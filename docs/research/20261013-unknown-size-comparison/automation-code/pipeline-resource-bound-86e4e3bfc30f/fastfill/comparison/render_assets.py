"""Canonical reference-asset renderer (stdlib planning; Blender 4+ rendering).

Run with Blender --background --factory-startup --threads 4 --python THIS --
--case case.json --output OUT --threads 4 --samples 8 --resolution 512.
Normal Python supports the identical CLI with --dry-run (no bpy import).

Portable case JSON: {uid, source, boundary: [[x,y],...], height, fixed: [],
expected_ids: [id,...], asset_mapping: [{prompt_id, asset_id, mesh_path, source,
layout_object: {scale: [1,1,1]}}], layouts: {rawGT: [...], candidate: [...]}}.
Every placement needs id/size/pos/yaw, in metres, Z-up, bottom-center, radians.
Missing/null height uses a shared 3.0 m render assumption. A finite positive
render_height overrides a valid/unknown height; invalid stated heights reject.
Receipts bind canonical JSON layouts, OBJ/MTL/texture bytes and published images.
--manifest accepts {cases: [case,...]}. --metric-inputs requires --uid, --stage,
and --asset-map; selects exactly one archived case, never the whole test set.
--asset-root relocates meshes to ROOT/asset_id/<archived filename>.

Geometry contract: sign(native scale), then Holo (-z,x,y) / Front (z,x,y),
union AABB bottom-center, fit this layout's size, canonical yaw, position.
OBJ normals use inverse transpose; reflections reverse complete face corners,
preserving UV indices. Positive native scale is replaced by bbox fit. No raw
rotation is reapplied. Reference asset identities are oracle evaluation data;
this is joint size/layout visualization, not native-scale Isaac Sim fidelity.
No geometry is inferred from descriptions, and no furniture box fallback exists.

Protocol v1: polygon floor, room-only cutaway walls, context doors/windows as
procedural structures, real meshes for all furniture, fixed room-only cameras
and lights, Cycles CPU, opaque RGB PNG, AgX. Perspective is the FID view; --top
adds an orthographic judging view. No scores or API calls are made here.
--denoise enables CPU OpenImageDenoise; use identical samples/settings per arm.
Only the documented f350... model.mtl -> raw_model.mtl override is allowed.
Inputs are read-only; transformed OBJ/MTL files live in a temporary directory.
Receipts retain every rejected case; images publish only after all views pass.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile


class RenderReject(ValueError):
    """A case/resource cannot be evaluated faithfully."""


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha256(value):
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RenderReject("invalid_number: expected finite number")
    return float(value)


def vector(value, length=3):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise RenderReject(f"invalid_vector: expected {length} numbers")
    return tuple(number(v) for v in value)


def source_point(v, source):
    x, y, z = v
    if source == "OptiScene_holodeck":
        return (-z, x, y)
    if source == "IL3D_3dfront":
        return (z, x, y)
    raise RenderReject(f"unsupported_source: {source}")


def vertex_bounds(vertices):
    points = [vector(v) for v in vertices]
    if not points:
        raise RenderReject("empty_mesh: no vertices")
    return tuple(tuple(fn(axis) for axis in zip(*points)) for fn in (min, max))


def matvec(matrix, value):
    return tuple(sum(a*b for a, b in zip(row, value)) for row in matrix)


@dataclass(frozen=True)
class AssetTransform:
    linear: tuple
    offset: tuple
    normal_matrix: tuple
    flip_winding: bool

    def point(self, value):
        return tuple(a+b for a, b in zip(matvec(self.linear, value), self.offset))

    def normal(self, value):
        transformed = matvec(self.normal_matrix, value)
        length = math.sqrt(sum(v*v for v in transformed))
        if length <= 1e-15:
            raise RenderReject("invalid_normal: zero length")
        return tuple(v/length for v in transformed)


def build_transform(raw_bounds, source, size, pos, yaw, native_scale=(1, 1, 1)):
    size, pos, native_scale = vector(size), vector(pos), vector(native_scale)
    yaw = number(yaw)
    if min(size) <= 0 or any(v == 0 for v in native_scale):
        raise RenderReject("invalid_scale: positive size and nonzero native scale required")
    signs = tuple(math.copysign(1, v) for v in native_scale)
    lo, hi = (vector(v) for v in raw_bounds)
    if any(b-a <= 1e-9 for a, b in zip(lo, hi)):
        raise RenderReject("degenerate_mesh: zero or reversed bbox extent")
    basis = tuple(zip(*(source_point(tuple(signs[i] if j == i else 0 for j in range(3)), source)
                        for i in range(3))))
    lower, upper = vertex_bounds(matvec(basis, v) for v in itertools.product(*zip(lo, hi)))
    pivot = ((lower[0]+upper[0])/2, (lower[1]+upper[1])/2, lower[2])
    fit = tuple(s/(b-a) for s, a, b in zip(size, lower, upper))
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    linear = tuple(tuple(sum(rotation[i][k]*fit[k]*basis[k][j] for k in range(3))
                         for j in range(3)) for i in range(3))
    normals = tuple(tuple(sum(rotation[i][k]/fit[k]*basis[k][j] for k in range(3))
                          for j in range(3)) for i in range(3))
    shift = matvec(rotation, tuple(pivot[i]*fit[i] for i in range(3)))
    determinant_sign = (-1 if source == "OptiScene_holodeck" else 1)*math.prod(signs)
    return AssetTransform(linear, tuple(pos[i]-shift[i] for i in range(3)), normals,
                          determinant_sign < 0)


def turn(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def triangulate_floor(boundary):
    """Ear clip a simple concave polygon, rejecting ambiguous/invalid boundaries."""
    points = [vector(p, 2) for p in boundary]
    n = len(points)
    if n < 3 or len(set(points)) != n:
        raise RenderReject("invalid_polygon: fewer than three or duplicate vertices")
    for i, a in enumerate(points):
        b = points[(i+1) % n]
        for j in range(i+1, n):
            if j in (i, (i+1) % n) or (j+1) % n == i:
                continue
            c, d = points[j], points[(j+1) % n]
            if (max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
                    and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))
                    and turn(a, b, c)*turn(a, b, d) <= 0 and turn(c, d, a)*turn(c, d, b) <= 0):
                raise RenderReject("invalid_polygon: intersecting edges")
    area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))
    indices = list(range(n)) if area > 0 else list(reversed(range(n)))
    triangles = []
    while len(indices) > 3:
        for k, b in enumerate(indices):
            a, c = indices[k-1], indices[(k+1) % len(indices)]
            if turn(points[a], points[b], points[c]) <= 1e-10:
                continue
            if any(all(turn(points[u], points[v], points[p]) >= -1e-10
                       for u, v in ((a, b), (b, c), (c, a)))
                   for p in indices if p not in (a, b, c)):
                continue
            triangles.append((a, b, c))
            indices = indices[:k]+indices[k+1:]
            break
        else:
            raise RenderReject("invalid_polygon: cannot triangulate")
    if abs(area) < 1e-10 or turn(*(points[i] for i in indices)) <= 1e-10:
        raise RenderReject("invalid_polygon: degenerate area")
    return triangles+[tuple(indices)]


def room_camera(boundary, height, view="perspective"):
    xs, ys = zip(*boundary)
    dx, dy = max(xs)-min(xs), max(ys)-min(ys)
    target = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, height/2)
    span = max(dx, dy)
    if view == "top":
        return {"view": view, "target": list(target), "location": [target[0], target[1], height+span*2],
                "ortho_scale": span*1.15}
    radius = math.sqrt(dx*dx+dy*dy+height*height)/2
    distance = radius/math.sin(math.radians(25))*1.12
    direction = (1, -1, 1.25)
    norm = math.sqrt(sum(v*v for v in direction))
    return {"view": view, "target": list(target), "fov_degrees": 50,
            "location": [target[i]+direction[i]*distance/norm for i in range(3)]}


def statement(line):
    tokens = line.split("#", 1)[0].strip().split(maxsplit=1)
    return (tokens[0], tokens[1] if len(tokens) > 1 else "") if tokens else ("", "")


def texture_spec(tail):
    """Read common Wavefront options, keeping whitespace in texture filenames."""
    tokens = shlex.split(tail)
    options = []
    counts = {"-bm": 1, "-clamp": 1, "-blendu": 1, "-blendv": 1, "-boost": 1,
              "-mm": 2, "-texres": 1, "-imfchan": 1, "-type": 1, "-cc": 1}
    while tokens and tokens[0].startswith("-"):
        flag = tokens.pop(0)
        options.append(flag)
        count = counts.get(flag)
        if flag in ("-s", "-o", "-t"):
            count = 0
            for token in tokens[:3]:
                try:
                    float(token)
                    count += 1
                except ValueError:
                    break
        if not count or len(tokens) <= count:
            raise RenderReject(f"unsupported_mtl_option: {flag}")
        options.extend(tokens[:count])
        tokens = tokens[count:]
    if not tokens:
        raise RenderReject("missing_texture: empty path")
    return " ".join(options), " ".join(tokens)


def read_material(path):
    lines, names, textures = [], set(), []
    content = path.read_bytes()
    for line in content.decode("utf-8").splitlines():
        key, tail = statement(line)
        if key == "newmtl":
            names.add(tail)
        if key.lower().startswith("map_") or key.lower() in ("bump", "disp", "decal", "refl", "norm"):
            options, filename = texture_spec(tail)
            texture = (path.parent / filename.replace("\\", "/")).resolve()
            if not texture.is_file():
                raise RenderReject(f"missing_texture: {texture}")
            textures.append(str(texture))
            line = " ".join(part for part in (key, options, str(texture)) if part)
        lines.append(line)
    return {"path": str(path), "names": sorted(names), "textures": textures,
            "sha256": hashlib.sha256(content).hexdigest(),
            "texture_sha256": {p: file_sha256(p) for p in dict.fromkeys(textures)},
            "text": "\n".join(lines)+"\n"}


def inspect_asset(mesh_path, asset_id):
    mesh = Path(mesh_path).resolve()
    if mesh.suffix.lower() != ".obj" or not mesh.is_file():
        raise RenderReject(f"no_asset: missing OBJ {mesh}")
    vertices, libraries, used, face_count = [], [], set(), 0
    with mesh.open() as stream:
        for line in stream:
            key, tail = statement(line)
            if key == "v":
                vertices.append(vector([float(v) for v in tail.split()[:3]]))
            elif key == "mtllib":
                libraries.extend(shlex.split(tail))
            elif key == "usemtl":
                used.add(tail)
            elif key == "f":
                face_count += 1
                if len(tail.split()) < 3:
                    raise RenderReject("invalid_face: fewer than three corners")
    if not face_count or not libraries or not used:
        raise RenderReject("incomplete_asset: faces, mtllib and usemtl required")
    materials, overrides = [], {}
    for library in dict.fromkeys(libraries):
        mtl = mesh.parent / library
        if not mtl.is_file() and asset_id == "f350c1ba-8040-4750-9e6e-f45cfbf916a4" and library == "model.mtl":
            mtl = mesh.parent / "raw_model.mtl"
            overrides[library] = "raw_model.mtl"
        if not mtl.is_file():
            raise RenderReject(f"missing_mtl: {mtl}")
        materials.append(read_material(mtl))
    defined = {name for mtl in materials for name in mtl["names"]}
    if not used.issubset(defined):
        raise RenderReject(f"undefined_material: {sorted(used-defined)}")
    return {"mesh_path": str(mesh), "bounds": vertex_bounds(vertices), "materials": materials,
            "asset_id": asset_id, "overrides": overrides, "vertices": len(vertices), "faces": face_count,
            "sha256": file_sha256(mesh)}


def prepare_obj(resource, transform, directory):
    """Write world-coordinate OBJ + resolved MTL copies, never touching raw files."""
    directory = Path(directory)
    names = [f"material-{i}.mtl" for i in range(len(resource["materials"]))]
    for name, material in zip(names, resource["materials"]):
        text = material["text"]
        # Native OBJ importer joins the directory to absolute map paths too.
        # Local aliases avoid this and also handle spaces in source filenames.
        for i, texture in enumerate(dict.fromkeys(material["textures"])):
            alias = f"{Path(name).stem}-texture-{i}{Path(texture).suffix}"
            try:
                (directory / alias).symlink_to(texture)
            except OSError:
                shutil.copyfile(texture, directory / alias)
            text = text.replace(texture+"\n", alias+"\n")
        (directory / name).write_text(text)
    destination = directory / "world.obj"
    with Path(resource["mesh_path"]).open() as source, destination.open("w") as target:
        target.write("mtllib "+" ".join(names)+"\n")
        for line in source:
            key, tail = statement(line)
            if key == "mtllib":
                continue
            if key in ("v", "vn"):
                fields = tail.split()
                xyz = vector([float(v) for v in fields[:3]])
                point = transform.point(xyz) if key == "v" else transform.normal(xyz)
                extra = fields[3:] if key == "v" else []
                line = key+" "+" ".join([format(v, ".12g") for v in point]+extra)+"\n"
            elif key == "f" and transform.flip_winding:
                line = "f "+" ".join(reversed(tail.split()))+"\n"
            target.write(line)
    return destination


def validate_placement(obj):
    if not isinstance(obj, dict) or not all(k in obj for k in ("id", "size", "pos", "yaw")):
        raise RenderReject("incomplete_placement: id/size/pos/yaw required")
    if not isinstance(obj["id"], str) or not obj["id"]:
        raise RenderReject("invalid_id")
    if min(vector(obj["size"])) <= 0:
        raise RenderReject("invalid_scale: size must be positive")
    vector(obj["pos"])
    number(obj["yaw"])
    if obj.get("tilted"):
        raise RenderReject("unsupported_tilt: canonical renderer supports yaw only")


def mapped_object(obj, mapping, source, asset_root, base_dir, cache):
    validate_placement(obj)
    row = mapping.get(obj["id"])
    if not row or not row.get("asset_id") or not row.get("mesh_path"):
        raise RenderReject(f"no_asset: {obj['id']}")
    path = Path(row["mesh_path"])
    if asset_root is not None:
        if Path(row["asset_id"]).name != row["asset_id"] or row["asset_id"] in (".", ".."):
            raise RenderReject("invalid_asset_id")
        path = Path(asset_root) / row["asset_id"] / path.name
    elif not path.is_absolute():
        path = Path(base_dir) / path
    key = (str(path.resolve()), row["asset_id"])
    if key not in cache:
        cache[key] = inspect_asset(*key)
    resource = cache[key]
    asset_source = row.get("source", source)
    scale = row.get("native_scale", row.get("layout_object", {}).get("scale", (1, 1, 1)))
    transform = build_transform(resource["bounds"], asset_source, obj["size"], obj["pos"], obj["yaw"], scale)
    return {"id": obj["id"], "size": list(obj["size"]), "pos": list(obj["pos"]), "yaw": obj["yaw"],
            "source": asset_source, "native_scale": list(scale), "resource": resource, "transform": transform}


def height_policy(case):
    original = case.get("height")
    for field in ("height", "render_height"):
        if field == "height" and original is None:
            continue
        if field not in case:
            continue
        try:
            if number(case[field]) <= 0:
                raise RenderReject("must be positive")
        except (ValueError, OverflowError) as exc:
            raise RenderReject(f"invalid_room_height: {field} must be finite and positive") from exc
    override = "render_height" in case
    imputed = not override and original is None
    effective = case["render_height"] if override else (3.0 if imputed else original)
    return {"input_height": original, "effective_height": float(effective), "imputed": imputed,
            "policy": "case_override" if override else ("unknown_uniform_3m" if imputed else "case_height")}


def validate_case_shape(case):
    if not isinstance(case, dict):
        raise RenderReject("incomplete_case: expected JSON object")
    if case.get("input_error"):
        raise RenderReject(str(case["input_error"]))
    if not isinstance(case.get("uid"), str) or not case["uid"]:
        raise RenderReject("incomplete_case: uid required")
    if case.get("stage") is not None and not isinstance(case["stage"], str):
        raise RenderReject("invalid_stage: expected string or null")
    if case.get("layout_variant", "provided") not in ("provided", "raw", "repaired"):
        raise RenderReject("invalid_layout_variant")
    for field in ("boundary", "fixed", "expected_ids"):
        if not isinstance(case.get(field), list):
            raise RenderReject(f"invalid_case: {field} must be a list")
    if not isinstance(case.get("layouts"), dict) or not case["layouts"]:
        raise RenderReject("incomplete_case: layouts must be a nonempty object")
    for name, objects in case["layouts"].items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise RenderReject("invalid_layout_name")
        if not isinstance(objects, list):
            raise RenderReject(f"invalid_layout: {name} must be a list")
        for obj in objects:
            validate_placement(obj)
    for obj in case["fixed"]:
        validate_placement(obj)


def plan_case(case, asset_root=None, base_dir="."):
    validate_case_shape(case)
    if case.get("schema_complete") is False or case.get("parsed") is False:
        raise RenderReject("incomplete_case: upstream schema/parse failure")
    boundary = [vector(p, 2) for p in case["boundary"]]
    triangles = triangulate_floor(boundary)
    provenance = height_policy(case)
    height = provenance["effective_height"]
    expected = case["expected_ids"]
    if not expected or any(not isinstance(i, str) or not i for i in expected) or len(set(expected)) != len(expected):
        raise RenderReject("incomplete_inventory: nonempty unique expected_ids required")
    rows = case.get("asset_mapping", [])
    if isinstance(rows, dict):
        if any(not isinstance(row, dict) for row in rows.values()):
            raise RenderReject("invalid_asset_mapping: expected object rows")
        rows = [{**row, "prompt_id": key} for key, row in rows.items()]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RenderReject("invalid_asset_mapping: expected list or object")
    mapping = {row["prompt_id"]: row for row in rows}
    if len(mapping) != len(rows):
        raise RenderReject("duplicate_id: asset mapping")
    cache, fixed, layouts = {}, [], {}
    fixed_ids = [obj["id"] for obj in case["fixed"]]
    if len(set(fixed_ids)) != len(fixed_ids) or set(fixed_ids) & set(expected):
        raise RenderReject("duplicate_id: fixed inventory")
    for obj in case["fixed"]:
        validate_placement(obj)
        structural = obj.get("structure") is True or obj.get("fixed_kind") == "structure"
        if structural and str(obj.get("category", "")).lower() in ("door", "window", "wall", "column", "beam"):
            fixed.append({**obj, "procedural_structure": True})
        else:
            fixed.append(mapped_object(obj, mapping, case["source"], asset_root, base_dir, cache))
    for name, objects in case["layouts"].items():
        ids = [obj["id"] for obj in objects]
        if len(set(ids)) != len(ids):
            raise RenderReject(f"duplicate_id: {name}")
        if set(ids) != set(expected):
            raise RenderReject(f"incomplete_inventory: {name}; missing={sorted(set(expected)-set(ids))}; extra={sorted(set(ids)-set(expected))}")
        layouts[name] = [mapped_object(obj, mapping, case["source"], asset_root, base_dir, cache) for obj in objects]
    return {"uid": case["uid"], "boundary": boundary, "height": height, "triangles": triangles,
            "fixed": fixed, "layouts": layouts, "height_provenance": provenance,
            "cameras": {v: room_camera(boundary, height, v) for v in ("perspective", "top")}}


def metric_case(record, asset_rows, repaired=False):
    """Explicit archive adapter; no candidate geometry fallback to GT."""
    return {"uid": record["uid"], "stage": record.get("stage"), "source": record["source"],
            "boundary": record["boundary"], "height": record.get("height"), "fixed": record["fixed"],
            **({"render_height": record["render_height"]} if "render_height" in record else {}),
            "layout_variant": "repaired" if repaired else "raw",
            "archive_record_sha256": json_sha256(record),
            "expected_ids": [obj["id"] for obj in record["judging_request"]["objects"]],
            "parsed": record.get("parsed", True), "schema_complete": record.get("schema_complete", True),
            "asset_mapping": [row for row in asset_rows if row["uid"] == record["uid"]],
            "layouts": {"rawGT": record["gt_canonicalobjects"],
                        "candidate": record["repaired_layout" if repaired else "raw_layout"]}}


def make_mesh(bpy, name, vertices, faces, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def material(bpy, name, color):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1)
    value.use_nodes = True
    shader = value.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = .72
    return value


def make_room(bpy, plan):
    floor = material(bpy, "shared-floor", (.55, .53, .48))
    wall = material(bpy, "shared-walls", (.76, .75, .71))
    vertices = [(x, y, 0) for x, y in plan["boundary"]]
    make_mesh(bpy, "polygon-floor", vertices, plan["triangles"], floor)
    points = plan["boundary"]
    area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))
    points = points if area > 0 else list(reversed(points))
    camera = plan["cameras"]["perspective"]["location"]
    for a, b in zip(points, points[1:]+points[:1]):
        # Keep only far-facing walls; no object-dependent framing or wall selection.
        if turn(a, b, camera) < 0:
            continue
        h = plan["height"]
        make_mesh(bpy, "cutaway-wall", [(a[0], a[1], 0), (b[0], b[1], 0),
                                       (b[0], b[1], h), (a[0], a[1], h)], [(0, 1, 2, 3)], wall)


def import_furniture(bpy, obj, directory):
    path = prepare_obj(obj["resource"], obj["transform"], directory)
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z",
                          use_split_objects=False, use_split_groups=False, validate_meshes=True)
    meshes = [value for value in set(bpy.data.objects)-before if value.type == "MESH"]
    if not meshes or sum(len(value.data.polygons) for value in meshes) == 0:
        raise RenderReject(f"import_empty: {obj['id']}")
    for value in meshes:
        value.name = obj["id"]
        if not value.data.materials or any(mat is None for mat in value.data.materials):
            raise RenderReject(f"import_material_missing: {obj['id']}")
    bounds = vertex_bounds(tuple(value.matrix_world @ v.co) for value in meshes for v in value.data.vertices)
    expected = vertex_bounds(obj["transform"].point(v) for v in itertools.product(*zip(*obj["resource"]["bounds"])))
    # A rotated irregular mesh need not fill all transformed bbox corners. Check containment.
    if any(bounds[0][i] < expected[0][i]-1e-4 or bounds[1][i] > expected[1][i]+1e-4 for i in range(3)):
        raise RenderReject(f"import_axis_mismatch: {obj['id']}")


def configure_scene(bpy, plan, resolution, samples, threads, denoise=False):
    from mathutils import Vector
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.seed = 0
    scene.cycles.use_denoising = denoise
    if denoise:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.use_adaptive_sampling = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new("shared-world")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (.8, .8, .8, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = .65
    cx, cy, _ = plan["cameras"]["perspective"]["target"]
    span = plan["cameras"]["top"]["ortho_scale"]/1.15
    for name, delta, strength in (("key", (-.35, -.4), 85), ("fill", (.4, .3), 45)):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.shape, light.size = strength*span*span, "DISK", span*.75
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = (cx+delta[0]*span, cy+delta[1]*span, plan["height"]+span*.55)
        obj.rotation_euler = (Vector((cx, cy, 0))-obj.location).to_track_quat("-Z", "Y").to_euler()


def validate_images(images):
    for img in images:
        if img.source == "FILE":
            # has_data is false for valid, lazily decoded images in Blender 5.2.
            # Access pixels before consulting it (reload alone is insufficient).
            decoded = img.pixels[:4]
            if len(decoded) < 4 or not img.has_data or min(img.size) <= 0:
                raise RenderReject(f"texture_decode_failed: {img.filepath}")


def render_layout(bpy, plan, objects, directory, views, resolution, samples, threads, denoise=False):
    from mathutils import Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_scene(bpy, plan, resolution, samples, threads, denoise)
    make_room(bpy, plan)
    structure_material = material(bpy, "shared-structure", (.30, .36, .39))
    with tempfile.TemporaryDirectory(prefix="optiscene-obj-") as temporary:
        for i, obj in enumerate(plan["fixed"]+objects):
            if obj.get("procedural_structure"):
                bpy.ops.mesh.primitive_cube_add(size=1, location=(obj["pos"][0], obj["pos"][1], obj["pos"][2]+obj["size"][2]/2))
                cube = bpy.context.object
                cube.name, cube.dimensions, cube.rotation_euler.z = obj["id"], obj["size"], obj["yaw"]
                cube.data.materials.append(structure_material)
            else:
                asset_dir = Path(temporary) / str(i)
                asset_dir.mkdir()
                import_furniture(bpy, obj, asset_dir)
        validate_images(bpy.data.images)
        camera_data = bpy.data.cameras.new("shared-camera")
        camera = bpy.data.objects.new("shared-camera", camera_data)
        bpy.context.scene.collection.objects.link(camera)
        bpy.context.scene.camera = camera
        for view in views:
            spec = plan["cameras"][view]
            camera.location = spec["location"]
            camera.rotation_euler = (Vector(spec["target"])-camera.location).to_track_quat("-Z", "Y").to_euler()
            camera_data.type = "ORTHO" if view == "top" else "PERSP"
            camera_data.lens_unit = "FOV"
            camera_data.angle = math.radians(50)
            camera_data.ortho_scale = plan["cameras"]["top"]["ortho_scale"]
            camera_data.clip_start, camera_data.clip_end = .01, 10000
            bpy.context.scene.render.filepath = str(directory / f"{view}.png")
            bpy.ops.render.render(write_still=True)


def write_ledger(output, ledger):
    temporary = output / "ledger.tmp"
    temporary.write_text(json.dumps(ledger, indent=2, allow_nan=False)+"\n")
    temporary.replace(output / "ledger.json")


def plan_receipt(case, plan):
    assets = []
    for label, objects in [*plan["layouts"].items(), ("fixed_context", plan["fixed"])]:
        for obj in objects:
            if "resource" not in obj:
                continue
            resource = obj["resource"]
            assets.append({"id": obj["id"], "layout": label, "asset_id": resource["asset_id"],
                           "mesh_path": resource["mesh_path"], "sha256": resource["sha256"],
                           "mtl_overrides": resource["overrides"], "flip_winding": obj["transform"].flip_winding,
                           "materials": [{k: m[k] for k in ("path", "sha256", "texture_sha256")}
                                         for m in resource["materials"]]})
    provenance = {"cameras": plan["cameras"], "height_provenance": plan["height_provenance"], "assets": assets,
            "input_sha256": json_sha256(case), "layout_variant": case.get("layout_variant", "provided"),
            "asset_mapping_sha256": json_sha256(case.get("asset_mapping", [])),
            "archive_record_sha256": case.get("archive_record_sha256"),
            "layout_sha256": {name: json_sha256(objects) for name, objects in case["layouts"].items()}}
    json_sha256(provenance)  # Reject nonfinite derived geometry before it can poison a ledger.
    return provenance


def publish_case(bpy, plan, output, name, views, resolution, samples, threads, denoise):
    with tempfile.TemporaryDirectory(prefix=".render-", dir=output) as temp:
        for label, objects in plan["layouts"].items():
            folder = Path(temp) / label
            folder.mkdir()
            render_layout(bpy, plan, objects, folder, views, resolution, samples, threads, denoise)
        # Hash every required output before publication: missing/empty files fail the entire case.
        hashes = {}
        for label in plan["layouts"]:
            hashes[label] = {}
            for view in views:
                path = Path(temp) / label / f"{view}.png"
                if not path.is_file() or not path.stat().st_size:
                    raise RenderReject(f"missing_render_output: {label}/{view}")
                hashes[label][view] = file_sha256(path)
        Path(temp).rename(output / name)
    return {"images": {label: {v: str(output / name / label / f"{v}.png") for v in views}
                       for label in plan["layouts"]}, "image_sha256": hashes,
            "blender_version": bpy.app.version_string, "status": "rendered"}


def run_cases(cases, output, *, dry_run=False, asset_root=None, base_dir=".",
              top=False, resolution=512, samples=8, threads=4, denoise=False, case_adapter=None,
              layout_variant="provided"):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (resolution, samples, threads)):
        raise RenderReject("invalid_render_settings: positive integers required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ledger.json").exists():
        raise RenderReject("output_exists: choose a fresh output directory")
    config = {"protocol": "canonical-asset-v1", "resolution": resolution, "samples": samples,
              "threads": threads, "device": "CPU", "transparent": False, "top": top, "denoise": denoise,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    ledger = {"schema_version": 1, "config": config, "cases": []}
    write_ledger(output, ledger)
    if not isinstance(cases, (list, tuple)):
        cases = [{"input_error": "invalid_cases: expected list"}]
    for i, case in enumerate(cases):
        metadata = case if isinstance(case, dict) else {}
        receipt = {k: metadata.get(k) if isinstance(metadata.get(k), str) else None for k in ("uid", "stage")}
        variant = metadata.get("layout_variant", layout_variant)
        receipt = {**receipt, "case_index": i, "status": "pending", "images": {}, "image_sha256": {},
                   "layout_variant": variant if isinstance(variant, str) else None}
        ledger["cases"].append(receipt)
        write_ledger(output, ledger)
        try:
            case = case_adapter(case) if case_adapter else case
            if isinstance(case, dict) and not case.get("input_error"):
                receipt["height_provenance"] = height_policy(case)
            plan = plan_case(case, asset_root=asset_root, base_dir=base_dir)
            receipt.update(plan_receipt(case, plan))
            if dry_run:
                receipt["status"] = "ready"
            else:
                import bpy
                receipt["status"] = "rendering"
                write_ledger(output, ledger)
                views = ["perspective", "top"] if top else ["perspective"]
                name = f"{i:04d}-"+hashlib.sha256(case["uid"].encode()).hexdigest()[:12]
                receipt.update(publish_case(bpy, plan, output, name, views, resolution, samples, threads, denoise))
        except Exception as exc:  # Case isolation; interrupts/exits and ledger I/O failures still propagate.
            receipt["status"] = "failed" if receipt["status"] == "rendering" else "rejected"
            receipt["reason"] = f"{type(exc).__name__}: {exc}"
        write_ledger(output, ledger)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", type=Path)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--metric-inputs", type=Path)
    parser.add_argument("--asset-map", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--uid")
    parser.add_argument("--stage")
    parser.add_argument("--repaired", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top", action="store_true")
    parser.add_argument("--denoise", action="store_true", help="CPU OpenImageDenoise, shared by every layout/view")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    path = args.case or args.manifest or args.metric_inputs
    if args.metric_inputs and not (args.uid and args.stage and args.asset_map):
        parser.error("--metric-inputs requires explicit --uid --stage --asset-map")
    adapter = None
    try:
        data = json.loads(path.read_text())
        cases = [data] if args.case else data["cases"]
        if not isinstance(cases, list):
            raise RenderReject("invalid_cases: expected list")
        if args.metric_inputs:
            cases = [c for c in cases if isinstance(c, dict) and c.get("uid") == args.uid and c.get("stage") == args.stage]
            if len(cases) != 1:
                raise RenderReject(f"expected exactly one case; got {len(cases)}")
            rows = json.loads(args.asset_map.read_text())
            adapter = lambda record: metric_case(record, rows, args.repaired)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        cases = [{"uid": args.uid, "stage": args.stage, "input_error": f"input_document: {exc}"}]
    ledger = run_cases(cases, args.output, dry_run=args.dry_run, asset_root=args.asset_root,
                       base_dir=path.resolve().parent, top=args.top, resolution=args.resolution,
                       samples=args.samples, threads=args.threads, denoise=args.denoise, case_adapter=adapter,
                       layout_variant=("repaired" if args.repaired else "raw") if args.metric_inputs else "provided")
    print(json.dumps({"ledger": str(args.output / "ledger.json"),
                      "statuses": [c["status"] for c in ledger["cases"]]}))
    return int(any(c["status"] not in ("ready", "rendered") for c in ledger["cases"]))


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(arguments))
