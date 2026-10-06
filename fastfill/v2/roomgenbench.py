"""Dynamic RoomGenBench assembly with explicit paths and truthful mesh/proxy receipts.

Only the upstream geometry helpers are imported. Its fixed five-scene CLI,
global paths, reference GT branch and source files are never changed. Asset
generation is a separate step; this adapter consumes generated GLB/sidecars.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

from .direct_layout import layout_to_roomgenbench
from .io import fingerprint, safe_output
from .schema import finite


FIT_POLICY = "roomgenbench_anisotropic_yaw_snap_tip"


def _read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _reference(root):
    source = Path(root).resolve() / "bench" / "assemble.py"
    if not source.is_file():
        raise FileNotFoundError(f"RoomGenBench assembler not found: {source}")
    spec = importlib.util.spec_from_file_location("_fastfill_roomgenbench_geometry", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"source": str(source), "sha256": fingerprint(source)}


def _handoff(path, require_placement):
    directory = Path(path).resolve()
    condition, layout = (_read_json(directory / name) for name in ("condition.json", "layout.json"))
    scene = layout_to_roomgenbench(condition, layout)
    if _read_json(directory / "roomgenbench_scene.json") != scene:
        raise ValueError("RoomGenBench handoff must match the canonical condition and layout")
    if require_placement and any(o["place_id"] is None for o in scene["objects"]):
        raise ValueError("this downstream policy requires explicit placement for every requested object")
    return condition, scene


def _place(parent):
    return "unknown" if parent is None else parent if parent in {"floor", "wall"} else "on_object"


def _extents(parts):
    import numpy as np
    vertices = np.vstack([part.vertices for part in parts])
    if not np.isfinite(vertices).all():
        raise ValueError("asset vertices must be finite")
    lengths = vertices.max(0) - vertices.min(0)
    if (lengths <= 0).any():
        raise ValueError("asset must have positive three-dimensional extents")
    return lengths.tolist()


def _asset_parts(module, obj, method, assets_dir):
    """Missing/failed/invalid assets retain a requested instance as a placeholder."""
    key = obj["asset_key"]
    sidecar, glb = assets_dir / f"{key}.json", assets_dir / f"{key}.glb"
    if not sidecar.is_file():
        return [], {"status": "missing", "reason": "sidecar_missing"}
    try:
        record = _read_json(sidecar)
        if (not isinstance(record, dict) or record.get("asset_key") != key
                or record.get("method") != method or not isinstance(record.get("status"), str)
                or record["status"] not in {"ok", "fallback", "failed"}):
            raise ValueError("sidecar requires matching asset_key/method and explicit ok/fallback/failed status")
    except (OSError, ValueError) as error:
        return [], {"status": "invalid_sidecar", "reason": str(error)}
    status = module.effective_status(method, sidecar)
    if status == "failed":
        return [], {"status": "failed", "reason": "method_reported_failure"}
    if not glb.is_file():
        return [], {"status": "missing", "reason": "glb_missing"}
    try:
        parts = module.load_glb_parts(glb)
        native = _extents(parts)
        sage_parts = [part.copy() for part in parts]
        for part in sage_parts:
            part.apply_transform(module.GLTF_TO_SAGE)
        fit_log = {"status": status, "native_size_gltf_m": native,
                   "native_size_sage_local_m": _extents(sage_parts)}
        fitted = module.fit_asset(parts, obj["dimensions"], fit_log)
        extents = _extents(fitted)
        target = [obj["dimensions"][k] for k in ("width", "length", "height")]
        # Upstream clamps thin native axes before scaling. Report a resulting
        # target mismatch rather than accepting the method's original ok status.
        if any(abs(a - b) > max(1e-9, 1e-5 * b) for a, b in zip(extents, target)):
            fit_log = {**fit_log, "status": "fit_mismatch", "source_status": status}
        return fitted, {**fit_log, "fitted_size_sage_local_m": extents,
                        "source_glb_sha256": fingerprint(glb), "source_sidecar_sha256": fingerprint(sidecar)}
    except Exception as error:
        return [], {"status": "failed", "reason": f"asset_load_or_fit_failed: {type(error).__name__}: {error}"}


def _box_parts(module, dimensions):
    import trimesh
    lengths = [dimensions[k] for k in ("width", "length", "height")]
    transform = trimesh.transformations.translation_matrix([0, 0, lengths[2] / 2])
    return [module.colored_box(lengths, transform, module.PLACEHOLDER_RGBA)]


def _objects(module, scene, method, assets_dir, output):
    logs = []
    for index, obj in enumerate(scene["objects"]):
        parts, fit_log = ([], {"status": "box"}) if method == "layout_boxes" else _asset_parts(module, obj, method, assets_dir)
        kind = "generated_fitted_mesh" if fit_log["status"] == "ok" else (
            "fitted_mesh_with_size_mismatch" if fit_log["status"] == "fit_mismatch" else
            "fallback_mesh" if parts else "target_bbox_proxy" if method == "layout_boxes" else "placeholder_bbox")
        parts = parts or _box_parts(module, obj["dimensions"])
        node = f"obj_{index:03d}"
        for part_index, part in enumerate(parts):
            name = node if len(parts) == 1 else f"{node}_p{part_index}"
            output.add_geometry(part, node_name=name, geom_name=name,
                                transform=module.SAGE_WORLD_TO_GLTF @ module.place_matrix(obj))
        logs = logs + [{"id": obj["id"], "asset_key": obj["asset_key"], "type": obj["type"],
                       "place": _place(obj["place_id"]), "support_parent": obj["place_id"],
                       "support_status": obj["support_status"], "geometry_kind": kind,
                       "node": node, **fit_log}]
    return logs


def _display_geometry(module, condition, output, display_height):
    """Display rectangle/fixed boxes are proxies; the original condition is retained."""
    import trimesh
    room = condition["room"]
    points = room["floor_polygon_xy_m"]
    low = [min(p[q] for p in points) for q in (0, 1)]
    high = [max(p[q] for p in points) for q in (0, 1)]
    floor = room.get("floor_z_m")
    display_floor = 0. if floor is None else floor
    transform = trimesh.transformations.translation_matrix([
        (low[0] + high[0]) / 2, (low[1] + high[1]) / 2, display_floor - .025])
    output.add_geometry(module.colored_box([high[0]-low[0], high[1]-low[1], .05], transform, module.FLOOR_RGBA),
                        node_name="display_room_extent_proxy", geom_name="display_room_extent_proxy",
                        transform=module.SAGE_WORLD_TO_GLTF)
    for index, fixed in enumerate(room.get("fixed_objects", [])):
        w, d, h = fixed["size_local_m"]
        # Same +X -> +Y conversion as the requested-object adapter.
        obj = {"position": dict(zip("xyz", fixed["bottom_center_m"])),
               "rotation": {"x": 0., "y": 0., "z": 180 * fixed["yaw_rad"] / 3.141592653589793 - 90},
               "dimensions": {"width": d, "length": w, "height": h}}
        node = f"fixed_bbox_{index:03d}"
        output.add_geometry(_box_parts(module, obj["dimensions"])[0], node_name=node, geom_name=node,
                            transform=module.SAGE_WORLD_TO_GLTF @ module.place_matrix(obj))
    height = room.get("height_m")
    return {"width": high[0] - low[0], "length": high[1] - low[1],
            "height": display_height if height is None else height,
            "height_source": "display_reference" if height is None else "condition",
            "source_height_m": height, "floor_known": room.get("floor_known", floor is not None),
            "boundary_known": room.get("boundary_known", True),
            "shell_geometry_kind": "rectangular_extent_display_proxy",
            "display_floor_z_m": display_floor, "source_floor_z_m": floor}


def assemble_handoff(handoff, output_dir, *, method="layout_boxes", assets_dir=None,
                     roomgenbench_root=None, require_placement=False, display_height_m=3.):
    """Consume an arbitrary FastFill handoff and optionally generated mesh assets.

    The returned receipt always uses all requested objects as its denominator.
    `generated_mesh_success` means every requested GLB loaded/fitted with status
    ok; it does not assert native dimensions, room feasibility or deployment.
    """
    if not isinstance(method, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", method) or method == "sage_gt":
        raise ValueError("method must be a short snake_case generated method or layout_boxes; sage_gt is unavailable")
    if method != "layout_boxes" and assets_dir is None:
        raise ValueError("generated mesh assembly requires --assets-dir")
    height = finite(display_height_m, "display reference height", positive=True)
    target = safe_output(output_dir)
    inputs = [Path(handoff).resolve()] + ([Path(assets_dir).resolve()] if assets_dir is not None else [])
    if any(target == source or target in source.parents or source in target.parents for source in inputs):
        raise ValueError("output directory must not overlap handoff or asset inputs")
    condition, scene = _handoff(handoff, require_placement)
    reference = Path(roomgenbench_root or Path(__file__).resolve().parents[2] / "RoomGenBench").resolve()
    if target == reference or target in reference.parents or reference in target.parents:
        raise ValueError("output directory must remain independent of the reference repository")
    module, source_info = _reference(reference)
    import trimesh
    output = trimesh.Scene()
    room = _display_geometry(module, condition, output, height)
    objects = _objects(module, scene, method, Path(assets_dir).resolve() if assets_dir is not None else None, output)
    counts = dict(Counter(obj["status"] for obj in objects))
    generated_success = method != "layout_boxes" and bool(objects) and counts.get("ok", 0) == len(objects)
    receipt = {"schema_version": "fastfill.roomgenbench-assembly.v1", "scene_key": scene["scene_key"],
        "scene": scene["scene_key"], "room_type": scene["room_type"], "walls": [],
        "method": method, "assembly_complete": True, "generated_mesh_success": generated_success,
        "requested_objects": len(objects), "counts": counts, "objects": objects, "room": room,
        "fixed_bbox_proxies": len(condition["room"].get("fixed_objects", [])),
        "condition": deepcopy(condition), "fit_policy": FIT_POLICY if method != "layout_boxes" else "target_bbox_proxy",
        "original_asset_geometry_acceptance": "not_checked", "support_verification": "not_checked",
        "validator": "not_attempted", "physics": "not_attempted", "host_commit": "not_attempted",
        "reference_assembler": source_info}
    # Build/serialize everything before creating the immutable output directory.
    payload = output.export(file_type="glb")
    metadata = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    target.mkdir(parents=True, exist_ok=False)
    (target / f'{scene["scene_key"]}.glb').write_bytes(payload)
    (target / f'{scene["scene_key"]}.json').write_text(metadata)
    (target / "receipt.json").write_text(metadata)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, help="directory from predict --export-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="layout_boxes")
    parser.add_argument("--assets-dir", help="method GLB/JSON sidecar directory")
    parser.add_argument("--roomgenbench-root")
    parser.add_argument("--require-placement", action="store_true")
    parser.add_argument("--display-height-m", type=float, default=3.)
    args = parser.parse_args(argv)
    receipt = assemble_handoff(args.handoff, args.output_dir, method=args.method, assets_dir=args.assets_dir,
        roomgenbench_root=args.roomgenbench_root, require_placement=args.require_placement, display_height_m=args.display_height_m)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "counts": receipt["counts"],
                      "generated_mesh_success": receipt["generated_mesh_success"]}))
    return 0 if args.method == "layout_boxes" or receipt["generated_mesh_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
