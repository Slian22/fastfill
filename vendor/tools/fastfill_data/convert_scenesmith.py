"""Convert scenesmith released example scenes into FastFillSample JSONL.

Source layout (see ``SCENESMITH_NOTES.md``): each scene under
``<data>/scenesmith_scenes/<subset>/scene_*/`` for the eight HuggingFace
subsets (Room, House, NoAgentMemory, NoAssetValidation, NoCritic,
NoObserveScene, NoSpecializedTools, NotGenerated; absent subset dirs skip
cleanly)
ships only the Drake directive ``combined_house/house.dmd.yaml`` (written by
``scenesmith/agent_utils/house.py::HouseScene.assemble``), per-room shell SDFs
under ``room_geometry/`` and per-asset SDF+mesh dirs under
``room_<name>/generated_assets/``. The rich ``house_state.json`` /
``scene_state.json`` that would carry explicit support-parent relations is
NOT shipped, so support parents are inferred geometrically from the real
poses (recorded as ``support=inferred_geometric`` in provenance notes).

Frames: scenesmith is already Z-up right-handed meters, and its mesh
canonicalization guarantees front=+Y, bottom at z=0, XY-centered model
origins (``mesh_canonicalization.py::canonicalize_mesh``) — so poses map to
the FastFill contract without any Y-up conversion; we only recenter the
floor polygon and apply the same offset to object positions.

Usage (from repo root):
    python tools/fastfill_data/convert_scenesmith.py \
        --data-dir <...>/data --out out/scenesmith_scenes.jsonl
"""

from __future__ import annotations

import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

if __package__ in (None, ""):  # executed directly as a script
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "tools")):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from fastfill_data.common import ConversionStats, converter_main

from scenesmith.growing_world.fastfill.schema import (
    FastFillSample,
    FloorLayout,
    FloorObjectSpec,
    LicenseTag,
    ProvenanceMeta,
    RoomContentLayout,
    RoomContext,
    SupportSurfaceSpec,
    SurfaceKind,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
    Vec2,
    Vec3,
)
from scenesmith.growing_world.fastfill.transforms import (
    ensure_ccw,
    normalize_deg,
    recenter_polygon,
    world_to_local_2d,
)

SOURCE_DATASET = "scenesmith_scenes"
SUBSETS: tuple[str, ...] = (
    "Room",
    "House",
    "NoAgentMemory",
    "NoAssetValidation",
    "NoCritic",
    "NoObserveScene",
    "NoSpecializedTools",
    "NotGenerated",
)

# An object is treated as upright when the rotated body +Z stays within
# ~25 degrees of world +Z (R[2,2] >= 0.9). Tilted manipulands (fallen bags,
# balls, scattered decor) cannot be represented by the yaw-only contract and
# are skipped + counted, never re-uprighted.
_UPRIGHT_MIN_R22 = 0.9
_FOOTPRINT_MARGIN_M = 0.05  # xy slack when testing support containment
_ABOVE_TOP_TOL_M = 0.10  # resting plane may exceed the parent top by this
_LEVEL_MERGE_M = 0.04  # resting-z gap below which two objects share a shelf
_TOP_KIND_TOL_M = 0.06  # resting plane within this of parent top => TOP
_FLOOR_REST_TOL_M = 0.05  # manipuland resting below this is floor-standing
_MIN_ASSET_EXTENT_M = 1e-3


# ---------------------------------------------------------------- dmd parsing


class _DmdLoader(yaml.SafeLoader):
    """SafeLoader that understands Drake's ``!AngleAxis`` rotation tag."""


def _angle_axis(loader: yaml.Loader, node: yaml.Node) -> dict:
    return loader.construct_mapping(node, deep=True)


_DmdLoader.add_constructor("!AngleAxis", _angle_axis)


def _yaw_and_uprightness(rotation: dict) -> tuple[float, float]:
    """AngleAxis -> (contract yaw_deg, R[2,2] uprightness) via Rodrigues."""
    angle = math.radians(float(rotation["angle_deg"]))
    axis = np.asarray(rotation["axis"], dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return 0.0, 1.0
    axis = axis / norm
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rot = np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)
    yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return normalize_deg(yaw), float(rot[2, 2])


# ------------------------------------------------------------ asset geometry


def _obj_vertex_bounds(
    path: Path, scale: tuple[float, float, float]
) -> tuple[list[float], list[float]] | None:
    """Scaled AABB of an OBJ file's vertices (Drake reads OBJ as Z-up)."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for line in path.read_text().splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        for i in range(3):
            v = float(parts[i + 1]) * scale[i]
            lo[i] = min(lo[i], v)
            hi[i] = max(hi[i], v)
    if lo[0] is math.inf:
        return None
    return lo, hi


def _asset_dimensions(sdf_path: Path) -> Vec3 | None:
    """Full [width, depth, height] extents from an asset SDF's collision
    meshes (union AABB in the model frame; joints on multi-link articulated
    assets are ignored — links are authored in place)."""
    try:
        root = ET.parse(sdf_path).getroot()
    except (ET.ParseError, OSError):
        return None
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for collision in root.iter("collision"):
        mesh = collision.find("./geometry/mesh")
        if mesh is None:
            continue
        uri = mesh.findtext("uri")
        scale_text = mesh.findtext("scale")
        scale = (
            tuple(float(s) for s in scale_text.split())
            if scale_text
            else (1.0, 1.0, 1.0)
        )
        obj_path = sdf_path.parent / uri
        if not obj_path.exists():
            continue
        bounds = _obj_vertex_bounds(obj_path, scale)  # type: ignore[arg-type]
        if bounds is None:
            continue
        pose_text = collision.findtext("pose")
        offset = (
            [float(v) for v in pose_text.split()[:3]] if pose_text else [0.0, 0.0, 0.0]
        )
        for i in range(3):
            lo[i] = min(lo[i], bounds[0][i] + offset[i])
            hi[i] = max(hi[i], bounds[1][i] + offset[i])
    if lo[0] is math.inf:
        return None
    dims = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    if min(dims) < _MIN_ASSET_EXTENT_M:
        return None
    return dims


# ----------------------------------------------------------------- room shell


def _parse_room_shell(
    sdf_path: Path,
) -> tuple[tuple[Vec2, ...], float | None] | None:
    """(floor rectangle CCW, ceiling height) from a room-geometry SDF.

    The shell writer emits an axis-aligned ``floor_collision`` box centered
    on the room frame plus per-wall collision boxes; ceiling height is the
    top of the tallest wall box. Returns None when no floor box exists.
    """
    try:
        root = ET.parse(sdf_path).getroot()
    except (ET.ParseError, OSError):
        return None
    floor: tuple[list[float], list[float]] | None = None
    wall_tops: list[float] = []
    for collision in root.iter("collision"):
        size_node = collision.find("./geometry/box/size")
        if size_node is None or size_node.text is None:
            continue
        size = [float(v) for v in size_node.text.split()]
        pose_text = collision.findtext("pose") or "0 0 0 0 0 0"
        pose = [float(v) for v in pose_text.split()]
        name = collision.get("name", "")
        if name == "floor_collision":
            floor = (size, pose)
        elif "wall" in name:
            wall_tops.append(pose[2] + size[2] / 2.0)
    if floor is None:
        return None
    (sx, sy, _), (px, py, *_rest) = floor
    polygon = ensure_ccw(
        (
            (px - sx / 2.0, py - sy / 2.0),
            (px + sx / 2.0, py - sy / 2.0),
            (px + sx / 2.0, py + sy / 2.0),
            (px - sx / 2.0, py + sy / 2.0),
        )
    )
    return polygon, (max(wall_tops) if wall_tops else None)


# ------------------------------------------------------------- scene parsing


@dataclass(frozen=True)
class _PosedObject:
    """One free body from house.dmd.yaml, in its room's frame (Z-up, m)."""

    object_id: str
    category: str
    asset_dir_id: str
    position: Vec3
    yaw_deg: float
    uprightness: float
    dimensions: Vec3


def _category_from_name(model_name: str, room_name: str) -> str:
    """``pet_store_dog_ball_toy_0_f0_1`` -> ``dog_ball_toy``."""
    stripped = model_name
    prefix = f"{room_name}_"
    if stripped.startswith(prefix):
        stripped = stripped[len(prefix) :]
    return re.sub(r"(_f?\d+)+$", "", stripped)


def _classify_model_path(rel_path: str) -> str:
    for kind in ("furniture", "manipuland", "wall_mounted", "ceiling_mounted"):
        if f"/{kind}/" in rel_path:
            return kind
    return "other"


def _parse_scene_objects(
    scene_dir: Path, directives: list[dict], stats: ConversionStats
) -> dict[str, dict[str, list[_PosedObject]]]:
    """Group the scene's free bodies as rooms[room]["furniture"/"manipuland"].

    Welded objects (wall/ceiling mounted — no ``default_free_body_pose``)
    and tilted/no-geometry bodies are skipped + counted.
    """
    rooms: dict[str, dict[str, list[_PosedObject]]] = {}
    for directive in directives:
        model = directive.get("add_model")
        if not isinstance(model, dict) or "file" not in model:
            continue
        name = str(model.get("name", ""))
        if name.startswith("room_geometry"):
            continue
        rel = str(model["file"]).replace("package://scene/", "")
        kind = _classify_model_path(rel)
        if "default_free_body_pose" not in model:
            stats.skip(f"welded_{kind}_object")
            continue
        if kind not in ("furniture", "manipuland"):
            stats.skip(f"free_{kind}_object")
            continue
        pose = next(iter(model["default_free_body_pose"].values()))
        frame_match = re.match(r"room_(.+)_frame$", pose.get("base_frame", ""))
        if frame_match is None:
            stats.skip("object_unexpected_base_frame")
            continue
        room_name = frame_match.group(1)
        yaw, uprightness = _yaw_and_uprightness(pose["rotation"])
        if uprightness < _UPRIGHT_MIN_R22:
            stats.skip(f"{kind}_tilted_pose")
            continue
        dims = _asset_dimensions(scene_dir / rel)
        if dims is None:
            stats.skip("asset_no_collision_geometry")
            continue
        bucket = rooms.setdefault(room_name, {"furniture": [], "manipuland": []})
        bucket[kind].append(
            _PosedObject(
                object_id=(
                    name[len(room_name) + 1 :]
                    if name.startswith(f"{room_name}_")
                    else name
                ),
                category=_category_from_name(name, room_name),
                asset_dir_id=Path(rel).parent.name,
                position=tuple(float(v) for v in pose["translation"]),
                yaw_deg=yaw,
                uprightness=uprightness,
                dimensions=dims,
            )
        )
    return rooms


# ---------------------------------------------------------- support inference


def _find_support_parent(
    manipuland: _PosedObject, furniture: list[_PosedObject]
) -> _PosedObject | None:
    """Smallest-footprint furniture whose (slightly expanded) footprint
    contains the manipuland's xy and whose z-range contains its resting z."""
    x, y, z = manipuland.position
    best: tuple[float, _PosedObject] | None = None
    for item in furniture:
        width, depth, height = item.dimensions
        fx, fy, fz = item.position
        local = world_to_local_2d((x, y), (fx, fy), item.yaw_deg)
        if abs(local[0]) > width / 2.0 + _FOOTPRINT_MARGIN_M:
            continue
        if abs(local[1]) > depth / 2.0 + _FOOTPRINT_MARGIN_M:
            continue
        rest = z - fz
        if rest < -_FLOOR_REST_TOL_M or rest > height + _ABOVE_TOP_TOL_M:
            continue
        area = width * depth
        if best is None or area < best[0]:
            best = (area, item)
    return best[1] if best is not None else None


def _cluster_levels(resting_zs: list[float]) -> list[float]:
    """Merge sorted resting heights into shelf levels (gap <= 4 cm)."""
    levels: list[list[float]] = []
    for z in sorted(resting_zs):
        if levels and z - levels[-1][-1] <= _LEVEL_MERGE_M:
            levels[-1].append(z)
        else:
            levels.append([z])
    return [sum(group) / len(group) for group in levels]


def _build_surface_layer(
    furniture: list[_PosedObject],
    manipulands: list[_PosedObject],
    stats: ConversionStats,
) -> tuple[tuple[SupportSurfaceSpec, ...], tuple[SurfaceObjectGroup, ...]]:
    """Inferred support surfaces + one FREE group per (parent, shelf level)."""
    by_parent: dict[str, list[_PosedObject]] = {}
    parents: dict[str, _PosedObject] = {}
    for manipuland in manipulands:
        parent = _find_support_parent(manipuland, furniture)
        if parent is None:
            on_floor = manipuland.position[2] < _FLOOR_REST_TOL_M
            stats.skip("manipuland_on_floor" if on_floor else "manipuland_unsupported")
            continue
        by_parent.setdefault(parent.object_id, []).append(manipuland)
        parents[parent.object_id] = parent
    surfaces: list[SupportSurfaceSpec] = []
    groups: list[SurfaceObjectGroup] = []
    for parent_id in sorted(by_parent):
        parent = parents[parent_id]
        placed = by_parent[parent_id]
        levels = _cluster_levels([m.position[2] for m in placed])
        members_by_level: dict[int, list[_PosedObject]] = {}
        for member in placed:
            nearest = min(
                range(len(levels)),
                key=lambda i: abs(member.position[2] - levels[i]),
            )
            members_by_level.setdefault(nearest, []).append(member)
        for index, level in enumerate(levels):
            members = members_by_level.get(index, [])
            if not members:  # pragma: no cover - defensive
                continue
            surface, group = _make_surface_group(parent, index, level, members)
            surfaces.append(surface)
            groups.append(group)
    return tuple(surfaces), tuple(groups)


def _make_surface_group(
    parent: _PosedObject,
    index: int,
    level: float,
    members: list[_PosedObject],
) -> tuple[SupportSurfaceSpec, SurfaceObjectGroup]:
    """One shelf level -> (SupportSurfaceSpec, FREE SurfaceObjectGroup)."""
    width, depth, height = parent.dimensions
    surface_id = f"{parent.object_id}_lvl{index}"
    kind = (
        SurfaceKind.TOP
        if abs(level - (parent.position[2] + height)) <= _TOP_KIND_TOL_M
        else SurfaceKind.SHELF
    )
    surface = SupportSurfaceSpec(
        surface_id=surface_id,
        parent_object_id=parent.object_id,
        kind=kind,
        height_m=max(level, 0.0),
        polygon_local=(
            (-width / 2.0, -depth / 2.0),
            (width / 2.0, -depth / 2.0),
            (width / 2.0, depth / 2.0),
            (-width / 2.0, depth / 2.0),
        ),
        capacity_max_objects=max(12, len(members)),
        source="asset",
    )
    objects = tuple(
        SurfaceObjectSpec(
            object_id=member.object_id,
            category=member.category,
            dimensions=member.dimensions,
            position_local=world_to_local_2d(
                (member.position[0], member.position[1]),
                (parent.position[0], parent.position[1]),
                parent.yaw_deg,
            ),
            z_local=member.position[2] - level,
            yaw_deg_local=normalize_deg(member.yaw_deg - parent.yaw_deg),
        )
        for member in members
    )
    group = SurfaceObjectGroup(
        group_id=f"{surface_id}_group", surface_id=surface_id, objects=objects
    )
    return surface, group


# -------------------------------------------------------------- sample build


def _build_sample(
    subset: str,
    scene_id: str,
    room_name: str,
    shell: tuple[tuple[Vec2, ...], float | None],
    objects: dict[str, list[_PosedObject]],
    stats: ConversionStats,
) -> FastFillSample:
    polygon_raw, ceiling = shell
    polygon, offset = recenter_polygon(polygon_raw)

    def _recentered(obj: _PosedObject) -> _PosedObject:
        return replace(
            obj,
            position=(
                obj.position[0] - offset[0],
                obj.position[1] - offset[1],
                obj.position[2],
            ),
        )

    furniture = [_recentered(obj) for obj in objects["furniture"]]
    manipulands = [_recentered(obj) for obj in objects["manipuland"]]
    floor_objects = tuple(
        FloorObjectSpec(
            object_id=obj.object_id,
            category=obj.category,
            dimensions=obj.dimensions,
            position_xy=(obj.position[0], obj.position[1]),
            z=0.0,
            yaw_deg=obj.yaw_deg,
        )
        for obj in furniture
    )
    surfaces, groups = _build_surface_layer(furniture, manipulands, stats)
    room_id = f"{subset.lower()}_{scene_id}_{room_name}"
    notes = f"subset={subset};support=inferred_geometric;dims=collision_mesh_aabb"
    if ceiling is None:
        notes += ";ceiling=default"
    if subset == "NotGenerated":
        notes += ";assets=hssd_retrieved_sha1_ids"
    provenance = ProvenanceMeta(
        source_dataset=SOURCE_DATASET,
        source_house_id=scene_id,
        source_room_id=room_name,
        source_asset_ids=tuple(
            sorted({o.asset_dir_id for o in furniture + manipulands})
        ),
        upstream_dataset="SceneSmith",
        license_tag=(
            LicenseTag.CC_BY_NC if subset == "NotGenerated" else LicenseTag.PERMISSIVE
        ),
        notes=notes,
    )
    context = RoomContext(
        room_id=room_id,
        room_type=room_name,
        floor_polygon=polygon,
        ceiling_height_m=ceiling if ceiling is not None else 2.5,
    )
    layout = RoomContentLayout(
        room_id=room_id,
        floor_layout=FloorLayout(
            room_id=room_id, objects=floor_objects, support_surfaces=surfaces
        ),
        surface_groups=groups,
    )
    return FastFillSample(
        sample_id=f"{SOURCE_DATASET}/{subset}/{scene_id}/{room_name}",
        room_context=context,
        layout=layout,
        provenance=provenance,
    )


# ---------------------------------------------------------------- conversion


def convert(
    data_dir: Path, limit: int | None, stats: ConversionStats
) -> Iterator[FastFillSample]:
    """Yield one FastFillSample per room across all subsets (deterministic
    order: subset, scene, room). Malformed pieces are skipped + counted."""
    root = data_dir / SOURCE_DATASET
    emitted = 0
    for subset in SUBSETS:
        for scene_dir in sorted((root / subset).glob("scene_*")):
            directive_path = scene_dir / "combined_house" / "house.dmd.yaml"
            if not directive_path.exists():
                stats.skip("scene_missing_directive")
                continue
            try:
                doc = yaml.load(directive_path.read_text(), Loader=_DmdLoader)
                directives = doc["directives"]
            except (yaml.YAMLError, KeyError, TypeError):
                stats.skip("scene_unparseable_directive")
                continue
            rooms = _parse_scene_objects(scene_dir, directives, stats)
            for room_name in sorted(rooms):
                if limit is not None and emitted >= limit:
                    return
                shell = _parse_room_shell(
                    scene_dir / "room_geometry" / f"room_geometry_{room_name}.sdf"
                )
                if shell is None:
                    stats.skip("room_missing_floor_geometry")
                    continue
                if not rooms[room_name]["furniture"]:
                    stats.skip("room_no_furniture")
                    continue
                yield _build_sample(
                    subset,
                    scene_dir.name,
                    room_name,
                    shell,
                    rooms[room_name],
                    stats,
                )
                emitted += 1


def main() -> None:
    converter_main(
        description=__doc__.split("\n", 1)[0],
        convert=convert,
        default_out="out/scenesmith_scenes.jsonl",
    )


if __name__ == "__main__":
    main()
