"""Convert IL3D layouts to FastFill JSONL (floor-standing furniture only).

Source: ``<data>/IL3D/layout/*.json`` — one scene per uuid-named file with
``objects`` (Y-up placements), ``meshes`` (triangulated floor) and a
``dataset`` field ("3D-FRONT" | "Synthetic Data" | "HSSD").

Format facts VERIFIED on the full 27,816-file corpus (2026-07-10):

- Exactly one roomId per file (0 multi-room files) and exactly one
  4-vertex rectangular floor mesh per room. Floor ``xyz`` vertices are in
  grid order, NOT boundary order, so the boundary loop is reconstructed
  from the triangle ``faces`` (grid order would shoelace to a bowtie).
- ``rotation`` is Euler ``[x, y, z]`` degrees. 54.5% of objects are the
  identity pattern (rx≈rz≈0, yaw = ry); 43.7% are the gimbal pattern
  (rx≈rz≈±180), which is algebraically a pure yaw of ``180 - ry``; the
  remaining ~1.8% are genuinely tilted and are skipped.
- ``bbox`` holds PRE-scale canonical asset extents (instances of one
  assetId share bbox values regardless of ``scale``). Axis order depends
  on the asset source (``object_path`` prefix): HSSD assets are always
  ``[x, y_up, z]``; 3D-FRONT assets are ``[x, z, y_up]`` when scale==1
  and ``[x, y_up, z]`` when scale!=1 (31,272 of 33,891 shared-asset
  pairs follow this swap; the ~8% residual is accepted noise). World
  extents are ``bbox * scale`` component-wise in matching Y-up axes.
- ~20% of objects sit above the floor (pendant lights, wall cabinets,
  on-surface manipulands) with no parent link — skipped and counted as
  ``elevated_no_parent`` per the WP1 converter contract.

Skip reasons counted in :class:`ConversionStats`: ``bad_json``,
``no_floor_mesh``, ``multiple_floor_meshes``, ``floor_boundary_failed``,
``degenerate_floor``, ``empty_room_after_filtering``, ``malformed_object``,
``missing_category``, ``degenerate_bbox``, ``tilted_rotation``,
``elevated_no_parent``, ``below_floor``.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional, Sequence

if __package__ in (None, ""):  # direct script execution: add import roots
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    sys.path.insert(0, str(_REPO_ROOT))

from fastfill_data.common import ConversionStats, converter_main
from scenesmith.growing_world.fastfill.schema import (
    FastFillSample,
    FloorLayout,
    FloorObjectSpec,
    LicenseTag,
    ProvenanceMeta,
    RoomContentLayout,
    RoomContext,
    Vec2,
    Vec3,
)
from scenesmith.growing_world.fastfill.transforms import (
    ensure_ccw,
    normalize_deg,
    polygon_area,
    recenter_polygon,
    yup_extents_to_dimensions,
    yup_to_zup_ground_point,
    yup_to_zup_point,
)

LAYOUT_SUBDIR = "IL3D/layout"

_ROT_TOL_DEG = 1.0  # tolerance for identity / gimbal pattern matching
_SCALE_ONE_TOL = 1e-5  # scale component "== 1" tolerance
_MIN_EXTENT_M = 1e-4  # reject degenerate bbox extents
_ELEVATION_EPS_M = 0.05  # z above this => on-surface candidate, skipped
_MIN_FLOOR_AREA_M2 = 1e-3
_WELD_DECIMALS = 6

_TRAILING_IDX_RE = re.compile(r"-\d+$")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _ang_close(angle: float, target: float) -> bool:
    return abs(normalize_deg(angle - target)) < _ROT_TOL_DEG


def effective_yaw_deg(rotation: Sequence[float]) -> Optional[float]:
    """Net yaw about +Y for IL3D Euler ``[x, y, z]`` degrees.

    ``(0, ry, 0)`` is a pure yaw ``ry``; ``(±180, ry, ±180)`` is the gimbal
    form of a pure yaw ``180 - ry`` (Rx(180)·Ry(ry)·Rz(180) == Ry(180-ry),
    identical under XYZ and ZYX orders). Anything else is a real tilt and
    returns ``None`` — the yaw-only contract cannot represent it.
    """
    rx, ry, rz = (float(v) for v in rotation)
    if _ang_close(rx, 0.0) and _ang_close(rz, 0.0):
        return normalize_deg(ry)
    if _ang_close(rx, 180.0) and _ang_close(rz, 180.0):
        return normalize_deg(180.0 - ry)
    return None


def extents_yup(
    bbox: Sequence[float], scale: Sequence[float], object_path: str
) -> Optional[Vec3]:
    """Scaled Y-up full extents ``(ex, ey_up, ez)`` from an IL3D object.

    Applies the verified source-dependent bbox axis order (module
    docstring) and multiplies by the instance scale. ``None`` for
    degenerate extents.
    """
    b0, b1, b2 = (float(v) for v in bbox)
    sx, sy, sz = (float(v) for v in scale)
    unit_scale = all(abs(s - 1.0) < _SCALE_ONE_TOL for s in (sx, sy, sz))
    if object_path.startswith("3D-FRONT") and unit_scale:
        ex, ey, ez = b0, b2, b1  # 3D-FRONT @ scale==1: bbox = [x, z, y_up]
    else:
        ex, ey, ez = b0, b1, b2  # bbox = [x, y_up, z]
    ex, ey, ez = ex * sx, ey * sy, ez * sz
    if min(ex, ey, ez) <= _MIN_EXTENT_M:
        return None
    return (ex, ey, ez)


def floor_boundary_ground(mesh: dict) -> Optional[tuple[Vec2, ...]]:
    """Boundary loop of a floor mesh as Y-up ground points ``(x, z)``.

    Welds duplicate vertices, collects edges used by exactly one triangle,
    and chains them into a single closed loop. ``None`` when the boundary
    is not one simple loop.
    """
    xyz = mesh.get("xyz") or []
    faces = mesh.get("faces") or []
    if len(xyz) < 3 or not faces:
        return None
    keys: list[Vec2] = []
    index_of: dict[Vec2, int] = {}
    remap: list[int] = []
    for v in xyz:
        key = (round(float(v[0]), _WELD_DECIMALS), round(float(v[2]), _WELD_DECIMALS))
        if key not in index_of:
            index_of[key] = len(keys)
            keys.append(key)
        remap.append(index_of[key])
    edge_count: Counter[tuple[int, int]] = Counter()
    for face in faces:
        idx = [remap[i] for i in face]
        for a, b in zip(idx, idx[1:] + idx[:1]):
            if a != b:
                edge_count[(min(a, b), max(a, b))] += 1
    boundary = [edge for edge, count in edge_count.items() if count == 1]
    return _chain_loop(boundary, keys)


def _chain_loop(
    boundary: list[tuple[int, int]], keys: list[Vec2]
) -> Optional[tuple[Vec2, ...]]:
    """Chain undirected boundary edges into one closed vertex loop."""
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if len(adjacency) < 3 or any(len(n) != 2 for n in adjacency.values()):
        return None
    start = boundary[0][0]
    loop = [start]
    prev, current = -1, start
    while True:
        nxt = [n for n in adjacency[current] if n != prev]
        if not nxt:
            return None
        prev, current = current, nxt[0]
        if current == start:
            break
        if len(loop) > len(adjacency):
            return None  # revisited a vertex without closing
        loop.append(current)
    if len(loop) != len(adjacency):
        return None  # more than one loop (hole / disconnected floor)
    return tuple(keys[i] for i in loop)


def _room_type_from_id(room_id: str) -> str:
    """``"LivingDiningRoom"`` -> ``"living dining room"``."""
    return _CAMEL_SPLIT_RE.sub(" ", room_id).lower()


def _unique_object_id(name: str, used: set[str]) -> str:
    object_id = name
    n = 2
    while object_id in used:
        object_id = f"{name}__{n}"
        n += 1
    used.add(object_id)
    return object_id


def _convert_object(
    obj: dict, offset: Vec2, used_ids: set[str], stats: ConversionStats
) -> Optional[FloorObjectSpec]:
    """One IL3D object -> FloorObjectSpec, or None (skip counted)."""
    try:
        name = str(obj["object_name"])
        position = obj["position"]
        rotation = obj["rotation"]
        bbox = obj["bbox"]
        scale = obj.get("scale") or (1.0, 1.0, 1.0)
        object_path = str(obj.get("object_path") or "")
        x, y, z = yup_to_zup_point(position)
    except (KeyError, TypeError, ValueError, IndexError):
        stats.skip("malformed_object")
        return None
    category = obj.get("label") or obj.get("category")
    if not category:
        stats.skip("missing_category")
        return None
    if z > _ELEVATION_EPS_M:
        stats.skip("elevated_no_parent")
        return None
    if z < -_ELEVATION_EPS_M:
        stats.skip("below_floor")
        return None
    yaw = effective_yaw_deg(rotation)
    if yaw is None:
        stats.skip("tilted_rotation")
        return None
    extents = extents_yup(bbox, scale, object_path)
    if extents is None:
        stats.skip("degenerate_bbox")
        return None
    return FloorObjectSpec(
        object_id=_unique_object_id(name, used_ids),
        category=str(category),
        asset_query=_TRAILING_IDX_RE.sub("", name),
        dimensions=yup_extents_to_dimensions(extents),
        position_xy=(x - offset[0], y - offset[1]),
        z=z,
        # +180: IL3D's yaw-0 front is -Y in the contract frame. Measured on
        # wall-backed categories over 300 real rooms: raw carry-through faces
        # the room only 8% of the time; with the flip ~92%. See fastfill
        # README facing contract.
        yaw_deg=normalize_deg(yaw + 180.0),
    )


def _room_floor_polygon(
    floor_meshes: list[dict], stats: ConversionStats
) -> Optional[tuple[tuple[Vec2, ...], Vec2]]:
    """Recentered CCW floor polygon + subtracted offset, or None (counted)."""
    if not floor_meshes:
        stats.skip("no_floor_mesh")
        return None
    if len(floor_meshes) > 1:
        stats.skip("multiple_floor_meshes")
        return None
    ground = floor_boundary_ground(floor_meshes[0])
    if ground is None:
        stats.skip("floor_boundary_failed")
        return None
    polygon = ensure_ccw([yup_to_zup_ground_point(p) for p in ground])
    if polygon_area(polygon) < _MIN_FLOOR_AREA_M2:
        stats.skip("degenerate_floor")
        return None
    return recenter_polygon(polygon)


def _room_provenance(
    stem: str, room_id: str, dataset_field: str, room_objects: list[dict]
) -> ProvenanceMeta:
    has_hssd = any(
        str(o.get("object_path") or "").startswith("HSSD") for o in room_objects
    )
    license_tag = LicenseTag.CC_BY_NC if has_hssd else LicenseTag.PERMISSIVE
    notes = "upstream_tou=3D-FRONT" if dataset_field == "3D-FRONT" else ""
    return ProvenanceMeta(
        source_dataset="il3d",
        source_house_id=stem,
        source_room_id=room_id,
        upstream_dataset=dataset_field,
        license_tag=license_tag,
        notes=notes,
    )


def _convert_room(
    stem: str,
    room_id: str,
    room_objects: list[dict],
    floor_meshes: list[dict],
    dataset_field: str,
    stats: ConversionStats,
) -> Optional[FastFillSample]:
    recentered = _room_floor_polygon(floor_meshes, stats)
    if recentered is None:
        return None
    polygon, offset = recentered
    used_ids: set[str] = set()
    specs: list[FloorObjectSpec] = []
    converted_asset_ids: list[str] = []
    for obj in room_objects:
        spec = _convert_object(obj, offset, used_ids, stats)
        if spec is None:
            continue
        specs.append(spec)
        if obj.get("assetId"):
            converted_asset_ids.append(str(obj["assetId"]))
    if not specs:
        stats.skip("empty_room_after_filtering")
        return None
    asset_ids = tuple(dict.fromkeys(converted_asset_ids))
    provenance = _room_provenance(
        stem, room_id, dataset_field, room_objects
    ).model_copy(update={"source_asset_ids": asset_ids})
    room_context = RoomContext(
        room_id=room_id,
        room_type=_room_type_from_id(room_id),
        floor_polygon=polygon,
    )
    layout = RoomContentLayout(
        room_id=room_id,
        floor_layout=FloorLayout(room_id=room_id, objects=tuple(specs)),
    )
    return FastFillSample(
        sample_id=f"il3d/{stem}/{room_id}",
        room_context=room_context,
        layout=layout,
        provenance=provenance,
    )


def iter_record_samples(
    stem: str, record: dict, stats: ConversionStats
) -> Iterator[FastFillSample]:
    """One parsed layout file -> one FastFillSample per roomId.

    The full corpus has exactly one roomId per file, but grouping by
    roomId keeps multi-room records correct if they ever appear.
    """
    objects = record.get("objects") or []
    meshes = record.get("meshes") or []
    dataset_field = str(record.get("dataset") or "")
    room_ids = sorted({str(o.get("roomId") or "") for o in objects})
    for room_id in room_ids:
        room_objects = [o for o in objects if str(o.get("roomId") or "") == room_id]
        floor_meshes = [
            m
            for m in meshes
            if m.get("type") == "floor" and str(m.get("roomId") or "") == room_id
        ]
        sample = _convert_room(
            stem, room_id, room_objects, floor_meshes, dataset_field, stats
        )
        if sample is not None:
            yield sample


def convert(
    data_dir: Path, limit: Optional[int], stats: ConversionStats
) -> Iterator[FastFillSample]:
    """Yield FastFillSamples from ``<data_dir>/IL3D/layout/*.json``."""
    layout_dir = data_dir / LAYOUT_SUBDIR
    emitted = 0
    for path in sorted(layout_dir.glob("*.json")):
        if limit is not None and emitted >= limit:
            return
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats.skip("bad_json")
            continue
        for sample in iter_record_samples(path.stem, record, stats):
            yield sample
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def main() -> None:
    converter_main(
        description="Convert IL3D layouts to FastFill JSONL",
        convert=convert,
        default_out="out/il3d.jsonl",
    )


if __name__ == "__main__":
    main()
