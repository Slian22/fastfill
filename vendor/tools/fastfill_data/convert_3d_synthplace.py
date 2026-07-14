"""Convert 3D-SynthPlace scenes to FastFill JSONL (floor-furniture layer only).

Source layout (verified on the real download, 17,528 files):
``<data>/3D-SynthPlace_indoor_scenes_dataset/3D-SynthPlace_Scenes_Assets/
scenes/<roomtype>-<N>.json`` with ``floor`` (Y-up ground-plane ``[x, z]``
vertices, meters), ``room`` (type string), ``objects`` (Y-up positions in
meters, ``bbox`` in base-100 units -> divide by 100 for meters, ``rotation``
in degrees) and a top-level ``source`` tag (``3DFRONT`` | ``HOLODECK``,
missing on 48 files — those are all-UUID, i.e. 3D-FRONT).

Caveats carried into ``provenance.notes`` / this docstring:

- FACING (per-source calibration): the source yaw convention vs our front=+Y
  contract differs BY UPSTREAM DATASET, so it is calibrated per source, not
  carried uniformly. Holodeck-Synth assets already face +Y after the Y-up->
  Z-up map, so ``yaw = normalize(rotation.y)``. 3D-FRONT assets face -Y, so
  ``yaw = normalize(rotation.y + 180)`` (sign preserved — NOT negated; the
  yup_to_zup map has determinant +1). Files with no ``source`` tag are all
  3D-FRONT (all-UUID asset ids) and take the +180 branch. Calibration is
  backed by a server-side full-corpus audit (17,528 scenes) and MUST be
  reconfirmed there before training; this converter no longer emits the
  generic ``yaw_facing=unverified_per_asset`` gate token for either subset.
- BBOX ORDER: the frozen contract maps ``bbox`` as ``[height, width, depth]``
  (``transforms.hwd_cm_to_dimensions``). This holds for the HOLODECK subset.
  3D-FRONT ships a MIX of axis orders (~73% HWD, ~27% DWH per mesh-vertex
  audit), so the uniform mapping is wrong for a large minority. Until the
  per-record permutation is resolved from ``bbox_vertices.npy`` on the
  server, 3D-FRONT records carry the ``bbox_axis=unverified_3dfront`` note
  so ``export_sft`` ISOLATES them (kept for dedup + contamination closure,
  never exported as training labels). Holodeck records export normally.
- No house ids exist in the source, so ``split_key`` is room-level.
- Elevated objects (|z_zup| > 0.10 m — 28 of 147,222 corpus-wide) are not
  floor layer and are skipped + counted; kept objects are floor-standing so
  ``z`` is emitted as 0.0.
- 25 corpus files contain ``NaN`` positions/rotations (e.g.
  ``living_room-2204.json``); those records fail finite-number validation
  and are skipped + counted (``invalid_position`` / ``invalid_rotation``).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Iterator, Sequence

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
)
from scenesmith.growing_world.fastfill.transforms import (
    Vec2,
    ensure_ccw,
    hwd_cm_to_dimensions,
    normalize_deg,
    polygon_area,
    recenter_polygon,
    yup_to_zup_ground_point,
    yup_to_zup_point,
)

SOURCE_DATASET = "3d_synthplace"
SCENES_SUBDIR = Path(
    "3D-SynthPlace_indoor_scenes_dataset/3D-SynthPlace_Scenes_Assets/scenes"
)
ELEVATED_Z_MAX_M = 0.10  # above this the object is not floor layer
_MIN_FLOOR_AREA_M2 = 1e-6

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}" r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_TRAILING_INDEX_RE = re.compile(r"-\d+$")
_SOURCE_TO_UPSTREAM = {"3DFRONT": "3D-FRONT", "HOLODECK": "Holodeck-Synth"}
_UPSTREAM_3DFRONT = "3D-FRONT"
_UPSTREAM_HOLODECK = "Holodeck-Synth"

#: Per-source yaw calibration to the front=+Y contract (degrees, added before
#: normalize). Holodeck faces +Y already (0); 3D-FRONT faces -Y (+180). Sign
#: is preserved — the yup_to_zup map has determinant +1. Server-audited.
_YAW_OFFSET_DEG = {_UPSTREAM_HOLODECK: 0.0, _UPSTREAM_3DFRONT: 180.0}

#: Isolation token for the 3D-FRONT subset: bbox axis order is a per-record
#: mix (HWD/DWH) not resolvable without mesh vertices, so export_sft keeps
#: these for dedup + contamination closure but never emits them as labels.
BBOX_UNVERIFIED_3DFRONT_NOTE = "bbox_axis=unverified_3dfront"

_NOTES_COMMON = "no house id in source (split is room-level)"
_NOTES_HOLODECK = (
    f"{_NOTES_COMMON}; yaw calibrated per source (holodeck: raw, front=+Y "
    "verified by server audit); bbox mapped [h,w,d] per contract (holodeck "
    "subset verified)"
)
_NOTES_3DFRONT = (
    f"{_NOTES_COMMON}; yaw calibrated per source (3d-front: raw+180, front=+Y "
    f"per server audit); {BBOX_UNVERIFIED_3DFRONT_NOTE} (axis order is a "
    "per-record HWD/DWH mix — isolated from export pending mesh-vertex "
    "resolution)"
)


def _notes_for(upstream: str) -> str:
    return _NOTES_3DFRONT if upstream == _UPSTREAM_3DFRONT else _NOTES_HOLODECK


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _parse_floor(raw: object) -> tuple[tuple[Vec2, ...], Vec2] | None:
    """Y-up ``[x, z]`` vertices -> (recentered CCW Z-up polygon, offset)."""
    if not isinstance(raw, list) or len(raw) < 3:
        return None
    for point in raw:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(_is_finite_number(v) for v in point)
        ):
            return None
    vertices = ensure_ccw([yup_to_zup_ground_point(p) for p in raw])
    if abs(polygon_area(vertices)) < _MIN_FLOOR_AREA_M2:
        return None
    return recenter_polygon(vertices)


def _object_skip_reason(raw: object) -> str | None:
    """Validate one raw object record; return a skip reason or None."""
    if not isinstance(raw, dict):
        return "object_not_a_dict"
    name = raw.get("object_name")
    if not isinstance(name, str) or not name:
        return "missing_object_name"
    position = raw.get("position")
    if not isinstance(position, dict) or not all(
        _is_finite_number(position.get(k)) for k in ("x", "y", "z")
    ):
        return "invalid_position"
    bbox = raw.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 3
        or not all(_is_finite_number(v) and v > 0 for v in bbox)
    ):
        return "invalid_bbox"
    rotation = raw.get("rotation")
    if not isinstance(rotation, dict) or not _is_finite_number(rotation.get("y", 0.0)):
        return "invalid_rotation"
    return None


def _parse_object(
    raw: object, offset: Vec2, yaw_offset_deg: float, stats: ConversionStats
) -> FloorObjectSpec | None:
    """One source object -> FloorObjectSpec in the recentered room frame.

    ``yaw_offset_deg`` is the per-source facing calibration (see
    ``_YAW_OFFSET_DEG``): added to the raw yaw before normalization so the
    object faces the front=+Y contract. Sign is preserved (never negated)."""
    reason = _object_skip_reason(raw)
    if reason is not None:
        stats.skip(reason)
        return None
    assert isinstance(raw, dict)  # narrowed by _object_skip_reason
    position = raw["position"]
    x, y, z = yup_to_zup_point((position["x"], position["y"], position["z"]))
    if abs(z) > ELEVATED_Z_MAX_M:
        stats.skip("elevated_object")
        return None
    name: str = raw["object_name"]
    description = raw.get("description")
    return FloorObjectSpec(
        object_id=name,
        category=_TRAILING_INDEX_RE.sub("", name),
        asset_query=description if isinstance(description, str) else "",
        dimensions=hwd_cm_to_dimensions(raw["bbox"]),
        position_xy=(x - offset[0], y - offset[1]),
        z=0.0,
        yaw_deg=normalize_deg(float(raw["rotation"].get("y", 0.0)) + yaw_offset_deg),
    )


def _collect_asset_ids(raw_objects: Sequence[object]) -> list[str]:
    """Unique ``assetId`` strings, in first-seen order (for upstream vote)."""
    asset_ids: list[str] = []
    for raw in raw_objects:
        asset_id = raw.get("assetId") if isinstance(raw, dict) else None
        if isinstance(asset_id, str) and asset_id and asset_id not in asset_ids:
            asset_ids.append(asset_id)
    return asset_ids


def _upstream_dataset(data: dict, asset_ids: Sequence[str]) -> str:
    """Prefer the file's ``source`` tag; fall back to the UUID heuristic
    (3D-FRONT asset ids are 8-4-4-4-12 hex UUIDs, Holodeck ids are not)."""
    source = data.get("source")
    if isinstance(source, str):
        mapped = _SOURCE_TO_UPSTREAM.get(source.strip().upper())
        if mapped is not None:
            return mapped
    uuid_votes = sum(1 for a in asset_ids if _UUID_RE.match(a))
    if asset_ids and uuid_votes * 2 >= len(asset_ids):
        return "3D-FRONT"
    return "Holodeck-Synth"


def _convert_scene(path: Path, stats: ConversionStats) -> FastFillSample | None:
    """One scene file -> FastFillSample (task/expected_* left empty)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stats.skip("unreadable_json")
        return None
    if not isinstance(data, dict):
        stats.skip("scene_not_a_dict")
        return None
    floor = _parse_floor(data.get("floor"))
    if floor is None:
        stats.skip("invalid_floor")
        return None
    polygon, offset = floor
    raw_objects = data.get("objects")
    if not isinstance(raw_objects, list):
        stats.skip("missing_objects")
        return None
    # Upstream must be resolved BEFORE parsing objects: it selects the yaw
    # calibration. The source tag decides it; the UUID vote is the fallback,
    # so collect asset ids up front.
    asset_ids = _collect_asset_ids(raw_objects)
    upstream = _upstream_dataset(data, asset_ids)
    yaw_offset = _YAW_OFFSET_DEG.get(upstream, 0.0)
    objects: list[FloorObjectSpec] = []
    for raw in raw_objects:
        spec = _parse_object(raw, offset, yaw_offset, stats)
        if spec is None:
            continue
        objects.append(spec)
    if not objects:
        stats.skip("no_valid_objects")
        return None
    stem = path.stem
    room = data.get("room")
    provenance = ProvenanceMeta(
        source_dataset=SOURCE_DATASET,
        source_house_id="",  # unknown in source -> room-level split key
        source_room_id=stem,
        source_asset_ids=tuple(asset_ids),
        upstream_dataset=upstream,
        license_tag=LicenseTag.LICENSE_PENDING,
        notes=_notes_for(upstream),
    )
    return FastFillSample(
        sample_id=f"{SOURCE_DATASET}/{stem}",
        room_context=RoomContext(
            room_id=stem,
            room_type=room if isinstance(room, str) else "",
            floor_polygon=polygon,
        ),
        layout=RoomContentLayout(
            room_id=stem,
            floor_layout=FloorLayout(room_id=stem, objects=tuple(objects)),
        ),
        provenance=provenance,
    )


def convert(
    data_dir: Path, limit: int | None, stats: ConversionStats
) -> Iterator[FastFillSample]:
    """Yield FastFillSamples from the real 3D-SynthPlace download."""
    # HF downloads land in two layouts: with or without the
    # 3D-SynthPlace_Scenes_Assets/ intermediate dir — probe both.
    candidates = (
        data_dir / SCENES_SUBDIR,
        data_dir / "3D-SynthPlace_indoor_scenes_dataset" / "scenes",
    )
    scenes_dir = next((c for c in candidates if c.is_dir()), None)
    if scenes_dir is None:
        raise FileNotFoundError(
            f"3D-SynthPlace scenes dir not found; tried: {[str(c) for c in candidates]}"
        )
    converted = 0
    for path in sorted(scenes_dir.glob("*.json")):
        if limit is not None and converted >= limit:
            break
        sample = _convert_scene(path, stats)
        if sample is None:
            continue
        converted += 1
        yield sample


def main() -> None:
    converter_main(
        "Convert 3D-SynthPlace scenes to FastFillSample JSONL.",
        convert,
        "out/3d_synthplace.jsonl",
    )


if __name__ == "__main__":
    main()
