"""Convert M3DLayout (3D-FRONT / Matterport3D / Infinigen) to FastFill JSONL.

Source containers live at
``<data>/M3DLayout/layout_dataset/M3DLayout_json/<split>_<subset>*.json`` with
shape ``{"source": str, "scenes": [{"scene_id", "objects": [{"category",
"location", "size", "rotation"}], "description": {...}}]}``.

Verified source semantics (hard-asserted in
``tests/unit/fastfill_data/test_convert_m3dlayout.py``):

- ``location`` is the object CENTER in a Y-up right-handed frame;
- ``size`` is HALF-extents ``[sx, sy_up, sz]`` — under this reading floor
  furniture bottoms (``location.y - size.y``) sit at 0 for ~82% of 3D-FRONT
  objects (the remainder are ceiling/pendant lamps) and median category
  heights are plausible (wardrobe 2.33 m, nightstand 0.55 m, dining table
  0.75 m); the full-extent reading puts every bottom mid-air and halves all
  heights (nightstand 0.28 m);
- ``rotation`` is a scalar yaw in radians about +Y. The contract yaw adds
  180°: M3DLayout's yaw-0 front is -Y in the contract frame (measured: raw
  carry-through makes wall-backed furniture face the room only ~5% of the
  time over 400 real 3dfront rooms; +180° flips that to ~95%).

M3DLayout carries NO floor polygon, so an axis-aligned rectangle covering all
converted object footprints plus ``FLOOR_MARGIN_M`` per side is synthesized
and recentered (``provenance.notes = "floor=synthesized_bbox"``). Ceiling /
elevated objects and Infinigen architecture + small manipulands are skipped
and counted per bucket (the manipuland counts are an audit datum for T2.2).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Script-mode bootstrap: make `fastfill_data` (tools/) and `scenesmith`
# (repo root, not pip-installed) importable when run as a plain script.
for _extra in (
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2],
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from fastfill_data.common import (  # noqa: E402
    ConversionStats,
    atomic_write_jsonl,
    finalize_sample,
    resolve_data_dir,
)
from scenesmith.growing_world.fastfill.schema import (  # noqa: E402
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
from scenesmith.growing_world.fastfill.transforms import (  # noqa: E402
    footprint_corners,
    normalize_deg,
    rad_to_yaw_deg,
    recenter_polygon,
    translate_points,
    yup_half_extents_to_dimensions,
    yup_to_zup_point,
)

JSON_SUBDIR = Path("M3DLayout") / "layout_dataset" / "M3DLayout_json"
SPLITS = ("3dfront", "mp3d", "infinigen")
SUBSETS = ("train", "val", "test")

SOURCE_DATASET = "m3dlayout"
UPSTREAM_BY_SOURCE = {
    "3DFront": "3D-FRONT",
    "Matterport3D": "Matterport3D",
    "Infinigen": "Infinigen",
}
LICENSE_BY_SOURCE = {
    # The HuggingFace dataset card licenses the whole M3DLayout package
    # CC-BY-NC-4.0; sub-source upstreams (e.g. Infinigen) may be permissive,
    # but the redistribution being converted here is NC. Research-only.
    "3DFront": LicenseTag.CC_BY_NC,
    "Matterport3D": LicenseTag.CC_BY_NC,
    "Infinigen": LicenseTag.CC_BY_NC,
}

FLOOR_MARGIN_M = 0.4  # synthesized floor rectangle margin per side
# Floor z contract: the Floor codec does not encode z, so any object kept
# with bottom above the snap band would silently train as floor-standing.
# Bottoms beyond FLOOR_SNAP_M are skipped as "elevated" instead.
FLOOR_SNAP_M = 0.02  # bottoms within this of z=0 snap to floor-standing

CEILING_CATEGORIES = frozenset({"ceiling_lamp", "pendant_lamp"})
INFINIGEN_ARCHITECTURE = frozenset(
    {
        "glass_panel_door",
        "lite_door",
        "louver_door",
        "panel_door",
        "window",
        "rug",
        "wall_art",
        "kitchen_space",
    }
)
INFINIGEN_MANIPULANDS = frozenset(
    {
        "plate",
        "cup",
        "fork",
        "spoon",
        "knife",
        "chopsticks",
        "bottle",
        "can",
        "jar",
        "bowl",
        "wineglass",
    }
)  # plus any "food_*" category (matched by prefix)

# 3D-FRONT room-name segment (lowercased, digits stripped) -> room_type.
# The six keys below are the complete vocabulary observed across
# 3dfront_{train,val,test}; unknown names fall through as-is.
THREEDFRONT_ROOM_TYPES = {
    "bedroom": "bedroom",
    "masterbedroom": "bedroom",
    "secondbedroom": "bedroom",
    "livingroom": "living_room",
    "diningroom": "dining_room",
    "livingdiningroom": "living_dining_room",
}

# "The room is (likely) a <noun phrase> ..." -> "<noun phrase>"
_ROOM_TYPE_SENTENCE_RE = re.compile(r"\bis (?:likely )?(?:a|an) ([a-z][a-z -]*)")
_ROOM_TYPE_CUT_RE = re.compile(
    r"\s+(?:that|which|with|and|used|where|featuring|containing"
    r"|connect\w*|includ\w*)\b"
)


@dataclass(frozen=True)
class _RawObject:
    """One source object after coordinate conversion, before recentering."""

    category: str
    position_xy: Vec2  # pre-recenter room frame (Z-up)
    dimensions: Vec3  # full [width, depth, height] extents, meters
    yaw_deg: float
    bottom_z: float  # object bottom above the floor plane


# ------------------------------------------------------------------- parsing


def _is_finite_vec3(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)
    )


def _parse_object(raw: object) -> _RawObject | None:
    """Convert one source object record; ``None`` when malformed."""
    if not isinstance(raw, dict):
        return None
    category = raw.get("category")
    location, size, rotation = (
        raw.get("location"),
        raw.get("size"),
        raw.get("rotation"),
    )
    if not isinstance(category, str) or not category:
        return None
    if not (_is_finite_vec3(location) and _is_finite_vec3(size)):
        return None
    if not isinstance(rotation, (int, float)) or not math.isfinite(rotation):
        return None
    if any(v <= 0.0 for v in size):  # type: ignore[union-attr]
        return None
    x, y, z_center = yup_to_zup_point(location)  # type: ignore[arg-type]
    dims = yup_half_extents_to_dimensions(size)  # type: ignore[arg-type]
    return _RawObject(
        category=category,
        position_xy=(x, y),
        dimensions=dims,
        # +180: M3DLayout's yaw-0 front is -Y in the contract frame. Measured
        # on wall-backed categories (wardrobe/cabinet/shelf/...) over 400 real
        # 3dfront rooms: raw carry-through faces the room only 5% of the time;
        # with the flip it is ~95%. See fastfill README facing contract.
        yaw_deg=normalize_deg(rad_to_yaw_deg(float(rotation)) + 180.0),
        bottom_z=z_center - dims[2] / 2.0,
    )


def _skip_bucket(obj: _RawObject, split: str) -> str | None:
    """Skip-bucket name for a non-floor object, or ``None`` to convert it."""
    if split == "infinigen":
        if obj.category in INFINIGEN_ARCHITECTURE:
            return f"architecture:{obj.category}"
        if obj.category in INFINIGEN_MANIPULANDS or obj.category.startswith("food_"):
            return f"manipuland:{obj.category}"
    if obj.category in CEILING_CATEGORIES:
        return "elevated"
    if obj.bottom_z > FLOOR_SNAP_M:
        return "elevated"
    return None


# ---------------------------------------------------------- room-type / ids


def _threedfront_room_type(scene_id: str) -> str:
    """``<uuid>_SecondBedroom-2658`` -> ``bedroom`` (see map above)."""
    if "_" not in scene_id:
        return "unknown"
    segment = scene_id.split("_", 1)[1]
    name = re.sub(r"\d+", "", re.sub(r"[-\d]+$", "", segment)).lower()
    return THREEDFRONT_ROOM_TYPES.get(name, name or "unknown")


def _room_type_from_description(description: object) -> str:
    """Extract a room type from templated description sentences.

    ``"The room is a bedroom."`` -> ``"bedroom"``; sentences without an
    ``is a <noun>`` pattern (or empty lists) yield ``"unknown"``.
    """
    if not isinstance(description, dict):
        return "unknown"
    global_desc = description.get("global_description")
    if not isinstance(global_desc, dict):
        return "unknown"
    for entry in global_desc.get("room_type") or []:
        match = _ROOM_TYPE_SENTENCE_RE.search(str(entry).lower())
        if match is None:
            continue
        phrase = _ROOM_TYPE_CUT_RE.split(match.group(1))[0].strip(" .,-")
        if phrase:
            return "_".join(phrase.split())
    return "unknown"


def room_type_for_scene(split: str, scene_id: str, description: object) -> str:
    if split == "3dfront":
        return _threedfront_room_type(scene_id)
    if split == "infinigen":
        return _room_type_from_description(description)
    return "unknown"  # mp3d scene ids ("<house>_region<N>") carry no type


def house_id_for_scene(split: str, scene_id: str) -> str:
    if split == "3dfront":
        return scene_id.split("_", 1)[0]
    if split == "mp3d":
        return scene_id.split("_region", 1)[0]
    return scene_id  # infinigen: one synthetic scene == one "house"


# ------------------------------------------------------------------ geometry


def _synthesize_floor_polygon(objects: list[_RawObject]) -> tuple[Vec2, ...]:
    """Axis-aligned CCW rectangle over all footprints + margin per side."""
    xs: list[float] = []
    ys: list[float] = []
    for obj in objects:
        for cx, cy in footprint_corners(obj.position_xy, obj.dimensions, obj.yaw_deg):
            xs.append(cx)
            ys.append(cy)
    min_x, max_x = min(xs) - FLOOR_MARGIN_M, max(xs) + FLOOR_MARGIN_M
    min_y, max_y = min(ys) - FLOOR_MARGIN_M, max(ys) + FLOOR_MARGIN_M
    return ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y))


def _floor_object_specs(
    objects: list[_RawObject], offset: Vec2
) -> tuple[FloorObjectSpec, ...]:
    positions = translate_points((o.position_xy for o in objects), offset)
    return tuple(
        FloorObjectSpec(
            object_id=f"obj_{i:03d}",
            category=obj.category,
            dimensions=obj.dimensions,
            position_xy=position,
            z=0.0,  # kept objects have bottom within FLOOR_SNAP_M of z=0
            yaw_deg=obj.yaw_deg,
        )
        for i, (obj, position) in enumerate(zip(objects, positions))
    )


# ---------------------------------------------------------------- conversion


def _keep_floor_objects(
    raw_objects: list[object], split: str, stats: ConversionStats
) -> list[_RawObject]:
    kept: list[_RawObject] = []
    for raw in raw_objects:
        obj = _parse_object(raw)
        if obj is None:
            stats.skip("malformed_object")
            continue
        bucket = _skip_bucket(obj, split)
        if bucket is not None:
            stats.skip(bucket)
            continue
        kept.append(obj)
    return kept


def convert_scene(
    scene: object,
    *,
    source: str,
    split: str,
    subset: str,
    stats: ConversionStats,
) -> FastFillSample | None:
    """Convert one M3DLayout scene; ``None`` (counted) when unconvertible."""
    if not isinstance(scene, dict) or not isinstance(scene.get("objects"), list):
        stats.skip("malformed_scene")
        return None
    scene_id = scene.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        stats.skip("malformed_scene")
        return None
    kept = _keep_floor_objects(scene["objects"], split, stats)
    if not kept:
        stats.skip("no_floor_objects")
        return None
    polygon, offset = recenter_polygon(_synthesize_floor_polygon(kept))
    provenance = ProvenanceMeta(
        source_dataset=SOURCE_DATASET,
        source_house_id=house_id_for_scene(split, scene_id),
        source_room_id=scene_id,
        upstream_dataset=UPSTREAM_BY_SOURCE.get(source, source),
        license_tag=LICENSE_BY_SOURCE.get(source, LicenseTag.UNKNOWN),
        notes="floor=synthesized_bbox",
    )
    room_context = RoomContext(
        room_id=scene_id,
        room_type=room_type_for_scene(split, scene_id, scene.get("description")),
        floor_polygon=polygon,
    )
    layout = RoomContentLayout(
        room_id=scene_id,
        floor_layout=FloorLayout(
            room_id=scene_id, objects=_floor_object_specs(kept, offset)
        ),
    )
    return FastFillSample(
        sample_id=f"{SOURCE_DATASET}/{split}/{subset}/{scene_id}",
        room_context=room_context,
        layout=layout,
        provenance=provenance,
    )


def _container_files(json_dir: Path, split: str, subset: str) -> list[Path]:
    requested = SPLITS if split == "all" else (split,)
    files: list[Path] = []
    for name in requested:
        files.extend(sorted(json_dir.glob(f"{name}_{subset}*.json")))
    return files


def iter_samples(
    data_dir: Path,
    limit: int | None,
    stats: ConversionStats,
    *,
    split: str = "all",
    subset: str = "train",
) -> Iterator[FastFillSample]:
    """Yield converted samples for the requested split/subset containers."""
    json_dir = data_dir / JSON_SUBDIR
    files = _container_files(json_dir, split, subset)
    if not files:
        raise FileNotFoundError(
            f"no M3DLayout containers for split={split!r} subset={subset!r} "
            f"under {json_dir}"
        )
    emitted = 0
    for path in files:
        container = json.loads(path.read_text(encoding="utf-8"))
        source = str(container.get("source", ""))
        file_split = path.name.split("_", 1)[0]
        for scene in container.get("scenes", []):
            if limit is not None and emitted >= limit:
                return
            sample = convert_scene(
                scene,
                source=source,
                split=file_split,
                subset=subset,
                stats=stats,
            )
            if sample is not None:
                emitted += 1
                yield sample


# ----------------------------------------------------------------------- CLI


def main() -> None:
    """Standard converter CLI (common.converter_main shape) + M3DLayout flags.

    ``common.converter_main`` has no extension point for extra arguments, so
    the --split/--subset flags require mirroring its argparse/report body.
    """
    parser = argparse.ArgumentParser(
        description="Convert M3DLayout rooms to FastFill JSONL"
    )
    parser.add_argument("--data-dir", default=None, help="dataset root")
    parser.add_argument(
        "--out", default="out/m3dlayout.jsonl", help="output JSONL path"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="max samples (None = all)"
    )
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--subset", choices=SUBSETS, default="train")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    stats = ConversionStats()
    out_path = Path(args.out)
    samples = iter_samples(
        data_dir, args.limit, stats, split=args.split, subset=args.subset
    )
    n = atomic_write_jsonl((finalize_sample(s) for s in samples), out_path)
    stats.converted = n
    report_path = out_path.with_suffix(".stats.json")
    report_path.write_text(json.dumps(stats.to_dict(), indent=2))
    print(f"wrote {n} samples -> {out_path}")
    print(f"stats -> {report_path}: {stats.to_dict()}")


if __name__ == "__main__":
    main()
