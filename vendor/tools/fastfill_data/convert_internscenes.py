"""Convert InternScenes (Gen + ScanNet Real2Sim) to FastFill JSONL.

Source layout under ``<data>/InternScenes``:

- Gen branch: ``InternScenes_Gen/Layout_info/<room_type>/<scene_id>/`` with
  ``layout.json`` (object list) and ``StructureMesh/boundary_points.json``
  (closed floor-boundary ring, z = floor plane). 14,089 scenes across five
  room-type folders; the numeric scene id is shared across folders when
  rooms belong to the same house (global house coordinates).
- ScanNet Real2Sim branch (``--branch scannet``, NOT converted by default):
  ``Layout_info/scannet/<scan_id>/layout.json`` (M3DLayout-style per-scene
  dirs, no boundary file). Room types come from ``Layout_info/
  room_types.csv`` (official ScanNet ``sceneType`` labels). Rescans of one
  space share the ``sceneXXXX`` prefix and therefore the house split key.
  Disabled by default pending two known fixes: the floor polygon is
  synthesized from the TARGET objects (answer-envelope leakage; the real
  ``StructureMesh/floor.glb`` should be parsed instead), and its 212-name
  category vocabulary needs a furniture allowlist (the Gen-derived
  manipuland blocklist lets shoes/backpacks/bags/persons through).

The Matterport3D Real2Sim branch is deliberately NOT converted: 1,346 of its
1,696 regions (all 72 of its houses) are the same physical regions already
in the pipeline via the M3DLayout mp3d containers. Converting them under a
second ``source_dataset`` would give the same physical house two different
split keys ("m3dlayout/<house>" vs "internscenes/<house>"), letting one
house straddle train/heldout. Revisit only with a shared-house split-key
scheme.

Verified source semantics (hard-asserted in
``tests/unit/fastfill_data/test_convert_internscenes.py``):

- ``bbox`` is an EmbodiedScan-style 9-tuple ``[cx, cy, cz, dx, dy, dz, rz,
  ry, rx]`` — Z-up meters, CENTER + FULL extents, rotations in radians
  (documented by the dataset README and consistent with measured door
  heights ~2.06 m, beds 2.03 x 1.66 m);
- yaw-0 front is local +X, so contract yaw = deg(rz) - 90 and the contract
  footprint is ``[width=dy, depth=dx, height=dz]``. Measured against
  nearest-boundary-wall facing on wall-backed categories (bed / couch / tv /
  toilet / refrigerator / oven / dishwasher): Gen 92.2% agreement at -90
  (next-best hypothesis 50.9%, n=918); ScanNet 86.7% (vs 53.2%, n=436);
- Gen floor plane sits at the boundary-ring z (~0.126 m), not 0; ScanNet
  floor-category bottoms sit at z=0 (median 0.000, p90 0.011, n=2,210).

Gen ``category`` is dirty: ~1.5k objects carry raw Infinigen factory names
(``comforterfactory(1227314)``); these are normalized before classification
and counted. Gen ``bbox`` sizes are occasionally corrupted (a 1,910 m tall
bed, ~12 beds with 8-16 m footprints) — implausible dimensions are skipped
and counted, as are degenerate boundary rings (~40 scenes with < 1 m^2
floor area). Architectural elements and small manipulands are skipped and
counted per bucket (audit datum, same policy as convert_m3dlayout).

License: the whole InternScenes HF dataset is gated CC BY-NC-SA 4.0
(InternScenes Community License) -> ``LicenseTag.CC_BY_NC`` (research route
only; the ShareAlike clause is recorded via the ``license_gate=`` note).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
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
    ensure_ccw,
    footprint_corners,
    normalize_deg,
    polygon_area,
    rad_to_yaw_deg,
    recenter_polygon,
    translate_points,
)

SOURCE_DATASET = "internscenes"
DATASET_SUBDIR = Path("InternScenes")
GEN_SUBDIR = DATASET_SUBDIR / "InternScenes_Gen" / "Layout_info"
REAL_SUBDIR = DATASET_SUBDIR / "Layout_info"
SCANNET_SUBDIR = REAL_SUBDIR / "scannet"
BRANCHES = ("gen", "scannet")

# Contract yaw = deg(rz) + this offset (yaw-0 front is local +X; measured,
# see module docstring). The same -90 rotation swaps the footprint to
# [width=dy, depth=dx].
YAW_OFFSET_DEG = -90.0
# Objects rotated about x/y beyond this cannot be represented by the
# yaw-only pose contract -> skipped and counted (same policy as IL3D).
TILT_MAX_DEG = 5.0

FLOOR_MARGIN_M = 0.4  # synthesized ScanNet floor rectangle margin per side
# Floor z contract: the Floor codec does not encode z, so any object kept
# with bottom above the snap band would silently train as floor-standing.
FLOOR_SNAP_M = 0.02  # bottoms within this of the floor plane are floor-standing
BELOW_FLOOR_M = 0.05  # bottoms below floor-minus-this are annotation glitches
GEN_CEILING_HEIGHT_M = 2.9  # measured in the dataset README (ceiling ~2.9 m)

# Corrupted-annotation guards (real Gen data contains a 1,910 m tall bed and
# beds with 8-16 m footprints; boundary rings with near-zero area exist too).
MAX_FOOTPRINT_DIM_M = 8.0  # widest plausible single furniture footprint side
MAX_HEIGHT_M = 3.5  # tallest plausible furniture (ceilings are ~2.9 m)
MIN_BOUNDARY_AREA_M2 = 1.0  # smallest plausible room floor area
# Category caps where the generic 8 m limit is too loose: real Gen beds are
# median 2.22 m / p95 2.39 m on their longest side, yet corrupted
# annotations reach 6 m — nothing legitimate is near 3 m. (Long cabinet
# runs up to ~5 m are real kitchen counter geometry — deliberately no cap.)
CATEGORY_MAX_FOOTPRINT_M = {"bed": 3.0}

# Fixed wall/ceiling-mounted architecture: never floor furniture. Wall-mounted
# leftovers not listed here (curtains, wall art, ...) still drop via the
# elevated-bottom check.
ARCHITECTURE_CATEGORIES = frozenset(
    {
        "blinds",
        "carpet",
        "ceiling",
        "curtain",
        "door",
        "doorframe",
        "faucet",
        "floor",
        "handle",
        "light",
        "mirror",
        "picture",
        "rug",
        "wall",
        "window",
    }
)
# Small graspables: surface-fill vocabulary, not floor furniture (counted,
# never converted — audit datum for T2.2, same policy as convert_m3dlayout).
MANIPULAND_CATEGORIES = frozenset(
    {
        "blanket",
        "book",
        "bottle",
        "bowl",
        "box",
        "can",
        "comforter",
        "cup",
        "decoration",
        "food",
        "jar",
        "knife",
        "pan",
        "pillow",
        "plate",
        "pot",
        "spoon",
        "towel",
        "toy",
        "utensil",
    }
)

# Gen room-type folders -> contract room types (complete folder vocabulary).
GEN_ROOM_TYPES = {
    "bathroom": "bathroom",
    "bedroom": "bedroom",
    "diningroom": "dining_room",
    "kitchen": "kitchen",
    "livingroom": "living_room",
}

# Official ScanNet sceneType labels -> contract room types. "Apartment" is a
# whole multi-room home in one scene and "Misc." is unlabeled — both are
# excluded (one layout must be one room).
SCANNET_SKIP_ROOM_TYPES = frozenset({"Apartment", "Misc."})
SCANNET_ROOM_TYPES = {
    "Bathroom": "bathroom",
    "Bedroom / Hotel": "bedroom",
    "Bookstore / Library": "library",
    "Classroom": "classroom",
    "Closet": "closet",
    "ComputerCluster": "computer_cluster",
    "Conference Room": "conference_room",
    "Copy/Mail Room": "copy_mail_room",
    "Dining Room": "dining_room",
    "Game room": "game_room",
    "Gym": "gym",
    "Hallway": "hallway",
    "Kitchen": "kitchen",
    "Laundry Room": "laundry_room",
    "Living room / Lounge": "living_room",
    "Lobby": "lobby",
    "Office": "office",
    "Stairs": "stairs",
    "Storage/Basement/Garage": "storage",
}

# Raw Infinigen factory names leaked into Gen categories, e.g.
# "comforterfactory(1227314)" / "boxcomforterfactory(8180331)".
_FACTORY_RE = re.compile(r"(.+?)factory\(\d+\)$")
_FACTORY_ALIASES = {
    "boxcomforter": "comforter",
    "coffeetable": "coffee_table",
    "comforter": "comforter",
    "sidetable": "side_table",
}

_NOTES_COMMON = "yaw=calibrated_wall_facing_-90;license_gate=hf_gated_cc_by_nc_sa"
GEN_NOTES = f"branch=gen;floor=boundary_polygon;{_NOTES_COMMON}"
SCANNET_NOTES = f"branch=scannet;floor=synthesized_bbox;{_NOTES_COMMON}"


@dataclass(frozen=True)
class _RawObject:
    """One source object after coordinate conversion, before recentering."""

    category: str
    model_uid: str
    position_xy: Vec2  # pre-recenter room frame
    dimensions: Vec3  # contract full [width, depth, height] = [dy, dx, dz]
    yaw_deg: float
    bottom_z: float  # object bottom relative to the floor plane
    tilt_deg: float  # max |ry|, |rx| magnitude in degrees


# ------------------------------------------------------------------- parsing


def normalize_category(raw: str) -> tuple[str, bool]:
    """Clean a category name; ``True`` when a factory name was normalized."""
    name = raw.strip().lower()
    match = _FACTORY_RE.fullmatch(name)
    if match is None:
        return name, False
    base = match.group(1)
    return _FACTORY_ALIASES.get(base, base), True


def _parse_object(raw: object, floor_z: float) -> tuple[_RawObject, bool] | None:
    """Convert one source object; ``None`` when malformed."""
    if not isinstance(raw, dict):
        return None
    category = raw.get("category")
    bbox = raw.get("bbox")
    if not isinstance(category, str) or not category.strip():
        return None
    if not (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 9
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox)
    ):
        return None
    cx, cy, cz, dx, dy, dz, rz, ry, rx = (float(v) for v in bbox)
    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        return None
    name, was_factory = normalize_category(category)
    model_uid = raw.get("model_uid")
    obj = _RawObject(
        category=name,
        model_uid=model_uid if isinstance(model_uid, str) else "",
        position_xy=(cx, cy),
        dimensions=(dy, dx, dz),  # -90 deg frame swap, see YAW_OFFSET_DEG
        yaw_deg=normalize_deg(rad_to_yaw_deg(rz) + YAW_OFFSET_DEG),
        bottom_z=cz - dz / 2.0 - floor_z,
        tilt_deg=max(
            abs(normalize_deg(math.degrees(ry))),
            abs(normalize_deg(math.degrees(rx))),
        ),
    )
    return obj, was_factory


def _skip_bucket(obj: _RawObject) -> str | None:
    """Skip-bucket name for a non-floor object, or ``None`` to convert it."""
    if obj.category in ARCHITECTURE_CATEGORIES:
        return f"architecture:{obj.category}"
    if obj.category in MANIPULAND_CATEGORIES:
        return f"manipuland:{obj.category}"
    width, depth, height = obj.dimensions
    footprint_cap = CATEGORY_MAX_FOOTPRINT_M.get(obj.category, MAX_FOOTPRINT_DIM_M)
    if width > footprint_cap or depth > footprint_cap or height > MAX_HEIGHT_M:
        return "implausible_dimensions"
    if obj.tilt_deg > TILT_MAX_DEG:
        return "tilted_rotation"
    if obj.bottom_z < -BELOW_FLOOR_M:
        return "below_floor"
    if obj.bottom_z > FLOOR_SNAP_M:
        return "elevated"
    return None


def _keep_floor_objects(
    raw_objects: object, floor_z: float, stats: ConversionStats
) -> list[_RawObject]:
    kept: list[_RawObject] = []
    if not isinstance(raw_objects, list):
        return kept
    for raw in raw_objects:
        parsed = _parse_object(raw, floor_z)
        if parsed is None:
            stats.skip("malformed_object")
            continue
        obj, was_factory = parsed
        if was_factory:
            stats.skip("category_factory_normalized")
        bucket = _skip_bucket(obj)
        if bucket is not None:
            stats.skip(bucket)
            continue
        kept.append(obj)
    return kept


# ------------------------------------------------------------------ geometry


def _load_boundary_polygon(
    scene_dir: Path,
) -> tuple[tuple[Vec2, ...], float] | None:
    """Gen floor boundary ring -> (open CCW polygon, floor plane z)."""
    path = scene_dir / "StructureMesh" / "boundary_points.json"
    try:
        points = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(points, list):
        return None
    ring: list[tuple[float, float, float]] = []
    for p in points:
        if not (
            isinstance(p, (list, tuple))
            and len(p) == 3
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
        ):
            return None
        ring.append((float(p[0]), float(p[1]), float(p[2])))
    if len(ring) >= 2 and ring[0][:2] == ring[-1][:2]:
        ring.pop()  # boundary rings repeat the first point at the end
    if len(ring) < 3:
        return None
    floor_z = statistics.median(p[2] for p in ring)
    return ensure_ccw([(p[0], p[1]) for p in ring]), floor_z


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
            z=0.0,  # kept objects have bottom within FLOOR_SNAP_M of the floor
            yaw_deg=obj.yaw_deg,
        )
        for i, (obj, position) in enumerate(zip(objects, positions))
    )


def _build_sample(
    *,
    sample_id: str,
    room_id: str,
    room_type: str,
    polygon: tuple[Vec2, ...],
    kept: list[_RawObject],
    offset: Vec2,
    ceiling_height_m: float | None,
    house_id: str,
    upstream: str,
    notes: str,
) -> FastFillSample:
    context_kwargs = {} if ceiling_height_m is None else {
        "ceiling_height_m": ceiling_height_m
    }
    return FastFillSample(
        sample_id=sample_id,
        room_context=RoomContext(
            room_id=room_id,
            room_type=room_type,
            floor_polygon=polygon,
            **context_kwargs,
        ),
        layout=RoomContentLayout(
            room_id=room_id,
            floor_layout=FloorLayout(
                room_id=room_id, objects=_floor_object_specs(kept, offset)
            ),
        ),
        provenance=ProvenanceMeta(
            source_dataset=SOURCE_DATASET,
            source_house_id=house_id,
            source_room_id=room_id,
            source_asset_ids=tuple(
                sorted({o.model_uid for o in kept if o.model_uid})
            ),
            upstream_dataset=upstream,
            license_tag=LicenseTag.CC_BY_NC,
            notes=notes,
        ),
    )


# ---------------------------------------------------------------- conversion


def convert_gen_scene(
    scene_dir: Path, room_folder: str, stats: ConversionStats
) -> FastFillSample | None:
    """Convert one Gen room; ``None`` (counted) when unconvertible."""
    try:
        raw_objects = json.loads((scene_dir / "layout.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        stats.skip("malformed_scene")
        return None
    boundary = _load_boundary_polygon(scene_dir)
    if boundary is None:
        stats.skip("malformed_boundary")
        return None
    raw_polygon, floor_z = boundary
    if polygon_area(raw_polygon) < MIN_BOUNDARY_AREA_M2:
        stats.skip("degenerate_boundary")
        return None
    kept = _keep_floor_objects(raw_objects, floor_z, stats)
    if not kept:
        stats.skip("no_floor_objects")
        return None
    polygon, offset = recenter_polygon(raw_polygon)
    scene_id = scene_dir.name
    room_id = f"{room_folder}_{scene_id}"
    return _build_sample(
        sample_id=f"{SOURCE_DATASET}/gen/{room_folder}/{scene_id}",
        room_id=room_id,
        room_type=GEN_ROOM_TYPES[room_folder],
        polygon=polygon,
        kept=kept,
        offset=offset,
        ceiling_height_m=GEN_CEILING_HEIGHT_M,
        # Same numeric id across room folders = same house (global coords),
        # so the house key must NOT include the room folder.
        house_id=f"gen_{scene_id}",
        upstream="InternScenes-Gen",
        notes=GEN_NOTES,
    )


def convert_scannet_scene(
    scene_dir: Path, room_type_raw: str | None, stats: ConversionStats
) -> FastFillSample | None:
    """Convert one ScanNet scan; ``None`` (counted) when unconvertible."""
    if room_type_raw is None:
        stats.skip("missing_room_type")
        return None
    if room_type_raw in SCANNET_SKIP_ROOM_TYPES:
        stats.skip(f"excluded_room_type:{room_type_raw}")
        return None
    try:
        raw_objects = json.loads((scene_dir / "layout.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        stats.skip("malformed_scene")
        return None
    # ScanNet floor plane is z=0 (measured: floor-category bottoms median
    # 0.000, p90 0.011 over 2,210 objects).
    kept = _keep_floor_objects(raw_objects, 0.0, stats)
    if not kept:
        stats.skip("no_floor_objects")
        return None
    polygon, offset = recenter_polygon(_synthesize_floor_polygon(kept))
    scan_id = scene_dir.name
    fallback = re.sub(r"[^a-z0-9]+", "_", room_type_raw.lower()).strip("_")
    return _build_sample(
        sample_id=f"{SOURCE_DATASET}/scannet/{scan_id}",
        room_id=scan_id,
        room_type=SCANNET_ROOM_TYPES.get(room_type_raw, fallback),
        polygon=polygon,
        kept=kept,
        offset=offset,
        ceiling_height_m=None,  # unknown for scans -> schema default
        # scene0101_02 is the third rescan of space scene0101: rescans of one
        # space must share the house split key.
        house_id=scan_id.split("_", 1)[0],
        upstream="ScanNet",
        notes=SCANNET_NOTES,
    )


def load_scannet_room_types(real_layout_dir: Path) -> dict[str, str]:
    """``room_types.csv`` -> {scan_id: official sceneType} (scannet rows)."""
    csv_path = real_layout_dir / "room_types.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"room type table missing: {csv_path}")
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return {
            row["scene"]: row["room_type"]
            for row in csv.DictReader(fh)
            if row.get("dataset") == "scannet" and row.get("scene")
        }


# ----------------------------------------------------------------- iteration


def _iter_gen(data_dir: Path, stats: ConversionStats) -> Iterator[FastFillSample]:
    gen_dir = data_dir / GEN_SUBDIR
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"InternScenes Gen branch missing: {gen_dir}")
    for room_folder in sorted(GEN_ROOM_TYPES):
        folder = gen_dir / room_folder
        if not folder.is_dir():
            continue
        for scene_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            sample = convert_gen_scene(scene_dir, room_folder, stats)
            if sample is not None:
                yield sample


def _iter_scannet(data_dir: Path, stats: ConversionStats) -> Iterator[FastFillSample]:
    scannet_dir = data_dir / SCANNET_SUBDIR
    if not scannet_dir.is_dir():
        raise FileNotFoundError(f"InternScenes ScanNet branch missing: {scannet_dir}")
    room_types = load_scannet_room_types(data_dir / REAL_SUBDIR)
    for scene_dir in sorted(p for p in scannet_dir.iterdir() if p.is_dir()):
        sample = convert_scannet_scene(
            scene_dir, room_types.get(scene_dir.name), stats
        )
        if sample is not None:
            yield sample


def iter_samples(
    data_dir: Path,
    limit: int | None,
    stats: ConversionStats,
    *,
    branch: str = "gen",
) -> Iterator[FastFillSample]:
    """Yield converted samples for the requested branch(es).

    Default is ``gen`` only: the ScanNet branch stays opt-in until its floor
    polygon comes from the real ``floor.glb`` (not the target objects) and
    its category vocabulary has a furniture allowlist (see module docstring).
    """
    iterators = {
        "gen": (_iter_gen,),
        "scannet": (_iter_scannet,),
        "all": (_iter_gen, _iter_scannet),
    }[branch]
    emitted = 0
    for iterator in iterators:
        for sample in iterator(data_dir, stats):
            if limit is not None and emitted >= limit:
                return
            emitted += 1
            yield sample


# ----------------------------------------------------------------------- CLI


def main() -> None:
    """Standard converter CLI (common.converter_main shape) + --branch.

    ``common.converter_main`` has no extension point for extra arguments, so
    the --branch flag requires mirroring its argparse/report body.
    """
    parser = argparse.ArgumentParser(
        description="Convert InternScenes (Gen + ScanNet) rooms to FastFill JSONL"
    )
    parser.add_argument("--data-dir", default=None, help="dataset root")
    parser.add_argument(
        "--out", default="out/internscenes.jsonl", help="output JSONL path"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="max samples (None = all)"
    )
    parser.add_argument(
        "--branch",
        choices=(*BRANCHES, "all"),
        default="gen",
        help="gen (default), scannet, or all. scannet is opt-in until its "
        "floor polygon reads the real floor.glb and its categories get a "
        "furniture allowlist; Matterport3D is excluded by design "
        "(1,346/1,696 regions duplicate the m3dlayout mp3d source)",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    stats = ConversionStats()
    out_path = Path(args.out)
    samples = iter_samples(data_dir, args.limit, stats, branch=args.branch)
    n = atomic_write_jsonl((finalize_sample(s) for s in samples), out_path)
    stats.converted = n
    report_path = out_path.with_suffix(".stats.json")
    report_path.write_text(json.dumps(stats.to_dict(), indent=2))
    print(f"wrote {n} samples -> {out_path}")
    print(f"stats -> {report_path}: {stats.to_dict()}")


if __name__ == "__main__":
    main()
