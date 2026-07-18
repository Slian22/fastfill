"""Convert MansionWorld (AI2-THOR/objathor) floors to FastFill JSONL.

Usage:
    python tools/fastfill_data/convert_mansionworld.py \
        --data-dir <...>/data --out out/mansionworld.jsonl --limit 100

Source layout: ``data/MansionWorld/mansionworld/<building>/floor_<n>.json``
(ProcTHOR-style scenes; Y-up, meters, ``rotation.y`` in degrees) plus the
objathor asset annotations at
``data/MansionWorld/mansion_patch/asset/annotations.json.gz`` — or the
uncompressed ``annotations.json`` — (assetId ->
``thor_metadata.assetMetadata.boundingBox`` min/max in Y-up meters, plus
category/description/scale/size fields; the measured boundingBox is used,
the GPT-4-estimated ``size`` is ignored).

MansionWorld is the SURFACE-layer source: every ``small_objects`` id encodes
its support parent (``"<small>-i|<parent>-j (<room>)"``), so this converter
emits SurfaceObjectGroups with explicit parent furniture.

Coordinate handling follows ``fastfill/README.md`` via ``transforms.py``:
ground points map ``(x, z)_yup -> (x, -z)_zup``; the room polygon is
re-centered on its centroid and the same offset is applied to all of the
room's objects; small objects are expressed in their parent's local frame.

Yaw caveat: AI2-THOR ``rotation.y`` (degrees) is carried through
``normalize_deg`` unchanged. The AI2-THOR yaw sign / front-axis convention
vs the contract (front=+Y at yaw 0, CCW about +Z) is UNVERIFIED — every
observed floor-object yaw in the data is a multiple of 90 deg, where
footprint de-rotation is sign-insensitive. Verify visually with
``tools/fastfill_data/visualize_sample.py`` before trusting yaw semantics.

Verified data facts this converter relies on (30 buildings, 3,591 floor
objects, 50,106 annotation entries — all with a bounding box):

- floor-object ``vertices`` is the axis-aligned world footprint in
  CENTIMETERS and is inflated by exactly +0.10 m per axis vs the asset's
  annotation bounding box for 99% of objects (collision padding), so
  ``FOOTPRINT_PADDING_M`` is subtracted.
- ``position.y`` is the object CENTER height: ``2 * y`` equals the
  annotation bbox height for 99% of floor objects.
- small-object BOTTOM heights (center y minus half the annotation bbox
  height) cluster to ~1 cm per receptacle while center heights spread
  ~16 cm, so the support-surface plane is the median BOTTOM height.
- some buildings drop the ``F<n>_`` room-id prefix on object ``roomId``
  fields; room matching strips the prefix on both sides.

Doors/windows schema found: ``doorSegment`` / ``windowSegment`` are 2-point
``[[x, z], [x, z]]`` polylines in WORLD ground meters (always present in the
sampled data); ``holePolygon`` is wall-local with ``y`` as height, giving the
window sill and hole height; ``room0``/``room1`` name the connected rooms.
"""

from __future__ import annotations

import functools
import gzip
import json
import math
import re
import statistics
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Direct-script bootstrap: make `fastfill_data.*` (tools/) and `scenesmith.*`
# (repo root) importable when run as `python tools/fastfill_data/...py`.
for _extra in (
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2],
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from fastfill_data.common import ConversionStats, converter_main  # noqa: E402

from scenesmith.growing_world.fastfill.schema import (  # noqa: E402
    DoorSpec,
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
    WindowSpec,
)
from scenesmith.growing_world.fastfill.transforms import (  # noqa: E402
    ensure_ccw,
    normalize_deg,
    recenter_polygon,
    rotate_2d,
    world_to_local_2d,
    yup_extents_to_dimensions,
    yup_to_zup_ground_point,
)

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Json = Mapping[str, Any]

CM_TO_M = 0.01
FOOTPRINT_PADDING_M = 0.10  # verified +0.10 m footprint inflation vs bbox
MIN_DIM_M = 0.01
SURFACE_INSET_FRAC = 0.05  # canonical top polygon: 5% inset per side
DEFAULT_CEILING_M = 2.5

MANSIONWORLD_RELPATH = Path("MansionWorld") / "mansionworld"
ANNOTATIONS_RELPATH = (
    Path("MansionWorld") / "mansion_patch" / "asset" / "annotations.json.gz"
)

_FLOOR_PREFIX_RE = re.compile(r"^F\d+_")
_TRAILING_INDEX_RE = re.compile(r"-\d+$")


# ---------------------------------------------------------------- annotations


@functools.lru_cache(maxsize=4)
def load_annotation_dims(annotations_path: str) -> dict[str, Vec3]:
    """assetId -> contract ``[width, depth, height]`` from objathor metadata.

    Uses ``thor_metadata.assetMetadata.boundingBox`` (Y-up meters); entries
    without a positive-extent box are dropped.
    """
    opener = gzip.open if annotations_path.endswith(".gz") else open
    with opener(annotations_path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    dims: dict[str, Vec3] = {}
    for asset_id, entry in raw.items():
        meta = (entry.get("thor_metadata") or {}).get("assetMetadata") or {}
        box = meta.get("boundingBox")
        if not box:
            continue
        extents = tuple(
            float(box["max"][axis]) - float(box["min"][axis])
            for axis in ("x", "y", "z")
        )
        if min(extents) <= 0.0:
            continue
        dims[asset_id] = yup_extents_to_dimensions(extents)
    return dims


# -------------------------------------------------------------------- helpers


def _strip_floor_prefix(room_id: str) -> str:
    """Room key for matching: some buildings drop the ``F<n>_`` prefix on
    object ``roomId`` while the rooms list keeps it."""
    return _FLOOR_PREFIX_RE.sub("", room_id)


def _category(object_name: str) -> str:
    """``"reception_desk-0"`` -> ``"reception_desk"``."""
    return _TRAILING_INDEX_RE.sub("", object_name)


def _ground_xy(position: Json, offset: Vec2) -> Vec2:
    """Y-up world position -> recentered contract room-frame (x, y)."""
    gx, gy = yup_to_zup_ground_point((position["x"], position["z"]))
    return (gx - offset[0], gy - offset[1])


def _parse_small_parent(small_id: str) -> tuple[str, str] | None:
    """``"tray-0|coffee_table-0 (F1_lobby)"`` -> ``("tray-0", "coffee_table-0")``."""
    head = small_id.split(" (", 1)[0]
    if "|" not in head:
        return None
    small_name, parent_name = head.split("|", 1)
    return small_name, parent_name


def _segment_center_width(
    segment: Sequence[Sequence[float]],
) -> tuple[Vec2, float] | None:
    """World-ground 2-point segment -> (contract-frame center, length)."""
    (x0, z0), (x1, z1) = segment[0], segment[1]
    width = math.hypot(float(x1) - float(x0), float(z1) - float(z0))
    if width <= 0.0:
        return None
    center = yup_to_zup_ground_point(((x0 + x1) / 2.0, (z0 + z1) / 2.0))
    return center, width


# ------------------------------------------------------------ floor furniture


def _footprint_local_extents(
    vertices_cm: Sequence[Sequence[float]], yaw_deg: float
) -> tuple[float, float] | None:
    """Local ``(width, depth)`` from the cm world footprint, de-rotated.

    General yaw is handled by rotating the footprint corners by ``-yaw``
    about the footprint center (for the observed 0/90/180/270 yaws this
    reduces to an axis swap). The verified +0.10 m collision padding is
    subtracted per axis and the result clamped to ``MIN_DIM_M``.
    """
    if not vertices_cm or len(vertices_cm) < 3:
        return None
    pts = [
        yup_to_zup_ground_point((v[0] * CM_TO_M, v[1] * CM_TO_M)) for v in vertices_cm
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    local = [rotate_2d((p[0] - center[0], p[1] - center[1]), -yaw_deg) for p in pts]
    lx = [p[0] for p in local]
    ly = [p[1] for p in local]
    width = (max(lx) - min(lx)) - FOOTPRINT_PADDING_M
    depth = (max(ly) - min(ly)) - FOOTPRINT_PADDING_M
    return (max(width, MIN_DIM_M), max(depth, MIN_DIM_M))


def _floor_object_spec(
    obj: Json,
    offset: Vec2,
    ann_dims: Mapping[str, Vec3],
    stats: ConversionStats,
    notes: list[str],
) -> FloorObjectSpec | None:
    """One ``floor_objects`` entry -> FloorObjectSpec, or None (skip+count).

    Width/depth come from the footprint ``vertices`` (padding-corrected);
    height is ``2 * position.y`` (center height). When either is missing the
    annotation bbox is the documented fallback (recorded in ``notes``);
    objects with neither source are skipped and counted — geometry is never
    fabricated.
    """
    name = str(obj["object_name"])
    yaw = normalize_deg(float(obj["rotation"]["y"]))
    ann = ann_dims.get(str(obj["assetId"]))
    footprint = _footprint_local_extents(obj.get("vertices") or (), yaw)
    if footprint is None:
        if ann is None:
            stats.skip("floor_object_no_dims")
            return None
        footprint = (ann[0], ann[1])
        notes.append(f"dims=annotation_bbox_fallback:{name}")
    height = 2.0 * float(obj["position"]["y"])
    if height <= MIN_DIM_M:
        if ann is None:
            stats.skip("floor_object_no_height")
            return None
        height = ann[2]
        notes.append(f"height=annotation_bbox_fallback:{name}")
    return FloorObjectSpec(
        object_id=name,
        category=_category(name),
        dimensions=(footprint[0], footprint[1], height),
        position_xy=_ground_xy(obj["position"], offset),
        z=0.0,
        yaw_deg=yaw,
    )


# ------------------------------------------------------------- doors/windows


def _door_specs(
    doors: Sequence[Json], room_id: str, offset: Vec2, stats: ConversionStats
) -> tuple[DoorSpec, ...]:
    """DoorSpecs for one room from the world-frame ``doorSegment`` lines."""
    room_key = _strip_floor_prefix(room_id)
    specs: list[DoorSpec] = []
    for door in doors:
        rooms_pair = (str(door.get("room0", "")), str(door.get("room1", "")))
        keys = tuple(_strip_floor_prefix(r) for r in rooms_pair)
        if room_key not in keys:
            continue
        segment = door.get("doorSegment")
        parsed = (
            _segment_center_width(segment) if segment and len(segment) >= 2 else None
        )
        if parsed is None:
            stats.skip("door_no_segment")
            continue
        (cx, cy), width = parsed
        other = rooms_pair[1] if keys[0] == room_key else rooms_pair[0]
        specs.append(
            DoorSpec(
                door_id=str(door.get("id", f"door_{len(specs)}")),
                center_xy=(cx - offset[0], cy - offset[1]),
                width_m=width,
                connects_room_id=other if other != room_id else "",
            )
        )
    return tuple(specs)


def _window_specs(
    windows: Sequence[Json], room_id: str, offset: Vec2, stats: ConversionStats
) -> tuple[WindowSpec, ...]:
    """WindowSpecs for one room; sill/height from the wall-local hole."""
    room_key = _strip_floor_prefix(room_id)
    specs: list[WindowSpec] = []
    for win in windows:
        win_room = str(win.get("roomId") or win.get("room0", ""))
        if _strip_floor_prefix(win_room) != room_key:
            continue
        segment = win.get("windowSegment")
        parsed = (
            _segment_center_width(segment) if segment and len(segment) >= 2 else None
        )
        if parsed is None:
            stats.skip("window_no_segment")
            continue
        (cx, cy), width = parsed
        hole_kwargs: dict[str, float] = {}
        heights = [float(p["y"]) for p in win.get("holePolygon") or []]
        if len(heights) >= 2 and max(heights) > min(heights):
            hole_kwargs = {
                "sill_height_m": max(min(heights), 0.0),
                "height_m": max(heights) - min(heights),
            }
        specs.append(
            WindowSpec(
                window_id=str(win.get("id", f"window_{len(specs)}")),
                center_xy=(cx - offset[0], cy - offset[1]),
                width_m=width,
                **hole_kwargs,
            )
        )
    return tuple(specs)


# ------------------------------------------------------------- surface layer


def _surface_polygon(parent: FloorObjectSpec) -> tuple[Vec2, ...]:
    """Canonical TOP polygon: parent footprint rect inset 5% per side."""
    half_w = max(parent.dimensions[0] * (0.5 - SURFACE_INSET_FRAC), MIN_DIM_M)
    half_d = max(parent.dimensions[1] * (0.5 - SURFACE_INSET_FRAC), MIN_DIM_M)
    return ((-half_w, -half_d), (half_w, -half_d), (half_w, half_d), (-half_w, half_d))


def _surface_object(
    small: Json,
    small_name: str,
    dims: Vec3,
    z_local: float,
    parent: FloorObjectSpec,
    offset: Vec2,
) -> SurfaceObjectSpec:
    """One small object in its parent surface's local frame."""
    room_xy = _ground_xy(small["position"], offset)
    return SurfaceObjectSpec(
        object_id=small_name,
        category=_category(small_name),
        dimensions=dims,
        position_local=world_to_local_2d(room_xy, parent.position_xy, parent.yaw_deg),
        z_local=z_local,
        yaw_deg_local=normalize_deg(float(small["rotation"]["y"]) - parent.yaw_deg),
    )


def _surface_and_group(
    parent: FloorObjectSpec,
    smalls: Sequence[tuple[str, Json]],
    offset: Vec2,
    ann_dims: Mapping[str, Vec3],
    stats: ConversionStats,
) -> tuple[SupportSurfaceSpec, SurfaceObjectGroup] | None:
    """Canonical TOP surface + object group for one receptacle.

    The surface plane is the median small-object BOTTOM height (center y
    minus half the annotation bbox height); bottoms cluster to ~1 cm per
    receptacle while centers spread ~16 cm (see module docstring). Small
    objects without annotation dims are skipped and counted (``no_dims``).
    """
    rows: list[tuple[Json, str, Vec3, float]] = []
    for small_name, small in smalls:
        dims = ann_dims.get(str(small["assetId"]))
        if dims is None:
            stats.skip("small_object_no_dims")
            continue
        bottom = float(small["position"]["y"]) - dims[2] / 2.0
        rows.append((small, small_name, dims, bottom))
    if not rows:
        stats.skip("surface_no_smalls_with_dims")
        return None
    height = max(statistics.median(r[3] for r in rows), 0.0)
    surface = SupportSurfaceSpec(
        surface_id=f"{parent.object_id}/top",
        parent_object_id=parent.object_id,
        kind=SurfaceKind.TOP,
        height_m=height,
        polygon_local=_surface_polygon(parent),
        source="canonical",
    )
    objects = tuple(
        _surface_object(small, small_name, dims, bottom - height, parent, offset)
        for small, small_name, dims, bottom in rows
    )
    group = SurfaceObjectGroup(
        group_id=f"{surface.surface_id}/group0",
        surface_id=surface.surface_id,
        objects=objects,
    )
    return surface, group


def _build_surfaces(
    smalls_by_parent: Mapping[str, Sequence[tuple[str, Json]]],
    floor_by_name: Mapping[str, FloorObjectSpec],
    offset: Vec2,
    ann_dims: Mapping[str, Vec3],
    stats: ConversionStats,
) -> tuple[tuple[SupportSurfaceSpec, ...], tuple[SurfaceObjectGroup, ...]]:
    """Surfaces + groups for every receptacle with a converted floor parent."""
    surfaces: list[SupportSurfaceSpec] = []
    groups: list[SurfaceObjectGroup] = []
    for parent_name in sorted(smalls_by_parent):
        parent = floor_by_name.get(parent_name)
        if parent is None:
            # Parent is a wall object or an unconverted floor object.
            stats.skip(
                "surface_parent_not_floor_object",
                len(smalls_by_parent[parent_name]),
            )
            continue
        try:
            built = _surface_and_group(
                parent, smalls_by_parent[parent_name], offset, ann_dims, stats
            )
        except (KeyError, TypeError, ValueError):
            stats.skip("surface_malformed")
            continue
        if built is not None:
            surfaces.append(built[0])
            groups.append(built[1])
    return tuple(surfaces), tuple(groups)


# ------------------------------------------------------------- per-room/file


@dataclass(frozen=True)
class _FloorData:
    """Pre-indexed content of one ``floor_<n>.json`` (room keys prefix-less)."""

    building: str
    rooms: tuple[Json, ...]
    floor_by_room: Mapping[str, tuple[Json, ...]]
    smalls_by_room: Mapping[str, Mapping[str, tuple[tuple[str, Json], ...]]]
    doors: tuple[Json, ...]
    windows: tuple[Json, ...]
    ceiling_height: float


def _index_floor_file(
    payload: Json, building: str, stats: ConversionStats
) -> _FloorData:
    """Group objects by room / receptacle; count wall + unparseable objects."""
    wall_objects = payload.get("wall_objects") or ()
    if wall_objects:
        stats.skip("wall_object", len(wall_objects))
    floor_by_room: dict[str, list[Json]] = {}
    for obj in payload.get("floor_objects") or ():
        floor_by_room.setdefault(
            _strip_floor_prefix(str(obj.get("roomId", ""))), []
        ).append(obj)
    smalls: dict[str, dict[str, list[tuple[str, Json]]]] = {}
    for small in payload.get("small_objects") or ():
        parsed = _parse_small_parent(str(small.get("id", "")))
        if parsed is None:
            stats.skip("small_object_unparseable_id")
            continue
        room_key = _strip_floor_prefix(str(small.get("roomId", "")))
        smalls.setdefault(room_key, {}).setdefault(parsed[1], []).append(
            (parsed[0], small)
        )
    ceiling = float(payload.get("wall_height") or 0.0)
    return _FloorData(
        building=building,
        rooms=tuple(payload.get("rooms") or ()),
        floor_by_room={k: tuple(v) for k, v in floor_by_room.items()},
        smalls_by_room={
            rk: {p: tuple(v) for p, v in d.items()} for rk, d in smalls.items()
        },
        doors=tuple(payload.get("doors") or ()),
        windows=tuple(payload.get("windows") or ()),
        ceiling_height=ceiling if ceiling > 0.0 else DEFAULT_CEILING_M,
    )


def _floor_specs_for_room(
    data: _FloorData,
    room_key: str,
    offset: Vec2,
    ann_dims: Mapping[str, Vec3],
    stats: ConversionStats,
    notes: list[str],
) -> tuple[FloorObjectSpec, ...]:
    specs: list[FloorObjectSpec] = []
    for obj in data.floor_by_room.get(room_key, ()):
        try:
            spec = _floor_object_spec(obj, offset, ann_dims, stats, notes)
        except (KeyError, TypeError, ValueError):
            stats.skip("floor_object_malformed")
            continue
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


def _room_asset_ids(data: _FloorData, room_key: str) -> tuple[str, ...]:
    floor_ids = [
        str(o.get("assetId", "")) for o in data.floor_by_room.get(room_key, ())
    ]
    small_ids = [
        str(s.get("assetId", ""))
        for pairs in data.smalls_by_room.get(room_key, {}).values()
        for _, s in pairs
    ]
    return tuple(dict.fromkeys(i for i in floor_ids + small_ids if i))


def _convert_room(
    data: _FloorData,
    room: Json,
    ann_dims: Mapping[str, Vec3],
    stats: ConversionStats,
) -> FastFillSample | None:
    """One room -> FastFillSample (rooms without floor furniture are skipped)."""
    room_id = str(room["id"])
    vertices = room.get("vertices") or ()
    if len(vertices) < 3:
        stats.skip("room_bad_polygon")
        return None
    ground = ensure_ccw([yup_to_zup_ground_point(v) for v in vertices])
    polygon, offset = recenter_polygon(ground)
    room_key = _strip_floor_prefix(room_id)
    # Facing audit (120 real buildings): wall-backed categories split BY
    # CATEGORY — cabinets/shelves/sofas/fridges face into the wall (0-11%
    # toward room) while toilets face the room 100%, i.e. objaverse assets
    # have per-asset canonical fronts. No global yaw offset can fix this;
    # per-asset canonicalization (annotations pose_z_rot_angle) is WP2 work.
    notes: list[str] = ["yaw_facing=unverified_per_asset"]
    specs = _floor_specs_for_room(data, room_key, offset, ann_dims, stats, notes)
    if not specs:
        stats.skip("room_no_floor_objects")
        return None
    floor_by_name = {s.object_id: s for s in specs}
    surfaces, groups = _build_surfaces(
        data.smalls_by_room.get(room_key, {}),
        floor_by_name,
        offset,
        ann_dims,
        stats,
    )
    provenance = ProvenanceMeta(
        source_dataset="mansionworld",
        source_house_id=data.building,
        source_room_id=room_id,
        source_asset_ids=_room_asset_ids(data, room_key),
        upstream_dataset="AI2-THOR/objathor",
        # Card declares CC BY 4.0, but the HF download gate adds a
        # "non-commercial research only" agreement the tag cannot express.
        # The note keeps that visible in lineage; a written policy ruling is
        # required before any commercial permissive checkpoint ships this.
        license_tag=LicenseTag.PERMISSIVE,
        notes="; ".join(
            [*notes, "license_gate=non_commercial_research_checkbox"]
        ),
    )
    context = RoomContext(
        room_id=room_id,
        room_type=str(room.get("roomType", "")),
        floor_polygon=polygon,
        ceiling_height_m=data.ceiling_height,
        doors=_door_specs(data.doors, room_id, offset, stats),
        windows=_window_specs(data.windows, room_id, offset, stats),
    )
    layout = RoomContentLayout(
        room_id=room_id,
        floor_layout=FloorLayout(
            room_id=room_id, objects=specs, support_surfaces=surfaces
        ),
        surface_groups=groups,
    )
    return FastFillSample(
        sample_id=f"mansionworld/{data.building}/{room_id}",
        room_context=context,
        layout=layout,
        provenance=provenance,
    )


def convert_floor_file(
    path: Path, ann_dims: Mapping[str, Vec3], stats: ConversionStats
) -> Iterator[FastFillSample]:
    """Convert one ``floor_<n>.json`` into per-room FastFillSamples."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stats.skip("floor_file_unreadable")
        return
    data = _index_floor_file(payload, path.parent.name, stats)
    for room in data.rooms:
        try:
            sample = _convert_room(data, room, ann_dims, stats)
        except (KeyError, TypeError, ValueError):
            stats.skip("room_malformed")
            continue
        if sample is not None:
            yield sample


def resolve_annotations_path(data_dir: Path) -> Path:
    """Locate the objathor annotations, preferring the shipped ``.json.gz``.

    Missing annotations are a hard error: without them every small object is
    skipped as ``small_object_no_dims`` and the Surface corpus silently
    vanishes (the pre-fix behaviour was an empty-dims fallback).
    """
    gz_path = data_dir / ANNOTATIONS_RELPATH
    if gz_path.exists():
        return gz_path
    plain_path = gz_path.with_suffix("")  # annotations.json.gz -> .json
    if plain_path.exists():
        return plain_path
    raise FileNotFoundError(
        f"MansionWorld objathor annotations not found at {gz_path} (or "
        "uncompressed annotations.json) — required for small-object "
        "dimensions and surface groups"
    )


def convert(
    data_dir: Path, limit: int | None, stats: ConversionStats
) -> Iterator[FastFillSample]:
    """Yield per-room samples from EVERY floor of every building.

    All ``floor_<n>.json`` files are converted (the earlier floor-1-only
    restriction dropped ~73% of MansionWorld rooms). Floor records from
    this source are yaw-gated at export anyway; the extra floors mainly
    add Surface supervision.
    """
    ann_path = resolve_annotations_path(data_dir)
    ann_dims: Mapping[str, Vec3] = load_annotation_dims(str(ann_path))
    if not ann_dims:
        raise ValueError(
            f"annotations file {ann_path} parsed to zero usable bounding "
            "boxes — every small object would be skipped as "
            "small_object_no_dims and the Surface corpus would silently "
            "vanish; the real file carries ~50k entries"
        )
    emitted = 0
    for floor_path in sorted((data_dir / MANSIONWORLD_RELPATH).glob("*/floor_*.json")):
        for sample in convert_floor_file(floor_path, ann_dims, stats):
            yield sample
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def main() -> None:
    converter_main(
        description="Convert MansionWorld floors to FastFill JSONL",
        convert=convert,
        default_out="out/mansionworld.jsonl",
    )


if __name__ == "__main__":
    main()
