"""CanonicalAssetResolver — deterministic bbox-proxy asset binding (V0).

Maps a requested :class:`FloorObjectSpec` to a :class:`ResolvedAsset` whose
``support_surfaces`` carry REAL (canonical-proxy) geometry: call 2 is always
conditioned on these, never on model-estimated surfaces. A real asset-library
resolver plugs in behind the same :class:`AssetResolver` protocol later.

Rules (all deterministic, no I/O):

* requested ``dimensions`` are clamped per-axis to +-40% of the canonical
  size (sanity band against model hallucination);
* surface heights are fractions of the FINAL (clamped) object height;
* surface polygons are the object footprint inset 5% per side (scale 0.90);
* a sink basin is an INSET forbidden region on the counter surface, not a
  separate surface;
* unknown category -> dims pass through unchanged, no surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

from scenesmith.growing_world.fastfill.interfaces import ResolvedAsset
from scenesmith.growing_world.fastfill.schema import (
    FloorObjectSpec,
    ForbiddenRegion,
    SupportSurfaceSpec,
    SurfaceKind,
    Vec2,
    Vec3,
)

DIM_CLAMP_BAND = 0.40  # requested dims may deviate +-40% from canonical
SURFACE_INSET_SCALE = 0.90  # 5% inset per side
BASIN_SCALE = 0.55  # sink basin footprint fraction of counter extents


@dataclass(frozen=True)
class SurfaceDef:
    """One canonical support surface on a category (geometry template)."""

    kind: SurfaceKind
    height_fraction: float  # surface plane as a fraction of object height
    capacity: int
    auto_fill: bool  # surface-bearing: fill even without wants_surface_fill
    basin: bool = False  # carve an inset-basin forbidden region (sinks)


@dataclass(frozen=True)
class CanonicalCategory:
    """Canonical size + surface templates for one furniture category."""

    dims: Vec3  # [width, depth, height] full extents, meters
    surfaces: tuple[SurfaceDef, ...] = ()


def _top(capacity: int, auto_fill: bool = True) -> SurfaceDef:
    return SurfaceDef(SurfaceKind.TOP, 1.0, capacity, auto_fill)


CATEGORY_TABLE: dict[str, CanonicalCategory] = {
    "bed": CanonicalCategory(
        (1.6, 2.0, 0.55), (SurfaceDef(SurfaceKind.TOP, 1.0, 4, False),)
    ),
    "wardrobe": CanonicalCategory((1.2, 0.6, 2.0)),
    "nightstand": CanonicalCategory((0.5, 0.4, 0.55), (_top(4),)),
    "desk": CanonicalCategory((1.2, 0.6, 0.75), (_top(8),)),
    "dining_table": CanonicalCategory((1.6, 0.9, 0.75), (_top(10),)),
    "coffee_table": CanonicalCategory((1.0, 0.6, 0.45), (_top(6),)),
    "sofa": CanonicalCategory(
        (2.0, 0.9, 0.8), (SurfaceDef(SurfaceKind.SEAT, 0.55, 3, False),)
    ),
    "chair": CanonicalCategory(
        (0.45, 0.5, 0.9), (SurfaceDef(SurfaceKind.SEAT, 0.5, 1, False),)
    ),
    "armchair": CanonicalCategory(
        (0.8, 0.8, 0.9), (SurfaceDef(SurfaceKind.SEAT, 0.5, 1, False),)
    ),
    "bench": CanonicalCategory(
        (1.2, 0.4, 0.45), (SurfaceDef(SurfaceKind.SEAT, 1.0, 3, False),)
    ),
    "bookshelf": CanonicalCategory(
        (0.8, 0.3, 1.8),
        (
            SurfaceDef(SurfaceKind.SHELF, 0.4, 6, True),
            SurfaceDef(SurfaceKind.SHELF, 0.7, 6, True),
        ),
    ),
    "shelf": CanonicalCategory(
        (0.8, 0.25, 1.5),
        (
            SurfaceDef(SurfaceKind.SHELF, 0.45, 5, True),
            SurfaceDef(SurfaceKind.SHELF, 0.75, 5, True),
        ),
    ),
    "tv_stand": CanonicalCategory((1.4, 0.4, 0.5), (_top(5),)),
    "sink_cabinet": CanonicalCategory(
        (0.8, 0.5, 0.85),
        (SurfaceDef(SurfaceKind.COUNTER, 1.0, 6, True, basin=True),),
    ),
    "kitchen_island": CanonicalCategory(
        (1.8, 0.9, 0.9), (SurfaceDef(SurfaceKind.COUNTER, 1.0, 8, True),)
    ),
    "toilet": CanonicalCategory((0.4, 0.65, 0.75)),
    "bathtub": CanonicalCategory((1.7, 0.75, 0.6)),
    "shower": CanonicalCategory((0.9, 0.9, 2.0)),
    "dresser": CanonicalCategory((1.0, 0.5, 0.8), (_top(6),)),
    "side_table": CanonicalCategory((0.45, 0.45, 0.55), (_top(4),)),
    "console_table": CanonicalCategory((1.2, 0.35, 0.8), (_top(5),)),
    "cabinet": CanonicalCategory((0.9, 0.45, 1.0), (_top(5),)),
    "refrigerator": CanonicalCategory((0.7, 0.7, 1.8)),
    "stove": CanonicalCategory(
        (0.6, 0.6, 0.9), (SurfaceDef(SurfaceKind.COUNTER, 1.0, 4, False),)
    ),
    "washing_machine": CanonicalCategory(
        (0.6, 0.6, 0.85), (SurfaceDef(SurfaceKind.TOP, 1.0, 3, False),)
    ),
}

# Common aliases the model may emit; resolved to canonical table keys.
CATEGORY_ALIASES: dict[str, str] = {
    "table": "dining_table",
    "counter": "sink_cabinet",
    "bedside_table": "nightstand",
    "couch": "sofa",
    "fridge": "refrigerator",
    "shelving": "shelf",
    "tv_cabinet": "tv_stand",
    "sink": "sink_cabinet",
    "end_table": "side_table",
}


def canonical_category(category: str) -> str | None:
    """Normalize a raw category string to a table key, or None if unknown."""
    key = category.strip().lower().replace(" ", "_").replace("-", "_")
    key = CATEGORY_ALIASES.get(key, key)
    return key if key in CATEGORY_TABLE else None


def _clamp_dims(requested: Vec3, canonical: Vec3) -> Vec3:
    """Clamp each requested extent to +-40% of the canonical extent."""
    out: list[float] = []
    for req, canon in zip(requested, canonical):
        lo = canon * (1.0 - DIM_CLAMP_BAND)
        hi = canon * (1.0 + DIM_CLAMP_BAND)
        out.append(min(max(float(req), lo), hi))
    return (out[0], out[1], out[2])


def _inset_rectangle(width: float, depth: float, scale: float) -> tuple[Vec2, ...]:
    """Axis-aligned rectangle centered at the origin, scaled footprint."""
    hw = width / 2.0 * scale
    hd = depth / 2.0 * scale
    return ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))


def _basin_region(obj_id: str, width: float, depth: float) -> ForbiddenRegion:
    return ForbiddenRegion(
        region_id=f"{obj_id}/basin",
        polygon=_inset_rectangle(width, depth, BASIN_SCALE),
        reason="sink basin (inset, no placement)",
    )


class CanonicalAssetResolver:
    """Deterministic category-table resolver (V0 canonical proxies)."""

    def resolve(self, obj: FloorObjectSpec) -> ResolvedAsset:
        """Bind ``obj`` to a canonical proxy asset with real surface geometry."""
        key = canonical_category(obj.category)
        if key is None:
            return ResolvedAsset(
                asset_id=f"canonical/{obj.category}",
                dimensions=obj.dimensions,
                support_surfaces=(),
                source="canonical",
            )
        entry = CATEGORY_TABLE[key]
        dims = _clamp_dims(obj.dimensions, entry.dims)
        return ResolvedAsset(
            asset_id=f"canonical/{key}",
            dimensions=dims,
            support_surfaces=self._build_surfaces(obj.object_id, entry, dims),
            source="canonical",
        )

    def is_surface_bearing(self, category: str) -> bool:
        """True when the category has surfaces that should be auto-filled."""
        key = canonical_category(category)
        if key is None:
            return False
        return any(s.auto_fill for s in CATEGORY_TABLE[key].surfaces)

    @staticmethod
    def _build_surfaces(
        obj_id: str, entry: CanonicalCategory, dims: Vec3
    ) -> tuple[SupportSurfaceSpec, ...]:
        width, depth, height = dims
        surfaces: list[SupportSurfaceSpec] = []
        kind_counts: dict[str, int] = {}
        for sdef in entry.surfaces:
            index = kind_counts.get(sdef.kind.value, 0)
            kind_counts[sdef.kind.value] = index + 1
            forbidden = (_basin_region(obj_id, width, depth),) if sdef.basin else ()
            surfaces.append(
                SupportSurfaceSpec(
                    surface_id=f"{obj_id}/{sdef.kind.value}{index}",
                    parent_object_id=obj_id,
                    kind=sdef.kind,
                    height_m=sdef.height_fraction * height,
                    polygon_local=_inset_rectangle(width, depth, SURFACE_INSET_SCALE),
                    forbidden_regions_local=forbidden,
                    capacity_max_objects=sdef.capacity,
                    source="canonical",
                )
            )
        return tuple(surfaces)
