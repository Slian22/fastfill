"""Structural (L0) and geometric (L1) checks for the FastFill validator.

Shared infrastructure for ``validator.py`` (T1.4): the violation collector,
the immutable derived-geometry index built once per ``validate`` call, and
the L0/L1 check functions. L2/L3 checks and the task-evidence rebuild live in
``validator.py`` next to the ``validate`` entry point.

Everything here is deterministic, Drake-free and pure with respect to the
frozen pydantic inputs; geometry helpers come from ``transforms.py`` (the
normative coordinate contract in ``README.md``) and polygon math is shapely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

from shapely.geometry import Polygon

from scenesmith.growing_world.fastfill.schema import (
    FloorObjectSpec,
    GroupPattern,
    RelationKind,
    RoomContentLayout,
    RoomContext,
    SemanticRelation,
    Severity,
    SupportSurfaceSpec,
    SurfaceObjectSpec,
    ValidatedTaskEvidence,
    Vec2,
    Violation,
    ViolationLayer,
)
from scenesmith.growing_world.fastfill.transforms import (
    footprint_corners,
    local_to_world_2d,
    normalize_deg,
    polygon_centroid,
)

# ------------------------------------------------------ contract thresholds

_FLOOR_CONTAINMENT_TOL_M = 0.05  # floor containment tolerance: 5 cm.
# Calibrated on real data: in 3D-SynthPlace ground truth, wall-flush
# furniture bboxes overshoot the floor polygon by up to 4.75 cm on 574/574
# measured cases (3D-FRONT-family floor polygons trace the inner floor,
# excluding the wall inset) — a 1 cm tolerance failed 100% of that
# human-curated corpus. 5 cm admits wall-flush placement, still far below
# any real placement error; repair clamps anything beyond it.
_SURFACE_CONTAINMENT_TOL_M = 0.01  # surfaces keep the tight 1 cm tolerance
_MIN_OVERLAP_AREA_M2 = 1e-4  # smallest intersection that counts as overlap
_FLOATING_MAX_ABS_Z_M = 0.02  # |z_local| beyond this -> L1_FLOATING
_NEAR_MAX_GAP_M = 0.6  # 'near' relation: footprint gap limit
_FACING_MAX_ANGLE_DEG = 60.0  # 'facing': front ray within 60 deg cone
_FACING_MAX_DIST_M = 3.5  # 'facing': target must be within this range
_CORRIDOR_HALF_WIDTH_M = 0.3  # 0.6 m wide door->target corridor
_OPERATING_ZONE_DEPTH_M = 0.6  # free zone in front of task furniture
_SURFACE_OCCUPANCY_MAX_RATIO = 0.9  # sum(footprints) <= 0.9 * surface area
_WINDOW_CLEARANCE_DEPTH_M = 0.9  # keep_clear windows act like doors

_UNCHECKED_RELATION_KINDS = frozenset(
    {RelationKind.INSIDE, RelationKind.ALIGNED_WITH, RelationKind.PAIRED_WITH}
)


# ---------------------------------------------------------------- collector


class _Collector:
    """Mutable scratch local to one ``validate`` call; inputs stay untouched."""

    __slots__ = ("violations", "evidence", "checks_run")

    def __init__(self) -> None:
        self.violations: list[Violation] = []
        self.evidence: list[ValidatedTaskEvidence] = []
        self.checks_run: int = 0

    def check(self, n: int = 1) -> None:
        self.checks_run += n

    def violation(
        self,
        code: str,
        severity: Severity,
        message: str,
        object_ids: Sequence[str] = (),
        surface_id: str = "",
        details: str = "",
    ) -> None:
        self.violations.append(
            Violation(
                code=code,
                layer=ViolationLayer(code.split("_", 1)[0]),
                severity=severity,
                message=message,
                object_ids=tuple(object_ids),
                surface_id=surface_id,
                details=details,
            )
        )


# --------------------------------------------------------- geometry helpers


def _is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _floor_values(obj: FloorObjectSpec) -> tuple[float, ...]:
    return (*obj.position_xy, obj.z, obj.yaw_deg, *obj.dimensions)


def _surface_obj_values(obj: SurfaceObjectSpec) -> tuple[float, ...]:
    return (*obj.position_local, obj.z_local, obj.yaw_deg_local, *obj.dimensions)


def _surface_values(surface: SupportSurfaceSpec) -> tuple[float, ...]:
    values: list[float] = [surface.height_m]
    for x, y in surface.polygon_local:
        values.extend((x, y))
    return tuple(values)


def _valid_polygon(vertices: Sequence[Vec2]) -> Polygon:
    poly = Polygon(vertices)
    return poly if poly.is_valid else poly.buffer(0)


@dataclass(frozen=True)
class _SurfaceEntry:
    """A support surface with its parent resolution and local-frame polygon.

    ``poly_local`` is the surface polygon expressed in the surface-LOCAL
    frame (origin at the surface-polygon centroid, per README) so it can be
    compared directly against surface-object footprints.
    """

    spec: SupportSurfaceSpec
    parent: Optional[FloorObjectSpec]  # finite floor-object parent, else None
    centroid_parent_local: Vec2  # surface centroid in the parent local frame
    poly_local: Optional[Polygon]
    poly_local_tol: Optional[Polygon]


@dataclass(frozen=True)
class _SurfaceObjEntry:
    spec: SurfaceObjectSpec
    group_id: str
    surface_id: str
    footprint_local: Optional[Polygon]


@dataclass(frozen=True)
class _WorldPose:
    centroid: Vec2
    footprint: Polygon
    yaw_deg: float


@dataclass(frozen=True)
class _Geometry:
    """Immutable derived-geometry index shared by all layers.

    Maps are keyed by id with first-occurrence-wins semantics, so duplicate
    ids (already FATAL at L0) cannot crash the geometric layers. Only objects
    with finite coordinates get footprints/world poses.
    """

    floor_poly: Polygon
    floor_poly_tol: Polygon
    floor_objects: dict[str, FloorObjectSpec]
    floor_footprints: dict[str, Polygon]
    surfaces: dict[str, _SurfaceEntry]
    surface_objects: dict[str, _SurfaceObjEntry]
    world_poses: dict[str, _WorldPose]


def _build_surface_entries(
    layout: RoomContentLayout, floor_objects: dict[str, FloorObjectSpec]
) -> dict[str, _SurfaceEntry]:
    entries: dict[str, _SurfaceEntry] = {}
    for surface in layout.floor_layout.support_surfaces:
        if surface.surface_id in entries:
            continue
        parent = floor_objects.get(surface.parent_object_id)
        if parent is not None and not _is_finite(_floor_values(parent)):
            parent = None
        poly_local: Optional[Polygon] = None
        poly_local_tol: Optional[Polygon] = None
        centroid: Vec2 = (0.0, 0.0)
        if _is_finite(_surface_values(surface)):
            centroid = polygon_centroid(surface.polygon_local)
            shifted = tuple(
                (x - centroid[0], y - centroid[1]) for x, y in surface.polygon_local
            )
            poly_local = _valid_polygon(shifted)
            poly_local_tol = poly_local.buffer(_SURFACE_CONTAINMENT_TOL_M)
        entries[surface.surface_id] = _SurfaceEntry(
            spec=surface,
            parent=parent,
            centroid_parent_local=centroid,
            poly_local=poly_local,
            poly_local_tol=poly_local_tol,
        )
    return entries


def _build_surface_object_entries(
    layout: RoomContentLayout,
) -> dict[str, _SurfaceObjEntry]:
    entries: dict[str, _SurfaceObjEntry] = {}
    for group in layout.surface_groups:
        for obj in group.objects:
            if obj.object_id in entries:
                continue
            footprint: Optional[Polygon] = None
            if _is_finite(_surface_obj_values(obj)):
                footprint = Polygon(
                    footprint_corners(
                        obj.position_local, obj.dimensions, obj.yaw_deg_local
                    )
                )
            entries[obj.object_id] = _SurfaceObjEntry(
                spec=obj,
                group_id=group.group_id,
                surface_id=group.surface_id,
                footprint_local=footprint,
            )
    return entries


def _build_world_poses(
    floor_objects: dict[str, FloorObjectSpec],
    floor_footprints: dict[str, Polygon],
    surfaces: dict[str, _SurfaceEntry],
    surface_objects: dict[str, _SurfaceObjEntry],
) -> dict[str, _WorldPose]:
    poses: dict[str, _WorldPose] = {}
    for object_id, footprint in floor_footprints.items():
        obj = floor_objects[object_id]
        poses[object_id] = _WorldPose(obj.position_xy, footprint, obj.yaw_deg)
    for object_id, entry in surface_objects.items():
        if object_id in poses or entry.footprint_local is None:
            continue
        surface = surfaces.get(entry.surface_id)
        if surface is None or surface.parent is None:
            continue
        cx, cy = surface.centroid_parent_local
        parent_local = (
            entry.spec.position_local[0] + cx,
            entry.spec.position_local[1] + cy,
        )
        world_xy = local_to_world_2d(
            parent_local, surface.parent.position_xy, surface.parent.yaw_deg
        )
        world_yaw = normalize_deg(surface.parent.yaw_deg + entry.spec.yaw_deg_local)
        footprint = Polygon(
            footprint_corners(world_xy, entry.spec.dimensions, world_yaw)
        )
        poses[object_id] = _WorldPose(world_xy, footprint, world_yaw)
    return poses


def _build_geometry(layout: RoomContentLayout, context: RoomContext) -> _Geometry:
    floor_poly = _valid_polygon(context.floor_polygon)
    floor_objects: dict[str, FloorObjectSpec] = {}
    floor_footprints: dict[str, Polygon] = {}
    for obj in layout.floor_layout.objects:
        if obj.object_id in floor_objects:
            continue  # duplicate ids are FATAL at L0; first occurrence wins
        floor_objects[obj.object_id] = obj
        if _is_finite(_floor_values(obj)):
            floor_footprints[obj.object_id] = Polygon(
                footprint_corners(obj.position_xy, obj.dimensions, obj.yaw_deg)
            )
    surfaces = _build_surface_entries(layout, floor_objects)
    surface_objects = _build_surface_object_entries(layout)
    return _Geometry(
        floor_poly=floor_poly,
        floor_poly_tol=floor_poly.buffer(_FLOOR_CONTAINMENT_TOL_M),
        floor_objects=floor_objects,
        floor_footprints=floor_footprints,
        surfaces=surfaces,
        surface_objects=surface_objects,
        world_poses=_build_world_poses(
            floor_objects, floor_footprints, surfaces, surface_objects
        ),
    )


def _all_object_ids(layout: RoomContentLayout) -> set[str]:
    ids = {o.object_id for o in layout.floor_layout.objects}
    for group in layout.surface_groups:
        ids.update(o.object_id for o in group.objects)
    return ids


def _iter_relations(layout: RoomContentLayout) -> Iterator[SemanticRelation]:
    yield from layout.semantic_relations
    for group in layout.surface_groups:
        yield from group.intra_relations


# ------------------------------------------------------------ L0 structural


def _l0_unique_ids(layout: RoomContentLayout, col: _Collector) -> None:
    ids: list[str] = [o.object_id for o in layout.floor_layout.objects]
    ids += [s.surface_id for s in layout.floor_layout.support_surfaces]
    for group in layout.surface_groups:
        ids.append(group.group_id)
        ids.extend(o.object_id for o in group.objects)
    seen: set[str] = set()
    duplicates: list[str] = []
    for identifier in ids:
        col.check()
        if identifier in seen and identifier not in duplicates:
            duplicates.append(identifier)
        seen.add(identifier)
    for identifier in duplicates:
        col.violation(
            "L0_DUPLICATE_ID",
            Severity.FATAL,
            f"id '{identifier}' is used more than once across the "
            "floor/surface layers",
            object_ids=(identifier,),
        )


def _l0_surface_parents(layout: RoomContentLayout, col: _Collector) -> None:
    floor_ids = {o.object_id for o in layout.floor_layout.objects}
    non_floor_ids = {s.surface_id for s in layout.floor_layout.support_surfaces}
    for group in layout.surface_groups:
        non_floor_ids.add(group.group_id)
        non_floor_ids.update(o.object_id for o in group.objects)
    for surface in layout.floor_layout.support_surfaces:
        col.check()
        parent_id = surface.parent_object_id
        if parent_id in floor_ids:
            continue
        if parent_id in non_floor_ids:
            col.violation(
                "L0_SURFACE_PARENT_NOT_FLOOR",
                Severity.FATAL,
                f"surface '{surface.surface_id}' is parented to non-floor id "
                f"'{parent_id}' (two-layer acyclicity guarantee)",
                object_ids=(parent_id,),
                surface_id=surface.surface_id,
            )
        else:
            col.violation(
                "L0_SURFACE_PARENT_MISSING",
                Severity.ERROR,
                f"surface '{surface.surface_id}' is parented to nonexistent "
                f"id '{parent_id}'",
                object_ids=(parent_id,),
                surface_id=surface.surface_id,
            )


def _l0_group_refs(layout: RoomContentLayout, col: _Collector) -> None:
    surface_ids = {s.surface_id for s in layout.floor_layout.support_surfaces}
    for group in layout.surface_groups:
        col.check()
        if group.surface_id not in surface_ids:
            col.violation(
                "L0_GROUP_SURFACE_MISSING",
                Severity.ERROR,
                f"group '{group.group_id}' targets unknown surface "
                f"'{group.surface_id}'",
                object_ids=(group.group_id,),
                surface_id=group.surface_id,
            )
        if group.anchor_object_id:
            col.check()
            member_ids = {o.object_id for o in group.objects}
            if group.anchor_object_id not in member_ids:
                col.violation(
                    "L0_ANCHOR_NOT_IN_GROUP",
                    Severity.ERROR,
                    f"group '{group.group_id}' anchor "
                    f"'{group.anchor_object_id}' is not one of its objects",
                    object_ids=(group.anchor_object_id,),
                    surface_id=group.surface_id,
                )


def _l0_claim_refs(layout: RoomContentLayout, col: _Collector) -> None:
    known = _all_object_ids(layout)
    for claim in layout.task_evidence_claims:
        for object_id in claim.object_ids:
            col.check()
            if object_id not in known:
                col.violation(
                    "L0_CLAIM_OBJECT_MISSING",
                    Severity.ERROR,
                    f"claim '{claim.claim_id}' references unknown object "
                    f"'{object_id}'",
                    object_ids=(object_id,),
                )


def _l0_relation_refs(layout: RoomContentLayout, col: _Collector) -> None:
    known = _all_object_ids(layout)
    for relation in _iter_relations(layout):
        for object_id in (relation.subject_id, relation.object_id):
            col.check()
            if object_id not in known:
                col.violation(
                    "L0_RELATION_REF_MISSING",
                    Severity.ERROR,
                    f"relation '{relation.relation.value}' references "
                    f"unknown id '{object_id}'",
                    object_ids=(object_id,),
                )


def _l0_finite(layout: RoomContentLayout, col: _Collector) -> None:
    rows: list[tuple[str, tuple[float, ...]]] = [
        (o.object_id, _floor_values(o)) for o in layout.floor_layout.objects
    ]
    rows += [
        (s.surface_id, _surface_values(s)) for s in layout.floor_layout.support_surfaces
    ]
    for group in layout.surface_groups:
        rows += [(o.object_id, _surface_obj_values(o)) for o in group.objects]
    for identifier, values in rows:
        col.check()
        if not _is_finite(values):
            col.violation(
                "L0_NONFINITE_COORDINATE",
                Severity.FATAL,
                f"'{identifier}' has non-finite coordinates or dimensions",
                object_ids=(identifier,),
            )


def _l0_room_ids(
    layout: RoomContentLayout, context: RoomContext, col: _Collector
) -> None:
    col.check(2)
    if layout.room_id != context.room_id:
        col.violation(
            "L0_ROOM_ID_MISMATCH",
            Severity.ERROR,
            f"layout room_id '{layout.room_id}' != context room_id "
            f"'{context.room_id}'",
        )
    if layout.floor_layout.room_id != layout.room_id:
        col.violation(
            "L0_ROOM_ID_MISMATCH",
            Severity.ERROR,
            f"floor_layout room_id '{layout.floor_layout.room_id}' != "
            f"layout room_id '{layout.room_id}'",
        )


def _l0_budget(
    layout: RoomContentLayout, context: RoomContext, col: _Collector
) -> None:
    budget = context.budget
    col.check()
    n_surfaces = len(layout.floor_layout.support_surfaces)
    if n_surfaces > budget.max_support_surfaces_per_room:
        col.violation(
            "L0_BUDGET_SURFACES_EXCEEDED",
            Severity.ERROR,
            f"{n_surfaces} support surfaces exceed the budget of "
            f"{budget.max_support_surfaces_per_room}",
        )
    per_surface: dict[str, int] = {}
    total = 0
    for group in layout.surface_groups:
        count = per_surface.get(group.surface_id, 0) + len(group.objects)
        per_surface[group.surface_id] = count
        total += len(group.objects)
    for surface_id, count in per_surface.items():
        col.check()
        if count > budget.max_surface_objects_per_surface:
            col.violation(
                "L0_BUDGET_SURFACE_OBJECTS_EXCEEDED",
                Severity.ERROR,
                f"surface '{surface_id}' holds {count} objects, budget is "
                f"{budget.max_surface_objects_per_surface}",
                surface_id=surface_id,
            )
    col.check()
    if total > budget.max_surface_objects_total:
        col.violation(
            "L0_BUDGET_TOTAL_OBJECTS_EXCEEDED",
            Severity.ERROR,
            f"{total} surface objects exceed the room budget of "
            f"{budget.max_surface_objects_total}",
        )


def _l0_patterns(layout: RoomContentLayout, col: _Collector) -> None:
    for group in layout.surface_groups:
        col.check()
        n = len(group.objects)
        params = group.pattern_params
        problem = ""
        if (
            group.pattern is GroupPattern.MATRIX
            and params.rows > 0
            and params.cols > 0
            and params.rows * params.cols != n
        ):
            problem = f"matrix {params.rows}x{params.cols} != {n} objects"
        elif group.pattern is GroupPattern.PAIRED and (n < 2 or n % 2 != 0):
            problem = f"paired needs an even object count >= 2, got {n}"
        elif group.pattern is GroupPattern.CIRCULAR and (n < 3 or params.radius_m <= 0):
            problem = (
                f"circular needs >= 3 objects and radius_m > 0, got {n} "
                f"objects with radius {params.radius_m}"
            )
        if problem:
            col.violation(
                "L0_PATTERN_INVALID",
                Severity.ERROR,
                f"group '{group.group_id}' is not expandable: {problem}",
                object_ids=(group.group_id,),
                surface_id=group.surface_id,
            )


def run_l0(layout: RoomContentLayout, context: RoomContext, col: _Collector) -> None:
    """All L0 structural checks, in deterministic order."""
    _l0_unique_ids(layout, col)
    _l0_surface_parents(layout, col)
    _l0_group_refs(layout, col)
    _l0_claim_refs(layout, col)
    _l0_relation_refs(layout, col)
    _l0_finite(layout, col)
    _l0_room_ids(layout, context, col)
    _l0_budget(layout, context, col)
    _l0_patterns(layout, col)


# ---------------------------------------------------- L1 geometry & support


def _l1_floor_bounds(geo: _Geometry, col: _Collector) -> None:
    for object_id, footprint in geo.floor_footprints.items():
        col.check()
        if footprint.is_empty:
            continue
        if not geo.floor_poly_tol.covers(footprint):
            col.violation(
                "L1_FLOOR_OUT_OF_BOUNDS",
                Severity.ERROR,
                f"floor object '{object_id}' footprint leaves the floor "
                "polygon (5 cm wall-flush tolerance)",
                object_ids=(object_id,),
            )


_FUNCTIONAL_OVERLAP_CLASSES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # Tucked seating: chair slides under the table/desk top — bbox footprints
    # overlap heavily while the meshes do not touch (SceneSmith's own COL
    # metric tolerates <1 mm mesh penetration; a zero-tolerance bbox check
    # over-fires here). Measured on 150 real 3D-FRONT rooms: chair x table
    # is the #1 GT "collision" (112 pairs), bed x nightstand #2 (87).
    (("chair", "stool", "bench"), ("table", "desk")),
    # Bedside tuck: nightstand inside the bed's half-extent bbox (headboard).
    (("nightstand",), ("bed",)),
    # Coffee table pushed against sofa (armrest/cushion bbox overhang).
    (("sofa", "couch"), ("table",)),
)


def _is_functional_overlap(category_a: str, category_b: str) -> bool:
    """True when the pair matches a whitelisted functional-overlap class."""
    a, b = category_a.lower(), category_b.lower()
    for class_x, class_y in _FUNCTIONAL_OVERLAP_CLASSES:
        if (any(k in a for k in class_x) and any(k in b for k in class_y)) or (
            any(k in b for k in class_x) and any(k in a for k in class_y)
        ):
            return True
    return False


def _l1_floor_collisions(geo: _Geometry, col: _Collector) -> None:
    ids = list(geo.floor_footprints)
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1 :]:
            col.check()
            area = (
                geo.floor_footprints[id_a].intersection(geo.floor_footprints[id_b]).area
            )
            if area <= _MIN_OVERLAP_AREA_M2:
                continue
            cat_a = geo.floor_objects[id_a].category
            cat_b = geo.floor_objects[id_b].category
            if _is_functional_overlap(cat_a, cat_b):
                col.violation(
                    "L1_FUNCTIONAL_OVERLAP",
                    Severity.WARNING,
                    f"floor objects '{id_a}' and '{id_b}' overlap by "
                    f"{area:.4f} m^2 (whitelisted functional pair — "
                    "recorded, not failing; repair must not separate them)",
                    object_ids=(id_a, id_b),
                )
                continue
            col.violation(
                "L1_FLOOR_COLLISION",
                Severity.ERROR,
                f"floor objects '{id_a}' and '{id_b}' overlap by " f"{area:.4f} m^2",
                object_ids=(id_a, id_b),
            )


def _clearance_rect(
    center: Vec2, width_m: float, depth_m: float, room_centroid: Vec2
) -> Optional[Polygon]:
    """Clearance rectangle extending from a wall opening INTO the room.

    The inward direction is inferred as pointing toward the floor-polygon
    centroid (T1.4 spec); openings coincident with the centroid are skipped.
    """
    cx, cy = center
    dx, dy = room_centroid[0] - cx, room_centroid[1] - cy
    norm = math.hypot(dx, dy)
    if norm < 1e-9 or depth_m <= 0 or width_m <= 0:
        return None
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux  # along-wall direction
    hw = width_m / 2.0
    return Polygon(
        (
            (cx - hw * px, cy - hw * py),
            (cx + hw * px, cy + hw * py),
            (cx + hw * px + depth_m * ux, cy + hw * py + depth_m * uy),
            (cx - hw * px + depth_m * ux, cy - hw * py + depth_m * uy),
        )
    )


def _l1_door_clearance(context: RoomContext, geo: _Geometry, col: _Collector) -> None:
    room_centroid = polygon_centroid(context.floor_polygon)
    zones: list[tuple[str, str, Polygon]] = []
    for door in context.doors:
        rect = _clearance_rect(
            door.center_xy, door.width_m, door.clearance_depth_m, room_centroid
        )
        if rect is not None:
            zones.append(("L1_DOOR_BLOCKED", door.door_id, rect))
    for window in context.windows:
        if not window.keep_clear:
            continue
        rect = _clearance_rect(
            window.center_xy,
            window.width_m,
            _WINDOW_CLEARANCE_DEPTH_M,
            room_centroid,
        )
        if rect is not None:
            zones.append(("L1_WINDOW_BLOCKED", window.window_id, rect))
    for code, zone_id, rect in zones:
        for object_id, footprint in geo.floor_footprints.items():
            col.check()
            area = rect.intersection(footprint).area
            if area > _MIN_OVERLAP_AREA_M2:
                col.violation(
                    code,
                    Severity.ERROR,
                    f"floor object '{object_id}' blocks the clearance zone "
                    f"of '{zone_id}'",
                    object_ids=(object_id,),
                    details=f"overlap={area:.4f} m^2",
                )


def _l1_forbidden_regions(
    context: RoomContext, geo: _Geometry, col: _Collector
) -> None:
    for region in context.forbidden_regions:
        region_poly = _valid_polygon(region.polygon)
        for object_id, footprint in geo.floor_footprints.items():
            col.check()
            area = region_poly.intersection(footprint).area
            if area > _MIN_OVERLAP_AREA_M2:
                col.violation(
                    "L1_FORBIDDEN_REGION",
                    Severity.ERROR,
                    f"floor object '{object_id}' intrudes into forbidden "
                    f"region '{region.region_id}'",
                    object_ids=(object_id,),
                    details=region.reason,
                )


def _l1_surface_objects(geo: _Geometry, col: _Collector) -> None:
    for object_id, entry in geo.surface_objects.items():
        col.check()
        z_local = entry.spec.z_local
        if math.isfinite(z_local) and abs(z_local) > _FLOATING_MAX_ABS_Z_M:
            col.violation(
                "L1_FLOATING",
                Severity.ERROR,
                f"surface object '{object_id}' floats at z_local="
                f"{z_local:.3f} m (limit {_FLOATING_MAX_ABS_Z_M} m)",
                object_ids=(object_id,),
                surface_id=entry.surface_id,
            )
        surface = geo.surfaces.get(entry.surface_id)
        if (
            surface is None
            or surface.poly_local_tol is None
            or entry.footprint_local is None
            or entry.footprint_local.is_empty
        ):
            continue
        col.check()
        if not surface.poly_local_tol.covers(entry.footprint_local):
            col.violation(
                "L1_SURFACE_OBJECT_OUT_OF_BOUNDS",
                Severity.ERROR,
                f"surface object '{object_id}' footprint leaves surface "
                f"'{entry.surface_id}' (1 cm tolerance)",
                object_ids=(object_id,),
                surface_id=entry.surface_id,
            )


def _objects_by_surface(geo: _Geometry) -> dict[str, list[str]]:
    by_surface: dict[str, list[str]] = {}
    for object_id, entry in geo.surface_objects.items():
        by_surface.setdefault(entry.surface_id, []).append(object_id)
    return by_surface


def _l1_surface_overlap(geo: _Geometry, col: _Collector) -> None:
    for surface_id, ids in _objects_by_surface(geo).items():
        for i, id_a in enumerate(ids):
            footprint_a = geo.surface_objects[id_a].footprint_local
            if footprint_a is None:
                continue
            for id_b in ids[i + 1 :]:
                footprint_b = geo.surface_objects[id_b].footprint_local
                if footprint_b is None:
                    continue
                col.check()
                area = footprint_a.intersection(footprint_b).area
                if area > _MIN_OVERLAP_AREA_M2:
                    col.violation(
                        "L1_SURFACE_COLLISION",
                        Severity.ERROR,
                        f"surface objects '{id_a}' and '{id_b}' overlap by "
                        f"{area:.4f} m^2 on '{surface_id}'",
                        object_ids=(id_a, id_b),
                        surface_id=surface_id,
                    )


def _l1_surface_load(geo: _Geometry, col: _Collector) -> None:
    for surface_id, ids in _objects_by_surface(geo).items():
        surface = geo.surfaces.get(surface_id)
        if surface is None:
            continue
        col.check()
        if len(ids) > surface.spec.capacity_max_objects:
            col.violation(
                "L1_SURFACE_CAPACITY_EXCEEDED",
                Severity.ERROR,
                f"surface '{surface_id}' holds {len(ids)} objects, capacity "
                f"is {surface.spec.capacity_max_objects}",
                object_ids=tuple(ids),
                surface_id=surface_id,
            )
        if surface.poly_local is None:
            continue
        col.check()
        used = sum(
            geo.surface_objects[i].footprint_local.area
            for i in ids
            if geo.surface_objects[i].footprint_local is not None
        )
        allowed = _SURFACE_OCCUPANCY_MAX_RATIO * surface.poly_local.area
        if used > allowed:
            col.violation(
                "L1_SURFACE_OCCUPANCY_EXCEEDED",
                Severity.ERROR,
                f"surface '{surface_id}' occupancy {used:.4f} m^2 exceeds "
                f"{allowed:.4f} m^2 "
                f"({_SURFACE_OCCUPANCY_MAX_RATIO:.0%} of surface area)",
                object_ids=tuple(ids),
                surface_id=surface_id,
            )


def run_l1(
    layout: RoomContentLayout,
    context: RoomContext,
    geo: _Geometry,
    col: _Collector,
) -> None:
    """All L1 geometry & support checks, in deterministic order."""
    del layout  # geometry is pre-indexed in ``geo``
    _l1_floor_bounds(geo, col)
    _l1_floor_collisions(geo, col)
    _l1_door_clearance(context, geo, col)
    _l1_forbidden_regions(context, geo, col)
    _l1_surface_objects(geo, col)
    _l1_surface_overlap(geo, col)
    _l1_surface_load(geo, col)
