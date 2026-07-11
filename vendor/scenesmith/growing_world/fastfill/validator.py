"""Deterministic, Drake-free validator for FastFill layouts (T1.4).

``validate(layout, context)`` re-derives every guarantee from geometry alone:

* L0 structural — id uniqueness, reference closure (two-layer acyclicity),
  finite coordinates, room-id consistency, output-budget compliance, pattern
  expandability (``validator_checks.run_l0``);
* L1 geometry & support — floor containment, pairwise collisions, door/window
  clearance, forbidden regions, plus the SupportSuite (surface containment,
  floating, same-surface overlap, occupancy, capacity)
  (``validator_checks.run_l1``);
* L2 function & semantics — expected furniture/manipuland coverage and
  geometric truth of declared semantic relations;
* L3 reachability & task evidence — door corridors, front operating zones,
  and an independent REBUILD of every task-evidence claim. Model self-claims
  never count: a layout only passes when no FATAL/ERROR violation remains AND
  every rebuilt evidence item verifies.

Pure function: inputs are frozen pydantic models and the validator returns a
new :class:`ValidationReport` without mutating anything. Geometry helpers come
from ``transforms.py`` (the normative coordinate contract in ``README.md``);
polygon math is shapely. No Drake, no model calls, target <150 ms per room.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional, Sequence

from shapely.geometry import LineString, Polygon

from scenesmith.growing_world.fastfill.schema import (
    EvidenceKind,
    FloorObjectSpec,
    RelationKind,
    RoomContentLayout,
    RoomContext,
    SemanticRelation,
    Severity,
    ValidatedTaskEvidence,
    ValidationReport,
    Vec2,
)
from scenesmith.growing_world.fastfill.transforms import (
    footprint_corners,
    front_direction,
)
from scenesmith.growing_world.fastfill.validator_checks import (
    _CORRIDOR_HALF_WIDTH_M,
    _FACING_MAX_ANGLE_DEG,
    _FACING_MAX_DIST_M,
    _FLOATING_MAX_ABS_Z_M,
    _MIN_OVERLAP_AREA_M2,
    _NEAR_MAX_GAP_M,
    _OPERATING_ZONE_DEPTH_M,
    _UNCHECKED_RELATION_KINDS,
    _all_object_ids,
    _build_geometry,
    _Collector,
    _Geometry,
    _iter_relations,
    _WorldPose,
    run_l0,
    run_l1,
)

# ------------------------------------------------- L2 function & semantics


def _norm_category(text: str) -> str:
    return text.lower().replace(" ", "").replace("_", "")


def _category_matches(expected: str, category: str) -> bool:
    """Normalized bidirectional substring match (T1.4 coverage rule)."""
    e, c = _norm_category(expected), _norm_category(category)
    return bool(e) and bool(c) and (e in c or c in e)


def _l2_expected_coverage(
    context: RoomContext, geo: _Geometry, col: _Collector
) -> None:
    for name in context.expected_furniture:
        col.check()
        if not any(
            _category_matches(name, o.category) for o in geo.floor_objects.values()
        ):
            col.violation(
                "L2_MISSING_EXPECTED_FURNITURE",
                Severity.ERROR,
                f"expected furniture '{name}' has no matching floor object",
                details=name,
            )
    for name in context.expected_manipulands:
        col.check()
        if not any(
            _category_matches(name, e.spec.category)
            for e in geo.surface_objects.values()
        ):
            col.violation(
                "L2_MISSING_EXPECTED_MANIPULAND",
                Severity.ERROR,
                f"expected manipuland '{name}' has no matching surface object",
                details=name,
            )


def _verdict_on(
    relation: SemanticRelation, geo: _Geometry
) -> tuple[Optional[bool], str]:
    entry = geo.surface_objects.get(relation.subject_id)
    if entry is None:
        return False, f"'{relation.subject_id}' is not a surface-layer object"
    surface = geo.surfaces.get(entry.surface_id)
    if surface is None:
        return False, f"surface '{entry.surface_id}' does not exist"
    holds = surface.spec.parent_object_id == relation.object_id
    return holds, (
        f"'{relation.subject_id}' sits on surface '{entry.surface_id}' "
        f"whose parent is '{surface.spec.parent_object_id}'"
    )


def _verdict_facing(
    subject: _WorldPose, target: _WorldPose
) -> tuple[Optional[bool], str]:
    dx = target.centroid[0] - subject.centroid[0]
    dy = target.centroid[1] - subject.centroid[1]
    dist = math.hypot(dx, dy)
    if dist > _FACING_MAX_DIST_M:
        return False, f"target {dist:.2f} m away (limit {_FACING_MAX_DIST_M})"
    if dist < 1e-9:
        return True, "subject and target are coincident"
    fx, fy = front_direction(subject.yaw_deg)
    cos_angle = max(-1.0, min(1.0, (fx * dx + fy * dy) / dist))
    angle = math.degrees(math.acos(cos_angle))
    return angle <= _FACING_MAX_ANGLE_DEG, (
        f"front ray off target by {angle:.1f} deg " f"(limit {_FACING_MAX_ANGLE_DEG})"
    )


def _relation_verdict(
    relation: SemanticRelation, geo: _Geometry
) -> tuple[Optional[bool], str]:
    """Geometric truth of one relation: True/False, or None if unchecked."""
    if relation.relation in _UNCHECKED_RELATION_KINDS:
        return None, (
            f"unchecked_relation_kind: '{relation.relation.value}' is not "
            "geometrically checked in v1"
        )
    if relation.relation is RelationKind.ON:
        return _verdict_on(relation, geo)
    subject = geo.world_poses.get(relation.subject_id)
    target = geo.world_poses.get(relation.object_id)
    if subject is None or target is None:
        return False, "world pose unresolvable for subject or object"
    if relation.relation is RelationKind.NEAR:
        gap = subject.footprint.distance(target.footprint)
        return gap < _NEAR_MAX_GAP_M, (
            f"footprint gap {gap:.3f} m (limit {_NEAR_MAX_GAP_M})"
        )
    return _verdict_facing(subject, target)


def _l2_relations(layout: RoomContentLayout, geo: _Geometry, col: _Collector) -> None:
    known = _all_object_ids(layout)
    for relation in _iter_relations(layout):
        if relation.subject_id not in known or relation.object_id not in known:
            continue  # already flagged as L0_RELATION_REF_MISSING
        col.check()
        verdict, detail = _relation_verdict(relation, geo)
        label = (
            f"{relation.subject_id} {relation.relation.value} " f"{relation.object_id}"
        )
        if verdict is None:
            col.violation(
                "L2_UNCHECKED_RELATION_KIND",
                Severity.WARNING,
                f"relation '{label}' not checked: {detail}",
                object_ids=(relation.subject_id, relation.object_id),
            )
        elif not verdict:
            col.violation(
                "L2_RELATION_VIOLATED",
                Severity.ERROR,
                f"relation '{label}' does not hold: {detail}",
                object_ids=(relation.subject_id, relation.object_id),
                details=detail,
            )


# ------------------------------------------------------------ L3 geometry


def _front_point(obj: FloorObjectSpec) -> Vec2:
    fx, fy = front_direction(obj.yaw_deg)
    half_depth = obj.dimensions[1] / 2.0
    return (
        obj.position_xy[0] + fx * half_depth,
        obj.position_xy[1] + fy * half_depth,
    )


def _corridor_clear(door_center: Vec2, target_id: str, geo: _Geometry) -> bool:
    corridor = LineString(
        (door_center, _front_point(geo.floor_objects[target_id]))
    ).buffer(_CORRIDOR_HALF_WIDTH_M)
    for object_id, footprint in geo.floor_footprints.items():
        if object_id == target_id:
            continue
        if corridor.intersection(footprint).area > _MIN_OVERLAP_AREA_M2:
            return False
    return True


def _reachable_from_any_door(
    target_id: str, context: RoomContext, geo: _Geometry
) -> tuple[bool, str]:
    if target_id not in geo.floor_footprints:
        return False, f"'{target_id}' is not a placeable floor object"
    if not context.doors:
        return True, "no doors in room context; corridor test is vacuous"
    for door in context.doors:
        if _corridor_clear(door.center_xy, target_id, geo):
            width = 2.0 * _CORRIDOR_HALF_WIDTH_M
            return True, (f"clear {width:.1f} m corridor from door '{door.door_id}'")
    return False, "no clear corridor from any door"


def _operating_zone(obj: FloorObjectSpec) -> Polygon:
    fx, fy = front_direction(obj.yaw_deg)
    offset = obj.dimensions[1] / 2.0 + _OPERATING_ZONE_DEPTH_M / 2.0
    center = (
        obj.position_xy[0] + fx * offset,
        obj.position_xy[1] + fy * offset,
    )
    return Polygon(
        footprint_corners(
            center,
            (obj.dimensions[0], _OPERATING_ZONE_DEPTH_M, 0.0),
            obj.yaw_deg,
        )
    )


def _operating_zone_free(target_id: str, geo: _Geometry) -> tuple[bool, str]:
    obj = geo.floor_objects.get(target_id)
    if obj is None or target_id not in geo.floor_footprints:
        return False, f"'{target_id}' is not a placeable floor object"
    zone = _operating_zone(obj)
    for object_id, footprint in geo.floor_footprints.items():
        if object_id == target_id:
            continue
        if zone.intersection(footprint).area > _MIN_OVERLAP_AREA_M2:
            return False, f"operating zone blocked by '{object_id}'"
    return True, "front operating zone free"


def _l3_reachability(context: RoomContext, geo: _Geometry, col: _Collector) -> None:
    for object_id, obj in geo.floor_objects.items():
        if not obj.required_by_task or object_id not in geo.floor_footprints:
            continue
        col.check()
        reachable, detail = _reachable_from_any_door(object_id, context, geo)
        if not reachable:
            col.violation(
                "L3_UNREACHABLE_TASK_TARGET",
                Severity.ERROR,
                f"no clear corridor from any door to '{object_id}'",
                object_ids=(object_id,),
                details=detail,
            )
        col.check()
        free, detail = _operating_zone_free(object_id, geo)
        if not free:
            col.violation(
                "L3_OPERATING_ZONE_BLOCKED",
                Severity.ERROR,
                f"front operating zone of '{object_id}' is blocked",
                object_ids=(object_id,),
                details=detail,
            )


# --------------------------------------------------- task evidence rebuild


def _placed_floor(object_id: str, geo: _Geometry) -> tuple[bool, str]:
    footprint = geo.floor_footprints.get(object_id)
    if footprint is None or footprint.is_empty:
        return False, f"floor object '{object_id}' has no computable footprint"
    if not geo.floor_poly_tol.covers(footprint):
        return False, (f"floor object '{object_id}' is outside the floor polygon")
    return True, f"floor object '{object_id}' placed inside the floor polygon"


def _placed_surface(object_id: str, geo: _Geometry) -> tuple[bool, str]:
    entry = geo.surface_objects[object_id]
    surface = geo.surfaces.get(entry.surface_id)
    if surface is None:
        return False, f"surface '{entry.surface_id}' does not exist"
    z_local = entry.spec.z_local
    if not math.isfinite(z_local) or abs(z_local) > _FLOATING_MAX_ABS_Z_M:
        return False, (
            f"'{object_id}' is not resting on the surface (z_local={z_local})"
        )
    if (
        surface.poly_local_tol is not None
        and entry.footprint_local is not None
        and not entry.footprint_local.is_empty
        and not surface.poly_local_tol.covers(entry.footprint_local)
    ):
        return False, (f"'{object_id}' footprint leaves surface '{entry.surface_id}'")
    return True, f"'{object_id}' rests on surface '{entry.surface_id}'"


def _object_placed(object_id: str, geo: _Geometry) -> tuple[bool, str]:
    if object_id in geo.floor_objects:
        return _placed_floor(object_id, geo)
    if object_id in geo.surface_objects:
        return _placed_surface(object_id, geo)
    return False, f"object '{object_id}' does not exist in the layout"


def _verify_ids(
    object_ids: Sequence[str], checker: Callable[[str], tuple[bool, str]]
) -> tuple[bool, str]:
    if not object_ids:
        return False, "claim lists no object ids to verify"
    verified = True
    details: list[str] = []
    for object_id in object_ids:
        ok, detail = checker(object_id)
        verified = verified and ok
        details.append(detail)
    return verified, "; ".join(details)


def _rebuild_one_claim(
    kind: EvidenceKind,
    object_ids: tuple[str, ...],
    relation: Optional[SemanticRelation],
    context: RoomContext,
    geo: _Geometry,
) -> tuple[bool, str, str]:
    """Return (verified, method, detail) for one claim, from geometry only."""
    if kind is EvidenceKind.OBJECT_PRESENT:
        verified, detail = _verify_ids(object_ids, lambda oid: _object_placed(oid, geo))
        return verified, "rebuilt_object_present", detail
    if kind is EvidenceKind.RELATION_HOLDS:
        if relation is None:
            return False, "rebuilt_relation_holds", "claim carries no relation"
        verdict, detail = _relation_verdict(relation, geo)
        if verdict is None:
            return False, "rebuilt_relation_holds", f"cannot verify: {detail}"
        return bool(verdict), "rebuilt_relation_holds", detail
    if kind is EvidenceKind.CLEARANCE_AVAILABLE:
        verified, detail = _verify_ids(
            object_ids, lambda oid: _operating_zone_free(oid, geo)
        )
        return verified, "rebuilt_clearance_available", detail
    verified, detail = _verify_ids(
        object_ids, lambda oid: _reachable_from_any_door(oid, context, geo)
    )
    return verified, "rebuilt_reachable", detail


def _rebuild_claims(
    layout: RoomContentLayout,
    context: RoomContext,
    geo: _Geometry,
    col: _Collector,
) -> None:
    """Independently re-verify every generator claim (self-reports ignored)."""
    for claim in layout.task_evidence_claims:
        col.check()
        verified, method, detail = _rebuild_one_claim(
            claim.kind, claim.object_ids, claim.relation, context, geo
        )
        col.evidence.append(
            ValidatedTaskEvidence(
                claim_id=claim.claim_id,
                task_requirement_id=claim.task_requirement_id,
                verified=verified,
                method=method,
                detail=detail,
                object_ids=claim.object_ids,
            )
        )


def _rebuild_presence(context: RoomContext, geo: _Geometry, col: _Collector) -> None:
    """Presence evidence for every expected item, claim or no claim."""
    for name in context.expected_furniture:
        col.check()
        matches = tuple(
            oid
            for oid, obj in geo.floor_objects.items()
            if _category_matches(name, obj.category)
        )
        col.evidence.append(
            ValidatedTaskEvidence(
                claim_id=f"rebuilt/furniture/{name}",
                task_requirement_id=name,
                verified=any(_placed_floor(oid, geo)[0] for oid in matches),
                method="rebuilt_presence",
                detail=(
                    f"matched floor objects: {list(matches)}"
                    if matches
                    else "no matching floor object"
                ),
                object_ids=matches,
            )
        )
    for name in context.expected_manipulands:
        col.check()
        matches = tuple(
            oid
            for oid, entry in geo.surface_objects.items()
            if _category_matches(name, entry.spec.category)
        )
        col.evidence.append(
            ValidatedTaskEvidence(
                claim_id=f"rebuilt/manipuland/{name}",
                task_requirement_id=name,
                verified=any(_placed_surface(oid, geo)[0] for oid in matches),
                method="rebuilt_presence",
                detail=(
                    f"matched surface objects: {list(matches)}"
                    if matches
                    else "no matching surface object"
                ),
                object_ids=matches,
            )
        )


# -------------------------------------------------------------- entry point


def validate(layout: RoomContentLayout, context: RoomContext) -> ValidationReport:
    """Run the deterministic L0-L3 validator on one room content layout.

    Pure function of ``(layout, context)``: repeated calls return identical
    reports (up to ``wall_time_ms``) and never mutate their inputs.
    ``passed`` is True iff no FATAL/ERROR violation was found AND every
    rebuilt task-evidence item verified — generator self-claims never count.
    """
    start = time.perf_counter()
    col = _Collector()

    run_l0(layout, context, col)

    geo = _build_geometry(layout, context)
    run_l1(layout, context, geo, col)

    _l2_expected_coverage(context, geo, col)
    _l2_relations(layout, geo, col)

    _l3_reachability(context, geo, col)

    _rebuild_claims(layout, context, geo, col)
    _rebuild_presence(context, geo, col)

    blocking = any(
        v.severity in (Severity.FATAL, Severity.ERROR) for v in col.violations
    )
    all_evidence_verified = all(e.verified for e in col.evidence)
    return ValidationReport(
        room_id=layout.room_id,
        passed=not blocking and all_evidence_verified,
        violations=tuple(col.violations),
        validated_task_evidence=tuple(col.evidence),
        checks_run=col.checks_run,
        wall_time_ms=(time.perf_counter() - start) * 1000.0,
    )
