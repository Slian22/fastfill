"""Bounded deterministic repair for FastFill layouts (T1.6).

``deterministic_repair`` is a pure function over the FROZEN schema types: it
takes the CURRENT validation report plus the room context and returns a new
layout with rule-based geometric fixes applied (the caller revalidates and
may call again). On a clean report the input layout is returned unchanged
with no steps (idempotency contract). Budgets: at most ``FLOOR_STEP_BUDGET``
floor fixes and ``SURFACE_STEP_BUDGET`` fixes per surface per call; budget
overruns and rule-less violations are surfaced via ``failed_surface_ids`` /
``gave_up`` — never silently dropped.

Violation codes handled here (the ``CODE_*`` constants) are verbatim from
``validator.py`` / ``validator_checks.py``; clearance-zone geometry mirrors
the validator's centroid-inward construction so repair subtracts exactly the
zones the validator flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

from scenesmith.growing_world.fastfill.interfaces import RepairOutcome
from scenesmith.growing_world.fastfill.schema import (
    FloorObjectSpec,
    RoomContentLayout,
    RoomContext,
    SupportSurfaceSpec,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
    ValidationReport,
    Vec2,
    Vec3,
    Violation,
)
from scenesmith.growing_world.fastfill.transforms import (
    polygon_centroid,
    rotate_2d,
)

_Spec = TypeVar("_Spec", FloorObjectSpec, SurfaceObjectSpec)

FLOOR_STEP_BUDGET = 20
SURFACE_STEP_BUDGET = 10
SEPARATION_MARGIN_M = 0.02
WINDOW_CLEARANCE_DEPTH_M = 0.9  # mirrors validator_checks
SURFACE_OCCUPANCY_MAX_RATIO = 0.9  # mirrors validator_checks

# ---- violation codes this engine repairs (verbatim from validator.py) ----
CODE_FLOOR_OOB = "L1_FLOOR_OUT_OF_BOUNDS"
CODE_FLOOR_COLLISION = "L1_FLOOR_COLLISION"
CODE_DOOR_BLOCKED = "L1_DOOR_BLOCKED"
CODE_WINDOW_BLOCKED = "L1_WINDOW_BLOCKED"
CODE_FORBIDDEN_REGION = "L1_FORBIDDEN_REGION"
CODE_SURFACE_OOB = "L1_SURFACE_OBJECT_OUT_OF_BOUNDS"
CODE_SURFACE_COLLISION = "L1_SURFACE_COLLISION"
CODE_SURFACE_FLOATING = "L1_FLOATING"
CODE_SURFACE_CAPACITY = "L1_SURFACE_CAPACITY_EXCEEDED"
CODE_SURFACE_OCCUPANCY = "L1_SURFACE_OCCUPANCY_EXCEEDED"
CODE_BUDGET_SURFACE_OBJECTS = "L0_BUDGET_SURFACE_OBJECTS_EXCEEDED"
CODE_BUDGET_TOTAL_OBJECTS = "L0_BUDGET_TOTAL_OBJECTS_EXCEEDED"


@dataclass
class _WorkState:
    """Function-local working buffers; entries are replaced with NEW frozen
    specs (``model_copy``) — the input layout is never mutated."""

    floor_objects: dict[str, FloorObjectSpec]
    floor_order: tuple[str, ...]
    groups: list[SurfaceObjectGroup]
    surfaces: dict[str, SupportSurfaceSpec]
    steps: list[str] = field(default_factory=list)
    failed_surfaces: set[str] = field(default_factory=set)
    gave_up: bool = False


# ------------------------------------------------------------ geometry helpers


def _nearest_position(
    allowed: BaseGeometry, point: Point, hw: float, hd: float
) -> Optional[Vec2]:
    """Nearest center position whose footprint fits in ``allowed``.

    Erodes ``allowed`` by the footprint circumradius first (guaranteed
    containment for any yaw), falling back to looser radii when the eroded
    region vanishes; ``None`` only when ``allowed`` itself is empty.
    """
    for radius in (math.hypot(hw, hd), max(hw, hd), min(hw, hd)):
        region = allowed.buffer(-radius)
        if not region.is_empty:
            target = nearest_points(region, point)[0]
            return (float(target.x), float(target.y))
    if allowed.is_empty:
        return None
    target = nearest_points(allowed, point)[0]
    return (float(target.x), float(target.y))


def _clearance_rect(
    center: Vec2, width_m: float, depth_m: float, room_centroid: Vec2
) -> Optional[Polygon]:
    """Keep-free rectangle from a wall opening into the room.

    Mirrors the validator's construction exactly: the inward direction points
    toward the floor-polygon centroid; degenerate openings return ``None``.
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


def _all_clearance_zones(ctx: RoomContext) -> list[Polygon]:
    centroid = polygon_centroid(ctx.floor_polygon)
    zones: list[Polygon] = []
    for door in ctx.doors:
        rect = _clearance_rect(
            door.center_xy, door.width_m, door.clearance_depth_m, centroid
        )
        if rect is not None:
            zones.append(rect)
    for window in ctx.windows:
        if not window.keep_clear:
            continue
        rect = _clearance_rect(
            window.center_xy,
            window.width_m,
            WINDOW_CLEARANCE_DEPTH_M,
            centroid,
        )
        if rect is not None:
            zones.append(rect)
    return zones


def _projected_half_extent(dimensions: Vec3, yaw_deg: float, unit: Vec2) -> float:
    """Half-extent of the yawed footprint projected onto ``unit`` (SAT)."""
    axis_x = rotate_2d((1.0, 0.0), yaw_deg)
    axis_y = rotate_2d((0.0, 1.0), yaw_deg)
    hw, hd = dimensions[0] / 2.0, dimensions[1] / 2.0
    return hw * abs(unit[0] * axis_x[0] + unit[1] * axis_x[1]) + hd * abs(
        unit[0] * axis_y[0] + unit[1] * axis_y[1]
    )


def _push_apart_position(
    lower_pos: Vec2,
    lower_dims: Vec3,
    lower_yaw: float,
    other_pos: Vec2,
    other_dims: Vec3,
    other_yaw: float,
) -> Optional[Vec2]:
    """New position for ``lower`` pushed along the centers line by
    overlap + margin; ``None`` when already separated along that axis."""
    dx = lower_pos[0] - other_pos[0]
    dy = lower_pos[1] - other_pos[1]
    dist = math.hypot(dx, dy)
    unit: Vec2 = (dx / dist, dy / dist) if dist > 1e-9 else (1.0, 0.0)
    overlap = (
        _projected_half_extent(other_dims, other_yaw, unit)
        + _projected_half_extent(lower_dims, lower_yaw, unit)
        - (dist if dist > 1e-9 else 0.0)
    )
    if overlap <= 0.0:
        return None
    push = overlap + SEPARATION_MARGIN_M
    return (lower_pos[0] + unit[0] * push, lower_pos[1] + unit[1] * push)


# ---------------------------------------------------------------- floor fixes


def _move_objects_inside(
    violation: Violation, state: _WorkState, allowed: BaseGeometry, tag: str
) -> None:
    for object_id in violation.object_ids:
        obj = state.floor_objects.get(object_id)
        if obj is None:
            state.steps.append(f"skip:{tag}:{object_id}:unknown_object")
            continue
        pos = _nearest_position(
            allowed,
            Point(obj.position_xy),
            obj.dimensions[0] / 2.0,
            obj.dimensions[1] / 2.0,
        )
        if pos is None:
            state.gave_up = True
            state.steps.append(f"skip:{tag}:{object_id}:no_valid_region")
            continue
        state.floor_objects[object_id] = obj.model_copy(update={"position_xy": pos})
        state.steps.append(
            f"{tag}:{object_id}:({obj.position_xy[0]:.2f},"
            f"{obj.position_xy[1]:.2f})->({pos[0]:.2f},{pos[1]:.2f})"
        )


def _fix_floor_oob(violation: Violation, state: _WorkState, ctx: RoomContext) -> None:
    _move_objects_inside(violation, state, Polygon(ctx.floor_polygon), "floor_oob")


def _fix_forbidden_region(
    violation: Violation, state: _WorkState, ctx: RoomContext
) -> None:
    allowed: BaseGeometry = Polygon(ctx.floor_polygon)
    if ctx.forbidden_regions:
        allowed = allowed.difference(
            unary_union([Polygon(r.polygon) for r in ctx.forbidden_regions])
        )
    _move_objects_inside(violation, state, allowed, "forbidden_region")


def _fix_clearance_blocked(
    violation: Violation, state: _WorkState, ctx: RoomContext
) -> None:
    allowed: BaseGeometry = Polygon(ctx.floor_polygon)
    zones = _all_clearance_zones(ctx)
    if zones:
        allowed = allowed.difference(unary_union(zones))
    _move_objects_inside(violation, state, allowed, "clearance_blocked")


def _lower_priority(a: _Spec, b: _Spec) -> tuple[_Spec, _Spec]:
    """(lower, other): not-required first, else smaller footprint, else id."""
    if a.required_by_task != b.required_by_task:
        return (a, b) if b.required_by_task else (b, a)
    area_a = a.dimensions[0] * a.dimensions[1]
    area_b = b.dimensions[0] * b.dimensions[1]
    if area_a != area_b:
        return (a, b) if area_a < area_b else (b, a)
    return (a, b) if a.object_id >= b.object_id else (b, a)


def _fix_floor_collision(
    violation: Violation, state: _WorkState, ctx: RoomContext
) -> None:
    if len(violation.object_ids) < 2:
        state.steps.append(f"skip:collision:{violation.object_ids}:need_pair")
        return
    a = state.floor_objects.get(violation.object_ids[0])
    b = state.floor_objects.get(violation.object_ids[1])
    if a is None or b is None:
        state.steps.append(f"skip:collision:{violation.object_ids[:2]}:unknown_object")
        return
    lower, other = _lower_priority(a, b)
    new_pos = _push_apart_position(
        lower.position_xy,
        lower.dimensions,
        lower.yaw_deg,
        other.position_xy,
        other.dimensions,
        other.yaw_deg,
    )
    if new_pos is None:
        state.steps.append(f"collision:{lower.object_id}:already_separated")
        return
    state.floor_objects[lower.object_id] = lower.model_copy(
        update={"position_xy": new_pos}
    )
    state.steps.append(
        f"collision:{lower.object_id}:pushed away from {other.object_id} "
        f"to ({new_pos[0]:.2f},{new_pos[1]:.2f})"
    )


# -------------------------------------------------------------- surface fixes


def _find_surface_object(
    state: _WorkState, surface_id: str, object_id: str
) -> Optional[tuple[int, int, SurfaceObjectSpec]]:
    for gi, group in enumerate(state.groups):
        if group.surface_id != surface_id:
            continue
        for oi, obj in enumerate(group.objects):
            if obj.object_id == object_id:
                return gi, oi, obj
    return None


def _replace_surface_object(
    state: _WorkState, gi: int, oi: int, new_obj: SurfaceObjectSpec
) -> None:
    group = state.groups[gi]
    objects = list(group.objects)
    objects[oi] = new_obj
    state.groups[gi] = group.model_copy(update={"objects": tuple(objects)})


def _fix_surface_oob(violation: Violation, state: _WorkState, ctx: RoomContext) -> None:
    surface = state.surfaces.get(violation.surface_id)
    if surface is None:
        state.steps.append(f"skip:surface_oob:{violation.surface_id}:unknown_surface")
        state.failed_surfaces.add(violation.surface_id)
        return
    # position_local is in the SURFACE-CENTROID frame (README contract); the
    # validator shifts polygon_local to that frame before containment, so
    # repair must clamp in the SAME shifted frame or it clamps to the wrong
    # region whenever polygon_local is not centroid-centered.
    cx, cy = polygon_centroid(surface.polygon_local)
    polygon = Polygon(
        tuple((x - cx, y - cy) for x, y in surface.polygon_local)
    )
    for object_id in violation.object_ids:
        located = _find_surface_object(state, violation.surface_id, object_id)
        if located is None:
            state.steps.append(f"skip:surface_oob:{object_id}:unknown_object")
            continue
        gi, oi, obj = located
        pos = _nearest_position(
            polygon,
            Point(obj.position_local),
            obj.dimensions[0] / 2.0,
            obj.dimensions[1] / 2.0,
        )
        if pos is None:
            state.failed_surfaces.add(violation.surface_id)
            continue
        _replace_surface_object(
            state, gi, oi, obj.model_copy(update={"position_local": pos})
        )
        state.steps.append(
            f"surface_oob:{object_id}:clamped to ({pos[0]:.2f},{pos[1]:.2f})"
        )


def _fix_surface_floating(
    violation: Violation, state: _WorkState, ctx: RoomContext
) -> None:
    for object_id in violation.object_ids:
        located = _find_surface_object(state, violation.surface_id, object_id)
        if located is None:
            state.steps.append(f"skip:floating:{object_id}:unknown_object")
            continue
        gi, oi, obj = located
        _replace_surface_object(state, gi, oi, obj.model_copy(update={"z_local": 0.0}))
        state.steps.append(f"floating:{object_id}:z_local {obj.z_local}->0.0")


def _fix_surface_collision(
    violation: Violation, state: _WorkState, ctx: RoomContext
) -> None:
    if len(violation.object_ids) < 2:
        state.steps.append(f"skip:surface_collision:{violation.object_ids}:need_pair")
        return
    first = _find_surface_object(state, violation.surface_id, violation.object_ids[0])
    second = _find_surface_object(state, violation.surface_id, violation.object_ids[1])
    if first is None or second is None:
        state.steps.append(
            f"skip:surface_collision:{violation.object_ids[:2]}:unknown_object"
        )
        return
    lower, other = _lower_priority(first[2], second[2])
    new_pos = _push_apart_position(
        lower.position_local,
        lower.dimensions,
        lower.yaw_deg_local,
        other.position_local,
        other.dimensions,
        other.yaw_deg_local,
    )
    if new_pos is None:
        state.steps.append(f"surface_collision:{lower.object_id}:already_separated")
        return
    gi, oi, _ = first if lower.object_id == first[2].object_id else second
    _replace_surface_object(
        state, gi, oi, lower.model_copy(update={"position_local": new_pos})
    )
    state.steps.append(
        f"surface_collision:{lower.object_id}:pushed away from "
        f"{other.object_id} to ({new_pos[0]:.2f},{new_pos[1]:.2f})"
    )


def _drop_from_groups(state: _WorkState, group_indices: list[int], need: int) -> int:
    """Drop up to ``need`` non-required objects, last-listed (decorative)
    first; required_by_task objects are NEVER dropped."""
    dropped: list[str] = []
    for gi in reversed(group_indices):
        group = state.groups[gi]
        keep = list(group.objects)
        for oi in range(len(keep) - 1, -1, -1):
            if need == 0:
                break
            if not keep[oi].required_by_task:
                dropped.append(keep[oi].object_id)
                del keep[oi]
                need -= 1
        state.groups[gi] = group.model_copy(update={"objects": tuple(keep)})
        if need == 0:
            break
    if dropped:
        state.steps.append(f"overflow:dropped {','.join(dropped)}")
    return need


def _fix_surface_overflow(
    violation: Violation, state: _WorkState, ctx: RoomContext
) -> None:
    if violation.surface_id:
        indices = [
            gi
            for gi, g in enumerate(state.groups)
            if g.surface_id == violation.surface_id
        ]
        cap = ctx.budget.max_surface_objects_per_surface
        surface = state.surfaces.get(violation.surface_id)
        if surface is not None:
            cap = min(cap, surface.capacity_max_objects)
    else:
        indices = list(range(len(state.groups)))
        cap = ctx.budget.max_surface_objects_total
    total = sum(len(state.groups[gi].objects) for gi in indices)
    if total <= cap:
        state.steps.append("overflow:already_within_cap")
        return
    remaining = _drop_from_groups(state, indices, total - cap)
    if remaining > 0:
        target = violation.surface_id or "room_total"
        state.steps.append(
            f"overflow:{target}:cannot reach cap without dropping "
            f"required_by_task objects ({remaining} over)"
        )
        if violation.surface_id:
            state.failed_surfaces.add(violation.surface_id)
        else:
            state.gave_up = True


def _fix_surface_occupancy(
    violation: Violation, state: _WorkState, ctx: RoomContext
) -> None:
    """Drop decorative objects until footprint area fits the occupancy cap."""
    surface = state.surfaces.get(violation.surface_id)
    if surface is None:
        state.steps.append(f"skip:occupancy:{violation.surface_id}:unknown_surface")
        state.failed_surfaces.add(violation.surface_id)
        return
    allowed_area = SURFACE_OCCUPANCY_MAX_RATIO * Polygon(surface.polygon_local).area
    indices = [
        gi for gi, g in enumerate(state.groups) if g.surface_id == violation.surface_id
    ]
    total = sum(
        o.dimensions[0] * o.dimensions[1]
        for gi in indices
        for o in state.groups[gi].objects
    )
    if total <= allowed_area:
        state.steps.append("occupancy:already_within_cap")
        return
    dropped: list[str] = []
    for gi in reversed(indices):
        group = state.groups[gi]
        keep = list(group.objects)
        for oi in range(len(keep) - 1, -1, -1):
            if total <= allowed_area:
                break
            if keep[oi].required_by_task:
                continue
            total -= keep[oi].dimensions[0] * keep[oi].dimensions[1]
            dropped.append(keep[oi].object_id)
            del keep[oi]
        state.groups[gi] = group.model_copy(update={"objects": tuple(keep)})
        if total <= allowed_area:
            break
    if dropped:
        state.steps.append(f"occupancy:dropped {','.join(dropped)}")
    if total > allowed_area:
        state.failed_surfaces.add(violation.surface_id)
        state.steps.append(
            f"occupancy:{violation.surface_id}:still over cap after dropping "
            "all non-required objects"
        )


# -------------------------------------------------------------------- driver


_RULES: dict[str, Callable[[Violation, _WorkState, RoomContext], None]] = {
    CODE_FLOOR_OOB: _fix_floor_oob,
    CODE_FLOOR_COLLISION: _fix_floor_collision,
    CODE_DOOR_BLOCKED: _fix_clearance_blocked,
    CODE_WINDOW_BLOCKED: _fix_clearance_blocked,
    CODE_FORBIDDEN_REGION: _fix_forbidden_region,
    CODE_SURFACE_OOB: _fix_surface_oob,
    CODE_SURFACE_COLLISION: _fix_surface_collision,
    CODE_SURFACE_FLOATING: _fix_surface_floating,
    CODE_SURFACE_CAPACITY: _fix_surface_overflow,
    CODE_SURFACE_OCCUPANCY: _fix_surface_occupancy,
    CODE_BUDGET_SURFACE_OBJECTS: _fix_surface_overflow,
    CODE_BUDGET_TOTAL_OBJECTS: _fix_surface_overflow,
}


def _note_unhandled(state: _WorkState, violation: Violation) -> None:
    state.steps.append(f"skip:{violation.code}:no_rule")
    if violation.surface_id:
        state.failed_surfaces.add(violation.surface_id)
    else:
        state.gave_up = True


def _rebuild(layout: RoomContentLayout, state: _WorkState) -> RoomContentLayout:
    new_floor = layout.floor_layout.model_copy(
        update={"objects": tuple(state.floor_objects[oid] for oid in state.floor_order)}
    )
    return layout.model_copy(
        update={
            "floor_layout": new_floor,
            "surface_groups": tuple(state.groups),
        }
    )


def deterministic_repair(
    layout: RoomContentLayout,
    report: ValidationReport,
    context: RoomContext,
) -> RepairOutcome:
    """Apply bounded rule-based fixes for ``report``'s error violations.

    Pure: returns a NEW layout (or the input unchanged when the report is
    clean); the caller revalidates and may iterate.
    """
    errors = report.errors()
    if not errors:
        return RepairOutcome(layout=layout)
    state = _WorkState(
        floor_objects={o.object_id: o for o in layout.floor_layout.objects},
        floor_order=tuple(o.object_id for o in layout.floor_layout.objects),
        groups=list(layout.surface_groups),
        surfaces={s.surface_id: s for s in layout.floor_layout.support_surfaces},
    )
    floor_steps = 0
    surface_steps: dict[str, int] = {}
    for violation in errors:
        handler = _RULES.get(violation.code)
        if handler is None:
            _note_unhandled(state, violation)
            continue
        if violation.surface_id:
            used = surface_steps.get(violation.surface_id, 0)
            if used >= SURFACE_STEP_BUDGET:
                state.failed_surfaces.add(violation.surface_id)
                state.steps.append(
                    f"budget:{violation.surface_id}:{violation.code}:skipped"
                )
                continue
            surface_steps[violation.surface_id] = used + 1
        else:
            if floor_steps >= FLOOR_STEP_BUDGET:
                state.gave_up = True
                state.steps.append(f"budget:floor:{violation.code}:skipped")
                continue
            floor_steps += 1
        handler(violation, state, context)
    return RepairOutcome(
        layout=_rebuild(layout, state),
        applied_steps=tuple(state.steps),
        failed_surface_ids=tuple(sorted(state.failed_surfaces)),
        gave_up=state.gave_up,
    )
