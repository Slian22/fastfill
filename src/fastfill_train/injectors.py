"""Programmatic violation injection for FastFill DPO stage 2.

Each injector is a pure function ``FastFillSample -> Injection | None``: it
returns a NEW ``RoomContentLayout`` perturbed to violate a specific validator
guarantee (inputs are frozen pydantic models and are never mutated), together
with the exact violation codes the vendored validator is expected to raise.
``None`` means the injector is not applicable to the sample (e.g.
``collision_pair`` needs at least two floor objects).

A pair is only a real preference signal when the perturbed layout actually
fails with an expected code AND the original layout produces none of them.
``verify_injection`` re-derives both facts with the vendored validator;
``run_injector`` combines injection + verification and returns ``None`` when
the guarantee cannot be established on a given sample.

Violation codes below are verbatim from ``validator.py`` /
``validator_checks.py`` (same convention as the vendored ``repair.py``).

Known codec limitation (documented, not a bug): ``floating_surface_object``
perturbs ``z_local``, which the text codec does not encode — the layout-level
negative is real and validator-verified, but its codec rendering equals the
clean completion, so the stage-2 builder skips (and counts) such pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import fastfill_train  # noqa: F401  (vendor path bootstrap)
from scenesmith.growing_world.fastfill.schema import (
    FastFillSample,
    FloorObjectSpec,
    RelationKind,
    RoomContentLayout,
    SupportSurfaceSpec,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
)
from scenesmith.growing_world.fastfill.transforms import (
    normalize_deg,
    polygon_centroid,
)
from scenesmith.growing_world.fastfill.validator import (
    _category_matches,  # exact expected-furniture matching semantics
    validate,
)

# ---- violation codes (verbatim from validator.py / validator_checks.py) ----
CODE_FLOOR_OOB = "L1_FLOOR_OUT_OF_BOUNDS"
CODE_FLOOR_COLLISION = "L1_FLOOR_COLLISION"
CODE_DOOR_BLOCKED = "L1_DOOR_BLOCKED"
CODE_FLOATING = "L1_FLOATING"
CODE_SURFACE_OOB = "L1_SURFACE_OBJECT_OUT_OF_BOUNDS"
CODE_SURFACE_COLLISION = "L1_SURFACE_COLLISION"
CODE_SURFACE_CAPACITY = "L1_SURFACE_CAPACITY_EXCEEDED"
CODE_SURFACE_OCCUPANCY = "L1_SURFACE_OCCUPANCY_EXCEEDED"
CODE_BUDGET_SURFACE_OBJECTS = "L0_BUDGET_SURFACE_OBJECTS_EXCEEDED"
CODE_MISSING_FURNITURE = "L2_MISSING_EXPECTED_FURNITURE"
CODE_RELATION_VIOLATED = "L2_RELATION_VIOLATED"

_FLOATING_Z_LOCAL_M = 0.3
_OOB_EXTRA_MARGIN_M = 1.0
_SURFACE_OOB_MARGIN_M = 0.5
_MAX_OVERFLOW_DUPLICATES = 24

FLOOR_LEVEL = "floor"
SURFACE_LEVEL = "surface"


@dataclass(frozen=True)
class Injection:
    """One programmatic negative: the perturbed layout + expected codes.

    ``level`` tells the DPO builder which SFT record kind the perturbation is
    visible in (``floor`` = floor-layout codec, ``surface`` = the codec of the
    group named by ``group_id``).
    """

    name: str
    bad_layout: RoomContentLayout
    expected_codes: tuple[str, ...]
    level: str
    group_id: str = ""


InjectorFn = Callable[[FastFillSample], Optional[Injection]]


# ------------------------------------------------ immutable layout rewrites


def _replace_floor_object(
    layout: RoomContentLayout, new_obj: FloorObjectSpec
) -> RoomContentLayout:
    objects = tuple(
        new_obj if o.object_id == new_obj.object_id else o
        for o in layout.floor_layout.objects
    )
    floor = layout.floor_layout.model_copy(update={"objects": objects})
    return layout.model_copy(update={"floor_layout": floor})


def _replace_group(
    layout: RoomContentLayout, new_group: SurfaceObjectGroup
) -> RoomContentLayout:
    groups = tuple(
        new_group if g.group_id == new_group.group_id else g
        for g in layout.surface_groups
    )
    return layout.model_copy(update={"surface_groups": groups})


def _replace_surface_object(
    layout: RoomContentLayout, group_id: str, new_obj: SurfaceObjectSpec
) -> RoomContentLayout:
    group = next(g for g in layout.surface_groups if g.group_id == group_id)
    objects = tuple(
        new_obj if o.object_id == new_obj.object_id else o for o in group.objects
    )
    return _replace_group(layout, group.model_copy(update={"objects": objects}))


def _drop_floor_object(
    layout: RoomContentLayout, object_id: str
) -> RoomContentLayout:
    """Remove a floor object and everything referencing it (no dangling ids)."""
    floor = layout.floor_layout
    objects = tuple(o for o in floor.objects if o.object_id != object_id)
    dropped_surfaces = {
        s.surface_id
        for s in floor.support_surfaces
        if s.parent_object_id == object_id
    }
    surfaces = tuple(
        s for s in floor.support_surfaces if s.surface_id not in dropped_surfaces
    )
    groups = tuple(
        g for g in layout.surface_groups if g.surface_id not in dropped_surfaces
    )
    removed = {object_id} | dropped_surfaces
    for group in layout.surface_groups:
        if group.surface_id in dropped_surfaces:
            removed.add(group.group_id)
            removed.update(o.object_id for o in group.objects)
    relations = tuple(
        r
        for r in layout.semantic_relations
        if r.subject_id not in removed and r.object_id not in removed
    )
    claims = tuple(
        c for c in layout.task_evidence_claims if removed.isdisjoint(c.object_ids)
    )
    return layout.model_copy(
        update={
            "floor_layout": floor.model_copy(
                update={"objects": objects, "support_surfaces": surfaces}
            ),
            "surface_groups": groups,
            "semantic_relations": relations,
            "task_evidence_claims": claims,
        }
    )


def _first_surface_target(
    sample: FastFillSample,
) -> Optional[tuple[SurfaceObjectGroup, SurfaceObjectSpec, SupportSurfaceSpec]]:
    """First (group, object, resolved surface) triple, else ``None``."""
    surfaces = {
        s.surface_id: s for s in sample.layout.floor_layout.support_surfaces
    }
    for group in sample.layout.surface_groups:
        surface = surfaces.get(group.surface_id)
        if group.objects and surface is not None:
            return group, group.objects[0], surface
    return None


# ------------------------------------------------------------ floor injectors


def _inject_out_of_bounds(sample: FastFillSample) -> Optional[Injection]:
    """Push a floor object past the +x edge of the floor polygon."""
    objects = sample.layout.floor_layout.objects
    if not objects:
        return None
    obj = objects[0]
    max_x = max(v[0] for v in sample.room_context.floor_polygon)
    moved = obj.model_copy(
        update={
            "position_xy": (
                max_x + obj.dimensions[0] + _OOB_EXTRA_MARGIN_M,
                obj.position_xy[1],
            )
        }
    )
    return Injection(
        name="out_of_bounds",
        bad_layout=_replace_floor_object(sample.layout, moved),
        expected_codes=(CODE_FLOOR_OOB,),
        level=FLOOR_LEVEL,
    )


def _inject_collision_pair(sample: FastFillSample) -> Optional[Injection]:
    """Move the second floor object onto the first one."""
    objects = sample.layout.floor_layout.objects
    if len(objects) < 2:
        return None
    moved = objects[1].model_copy(update={"position_xy": objects[0].position_xy})
    return Injection(
        name="collision_pair",
        bad_layout=_replace_floor_object(sample.layout, moved),
        expected_codes=(CODE_FLOOR_COLLISION,),
        level=FLOOR_LEVEL,
    )


def _inject_blocked_door(sample: FastFillSample) -> Optional[Injection]:
    """Move a floor object into a door clearance rectangle."""
    context = sample.room_context
    objects = sample.layout.floor_layout.objects
    if not context.doors or not objects:
        return None
    centroid = polygon_centroid(context.floor_polygon)
    for door in context.doors:
        dx = centroid[0] - door.center_xy[0]
        dy = centroid[1] - door.center_xy[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-9 or door.clearance_depth_m <= 0:
            continue  # validator skips such doors too
        half_depth = door.clearance_depth_m / 2.0
        target = (
            door.center_xy[0] + dx / norm * half_depth,
            door.center_xy[1] + dy / norm * half_depth,
        )
        moved = objects[0].model_copy(update={"position_xy": target})
        return Injection(
            name="blocked_door",
            bad_layout=_replace_floor_object(sample.layout, moved),
            expected_codes=(CODE_DOOR_BLOCKED,),
            level=FLOOR_LEVEL,
        )
    return None


def _inject_wrong_facing(sample: FastFillSample) -> Optional[Injection]:
    """Rotate the subject of a facing relation by 180 degrees.

    Only applicable when the sample declares a FACING relation whose subject
    is a floor object (samples without facing relations are skipped: without
    one, the validator has nothing to fail on).
    """
    floor_objects = {o.object_id: o for o in sample.layout.floor_layout.objects}
    for relation in sample.layout.semantic_relations:
        if relation.relation is not RelationKind.FACING:
            continue
        subject = floor_objects.get(relation.subject_id)
        if subject is None:
            continue
        rotated = subject.model_copy(
            update={"yaw_deg": normalize_deg(subject.yaw_deg + 180.0)}
        )
        return Injection(
            name="wrong_facing",
            bad_layout=_replace_floor_object(sample.layout, rotated),
            expected_codes=(CODE_RELATION_VIOLATED,),
            level=FLOOR_LEVEL,
        )
    return None


def _inject_missing_required(sample: FastFillSample) -> Optional[Injection]:
    """Drop the only floor object matching an expected-furniture entry."""
    objects = sample.layout.floor_layout.objects
    for name in sample.room_context.expected_furniture:
        matches = [o for o in objects if _category_matches(name, o.category)]
        if len(matches) != 1:
            continue  # dropping one of several matches would not violate
        return Injection(
            name="missing_required",
            bad_layout=_drop_floor_object(sample.layout, matches[0].object_id),
            expected_codes=(CODE_MISSING_FURNITURE,),
            level=FLOOR_LEVEL,
        )
    return None


# ---------------------------------------------------------- surface injectors


def _inject_floating_surface_object(
    sample: FastFillSample,
) -> Optional[Injection]:
    """Lift a surface object to z_local 0.3 m (floating above its surface)."""
    target = _first_surface_target(sample)
    if target is None:
        return None
    group, obj, _ = target
    lifted = obj.model_copy(update={"z_local": _FLOATING_Z_LOCAL_M})
    return Injection(
        name="floating_surface_object",
        bad_layout=_replace_surface_object(sample.layout, group.group_id, lifted),
        expected_codes=(CODE_FLOATING,),
        level=SURFACE_LEVEL,
        group_id=group.group_id,
    )


def _inject_surface_off_polygon(sample: FastFillSample) -> Optional[Injection]:
    """Move a small object outside its surface's local polygon."""
    target = _first_surface_target(sample)
    if target is None:
        return None
    group, obj, surface = target
    centroid = polygon_centroid(surface.polygon_local)
    max_x = max(x - centroid[0] for x, _ in surface.polygon_local)
    moved = obj.model_copy(
        update={
            "position_local": (
                max_x + obj.dimensions[0] + _SURFACE_OOB_MARGIN_M,
                0.0,
            )
        }
    )
    return Injection(
        name="surface_off_polygon",
        bad_layout=_replace_surface_object(sample.layout, group.group_id, moved),
        expected_codes=(CODE_SURFACE_OOB,),
        level=SURFACE_LEVEL,
        group_id=group.group_id,
    )


def _inject_wrong_parent(sample: FastFillSample) -> Optional[Injection]:
    """Repoint a group at another existing surface (needs >= 2 surfaces).

    The wrong parent shows up geometrically (out of bounds / collision /
    capacity / occupancy on the new surface) or semantically (a violated ON
    relation); which code fires depends on the sample, so all plausible codes
    are expected and verification keeps only samples where one actually fires.
    """
    surfaces = sample.layout.floor_layout.support_surfaces
    if len({s.surface_id for s in surfaces}) < 2:
        return None
    surface_ids = {s.surface_id for s in surfaces}
    for group in sample.layout.surface_groups:
        if not group.objects or group.surface_id not in surface_ids:
            continue
        other = next(
            (s for s in surfaces if s.surface_id != group.surface_id), None
        )
        if other is None:
            continue
        repointed = group.model_copy(update={"surface_id": other.surface_id})
        return Injection(
            name="wrong_parent",
            bad_layout=_replace_group(sample.layout, repointed),
            expected_codes=(
                CODE_SURFACE_OOB,
                CODE_SURFACE_COLLISION,
                CODE_SURFACE_CAPACITY,
                CODE_SURFACE_OCCUPANCY,
                CODE_RELATION_VIOLATED,
            ),
            level=SURFACE_LEVEL,
            group_id=group.group_id,
        )
    return None


def _inject_surface_overflow(sample: FastFillSample) -> Optional[Injection]:
    """Duplicate small objects past the surface's capacity_max_objects."""
    target = _first_surface_target(sample)
    if target is None:
        return None
    group, _, surface = target
    on_surface = {
        o.object_id
        for g in sample.layout.surface_groups
        if g.surface_id == group.surface_id
        for o in g.objects
    }
    needed = surface.capacity_max_objects - len(on_surface) + 1
    if needed <= 0 or needed > _MAX_OVERFLOW_DUPLICATES:
        return None  # already at capacity (original would violate) / absurd
    base = group.objects[-1]
    duplicates = tuple(
        base.model_copy(update={"object_id": f"{base.object_id}+dup{i}"})
        for i in range(needed)
    )
    overflowed = group.model_copy(update={"objects": group.objects + duplicates})
    return Injection(
        name="surface_overflow",
        bad_layout=_replace_group(sample.layout, overflowed),
        expected_codes=(CODE_SURFACE_CAPACITY, CODE_BUDGET_SURFACE_OBJECTS),
        level=SURFACE_LEVEL,
        group_id=group.group_id,
    )


# --------------------------------------------------------- registry & runner

ALL_INJECTORS: dict[str, InjectorFn] = {
    "out_of_bounds": _inject_out_of_bounds,
    "collision_pair": _inject_collision_pair,
    "blocked_door": _inject_blocked_door,
    "wrong_facing": _inject_wrong_facing,
    "missing_required": _inject_missing_required,
    "floating_surface_object": _inject_floating_surface_object,
    "surface_off_polygon": _inject_surface_off_polygon,
    "wrong_parent": _inject_wrong_parent,
    "surface_overflow": _inject_surface_overflow,
}


def original_violation_codes(sample: FastFillSample) -> frozenset[str]:
    """Violation codes of the UNPERTURBED layout (compute once per sample)."""
    report = validate(sample.layout, sample.room_context)
    return frozenset(v.code for v in report.violations)


def verify_injection(
    sample: FastFillSample,
    injection: Injection,
    original_codes: Optional[frozenset[str]] = None,
) -> bool:
    """True iff the pair is a real preference signal, not noise.

    The perturbed layout must produce at least one expected code and the
    original layout must produce none of them (both re-derived with the
    vendored validator).
    """
    expected = set(injection.expected_codes)
    bad_report = validate(injection.bad_layout, sample.room_context)
    if not expected & {v.code for v in bad_report.violations}:
        return False
    if original_codes is None:
        original_codes = original_violation_codes(sample)
    return not (expected & set(original_codes))


def run_injector(
    name: str,
    sample: FastFillSample,
    original_codes: Optional[frozenset[str]] = None,
) -> Optional[Injection]:
    """Run one injector by name and keep only validator-verified negatives."""
    if name not in ALL_INJECTORS:
        raise KeyError(
            f"unknown injector {name!r}; known: {sorted(ALL_INJECTORS)}"
        )
    injection = ALL_INJECTORS[name](sample)
    if injection is None:
        return None
    if not verify_injection(sample, injection, original_codes):
        return None
    return injection
