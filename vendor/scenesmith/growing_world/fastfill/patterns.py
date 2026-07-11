"""Deterministic expansion of structured surface-group patterns (v1.4 §2.1).

The model only *declares* a pattern primitive (matrix / paired / circular)
with parameters; concrete object placements are produced here, so pattern
geometry is exactly reproducible and never hallucinated.

Conventions (all in the support-surface local frame):

* the pattern center is the ``anchor_object_id`` object's position when that
  object is in the group (the anchor keeps its own pose), else ``(0, 0)``;
* MATRIX: row-major grid centered on the pattern center, row 0 at +Y
  (columns step +X by ``spacing_x_m``, rows step -Y by ``spacing_y_m``);
  a single non-anchor template is replicated to fill all ``rows x cols``
  cells, otherwise the given non-anchor objects fill cells in order and
  extras keep their positions;
* PAIRED: two placements at center +- (spacing_x/2, 0); one template is
  duplicated (ids ``_a``/``_b``), spacing 0 falls back to 1.5x object width;
* CIRCULAR: the non-anchor objects sit evenly on the circle of
  ``radius_m``, starting at +Y and proceeding CCW, each yawed to face the
  pattern center;
* FREE: passthrough.

Every function returns NEW objects (``model_copy``), never mutates input.
"""

from __future__ import annotations

import math
from typing import Sequence

from scenesmith.growing_world.fastfill.schema import (
    GroupPattern,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
    Vec2,
)
from scenesmith.growing_world.fastfill.transforms import normalize_deg

PAIRED_SPACING_WIDTH_FACTOR = 1.5  # fallback when spacing_x_m is 0


def _pattern_center(group: SurfaceObjectGroup) -> Vec2:
    for obj in group.objects:
        if obj.object_id == group.anchor_object_id:
            return obj.position_local
    return (0.0, 0.0)


def _split_anchor(
    group: SurfaceObjectGroup,
) -> tuple[tuple[SurfaceObjectSpec, ...], tuple[SurfaceObjectSpec, ...]]:
    anchors = tuple(o for o in group.objects if o.object_id == group.anchor_object_id)
    rest = tuple(o for o in group.objects if o.object_id != group.anchor_object_id)
    return anchors, rest


def _at(
    obj: SurfaceObjectSpec, pos: Vec2, object_id: str | None = None
) -> SurfaceObjectSpec:
    update: dict = {"position_local": pos}
    if object_id is not None:
        update["object_id"] = object_id
    return obj.model_copy(update=update)


def _expand_matrix(group: SurfaceObjectGroup) -> tuple[SurfaceObjectSpec, ...]:
    params = group.pattern_params
    rows, cols = max(params.rows, 1), max(params.cols, 1)
    cx, cy = _pattern_center(group)
    anchors, rest = _split_anchor(group)
    cells: list[Vec2] = [
        (
            cx + (c - (cols - 1) / 2.0) * params.spacing_x_m,
            cy + ((rows - 1) / 2.0 - r) * params.spacing_y_m,
        )
        for r in range(rows)
        for c in range(cols)
    ]
    if len(rest) == 1 and len(cells) > 1:
        template = rest[0]
        placed = tuple(
            _at(
                template,
                cells[r * cols + c],
                f"{template.object_id}_r{r}c{c}",
            )
            for r in range(rows)
            for c in range(cols)
        )
        return anchors + placed
    placed = tuple(
        _at(obj, cells[i]) if i < len(cells) else obj for i, obj in enumerate(rest)
    )
    return anchors + placed


def _expand_paired(group: SurfaceObjectGroup) -> tuple[SurfaceObjectSpec, ...]:
    anchors, rest = _split_anchor(group)
    if not rest:
        return group.objects
    cx, cy = _pattern_center(group)
    spacing = group.pattern_params.spacing_x_m
    if spacing <= 0.0:
        spacing = rest[0].dimensions[0] * PAIRED_SPACING_WIDTH_FACTOR
    left: Vec2 = (cx - spacing / 2.0, cy)
    right: Vec2 = (cx + spacing / 2.0, cy)
    if len(rest) == 1:
        template = rest[0]
        placed = (
            _at(template, left, f"{template.object_id}_a"),
            _at(template, right, f"{template.object_id}_b"),
        )
    else:
        placed = (_at(rest[0], left), _at(rest[1], right)) + rest[2:]
    return anchors + placed


def _expand_circular(group: SurfaceObjectGroup) -> tuple[SurfaceObjectSpec, ...]:
    anchors, rest = _split_anchor(group)
    if not rest:
        return group.objects
    cx, cy = _pattern_center(group)
    radius = group.pattern_params.radius_m
    count = len(rest)
    placed = []
    for i, obj in enumerate(rest):
        angle_deg = 90.0 + 360.0 * i / count
        rad = math.radians(angle_deg)
        pos: Vec2 = (cx + radius * math.cos(rad), cy + radius * math.sin(rad))
        # front (+Y at yaw 0) toward the center: yaw = angle + 90 (derived).
        placed.append(
            obj.model_copy(
                update={
                    "position_local": pos,
                    "yaw_deg_local": normalize_deg(angle_deg + 90.0),
                }
            )
        )
    return anchors + tuple(placed)


_EXPANDERS = {
    GroupPattern.MATRIX: _expand_matrix,
    GroupPattern.PAIRED: _expand_paired,
    GroupPattern.CIRCULAR: _expand_circular,
}


def expand_group(group: SurfaceObjectGroup) -> SurfaceObjectGroup:
    """Expand one group's pattern into concrete placements (FREE: passthrough)."""
    expander = _EXPANDERS.get(group.pattern)
    if expander is None:
        return group
    return group.model_copy(update={"objects": expander(group)})


def expand_patterns(
    groups: Sequence[SurfaceObjectGroup],
) -> tuple[SurfaceObjectGroup, ...]:
    """Deterministically expand every non-FREE group in ``groups``."""
    return tuple(expand_group(g) for g in groups)
