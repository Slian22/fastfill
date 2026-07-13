"""Prompt pairs (system, user) for FastFill's model calls (line codec).

TRAINING PARITY IS THE CONTRACT: the served model is fine-tuned on
``export_sft`` records rendered as a single user message
``<instruction>\\n<codec input>`` with NO system message, so the runtime
builders emit exactly that shape (system = "" — the backend then sends only
the user turn). Instructions and codec come from ``codec.py``, the single
source both training export and runtime read, so the two sides cannot
drift.

Calls are PER TARGET: one floor call per room and one surface call PER
support surface, matching the one-support-per-record training shape. The
repair prompt reuses the surface instruction plus the failed groups and the
violation text; it has no training analog and relies on the model
generalizing the codec.
"""

from __future__ import annotations

from typing import Sequence

from scenesmith.growing_world.fastfill.codec import (
    FLOOR_INSTRUCTION,
    SURFACE_INSTRUCTION,
    encode_room_context,
    encode_support_context,
    encode_surface_groups,
)
from scenesmith.growing_world.fastfill.schema import (
    RoomContext,
    SupportContext,
    SurfaceObjectGroup,
)


def _training_floor_context(ctx: RoomContext) -> RoomContext:
    """Project onto the v1 training distribution.

    Every converter leaves task/expected/style empty, so 100% of floor
    training prompts render ``task -`` / ``furn -`` / ``manip -``. Populated
    values would be out-of-distribution tokens for the fine-tuned model —
    stripped here until the exporter populates them (v1.2 task
    conditioning). Geometry (doors/windows/portals/forbidden) stays: it
    appears in training whenever the source has it.
    """
    return ctx.model_copy(
        update={
            "task": "",
            "expected_furniture": (),
            "expected_manipulands": (),
            "style_hint": "",
        }
    )


def _training_support_context(context: SupportContext) -> SupportContext:
    """Project onto the v1 training distribution (see floor variant)."""
    return context.model_copy(
        update={
            "task": "",
            "expected_manipulands_here": (),
            "neighbor_objects": (),
            "desired_groups": (),
        }
    )


def build_floor_prompt(ctx: RoomContext) -> tuple[str, str]:
    """(system, user) for call 1 — byte-identical to the training prompt."""
    encoded = encode_room_context(_training_floor_context(ctx))
    return "", f"{FLOOR_INSTRUCTION}\n{encoded}"


def build_surface_prompt(context: SupportContext) -> tuple[str, str]:
    """(system, user) for ONE surface call — matches training records."""
    encoded = encode_support_context(_training_support_context(context))
    return "", f"{SURFACE_INSTRUCTION}\n{encoded}"


def build_repair_prompt(
    failed_groups: Sequence[SurfaceObjectGroup],
    violations_text: str,
    context: SupportContext,
) -> tuple[str, str]:
    """(system, user) for one per-surface semantic-repair call."""
    encoded = encode_support_context(_training_support_context(context))
    user = (
        f"{SURFACE_INSTRUCTION}\n"
        f"{encoded}\n"
        "The previous placement below failed validation. Output a corrected "
        "placement for this surface in the same line format (group header "
        "line then one object per line), fixing the violations.\n"
        "PREVIOUS PLACEMENT:\n"
        f"{encode_surface_groups(failed_groups)}\n"
        "VIOLATIONS:\n"
        f"{violations_text}"
    )
    return "", user
