"""RoomContext builders: compile growing_world shell data into FastFill input.

Two entry points:

- :func:`build_room_context` — pure compiler from a ``PlacedRoom``-shaped
  object (duck-typed so this module never imports the Drake-bound
  ``agent_utils.house`` chain) + task/expected metadata into a
  :class:`RoomContext`.
- :class:`FastFillContentHook` — the v1 ``ContentHooks`` backend (T1.7):
  writes ``room_context.json`` per room and, when an LLM config is
  available, runs the full generate → validate → repair loop and writes
  ``room_content_layout.json`` + ``validation_report.json``.

The v2 worldedge slot backend compiles to the SAME RoomContext schema
(T1.9 alignment); nothing downstream may depend on which backend produced it.
"""

from __future__ import annotations

import json
import logging
import os

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from scenesmith.growing_world.hooks import ContentHooks, PopulateContext
from scenesmith.growing_world.fastfill.schema import (
    DoorSpec,
    NeighborSummary,
    OutputBudget,
    RoomContext,
    WindowSpec,
)

if TYPE_CHECKING:
    from scenesmith.agent_utils.house import RoomGeometry

console_logger = logging.getLogger(__name__)

_DOOR_CLEARANCE_M = 0.9
_MAX_REPAIR_ROUNDS = 3  # mirrors eval_layout / fastfill_api_smoke / sanitize


class _StrictValidationFailure(RuntimeError):
    """Raised in strict mode when a room stays invalid after full repair
    (already counted in failure_count when raised)."""


def _room_center(placed: Any) -> tuple[float, float]:
    """Global center of a PlacedRoom (position is the min corner)."""
    return (
        placed.position[0] + placed.width / 2.0,
        placed.position[1] + placed.depth / 2.0,
    )


def _opening_center(wall: Any, opening: Any) -> tuple[float, float]:
    """Global center of an opening: lerp along the wall segment.

    ``position_along_wall`` is the left-edge distance from ``start_point``
    (see ``portals.cut_opening``), so the center sits half a width further.
    """
    t = (opening.position_along_wall + opening.width / 2.0) / wall.length
    sx, sy = wall.start_point
    ex, ey = wall.end_point
    return (sx + (ex - sx) * t, sy + (ey - sy) * t)


def _extract_openings(
    placed: Any,
) -> tuple[tuple[DoorSpec, ...], tuple[WindowSpec, ...]]:
    """Doors/windows in the ROOM frame from the placed room's wall openings."""
    cx, cy = _room_center(placed)
    doors: list[DoorSpec] = []
    windows: list[WindowSpec] = []
    for wall in placed.walls:
        direction = str(wall.direction.value)
        for opening in wall.openings:
            gx, gy = _opening_center(wall, opening)
            center = (gx - cx, gy - cy)
            kind = str(opening.opening_type.value).lower()
            if kind == "door":
                faces = list(getattr(wall, "faces_rooms", []) or [])
                doors.append(
                    DoorSpec(
                        door_id=str(opening.opening_id),
                        center_xy=center,
                        width_m=float(opening.width),
                        clearance_depth_m=_DOOR_CLEARANCE_M,
                        wall=direction,  # type: ignore[arg-type]
                        connects_room_id=faces[0] if faces else "",
                    )
                )
            elif kind == "window":
                windows.append(
                    WindowSpec(
                        window_id=str(opening.opening_id),
                        center_xy=center,
                        width_m=float(opening.width),
                        sill_height_m=float(opening.sill_height),
                        height_m=float(opening.height),
                        wall=direction,  # type: ignore[arg-type]
                    )
                )
    return tuple(doors), tuple(windows)


def _compile_expected(task: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Best-effort expected buckets from the task via the Task Compiler.

    Import is lazy and failure-tolerant: fastfill must work (with empty
    expected lists) even where task_conditioning's dependencies are absent.
    """
    if not task:
        return (), ()
    try:
        from scenesmith.task_conditioning import compile_task

        req = compile_task(task)
        if not req.is_actionable:
            return (), ()
        return tuple(req.required_furniture), tuple(req.required_objects)
    except Exception as exc:  # noqa: BLE001 — advisory enrichment only
        console_logger.warning("task compile failed (%s); expected lists empty", exc)
        return (), ()


def build_room_context(
    *,
    room_id: str,
    room_type: str,
    placed_room: Any = None,
    fallback_dims: tuple[float, float] | None = None,
    ceiling_height_m: float = 2.5,
    task: str | None = None,
    neighbors: Iterable[tuple[str, str, str]] = (),
    expected_furniture: tuple[str, ...] | None = None,
    expected_manipulands: tuple[str, ...] | None = None,
    budget: OutputBudget | None = None,
) -> RoomContext:
    """Compile shell data into the frozen RoomContext contract.

    ``placed_room`` gives exact door/window coordinates; without it,
    ``fallback_dims`` (length_x, depth_y) yields a bare rectangle context.
    When expected lists are not supplied they are compiled from ``task``.
    """
    if placed_room is not None:
        half_l = placed_room.width / 2.0
        half_d = placed_room.depth / 2.0
        doors, windows = _extract_openings(placed_room)
    elif fallback_dims is not None:
        half_l, half_d = fallback_dims[0] / 2.0, fallback_dims[1] / 2.0
        doors, windows = (), ()
    else:
        raise ValueError("need placed_room or fallback_dims")

    if expected_furniture is None or expected_manipulands is None:
        furn, manip = _compile_expected(task)
        expected_furniture = furn if expected_furniture is None else expected_furniture
        expected_manipulands = (
            manip if expected_manipulands is None else expected_manipulands
        )

    return RoomContext(
        room_id=room_id,
        room_type=room_type,
        floor_polygon=(
            (-half_l, -half_d),
            (half_l, -half_d),
            (half_l, half_d),
            (-half_l, half_d),
        ),
        ceiling_height_m=ceiling_height_m,
        doors=doors,
        windows=windows,
        neighbors=tuple(
            NeighborSummary(room_id=nid, room_type=ntype, direction=ndir)  # type: ignore[arg-type]
            for nid, ntype, ndir in neighbors
        ),
        task=task or "",
        expected_furniture=tuple(expected_furniture),
        expected_manipulands=tuple(expected_manipulands),
        budget=budget or OutputBudget(),
    )


class FastFillContentHook(ContentHooks):
    """v1 fastfill backend for the growth loop (T1.7 smoke: three JSONs).

    Always writes ``fastfill/room_context.json``. With ``generate=True`` it
    runs the fixed-call pipeline (generate → validate → deterministic repair
    → revalidate) and writes ``room_content_layout.json`` +
    ``validation_report.json``.

    Failure policy: every generation failure writes a structured
    ``generation_error.json`` and increments ``failure_count`` — a room can
    never stay empty silently. With ``strict=False`` (production default)
    the growth loop survives; ``strict=True`` (or ``$FASTFILL_STRICT=1``)
    re-raises so smoke runs and CI fail fast.
    """

    def __init__(
        self,
        world_dir: Path,
        *,
        generate: bool = False,
        strict: bool | None = None,
        llm_config_path: str | None = None,
    ) -> None:
        self.world_dir = Path(world_dir)
        self.generate = generate
        if strict is None:
            strict = os.environ.get("FASTFILL_STRICT", "") not in ("", "0", "false")
        self.strict = strict
        self.llm_config_path = llm_config_path
        self.failure_count = 0
        self.success_count = 0

    def populate_room(
        self,
        *,
        room_id: str,
        room_type: str,
        room_geometry: "RoomGeometry",
        prompt: str,
        context: PopulateContext | None = None,
    ) -> None:
        out_dir = self.world_dir / "rooms" / room_id / "fastfill"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctx = build_room_context(
            room_id=room_id,
            room_type=room_type,
            placed_room=context.placed_room if context else None,
            fallback_dims=(room_geometry.length, room_geometry.width),
            task=context.task if context else None,
            neighbors=context.neighbors if context else (),
        )
        (out_dir / "room_context.json").write_text(ctx.model_dump_json(indent=2))
        console_logger.info(
            "[fastfill] room '%s': context written (%d doors, %d windows)",
            room_id,
            len(ctx.doors),
            len(ctx.windows),
        )
        if self.generate:
            self._generate_content(ctx, out_dir)

    def _generate_content(self, ctx: RoomContext, out_dir: Path) -> None:
        """Fixed-call content pipeline; imports deferred (heavy/optional)."""
        try:
            from dataclasses import asdict

            from scenesmith.growing_world.fastfill.asset_resolver import (
                CanonicalAssetResolver,
            )
            from scenesmith.growing_world.fastfill.generator import (
                FastFillGenerator,
                OpenAIChatBackend,
            )
            from scenesmith.growing_world.fastfill.repair import (
                deterministic_repair,
            )
            from scenesmith.growing_world.fastfill.validator import validate

            generator = FastFillGenerator(
                llm=OpenAIChatBackend(config_path=self.llm_config_path),
                resolver=CanonicalAssetResolver(),
                surface_failure_mode="raise" if self.strict else "skip",
            )
            result = generator.generate_room(ctx)
            # Iterated repair -> revalidate, same depth as the eval harness
            # and smoke script (a single round would under-repair rooms the
            # acceptance metrics certified as passing).
            layout = result.layout
            final_report = validate(layout, ctx)
            repair_steps: list[str] = []
            for _ in range(_MAX_REPAIR_ROUNDS):
                if final_report.passed:
                    break
                outcome = deterministic_repair(layout, final_report, ctx)
                repair_steps.extend(outcome.applied_steps)
                if outcome.layout == layout:
                    break  # no progress possible
                layout = outcome.layout
                final_report = validate(layout, ctx)
            # Surfaces still failing after deterministic repair get one
            # semantic-repair pass (per-surface model calls), same as the
            # smoke script — production must not run a weaker pipeline than
            # the one the acceptance metrics were measured on.
            semantic_used = False
            failed_surfaces = sorted(
                {v.surface_id for v in final_report.errors() if v.surface_id}
            )
            if failed_surfaces:
                violations_text = "\n".join(
                    f"{v.code} [{v.surface_id}]: {v.message}"
                    for v in final_report.errors()
                    if v.surface_id
                )
                layout = generator.semantic_repair(
                    layout,
                    tuple(failed_surfaces),
                    violations_text,
                    result.support_contexts,
                )
                semantic_used = True
                final_report = validate(layout, ctx)
            # A skipped surface (non-strict mode swallows per-surface
            # GenerationErrors into trace notes) is a generation failure
            # even when the PARTIAL layout validates — the validator never
            # requires every declared surface to have output, so
            # passed=True with missing surfaces must not count as success.
            surface_failures = tuple(
                note
                for note in result.trace.notes
                if "generation failed" in note
            )
            room_success = final_report.passed and not surface_failures
            (out_dir / "room_content_layout.json").write_text(
                layout.model_dump_json(indent=2)
            )
            (out_dir / "validation_report.json").write_text(
                final_report.model_dump_json(indent=2)
            )
            (out_dir / "generation_trace.json").write_text(
                json.dumps(
                    {
                        "room_id": ctx.room_id,
                        "trace": asdict(result.trace),
                        "repair_steps": repair_steps,
                        "semantic_repair_used": semantic_used,
                        "semantic_repair_surfaces": failed_surfaces,
                        "passed": final_report.passed,
                        "surface_failures": len(surface_failures),
                        "room_success": room_success,
                        "violations": len(final_report.violations),
                    },
                    indent=2,
                )
            )
            console_logger.info(
                "[fastfill] room '%s': %s (%d violations, %d surface "
                "failures, llm_calls=%d)",
                ctx.room_id,
                "PASSED" if room_success else "FAILED",
                len(final_report.violations),
                len(surface_failures),
                result.trace.llm_calls,
            )
            if not room_success:
                # A room that stays invalid after the full repair ladder —
                # or lost surfaces to generation failures — is a failure,
                # not a quiet FAILED file on disk.
                self.failure_count += 1
                if self.strict:
                    raise _StrictValidationFailure(
                        f"room '{ctx.room_id}' failed: "
                        f"{len(final_report.errors())} errors, "
                        f"{len(surface_failures)} surface generation failures"
                    )
            else:
                self.success_count += 1
        except Exception as exc:  # noqa: BLE001 — counted + surfaced, never silent
            if not isinstance(exc, _StrictValidationFailure):
                self.failure_count += 1
            console_logger.exception(
                "[fastfill] room '%s': generation failed (failure #%d)",
                ctx.room_id,
                self.failure_count,
            )
            (out_dir / "generation_error.json").write_text(
                json.dumps(
                    {
                        "room_id": ctx.room_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "strict": self.strict,
                    },
                    indent=2,
                )
            )
            if self.strict:
                raise


def dump_context_json(ctx: RoomContext) -> str:
    """Stable JSON for logs/tests."""
    return json.dumps(json.loads(ctx.model_dump_json()), indent=2, sort_keys=True)
