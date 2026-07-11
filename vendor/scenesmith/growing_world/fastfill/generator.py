"""FastFill two-stage API generation (§2 call budget: 2 + <=1 repair).

``FastFillGenerator`` orchestrates the fixed call sequence: call 1 produces
floor furniture, the deterministic asset resolver binds real dimensions and
support-surface geometry, call 2 fills all surfaces in ONE batched call, and
``semantic_repair`` regenerates only failed groups in at most one extra call.
Each model call retries at most ``max_retries_per_call`` times on a
parse/schema failure (the error text is appended to the user message), then
raises :class:`GenerationError`.

``OpenAIChatBackend`` is the real backend; it imports ``openai`` lazily so
unit tests (which inject a fake ``LLMBackend``) never touch the package.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from pydantic import ValidationError

from scenesmith.growing_world.fastfill.interfaces import (
    AssetResolver,
    GenerationResult,
    GenerationTrace,
    LLMBackend,
)
from scenesmith.growing_world.fastfill.patterns import expand_patterns
from scenesmith.growing_world.fastfill.prompts import (
    build_floor_prompt,
    build_repair_prompt,
    build_surface_prompt,
)
from scenesmith.growing_world.fastfill.schema import (
    FloorLayout,
    FloorObjectSpec,
    OutputBudget,
    RoomContentLayout,
    RoomContext,
    SupportContext,
    SurfaceObjectGroup,
)

ENV_BASE_URL = "FASTFILL_LLM_BASE_URL"
ENV_API_KEY = "FASTFILL_LLM_API_KEY"
ENV_MODEL = "FASTFILL_LLM_MODEL"
LLM_CONFIG_ROLE = "fastfill"  # role key in growing_world llm_config models

_DEFAULT_SURFACE_MAX_TOKENS = OutputBudget().max_output_tokens_surface
_MAX_ERROR_CHARS = 600
_MAX_NEIGHBOR_CATEGORIES = 6

_T = TypeVar("_T")


class GenerationError(RuntimeError):
    """A model call failed to yield schema-valid JSON within the retry budget."""


# ------------------------------------------------------------ JSON extraction


def extract_first_json(text: str) -> object:
    """Parse the first balanced JSON array/object embedded in ``text``.

    Robust to markdown fences and surrounding prose: scans from the first
    ``[`` or ``{`` with string/escape awareness. Raises ``ValueError`` when
    no parsable JSON value is present.
    """
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        raise ValueError("no JSON array/object found in model output")
    start = min(starts)
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in model output")


# -------------------------------------------------------------- real backend


@dataclass(frozen=True)
class _BackendConfig:
    base_url: str
    api_key: str
    model: str


def _resolve_backend_config(
    base_url: str | None,
    api_key: str | None,
    model: str | None,
    config_path: str | None,
) -> _BackendConfig:
    """Explicit args > FASTFILL_LLM_* env > growing_world llm_config JSON."""
    base_url = base_url or os.environ.get(ENV_BASE_URL) or ""
    api_key = api_key or os.environ.get(ENV_API_KEY) or ""
    model = model or os.environ.get(ENV_MODEL) or ""
    if base_url and api_key and model:
        return _BackendConfig(base_url=base_url, api_key=api_key, model=model)
    from scenesmith.growing_world.llm import load_llm_config

    try:
        cfg = load_llm_config(config_path)
    except (FileNotFoundError, ValueError) as err:
        raise GenerationError(
            "LLM backend not configured: pass base_url/api_key/model "
            f"explicitly, set ${ENV_BASE_URL}/${ENV_API_KEY}/${ENV_MODEL}, "
            f"or provide a growing_world llm_config JSON ({err})"
        ) from err
    return _BackendConfig(
        base_url=base_url or cfg.base_url,
        api_key=api_key or cfg.api_key,
        model=model or cfg.model(LLM_CONFIG_ROLE),
    )


class OpenAIChatBackend:
    """OpenAI-compatible Chat Completions backend (:class:`LLMBackend`)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        config_path: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        cfg = _resolve_backend_config(base_url, api_key, model, config_path)
        # Lazy import: unit tests inject fake backends and must never
        # require the openai package to be installed.
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self._model = cfg.model
        self._temperature = temperature

    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""


# ----------------------------------------------------------------- generator


@dataclass(frozen=True)
class _CallStats:
    """Accounting for one retried model call (feeds GenerationTrace)."""

    calls: int = 0
    retries: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0


class FastFillGenerator:
    """The two fixed generation calls plus one optional semantic repair."""

    def __init__(
        self,
        llm: LLMBackend,
        resolver: AssetResolver,
        *,
        max_retries_per_call: int = 1,
    ) -> None:
        self._llm = llm
        self._resolver = resolver
        self._max_retries = max_retries_per_call

    # ------------------------------------------------------------- call 1

    def generate_floor(self, context: RoomContext) -> FloorLayout:
        """Call 1: floor furniture from the room context."""
        layout, _ = self._generate_floor(context)
        return layout

    def _generate_floor(self, context: RoomContext) -> tuple[FloorLayout, _CallStats]:
        system, user = build_floor_prompt(context)
        return self._call_with_retry(
            system,
            user,
            max_tokens=context.budget.max_output_tokens_floor,
            parse=lambda text: self._parse_floor(text, context.room_id),
        )

    @staticmethod
    def _parse_floor(text: str, room_id: str) -> FloorLayout:
        data = extract_first_json(text)
        if isinstance(data, dict):
            data = data.get("objects", data)
        if not isinstance(data, list) or not data:
            raise ValueError("expected a non-empty JSON array of floor objects")
        objects = tuple(FloorObjectSpec.model_validate(item) for item in data)
        ids = [o.object_id for o in objects]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate object_id values in floor output: {ids}")
        return FloorLayout(room_id=room_id, objects=objects)

    # ----------------------------------------------------------- resolver

    def resolve_and_contextualize(
        self, context: RoomContext, floor: FloorLayout
    ) -> tuple[FloorLayout, tuple[SupportContext, ...]]:
        """Bind resolver dims + surfaces; build call-2 contexts (budgeted)."""
        bound, contexts, _ = self._resolve_and_contextualize(context, floor)
        return bound, contexts

    def _resolve_and_contextualize(
        self, context: RoomContext, floor: FloorLayout
    ) -> tuple[FloorLayout, tuple[SupportContext, ...], tuple[str, ...]]:
        bound_objects: list[FloorObjectSpec] = []
        candidates: list[tuple[bool, SupportContext]] = []
        categories = [o.category for o in floor.objects]
        for obj in floor.objects:
            resolved = self._resolver.resolve(obj)
            bound_objects.append(
                obj.model_copy(update={"dimensions": resolved.dimensions})
            )
            if not self._object_wants_surfaces(obj):
                continue
            neighbors = tuple(c for c in categories if c != obj.category)[
                :_MAX_NEIGHBOR_CATEGORIES
            ]
            manipulands = context.expected_manipulands if obj.required_by_task else ()
            for surface in resolved.support_surfaces:
                candidates.append(
                    (
                        obj.required_by_task,
                        SupportContext(
                            surface=surface,
                            parent_category=obj.category,
                            parent_dimensions=resolved.dimensions,
                            room_type=context.room_type,
                            task=context.task,
                            expected_manipulands_here=manipulands,
                            neighbor_objects=neighbors,
                        ),
                    )
                )
        contexts, notes = self._prune_to_budget(context, candidates)
        bound_floor = floor.model_copy(
            update={
                "objects": tuple(bound_objects),
                "support_surfaces": tuple(c.surface for c in contexts),
            }
        )
        return bound_floor, contexts, notes

    def _object_wants_surfaces(self, obj: FloorObjectSpec) -> bool:
        if obj.wants_surface_fill:
            return True
        checker = getattr(self._resolver, "is_surface_bearing", None)
        return bool(checker(obj.category)) if callable(checker) else False

    @staticmethod
    def _prune_to_budget(
        context: RoomContext,
        candidates: Sequence[tuple[bool, SupportContext]],
    ) -> tuple[tuple[SupportContext, ...], tuple[str, ...]]:
        """Cap surfaces at the budget, task-required parents first; the
        dropped count is reported in trace notes (no silent caps)."""
        ordered = sorted(
            range(len(candidates)),
            key=lambda i: (not candidates[i][0], i),
        )
        limit = context.budget.max_support_surfaces_per_room
        kept = [candidates[i][1] for i in ordered[:limit]]
        dropped = len(candidates) - len(kept)
        if (
            kept
            and context.expected_manipulands
            and not any(c.expected_manipulands_here for c in kept)
        ):
            kept[0] = kept[0].model_copy(
                update={"expected_manipulands_here": context.expected_manipulands}
            )
        notes = (
            (
                f"pruned {dropped} support surfaces over "
                f"max_support_surfaces_per_room={limit} "
                "(task-required parents kept first)",
            )
            if dropped
            else ()
        )
        return tuple(kept), notes

    # ------------------------------------------------------------- call 2

    def generate_surfaces(
        self,
        contexts: list[SupportContext],
        *,
        max_tokens: int = _DEFAULT_SURFACE_MAX_TOKENS,
    ) -> list[SurfaceObjectGroup]:
        """Call 2: fill all support surfaces in ONE batched call."""
        groups, _ = self._generate_surfaces(contexts, max_tokens=max_tokens)
        return groups

    def _generate_surfaces(
        self,
        contexts: Sequence[SupportContext],
        *,
        max_tokens: int = _DEFAULT_SURFACE_MAX_TOKENS,
    ) -> tuple[list[SurfaceObjectGroup], _CallStats]:
        if not contexts:
            return [], _CallStats()
        system, user = build_surface_prompt(contexts)
        groups, stats = self._call_with_retry(
            system, user, max_tokens=max_tokens, parse=self._parse_groups
        )
        return list(expand_patterns(groups)), stats

    @staticmethod
    def _parse_groups(text: str) -> tuple[SurfaceObjectGroup, ...]:
        data = extract_first_json(text)
        if isinstance(data, dict):
            data = data.get("groups", data)
        if not isinstance(data, list):
            raise ValueError("expected a JSON array of surface object groups")
        return tuple(SurfaceObjectGroup.model_validate(item) for item in data)

    # ---------------------------------------------------------- full room

    def generate_room(self, context: RoomContext) -> GenerationResult:
        """Assemble the full room: 2 model calls on the happy path."""
        floor_raw, floor_stats = self._generate_floor(context)
        bound_floor, contexts, notes = self._resolve_and_contextualize(
            context, floor_raw
        )
        if contexts:
            groups, surface_stats = self._generate_surfaces(
                contexts, max_tokens=context.budget.max_output_tokens_surface
            )
        else:
            groups, surface_stats = [], _CallStats()
            notes += ("no surface contexts; surface call skipped",)
        layout = RoomContentLayout(
            room_id=context.room_id,
            floor_layout=bound_floor,
            surface_groups=tuple(groups),
            meta=context.meta,
        )
        trace = GenerationTrace(
            llm_calls=floor_stats.calls + surface_stats.calls,
            retries=floor_stats.retries + surface_stats.retries,
            prompt_chars=floor_stats.prompt_chars + surface_stats.prompt_chars,
            completion_chars=(
                floor_stats.completion_chars + surface_stats.completion_chars
            ),
            notes=notes,
        )
        return GenerationResult(
            layout=layout,
            floor_layout_raw=floor_raw,
            support_contexts=contexts,
            trace=trace,
        )

    # ------------------------------------------------------------- repair

    def semantic_repair(
        self,
        layout: RoomContentLayout,
        failed_surface_ids: Sequence[str],
        violations_text: str,
        contexts: Sequence[SupportContext],
    ) -> RoomContentLayout:
        """Regenerate ONLY the failed surfaces' groups (one model call)."""
        failed = set(failed_surface_ids)
        if not failed:
            return layout
        failed_groups = tuple(
            g for g in layout.surface_groups if g.surface_id in failed
        )
        relevant = tuple(c for c in contexts if c.surface.surface_id in failed)
        system, user = build_repair_prompt(failed_groups, violations_text, relevant)
        groups, _ = self._call_with_retry(
            system,
            user,
            max_tokens=_DEFAULT_SURFACE_MAX_TOKENS,
            parse=self._parse_groups,
        )
        kept = tuple(g for g in layout.surface_groups if g.surface_id not in failed)
        return layout.model_copy(
            update={"surface_groups": kept + expand_patterns(groups)}
        )

    # ------------------------------------------------------------ plumbing

    def _call_with_retry(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        parse: Callable[[str], _T],
    ) -> tuple[_T, _CallStats]:
        calls = retries = prompt_chars = completion_chars = 0
        current_user = user
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            text = self._llm.complete(
                system=system, user=current_user, max_tokens=max_tokens
            )
            calls += 1
            prompt_chars += len(system) + len(current_user)
            completion_chars += len(text)
            try:
                value = parse(text)
            except (ValueError, ValidationError) as err:
                last_error = err
                if attempt < self._max_retries:
                    retries += 1
                    current_user = (
                        f"{user}\n\nYour previous response was invalid:\n"
                        f"{str(err)[:_MAX_ERROR_CHARS]}\n"
                        "Return ONLY corrected JSON matching the required "
                        "shape."
                    )
                continue
            return value, _CallStats(
                calls=calls,
                retries=retries,
                prompt_chars=prompt_chars,
                completion_chars=completion_chars,
            )
        raise GenerationError(
            f"model output stayed invalid after {calls} attempts: {last_error}"
        )
