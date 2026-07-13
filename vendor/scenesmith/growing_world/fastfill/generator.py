"""FastFill generation over the training line codec (per-support calls).

``FastFillGenerator`` orchestrates the call sequence: call 1 produces floor
furniture (line codec, one call per room), the deterministic asset resolver
binds real dimensions and support-surface geometry, then ONE call PER
support surface fills that surface — matching the one-support-per-record
SFT shape — and ``semantic_repair`` regenerates failed surfaces with one
extra call per failed surface. Model output is parsed with the SAME codec
used to build the training labels (``decode_floor_layout`` /
``decode_surface_groups``, plan prefixes stripped first); JSON is gone from
the runtime protocol. Each model call retries at most
``max_retries_per_call`` times on a parse/schema failure (the error text is
appended to the user message), then raises :class:`GenerationError`.

Per-surface failures follow ``surface_failure_mode``: ``"skip"`` (default)
records the failure in the trace notes and leaves that surface empty —
production world growth survives; ``"raise"`` propagates (smoke/CI
fail-fast).

Budget note: ``OutputBudget.max_output_tokens_surface`` is the cap for EACH
per-surface call (the historical single batched surface call no longer
exists).

``OpenAIChatBackend`` is the real backend; it imports ``openai`` lazily so
unit tests (which inject a fake ``LLMBackend``) never touch the package.
An empty system prompt is NOT sent (training used a single user turn), the
default temperature is 0.0 (deterministic eval parity), and Qwen thinking
is explicitly disabled via ``chat_template_kwargs`` (override with
``$FASTFILL_LLM_EXTRA_BODY``, e.g. ``{}`` for servers that reject the
field).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from pydantic import ValidationError

from scenesmith.growing_world.fastfill.codec import (
    decode_floor_layout,
    decode_surface_groups,
    strip_plan_prefix,
)
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
ENV_EXTRA_BODY = "FASTFILL_LLM_EXTRA_BODY"
LLM_CONFIG_ROLE = "fastfill"  # role key in growing_world llm_config models

# Qwen chat templates default to thinking mode; training completions carry
# no thinking, so it must be off for parity (vLLM honors this kwarg).
DEFAULT_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}

_DEFAULT_SURFACE_MAX_TOKENS = OutputBudget().max_output_tokens_surface
_MAX_ERROR_CHARS = 600
_MAX_NEIGHBOR_CATEGORIES = 6
_SURFACE_FAILURE_MODES = ("skip", "raise")

_T = TypeVar("_T")


class GenerationError(RuntimeError):
    """A model call failed to yield codec-valid output within the retry budget."""


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


def _resolve_extra_body(extra_body: dict | None) -> dict:
    if extra_body is not None:
        return extra_body
    env = os.environ.get(ENV_EXTRA_BODY)
    if env is not None:
        try:
            parsed = json.loads(env)
        except json.JSONDecodeError as err:
            raise GenerationError(
                f"${ENV_EXTRA_BODY} is not valid JSON: {err}"
            ) from err
        if not isinstance(parsed, dict):
            raise GenerationError(f"${ENV_EXTRA_BODY} must be a JSON object")
        return parsed
    return DEFAULT_EXTRA_BODY


class OpenAIChatBackend:
    """OpenAI-compatible Chat Completions backend (:class:`LLMBackend`)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        config_path: str | None = None,
        temperature: float = 0.0,
        extra_body: dict | None = None,
    ) -> None:
        cfg = _resolve_backend_config(base_url, api_key, model, config_path)
        # Lazy import: unit tests inject fake backends and must never
        # require the openai package to be installed.
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self._model = cfg.model
        self._temperature = temperature
        self._extra_body = _resolve_extra_body(extra_body)

    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        # Training rendered a single user turn; an empty system prompt is
        # omitted entirely so the served prompt matches training bytes.
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        kwargs: dict = {}
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self._temperature,
            **kwargs,
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            # The line codec is prefix-closed: a truncated completion still
            # decodes, silently losing tail objects. Fail as a parse error
            # so _call_with_retry retries instead of accepting it.
            raise ValueError(
                f"completion truncated at max_tokens={max_tokens} "
                "(finish_reason=length)"
            )
        return choice.message.content or ""


# ----------------------------------------------------------------- generator


@dataclass(frozen=True)
class _CallStats:
    """Accounting for one retried model call (feeds GenerationTrace)."""

    calls: int = 0
    retries: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0

    def __add__(self, other: "_CallStats") -> "_CallStats":
        return _CallStats(
            calls=self.calls + other.calls,
            retries=self.retries + other.retries,
            prompt_chars=self.prompt_chars + other.prompt_chars,
            completion_chars=self.completion_chars + other.completion_chars,
        )


class FastFillGenerator:
    """Floor call + per-surface calls + per-failed-surface semantic repair."""

    def __init__(
        self,
        llm: LLMBackend,
        resolver: AssetResolver,
        *,
        max_retries_per_call: int = 1,
        surface_failure_mode: str = "skip",
    ) -> None:
        if surface_failure_mode not in _SURFACE_FAILURE_MODES:
            raise ValueError(
                f"surface_failure_mode must be one of {_SURFACE_FAILURE_MODES}"
            )
        self._llm = llm
        self._resolver = resolver
        self._max_retries = max_retries_per_call
        self._surface_failure_mode = surface_failure_mode

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
        _, layout_text = strip_plan_prefix(text)
        floor = decode_floor_layout(layout_text, room_id)
        if not floor.objects:
            raise ValueError("expected at least one floor object line")
        return floor

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

    # ---------------------------------------------------- per-surface calls

    @staticmethod
    def _parse_groups(text: str) -> tuple[SurfaceObjectGroup, ...]:
        _, layout_text = strip_plan_prefix(text)
        groups = decode_surface_groups(layout_text)
        if not groups:
            raise ValueError("expected at least one group header line")
        if not any(group.objects for group in groups):
            raise ValueError("expected at least one surface object line")
        return groups

    @staticmethod
    def _rebind_groups(
        groups: Sequence[SurfaceObjectGroup], surface_id: str, prefix: str
    ) -> tuple[SurfaceObjectGroup, ...]:
        """Pin groups to the requested surface and uniquify ids.

        The model may echo a wrong surface_id, repeat a group id within one
        completion, and every training label starts at ``g0`` — so ids are
        namespaced ``<prefix>_<position>_<model id>``: unique within the
        call by position, unique room-wide by the per-call prefix.
        """
        rebound: list[SurfaceObjectGroup] = []
        for position, group in enumerate(groups):
            new_gid = f"{prefix}_{position}_{group.group_id}"
            objects = tuple(
                obj.model_copy(
                    update={
                        "object_id": (
                            f"{new_gid}/{obj.object_id.split('/', 1)[-1]}"
                        )
                    }
                )
                for obj in group.objects
            )
            anchor = group.anchor_object_id
            rebound.append(
                group.model_copy(
                    update={
                        "group_id": new_gid,
                        "surface_id": surface_id,
                        "objects": objects,
                        "anchor_object_id": (
                            f"{new_gid}/{anchor.split('/', 1)[-1]}" if anchor else ""
                        ),
                    }
                )
            )
        return tuple(rebound)

    def generate_surfaces(
        self,
        contexts: list[SupportContext],
        *,
        max_tokens: int = _DEFAULT_SURFACE_MAX_TOKENS,
    ) -> list[SurfaceObjectGroup]:
        """Fill every support surface, ONE model call per surface."""
        groups, _, _ = self._generate_surfaces(contexts, max_tokens=max_tokens)
        return groups

    def _generate_surfaces(
        self,
        contexts: Sequence[SupportContext],
        *,
        max_tokens: int = _DEFAULT_SURFACE_MAX_TOKENS,
    ) -> tuple[list[SurfaceObjectGroup], _CallStats, tuple[str, ...]]:
        all_groups: list[SurfaceObjectGroup] = []
        stats = _CallStats()
        notes: tuple[str, ...] = ()
        for index, context in enumerate(contexts):
            surface_id = context.surface.surface_id
            system, user = build_surface_prompt(context)
            try:
                groups, call_stats = self._call_with_retry(
                    system, user, max_tokens=max_tokens, parse=self._parse_groups
                )
            except GenerationError as err:
                if self._surface_failure_mode == "raise":
                    raise
                failed_stats = getattr(err, "stats", None)
                stats = stats + (
                    failed_stats
                    if isinstance(failed_stats, _CallStats)
                    else _CallStats(calls=self._max_retries + 1)
                )
                notes += (f"surface '{surface_id}' generation failed: {err}",)
                continue
            stats = stats + call_stats
            rebound = self._rebind_groups(groups, surface_id, f"s{index}")
            all_groups.extend(expand_patterns(rebound))
        return all_groups, stats, notes

    # ---------------------------------------------------------- full room

    def generate_room(self, context: RoomContext) -> GenerationResult:
        """Assemble the full room: 1 floor call + 1 call per surface."""
        floor_raw, floor_stats = self._generate_floor(context)
        bound_floor, contexts, notes = self._resolve_and_contextualize(
            context, floor_raw
        )
        if contexts:
            groups, surface_stats, surface_notes = self._generate_surfaces(
                contexts, max_tokens=context.budget.max_output_tokens_surface
            )
            notes += surface_notes
        else:
            groups, surface_stats = [], _CallStats()
            notes += ("no surface contexts; surface calls skipped",)
        layout = RoomContentLayout(
            room_id=context.room_id,
            floor_layout=bound_floor,
            surface_groups=tuple(groups),
            meta=context.meta,
        )
        total = floor_stats + surface_stats
        trace = GenerationTrace(
            llm_calls=total.calls,
            retries=total.retries,
            prompt_chars=total.prompt_chars,
            completion_chars=total.completion_chars,
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
        """Regenerate failed surfaces' groups, one model call per surface.

        A surface whose repair call fails (or that has no matching context)
        keeps its original groups under ``surface_failure_mode="skip"``;
        ``"raise"`` propagates the failure.
        """
        failed = set(failed_surface_ids)
        if not failed:
            return layout
        by_surface = {c.surface.surface_id: c for c in contexts}
        kept = list(
            g for g in layout.surface_groups if g.surface_id not in failed
        )
        for index, surface_id in enumerate(sorted(failed)):
            old_groups = tuple(
                g for g in layout.surface_groups if g.surface_id == surface_id
            )
            context = by_surface.get(surface_id)
            if context is None:
                if self._surface_failure_mode == "raise":
                    raise GenerationError(
                        f"no support context for failed surface '{surface_id}'"
                    )
                kept.extend(old_groups)
                continue
            system, user = build_repair_prompt(
                old_groups, violations_text, context
            )
            try:
                groups, _ = self._call_with_retry(
                    system,
                    user,
                    max_tokens=_DEFAULT_SURFACE_MAX_TOKENS,
                    parse=self._parse_groups,
                )
            except GenerationError:
                if self._surface_failure_mode == "raise":
                    raise
                kept.extend(old_groups)
                continue
            rebound = self._rebind_groups(groups, surface_id, f"r{index}")
            kept.extend(expand_patterns(rebound))
        return layout.model_copy(update={"surface_groups": tuple(kept)})

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
            except (ValueError, IndexError, ValidationError) as err:
                last_error = err
                if attempt < self._max_retries:
                    retries += 1
                    current_user = (
                        f"{user}\n\nYour previous response was invalid:\n"
                        f"{str(err)[:_MAX_ERROR_CHARS]}\n"
                        "Output ONLY the corrected layout lines in the exact "
                        "plain-text line format required by the instruction "
                        "— no JSON, no prose, no markdown."
                    )
                continue
            return value, _CallStats(
                calls=calls,
                retries=retries,
                prompt_chars=prompt_chars,
                completion_chars=completion_chars,
            )
        error = GenerationError(
            f"model output stayed invalid after {calls} attempts: {last_error}"
        )
        # Real call accounting must survive the failure (trace correctness).
        error.stats = _CallStats(  # type: ignore[attr-defined]
            calls=calls,
            retries=retries,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars,
        )
        raise error
