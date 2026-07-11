"""Protocols and small shared value types between FastFill components.

Kept import-light on purpose: generator, validator, repair, resolvers and
context backends all depend on this module plus ``schema.py`` and nothing
else, so any piece can be swapped (API model vs local model, canonical bbox
resolver vs real asset library, v1 hook backend vs v2 slot backend) without
touching the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from scenesmith.growing_world.fastfill.schema import (
    FloorLayout,
    FloorObjectSpec,
    RoomContentLayout,
    RoomContext,
    SupportContext,
    SupportSurfaceSpec,
    SurfaceObjectGroup,
    Vec3,
)


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal chat-completion surface FastFill needs.

    Implementations: OpenAI-compatible API client (V0), vLLM guided-JSON
    endpoint (V2) — both behind the same call shape.
    """

    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        """Return the raw completion text for one system+user exchange."""
        ...


@dataclass(frozen=True)
class ResolvedAsset:
    """Deterministic asset binding for one floor object (call-2 precondition).

    ``support_surfaces`` carries REAL geometry (or canonical proxies in V0);
    call 2 is always conditioned on these, never on model-estimated bboxes.
    """

    asset_id: str
    dimensions: Vec3  # bound asset's actual [width, depth, height]
    support_surfaces: tuple[SupportSurfaceSpec, ...] = ()
    source: str = "canonical"  # "canonical" | library name


@runtime_checkable
class AssetResolver(Protocol):
    """Maps a requested floor object to a concrete (or proxy) asset."""

    def resolve(self, obj: FloorObjectSpec) -> ResolvedAsset: ...


@runtime_checkable
class RoomContextBackend(Protocol):
    """Builds RoomContext from a host system (v1 hooks / v2 slots)."""

    def build(self) -> RoomContext: ...


@dataclass(frozen=True)
class GenerationTrace:
    """Book-keeping for one room generation (budget accounting, T1.10)."""

    llm_calls: int = 0
    retries: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    """Everything the pipeline produced for one room, pre-validation."""

    layout: RoomContentLayout
    floor_layout_raw: FloorLayout  # call-1 output before resolver rebinding
    support_contexts: tuple[SupportContext, ...]
    trace: GenerationTrace = field(default_factory=GenerationTrace)


@dataclass(frozen=True)
class RepairOutcome:
    """Deterministic-repair result. ``layout`` is a new object (immutability
    contract); ``failed_surface_ids`` feed the single batched semantic-repair
    call; ``gave_up`` marks layouts that stay failed after all budgets."""

    layout: RoomContentLayout
    applied_steps: tuple[str, ...] = ()
    failed_surface_ids: tuple[str, ...] = ()
    gave_up: bool = False


class FastFillGeneratorProtocol(Protocol):
    """The two fixed generation calls (§2 call budget: 2 + ≤1 repair)."""

    def generate_floor(self, context: RoomContext) -> FloorLayout: ...

    def generate_surfaces(
        self, contexts: list[SupportContext]
    ) -> list[SurfaceObjectGroup]: ...
