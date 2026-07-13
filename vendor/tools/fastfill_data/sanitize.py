"""Canonical label sanitizer for FastFill corpora.

Pipeline position:  convert -> deduplicate.py --contamination-list (raw
hashes: identical cross-source copies still collide, so the closure has
maximum recall) -> SANITIZE (restamps hashes from repaired geometry) ->
deduplicate.py again (post-repair hashes may newly collide) -> export.
Every downstream consumer (SFT export, DPO builders, eval ground truths)
must read the CLEANED samples this tool emits — cleaning only the exported
SFT text would let dirty labels re-enter through ``build_dpo_data`` stage
2, which renders positives from raw samples.

Per sample:

- FLOOR layer: quantize to codec precision (cm / whole degrees,
  id-preserving), validate floor-scope (semantic conditioning stripped —
  geometry only), apply bounded ``deterministic_repair`` rounds with
  re-quantization, revalidate, then gate on an encode -> decode ->
  revalidate round-trip. Samples whose floor stays invalid are DROPPED and
  counted per source x violation code.
- SURFACE layer: each group is validated in a synthetic single-parent room
  built from the REAL parent surface (floor violations of the original room
  cannot kill a valid surface group), repaired, revalidated, and codec
  round-tripped. Unrepairable groups are dropped and counted; ``z_local``
  is zeroed (the codec deliberately does not encode it).
- Provenance hashes + split key are RESTAMPED from the cleaned geometry
  (repair moves objects), and ``sanitized=v1`` is appended to the notes so
  the exporter can flag unsanitized inputs.

    python tools/fastfill_data/sanitize.py \
        --in out/conv/*.jsonl --out out/sanitized.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# Script-mode bootstrap: make `fastfill_data` (tools/) and `scenesmith`
# (repo root, not pip-installed) importable when run as a plain script.
for _extra in (
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2],
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from fastfill_data.common import read_jsonl  # noqa: E402
from scenesmith.growing_world.fastfill.codec import (  # noqa: E402
    decode_floor_layout,
    decode_surface_groups,
    encode_floor_layout,
    encode_surface_groups,
)
from scenesmith.growing_world.fastfill.provenance import stamp_hashes  # noqa: E402
from scenesmith.growing_world.fastfill.repair import (  # noqa: E402
    deterministic_repair,
)
from scenesmith.growing_world.fastfill.schema import (  # noqa: E402
    Anchor,
    FastFillSample,
    FloorLayout,
    FloorObjectSpec,
    OutputBudget,
    PatternParams,
    RoomContentLayout,
    RoomContext,
    SupportSurfaceSpec,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
    ValidationReport,
)
from scenesmith.growing_world.fastfill.transforms import normalize_deg  # noqa: E402
from scenesmith.growing_world.fastfill.validator import validate  # noqa: E402

SANITIZED_NOTE = "sanitized=v1"
MAX_REPAIR_ROUNDS = 3
_SCOPE_MARGIN_M = 5.0  # synthetic-room slack around the parent footprint


# ------------------------------------------------------- codec quantization


def _q_m(value: float) -> float:
    """Quantize meters to the codec's cm grid (matches encode->decode)."""
    return int(round(float(value) * 100.0)) / 100.0

def _q_vec2(v: Sequence[float]) -> tuple[float, float]:
    return (_q_m(v[0]), _q_m(v[1]))

def _q_vec3(v: Sequence[float]) -> tuple[float, float, float]:
    return (_q_m(v[0]), _q_m(v[1]), _q_m(v[2]))

def _q_deg(value: float) -> float:
    return float(normalize_deg(int(round(float(value)))))


def quantize_floor(layout: FloorLayout) -> FloorLayout:
    """Codec-precision copy of a floor layout, ORIGINAL object ids kept."""
    objects = tuple(
        obj.model_copy(
            update={
                "dimensions": _q_vec3(obj.dimensions),
                "position_xy": _q_vec2(obj.position_xy),
                "yaw_deg": _q_deg(obj.yaw_deg),
            }
        )
        for obj in layout.objects
    )
    return layout.model_copy(update={"objects": objects})


def quantize_group(group: SurfaceObjectGroup) -> SurfaceObjectGroup:
    """Codec-precision copy of a group; ``z_local`` zeroed (not encoded)."""
    objects = tuple(
        obj.model_copy(
            update={
                "dimensions": _q_vec3(obj.dimensions),
                "position_local": _q_vec2(obj.position_local),
                "yaw_deg_local": _q_deg(obj.yaw_deg_local),
                "z_local": 0.0,
            }
        )
        for obj in group.objects
    )
    params = group.pattern_params
    q_params = PatternParams(
        rows=params.rows,
        cols=params.cols,
        spacing_x_m=_q_m(params.spacing_x_m),
        spacing_y_m=_q_m(params.spacing_y_m),
        radius_m=_q_m(params.radius_m),
    )
    return group.model_copy(update={"objects": objects, "pattern_params": q_params})


# ------------------------------------------------------------- floor scope


def _geometry_only(context: RoomContext) -> RoomContext:
    """Strip semantic conditioning: the sanitizer judges geometry only."""
    return context.model_copy(
        update={"task": "", "expected_furniture": (), "expected_manipulands": ()}
    )


def _repair_until_pass(
    layout: RoomContentLayout, context: RoomContext
) -> tuple[RoomContentLayout, ValidationReport, ValidationReport]:
    """(final layout, pre report, post report) with bounded repair rounds."""
    pre = validate(layout, context)
    current, report = layout, pre
    for _ in range(MAX_REPAIR_ROUNDS):
        if report.passed:
            break
        outcome = deterministic_repair(current, report, context)
        requantized = outcome.layout.model_copy(
            update={
                "floor_layout": quantize_floor(outcome.layout.floor_layout),
                "surface_groups": tuple(
                    quantize_group(g) for g in outcome.layout.surface_groups
                ),
            }
        )
        if requantized == current:
            break  # no progress possible
        current = requantized
        report = validate(current, context)
    return current, pre, report


def sanitize_floor(
    sample: FastFillSample,
) -> tuple[FloorLayout | None, str, ValidationReport | None]:
    """(cleaned floor | None, drop_reason, pre-repair report)."""
    if not sample.layout.floor_layout.objects:
        return None, "empty_floor", None
    floor = quantize_floor(sample.layout.floor_layout)
    context = _geometry_only(sample.room_context)
    scope = RoomContentLayout(room_id=context.room_id, floor_layout=floor)
    final, pre, post = _repair_until_pass(scope, context)
    if not post.passed:
        return None, "unrepaired", pre
    cleaned = final.floor_layout
    try:
        decoded = decode_floor_layout(encode_floor_layout(cleaned), context.room_id)
    except ValueError:
        return None, "codec_error", pre
    round_trip = RoomContentLayout(room_id=context.room_id, floor_layout=decoded)
    if not validate(round_trip, context).passed:
        return None, "roundtrip_fail", pre
    return cleaned, "", pre


# ----------------------------------------------------------- surface scope


def build_surface_scope(
    surface: SupportSurfaceSpec,
    parent: FloorObjectSpec,
    groups: Sequence[SurfaceObjectGroup],
    room_type: str,
    budget: OutputBudget | None = None,
) -> tuple[RoomContentLayout, RoomContext]:
    """Synthetic single-parent room for surface-scoped validation.

    The REAL surface geometry and parent dimensions are kept; the parent is
    re-centered in a generous empty room so original floor-layer violations
    (parent OOB, floor collisions) cannot fail a valid surface group. ALL
    groups of one surface must be validated together — capacity, occupancy
    and cross-group collisions aggregate per surface. ``budget`` should be
    the sample's own room budget so L0 limits match the real room.
    """
    scoped_parent = parent.model_copy(
        update={
            "position_xy": (0.0, 0.0),
            "yaw_deg": 0.0,
            "anchor": Anchor.FREE,
            "required_by_task": False,
            "wants_surface_fill": (),
        }
    )
    half = max(parent.dimensions[0], parent.dimensions[1]) / 2.0 + _SCOPE_MARGIN_M
    base_budget = budget or OutputBudget()
    # A converter may declare a surface capacity above the room budget's
    # per-surface default (e.g. scenesmith scenes: max(12, len(members)));
    # the GT's own capacity is authoritative in surface scope, otherwise
    # valid labels get dropped on L0 budget violations.
    scope_budget = base_budget.model_copy(
        update={
            "max_surface_objects_per_surface": max(
                base_budget.max_surface_objects_per_surface,
                surface.capacity_max_objects,
            ),
            "max_surface_objects_total": max(
                base_budget.max_surface_objects_total,
                surface.capacity_max_objects,
            ),
        }
    )
    context = RoomContext(
        room_id="surface_scope",
        room_type=room_type or "room",
        floor_polygon=((-half, -half), (half, -half), (half, half), (-half, half)),
        budget=scope_budget,
    )
    layout = RoomContentLayout(
        room_id="surface_scope",
        floor_layout=FloorLayout(
            room_id="surface_scope",
            objects=(scoped_parent,),
            support_surfaces=(surface,),
        ),
        surface_groups=tuple(groups),
    )
    return layout, context


def surface_reports_with_repair(
    groups: Sequence[SurfaceObjectGroup],
    surface: SupportSurfaceSpec,
    parent: FloorObjectSpec,
    room_type: str,
    budget: OutputBudget | None = None,
) -> tuple[ValidationReport, ValidationReport, tuple[SurfaceObjectGroup, ...]]:
    """(pre report, post report, repaired groups) in the surface scope."""
    layout, context = build_surface_scope(surface, parent, groups, room_type, budget)
    final, pre, post = _repair_until_pass(layout, context)
    return pre, post, final.surface_groups


def _resolve_surface_parent(
    sample_floor: FloorLayout, group: SurfaceObjectGroup
) -> tuple[SupportSurfaceSpec, FloorObjectSpec] | None:
    surface = next(
        (
            s
            for s in sample_floor.support_surfaces
            if s.surface_id == group.surface_id
        ),
        None,
    )
    if surface is None:
        return None
    parent = next(
        (
            o
            for o in sample_floor.objects
            if o.object_id == surface.parent_object_id
        ),
        None,
    )
    if parent is None:
        return None
    return surface, parent


def sanitize_surface_set(
    groups: Sequence[SurfaceObjectGroup],
    surface: SupportSurfaceSpec,
    parent: FloorObjectSpec,
    room_type: str,
    budget: OutputBudget,
) -> tuple[tuple[SurfaceObjectGroup, ...], str, ValidationReport | None]:
    """Sanitize ALL groups of one surface together.

    (kept groups, drop_reason, pre-repair report) — validating groups one
    at a time would miss cross-group collisions/capacity on the same
    surface, so the whole per-surface set passes or is dropped as a unit.
    """
    quantized = tuple(quantize_group(g) for g in groups)
    pre, post, repaired = surface_reports_with_repair(
        quantized, surface, parent, room_type, budget
    )
    if not post.passed:
        return (), "unrepaired", pre
    wanted = {g.group_id for g in groups}
    kept = tuple(
        g for g in repaired if g.group_id in wanted and g.objects
    )
    if not kept:
        return (), "emptied_by_repair", pre
    try:
        decoded = decode_surface_groups(encode_surface_groups(kept))
    except ValueError:
        return (), "codec_error", pre
    rt_pre, _, _ = surface_reports_with_repair(
        decoded, surface, parent, room_type, budget
    )
    if not rt_pre.passed:
        return (), "roundtrip_fail", pre
    return kept, "", pre


# ------------------------------------------------------------- per sample


class _Stats:
    """Per-source accounting: nothing is dropped silently."""

    def __init__(self) -> None:
        self.per_source: dict[str, dict] = {}

    def _bucket(self, source: str) -> dict:
        return self.per_source.setdefault(
            source,
            {
                "input": 0,
                "kept": 0,
                "floor_repaired": 0,
                "floor_dropped": {},
                "floor_pre_violations": {},
                "groups_input": 0,
                "groups_kept": 0,
                "groups_repaired": 0,
                "groups_dropped": {},
                "groups_pre_violations": {},
            },
        )

    @staticmethod
    def _bump(counter: dict[str, int], key: str, n: int = 1) -> None:
        counter[key] = counter.get(key, 0) + n

    def record_violations(
        self, source: str, kind: str, report: ValidationReport | None
    ) -> None:
        if report is None or report.passed:
            return
        bucket = self._bucket(source)
        for violation in report.errors():
            self._bump(bucket[f"{kind}_pre_violations"], violation.code)
        bucket[f"{kind}_repaired"] += 1

    def to_dict(self) -> dict:
        totals = {
            "input": 0,
            "kept": 0,
            "groups_input": 0,
            "groups_kept": 0,
        }
        for bucket in self.per_source.values():
            for key in totals:
                totals[key] += bucket[key]
            bucket["floor_dropped"] = dict(sorted(bucket["floor_dropped"].items()))
            bucket["groups_dropped"] = dict(sorted(bucket["groups_dropped"].items()))
        return {
            "totals": totals,
            "per_source": dict(sorted(self.per_source.items())),
        }


def _with_sanitized_note(sample: FastFillSample) -> str:
    notes = sample.provenance.notes
    return f"{notes}; {SANITIZED_NOTE}" if notes else SANITIZED_NOTE


def sanitize_sample(
    sample: FastFillSample, stats: _Stats
) -> FastFillSample | None:
    """Cleaned sample, or ``None`` when the floor layer is unrepairable."""
    source = sample.provenance.source_dataset
    bucket = stats._bucket(source)
    bucket["input"] += 1

    floor, reason, pre = sanitize_floor(sample)
    if floor is None:
        stats._bump(bucket["floor_dropped"], reason)
        if pre is not None:
            for violation in pre.errors():
                stats._bump(bucket["floor_pre_violations"], violation.code)
        return None
    stats.record_violations(source, "floor", pre)

    # Groups sharing a surface validate as ONE set (cross-group collisions,
    # capacity and occupancy aggregate per surface).
    by_surface: dict[str, list[SurfaceObjectGroup]] = {}
    for group in sample.layout.surface_groups:
        bucket["groups_input"] += 1
        if not group.objects:
            stats._bump(bucket["groups_dropped"], "empty")
            continue
        if _resolve_surface_parent(floor, group) is None:
            stats._bump(bucket["groups_dropped"], "unresolved")
            continue
        by_surface.setdefault(group.surface_id, []).append(group)

    kept_groups: list[SurfaceObjectGroup] = []
    for surface_id in sorted(by_surface):
        groups = by_surface[surface_id]
        surface, parent = _resolve_surface_parent(floor, groups[0])  # type: ignore[misc]
        kept, group_reason, group_pre = sanitize_surface_set(
            groups,
            surface,
            parent,
            sample.room_context.room_type,
            sample.room_context.budget,
        )
        if not kept:
            stats._bump(bucket["groups_dropped"], group_reason, n=len(groups))
            if group_pre is not None:
                for violation in group_pre.errors():
                    stats._bump(bucket["groups_pre_violations"], violation.code)
            continue
        stats.record_violations(source, "groups", group_pre)
        dropped = len(groups) - len(kept)
        if dropped:
            stats._bump(bucket["groups_dropped"], "emptied_by_repair", n=dropped)
        bucket["groups_kept"] += len(kept)
        kept_groups.extend(kept)

    provenance = sample.provenance.model_copy(
        update={"notes": _with_sanitized_note(sample)}
    )
    stamped = stamp_hashes(
        provenance, sample.room_context.floor_polygon, floor.objects
    )
    bucket["kept"] += 1
    return sample.model_copy(
        update={
            "layout": sample.layout.model_copy(
                update={
                    "floor_layout": floor,
                    "surface_groups": tuple(kept_groups),
                    "meta": stamped,
                }
            ),
            "provenance": stamped,
        }
    )


def sanitize(inputs: Sequence[Path], out_path: Path) -> dict:
    """Stream every input into cleaned JSONL; write and return the report."""
    stats = _Stats()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for path in inputs:
            for sample in read_jsonl(path):
                cleaned = sanitize_sample(sample, stats)
                if cleaned is not None:
                    out.write(cleaned.model_dump_json() + "\n")
    report = {
        "inputs": [str(p) for p in inputs],
        "max_repair_rounds": MAX_REPAIR_ROUNDS,
        **stats.to_dict(),
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanitize FastFill JSONL labels (validate/repair/round-trip)"
    )
    parser.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        required=True,
        help="converted FastFillSample JSONL files",
    )
    parser.add_argument("--out", required=True, help="cleaned JSONL path")
    args = parser.parse_args()

    out_path = Path(args.out)
    report = sanitize([Path(p) for p in args.inputs], out_path)
    print(f"report -> {out_path.with_suffix('.report.json')}")
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
