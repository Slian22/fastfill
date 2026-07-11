"""Build DPO preference pairs for the FastFill planner (two stages).

``stage2`` — programmatic hard negatives: for each ``FastFillSample`` the
injectors in ``injectors.py`` perturb the ground-truth layout into a
validator-failing variant. ``chosen`` is the record rendered exactly like SFT
(vendored ``export_sft`` instructions + codec encoders, completion side via
``templates.render_sft_example``); ``rejected`` is the same prompt with the
injected layout encoded by the same codec/template. Every emitted pair is
validator-verified (perturbed layout fails with an expected code, original
produces none of them) — unverifiable or unrenderable attempts are counted,
never silently dropped.

``stage1`` — model self-negatives: pairs a floor SFT record (``--contexts``)
with a model completion (``--generations``, rows ``{"uid", "completion"}``
from ``eval_layout --dump-generations``). ``rejected`` is kept ONLY when the
completion fails to parse (rejected = raw text) or parses but fails the
vendored validator while the ground truth passes. When ``--samples`` provides
the paired ``FastFillSample`` the RoomContext is rebuilt for full validation
("validated" mode, ``expected_manipulands`` stripped since floor records
carry no surface objects); otherwise only parse success can be checked
("parse_only" mode) — the stats file records which mode judged each row.

Usage::

    PYTHONPATH=src python3 -m fastfill_train.build_dpo_data stage2 \
        --in data/samples.jsonl --out data/dpo/stage2_pairs.jsonl \
        --template direct --per-sample 2 --seed 42

    PYTHONPATH=src python3 -m fastfill_train.build_dpo_data stage1 \
        --contexts data/sft/floor_sft.jsonl --generations gens.jsonl \
        --out data/dpo/stage1_pairs.jsonl --samples data/samples.jsonl

A ``<out>.stats.json`` report is written next to ``--out`` for both stages.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, TextIO

import fastfill_train  # noqa: F401  (vendor path bootstrap)
from fastfill_data.export_sft import (
    FLOOR_INSTRUCTION,
    SURFACE_INSTRUCTION,
    build_support_context,
)
from scenesmith.growing_world.fastfill.codec import (
    decode_floor_layout,
    encode_floor_layout,
    encode_room_context,
    encode_support_context,
    encode_surface_groups,
)
from scenesmith.growing_world.fastfill.schema import (
    FastFillSample,
    FloorLayout,
    RoomContentLayout,
    RoomContext,
    SurfaceObjectGroup,
    ValidationReport,
)
from scenesmith.growing_world.fastfill.validator import validate

from fastfill_train.data import read_records
from fastfill_train.injectors import (
    ALL_INJECTORS,
    FLOOR_LEVEL,
    Injection,
    original_violation_codes,
    verify_injection,
)
from fastfill_train.templates import TEMPLATES, render_sft_example, split_completion

DEFAULT_PER_SAMPLE = 2
DEFAULT_SEED = 42


def _bump(counter: dict[str, int], key: str, n: int = 1) -> None:
    counter[key] = counter.get(key, 0) + n


def _load_samples(path: str | Path) -> list[FastFillSample]:
    return [FastFillSample.model_validate(r) for r in read_records([path])]


def _write_stats(out_path: Path, stats: dict) -> Path:
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats_path


# -------------------------------------------------------------------- stage 2


def _floor_record(sample: FastFillSample) -> dict:
    """SFT floor record (same shape as vendored ``export_sft._floor_record``)."""
    return {
        "uid": sample.sample_id,
        "split_key": sample.provenance.split_key,
        "source_dataset": sample.provenance.source_dataset,
        "instruction": FLOOR_INSTRUCTION,
        "input": encode_room_context(sample.room_context),
        "output": encode_floor_layout(sample.layout.floor_layout),
    }


def _surface_record(sample: FastFillSample, group: SurfaceObjectGroup) -> Optional[dict]:
    """SFT surface record for one group; ``None`` if its context is unresolved."""
    context = build_support_context(sample, group)
    if context is None:
        return None
    return {
        "uid": f"{sample.sample_id}#{group.group_id}",
        "split_key": sample.provenance.split_key,
        "source_dataset": sample.provenance.source_dataset,
        "instruction": SURFACE_INSTRUCTION,
        "input": encode_support_context(context),
        "output": encode_surface_groups([group]),
    }


def _pair_record(
    sample: FastFillSample, injection: Injection
) -> tuple[Optional[dict], Optional[str], str]:
    """(chosen SFT record, rejected output codec text, skip_reason)."""
    if injection.level == FLOOR_LEVEL:
        record = _floor_record(sample)
        bad_output = encode_floor_layout(injection.bad_layout.floor_layout)
        return record, bad_output, ""
    group = next(
        g for g in sample.layout.surface_groups if g.group_id == injection.group_id
    )
    record = _surface_record(sample, group)
    if record is None:
        return None, None, "surface_context_unresolved"
    bad_group = next(
        g
        for g in injection.bad_layout.surface_groups
        if g.group_id == injection.group_id
    )
    return record, encode_surface_groups([bad_group]), ""


def _stage2_row(
    sample: FastFillSample, injection: Injection, template: str
) -> tuple[Optional[dict], str]:
    """Render one DPO pair row; ``(None, reason)`` when it must be skipped."""
    try:
        record, bad_output, reason = _pair_record(sample, injection)
    except ValueError:  # codec-illegal token (e.g. '|' in a category)
        return None, "codec_error"
    if record is None or bad_output is None:
        return None, reason
    room_id = sample.room_context.room_id
    chosen = render_sft_example(record, template, room_id=room_id)
    rejected = render_sft_example(
        {**record, "output": bad_output}, template, room_id=room_id
    )
    if chosen.assistant == rejected.assistant:
        return None, "identical_completion"  # e.g. z_local is not encoded
    row = {
        "prompt": [{"role": "user", "content": chosen.user}],
        "chosen": [{"role": "assistant", "content": chosen.assistant}],
        "rejected": [{"role": "assistant", "content": rejected.assistant}],
        "split_key": sample.provenance.split_key,
        "uid": record["uid"],
        "injector": injection.name,
        "expected_codes": list(injection.expected_codes),
    }
    return row, ""


def _emit_sample_pairs(
    sample: FastFillSample,
    rotated_names: list[str],
    template: str,
    per_sample: int,
    out: TextIO,
    per_injector: dict[str, int],
    skipped: dict[str, int],
) -> int:
    """Try injectors in round-robin order until ``per_sample`` pairs emitted."""
    original_codes = original_violation_codes(sample)
    emitted = 0
    for name in rotated_names:
        if emitted >= per_sample:
            break
        injection = ALL_INJECTORS[name](sample)
        if injection is None:
            _bump(skipped, f"not_applicable:{name}")
            continue
        if not verify_injection(sample, injection, original_codes):
            _bump(skipped, f"verification_failed:{name}")
            continue
        row, reason = _stage2_row(sample, injection, template)
        if row is None:
            _bump(skipped, f"{reason}:{name}")
            continue
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        _bump(per_injector, name)
        emitted += 1
    return emitted


def run_stage2(
    in_path: Path, out_path: Path, template: str, per_sample: int, seed: int
) -> dict:
    """Build stage-2 pairs from FastFillSample JSONL; return the stats dict."""
    names = sorted(ALL_INJECTORS)
    random.Random(seed).shuffle(names)
    counts: dict[str, int] = {"input_samples": 0, "pairs": 0}
    per_injector: dict[str, int] = {}
    skipped: dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for index, sample in enumerate(_load_samples(in_path)):
            counts["input_samples"] += 1
            start = index % len(names)
            rotated = names[start:] + names[:start]
            emitted = _emit_sample_pairs(
                sample, rotated, template, per_sample, out, per_injector, skipped
            )
            counts["pairs"] += emitted
            if emitted == 0:
                _bump(skipped, "sample_yielded_no_pairs")
            elif emitted < per_sample:
                _bump(skipped, "sample_underfilled")
    stats = {
        "stage": "stage2",
        "in_path": str(in_path),
        "out_path": str(out_path),
        "template": template,
        "per_sample": per_sample,
        "seed": seed,
        "counts": counts,
        "per_injector": dict(sorted(per_injector.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    _write_stats(out_path, stats)
    return stats


# -------------------------------------------------------------------- stage 1


def _floor_only_report(
    codec_text: str, context: RoomContext
) -> ValidationReport:
    """Validate a floor-codec completion as a floor-only room layout."""
    floor = decode_floor_layout(codec_text, context.room_id)
    layout = RoomContentLayout(room_id=context.room_id, floor_layout=floor)
    return validate(layout, context)


def _judge_generation(
    record: dict, completion: str, sample: Optional[FastFillSample]
) -> tuple[str, list[str]]:
    """Classify one model completion against its ground-truth record.

    Returns ``(verdict, violation_codes)`` with verdict one of:
    ``reject_parse_failure``, ``reject_validation_failure``,
    ``skip_generation_passes``, ``skip_ground_truth_fails``,
    ``skip_parse_only_passes``.
    """
    _, layout_text = split_completion(completion)
    room_id = sample.room_context.room_id if sample else "eval"
    try:
        decode_floor_layout(layout_text, room_id)
    except (ValueError, IndexError):
        return "reject_parse_failure", []
    if sample is None:
        return "skip_parse_only_passes", []
    # Floor records carry no surface objects, so expected_manipulands would
    # fail for ground truth and generation alike — strip them for both sides.
    context = sample.room_context.model_copy(update={"expected_manipulands": ()})
    if not _floor_only_report(record["output"], context).passed:
        return "skip_ground_truth_fails", []
    gen_report = _floor_only_report(layout_text, context)
    if gen_report.passed:
        return "skip_generation_passes", []
    return "reject_validation_failure", [v.code for v in gen_report.violations]


def _stage1_row(
    record: dict, completion: str, template: str, verdict: str, codes: list[str]
) -> dict:
    chosen = render_sft_example(record, template)
    return {
        "prompt": [{"role": "user", "content": chosen.user}],
        "chosen": [{"role": "assistant", "content": chosen.assistant}],
        "rejected": [{"role": "assistant", "content": completion}],
        "split_key": record.get("split_key", ""),
        "uid": record["uid"],
        "reason": verdict.removeprefix("reject_"),
        "violation_codes": codes,
    }


def run_stage1(
    contexts_path: Path,
    generations_path: Path,
    out_path: Path,
    samples_path: Optional[Path],
    template: str,
) -> dict:
    """Pair SFT ground truths with failing model completions; return stats."""
    contexts = {r["uid"]: r for r in read_records([contexts_path])}
    samples: dict[str, FastFillSample] = {}
    if samples_path is not None:
        samples = {s.sample_id: s for s in _load_samples(samples_path)}
    counts: dict[str, int] = {"generations": 0, "pairs": 0}
    modes: dict[str, int] = {"validated": 0, "parse_only": 0}
    skipped: dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for gen in read_records([generations_path]):
            if "uid" not in gen or "completion" not in gen:
                raise ValueError(f"generation row needs uid+completion: {gen!r}")
            counts["generations"] += 1
            record = contexts.get(gen["uid"])
            if record is None:
                _bump(skipped, "unknown_uid")
                continue
            sample = samples.get(gen["uid"])
            _bump(modes, "validated" if sample is not None else "parse_only")
            verdict, codes = _judge_generation(record, gen["completion"], sample)
            if not verdict.startswith("reject_"):
                _bump(skipped, verdict.removeprefix("skip_"))
                continue
            row = _stage1_row(record, gen["completion"], template, verdict, codes)
            if row["chosen"][0]["content"] == row["rejected"][0]["content"]:
                _bump(skipped, "identical_completion")
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["pairs"] += 1
    stats = {
        "stage": "stage1",
        "contexts_path": str(contexts_path),
        "generations_path": str(generations_path),
        "samples_path": str(samples_path) if samples_path else None,
        "out_path": str(out_path),
        "template": template,
        "counts": counts,
        "judge_modes": modes,
        "judge_mode_note": (
            "validated = RoomContext rebuilt from --samples, full validator; "
            "parse_only = no matching sample, only parse success checked "
            "(decodable completions are counted as passing)"
        ),
        "skipped": dict(sorted(skipped.items())),
    }
    _write_stats(out_path, stats)
    return stats


# ------------------------------------------------------------------------ CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fastfill_train.build_dpo_data",
        description="Build FastFill DPO preference pairs (stage1 / stage2)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p2 = sub.add_parser("stage2", help="injector-based hard negatives")
    p2.add_argument("--in", dest="in_path", required=True, help="samples JSONL")
    p2.add_argument("--out", required=True, help="output pairs JSONL")
    p2.add_argument("--template", choices=TEMPLATES, default="direct")
    p2.add_argument("--per-sample", type=int, default=DEFAULT_PER_SAMPLE)
    p2.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p1 = sub.add_parser("stage1", help="model self-negatives vs ground truth")
    p1.add_argument("--contexts", required=True, help="floor SFT JSONL")
    p1.add_argument("--generations", required=True, help="{uid,completion} JSONL")
    p1.add_argument("--out", required=True, help="output pairs JSONL")
    p1.add_argument(
        "--samples",
        default=None,
        help="FastFillSample JSONL for full validation (else parse-only mode)",
    )
    p1.add_argument("--template", choices=TEMPLATES, default="direct")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "stage2":
        if args.per_sample < 1:
            raise SystemExit("--per-sample must be >= 1")
        stats = run_stage2(
            Path(args.in_path),
            Path(args.out),
            args.template,
            args.per_sample,
            args.seed,
        )
    else:
        stats = run_stage1(
            Path(args.contexts),
            Path(args.generations),
            Path(args.out),
            Path(args.samples) if args.samples else None,
            args.template,
        )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
