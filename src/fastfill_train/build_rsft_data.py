"""RSFT (rejection-sampling fine-tuning) data builder with anti-collapse gates.

Samples K candidates per context (produced by ``eval_layout
--dump-generations`` run K times, or any {"uid","completion"} JSONL with
repeated uids), keeps only candidates that clear HARD gates, ranks the
survivors, and emits new SFT records from the model's own best outputs.

Hard gates (any failure -> candidate rejected, counted):
1. parses via the codec;
2. passes the validator pre-repair (covers expected-furniture coverage,
   collisions, bounds, door clearance);
3. object count >= ``min_objects`` AND >= ``min_gt_ratio`` x ground-truth
   count — the anti-sparse-collapse floor (validator-picked selection would
   otherwise teach "fewer objects is safer").

Collapse alarm: if the fraction of contexts with zero surviving candidates
exceeds ``max_dry_rate``, stats carry ``collapse_alarm: true`` and the
process exits with code 2 — stop the RSFT round instead of training on a
collapsed distribution. Mean object-count delta vs ground truth is always
reported so shrinkage is visible before it crosses the gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import fastfill_train  # noqa: F401  (vendor path bootstrap)
from fastfill_train.build_dpo_data import (
    _bump,
    _floor_only_report,
    _load_samples,
    _write_stats,
)
from fastfill_train.data import read_records
from fastfill_train.templates import split_completion
from scenesmith.growing_world.fastfill.codec import decode_floor_layout
from scenesmith.growing_world.fastfill.schema import FastFillSample

DEFAULT_MIN_OBJECTS = 3
DEFAULT_MIN_GT_RATIO = 0.6
DEFAULT_MAX_DRY_RATE = 0.5


def _floor_context(sample: FastFillSample):
    """Floor-only RoomContext (expected_manipulands stripped, like stage 1)."""
    return sample.room_context.model_copy(update={"expected_manipulands": ()})


def _candidate_score(
    n_objects: int, n_gt: int, completion: str
) -> tuple[int, int]:
    """Rank key (lower = better): object-count distance to GT, then length."""
    return (abs(n_objects - n_gt), len(completion))


def run_rsft(
    contexts_path: Path,
    generations_path: Path,
    samples_path: Path,
    out_path: Path,
    *,
    min_objects: int = DEFAULT_MIN_OBJECTS,
    min_gt_ratio: float = DEFAULT_MIN_GT_RATIO,
    max_dry_rate: float = DEFAULT_MAX_DRY_RATE,
) -> dict:
    """Select the best gate-passing candidate per context; return stats."""
    records = {r["uid"]: r for r in read_records([contexts_path])}
    samples = {s.sample_id: s for s in _load_samples(samples_path)}
    by_uid: dict[str, list[str]] = {}
    for gen in read_records([generations_path]):
        by_uid.setdefault(str(gen["uid"]), []).append(str(gen["completion"]))

    counts: dict[str, int] = {"contexts": 0, "candidates": 0, "selected": 0}
    rejects: dict[str, int] = {}
    object_deltas: list[int] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for uid, completions in by_uid.items():
            record, sample = records.get(uid), samples.get(uid)
            if record is None or sample is None:
                _bump(rejects, "unknown_uid", len(completions))
                continue
            counts["contexts"] += 1
            n_gt = len(sample.layout.floor_layout.objects)
            context = _floor_context(sample)
            floor_min = max(min_objects, int(min_gt_ratio * n_gt))
            survivors: list[tuple[tuple[int, int], int, str]] = []
            for completion in completions:
                counts["candidates"] += 1
                _, codec_text = split_completion(completion)
                try:
                    layout = decode_floor_layout(codec_text, context.room_id)
                except Exception:  # noqa: BLE001 — codec parse is the gate
                    _bump(rejects, "gate_parse")
                    continue
                n_objects = len(layout.objects)
                if n_objects < floor_min:
                    _bump(rejects, "gate_object_count_floor")
                    continue
                if not _floor_only_report(codec_text, context).passed:
                    _bump(rejects, "gate_validator")
                    continue
                survivors.append(
                    (_candidate_score(n_objects, n_gt, completion), n_objects, completion)
                )
            if not survivors:
                _bump(rejects, "context_dry")
                continue
            survivors.sort(key=lambda item: item[0])
            _, n_best, best = survivors[0]
            object_deltas.append(n_best - n_gt)
            out.write(
                json.dumps(
                    {
                        "uid": f"{uid}::rsft",  # NEVER '#': that marks surface records
                        "split_key": record.get("split_key", ""),
                        "source_dataset": record.get("source_dataset", ""),
                        "instruction": record["instruction"],
                        "input": record["input"],
                        "output": best,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts["selected"] += 1

    dry = rejects.get("context_dry", 0)
    dry_rate = dry / counts["contexts"] if counts["contexts"] else 1.0
    stats = {
        "stage": "rsft",
        "counts": counts,
        "rejects": rejects,
        "gates": {
            "min_objects": min_objects,
            "min_gt_ratio": min_gt_ratio,
            "max_dry_rate": max_dry_rate,
        },
        "dry_rate": round(dry_rate, 4),
        "mean_object_delta_vs_gt": (
            round(sum(object_deltas) / len(object_deltas), 3)
            if object_deltas
            else None
        ),
        "collapse_alarm": dry_rate > max_dry_rate,
        "note": (
            "collapse_alarm=true means too few contexts produced any "
            "gate-passing candidate — do NOT train on this round; raise K, "
            "fix the sampler, or stop RSFT"
        ),
    }
    _write_stats(out_path, stats)
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="RSFT data builder: validator-gated best-of-K selection"
    )
    parser.add_argument("--contexts", required=True, help="floor SFT JSONL")
    parser.add_argument(
        "--generations", required=True, help="{uid,completion} JSONL, K rows/uid"
    )
    parser.add_argument("--samples", required=True, help="FastFillSample JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-objects", type=int, default=DEFAULT_MIN_OBJECTS)
    parser.add_argument("--min-gt-ratio", type=float, default=DEFAULT_MIN_GT_RATIO)
    parser.add_argument("--max-dry-rate", type=float, default=DEFAULT_MAX_DRY_RATE)
    args = parser.parse_args(argv)
    stats = run_rsft(
        Path(args.contexts),
        Path(args.generations),
        Path(args.samples),
        Path(args.out),
        min_objects=args.min_objects,
        min_gt_ratio=args.min_gt_ratio,
        max_dry_rate=args.max_dry_rate,
    )
    print(json.dumps(stats, indent=2))
    if stats["collapse_alarm"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
