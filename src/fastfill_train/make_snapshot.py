"""Freeze a stage-0 snapshot: house-first split + leakage check + hashes.

Replaces the (wrong) ``shuf | head`` runbook step. Splits by ``split_key``
hash (same rule training uses) into train / heldout / TEST, writes the
JSONL files + a sidecar with source hashes, seed, rule and per-source-layer
counts, and HARD-FAILS if any split_key lands on more than one side.

The TEST split (``--test-fraction``, default 0.05) is the final untouched
set: heldout is for learning curves and tuning decisions; test is read once
at the very end. Mix floor + surface inputs to decide a unified base; pass
floor only for an explicitly Floor-only decision (recorded in the sidecar).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import fastfill_train  # noqa: F401
from fastfill_train.data import read_records, split_bucket
from fastfill_train.templates import record_layer as _layer

_BUCKETS = 10_000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _assign(record: dict, val_fraction: float, test_fraction: float, seed: int) -> str:
    key = record.get("split_key")
    if not key:
        # A uid fallback would bucket a room's floor and surface records
        # independently (uid vs uid#group) — silent house leakage. The
        # post-fix pipeline stamps split_key on every record; missing keys
        # mean a stale or broken export.
        raise SystemExit(
            f"record {record.get('uid')!r} has no split_key — re-export with "
            "the current pipeline before freezing a snapshot"
        )
    bucket = split_bucket(key, seed)
    if bucket < int(val_fraction * _BUCKETS):
        return "heldout"
    if bucket < int((val_fraction + test_fraction) * _BUCKETS):
        return "test"
    return "train"


def _source_layer_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in rows:
        key = f"{record.get('source_dataset', '')}/{_layer(record)}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="freeze stage-0 snapshot")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help="final untouched test split (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "heldout": [], "test": []}
    for record in read_records(args.inputs):
        splits[_assign(record, args.val_fraction, args.test_fraction, args.seed)].append(
            record
        )
    train = splits["train"]
    if args.max_train and len(train) > args.max_train:
        rng = random.Random(args.seed)
        rng.shuffle(train)  # subsample AFTER the house split — no leakage
        splits["train"] = train[: args.max_train]

    keys = {
        name: {r.get("split_key") for r in rows} - {None, ""}
        for name, rows in splits.items()
    }
    for a, b in (("train", "heldout"), ("train", "test"), ("heldout", "test")):
        overlap = keys[a] & keys[b]
        if overlap:
            raise SystemExit(
                f"HOUSE LEAKAGE: {len(overlap)} split_keys in both {a} and {b}"
            )

    files: dict[str, Path] = {}
    for name, rows in splits.items():
        if name == "test" and args.test_fraction <= 0:
            continue
        path = out / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        files[name] = path
    sidecar = {
        "inputs": {p: _sha(Path(p)) for p in args.inputs},
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "rule": "house-first sha1(split_key) bucket (data.split_bucket)",
        "counts": {name: len(splits[name]) for name in files},
        "per_source_layer": {
            name: _source_layer_counts(splits[name]) for name in files
        },
        "hashes": {name: _sha(path) for name, path in files.items()},
        "leakage_check": "passed",
        "test_split_policy": (
            "heldout drives learning curves and tuning; test.jsonl is read "
            "ONCE for the final report"
        ),
    }
    (out / "SNAPSHOT.json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar["counts"]), "->", out)


if __name__ == "__main__":
    main()
