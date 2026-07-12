"""Freeze a stage-0 snapshot: house-first split + leakage check + hashes.

Replaces the (wrong) ``shuf | head`` runbook step. Splits by ``split_key``
hash (same rule training uses), writes train/heldout JSONL + a sidecar with
source hashes, seed, rule and counts, and HARD-FAILS if any split_key lands
on both sides. Mix floor + surface inputs to decide a unified base; pass
floor only for an explicitly Floor-only decision (recorded in the sidecar).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import fastfill_train  # noqa: F401
from fastfill_train.data import is_validation_key, read_records


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="freeze stage-0 snapshot")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, heldout = [], []
    for record in read_records(args.inputs):
        key = record.get("split_key") or record.get("uid", "")
        (heldout if is_validation_key(key, args.val_fraction, args.seed) else train).append(record)
    if args.max_train and len(train) > args.max_train:
        rng = random.Random(args.seed)
        rng.shuffle(train)  # subsample AFTER the house split — no leakage
        train = train[: args.max_train]

    overlap = {r.get("split_key") for r in train} & {
        r.get("split_key") for r in heldout
    } - {None, ""}
    if overlap:
        raise SystemExit(f"HOUSE LEAKAGE: {len(overlap)} split_keys on both sides")

    for name, rows in (("train.jsonl", train), ("heldout.jsonl", heldout)):
        (out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    sidecar = {
        "inputs": {p: _sha(Path(p)) for p in args.inputs},
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "rule": "house-first sha1(split_key) bucket (data.is_validation_key)",
        "counts": {"train": len(train), "heldout": len(heldout)},
        "layers": sorted({("surface" if "#" in str(r.get("uid","")) else "floor") for r in train}),
        "train_sha": _sha(out / "train.jsonl"),
        "heldout_sha": _sha(out / "heldout.jsonl"),
        "leakage_check": "passed",
    }
    (out / "SNAPSHOT.json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar["counts"]), "->", out)


if __name__ == "__main__":
    main()
