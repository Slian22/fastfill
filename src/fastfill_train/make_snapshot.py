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


def _license_counts(rows: list[dict]) -> dict[str, int]:
    """License lineage per split — a permissive snapshot with any non-
    permissive tag here was frozen from the wrong export route."""
    counts: dict[str, int] = {}
    for record in rows:
        tag = record.get("license") or "unrecorded"
        counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


_LICENSE_MODES = ("permissive", "research")
_ALLOWED_BY_MODE = {
    "permissive": frozenset({"permissive"}),
    "research": frozenset({"permissive", "cc_by_nc"}),
}


def _enforce_license_mode(
    splits: dict[str, list[dict]], license_mode: str, allow_unresolved: bool
) -> None:
    """Fail-closed license gate at freeze time: a permissive snapshot with
    NC / pending / missing-license records must never come into existence —
    counting alone lets the wrong export route freeze silently."""
    allowed = set(_ALLOWED_BY_MODE[license_mode])
    if allow_unresolved and license_mode == "research":
        allowed |= {"license_pending", "unknown"}
    bad: dict[str, int] = {}
    for rows in splits.values():
        for record in rows:
            tag = record.get("license") or "unrecorded"
            if tag not in allowed:
                bad[tag] = bad.get(tag, 0) + 1
    if bad:
        raise SystemExit(
            f"license gate ({license_mode}): refusing to freeze — input "
            f"contains disallowed license tags {dict(sorted(bad.items()))}. "
            "Re-export with the matching --license-mode (records must carry "
            "a license field), or pass --allow-unresolved-licenses in "
            "research mode for pending/unknown tags."
        )


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
    parser.add_argument(
        "--license-mode",
        choices=_LICENSE_MODES,
        default="permissive",
        help=(
            "fail-closed gate: every input record's license field must be "
            "allowed by this mode (permissive: PERMISSIVE only; research: "
            "+ CC_BY_NC); recorded in the sidecar"
        ),
    )
    parser.add_argument(
        "--allow-unresolved-licenses",
        action="store_true",
        help="DANGER: research mode only — also accept pending/unknown tags",
    )
    args = parser.parse_args()
    if not 0.0 <= args.val_fraction < 1.0 or not 0.0 <= args.test_fraction < 1.0:
        raise SystemExit("--val-fraction/--test-fraction must be in [0, 1)")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise SystemExit("val_fraction + test_fraction must be < 1")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "heldout": [], "test": []}
    for record in read_records(args.inputs):
        splits[_assign(record, args.val_fraction, args.test_fraction, args.seed)].append(
            record
        )
    _enforce_license_mode(splits, args.license_mode, args.allow_unresolved_licenses)
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
    # The full split-key universe lets downstream consumers (build_dpo_data)
    # verify their input corpus is the one this snapshot was frozen from.
    split_keys_path = out / "SPLIT_KEYS.json"
    split_keys = sorted(set().union(*keys.values()))
    split_keys_path.write_text(json.dumps(split_keys, indent=0))
    sidecar = {
        "inputs": {p: _sha(Path(p)) for p in args.inputs},
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "rule": "house-first sha1(split_key) bucket (data.split_bucket)",
        "license_mode": args.license_mode,
        "allow_unresolved_licenses": args.allow_unresolved_licenses,
        "counts": {name: len(splits[name]) for name in files},
        "per_source_layer": {
            name: _source_layer_counts(splits[name]) for name in files
        },
        "license_counts": {
            name: _license_counts(splits[name]) for name in files
        },
        "hashes": {name: _sha(path) for name, path in files.items()},
        "split_keys_file": split_keys_path.name,
        "split_keys_sha256": _sha(split_keys_path),
        "n_split_keys": len(split_keys),
        "leakage_check": "passed",
        "test_split_policy": (
            "heldout drives learning curves and tuning; test.jsonl is read "
            "ONCE for the final report"
        ),
    }
    # Immutable fingerprint over everything above — downstream tools verify
    # a sidecar by recomputing this over the sorted payload.
    sidecar["snapshot_id"] = hashlib.sha256(
        json.dumps(sidecar, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    (out / "SNAPSHOT.json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar["counts"]), "->", out)


if __name__ == "__main__":
    main()
