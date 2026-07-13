"""Dataset loading and house-first splitting for SFT / DPO runs.

Consumes the JSONL files produced by the vendored ``export_sft.py``
(records: uid / split_key / source_dataset / instruction / input / output)
and DPO pair files produced by ``build_dpo_data.py`` (records: prompt /
chosen / rejected, already in chat-messages form).

The split is HOUSE-FIRST (铁律 4): membership is decided by hashing
``split_key`` (source house), never per-record randomness, so rooms of one
house — and floor/surface records of one room — can never straddle
train/val. (OptiScene's released loader uses a random 80/20 row split; we
deliberately do not copy that.)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from fastfill_train.templates import render_sft_example

_SPLIT_BUCKETS = 10_000


def read_records(paths: Iterable[str | Path]) -> Iterator[dict]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: bad JSONL line") from exc


def is_validation_key(split_key: str, val_fraction: float, seed: int) -> bool:
    """Deterministic house-level split membership."""
    digest = hashlib.sha1(f"{seed}:{split_key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % _SPLIT_BUCKETS
    return bucket < int(val_fraction * _SPLIT_BUCKETS)


def split_house_first(
    records: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    val: list[dict] = []
    for record in records:
        key = record.get("split_key") or record.get("uid", "")
        (val if is_validation_key(key, val_fraction, seed) else train).append(
            record
        )
    return train, val


def render_messages(records: list[dict], template: str) -> list[dict]:
    """SFT records -> TRL prompt-completion rows (COMPLETION-ONLY loss).

    Deliberately NOT the plain "messages" format: with messages TRL trains
    on the full sequence, so the model also learns to echo the RoomContext
    (observed in smoke: completions regurgitating the input). The
    prompt/completion split makes TRL mask the prompt from the loss.
    """
    rows = []
    for r in records:
        ex = render_sft_example(r, template)
        rows.append(
            {
                "prompt": [{"role": "user", "content": ex.user}],
                "completion": [{"role": "assistant", "content": ex.assistant}],
            }
        )
    return rows


def load_sft_datasets(
    dataset_files: Iterable[str | Path],
    template: str,
    val_fraction: float,
    seed: int,
):
    """Build (train, val) HF Datasets. Imports ``datasets`` lazily."""
    from datasets import Dataset

    records = list(read_records(dataset_files))
    if not records:
        raise ValueError("no records found in dataset_files")
    train_recs, val_recs = split_house_first(records, val_fraction, seed)
    if not train_recs:
        raise ValueError("house-first split left the training set empty")
    train = Dataset.from_list(render_messages(train_recs, template))
    val = Dataset.from_list(render_messages(val_recs, template)) if val_recs else None
    return train, val


def load_dpo_datasets(
    dataset_files: Iterable[str | Path], val_fraction: float, seed: int
):
    """DPO pair JSONL -> (train, val) HF Datasets with prompt/chosen/rejected."""
    from datasets import Dataset

    records = list(read_records(dataset_files))
    for record in records:
        missing = {"prompt", "chosen", "rejected"} - set(record)
        if missing:
            raise ValueError(f"DPO record missing fields: {sorted(missing)}")
    train_recs, val_recs = split_house_first(records, val_fraction, seed)
    keep = ("prompt", "chosen", "rejected")
    train = Dataset.from_list([{k: r[k] for k in keep} for r in train_recs])
    val = (
        Dataset.from_list([{k: r[k] for k in keep} for r in val_recs])
        if val_recs
        else None
    )
    return train, val
