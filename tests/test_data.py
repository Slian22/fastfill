"""Tests for dataset reading and the house-first split (no torch/datasets)."""

from __future__ import annotations

import json
from pathlib import Path

import fastfill_train  # noqa: F401  (vendor path bootstrap)
import pytest

from fastfill_train.data import read_records, render_messages, split_house_first


def _records(houses: int = 10, per_house: int = 3) -> list[dict]:
    return [
        {
            "uid": f"h{h}_r{i}",
            "split_key": f"house{h}",
            "source_dataset": "test_ds",
            "instruction": "Place furniture.",
            "input": f"room bedroom id=h{h}_r{i}",
            "output": "bed|160,200,50|0,0|0|W",
        }
        for h in range(houses)
        for i in range(per_house)
    ]


# ------------------------------------------------------------- house-first split


def test_records_sharing_split_key_never_straddle_splits() -> None:
    records = _records()
    for seed in range(20):
        train, val = split_house_first(records, val_fraction=0.3, seed=seed)
        assert len(train) + len(val) == len(records)
        train_keys = {r["split_key"] for r in train}
        val_keys = {r["split_key"] for r in val}
        assert not train_keys & val_keys  # a house is train XOR val


def test_split_is_deterministic_per_seed() -> None:
    records = _records()
    first_train, first_val = split_house_first(records, 0.3, seed=7)
    again_train, again_val = split_house_first(records, 0.3, seed=7)
    assert first_train == again_train
    assert first_val == again_val


def test_val_fraction_zero_yields_no_val() -> None:
    records = _records()
    for seed in range(10):
        train, val = split_house_first(records, val_fraction=0.0, seed=seed)
        assert val == []
        assert len(train) == len(records)


def test_split_falls_back_to_uid_without_split_key() -> None:
    records = [{"uid": "only_record"}, {"uid": "only_record"}]
    for seed in range(10):
        train, val = split_house_first(records, val_fraction=0.5, seed=seed)
        assert (len(train), len(val)) in ((2, 0), (0, 2))  # never straddles


# ------------------------------------------------------------------ read_records


def test_read_records_yields_rows_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "ok.jsonl"
    path.write_text('{"uid": "a"}\n\n{"uid": "b"}\n', encoding="utf-8")
    rows = list(read_records([path]))
    assert rows == [{"uid": "a"}, {"uid": "b"}]


def test_read_records_raises_with_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"uid": "a"}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        list(read_records([path]))
    message = str(excinfo.value)
    assert str(path) in message
    assert ":2:" in message  # the offending line number


def test_read_records_spans_multiple_files(tmp_path: Path) -> None:
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    first.write_text('{"uid": "a"}\n', encoding="utf-8")
    second.write_text('{"uid": "b"}\n', encoding="utf-8")
    assert [r["uid"] for r in read_records([first, second])] == ["a", "b"]


# --------------------------------------------------------------- render_messages


def test_render_messages_chat_shape():
    import json

    record = _records(1, 1)[0]
    rows = render_messages([record], "direct")
    assert set(rows[0]) == {"prompt", "completion"}  # completion-only loss
    assert rows[0]["prompt"][0]["role"] == "user"
    assert rows[0]["completion"][0]["role"] == "assistant"
    assert rows[0]["completion"][0]["content"] == record["output"]
    json.dumps(rows)  # serializable

def test_render_messages_rows_are_json_serializable() -> None:
    rows = render_messages(_records(houses=1, per_house=1), "plan")
    assert json.loads(json.dumps(rows)) == rows
