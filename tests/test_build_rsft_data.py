"""RSFT builder: hard gates select good candidates, collapse alarm fires."""

import json

from test_injectors import make_clean_sample

from fastfill_train.build_dpo_data import _floor_record
from fastfill_train.build_rsft_data import run_rsft
from scenesmith.growing_world.fastfill.codec import encode_floor_layout


def _setup(tmp_path, completions_by_uid):
    sample = make_clean_sample("s0", "house0")
    record = _floor_record(sample)
    (tmp_path / "ctx.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "samples.jsonl").write_text(sample.model_dump_json() + "\n")
    rows = [
        {"uid": record["uid"], "completion": c}
        for c in completions_by_uid
    ]
    (tmp_path / "gens.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    return sample, record


def test_selects_valid_candidate_and_rejects_bad(tmp_path):
    sample, record = _setup(
        tmp_path,
        [
            record_output := None,  # placeholder replaced below
        ],
    )
    good = encode_floor_layout(sample.layout.floor_layout)
    sparse = "\n".join(good.splitlines()[:1])  # too few objects
    garbage = "not a layout at all"
    oob = good.replace("|0,", "|900,", 1)  # push an object far away
    (tmp_path / "gens.jsonl").write_text(
        "\n".join(
            json.dumps({"uid": record["uid"], "completion": c})
            for c in (garbage, sparse, oob, good)
        )
    )
    stats = run_rsft(
        tmp_path / "ctx.jsonl",
        tmp_path / "gens.jsonl",
        tmp_path / "samples.jsonl",
        tmp_path / "rsft.jsonl",
        min_objects=2,
    )
    assert stats["counts"]["selected"] == 1
    assert stats["rejects"]["gate_parse"] >= 1
    assert stats["rejects"]["gate_object_count_floor"] >= 1
    assert not stats["collapse_alarm"]
    row = json.loads((tmp_path / "rsft.jsonl").read_text())
    assert row["output"] == good
    assert row["uid"].endswith("#rsft")


def test_collapse_alarm_when_all_candidates_fail(tmp_path):
    sample, record = _setup(tmp_path, ["junk one", "junk two"])
    stats = run_rsft(
        tmp_path / "ctx.jsonl",
        tmp_path / "gens.jsonl",
        tmp_path / "samples.jsonl",
        tmp_path / "rsft.jsonl",
    )
    assert stats["counts"]["selected"] == 0
    assert stats["collapse_alarm"] is True
