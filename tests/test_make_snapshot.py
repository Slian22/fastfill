"""Tests for the stage-0 snapshot freezer: three-way house-first split,
leakage guard, sidecar accounting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fastfill_train  # noqa: F401  (vendor path bootstrap)

from fastfill_train.data import split_bucket
from fastfill_train.make_snapshot import main


def _write_records(path: Path, n: int = 200) -> None:
    rows = []
    for i in range(n):
        rows.append(
            json.dumps(
                {
                    "uid": f"s{i}",
                    "split_key": f"src/house_{i}",
                    "source_dataset": "src",
                    "license": "permissive",
                    "instruction": "x",
                    "input": "room bedroom id=r",
                    "output": "bed|160,200,55|0,0|0|W",
                }
            )
        )
        rows.append(
            json.dumps(
                {
                    "uid": f"s{i}#g0",
                    "split_key": f"src/house_{i}",
                    "source_dataset": "src",
                    "license": "permissive",
                    "instruction": "x",
                    "input": "surface s kind=top",
                    "output": "group g0 surface=s pattern=free anchor=-",
                }
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run_main(tmp_path: Path, argv: list[str], monkeypatch) -> dict:
    monkeypatch.setattr(sys, "argv", ["make_snapshot"] + argv)
    main()
    return json.loads((tmp_path / "SNAPSHOT.json").read_text())


def test_three_way_split_is_disjoint_and_complete(tmp_path: Path, monkeypatch):
    in_path = tmp_path / "in.jsonl"
    _write_records(in_path)
    sidecar = _run_main(
        tmp_path,
        [
            "--in", str(in_path),
            "--out-dir", str(tmp_path),
            "--val-fraction", "0.1",
            "--test-fraction", "0.1",
            "--seed", "42",
        ],
        monkeypatch,
    )
    counts = sidecar["counts"]
    assert set(counts) == {"train", "heldout", "test"}
    assert sum(counts.values()) == 400
    assert counts["heldout"] > 0 and counts["test"] > 0
    # floor + surface rows of one house always land on the same side
    seen: dict[str, str] = {}
    for name in ("train", "heldout", "test"):
        for line in (tmp_path / f"{name}.jsonl").read_text().splitlines():
            record = json.loads(line)
            key = record["split_key"]
            assert seen.setdefault(key, name) == name
    # assignment matches the shared bucket rule training uses
    for key, side in seen.items():
        bucket = split_bucket(key, 42)
        expected = (
            "heldout" if bucket < 1000 else "test" if bucket < 2000 else "train"
        )
        assert side == expected
    assert sidecar["per_source_layer"]["train"]["src/floor"] > 0
    assert sidecar["per_source_layer"]["train"]["src/surface"] > 0


def test_zero_test_fraction_writes_no_test_file(tmp_path: Path, monkeypatch):
    in_path = tmp_path / "in.jsonl"
    _write_records(in_path, n=50)
    sidecar = _run_main(
        tmp_path,
        [
            "--in", str(in_path),
            "--out-dir", str(tmp_path),
            "--test-fraction", "0",
        ],
        monkeypatch,
    )
    assert not (tmp_path / "test.jsonl").exists()
    assert set(sidecar["counts"]) == {"train", "heldout"}


def test_max_train_subsamples_after_split(tmp_path: Path, monkeypatch):
    in_path = tmp_path / "in.jsonl"
    _write_records(in_path, n=100)
    sidecar = _run_main(
        tmp_path,
        [
            "--in", str(in_path),
            "--out-dir", str(tmp_path),
            "--max-train", "20",
        ],
        monkeypatch,
    )
    assert sidecar["counts"]["train"] == 20


def test_license_gate_fails_closed(tmp_path: Path, monkeypatch):
    """A permissive freeze must refuse NC / unrecorded license records."""
    import pytest

    in_path = tmp_path / "in.jsonl"
    rows = [
        json.dumps(
            {
                "uid": "s0",
                "split_key": "src/h0",
                "source_dataset": "src",
                "license": "cc_by_nc",
                "instruction": "x",
                "input": "room bedroom id=r",
                "output": "bed|160,200,55|0,0|0|W",
            }
        )
    ]
    in_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_snapshot", "--in", str(in_path), "--out-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="license gate"):
        main()
    # research mode accepts the same input.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_snapshot",
            "--in", str(in_path),
            "--out-dir", str(tmp_path / "res"),
            "--license-mode", "research",
        ],
    )
    main()
    sidecar = json.loads((tmp_path / "res" / "SNAPSHOT.json").read_text())
    assert sidecar["license_mode"] == "research"


def test_sidecar_carries_fingerprint_and_split_keys(tmp_path: Path, monkeypatch):
    in_path = tmp_path / "in.jsonl"
    _write_records(in_path, n=50)
    sidecar = _run_main(
        tmp_path,
        ["--in", str(in_path), "--out-dir", str(tmp_path)],
        monkeypatch,
    )
    assert sidecar["snapshot_id"]
    assert sidecar["license_mode"] == "permissive"
    keys_path = tmp_path / sidecar["split_keys_file"]
    keys = json.loads(keys_path.read_text())
    assert len(keys) == sidecar["n_split_keys"] == 50


def test_fraction_range_is_validated(tmp_path: Path, monkeypatch):
    import pytest

    in_path = tmp_path / "in.jsonl"
    _write_records(in_path, n=10)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_snapshot",
            "--in", str(in_path),
            "--out-dir", str(tmp_path),
            "--val-fraction", "-0.5",
        ],
    )
    with pytest.raises(SystemExit, match="fraction"):
        main()
