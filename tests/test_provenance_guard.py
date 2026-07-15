"""Training-entry data-provenance gate: dataset files must verify against
their snapshot / pair-builder sidecars before any GPU time is spent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import fastfill_train  # noqa: F401,E402  (vendor path bootstrap)
from fastfill_train.config import load_config  # noqa: E402
from fastfill_train.provenance_guard import (  # noqa: E402
    _sha,
    verify_training_data,
    write_provenance,
)

CONFIGS = REPO / "configs"


def _cfg(tmp_path: Path, dataset: Path, allow: bool = False):
    cfg = load_config(
        CONFIGS / "sft_smoke.yaml",
        [
            f"dataset_files=[{dataset}]",
            f"output_dir={tmp_path / 'out'}",
            f"allow_unverified_data={'true' if allow else 'false'}",
        ],
    )
    return cfg


def _snapshot_dir(tmp_path: Path) -> Path:
    import hashlib

    out = tmp_path / "snap"
    out.mkdir()
    train = out / "train.jsonl"
    train.write_text('{"uid": "s0"}\n', encoding="utf-8")
    payload = {
        "hashes": {"train": _sha(train)},
        "license_mode": "permissive",
        "license_counts": {"train": {"permissive": 1}},
    }
    payload["snapshot_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    (out / "SNAPSHOT.json").write_text(json.dumps(payload))
    return out


def test_verified_snapshot_file_passes(tmp_path: Path) -> None:
    snap = _snapshot_dir(tmp_path)
    provenance = verify_training_data(_cfg(tmp_path, snap / "train.jsonl"))
    assert provenance["verified"] is True
    (entry,) = provenance["files"]
    assert entry["snapshot_id"]  # verified fingerprint, recomputed
    assert provenance["snapshot_id"] == entry["snapshot_id"]
    assert entry["license_mode"] == "permissive"
    path = write_provenance(_cfg(tmp_path, snap / "train.jsonl"), provenance)
    assert json.loads(path.read_text())["verified"] is True


def test_modified_file_fails(tmp_path: Path) -> None:
    snap = _snapshot_dir(tmp_path)
    (snap / "train.jsonl").write_text('{"uid": "tampered"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="hash"):
        verify_training_data(_cfg(tmp_path, snap / "train.jsonl"))


def test_missing_sidecar_fails_closed(tmp_path: Path) -> None:
    loose = tmp_path / "loose.jsonl"
    loose.write_text('{"uid": "s0"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="provenance"):
        verify_training_data(_cfg(tmp_path, loose))
    provenance = verify_training_data(_cfg(tmp_path, loose, allow=True))
    assert provenance["verified"] is False


def test_dpo_pairs_verify_via_stats_sidecar(tmp_path: Path) -> None:
    import hashlib

    pairs = tmp_path / "stage2_pairs.jsonl"
    pairs.write_text('{"uid": "p0"}\n', encoding="utf-8")
    stats = {
        "snapshot_id": "abc123",
        "license_mode": "permissive",
        "out_sha256_16": hashlib.sha256(pairs.read_bytes()).hexdigest()[:16],
    }
    pairs.with_suffix(".stats.json").write_text(json.dumps(stats))
    provenance = verify_training_data(_cfg(tmp_path, pairs))
    (entry,) = provenance["files"]
    assert entry["snapshot_id"] == "abc123"


def test_dpo_pairs_without_snapshot_id_fail(tmp_path: Path) -> None:
    pairs = tmp_path / "stage2_pairs.jsonl"
    pairs.write_text('{"uid": "p0"}\n', encoding="utf-8")
    pairs.with_suffix(".stats.json").write_text(json.dumps({"snapshot_id": None}))
    with pytest.raises(SystemExit, match="snapshot_id"):
        verify_training_data(_cfg(tmp_path, pairs))


def test_forged_snapshot_id_fails(tmp_path: Path) -> None:
    snap = _snapshot_dir(tmp_path)
    sidecar_path = snap / "SNAPSHOT.json"
    payload = json.loads(sidecar_path.read_text())
    payload["snapshot_id"] = "deadbeefdeadbeef"
    sidecar_path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="does not recompute"):
        verify_training_data(_cfg(tmp_path, snap / "train.jsonl"))


def test_resume_chains_previous_provenance(tmp_path: Path) -> None:
    snap = _snapshot_dir(tmp_path)
    cfg = _cfg(tmp_path, snap / "train.jsonl")
    first = verify_training_data(cfg)
    write_provenance(cfg, first)
    second = verify_training_data(cfg)
    path = write_provenance(cfg, second)
    payload = json.loads(path.read_text())
    assert payload["verified"] is True
    assert payload["previous"]["verified"] is True  # both lineages visible


def test_dpo_base_and_pairs_must_share_snapshot(tmp_path: Path) -> None:
    from fastfill_train.provenance_guard import check_base_snapshot_consistency

    base_dir = tmp_path / "base_model"
    base_dir.mkdir()
    (base_dir / "DATA_PROVENANCE.json").write_text(
        json.dumps({"snapshot_id": "snap_a"})
    )
    from dataclasses import replace

    cfg = replace(
        _cfg(tmp_path, tmp_path / "unused.jsonl", allow=True),
        model_name_or_path=str(base_dir),
    )
    with pytest.raises(SystemExit, match="share one snapshot"):
        check_base_snapshot_consistency(cfg, {"snapshot_id": "snap_b"})
    # Matching id passes; missing lineage on either side is soft.
    check_base_snapshot_consistency(cfg, {"snapshot_id": "snap_a"})
    check_base_snapshot_consistency(cfg, {"snapshot_id": None})
