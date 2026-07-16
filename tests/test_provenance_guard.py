"""Training-entry data-provenance gate: dataset files must verify against
their snapshot / pair-builder sidecars before any GPU time is spent."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import fastfill_train  # noqa: F401,E402  (vendor path bootstrap)
from fastfill_train.config import load_config  # noqa: E402
from fastfill_train.data import SNAPSHOT_SCHEMA_VERSION  # noqa: E402
from fastfill_train.provenance_guard import (  # noqa: E402
    _sha,
    verify_training_data,
    write_provenance,
)

CONFIGS = REPO / "configs"


def _fingerprinted(payload: dict) -> dict:
    payload = dict(payload)
    payload["snapshot_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def _cfg(tmp_path: Path, dataset: Path, allow: bool = False):
    return load_config(
        CONFIGS / "sft_smoke.yaml",
        [
            f"dataset_files=[{dataset}]",
            f"output_dir={tmp_path / 'out'}",
            f"allow_unverified_data={'true' if allow else 'false'}",
        ],
    )


def _snapshot_dir(tmp_path: Path, name: str = "snap") -> Path:
    """A minimal but SCHEMA-COMPLETE snapshot dir with a valid fingerprint."""
    out = tmp_path / name
    out.mkdir()
    train = out / "train.jsonl"
    train.write_text('{"uid": "s0"}\n', encoding="utf-8")
    keys = out / "SPLIT_KEYS.json"
    keys.write_text(
        json.dumps(
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "split_keys": ["house0"],
                "geometry_hashes": [],
                "record_hashes": [],
            }
        )
    )
    sidecar = _fingerprinted(
        {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "seed": 42,
            "val_fraction": 0.1,
            "test_fraction": 0.0,
            "rule": "house-first sha1(split_key) bucket (data.split_bucket)",
            "inputs": {"floor_sft.jsonl": "0" * 16},
            "hashes": {"train": _sha(train)},
            "license_mode": "permissive",
            "license_counts": {"train": {"permissive": 1}},
            "split_keys_file": keys.name,
            "split_keys_sha256": _sha(keys),
            "leakage_check": "passed",
            "counts": {"train": 1},
        }
    )
    (out / "SNAPSHOT.json").write_text(json.dumps(sidecar))
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


def test_minimal_handwritten_sidecar_fails_schema(tmp_path: Path) -> None:
    """A sidecar with a self-consistent fingerprint but missing required
    fields (seed, rule, SPLIT_KEYS, ...) must not verify."""
    out = tmp_path / "snap"
    out.mkdir()
    train = out / "train.jsonl"
    train.write_text('{"uid": "s0"}\n', encoding="utf-8")
    sidecar = _fingerprinted(
        {"hashes": {"train": _sha(train)}, "license_mode": "permissive"}
    )
    (out / "SNAPSHOT.json").write_text(json.dumps(sidecar))
    with pytest.raises(SystemExit, match="missing"):
        verify_training_data(_cfg(tmp_path, train))


def test_forged_snapshot_id_fails(tmp_path: Path) -> None:
    snap = _snapshot_dir(tmp_path)
    sidecar_path = snap / "SNAPSHOT.json"
    payload = json.loads(sidecar_path.read_text())
    payload["snapshot_id"] = "deadbeefdeadbeef"
    sidecar_path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="does not recompute"):
        verify_training_data(_cfg(tmp_path, snap / "train.jsonl"))


def _pairs_with_stats(tmp_path: Path, snap: Path) -> Path:
    pairs = tmp_path / "stage2_pairs.jsonl"
    pairs.write_text('{"uid": "p0"}\n', encoding="utf-8")
    sidecar = json.loads((snap / "SNAPSHOT.json").read_text())
    stats = {
        "snapshot": str(snap / "SNAPSHOT.json"),
        "snapshot_id": sidecar["snapshot_id"],
        "license_mode": "permissive",
        "out_sha256_16": _sha(pairs),
    }
    stats["stats_id"] = hashlib.sha256(
        json.dumps(stats, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    pairs.with_suffix(".stats.json").write_text(json.dumps(stats))
    return pairs


def test_dpo_pairs_verify_via_stats_sidecar(tmp_path: Path) -> None:
    snap = _snapshot_dir(tmp_path)
    pairs = _pairs_with_stats(tmp_path, snap)
    provenance = verify_training_data(_cfg(tmp_path, pairs))
    (entry,) = provenance["files"]
    assert entry["snapshot_id"] == json.loads(
        (snap / "SNAPSHOT.json").read_text()
    )["snapshot_id"]


def test_dpo_stats_without_fingerprint_fail(tmp_path: Path) -> None:
    snap = _snapshot_dir(tmp_path)
    pairs = _pairs_with_stats(tmp_path, snap)
    stats_path = pairs.with_suffix(".stats.json")
    stats = json.loads(stats_path.read_text())
    del stats["stats_id"]
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(SystemExit, match="stats_id"):
        verify_training_data(_cfg(tmp_path, pairs))


def test_dpo_stats_forged_snapshot_id_fails(tmp_path: Path) -> None:
    """Hand-written stats vouching for arbitrary pairs must not verify:
    the back-referenced SNAPSHOT must exist and match."""
    snap = _snapshot_dir(tmp_path)
    pairs = _pairs_with_stats(tmp_path, snap)
    stats_path = pairs.with_suffix(".stats.json")
    stats = json.loads(stats_path.read_text())
    stats["snapshot_id"] = "totally-fake"
    stats["stats_id"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in stats.items() if k != "stats_id"}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()[:16]
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(SystemExit, match="snapshot"):
        verify_training_data(_cfg(tmp_path, pairs))


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


def test_resume_from_external_checkpoint_is_recorded(tmp_path: Path) -> None:
    from dataclasses import replace

    from fastfill_train.provenance_guard import attach_resume_lineage

    other_run = tmp_path / "other_run"
    (other_run / "checkpoint-100").mkdir(parents=True)
    (other_run / "DATA_PROVENANCE.json").write_text(
        json.dumps({"snapshot_id": "snap_other"})
    )
    snap = _snapshot_dir(tmp_path)
    cfg = replace(
        _cfg(tmp_path, snap / "train.jsonl"),
        resume_from_checkpoint=str(other_run / "checkpoint-100"),
    )
    provenance = attach_resume_lineage(cfg, verify_training_data(cfg))
    assert provenance["resumed_from"]["provenance"]["snapshot_id"] == "snap_other"


def test_string_allow_unverified_is_rejected(tmp_path: Path) -> None:
    from dataclasses import replace

    snap = _snapshot_dir(tmp_path)
    cfg = replace(
        _cfg(tmp_path, snap / "train.jsonl"), allow_unverified_data="false"
    )
    with pytest.raises(SystemExit, match="YAML boolean"):
        verify_training_data(cfg)


def test_dpo_base_and_pairs_must_share_snapshot(tmp_path: Path) -> None:
    from dataclasses import replace

    from fastfill_train.provenance_guard import check_base_snapshot_consistency

    base_dir = tmp_path / "base_model"
    base_dir.mkdir()
    (base_dir / "DATA_PROVENANCE.json").write_text(
        json.dumps({"snapshot_id": "snap_a", "license_mode": "permissive"})
    )
    cfg = replace(
        _cfg(tmp_path, tmp_path / "unused.jsonl", allow=True),
        model_name_or_path=str(base_dir),
    )
    with pytest.raises(SystemExit, match="EXACTLY one shared snapshot"):
        check_base_snapshot_consistency(cfg, {"snapshot_id": "snap_b"})
    with pytest.raises(SystemExit, match="routes must match exactly"):
        check_base_snapshot_consistency(
            cfg, {"snapshot_id": "snap_a", "license_mode": "research"}
        )
    # Matching id+mode passes; pairs without lineage are soft.
    check_base_snapshot_consistency(
        cfg, {"snapshot_id": "snap_a", "license_mode": "permissive"}
    )
    check_base_snapshot_consistency(cfg, {"snapshot_id": None})


def test_dpo_base_mixed_lineage_is_rejected(tmp_path: Path) -> None:
    """Set membership is not enough: a base whose history mixes a research
    checkpoint still carries NC-derived weights."""
    from dataclasses import replace

    from fastfill_train.provenance_guard import check_base_snapshot_consistency

    base_dir = tmp_path / "mixed_base"
    base_dir.mkdir()
    (base_dir / "DATA_PROVENANCE.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_p",
                "license_mode": "permissive",
                "resumed_from": {
                    "provenance": {
                        "snapshot_id": "snap_r",
                        "license_mode": "research",
                    }
                },
            }
        )
    )
    cfg = replace(
        _cfg(tmp_path, tmp_path / "unused.jsonl", allow=True),
        model_name_or_path=str(base_dir),
    )
    with pytest.raises(SystemExit, match="EXACTLY one shared snapshot"):
        check_base_snapshot_consistency(
            cfg, {"snapshot_id": "snap_p", "license_mode": "permissive"}
        )


def test_dpo_base_empty_or_unverified_lineage_rejected(tmp_path: Path) -> None:
    from dataclasses import replace

    from fastfill_train.provenance_guard import check_base_snapshot_consistency

    empty_base = tmp_path / "empty_base"
    empty_base.mkdir()
    (empty_base / "DATA_PROVENANCE.json").write_text("{}")
    cfg = replace(
        _cfg(tmp_path, tmp_path / "unused.jsonl", allow=True),
        model_name_or_path=str(empty_base),
    )
    with pytest.raises(SystemExit, match="empty lineage"):
        check_base_snapshot_consistency(cfg, {"snapshot_id": "snap_a"})

    unverified_base = tmp_path / "unverified_base"
    unverified_base.mkdir()
    (unverified_base / "DATA_PROVENANCE.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_a",
                "license_mode": "permissive",
                "previous": {"verified": False},
            }
        )
    )
    cfg = replace(cfg, model_name_or_path=str(unverified_base))
    with pytest.raises(SystemExit, match="verified=false"):
        check_base_snapshot_consistency(
            cfg, {"snapshot_id": "snap_a", "license_mode": "permissive"}
        )


def test_resume_across_license_routes_is_rejected(tmp_path: Path) -> None:
    """A research checkpoint must not resume into a permissive run — the
    weights themselves would carry NC-derived training."""
    from dataclasses import replace

    from fastfill_train.provenance_guard import attach_resume_lineage

    research_run = tmp_path / "research_run"
    (research_run / "checkpoint-50").mkdir(parents=True)
    (research_run / "DATA_PROVENANCE.json").write_text(
        json.dumps({"snapshot_id": "snap_r", "license_mode": "research"})
    )
    snap = _snapshot_dir(tmp_path)  # permissive
    cfg = replace(
        _cfg(tmp_path, snap / "train.jsonl"),
        resume_from_checkpoint=str(research_run / "checkpoint-50"),
    )
    provenance = verify_training_data(cfg)
    assert provenance["license_mode"] == "permissive"
    with pytest.raises(SystemExit, match="across license routes"):
        attach_resume_lineage(cfg, provenance)


def test_record_identity_includes_layer() -> None:
    """Flipping floor->surface must change the record identity — templates
    route PLAN-rendering on layer, so an unchanged hash would poison
    plan/plan_nl DPO chosens."""
    from fastfill_train.data import record_identity_sha

    record = {
        "uid": "s0",
        "split_key": "house0",
        "license": "permissive",
        "layer": "floor",
        "instruction": "x",
        "input": "room",
        "output": "bed|1",
    }
    flipped = dict(record, layer="surface")
    assert record_identity_sha(record) != record_identity_sha(flipped)


def test_dpo_base_without_provenance_is_hard_error(tmp_path: Path) -> None:
    from dataclasses import replace

    from fastfill_train.provenance_guard import check_base_snapshot_consistency

    bare_base = tmp_path / "bare_model"
    bare_base.mkdir()
    cfg = replace(
        _cfg(tmp_path, tmp_path / "unused.jsonl", allow=True),
        model_name_or_path=str(bare_base),
    )
    with pytest.raises(SystemExit, match="DATA_PROVENANCE"):
        check_base_snapshot_consistency(cfg, {"snapshot_id": "snap_a"})


def test_boolean_resume_across_license_routes_is_rejected(tmp_path: Path) -> None:
    """resume_from_checkpoint=true resumes output_dir's latest checkpoint —
    an output_dir holding a research run's weights must not continue as a
    permissive run (round-7 covered only explicit checkpoint paths)."""
    from dataclasses import replace

    from fastfill_train.provenance_guard import attach_resume_lineage

    snap = _snapshot_dir(tmp_path)  # permissive
    cfg = replace(
        _cfg(tmp_path, snap / "train.jsonl"), resume_from_checkpoint=True
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "DATA_PROVENANCE.json").write_text(
        json.dumps({"snapshot_id": "snap_r", "license_mode": "research"})
    )
    provenance = verify_training_data(cfg)
    with pytest.raises(SystemExit, match="across license routes"):
        attach_resume_lineage(cfg, provenance)


def test_boolean_resume_same_route_passes(tmp_path: Path) -> None:
    """Crash-resume of the SAME run (same route in output_dir) stays legal."""
    from dataclasses import replace

    from fastfill_train.provenance_guard import attach_resume_lineage

    snap = _snapshot_dir(tmp_path)
    cfg = replace(
        _cfg(tmp_path, snap / "train.jsonl"), resume_from_checkpoint=True
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "DATA_PROVENANCE.json").write_text(
        json.dumps({"snapshot_id": "snap_p", "license_mode": "permissive"})
    )
    provenance = attach_resume_lineage(cfg, verify_training_data(cfg))
    assert provenance["resumed_from"]["provenance"]["license_mode"] == "permissive"
