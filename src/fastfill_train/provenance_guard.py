"""Data-provenance gate for training entrypoints.

``verify_training_data`` binds a training run to the frozen snapshot its
dataset files came from: each file must sit next to a make_snapshot
``SNAPSHOT.json`` whose recorded hash matches the file's current content.
The returned provenance dict (snapshot_id, license_mode, file hashes, code
commit) is written into the run's output dir as ``DATA_PROVENANCE.json`` so
a model directory can always prove which data and code produced it — a
mutable path like ``data/full/train.jsonl`` in the config proves nothing
once the directory has been rebuilt.

Fail-closed: a missing sidecar, an unrecorded file, or a hash mismatch
aborts the run unless ``cfg.allow_unverified_data`` is explicitly set
(smoke/ad-hoc data only).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _verify_one(path: Path) -> dict:
    """Verify one dataset file against its provenance sidecar.

    Two sidecar shapes are accepted: a make_snapshot ``SNAPSHOT.json`` in
    the same directory (SFT splits), or the pair builder's
    ``<file>.stats.json`` (DPO pairs — must itself record a snapshot_id,
    i.e. the pairs were built against a frozen snapshot).
    """
    sidecar_path = path.parent / "SNAPSHOT.json"
    stats_path = path.with_suffix(".stats.json")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        recorded = (sidecar.get("hashes") or {}).get(path.stem)
        if recorded is None:
            raise SystemExit(
                f"data provenance: {sidecar_path} does not record a hash for "
                f"'{path.stem}' — file is not part of this snapshot"
            )
        actual = _sha(path)
        if actual != recorded:
            raise SystemExit(
                f"data provenance: {path} hash {actual} != snapshot's "
                f"{recorded} — the file changed after freezing (directory "
                "rebuilt?); re-freeze or point at the right snapshot"
            )
        return {
            "file": str(path),
            "sha256_16": actual,
            "snapshot": str(sidecar_path),
            "snapshot_id": sidecar.get("snapshot_id"),
            "license_mode": sidecar.get("license_mode"),
            "license_counts": (sidecar.get("license_counts") or {}).get(
                path.stem
            ),
        }
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if not stats.get("snapshot_id"):
            raise SystemExit(
                f"data provenance: {stats_path} has no snapshot_id — these "
                "pairs were built without a frozen snapshot (smoke only); "
                "rebuild with --snapshot, or set allow_unverified_data=true"
            )
        recorded = stats.get("out_sha256_16")
        actual = _sha(path)
        if recorded is not None and actual != recorded:
            raise SystemExit(
                f"data provenance: {path} hash {actual} != pair-builder's "
                f"{recorded} — pairs changed after building"
            )
        return {
            "file": str(path),
            "sha256_16": actual,
            "stats": str(stats_path),
            "snapshot_id": stats.get("snapshot_id"),
            "license_mode": stats.get("license_mode"),
        }
    raise SystemExit(
        f"data provenance: no SNAPSHOT.json next to {path} and no "
        f"{stats_path.name} — train only on make_snapshot / build_dpo_data "
        "output, or set allow_unverified_data=true for an explicitly "
        "unverified smoke run"
    )


def verify_training_data(cfg) -> dict:
    """Verify every dataset file; return the provenance payload."""
    if cfg.allow_unverified_data:
        return {
            "verified": False,
            "note": "allow_unverified_data=true — smoke/ad-hoc data",
            "files": [str(f) for f in cfg.dataset_files],
            "code_commit": _git_commit(),
        }
    return {
        "verified": True,
        "files": [_verify_one(Path(f)) for f in cfg.dataset_files],
        "code_commit": _git_commit(),
    }


def write_provenance(cfg, provenance: dict) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "DATA_PROVENANCE.json"
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return path
