"""Data-provenance gate for training entrypoints.

``verify_training_data`` binds a training run to the frozen snapshot its
dataset files came from: each file must sit next to a make_snapshot
``SNAPSHOT.json`` (or a build_dpo_data ``.stats.json``) whose recorded hash
matches the file's current content, and the sidecar's own ``snapshot_id``
fingerprint must recompute — a hand-written sidecar with the right file
hash but a fabricated fingerprint fails. All dataset files of one run must
agree on snapshot_id and license_mode. The returned provenance dict
(snapshot_id, license_mode, file hashes, code commit) is written into the
run's output dir as ``DATA_PROVENANCE.json``; on resume the previous
payload is chained under ``previous`` so mixed-data lineage stays visible.

Fail-closed: a missing sidecar, an unrecorded file, a hash or fingerprint
mismatch, or inconsistent lineage across files aborts the run unless
``cfg.allow_unverified_data`` is explicitly set (smoke/ad-hoc data only).
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


def _verified_snapshot_id(sidecar_path: Path, sidecar: dict) -> str:
    """The sidecar's snapshot_id, after recomputing the fingerprint."""
    snapshot_id = sidecar.get("snapshot_id")
    if not snapshot_id:
        raise SystemExit(
            f"data provenance: {sidecar_path} has no snapshot_id — legacy "
            "snapshots cannot be verified; re-freeze with the current "
            "make_snapshot, or set allow_unverified_data=true"
        )
    payload = {k: v for k, v in sidecar.items() if k != "snapshot_id"}
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    if recomputed != snapshot_id:
        raise SystemExit(
            f"data provenance: {sidecar_path} snapshot_id {snapshot_id} does "
            f"not recompute (got {recomputed}) — sidecar was edited"
        )
    return snapshot_id


def _verify_one(path: Path) -> dict:
    """Verify one dataset file against its provenance sidecar.

    Two sidecar shapes are accepted: a make_snapshot ``SNAPSHOT.json`` in
    the same directory (SFT splits), or the pair builder's
    ``<file>.stats.json`` (DPO pairs — must record a snapshot_id AND the
    pairs file's content hash).
    """
    sidecar_path = path.parent / "SNAPSHOT.json"
    stats_path = path.with_suffix(".stats.json")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        snapshot_id = _verified_snapshot_id(sidecar_path, sidecar)
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
        license_mode = sidecar.get("license_mode")
        if not license_mode:
            raise SystemExit(
                f"data provenance: {sidecar_path} records no license_mode — "
                "re-freeze with the current make_snapshot"
            )
        return {
            "file": str(path),
            "sha256_16": actual,
            "snapshot": str(sidecar_path),
            "snapshot_id": snapshot_id,
            "license_mode": license_mode,
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
        if recorded is None:
            raise SystemExit(
                f"data provenance: {stats_path} records no out_sha256_16 — "
                "rebuild the pairs with the current build_dpo_data"
            )
        actual = _sha(path)
        if actual != recorded:
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


def _check_consistency(entries: list[dict]) -> None:
    """One run must not mix snapshots or license routes across files."""
    for field in ("snapshot_id", "license_mode"):
        values = {e.get(field) for e in entries if e.get(field) is not None}
        if len(values) > 1:
            raise SystemExit(
                f"data provenance: dataset files disagree on {field}: "
                f"{sorted(values)} — one training run must use one snapshot "
                "and one license route"
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
    entries = [_verify_one(Path(f)) for f in cfg.dataset_files]
    _check_consistency(entries)
    return {
        "verified": True,
        "files": entries,
        "snapshot_id": next(
            (e["snapshot_id"] for e in entries if e.get("snapshot_id")), None
        ),
        "license_mode": next(
            (e["license_mode"] for e in entries if e.get("license_mode")), None
        ),
        "code_commit": _git_commit(),
    }


def write_provenance(cfg, provenance: dict) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "DATA_PROVENANCE.json"
    if path.exists():
        # Resume / rerun into the same output dir: chain the previous
        # payload so weights trained on data A then data B show BOTH.
        previous = json.loads(path.read_text(encoding="utf-8"))
        provenance = {**provenance, "previous": previous}
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return path


def model_dir_snapshot_ids(model_dir: str) -> set[str]:
    """All snapshot_ids recorded (at any nesting) in a model directory's
    DATA_PROVENANCE.json — merged dirs nest base/adapter payloads."""
    path = Path(model_dir) / "DATA_PROVENANCE.json"
    if not path.exists():
        return set()
    found: set[str] = set()

    def _walk(node) -> None:
        if isinstance(node, dict):
            value = node.get("snapshot_id")
            if isinstance(value, str) and value:
                found.add(value)
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(json.loads(path.read_text(encoding="utf-8")))
    return found


def check_base_snapshot_consistency(cfg, provenance: dict) -> None:
    """DPO must train on pairs from the SAME snapshot its base was tuned on.

    Soft when either side carries no lineage (HF hub base, legacy dirs);
    hard when both sides record snapshot_ids and they do not intersect.
    """
    pairs_id = provenance.get("snapshot_id")
    if not pairs_id:
        return
    base_ids = model_dir_snapshot_ids(cfg.model_name_or_path)
    if base_ids and pairs_id not in base_ids:
        raise SystemExit(
            f"data provenance: DPO pairs come from snapshot {pairs_id} but "
            f"the base model {cfg.model_name_or_path} records "
            f"{sorted(base_ids)} — base and pairs must share one snapshot "
            "(set allow_unverified_data=true only for smoke)"
        )
