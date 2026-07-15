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

from fastfill_train.data import SNAPSHOT_REQUIRED_FIELDS


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


def _fingerprint(payload: dict, exclude: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in payload.items() if k != exclude}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()[:16]


def _validated_sidecar(sidecar_path: Path) -> dict:
    """Load + fully validate a SNAPSHOT.json: schema, fields, fingerprint."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict) or any(
        k not in sidecar for k in SNAPSHOT_REQUIRED_FIELDS
    ):
        missing = [
            k
            for k in SNAPSHOT_REQUIRED_FIELDS
            if not isinstance(sidecar, dict) or k not in sidecar
        ]
        raise SystemExit(
            f"data provenance: {sidecar_path} is not a current make_snapshot "
            f"sidecar (missing {missing}) — re-freeze with the current "
            "make_snapshot, or set allow_unverified_data=true"
        )
    if _fingerprint(sidecar, "snapshot_id") != sidecar["snapshot_id"]:
        raise SystemExit(
            f"data provenance: {sidecar_path} snapshot_id "
            f"{sidecar['snapshot_id']} does not recompute — sidecar was edited"
        )
    return sidecar


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
        sidecar = _validated_sidecar(sidecar_path)
        snapshot_id = sidecar["snapshot_id"]
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
            "snapshot_id": snapshot_id,
            "license_mode": sidecar["license_mode"],
            "license_counts": (sidecar.get("license_counts") or {}).get(
                path.stem
            ),
        }
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for field in ("snapshot_id", "out_sha256_16", "license_mode", "stats_id"):
            if not stats.get(field):
                raise SystemExit(
                    f"data provenance: {stats_path} has no {field} — rebuild "
                    "the pairs with the current build_dpo_data (--snapshot "
                    "required), or set allow_unverified_data=true"
                )
        if _fingerprint(stats, "stats_id") != stats["stats_id"]:
            raise SystemExit(
                f"data provenance: {stats_path} stats_id does not recompute "
                "— stats file was edited"
            )
        actual = _sha(path)
        if actual != stats["out_sha256_16"]:
            raise SystemExit(
                f"data provenance: {path} hash {actual} != pair-builder's "
                f"{stats['out_sha256_16']} — pairs changed after building"
            )
        # Follow the back-reference: the SNAPSHOT the pairs were built
        # against must still exist, validate, and match the recorded id.
        snapshot_ref = Path(stats.get("snapshot") or "")
        if not snapshot_ref.exists():
            raise SystemExit(
                f"data provenance: {stats_path} references snapshot "
                f"{snapshot_ref} which does not exist — restore it or set "
                "allow_unverified_data=true"
            )
        ref_sidecar = _validated_sidecar(snapshot_ref)
        if ref_sidecar["snapshot_id"] != stats["snapshot_id"]:
            raise SystemExit(
                f"data provenance: {stats_path} snapshot_id "
                f"{stats['snapshot_id']} != {snapshot_ref}'s "
                f"{ref_sidecar['snapshot_id']} — snapshot dir was replaced"
            )
        return {
            "file": str(path),
            "sha256_16": actual,
            "stats": str(stats_path),
            "snapshot": str(snapshot_ref),
            "snapshot_id": stats["snapshot_id"],
            "license_mode": stats["license_mode"],
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
    if cfg.allow_unverified_data is True:
        return {
            "verified": False,
            "note": "allow_unverified_data=true — smoke/ad-hoc data",
            "files": [str(f) for f in cfg.dataset_files],
            "code_commit": _git_commit(),
        }
    if cfg.allow_unverified_data:  # truthy non-bool, e.g. the string "false"
        raise SystemExit(
            "allow_unverified_data must be a YAML boolean (true/false), got "
            f"{cfg.allow_unverified_data!r} — quoted strings are truthy and "
            "would silently skip verification"
        )
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


def attach_resume_lineage(cfg, provenance: dict) -> dict:
    """Record the resumed checkpoint's own data lineage.

    ``resume_from_checkpoint`` as a path may live in ANOTHER run's output
    dir — weights carry that run's data too, so its DATA_PROVENANCE.json is
    embedded as ``resumed_from`` (or flagged missing)."""
    resume = cfg.resume_from_checkpoint
    if not isinstance(resume, str) or not resume:
        return provenance
    ckpt = Path(resume)
    for candidate in (ckpt / "DATA_PROVENANCE.json", ckpt.parent / "DATA_PROVENANCE.json"):
        if candidate.exists():
            return {
                **provenance,
                "resumed_from": {
                    "checkpoint": resume,
                    "provenance": json.loads(candidate.read_text(encoding="utf-8")),
                },
            }
    return {
        **provenance,
        "resumed_from": {
            "checkpoint": resume,
            "provenance": None,
            "note": "no DATA_PROVENANCE.json found next to the checkpoint",
        },
    }


def write_provenance(cfg, provenance: dict) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "DATA_PROVENANCE.json"
    if path.exists():
        # Rerun into the same output dir: chain the previous payload so
        # weights trained on data A then data B show BOTH. (For a resume
        # from a DIFFERENT dir, see attach_resume_lineage.)
        previous = json.loads(path.read_text(encoding="utf-8"))
        provenance = {
            **provenance,
            "previous": previous,
            "previous_note": "earlier payload found in this output_dir",
        }
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
    """DPO must train on pairs from the SAME snapshot/route its base used.

    Hard requirements when the pairs carry a snapshot_id and the base is a
    local model dir: the base must have a DATA_PROVENANCE.json, the pairs'
    snapshot_id must appear in its lineage, and license modes must not mix.
    HF hub ids (no local dir) stay soft — they carry no lineage to check.
    """
    pairs_id = provenance.get("snapshot_id")
    if not pairs_id:
        return
    base_dir = Path(cfg.model_name_or_path)
    if not base_dir.is_dir():
        return  # HF hub id — no local lineage to compare
    if not (base_dir / "DATA_PROVENANCE.json").exists():
        raise SystemExit(
            f"data provenance: base model {base_dir} has no "
            "DATA_PROVENANCE.json — retrain/merge with the current tooling "
            "or set allow_unverified_data=true for smoke"
        )
    base_ids = model_dir_snapshot_ids(cfg.model_name_or_path)
    if base_ids and pairs_id not in base_ids:
        raise SystemExit(
            f"data provenance: DPO pairs come from snapshot {pairs_id} but "
            f"the base model {cfg.model_name_or_path} records "
            f"{sorted(base_ids)} — base and pairs must share one snapshot "
            "(set allow_unverified_data=true only for smoke)"
        )
    pairs_mode = provenance.get("license_mode")
    base_modes = _model_dir_license_modes(cfg.model_name_or_path)
    if pairs_mode and base_modes and pairs_mode not in base_modes:
        raise SystemExit(
            f"data provenance: DPO pairs are {pairs_mode}-route but the base "
            f"model records {sorted(base_modes)} — license routes must not mix"
        )


def _model_dir_license_modes(model_dir: str) -> set[str]:
    path = Path(model_dir) / "DATA_PROVENANCE.json"
    if not path.exists():
        return set()
    found: set[str] = set()

    def _walk(node) -> None:
        if isinstance(node, dict):
            value = node.get("license_mode")
            if isinstance(value, str) and value:
                found.add(value)
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(json.loads(path.read_text(encoding="utf-8")))
    return found
