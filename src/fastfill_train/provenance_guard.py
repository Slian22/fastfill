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
import os
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
    """Record the resumed checkpoint's own data lineage — and refuse route
    contamination.

    Two official resume forms are covered: an explicit checkpoint path
    (may live in ANOTHER run's output dir) and ``resume_from_checkpoint:
    true`` (Trainer picks the latest checkpoint inside ``output_dir`` — the
    dir may hold a previous run's weights from a different route). Either
    way the resumed lineage is embedded as ``resumed_from``, and a lineage
    recording a DIFFERENT license route than the current run is a hard
    error: resuming research/NC weights into a permissive run puts
    NC-derived weights into the permissive model, which no report field
    can undo."""
    resume = cfg.resume_from_checkpoint
    if resume is False or resume is None or resume == "":
        return provenance
    if resume is True:
        # Trainer will resume the latest checkpoint in output_dir; its
        # lineage is the DATA_PROVENANCE.json already sitting there.
        candidates = (Path(cfg.output_dir) / "DATA_PROVENANCE.json",)
        checkpoint_label = f"{cfg.output_dir} (latest checkpoint)"
    else:
        ckpt = Path(str(resume))
        candidates = (
            ckpt / "DATA_PROVENANCE.json",
            ckpt.parent / "DATA_PROVENANCE.json",
        )
        checkpoint_label = str(resume)
    for candidate in candidates:
        if candidate.exists():
            resumed = json.loads(candidate.read_text(encoding="utf-8"))
            current_mode = provenance.get("license_mode")
            resumed_modes = {
                v
                for v in _walk_values(resumed, "license_mode")
                if isinstance(v, str) and v
            }
            if current_mode and resumed_modes and resumed_modes != {current_mode}:
                raise SystemExit(
                    f"data provenance: checkpoint {checkpoint_label} records "
                    f"license modes {sorted(resumed_modes)} but this run is "
                    f"{current_mode} — resuming across license routes "
                    "contaminates the weights (set allow_unverified_data="
                    "true only for smoke)"
                )
            current_id = provenance.get("snapshot_id")
            resumed_ids = {
                v
                for v in _walk_values(resumed, "snapshot_id")
                if isinstance(v, str) and v
            }
            if current_id and resumed_ids and resumed_ids != {current_id}:
                raise SystemExit(
                    f"data provenance: checkpoint {checkpoint_label} was "
                    f"trained on snapshot(s) {sorted(resumed_ids)} but this "
                    f"run uses {current_id} — warm-starting across snapshots "
                    "mixes corpora in the weights (train fresh, or set "
                    "allow_unverified_data=true only for smoke)"
                )
            return {
                **provenance,
                "resumed_from": {
                    "checkpoint": checkpoint_label,
                    "provenance": resumed,
                },
            }
    # No DATA_PROVENANCE next to the checkpoint: the weights are
    # unverifiable — refuse rather than record a null lineage.
    if cfg.allow_unverified_data is True:
        return {
            **provenance,
            "resumed_from": {
                "checkpoint": checkpoint_label,
                "provenance": None,
                "note": "no DATA_PROVENANCE.json found next to the checkpoint",
            },
        }
    raise SystemExit(
        f"data provenance: checkpoint {checkpoint_label} has no "
        "DATA_PROVENANCE.json — refusing to resume unverifiable weights "
        "(set allow_unverified_data=true only for smoke)"
    )


def write_provenance(cfg, provenance: dict) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "DATA_PROVENANCE.json"
    # Multi-GPU launches (accelerate/deepspeed) run this in EVERY rank;
    # only rank 0 owns the file — concurrent writes corrupt it and
    # sequential ones self-chain a fake `previous` ladder.
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    if rank not in ("", "0"):
        return path
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))

        def _core(payload: dict) -> dict:
            return {
                k: v
                for k, v in payload.items()
                if k not in ("previous", "previous_note")
            }

        if _core(previous) == _core(provenance):
            # Identical rerun (restart of the same run): keep the existing
            # payload instead of chaining a duplicate lineage level.
            return path
        # Rerun into the same output dir with DIFFERENT data: chain the
        # previous payload so weights trained on data A then B show BOTH.
        provenance = {
            **provenance,
            "previous": previous,
            "previous_note": "earlier payload found in this output_dir",
        }
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return path


def _walk_values(payload, key: str) -> set:
    """Every str/bool value of ``key`` (at any nesting) in a payload."""
    found: set = set()

    def _walk(node) -> None:
        if isinstance(node, dict):
            if key in node and isinstance(node[key], (str, bool)):
                found.add(node[key])
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(payload)
    return found


def model_dir_snapshot_ids(model_dir: str) -> set[str]:
    """All snapshot_ids recorded (at any nesting) in a model directory's
    DATA_PROVENANCE.json — merged dirs nest base/adapter payloads."""
    path = Path(model_dir) / "DATA_PROVENANCE.json"
    if not path.exists():
        return set()
    return {
        v
        for v in _walk_values(
            json.loads(path.read_text(encoding="utf-8")), "snapshot_id"
        )
        if isinstance(v, str) and v
    }


def check_base_snapshot_consistency(cfg, provenance: dict) -> None:
    """DPO must train on pairs from EXACTLY the snapshot/route its base used.

    Hard requirements when the pairs carry a snapshot_id and the base is a
    local model dir: the base's lineage must be non-empty, contain no
    verified=false entries, and its snapshot_id/license_mode sets must
    EQUAL the pairs' — set membership is not enough (a base that mixes a
    research checkpoint into its history still carries NC-derived weights).
    HF hub ids (no local dir) stay soft — they carry no lineage to check.
    """
    pairs_id = provenance.get("snapshot_id")
    if not pairs_id:
        return
    base_dir = Path(cfg.model_name_or_path)
    if not base_dir.is_dir():
        return  # HF hub id — no local lineage to compare
    prov_path = base_dir / "DATA_PROVENANCE.json"
    if not prov_path.exists():
        raise SystemExit(
            f"data provenance: base model {base_dir} has no "
            "DATA_PROVENANCE.json — retrain/merge with the current tooling "
            "or set allow_unverified_data=true for smoke"
        )
    base_payload = json.loads(prov_path.read_text(encoding="utf-8"))
    base_ids = {
        v
        for v in _walk_values(base_payload, "snapshot_id")
        if isinstance(v, str) and v
    }
    if not base_ids:
        raise SystemExit(
            f"data provenance: base model {base_dir} has an empty lineage "
            "(no snapshot_id anywhere in DATA_PROVENANCE.json) — not "
            "acceptable for production DPO"
        )
    if False in _walk_values(base_payload, "verified"):
        raise SystemExit(
            f"data provenance: base model {base_dir} lineage contains a "
            "verified=false entry — it was trained on unverified data and "
            "cannot back a production DPO"
        )
    if base_ids != {pairs_id}:
        raise SystemExit(
            f"data provenance: DPO pairs come from snapshot {pairs_id} but "
            f"the base model {cfg.model_name_or_path} records "
            f"{sorted(base_ids)} — base and pairs must use EXACTLY one "
            "shared snapshot (set allow_unverified_data=true only for smoke)"
        )
    pairs_mode = provenance.get("license_mode")
    base_modes = {
        v
        for v in _walk_values(base_payload, "license_mode")
        if isinstance(v, str) and v
    }
    if pairs_mode and base_modes != {pairs_mode}:
        raise SystemExit(
            f"data provenance: DPO pairs are {pairs_mode}-route but the base "
            f"model records {sorted(base_modes) or ['nothing']} — license "
            "routes must match exactly"
        )
