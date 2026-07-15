"""Build DPO preference pairs for the FastFill planner (two stages).

``stage2`` — programmatic hard negatives: for each ``FastFillSample`` the
injectors in ``injectors.py`` perturb the ground-truth layout into a
validator-failing variant. ``chosen`` is the record rendered exactly like SFT
(vendored ``export_sft`` instructions + codec encoders, completion side via
``templates.render_sft_example``); ``rejected`` is the same prompt with the
injected layout encoded by the same codec/template. Every emitted pair is
validator-verified (perturbed layout fails with an expected code, original
produces none of them) — unverifiable or unrenderable attempts are counted,
never silently dropped.

``stage1`` — model self-negatives: pairs a floor SFT record (``--contexts``)
with a model completion (``--generations``, rows ``{"uid", "completion"}``
from ``eval_layout --dump-generations``). ``rejected`` is kept ONLY when the
completion fails to parse (rejected = raw text) or parses but fails the
vendored validator while the ground truth passes. When ``--samples`` provides
the paired ``FastFillSample`` the RoomContext is rebuilt for full validation
("validated" mode, ``expected_manipulands`` stripped since floor records
carry no surface objects); otherwise only parse success can be checked
("parse_only" mode) — the stats file records which mode judged each row.

Usage::

    PYTHONPATH=src python3 -m fastfill_train.build_dpo_data stage2 \
        --in data/samples.jsonl --out data/dpo/stage2_pairs.jsonl \
        --template direct --per-sample 2 --seed 42

    PYTHONPATH=src python3 -m fastfill_train.build_dpo_data stage1 \
        --contexts data/sft/floor_sft.jsonl --generations gens.jsonl \
        --out data/dpo/stage1_pairs.jsonl --samples data/samples.jsonl

A ``<out>.stats.json`` report is written next to ``--out`` for both stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TextIO

from pydantic import ValidationError

import fastfill_train  # noqa: F401  (vendor path bootstrap)
from fastfill_data.export_sft import (
    BBOX_UNVERIFIED_NOTE,
    FLOOR_INSTRUCTION,
    SURFACE_INSTRUCTION,
    UNVERIFIED_YAW_NOTE,
    build_support_context,
)
from fastfill_data.sanitize import (
    FLOOR_UNREPAIRED_NOTE,
    has_note,
    has_sanitized_note,
)
from scenesmith.growing_world.fastfill.codec import (
    decode_floor_layout,
    encode_floor_layout,
    encode_room_context,
    encode_support_context,
    encode_surface_groups,
)
from scenesmith.growing_world.fastfill.schema import (
    FastFillSample,
    LicenseTag,
    RoomContentLayout,
    RoomContext,
    SurfaceObjectGroup,
    ValidationReport,
)
from scenesmith.growing_world.fastfill.validator import validate

from fastfill_train.data import (
    SNAPSHOT_REQUIRED_FIELDS,
    read_records,
    record_identity_sha,
    split_bucket,
)
from fastfill_train.injectors import (
    ALL_INJECTORS,
    FLOOR_LEVEL,
    Injection,
    original_violation_codes,
    verify_injection,
)
from fastfill_train.templates import TEMPLATES, render_sft_example, split_completion

DEFAULT_PER_SAMPLE = 2
DEFAULT_SEED = 42
_SPLIT_BUCKETS = 10_000
# Above this fraction of foreign split_keys the snapshot clearly belongs to
# a different corpus (legit dropouts — e.g. codec-error-only samples never
# exported — stay far below it).
_FOREIGN_KEY_LIMIT = 0.5


def _bump(counter: dict[str, int], key: str, n: int = 1) -> None:
    counter[key] = counter.get(key, 0) + n


@dataclass(frozen=True)
class SnapshotFilter:
    """Holdout predicate + corpus-identity data from a make_snapshot dir."""

    is_holdout: Callable[[str], bool]
    known_keys: frozenset[str]
    known_records: frozenset[str]  # record_identity_sha manifest
    snapshot_id: str
    license_mode: Optional[str]
    allow_unresolved: bool

    def is_foreign_key(self, split_key: str) -> bool:
        return split_key not in self.known_keys

    def authorizes(self, record: dict) -> bool:
        """True when this exact record content was frozen into the corpus."""
        return record_identity_sha(record) in self.known_records


def load_holdout_filter(snapshot_path: Path | None) -> Optional[SnapshotFilter]:
    """Build a :class:`SnapshotFilter` from a make_snapshot sidecar.

    DPO chosen labels ARE training labels: pairs built from the full corpus
    would put Stage-0 heldout/test ground truths into DPO train (the DPO
    loader's own val split uses a smaller fraction, so heldout buckets land
    on the train side). Without a snapshot no filtering happens — callers
    should treat that as smoke-only.

    Validation is fail-closed: full sidecar schema (schema_version + every
    required field), fraction/seed value ranges, the recorded
    ``snapshot_id`` fingerprint, the ``SPLIT_KEYS.json`` hash AND its
    record-identity manifest are all verified — legacy or hand-edited
    sidecars are rejected.
    """
    if snapshot_path is None:
        return None
    snapshot_path = Path(snapshot_path)
    sidecar = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict) or any(
        k not in sidecar for k in SNAPSHOT_REQUIRED_FIELDS
    ):
        missing = [
            k
            for k in SNAPSHOT_REQUIRED_FIELDS
            if not isinstance(sidecar, dict) or k not in sidecar
        ]
        raise ValueError(
            f"{snapshot_path} is not a current make_snapshot sidecar "
            f"(missing {missing}) — legacy/hand-written sidecars are not "
            "accepted; re-freeze with the current make_snapshot"
        )
    if "split_bucket" not in str(sidecar["rule"]):
        raise ValueError(
            f"{snapshot_path} uses split rule {sidecar['rule']!r}; this "
            "filter only reproduces the data.split_bucket house-first rule"
        )
    seed = sidecar["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{snapshot_path}: seed must be an int, got {seed!r}")
    val = float(sidecar["val_fraction"])
    test = float(sidecar.get("test_fraction", 0.0))
    if not (0.0 <= val < 1.0 and 0.0 <= test < 1.0 and val + test < 1.0):
        raise ValueError(
            f"{snapshot_path}: fractions out of range "
            f"(val={val}, test={test}; each in [0,1), sum < 1)"
        )
    snapshot_id = sidecar["snapshot_id"]
    payload = {k: v for k, v in sidecar.items() if k != "snapshot_id"}
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    if recomputed != snapshot_id:
        raise ValueError(
            f"{snapshot_path}: snapshot_id mismatch (recorded "
            f"{snapshot_id}, recomputed {recomputed}) — sidecar was "
            "edited after freezing"
        )
    keys_path = snapshot_path.parent / sidecar["split_keys_file"]
    if not keys_path.exists():
        raise ValueError(
            f"{snapshot_path} references {sidecar['split_keys_file']} but it "
            "is missing next to the sidecar — snapshot dir is incomplete"
        )
    digest = hashlib.sha256(keys_path.read_bytes()).hexdigest()[:16]
    if digest != sidecar.get("split_keys_sha256"):
        raise ValueError(
            f"{keys_path}: content hash {digest} does not match the "
            f"sidecar's split_keys_sha256 — key list was modified"
        )
    keys_payload = json.loads(keys_path.read_text(encoding="utf-8"))
    if not isinstance(keys_payload, dict) or "record_hashes" not in keys_payload:
        raise ValueError(
            f"{keys_path} has no record_hashes manifest — re-freeze with "
            "the current make_snapshot (content-level authorization is "
            "required for pair building)"
        )
    cutoff = int((val + test) * _SPLIT_BUCKETS)

    def is_holdout(split_key: str) -> bool:
        return split_bucket(split_key, seed) < cutoff

    return SnapshotFilter(
        is_holdout=is_holdout,
        known_keys=frozenset(keys_payload.get("split_keys") or ()),
        known_records=frozenset(keys_payload["record_hashes"]),
        snapshot_id=snapshot_id,
        license_mode=sidecar.get("license_mode"),
        allow_unresolved=bool(sidecar.get("allow_unresolved_licenses")),
    )


def _enforce_route_match(
    snapshot_filter: "SnapshotFilter",
    license_mode: str,
    allow_unresolved: bool,
    snapshot: Path,
) -> None:
    """Pairs must follow the snapshot's license route exactly — a research
    invocation over a permissive snapshot would smuggle NC chosen labels
    into a permissive model's DPO stage."""
    if snapshot_filter.license_mode and license_mode != snapshot_filter.license_mode:
        raise SystemExit(
            f"--license-mode {license_mode} != snapshot's "
            f"{snapshot_filter.license_mode} ({snapshot}) — pairs must "
            "follow the snapshot's route"
        )
    if allow_unresolved != snapshot_filter.allow_unresolved:
        raise SystemExit(
            f"--allow-unresolved-licenses={allow_unresolved} != snapshot's "
            f"{snapshot_filter.allow_unresolved} ({snapshot}) — unresolved-"
            "license policy must match the frozen corpus"
        )


def _enforce_corpus_match(
    foreign: int, total: int, stats_target: dict, snapshot: Path
) -> None:
    """Hard-fail when the input corpus is clearly not the snapshot's."""
    if total and foreign / total > _FOREIGN_KEY_LIMIT:
        raise SystemExit(
            f"corpus mismatch: {foreign}/{total} input split_keys are not in "
            f"{snapshot}'s SPLIT_KEYS — this snapshot was frozen from a "
            "different corpus"
        )
    if foreign:
        stats_target["excluded_foreign_split_key"] = foreign


def _load_samples(path: str | Path) -> list[FastFillSample]:
    return [FastFillSample.model_validate(r) for r in read_records([path])]


def _write_stats(out_path: Path, stats: dict) -> Path:
    # Content hash of the pairs file binds the stats (and its snapshot_id /
    # license_mode lineage) to these exact pairs for the training-side gate;
    # stats_id fingerprints the stats payload itself so a hand-written stats
    # file cannot vouch for arbitrary pairs.
    if out_path.exists():
        stats["out_sha256_16"] = hashlib.sha256(
            out_path.read_bytes()
        ).hexdigest()[:16]
    stats["stats_id"] = hashlib.sha256(
        json.dumps(stats, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats_path


# -------------------------------------------------------------------- stage 2


def _floor_record(sample: FastFillSample) -> dict:
    """SFT floor record (same shape as vendored ``export_sft._floor_record``)."""
    return {
        "uid": sample.sample_id,
        "split_key": sample.provenance.split_key,
        "source_dataset": sample.provenance.source_dataset,
        "license": sample.provenance.license_tag.value,
        "room_type": sample.room_context.room_type,
        "layer": "floor",  # explicit layer: a floor uid may contain '#'
        "instruction": FLOOR_INSTRUCTION,
        "input": encode_room_context(sample.room_context),
        "output": encode_floor_layout(sample.layout.floor_layout),
    }


def _surface_record(sample: FastFillSample, group: SurfaceObjectGroup) -> Optional[dict]:
    """SFT surface record for one group; ``None`` if its context is unresolved."""
    context = build_support_context(sample, group)
    if context is None:
        return None
    return {
        "uid": f"{sample.sample_id}#{group.group_id}",
        "split_key": sample.provenance.split_key,
        "source_dataset": sample.provenance.source_dataset,
        "license": sample.provenance.license_tag.value,
        "room_type": sample.room_context.room_type,
        "layer": "surface",
        "instruction": SURFACE_INSTRUCTION,
        "input": encode_support_context(context),
        "output": encode_surface_groups([group]),
    }


def _pair_record(
    sample: FastFillSample, injection: Injection
) -> tuple[Optional[dict], Optional[str], str]:
    """(chosen SFT record, rejected output codec text, skip_reason)."""
    if injection.level == FLOOR_LEVEL:
        record = _floor_record(sample)
        bad_output = encode_floor_layout(injection.bad_layout.floor_layout)
        return record, bad_output, ""
    group = next(
        g for g in sample.layout.surface_groups if g.group_id == injection.group_id
    )
    record = _surface_record(sample, group)
    if record is None:
        return None, None, "surface_context_unresolved"
    bad_group = next(
        g
        for g in injection.bad_layout.surface_groups
        if g.group_id == injection.group_id
    )
    return record, encode_surface_groups([bad_group]), ""


def _stage2_row(
    sample: FastFillSample,
    injection: Injection,
    template: str,
    snapshot_filter: Optional[SnapshotFilter] = None,
) -> tuple[Optional[dict], str]:
    """Render one DPO pair row; ``(None, reason)`` when it must be skipped."""
    try:
        record, bad_output, reason = _pair_record(sample, injection)
    except ValueError:  # codec-illegal token (e.g. '|' in a category)
        return None, "codec_error"
    if record is None or bad_output is None:
        return None, reason
    if snapshot_filter is not None and not snapshot_filter.authorizes(record):
        # Record-level authorization: the chosen content (incl. surface
        # groups, categories, licenses) must be byte-identical to what the
        # snapshot froze — a tampered sample sharing key/geometry with the
        # corpus must not smuggle a modified chosen label in.
        return None, "unauthorized_record"
    room_id = sample.room_context.room_id
    chosen = render_sft_example(record, template, room_id=room_id)
    rejected = render_sft_example(
        {**record, "output": bad_output}, template, room_id=room_id
    )
    if chosen.assistant == rejected.assistant:
        return None, "identical_completion"  # e.g. z_local is not encoded
    row = {
        "prompt": [{"role": "user", "content": chosen.user}],
        "chosen": [{"role": "assistant", "content": chosen.assistant}],
        "rejected": [{"role": "assistant", "content": rejected.assistant}],
        "split_key": sample.provenance.split_key,
        "license": sample.provenance.license_tag.value,
        "uid": record["uid"],
        "injector": injection.name,
        "expected_codes": list(injection.expected_codes),
    }
    return row, ""


def _emit_sample_pairs(
    sample: FastFillSample,
    rotated_names: list[str],
    template: str,
    per_sample: int,
    out: TextIO,
    per_injector: dict[str, int],
    skipped: dict[str, int],
    snapshot_filter: Optional[SnapshotFilter] = None,
) -> int:
    """Try injectors in round-robin order until ``per_sample`` pairs emitted."""
    original_codes = original_violation_codes(sample)
    # Same floor-label gates as export_sft: unverified-yaw, sanitizer-unrepaired,
    # and unverified-bbox 3D-FRONT floors must not contribute FLOOR-level chosen
    # labels (surface pairs stay legal).  Each reason gets its own stats key so
    # .stats.json reflects the actual exclusion cause, not a stale label.
    notes = sample.provenance.notes
    if UNVERIFIED_YAW_NOTE in notes:
        skip_floor_reason: str | None = "excluded_unverified_yaw_floor"
    elif BBOX_UNVERIFIED_NOTE in notes:
        skip_floor_reason = "excluded_bbox_unverified_floor"
    elif has_note(notes, FLOOR_UNREPAIRED_NOTE):
        skip_floor_reason = "excluded_floor_unrepaired"
    else:
        skip_floor_reason = None
    emitted = 0
    for name in rotated_names:
        if emitted >= per_sample:
            break
        injection = ALL_INJECTORS[name](sample)
        if injection is None:
            _bump(skipped, f"not_applicable:{name}")
            continue
        if skip_floor_reason is not None and injection.level == FLOOR_LEVEL:
            _bump(skipped, f"{skip_floor_reason}:{name}")
            continue
        if not verify_injection(sample, injection, original_codes):
            _bump(skipped, f"verification_failed:{name}")
            continue
        row, reason = _stage2_row(sample, injection, template, snapshot_filter)
        if row is None:
            _bump(skipped, f"{reason}:{name}")
            continue
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        _bump(per_injector, name)
        emitted += 1
    return emitted


LICENSE_MODES = ("permissive", "research")
_LICENSE_BY_VALUE = {tag.value: tag for tag in LicenseTag}


def _license_exclusion(
    tag: LicenseTag, license_mode: str, allow_unresolved: bool
) -> str | None:
    """Mirror of export_sft's allowlist: permissive = PERMISSIVE only;
    research adds CC_BY_NC; unresolved tags (LICENSE_PENDING/UNKNOWN) need
    the explicit override. Stage-2 chosen completions are training labels,
    so they must pass the same gate as the SFT export — the old exact-NC
    check let pending/unknown samples re-enter a permissive DPO corpus."""
    if tag is LicenseTag.PERMISSIVE:
        return None
    if license_mode == "research" and (
        tag is LicenseTag.CC_BY_NC or allow_unresolved
    ):
        return None
    return f"excluded_license_{tag.value}"


def run_stage2(
    in_path: Path,
    out_path: Path,
    template: str,
    per_sample: int,
    seed: int,
    license_mode: str = "permissive",
    allow_unresolved: bool = False,
    snapshot: Path | None = None,
    require_sanitized: bool = True,
) -> dict:
    """Build stage-2 pairs from FastFillSample JSONL; return the stats dict.

    Applies the SAME sample-level 铁律 gates as export_sft (license
    allowlist, unverified-yaw / unrepaired floors, sanitized-note hard gate)
    plus the ``--snapshot`` holdout filter — stage-2 chosen completions are
    training labels and must obey the training-data rules, not just the SFT
    export.
    """
    if license_mode not in LICENSE_MODES:
        raise ValueError(
            f"license_mode must be one of {LICENSE_MODES}, got {license_mode!r}"
        )
    snapshot_filter = load_holdout_filter(snapshot)
    if snapshot_filter is not None:
        _enforce_route_match(snapshot_filter, license_mode, allow_unresolved, snapshot)
    names = sorted(ALL_INJECTORS)
    random.Random(seed).shuffle(names)
    counts: dict[str, int] = {"input_samples": 0, "pairs": 0}
    per_injector: dict[str, int] = {}
    skipped: dict[str, int] = {}
    foreign = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for index, sample in enumerate(_load_samples(in_path)):
            counts["input_samples"] += 1
            if snapshot_filter is not None:
                key = sample.provenance.split_key
                if snapshot_filter.is_foreign_key(key):
                    foreign += 1
                    continue
                if snapshot_filter.is_holdout(key):
                    _bump(skipped, "excluded_holdout")
                    continue
            if require_sanitized and not has_sanitized_note(
                sample.provenance.notes
            ):
                _bump(skipped, "excluded_unsanitized")
                continue
            license_reason = _license_exclusion(
                sample.provenance.license_tag, license_mode, allow_unresolved
            )
            if license_reason is not None:
                _bump(skipped, license_reason)
                continue
            start = index % len(names)
            rotated = names[start:] + names[:start]
            emitted = _emit_sample_pairs(
                sample,
                rotated,
                template,
                per_sample,
                out,
                per_injector,
                skipped,
                snapshot_filter,
            )
            counts["pairs"] += emitted
            if emitted == 0:
                _bump(skipped, "sample_yielded_no_pairs")
            elif emitted < per_sample:
                _bump(skipped, "sample_underfilled")
    if snapshot_filter is not None:
        _enforce_corpus_match(foreign, counts["input_samples"], skipped, snapshot)
    stats = {
        "stage": "stage2",
        "in_path": str(in_path),
        "out_path": str(out_path),
        "template": template,
        "per_sample": per_sample,
        "seed": seed,
        "license_mode": license_mode,
        "allow_unresolved_licenses": allow_unresolved,
        "snapshot": str(snapshot) if snapshot else None,
        "snapshot_id": snapshot_filter.snapshot_id if snapshot_filter else None,
        "require_sanitized": require_sanitized,
        "counts": counts,
        "per_injector": dict(sorted(per_injector.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    if snapshot is None:
        stats["warning"] = (
            "no --snapshot: pairs may contain Stage-0 heldout/test ground "
            "truths — smoke use only"
        )
    _write_stats(out_path, stats)
    return stats


# -------------------------------------------------------------------- stage 1


def _floor_only_report(
    codec_text: str, context: RoomContext
) -> ValidationReport:
    """Validate a floor-codec completion as a floor-only room layout."""
    floor = decode_floor_layout(codec_text, context.room_id)
    layout = RoomContentLayout(room_id=context.room_id, floor_layout=floor)
    return validate(layout, context)


def _judge_generation(
    record: dict, completion: str, sample: Optional[FastFillSample]
) -> tuple[str, list[str]]:
    """Classify one model completion against its ground-truth record.

    Returns ``(verdict, violation_codes)`` with verdict one of:
    ``reject_parse_failure``, ``reject_validation_failure``,
    ``skip_generation_passes``, ``skip_ground_truth_fails``,
    ``skip_parse_only_passes``.
    """
    _, layout_text = split_completion(completion)
    room_id = sample.room_context.room_id if sample else "eval"
    try:
        decode_floor_layout(layout_text, room_id)
    except (ValueError, IndexError, ValidationError):
        return "reject_parse_failure", []
    if sample is None:
        return "skip_parse_only_passes", []
    # Floor records carry no surface objects, so expected_manipulands would
    # fail for ground truth and generation alike — strip them for both sides.
    context = sample.room_context.model_copy(update={"expected_manipulands": ()})
    if not _floor_only_report(record["output"], context).passed:
        return "skip_ground_truth_fails", []
    gen_report = _floor_only_report(layout_text, context)
    if gen_report.passed:
        return "skip_generation_passes", []
    return "reject_validation_failure", [v.code for v in gen_report.violations]


def _stage1_row(
    record: dict, completion: str, template: str, verdict: str, codes: list[str]
) -> dict:
    chosen = render_sft_example(record, template)
    return {
        "prompt": [{"role": "user", "content": chosen.user}],
        "chosen": [{"role": "assistant", "content": chosen.assistant}],
        "rejected": [{"role": "assistant", "content": completion}],
        "split_key": record.get("split_key", ""),
        "license": record.get("license", "unrecorded"),
        "uid": record["uid"],
        "reason": verdict.removeprefix("reject_"),
        "violation_codes": codes,
    }


def run_stage1(
    contexts_path: Path,
    generations_path: Path,
    out_path: Path,
    samples_path: Optional[Path],
    template: str,
    max_reject_codes: int = 2,
    snapshot: Path | None = None,
    license_mode: str = "permissive",
    allow_unresolved: bool = False,
) -> dict:
    """Pair SFT ground truths with failing model completions; return stats.

    Near-miss gate (Architect-Ant precedent): model-pair DPO scored higher
    on rules but produced WORSE layouts when preference pairs contained
    many non-target differences (shortcut/reward-hacking risk). Rejected
    completions failing with more than ``max_reject_codes`` distinct
    violation codes are therefore skipped and counted — stage 2's
    single-factor synthetic pairs are the primary recipe; stage 1 is the
    near-miss complement (OptiScene reported +7pp from model pairs, so it
    stays available as an ablation arm; disable the gate with
    --max-reject-codes 0).

    ``--snapshot`` drops contexts whose split_key falls in the Stage-0
    heldout/test buckets — their ground truths must never become chosen
    labels. Chosen labels also pass the license allowlist: exported context
    records carry a ``license`` field; records without one are unresolved
    and fail closed.
    """
    if license_mode not in LICENSE_MODES:
        raise ValueError(
            f"license_mode must be one of {LICENSE_MODES}, got {license_mode!r}"
        )
    snapshot_filter = load_holdout_filter(snapshot)
    if snapshot_filter is not None:
        _enforce_route_match(snapshot_filter, license_mode, allow_unresolved, snapshot)
    contexts: dict[str, dict] = {}
    holdout_uids: set[str] = set()
    license_skipped: dict[str, int] = {}
    foreign = 0
    n_context_records = 0
    for record in read_records([contexts_path]):
        uid = str(record["uid"])
        n_context_records += 1
        if snapshot_filter is not None:
            key = record.get("split_key", "")
            if snapshot_filter.is_foreign_key(key) or not snapshot_filter.authorizes(
                record
            ):
                foreign += 1
                continue
            if snapshot_filter.is_holdout(key):
                holdout_uids.add(uid)
                continue
        tag_value = record.get("license")
        tag = _LICENSE_BY_VALUE.get(tag_value, LicenseTag.UNKNOWN)
        license_reason = _license_exclusion(tag, license_mode, allow_unresolved)
        if license_reason is not None:
            _bump(license_skipped, license_reason)
            continue
        contexts[uid] = record
    if snapshot_filter is not None:
        _enforce_corpus_match(foreign, n_context_records, license_skipped, snapshot)
    samples: dict[str, FastFillSample] = {}
    if samples_path is not None:
        samples = {s.sample_id: s for s in _load_samples(samples_path)}
    counts: dict[str, int] = {"generations": 0, "pairs": 0}
    modes: dict[str, int] = {"validated": 0, "parse_only": 0}
    skipped: dict[str, int] = dict(license_skipped)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for gen in read_records([generations_path]):
            if "uid" not in gen or "completion" not in gen:
                raise ValueError(f"generation row needs uid+completion: {gen!r}")
            counts["generations"] += 1
            if not str(gen["completion"]).strip():
                _bump(skipped, "empty_completion")
                continue
            if str(gen["uid"]) in holdout_uids:
                _bump(skipped, "excluded_holdout")
                continue
            record = contexts.get(gen["uid"])
            if record is None:
                _bump(skipped, "unknown_uid")
                continue
            sample = samples.get(gen["uid"])
            _bump(modes, "validated" if sample is not None else "parse_only")
            verdict, codes = _judge_generation(record, gen["completion"], sample)
            if not verdict.startswith("reject_"):
                _bump(skipped, verdict.removeprefix("skip_"))
                continue
            distinct_codes = len(set(codes))
            if (
                verdict == "reject_validation_failure"
                and max_reject_codes > 0
                and distinct_codes > max_reject_codes
            ):
                _bump(skipped, "too_many_violations_not_near_miss")
                continue
            row = _stage1_row(record, gen["completion"], template, verdict, codes)
            if row["chosen"][0]["content"] == row["rejected"][0]["content"]:
                _bump(skipped, "identical_completion")
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["pairs"] += 1
    stats = {
        "stage": "stage1",
        "contexts_path": str(contexts_path),
        "generations_path": str(generations_path),
        "samples_path": str(samples_path) if samples_path else None,
        "out_path": str(out_path),
        "template": template,
        "max_reject_codes": max_reject_codes,
        "license_mode": license_mode,
        "allow_unresolved_licenses": allow_unresolved,
        "snapshot": str(snapshot) if snapshot else None,
        "snapshot_id": snapshot_filter.snapshot_id if snapshot_filter else None,
        "counts": counts,
        "judge_modes": modes,
        "judge_mode_note": (
            "validated = RoomContext rebuilt from --samples, full validator; "
            "parse_only = no matching sample, only parse success checked "
            "(decodable completions are counted as passing)"
        ),
        "skipped": dict(sorted(skipped.items())),
    }
    if snapshot is None:
        stats["warning"] = (
            "no --snapshot: pairs may contain Stage-0 heldout/test ground "
            "truths — smoke use only"
        )
    _write_stats(out_path, stats)
    return stats


# ------------------------------------------------------------------------ CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fastfill_train.build_dpo_data",
        description="Build FastFill DPO preference pairs (stage1 / stage2)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p2 = sub.add_parser("stage2", help="injector-based hard negatives")
    p2.add_argument("--in", dest="in_path", required=True, help="samples JSONL")
    p2.add_argument("--out", required=True, help="output pairs JSONL")
    p2.add_argument("--template", choices=TEMPLATES, default="direct")
    p2.add_argument("--per-sample", type=int, default=DEFAULT_PER_SAMPLE)
    p2.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p2.add_argument(
        "--license-mode",
        choices=LICENSE_MODES,
        default="permissive",
        help=(
            "permissive: chosen labels only from PERMISSIVE samples (铁律 2, "
            "default, mirrors export_sft); research: also CC_BY_NC"
        ),
    )
    p2.add_argument(
        "--allow-unresolved-licenses",
        action="store_true",
        help=(
            "DANGER: in research mode, also use LICENSE_PENDING/UNKNOWN "
            "samples as chosen labels"
        ),
    )
    p2.add_argument(
        "--snapshot",
        default=None,
        help=(
            "make_snapshot SNAPSHOT.json — drop samples in the Stage-0 "
            "heldout/test buckets (REQUIRED for real training runs)"
        ),
    )
    p2.add_argument(
        "--allow-no-snapshot",
        action="store_true",
        help="smoke only: build pairs without heldout/test filtering",
    )
    p2.add_argument(
        "--require-sanitized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hard gate on the sanitize.py 'sanitized=v1' note (default on)",
    )
    p1 = sub.add_parser("stage1", help="model self-negatives vs ground truth")
    p1.add_argument("--contexts", required=True, help="floor SFT JSONL")
    p1.add_argument("--generations", required=True, help="{uid,completion} JSONL")
    p1.add_argument("--out", required=True, help="output pairs JSONL")
    p1.add_argument(
        "--samples",
        default=None,
        help="FastFillSample JSONL for full validation (else parse-only mode)",
    )
    p1.add_argument("--template", choices=TEMPLATES, default="direct")
    p1.add_argument(
        "--license-mode",
        choices=LICENSE_MODES,
        default="permissive",
        help=(
            "permissive: chosen labels only from records tagged permissive "
            "(exported license field; missing tag fails closed); research: "
            "also cc_by_nc"
        ),
    )
    p1.add_argument(
        "--allow-unresolved-licenses",
        action="store_true",
        help=(
            "DANGER: in research mode, also use records with pending/"
            "unknown/missing license tags as chosen labels"
        ),
    )
    p1.add_argument(
        "--snapshot",
        default=None,
        help=(
            "make_snapshot SNAPSHOT.json — drop contexts in the Stage-0 "
            "heldout/test buckets (REQUIRED for real training runs)"
        ),
    )
    p1.add_argument(
        "--allow-no-snapshot",
        action="store_true",
        help="smoke only: build pairs without heldout/test filtering",
    )
    p1.add_argument(
        "--max-reject-codes",
        type=int,
        default=2,
        help=(
            "near-miss gate: skip rejected completions with more distinct "
            "violation codes than this (0 disables; Architect-Ant precedent)"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    # Normalize: --snapshot '' must not slip past the gate and then decay to
    # None at the Path conversion below.
    snapshot_arg = (args.snapshot or "").strip() or None
    if snapshot_arg is None and not args.allow_no_snapshot:
        raise SystemExit(
            "--snapshot is required: pairs built from the full corpus leak "
            "Stage-0 heldout/test ground truths into DPO train. Pass the "
            "snapshot sidecar, or --allow-no-snapshot for a smoke build."
        )
    snapshot_path = Path(snapshot_arg) if snapshot_arg else None
    if args.command == "stage2":
        if args.per_sample < 1:
            raise SystemExit("--per-sample must be >= 1")
        stats = run_stage2(
            Path(args.in_path),
            Path(args.out),
            args.template,
            args.per_sample,
            args.seed,
            args.license_mode,
            args.allow_unresolved_licenses,
            snapshot_path,
            args.require_sanitized,
        )
    else:
        stats = run_stage1(
            Path(args.contexts),
            Path(args.generations),
            Path(args.out),
            Path(args.samples) if args.samples else None,
            args.template,
            args.max_reject_codes,
            snapshot_path,
            args.license_mode,
            args.allow_unresolved_licenses,
        )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
