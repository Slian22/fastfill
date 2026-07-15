"""Acceptance harness for FastFill layout models (T3.2).

Mirrors what ``scenesmith/scripts/fastfill_api_smoke.py`` measures, but
drives ANY OpenAI-compatible endpoint over a JSONL eval set (the
``export_sft.py`` record shape), or scores a pre-generated completions file
offline. Per floor record: build the SAME user message as training, obtain a
completion, ``split_completion`` -> ``decode_floor_layout`` (parse), and —
when ``--samples`` provides the matching ``FastFillSample`` — validate,
apply iterated ``deterministic_repair`` (same round cap as the smoke
script), and revalidate.

Documented scoring decisions:

- ``parse_ok`` requires at least one decoded object (an empty completion is
  a failure, not a trivially-parsed layout).
- Floor pass rates are computed over records WITH a matching ``--samples``
  entry (``n_validated``); parse failures inside that set count as failed
  passes, so a degrading model cannot hide behind unparseable output.
- ``expected_manipulands`` are stripped from the room context before
  validation: only the floor layer is being scored, so missing surface
  objects must not fail an otherwise-correct floor layout.
- Surface records (uid contains ``#``) are decoded with
  ``decode_surface_groups``; when ``--samples`` provides the parent sample
  they are ALSO geometrically validated in the vendored sanitizer's
  surface scope (real parent surface, synthetic room) with iterated
  deterministic repair — reported as ``surface_pass_pre_repair`` /
  ``surface_pass_post_repair`` over ``n_validated_surface``. Without
  samples they stay parse-only.
- ``violations_top`` counts PRE-repair violation codes: what the model
  itself gets wrong, before repair masks it.
- ``expected_furniture_coverage`` is derived from the record input's
  ``furn`` line, over parsed floor records only.

Record selection is DETERMINISTIC and recorded: ``--sample-mode``
(``stratified`` by source x room_type x layer by default, or ``head`` /
``shuffle``) with ``--seed``; the evaluated uid list and its sha256 are
written next to ``--out`` as ``<out>.uids.json``. Live generation defaults
to temperature 0.0 so adapter/merged/runtime comparisons are reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import fastfill_train  # noqa: F401  (vendor path bootstrap)
from fastfill_data.sanitize import surface_reports_with_repair
from pydantic import ValidationError
from scenesmith.growing_world.fastfill.codec import (
    decode_floor_layout,
    decode_surface_groups,
)
from scenesmith.growing_world.fastfill.patterns import expand_patterns
from scenesmith.growing_world.fastfill.repair import deterministic_repair
from scenesmith.growing_world.fastfill.schema import (
    FastFillSample,
    FloorLayout,
    FloorObjectSpec,
    RoomContentLayout,
    SupportSurfaceSpec,
    ValidationReport,
)
from scenesmith.growing_world.fastfill.validator import validate

from fastfill_train.data import read_records
from fastfill_train.templates import (
    TEMPLATES,
    record_layer,
    render_sft_example,
    split_completion,
)

MAX_REPAIR_ROUNDS = 3  # mirrors scenesmith/scripts/fastfill_api_smoke.py
LIVE_MAX_TOKENS = 1200
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
# Same default as the runtime OpenAIChatBackend: Qwen thinking off. Override
# with --extra-body '{}' for servers that reject the field.
DEFAULT_EXTRA_BODY = '{"chat_template_kwargs": {"enable_thinking": false}}'
SAMPLE_MODES = ("stratified", "shuffle", "head")
TOP_VIOLATIONS = 10
# Bump on any change to scoring/repair/validator semantics: teacher/student
# reports from different harness versions are not comparable for T3.2.
HARNESS_VERSION = "fastfill-eval/2"


@dataclass(frozen=True)
class Generation:
    """One model completion for one record (latency/tokens live-mode only)."""

    completion: str
    latency_ms: Optional[float] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None

    @property
    def truncated(self) -> bool:
        """The line codec is prefix-decodable: a completion cut at a line
        boundary parses cleanly with tail objects silently missing — so a
        length-stopped completion is a FAILURE, never a valid layout."""
        return self.finish_reason == "length"


@dataclass(frozen=True)
class RecordResult:
    """Per-record scoring outcome.

    ``validated`` means a matching ``--samples`` entry existed (floor
    records only); parse failures inside that set keep ``validated=True``
    with both pass flags False.
    """

    uid: str
    source_dataset: str
    kind: str  # "floor" | "surface"
    completion_chars: int
    parse_ok: bool
    latency_ms: Optional[float] = None
    validated: bool = False
    pass_pre_repair: bool = False
    pass_post_repair: bool = False
    violation_codes: tuple[str, ...] = ()
    furniture_coverage: Optional[float] = None


# ----------------------------------------------------------------- loading


def load_generations(path: str | Path) -> dict[str, Generation]:
    """Offline generations JSONL ({"uid","completion"} rows) -> uid map."""
    generations: dict[str, Generation] = {}
    for row in read_records([path]):
        missing = {"uid", "completion"} - set(row)
        if missing:
            raise ValueError(
                f"{path}: generation row missing fields {sorted(missing)}"
            )
        generations[str(row["uid"])] = Generation(
            completion=str(row["completion"]),
            finish_reason=row.get("finish_reason"),
        )
    return generations


def load_samples(path: str | Path) -> dict[str, FastFillSample]:
    """FastFillSample JSONL -> map keyed by sample_id (uid match key)."""
    samples: dict[str, FastFillSample] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = FastFillSample.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(f"{path}:{line_no}: bad FastFillSample") from exc
            samples[sample.sample_id] = sample
    return samples


def generate_live(
    records: Sequence[dict],
    endpoint: str,
    model: str,
    api_key: Optional[str],
    template: str,
    temperature: float = DEFAULT_TEMPERATURE,
    extra_body: Optional[dict] = None,
) -> dict[str, Generation]:
    """Query an OpenAI-compatible endpoint with the training user message."""
    from openai import OpenAI  # lazy: not required for offline scoring

    if extra_body is None:
        extra_body = json.loads(DEFAULT_EXTRA_BODY)
    client = OpenAI(base_url=endpoint, api_key=api_key or "EMPTY")
    generations: dict[str, Generation] = {}
    for record in records:
        user = render_sft_example(record, template).user
        start = time.perf_counter()
        kwargs: dict = {"extra_body": extra_body} if extra_body else {}
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=LIVE_MAX_TOKENS,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        choice = response.choices[0]
        completion = choice.message.content or ""
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "completion_tokens", None) if usage else None
        generations[str(record["uid"])] = Generation(
            completion,
            latency_ms,
            tokens,
            getattr(choice, "finish_reason", None),
        )
    return generations


def dump_generations(
    generations: dict[str, Generation], records: Sequence[dict], path: str | Path
) -> None:
    """Write {"uid","completion"[,"finish_reason"]} rows (stage-1 shape)."""
    lines = []
    for record in records:
        uid = str(record["uid"])
        if uid not in generations:
            continue
        generation = generations[uid]
        row: dict = {"uid": uid, "completion": generation.completion}
        if generation.finish_reason is not None:
            row["finish_reason"] = generation.finish_reason
        lines.append(json.dumps(row, ensure_ascii=False))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ----------------------------------------------------------------- scoring


def _is_surface_uid(uid: str) -> bool:
    return "#" in uid  # surface uids: <sample>#<group>


def _is_surface_record(record: dict) -> bool:
    """Canonical layer routing (shared with templates/make_snapshot)."""
    return record_layer(record) == "surface"


def _norm_category(text: str) -> str:
    return text.lower().replace(" ", "").replace("_", "")


def _category_matches(expected: str, category: str) -> bool:
    """Bidirectional normalized substring match (validator coverage rule)."""
    e, c = _norm_category(expected), _norm_category(category)
    return bool(e) and bool(c) and (e in c or c in e)


def _expected_furniture(record_input: str) -> tuple[str, ...]:
    """Parse the ``furn a,b,c`` line of an encoded RoomContext ('-' = none)."""
    for line in record_input.splitlines():
        if line.startswith("furn "):
            token = line[len("furn ") :].strip()
            if not token or token == "-":
                return ()
            return tuple(t for t in token.split(",") if t)
    return ()


def _furniture_coverage(
    expected: tuple[str, ...], categories: Sequence[str]
) -> Optional[float]:
    if not expected:
        return None
    hit = sum(
        1 for name in expected if any(_category_matches(name, c) for c in categories)
    )
    return hit / len(expected)


def _validate_with_repair(
    floor: FloorLayout, sample: FastFillSample
) -> tuple[ValidationReport, ValidationReport]:
    """(pre-repair report, post-repair report) with iterated repair."""
    context = sample.room_context.model_copy(update={"expected_manipulands": ()})
    layout = RoomContentLayout(room_id=context.room_id, floor_layout=floor)
    report_pre = validate(layout, context)
    current, report = layout, report_pre
    for _ in range(MAX_REPAIR_ROUNDS):
        if report.passed:
            break
        outcome = deterministic_repair(current, report, context)
        if outcome.layout == current:
            break  # no progress possible
        current = outcome.layout
        report = validate(current, context)
    return report_pre, report


def score_floor(
    record: dict, generation: Generation, sample: Optional[FastFillSample]
) -> RecordResult:
    """Score one floor record: parse, then validate+repair when possible."""
    _, layout_text = split_completion(generation.completion)
    room_id = sample.room_context.room_id if sample else "eval"
    floor: Optional[FloorLayout] = None
    if not generation.truncated:
        try:
            floor = decode_floor_layout(layout_text, room_id)
        except (ValueError, IndexError, ValidationError):
            floor = None
    parse_ok = floor is not None and bool(floor.objects)
    coverage = (
        _furniture_coverage(
            _expected_furniture(record.get("input", "")),
            [obj.category for obj in floor.objects],
        )
        if parse_ok and floor is not None
        else None
    )
    pass_pre = pass_post = False
    codes: tuple[str, ...] = ()
    if sample is not None and parse_ok and floor is not None:
        report_pre, report_post = _validate_with_repair(floor, sample)
        pass_pre, pass_post = report_pre.passed, report_post.passed
        codes = tuple(v.code for v in report_pre.violations)
    return RecordResult(
        uid=str(record["uid"]),
        source_dataset=str(record.get("source_dataset", "")),
        kind="floor",
        completion_chars=len(generation.completion),
        parse_ok=parse_ok,
        latency_ms=generation.latency_ms,
        validated=sample is not None,
        pass_pre_repair=pass_pre,
        pass_post_repair=pass_post,
        violation_codes=codes,
        furniture_coverage=coverage,
    )


def _surface_ground_truth(
    uid: str, sample: Optional[FastFillSample]
) -> Optional[tuple[SupportSurfaceSpec, FloorObjectSpec]]:
    """(surface, parent) for a surface uid ``<sample>#<group>``; None when
    the sample, group, surface, or parent cannot be resolved."""
    if sample is None or "#" not in uid:
        return None
    # rsplit: mansionworld sample_ids contain "#" (building names like
    # "..._fp001#3"); group_ids never do, so the LAST "#" is the separator.
    group_id = uid.rsplit("#", 1)[1]
    group = next(
        (g for g in sample.layout.surface_groups if g.group_id == group_id), None
    )
    if group is None:
        return None
    floor = sample.layout.floor_layout
    surface = next(
        (s for s in floor.support_surfaces if s.surface_id == group.surface_id),
        None,
    )
    if surface is None:
        return None
    parent = next(
        (o for o in floor.objects if o.object_id == surface.parent_object_id),
        None,
    )
    if parent is None:
        return None
    return surface, parent


def _rebind_groups(groups, surface_id: str, prefix: str = "eval"):
    """Pin groups to ``surface_id`` and uniquify ids by position.

    Mirrors ``FastFillGenerator._rebind_groups`` (generator.py): the model
    may repeat a group id within one completion and every label starts at
    ``g0``, so ids are namespaced ``<prefix>_<position>_<model id>`` — unique
    within the call by position. Object/anchor ids are rebased on the new
    group id, matching runtime so eval and runtime agree on L0_DUPLICATE_ID."""
    rebound = []
    for position, group in enumerate(groups):
        new_gid = f"{prefix}_{position}_{group.group_id}"
        objects = tuple(
            obj.model_copy(
                update={"object_id": f"{new_gid}/{obj.object_id.split('/', 1)[-1]}"}
            )
            for obj in group.objects
        )
        anchor = group.anchor_object_id
        rebound.append(
            group.model_copy(
                update={
                    "group_id": new_gid,
                    "surface_id": surface_id,
                    "objects": objects,
                    "anchor_object_id": (
                        f"{new_gid}/{anchor.split('/', 1)[-1]}" if anchor else ""
                    ),
                }
            )
        )
    return tuple(rebound)


def score_surface(
    record: dict, generation: Generation, sample: Optional[FastFillSample]
) -> RecordResult:
    """Score one surface record: parse, then surface-scope validate+repair.

    Decoded groups are pinned to the ground-truth surface (mirrors the
    runtime's per-support rebinding) and validated in the sanitizer's
    synthetic single-parent room, so floor-layer noise cannot leak in.
    """
    _, layout_text = split_completion(generation.completion)
    groups: tuple = ()
    if not generation.truncated:
        try:
            groups = decode_surface_groups(layout_text)
        except (ValueError, IndexError, ValidationError):
            groups = ()
    parse_ok = any(group.objects for group in groups)
    truth = _surface_ground_truth(str(record["uid"]), sample)
    pass_pre = pass_post = False
    codes: tuple[str, ...] = ()
    if truth is not None and parse_ok:
        surface, parent = truth
        # Mirror runtime's _rebind_groups (generator.py): pin the surface AND
        # uniquify group/object ids by position, so a model that repeats a
        # group id in one completion is not failed on L0_DUPLICATE_ID (runtime
        # namespaces those ids and passes). Then expand declared patterns into
        # concrete placements before validation (generator.py:441), so a legal
        # 1-template MATRIX is not failed on L0_PATTERN_INVALID (rows*cols!=n).
        pinned = _rebind_groups(groups, surface.surface_id)
        expanded = expand_patterns(pinned)
        room_type = str(record.get("room_type", "")) or "room"
        pre, post, _ = surface_reports_with_repair(
            expanded,
            surface,
            parent,
            room_type,
            sample.room_context.budget if sample else None,
        )
        pass_pre, pass_post = pre.passed, post.passed
        codes = tuple(v.code for v in pre.violations)
    return RecordResult(
        uid=str(record["uid"]),
        source_dataset=str(record.get("source_dataset", "")),
        kind="surface",
        completion_chars=len(generation.completion),
        parse_ok=parse_ok,
        latency_ms=generation.latency_ms,
        validated=truth is not None,
        pass_pre_repair=pass_pre,
        pass_post_repair=pass_post,
        violation_codes=codes,
    )


def evaluate(
    records: Sequence[dict],
    generations: dict[str, Generation],
    samples: dict[str, FastFillSample],
) -> tuple[list[RecordResult], int]:
    """Score every record with a generation; return (results, n_missing)."""
    results: list[RecordResult] = []
    missing = 0
    for record in records:
        uid = str(record["uid"])
        generation = generations.get(uid)
        if generation is None:
            missing += 1
            continue
        if _is_surface_record(record):
            sample = samples.get(uid.rsplit("#", 1)[0])
            results.append(score_surface(record, generation, sample))
        else:
            results.append(score_floor(record, generation, samples.get(uid)))
    return results, missing


# ------------------------------------------------------------ record selection


def _record_layer(record: dict) -> str:
    return "surface" if _is_surface_record(record) else "floor"


def _record_room_type(record: dict) -> str:
    """room_type field (new exports) or parsed from the codec input line."""
    explicit = str(record.get("room_type", ""))
    if explicit:
        return explicit
    for line in str(record.get("input", "")).splitlines():
        if line.startswith("room "):
            token = line[len("room ") :].strip()
            return token.split(" id=", 1)[0].strip()
    return ""


def select_records(
    records: list[dict], limit: Optional[int], mode: str, seed: int
) -> list[dict]:
    """Deterministic eval-slice selection.

    ``head``: first N in file order (the old behavior — file order is
    lexicographic by source room file, so this slice is NOT representative;
    kept only for reproducing old runs). ``shuffle``: seeded global shuffle.
    ``stratified``: proportional allocation over source x room_type x layer
    strata (each stratum seeded-shuffled; largest-remainder rounding), so
    small strata are represented and reruns are byte-identical.
    """
    if mode not in SAMPLE_MODES:
        raise ValueError(f"unknown sample mode {mode!r}; expected {SAMPLE_MODES}")
    if limit is None or limit >= len(records):
        return list(records)
    if mode == "head":
        return records[:limit]
    if mode == "shuffle":
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        return shuffled[:limit]
    strata: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        key = (
            str(record.get("source_dataset", "")),
            _record_room_type(record),
            _record_layer(record),
        )
        strata.setdefault(key, []).append(record)
    for key in sorted(strata):
        random.Random(f"{seed}:{key}").shuffle(strata[key])
    total = len(records)
    quotas: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str, str]]] = []
    for key in sorted(strata):
        exact = limit * len(strata[key]) / total
        quotas[key] = int(exact)
        remainders.append((exact - int(exact), key))
    shortfall = limit - sum(quotas.values())
    for _, key in sorted(remainders, key=lambda r: (-r[0], r[1]))[:shortfall]:
        quotas[key] += 1
    selected: list[dict] = []
    for key in sorted(strata):
        selected.extend(strata[key][: min(quotas[key], len(strata[key]))])
    # Under-filled strata (quota > size) leave a gap: top up deterministically.
    if len(selected) < limit:
        chosen = {str(r["uid"]) for r in selected}
        leftovers = [r for r in records if str(r["uid"]) not in chosen]
        random.Random(seed).shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])
    return selected[:limit]


def write_uid_manifest(
    records: Sequence[dict], path: Path, settings: dict
) -> dict:
    """Persist the exact evaluated uid list + its sha256 next to the report."""
    uids = [str(r["uid"]) for r in records]
    digest = hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest()
    manifest = {**settings, "n": len(uids), "uids_sha256": digest, "uids": uids}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ----------------------------------------------------------------- reporting


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _per_source(results: Sequence[RecordResult]) -> dict[str, dict]:
    by_source: dict[str, list[RecordResult]] = {}
    for result in results:
        by_source.setdefault(result.source_dataset, []).append(result)
    out: dict[str, dict] = {}
    for source, rs in sorted(by_source.items()):
        floor = [r for r in rs if r.kind == "floor"]
        surface = [r for r in rs if r.kind == "surface"]
        validated = [r for r in floor if r.validated]
        validated_surface = [r for r in surface if r.validated]
        out[source] = {
            "n": len(rs),
            "parse_rate": _mean([float(r.parse_ok) for r in floor]),
            "surface_parse_rate": _mean([float(r.parse_ok) for r in surface]),
            "pass_post_repair": _mean(
                [float(r.pass_post_repair) for r in validated]
            ),
            "surface_pass_post_repair": _mean(
                [float(r.pass_post_repair) for r in validated_surface]
            ),
        }
    return out


def build_report(results: Sequence[RecordResult], n_missing: int) -> dict:
    """Aggregate per-record results into the report.json dict."""
    floor = [r for r in results if r.kind == "floor"]
    surface = [r for r in results if r.kind == "surface"]
    validated = [r for r in floor if r.validated]
    validated_surface = [r for r in surface if r.validated]
    codes = Counter(code for r in results for code in r.violation_codes)
    chars = [r.completion_chars for r in results]
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]

    def _pct(vals, q):
        if not vals:
            return None
        s = sorted(vals)
        return s[min(len(s) - 1, int(q * len(s)))]
    coverages = [
        r.furniture_coverage for r in floor if r.furniture_coverage is not None
    ]
    return {
        "harness": HARNESS_VERSION,
        "n": len(results),
        "n_floor": len(floor),
        "n_surface": len(surface),
        "n_missing_generation": n_missing,
        # Hash of the uids that were actually scored (selection tells you
        # what was ASKED for; missing generations drop out of scoring).
        "scored_uids_sha256": hashlib.sha256(
            "\n".join(sorted(r.uid for r in results)).encode("utf-8")
        ).hexdigest(),
        "parse_rate": _mean([float(r.parse_ok) for r in floor]),
        "surface_parse_rate": _mean([float(r.parse_ok) for r in surface]),
        "n_validated": len(validated),
        "n_validated_surface": len(validated_surface),
        "pass_pre_repair": _mean([float(r.pass_pre_repair) for r in validated]),
        "pass_post_repair": _mean([float(r.pass_post_repair) for r in validated]),
        "surface_pass_pre_repair": _mean(
            [float(r.pass_pre_repair) for r in validated_surface]
        ),
        "surface_pass_post_repair": _mean(
            [float(r.pass_post_repair) for r in validated_surface]
        ),
        "violations_top": dict(codes.most_common(TOP_VIOLATIONS)),
        "expected_furniture_coverage": _mean(coverages),
        "completion_chars_mean": _mean(chars),
        "completion_chars_p95": _pct(chars, 0.95),
        "latency_ms_p50": _pct(latencies, 0.50),
        "latency_ms_p95": _pct(latencies, 0.95),
        "completion_chars_median": (statistics.median(chars) if chars else None),
        "latency_ms_mean": _mean(latencies),
        "per_source": _per_source(results),
    }


# Identity fields a formal T3.2 teacher report MUST carry and match. The
# selection block proves what was asked for; the top-level fields prove what
# was actually scored and by which harness. "Present on both sides or it
# counts as a mismatch" — optional-field skipping allowed a teacher with
# only uids_sha256 to pass as a formal comparison.
_TEACHER_SELECTION_FIELDS = (
    "uids_sha256",
    "temperature",
    "sample_mode",
    "limit",
    "template",
    "records_sha256",
    "samples_sha256",
    "generation_mode",
)
_TEACHER_REPORT_FIELDS = (
    "harness",
    "n",
    "n_validated",
    "n_missing_generation",
    "scored_uids_sha256",
)


def _valid_rate(value: object) -> bool:
    """A pass rate must be a finite non-bool float in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


_MISSING = object()


def _identity_mismatches(report: dict, teacher: dict) -> list[str]:
    mismatches: list[str] = []
    teacher_sel = teacher.get("selection") or {}
    student_sel = report.get("selection") or {}
    for key in _TEACHER_SELECTION_FIELDS:
        t_val = teacher_sel.get(key, _MISSING)
        s_val = student_sel.get(key, _MISSING)
        if t_val is _MISSING or s_val is _MISSING:
            mismatches.append(
                f"selection.{key} missing on "
                f"{'teacher' if t_val is _MISSING else 'student'} — "
                "regenerate both reports with the current harness"
            )
        elif t_val != s_val:
            mismatches.append(
                f"selection.{key}: teacher {t_val!r} != student {s_val!r}"
            )
    for key in _TEACHER_REPORT_FIELDS:
        t_val = teacher.get(key, _MISSING)
        s_val = report.get(key, _MISSING)
        if t_val is _MISSING or s_val is _MISSING:
            mismatches.append(
                f"{key} missing on "
                f"{'teacher' if t_val is _MISSING else 'student'}"
            )
        elif t_val != s_val:
            mismatches.append(f"{key}: teacher {t_val!r} != student {s_val!r}")
    return mismatches


def apply_teacher(
    report: dict, teacher_path: str | Path, *, allow_mismatch: bool = False
) -> dict:
    """Attach the student/teacher pass_post_repair ratio (T3.2 acceptance).

    Accepts either a prior eval_layout report (``pass_post_repair``) or a
    fastfill_api_smoke summary (``pass_rate_post_repair``). A teacher file
    with neither key — or a non-finite / out-of-range / boolean rate — is a
    hard error.

    Identity check: a formal T3.2 ratio needs the SAME harness version on
    the SAME samples, with the SAME number actually scored. Every field in
    ``_TEACHER_SELECTION_FIELDS`` + ``_TEACHER_REPORT_FIELDS`` must be
    present on both sides and equal; anything else raises unless
    ``allow_mismatch`` — and is then recorded as a trend-only comparison.
    """
    teacher = json.loads(Path(teacher_path).read_text(encoding="utf-8"))
    teacher_pass = teacher.get(
        "pass_post_repair", teacher.get("pass_rate_post_repair")
    )
    if not _valid_rate(teacher_pass):
        raise ValueError(
            f"teacher report {teacher_path} has no valid pass_post_repair / "
            f"pass_rate_post_repair (got {teacher_pass!r}; need a finite "
            "float in [0, 1])"
        )
    if float(teacher_pass) == 0.0 and not allow_mismatch:
        raise ValueError(
            f"teacher report {teacher_path} has pass_post_repair == 0 — a "
            "teacher that passes nothing is not a usable T3.2 baseline "
            "(--allow-teacher-mismatch for a trend-only record)"
        )
    student_pass = report.get("pass_post_repair")
    if not _valid_rate(student_pass) and not allow_mismatch:
        raise ValueError(
            f"student pass_post_repair is not a valid rate ({student_pass!r})"
            " — the eval scored no validated floor records; a formal T3.2 "
            "ratio cannot be computed"
        )
    mismatches = _identity_mismatches(report, teacher)
    if mismatches and not allow_mismatch:
        raise ValueError(
            "teacher/student identity mismatch — T3.2 requires the same "
            "eval_layout harness on the same UID manifest: "
            + "; ".join(mismatches)
            + " (pass --allow-teacher-mismatch for an explicitly "
            "trend-only ratio)"
        )
    ratio = None
    if float(teacher_pass) > 0 and _valid_rate(student_pass):
        ratio = float(student_pass) / float(teacher_pass)
    return {
        **report,
        "teacher": {
            "report": str(teacher_path),
            "pass_post_repair": teacher_pass,
            "ratio_pass_post_repair": ratio,
            "identity_mismatches": mismatches,
            "trend_only": bool(mismatches),
        },
    }


# ----------------------------------------------------------------------- CLI


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print_summary(report: dict) -> None:
    print(
        f"n={report['n']} (floor={report['n_floor']} "
        f"surface={report['n_surface']} "
        f"missing={report['n_missing_generation']})"
    )
    print(
        f"parse_rate={_fmt(report['parse_rate'])} "
        f"surface_parse_rate={_fmt(report['surface_parse_rate'])}"
    )
    print(
        f"pass_pre_repair={_fmt(report['pass_pre_repair'])} "
        f"pass_post_repair={_fmt(report['pass_post_repair'])} "
        f"(n_validated={report['n_validated']})"
    )
    print(
        f"surface_pass_pre_repair={_fmt(report['surface_pass_pre_repair'])} "
        f"surface_pass_post_repair={_fmt(report['surface_pass_post_repair'])} "
        f"(n_validated_surface={report['n_validated_surface']})"
    )
    teacher = report.get("teacher")
    if teacher is not None:
        print(
            "teacher ratio (pass_post_repair): "
            f"student {_fmt(report['pass_post_repair'])} / "
            f"teacher {_fmt(teacher['pass_post_repair'])} = "
            f"{_fmt(teacher['ratio_pass_post_repair'])} (target 0.8-0.9)"
        )


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastFill layout eval: parse / validate / repair metrics"
    )
    parser.add_argument("--records", required=True, help="export_sft JSONL")
    parser.add_argument("--samples", default=None, help="FastFillSample JSONL")
    parser.add_argument("--endpoint", default=None, help="OpenAI-compatible URL")
    parser.add_argument("--model", default=None, help="model name (live mode)")
    parser.add_argument("--api-key", default=None, help="API key (live mode)")
    parser.add_argument(
        "--generations", default=None, help='offline {"uid","completion"} JSONL'
    )
    parser.add_argument("--template", default="direct", choices=TEMPLATES)
    parser.add_argument("--limit", type=int, default=None, help="max records")
    parser.add_argument(
        "--sample-mode",
        default="stratified",
        choices=SAMPLE_MODES,
        help=(
            "how --limit selects records: stratified (source x room_type x "
            "layer, default), shuffle, or head (old non-representative slice)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="selection seed"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="live sampling temperature (0.0 = deterministic comparisons)",
    )
    parser.add_argument(
        "--extra-body",
        default=DEFAULT_EXTRA_BODY,
        help=(
            "JSON object merged into live requests (default disables Qwen "
            "thinking; pass '{}' for servers that reject the field)"
        ),
    )
    parser.add_argument(
        "--dump-generations", default=None, help="write generations JSONL here"
    )
    parser.add_argument(
        "--teacher-report",
        default=None,
        help="teacher eval_layout report.json to ratio (same harness+UIDs)",
    )
    parser.add_argument(
        "--allow-teacher-mismatch",
        action="store_true",
        help=(
            "accept a teacher report from a different sample set/harness "
            "(ratio is marked trend_only; NOT valid for T3.2 acceptance)"
        ),
    )
    parser.add_argument("--out", required=True, help="report.json path")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.generations is None and not (args.endpoint and args.model):
        raise SystemExit(
            "provide --generations for offline scoring, or --endpoint and "
            "--model for live mode"
        )
    records = select_records(
        list(read_records([args.records])), args.limit, args.sample_mode, args.seed
    )
    out = Path(args.out)
    manifest = write_uid_manifest(
        records,
        out.with_suffix(".uids.json"),
        {
            "records_path": str(args.records),
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "limit": args.limit,
        },
    )
    if args.generations is not None:
        generations = load_generations(args.generations)
    else:
        try:
            extra_body = json.loads(args.extra_body) if args.extra_body else {}
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--extra-body is not valid JSON: {exc}") from exc
        if not isinstance(extra_body, dict):
            raise SystemExit("--extra-body must be a JSON object")
        generations = generate_live(
            records,
            args.endpoint,
            args.model,
            args.api_key,
            args.template,
            args.temperature,
            extra_body,
        )
    if args.dump_generations:
        dump_generations(generations, records, args.dump_generations)
    samples = load_samples(args.samples) if args.samples else {}
    results, n_missing = evaluate(records, generations, samples)
    report = build_report(results, n_missing)
    tokens = [
        generations[str(r["uid"])].completion_tokens
        for r in records
        if str(r["uid"]) in generations
        and generations[str(r["uid"])].completion_tokens is not None
    ]
    report["usage_completion_tokens_total"] = sum(tokens) if tokens else None
    report["usage_completion_tokens_mean"] = _mean(tokens) if tokens else None
    report["n_truncated"] = sum(
        1
        for r in records
        if str(r["uid"]) in generations and generations[str(r["uid"])].truncated
    )
    report["selection"] = {
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "limit": args.limit,
        "temperature": args.temperature,
        "template": args.template,
        "uids_sha256": manifest["uids_sha256"],
        # Content hash of the records file: same UID list over edited
        # records must not pass as the same manifest.
        "records_sha256": hashlib.sha256(
            Path(args.records).read_bytes()
        ).hexdigest(),
        # Samples drive floor validation — a different ground-truth file
        # changes every pass rate, so it is part of the eval's identity.
        "samples_sha256": (
            hashlib.sha256(Path(args.samples).read_bytes()).hexdigest()
            if args.samples
            else None
        ),
        "generation_mode": "live" if args.endpoint else "offline",
        "endpoint_model": args.model,
        "extra_body_sha256": (
            hashlib.sha256(str(args.extra_body).encode("utf-8")).hexdigest()[:16]
            if args.endpoint
            else None
        ),
        "uids_manifest": str(out.with_suffix(".uids.json")),
    }
    if args.teacher_report:
        report = apply_teacher(
            report, args.teacher_report, allow_mismatch=args.allow_teacher_mismatch
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
