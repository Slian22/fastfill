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
  ``decode_surface_groups`` and scored parse-only, reported separately as
  ``surface_parse_rate`` (their validation needs the parent surface frame,
  out of scope here).
- ``violations_top`` counts PRE-repair violation codes: what the model
  itself gets wrong, before repair masks it.
- ``expected_furniture_coverage`` is derived from the record input's
  ``furn`` line, over parsed floor records only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import fastfill_train  # noqa: F401  (vendor path bootstrap)
from pydantic import ValidationError
from scenesmith.growing_world.fastfill.codec import (
    decode_floor_layout,
    decode_surface_groups,
)
from scenesmith.growing_world.fastfill.repair import deterministic_repair
from scenesmith.growing_world.fastfill.schema import (
    FastFillSample,
    FloorLayout,
    RoomContentLayout,
    ValidationReport,
)
from scenesmith.growing_world.fastfill.validator import validate

from fastfill_train.data import read_records
from fastfill_train.templates import TEMPLATES, render_sft_example, split_completion

MAX_REPAIR_ROUNDS = 3  # mirrors scenesmith/scripts/fastfill_api_smoke.py
LIVE_TEMPERATURE = 0.2
LIVE_MAX_TOKENS = 1200
TOP_VIOLATIONS = 10


@dataclass(frozen=True)
class Generation:
    """One model completion for one record (latency only in live mode)."""

    completion: str
    latency_ms: Optional[float] = None


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
        generations[str(row["uid"])] = Generation(completion=str(row["completion"]))
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
) -> dict[str, Generation]:
    """Query an OpenAI-compatible endpoint with the training user message."""
    from openai import OpenAI  # lazy: not required for offline scoring

    client = OpenAI(base_url=endpoint, api_key=api_key or "EMPTY")
    generations: dict[str, Generation] = {}
    for record in records:
        user = render_sft_example(record, template).user
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user}],
            temperature=LIVE_TEMPERATURE,
            max_tokens=LIVE_MAX_TOKENS,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        completion = response.choices[0].message.content or ""
        generations[str(record["uid"])] = Generation(completion, latency_ms)
    return generations


def dump_generations(
    generations: dict[str, Generation], records: Sequence[dict], path: str | Path
) -> None:
    """Write {"uid","completion"} rows (the build_dpo_data stage-1 shape)."""
    lines = [
        json.dumps(
            {"uid": record["uid"], "completion": generations[record["uid"]].completion},
            ensure_ascii=False,
        )
        for record in records
        if record["uid"] in generations
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ----------------------------------------------------------------- scoring


def _is_surface_uid(uid: str) -> bool:
    return "#" in uid  # surface uids: <sample>#<group>


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
    try:
        floor: Optional[FloorLayout] = decode_floor_layout(layout_text, room_id)
    except (ValueError, ValidationError):
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


def score_surface(record: dict, generation: Generation) -> RecordResult:
    """Score one surface record: parse-only (see module docstring)."""
    _, layout_text = split_completion(generation.completion)
    try:
        groups = decode_surface_groups(layout_text)
        parse_ok = any(group.objects for group in groups)
    except (ValueError, ValidationError):
        parse_ok = False
    return RecordResult(
        uid=str(record["uid"]),
        source_dataset=str(record.get("source_dataset", "")),
        kind="surface",
        completion_chars=len(generation.completion),
        parse_ok=parse_ok,
        latency_ms=generation.latency_ms,
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
        if _is_surface_uid(uid):
            results.append(score_surface(record, generation))
        else:
            results.append(score_floor(record, generation, samples.get(uid)))
    return results, missing


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
        out[source] = {
            "n": len(rs),
            "parse_rate": _mean([float(r.parse_ok) for r in floor]),
            "surface_parse_rate": _mean([float(r.parse_ok) for r in surface]),
            "pass_post_repair": _mean(
                [float(r.pass_post_repair) for r in validated]
            ),
        }
    return out


def build_report(results: Sequence[RecordResult], n_missing: int) -> dict:
    """Aggregate per-record results into the report.json dict."""
    floor = [r for r in results if r.kind == "floor"]
    surface = [r for r in results if r.kind == "surface"]
    validated = [r for r in floor if r.validated]
    codes = Counter(code for r in floor for code in r.violation_codes)
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
        "n": len(results),
        "n_floor": len(floor),
        "n_surface": len(surface),
        "n_missing_generation": n_missing,
        "parse_rate": _mean([float(r.parse_ok) for r in floor]),
        "surface_parse_rate": _mean([float(r.parse_ok) for r in surface]),
        "n_validated": len(validated),
        "pass_pre_repair": _mean([float(r.pass_pre_repair) for r in validated]),
        "pass_post_repair": _mean([float(r.pass_post_repair) for r in validated]),
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


def apply_teacher(report: dict, teacher_path: str | Path) -> dict:
    """Attach the student/teacher pass_post_repair ratio (T3.2 acceptance)."""
    teacher = json.loads(Path(teacher_path).read_text(encoding="utf-8"))
    teacher_pass = teacher.get("pass_post_repair")
    student_pass = report.get("pass_post_repair")
    ratio = None
    if (
        isinstance(teacher_pass, (int, float))
        and teacher_pass > 0
        and isinstance(student_pass, (int, float))
    ):
        ratio = student_pass / teacher_pass
    return {
        **report,
        "teacher": {
            "report": str(teacher_path),
            "pass_post_repair": teacher_pass,
            "ratio_pass_post_repair": ratio,
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
        "--dump-generations", default=None, help="write generations JSONL here"
    )
    parser.add_argument(
        "--teacher-report", default=None, help="previous report.json to ratio"
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
    records = list(read_records([args.records]))
    if args.limit is not None:
        records = records[: args.limit]
    if args.generations is not None:
        generations = load_generations(args.generations)
    else:
        generations = generate_live(
            records, args.endpoint, args.model, args.api_key, args.template
        )
    if args.dump_generations:
        dump_generations(generations, records, args.dump_generations)
    samples = load_samples(args.samples) if args.samples else {}
    results, n_missing = evaluate(records, generations, samples)
    report = build_report(results, n_missing)
    if args.teacher_report:
        report = apply_teacher(report, args.teacher_report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
