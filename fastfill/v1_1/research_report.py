"""Synthesize verified V1.1 evaluations with prespecified, dev-only inference.

The fixed primary outcome is repaired zero-collision acceptance over all requests.
Inference pairs rooms and resamples whole (source, group) clusters. Test outcomes
are descriptive; this report never selects a winner or launches more training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import groupby
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

from fastfill.v1_1 import evaluation_protocol as protocol


MODELS = ("sft", "dpo1", "final")
DEV_SPLITS = ("dev_rooms", "dev_constrained_rooms")
TEST_SPLITS = ("test_rooms", "test_constrained_rooms")
COMPARISONS = (("sft", "dpo1"), ("dpo1", "final"))
INFERENCE = {
    "primary": "zero_collision_accepted after repair; failures count false",
    "comparisons": ["dpo1 - sft", "final - dpo1"],
    "selection_data": "dev only; ordinary and constrained dev reported separately",
    "bootstrap": "paired whole (source, group) clusters, room-weighted mean difference",
    "bootstrap_draws": 2000, "bootstrap_seed": 42, "interval": "95% percentile",
    "evidence_rule": "supported if lower > 0; regression if upper < 0; otherwise inconclusive; at least two clusters required",
    "guards": "Coverage and size changes are descriptive warnings; no quality threshold or automatic winner",
    "multiplicity": "Four prespecified dev comparisons; intervals are descriptive and not adjusted for multiplicity",
    "test_policy": "No test-based selection, tuning, or automatic training",
    "causal_scope": "Stage comparisons do not isolate the effect of cleaning; a matched old-data control is absent",
}


def _read(path: Path) -> dict[str, Any]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return content


def _successful(row: dict[str, Any]) -> bool:
    return row.get("error") is None and row.get("parsed") is True


def _size(row: dict[str, Any]) -> float | None:
    if not _successful(row):
        return 0.0
    size = row.get("size_iou_3d")
    return float(size) if type(size) in (int, float) and math.isfinite(size) and 0 <= size <= 1 else None


def _constraint_rate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    outcomes = [bool(value) and _successful(row) for row in rows for _, value in row.get(field, [])]
    return {"constraints": len(outcomes), "rate": sum(outcomes) / len(outcomes) if outcomes else None}


def describe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if _successful(row)]
    sizes = [_size(row) for row in rows]
    return {
        "rooms": len(rows), "complete_rooms": len(complete), "complete_rate": len(complete) / len(rows),
        "zero_collision_accepted_all_requests": sum(_successful(row) and row.get("zero_collision_accepted") is True for row in rows) / len(rows),
        "raw_main_collision_rate_complete": sum(row.get("coll_main", 0) > 0 for row in complete) / len(complete) if complete else None,
        "raw_main_collision_denominator": len(complete),
        "raw_zero_main_collision_all_requests": sum(_successful(row) and row.get("coll_main") == 0 for row in rows) / len(rows),
        "size_iou_all_requests": sum(sizes) / len(rows) if all(size is not None for size in sizes) else None,
        "size_iou_missing_successful_rooms": sum(size is None for size in sizes),
        "constraints_raw": _constraint_rate(rows, "constraints"),
        "constraints_repaired": _constraint_rate(rows, "constraints_repaired"),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _cluster_interval(pairs: list[tuple[tuple[str, str], int]]) -> tuple[int, list[float] | None]:
    grouped = [list(items) for _, items in groupby(sorted(pairs), key=lambda item: item[0])]
    aggregates = [(sum(value for _, value in items), len(items)) for items in grouped]
    if len(aggregates) < 2:
        return len(aggregates), None
    rng = random.Random(INFERENCE["bootstrap_seed"])
    samples = []
    for _ in range(INFERENCE["bootstrap_draws"]):
        sampled = rng.choices(aggregates, k=len(aggregates))
        samples.append(sum(total for total, _ in sampled) / sum(count for _, count in sampled))
    return len(aggregates), [_quantile(samples, 0.025), _quantile(samples, 0.975)]


def _paired_rows(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> None:
    fields = ("uid", "source", "group")
    if not before or len(before) != len(after):
        raise ValueError("Invalid paired cohort length")
    for first, second in zip(before, after):
        if any(not isinstance(first.get(key), str) or not first[key] or first[key] != second.get(key) for key in fields):
            raise ValueError("Invalid paired cohort: exact ordered uid/source/group equality required")


def _common_sizes(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [(_size(first), _size(second)) for first, second in zip(before, after)
             if _successful(first) and _successful(second) and _size(first) is not None and _size(second) is not None]
    baseline = sum(first for first, _ in pairs) / len(pairs) if pairs else None
    candidate = sum(second for _, second in pairs) / len(pairs) if pairs else None
    return {"rooms": len(pairs), "baseline": baseline, "candidate": candidate,
            "difference": candidate - baseline if pairs else None,
            "interpretation": "Conditional on both outputs completing with size metrics; selection bias is possible"}


def _guard_warnings(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    coverage = ["Candidate complete rate decreased; inspect failure types and coverage before further research."] if after["complete_rate"] < before["complete_rate"] else []
    sizes = (before["size_iou_all_requests"], after["size_iou_all_requests"])
    if None in sizes:
        return coverage + ["Size comparison unavailable because successful rows lack valid size IoU."]
    return coverage + (["Candidate all-request size IoU decreased; inspect size behavior before further research."] if sizes[1] < sizes[0] else [])


def compare_rows(before: list[dict[str, Any]], after: list[dict[str, Any]], split: str,
                 baseline: str, candidate: str) -> dict[str, Any]:
    if split not in DEV_SPLITS:
        raise ValueError("Inferential comparisons are restricted to dev splits")
    _paired_rows(before, after)
    differences = [int(_successful(second) and second.get("zero_collision_accepted") is True)
                   - int(_successful(first) and first.get("zero_collision_accepted") is True)
                   for first, second in zip(before, after)]
    clusters, interval = _cluster_interval([((row["source"], row["group"]), delta) for row, delta in zip(before, differences)])
    evidence = "supported" if interval and interval[0] > 0 else "regression" if interval and interval[1] < 0 else "inconclusive"
    first, second = describe_rows(before), describe_rows(after)
    sources = sorted({row["source"] for row in before})
    return {
        "split": split, "baseline": baseline, "candidate": candidate,
        "primary": {"rooms": len(before), "clusters": clusters, "difference": sum(differences) / len(differences),
                    "baseline_rate": first["zero_collision_accepted_all_requests"],
                    "candidate_rate": second["zero_collision_accepted_all_requests"], "ci95": interval, "evidence": evidence},
        "descriptive": {"baseline": first, "candidate": second,
                        "complete_rate_difference": second["complete_rate"] - first["complete_rate"],
                        "size_iou_all_requests_difference": second["size_iou_all_requests"] - first["size_iou_all_requests"]
                        if first["size_iou_all_requests"] is not None and second["size_iou_all_requests"] is not None else None,
                        "common_complete_size_iou": _common_sizes(before, after)},
        "by_source": {source: {"baseline": describe_rows([row for row in before if row["source"] == source]),
                               "candidate": describe_rows([row for row in after if row["source"] == source])} for source in sources},
        "guard_warnings": _guard_warnings(first, second),
        "recommendation": "Inspect dev evidence and guard warnings before proposing a new experiment; no automatic winner or training action.",
    }


def _verified_cell(output: Path, frozen: dict[str, Any], tag: str, split: str,
                   state: dict[str, Any]) -> dict[str, Any]:
    target = output / tag / split
    cells = state.get("cells", {})
    if not isinstance(cells, dict) or not isinstance(cells.get(split, {}), dict):
        raise ValueError(f"Invalid evaluation receipt for {tag}/{split}")
    cell = cells.get(split, {})
    expected = {"schema_version": frozen["schema_version"], "dataset": frozen["datasets"][split],
                "model": state.get("model_identity", {}), "settings": frozen["settings"], "code_hashes": frozen["code_hashes"]}
    valid = state.get("validation_complete", True) and cell.get("status") == "complete" and cell.get("identity") == expected and protocol.evaluation_complete(target, expected)
    return {"status": "complete" if valid else "incomplete", "out_dir": str(target),
            "expected_rooms": frozen["datasets"][split]["line_count"],
            "metrics": _read(target / "metrics.json") if valid else None}


def _dev_rows(path: Path) -> list[dict[str, Any]]:
    fields = ("uid", "source", "group", "error", "parsed", "zero_collision_accepted", "coll_main", "size_iou_3d", "constraints", "constraints_repaired")
    with path.open(encoding="utf-8") as stream:
        return [{key: row[key] for key in fields if key in row} for row in map(json.loads, stream)]


def build_report(output: str | Path) -> dict[str, Any]:
    output = Path(output).resolve()
    frozen = _read(output / "protocol.json")
    states = {tag: _read(output / "state" / f"{tag}.json") if (output / "state" / f"{tag}.json").exists() else {} for tag in MODELS}
    evaluations = {f"{tag}/{split}": _verified_cell(output, frozen, tag, split, states[tag]) for tag in MODELS for split in protocol.SPLITS}
    complete = sum(item["status"] == "complete" for item in evaluations.values())
    dev_ready = all(evaluations[f"{tag}/{split}"]["status"] == "complete" for tag in MODELS for split in DEV_SPLITS)
    errors, comparisons = [], []
    if dev_ready:
        try:
            rows = {f"{tag}/{split}": _dev_rows(output / tag / split / "predictions.jsonl") for tag in MODELS for split in DEV_SPLITS}
            comparisons = [compare_rows(rows[f"{before}/{split}"], rows[f"{after}/{split}"], split, before, after)
                           for split in DEV_SPLITS for before, after in COMPARISONS]
        except (ValueError, KeyError, TypeError) as exc:
            errors = [f"Invalid paired dev evidence: {exc}"]
            dev_ready = False
    verified = complete == 12 and not errors
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if verified else "incomplete", "complete": verified,
            "completed_evaluations": complete, "expected_evaluations": 12, "dev_ready": dev_ready,
            "inference_protocol": INFERENCE, "dev_comparisons": comparisons,
            "evaluations": evaluations, "errors": errors,
            "limitations": ["These intervals describe four prespecified comparisons without multiplicity adjustment.",
                            "The size metric uses reference dimensions and is not a category-level realism assessment.",
                            "Repaired acceptance includes postprocessing; raw metrics are reported separately.",
                            INFERENCE["causal_scope"], "This report does not select a winner, tune on test data, or launch subsequent training."]}


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# FastFill V1.1 research synthesis", "", f"Status: {summary['status']}. Verified full evaluations: {summary['completed_evaluations']}/12.", "",
             "No test-based selection, tuning, or automatic training. Final synthesis requires all 12 evaluations; provisional dev comparisons require all six dev evaluations.", "",
             "Primary outcome: repaired zero-collision acceptance over all requests. Failures count as false. Paired bootstrap resamples whole (source, group) clusters, with 2,000 draws, seed 42, and 95% percentile intervals.", "",
             "| Dev split | Comparison | Rooms / clusters | Difference | 95% interval | Evidence |",
             "| --- | --- | ---: | ---: | --- | --- |"]
    for item in summary.get("dev_comparisons", []):
        primary = item["primary"]
        interval = primary["ci95"]
        ci = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "Unavailable: fewer than two clusters"
        lines.append(f"| {item['split']} | {item['candidate']} − {item['baseline']} | {primary['rooms']} / {primary['clusters']} | {primary['difference']:.4f} | {ci} | {primary['evidence']} |")
    lines += ["", "Coverage/size guard warnings are descriptive; they do not redefine the primary evidence status or set a quality threshold.", ""]
    lines += [f"- {item['split']}, {item['candidate']}: {warning}" for item in summary.get("dev_comparisons", []) for warning in item["guard_warnings"]]
    lines += ["", "## Descriptive metrics", "", "Full population/source, raw/repaired, complete-rate, size-IoU, and constraint metrics are retained in research-summary.json. Test metrics are descriptive only. Common-complete-pair size IoU includes its selected denominator; failed requests score zero in all-request size IoU.", "",
              "## Limitations", ""] + [f"- {item}" for item in summary.get("limitations", [])]
    lines += ["", "## Pending or invalid evaluations", ""]
    lines += [f"- {name}: {cell['status']}" for name, cell in summary.get("evaluations", {}).items() if cell["status"] != "complete"]
    lines += [f"- {error}" for error in summary.get("errors", [])]
    return "\n".join(lines) + "\n"


def _write_markdown(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def generate_report(output: str | Path) -> dict[str, Any]:
    summary = build_report(output)
    protocol.write_json_atomic(Path(output) / "research-summary.json", summary)
    _write_markdown(Path(output) / "research-summary.md", _markdown(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        summary = generate_report(args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        summary = {"status": "incomplete", "complete": False, "completed_evaluations": 0,
                   "errors": [f"Cannot validate research inputs: {exc}"]}
        protocol.write_json_atomic(Path(args.output) / "research-summary.json", summary)
        _write_markdown(Path(args.output) / "research-summary.md", _markdown(summary))
    print(json.dumps({key: summary[key] for key in ("status", "complete", "completed_evaluations")}, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
