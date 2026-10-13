"""Independent, hash-bound full evaluation of the three fixed V1.1 checkpoints.

Prepare a new output directory once; run each model in a separate bounded job.
Every model evaluates dev, constrained dev, test, and constrained test in order.
No training, job submission, checkpoint selection, or test-based tuning occurs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

from fastfill.v1_1 import evaluation_protocol as protocol

MODELS = {"sft": "sft_merged", "dpo1": "dpo1_merged", "final": "final_merged"}
INTERPRETATION = (
    "Completion verifies all 12 full-cohort evaluations; it does not assert model quality "
    "or select a test-set winner. Checkpoints and settings are fixed before evaluation. "
    "Parse, generation, and geometry failures remain in the room denominators. "
    "Raw and repaired metrics, sources, and held-out populations are reported separately."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _evaluation_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    from fastfill.v1_1.evaluate_v1_1 import evaluation_code_hashes, evaluation_settings

    args = argparse.Namespace(all_rows=True, gt_only=False, max_len=40960,
                              max_new_tokens=8192, batch=16, tp=1, use_vllm=True,
                              chunk_size=256, workers=32, limit=None)
    return evaluation_settings(args), evaluation_code_hashes()


def prepare(root: str | Path, data: str | Path, run: str | Path, output: str | Path,
            *, python: str = sys.executable, lineage_mode: str = "auto") -> dict[str, Any]:
    root, data, run, output = (Path(item).resolve() for item in (root, data, run, output))
    if any(output == model or output.is_relative_to(model) for model in ((run / v).resolve() for v in MODELS.values())):
        raise ValueError("Evaluation output must be isolated from model directories")
    if output == run or output == data or output.is_relative_to(data):
        raise ValueError("Evaluation output must be isolated from run and dataset roots")
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("Run prepare using the Python module in the requested code root")
    if lineage_mode not in {"auto", "clean", "legacy"}:
        raise ValueError("Unknown lineage mode")
    if lineage_mode == "auto":
        lineage_mode = "clean" if run.name.startswith("fastfill_v1_1_clean_") else "legacy"
    configuration = {"schema_version": 1, "root": str(root), "data": str(data),
                     "run": str(run), "output": str(output), "python": str(Path(python).resolve()),
                     "models": MODELS, "splits": list(protocol.SPLITS), "lineage_mode": lineage_mode}
    with _lock(output / ".prepare.lock"):
        target = output / "runner.json"
        if target.exists() and _read(target) != configuration:
            raise ValueError("Frozen runner configuration differs; use a new output directory")
        settings, code_hashes = _evaluation_contract()
        protocol.freeze_protocol(data, output, settings, code_hashes=code_hashes)
        protocol.write_json_atomic(target, configuration)
    return report(output)


def _validate_protocol(configuration: dict[str, Any], output: Path) -> dict[str, Any]:
    if Path(configuration["root"]) != Path(__file__).resolve().parents[2]:
        raise ValueError("Evaluation must run from its frozen code root")
    if str(output) != configuration["output"]:
        raise ValueError("Evaluation output differs from frozen runner configuration")
    settings, code_hashes = _evaluation_contract()
    return protocol.freeze_protocol(configuration["data"], output, settings, code_hashes=code_hashes)


def _model_stats(model: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {name: ((Path(model["path"]) / name).stat().st_size,
                   (Path(model["path"]) / name).stat().st_mtime_ns)
            for name in model["files"]}


def _identity(frozen: dict[str, Any], model: dict[str, Any], split: str) -> dict[str, Any]:
    return {"schema_version": frozen["schema_version"], "dataset": frozen["datasets"][split],
            "model": model, "settings": frozen["settings"], "code_hashes": frozen["code_hashes"]}


def _command(configuration: dict[str, Any], model: dict[str, Any], split: str, output: Path) -> list[str]:
    return [configuration["python"], "-u", "-m", "fastfill.v1_1.evaluate_v1_1",
            "--model", model["path"], "--rooms", str(Path(configuration["data"]) / f"{split}.jsonl"),
            "--out", str(output), "--all-rows", "--resume-evaluation", "--chunk-size", "256",
            "--max-new-tokens", "8192", "--max-len", "40960", "--tp", "1",
            "--batch", "16", "--workers", "32", "--use-vllm"]


def _save_state(path: Path, state: dict[str, Any], **changes: Any) -> dict[str, Any]:
    updated = {**state, **changes, "updated_at": _now()}
    protocol.write_json_atomic(path, updated)
    return updated


def _run_split(configuration: dict[str, Any], frozen: dict[str, Any], model: dict[str, Any],
               split: str, target: Path, executor: Callable[..., Any]) -> dict[str, Any]:
    identity = _identity(frozen, model, split)
    command = _command(configuration, model, split, target)
    receipt = {"identity": identity, "command": command, "out_dir": str(target)}
    if protocol.evaluation_complete(target, identity):
        return {**receipt, "status": "complete", "verified_at": _now()}
    target.mkdir(parents=True, exist_ok=True)
    # The evaluator owns its output directory and requires it empty on first use.
    root, tag = target.parents[1], target.parent.name
    protocol.write_json_atomic(root / "state" / "identities" / tag / f"{split}.json", identity)
    log = root / "logs" / tag / f"{split}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": configuration["root"], "PYTHONUNBUFFERED": "1"}
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"started_at": _now(), "argv": command}) + "\n")
        stream.flush()
        executor(command, check=True, shell=False, cwd=configuration["root"], env=env,
                 stdout=stream, stderr=subprocess.STDOUT)
    if not protocol.evaluation_complete(target, identity):
        raise RuntimeError(f"Evaluator exited without verified full-cohort completion: {target}")
    return {**receipt, "status": "complete", "verified_at": _now(), "log": str(log)}


def _run_locked(output: Path, tag: str, executor: Callable[..., Any]) -> None:
    state_path = output / "state" / f"{tag}.json"
    state = _read(state_path) if state_path.exists() else {"tag": tag, "cells": {}}
    state = _save_state(state_path, state, status="running", error=None, started_at=_now(), validation_complete=False,
                        attempt=int(state.get("attempt", 0)) + 1)
    active = None
    try:
        configuration = _read(output / "runner.json")
        frozen = _validate_protocol(configuration, output)
        lineage = _lineage(configuration, tag)
        if state.get("training_lineage") not in (None, lineage):
            raise ValueError("Frozen training lineage changed; use a new evaluation directory")
        model = protocol.model_identity(Path(configuration["run"]) / MODELS[tag])
        if state.get("model_identity") not in (None, model):
            raise ValueError("Frozen model identity changed; use a new evaluation directory")
        model_stats = _model_stats(model)
        state = _save_state(state_path, state, model_identity=model, training_lineage=lineage, validation_complete=True)
        for split in protocol.SPLITS:
            active = split
            target = output / tag / split
            running = {"status": "running", "started_at": _now(), "out_dir": str(target),
                       "identity": _identity(frozen, model, split),
                       "command": _command(configuration, model, split, target)}
            state = _save_state(state_path, state, cells={**state["cells"], split: running})
            if _model_stats(model) != model_stats:
                raise ValueError("Model changed during evaluation")
            result = _run_split(configuration, frozen, model, split, target, executor)
            if _model_stats(model) != model_stats:
                raise ValueError("Model changed during evaluation")
            state = _save_state(state_path, state, cells={**state["cells"], split: result})
        state = _save_state(state_path, state, validation_complete=False)
        if _lineage(configuration, tag) != lineage:
            raise ValueError("Training lineage changed during evaluation")
        _save_state(state_path, state, status="complete", completed_at=_now(), validation_complete=True)
    except BaseException as exc:
        failure = {"error": f"{type(exc).__name__}: {exc}", "returncode": getattr(exc, "returncode", None)}
        cells = state["cells"]
        if active is not None:
            cells = {**cells, active: {**cells.get(active, {}), **failure, "status": "failed"}}
        _save_state(state_path, state, **failure, status="failed", cells=cells)
        raise


def _lineage(configuration: dict[str, Any], tag: str) -> dict[str, Any] | None:
    run = Path(configuration["run"])
    mode = configuration.get("lineage_mode", "clean" if run.name.startswith("fastfill_v1_1_clean_") else "legacy")
    if mode == "legacy":
        return None
    if mode != "clean":
        raise ValueError("Invalid frozen lineage mode")
    from fastfill.v1_1.training_guard import verify_training_lineage

    return verify_training_lineage(run, tag)


def run_model(output: str | Path, tag: str, *, executor: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    if tag not in MODELS:
        raise ValueError(f"Unknown model tag: {tag}")
    output = Path(output).resolve()
    try:
        with _lock(output / "state" / f".{tag}.lock"):
            _run_locked(output, tag, executor)
    finally:
        report(output)
    return report(output)


def _cell_report(output: Path, frozen: dict[str, Any], tag: str, split: str,
                 state: dict[str, Any]) -> dict[str, Any]:
    target = output / tag / split
    cell = state.get("cells", {}).get(split, {})
    result = {"status": cell.get("status", "pending"), "out_dir": str(target),
              "expected_rooms": frozen["datasets"][split]["line_count"],
              "metrics": None, "error": cell.get("error"), "returncode": cell.get("returncode")}
    if cell.get("status") == "complete":
        expected = _identity(frozen, state.get("model_identity", {}), split)
        valid = state.get("validation_complete", True) and cell.get("identity") == expected and protocol.evaluation_complete(target, expected)
        result = {**result, "status": "complete" if valid else "incomplete",
                  "metrics": _read(target / "metrics.json") if valid else None}
    for progress_name in ("progress.json", "evaluation_progress.json"):
        path = target / progress_name
        if path.exists():
            try:
                result = {**result, "progress": _read(path)}
            except (ValueError, OSError) as exc:
                result = {**result, "progress_error": f"{type(exc).__name__}: {exc}"}
    return result


def _markdown(summary: dict[str, Any]) -> str:
    rows = ["# FastFill V1.1 full-cohort evaluation", "",
            f"Verified evaluations: {summary['completed_evaluations']}/12. Overall complete: {summary['complete']}.",
            "", INTERPRETATION, "", "| Model | Split | Status | Full rooms |", "| --- | --- | --- | ---: |"]
    for name, cell in summary["evaluations"].items():
        tag, split = name.split("/")
        rows.append(f"| {tag} | {split} | {cell['status']} | {cell['expected_rooms']} |")
    rows += ["", "## Population and source metrics", "",
             "Rates below use the evaluator's stated denominators. Generation and parsing failures are retained; "
             "collision and object metrics conditional on successful parsing must be read together with complete_rate.", ""]
    for name, cell in summary["evaluations"].items():
        if cell["error"]:
            rows += [f"{name}: {cell['error']}", ""]
        if cell["metrics"] is None:
            continue
        rows += [f"### {name}", "", "| Population/source | Rooms | Parse | Complete | Raw valid | Repaired valid | Accepted | Zero collision accepted |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for population, metrics in cell["metrics"].items():
            if not isinstance(metrics, dict):
                continue
            keys = ("rooms", "parse_rate", "complete_rate", "valid_rate", "valid_rate_repaired",
                    "accepted_rate", "zero_collision_accepted_rate")
            values = [str(metrics.get(key, "—")) for key in keys]
            label = str(population).replace("|", "\\|").replace("\n", " ")
            rows.append("| " + " | ".join([label, *values]) + " |")
        rows += ["", "Full raw/repaired metrics and error counts are retained in report.json.", ""]
    return "\n".join(rows) + "\n"


def _write_text_atomic(path: Path, payload: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def report(output: str | Path) -> dict[str, Any]:
    output = Path(output).resolve()
    with _lock(output / ".report.lock"):
        frozen = _read(output / "protocol.json")
        states = {tag: _read(output / "state" / f"{tag}.json")
                  if (output / "state" / f"{tag}.json").exists() else {"status": "pending"}
                  for tag in MODELS}
        evaluations = {f"{tag}/{split}": _cell_report(output, frozen, tag, split, states[tag])
                       for tag in MODELS for split in protocol.SPLITS}
        completed = sum(cell["status"] == "complete" for cell in evaluations.values())
        summary = {"schema_version": 1, "updated_at": _now(), "complete": completed == 12,
                   "completed_evaluations": completed, "expected_evaluations": 12,
                   "models": {tag: {key: state.get(key) for key in ("status", "attempt", "error", "updated_at")}
                              for tag, state in states.items()},
                   "evaluations": evaluations, "interpretation": INTERPRETATION}
        protocol.write_json_atomic(output / "report.json", summary)
        _write_text_atomic(output / "report.md", _markdown(summary))
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for flag in ("root", "data", "output"):
        prepare_parser.add_argument(f"--{flag}", required=True)
    prepare_parser.add_argument("--run", "--run-dir", dest="run", required=True)
    prepare_parser.add_argument("--python", default=sys.executable)
    prepare_parser.add_argument("--lineage-mode", choices=("auto", "clean", "legacy"), default="auto")
    run_parser = commands.add_parser("run-model")
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--tag", choices=tuple(MODELS), required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root, args.data, args.run, args.output, python=args.python, lineage_mode=args.lineage_mode)
    elif args.command == "run-model":
        result = run_model(args.output, args.tag)
    else:
        result = report(args.output)
    print(json.dumps({"complete": result["complete"], "completed_evaluations": result["completed_evaluations"],
                      "report": str(Path(args.output).resolve() / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
