"""Verify persisted outputs and training provenance before research synthesis.

This gate can inspect an older frozen experiment without changing its evaluator.
The interpretation addendum is bound to the exact four cohort hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from fastfill.v1_1.evaluation_protocol import file_identity, write_json_atomic


TAGS = ("sft", "dpo1", "final")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _verify_outputs(output: Path, tag: str, workers: int) -> dict[str, Any]:
    runner = _read(output / "runner.json")
    command = [sys.executable, str(Path(__file__).with_name("verify_outputs.py")), "--output", str(output),
               "--tag", tag, "--scoring-root", runner["root"], "--workers", str(workers),
               "--reuse-verified"]
    subprocess.run(command, check=True)
    return _read(output / "verification" / f"{tag}.json")


def _verify_lineage(output: Path, tag: str) -> dict[str, Any]:
    from fastfill.v1_1.training_guard import verify_training_lineage

    runner = _read(output / "runner.json")
    return verify_training_lineage(runner["run"], tag)


def _synthesize_frozen(output: Path) -> dict[str, Any]:
    runner = _read(output / "runner.json")
    root = Path(runner["root"]).resolve()
    code = (
        "import json,sys,pathlib; root=pathlib.Path(sys.argv[1]).resolve(); "
        "sys.path.insert(0,str(root)); "
        "from fastfill.v1_1 import research_report as report; "
        "assert pathlib.Path(report.__file__).resolve().is_relative_to(root); "
        "summary=report.build_report(sys.argv[2]); "
        "print(json.dumps({'summary':summary,'markdown':report._markdown(summary)},allow_nan=False))"
    )
    result = subprocess.run([runner["python"], "-I", "-c", code, str(root), str(output)],
                            check=True, capture_output=True, text=True, cwd=root)
    return json.loads(result.stdout)


def _interpretation(output: Path, path: Path) -> dict[str, Any]:
    value = _read(path)
    frozen = _read(output / "protocol.json")
    expected = {split: entry["sha256"] for split, entry in frozen["datasets"].items()}
    if value.get("schema_version") != 1 or value.get("dataset_sha256") != expected:
        raise ValueError("Interpretation addendum does not match the frozen cohort")
    limits = value.get("limitations")
    if value.get("research_status") != "diagnostic_only" or not isinstance(limits, list) or not limits or any(
        not isinstance(item, str) or not item.strip() for item in limits
    ):
        raise ValueError("Invalid research interpretation addendum")
    return value


def _write_text(path: Path, text: str) -> None:
    import tempfile

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def generate_audited_report(output, interpretation, workers=32, *, verifier=_verify_outputs,
                            lineage_verifier=_verify_lineage, synthesizer=_synthesize_frozen):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    output, interpretation = Path(output).resolve(), Path(interpretation).resolve()
    addendum_identity = file_identity(interpretation)
    addendum = _interpretation(output, interpretation)
    errors, checks = [], {}
    for tag in TAGS:
        try:
            check = verifier(output, tag, workers)
            if not isinstance(check, dict) or check.get("status") != "complete":
                raise ValueError("Independent scoring verification is not complete")
            checks[f"outputs/{tag}"] = check
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(f"{tag} output verification: {type(exc).__name__}: {exc}")
    for tag in ("dpo1", "final"):
        try:
            checks[f"lineage/{tag}"] = lineage_verifier(output, tag)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            errors.append(f"{tag} training lineage: {type(exc).__name__}: {exc}")
    synthesis = {}
    if not errors:
        try:
            synthesis = synthesizer(output)
            if (not isinstance(synthesis, dict) or not isinstance(synthesis.get("summary"), dict)
                    or not isinstance(synthesis.get("markdown"), str)):
                raise ValueError("Malformed frozen synthesis")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(f"Frozen synthesis: {type(exc).__name__}: {exc}")
    if file_identity(interpretation) != addendum_identity:
        errors.append("Interpretation addendum changed during verification")
    summary = synthesis.get("summary", {}) if not errors else {}
    complete = summary.get("complete") is True and not errors
    result = {**summary, "schema_version": 1, "status": "complete" if complete else "incomplete",
              "complete": complete, "research_status": addendum["research_status"],
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "errors": [*summary.get("errors", []), *errors],
              "dev_comparisons": summary.get("dev_comparisons", []) if complete else [],
              "interpretation": addendum,
              "audit": {"status": "passed" if complete else "failed", "checks": checks,
                        "addendum": addendum_identity, "gate_code": file_identity(__file__)}}
    prefix = ("# Independently verified research report\n\n"
              f"Coverage/integrity status: **{result['status']}**. Research scope: **diagnostic only**.\n\n"
              + "\n\n".join(addendum["limitations"]) + "\n\n")
    body = synthesis.get("markdown", "") if complete else "No inferential result is published.\n\n" + "\n".join(
        f"- {error}" for error in result["errors"])
    write_json_atomic(output / "research-summary.json", result)
    _write_text(output / "research-summary.md", prefix + body.rstrip() + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interpretation", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        result = generate_audited_report(args.output, args.interpretation, args.workers)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"complete": False, "status": "incomplete", "dev_comparisons": [], "errors": [str(exc)]}
        write_json_atomic(Path(args.output) / "research-summary.json", result)
        _write_text(Path(args.output) / "research-summary.md", "# Research report incomplete\n\n" + str(exc) + "\n")
    print(json.dumps({key: result[key] for key in ("complete", "status")}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
