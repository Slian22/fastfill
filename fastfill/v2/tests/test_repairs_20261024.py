"""Current audit regressions: temporary inputs and fake transports only."""
import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from fastfill.v2 import autorun, llm_baseline, roomgenbench
from fastfill.v2.evaluate import project_minimal, run_evaluation
from fastfill.v2.io import fingerprint
from fastfill.v2.ops.stratify import collect
from fastfill.v2.tests.test_evaluate_fix20261006 import real_rows
from fastfill.v2.tests.test_llm_baseline import _answer_from_targets


@pytest.mark.parametrize("key", ["../reference/protected", "nested/name", "..\\protected", "", ".", "..", 7, None, "absolute"])
def test_benchmark_request_keys_cannot_escape_or_create_partial_outputs(tmp_path, key):
    reference = tmp_path / "reference"
    source = reference / "scenes"
    source.mkdir(parents=True)
    victim = reference / "protected.json"
    victim.write_text('{"original": true}\n')
    if key == "absolute":
        key = str(victim.with_suffix(""))
    scene = {"scene_key": "safe_scene", "room_type": "study",
             "room": {"dimensions": {"width": 5., "length": 4., "height": 3.}},
             "objects": [{"id": "desk", "type": "desk", "description": "desk", "place_id": "floor"}]}
    (source / "a.json").write_text(json.dumps(scene))
    (source / "b.json").write_text(json.dumps({**scene, "scene_key": key}))
    before = {p: p.read_bytes() for p in reference.rglob("*.json")}
    out = tmp_path / "requests"
    with pytest.raises(ValueError, match="scene_key"):
        roomgenbench.write_benchmark_requests(source, out, roomgenbench_root=reference)
    assert not out.exists()
    assert {p: p.read_bytes() for p in reference.rglob("*.json")} == before


@pytest.mark.parametrize("changed", ["fit", "report"])
def test_stratify_cache_invalidates_every_consumed_input(tmp_path, changed):
    rows = real_rows()
    data, fit, predictions = (tmp_path / name for name in ("rows.jsonl", "fit.jsonl", "predictions.jsonl"))
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fit.write_bytes(data.read_bytes())
    predictions.write_text("".join(json.dumps(r["target"]) + "\n" for r in rows))
    out, cache = tmp_path / "evaluation", tmp_path / "cache"
    run_evaluation(data, out, predictions=predictions, baseline_fit=fit)
    assert collect(out, workers=1, cache=cache)[2]["reproduces_report_exactly"]
    if changed == "fit":
        modified = deepcopy(rows)
        for row in modified:
            for obj in row["target"]["objects"]:
                obj["target_size_local_m"] = [None if x is None else x * 2 for x in obj["target_size_local_m"]]
        fit.write_text("".join(json.dumps(r) + "\n" for r in modified))
    else:
        path = out / "report.json"
        report = json.loads(path.read_text())
        report["model"]["reference"]["log_size_error"]["mean"] = 123.
        path.write_text(json.dumps(report))
    cached, fresh = collect(out, workers=1, cache=cache), collect(out, workers=1)
    assert not fresh[2]["reproduces_report_exactly"]
    assert cached == fresh


@pytest.mark.parametrize("failure", ["transport", "malformed"])
def test_versioned_harness_retains_valid_layout_on_failed_repair(tmp_path, failure):
    module = importlib.import_module("fastfill.v2.llm_baseline_v2")
    row = next(p for p in map(project_minimal, real_rows()) if p is not None)
    answer = json.loads(_answer_from_targets(row).removeprefix("```json\n").removesuffix("\n```"))
    answer["objects"][0]["x"] = -100.
    first = json.dumps(answer)
    data, env = tmp_path / "data.jsonl", tmp_path / "dummy.env"
    data.write_text(json.dumps(row) + "\n")
    env.write_text("OPENAI_MODEL=unused\n")
    seen = []
    def ask(*args):
        seen.append(len(args[-1]))
        if len(seen) == 1:
            return first
        if failure == "transport":
            raise ConnectionError("synthetic repair failure")
        return "malformed repair"
    old_hash = fingerprint(llm_baseline.__file__)
    summary = module.run(data, tmp_path / "out", env, mode="harness", workers=1, ask=ask)
    prediction = json.loads((tmp_path / "out/predictions.jsonl").read_text())
    assert seen == [1, 3]
    assert summary["failed"] == 0 and summary["harness_version"] == "repair-v2"
    assert summary["implementation_sha256"] == fingerprint(module.__file__)
    assert summary["legacy_implementation_sha256"] == old_hash == fingerprint(llm_baseline.__file__)
    assert prediction["layout"] == llm_baseline.to_layout(first, row["condition"])
    assert prediction["first_answer_valid"] is True and prediction["attempts"] == 1
    assert prediction["repair_error"] and prediction["problems_final"] > 0


@pytest.mark.parametrize("setting,value", [("repairs", -1), ("repairs", True), ("repairs", 1.5),
                                          ("retries", -1), ("retries", False), ("retries", "1"),
                                          ("workers", 0), ("workers", -1), ("workers", True), ("workers", None),
                                          ("max_samples", 0), ("max_samples", -1),
                                          ("max_samples", True), ("max_samples", 1.5)])
def test_versioned_harness_rejects_invalid_counts_before_any_io(tmp_path, setting, value):
    module = importlib.import_module("fastfill.v2.llm_baseline_v2")
    output = tmp_path / "out"
    calls = []
    with pytest.raises(ValueError, match=setting):
        module.run(tmp_path / "missing-data", output, tmp_path / "missing-env",
                   ask=lambda *args: calls.append(args), **{setting: value})
    assert not output.exists() and calls == []


@pytest.mark.parametrize("case", ["prompt", "zero_counts", "retry_initial", "unanswered", "repair", "later_repair_failure"])
def test_versioned_harness_normal_and_initial_failure_contracts(tmp_path, case):
    module = importlib.import_module("fastfill.v2.llm_baseline_v2")
    row = next(p for p in map(project_minimal, real_rows()) if p is not None)
    good = _answer_from_targets(row)
    bad = json.loads(good.removeprefix("```json\n").removesuffix("\n```"))
    bad["objects"][0]["x"] = -100.
    improved = deepcopy(bad)
    improved["objects"][0]["x"] = -50.
    data, env = tmp_path / "rows.jsonl", tmp_path / "api.env"
    data.write_text(json.dumps(row) + "\n")
    env.write_text("OPENAI_MODEL=unused\n")
    seen = []
    def ask(*args):
        seen.append(len(args[-1]))
        if case == "unanswered" or (case == "later_repair_failure" and len(seen) == 3):
            raise ConnectionError("synthetic failure")
        if case == "retry_initial" and len(seen) == 1:
            return "invalid first answer"
        if case in ("zero_counts", "repair", "later_repair_failure") and len(seen) == 1:
            return json.dumps(bad)
        return json.dumps(improved) if case == "later_repair_failure" else good
    summary = module.run(data, tmp_path / "out", env, workers=1, ask=ask,
                         mode="harness" if case in ("zero_counts", "repair", "later_repair_failure") else "prompt",
                         **({"repairs": 0, "retries": 0} if case == "zero_counts" else {}))
    prediction = json.loads((tmp_path / "out/predictions.jsonl").read_text())
    assert seen == {"prompt": [1], "zero_counts": [1], "retry_initial": [1, 1], "unanswered": [1, 1],
                    "repair": [1, 3], "later_repair_failure": [1, 3, 5]}[case]
    assert summary["api_calls"] == len(seen)
    assert summary["failed"] == summary["unanswered"] == int(case == "unanswered")
    assert summary["first_answer_invalid"] == int(case == "retry_initial")
    if case != "unanswered":
        expected = json.dumps(improved) if case == "later_repair_failure" else json.dumps(bad) if case == "zero_counts" else good
        assert prediction["layout"] == llm_baseline.to_layout(expected, row["condition"])
        assert prediction["repair_rounds"] == int(case in ("repair", "later_repair_failure"))
        assert bool(prediction["repair_error"]) is (case == "later_repair_failure")
    else:
        assert prediction["layout"] is None and prediction["first_answer_valid"] is None


def test_legacy_autorun_never_deletes_or_rebuys_unbound_baselines(tmp_path):
    env, data = tmp_path / "dummy.env", tmp_path / "rows.jsonl"
    env.write_text("OPENAI_MODEL=unused\n")
    data.write_text("{}\n")
    (tmp_path / "fastfill/v2").mkdir(parents=True)
    shutil.copy(llm_baseline.__file__, tmp_path / "fastfill/v2/llm_baseline.py")
    pilot = autorun.Autopilot(argparse.Namespace(repo=str(tmp_path), llm_env=str(env), llm_rows="300"))
    for mode in ("prompt", "harness"):
        path = pilot.runs / f"llm-{mode}-300"
        path.mkdir()
        (path / "summary.json").write_text('{"historical": true}\n')
        (path / "predictions.jsonl").write_text('{"layout": null}\n')
    before = {p: p.read_bytes() for p in pilot.runs.rglob("*") if p.is_file()}
    calls = []
    pilot.sh = lambda *a, **k: calls.append(a)
    pilot.llm_baselines(data)
    assert calls == []
    for path, content in before.items():
        assert path.read_bytes() == content
    assert pilot.llm_reports == {}


def test_versioned_autorun_keeps_history_and_requires_new_path_for_input_drift(tmp_path):
    module = importlib.import_module("fastfill.v2.llm_baseline_v2")
    (tmp_path / "fastfill/v2").mkdir(parents=True)
    for code in Path(llm_baseline.__file__).parent.glob("*.py"):
        shutil.copy(code, tmp_path / "fastfill/v2" / code.name)
    row = next(p for p in map(project_minimal, real_rows()) if p is not None)
    data, env = tmp_path / "data.jsonl", tmp_path / "dummy.env"
    data.write_text(json.dumps(row) + "\n")
    env.write_text("OPENAI_MODEL=unused\n")
    pilot = autorun.Autopilot(argparse.Namespace(repo=str(tmp_path), llm_env=str(env), llm_rows="300",
                                               llm_baseline="repair-v2"))
    historical = pilot.runs / "llm-harness-300"
    historical.mkdir()
    (historical / "predictions.jsonl").write_text("historical paid result\n")
    calls = []
    def sh(args, **kwargs):
        args = list(map(str, args)); calls.append(args[2])
        get = lambda flag: args[args.index(flag) + 1]
        if args[2] == "fastfill.v2.llm_baseline_v2":
            module.run(get("--data"), get("--output"), get("--env"), mode=get("--mode"), workers=1,
                       max_samples=int(get("--max-samples")),
                       ask=lambda *args: _answer_from_targets(row))
        elif args[2] == "fastfill.v2.evaluate":
            run_evaluation(get("--data"), get("--output"), predictions=get("--predictions"))
        else:
            pytest.fail(f"unexpected executable: {args[2]}")
    pilot.sh = sh
    pilot.llm_baselines(data)
    assert calls == ["fastfill.v2.llm_baseline_v2", "fastfill.v2.evaluate"] * 2
    assert (pilot.runs / "llm-repair-v2-harness-300/predictions.jsonl").is_file()
    calls.clear()
    pilot.llm_baselines(data)
    assert calls == []
    # An evaluator change may create a new report, but never rewrites a previous report or pays again.
    old_report = Path(pilot.llm_reports["LLM harness"])
    old_report.write_text(json.dumps({**json.loads(old_report.read_text()), "implementation_sha256": "old"}))
    content = old_report.read_bytes()
    pilot.llm_baselines(data)
    assert calls == ["fastfill.v2.evaluate"] and old_report.read_bytes() == content
    assert Path(pilot.llm_reports["LLM harness"]) != old_report
    before = {p: p.read_bytes() for p in pilot.runs.rglob("*") if p.is_file() and pilot.dir not in p.parents}
    calls.clear()
    env.write_text("OPENAI_MODEL=different\n")
    pilot.llm_baselines(data)
    assert calls == [] and pilot.llm_reports == {}
    assert all(path.read_bytes() == value for path, value in before.items())
    pilot.a.llm_run_tag = "model-b"
    pilot.llm_baselines(data)
    assert calls == ["fastfill.v2.llm_baseline_v2", "fastfill.v2.evaluate"] * 2
    assert (pilot.runs / "llm-repair-v2-harness-300-model-b/summary.json").is_file()
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("failure", ["generate", "rows", "evaluate", "one_generate", "one_evaluate", "none"])
def test_structured_wrapper_propagates_failure_and_stops_each_chain(tmp_path, failure):
    (tmp_path / "env/bin").mkdir(parents=True)
    rows = tmp_path / "runs/llm-prompt-300/rows.jsonl"
    rows.parent.mkdir(parents=True)
    rows.write_text("{}\n")
    stub = tmp_path / "env/bin/python"
    stub.write_text(f"#!{sys.executable}\n" + '''import os, sys
from pathlib import Path
args=sys.argv[1:]
module=args[args.index('-m')+1]
with open('calls.txt','a') as stream: stream.write(module+'\\n')
failure=os.environ['TEST_FAILURE']
out=Path(args[args.index('--output')+1])
if module.endswith('llm_structured'):
    if failure == 'generate' or failure == 'one_generate' and out.name == 'llm-structured-300': raise SystemExit(42)
    out.mkdir(parents=True)
    (out/'rows.jsonl').write_text('DIFFERENT' if failure == 'rows' else '{}\\n')
elif failure == 'evaluate' or failure == 'one_evaluate' and out.name == 'llm-structured-300-eval': raise SystemExit(42)
''')
    stub.chmod(0o755)
    source = Path(__file__).parents[1] / "ops/run_structured.sh"
    result = subprocess.run(["bash", "-c", 'cd() { builtin cd "$OPS_TEST_WORK"; }; source "$OPS_SRC"'],
                            env={**os.environ, "OPS_TEST_WORK": str(tmp_path), "OPS_SRC": str(source), "TEST_FAILURE": failure},
                            text=True, capture_output=True)
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert (result.returncode == 0) is (failure == "none")
    assert ("ALL-DONE" in result.stdout) is (failure == "none")
    assert calls.count("fastfill.v2.evaluate") == {"generate": 0, "rows": 0, "one_generate": 1}.get(failure, 2)
