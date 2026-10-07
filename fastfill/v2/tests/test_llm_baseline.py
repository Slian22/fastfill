"""LLM baseline: three-field prompt carries no target geometry; answers convert to layouts exactly."""
import json
import math

from fastfill.v2 import llm_baseline
from fastfill.v2.evaluate import project_minimal, run_evaluation
from fastfill.v2.tests.test_evaluate_fix20261006 import real_rows


def _answer_from_targets(row):
    objects = []
    for t in row["target"]["objects"]:
        depth, width, height = t["target_size_local_m"]
        x, y, z = t["bottom_center_m"]
        objects.append({"id": t["id"], "width_m": width, "depth_m": depth, "height_m": height,
                        "x": x, "y": y, "z": z, "facing_deg": math.degrees(t["yaw_rad"])})
    return "```json\n" + json.dumps({"objects": objects}) + "\n```"


def test_prompt_is_three_field_and_answers_round_trip_through_evaluate(tmp_path):
    rows = [p for p in map(project_minimal, real_rows()) if p is not None][:2]
    assert rows
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for row in rows:
        text = llm_baseline.prompt(row["condition"])
        assert all(f"{t['bottom_center_m'][0]:.6f}" not in text for t in row["target"]["objects"])
    (tmp_path / "api.env").write_text("export OPENAI_BASE_URL=http://unused\nOPENAI_API_KEY='k'\nOPENAI_MODEL=m\n")
    by_prompt = {llm_baseline.prompt(r["condition"]): _answer_from_targets(r) for r in rows}
    summary = llm_baseline.run(data, tmp_path / "llm", tmp_path / "api.env", ask=lambda env, model, messages: by_prompt[messages[0]["content"]])
    assert summary["failed"] == 0 and summary["model"] == "m"
    report = run_evaluation(tmp_path / "llm/rows.jsonl", tmp_path / "eval", predictions=tmp_path / "llm/predictions.jsonl")
    reference = report["model"]["reference"]
    assert reference["bottom_center_error_m"]["mean"] < 1e-6 and reference["log_size_error"]["mean"] < 1e-6
    assert report["collapse"]["predicted"] == report["collapse"]["ground_truth"]


def test_unparseable_answers_are_recorded_failures_not_dropped(tmp_path):
    rows = [p for p in map(project_minimal, real_rows()) if p is not None][:1]
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps(rows[0]) + "\n")
    (tmp_path / "api.env").write_text("OPENAI_BASE_URL=x\nOPENAI_API_KEY=k\n")
    summary = llm_baseline.run(data, tmp_path / "llm", tmp_path / "api.env", model="m", ask=lambda *a: "no json here")
    assert summary == {**summary, "rows": 1, "failed": 1}
    assert json.loads((tmp_path / "llm/predictions.jsonl").read_text())["layout"] is None


def test_harness_reports_concrete_problems_and_keeps_the_repaired_layout(tmp_path):
    row = [p for p in map(project_minimal, real_rows()) if p is not None][0]
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps(row) + "\n")
    (tmp_path / "api.env").write_text("OPENAI_BASE_URL=x\nOPENAI_API_KEY=k\nOPENAI_MODEL=m\n")
    good = _answer_from_targets(row)
    bad = json.loads(good.strip("`\njson"))
    bad["objects"][1].update(x=bad["objects"][0]["x"], y=bad["objects"][0]["y"], z=bad["objects"][0]["z"])  # stack two objects
    seen = []
    def ask(env, model, messages):
        seen.append(messages)
        return json.dumps(bad) if len(seen) == 1 else good
    summary = llm_baseline.run(data, tmp_path / "h", tmp_path / "api.env", mode="harness", ask=ask)
    assert summary["api_calls"] == 2 and summary["mean_problems_first"] >= 1
    assert "overlap" in seen[1][-1]["content"] and seen[1][-2]["role"] == "assistant"
    prediction = json.loads((tmp_path / "h/predictions.jsonl").read_text())
    assert prediction["repair_rounds"] == 1 and prediction["problems_final"] <= summary["mean_problems_first"]
    once = llm_baseline.run(data, tmp_path / "p", tmp_path / "api.env", mode="prompt", ask=lambda *a: json.dumps(bad))
    assert once["api_calls"] == 1


def test_rate_limiter_spaces_requests():
    limiter = llm_baseline.RateLimiter(rpm=600)  # 0.1 s apart
    import time
    start = time.monotonic()
    for _ in range(4):
        limiter.wait()
    assert time.monotonic() - start >= .29


def test_reasoning_effort_defaults_to_medium_and_is_recorded(tmp_path):
    assert llm_baseline.request_parameters({}, "gpt-6.1-sol") == {"model": "gpt-6.1-sol", "reasoning_effort": "medium"}
    assert llm_baseline.request_parameters({"FASTFILL_LLM_REASONING_EFFORT": ""}, "m") == {"model": "m"}
    row = [p for p in map(project_minimal, real_rows()) if p is not None][0]
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "api.env").write_text("OPENAI_BASE_URL=https://relay/v1\nOPENAI_API_KEY=secret-test-key\nOPENAI_MODEL=gpt-6.1-sol\n")
    summary = llm_baseline.run(tmp_path / "rows.jsonl", tmp_path / "o", tmp_path / "api.env", ask=lambda *a: _answer_from_targets(row))
    assert summary["request_parameters"]["reasoning_effort"] == "medium" and summary["endpoint"] == "https://relay/v1/chat/completions"
    assert "secret-test-key" not in json.dumps(summary) + (tmp_path / "o/summary.json").read_text()
