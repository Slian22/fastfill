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
    summary = llm_baseline.run(data, tmp_path / "llm", tmp_path / "api.env", ask=lambda env, model, user: by_prompt[user])
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
