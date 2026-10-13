"""Behavioral tests for the independent full-cohort evaluation controller."""
import json
import argparse
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from fastfill.v1_1 import automate
from fastfill.v1_1 import evaluation_protocol as protocol


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data, run, out = (tmp_path / part for part in ("data", "run", "evaluation"))
    data.mkdir()
    for split in protocol.SPLITS:
        (data / f"{split}.jsonl").write_text('{"uid":"same"}\n{"uid":"same"}\n')
    for name in ("sft_merged", "dpo1_merged", "final_merged"):
        model = run / name
        model.mkdir(parents=True)
        for config in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (model / config).write_text('{}')
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        (model / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"0000")
    settings = {"all_rows": True, "gt_only": False, "max_len": 40960,
                "max_new_tokens": 8192, "batch": 16, "tp": 1,
                "use_vllm": True, "chunk_size": 256}
    monkeypatch.setattr(automate, "_evaluation_contract", lambda: (settings, {"eval.py": "frozen"}))
    root = Path(__file__).resolve().parents[3]
    automate.prepare(root, data, run, out, python=sys.executable)
    return out, data, run


def fake_evaluator(argv, **kwargs):
    target = Path(argv[argv.index("--out") + 1])
    identity_path = target.parents[1] / "state" / "identities" / target.parent.name / f"{target.name}.json"
    identity = json.loads(identity_path.read_text())
    count = identity["dataset"]["line_count"]
    (target / "predictions.jsonl").write_text(
        ''.join(json.dumps({"uid": "same", "output": None, "error": "prompt_too_long"}) + '\n'
                for _ in range(count))
    )
    (target / "metrics.json").write_text(json.dumps({
        "in_dist": {"rooms": 1, "accepted_rate": 0, "errors": {"prompt_too_long": 1}},
        "held_out": {"rooms": count - 1, "accepted_rate": 0},
        "SceneSmith": {"rooms": count - 1, "accepted_rate": 0},
    }))
    protocol.finalize_evaluation(target, identity)
    return subprocess.CompletedProcess(argv, 0)


def test_prepare_freezes_original_cohorts_and_rejects_drift(prepared):
    out, data, _ = prepared
    frozen = json.loads((out / "protocol.json").read_text())
    assert set(frozen["datasets"]) == set(protocol.SPLITS)
    assert all(item["line_count"] == 2 for item in frozen["datasets"].values())
    (data / "test_rooms.jsonl").write_text('{"uid":"new"}\n')
    with pytest.raises(ValueError, match="protocol|dataset|cohort"):
        automate.run_model(out, "sft", executor=fake_evaluator)
    assert automate.report(out)["complete"] is False


def test_prepare_rejects_output_inside_model_and_reconfigured_run(prepared):
    out, data, run = prepared
    root = Path(__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="isolated|model|output"):
        automate.prepare(root, data, run, run / "sft_merged", python=sys.executable)
    with pytest.raises(ValueError, match="config|runner"):
        automate.prepare(root, data, run / "different", out, python=sys.executable)


def test_full_commands_are_ordered_and_explicit_with_no_limit(prepared):
    out, _, _ = prepared
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        assert kwargs["check"] is True and kwargs["shell"] is False
        return fake_evaluator(argv, **kwargs)
    automate.run_model(out, "sft", executor=execute)
    assert [Path(cmd[cmd.index("--rooms") + 1]).stem for cmd in calls] == list(protocol.SPLITS)
    for cmd in calls:
        assert "--all-rows" in cmd and "--resume-evaluation" in cmd and "--limit" not in cmd
        assert cmd[cmd.index("--max-new-tokens") + 1] == "8192"
        assert cmd[cmd.index("--max-len") + 1] == "40960"
        assert cmd[cmd.index("--chunk-size") + 1] == "256"
    summary = automate.report(out)
    assert summary["complete"] is False and summary["completed_evaluations"] == 4


def test_resume_skips_only_verified_cells_and_preserves_other_models(prepared):
    out, _, _ = prepared
    automate.run_model(out, "sft", executor=fake_evaluator)
    automate.run_model(out, "dpo1", executor=fake_evaluator)
    def should_not_run(*args, **kwargs):
        raise AssertionError("verified evaluations must not run again")
    automate.run_model(out, "sft", executor=should_not_run)
    assert automate.report(out)["completed_evaluations"] == 8


def test_legacy_metrics_without_receipt_cannot_mark_complete(prepared):
    out, _, _ = prepared
    def legacy(argv, **kwargs):
        target = Path(argv[argv.index("--out") + 1])
        (target / "metrics.json").write_text('{"in_dist":{"rooms":200},"held_out":null}')
        return subprocess.CompletedProcess(argv, 0)
    with pytest.raises(RuntimeError, match="completion|verified"):
        automate.run_model(out, "sft", executor=legacy)
    result = automate.report(out)
    assert result["completed_evaluations"] == 0
    assert result["evaluations"]["sft/dev_rooms"]["status"] == "failed"


def test_subprocess_failure_stops_split_chain_and_records_report(prepared):
    out, _, _ = prepared
    calls = []
    def fail(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(17, argv)
    with pytest.raises(subprocess.CalledProcessError):
        automate.run_model(out, "sft", executor=fail)
    assert len(calls) == 1
    result = json.loads((out / "report.json").read_text())
    assert result["complete"] is False
    assert result["models"]["sft"]["status"] == "failed"
    assert result["evaluations"]["sft/dev_rooms"]["returncode"] == 17
    assert result["evaluations"]["sft/test_rooms"]["status"] == "pending"


def test_all_twelve_results_include_failures_sources_and_heldout(prepared):
    out, _, _ = prepared
    for tag in automate.MODELS:
        automate.run_model(out, tag, executor=fake_evaluator)
    result = automate.report(out)
    assert result["complete"] is True and result["completed_evaluations"] == 12
    assert len(result["evaluations"]) == 12
    test = result["evaluations"]["final/test_rooms"]
    assert test["expected_rooms"] == 2
    assert test["metrics"]["held_out"]["rooms"] == 1
    assert test["metrics"]["SceneSmith"]["rooms"] == 1
    assert test["metrics"]["in_dist"]["errors"] == {"prompt_too_long": 1}
    assert "winner" not in result
    assert "does not" in result["interpretation"]
    assert "SceneSmith" in (out / "report.md").read_text()


def test_tampered_artifacts_invalidate_report_without_model_rehash(prepared, monkeypatch):
    out, _, _ = prepared
    automate.run_model(out, "sft", executor=fake_evaluator)
    def no_hash(*args):
        raise AssertionError("report must use captured model receipt")
    monkeypatch.setattr(protocol, "model_identity", no_hash)
    (out / "sft" / "test_rooms" / "predictions.jsonl").write_text('tampered')
    result = automate.report(out)
    assert result["completed_evaluations"] == 3
    assert result["evaluations"]["sft/test_rooms"]["status"] == "incomplete"


def test_missing_final_model_is_failure_not_success(prepared):
    out, _, run = prepared
    (run / "final_merged" / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="weight"):
        automate.run_model(out, "final", executor=fake_evaluator)
    assert automate.report(out)["models"]["final"]["status"] == "failed"


def test_model_mutation_between_split_evaluations_stops_run(prepared):
    out, _, run = prepared
    def mutate(argv, **kwargs):
        result = fake_evaluator(argv, **kwargs)
        (run / "sft_merged" / "config.json").write_text('{"changed":true}')
        return result
    with pytest.raises(ValueError, match="model|Model"):
        automate.run_model(out, "sft", executor=mutate)
    assert automate.report(out)["models"]["sft"]["status"] == "failed"


def test_model_identity_change_on_resume_fails_closed(prepared):
    out, _, run = prepared
    automate.run_model(out, "sft", executor=fake_evaluator)
    (run / "sft_merged" / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity|model"):
        automate.run_model(out, "sft", executor=fake_evaluator)


def test_unknown_model_tag_is_rejected(prepared):
    with pytest.raises(ValueError, match="tag"):
        automate.run_model(prepared[0], "../../outside", executor=fake_evaluator)


def test_actual_evaluator_contract_matches_command_values():
    settings, hashes = automate._evaluation_contract()
    assert settings == {"all_rows": True, "gt_only": False, "max_len": 40960,
                        "max_new_tokens": 8192, "batch": 16, "tp": 1,
                        "use_vllm": True, "chunk_size": 256}
    assert any(name.endswith("evaluate_v1_1.py") for name in hashes)
    assert any(name.endswith("automate.py") for name in hashes)


def test_runner_works_with_real_resumable_evaluator_initial_directory(prepared, monkeypatch):
    from fastfill.v1_1 import evaluate_v1_1 as evaluator
    out, _, _ = prepared
    monkeypatch.setattr(evaluator, "evaluation_code_hashes", lambda: {"eval.py": "frozen"})
    monkeypatch.setattr(evaluator, "_load_tokenizer", lambda model: object())
    monkeypatch.setattr(evaluator, "messages_v1_1", lambda room, *args, **kwargs: room["uid"])
    monkeypatch.setattr(evaluator, "prompt_text", lambda tokenizer, messages: messages)
    monkeypatch.setattr(evaluator, "_score_worker", lambda task: {"uid": task[0]["uid"], "accepted": False})
    monkeypatch.setattr(evaluator, "summarize_v1_1", lambda rows: {
        "in_dist": {"rooms": len(rows), "accepted_rate": 0}, "held_out": None})
    class Generator:
        def __init__(self, args):
            pass
        def generate(self, prompts):
            return [None for _ in prompts]
    monkeypatch.setattr(evaluator, "_GenerationSession", Generator)
    def execute(argv, **kwargs):
        value = lambda flag: argv[argv.index(flag) + 1]
        evaluator.run_resumable_evaluation(argparse.Namespace(
            rooms=value("--rooms"), model=value("--model"), out=value("--out"),
            all_rows=True, resume_evaluation=True, gt_only=False, use_vllm=True,
            max_len=int(value("--max-len")), max_new_tokens=int(value("--max-new-tokens")),
            chunk_size=int(value("--chunk-size")), tp=1, batch=16, workers=1, limit=None))
        return subprocess.CompletedProcess(argv, 0)
    automate.run_model(out, "sft", executor=execute)
    assert automate.report(out)["completed_evaluations"] == 4


def test_cli_preparation_and_report_use_frozen_paths(prepared, monkeypatch, capsys):
    out, data, run = prepared
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(sys, "argv", ["automate", "prepare", "--root", str(root),
                                    "--data", str(data), "--run-dir", str(run),
                                    "--output", str(out)])
    automate.main()
    assert json.loads(capsys.readouterr().out)["completed_evaluations"] == 0
    monkeypatch.setattr(sys, "argv", ["automate", "report", "--output", str(out)])
    automate.main()
    assert json.loads(capsys.readouterr().out)["complete"] is False


def test_corrupt_progress_is_reported_without_marking_complete(prepared):
    out, _, _ = prepared
    target = out / "sft" / "dev_rooms"
    target.mkdir(parents=True)
    (target / "progress.json").write_text('broken')
    cell = automate.report(out)["evaluations"]["sft/dev_rooms"]
    assert "progress_error" in cell and cell["status"] == "pending"


def test_sbatch_is_bounded_and_has_separate_gpu_resources():
    script = (Path(__file__).parents[1] / "isambard_v1_1_evaluate.sbatch").read_text()
    assert "#SBATCH --gpus=1" in script
    assert "#SBATCH --time=06:00:00" in script
    assert "#SBATCH --cpus-per-task=32" in script
    assert "#SBATCH --mem-per-gpu=115000M" in script
    assert "sbatch " not in script
    assert "run-model" in script
