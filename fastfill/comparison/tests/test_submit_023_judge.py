"""The paid full-run gate is exercised with local fake Slurm and stored API records only."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fastfill.comparison import server_api_judge as judge
from fastfill.comparison.tests.test_judge_batch import bundle, write


@pytest.fixture
def submission(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    project = tmp_path / "project"
    run = project / "runs/unknownsize-20261013"
    root = bundle(run / "gpt-863099194937c5259a5aa8ac269f7a86f5b40cf752eabb9b789743d655ebcbb3")
    base = run / "judge-023-astra-high"
    smoke = base / "smoke"
    smoke.mkdir(parents=True)
    (project / "env/bin").mkdir(parents=True)
    (project / "env/bin/python").symlink_to(sys.executable)
    env_file = tmp_path / "api.env"
    env_file.write_text("OPENAI_API_KEY=local-test-only\nOPENAI_BASE_URL=https://example.invalid\nOPENAI_MODEL=gpt-6-astra\n")
    env_file.chmod(0o600)
    script = tmp_path / "submit.sh"
    script.write_text((repo / "fastfill/comparison/submit_023_judge.sh").read_text().replace(
        "PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill", f"PROJECT_ROOT='{project}'").replace(
        "ENV_FILE=/home/u6uv/shanliantian.u6uv/.fastfill_api.env", f"ENV_FILE='{env_file}'"))
    commands = tmp_path / "bin"
    commands.mkdir()
    (commands / "squeue").write_text("#!/bin/sh\nexit 0\n")
    (commands / "sbatch").write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SUBMITTED"\nprintf "12345\\n"\n')
    for command in commands.iterdir():
        command.chmod(0o755)
    submitted = tmp_path / "submitted"
    env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"], "SUBMITTED": str(submitted), "USER": "local"}

    api = root / "api-bundle"
    manifest = judge.load_manifest(api / "manifest.json")
    cases = [{**c, "prompt_file": str(api / c["prompt_file"]),
              "images": [{**i, "path": str(api / i["path"])} for i in c["images"]]}
             for c in manifest["cases"][:2]]
    write(smoke / "manifest.json", {"schema_version": 1, "expected_model": "gpt-6-astra", "cases": cases})
    config = replace(judge.load_config(env_file), reasoning_effort="high")
    records = []
    for case in cases:
        identity, _ = judge.prepare_case(case, smoke, config)
        stem = judge.sha256(case["id"].encode())
        attempt, result = smoke / "results" / (stem + ".attempt.json"), smoke / "results" / (stem + ".result.json")
        scores = {"functionality_score": 7, "layout_score": 6, "aesthetics_score": 5}
        write(attempt, {**identity, "status": "attempted"})
        write(result, {**identity, "status": "completed", "returned_model": config.model,
                       "raw_final_text": json.dumps(scores), "scores": scores})
        records.append((attempt, result))
    write(base / "smoke.completion.json", {"exit_code": 0})
    return script, repo, base, env, submitted, records


@pytest.mark.parametrize("failure", ["failed_job", "no_completion", "empty_results", "partial", "failed_case",
                                     "invalid_scores", "wrong_model", "wrong_request", "changed_manifest", "changed_image"])
def test_full_submission_rejects_unsuccessful_or_changed_smoke(submission, failure):
    script, repo, base, env, submitted, records = submission
    if failure == "failed_job":
        write(base / "smoke.completion.json", {"exit_code": 1})
    elif failure == "no_completion":
        (base / "smoke.completion.json").unlink()
    elif failure == "empty_results":
        for pair in records:
            for path in pair:
                path.unlink()
    elif failure == "partial":
        records[1][1].unlink()
    elif failure in ("failed_case", "invalid_scores", "wrong_model", "wrong_request"):
        path = records[0][1]
        record = json.loads(path.read_text())
        if failure == "failed_case":
            record.update(status="failed", error="timeout")
        elif failure == "invalid_scores":
            record["scores"]["functionality_score"] = 99
            record["raw_final_text"] = json.dumps(record["scores"])
        elif failure == "wrong_model":
            record["returned_model"] = "other-model"
        else:
            record["request_sha256"] = "a" * 64
        write(path, record)
    else:
        manifest_path = base / "smoke/manifest.json"
        manifest = judge.load_manifest(manifest_path)
        if failure == "changed_manifest":
            manifest["cases"].reverse()
            write(manifest_path, manifest)
        else:
            Path(manifest["cases"][0]["images"][0]["path"]).write_bytes(b"changed artifact")
    result = subprocess.run(["bash", str(script), str(repo), "full"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not submitted.exists()
    assert "local-test-only" not in result.stdout + result.stderr


def test_full_submission_accepts_two_verified_completed_smoke_cases(submission):
    script, repo, _, env, submitted, _ = submission
    result = subprocess.run(["bash", str(script), str(repo), "full"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = submitted.read_text()
    assert "--expected-total 696" in command and "--workers 4" in command
    assert "--model gpt-6-astra --reasoning-effort high" in command
    assert "local-test-only" not in result.stdout + result.stderr + command
