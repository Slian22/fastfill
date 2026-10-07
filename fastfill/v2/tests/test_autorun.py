"""Autopilot decisions: selection score, the rescaled next-run configuration and the final acceptance."""
import json
from pathlib import Path

from fastfill.v2.autorun import next_config, score


def _report(size, yaw, pos, central=.17, wall=.66, overlap=.01, out=0.):
    collapse = lambda c, w, o, r: {"central_quarter_fraction": c, "mean_nearest_wall_distance_m": w,
                                   "bev_overlap_rate_iou_gt_0.3": o, "out_of_room_fraction": r}
    return {"model": {"reference": {"log_size_error": {"mean": size}, "yaw_error_rad": {"mean": yaw},
                                    "bottom_center_error_m": {"mean": pos}}},
            "baselines": {"category_median_size": {"log_size_error": {"mean": .4}}, "uniform_yaw": {"yaw_error_rad": {"mean": .7}},
                          "room_center_position": {"bottom_center_error_m": {"mean": 2.5}}},
            "collapse": {"predicted_matched": collapse(central, wall, overlap, out), "ground_truth": collapse(.17, .66, .01, 0.)}}


def test_score_prefers_accurate_layouts_and_punishes_collapse_or_leaving_the_room():
    good, worse = score(_report(.3, .6, 2.0)), score(_report(.35, .6, 2.0))
    assert good < worse
    assert score(_report(.3, .6, 2.0, central=.9, wall=2.0)) > good + .7          # everything in the centre
    assert score(_report(.3, .6, 2.0, overlap=.2)) > good + .85                    # stacked
    assert score(_report(.3, .6, 2.0, out=.5)) > good + 2                          # pushed out of the room


def test_next_config_keeps_the_model_and_loss_and_rescales_to_seven_gpus():
    config = {"model": {"position_head": "grid_residual"}, "loss": {"position_cell": .5},
              "training": {"steps": 3887, "batch_size": 1, "gradient_accumulation_steps": 32, "resume": "x",
                           "checkpoint_every": 500, "validate_every": 500},
              "optimizer": {"warmup_steps": 117}}
    out = next_config(config, world_size=7, train_rows=124584, epochs=5)
    assert out["model"] == config["model"] and out["loss"] == config["loss"]
    t = out["training"]
    assert t["gradient_accumulation_steps"] == 14 and t["batch_size"] == 1 and t["resume"] is None
    assert t["steps"] == -(-5 * 124584 // (7 * 14)) and out["optimizer"]["warmup_steps"] == round(.03 * t["steps"])
    assert json.loads(json.dumps(config))["training"]["steps"] == 3887  # input untouched


def test_adopt_restores_the_resume_chain(tmp_path):
    import argparse
    from fastfill.v2.autorun import Autopilot
    pilot = Autopilot(argparse.Namespace(repo=str(tmp_path)))
    base = tmp_path / "runs" / "job"
    for suffix in ("", "-resume1", "-resume2"):
        (tmp_path / "runs" / f"job{suffix}").mkdir(parents=True, exist_ok=True)
        (tmp_path / "runs" / f"job{suffix}.log").write_text("")
    job = pilot.adopt({"outputs": [str(base)], "output": str(base)})
    assert job["resumes"] == 2 and job["output"].endswith("job-resume2") and len(job["outputs"]) == 3


def _phase_f(tmp_path, *, failed=0, over=0, upload_error=False, rows_late=False):
    """Autopilot.run end to end with stubbed processes: a finished prior run, data unchanged, phase F checks."""
    import argparse
    from unittest.mock import patch
    from fastfill.v2 import autorun

    class Proc:
        def __init__(self, code=0): self.code = code
        def wait(self): return self.code
        def poll(self): return self.code

    class Thread:  # the LLM thread: never finishes when rows arrive late, else nothing to do
        def __init__(self, *a, **k): pass
        def start(self): pass
        def join(self, timeout): pass

    old = tmp_path / "old"
    old.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        (old / name).write_text("{}\n")
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "NEXT_DATA.json").write_text(json.dumps({"name": "../old", "repo": "r", "revision": "v", "folder": "f", "commit": "c",
        "sha256": {n: autorun.sha256(old / n) for n in ("train.jsonl", "validation.jsonl", "test.jsonl")}}))
    a = argparse.Namespace(repo=str(tmp_path), current_data=str(old), data_root=str(tmp_path / "data"), data_wait_h=0., epochs=5,
                           eval_rows="3000", model_repo="mock/model", llm_env=str(tmp_path / "noenv"), llm_rows="300",
                           current=[("prior", "c", "1,2,3,4")])
    pilot = autorun.Autopilot(a)
    checkpoint = tmp_path / "runs/prior/model-step-10"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint.parent / "run_manifest_start.json").write_text(json.dumps({"config": {"training": {}, "optimizer": {}, "loss": {"position_cell": .5}}}))
    pilot.supervise = lambda *a: None
    pilot.evaluate_candidates = lambda *a, **k: {"checkpoint": str(checkpoint), "report": "x", "score": 1.}
    pilot.launch = lambda *a: Proc()
    if upload_error:
        def upload(*a): raise OSError("403 Forbidden")
        pilot.upload = upload
    else:
        pilot.upload = lambda *a: None
    commands = []
    def sh(args, **kw):
        args = list(map(str, args))
        commands.append(args)
        if args[0] == "bash":
            for i in range(5):
                room = pilot.runs / "roomgenbench" / args[-1] / f"r{i}.layout_boxes"
                room.mkdir(parents=True, exist_ok=True)
                (room / "receipt.json").write_text("{}")
        if "fastfill.v2.evaluate" in args:
            if rows_late:  # the LLM rows land while the test set is evaluated
                rows = pilot.runs / "llm-prompt-300/rows.jsonl"
                rows.parent.mkdir(parents=True, exist_ok=True)
                rows.write_text("{}\n")
            out = Path(args[args.index("--output") + 1])
            out.mkdir(parents=True)  # evaluate refuses an existing output
            (out / "report.json").write_text(json.dumps({**_report(.3, .6, 2.), "requests": 10, "inference_failed_requests": failed,
                                                         "over_capacity_requests": over}))
        return Proc()
    pilot.sh = sh
    with patch.object(autorun.threading, "Thread", Thread):
        pilot.run()
    return pilot, commands



def test_phase_f_reports_over_capacity_requests_without_failing_on_them(tmp_path):
    pilot, commands = _phase_f(tmp_path, failed=5, over=5)
    assert pilot.status["phase"] == "done" and pilot.status["accepted"]
    decodes = {a[a.index("--grid-decode") + 1] for a in commands if "--grid-decode" in a}
    assert decodes == {"spread", "argmax"}  # the model alone is evaluated even without LLM rows
    assert "5 (5 over capacity)" in (pilot.dir / "SUMMARY.md").read_text()
    pilot, _ = _phase_f(tmp_path / "real", failed=6, over=5)
    assert pilot.status["failures"] == ["test: 1 of 10 requests without a layout"]


def test_phase_f_survives_an_upload_error_and_late_llm_rows(tmp_path):
    pilot, commands = _phase_f(tmp_path, upload_error=True, rows_late=True)
    assert pilot.status["phase"] == "done-with-failures" and pilot.status["failures"] == ["upload failed: OSError: 403 Forbidden"]
    assert "upload failed" in (pilot.dir / "SUMMARY.md").read_text()
    assert any("test-" in " ".join(a) for a in commands)  # the test set was still evaluated after the failed upload


def test_phase_f_can_run_again_after_a_failed_upload(tmp_path):
    _phase_f(tmp_path, upload_error=True)
    pilot, commands = _phase_f(tmp_path)  # e.g. after fixing the credentials
    assert pilot.status["phase"] == "done" and pilot.status["uploaded_model"] == "mock/model/main7-cell05-old-e5"
    assert not any(a[0] == "bash" for a in commands)  # the same checkpoint's five rooms are not exported twice


def test_no_gpu_evaluation_starts_beside_a_running_training(tmp_path, monkeypatch):
    import argparse
    import pytest
    from fastfill.v2 import autorun
    pilot = autorun.Autopilot(argparse.Namespace(repo=str(tmp_path), current_data=str(tmp_path), eval_rows="3000"))
    (tmp_path / "run/model-step-5").mkdir(parents=True)
    monkeypatch.setattr(autorun, "any_training_alive", lambda: True)
    monkeypatch.setattr(pilot, "sh", lambda *a, **k: pytest.fail("a GPU job was started"))
    with pytest.raises(RuntimeError, match="while a fastfill.v2.train process runs"):
        pilot.evaluate_candidates([[str(tmp_path / "run")]], "select1", data=tmp_path / "sample.jsonl")
