"""Server autopilot: finish the running pair, select, retrain on the newest data, evaluate, publish.

Phases (state in <runs>/autorun/STATUS.json, log in <runs>/autorun/autorun.log):
  A  wait for the running training runs; a run that dies without run_manifest.json is resumed from its
     newest state-step into <run>-resume<k> (at most twice per run)
  B  evaluate the newest three model-step checkpoints of every run on the validation set (three-field
     projection, collision-aware spread decoding), one GPU each
  C  select the winner (``score``), export its RoomGenBench hand-off for the five benchmark rooms (CPU)
  D  wait for <data root>/NEXT_DATA.json (written after a rebuild is uploaded to Hugging Face) at most
     ``--data-wait-h`` hours; download and verify it and pull the matching code, else keep the current data
  E  train the winner's configuration on all GPUs for ``--epochs`` epochs (same global batch), with the
     same crash-resume rule as A
  F  evaluate its newest checkpoints, select, evaluate the best on the test set, export RoomGenBench,
     upload the best model directory to a private Hugging Face model repository

Every decision and command is logged; a failing step stops the autopilot with STATUS phase "failed".
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

ENV = {"PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": "/home/jovyan/shanliantian/.huggingface"}


def score(report):
    """Lower is better. Errors relative to the report's own trivial baselines, plus distribution penalties.

    size: log-size error / per-category median size; yaw: yaw error / constant yaw; position: bottom-centre
    error / room-centre placement; |central-quarter fraction - GT|; |mean wall distance - GT| / GT;
    overlap above GT + 1 point and objects outside the room above GT count fivefold.
    """
    ref, base = report["model"]["reference"], report["baselines"]
    pred, gt = report["collapse"]["predicted"], report["collapse"]["ground_truth"]
    ratio = lambda a, b: a / b if a is not None and b else 1.
    s = (ratio(ref["log_size_error"]["mean"], base["category_median_size"]["log_size_error"]["mean"])
         + ratio(ref["yaw_error_rad"].get("mean"), base["uniform_yaw"]["yaw_error_rad"].get("mean"))
         + ratio(ref["bottom_center_error_m"]["mean"], base["room_center_position"]["bottom_center_error_m"]["mean"])
         + abs(pred["central_quarter_fraction"] - gt["central_quarter_fraction"])
         + abs(pred["mean_nearest_wall_distance_m"] - gt["mean_nearest_wall_distance_m"]) / max(gt["mean_nearest_wall_distance_m"], 1e-6)
         + 5 * max(0., pred["bev_overlap_rate_iou_gt_0.3"] - gt["bev_overlap_rate_iou_gt_0.3"] - .01))
    if "out_of_room_fraction" in pred:
        s += 5 * max(0., pred["out_of_room_fraction"] - gt.get("out_of_room_fraction", 0.))
    return s


def next_config(config, *, world_size, train_rows, epochs, global_batch=96):
    """The winner's configuration rescaled to ``world_size`` GPUs at batch 1 and about ``global_batch``."""
    config = json.loads(json.dumps(config))
    training = config["training"]
    accumulation = max(1, round(global_batch / world_size))
    steps = math.ceil(epochs * train_rows / (world_size * accumulation))
    training.update(batch_size=1, gradient_accumulation_steps=accumulation, steps=steps, resume=None,
                    checkpoint_every=500, validate_every=500)
    config["optimizer"]["warmup_steps"] = round(.03 * steps)
    return config


def step_of(path):
    return int(str(path).rsplit("-", 1)[1])


def complete_states(outputs):
    """Saved states of every output of one run, oldest first. state-step-N counts as complete once model-step-N
    exists (train.py exports it after save_state and the barrier), so a state torn by a crash is skipped."""
    states = [p for o in outputs for p in glob.glob(f"{o}/state-step-*")
              if Path(p).with_name(f"model-step-{step_of(p)}").is_dir()]
    return sorted(states, key=step_of)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


class Autopilot:
    def __init__(self, args):
        self.a, self.root = args, Path(args.repo)
        self.runs = self.root / "runs"
        self.dir = self.runs / "autorun"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status = {"phase": "start", "history": []}
        self.python = str(self.root / "env/bin/python")

    def log(self, message, **fields):
        line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message, **fields}
        with (self.dir / "autorun.log").open("a") as stream:
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.status["history"].append(line)
        (self.dir / "STATUS.json").write_text(json.dumps(self.status, indent=2, ensure_ascii=False) + "\n")

    def phase(self, name, **fields):
        self.status.update(phase=name, **fields)
        self.log(f"phase {name}", **fields)

    def sh(self, args, *, gpus=None, log=None, wait=True):
        env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": gpus if gpus is not None else ""}
        self.log("run", args=" ".join(map(str, args)), gpus=gpus)
        out = open(self.root / log, "w") if log else subprocess.DEVNULL  # relative logs live under the repo
        process = subprocess.Popen(list(map(str, args)), cwd=self.root, env=env, stdout=out, stderr=subprocess.STDOUT)
        if not wait:
            return process
        if process.wait():
            raise RuntimeError(f"command failed ({process.returncode}): {' '.join(map(str, args))}")
        return process

    # -- training ---------------------------------------------------------------------------------
    def launch(self, config, data, output, gpus):
        world = len(gpus.split(","))
        return self.sh([self.python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
                        "--module", "fastfill.v2.train", "--config", config, "--data", f"{data}/train.jsonl",
                        "--validation", f"{data}/validation.jsonl", "--expect-data-sha256", sha256(f"{data}/train.jsonl"),
                        "--expect-validation-sha256", sha256(f"{data}/validation.jsonl"), "--output", output]
                       + list(self.extra_resume), gpus=gpus, log=f"{output}.log", wait=False)

    def out(self, name):
        return str(self.root / "runs" / name)

    def supervise(self, jobs):
        """jobs: name -> dict(process|None, config, data, gpus, output). Resume crashed runs twice."""
        while True:
            pending = 0
            for name, job in jobs.items():
                done = (Path(job["output"]) / "run_manifest.json").is_file()
                alive = job["process"] is not None and job["process"].poll() is None
                if job["process"] is None and not done:  # adopted from an earlier launcher: watch its log
                    alive = time.time() - Path(f"{job['output']}.log").stat().st_mtime < 1800
                if done or alive:
                    pending += not done
                    continue
                states = complete_states(job["outputs"])
                if job.get("resumes", 0) >= 2 or not states:
                    raise RuntimeError(f"{name} stopped without a final manifest and cannot be resumed")
                job["resumes"] = job.get("resumes", 0) + 1
                job["outputs"].append(f"{job['outputs'][0]}-resume{job['resumes']}")
                self.extra_resume = ("--resume", states[-1])
                self.log(f"{name} died; resuming", state=states[-1], output=job["outputs"][-1])
                job["process"] = self.launch(job["config"], job["data"], job["outputs"][-1], job["gpus"])
                self.extra_resume = ()
                job["output"] = job["outputs"][-1]
                pending += 1
            if not pending:
                return
            time.sleep(120)

    # -- evaluation -------------------------------------------------------------------------------
    def eval_data(self):
        """A seeded random sample of --eval-rows validation rows: the file is grouped by source, so its head
        (what --max-samples reads) leaves whole sources out."""
        path = self.dir / f"validation-{Path(self.data).name}-{self.a.eval_rows}.jsonl"
        if not path.is_file():
            with open(f"{self.data}/validation.jsonl") as stream:
                lines = [line.rstrip("\n") + "\n" for line in stream if line.strip()]
            random.Random(0).shuffle(lines)
            path.write_text("".join(lines[:int(self.a.eval_rows)]))
        return path

    def evaluate_candidates(self, runs, tag, *, newest=3, gpus=("1", "2", "3", "4", "5", "6", "7")):
        """runs: one list of output directories (the run and its resumes) per run; its newest checkpoints compete."""
        candidates, data = [], self.eval_data()
        for outputs in runs:
            candidates += sorted((p for o in outputs for p in glob.glob(f"{o}/model-step-*")), key=step_of)[-newest:]
        results = []
        for start in range(0, len(candidates), len(gpus)):
            batch = candidates[start:start + len(gpus)]
            jobs = []
            for gpu, checkpoint in zip(gpus, batch):
                out = self.runs / f"{tag}-{Path(checkpoint).parent.name}-{Path(checkpoint).name}"
                if (out / "report.json").is_file():  # a restarted autopilot reuses finished evaluations
                    jobs.append((checkpoint, out, None))
                    continue
                jobs.append((checkpoint, out, self.sh([self.python, "-m", "fastfill.v2.evaluate", "--checkpoint", checkpoint,
                    "--data", data, "--projection", "minimal",
                    "--device", "cuda", "--grid-decode", "spread", "--output", out], gpus=gpu, log=f"{out}.log", wait=False)))
            for checkpoint, out, process in jobs:
                if process is not None and process.wait():
                    self.log("evaluation failed", checkpoint=checkpoint)
                    continue
                try:
                    value = score(json.loads((out / "report.json").read_text()))
                except (KeyError, TypeError, ValueError) as error:
                    self.log("evaluation unscorable", checkpoint=checkpoint, error=repr(error))
                    continue
                results.append({"checkpoint": checkpoint, "report": str(out / "report.json"), "score": value})
        if not results:
            raise RuntimeError("no checkpoint could be evaluated")
        results.sort(key=lambda r: r["score"])
        (self.dir / f"{tag}-selection.json").write_text(json.dumps(results, indent=2) + "\n")
        self.log(f"{tag} selection", winner=results[0], ranking=[(r["checkpoint"], round(r["score"], 4)) for r in results])
        return results[0]

    def roomgenbench(self, checkpoint, tag):
        return self.sh(["bash", "runs/roomgenbench/run_checkpoint.sh", checkpoint, tag],
                       log=self.runs / "roomgenbench" / f"{tag}.log", wait=False)

    # -- data ---------------------------------------------------------------------------------------
    def next_data(self):
        marker, deadline = Path(self.a.data_root) / "NEXT_DATA.json", time.time() + 3600 * self.a.data_wait_h
        while not marker.is_file() and time.time() < deadline:
            time.sleep(120)
        if not marker.is_file():
            self.log("no new data before the deadline; keeping the current data", data=self.data)
            return
        spec = json.loads(marker.read_text())
        missing = {"train.jsonl", "validation.jsonl", "test.jsonl"} - set(spec["sha256"])
        if missing:  # phase F reads test.jsonl only ~10 h later
            raise RuntimeError(f"NEXT_DATA.json does not pin {sorted(missing)}")
        target = Path(self.a.data_root) / spec["name"]
        if not target.is_dir():
            staging = Path(self.a.data_root) / f".download-{spec['name']}"
            self.sh([str(self.root / "env/bin/hf"), "download", spec["repo"], "--repo-type", "dataset", "--revision",
                     spec["revision"], "--include", f"{spec['folder']}/*", "--local-dir", staging], log=self.dir / "hf-download.log")
            (staging / spec["folder"]).rename(target)
        for name, digest in spec["sha256"].items():
            if sha256(target / name) != digest:
                raise RuntimeError(f"{target / name} does not match NEXT_DATA.json")
        self.sh(["git", "pull", "--ff-only", "origin", "main"], log=self.dir / "git-pull-data.log")
        self.sh(["git", "merge-base", "--is-ancestor", spec["commit"], "HEAD"])
        self.data = str(target)
        self.log("switched to new data", data=self.data, commit=spec["commit"])

    def upload(self, checkpoint, name):
        """<name>/model-step-N plus <name>/tokenizer and the run manifests: the layout predict/evaluate load
        (they read the tokenizer from the checkpoint's parent)."""
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(self.a.model_repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=Path(checkpoint).parent, path_in_repo=name, repo_id=self.a.model_repo, repo_type="model",
                          allow_patterns=[f"{Path(checkpoint).name}/*", "tokenizer/*", "run_manifest*.json"],
                          commit_message=f"FastFill v2 {name}")
        self.log("uploaded model", repo=self.a.model_repo, path=name)

    def run(self):
        self.extra_resume, self.data = (), str(Path(self.a.current_data).resolve())
        self.phase("A-wait-current-runs")
        jobs = {name: {"process": None, "config": config, "data": self.data, "gpus": gpus, "output": self.out(name),
                       "outputs": [self.out(name)]} for name, config, gpus in self.a.current}
        self.supervise(jobs)
        self.sh(["git", "pull", "--ff-only", "origin", "main"], log=self.dir / "git-pull-a.log")
        self.phase("B-evaluate-current")
        winner = self.evaluate_candidates([job["outputs"] for job in jobs.values()], "select1")
        self.phase("C-selected", winner=winner)
        handoff = self.roomgenbench(winner["checkpoint"], "best-round1")
        # its five rooms each start a new process from the repo; finish them before phase D pulls new code
        self.log("round-1 RoomGenBench hand-off finished", returncode=handoff.wait())
        self.phase("D-wait-data")
        self.next_data()
        run_dir = Path(winner["checkpoint"]).parent
        config = json.loads((run_dir / "run_manifest_start.json").read_text())["config"]
        rows = sum(1 for _ in open(f"{self.data}/train.jsonl"))
        config = next_config(config, world_size=7, train_rows=rows, epochs=self.a.epochs)
        name = f"main7-cell{config['loss']['position_cell']:g}-{Path(self.data).name}-e{self.a.epochs}".replace(".", "")
        config_path = self.dir / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        self.phase("E-train", run=name, config=str(config_path), data=self.data, steps=config["training"]["steps"])
        job = {"config": str(config_path), "data": self.data, "gpus": "1,2,3,4,5,6,7", "output": self.out(name),
               "outputs": [self.out(name)]}
        job["process"] = self.launch(job["config"], self.data, job["output"], job["gpus"])
        self.supervise({name: job})
        self.phase("F-evaluate-final")
        best = self.evaluate_candidates([job["outputs"]], "select2", newest=4)
        self.upload(best["checkpoint"], name)  # before the ~7 h single-GPU test evaluation, which may fail
        self.roomgenbench(best["checkpoint"], f"best-{name}")
        out = self.runs / f"test-{name}"
        self.sh([self.python, "-m", "fastfill.v2.evaluate", "--checkpoint", best["checkpoint"], "--data", f"{self.data}/test.jsonl",
                 "--projection", "minimal", "full", "--device", "cuda", "--grid-decode", "spread", "--output", out],
                gpus="1", log=f"{out}.log")
        self.phase("done", best=best, test_report=str(out / "report.json"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="/home/jovyan/shanliantian/fastfill")
    p.add_argument("--current", nargs=3, action="append", metavar=("RUN", "CONFIG", "GPUS"), required=True)
    p.add_argument("--current-data", required=True)
    p.add_argument("--data-root", default="/home/jovyan/shanliantian/fastfill/data")
    p.add_argument("--data-wait-h", type=float, default=2.)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eval-rows", default="3000")
    p.add_argument("--model-repo", default="liantian/fastfill-v2-models")
    os.environ.setdefault("HF_HOME", ENV["HF_HOME"])  # the in-process Hugging Face upload uses the logged-in home
    pilot = Autopilot(p.parse_args(argv))
    try:
        pilot.run()
    except Exception as error:
        pilot.phase("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
