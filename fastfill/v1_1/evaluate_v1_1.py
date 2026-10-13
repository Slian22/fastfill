"""Evaluate FastFill v1.1 checkpoints on dev_rooms.jsonl / test_rooms.jsonl / test_constrained_rooms.jsonl.

Reports raw model predictions, repaired predictions (`repair_v1_1`), and reference GT side-by-side:
- `parse_rate`, `complete_rate`
- `valid_rate`, `valid_rate_repaired` (no OOB, no support fail, no ceiling fail)
- `inside_1mm_rate`, `inside_1mm_rate_repaired`
- `accepted_rate` (v1 acceptance: valid + inside 1mm + no fixed_blocking + all constraints hold)
- `zero_collision_accepted_rate` (accepted + 0 main collisions + 0 severe collisions)
- `rooms_with_main`, `rooms_with_main_repaired`, `rooms_with_severe`, `rooms_with_severe_repaired`
- `rooms_with_fixed_collision`, `mean_oor`, `mean_oor_repaired`, `mean_size_iou_3d`
- `constraint_satisfaction`, `constraint_satisfaction_repaired`
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from shapely.geometry import Polygon

from fastfill.evaluate import BUCKETS, inside
from fastfill.scene import footprint, prompt_text
from fastfill.v1_1.postprocess import check_v1_1, repair_v1_1
from fastfill.v1_1.scene_v1_1 import apply_v1_1, gt_placements_v1_1, messages_v1_1, parse_v1_1, prior_size
from fastfill.validate import check, holds
import math

DEFAULT_PY_VLLM = "/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python"


def _hf_generation_budget(chunk: list[tuple[int, int]], max_len: int, max_new_tokens: int) -> int:
    """Padding expands every input to the longest prompt in the batch."""
    return min(max_new_tokens, max_len - max(length for length, _ in chunk))


def _effective_tp(requested_tp: int) -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis.strip():
        n_gpus = len([x for x in vis.split(",") if x.strip()])
    else:
        try:
            import torch

            n_gpus = torch.cuda.device_count()
        except Exception:
            n_gpus = 1
    n_gpus = max(1, n_gpus)
    tp = min(max(1, requested_tp), n_gpus)
    while tp > 1 and n_gpus % tp != 0:
        tp -= 1
    return tp


def generate_v1_1(
    model: str,
    prompts: list[str],
    max_len: int,
    batch: int = 16,
    tp: int = 1,
    use_vllm: bool = True,
    max_new_tokens: int = 4096,
) -> list[str | None]:
    res: list[str | None] = [None] * len(prompts)
    if not prompts:
        return res
    eff_tp = _effective_tp(tp)

    if use_vllm:
        try:
            from vllm import LLM, SamplingParams

            llm = LLM(
                model=model,
                max_model_len=max_len,
                tensor_parallel_size=eff_tp,
                disable_custom_all_reduce=True,
            )
            tok = llm.get_tokenizer()
            fit = [(i, max_len - len(tok(p)["input_ids"])) for i, p in enumerate(prompts)]
            fit = [(i, n) for i, n in fit if n > 0]
            outs = llm.generate(
                [prompts[i] for i, _ in fit],
                [SamplingParams(temperature=0, max_tokens=min(max_new_tokens, n)) for _, n in fit],
            )
            for (i, _), o in zip(fit, outs):
                res[i] = o.outputs[0].text
            return res
        except ImportError:
            if Path(DEFAULT_PY_VLLM).is_file() and Path(sys.executable).resolve() != Path(DEFAULT_PY_VLLM).resolve():
                try:
                    with tempfile.TemporaryDirectory(prefix="fastfill_vllm_") as tmpd:
                        in_p = Path(tmpd) / "prompts.json"
                        out_p = Path(tmpd) / "outputs.json"
                        in_p.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
                        helper = (
                            "import json, sys\n"
                            "from vllm import LLM, SamplingParams\n"
                            "model, in_p, out_p, max_len, tp, max_new = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])\n"
                            "prompts = json.load(open(in_p, encoding='utf-8'))\n"
                            "res = [None] * len(prompts)\n"
                            "llm = LLM(model=model, max_model_len=max_len, tensor_parallel_size=tp, disable_custom_all_reduce=True)\n"
                            "tok = llm.get_tokenizer()\n"
                            "fit = [(i, max_len - len(tok(p)['input_ids'])) for i, p in enumerate(prompts)]\n"
                            "fit = [(i, n) for i, n in fit if n > 0]\n"
                            "outs = llm.generate([prompts[i] for i, _ in fit], [SamplingParams(temperature=0, max_tokens=min(max_new, n)) for _, n in fit])\n"
                            "for (i, _), o in zip(fit, outs):\n"
                            "    res[i] = o.outputs[0].text\n"
                            "json.dump(res, open(out_p, 'w', encoding='utf-8'), ensure_ascii=False)\n"
                        )
                        subprocess.run(
                            [DEFAULT_PY_VLLM, "-c", helper, model, str(in_p), str(out_p), str(max_len), str(eff_tp), str(max_new_tokens)],
                            check=True,
                        )
                        return json.loads(out_p.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[generate_v1_1] External PY_VLLM fallback note: {exc}; falling back to HF generate.", flush=True)
        except Exception as exc:
            print(f"[generate_v1_1] vLLM error ({exc}); falling back to batched HF generate.", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev)
    m.eval()
    lengths = sorted((len(tok(p, add_special_tokens=False)["input_ids"]), i) for i, p in enumerate(prompts))
    fit_items = [(length, i) for length, i in lengths if length < max_len]
    bs = max(1, batch)
    with torch.inference_mode():
        for k in range(0, len(fit_items), bs):
            chunk = fit_items[k : k + bs]
            idx = [i for _, i in chunk]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(
                m.device
            )
            out = m.generate(**enc, max_new_tokens=_hf_generation_budget(chunk, max_len, max_new_tokens), do_sample=False)
            for i, t in zip(idx, tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)):
                res[i] = t
    return res


def _load_tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, padding_side="left")


class _GenerationSession:
    """One model load per evaluator process; authoritative runs never switch backends silently."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        if args.use_vllm:
            from vllm import LLM

            self.model = LLM(model=args.model, max_model_len=args.max_len,
                tensor_parallel_size=_effective_tp(args.tp), disable_custom_all_reduce=True)
            self.tokenizer = self.model.get_tokenizer()
        else:
            import torch
            from transformers import AutoModelForCausalLM

            self.tokenizer = _load_tokenizer(args.model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.model.eval()

    def generate(self, prompts: list[str]) -> list[str | None]:
        if self.args.use_vllm:
            from vllm import SamplingParams

            fit = [(i, self.args.max_len - len(self.tokenizer(p)["input_ids"])) for i, p in enumerate(prompts)]
            fit = [(i, n) for i, n in fit if n > 0]
            result: list[str | None] = [None] * len(prompts)
            if fit:
                outputs = self.model.generate([prompts[i] for i, _ in fit],
                    [SamplingParams(temperature=0, max_tokens=min(self.args.max_new_tokens, n)) for _, n in fit])
                if len(outputs) != len(fit):
                    raise ValueError("Generation output count does not match prompt count")
                for (i, _), output in zip(fit, outputs):
                    result[i] = output.outputs[0].text if output.outputs else None
            return result
        return self._generate_hf(prompts)

    def _generate_hf(self, prompts: list[str]) -> list[str | None]:
        import torch

        result: list[str | None] = [None] * len(prompts)
        lengths = sorted((len(self.tokenizer(p, add_special_tokens=False)["input_ids"]), i)
                         for i, p in enumerate(prompts))
        fit = [(length, i) for length, i in lengths if length < self.args.max_len]
        with torch.inference_mode():
            for start in range(0, len(fit), self.args.batch):
                chunk = fit[start:start + self.args.batch]
                indices = [i for _, i in chunk]
                encoded = self.tokenizer([prompts[i] for i in indices], return_tensors="pt", padding=True,
                                         add_special_tokens=False).to(self.model.device)
                outputs = self.model.generate(**encoded, do_sample=False,
                    max_new_tokens=_hf_generation_budget(chunk, self.args.max_len, self.args.max_new_tokens))
                texts = self.tokenizer.batch_decode(outputs[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                if len(texts) != len(indices):
                    raise ValueError("Generation output count does not match prompt count")
                for index, text in zip(indices, texts):
                    result[index] = text
        return result


def _size_iou_3d(room: dict[str, Any], pl: dict[str, dict[str, Any]]) -> float:
    ious = []
    for o in room["objects"]:
        oid = o["id"]
        if oid not in pl:
            continue
        gs = o["size"]
        ps = pl[oid]["size"]
        inter = min(gs[0], ps[0]) * min(gs[1], ps[1]) * min(gs[2], ps[2])
        vol_g = gs[0] * gs[1] * gs[2]
        vol_p = ps[0] * ps[1] * ps[2]
        union = vol_g + vol_p - inter
        ious.append(inter / union if union > 0 else 0.0)
    return sum(ious) / len(ious) if ious else 0.0


def _sizes_realistic(room: dict[str, Any], pl: dict[str, dict[str, Any]]) -> bool:
    for o in room["objects"]:
        oid = o["id"]
        if oid not in pl:
            return False
        gs = o.get("size") or prior_size(o.get("category"))
        ps = pl[oid]["size"]
        for p_dim, g_dim in zip(ps, gs):
            if float(p_dim) < min(float(g_dim) * 0.9, max(0.05, 0.35 * float(g_dim))):
                return False
    return True


def geometry_v1_1(room: dict[str, Any]) -> dict[str, Any]:
    c = check_v1_1(room)
    return {
        "valid": c["valid"],
        "oob_objects": len(c["oob"]),
        "support_fail": len(c["support_fail"]),
        "ceiling_objects": len(c["ceiling"]),
        "oor": c["oor"],
        "coll_fixed": len(c["fixed_collisions"]),
        "fixed_blocking": len(c["fixed_blocking"]),
        **{f"coll_{k}": len(c["collisions"][k]) for k in BUCKETS[:4]},
    }


def judge_v1_1(room: dict[str, Any], pl: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cons = room.get("constraints") or []
    pred = apply_v1_1(room, pl)
    r = geometry_v1_1(pred)
    r["size_iou_3d"] = _size_iou_3d(room, pl)
    r["constraints"] = [[c[0], holds(c, pred["objects"], room["boundary"])] for c in cons]

    floor = Polygon(room["boundary"])
    rep = repair_v1_1(room, pl, resolve_collisions=True)
    fixed = apply_v1_1(room, rep)
    c_rep = check_v1_1(fixed)

    r["inside_1mm"] = inside(pred, floor)
    r["inside_1mm_repaired"] = inside(fixed, floor)
    r["valid_repaired"] = c_rep["valid"]
    r["oor_repaired"] = c_rep["oor"]
    r["coll_main_repaired"] = len(c_rep["collisions"]["main"])
    r["coll_severe_repaired"] = len(c_rep["collisions"]["severe"])
    r["coll_fixed_repaired"] = len(c_rep["fixed_collisions"])
    r["support_fail_repaired"] = len(c_rep["support_fail"])
    r["oob_objects_repaired"] = len(c_rep["oob"])
    r["ceiling_objects_repaired"] = len(c_rep["ceiling"])
    r["constraints_repaired"] = [[k[0], holds(k, fixed["objects"], room["boundary"])] for k in cons]

    def _obj_repaired(oid: str) -> bool:
        if oid not in rep or oid not in pl:
            return True
        pos_diff = any(abs(float(rep[oid]["pos"][a]) - float(pl[oid]["pos"][a])) > 1e-4 for a in range(3))
        size_diff = any(abs(float(rep[oid]["size"][a]) - float(pl[oid]["size"][a])) > 1e-4 for a in range(3))
        yaw_diff = abs((float(rep[oid]["yaw"]) - float(pl[oid]["yaw"]) + math.pi) % (2 * math.pi) - math.pi) > 1e-3
        on_diff = rep[oid].get("on") != pl[oid].get("on")
        return pos_diff or size_diff or yaw_diff or on_diff

    r["repaired_objects"] = sum(_obj_repaired(o["id"]) for o in room["objects"] if o["id"] in pl)
    r["size_realistic"] = _sizes_realistic(room, rep)
    r["accepted"] = (
        c_rep["valid"]
        and r["inside_1mm_repaired"]
        and not c_rep["fixed_blocking"]
        and r["size_realistic"]
        and all(v for _, v in r["constraints_repaired"])
    )
    r["zero_collision_accepted"] = r["accepted"] and r["coll_main_repaired"] == 0 and r["coll_severe_repaired"] == 0
    return r


def score_v1_1(room: dict[str, Any], text: str | None, error: str | None = None) -> dict[str, Any]:
    ids = [o["id"] for o in room["objects"]]
    gt = judge_v1_1(room, gt_placements_v1_1(room))
    r: dict[str, Any] = {
        "uid": room["uid"],
        "source": room["source"],
        "group": room["group"],
        "n": len(ids),
        "held_out": bool(room.get("meta", {}).get("eval_only")),
        "gt": gt,
    }
    pl, err = ({}, error) if error else parse_v1_1(text, ids) if text is not None else (None, None)
    r["error"] = err
    r["parsed"] = not error and err not in ("truncated_or_bad_json", "bad_json")
    cons = room.get("constraints") or []
    if err is not None:
        return {
            **r,
            "valid": False,
            "valid_repaired": False,
            "accepted": False,
            "zero_collision_accepted": False,
            "constraints": [[c[0], False] for c in cons],
            "constraints_repaired": [[c[0], False] for c in cons],
        }
    return {**r, **(gt if text is None else judge_v1_1(room, pl))}


def _score_worker(args: tuple[dict[str, Any], str | None, str | None]) -> dict[str, Any]:
    room, text, err = args
    return score_v1_1(room, text, err)


def summarize_v1_1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def agg(rs: list[dict[str, Any]]) -> dict[str, Any] | None:
        n = len(rs)
        if not n:
            return None
        ok = [r for r in rs if r["error"] is None]
        objs = sum(r["n"] for r in ok)
        s: dict[str, Any] = {
            "rooms": n,
            "groups": len({r["group"] for r in rs}),
            "parse_rate": sum(r["parsed"] for r in rs) / n,
            "complete_rate": len(ok) / n,
            "errors": dict(collections.Counter(r["error"] for r in rs if r["error"])),
        }
        for k, key in (
            ("valid_rate", "valid"),
            ("valid_rate_repaired", "valid_repaired"),
            ("inside_1mm_rate", "inside_1mm"),
            ("inside_1mm_rate_repaired", "inside_1mm_repaired"),
            ("accepted_rate", "accepted"),
            ("zero_collision_accepted_rate", "zero_collision_accepted"),
        ):
            s[k] = sum(r.get(key, False) for r in rs) / n
            s[f"{k}_gt"] = sum(r["gt"][key] for r in rs) / n

        rate = lambda num_v, den_v: num_v / den_v if den_v else None
        for k, key in (
            ("repaired_object_rate", "repaired_objects"),
            ("oob_object_rate", "oob_objects"),
            ("oob_object_rate_repaired", "oob_objects_repaired"),
            ("support_fail_rate", "support_fail"),
            ("support_fail_rate_repaired", "support_fail_repaired"),
            ("ceiling_object_rate", "ceiling_objects"),
            ("ceiling_object_rate_repaired", "ceiling_objects_repaired"),
        ):
            s[k] = rate(sum(r[key] for r in ok), objs)
            s[f"{k}_gt"] = rate(sum(r["gt"][key] for r in ok), objs)

        for k in BUCKETS:
            name = "fixed_collision" if k == "fixed" else k
            s[f"rooms_with_{name}"] = rate(sum(r[f"coll_{k}"] > 0 for r in ok), len(ok))
            s[f"rooms_with_{name}_gt"] = rate(sum(r["gt"][f"coll_{k}"] > 0 for r in ok), len(ok))
        s["rooms_with_main_repaired"] = rate(sum(r["coll_main_repaired"] > 0 for r in ok), len(ok))
        s["rooms_with_severe_repaired"] = rate(sum(r["coll_severe_repaired"] > 0 for r in ok), len(ok))
        s["mean_oor"] = rate(sum(r["oor"] for r in ok), len(ok))
        s["mean_oor_repaired"] = rate(sum(r["oor_repaired"] for r in ok), len(ok))
        s["mean_oor_gt"] = rate(sum(r["gt"]["oor"] for r in ok), len(ok))
        s["mean_size_iou_3d"] = rate(sum(r["size_iou_3d"] for r in ok), len(ok))

        by_c, by_r = collections.defaultdict(list), collections.defaultdict(list)
        for r in rs:
            for t, v in r.get("constraints", []):
                by_c[t].append(v)
            for t, v in r.get("constraints_repaired", []):
                by_r[t].append(v)
        s["constraint_satisfaction"] = {t: sum(v) / len(v) for t, v in sorted(by_c.items())}
        s["constraint_satisfaction_repaired"] = {t: sum(v) / len(v) for t, v in sorted(by_r.items())}
        return s

    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    return {
        "in_dist": agg([r for r in rows if not r["held_out"]]),
        "held_out": agg([r for r in rows if r["held_out"]]),
        **{k: agg(v) for k, v in sorted(by.items())},
    }


def evaluation_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in (
        "all_rows", "gt_only", "max_len", "max_new_tokens", "batch", "tp", "use_vllm", "chunk_size")}


def evaluation_code_hashes() -> dict[str, str]:
    """Bind all Python dependencies, including geometry, parsing and repair code."""
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py")) if "tests" not in path.relative_to(root).parts}


def get_evaluation_identity(args: argparse.Namespace) -> dict[str, Any]:
    from fastfill.v1_1.evaluation_protocol import evaluation_identity

    return evaluation_identity(args.rooms, args.model, evaluation_settings(args), code_hashes=evaluation_code_hashes())


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _seal_chunk(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "content_sha256": _json_digest(value)}


def _check_chunk_digest(saved: Any, path: Path) -> None:
    if not isinstance(saved, dict) or saved.get("content_sha256") != _json_digest(
            {key: value for key, value in saved.items() if key != "content_sha256"}):
        raise ValueError(f"Saved chunk content digest missing or integrity mismatch: {path}")


def _room_chunks(path: str, chunk_size: int):
    with open(path, encoding="utf-8") as handle:
        chunk = []
        for line in handle:
            chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _load_chunk(path: Path, expected: dict[str, Any], field: str) -> dict[str, Any]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ValueError(f"Saved chunk integrity mismatch: {path}")
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Saved chunk identity or ordered cohort mismatch: {path}")
    if not isinstance(saved.get(field), list) or len(saved[field]) != len(expected["uids"]):
        raise ValueError(f"Saved chunk output count mismatch: {path}")
    _check_chunk_digest(saved, path)
    return saved


def _score_chunk(rooms: list[dict[str, Any]], texts: list[str | None], workers: int) -> list[dict[str, Any]]:
    tasks = [(room, text, "missing_generation" if text is None else None) for room, text in zip(rooms, texts)]
    if len(tasks) > 8 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            return list(pool.map(_score_worker, tasks, chunksize=max(1, len(tasks) // (workers * 4))))
    return [_score_worker(task) for task in tasks]


def _publish_evaluation(out: Path, chunks: list[tuple[Path, Path]], identity: dict[str, Any]) -> None:
    from fastfill.v1_1.evaluation_protocol import finalize_evaluation, write_json_atomic

    rows = []
    temporary = out / f"predictions.jsonl.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        for generated_path, scored_path in chunks:
            generated = json.loads(generated_path.read_text(encoding="utf-8"))
            scored = json.loads(scored_path.read_text(encoding="utf-8"))
            _check_chunk_digest(generated, generated_path)
            _check_chunk_digest(scored, scored_path)
            if scored.get("generation_sha256") != _json_digest(generated):
                raise ValueError(f"Saved generation/scoring chunk integrity mismatch: {scored_path}")
            for offset, (row, text) in enumerate(zip(scored["rows"], generated["outputs"])):
                handle.write(json.dumps({**row, "output": text, "evaluation_row": generated["start"] + offset},
                                        ensure_ascii=False) + "\n")
            rows.extend(scored["rows"])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, out / "predictions.jsonl")
    summary = summarize_v1_1(rows)
    write_json_atomic(out / "metrics.json", summary)
    # The completion receipt is the commit point; partial artifacts are never complete.
    finalize_evaluation(out, identity)
    print(json.dumps({k: summary[k] for k in ("in_dist", "held_out")}, indent=2), flush=True)


def _run_resumable_locked(args: argparse.Namespace, out: Path) -> None:
    from fastfill.v1_1.evaluation_protocol import evaluation_complete, write_json_atomic

    identity = get_evaluation_identity(args)
    manifest_path = out / "evaluation_manifest.json"
    expected_manifest = {"identity": identity, "chunk_size": args.chunk_size}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected_manifest:
            raise ValueError("Evaluation manifest identity changed; choose a new output directory")
    else:
        if any(path.name != ".evaluation.lock" for path in out.iterdir()):
            raise ValueError("Refusing to overwrite existing/legacy evaluation outputs; choose a new output directory")
        write_json_atomic(manifest_path, expected_manifest)
    if evaluation_complete(out, identity):
        print(f"[evaluation] Verified complete: {out}", flush=True)
        return
    if (out / "completion.json").exists():
        raise ValueError("Completed evaluation artifacts failed identity/integrity validation")
    chunks_dir = out / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    session, tokenizer = None, None
    chunks = []
    completed = 0
    for chunk_index, rooms in enumerate(_room_chunks(args.rooms, args.chunk_size)):
        metadata = {"identity_sha256": _json_digest(identity), "start": completed, "uids": [r["uid"] for r in rooms]}
        generated_path = chunks_dir / f"{chunk_index:06d}.generated.json"
        scored_path = chunks_dir / f"{chunk_index:06d}.scored.json"
        if generated_path.exists():
            generated = _load_chunk(generated_path, metadata, "outputs")
        else:
            if scored_path.exists():
                raise ValueError(f"Saved scored chunk has no generation: {scored_path}")
            if session is None:
                tokenizer = _load_tokenizer(args.model)
                session = _GenerationSession(args)
            prompts = [prompt_text(tokenizer, messages_v1_1(room, room.get("constraints"), with_target=False)) for room in rooms]
            outputs = session.generate(prompts)
            if len(outputs) != len(rooms) or any(text is not None and not isinstance(text, str) for text in outputs):
                raise ValueError("Generation output count/type does not match chunk")
            generated = _seal_chunk({**metadata, "outputs": outputs})
            write_json_atomic(generated_path, generated)
        score_metadata = {**metadata, "generation_sha256": _json_digest(generated)}
        if scored_path.exists():
            scored = _load_chunk(scored_path, score_metadata, "rows")
        else:
            scored = _seal_chunk({**score_metadata, "rows": _score_chunk(rooms, generated["outputs"], args.workers)})
            write_json_atomic(scored_path, scored)
        if [r.get("uid") for r in scored["rows"]] != metadata["uids"]:
            raise ValueError(f"Scored chunk ordered cohort mismatch: {scored_path}")
        chunks.append((generated_path, scored_path))
        completed += len(rooms)
        write_json_atomic(out / "progress.json", {
            "status": "running", "completed_rows": completed,
            "expected_rows": identity["dataset"]["line_count"],
            "chunk": chunk_index, "identity_sha256": _json_digest(identity),
        })
        print(f"[evaluation] Scored {completed}/{identity['dataset']['line_count']} rows; chunk {chunk_index}", flush=True)
    if completed != identity["dataset"]["line_count"]:
        raise ValueError("Evaluation cohort changed while running")
    if get_evaluation_identity(args) != identity:
        raise ValueError("Evaluation identity changed while running; no metrics will be published")
    _publish_evaluation(out, chunks, identity)
    write_json_atomic(out / "progress.json", {
        "status": "complete", "completed_rows": completed,
        "expected_rows": completed, "identity_sha256": _json_digest(identity),
    })


def run_resumable_evaluation(args: argparse.Namespace) -> None:
    if not args.all_rows or args.limit is not None or args.gt_only or not args.model:
        raise ValueError("--resume-evaluation requires --all-rows, a model, no --limit and no --gt-only")
    for key in ("chunk_size", "max_len", "max_new_tokens", "batch", "tp", "workers"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    import fcntl

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another evaluator already owns {out}") from exc
        _run_resumable_locked(args, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rooms", required=True, help="dev_rooms.jsonl / test_rooms.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="merged checkpoint dir or HF id")
    ap.add_argument("--gt-only", action="store_true")
    ap.add_argument("--max_len", "--max-len", dest="max_len", type=int, default=40960)
    ap.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit", "--max-samples", "--max_samples", dest="limit", type=int, default=None)
    ap.add_argument("--use-vllm", dest="use_vllm", action="store_true", default=True)
    ap.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--all-rows", action="store_true", help="Keep every dataset row, including duplicate prompts and UIDs")
    ap.add_argument("--resume-evaluation", action="store_true", help="Persist identity-bound chunks and a full-cohort completion receipt")
    ap.add_argument("--chunk-size", type=int, default=256)
    a = ap.parse_args()

    if a.resume_evaluation:
        run_resumable_evaluation(a)
        return

    uniq: dict[str, dict[str, Any]] = {}
    all_rooms = []
    for line in open(a.rooms, encoding="utf-8"):
        r = json.loads(line)
        if a.all_rows:
            all_rooms.append(r)
        else:
            uniq.setdefault(messages_v1_1(r, r.get("constraints"), with_target=False)[1]["content"], r)
    rooms = (all_rooms if a.all_rows else list(uniq.values()))[: a.limit]

    if a.gt_only:
        texts: list[str | None] = [None] * len(rooms)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.model)
        prompts = [prompt_text(tok, messages_v1_1(r, r.get("constraints"), with_target=False)) for r in rooms]
        texts = generate_v1_1(
            a.model,
            prompts,
            a.max_len,
            a.batch,
            tp=a.tp,
            use_vllm=a.use_vllm,
            max_new_tokens=a.max_new_tokens,
        )

    if len(texts) != len(rooms):
        raise ValueError("Generation output count does not match room count")
    tasks = [(r, t, "prompt_too_long" if t is None and not a.gt_only else None) for r, t in zip(rooms, texts)]
    if len(tasks) > 8 and a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            rows = list(pool.map(_score_worker, tasks, chunksize=max(1, len(tasks) // (a.workers * 4))))
    else:
        rows = [_score_worker(task) for task in tasks]

    os.makedirs(a.out, exist_ok=True)
    for pred_name in ("predictions.jsonl", "eval_v1_1_predictions.jsonl"):
        tmp_pred = f"{a.out}/{pred_name}.tmp.{os.getpid()}"
        with open(tmp_pred, "w", encoding="utf-8") as f:
            for r, t in zip(rows, texts):
                f.write(json.dumps({**r, "output": t}, ensure_ascii=False) + "\n")
        os.replace(tmp_pred, f"{a.out}/{pred_name}")
    summary = summarize_v1_1(rows)
    for sum_name in ("metrics.json", "eval_v1_1_summary.json"):
        tmp_sum = f"{a.out}/{sum_name}.tmp.{os.getpid()}"
        with open(tmp_sum, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp_sum, f"{a.out}/{sum_name}")
    print(json.dumps({k: summary[k] for k in ("in_dist", "held_out")}, indent=2))


if __name__ == "__main__":
    if sys.argv[1:]:
        main()
    else:
        from fastfill.v1_1.scene_v1_1 import target_json_v1_1

        room = {
            "uid": "u",
            "source": "S",
            "group": "g",
            "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
            "boundary_type": "polygon",
            "height": 2.8,
            "fixed": [],
            "constraints": [["faces", "chair_1", "box_1"]],
            "objects": [
                {"id": "chair_1", "category": "chair", "size": [0.5, 0.5, 0.9], "pos": [1, 0.22, 0], "yaw": 0.0},
                {"id": "box_1", "category": "box", "size": [0.06, 0.06, 0.3], "pos": [1.35, 0.03, 0], "yaw": 0.0},
            ],
        }
        s = score_v1_1(room, target_json_v1_1(room))
        assert s["accepted"] and s["zero_collision_accepted"] and abs(s["size_iou_3d"] - 1.0) < 1e-6, s
        print("evaluate_v1_1.py self-check ok")
