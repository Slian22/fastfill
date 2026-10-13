"""Build DPO preference pairs for FastFill v1.1 (Stage 1 on-policy vLLM + Stage 2 geometric hard negatives).

Stage 1 (`--mode stage1_vllm`):
  Uses vLLM (`temperature=0.85, n=4`) on clean training rooms to sample candidate layouts, validates each candidate
  with `fastfill.validate.check`, and pairs:
  - `chosen`: a zero-collision valid candidate (or the clean ground-truth reference layout)
  - `rejected`: a candidate that parses (or near-misses) but exhibits `main` collision, `fixed_blocking`, `oob`, or `support_fail`.

Stage 2 (`--mode stage2_hardneg`):
  Constructs OptiScene-style targeted single-object/multi-object geometric perturbations from clean training rooms,
  verified by `fastfill.validate.check`:
  1. `collision_shift`: moves one floor object onto another floor object's footprint (`main` / `severe` collision).
  2. `door_block`: moves a floor object directly onto a `fixed` door/column box (`fixed_blocking`).
  3. `wall_oob`: pushes an edge object 25-45 cm through the room boundary wall (`oob`).
  4. `support_z_corrupt`: corrupts a supported child's `z` by +0.35 m (floating) or -0.25 m (sunk into parent).
  5. `size_corrupt`: scales an object's `[sx, sy, sz]` by 2.2x so it collides with adjacent furniture or walls.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from fastfill.scene import num, prompt_text
from fastfill.v1_1.scene_v1_1 import parse_v1_1
from fastfill.validate import check
from fastfill.v1_1.evaluation_protocol import file_identity, model_identity, write_json_atomic


@contextmanager
def _generation_lock(out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with (out_file.parent / f'.{out_file.name}.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _training_generation_identity(data_dir: Path, settings: dict) -> dict:
    package = Path(__file__).resolve().parents[1]
    modules = ('v1_1/build_dpo.py', 'v1_1/audit_pairs.py', 'v1_1/scene_v1_1.py',
               'v1_1/evaluation_protocol.py', 'validate.py', 'scene.py', 'anchors.py')
    return {'train': {'path': str((data_dir / 'train.jsonl').resolve()), **file_identity(data_dir / 'train.jsonl')},
            'settings': settings,
            'code_sha256': {name: file_identity(package / name)['sha256'] for name in modules}}


def _generation_identity(data_dir: Path, model_path: str, settings: dict) -> dict:
    return {**_training_generation_identity(data_dir, settings),
            'protocol': 'fastfill_v1.1_stage1_cache_v1', 'model': model_identity(model_path)}


def _reference_layout(messages: list) -> dict | None:
    # Lazy import avoids the audit -> message parser import cycle. Generation and
    # sanitization deliberately use exactly the same registered acceptance policy.
    from fastfill.v1_1.audit_pairs import _violations
    ids = [o['id'] for o in json.loads(messages[1]['content'])['objects']]
    reference, error = parse_v1_1(messages[2]['content'], ids)
    if error:
        return None
    defects, _ = _violations(messages[:2], messages[2]['content'], reference)
    return None if defects else reference


def _verified_pair(record: dict, chosen: str, rejected: str, kind: str) -> dict | None:
    from fastfill.v1_1.audit_pairs import audit_pair
    pair = {'uid': record['uid'], 'source': record['source'], 'reject_kind': kind,
            'prompt': record['messages'][:2], 'chosen': chosen, 'rejected': rejected}
    return pair if audit_pair(pair, record)['keep'] else None


def _read_generation_cache(path: Path, identity: dict, references: dict) -> dict:
    if not path.exists():
        return {'identity': identity, 'pairs': [], 'attempted_uids': []}
    try:
        cache = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Legacy or unbound partial cache; use a new output: {path}') from exc
    if not isinstance(cache, dict) or cache.get('identity') != identity:
        raise ValueError(f'Legacy/unbound cache or generation identity differs: {path}')
    pairs, attempted = cache.get('pairs'), cache.get('attempted_uids')
    if not isinstance(pairs, list) or not isinstance(attempted, list) or \
            any(not isinstance(uid, str) or uid not in references for uid in attempted) or len(set(attempted)) != len(attempted):
        raise ValueError(f'Invalid cached pair/attempt identities: {path}')
    from fastfill.v1_1.audit_pairs import audit_pair
    seen = set()
    for pair in pairs:
        uid = pair.get('uid') if isinstance(pair, dict) else None
        if not isinstance(uid, str) or uid not in references or uid in seen or uid not in attempted:
            raise ValueError(f'Invalid or duplicate cached pair identity: {path}')
        if not audit_pair(pair, references[uid])['keep']:
            raise ValueError(f'Cached pair fails independent semantic validation: {uid}')
        seen.add(uid)
    return cache


def _publish_pairs(out_file: Path, pairs: list[dict], *, allow_existing: bool = True,
                   sort_keys: bool = True) -> None:
    payload = ''.join(json.dumps(pair, ensure_ascii=False, allow_nan=False, sort_keys=sort_keys) + '\n' for pair in pairs)
    if out_file.exists():
        if not allow_existing:
            raise FileExistsError(f'Refusing to replace existing output: {out_file}')
        if out_file.read_text(encoding='utf-8') != payload:
            raise ValueError(f'Existing output differs from verified cached pairs: {out_file}')
        return
    descriptor, name = tempfile.mkstemp(prefix=f'.{out_file.name}.', dir=out_file.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Both paths share a filesystem. Link publishes complete bytes atomically
        # and, unlike replace, cannot overwrite an output that appeared meanwhile.
        os.link(temporary, out_file)
    finally:
        temporary.unlink(missing_ok=True)


def _room_from_v1_1_messages(messages: list[dict[str, str]], asst_text: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    u = json.loads(messages[1]["content"])
    target_str = asst_text if asst_text is not None else messages[2]["content"]
    ids = [o["id"] for o in u["objects"]]
    pl, err = parse_v1_1(target_str, ids)
    if err is not None:
        return None
    objs = []
    raw_pl = []
    for o in u["objects"]:
        p = pl[o["id"]]
        cat = o.get("category") or o["id"].rsplit("_", 1)[0].replace("_", " ")
        objs.append(
            {
                "id": o["id"],
                "category": cat,
                "size": p["size"],
                "pos": p["pos"],
                "yaw": p["yaw"],
                "parent": p["on"],
            }
        )
        raw_pl.append(
            {
                "id": o["id"],
                **({"on": p["on"]} if p.get("on") else {}),
                "size": [max(num(v), 0.01) for v in p["size"]],
                "pos": [num(v) for v in p["pos"]],
                "yaw": round(math.degrees(p["yaw"])) % 360,
            }
        )
    fixed = [
        {
            "id": fx["id"],
            "category": fx["id"].rsplit("_", 1)[0].replace("_", " "),
            "size": fx["size"],
            "pos": fx["pos"],
            "yaw": math.radians(fx["yaw"]),
        }
        for fx in u.get("fixed", [])
    ]
    room = {
        "boundary_type": u["boundary_type"],
        "boundary": u["boundary"],
        "height": u.get("height"),
        "fixed": fixed,
        "objects": objs,
        "constraints": u.get("constraints") or [],
    }
    return room, raw_pl


def _make_hard_negative(messages: list[dict[str, str]], rng: random.Random) -> tuple[str, str] | None:
    """Create a verified geometric hard-negative assistant JSON string and its perturbation kind."""
    parsed = _room_from_v1_1_messages(messages)
    if parsed is None:
        return None
    room, pls = parsed
    if len(pls) < 2:
        return None
    c0 = check(room)
    if not c0["valid"] or c0["collisions"]["main"] or c0["fixed_blocking"] or \
            _reference_layout(messages) is None:
        return None

    floor_indices = [i for i, p in enumerate(pls) if not p.get("on")]
    child_indices = [i for i, p in enumerate(pls) if p.get("on")]
    fixed_doors = [
        fx for fx in room["fixed"]
        if not {"window", "windows"}.intersection(fx["category"].split())
    ]

    ops = ["collision_shift", "wall_oob", "size_corrupt"]
    if len(floor_indices) >= 2:
        ops.insert(0, "collision_shift")
    if fixed_doors and floor_indices:
        ops.insert(0, "door_block")
    if child_indices:
        ops.insert(0, "support_z_corrupt")
    rng.shuffle(ops)

    for op in ops:
        bad_pls = copy.deepcopy(pls)
        if op == "collision_shift" and len(floor_indices) >= 2:
            i, j = rng.sample(floor_indices, 2)
            # Move object i onto object j's XY center with slight offset (guaranteed 3D box collision)
            bad_pls[i]["pos"][0] = num(bad_pls[j]["pos"][0] + rng.uniform(-0.08, 0.08))
            bad_pls[i]["pos"][1] = num(bad_pls[j]["pos"][1] + rng.uniform(-0.08, 0.08))
        elif op == "door_block" and fixed_doors and floor_indices:
            i = rng.choice(floor_indices)
            door = rng.choice(fixed_doors)
            bad_pls[i]["pos"][0] = num(door["pos"][0])
            bad_pls[i]["pos"][1] = num(door["pos"][1])
        elif op == "support_z_corrupt" and child_indices:
            i = rng.choice(child_indices)
            delta = rng.choice([-0.22, 0.35])
            bad_pls[i]["pos"][2] = num(max(0.0, bad_pls[i]["pos"][2] + delta))
        elif op == "wall_oob" and floor_indices:
            i = rng.choice(floor_indices)
            xs = [pt[0] for pt in room["boundary"]]
            ys = [pt[1] for pt in room["boundary"]]
            if rng.random() < 0.5:
                bad_pls[i]["pos"][0] = num(max(xs) + 0.35)
            else:
                bad_pls[i]["pos"][1] = num(min(ys) - 0.35)
        elif op == "size_corrupt" and floor_indices:
            i = rng.choice(floor_indices)
            bad_pls[i]["size"] = [num(v * 2.1) for v in bad_pls[i]["size"]]
        else:
            continue

        bad_str = json.dumps({"placements": bad_pls}, separators=(",", ":"), ensure_ascii=False)
        bad_parsed = _room_from_v1_1_messages(messages, bad_str)
        if bad_parsed is None:
            continue
        c_bad = check(bad_parsed[0])
        if (
            not c_bad["valid"]
            or c_bad["collisions"]["main"]
            or c_bad["collisions"]["severe"]
            or c_bad["fixed_blocking"]
        ):
            return bad_str, op
    return None


def build_stage2_hardneg(data_dir: Path, out_file: Path, max_pairs: int, seed: int) -> None:
    if max_pairs < 1:
        raise ValueError('max_pairs must be positive')
    with _generation_lock(out_file):
        if out_file.exists() or out_file.is_symlink():
            raise FileExistsError(f'Refusing to replace existing output: {out_file}')
        settings = {'mode': 'stage2_hardneg', 'max_pairs': max_pairs, 'seed': seed}
        identity = _training_generation_identity(data_dir, settings)
        pairs, by_kind = _stage2_pairs(data_dir, max_pairs, seed)
        if _training_generation_identity(data_dir, settings) != identity:
            raise ValueError('Generation identity changed during Stage 2; output not published')
        _publish_pairs(out_file, pairs, allow_existing=False, sort_keys=False)
    print(f'[stage2_hardneg] Wrote {len(pairs)} DPO pairs to {out_file}: {by_kind}', flush=True)


def _stage2_pairs(data_dir: Path, max_pairs: int, seed: int) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(seed)
    excl = {"oob_objects", "overlapping_furniture", "fixed_collision", "severe_collision", "main_collision"}
    rows = []
    with open(data_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not (set(r.get("flags", {})) & excl):
                rows.append(r)
    rng.shuffle(rows)
    pairs = []
    by_kind: dict[str, int] = {}
    for record in rows:
        if len(pairs) >= max_pairs:
            break
        result = _make_hard_negative(record['messages'], rng)
        if result is None:
            continue
        bad_str, kind = result
        pair = _verified_pair(record, record['messages'][2]['content'], bad_str, kind)
        if pair is not None:
            by_kind[kind] = by_kind.get(kind, 0) + 1
            pairs = [*pairs, pair]
    return pairs, by_kind


def build_stage1_vllm(
    model_path: str,
    data_dir: Path,
    out_file: Path,
    max_pairs: int,
    n_samples: int,
    max_len: int,
    tp: int,
    seed: int,
) -> None:
    if min(max_pairs, n_samples, max_len, tp) < 1:
        raise ValueError('max_pairs, n_samples, max_len, and tp must be positive')
    with _generation_lock(out_file):
        _build_stage1_vllm(model_path, data_dir, out_file, max_pairs, n_samples, max_len, tp, seed)


def _build_stage1_vllm(model_path: str, data_dir: Path, out_file: Path, max_pairs: int,
                       n_samples: int, max_len: int, tp: int, seed: int) -> None:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    vis_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    eff_tp = max(1, min(tp, vis_gpus))

    settings = {'max_pairs': max_pairs, 'n_samples': n_samples, 'max_len': max_len,
                'tp': tp, 'effective_tp': eff_tp, 'seed': seed, 'temperature': .85, 'top_p': .95}
    identity = _generation_identity(data_dir, model_path, settings)

    rng = random.Random(seed)
    excl = {"oob_objects", "overlapping_furniture", "fixed_collision", "severe_collision", "main_collision"}
    rows = []
    with open(data_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not (set(r.get("flags", {})) & excl):
                rows.append(r)
    references = {r['uid']: r for r in rows}
    if len(references) != len(rows):
        raise ValueError('Duplicate training UIDs cannot identify generation pairs')
    rng.shuffle(rows)
    # Sample up to 2x max_pairs candidate prompts so even if >50% of rooms have 4/4 clean generations, we hit max_pairs
    cand_rows = rows[: min(len(rows), int(max_pairs * 2))]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = out_file.with_suffix(out_file.suffix + ".partial")
    if out_file.exists() and not partial_file.exists():
        raise ValueError(f'Existing output has no bound generation cache: {out_file}')
    cache = _read_generation_cache(partial_file, identity, references)
    pairs = cache['pairs']
    attempted = set(cache['attempted_uids'])
    written = len(pairs)
    if written > max_pairs:
        raise ValueError('Cached pairs exceed registered max_pairs')
    write_json_atomic(partial_file, cache)

    if written >= max_pairs:
        if _generation_identity(data_dir, model_path, settings) != identity:
            raise ValueError('Generation identity changed during cache validation')
        _publish_pairs(out_file, pairs)
        print(f"[stage1_vllm] Already reached {written} pairs -> {out_file}", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(model_path)
    prompts = [prompt_text(tok, r["messages"][:2]) for r in cand_rows]
    p_lens = [len(ids) for ids in tok(prompts, add_special_tokens=False)["input_ids"]]
    fit = [
        (i, min(4096, max_len - p_lens[i]))
        for i in range(len(prompts))
        if max_len - p_lens[i] >= 64 and cand_rows[i]["uid"] not in attempted
        and _reference_layout(cand_rows[i]['messages']) is not None
    ]

    from fastfill.v1_1.audit_pairs import _violations
    llm = LLM(model=model_path, max_model_len=max_len, tensor_parallel_size=eff_tp,
              disable_custom_all_reduce=True) if fit else None
    chunk_size = 2048
    for start in range(0, len(fit), chunk_size):
        if written >= max_pairs:
            break
        sub_fit = fit[start:start + chunk_size]
        sampling = [SamplingParams(temperature=.85, top_p=.95, n=n_samples, max_tokens=n, seed=seed + i)
                    for i, n in sub_fit]
        outputs = llm.generate([prompts[i] for i, _ in sub_fit], sampling)
        if len(outputs) != len(sub_fit):
            raise ValueError('Generation output count does not match requested prompts')
        for (index, _), output in zip(sub_fit, outputs):
            if written >= max_pairs:
                break
            record = cand_rows[index]
            reference = _reference_layout(record['messages'])
            model_chosen, model_rejected, reject_kind = None, None, None
            for candidate in output.outputs:
                text = candidate.text.strip()
                defects, parsed = _violations(record['messages'][:2], text, reference)
                if not parsed:  # retain the registered parseable-candidate generation strategy
                    continue
                if not defects and model_chosen is None:
                    model_chosen = text
                elif defects and model_rejected is None:
                    model_rejected, reject_kind = text, '+'.join(defects)
            if model_rejected is not None:
                chosen = model_chosen if model_chosen is not None else record['messages'][2]['content']
                pair = _verified_pair(record, chosen, model_rejected, reject_kind)
                if pair is not None:
                    pairs = [*pairs, pair]
                    written += 1
            attempted = attempted | {record['uid']}
        # Identity and rows form one atomic document, so an interrupted write can
        # never leave a complete-looking prefix from another model or policy.
        write_json_atomic(partial_file, {'identity': identity, 'pairs': pairs,
                                         'attempted_uids': sorted(attempted)})
    if _generation_identity(data_dir, model_path, settings) != identity:
        raise ValueError('Generation identity changed during inference; output not published')
    _publish_pairs(out_file, pairs)
    by_kind = {}
    for pair in pairs:
        kind = pair['reject_kind']
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(f'[stage1_vllm] Wrote {written} independently verified pairs to {out_file}: {by_kind}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["stage1_vllm", "stage2_hardneg"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=str, default="", help="complete local model snapshot (hash-bound before generation)")
    ap.add_argument("--max_pairs", type=int, default=12000)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "stage2_hardneg":
        build_stage2_hardneg(args.data, args.out, args.max_pairs, args.seed)
    else:
        if not args.model:
            ap.error("--model is required for --mode stage1_vllm")
        build_stage1_vllm(args.model, args.data, args.out, args.max_pairs, args.n_samples, args.max_len, args.tp, args.seed)


if __name__ == "__main__":
    main()
