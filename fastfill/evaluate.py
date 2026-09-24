"""Generate layouts for held-out rooms and score them.

    python -m fastfill.evaluate --model outputs/ff-v1-merged --rooms data/fastfill_v1/test_rooms.jsonl --out eval/ff-v1
    python -m fastfill.evaluate --gt-only --rooms data/fastfill_v1/test_rooms.jsonl --out eval/gt   # reference layouts

Rooms come from fastfill.build (already canonical: ids and order as the model saw them in training).
Every request is counted: unparsable or incomplete outputs are invalid and fail all their constraints (no retries,
no filtering); identical requests are scored once. Metrics are reported per source and pooled separately for
in-distribution sources and held-out (eval-only) sources.
Collision buckets and OOR are reported next to the GT value on the same rooms.
Uses vLLM when installed, else HF generate. Greedy decoding.
"""
import argparse
import collections
import copy
import json
import os

from shapely.geometry import Polygon

from fastfill.interface import snap_inside
from fastfill.scene import apply, footprint, gt_placements, messages, parse, prompt_text
from fastfill.validate import check, holds

BUCKETS = ("main", "severe", "tuck", "contained")


def generate(model, prompts, max_new_tokens, batch):
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model, max_model_len=8192)
        outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_new_tokens))
        return [o.outputs[0].text for o in outs]
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="auto")
        res = []
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
            out = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            res += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return res


def geometry(room):
    c = check(room)
    return {"valid": c["valid"], "oob_objects": len(c["oob"]), "support_fail": len(c["support_fail"]),
            "ceiling_objects": len(c["ceiling"]), "oor": c["oor"],
            **{f"coll_{k}": len(c["collisions"][k]) for k in BUCKETS}}


def score(room, text):
    ids = [o["id"] for o in room["objects"]]
    r = {"uid": room["uid"], "source": room["source"], "group": room["group"], "n": len(ids),
         "held_out": bool(room.get("meta", {}).get("eval_only")), "gt": geometry(room)}
    pl, err = parse(text, ids) if text is not None else (gt_placements(room), None)
    r["error"] = err
    r["parsed"] = err not in ("truncated_or_bad_json", "bad_json")
    cons = room.get("constraints") or []
    if err is not None:        # a failed request fails every constraint it was asked to satisfy
        return {**r, "valid": False, "constraints": [[c[0], False] for c in cons],
                "constraints_repaired": [[c[0], False] for c in cons]}
    pred = apply(room, pl)
    g = geometry(pred)
    r["constraints"] = [[c[0], holds(c, pred["objects"], room["boundary"])] for c in cons]
    # the raw proposal above is the model's result; the interface's <= 10 cm push back inside is reported apart
    floor = Polygon(room["boundary"])
    rep = copy.deepcopy(pl)
    snap_inside(rep, room, floor)
    fixed = apply(room, rep)
    r["inside_1mm"], r["inside_1mm_repaired"] = inside(pred, floor), inside(fixed, floor)
    r["valid_repaired"] = check(fixed)["valid"]
    r["constraints_repaired"] = [[c[0], holds(c, fixed["objects"], room["boundary"])] for c in cons]   # repair can break one
    r["repaired_objects"] = sum(rep[i]["pos"] != pl[i]["pos"] for i in pl)
    return {**r, **g}


def inside(room, floor, tol=1e-3):
    """Every object, on others too, inside the boundary within 1 mm (EmbodiedGen's containment rule)."""
    grown = floor.buffer(tol)
    return all(grown.covers(Polygon(footprint(o))) for o in room["objects"])


def summarize(rows):
    def agg(rs):
        n = len(rs)
        if not n:
            return None
        ok = [r for r in rs if r["error"] is None]
        objs = sum(r["n"] for r in ok)
        s = {"rooms": n, "groups": len({r["group"] for r in rs}),
             "parse_rate": sum(r["parsed"] for r in rs) / n, "complete_rate": len(ok) / n,
             "valid_rate": sum(r["valid"] for r in rs) / n,
             # after the interface's deterministic repair (not model skill); a failed request stays failed
             "valid_rate_repaired": sum(r.get("valid_repaired", False) for r in rs) / n,
             "inside_1mm_rate": sum(r.get("inside_1mm", False) for r in rs) / n,
             "inside_1mm_rate_repaired": sum(r.get("inside_1mm_repaired", False) for r in rs) / n,
             "repaired_object_rate": (sum(r["repaired_objects"] for r in ok) / objs) if objs else None,
             "errors": dict(collections.Counter(r["error"] for r in rs if r["error"]))}
        # geometry is measured on complete answers only; with none it is unknown (None), not a perfect 0
        rate = lambda num, den: num / den if den else None
        s["oob_object_rate"] = rate(sum(r["oob_objects"] for r in ok), objs)
        s["support_fail_rate"] = rate(sum(r["support_fail"] for r in ok), objs)
        s["ceiling_object_rate"] = rate(sum(r["ceiling_objects"] for r in ok), objs)
        for k in BUCKETS:   # share of rooms with a collision of this kind: model vs GT on the same (complete) rooms
            s[f"rooms_with_{k}"] = rate(sum(r[f"coll_{k}"] > 0 for r in ok), len(ok))
            s[f"rooms_with_{k}_gt"] = rate(sum(r["gt"][f"coll_{k}"] > 0 for r in ok), len(ok))
        s["mean_oor"] = rate(sum(r["oor"] for r in ok), len(ok))
        s["mean_oor_gt"] = rate(sum(r["gt"]["oor"] for r in ok), len(ok))
        by_c, by_r = collections.defaultdict(list), collections.defaultdict(list)
        for r in rs:                   # every request, failed ones included
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
    # eval-only sources (SpatialGen, SceneSmith) are out of distribution: never pooled with the rest
    return {"in_dist": agg([r for r in rows if not r["held_out"]]), "held_out": agg([r for r in rows if r["held_out"]]),
            **{k: agg(v) for k, v in sorted(by.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", required=True, help="dev_rooms.jsonl / test_rooms.jsonl from fastfill.build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="merged checkpoint dir or HF id")
    ap.add_argument("--gt-only", action="store_true", help="score the reference layouts themselves")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    uniq = {}                          # identical requests (same room text and constraints) are scored once
    for line in open(a.rooms):
        r = json.loads(line)
        uniq.setdefault(messages(r, r.get("constraints"), with_target=False)[1]["content"], r)
    rooms = list(uniq.values())[:a.limit]
    if a.gt_only:
        texts = [None] * len(rooms)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        prompts = [prompt_text(tok, messages(r, r.get("constraints"), with_target=False)) for r in rooms]
        texts = generate(a.model, prompts, a.max_new_tokens, a.batch)
    rows = [score(r, t) for r, t in zip(rooms, texts)]
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/predictions.jsonl", "w") as f:
        for r, t in zip(rows, texts):
            f.write(json.dumps({**r, "output": t}, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    json.dump(summary, open(f"{a.out}/metrics.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("in_dist", "held_out")}, indent=1))


if __name__ == "__main__":
    main()
