"""IR rooms -> filtered, deduplicated, leakage-safe SFT files (spec: FastFill I/O design review, 2026-09-24).

    python -m fastfill.build --ir /Volumes/harddisk/fastfill_ir --out data/fastfill_v1 [--sources ...] [--constraint_frac 0.3]

Per room: prep (reject or clean, see `prep`) -> cross-source dedup by group -> split by group/content
-> train: rot90 augmentation -> canonical order/ids -> optional constraints (S2) -> field dropout -> messages
   dev/test: canonical -> messages (no augmentation), IR written to {dev,test}_rooms.jsonl for fastfill.evaluate,
   test_constrained_rooms.jsonl adds 1-4 GT-true constraints per test room (rooms without any are skipped).
Eval-only sources (meta.eval_only) go to test with their whole split component (split.assign_splits).
stats.json counts every rejection reason, dropped object and inferred vs evidence support per source.
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import re

from shapely.geometry import Point, Polygon

from fastfill.anchors import FLAT, FLOOR_Z, is_structure, scope_objects
from fastfill.scene import canonical, clean_boundary, footprint, head, messages, norm_cat, num, rot90, shown_size
from fastfill.split import assign_splits
from fastfill.validate import check, holds

GENERIC = re.compile(r"^(other\w*|objects?|unknown)( |$)")
FLOOR_STRUCT = re.compile(r"\b(column|pillar|stair|staircase)s?\b")
# dropped boxes that do not block the floor: openings and wall/floor finishes (plant_floor -> plant is not exempt)
FLUSH = re.compile(r"^(doors?|doorframes?|windows?|curtains?|drapes?|blinds?|frames?|openings?|baseboards?|rails?|"
                   r"railings?|radiators?|beams?|shades?|screening|designs?|molds?|moldings?|panels?|paneling|combination|"
                   r"walls?|floors?|ceilings?|skirting|trim|tiles?|boards?)$")


WALL_BAND_M = 0.10       # a flagged structure box within this band of the boundary keeps to the walls
INTRUSION_M2 = 0.05      # ... unless more than this much of it stands further inside the room


def _words(category):
    return re.sub(r"[^a-z]+", " ", (category or "").lower()).split()


def _flush(o, inner):
    """A dropped floor box that leaves no hole: a flat covering, or a structural opening/finish ('bed frame' is not).
    Source-flagged architecture (stairs/pillars were rejected before, floor_structure) is flush when its name says
    wall/floor/door/window/railing..., or when it keeps to the walls: under INTRUSION_M2 inside `inner` (the room
    shrunk by WALL_BAND_M). A windowsill block or a fence standing in the room is an obstacle whatever its label."""
    if head(o) in FLAT:
        return True
    if o.get("structure"):
        w = _words(o["category"])
        return (any(FLUSH.match(x) for x in w) and not {"partition", "screen", "clothes"} & set(w)) or \
            Polygon(footprint(o)).intersection(inner).area < INTRUSION_M2
    w = _words(o["category"])
    while len(w) > 1 and w[-1] in ("floor", "wall", "ceiling"):   # placement suffix: plant_floor, bookshelf_wall
        w.pop()
    return bool(w) and is_structure(o["category"]) and not {"partition", "screen", "clothes"} & set(w) \
        and bool(FLUSH.match(w[-1]))
FRONT = {"chair", "armchair", "seat", "sofa", "couch", "loveseat", "bed", "desk", "tv", "television", "monitor", "toilet",
         "piano"}
PRIORITY = ["on", "between", "faces", "against_wall", "near"]   # availability in rooms, rarest first


def _drop_with_dependants(objs, bad):
    bad = set(bad)
    while True:
        more = {o["id"] for o in objs if o.get("parent") in bad} - bad
        if not more:
            return [o for o in objs if o["id"] not in bad]
        bad |= more


def prep(room, a):
    """IR room -> (clean room, None) or (None, reject reason)."""
    meta = room.get("meta", {})
    if meta.get("front_known") is not True:
        return None, "front_unknown"
    if meta.get("n_incomplete"):                     # objects without a box would be invisible obstacles
        return None, "incomplete_objects"
    if len({o["id"] for o in room["objects"]}) != len(room["objects"]):
        return None, "duplicate_source_ids"
    if room.get("boundary_type") not in a.boundary_types or not room.get("boundary"):
        return None, "boundary"
    if any(o["pos"][2] <= FLOOR_Z and FLOOR_STRUCT.search(" ".join(_words(o["category"])))
           and (o.get("structure") or is_structure(o["category"])) for o in room["objects"]):
        return None, "floor_structure"
    r = scope_objects(room, keep=a.source_anchors.get(room["source"], a.anchors))
    objs = r["objects"]
    if any(o.get("front_known") is False for o in objs):
        return None, "front_unreliable_object"
    for o in objs:
        vals = list(o["size"]) + list(o["pos"]) + [o["yaw"]]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals) or min(o["size"]) <= 0:
            return None, "bad_numbers"
    # generic labels, tilted boxes and floating/sunk floor boxes are dropped with everything they support
    objs = _drop_with_dependants(objs, [o["id"] for o in objs if GENERIC.match(norm_cat(o["category"]))
                                        or not norm_cat(o["category"]) or o.get("tilted")
                                        or (not o.get("parent") and abs(o["pos"][2]) > FLOOR_Z)])
    byid = {o["id"]: o for o in objs}                # keep only top-surface support
    objs = _drop_with_dependants(objs, [
        o["id"] for o in objs if o.get("parent") and (
            (p := byid.get(o["parent"])) is None or abs(o["pos"][2] - p["pos"][2] - p["size"][2]) > 0.05
            or not Polygon(footprint(p)).buffer(0.05).contains(Point(o["pos"][:2])))])
    # every dropped box that stood on the floor (at any stage: structure, wall/ceiling, generic, tilted, sunk)
    # leaves an unexplained hole in the reference layout unless it is an opening or a flush finish
    kept = {o["id"] for o in objs}
    children = {o["id"] for o in r["objects"] if o.get("parent")}
    shell = Polygon(room["boundary"]).buffer(0)        # area counted is the part of the box on this room's floor
    inner = shell.buffer(-WALL_BAND_M)
    hidden = sum(Polygon(footprint(o)).intersection(shell).area for o in room["objects"]
                 if o["id"] not in kept and o["id"] not in children and not o.get("parent")
                 and o["pos"][2] <= FLOOR_Z and o["size"][2] > 0.05 and not _flush(o, inner))
    if hidden >= a.hidden_max:
        return None, "hidden_floor_obstacle"
    if not a.min_objects <= len(objs) <= a.max_objects:
        return None, "object_count"
    # snap to the DISPLAYED numbers (cm, integer degrees) so checks, constraints and eval see what the model sees;
    # z: floor 0, child = parent top
    objs = [{**o, "size": shown_size(o), "yaw": math.radians(round(math.degrees(o["yaw"])) % 360)} for o in objs]
    byid = {o["id"]: o for o in objs}
    done = set()

    def z_of(o):
        if o["id"] not in done:
            p = byid.get(o.get("parent"))
            o["pos"] = [num(o["pos"][0]), num(o["pos"][1]), num(z_of(p) + p["size"][2]) if p else 0]
            done.add(o["id"])
        return o["pos"][2]
    for o in objs:
        z_of(o)
    b = clean_boundary(room["boundary"])
    if b is None or len(b) > a.max_vertices or Polygon(b).area < 1.0:
        return None, "boundary_shape"
    out = {**r, "boundary": b, "objects": objs}
    if check(out, oob_tol=a.oob_tol)["oob"]:
        return None, "gt_out_of_bounds"
    area = Polygon(b).area
    if any(not o.get("parent") and head(o) not in FLAT and o["size"][2] >= 0.5
           and o["size"][0] * o["size"][1] >= 0.8 * area for o in objs):   # composite built-in boxing the room
        return None, "room_filling_object"
    h = out.get("height")
    if h and any(o["pos"][2] + o["size"][2] > h + 0.05 for o in objs):   # height contradicts the boxes: unknown
        out = {**out, "height": None, "meta": {**out.get("meta", {}), "height_dropped": True}}
    return out, None


def dedup(rooms):
    """Same group in several sources (e.g. a ScanNet scan in Scan2CAD and InternScenes): keep the source whose
    rooms of that group keep more objects; ties go to a polygon boundary, then source name."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rooms:
        by[r["group"]][r["source"]].append(r)
    keep, dropped = [], collections.Counter()
    for g, srcs in by.items():
        best = max(srcs, key=lambda s: (sum(len(r["objects"]) for r in srcs[s]),
                                        any(r["boundary_type"] == "polygon" for r in srcs[s]), s))
        keep += srcs[best]
        for s, rs in srcs.items():
            if s != best:
                dropped[s] += len(rs)
    return keep, dropped


def rng_for(uid, salt):
    return random.Random(int(hashlib.sha1(f"{salt}:{uid}".encode()).hexdigest(), 16))


def extract_constraints(room, rng, max_k=4):
    """1..max_k relations that hold on the reference layout, one per type first (round robin, rarest type first).
    Only a few relations are given, never the full relation graph: the model must still lay out everything else."""
    b = room["boundary"]
    objs = room["objects"]
    fps = {o["id"]: Polygon(footprint(o)) for o in objs}
    cand = collections.defaultdict(list)
    objs = [o for o in objs if head(o) not in FLAT]  # rugs are not meaningful constraint arguments
    for o in objs:
        if o.get("parent"):
            cand["on"].append(["on", o["id"], o["parent"]])
        if room["boundary_type"] == "polygon" and not o.get("parent") and holds(["against_wall", o["id"]], objs, b):
            cand["against_wall"].append(["against_wall", o["id"]])
    for x in objs:
        for y in objs:
            if x is y or x.get("parent") or y.get("parent"):
                continue
            if head(x) in FRONT and not fps[x["id"]].intersects(fps[y["id"]]) and \
                    holds(["faces", x["id"], y["id"]], objs, b):
                cand["faces"].append(["faces", x["id"], y["id"]])
            if x["id"] < y["id"]:
                gap = fps[x["id"]].distance(fps[y["id"]])
                d = next((d for d in (0.1, 0.2, 0.3, 0.5) if d >= gap), None)
                if d is not None and holds(["near", x["id"], y["id"], d], objs, b):   # the one checker decides
                    cand["near"].append(["near", x["id"], y["id"], d])
    floor = [o for o in objs if not o.get("parent")]
    for i, p in enumerate(floor):
        for q in floor[i + 1:]:
            if norm_cat(p["category"]) != norm_cat(q["category"]):
                continue
            for m in floor:
                if m is p or m is q or fps[m["id"]].intersects(Point(p["pos"][:2])) or \
                        fps[m["id"]].intersects(Point(q["pos"][:2])):   # endpoint inside/on m: trivially true
                    continue
                if holds(["between", m["id"], p["id"], q["id"]], objs, b):
                    cand["between"].append(["between", m["id"], p["id"], q["id"]])
    types = [t for t in PRIORITY if cand[t]]         # rarest types first, so all five are covered
    for t in types:
        rng.shuffle(cand[t])
    k, out = rng.randint(1, max_k), []
    while len(out) < k and any(cand[t] for t in types):
        for t in types:
            if cand[t] and len(out) < k:
                out.append(cand[t].pop())
    return out


def count_constraints(st, split, room, cons):
    """stats: constraints per type; 'on' split by evidence (support given by the source vs box-contact inference)."""
    objs = {o["id"]: o for o in room["objects"]}
    for c in cons or []:
        t = c[0]
        if t == "on":
            t += "(inferred)" if objs[c[1]].get("anchor_inferred") else "(source)"
        st[f"{split}_constraints:{t}"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="dir of <source>.jsonl from fastfill.adapters")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="*", default=None, help="default: every file in --ir")
    ap.add_argument("--min_objects", type=int, default=2)
    ap.add_argument("--max_objects", type=int, default=40)
    ap.add_argument("--max_vertices", type=int, default=24)
    ap.add_argument("--boundary_types", nargs="*", default=["polygon", "hull"])
    ap.add_argument("--anchors", nargs="*", default=["floor", "object"], help="object anchors kept (see anchors.py)")
    ap.add_argument("--oob_tol", type=float, default=0.10, help="GT rooms with objects outside boundary+tol are rejected")
    ap.add_argument("--hidden_max", type=float, default=0.3, help="max m2 of dropped floor objects")
    ap.add_argument("--cap", nargs="*", default=[], help="SOURCE=N: keep N train rooms of SOURCE (hash-sampled)")
    ap.add_argument("--source_anchors", nargs="*", default=[], help="SOURCE=a,b: anchors kept for that source")
    ap.add_argument("--dev", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--constraint_frac", type=float, default=0.0, help="share of train rooms given as S2 (v1.1: 0.3)")
    ap.add_argument("--no_rot90", action="store_true")
    ap.add_argument("--field_dropout", type=float, default=0.2, help="train: drop room_type / height each with p")
    ap.add_argument("--desc_dropout", type=float, default=0.5, help="train: drop all desc of a room with p")
    a = ap.parse_args()
    a.source_anchors = {k: v.split(",") for k, v in (x.split("=") for x in a.source_anchors)}
    caps = {k: int(v) for k, v in (x.split("=") for x in a.cap)}

    files = sorted(glob.glob(f"{a.ir}/*.jsonl"))
    if a.sources:
        files = [f for f in files if os.path.basename(f)[:-6] in a.sources]
    rooms, stats = [], collections.defaultdict(collections.Counter)
    for f in files:
        for line in open(f):
            r0 = json.loads(line)
            st = stats[r0["source"]]
            r, why = prep(r0, a)
            st["scanned"] += 1
            if why:
                st["rejected:" + why] += 1
                continue
            st["objects_dropped"] += len(r0["objects"]) - len(r["objects"])
            st["supports_inferred"] += sum(bool(o.get("anchor_inferred")) for o in r["objects"] if o.get("parent"))
            st["supports_evidence"] += sum(not o.get("anchor_inferred") for o in r["objects"] if o.get("parent"))
            rooms.append(r)
    rooms, dropped = dedup(rooms)
    for s, n in dropped.items():
        stats[s]["dedup_dropped"] += n

    splits = assign_splits(rooms, a.dev, a.test)
    for src, n in caps.items():                    # hash order, not file order (files are sorted by type)
        idx = sorted((i for i, r in enumerate(rooms) if r["source"] == src and splits[i] == "train"),
                     key=lambda i: hashlib.sha1(rooms[i]["uid"].encode()).hexdigest())
        for i in idx[n:]:
            splits[i] = "capped"
    os.makedirs(a.out, exist_ok=True)
    outs = {s: open(f"{a.out}/{s}.jsonl", "w") for s in ("train", "dev", "test")}
    ir_outs = {s: open(f"{a.out}/{s}.jsonl", "w")
               for s in ("dev_rooms", "test_rooms", "dev_constrained_rooms", "test_constrained_rooms")}
    for r, s in zip(rooms, splits):
        st = stats[r["source"]]
        st["height_dropped"] += bool(r.get("meta", {}).get("height_dropped"))
        if s == "capped":
            st["capped"] += 1
            continue
        if s == "train":
            # one random stream per purpose: --constraint_frac changes nothing but the constraints (clean ablation:
            # v1.1 minus its constraints is v1.0 byte for byte)
            aug, s2, drop = (rng_for(r["uid"], k) for k in ("aug", "s2", "dropout"))
            c = canonical(r if a.no_rot90 else rot90(r, aug.randrange(4)))
            cons = extract_constraints(c, s2) if s2.random() < a.constraint_frac else None
            st["train_S2" if cons else "train_S1"] += 1
            count_constraints(st, "train", c, cons)
            if drop.random() < a.field_dropout:
                c = {**c, "room_type": None}
            if drop.random() < a.field_dropout:
                c = {**c, "height": None}
            m = messages(c, cons, with_desc=drop.random() >= a.desc_dropout)
        else:
            c = canonical(r)
            st[s] += 1
            m = messages(c)
            ir_outs[f"{s}_rooms"].write(json.dumps(c, ensure_ascii=False) + "\n")
            cons = extract_constraints(c, rng_for(r["uid"], "eval"))
            if cons:
                count_constraints(st, s, c, cons)
                ir_outs[f"{s}_constrained_rooms"].write(json.dumps({**c, "constraints": cons}, ensure_ascii=False) + "\n")
        outs[s].write(json.dumps({"uid": r["uid"], "source": r["source"], "messages": m}, ensure_ascii=False) + "\n")
    for fh in list(outs.values()) + list(ir_outs.values()):
        fh.close()
    json.dump({k: dict(v) for k, v in sorted(stats.items())}, open(f"{a.out}/stats.json", "w"), indent=1)
    # manifest: what was built, from which code, with which arguments (sha256 of every output and code file)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump({"args": vars(a),
               "files": {f: {"sha256": sha(f"{a.out}/{f}"), "lines": sum(1 for _ in open(f"{a.out}/{f}"))}
                         for f in sorted(os.listdir(a.out)) if f.endswith((".jsonl", "stats.json"))},
               "code_sha256": {os.path.relpath(p, here): sha(p)
                               for p in sorted(glob.glob(f"{here}/**/*.py", recursive=True))}},
              open(f"{a.out}/MANIFEST.json", "w"), indent=1)
    for k, v in sorted(stats.items()):
        print(f"{k:24s} {dict(v)}")


if __name__ == "__main__":
    main()
