"""Everything asked for before syncing to the server, for one or more built dirs.
usage: qa_report.py IR_DIR BUILD_DIR [BUILD_DIR ...]  -> prints a summary, writes BUILD_DIR/QA.json"""
import collections, copy, json, os, sys
from shapely.geometry import Polygon
from fastfill.evaluate import score, summarize, inside
from fastfill.interface import snap_inside
from fastfill.scene import apply, parse
from fastfill.split import content_key, layout_key
from fastfill.validate import holds

IR, BUILDS = sys.argv[1], sys.argv[2:]
EVAL_ONLY = {"SpatialGen", "SceneSmith"}
ir_group, ir_alias = {}, {}
for f in os.listdir(IR):
    if f.endswith(".jsonl"):
        for line in open(f"{IR}/{f}"):
            r = json.loads(line)
            ir_group[r["uid"]] = r["group"]
            ir_alias[r["uid"]] = r.get("meta", {}).get("group_aliases") or []


def text_room(row):
    u = json.loads(row["messages"][1]["content"])
    pl, err = parse(row["messages"][2]["content"], [o["id"] for o in u["objects"]])
    return u, pl, err


for D in BUILDS:
    q = {"build": D}
    worlds, rooms, keys = (collections.defaultdict(set) for _ in range(3))
    cons_n = collections.defaultdict(collections.Counter)            # split -> type -> n
    cons_ok = collections.Counter(); cons_all = collections.Counter()
    closure = collections.Counter(); inside_train = collections.Counter()
    rows = collections.Counter(); s2 = 0; per_src = collections.Counter(); comp_links = collections.defaultdict(list)
    for line in open(f"{D}/train.jsonl"):
        row = json.loads(line); rows["train"] += 1; per_src[row["source"]] += 1
        uid = row["uid"]
        rooms["train"].add(uid); worlds["train"].add(ir_group.get(uid, "?" + uid))
        keys["train_alias"].update(ir_alias.get(uid, []))
        u, pl, err = text_room(row)
        closure["train_ok" if err is None else "train_" + err] += 1
        if err:
            continue
        objs = [{"id": o["id"], "size": o["size"], "pos": pl[o["id"]]["pos"], "yaw": pl[o["id"]]["yaw"],
                 "parent": pl[o["id"]]["on"]} for o in u["objects"]]
        room = {"source": row["source"], "boundary": u["boundary"], "boundary_type": u["boundary_type"], "objects": objs}
        ck, lk = content_key(room), layout_key(room)
        keys["train_c"].add(ck); keys["train_l"].add(lk)
        comp_links["train"].append(("u:" + uid, ["g:" + ir_group.get(uid, "?"), f"c:{ck}", f"l:{lk}"] + ["a:" + x for x in ir_alias.get(uid, [])]))
        floor = Polygon(u["boundary"]); rep = copy.deepcopy(pl); snap_inside(rep, room, floor)
        inside_train["raw"] += inside(room, floor); inside_train["repaired"] += inside(apply(room, rep), floor)
        if u.get("constraints"):
            s2 += 1
            for c in u["constraints"]:
                cons_n["train"][c[0]] += 1; cons_all["train"] += 1; cons_ok["train"] += holds(c, objs, u["boundary"])
    for s in ("dev", "test"):
        rows[s] = sum(1 for _ in open(f"{D}/{s}.jsonl"))
        for line in open(f"{D}/{s}_rooms.jsonl"):
            r = json.loads(line)
            rooms[s].add(r["uid"]); worlds[s].add(r["group"])
            keys[f"{s}_alias"].update(r.get("meta", {}).get("group_aliases") or [])
            ck, lk = content_key(r), layout_key(r)
            keys[f"{s}_c"].add(ck); keys[f"{s}_l"].add(lk)
            comp_links[s].append(("u:" + r["uid"], ["g:" + r["group"], f"c:{ck}", f"l:{lk}"]
                                  + ["a:" + x for x in r.get("meta", {}).get("group_aliases") or []]))
        for line in open(f"{D}/{s}_constrained_rooms.jsonl"):
            r = json.loads(line)
            for c in r["constraints"]:
                cons_n[s][c[0]] += 1; cons_all[s] += 1; cons_ok[s] += holds(c, r["objects"], r["boundary"])
    # world count the reviewer asked for: connected components over group / alias / furniture key / layout key
    def components(split):
        parent = {}
        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for node, links in comp_links[split]:
            for l in links:
                if l and not l.endswith("None"):
                    parent[find(node)] = find(l)
        return len({find(node) for node, _ in comp_links[split]})
    leak = {}
    for s in ("dev", "test"):
        leak[f"train&{s} worlds"] = len(worlds["train"] & worlds[s])
        leak[f"train&{s} aliases"] = len(keys["train_alias"] & keys[f"{s}_alias"])
        leak[f"train&{s} furniture keys"] = len((keys["train_c"] & keys[f"{s}_c"]) - {None})
        leak[f"train&{s} layout keys"] = len((keys["train_l"] & keys[f"{s}_l"]) - {None})
    leak["held-out sources in train (rows)"] = sum(per_src[s] for s in EVAL_ONLY)
    q["rows"] = dict(rows); q["train_rows_with_constraints"] = s2
    q["distinct groups (house/scan)"] = {s: len(v) for s, v in worlds.items()}
    q["linked components (group+alias+furniture+layout)"] = {s: components(s) for s in ("train", "dev", "test")}; q["rooms"] = {s: len(v) for s, v in rooms.items()}
    q["constraints_by_type"] = {s: dict(sorted(v.items())) for s, v in cons_n.items()}
    q["constraint_reference_validation"] = {s: f"{cons_ok[s]}/{cons_all[s]}" for s in cons_all}
    q["object_set_closure"] = dict(closure)
    q["leakage (all must be 0)"] = leak
    q["train reference: all objects inside walls within 1 mm"] = {k: f"{v}/{rows['train']}" for k, v in inside_train.items()}
    gt = {}
    for s in ("dev", "test"):
        summ = summarize([score(r, None) for r in map(json.loads, open(f"{D}/{s}_rooms.jsonl"))])
        for part in ("in_dist", "held_out"):
            p = summ.get(part)
            if p:
                gt[f"{s}/{part}"] = {k: round(p[k], 4) for k in ("rooms", "complete_rate", "valid_rate", "valid_rate_repaired",
                                                                   "inside_1mm_rate", "inside_1mm_rate_repaired")}
    q["reference legality raw vs repaired"] = gt
    st = json.load(open(f"{D}/stats.json"))
    q["rejections_by_source"] = {src: {k[9:]: v for k, v in sorted(d.items()) if k.startswith("rejected:")}
                                 | {"scanned": d.get("scanned", 0), "kept_train": d.get("train_S1", 0) + d.get("train_S2", 0),
                                    "dev": d.get("dev", 0), "test": d.get("test", 0)} for src, d in st.items()}
    man = json.load(open(f"{D}/MANIFEST.json"))
    q["manifest_sha256"] = {f: v["sha256"][:16] for f, v in man["files"].items()}
    json.dump(q, open(f"{D}/QA.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in q.items() if k not in ("rejections_by_source", "manifest_sha256")},
                     indent=1, ensure_ascii=False))
