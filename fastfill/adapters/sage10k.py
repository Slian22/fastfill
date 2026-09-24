"""SAGE-10k (agentic generation, TRELLIS assets): 9,987 single-room scenes + 13 multi-room scenes (40 rooms).

Aggregate json_single.json / json_multi_split.json (aggregate_scenes.py): meters, Z-up, room_boundary = ordered wall
centreline endpoints (all rooms are 4-corner rectangles), position = bottom center (floor objects z = 0, equals the
cm footprint center of placement_constraints), rotation = Euler deg applied as Rz Ry Rx (kits/tex_utils_local.py),
dimensions = local X/Y/Z extents. Front = local +Y (front test), so front_offset_deg = 90.
Support: the aggregate drops place_id; it is read from the per-scene layout_*.json when present
(floor | wall | <instance id>), otherwise anchors are left to fastfill.anchors.
Tilt: floor/wall objects are exactly upright; objects resting on objects are physics-settled with x/y tilt.
Tilt <= TILT_DEG is treated as settling noise (zeroed); larger tilt is kept and flagged.
"""
import collections
import json
import math
import os

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "SAGE-10k"
TILT_DEG = 2.0


def tilt_deg(rot):
    return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(math.radians(rot["x"])) * math.cos(math.radians(rot["y"]))))))


def load(root):
    base = os.path.join(root, "liantian__3D_Room_Scene_Collections/SAGE10k")
    sdir = os.path.join(base, "sage10k/scenes")
    layouts = {d.split("_", 2)[-1]: os.path.join(sdir, d, d.split("_", 2)[-1] + ".json")
               for d in (os.listdir(sdir) if os.path.isdir(sdir) else [])}
    for fn in ("json_single.json", "json_multi_split.json"):
        for rec in iter_records(os.path.join(base, fn)):
            sid, k = rec["scene_id"], rec["room_index"]
            fids = [f["furniture_instance_id"] for f in rec["furniture"]]
            count = collections.Counter(fids)
            place = None
            if sid in layouts and os.path.exists(layouts[sid]):
                with open(layouts[sid]) as fh:
                    objs = json.load(fh)["rooms"][k]["objects"]
                if [o["id"] for o in objs] == fids:
                    place = [o.get("place_id") for o in objs]
            n_noise = 0
            furn, seen = [], collections.Counter()
            for j, f in enumerate(rec["furniture"]):
                p = place[j] if place else None
                if p in ("floor", "wall"):
                    sup = {"anchor": p}
                elif count.get(p) == 1:
                    sup = {"anchor": "object", "parent": p}
                else:
                    sup = {}               # no evidence, dangling or repeated parent id: inferred downstream
                f = {**f, "_support": sup}
                seen[fids[j]] += 1
                if seen[fids[j]] > 1:      # raw ids repeat for copied small objects (663 rooms): keep ids unique
                    f["furniture_instance_id"] = f"{fids[j]}#{seen[fids[j]]}"
                r = f.get("furniture_rotation")
                if r and (r["x"] or r["y"]) and tilt_deg(r) <= TILT_DEG:
                    f["furniture_rotation"] = {"x": 0, "y": 0, "z": r["z"]}
                    n_noise += 1
                furn.append(f)

            yield convert_room(
                {**rec, "furniture": furn, "room_boundary": [[q["x"], q["y"]] for q in rec.get("room_boundary") or []]},
                source=SOURCE, uid=f"SAGE-10k::{sid}::{k}", group=f"sage:{sid}",
                subset="multi" if fn == "json_multi_split.json" else "single", boundary_type="polygon",
                center_z=False, front_offset_deg=90, extra=lambda f: f["_support"],
                meta={"scene_id": sid, "room_index": k, "front_known": True, "tilt_threshold_deg": TILT_DEG,
                      "n_tilt_noise_zeroed": n_noise, "anchor_source": "place_id" if place is not None else "inferred",
                      "building_style": rec.get("building_style")})
