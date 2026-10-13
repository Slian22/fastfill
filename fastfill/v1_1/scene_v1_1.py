"""FastFill v1.1 protocol, serialization, canonicalization, and parser.

Differences from FastFill v1 (fastfill/scene.py):
1. `user["objects"]` does NOT include `"size": [sx, sy, sz]`. Each requested object has `{"id": id, "desc"?: desc}`.
2. `user["fixed"]` (doors, windows, columns, stairs) STILL includes `"size": [sx, sy, sz]`, `"pos": [x, y, z]`, `"yaw": deg`.
3. `assistant["placements"]` outputs `"size": [sx, sy, sz]` alongside `"id"`, optional `"on"`, `"pos": [x, y, z]`, `"yaw": deg`:
   `{"placements": [{"id": id, "on"?: parent_id, "size": [sx, sy, sz], "pos": [x, y, z], "yaw": deg}, ...]}`.
4. At inference time (when ground-truth `size` is unknown), `canonical_v1_1` orders objects by:
   floor objects before supported (`on`) objects, then category prior footprint area descending, then category slug.
"""
from __future__ import annotations

import json
import math
from typing import Any

from fastfill.scene import (
    MAX_COORD,
    _isnum,
    clean_boundary,
    footprint,
    head,
    norm_cat,
    norm_room_type,
    num,
    rot90,
    short_desc,
    slug,
)

SYSTEM_PROMPT_V1_1 = """You are a room layout model. Choose a realistic 3D bounding box size and place every listed object in the room without collisions; do not add or drop objects.
Frame: meters, right-handed, Z up, floor z=0. boundary: floor polygon [[x,y],...], counter-clockwise; boundary_type "polygon" = walls, "hull" = convex hull of a scanned floor (the real floor may be smaller). height: ceiling height. A missing key means unknown.
fixed: boxes already in the room that cannot move (doors, windows, columns, stairs, ...), each with id, size, pos and yaw as below; keep placements clear of them and do not block doors; they are not part of the answer.
Object: id (category_index), optional desc.
Output only JSON {"placements":[{"id":id,"on":id,"size":[sx,sy,sz],"pos":[x,y,z],"yaw":deg},...]}, every id once, in the listed order. on: the object it stands on, z = that object's top; omit on for the floor, z = 0. size: [sx,sy,sz] in meters along its local X,Y,Z (local +X = its front). pos: bottom center of the box. yaw: integer degrees in [0,360), counter-clockwise from room +X to the object's front.
constraints (optional, all must hold): ["on",a,b]; ["faces",a,b] a's front within 30 degrees of the direction to b's center; ["near",a,b,d] footprint gap at most d; ["against_wall",a] one whole side of a's footprint within 0.1 of a wall; ["between",a,b,c] the segment between the centers of b and c crosses a."""

# Category prior footprint areas (m^2) for deterministic large-to-small ordering at inference when sizes are unknown
CATEGORY_PRIOR_AREA: dict[str, float] = {
    "bed": 3.60, "king_size_bed": 4.20, "double_bed": 3.50, "single_bed": 2.00, "bunk_bed": 2.40,
    "sofa": 2.20, "l_shaped_sofa": 3.80, "three_seat_sofa": 2.40, "two_seat_sofa": 1.60, "loveseat": 1.50,
    "dining_table": 1.80, "conference_table": 3.20, "pool_table": 3.50, "ping_pong_table": 4.10,
    "wardrobe": 1.20, "closet": 1.30, "cabinet": 0.80, "kitchen_cabinet": 1.20, "cupboard": 0.90,
    "counter": 1.50, "kitchen_island": 1.80, "bar": 1.40, "reception_desk": 1.60,
    "desk": 0.90, "table": 0.85, "coffee_table": 0.75, "dressing_table": 0.65, "vanity": 0.70,
    "tv_stand": 0.75, "media_console": 0.75, "bookshelf": 0.60, "bookcase": 0.65, "shelf": 0.45,
    "bathtub": 1.20, "shower": 1.00, "jacuzzi": 1.80, "refrigerator": 0.65, "fridge": 0.65,
    "washing_machine": 0.42, "dryer": 0.42, "dishwasher": 0.42, "stove": 0.45, "oven": 0.45,
    "treadmill": 1.50, "gym_equipment": 1.40, "exercise_bike": 0.65, "piano": 1.10,
    "armchair": 0.65, "lounge_chair": 0.65, "recliner": 0.75, "office_chair": 0.40, "chair": 0.32,
    "bench": 0.55, "ottoman": 0.35, "stool": 0.18, "bar_stool": 0.18,
    "nightstand": 0.22, "side_table": 0.24, "end_table": 0.24, "toilet": 0.30, "sink": 0.30,
    "washbasin": 0.30, "urinal": 0.18, "floor_lamp": 0.14, "plant": 0.16, "trash_can": 0.10,
    "tv": 0.18, "monitor": 0.12, "laptop": 0.08, "microwave": 0.18, "lamp": 0.07,
}

# Category prior [sx, sy, sz] (m) used only as a fallback if a model ever omits/corrupts size
CATEGORY_PRIOR_SIZE: dict[str, list[float]] = {
    "bed": [2.05, 1.65, 0.95], "king_size_bed": [2.10, 1.90, 1.05], "single_bed": [2.00, 1.05, 0.90],
    "sofa": [0.90, 2.00, 0.85], "armchair": [0.80, 0.80, 0.85], "chair": [0.55, 0.52, 0.85],
    "office_chair": [0.60, 0.60, 0.95], "dining_table": [0.95, 1.60, 0.76], "table": [0.80, 1.20, 0.75],
    "coffee_table": [0.65, 1.15, 0.45], "desk": [0.65, 1.30, 0.75], "nightstand": [0.45, 0.48, 0.55],
    "side_table": [0.48, 0.48, 0.55], "wardrobe": [0.60, 1.60, 2.10], "cabinet": [0.50, 1.00, 0.85],
    "bookshelf": [0.38, 0.95, 1.85], "tv_stand": [0.45, 1.60, 0.50], "bathtub": [0.80, 1.65, 0.60],
    "shower": [0.95, 0.95, 2.05], "toilet": [0.65, 0.42, 0.78], "sink": [0.50, 0.60, 0.85],
    "refrigerator": [0.70, 0.75, 1.78], "washing_machine": [0.62, 0.60, 0.85], "stove": [0.62, 0.65, 0.88],
    "bench": [0.45, 1.20, 0.48], "stool": [0.38, 0.38, 0.46], "ottoman": [0.55, 0.55, 0.42],
    "floor_lamp": [0.35, 0.35, 1.55], "lamp": [0.25, 0.25, 0.42], "plant": [0.38, 0.38, 0.75],
    "trash_can": [0.28, 0.28, 0.40], "tv": [0.15, 1.15, 0.70], "monitor": [0.18, 0.55, 0.42],
}


def prior_area(cat: str) -> float:
    s = slug(cat)
    if s in CATEGORY_PRIOR_AREA:
        return CATEGORY_PRIOR_AREA[s]
    h = slug(head({"category": cat}))
    return CATEGORY_PRIOR_AREA.get(h, 0.25)


def prior_size(cat: str) -> list[float]:
    s = slug(cat)
    if s in CATEGORY_PRIOR_SIZE:
        return list(CATEGORY_PRIOR_SIZE[s])
    h = slug(head({"category": cat}))
    return list(CATEGORY_PRIOR_SIZE.get(h, [0.50, 0.50, 0.60]))


def canonical_v1_1(room: dict[str, Any]) -> dict[str, Any]:
    """Order objects deterministically when ground-truth sizes are unknown at inference time:
    floor/wall objects before supported (`on`/`parent`) objects, then category prior area descending, then slug."""
    objs = list(room.get("objects") or [])
    indexed = list(enumerate(objs))
    indexed.sort(
        key=lambda pair: (
            0 if not pair[1].get("parent") else 1,
            -prior_area(pair[1].get("category") or pair[1]["id"].rsplit("_", 1)[0].replace("_", " ")),
            slug(pair[1].get("category") or pair[1]["id"].rsplit("_", 1)[0].replace("_", " ")),
            pair[0],
        )
    )
    return {**room, "objects": [o for _, o in indexed]}


def user_json_v1_1(room: dict[str, Any], constraints: list[Any] | None = None, with_desc: bool = True) -> str:
    """Serialize `user` JSON for FastFill v1.1 (no `size` in `objects`, `fixed` keeps `size`)."""
    d: dict[str, Any] = {}
    rt = norm_room_type(room.get("room_type"))
    if rt:
        d["room_type"] = rt
    d["boundary_type"] = room["boundary_type"]
    d["boundary"] = [[num(x), num(y)] for x, y in room["boundary"]]
    if room.get("height"):
        d["height"] = num(room["height"])
    if room.get("fixed"):
        d["fixed"] = [
            {
                "id": o["id"],
                "size": [max(num(v), 0.01) for v in o["size"]],
                "pos": [num(v) for v in o["pos"]],
                "yaw": round(math.degrees(o["yaw"])) % 360,
            }
            for o in room["fixed"]
        ]
    objs = []
    for o in room["objects"]:
        item: dict[str, Any] = {"id": o["id"]}
        if with_desc and o.get("desc"):
            sd = short_desc(o["desc"])
            if sd:
                item["desc"] = sd
        objs.append(item)
    d["objects"] = objs
    if constraints:
        d["constraints"] = constraints
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def target_json_v1_1(room: dict[str, Any]) -> str:
    """Serialize `assistant` JSON for FastFill v1.1 (`placements` includes `size`: `[sx, sy, sz]`)."""
    pl = [
        {
            "id": o["id"],
            **({"on": o["parent"]} if o.get("parent") else {}),
            "size": [max(num(v), 0.01) for v in o["size"]],
            "pos": [num(v) for v in o["pos"]],
            "yaw": round(math.degrees(o["yaw"])) % 360,
        }
        for o in room["objects"]
    ]
    return json.dumps({"placements": pl}, separators=(",", ":"), ensure_ascii=False)


def messages_v1_1(
    room: dict[str, Any],
    constraints: list[Any] | None = None,
    with_desc: bool = True,
    with_target: bool = True,
) -> list[dict[str, str]]:
    m = [
        {"role": "system", "content": SYSTEM_PROMPT_V1_1},
        {"role": "user", "content": user_json_v1_1(room, constraints, with_desc)},
    ]
    if with_target:
        m.append({"role": "assistant", "content": target_json_v1_1(room)})
    return m


def convert_v1_messages_to_v1_1(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deterministically convert a FastFill v1 [system, user, assistant] message triple to FastFill v1.1."""
    user_d = json.loads(messages[1]["content"])
    size_by_id: dict[str, list[float]] = {}
    new_objs = []
    for o in user_d["objects"]:
        size_by_id[o["id"]] = o["size"]
        item = {"id": o["id"]}
        if "desc" in o:
            item["desc"] = o["desc"]
        new_objs.append(item)
    user_d["objects"] = new_objs
    out = [
        {"role": "system", "content": SYSTEM_PROMPT_V1_1},
        {"role": "user", "content": json.dumps(user_d, separators=(",", ":"), ensure_ascii=False)},
    ]
    if len(messages) >= 3 and messages[2].get("content"):
        asst_d = json.loads(messages[2]["content"])
        new_pl = []
        for p in asst_d["placements"]:
            pid = p["id"]
            new_p: dict[str, Any] = {"id": pid}
            if "on" in p and p["on"] is not None:
                new_p["on"] = p["on"]
            new_p["size"] = size_by_id[pid]
            new_p["pos"] = p["pos"]
            new_p["yaw"] = p["yaw"]
            new_pl.append(new_p)
        out.append(
            {
                "role": "assistant",
                "content": json.dumps({"placements": new_pl}, separators=(",", ":"), ensure_ascii=False),
            }
        )
    return out


def parse_v1_1(text: str | None, ids: list[str]) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Never raises. -> (placements {id: {size: [sx,sy,sz], pos: [x,y,z], yaw: rad, on: str|None}}, error | None).
    error: truncated_or_bad_json | bad_json | bad_item | bad_number | bad_size | duplicate_id | missing_ids |
    extra_ids | bad_support_ref."""
    s = text.split("</think>", 1)[-1] if isinstance(text, str) else ""
    i = s.find("{")
    try:
        items = json.JSONDecoder().raw_decode(s, i)[0]["placements"] if i >= 0 else None
    except (ValueError, KeyError, TypeError, IndexError, RecursionError):
        return {}, "truncated_or_bad_json"
    if not isinstance(items, list):
        return {}, "bad_json"
    out: dict[str, dict[str, Any]] = {}
    err: str | None = None
    for p in items:
        if not isinstance(p, dict):
            err = err or "bad_item"
            continue
        size, pos, yaw, on = p.get("size"), p.get("pos"), p.get("yaw"), p.get("on")
        if not (
            isinstance(p.get("id"), str)
            and isinstance(size, list)
            and len(size) == 3
            and all(map(_isnum, size))
            and isinstance(pos, list)
            and len(pos) == 3
            and all(map(_isnum, pos))
            and _isnum(yaw)
            and (on is None or isinstance(on, str))
        ):
            err = err or "bad_item"
            continue
        try:
            s_vals = [float(v) for v in size]
            p_vals = [float(v) for v in pos + [yaw]]
        except OverflowError:
            s_vals, p_vals = [math.inf], [math.inf]
        if not all(math.isfinite(v) and abs(v) < MAX_COORD for v in s_vals + p_vals):
            err = err or "bad_number"
            continue
        if not all(0.01 <= v <= 50.0 for v in s_vals):
            err = err or "bad_size"
            continue
        if p["id"] in out:
            err = err or "duplicate_id"
            continue
        out[p["id"]] = {
            "size": s_vals,
            "pos": p_vals[:3],
            "yaw": math.radians(p_vals[3] % 360),
            "on": on,
        }
    if err is None and set(out) != set(ids):
        err = "missing_ids" if set(ids) - set(out) else "extra_ids"
    if err is None:
        for pid, p in out.items():
            seen, cur = {pid}, p["on"]
            while cur is not None:
                if cur not in out or cur in seen:
                    return out, "bad_support_ref"
                seen.add(cur)
                cur = out[cur]["on"]
    return out, err


def gt_placements_v1_1(room: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reference layout in parse_v1_1() form."""
    return {
        o["id"]: {
            "size": list(o["size"]),
            "pos": list(o["pos"]),
            "yaw": o["yaw"],
            "on": o.get("parent"),
        }
        for o in room["objects"]
    }


def apply_v1_1(room: dict[str, Any], pl: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Predicted layout in IR form; size, pos, yaw, parent come from the prediction."""
    return {
        **room,
        "objects": [
            {
                **o,
                "size": pl[o["id"]]["size"],
                "pos": pl[o["id"]]["pos"],
                "yaw": pl[o["id"]]["yaw"],
                "parent": pl[o["id"]]["on"],
            }
            for o in room["objects"]
            if o["id"] in pl
        ],
    }


if __name__ == "__main__":
    sample_v1 = [
        {"role": "system", "content": "old"},
        {
            "role": "user",
            "content": '{"room_type":"bed room","boundary_type":"polygon","boundary":[[0,0],[4,0],[4,3],[0,3]],"height":2.7,"fixed":[{"id":"door_1","size":[0.1,0.9,2.1],"pos":[0.05,0.8,0],"yaw":0}],"objects":[{"id":"desk_1","size":[1.2,0.6,0.75],"desc":"wooden desk"},{"id":"lamp_1","size":[0.2,0.2,0.4]}]}',
        },
        {
            "role": "assistant",
            "content": '{"placements":[{"id":"desk_1","pos":[2,0.3,0],"yaw":90},{"id":"lamp_1","on":"desk_1","pos":[2,0.3,0.75],"yaw":0}]}',
        },
    ]
    v11 = convert_v1_messages_to_v1_1(sample_v1)
    u = json.loads(v11[1]["content"])
    assert "size" not in u["objects"][0] and u["objects"][0]["desc"] == "wooden desk"
    assert u["fixed"][0]["size"] == [0.1, 0.9, 2.1]
    pl, err = parse_v1_1(v11[2]["content"], ["desk_1", "lamp_1"])
    assert err is None and pl["desk_1"]["size"] == [1.2, 0.6, 0.75] and pl["lamp_1"]["on"] == "desk_1"
    print("scene_v1_1.py self-check ok")
