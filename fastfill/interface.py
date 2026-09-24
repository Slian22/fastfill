"""WorldEdge request (scope doc, Appendix B) <-> FastFill model text.

    room, ctx = request_to_room(request)
    prompt = prompt_text(tok, messages(room, room["constraints"], with_target=False))
    ... one generation call ...
    result = placements_from_text(text, room, ctx)

Model side (scene.py): room frame with the boundary AABB min corner at (0, 0), object local +X = semantic front,
integer-degree yaw inside the text. Interface side: the request's own frame (boundary_xy, floor_z), position_m of
the box bottom center, yaw_rad about +Z of the object's own local frame, support_parent = floor_id or the id of the
object it stands on, support_surface = "top". Objects anchored to walls or ceilings ("anchor": "wall" | "ceiling")
and fronts that are not along a horizontal local axis are returned as unsupported instead of being sent to the model,
and so is any constraint that names one of them.

layout_constraints (Appendix B names; "hard" is not read, every constraint is sent as must-hold):
    {"type": "faces", "subject", "target"}         {"type": "between", "subject", "anchors": [a, b]}
    {"type": "supported_by", "subject", "parent", "surface": "top"}      {"type": "against_wall", "subject"}
    {"type": "near", "subject", "target", "max_gap_m"}   footprint gap, NOT centre distance (trained on 0.1-0.5)
    {"type": "keepout", "polygon_xy": [[x, y], ...]}      cut out of the floor, padded by KEEPOUT_PAD
"""
import copy
import math

from shapely.affinity import translate
from shapely.errors import ShapelyError
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points, unary_union

from fastfill.scene import apply, canonical, clean_boundary, footprint, num, parse
from fastfill.validate import check, holds

HULL_SOURCES = {"convex_hull", "hull", "scan_hull", "floor_hull"}
KEEPOUT_PAD = 0.10   # = validate.check's oob tolerance: a layout it accepts stays out of the keepout itself
SNAP_TOL = 0.10      # training layouts cross walls by up to this much (40% of test rooms by > 1 mm); EmbodiedGen allows 1 mm


def request_to_room(req):
    """-> (canonical model room with constraints, ctx for mapping the answer back). Raises ValueError on bad input."""
    try:
        return _request_to_room(req)
    except (KeyError, TypeError, IndexError, AttributeError, ShapelyError) as e:
        raise ValueError(f"bad_request: {e!r}") from e


def _finite(v, lo=None):
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and (lo is None or v >= lo)):
        raise ValueError(f"bad_number: {v!r}")
    return float(v)


def _request_to_room(req):
    rm = req["room"]
    cons_in = req.get("layout_constraints") or []
    pts = rm["boundary_xy"]
    keepouts = [c["polygon_xy"] for c in cons_in if c["type"] == "keepout"]
    for p in [q for poly in [pts, *keepouts] for q in poly]:
        _finite(p[0]), _finite(p[1])
    if rm.get("height_m") is not None:
        _finite(rm["height_m"], lo=0)
    walls = Polygon(pts)                    # against_wall is checked against these, not against keepout cuts
    floor = walls                           # what answers are snapped into: the walls minus the keepouts themselves
    if keepouts:        # cut out, padded, from what the model sees: it treats a keepout as wall
        if not floor.is_valid:
            raise ValueError("bad_boundary")
        cut = unary_union([Polygon(k) for k in keepouts])
        seen = floor.difference(cut.buffer(KEEPOUT_PAD, join_style="mitre"))
        if seen.is_empty or seen.geom_type != "Polygon" or seen.interiors:
            raise ValueError("keepout_not_representable")      # fills or splits the room, or leaves an island
        floor = floor.difference(cut)
        pts = seen.exterior.coords[:-1]
    x0, y0 = (min(p[i] for p in pts) for i in (0, 1))
    boundary = clean_boundary([[x - x0, y - y0] for x, y in pts])
    if boundary is None:
        raise ValueError("bad_boundary")
    ctx = {"x0": x0, "y0": y0, "floor_z": _finite(rm.get("floor_z", 0.0)), "floor_id": rm.get("floor_id", "floor"),
           "floor": translate(floor, -x0, -y0), "walls": [[x - x0, y - y0] for x, y in walls.exterior.coords[:-1]],
           "front_k": {}, "order": [], "unsupported": []}
    ids = [o["id"] for o in req["objects_to_place"]]
    if len(set(ids)) < len(ids) or ctx["floor_id"] in ids:
        raise ValueError("duplicate_object_id")
    objs = []
    for o in req["objects_to_place"]:
        fx, fy, fz = (_finite(v) for v in (o.get("semantic_front_local") or [1, 0, 0]))
        phi, h = math.atan2(fy, fx), math.hypot(fx, fy)
        k = round(phi / (math.pi / 2)) % 4
        if o.get("anchor") in ("wall", "ceiling"):
            ctx["unsupported"].append({"id": o["id"], "reason": "unsupported_anchor"})
            continue
        if h < 1e-9 or abs(fz) > 1e-3 * h or abs(math.remainder(phi - k * math.pi / 2, 2 * math.pi)) > 1e-3:
            ctx["unsupported"].append({"id": o["id"], "reason": "front_not_axis_aligned"})
            continue
        sx, sy, sz = o["size_xyz_m"]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
                   for v in (sx, sy, sz)):
            raise ValueError(f"bad_size: {o['id']} {o['size_xyz_m']}")
        if k % 2:                                   # front along local Y: the model's local X is that axis
            sx, sy = sy, sx
        ctx["front_k"][o["id"]] = k
        ctx["order"].append(o["id"])
        objs.append({"id": o["id"], "ext_id": o["id"], "category": o["category"], "size": [sx, sy, sz],
                     "desc": o.get("desc")})
    room = canonical({"room_type": rm.get("room_type"), "boundary": boundary, "height": rm.get("height_m"),
                      "boundary_type": "hull" if str(rm.get("boundary_source")).lower() in HULL_SOURCES else "polygon",
                      "objects": objs})
    m = {o["ext_id"]: o["id"] for o in room["objects"]}
    skipped = {u["id"] for u in ctx["unsupported"]}
    cons, ctx["checks"] = [], []            # what the model is shown / what the answer is accepted against
    for c in cons_in:
        t = c["type"]
        if t == "keepout":
            continue
        s = c["subject"]
        refs = [s] + [c[k] for k in ("target", "parent") if k in c] + list(c.get("anchors", []))
        if any(r not in m and r not in skipped for r in refs):
            raise ValueError(f"constraint on an unknown object: {c}")
        if any(r in skipped for r in refs):         # the rest of the room is still placed
            ctx["unsupported"].append({"constraint": c, "reason": "references_unsupported_object"})
            continue
        if t == "faces":
            k = ["faces", m[s], m[c["target"]]]
        elif t == "between" and len(c["anchors"]) == 2:
            k = ["between", m[s]] + [m[a] for a in c["anchors"]]
        elif t == "supported_by" and c.get("surface", "top") == "top":
            k = ["on", m[s], m[c["parent"]]]
        elif t == "near":
            k = ["near", m[s], m[c["target"]], _finite(c["max_gap_m"], lo=0)]
        elif t == "against_wall":
            k = ["against_wall", m[s]]
        else:
            raise ValueError(f"unsupported constraint: {c}")
        # the model sees cm (near rounded DOWN, so meeting what it sees meets the request); acceptance uses the
        # request's own number; hard=false constraints are reported but do not fail the layout
        cons.append(k[:3] + [num(math.floor(k[3] * 100 + 1e-9) / 100)] if t == "near" else k)
        ctx["checks"].append((k, c.get("hard", True) is not False))
    room["constraints"] = cons
    return room, ctx


def snap_inside(pl, room, floor):
    """Move each object that crosses `floor` (model frame) by at most SNAP_TOL back inside, with what stands on it.
    Lower objects first, so a child is judged after its parent moved; a child is only moved as far as its centre
    stays over its parent (validate.check's support rule), else it is left for validation to reject."""
    inside = floor.buffer(1e-6)
    size = {o["id"]: o["size"] for o in room["objects"]}
    kids = {}
    for i, p in pl.items():
        if p["on"]:
            kids.setdefault(p["on"], []).append(i)
    for i in sorted(pl, key=lambda i: pl[i]["pos"][2]):
        p = pl[i]
        box = lambda dx, dy: Polygon(footprint({"pos": [p["pos"][0] + dx, p["pos"][1] + dy], "size": size[i], "yaw": p["yaw"]}))
        if inside.covers(box(0, 0)):
            continue
        q = pl.get(p["on"])
        seat = Polygon(footprint({**q, "size": size[p["on"]]})).buffer(0.05) if q else None
        push = [0.0, 0.0]              # per axis, the largest move any outside corner needs to reach the boundary
        for c in box(0, 0).exterior.coords[:-1]:
            if not floor.covers(Point(c)):
                q = nearest_points(floor.boundary, Point(c))[0]
                for k, d in enumerate((q.x - c[0], q.y - c[1])):
                    if abs(d) > abs(push[k]):
                        push[k] = d
        if math.hypot(*push) > SNAP_TOL:
            continue
        for dx, dy in (push, (push[0], 0), (0, push[1])):
            if inside.covers(box(dx, dy)) and (seat is None or seat.contains(Point(p["pos"][0] + dx, p["pos"][1] + dy))):
                todo = [i]
                while todo:            # parse() rejects support cycles
                    j = todo.pop()
                    pl[j]["pos"] = [pl[j]["pos"][0] + dx, pl[j]["pos"][1] + dy, pl[j]["pos"][2]]
                    todo += kids.get(j, [])
                break


def placements_from_text(text, room, ctx):
    """Model answer -> {"placements": after repair, "raw_placements": as the model proposed, "validation":
    {"raw": ..., "repaired": ...}, "error": code | None, "unsupported": [...]}, placements in request order.
    Repair = floor objects crossing a wall or a keepout by at most SNAP_TOL moved back inside (snap_inside); it is
    deterministic post-processing, so the raw proposal and its validation are always returned next to it."""
    ids = [o["id"] for o in room["objects"]]
    pl, err = parse(text, ids)
    if err is not None:
        return {"placements": [], "raw_placements": [], "validation": None, "error": err,
                "unsupported": ctx["unsupported"]}
    raw = copy.deepcopy(pl)
    snap_inside(pl, room, ctx["floor"])
    return {"placements": _to_request(pl, room, ctx), "raw_placements": _to_request(raw, room, ctx),
            "validation": {"raw": validation(raw, room, ctx), "repaired": validation(pl, room, ctx)},
            "error": None, "unsupported": ctx["unsupported"]}


def _to_request(pl, room, ctx):
    model2ext = {o["id"]: o["ext_id"] for o in room["objects"]}
    by_ext = {}
    for mid, p in pl.items():
        ext = model2ext[mid]
        yaw = (p["yaw"] - ctx["front_k"][ext] * math.pi / 2) % (2 * math.pi)   # model +X (front) -> asset frame
        item = {"id": ext, "position_m": [p["pos"][0] + ctx["x0"], p["pos"][1] + ctx["y0"], p["pos"][2] + ctx["floor_z"]],
                "yaw_rad": yaw, "support_parent": model2ext[p["on"]] if p["on"] else ctx["floor_id"]}
        if p["on"]:
            item["support_surface"] = "top"
        by_ext[ext] = item
    return [by_ext[i] for i in ctx["order"]]


def validation(pl, room, ctx):
    """The canonical checks (validate.check / validate.holds) on a placement, with request ids. ok = valid (inside
    boundary + 0.1, supported, under the ceiling), every object (on others too) within 1 mm of the walls minus the
    keepouts, and every hard constraint holding with the request's own numbers against the real walls.
    Collisions are reported, not gated (the reference layouts themselves have 7.2% rooms with a severe one)."""
    ext = {o["id"]: o["ext_id"] for o in room["objects"]}
    lay = apply(room, pl)
    c = check(lay)
    grown = ctx["floor"].buffer(1e-3)
    outside = [ext[o["id"]] for o in lay["objects"] if not grown.covers(Polygon(footprint(o)))]
    cons = [{"constraint": [k[0]] + [ext.get(v, v) for v in k[1:]], "hard": hard,
             "holds": holds(k, lay["objects"], ctx["walls"])} for k, hard in ctx["checks"]]
    return {"ok": c["valid"] and not outside and all(x["holds"] for x in cons if x["hard"]),
            "oob": [ext[i] for i in c["oob"]],
            "support_fail": [ext[i] for i in c["support_fail"]], "ceiling": [ext[i] for i in c["ceiling"]],
            "outside_1mm": outside, "collisions": {k: [[ext[a], ext[b]] for a, b in v] for k, v in c["collisions"].items()},
            "constraints": cons}


if __name__ == "__main__":
    import json

    from fastfill.scene import messages, target_json
    from fastfill.validate import holds

    req = {  # scope doc Appendix B, room moved to (10, 20) and one chair modelled with its front along local +Y
        "room": {"id": "new_room", "room_type": "meeting room", "boundary_xy": [[10, 20], [14, 20], [14, 24], [10, 24]],
                 "boundary_source": "specified_polygon", "floor_id": "room_floor", "floor_z": 0.0, "height_m": 2.8},
        "objects_to_place": [
            {"id": "table_1", "category": "table", "size_xyz_m": [1.4, 0.8, 0.75]},
            {"id": "chair_1", "category": "chair", "size_xyz_m": [0.55, 0.5, 0.9], "semantic_front_local": [1, 0, 0]},
            {"id": "chair_2", "category": "chair", "size_xyz_m": [0.5, 0.55, 0.9], "semantic_front_local": [0, 1, 0]},
            {"id": "cup_1", "category": "cup", "size_xyz_m": [0.08, 0.08, 0.1]},
            {"id": "clock_1", "category": "clock", "size_xyz_m": [0.3, 0.05, 0.3], "anchor": "wall"}],
        "layout_constraints": [
            {"type": "faces", "subject": "chair_1", "target": "chair_2", "hard": True},
            {"type": "faces", "subject": "chair_2", "target": "chair_1", "hard": True},
            {"type": "between", "subject": "table_1", "anchors": ["chair_1", "chair_2"], "hard": True},
            {"type": "supported_by", "subject": "cup_1", "parent": "table_1", "surface": "top", "hard": True}]}
    room, ctx = request_to_room(req)
    assert [o["id"] for o in room["objects"]] == ["table_1", "chair_1", "chair_2", "cup_1"], room["objects"]
    assert room["objects"][2]["size"] == [0.55, 0.5, 0.9]            # chair_2 re-framed: front axis first
    assert ctx["unsupported"] == [{"id": "clock_1", "reason": "unsupported_anchor"}]
    assert room["constraints"][3] == ["on", "cup_1", "table_1"]
    # the model answers in its own frame: chair_1 at x=0.9 facing +X, chair_2 at x=3.1 facing -X, cup on the table
    ans = {"table_1": ([2, 2, 0], 0), "chair_1": ([0.9, 2, 0], 0), "chair_2": ([3.1, 2, 0], math.pi),
           "cup_1": ([1.7, 2, 0.75], 0)}
    placed = {**room, "objects": [{**o, "pos": ans[o["id"]][0], "yaw": ans[o["id"]][1],
                                   "parent": "table_1" if o["id"] == "cup_1" else None} for o in room["objects"]]}
    assert all(holds(c, placed["objects"], placed["boundary"]) for c in room["constraints"])
    text = target_json(placed)
    assert "constraints" in messages(room, room["constraints"], with_target=False)[1]["content"]
    out = placements_from_text(text, room, ctx)
    got = {p["id"]: p for p in out["placements"]}
    assert out["error"] is None and [p["id"] for p in out["placements"]] == ["table_1", "chair_1", "chair_2", "cup_1"]
    assert got["chair_1"]["position_m"] == [10.9, 22, 0] and got["chair_1"]["yaw_rad"] == 0
    assert abs(got["chair_2"]["yaw_rad"] - math.pi / 2) < 1e-9           # front on local +Y: asset turned by 90 deg
    assert got["cup_1"]["support_parent"] == "table_1" and got["cup_1"]["support_surface"] == "top"
    assert got["table_1"]["support_parent"] == "room_floor" and "support_surface" not in got["table_1"]
    assert placements_from_text("not json", room, ctx)["error"] == "bad_json"
    # table 3 cm through the east wall (x up to 4.03): moved back inside, the cup on it moves along; 20 cm is left alone
    over = {**placed, "objects": [{**o, "pos": [3.33 if o["id"] == "table_1" else 3.0 if o["id"] == "cup_1" else o["pos"][0],
                                                *o["pos"][1:]]} for o in placed["objects"]]}
    got = {p["id"]: p["position_m"] for p in placements_from_text(target_json(over), room, ctx)["placements"]}
    assert abs(got["table_1"][0] - 13.3) < 1e-9 and abs(got["cup_1"][0] - 12.97) < 1e-9, got
    far = {**over, "objects": [{**o, "pos": [3.5, *o["pos"][1:]]} if o["id"] == "table_1" else o for o in over["objects"]]}
    assert placements_from_text(target_json(far), room, ctx)["placements"][0]["position_m"][0] == 13.5

    # a doorway keepout on the west wall becomes a notch (padded) in the floor the model sees; the frame is unchanged
    door = {"type": "keepout", "polygon_xy": [[10, 21.5], [11, 21.5], [11, 22.5], [10, 22.5]]}
    room2, ctx2 = request_to_room({**req, "layout_constraints": req["layout_constraints"] + [door]})
    assert (ctx2["x0"], ctx2["y0"]) == (10, 20) and [0, 1.4] in room2["boundary"] and [1.1, 2.6] in room2["boundary"]
    assert room2["constraints"] == room["constraints"]
    # a constraint naming an unsupported object is skipped and reported; the rest of the room is still placed
    hang = {"type": "near", "subject": "clock_1", "target": "table_1", "max_gap_m": 0.5}
    room3, ctx3 = request_to_room({**req, "layout_constraints": [hang]})
    assert room3["constraints"] == [] and ctx3["unsupported"][1] == {"constraint": hang, "reason": "references_unsupported_object"}
    up = {**req, "objects_to_place": [{**req["objects_to_place"][0], "semantic_front_local": [0, 0, 1]}], "layout_constraints": []}
    assert request_to_room(up)[1]["unsupported"] == [{"id": "table_1", "reason": "front_not_axis_aligned"}]
    bad = [{**req, "objects_to_place": req["objects_to_place"] * 2},                                  # duplicate ids
           {**req, "layout_constraints": [{"type": "near", "id": "cup_1", "target": "table_1", "max_distance_m": 1.5}]},
           {**req, "layout_constraints": [{"type": "between", "subject": "table_1", "anchors": ["chair_1"]}]},
           {**req, "room": {**req["room"], "floor_z": None}},
           {**req, "objects_to_place": [{**req["objects_to_place"][0], "size_xyz_m": [1.4, -0.8, 0.75]}]},
           {**req, "objects_to_place": [{**req["objects_to_place"][0], "size_xyz_m": [1.4, float("nan"), 0.75]}]},
           {**req, "layout_constraints": [{**door, "polygon_xy": [[9, 19], [15, 19], [15, 25], [9, 25]]}]}]  # fills the room
    bad += [{**req, "room": {**req["room"], "floor_z": float("inf")}},
            {**req, "room": {**req["room"], "height_m": float("nan")}},
            {**req, "layout_constraints": [{"type": "near", "subject": "cup_1", "target": "table_1", "max_gap_m": -0.1}]}]
    for b in bad:
        try:
            request_to_room(b)
            raise AssertionError(b)
        except ValueError:
            pass

    # near: the model is shown the threshold rounded down to cm, the answer is accepted against the request's own
    # number; a failed hard constraint fails the layout (after repair), a failed soft one is only reported
    near = lambda gap, hard=True: {"type": "near", "subject": "chair_1", "target": "chair_2", "max_gap_m": gap, "hard": hard}
    r5, c5 = request_to_room({**req, "layout_constraints": [near(0.126)]})
    assert r5["constraints"] == [["near", "chair_1", "chair_2", 0.12]] and c5["checks"][0][0][3] == 0.126
    apart = target_json(placed)                        # chairs 1.7 m apart: fails any near <= 0.5
    v = placements_from_text(apart, *request_to_room({**req, "layout_constraints": [near(0.1)]}))["validation"]
    assert not v["repaired"]["ok"] and v["repaired"]["constraints"][0]["holds"] is False, v
    v = placements_from_text(apart, *request_to_room({**req, "layout_constraints": [near(0.1, hard=False)]}))["validation"]
    assert v["repaired"]["ok"], v
    # a cup hanging 3 cm over the east wall from a table flush with it: the cup is pushed in, staying on the table
    edge = {**placed, "objects": [{**o, "pos": [3.3, 2, 0] if o["id"] == "table_1" else [3.99, 2, 0.75] if o["id"] == "cup_1"
                                   else o["pos"]} for o in placed["objects"]]}
    out = placements_from_text(target_json(edge), room, ctx)
    cup = {p["id"]: p for p in out["placements"]}["cup_1"]["position_m"]
    assert out["validation"]["raw"]["outside_1mm"] == ["cup_1"] and out["validation"]["repaired"]["ok"], out["validation"]
    assert abs(cup[0] - 13.96) < 1e-9, cup
    print("interface.py self-check ok")
    print(json.dumps(out["placements"][2]))
