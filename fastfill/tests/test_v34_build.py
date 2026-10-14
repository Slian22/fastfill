"""Data v3.4 build fixes (source-fidelity audit 2026-10-14, synthesis section 4: B1, B2, B3, B11).

Fixtures copy the geometry of the audit's examples from /Volumes/harddisk/fastfill_ir. "Before" values are what
v3.3 build.prep produced on them (git 9f72e9944); the comments give them, the tests assert the v3.4 behaviour.
"""
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import pytest
from shapely.geometry import Polygon

from fastfill import anchors, build, scene
from fastfill.comparison import contract
from fastfill.validate import check

ARGS = SimpleNamespace(min_objects=1, max_vertices=0, reject_flagged=[], boundary_types=["polygon", "hull"],
                       anchors=["floor", "object"], oob_tol=0.1, hidden_max=0.3, source_anchors={})


def room(objects, boundary, source="Fixture", btype="polygon", height=2.8):
    return {"uid": "fixture:v34", "source": source, "group": "fixture:v34", "meta": {"front_known": True},
            "boundary_type": btype, "boundary": boundary, "height": height, "objects": objects}


def obj(i, category, size, pos, yaw=0.0, **kw):
    return {"id": i, "category": category, "size": size, "pos": pos, "yaw": yaw, **kw}


def prep(r):
    out, why = build.prep(json.loads(json.dumps(r)), ARGS)
    assert why is None, why
    return out, {o["id"]: o for o in out["objects"]}


def legacy(o):
    """v3.3 rounding: centre and size rounded on their own (scene.num / scene.shown_size)."""
    return {**o, "pos": [scene.num(o["pos"][0]), scene.num(o["pos"][1]), 0], "size": scene.shown_size(o),
            "yaw": math.radians(round(math.degrees(o["yaw"])) % 360)}


def pokes(o, boundary):
    return not Polygon(boundary).buffer(build.POKE_M).covers(Polygon(scene.footprint(o)))


def overlap(a, b):
    return Polygon(scene.footprint(a)).intersection(Polygon(scene.footprint(b))).area


def on_grid(o):
    """Every number the model is shown equals the stored one (cm, integer degrees)."""
    return o["pos"] == [scene.num(v) for v in o["pos"]] and o["size"] == scene.shown_size(o) and \
        math.isclose(math.degrees(o["yaw"]), round(math.degrees(o["yaw"])), abs_tol=1e-9)


# ---------------------------------------------------------------- B1 rounding

def test_b1_washing_machine_flush_with_wall_stays_inside():
    """spatiallm:scene_000467::room_2 bbox_0 (audit obj_0001): the official face is 0.004 mm inside the wall at y=4.
    v3.3: centre 3.66597 -> 3.67 and size 0.66806 -> 0.67, so the face moved to 4.005 (a 5 mm R5 poke)."""
    wm = obj("wm", "washing machine", [0.6680599975585938, 0.607989990234375, 0.847916015625],
             [0.3865168025289911, 3.665970003890757, 0.045714572187500024], 4.71238898038469)
    b = [[1.3, 4.0], [0, 4.0], [0, 0], [1.3, 0]]
    r = room([wm, obj("d", "door", [2.095, 0.1, 2.251], [1.3, 2.352, 0], math.pi / 2, structure=True),
              obj("w", "window", [0.986, 0.1, 2.0], [0.807, 0.0, 0.3], 0.0, structure=True)], b)
    assert pokes(legacy(wm), b)                                      # before
    out, objs = prep(r)
    got = objs["wm"]
    assert not pokes(got, out["boundary"]) and on_grid(got)          # after
    assert got["pos"] == [0.39, 3.67, 0] and got["size"] == [0.66, 0.61, 0.85]
    assert not check(out)["oob"]


def test_b1_flush_cabinet_run_keeps_contacts_and_walls():
    """Three cabinets flush with each other and with the back wall, all faces off the cm grid. v3.3 rounding pushed
    the run 5 mm through the wall and 1 cm into each other; v3.4 rounds faces and creates neither."""
    w, d = 0.6363, 0.6066
    cabs = [obj(f"c{k}", "cabinet", [w, d, 0.9], [w / 2 + k * w, 2.0 - d / 2, 0]) for k in range(3)]
    b = [[0, 0], [3, 0], [3, 2], [0, 2]]
    before = [legacy(c) for c in cabs]
    assert any(pokes(c, b) for c in before) and overlap(before[0], before[1]) > 0.004
    out, objs = prep(room(cabs, b))
    after = [objs[f"c{k}"] for k in range(3)]
    assert not any(pokes(c, out["boundary"]) for c in after) and all(map(on_grid, after))
    assert max(overlap(after[0], after[1]), overlap(after[1], after[2])) < 1e-9
    for c, s in zip(after, cabs):                                    # sizes move at most 1 cm off the source
        assert all(abs(x - y) <= 0.0101 for x, y in zip(c["size"], s["size"]))


def test_b1_rotated_box_is_nudged_off_a_rounding_poke():
    """A box at 29.7 deg whose corner is 0.4 mm inside the wall: v3.3 (centre/size/yaw rounded on their own) put
    the corner 6.5 mm through it; v3.4 keeps the plain rounding but moves the box 1-2 cm until it fits."""
    o = obj("t", "tv stand", [0.9966, 0.4977, 0.7], [2.0037, 0, 0], math.radians(29.7))
    o["pos"][1] = 0.0004 - min(c[1] for c in scene.footprint(o))
    b = [[0, 0], [4, 0], [4, 3], [0, 3]]
    assert pokes(legacy(o), b)                                       # before
    out, objs = prep(room([o], b))
    got = objs["t"]
    assert not pokes(got, out["boundary"]) and on_grid(got)
    assert got["size"] == scene.shown_size(o)                        # rotated: plain size rounding kept
    assert max(abs(got["pos"][i] - legacy(o)["pos"][i]) for i in (0, 1)) <= 0.0201


def test_b1_box_inside_the_cleaned_wall_but_not_the_raw_one_does_not_grow_through_it():
    """spatiallm:scene_001422::room_3 bbox_1: the wardrobe face is 2.1 mm past the raw wall (y=3.095) but inside the
    cleaned one (3.10). Checking only the raw wall let a growing face make it 0.61 wide, 5 mm through 3.10, unflagged."""
    w = obj("w", "wardrobe", [0.604145797615216, 1.8029232866296145, 2.768431332], [2.0785435, 2.7950000000000004,
                                                                                            0.0], 4.71238898038469)
    b = [[0.0, 0.0], [3.0499999999999994, 0.0], [3.0499999999999994, 3.095], [0.0, 3.095]]   # IR values
    assert pokes(w, b)                                               # source: through the raw wall only
    out, objs = prep(room([w], b, "SpatialLM"))
    assert out["boundary"][2] == [3.05, 3.1] and not pokes(w, out["boundary"])
    assert objs["w"]["size"][:2] == [0.6, 1.8] and not pokes(objs["w"], out["boundary"])   # v3.3's box, in bounds


def test_b1_rounding_does_not_push_a_marginal_pair_over_the_collision_threshold():
    """Two boxes at 30 deg overlapping 0.0094 m2 (under validate.check's 0.01): v3.3 rounding made it 0.0124 m2,
    a new main collision (the audit's 88 marginal pairs); v3.4 keeps it under."""
    yaw = math.radians(30.0243)
    a = obj("a", "cabinet", [0.50174, 0.50257, 0.8], [1.50517, 1.50286, 0], yaw)
    c = obj("b", "bookcase", [0.50392, 0.50354, 0.8], [1.92491, 1.74396, 0], yaw)
    assert overlap(a, c) <= 0.01 < overlap(legacy(a), legacy(c))    # before
    out, objs = prep(room([a, c], [[0, 0], [4, 0], [4, 4], [0, 4]]))
    assert overlap(objs["a"], objs["b"]) <= 0.01 and not check(out)["collisions"]["main"]


def test_b1_clean_cm_room_is_unchanged():
    r = room([obj("a", "chair", [1, 1, 1], [2, 2, 0]), obj("t", "table", [1.2, 0.8, 0.75], [3.4, 1.6, 0], math.pi / 2)],
             [[0, 0], [5, 0], [5, 5], [0, 5]])
    out, objs = prep(r)
    assert {k: (o["pos"], o["size"]) for k, o in objs.items()} == {"a": ([2, 2, 0], [1, 1, 1]),
                                                                    "t": ([3.4, 1.6, 0], [1.2, 0.8, 0.75])}
    assert out["meta"] == {"front_known": True, "flags": {}, "data_version": "v3.4"}


def test_b1_cm_faces_give_cm_centres_and_any_cm_width():
    # faces 333.194..399.9996 cm: rounded faces 333 / 400 put the centre on a half cm; the low face gives up 1 cm
    assert build._cm_faces(333.194, 399.9996)[0][2:] == (334.0, 400.0)
    widths = {round(b - a) for lo in (10.0, 10.3, 10.6) for w in range(40, 50)
              for _, _, a, b in build._cm_faces(lo, lo + w + 0.1)[:1]}
    assert {w % 2 for w in widths} == {0, 1}                         # odd widths too, not only even ones
    for _, _, a, b in build._cm_faces(12.34, 56.78):
        assert (a + b) / 2 == round((a + b) / 2) and b - a == round(b - a)


# ---------------------------------------------------------------- B2 floating boxes

def test_b2_wardrobe_over_sofa_goes_on_the_sofa():
    """spatiallm:scene_010932::room_3: wardrobe 0.45-2.60 m over a 0.45 m combination sofa. v3.3: the wardrobe was
    snapped to z=0 through the sofa (severe main pair 0.90 m2)."""
    sofa = obj("sofa", "combination sofa", [2.0, 2.4, 0.45], [1.2, 1.0, 0.0], math.pi / 2)
    ward = obj("ward", "wardrobe", [0.469, 1.929, 2.15], [2.166, 1.034, 0.45], math.pi)
    out, objs = prep(room([sofa, ward], [[0, 0], [2.4, 0], [2.4, 4.0], [0, 4.0]]))
    assert objs["ward"]["parent"] == "sofa" and objs["ward"]["pos"][2] == 0.45
    assert not check(out)["collisions"]["main"] and not check(out)["support_fail"]


def test_b2_table_and_chairs_on_a_bed_platform():
    """structured3d_02509_room_687234: table bbox 77 (0.45-1.33 m) and chairs stand on a 0.45 m bed platform (26).
    v3.3: snapped to z=0 into the platform (severe 0.22 m2 pair)."""
    bed = obj("26", "bed", [0.788, 1.628, 0.45], [3.016, 0.786, 0.0], math.pi)
    table = obj("77", "table", [0.44, 0.499, 0.884], [3.029, 0.806, 0.45], math.pi / 2)
    chair = obj("22", "chair", [0.467, 0.402, 0.451], [3.034, 0.273, 0.45], math.pi / 2)
    b = [[0, 0], [3.41, 0], [3.41, 1.6], [2.63, 1.6], [2.63, 2.42], [0, 2.42]]
    out, objs = prep(room([bed, table, chair], b))
    assert {k: (objs[k]["parent"], objs[k]["pos"][2]) for k in ("77", "22")} == {"77": ("26", 0.45), "22": ("26", 0.45)}
    assert not check(out)["collisions"]["main"]
    assert "support_lowered" not in out["meta"]["flags"]             # true contact: nothing lowered


def test_b2_floating_oven_is_lowered_onto_the_cabinet_below():
    """InternScenes arkit 47895976: oven 1 floats 0.14 m above cabinet 15. v3.3: snapped to z=0 inside the cabinet."""
    cab = obj("15", "cabinet", [0.62, 0.599, 0.289], [1.108, 0.339, 0.0], 1.583)
    oven = obj("1", "oven", [0.646, 0.599, 0.609], [1.113, 0.353, 0.432], 1.583)
    out, objs = prep(room([cab, oven], [[0, 0], [5.9, 0], [5.9, 6.9], [0, 6.9]], "InternScenes_arkit", "hull"))
    assert objs["1"]["parent"] == "15" and objs["1"]["pos"][2] == 0.29 and objs["1"]["support_gap"] == 0.143
    assert out["meta"]["flags"]["support_lowered"] == 1 and not check(out)["collisions"]["main"]


def test_b2_keeps_the_floor_and_contact_rules():
    infer = lambda *os: {o["id"]: (o["anchor"], o.get("parent"))
                         for o in anchors.infer_anchors({"objects": [dict(o) for o in os]})["objects"]}
    rug = obj("r", "rug", [3, 2, 0.02], [1, 1, 0])
    assert infer(obj("s", "sofa", [2, 0.9, 0.8], [1, 1, 0.07]), rug)["s"] == ("floor", None)   # a rug is no support
    low = obj("x", "box", [0.3, 0.3, 0.25], [0.0, 0.0, 0.0])                # under 1/4 of the chair: not its support
    assert infer(obj("c", "chair", [0.6, 0.6, 0.8], [0.2, 0.2, 0.3]), low)["c"] == ("floor", None)
    desk, lamp = obj("d", "desk", [1.2, 0.6, 0.75], [1, 1, 0]), obj("l", "table lamp", [0.2, 0.2, 0.4], [1.2, 1, 0.77])
    got = anchors.infer_anchors({"objects": [dict(desk), dict(lamp)]})["objects"]
    assert got[1]["parent"] == "d" and "support_gap" not in got[1]     # true contact: the 5 cm rule, unchanged
    floating_fixed = obj("v", "vanity cabinet", [0.6, 0.4, 0.5], [3, 3, 0.35])   # wall-hung unit stays fixed
    assert infer(floating_fixed, obj("b", "box", [0.6, 0.4, 0.2], [3, 3, 0]))["v"] == ("fixed", None)


# ---------------------------------------------------------------- B3 doors

def test_b3_sage_door_prop_is_not_a_door():
    """SAGE-10k layout_45e3962a: room_..._door_7690d30b is furniture 11 cm off the wall (source anchor wall). v3.3
    showed it as a second door (fixed box with a door word, so a downstream door strip)."""
    real = obj("door_08775e76", "door", [0.95, 0.1, 2.05], [0.925, 4.6, 0.0], 0.0, structure=True)
    prop = obj("prop", "door", [0.077, 1.042, 2.083], [3.581, 0.902, 0.371], math.pi, anchor="wall")
    floor_prop = obj("prop2", "doorway_1", [0.1, 0.9, 2.0], [1.5, 2.0, 0.0], 0.0)
    bench = obj("bench", "bench", [0.5, 1.5, 0.75], [2.85, 3.85, 0])
    r = room([real, prop, floor_prop, bench], [[3.7, 0], [3.7, 4.6], [0, 4.6], [0, 0]], "SAGE-10k")
    assert [f["id"] for f in anchors.fixed_geometry(r)] == ["door_08775e76"]
    out, objs = prep(r)
    assert [f["id"] for f in out["fixed"]] == ["door_08775e76"]
    assert set(objs) == {"bench", "prop2"}                           # a floor-standing prop is placed like furniture
    # every other source keeps named doors as doors: InternScenes annotates doors only as boxes, and unflagged door
    # boxes elsewhere are real doors / frames / shower doors (InteriorGS, IL3D, MultiScan, HSSD200 stay as in v3.3)
    named = obj("d9", "door", [0.9, 0.1, 2.0], [1.0, 0.0, 0.0])
    for src, cat in [("InteriorGS", "Door frame"), ("IL3D_3dfront", "door"), ("MultiScan", "glass door"),
                     ("HSSD200", "shower_door")]:
        assert anchors.structural({**named, "category": cat}, src)
    r2 = room([named, bench], [[0, 0], [3.7, 0], [3.7, 4.6], [0, 4.6]], "InternScenes_scannet", "hull")
    assert [f["id"] for f in anchors.fixed_geometry(r2)] == ["d9"]
    assert anchors.structural(named) and not anchors.structural(named, "SAGE-10k")
    assert anchors.structural({**named, "category": "window"}, "SAGE-10k")   # only doors change


# ---------------------------------------------------------------- B11 far outside

def test_b11_object_far_outside_rejects_the_room(tmp_path, monkeypatch):
    """InternScenes arkit 43828265: bed 3 lies wholly outside the scanned floor (2.88 m). v3.3 kept the room with only
    an oob_objects flag; v3.4 flags it oob_far and every build rejects that flag."""
    toilet = obj("0", "toilet", [0.55, 0.38, 0.76], [1.1, 0.32, 0])
    bed = obj("3", "bed", [2.04, 1.43, 0.59], [-0.97, 3.29, 0])
    b = [[0, 0], [3.2, 0], [3.2, 2.2], [0, 2.2]]
    far = room([toilet, bed], b, "InternScenes_arkit", "hull")
    out, _ = prep(far)                                               # replay args without it: flagged only
    assert out["meta"]["flags"]["oob_objects"] == 1 and out["meta"]["flags"]["oob_far"] == 2.69
    rejecting = SimpleNamespace(**{**vars(ARGS), "reject_flagged": ["oob_far"]})
    assert build.prep(json.loads(json.dumps(far)), rejecting) == (None, "flagged:oob_far")
    near = obj("c", "cabinet", [0.6, 0.5, 0.8], [3.2, 1.0, 0])    # 0.3 m out: kept, flagged as before
    out, _ = prep(room([toilet, near], b, "InternScenes_arkit", "hull"))
    assert out["meta"]["flags"] == {"oob_objects": 1}
    assert build._poke_depth(near, Polygon(b)) == pytest.approx(0.3) and 0.3 < build.FAR_OOB_M
    ir = tmp_path / "ir"                                             # a build always rejects oob_far
    ir.mkdir()
    (ir / "Fixture.jsonl").write_text(json.dumps({**far, "uid": "f:far", "group": "f:far"}) + "\n" +
                                      json.dumps({**room([toilet], b), "uid": "f:ok", "group": "f:ok"}) + "\n")
    monkeypatch.setattr(sys, "argv", ["build", "--ir", str(ir), "--out", str(tmp_path / "out")])
    build.main()
    assert json.loads((tmp_path / "out" / "stats.json").read_text())["InternScenes_arkit"]["rejected:flagged:oob_far"] == 1
    assert json.loads((tmp_path / "out" / "MANIFEST.json").read_text())["args"]["reject_flagged"] == ["oob_far"]


# ---------------------------------------------------------------- version and unchanged text protocol

def test_data_version_is_recorded(tmp_path, monkeypatch):
    ir = tmp_path / "ir"
    ir.mkdir()
    r = room([obj("a", "chair", [1, 1, 1], [2, 2, 0])], [[0, 0], [5, 0], [5, 5], [0, 5]])
    (ir / "Fixture.jsonl").write_text(json.dumps(r) + "\n")
    monkeypatch.setattr(sys, "argv", ["build", "--ir", str(ir), "--out", str(tmp_path / "out")])
    build.main()
    assert build.DATA_VERSION == "v3.4"
    assert json.loads((tmp_path / "out" / "MANIFEST.json").read_text())["data_version"] == "v3.4"
    sidecars = [json.loads(line) for name in ("dev_rooms", "test_rooms")
                for line in (tmp_path / "out" / f"{name}.jsonl").read_text().splitlines()]
    assert all(s["meta"]["data_version"] == "v3.4" for s in sidecars)


GOLDEN = {   # sha256 of the outputs below at git 9f72e9944 (before v3.4); the comparison runtime reads these formats
    "num": "a04a387d52a611c32d80ccb3f63b69c1d257f75a1b02e6f153e372cf7e1fa4b0",
    "shown_size": "55362ecb05791238dde4814e619604276bd0db30d7153c8b77393715487c381d",
    "messages": "e1abf5093a7e70e06cb3e4aa1d6e15d3ce4039d6efc328b6972ee06024ecea5c",
    "parse": "397ce063aef7468201da43f6836399326d234cd34f558c8dd3e29e6e6fbca8e9",
    "contract_joint": "a4812a3ee120aef685743827e238532c1f25b2f9788c2ce61beb6fd6fb9e0cf0",
    "contract_placement": "10bfd323b3a7c722223cfa4fd1113b0cfa3eecf452a71ec1af5da2bdefe8d5af",
    "sidecar": "bc4fcf6f142a2286e0c3114b0c7658bfce9418a9c7cfc82e9bcc0d8515518d7e",
}
GOLDEN_ROOM = {
    "uid": "fixture:golden", "source": "Fixture", "group": "fixture:golden", "room_type": "Bedroom",
    "boundary_type": "polygon", "boundary": [[0, 0], [4.2, 0], [4.2, 3.6], [0, 3.6]], "height": 2.7,
    "fixed": [{"id": "d", "category": "door", "size": [0.9, 0.1, 2.1], "pos": [1.0, 0.0, 0.0], "yaw": 0.0}],
    "objects": [
        {"id": "bed", "category": "King-size Bed", "size": [2.04, 1.83, 0.95], "pos": [2.1, 2.67, 0], "yaw": math.pi / 2,
         "parent": None, "desc": "A wooden bed placed against the wall."},
        {"id": "ns", "category": "nightstand", "size": [0.45, 0.4, 0.55], "pos": [0.76, 3.37, 0], "yaw": 3 * math.pi / 2,
         "parent": None},
        {"id": "lamp", "category": "table lamp", "size": [0.2, 0.2, 0.4], "pos": [0.76, 3.37, 0.55], "yaw": 0.0,
         "parent": "ns"}]}


def test_text_protocol_and_comparison_formats_are_byte_identical():
    h = lambda x: hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    c = scene.canonical(GOLDEN_ROOM)
    m = scene.messages(c, [["on", "table_lamp_1", "nightstand_1"]])
    bound, side = contract.bind_room(dict(GOLDEN_ROOM))
    got = {"num": [scene.num(v) for v in (0, -0.001, 2.704, 3.66597, 0.66806, 1.005, -0.0049, 12.345)],
           "shown_size": scene.shown_size({"size": [0.66806, 0.004, 1.995]}), "messages": m,
           "parse": scene.parse(m[2]["content"], [o["id"] for o in c["objects"]]),
           "contract_joint": contract.messages(bound, "joint"),
           "contract_placement": contract.messages(bound, "placement",
                                                   {o["id"]: [0.5, 0.5, 0.5] for o in bound["objects"]}),
           "sidecar": side}
    assert {k: h(v) for k, v in got.items()} == GOLDEN


def test_v34_rooms_are_exactly_what_the_model_is_shown():
    """Every v3.4 number is already a displayed number, so the unchanged renderers show the IR exactly and the
    target parses back to it."""
    rooms = [room([obj("wm", "washing machine", [0.6680599975585938, 0.607989990234375, 0.847916015625],
                       [0.3865168025289911, 3.665970003890757, 0.0457], 4.71238898038469),
                   obj("t", "tv stand", [0.9966, 0.4977, 0.7], [0.7, 1.5, 0], math.radians(29.7)),
                   obj("v", "vase", [0.1234, 0.1111, 0.3], [0.7, 1.5, 0.7])], [[1.3, 4.0], [0, 4.0], [0, 0], [1.3, 0]])]
    for r in rooms:
        out, objs = prep(r)
        assert all(map(on_grid, out["objects"] + out["fixed"]))
        c = scene.canonical(out)
        pl, err = scene.parse(scene.target_json(c), [o["id"] for o in c["objects"]])
        assert err is None
        assert {i: (p["pos"], p["on"]) for i, p in pl.items()} == {o["id"]: (o["pos"], o.get("parent"))
                                                                    for o in c["objects"]}
