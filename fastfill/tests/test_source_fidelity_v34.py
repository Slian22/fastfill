"""Data v3.4 adapter fixes B4-B9 of the 2026-10-14 source-fidelity audit; fixtures are the audit's own examples,
copied from the raw / unified files (no corpus read here).

See docs/research/20261013-unknown-size-comparison/evidence/source-fidelity-audit-20261014/synthesis.md section 4.
"""
import csv
import io
import itertools
import json
import math
import tarfile
from unittest.mock import patch

import numpy as np
import pytest
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from fastfill.adapters import hssd200, il3d_3dfront, internscenes, multiscan, optiscene_holodeck, sage10k, structured3d
from fastfill.adapters.unified import convert_room
from fastfill.scene import footprint


def fp(o):
    return Polygon(footprint(o))


def poke(o, ring):
    """Depth (m) of the object's footprint outside the room polygon."""
    room = Polygon(ring)
    return max((room.exterior.distance(Point(q)) for q in fp(o).exterior.coords if not room.covers(Point(q))),
               default=0.0)


# ---- B4 Holodeck: the official bbox width / depth carry a 0.10 m buffer ---------------------------------------------

BATHROOM_101 = {  # OptiScene_exported/holodeck.json, scene HOLODECK-bathroom-101 (audit: bathtub-0 pokes 4.67 cm)
    "scene_id": "HOLODECK-bathroom-101", "provenance": {"source_file": "scenes/bathroom-101.json"},
    "rooms": [{"room_id": "HOLODECK-bathroom-101::room_0", "room_type": "bathroom", "room_height": None,
               "room_boundary": [[-1.5, -2.0], [-1.5, 2.0], [1.5, 2.0], [1.5, -2.0]], "furniture": [
                   {"furniture_category": cat, "furniture_instance_id": f"HOLODECK-bathroom-101::obj_{k}",
                    "source_fields": {"layout_object": {"position": {"x": x, "y": 0, "z": z},
                                                        "rotation": {"x": 0, "y": ry, "z": 0}, "bbox": bbox}}}
                   for k, (cat, x, z, ry, bbox) in enumerate([
                       ("bathtub", 0.7, -1.1317, 180, [71, 83, 183]),
                       ("vanity_cabinet_with_sink", 1.2209, 0.8, 90, [94, 172, 65]),
                       ("toilet", -1.1472, 1.6, 270, [68, 56, 80]),
                       ("towel_rack", -1.2003, 0.2, 270, [105, 168, 69]),
                       ("laundry_basket", 0.5, 1.7231, 0, [58, 58, 64])])]}]}


def test_holodeck_unpadded_boxes_are_the_mesh_and_stay_inside_the_floor():
    with patch.object(optiscene_holodeck, "iter_records", return_value=iter([BATHROOM_101])):
        ir, = optiscene_holodeck.load("unused")
    tub = ir["objects"][0]
    assert tub["size"] == pytest.approx([1.73, 0.73, 0.71])          # Objathor mesh 1.727 x 0.726, height exact
    assert abs(tub["size"][0] - 1.727) < 0.005 and abs(tub["size"][1] - 0.726) < 0.005
    assert max(poke(o, ir["boundary"]) for o in ir["objects"]) <= 0.005
    assert "minus its 0.10 m" in ir["meta"]["size_note"]
    # the padded box (what v3.2/v3.3 shipped) went 4.67 cm through the wall
    padded = convert_room({**BATHROOM_101["rooms"][0], "furniture": [optiscene_holodeck.raw_furniture(
        BATHROOM_101["rooms"][0]["furniture"][0])]}, source="t", uid="t", group="g", front_offset_deg=-90)
    assert poke(padded["objects"][0], padded["boundary"]) == pytest.approx(0.0467, abs=1e-4)


def test_holodeck_raw_frame_is_unchanged_for_the_comparison_contract():
    # comparison/tests/test_optiscene_eval.py relies on raw_furniture = their text frame (padded, as their model sees it)
    g = optiscene_holodeck.raw_furniture({"source_fields": {"layout_object": {
        "position": {"x": 1.25, "y": 0.0, "z": -0.4}, "rotation": {"x": 0, "y": 90, "z": 0}, "bbox": [75, 77, 174]}}})
    assert g["furniture_size"] == {"width": 0.77, "length": 1.74, "height": 0.75}
    assert optiscene_holodeck.unpad(g)["furniture_size"] == pytest.approx({"width": 0.67, "length": 1.64, "height": 0.75})
    # mirrors / dishwashers 3-5 mm deep have a 10 cm padded depth: kept at MIN_DIM, never 0
    thin = optiscene_holodeck.unpad({"furniture_size": {"width": 0.18, "length": 0.10, "height": 0.13}})
    assert thin["furniture_size"]["length"] == optiscene_holodeck.MIN_DIM


# ---- B5 IL3D: wrong assets.json extents -> the layout bbox (= 3D-FRONT size) -----------------------------------------

CHAIR = "6914bc9b-d7f3-3d85-9fdf-a8e30b54ed35"     # Dining Chair, unit-cube normalised in assets.json
ASSETS = {CHAIR: {"meta_data": {"width": 2.0, "length": 1.9376716613769531, "height": 2.0}},
          "wardrobe": {"meta_data": {"width": 2.382565975189209, "length": 2.460407018661499,
                                     "height": 0.46157801151275635}},
          "lamp": {"meta_data": {"width": 2.068, "length": 2.327, "height": 0.356}}}
KIDS_ROOM = [[0.0, 0.0], [3.5326, 0.0], [3.5326, 3.1021], [0.0, 3.1021]]    # il3d:b72fe1d0-86d4-11f0-a478-60cf84ae2082


def il3d(asset, scale, bbox, pos=(2.770445, 0.985361), yaw=45.0):
    return {"furniture_category": "Dining Chair", "furniture_instance_id": "obj_1", "furniture_asset_id": asset,
            "furniture_position": {"x": pos[0], "y": pos[1], "z": 0.0}, "furniture_rotation": {"x": 0.0, "y": 0.0, "z": yaw},
            "placement_flags": {"on_floor": True}, "source_fields": {"layout_object": {"scale": scale, "bbox": bbox}}}


def il3d_ir(f, size_from_bbox):
    f = {**f, "furniture_size": size_from_bbox[0], "size_from_bbox": size_from_bbox[1]}
    return convert_room({"room_boundary": KIDS_ROOM, "furniture": [f]}, source="t", uid="t", group="g",
                        front_offset_deg=-90, extra=il3d_3dfront.il3d_extra)["objects"][0]


def test_il3d_unit_cube_chair_takes_its_layout_bbox():
    f = il3d(CHAIR, [1, 1, 1], [0.47773399353027346, 0.5041379928588867, 0.7499700164794922])   # bbox [x, z, up]
    size, from_bbox = il3d_3dfront.checked_size(f, ASSETS[CHAIR])
    assert from_bbox and size == pytest.approx({"width": 0.4777, "length": 0.5041, "height": 0.7500}, abs=1e-4)
    fixed, old = il3d_ir(f, (size, True)), il3d_ir(f, (il3d_3dfront.placed_size(f, ASSETS[CHAIR]), False))
    assert fixed["size_from_bbox"] is True and "size_from_bbox" not in old
    assert poke(fixed, KIDS_ROOM) == 0 and poke(old, KIDS_ROOM) > 0.5          # audit: 2.0 x 2.0 x 1.94 m, 0.65 m out
    # a non-integer scale: the bbox is the unscaled [x, up, z] (il3d:b59bc802...)
    g = il3d(CHAIR, [1.037, 1.105, 1.114], [0.478, 0.75, 0.504])
    size, from_bbox = il3d_3dfront.checked_size(g, ASSETS[CHAIR])
    assert from_bbox and size == pytest.approx({"width": 0.478 * 1.037, "length": 0.504 * 1.114, "height": 0.75 * 1.105})


def test_il3d_right_assets_json_is_kept_whatever_the_bbox_order():
    wardrobe = il3d("wardrobe", [1, 1, 1], [2.3825700378417967, 0.4615779876708984, 2.4604100036621093])
    lamp = il3d("lamp", [1, 1, 1], [2.032, 2.327, 0.32])    # 2% of int-scale bboxes are [x, up, z]: floor lamp 2.33 m
    for f, a in ((wardrobe, ASSETS["wardrobe"]), (lamp, ASSETS["lamp"])):
        assert il3d_3dfront.checked_size(f, a) == (il3d_3dfront.placed_size(f, a), False)
    assert il3d_3dfront.checked_size(lamp, ASSETS["lamp"])[0]["height"] == pytest.approx(2.327)
    assert il3d_3dfront.checked_size(il3d(CHAIR, [1, 1, 1], None), ASSETS[CHAIR])[1] is False   # no bbox: as before


# ---- B6 HSSD: a URDF box under the rigid boxes of its own modules is a leftover copy ----------------------------------

SCENE = "108294465_176709960"
KITCHEN = {"scene_id": SCENE, "region_id": "0", "region_name": "kitchen", "room_type": "kitchen",
           "poly_loop_xz": "[[-4.99625, -9.4052], [-4.99625, -4.74546], [-7.70598, -4.78271], [-7.70598, -9.4052]]",
           "floor_height": "0.0", "ceiling_height": "2.8", "extrusion_height": "2.8"}
RIGID = [  # objects.csv rows of the kitchen modules under "YOO OH Kitchen" (instance 358)
    ("13", "fridge", "[-7.294058, 0.958471, -8.968253]", "[0.374407, 0.958471, 0.380988]",
     "[0.7071068, 0.0, 0.70710676, 5e-08]", "[-6.913071, 1.916942, -8.593846]"),
    ("18", "cabinet", "[-7.40598, 0.449214, -8.35157]", "[0.18734, 0.449213, 0.3]",
     "[0.7071068, 0.0, -0.70710676, -4e-08]", "[-7.10598, 0.898427, -8.16423]"),
    ("45", "cabinet", "[-7.40598, 0.449214, -6.53011]", "[0.135791, 0.449213, 0.3]",
     "[0.7071068, 0.0, 0.70710676, 4e-08]", "[-7.10598, 0.898427, -6.394319]"),
    ("46", "cabinet", "[-7.28246, 0.449214, -5.971398]", "[0.42352, 0.449213, 0.422925]",
     "[-4e-08, 0.0, 1.0, 6e-08]", "[-6.85894, 0.898427, -5.548473]"),
    ("56", "sink_cabinet", "[-7.40598, 0.596397, -7.065917]", "[0.400009, 0.59694, 0.3]",
     "[0.7071068, 0.0, 0.70710676, 4e-08]", "[-7.10598, 1.193337, -6.665908]"),
    ("233", "cabinet", "[-7.50926, 2.198829, -8.750905]", "[0.227907, 0.19672, 0.254051]",
     "[0.49999999, 0.49999995, 0.50000007, -0.49999999]", "[-7.31254, 2.45288, -8.522998]"),
    ("235", "cabinet", "[-7.50926, 1.91765, -8.339725]", "[0.189154, 0.19672, 0.546049]",
     "[0.49999997, 0.49999995, 0.50000009, -0.49999999]", "[-7.31254, 2.463699, -8.150571]"),
    ("236", "cabinet", "[-7.50926, 2.108149, -7.985618]", "[0.16929, 0.19672, 0.355549]",
     "[0.49999998, 0.49999995, 0.50000008, -0.49999999]", "[-7.31254, 2.463699, -7.816327]"),
    ("237", "cabinet", "[-7.50926, 2.108149, -7.650477]", "[0.16929, 0.19672, 0.355549]",
     "[0.49999998, 0.49999995, 0.50000008, -0.49999999]", "[-7.31254, 2.463699, -7.481186]"),
    ("238", "cabinet", "[-7.50926, 2.108149, -7.278895]", "[0.208382, 0.19672, 0.355549]",
     "[0.49999998, 0.49999995, 0.50000008, -0.49999999]", "[-7.31254, 2.463699, -7.070513]"),
    ("239", "cabinet", "[-7.50926, 1.91765, -6.499862]", "[0.189154, 0.19672, 0.546049]",
     "[0.49999997, 0.49999995, 0.50000009, -0.49999999]", "[-7.31254, 2.463699, -6.310708]"),
    ("240", "cabinet", "[-7.50926, 1.91765, -6.113194]", "[0.189154, 0.19672, 0.546049]",
     "[0.49999997, 0.49999995, 0.50000009, -0.49999999]", "[-7.31254, 2.463699, -5.92404]"),
    ("241", "cabinet", "[-7.50926, 2.108149, -6.885859]", "[0.208382, 0.19672, 0.355549]",
     "[0.49999998, 0.49999995, 0.50000008, -0.49999999]", "[-7.31254, 2.463699, -6.677478]"),
    ("253", "cooker", "[-7.40598, 0.457219, -7.815143]", "[0.349153, 0.456694, 0.3]",
     "[0.7071068, 0.0, 0.70710676, 4e-08]", "[-7.10598, 0.913913, -7.46599]")]
YOO = "0df79a116fc0b650af57895b8c76d70b91dec6af"
YOO_URDF = (np.array([-1.37507021427155, -9.742003078532121e-08, -0.29810720682142605]),
            np.array([1.37506687641144, 2.4025320083046866, 0.29810834958436794]), "urdf")   # _urdf_bounds of its URDF


def hssd_rows():
    base = {"scene_id": SCENE, "is_architectural": "False", "bbox_source": "mesh", "non_uniform_scale": "[1.0, 1.0, 1.0]",
            "render_asset": "", "urdf_path": ""}
    rows = [{**base, "instance_id": i, "instance_kind": "rigid", "template_name": f"t{i}", "category": c,
             "region_id": "0", "region_source": "inside", "obb_center": oc, "obb_half_extents": he,
             "obb_rotation_wxyz": q, "aabb_max": am, "translation": oc} for i, c, oc, he, q, am in RIGID]
    return rows + [{**base, "instance_id": "358", "instance_kind": "articulated", "template_name": YOO, "category": "cabinet",
                    "region_id": "", "region_source": "none", "obb_center": "", "obb_half_extents": "", "aabb_max": "",
                    "obb_rotation_wxyz": "[0.7071102, 0.0, 0.7071033, 0.0]", "translation": "[-7.39406, 0.011, -7.545287]",
                    "urdf_path": f"urdf/{YOO}/{YOO}.urdf"}]


def csv_stream(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return stream


def hssd_kitchen():
    files = {"objects.csv": hssd_rows(), "regions.csv": [KITCHEN],
             "asset_catalog.csv": [{"template_name": "unused", "local_bbox_min": ""}]}
    with patch("builtins.open", side_effect=lambda path: csv_stream(files[path.rsplit("/", 1)[-1]])), \
            patch.object(hssd200, "_model", side_effect=lambda r, *_: YOO_URDF if r["template_name"] == YOO else None):
        ir, = hssd200.load("unused")
    return ir


def test_hssd_urdf_kitchen_unit_under_its_rigid_modules_is_dropped():
    ir = hssd_kitchen()
    assert sorted(o["id"].split(":")[1] for o in ir["objects"]) == sorted(r[0] for r in RIGID)
    assert ir["meta"]["n_articulated_covered_dropped"] == 1 and ir["meta"]["n_articulated_twin_dropped"] == 0
    with patch.object(hssd200, "COVER", 1.01):          # the twin test alone keeps the 2.75 x 0.6 x 2.4 m unit
        kept = hssd_kitchen()
    unit = next(o for o in kept["objects"] if o["id"].endswith(":358"))
    assert unit["box_src"] == "urdf" and unit["size"] == pytest.approx([0.596, 2.750, 2.403], abs=1e-3)
    others = [o for o in kept["objects"] if o is not unit]
    covered = unary_union([fp(o) for o in others]).intersection(fp(unit)).area / fp(unit).area
    assert covered == pytest.approx(0.98, abs=0.01)                                  # audit: 98%


def test_hssd_cover_ignores_boxes_that_only_touch_it_in_height():
    box = lambda x, y0, y1: (Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]), y0, y1)
    a = box(0, 0.0, 0.9)
    assert hssd200._covered(a, [box(0, 0.0, 0.04)]) == 0          # rug under the cabinet
    assert hssd200._covered(a, [box(0, 0.9, 1.5)]) == 0            # item standing on it
    assert hssd200._covered(a, [box(0.25, 0.0, 0.9), box(-0.5, 0.5, 2.0)]) == pytest.approx(1.0)
    assert hssd200._covered(a, [box(0.5, 0.0, 0.9)]) == pytest.approx(0.5)


# ---- B7 tilted boxes: minimum-area rectangle instead of the box aligned to the heading --------------------------------

def tilted_corners_hull(b):
    """Exact footprint of an InternScenes bbox [x, y, z, sx, sy, sz, a_z, a_x, a_y] (all 8 corners projected)."""
    R = internscenes.rot_zxy(b[6], b[7], b[8])
    c = np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * b[3:6] @ R.T + b[:3]
    return MultiPoint(c[:, :2].tolist()).convex_hull


def internscenes_fp(b):
    f = internscenes.canon({"furniture_category": "x", "source_fields": {"bbox": b}})
    s = f["furniture_size"]
    return fp({"pos": b[:2], "size": [s["width"], s["length"]], "yaw": math.radians(f["furniture_rotation"]["z"])})


TOWEL = [-1.9512139360354914, 1.0812129739502754, 0.12443013014195747, 0.1893758247121573, 0.2873404697823525,
         0.8799594640731813, 1.1619592923914526, 0.7496078454017815, -1.8211352390902058]   # ScanNet scene0051_00 id 46
VANITY = [-2.0717500926174157, 1.7554297845851898, 0.38261919713306425, 0.38724171834135035, 0.795435510240674,
          0.7568351328372955, -0.009232779595207375, 0.0, -0.0]                                  # id 13
EXTINGUISHER = [0.9566598616696089, -0.5921732766785639, 0.2577081845654239, 0.1417363834345342, 0.1740769174752237,
                0.5463708639144897, 3.1292889293322186, -0.2273482908165158, 0.5709481267097896]  # scene0158_00 id 30
SHELF = [1.0179192620549233, -1.2329542017415072, 0.379255731410861, 0.3557287399669886, 1.0950513744046688,
         0.7689835764467716, 3.141592653589793, 0.0, -0.0]                                        # id 31


@pytest.mark.parametrize("tilted, other, eval_v32", [(TOWEL, VANITY, 0.0347), (EXTINGUISHER, SHELF, 0.022)])
def test_internscenes_compound_tilt_no_longer_inflates_into_a_collision(tilted, other, eval_v32):
    official = tilted_corners_hull(tilted).intersection(tilted_corners_hull(other)).area
    new = internscenes_fp(tilted).intersection(internscenes_fp(other)).area
    with patch.object(internscenes, "min_area_yaw", lambda xy, yaw: yaw):        # v3.2: aligned to the heading
        old = internscenes_fp(tilted).intersection(internscenes_fp(other)).area
    assert old == pytest.approx(eval_v32, abs=2e-3) and old > 0.01               # the checker's collision area
    assert official - 1e-9 <= new < 0.01
    assert internscenes_fp(tilted).buffer(1e-9).covers(tilted_corners_hull(tilted))   # still encloses the box


def test_tilted_bbox_keeps_heading_for_single_axis_tilt_and_turns_less_than_45_deg():
    R = internscenes.rot_zxy(0.7, math.radians(30), 0.0)        # tilted about local X only: projection is a rectangle
    assert internscenes.tilted_bbox(R, [1.0, 0.2, 0.1])[1] == pytest.approx(math.degrees(0.7), abs=1e-9)
    R = internscenes.rot_zxy(TOWEL[6], TOWEL[7], TOWEL[8])
    size, yaw, _, _ = internscenes.tilted_bbox(R, TOWEL[3:6])
    turn = (math.radians(yaw) - internscenes.heading(R) + math.pi) % (2 * math.pi) - math.pi
    assert 1e-3 < abs(turn) <= math.pi / 4


WINE = {"furniture_instance_id": "room_fe4e6dcc_wineglass_platform6b_86281a09", "furniture_category": "wineglass_platform6b",
        "furniture_position": {"x": 4.211807923423423, "y": 2.064914295035101, "z": 0.7018272653751505},
        "furniture_rotation": {"x": 45.262611412800354, "y": -4.468024297101949, "z": 95.02923156296241},
        "furniture_size": {"width": 0.11881484958977939, "length": 0.11888163890118929, "height": 0.2868252223265891}}
BOTTLE = {"furniture_instance_id": "room_fe4e6dcc_bottle_plat6_b6fe148a", "furniture_category": "bottle_plat6",
          "furniture_position": {"x": 4.196565783426864, "y": 2.2107897029742243, "z": 0.7151438920959152},
          "furniture_rotation": {"x": -27.268191713107004, "y": -29.913221669060796, "z": -134.20784092173906},
          "furniture_size": {"width": 0.13316753588115618, "length": 0.11767238081155011, "height": 0.3689287298312099}}


def sage_overlap():
    a, b = convert_room({"room_boundary": [[0, 0], [9, 0], [9, 9], [0, 9]], "furniture": [
        sage10k.upright_box(WINE), sage10k.upright_box(BOTTLE)]}, source="t", uid="t", group="g",
        front_offset_deg=90)["objects"]
    return fp(a).intersection(fp(b)).area


def test_sage_tilted_pair_of_layout_a0910029_no_longer_collides():
    with patch.object(sage10k, "min_area_yaw", lambda xy, yaw: yaw):
        assert sage_overlap() > 0.015                       # audit: GT 168 cm2, raw boxes 17 cm2
    assert sage_overlap() < 0.003


# ---- B8 MultiScan: every floor-standing object up to 0.5 m decides hull vs proxy -------------------------------------

SOFA_25 = {  # objects.csv, scene_00054_00 object 25 (its box starts 6.9 cm above the floor level, 9.4 cm outside)
    "object_id": "25", "category": "sofa", "is_architectural": "False", "is_opening": "False",
    "obb_center": "[2.897955020593432, -1.0247573199479159, -1.4977949857711794]",
    "obb_half_extents": "[1.049719694941359, 0.13954603672027616, 0.33666266317584587]",
    "obb_axes": "[0.15190213464362803, -0.9883955389876612, -2.194678969641786e-16, 0.0, -2.220446049250313e-16, 1.0, "
                "-0.9883955389876612, -0.15190213464362803, -3.37290494742133e-17]",
    "front": "[-0.9883955389876612, -0.15190213464362803, -3.37290494742133e-17]"}
REGION_54 = {"poly_loop": "[[-5.717621, 2.07195], [-4.668023, -4.231767], [3.395312, -2.889185], [2.345714, 3.414532]]",
             "wall_segments": json.dumps([{"start": [2.547813314390713, 3.5182531751594825],
                                           "end": [-4.869720605535319, 2.222872685309302]}])}
FLOOR_54 = -1.7062410116195683     # median bottom of its rests_on_floor objects


def test_multiscan_sofa_above_5cm_now_makes_the_room_proxy():
    sofa = multiscan.box(SOFA_25)
    sofa["furniture_position"]["z"] -= FLOOR_54
    assert sofa["furniture_position"]["z"] == pytest.approx(0.0689, abs=1e-4)
    assert multiscan.boundary(REGION_54, [sofa])[1:] == ("proxy", 1)
    assert multiscan.boundary(REGION_54, [sofa], floor_z=0.05)[1:] == ("hull", 0)       # v3.2's z < 0.05 rule
    sofa["furniture_position"]["z"] = 0.6                                                 # hanging: not tested
    assert multiscan.boundary(REGION_54, [sofa])[1:] == ("hull", 0)


# ---- B9 Structured3D: no nearest-assigned objects outside the room, only topology-linked openings --------------------

def s3d_obj(i, cat, method, pos, coeffs):
    return {"category": cat, "instance_id": i, "assignment_method": method, "position": dict(zip("xyz", pos)),
            "raw_basis": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "raw_coeffs": coeffs}


def s3d_rec(sample, house, room, poly, objs, doors=()):
    return {"sample_id": sample, "provenance": {"source_house_id": house, "split": "train"},
            "room_origin_house_m": {"x": 0, "y": 0, "z": 0},
            "room_context": {"room_id": room, "room_type": "balcony", "ceiling_height_m": 2.8, "floor_polygon": poly,
                             "doors": list(doors), "windows": []},
            "layout": {"floor_layout": {"objects": objs}}}


def test_structured3d_nearest_panel_outside_the_balcony_is_dropped():
    rec = s3d_rec("structured3d_00275_room_606054", "scene_00275", "606054",
                  [[-1.896194, -0.635749], [1.896011, -0.63613], [1.895998, 0.635941], [-1.896194, 0.635941]], [
                      s3d_obj(20, "table", "nearest", (-0.510036, -0.879079, 2.1985), [2380.865, 0.0001, 2198.5000416927]),
                      s3d_obj(123, "table", "inside", (-0.074014, -0.270939, 0.243432), [225.0, 225.0, 243.432])])
    ir = structured3d.convert(rec)
    assert [o["id"] for o in ir["objects"]] == ["123"] and ir["meta"]["n_nearest_outside_dropped"] == 1
    panel = structured3d._furniture(rec["layout"]["floor_layout"]["objects"][0])     # what v3.2 kept: 4.76 x 0 x 4.40 m
    assert panel["furniture_size"]["length"] == pytest.approx(4.76173)


def topology_tar(path):
    """Minimal annotation_3d of scene_01827: door 15 joins rooms 474 / 187, door 20 opens room 549 into 474."""
    planes = ["floor", "wall", "wall", "wall", "door", "door", "window"]  # walls 1: room 549, 2: 474, 3: 187
    lines = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1]]
    ann = {"scene_id": "scene_01827", "annotation_3d": {
        "planes": [{"ID": i, "type": t} for i, t in enumerate(planes)], "planeLineMatrix": lines,
        "semantics": [{"ID": 549, "type": "store room", "planeID": [0, 1]}, {"ID": 474, "type": "kitchen", "planeID": [2]},
                      {"ID": 187, "type": "hall", "planeID": [3]}, {"ID": 15, "type": "door", "planeID": [4]},
                      {"ID": 20, "type": "door", "planeID": [5]}, {"ID": 9, "type": "window", "planeID": [6]}]}}
    with tarfile.open(path, "w:gz") as t:
        for name, data in (("structure_topology/scene_01827.json", json.dumps(ann).encode()),
                           ("structure_topology/._scene_01827.json", b"\x00\x05\x16\x07 AppleDouble")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


def test_structured3d_door_of_two_other_rooms_is_not_attached(tmp_path):
    topology_tar(tmp_path / "t.tar.gz")
    links = structured3d.topology_links(str(tmp_path / "t.tar.gz"))
    assert links[("scene_01827", "549")] == {20} and links[("scene_01827", "474")] == {15, 20}
    assert links[("scene_01827", "187")] == {15}
    doors = [{"position": {"x": -1.2771, "y": -0.414551, "z": 1.0}, "width": 0.737891, "height": 2.0,
              "source_semantic_id": 15},
             {"position": {"x": -0.058155, "y": 0.414551, "z": 1.0}, "width": 1.5, "height": 2.0, "source_semantic_id": 20}]
    rec = s3d_rec("structured3d_01827_room_549", "scene_01827", "549",
                  [[0.808154, -0.364551], [-0.808155, -0.364551], [-0.808155, 0.364551], [0.808154, 0.364551]],
                  [s3d_obj(4, "cabinet", "inside", (0.004959, -0.045694, 1.25029), [812.3110471520031, 336.9225, 1250.29])],
                  doors)
    ir = structured3d.convert(rec, links)
    assert [o["id"] for o in ir["objects"]] == ["4", "door_1"] and ir["meta"]["n_openings_unlinked_dropped"] == 1
    assert [o["id"] for o in structured3d.convert(rec)["objects"]] == ["4", "door_0", "door_1"]   # without topology


def test_adapters_cli_refuses_to_overwrite_existing_ir(tmp_path):
    """The frozen v3.2/v3.3 IR must never be replaced by a v3.4 adapter run."""
    from fastfill import adapters
    frozen = tmp_path / "MultiScan.jsonl"
    frozen.write_text("frozen\n")
    argv = ["fastfill.adapters", "--root", str(tmp_path), "--out", str(tmp_path), "--only", "MultiScan"]
    with patch("sys.argv", argv), patch.object(multiscan, "load", side_effect=AssertionError("load called")):
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            adapters.main()
    assert frozen.read_text() == "frozen\n" and sorted(p.name for p in tmp_path.iterdir()) == ["MultiScan.jsonl"]
