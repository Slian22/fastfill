"""Source-shaped IL3D regressions; no mounted data or frozen release is required.

Coordinates are from 3DFront_exported/3dfront.jsonl and the 2026-09-28
designer/opening-target-join.json audit. See the 20261013 correction report.
Profiles of the arch, rounding and small-angle fixtures are verbatim export rows.
"""
import copy
import math

import pytest
from shapely.geometry import Polygon

from fastfill.adapters import il3d_3dfront as il
from fastfill.adapters.unified import convert_room


# house / room / window index, horizontal endpoints, bottom/top, exported width.
# First case was also replayed from raw Window constructid 523022 + WallInner.
SOURCE_WINDOWS = [
    ("71b4daf5/SecondBedroom-198098/2", (-0.01947, 4.67276), (0.24187, 4.40848),
     0.9, 2.4, 0.26428),
    ("cbce76fc/LivingDiningRoom-515/3", (2.04573, -8.88648), (3.08161, -7.94477),
     0.19999, 2.50001, 1.03588),
    ("258b12fb/LivingDiningRoom-35273/0", (-14.85383, 3.09106), (-9.68166, 6.07722),
     0.01997, 3.30003, 5.17217),
    ("eb1f5725/SecondBedroom-25551/0", (1.94122, 4.47898), (2.53029, 3.81648),
     0.25, 2.25, 0.6625),
    ("eb1f5725/SecondBedroom-25551/1", (0.27419, 3.83604), (0.8903, 4.45321),
     0.25, 2.25, 0.61717),
]


# Real arched window: 3dfront.jsonl line 40854 (byte offset 269037452), house
# ba1ea4ea-d448-4abb-a7fd-de1ae548f136 KidsRoom-194027 window_0 (v3.2 test split).
# Its outline touches itself at mm scale near (y, z) = (-2.771, 2.447).
ARCH = {"opening_type": "Window", "wall_id": None, "width": 1.78824, "height": 1.86953,
        "sill_height": 0.57779, "position": {"x": -3.49246, "y": -2.75924, "z": 1.512555}}
ARCH_YZ = [  # (y, z) of all 115 points; x = -3.49246 for every point
    (-1.86512, 0.57779), (-1.92588, 0.57779), (-1.92992, 0.57779), (-2.03426, 0.57779),
    (-3.58855, 0.57779), (-3.5926, 0.57779), (-3.65336, 0.57779), (-3.65336, 0.58019),
    (-3.65336, 0.61644), (-3.5926, 0.61644), (-3.58856, 0.61644), (-3.58805, 0.7176),
    (-3.58805, 0.73274), (-3.58767, 0.76978), (-3.58767, 0.77539), (-3.58729, 0.97308),
    (-3.58729, 1.00274), (-3.58691, 1.03987), (-3.58679, 1.04548), (-3.58464, 1.2405),
    (-3.58439, 1.2698), (-3.58401, 1.30765), (-3.58388, 1.31326), (-3.58173, 1.50881),
    (-3.58148, 1.5382), (-3.58098, 1.57596), (-3.58098, 1.58157), (-3.57883, 1.77864),
    (-3.57858, 1.8082), (-3.5782, 1.84044), (-3.5782, 1.84534), (-3.57175, 1.89467),
    (-3.57087, 1.90206), (-3.56026, 1.95656), (-3.55862, 1.96475), (-3.53904, 2.01427),
    (-3.53613, 2.02175), (-3.51061, 2.07126), (-3.5067, 2.07874), (-3.46615, 2.14072),
    (-3.46008, 2.15007), (-3.40715, 2.20662), (-3.39919, 2.21508), (-3.3522, 2.25497),
    (-3.34513, 2.26103), (-3.29334, 2.29629), (-3.28563, 2.30155), (-3.22929, 2.33423),
    (-3.22083, 2.33912), (-3.15956, 2.36531), (-3.15034, 2.36922), (-3.0729, 2.3979),
    (-3.06128, 2.40217), (-2.97891, 2.42257), (-2.96653, 2.42559), (-2.89453, 2.43655),
    (-2.88367, 2.43824), (-2.80914, 2.44465), (-2.79802, 2.44563), (-2.76429, 2.44705),
    (-2.79713, 2.44732), (-2.72551, 2.44581), (-2.72046, 2.44563), (-2.64592, 2.43922),
    (-2.63481, 2.43824), (-2.56268, 2.4272), (-2.55194, 2.42559), (-2.46958, 2.4052),
    (-2.4572, 2.40217), (-2.37976, 2.3735), (-2.36814, 2.36922), (-2.30687, 2.34304),
    (-2.29765, 2.33912), (-2.24131, 2.30644), (-2.23284, 2.30155), (-2.18105, 2.26628),
    (-2.17335, 2.26103), (-2.12635, 2.22104), (-2.11928, 2.21508), (-2.06635, 2.15853),
    (-2.05839, 2.15007), (-2.01784, 2.088), (-2.01178, 2.07874), (-1.98613, 2.02914),
    (-1.98234, 2.02175), (-1.96276, 1.97215), (-1.95986, 1.96475), (-1.94925, 1.91026),
    (-1.9476, 1.90206), (-1.94129, 1.85273), (-1.94028, 1.84534), (-1.94003, 1.81301),
    (-1.94003, 1.8082), (-1.93788, 1.61113), (-1.9375, 1.58157), (-1.93712, 1.54381),
    (-1.93699, 1.5382), (-1.93485, 1.34264), (-1.93459, 1.31326), (-1.93421, 1.2755),
    (-1.93409, 1.2698), (-1.93194, 1.07469), (-1.93169, 1.04548), (-1.93118, 1.00826),
    (-1.93118, 1.00274), (-1.9308, 0.80505), (-1.9308, 0.77539), (-1.93042, 0.73826),
    (-1.93042, 0.73274), (-1.93005, 0.63157), (-1.92992, 0.61644), (-1.90971, 0.61644),
    (-1.90567, 0.61644), (-1.86512, 0.61644), (-1.86512, 0.61403),
]

# Real 5-decimal windows whose span differs from the export width only at the
# rounding level (lines 48 and 152): export width and historical angle stay.
ROUNDING_WINDOWS = [
    ("Bedroom-7177/0", 1.63119, 0.9, 1.5, 0.0,
     [[4.20662, 1.52178, 0.9], [4.20662, 1.52178, 2.4], [4.20662, -0.10941, 2.4], [4.20663, -0.10941, 0.9]]),
    ("SecondBedroom-33125/0", 1.81701, 0.50001, 1.59998, 89.89625653761121,
     [[-2.44575, 4.96255, 0.50001], [-2.44575, 4.96255, 2.09999],
      [-0.62874, 4.95926, 2.09999], [-0.62874, 4.95926, 0.50001]]),
]


def window(a, b, bottom, top, width):
    return {"opening_type": "Window", "width": width, "height": top - bottom,
            "position": {"x": (a[0] + b[0]) / 2, "y": (a[1] + b[1]) / 2,
                         "z": (bottom + top) / 2}, "sill_height": bottom,
            "profile": [[*a, bottom], [*a, top], [*b, top], [*b, bottom]]}


def emitted(opening, kind="window", shift=(0, 0)):
    groups = ([opening], []) if kind == "door" else ([], [opening])
    objects, skipped = il.openings(groups, shift, "room")
    assert skipped == 0 and len(objects) == 1
    return objects[0]


@pytest.mark.parametrize("case,a,b,bottom,top,width", SOURCE_WINDOWS,
                         ids=[case[0] for case in SOURCE_WINDOWS])
def test_real_slanted_windows_use_wall_span(case, a, b, bottom, top, width):
    source = window(a, b, bottom, top, width)
    before = copy.deepcopy(source)
    obj = emitted(source, shift=(2, -3))
    assert obj["furniture_size"] == pytest.approx(
        {"width": il.OPENING_T, "length": math.dist(a, b), "height": top - bottom})
    yaw = math.radians(obj["furniture_rotation"]["z"] - 90)
    assert abs(math.cos(yaw) * (b[1] - a[1]) - math.sin(yaw) * (b[0] - a[0])) < 1e-10
    assert obj["furniture_instance_id"] == "room::window_0"
    assert obj["furniture_position"] == pytest.approx(
        {"x": source["position"]["x"] - 2, "y": source["position"]["y"] + 3, "z": bottom})
    assert obj["structure"] and source == before


@pytest.mark.parametrize("a,b,width", [((7.4, 5), (8.6, 5), 1.2), ((2, 3), (2, 4.2), 1.2)])
def test_axis_aligned_window_keeps_export_precision(a, b, width):
    obj = emitted(window(a, b, 0.9, 2.4, width))
    assert obj["furniture_size"]["length"] == width
    assert obj["furniture_position"]["z"] == 0.9


@pytest.mark.parametrize("reverse,closed", [(False, False), (True, False), (False, True)])
def test_subdivided_vertical_profile_uses_full_span_not_longest_edge(reverse, closed):
    source = window((0, 0), (0.6, 0.8), 0.9, 3.4, 0.8)
    profile = [[0, 0, 0.9], [0, 0, 3.4], [0.3, 0.4, 3.4],
               [0.6, 0.8, 3.4], [0.6, 0.8, 0.9], [0.3, 0.4, 0.9]]
    if reverse:
        profile = profile[::-1]
    if closed:
        profile = profile + [profile[0]]
    obj = emitted({**source, "profile": profile})
    assert obj["furniture_size"]["length"] == pytest.approx(1.0)
    assert obj["furniture_size"]["height"] == 2.5


def test_rounded_subdivisions_use_full_extent_for_plane_direction():
    _, a, b, bottom, top, width = SOURCE_WINDOWS[0]
    xy = [[round(a[k] + (b[k] - a[k]) * t / 10, 5) for k in range(2)] for t in range(11)]
    profile = [[*p, top] for p in xy] + [[*p, bottom] for p in xy[::-1]]
    source = {**window(a, b, bottom, top, width), "profile": profile}
    assert emitted(source)["furniture_size"]["length"] == pytest.approx(math.dist(a, b))


@pytest.mark.parametrize("vector", [(2, 0), (-2, 0), (0, 2), (0, -2), (1.2, 1.6), (-1.2, -1.6)])
def test_full_extent_keeps_legacy_directed_wall_axis_exactly(vector):
    # The subdivided lower edge runs A->B, but only the full upper B->A edge
    # matches the exported width. The two axes bound the same geometry.
    a, b = (3.0, 4.0), (3.0 + vector[0], 4.0 + vector[1])
    source = window(a, b, 0.9, 2.4, max(abs(v) for v in vector))
    bottom = [[a[k] + vector[k] * t / 3 for k in range(2)] + [0.9] for t in range(4)]
    source = {**source, "profile": bottom + [[*b, 2.4], [*a, 2.4]]}
    expected_angle = math.atan2(a[1] - b[1], a[0] - b[0])
    angle, width, _ = il._opening_frame("window", source["profile"], source["width"])
    assert angle == expected_angle
    assert width == pytest.approx(math.dist(a, b))
    obj = emitted(source)
    assert obj["furniture_rotation"]["z"] == math.degrees(expected_angle) + 90
    ir = convert_room({"furniture": [obj]}, source=il.SOURCE, uid="room", group="house", front_offset_deg=-90)
    expected_yaw = math.radians((math.degrees(expected_angle) + 90) - 90) % (2 * math.pi)
    assert ir["objects"][0]["yaw"] == expected_yaw


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("epsilon", [3e-13, 1e-5], ids=["same-axis", "genuine-angle-change"])
def test_near_axis_preserves_exact_angle_only_below_tolerance(sign, epsilon):
    a, middle, b = (0, 0), (sign * 1.1, sign * 1.1), (sign * 2, sign * (2 + epsilon))
    source = window(a, b, 0.9, 2.4, 2 + epsilon)
    source = {**source, "profile": [[*a, 0.9], [*middle, 0.9], [*b, 0.9], [*b, 2.4], [*a, 2.4]]}
    # The A->middle edge is nearest exported width. Full-span direction differs
    # only by rounding in one case, and by real source geometry in the other.
    old_angle = math.atan2(middle[1], middle[0])
    full_angle = math.atan2(b[1], b[0])
    angle, width, _ = il._opening_frame("window", source["profile"], source["width"])
    assert angle == (old_angle if epsilon < 1e-12 else full_angle)
    assert width == pytest.approx(math.dist(a, b))


@pytest.mark.parametrize("kind", ["door", "window"])
@pytest.mark.parametrize("angle", [0, math.pi / 2, math.pi / 4])
@pytest.mark.parametrize("width,depth", [(0.9, 0.24), (0.2, 0.8)])
def test_horizontal_footprints_retain_width_edge_and_depth(kind, angle, width, depth):
    # Deep/narrow footprints prove that longest-edge replacement is unsafe.
    c, s = math.cos(angle), math.sin(angle)
    profile = [[5 + c*x - s*y, 7 + s*x + c*y]
               for x, y in [(-width/2, -depth/2), (width/2, -depth/2),
                            (width/2, depth/2), (-width/2, depth/2)]]
    source = {"width": width, "height": 2.1, "sill_height": None,
              "position": {"x": 5, "y": 7, "z": 4.05}, "profile": profile,
              "listed_on": "neighbour"}
    before = copy.deepcopy(source)
    obj = emitted(source, kind, shift=(5, 5))
    assert obj["furniture_size"] == pytest.approx({"width": depth, "length": width, "height": 2.1})
    assert obj["furniture_position"] == pytest.approx({"x": 0, "y": 2, "z": 3})
    assert obj["furniture_rotation"]["z"] == pytest.approx(math.degrees(angle) + 90)
    assert obj["listed_on"] == "neighbour" and source == before


def test_nonplanar_3d_profile_does_not_claim_wall_span():
    source = window((0, 0), (0.6, 0.8), 0.9, 2.4, 0.8)
    source = {**source, "profile": [[0, 0, 0.9], [0.1, 0, 2.4], [0.6, 0.8, 2.4], [0.6, 0.8, 0.9]]}
    assert emitted(source)["furniture_size"]["length"] == 0.8


def test_real_self_touching_arch_is_kept_with_export_width():
    profile = [[-3.49246, y, z] for y, z in ARCH_YZ]
    outline = Polygon(ARCH_YZ)
    assert len(profile) == 115 and not outline.is_valid and outline.area == pytest.approx(2.862, abs=1e-3)
    source = {**ARCH, "profile": profile}
    obj = emitted(source)
    # Same bytes as the historical adapter: the full span equals the export width.
    assert obj["furniture_size"] == {"width": il.OPENING_T, "length": 1.78824, "height": 1.86953}
    assert obj["furniture_position"] == {"x": -3.49246, "y": -2.75924, "z": 0.57779}
    assert obj["furniture_rotation"] == {"x": 0.0, "y": 0.0, "z": 0.0}


@pytest.mark.parametrize("case,width,bottom,height,rotation,profile", ROUNDING_WINDOWS,
                         ids=[case[0] for case in ROUNDING_WINDOWS])
def test_rounding_level_span_keeps_export_width_and_angle(case, width, bottom, height, rotation, profile):
    source = {"width": width, "height": height, "sill_height": bottom, "profile": profile,
              "position": {"x": 0, "y": 0, "z": bottom + height / 2}}
    obj = emitted(source)
    assert obj["furniture_size"]["length"] == width
    assert obj["furniture_rotation"]["z"] == rotation


def test_genuine_small_wall_angle_beyond_rounding_is_corrected():
    # Kitchen-54151 window_0 (line 23796): a 0.41 deg wall, span - width = 2.6e-5 > rounding.
    profile = [[4.4583, 5.64685, 0.85], [4.4583, 5.64685, 1.196], [4.4583, 5.64685, 1.998],
               [4.4583, 5.64685, 2.55], [5.00717, 5.6508, 2.55], [5.45873, 5.65405, 2.55],
               [5.45873, 5.65405, 1.998], [5.45873, 5.65405, 1.196], [5.45873, 5.65405, 0.85],
               [5.00717, 5.6508, 0.85]]
    source = {"width": 1.00043, "height": 1.7, "sill_height": 0.85, "profile": profile,
              "position": {"x": 4.958515, "y": 5.65045, "z": 1.7}}
    length = emitted(source)["furniture_size"]["length"]
    assert length == pytest.approx(math.dist((4.4583, 5.64685), (5.45873, 5.65405)), abs=1e-12)
    assert length - source["width"] > il.OPENING_ROUND_TOL


@pytest.mark.parametrize("profile", [None, [], [None], [[0]], [[0, 0]] * 4,
    [[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 0, 2], [1, 1]],
    [[0, 0], [1, 0], [1, float("nan")]], [[0, 0], [1, 0], [1, float("inf")]],
    [[0, 0], [1, 0], [1, "bad"]], [[0, 0], [1, 0], [True, 1]],
    [[0, 0], [1, 0], [10**400, 1]], [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 0, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]])
def test_degenerate_profiles_are_skipped(profile):
    source = {**window((0, 0), (0.6, 0.8), 0.9, 2.4, 0.8), "profile": profile}
    assert il.openings(([], [source]), (0, 0), "room") == ([], 1)


@pytest.mark.parametrize("field,value", [("position", None), ("width", 0), ("width", float("inf")),
    ("height", 0), ("height", 0.05), ("height", float("nan")),
    ("width", True), ("height", 10**400)])
def test_invalid_dimensions_and_sill_slabs_are_skipped(field, value):
    source = {**window((0, 0), (0.6, 0.8), 0.9, 2.4, 0.8), field: value}
    assert il.openings(([], [source]), (0, 0), "room") == ([], 1)


def test_corrected_dimensions_reach_ir_and_merging_keeps_original_index():
    small = window((0.15, 0.2), (0.45, 0.6), 0.9, 2.4, 0.4)
    large = window((0, 0), (0.6, 0.8), 0.9, 2.4, 0.8)
    source = ([], [small, large])
    before = copy.deepcopy(source)
    objects, skipped = il.openings(source, (0, 0), "room")
    assert skipped == 1 and len(objects) == 1
    ir = convert_room({"furniture": objects, "room_boundary": [[0, 0], [3, 0], [3, 3], [0, 3]]},
                      source=il.SOURCE, uid="room", group="house", front_offset_deg=-90, extra=il.il3d_extra)
    obj = ir["objects"][0]
    assert obj["id"] == "room::window_1" and obj["structure"]
    assert obj["size"] == pytest.approx([1, il.OPENING_T, 1.5])
    assert obj["pos"] == pytest.approx([0.3, 0.4, 0.9])
    assert obj["yaw"] == pytest.approx(math.atan2(0.8, 0.6))
    assert source == before


@pytest.mark.parametrize("second_bottom", [1.52, 1.8])
def test_wider_footprints_do_not_merge_windows_at_separate_heights(second_bottom):
    first = window((0, 0), (0.6, 0.8), 0.9, 1.5, 0.8)
    second = window((0.27, 0.36), (0.87, 1.16), second_bottom, second_bottom + 0.6, 0.8)
    objects, skipped = il.openings(([], [first, second]), (0, 0), "room")
    assert skipped == 0
    assert {o["furniture_instance_id"] for o in objects} == {"room::window_0", "room::window_1"}


@pytest.mark.parametrize("gap", [0, 0.005, 0.01])
def test_touching_or_near_window_pieces_merge_as_one_opening(gap):
    # Bay glass on its slab is one opening, as in the historical adapter. The export
    # rounds height independently: 0.9 + 0.8 > 1.7 in floating point.
    first = {**window((0, 0), (0.6, 0.8), 0.9, 1.7, 0.8), "height": 0.8}
    second = {**window((0.27, 0.36), (0.87, 1.16), 1.7 + gap, 2.5, 0.8), "height": 0.8 - gap}
    objects, skipped = il.openings(([], [first, second]), (0, 0), "room")
    assert skipped == 1 and [o["furniture_instance_id"] for o in objects] == ["room::window_0"]


def test_load_preserves_source_room_join_neighbour_door_and_floor_reference(monkeypatch, tmp_path):
    ring = [[5, 5], [9, 5], [9, 8], [5, 8]]
    door = {"width": 0.9, "height": 2.1, "position": {"x": 6.45, "y": 8.12, "z": 4.05},
            "profile": [[6, 8], [6.9, 8], [6.9, 8.24], [6, 8.24]]}
    upstairs = {**door, "position": {**door["position"], "z": 7.05}}
    win = window((5, 5), (5.6, 5.8), 3.9, 5.4, 0.8)
    rooms = [("own", ring, 3, [("asset", 6, 6)], ([], [win])),
             ("neighbour", ring, 3, [], ([door, upstairs], []))]
    furniture = {"furniture_asset_id": "asset", "furniture_instance_id": "chair",
                 "furniture_category": "chair", "furniture_position": {"x": 1, "y": 1, "z": 3},
                 "furniture_rotation": {"x": 0, "y": 0, "z": 0},
                 "source_fields": {"layout_object": {"scale": [1, 1, 1]}}}
    scene = {"scene_id": "scene", "rooms": [{"furniture": [furniture], "room_height": 3,
             "source_fields": {"original_house_file": "house.json", "original_roomId": "own"}}]}
    before = copy.deepcopy((rooms, scene))
    house = tmp_path / il.HOUSES / "house.json"
    house.parent.mkdir(parents=True)
    house.write_text("{}")
    monkeypatch.setattr(il, "load_assets", lambda root: {"asset": {"meta_data": {
        "width": 0.5, "length": 1, "height": 0.5}}})
    monkeypatch.setattr(il, "export_rooms", lambda root: {"house": rooms})
    monkeypatch.setattr(il, "floor_polygons", lambda path: {"own": (ring, 3)})
    monkeypatch.setattr(il, "iter_records", lambda path: iter([scene]))
    ir, = il.load(tmp_path)
    assert ir["uid"] == "il3d:scene" and ir["group"] == "3dfront:house"
    assert ir["meta"]["source_room_id"] == "own"
    assert ir["meta"]["n_doors_listed_on_neighbour"] == 1
    d, w = ir["objects"][1:]
    assert d["id"] == "il3d:scene::door_0" and d["listed_on"] == "neighbour"
    assert d["pos"] == pytest.approx([1.45, 3.12, 3])
    assert d["size"] == pytest.approx([0.9, 0.24, 2.1])
    assert w["id"] == "il3d:scene::window_0"
    assert w["pos"] == pytest.approx([0.3, 0.4, 3.9])
    assert w["size"] == pytest.approx([1, il.OPENING_T, 1.5])
    assert (rooms, scene) == before
