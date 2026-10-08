"""vlm_judge projection math (plan and z-buffer agree with validation.footprint and local +X front) and anchors."""
import math

import numpy as np
from PIL import Image

from fastfill.v2.direct_layout import layout_to_roomgenbench, request_to_condition
from fastfill.v2.ops import vlm_judge as vj
from fastfill.v2.validation import _geometry, footprint, validate_scene


def room(n=1):
    condition = request_to_condition({"room_type": "living room", "room_size_m": [4., 3., 2.6],
                                      "furniture_list": [{"category": "sofa", "count": n}]})
    return {"condition": condition}


def box(yaw, ident="obj_0000", pos=(2.2, 1.3, 0.)):
    return {"id": ident, "target_size_local_m": [1.2, .6, .8], "bottom_center_m": list(pos), "yaw_rad": yaw}


def test_box_lands_where_validation_footprint_says(tmp_path):
    row, obj = room(), box(.7)
    expected = np.array(footprint(_geometry(obj, "target")).exterior.coords[:4])
    assert np.allclose(vj.corners_xy(obj["target_size_local_m"], obj["bottom_center_m"], obj["yaw_rad"]), expected)
    its, _ = vj.items(row, {"schema_version": "fastfill.v2", "objects": [obj]})
    vj.render_panel(tmp_path / "p.png", row, its)
    image = np.asarray(Image.open(tmp_path / "p.png").convert("RGB"))
    assert image.shape == (vj.PANEL_H, vj.PANEL_W, 3)
    x0, x1, y0, y1 = vj.plan_extent(row)
    c, s = math.cos(obj["yaw_rad"]), math.sin(obj["yaw_rad"])

    def pixel(lx, ly):  # local footprint coordinates -> plan pixel colour
        x, y = obj["bottom_center_m"][0] + c * lx - s * ly, obj["bottom_center_m"][1] + s * lx + c * ly
        return image[int((y1 - y) / (y1 - y0) * vj.PLAN_PX), int((x - x0) / (x1 - x0) * vj.PLAN_PX)]
    blue = vj.PLACE_RGB["floor"]
    assert np.abs(pixel(-.3, -.15).astype(int) - blue).max() <= 2  # inside, away from the label and arrow
    assert np.abs(pixel(.85, 0).astype(int) - blue).max() > 60  # 0.25 m in front: empty floor
    for eye, target in vj.cameras(row):  # z-buffer: the top centre shows the top face
        img, _ = vj.rasterize(vj.view_primitives(row, its, eye), eye, target)
        x, y, _ = vj.project([[*obj["bottom_center_m"][:2], .8]], eye, target)
        top = np.round(vj._shade(blue, (0, 0, 1)) * 255)
        assert np.abs(img[int(y[0]), int(x[0])].astype(int) - top).max() <= 1


def test_front_arrow_points_along_local_x():
    for yaw in (0., .7, math.pi / 2, -2.5, math.pi):
        obj = box(yaw)
        tail, head = vj.front_arrow(obj["target_size_local_m"], obj["bottom_center_m"], yaw)
        corners = vj.corners_xy(obj["target_size_local_m"], obj["bottom_center_m"], yaw)
        assert np.allclose(head - tail, .6 * np.array((math.cos(yaw), math.sin(yaw))))
        assert np.allclose(head, corners[1:3].mean(0))  # the front edge (local +X)


def test_anchors_are_deterministic_and_valid():
    row = room(4)
    gt = {"schema_version": "fastfill.v2", "objects": [box(.3 * i, f"obj_{i:04d}", (.8 + .8 * i, 1. + .2 * i, 0.)) for i in range(4)]}
    first, again, other = vj.anchors(row, 7, gt, 0), vj.anchors(row, 7, gt, 0), vj.anchors(row, 7, gt, 1)
    assert first == again and first["random"] != other["random"] and set(first) == set(vj.ANCHORS)
    assert first["identical"] == gt
    moved = [a["bottom_center_m"] != b["bottom_center_m"] for a, b in zip(first["shuffled"]["objects"], gt["objects"])]
    assert all(moved)
    assert sorted(o["bottom_center_m"] for o in first["shuffled"]["objects"]) == sorted(o["bottom_center_m"] for o in gt["objects"])
    assert "boundary" not in {c["code"] for c in validate_scene(row["condition"], first["random"]["objects"])["checks"]
                              if c["status"] == "violation"}
    mirrored = first["mirrored"]["objects"][1]
    assert np.isclose(mirrored["bottom_center_m"][0], 4. - gt["objects"][1]["bottom_center_m"][0])
    assert np.isclose(math.cos(mirrored["yaw_rad"]), -math.cos(.3)) and np.isclose(math.sin(mirrored["yaw_rad"]), math.sin(.3))
    for layout in first.values():
        layout_to_roomgenbench(row["condition"], layout)  # every anchor is a valid hand-off layout
