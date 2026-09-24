"""HSSD-200 (XXXpilar/hssd_clean rebuild of hssd-hab): 168 designer scenes, 2,351 annotated regions.

Source (README / dataset_manifest.json): meters, Y-up, right-handed, scene-global. objects.csv gives the tight
OBB of every rigid instance: obb_center (box center), obb_half_extents (already scaled) along the asset .glb
axes, obb_rotation_wxyz (world <- glb). regions.csv gives the official region polygon in XZ, floor/ceiling height
and the human region label. Converted here to Z-up by (x, y, z) -> (x, -z, y), z measured from the region floor.

Object frame: the glb axis that maps to world up is the asset's up. Wall-contact statistics (front test on
beds/toilets/cabinets/wardrobes, also per template) show two authoring frames: up=+Y with front=+Z, and the
same frame turned -90 deg about X (up=-Z, front=+Y). Other up axes (33 objects) have no known front -> yaw
missing (incomplete); objects whose no axis is vertical (94: books, boxes, pillows) are tilted.
Articulated instances (3,626 cabinets, wardrobes, fridges, ...) have no box and no region in the rebuild;
they are assigned to regions by their translation and counted as incomplete, so those rooms are rejected
by build instead of silently losing furniture. is_architectural instances (doors, windows, stairs ...) are kept
as "structure" objects (never placed; a floor staircase rejects the room in build); boxless ones are dropped.
Category 'unknown' (lexicon id 0) is a missing category -> incomplete. Boxes lying wholly below the region floor
belong to a lower storey without a region at that XY (the source assigns by XY alone) -> dropped, counted in meta.
"""
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from shapely.geometry import Point, Polygon

from fastfill.adapters.unified import convert_room

SOURCE = "HSSD200"
T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])      # Y-up -> Z-up
FRONT = {(1, 1): (2, 1), (2, -1): (1, 1)}              # (up axis, sign) -> (front axis, sign) in glb frame


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _furniture(r, floor):
    """objects.csv row -> unified-schema furniture dict (Z-up, center z above the region floor)."""
    cat = None if r["category"].startswith("unknown") else r["category"] or None   # lexicon id 0 = no category
    f = {"furniture_category": cat, "furniture_instance_id": f"{r['scene_id']}:{r['instance_id']}",
         "furniture_position": None, "furniture_rotation": None, "furniture_size": None}
    if not r["obb_center"]:
        return f                                         # articulated / missing template: no box
    C = T @ _rot(json.loads(r["obb_rotation_wxyz"]))
    ext = 2 * np.array(json.loads(r["obb_half_extents"]))
    c = T @ np.array(json.loads(r["obb_center"]))
    k = int(np.argmax(abs(C[2])))
    f["furniture_position"] = {"x": c[0], "y": c[1], "z": c[2] - floor}
    if abs(C[2, k]) < 0.999:                             # no vertical axis: tilted
        f["furniture_rotation"] = {"x": None, "y": None, "z": 0.0}
        f["furniture_size"] = {"width": ext[0], "length": ext[1], "height": ext[2]}
        f["rot_wxyz_yup"] = json.loads(r["obb_rotation_wxyz"])   # full pose kept; size is along glb axes
        return f
    fr = FRONT.get((k, int(np.sign(C[2, k]))))
    if fr is None:                                       # unknown authoring frame: front unknown
        f["furniture_rotation"] = {"x": 0, "y": 0, "z": None}
        f["furniture_size"] = {"width": ext[0], "length": ext[1], "height": ext[2]}
        return f
    a, s = fr
    o = 3 - a - k
    d = s * C[:, a]
    f["furniture_rotation"] = {"x": 0, "y": 0, "z": math.degrees(math.atan2(d[1], d[0]))}
    f["furniture_size"] = {"width": float(ext[a]), "length": float(ext[o]), "height": float(ext[k])}
    return f


def load(root):
    base = os.path.join(root, "XXXpilar__hssd_clean")
    regions = {}
    for g in csv.DictReader(open(os.path.join(base, "regions.csv"))):
        g["poly"] = [[x, -z] for x, z in json.loads(g["poly_loop_xz"])]
        regions[(g["scene_id"], g["region_id"])] = g
    by_scene = defaultdict(list)
    for key, g in regions.items():
        by_scene[key[0]].append((key, Polygon(g["poly"]), float(g["floor_height"]), float(g["ceiling_height"])))

    furn, n_arch, n_below = defaultdict(list), defaultdict(int), defaultdict(int)
    for r in csv.DictReader(open(os.path.join(base, "objects.csv"))):
        if r["region_source"] in ("inside", "nearest"):
            keys = [(r["scene_id"], r["region_id"])]
            # the source assigns by XY only when one polygon matches: boxes wholly below this region's
            # floor stand on a lower storey that has no region there -> not in this room
            if r["obb_center"] and json.loads(r["aabb_max"])[1] < float(regions[keys[0]]["floor_height"]) - 0.05:
                n_below[keys[0]] += 1
                continue
        elif not r["obb_center"]:                        # boxless instance: locate by translation
            x, y, z = json.loads(r["translation"])
            keys = [k for k, p, fl, ce in by_scene[r["scene_id"]] if p.contains(Point(x, -z)) and fl - 0.3 <= y <= ce]
        else:
            continue                                     # outdoor props outside every region
        for key in keys:
            f = _furniture(r, float(regions[key]["floor_height"]))
            if r["is_architectural"] == "True":
                # kept as "structure" (never placed) so build sees floor stairs; without a box or pose it cannot
                # be checked and would read as an incomplete object: dropped and counted, as before
                if f["furniture_rotation"] is None or f["furniture_rotation"]["z"] is None:
                    n_arch[key] += 1
                    continue
                f["furniture_category"], f["structure"] = r["category"], True
            furn[key].append(f)

    for key, g in regions.items():
        room = {"room_id": f"{key[0]}:{key[1]}", "room_type": g["room_type"], "room_boundary": g["poly"],
                "room_height": float(g["extrusion_height"]), "furniture": furn[key]}
        yield convert_room(room, source=SOURCE, uid=f"{SOURCE}:{key[0]}:{key[1]}", group=f"hssd:{key[0]}",
                           boundary_type="polygon", center_z=True,
                           extra=lambda f: {**({"rot_wxyz_yup": f["rot_wxyz_yup"]} if "rot_wxyz_yup" in f else {}),
                                            **({"structure": True} if f.get("structure") else {})},
                           meta={"region_name": g["region_name"], "floor_height": float(g["floor_height"]),
                                 "n_architectural_dropped": n_arch[key], "n_below_floor_dropped": n_below[key],
                                 "front_known": True})


if __name__ == "__main__":
    # glb frame turned -90 deg about X (up=-Z), yawed 90 deg in the scene: front must point to world +Y (Z-up)
    q_x = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]                # Rx(+90): glb -Z -> +Y up
    R = _rot([math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]) @ _rot(q_x)  # then yaw 90 about +Y
    w = math.sqrt(1 + np.trace(R)) / 2
    q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    row = {"category": "toilet", "scene_id": "s", "instance_id": "0", "obb_center": "[1, 0.4, -2]",
           "obb_half_extents": "[0.2, 0.35, 0.4]", "obb_rotation_wxyz": json.dumps(q)}
    f = _furniture(row, 0.0)
    # canonical front +Z yawed 90 about +Y -> Habitat +X -> Z-up +X; sizes: front 0.7, side 0.4, up 0.8
    assert abs(f["furniture_rotation"]["z"]) < 1e-6, f
    assert np.allclose([f["furniture_size"][k] for k in ("width", "length", "height")], [0.7, 0.4, 0.8]), f
    assert np.allclose([f["furniture_position"][k] for k in "xyz"], [1, 2, 0.4]), f
    print("hssd200.py self-check ok")
