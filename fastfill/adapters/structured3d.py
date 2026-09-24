"""Structured3D (imChuling export, structured3d.jsonl: 22,894 rooms / 3,488 houses, official train/val/test).

Export (kits/convert_structured3d.py): meters, Z-up, room-local (floor polygon centroid, z from floor_z);
position = bbox_3d centroid (box center); the exported size/yaw come from a heuristic axis canonicalization
(row most aligned with world X), which discards the model frame. The raw bbox_3d fields are kept per object:
raw_basis rows = box axes in world, raw_coeffs = half extents (mm) along those rows. Row 2 is the model's up;
wall-contact statistics on beds / TVs / pictures / toilets put the back on raw +Y, so front = raw -Y.
Objects whose raw row 2 is not world +Z (~31k, almost all 'unknown'/'otherprop') are tilted relative to their
model pose -> tilted=True (raw_basis kept). Category 'unknown' (41%: occluded in every view, or the corrupt
perspective_full_09.zip for scenes 1600-1799) keeps its box with category "unknown": build.prep drops it as a
generic label and rejects the room only if dropped floor boxes leave a real hole (hidden_floor_obstacle).
room_type 'undefined' -> None.
Boundary = floor polygon traced from annotation_3d junctions/lines.
"""
import math
import os

import numpy as np

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "Structured3D"


def _furniture(o):
    B = np.array(o["raw_basis"], float)
    ext = 2 * np.abs(np.array(o["raw_coeffs"], float)) / 1000
    f = {"furniture_category": o["category"],
         "furniture_instance_id": str(o["instance_id"]), "furniture_position": o["position"],
         "furniture_size": {"width": ext[1], "length": ext[0], "height": ext[2]}, "raw_basis": o["raw_basis"]}
    if B[2, 2] < 0.999:                                   # model up is not world up
        f["furniture_rotation"] = {"x": None, "y": None, "z": 0.0}
    else:
        front = -B[1]
        f["furniture_rotation"] = {"x": 0, "y": 0, "z": math.degrees(math.atan2(front[1], front[0]))}
    return f


def load(root):
    path = os.path.join(root, "imChuling__3D_Room_Collections/Structured3D_exported/structured3d.jsonl")
    for rec in iter_records(path):
        rc, prov = rec["room_context"], rec["provenance"]
        objs = rec["layout"]["floor_layout"]["objects"]
        room = {"room_id": rc["room_id"], "room_type": None if rc["room_type"] == "undefined" else rc["room_type"],
                "room_boundary": rc["floor_polygon"], "room_height": rc["ceiling_height_m"],
                "furniture": [_furniture(o) for o in objs]}
        yield convert_room(room, source=SOURCE, uid=rec["sample_id"], group=f"structured3d:{prov['source_house_id']}",
                           subset=prov["split"], boundary_type="polygon", center_z=True,
                           extra=lambda f: {"raw_basis": f["raw_basis"]} if f["furniture_rotation"]["x"] is None else {},
                           meta={"front_known": True, "room_origin_house_m": rec["room_origin_house_m"],
                                 "n_nearest_assigned": sum(o.get("assignment_method") == "nearest" for o in objs),
                                 "n_unknown_category": sum(o["category"] == "unknown" for o in objs)})


if __name__ == "__main__":
    # bed rotated 90 deg: raw row0 = world +Y, row1 = world -X (back on raw +Y -> world -X) -> front world +X, yaw 0
    f = _furniture({"category": "bed", "instance_id": 1, "position": {"x": 0, "y": 0, "z": 0.5},
                    "raw_basis": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], "raw_coeffs": [800, 1000, 500]})
    assert abs(f["furniture_rotation"]["z"]) < 1e-9 and f["furniture_size"]["width"] == 2.0, f
    f = _furniture({"category": "unknown", "instance_id": 2, "position": {"x": 0, "y": 0, "z": 1},
                    "raw_basis": [[0, 0, -1], [0, 1, 0], [1, 0, 0]], "raw_coeffs": [1, 1, 1]})
    assert f["furniture_category"] == "unknown" and f["furniture_rotation"]["x"] is None
    print("structured3d.py self-check ok")
