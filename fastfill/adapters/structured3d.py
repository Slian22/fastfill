"""Structured3D (imChuling export, structured3d.jsonl: 22,894 rooms / 3,488 houses, official train/val/test).

Export (kits/convert_structured3d.py): meters, Z-up, room-local (floor polygon centroid, z from floor_z);
position = bbox_3d centroid (box center); the exported size/yaw come from a heuristic axis canonicalization
(row most aligned with world X), which discards the model frame. The raw bbox_3d fields are kept per object:
raw_basis rows = box axes in world, raw_coeffs = half extents (mm) along those rows. Row 2 is the model's up;
wall-contact statistics on beds / TVs / pictures / toilets put the back on raw +Y, so front = raw -Y.
Objects whose raw row 2 is not world +Z (~31k, almost all 'unknown'/'otherprop') are tilted relative to their
model pose -> tilted=True (raw_basis kept), boxed by the upright envelope of their rotated corners. Yaw follows
raw -Y when horizontal; when that front is near vertical, use raw X and mark front_known=False.
Category 'unknown' (41%: occluded in every view, or the corrupt
perspective_full_09.zip for scenes 1600-1799) keeps its box with category "unknown": build.prep drops it as a
generic label and rejects the room only if dropped floor boxes leave a real hole (hidden_floor_obstacle).
room_type 'undefined' -> None.
Boundary = floor polygon traced from annotation_3d junctions/lines.
Doors/windows: room_context.doors/windows (same room frame, position = opening centre on the wall centreline,
~0.1 m outside the floor polygon; width, height, no direction/thickness) -> structure boxes, see `openings`.
Room membership: the export gives a room every object, door and window whose centre lies within 0.5 m of it when the
centre is in no room (assignment_method "nearest"). Such objects (13,921 of 13,977 nearest-assigned have their centre
outside the room; e.g. 00275_room_606054 bbox 20, a 4.76 x 0.00 x 4.40 m "table" poking 1.02 m) are dropped, and a door
or window is kept only if the official topology links it to the room's walls (`topology_links`: 175 of 42,569 doors and
408 of 23,138 windows are not, e.g. 01827_room_549 door semantic 15 of rooms 474 / 187); both counted in meta.
"""
import json
import math
import os
import tarfile

import numpy as np
from shapely.geometry import Point, Polygon

from fastfill.adapters.internscenes import tilted_bbox
from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "Structured3D"
DIR = "imChuling__3D_Room_Collections/Structured3D_exported"


def _furniture(o):
    B = np.array(o["raw_basis"], float)
    ext = 2 * np.abs(np.array(o["raw_coeffs"], float)) / 1000
    f = {"furniture_category": o["category"],
         "furniture_instance_id": str(o["instance_id"]), "furniture_position": o["position"],
         "furniture_size": {"width": ext[1], "length": ext[0], "height": ext[2]}, "raw_basis": o["raw_basis"]}
    if B[2, 2] < 0.999:                                   # model up is not world up
        R = np.stack([-B[1], B[0], B[2]], axis=1)          # semantic front first
        size, yaw, k, _ = tilted_bbox(R, ext[[1, 0, 2]])
        return {**f, "furniture_size": dict(zip(("width", "length", "height"), size)),
                "furniture_rotation": {"x": None, "y": None, "z": yaw},
                **({"front_known": False} if k == 0 else {})}
    front = -B[1]
    return {**f, "furniture_rotation": {"x": 0, "y": 0, "z": math.degrees(math.atan2(front[1], front[0]))}}


OPENING_T = 0.10     # opening box thickness (the export gives none)
SAME_M = 0.05        # an object box centred this close (xy and z) to an opening, same size, is that opening's box


def _edge_deg(poly, x, y):
    """Direction (deg) of the floor-polygon edge nearest to (x, y)."""
    def dist(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = min(1.0, max(0.0, ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy or 1)))
        return math.hypot(x - a[0] - t * dx, y - a[1] - t * dy)
    a, b = min(zip(poly, poly[1:] + poly[:1]), key=lambda e: dist(*e))
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def openings(rc, objs, linked=None):
    """room_context doors/windows -> (structure furniture, objs without the boxes they replace). Box = [width along
    the wall, OPENING_T, height] centred at the given position (z = centre -> bottom = sill), yaw = nearest floor
    edge. ~15% of openings already carry an object box (unknown / otherstructure, same centre and size): that box is
    replaced by the labelled opening instead of duplicated. linked (a set of source_semantic_id): only those
    openings are kept; ids keep the source index."""
    out, dup = [], set()
    for kind in ("door", "window"):
        for k, d in enumerate(rc.get(kind + "s") or []):
            if linked is not None and d.get("source_semantic_id") not in linked:
                continue
            p, wh = d["position"], sorted([d["width"], d["height"]])
            for i, o in enumerate(objs):
                q, ext = o["position"], sorted(2 * abs(c) / 1000 for c in o["raw_coeffs"])[1:]
                if math.hypot(q["x"] - p["x"], q["y"] - p["y"]) < SAME_M and abs(q["z"] - p["z"]) < SAME_M \
                        and all(abs(u - v) < 0.3 for u, v in zip(ext, wh)):
                    dup.add(i)
            out.append({"furniture_category": kind, "furniture_instance_id": f"{kind}_{k}", "structure": True,
                        "furniture_position": p,
                        "furniture_rotation": {"x": 0, "y": 0, "z": _edge_deg(rc["floor_polygon"], p["x"], p["y"])},
                        "furniture_size": {"width": d["width"], "length": OPENING_T, "height": d["height"]}})
    return out, [o for i, o in enumerate(objs) if i not in dup]


def topology_links(path):
    """structure_topology.tar.gz (the official annotation_3d per scene) -> {(scene id, room id): ids of the doors /
    windows whose planes share a line with one of the room's wall planes}."""
    out = {}
    with tarfile.open(path) as t:
        for m in t:
            if not (m.isfile() and m.name.endswith(".json")) or m.name.rsplit("/", 1)[-1].startswith("._"):
                continue
            d = json.load(t.extractfile(m))
            a = d["annotation_3d"]
            pl = np.asarray(a["planeLineMatrix"], dtype=bool)
            ops = {e["ID"]: pl[e["planeID"]].any(axis=0) for e in a["semantics"] if e["type"] in ("door", "window")}
            for e in a["semantics"]:
                if e["type"] not in ("door", "window"):
                    walls = pl[[p for p in e["planeID"] if a["planes"][p]["type"] == "wall"]].any(axis=0)
                    out[(d["scene_id"], str(e["ID"]))] = {i for i, ln in ops.items() if (ln & walls).any()}
    return out


def convert(rec, links=None):
    """Export record -> IR room, without the nearest-assigned objects whose centre is outside the room (module doc).
    links (topology_links): keep only the doors / windows the official topology puts on this room's walls; None keeps
    all."""
    rc, prov = rec["room_context"], rec["provenance"]
    linked = None if links is None else links.get((prov["source_house_id"], str(rc["room_id"])), set())
    n_unlinked = 0 if linked is None else sum(d.get("source_semantic_id") not in linked
                                              for k in ("doors", "windows") for d in rc.get(k) or [])
    floor = Polygon(rc["floor_polygon"])
    objs = [o for o in rec["layout"]["floor_layout"]["objects"] if o.get("assignment_method") != "nearest"
            or floor.covers(Point(o["position"]["x"], o["position"]["y"]))]
    n_outside = len(rec["layout"]["floor_layout"]["objects"]) - len(objs)
    ops, objs = openings(rc, objs, linked)
    room = {"room_id": rc["room_id"], "room_type": None if rc["room_type"] == "undefined" else rc["room_type"],
            "room_boundary": rc["floor_polygon"], "room_height": rc["ceiling_height_m"],
            "furniture": [_furniture(o) for o in objs] + ops}
    return convert_room(room, source=SOURCE, uid=rec["sample_id"], group=f"structured3d:{prov['source_house_id']}",
                        subset=prov["split"], boundary_type="polygon", center_z=True,
                        extra=lambda f: {"structure": True} if f.get("structure") else
                        {**({"raw_basis": f["raw_basis"]} if f["furniture_rotation"]["x"] is None else {}),
                         **({"front_known": False} if f.get("front_known") is False else {})},
                        meta={"front_known": True, "room_origin_house_m": rec["room_origin_house_m"],
                              "n_nearest_assigned": sum(o.get("assignment_method") == "nearest" for o in objs),
                              "n_nearest_outside_dropped": n_outside, "n_openings_unlinked_dropped": n_unlinked,
                              "n_unknown_category": sum(o["category"] == "unknown" for o in objs)})


def load(root):
    links = topology_links(os.path.join(root, DIR, "structure_topology.tar.gz"))
    for rec in iter_records(os.path.join(root, DIR, "structured3d.jsonl")):
        yield convert(rec, links)


if __name__ == "__main__":
    # bed rotated 90 deg: raw row0 = world +Y, row1 = world -X (back on raw +Y -> world -X) -> front world +X, yaw 0
    f = _furniture({"category": "bed", "instance_id": 1, "position": {"x": 0, "y": 0, "z": 0.5},
                    "raw_basis": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], "raw_coeffs": [800, 1000, 500]})
    assert abs(f["furniture_rotation"]["z"]) < 1e-9 and f["furniture_size"]["width"] == 2.0, f
    f = _furniture({"category": "unknown", "instance_id": 2, "position": {"x": 0, "y": 0, "z": 1},
                    "raw_basis": [[0, 0, -1], [0, 1, 0], [1, 0, 0]], "raw_coeffs": [1, 1, 1]})
    assert f["furniture_category"] == "unknown" and f["furniture_rotation"]["x"] is None
    # 4 x 3 room; door on the bottom wall already boxed as 'otherstructure' (replaced, not duplicated), window on the
    # right wall (x = 4) 0.1 m outside the floor, centre z 1.5, height 1.4 -> sill 0.8
    rec = {"sample_id": "t", "provenance": {"source_house_id": 0, "split": "train"}, "room_origin_house_m": [0, 0, 0],
           "room_context": {"room_id": "r", "room_type": "undefined", "ceiling_height_m": 2.8,
                            "floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                            "doors": [{"position": {"x": 1, "y": -0.06, "z": 1.1}, "width": 0.7, "height": 2.2}],
                            "windows": [{"position": {"x": 4.1, "y": 1.5, "z": 1.5}, "width": 1.6, "height": 1.4}]},
           "layout": {"floor_layout": {"objects": [
               {"category": "otherstructure", "instance_id": 9, "position": {"x": 1, "y": -0.06, "z": 1.1},
                "raw_basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "raw_coeffs": [350, 60, 1100]}]}}}
    ir = convert(rec)
    door, win = ir["objects"]
    close = lambda a, b: all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    assert door["category"] == "door" and door["structure"] and close(door["pos"], [1, -0.06, 0]), door
    assert close(door["size"], [0.7, OPENING_T, 2.2]) and abs(door["yaw"]) < 1e-9, door
    assert win["category"] == "window" and close(win["pos"], [4.1, 1.5, 0.8]) and close(win["size"], [1.6, OPENING_T, 1.4])
    assert abs(win["yaw"] - math.pi / 2) < 1e-9 and not win["tilted"], win
    from fastfill.anchors import fixed_geometry
    assert {o["category"] for o in fixed_geometry(ir)} == {"door", "window"}
    print("structured3d.py self-check ok")
