"""InteriorGS (Manycore designer scenes, 1,000 houses / 6,607 rooms). View used: interiorgs.json only (full
scenes, every room once); json_single.json + multiroom/*.json repackage the same rooms and are ignored.

Export (aggregate_scenes.py): meters, Z-up, scene-global XY, room_boundary = structure.rooms[].profile (floor
polygon), room_height from the room_box. The exported furniture_rotation/size are longest-edge geometry
([0, 180) yaw, no front), so this adapter re-derives pose from source_fields.bounding_box instead:
corners 0-3 are the bottom face in a fixed local order and edge c1-c2 is the object's BACK (front test on all
1,000 scenes: bed 0.85, toilet 0.94, painting 0.90 touch the wall with c1-c2, c3-c0 ~0.00), so
front = midpoint(c3, c0) - midpoint(c1, c2); IR size = [|c0c1| (depth, along front), |c1c2|, height].
Caveat: custom cabinetry (labels wardrobe / cabinet / wall cabinet) does not follow this frame (wall side ~uniform),
so their yaw is only reliable up to the box axes: those objects get front_known=False.
Boxes whose top face is shifted from the bottom face or whose bottom is not level are marked tilted.
"""
import math
import os
import re

from shapely.geometry import Polygon

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "InteriorGS"
GROUP = {"0755_840824": "structured3d:scene_01482",   # same design as a Structured3D house (room boundary + furniture match)
         # duplicated InteriorGS houses that split.py's content_key does not link (rooms differ by a few objects)
         "0565_840943": "interiorgs:0478_840944", "0684_841508": "interiorgs:0521_840344"}
TILT_TOL = 0.02   # m
# Only the generic custom-cabinetry labels: TV / basin / shoe / wine / display / storage cabinets follow the corner
# order (back edge c1-c2 on the wall 0.51-0.82, front edge c3-c0 <= 0.10), plain wardrobe / cabinet / wall cabinet do not
UNRELIABLE_FRONT = re.compile(r"^(wardrobe|cabinet|wall cabinet)s?( doors)?$", re.I)


def _pose(f):
    """Export furniture -> unified-schema furniture with front-aware yaw/size, or None fields if no bbox."""
    bb = (f.get("source_fields") or {}).get("bounding_box")
    if not bb or len(bb) != 8:
        return {**f, "furniture_rotation": None}
    c = [(p["x"], p["y"], p["z"]) for p in bb]
    bz = [p[2] for p in c[:4]]
    tilted = (max(bz) - min(bz) > TILT_TOL or
              max(math.hypot(c[i][0] - c[i + 4][0], c[i][1] - c[i + 4][1]) for i in range(4)) > TILT_TOL)
    fx = (c[3][0] + c[0][0] - c[1][0] - c[2][0]) / 2
    fy = (c[3][1] + c[0][1] - c[1][1] - c[2][1]) / 2
    rot = None if tilted else 0.0
    return {**f,
            "furniture_position": {"x": sum(p[0] for p in c[:4]) / 4, "y": sum(p[1] for p in c[:4]) / 4, "z": min(bz)},
            "furniture_rotation": {"x": rot, "y": rot, "z": math.degrees(math.atan2(fy, fx))},
            "furniture_size": {"width": math.hypot(fx, fy),
                               "length": math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1]),
                               "height": max(p[2] for p in c) - min(bz)}}


def load(root):
    path = os.path.join(root, "imChuling__3D_Room_Collections/InteriorGS_exported/interiorgs.json")
    for scene in iter_records(path):
        sid = scene["scene_id"]
        for room in scene["rooms"]:
            ir = convert_room({**room, "furniture": [_pose(f) for f in room["furniture"]]},
                              source=SOURCE, uid=f"interiorgs:{sid}::{room['room_id']}", group=GROUP.get(sid, f"interiorgs:{sid}"),
                              boundary_type="polygon",
                              extra=lambda f: {"front_known": False} if UNRELIABLE_FRONT.search(f["furniture_category"]) else {},
                              meta={"scene_id": sid, "front_known": True, "front_source": "bbox corner order",
                                    "room_type_available": False, "view": "interiorgs.json"})
            b = ir["boundary"]
            if b and not Polygon(b).is_valid:     # self-intersecting source profile: cannot prove in-bounds
                ir["boundary_type"] = None
                ir["meta"]["boundary_invalid"] = True
            yield ir


if __name__ == "__main__":
    # bed-like box whose back edge c1-c2 lies on the wall x=0 (clockwise corners): front = +X, depth 0.6, width 2
    f = {"source_fields": {"bounding_box": [{"x": x, "y": y, "z": z} for z in (0.0, 1.0) for x, y in
                                            ((0.6, -1.0), (0.0, -1.0), (0.0, 1.0), (0.6, 1.0))]}}
    p = _pose(f)
    assert abs(p["furniture_rotation"]["z"]) < 1e-9 and p["furniture_rotation"]["x"] == 0.0
    assert abs(p["furniture_size"]["width"] - 0.6) < 1e-9 and abs(p["furniture_size"]["length"] - 2.0) < 1e-9
    assert p["furniture_position"] == {"x": 0.3, "y": 0.0, "z": 0.0}
    print("interiorgs.py self-check ok")
