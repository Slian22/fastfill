"""SceneSmith example scenes (179 Room scenes + 31 Houses split into 185 rooms). Eval-only: too small, and
they are the paper's evaluation scenes (prompts 0-99 are the SceneEval-100 benchmark).

Export (kits/scenesmith_aggregate.py): meters, Z-up, furniture_position = DMD model-frame origin, rotation =
AngleAxis -> XYZ Euler deg (R = Rz Ry Rx), size = asset glTF AABB in the Z-up asset frame (x, y, z).
Asset origin, checked on geometry_metadata AABBs (not in the aggregate): furniture/manipuland = bottom center;
wall_mounted = back face (bbox y in [0, L]), vertical center; ceiling_mounted = top center.
Front = local +Y (wall-mounted assets extend into +Y; front test on beds/wardrobes).
Boundary = floor-plan AABB; every room has exactly north/south/east/west walls, so it is the real rectangle.
Free bodies are physics-settled: x/y tilt <= TILT_DEG is settling noise and is zeroed, larger tilt is kept.
"""
import math
import os

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "SceneSmith"
TILT_DEG = 2.0
ANCHOR = {"furniture": "floor", "wall_mounted": "wall", "ceiling_mounted": "ceiling"}   # manipuland: inferred


def tilt_deg(rot):
    return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(math.radians(rot["x"])) * math.cos(math.radians(rot["y"]))))))


def fix(f):
    """Origin -> bottom center, settling tilt -> upright. Returns (furniture dict, noise_zeroed)."""
    rot, p, s, g = f.get("furniture_rotation"), f.get("furniture_position"), f.get("furniture_size"), f["source_fields"]["asset_group"]
    if not (rot and p and s):
        return f, False
    p = dict(p)
    yaw = math.radians(rot["z"])
    if g == "wall_mounted":
        p["x"] -= math.sin(yaw) * s["length"] / 2
        p["y"] += math.cos(yaw) * s["length"] / 2
        p["z"] -= s["height"] / 2
    elif g == "ceiling_mounted":
        p["z"] -= s["height"]
    noise = (rot["x"] or rot["y"]) and tilt_deg(rot) <= TILT_DEG
    if noise:
        rot = {"x": 0, "y": 0, "z": rot["z"]}
    return {**f, "furniture_position": p, "furniture_rotation": rot}, bool(noise)


def load(root):
    base = os.path.join(root, "imChuling__3D_Room_Collections/SceneSmith_exported")
    for fn in ("json_single.json", "multiroom/multiroom_split.json"):
        for scene in iter_records(os.path.join(base, fn)):
            parent = scene["parent_scene_id"]
            for room in scene["rooms"]:
                fixed = [fix(f) for f in room["furniture"]]
                room = {**room, "furniture": [f for f, _ in fixed]}
                ir = convert_room(
                    room, source=SOURCE, uid=f"SceneSmith::{scene['scene_id']}", group=f"scenesmith:{parent}",
                    subset=scene["provenance"]["subset"], boundary_type="polygon", center_z=False,
                    front_offset_deg=90,
                    extra=lambda f: {"anchor": ANCHOR.get(f["source_fields"]["asset_group"]),
                                     "asset_group": f["source_fields"]["asset_group"]},
                    meta={"eval_only": True, "front_known": True, "file": fn, "scene_id": scene["scene_id"],
                          "parent_scene_id": parent, "tilt_threshold_deg": TILT_DEG,
                          "n_tilt_noise_zeroed": sum(z for _, z in fixed), "room_type_source": "dmd_room_frame"})
                yield ir
