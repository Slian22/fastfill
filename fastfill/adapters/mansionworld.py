"""MansionWorld (AI2-THOR / Objathor procedural buildings), room view json_room.json (27,884 rooms, 1,000 building
variants of 300 floor plans; building/floor views are the same rooms regrouped).

Export (kits/convert_mansionworld.py, enrich_wall_objects.py): meters, Z-up, (x, z)_yup -> (x, -z), room polygon
recentred on its area centroid. Floor objects: position = footprint center, z = 0 (bottom), size = padded cm
footprint minus 0.10 m, de-rotated (yaws are multiples of 90), height = 2 * center y. Yaw = THOR rotation.y,
which the (x, -z) map turns into CCW yaw about +Z of asset local +X; front = Unity +Z = local -Y.
Exception: the single custom asset of `toilet_suite` (a stall) is authored backwards: its +Y side touches the wall
and its paired toilet_suite_door lies on the room side, so it gets +180 deg.
Wall objects: z = bottom; xy were recentred on the vertex mean instead of the area centroid (bug), corrected here;
width/length had the floor-footprint padding subtracted although wall footprints are unpadded (bug), added back here.
Surface (small) objects are excluded: the export kept only rotation.y, and raw small objects carry +-90 deg x/z
rotations (real tilts / axis fixes) that cannot be recovered from the export.
"""
import os

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "MansionWorld"
PAD, MIN_DIM = 0.10, 0.01   # enrich_wall_objects.py FOOTPRINT_PADDING_M / MIN_DIM_M


def fix(f, vx, vy):
    f = dict(f)
    if f["source_fields"].get("mount_type") == "wall" and f.get("furniture_position"):
        p = f["furniture_position"]
        f["furniture_position"] = {**p, "x": p["x"] + vx, "y": p["y"] + vy}
        # enrich_wall_objects.py also subtracts the 0.10 m floor-footprint padding, but wall footprints are unpadded
        # (raw sample: footprint == annotation bbox for 1,652/1,652). Values clamped to 0.01 stay (true extent <= 0.11).
        s = f.get("furniture_size")
        if s:
            f["furniture_size"] = {**s, **{k: s[k] + PAD for k in ("width", "length") if s[k] > MIN_DIM + 1e-6}}
    if f["furniture_category"] == "toilet_suite" and f.get("furniture_rotation"):
        f["furniture_rotation"] = {**f["furniture_rotation"], "z": f["furniture_rotation"]["z"] + 180}
    if f["furniture_category"].startswith("vertical_core_"):   # stair / elevator cores: let the structure filter see them
        f["furniture_category"] = f["furniture_category"].replace("_", " ")
    return f


def load(root):
    path = os.path.join(root, "imChuling__3D_Room_Collections/MansionWorld_exported/json_room.json")
    for scene in iter_records(path):
        prov = scene["provenance"]
        building = prov["source_house_id"]
        for room in scene["rooms"]:
            b = room.get("room_boundary") or []
            vx, vy = (sum(p[0] for p in b) / len(b), sum(p[1] for p in b) / len(b)) if b else (0.0, 0.0)
            yield convert_room(
                {**room, "furniture": [fix(f, vx, vy) for f in room["furniture"]]}, source=SOURCE,
                uid=f"MansionWorld::{scene['scene_id']}", group="mansionworld:" + building.split("#")[0],
                boundary_type="polygon", center_z=False, front_offset_deg=-90,
                extra=lambda f: {"anchor": "wall" if f["source_fields"].get("mount_type") == "wall" else "floor"},
                meta={"building": building, "room_id": room["room_id"], "geometry_hash": prov.get("geometry_hash"),
                      "front_known": True, "surface_objects_excluded": sum(
                          len(g.get("objects") or []) for g in room["source_fields"].get("surface_groups") or []),
                      "license_note": "CC BY 4.0 + HF gate: non-commercial research"})
