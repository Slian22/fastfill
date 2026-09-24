"""SpatialLM-Dataset layouts (Manycore/Kujiale designer houses: 12,328 scenes, 54,775 rooms).
View used: spatiallm.json only (scene records, every room once); json_room.json and multiroom/*.json are
re-packagings of the same rooms (split_views.py) and are ignored.

Export (aggregate_spatiallm.py): source Bbox(category, px,py,pz, angle_z, sx,sy,sz) -> furniture_position = box
CENTER, furniture_size = full scale along local x/y/z, rotation.z = degrees(angle_z); the format is yaw-only, so
the exported x/y = null mean "not tilted" (set to 0 here). Meters, Z-up, scene-global XY.
room_boundary = ring of wall start points (wall_polygon); self-intersecting rings get boundary_type None.
Front = local -Y (front test: toilets/beds touch the wall with local +Y), same as the other Manycore exports.

Cross-export duplicates (same Manycore design; exact floor area / boundary edges + identical pairwise distances of
floor furniture, see the g2-layout-mixed assessment): scenes holding a SpatialGen test room are dropped (SpatialGen
is eval-only), scenes matching a Structured3D / InteriorGS house take that house's group key.
"""
import os

from shapely.geometry import Polygon

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "SpatialLM"
SPATIALGEN_DUP = {f"scene_{n:06d}" for n in (
    646, 692, 701, 713, 734, 738, 773, 792, 793, 806, 828, 832, 1569, 1585, 1591, 1599, 1601, 1610, 1612, 1758,
    4021, 4053, 7537, 7629, 7989, 8143, 8147, 8729, 8815, 11483,
    # review pass: boundary-free match (>= 3 floor objects with identical dims at identical relative positions under
    # one axis-aligned rigid map, >= half of the smaller set) or same floor area +-0.1 m2 with >= 70% object match
    82, 537, 644, 667, 671, 740, 745, 822, 827, 841, 843, 1579, 1606, 1608, 1624, 1626, 1725, 1745, 4027, 4099,
    4111, 4112, 4165, 5211, 6058, 6252, 7362, 7478, 7498, 7622, 7729, 7736, 9248, 9360, 9771, 10350, 10394, 11383,
    11482, 12106,
    4359)}   # pre-launch review: design twin of spatialgen_scene_00039::room_0
GROUP = {"scene_003636": "structured3d:scene_00729", "scene_007839": "structured3d:scene_02605",
         "scene_009460": "structured3d:scene_01438", "scene_011931": "structured3d:scene_00402",
         "scene_012118": "structured3d:scene_01556", "scene_004716": "interiorgs:0304_840560",
         "scene_011878": "interiorgs:0553_840921",
         # review pass: same floor area to 0.01 m2 and 4-6 floor objects with identical dims at identical relative positions
         "scene_005720": "interiorgs:0785_841229", "scene_006253": "interiorgs:0536_840884",
         "scene_010373": "interiorgs:0817_841276"}


def load(root):
    path = os.path.join(root, "imChuling__3D_Room_Collections/SpatialLM_exported/spatiallm.json")
    for scene in iter_records(path):
        sid = scene["scene_id"]
        if sid in SPATIALGEN_DUP:
            continue
        for room in scene["rooms"]:
            furn = [{**f, "furniture_category": f["furniture_category"].replace("_", " "),
                     "furniture_rotation": {**f["furniture_rotation"], "x": 0, "y": 0}}
                    for f in room["furniture"]]
            ir = convert_room({**room, "furniture": furn}, source=SOURCE, uid=f"spatiallm:{sid}::{room['room_id']}",
                              group=GROUP.get(sid, f"spatiallm:{sid}"), boundary_type="polygon", center_z=True, front_offset_deg=-90,
                              meta={"scene_id": sid, "front_known": True, "source_splits": room.get("source_splits"),
                                    "view": "spatiallm.json"})
            b = ir["boundary"]
            if b and not Polygon(b).is_valid:     # wall ring crosses/touches itself: cannot prove in-bounds
                ir["boundary_type"] = None
                ir["meta"]["boundary_invalid"] = True
            yield ir
