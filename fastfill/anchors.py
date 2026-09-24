"""Object scope for training: drop structure, infer support anchors, keep a support-closed subset.

anchor = "floor" | "object" | "wall" | "ceiling". Adapters set anchor/parent from source evidence when the
source has it (placement flags, explicit parent ids); otherwise `infer_anchors` derives them from boxes:
  floor    bottom at most FLOOR_Z = 10 cm above z=0 (sunk boxes too, so build.prep can drop and count them)
  ceiling  hanging categories (pendant, chandelier, ...), or top within 10 cm of the room height above 1 m
  wall     wall art (painting, mirror, ...), or anything else elevated without a support below
  object   bottom within 5 cm of the top of a floor/object box whose footprint contains its center;
           seats, beds and rugs are never parents (bbox top = backrest / headboard; a rug is not a surface)
Box-level contact is weak evidence of support; build.py counts evidence and inferred supports separately.
"""
import re

from shapely.geometry import Point, Polygon

from fastfill.scene import footprint, head, norm_cat

FLOOR_Z = 0.10   # bottoms up to 10 cm above the floor count as standing on it (build.prep snaps them to z=0)
# Strong architectural words mark structure anywhere in the name; floor/wall/ceiling/column are usually modifiers
# (plant_floor, floor lamp, wall_shelving_unit, locker_column) and mark structure only in purely architectural
# names ("wall", "Wall design", "floor mold").
STRUCTURE = re.compile(r"\b(door|window|curtain|drape|blind|beam|pillar|stair|staircase|railing|rail|baseboard|radiator|"
                       r"opening|doorframe|partition|fireplace|otherstructure|structure|doorway|arch|archway|bulkhead|"
                       r"niche|fence|step|drain|pipe|socket|outlet|tile)s?\b")
# furniture whose name borrows a structure word: fireplace_console, pillar_candle, clothes rail, portable_partition_panel
NOT_STRUCTURE = re.compile(r"\b(lamp|light|fan|cabinet|shelf|mat|rug|towel|chair|table|console|stool|candle|figurine|box|"
                           r"plant|planter|basket|display|station|clothes|freestanding|portable)s?\b")
SURFACE = r"(wall|floor|ceiling|column)s?"
ARCH = (r"(panel|paneling|design|combination|mold|molding|moulding|background|trim|skirting|board|tile|cladding|covering|"
        r"decoration|sticker|styling|decorative|decor|structural|support|concrete|low|half|suspended)s?")
HANGING = re.compile(r"\b(pendant|chandelier|ceiling|downlight|hanging)\b")
WALL_ART = re.compile(r"\b(painting|picture|poster|artwork|art|mirror|whiteboard|blackboard|frame)s?\b")
FLAT = {"rug", "carpet", "mat", "doormat", "bathmat", "placemat", "playmat", "floormat", "yogamat", "exercisemat",
        "fitnessmat"}
FLAT |= {w + "s" for w in FLAT}
NO_TOP = {"sofa", "couch", "loveseat", "sectional", "bed", "armchair", "chair", "bench"} | FLAT  # not a support surface


def is_structure(category):
    c = re.sub(r"[^a-z]+", " ", (category or "").lower())
    if NOT_STRUCTURE.search(c):
        return False
    return bool(STRUCTURE.search(c)) or (bool(re.search(rf"\b{SURFACE}\b", c))
                                         and not re.sub(rf"\b({SURFACE}|{ARCH})\b", " ", c).strip())


def infer_anchors(room, tol=0.05):
    objs = room["objects"]
    fps = {o["id"]: Polygon(footprint(o)) for o in objs}
    height = room.get("height")
    for o in sorted(objs, key=lambda o: o["pos"][2]):
        if o.get("anchor"):
            continue
        z, nc = o["pos"][2], norm_cat(o["category"])
        if z <= FLOOR_Z:                 # one-sided: sunk boxes stay floor so build.prep drops and counts them
            o["anchor"] = "floor"
            continue
        if HANGING.search(nc):
            o["anchor"] = "ceiling"
            continue
        if WALL_ART.search(nc):
            o["anchor"] = "wall"
            continue
        c = Point(o["pos"][0], o["pos"][1])
        below = [p for p in objs if p is not o and p.get("anchor") in ("floor", "object") and head(p) not in NO_TOP
                 and abs(p["pos"][2] + p["size"][2] - z) < tol and fps[p["id"]].contains(c)]
        if below:
            o["anchor"], o["parent"] = "object", max(below, key=lambda p: p["pos"][2] + p["size"][2])["id"]
            o["anchor_inferred"] = True
        elif height and z + o["size"][2] > height - 2 * tol and z > 1.0:
            o["anchor"] = "ceiling"
        else:
            o["anchor"] = "wall"
    return room


def scope_objects(room, keep=("floor", "object")):
    """Structure removed, anchors inferred, objects outside `keep` dropped together with their dependants."""
    objs = [dict(o) for o in room["objects"] if not (o.get("structure") or is_structure(o["category"]))]
    room = infer_anchors({**room, "objects": objs})
    kept, ids = [], set()
    for o in sorted(room["objects"], key=lambda o: o["pos"][2]):   # parents sit lower than children
        if o["anchor"] in keep and (o.get("parent") is None or o["parent"] in ids):
            kept.append(o)
            ids.add(o["id"])
    order = {o["id"]: k for k, o in enumerate(room["objects"])}
    kept.sort(key=lambda o: order[o["id"]])
    return {**room, "objects": kept}


if __name__ == "__main__":
    room = {"height": 2.6, "objects": [
        {"id": "desk", "category": "desk", "size": [1.2, .6, .75], "pos": [1, 1, 0], "yaw": 0},
        {"id": "lamp", "category": "table lamp", "size": [.2, .2, .4], "pos": [1.2, 1, .75], "yaw": 0},
        {"id": "pic", "category": "painting", "size": [.8, .05, .6], "pos": [1, 0, 1.4], "yaw": 0},
        {"id": "cl", "category": "pendant lamp", "size": [.5, .5, .3], "pos": [2, 2, 2.0], "yaw": 0},
        {"id": "sofa", "category": "sofa", "size": [2, .9, .8], "pos": [3, 3, 0], "yaw": 0},
        {"id": "art", "category": "picture frame", "size": [.6, .05, .5], "pos": [3, 3, .8], "yaw": 0},
        {"id": "vase", "category": "vase", "size": [.1, .1, .2], "pos": [1, 0, 2.0], "yaw": 0, "anchor": "object", "parent": "pic"},
        {"id": "win", "category": "window", "size": [1, .1, 1.2], "pos": [3, 0, .9], "yaw": 0}]}
    s = scope_objects(room)
    assert [o["id"] for o in s["objects"]] == ["desk", "lamp", "sofa"], s
    assert s["objects"][1]["parent"] == "desk" and s["objects"][1]["anchor"] == "object"
    a = {o["id"]: o["anchor"] for o in infer_anchors({**room, "objects": [dict(o) for o in room["objects"]]})["objects"]}
    assert a["pic"] == "wall" and a["cl"] == "ceiling" and a["art"] == "wall", a   # art above a sofa is not "on" it
    assert is_structure("window") and is_structure("door(screen door, screen)") and not is_structure("floor lamp")
    for c in ("wall", "floor", "Wall design", "ceiling combination", "floor mold", "otherstructure", "room_partition",
              "radiator", "support_column", "vertical core stair 1", "door(screen door, screen)"):
        assert is_structure(c), c
    for c in ("plant_floor", "floor_plant_large", "locker_column", "floor_bookshelf", "bookshelf_wall_unit",
              "floor-standing air purifier", "wall_hung_toilet", "floor lamp", "wall_cabinet", "ceiling fan", "wall_mirror"):
        assert not is_structure(c), c
    sofa = {"id": "s", "category": "sofa", "size": [2, .9, .8], "pos": [1, 1, 0.07], "yaw": 0}
    rug = {"id": "r", "category": "rug", "size": [3, 2, .02], "pos": [1, 1, 0], "yaw": 0}
    got = {o["id"]: (o["anchor"], o.get("parent")) for o in infer_anchors({"objects": [dict(sofa), dict(rug)]})["objects"]}
    assert got == {"s": ("floor", None), "r": ("floor", None)}, got   # 7 cm up = standing, not on the rug
    print("anchors.py self-check ok")
