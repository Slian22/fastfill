"""IL3D, 3D-FRONT subset (20,274 designer rooms): the canonical copy of 3D-FRONT rooms.

Verified conventions (scratch checks against IL3D's own USDZ meshes and the 3D-FRONT export):
- position = the 3D-FUTURE asset origin (IL3D copies the raw 3D-FRONT `pos`); every sampled mesh has its origin
  at the bottom center (y_min = 0, x/z centered), so it is the IR bottom center. XY are room-local (IL3D moved each
  room so its own floor mesh AABB starts at 0).
- size: the layout `bbox` is NOT usable as-is: for int-scale placements it is [x, z, y] of the mesh, otherwise
  [x, y, z] unscaled, and the IL3D export reads it as [w, h, l] for all. assets.json meta_data width/length/height
  equal the mesh extents along x / y(up) / z for all sampled assets, and the official USD script applies `scale` in
  the asset frame, so the placed local extents are meta x |scale|.
- rotation: the export's furniture_rotation.z = yaw of mesh +X (from the full rotateXYZ matrix; tilted -> x/y null).
  3D-FUTURE meshes face +Z (y-up) = IR local -Y, hence front_offset_deg = -90. Only x-mirrors (scale x < 0) occur,
  which do not flip the front.
- rooms: IL3D groups a house's objects by room TYPE, so 1,300 IL3D "rooms" merge objects of 2-10 source rooms
  (e.g. three bedrooms in one), and its single floor rectangle may belong to yet another room. Every IL3D object is
  matched to its 3D-FRONT export placement (same asset id and position under one per-room frame shift), which gives
  its true source room; each IL3D room is split into one IR room per source room.
- boundary / height: the export's convex hull of the source room's Floor meshes (moved into IL3D's frame by the
  shift) and its floor-to-ceiling height (IL3D's supplemented height when the export has none and the room is
  IL3D's own floor room).
- anchors: assets.json placement flags are per ASSET, so they are used only when a single flag agrees with z.
"""
import collections
import json
import os

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "IL3D_3dfront"
IL3D = "imChuling__3D_Room_Collections/IL3D_exported"
F3D = "BillLin66__3D_Room_Collections/3DFront_exported/3dfront.jsonl"


def load_assets(root):
    with open(os.path.join(root, IL3D, "IL3D/assets.json")) as f:
        return {a["model_id"]: a for a in json.load(f)}


def placed_size(f, asset):
    """Placed extents along mesh x, z, y = IR local X, Y, Z (before the front offset); None if unknown."""
    m = (asset or {}).get("meta_data") or {}
    s = f["source_fields"]["layout_object"].get("scale")
    if not (s and len(s) == 3 and all(isinstance(m.get(k), (int, float)) for k in ("width", "length", "height"))):
        return None
    return {"width": m["width"] * abs(s[0]), "length": m["height"] * abs(s[2]), "height": m["length"] * abs(s[1])}


def flag_anchor(f):
    fl = {k for k, v in (f.get("placement_flags") or {}).items() if v}
    z = (f.get("furniture_position") or {}).get("z")
    if z is None or len(fl) != 1:
        return None
    if fl == {"on_floor"} and z < 0.05:
        return "floor"
    if fl == {"on_ceiling"} and z > 1.0:
        return "ceiling"
    if fl == {"on_wall"} and z >= 0.05:
        return "wall"
    return None


def il3d_extra(f):
    return {"anchor": flag_anchor(f), "desc": f.get("description")}


def export_rooms(root):
    """house uid -> [(source room id, hull in house frame, height, [(jid, x, y), ...])]."""
    out = collections.defaultdict(list)
    for line in open(os.path.join(root, F3D)):
        d = json.loads(line)
        r = d["room"]
        pts = [(f["source_fields"]["jid"], f["furniture_position"]["x"], f["furniture_position"]["y"])
               for f in r["furniture"] if f["source_fields"].get("jid") and f["furniture_position"]]
        out[d["house_uid"]].append((d["provenance"]["source_room_id"], r["room_boundary"], r["room_height"], pts))
    return out


def frame_shift(furniture, pts, tol=0.02):
    """Shift s with export_pos = il3d_pos + s supported by the most shared placements; (s, votes) or (None, 0)."""
    pairs = [((f["furniture_position"]["x"], f["furniture_position"]["y"]), pts[f["furniture_asset_id"]])
             for f in furniture if f.get("furniture_position") and f.get("furniture_asset_id") in pts]
    best, votes = None, 0
    for p, cands in pairs:
        for q in cands:
            s = (q[0] - p[0], q[1] - p[1])
            n = sum(any(abs(c[0] - a[0] - s[0]) < tol and abs(c[1] - a[1] - s[1]) < tol for c in cs) for a, cs in pairs)
            if n > votes:
                best, votes = s, n
    return best, votes


def split_by_source_room(furniture, rooms, own=None, tol=0.02):
    """-> (shift, {export room index: [furniture]}, [unassigned furniture])."""
    pts = collections.defaultdict(list)
    for k, (_, _, _, placements) in enumerate(rooms):
        for jid, x, y in placements:
            pts[jid].append((x, y, k))
    shift, _ = frame_shift(furniture, {j: [(x, y) for x, y, _ in v] for j, v in pts.items()}, tol)
    hits = []
    for f in furniture:
        p = f.get("furniture_position")
        hits.append({k for x, y, k in pts.get(f.get("furniture_asset_id"), ())
                     if shift and p and abs(x - p["x"] - shift[0]) < tol and abs(y - p["y"] - shift[1]) < tol})
    # a placement listed in several source rooms goes to the room most of this room's objects come from,
    # ties to IL3D's own room
    votes = collections.Counter(k for h in hits for k in h)
    for k, r in enumerate(rooms):
        votes[k] += 0.5 * (r[0] == own)
    parts, rest = collections.defaultdict(list), []
    for f, h in zip(furniture, hits):
        (parts[max(sorted(h), key=votes.__getitem__)] if h else rest).append(f)
    return shift, parts, rest


def load(root):
    assets = load_assets(root)
    exp = export_rooms(root)
    for scene in iter_records(os.path.join(root, IL3D, "3dfront.json")):
        for room in scene["rooms"]:
            sf = room["source_fields"]
            house = sf["original_house_file"][:-len(".json")]
            furniture = []
            for f in room["furniture"]:
                f = dict(f)
                a = assets.get(f.get("furniture_asset_id"))
                # 3D-FUTURE category; for the ~12k assets without one (IL3D label "Others": potted plants, fridges,
                # pianos ...) the asset's own assets.json meta_data category
                f["furniture_category"] = (f.get("asset_category") or ((a or {}).get("meta_data") or {}).get("category")
                                           or f.get("furniture_category"))
                f["furniture_size"] = placed_size(f, a)
                furniture.append(f)
            rooms = exp.get(house, [])
            shift, parts, rest = split_by_source_room(furniture, rooms, sf.get("original_roomId"))
            if len(parts) == 1:          # unassigned objects (NaN position) can only belong to the one source room
                parts = {k: v + rest for k, v in parts.items()}
            for k, fl in (parts or {None: furniture}).items():
                srid, hull, height, _ = rooms[k] if k is not None else (None, None, None, None)
                if height is None and srid == sf.get("original_roomId"):
                    height = room.get("room_height")
                boundary = [[x - shift[0], y - shift[1]] for x, y in hull] if hull else None
                uid = f"il3d:{scene['scene_id']}" + (f"::{srid}" if len(parts) > 1 else "")
                ir = convert_room({**room, "furniture": fl, "room_boundary": boundary, "room_height": height},
                                  source=SOURCE, uid=uid, group=f"3dfront:{house.lower()}", subset="3D-FRONT",
                                  boundary_type="hull", front_offset_deg=-90, extra=il3d_extra,
                                  meta={"house": house, "source_room_id": srid, "il3d_room_id": sf.get("original_roomId"),
                                        "il3d_scene_id": scene["scene_id"], "split_from_il3d_room": len(parts) > 1,
                                        "boundary_from": "3dfront_export_floor_hull", "front_known": True,
                                        "size_from": "assets.json mesh extents x layout scale"})
                if len(parts) > 1 and rest:  # objects whose source room is unknown: the split rooms are incomplete
                    ir["meta"]["n_incomplete"] += len(rest)
                yield ir


if __name__ == "__main__":
    f = {"furniture_position": {"x": 1.0, "y": 2.0, "z": 0.0}, "furniture_asset_id": "a",
         "placement_flags": {"on_floor": True}, "source_fields": {"layout_object": {"scale": [-2, 1, 1]}}}
    assert placed_size(f, {"meta_data": {"width": 1, "length": 0.5, "height": 3}}) == {"width": 2, "length": 3, "height": 0.5}
    assert flag_anchor(f) == "floor"
    g = {**f, "furniture_position": {"x": 3.0, "y": 2.0, "z": 0.0}, "furniture_asset_id": "b"}
    h = {**f, "furniture_position": None}
    assert frame_shift([f, g], {"a": [(6.0, 7.0), (0.0, 0.0)], "b": [(8.0, 7.0)]}) == ((5.0, 5.0), 2)
    shift, parts, rest = split_by_source_room([f, g, h], [("r0", None, None, [("a", 6.0, 7.0)]),
                                                          ("r1", None, None, [("b", 8.0, 7.0), ("a", 0.0, 0.0)])])
    assert shift == (5.0, 5.0) and parts == {0: [f], 1: [g]} and rest == [h]
    print("il3d_3dfront self-check ok")
