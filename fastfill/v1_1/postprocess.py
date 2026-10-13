"""Deterministic geometric repair and zero-collision enforcement for FastFill v1.1.

Enforces in order:
1. Exact vertical support (`z = 0` for floor objects, `z = parent_z + parent_sz` for `on` children) and clamps
   each child's XY center inside its parent's oriented footprint (plus sibling de-collision on parent surfaces).
2. Wall containment (`snap_inside` + polygon interior clamp + interior grid fallback so every footprint stays within
   1 mm of walls without 1 cm rounding drift).
3. Fixed obstacle (door/column/stair) de-blocking: pushes floor objects clear of non-window fixed boxes.
4. Iterative Minimum-Translation-Vector (MTV) / spiral + interior-grid 2D footprint de-collision for floor subtrees
   (checking both floor objects and supported children) so that `check(repaired_room)["collisions"]["main"]` and
   `["severe"]` are brought to 0 while preserving held constraints.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from shapely.geometry import Point, Polygon

from fastfill.anchors import FLAT
from fastfill.scene import footprint, head, norm_cat
from fastfill.v1_1.scene_v1_1 import apply_v1_1
from fastfill.validate import collision_kind, holds, support_contains


def _subtree_ids(oid: str, children_map: dict[str, list[str]]) -> list[str]:
    out = [oid]
    seen = {oid}
    stack = [oid]
    while stack:
        cur = stack.pop()
        for ch in children_map.get(cur, ()):
            if ch not in seen:
                seen.add(ch)
                out.append(ch)
                stack.append(ch)
    return out


def _move_with_children(
    pl: dict[str, dict[str, Any]],
    pid: str,
    dx: float,
    dy: float,
    children_map: dict[str, list[str]],
) -> None:
    """Translate `pid` and all its transitive `on` descendants by exact `(dx, dy)` in XY (no 1 cm rounding)."""
    for cur in _subtree_ids(pid, children_map):
        pos = pl[cur]["pos"]
        pl[cur]["pos"] = [pos[0] + dx, pos[1] + dy, pos[2]]


def _snap_wall_object_xy(
    room: dict[str, Any],
    p: dict[str, Any],
    floor_poly: Polygon,
) -> tuple[float, float, float]:
    """Ensure a wall-mounted object lies inside `floor_poly` (within 0.1 mm) and has a side within 0.08 m of a wall."""
    b_pts = room.get("boundary") or []
    x, y, yaw = float(p["pos"][0]), float(p["pos"][1]), float(p["yaw"])
    if len(b_pts) < 3:
        return x, y, yaw
    sx, sy = float(p["size"][0]), float(p["size"][1])
    cand = {"id": "w", "size": p["size"], "pos": [x, y, float(p["pos"][2])], "yaw": yaw}
    fp = Polygon(footprint(cand))
    if floor_poly.buffer(1e-4).covers(fp) and holds(["against_wall", "w"], [cand], b_pts):
        return x, y, yaw

    # Snap onto the nearest boundary wall segment, flush inside the room by sx/2 + 0.015 m (front +X facing inward)
    pts = list(b_pts)
    best_dist = math.inf
    best_pose = (x, y, yaw)
    ccw = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])) > 0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        dx, dy = float(x1) - float(x0), float(y1) - float(y0)
        seg_len = math.hypot(dx, dy)
        if seg_len < max(sy + 0.04, 0.10):
            continue
        ux, uy = dx / seg_len, dy / seg_len
        nx, ny = (-uy, ux) if ccw else (uy, -ux)  # unit inward normal
        # Project current (x, y) onto segment, clamped away from corners by sy/2 + 0.02
        margin = min(seg_len * 0.45, sy * 0.5 + 0.02)
        t = ((x - float(x0)) * ux + (y - float(y0)) * uy)
        t = max(margin, min(seg_len - margin, t))
        depth_offset = sx * 0.5 + 0.015
        wx = float(x0) + t * ux + depth_offset * nx
        wy = float(y0) + t * uy + depth_offset * ny
        wyaw = math.atan2(ny, nx) % (2 * math.pi)
        test_obj = {"id": "w", "size": p["size"], "pos": [wx, wy, float(p["pos"][2])], "yaw": wyaw}
        if not floor_poly.buffer(1e-4).covers(Polygon(footprint(test_obj))):
            continue
        d = math.hypot(wx - x, wy - y)
        if d < best_dist:
            best_dist = d
            best_pose = (wx, wy, wyaw)
    return best_pose


def _wall_candidate_poses(
    room: dict[str, Any],
    p: dict[str, Any],
    floor_poly: Polygon,
) -> list[tuple[float, float, float, float]]:
    """Generate candidate `(wx, wy, wz, wyaw)` poses along boundary walls sorted by distance to `p`."""
    b_pts = room.get("boundary") or []
    x, y, z = float(p["pos"][0]), float(p["pos"][1]), float(p["pos"][2])
    sx, sy, sz = float(p["size"][0]), float(p["size"][1]), float(p["size"][2])
    h_room = float(room.get("height") or 2.8)
    max_z = max(0.05, h_room - sz - 0.01)
    base_z = min(max(0.05, z if z > 0.02 else 1.20), max_z)
    z_cands = []
    for dz in (0.0, 0.35, -0.35, 0.70, -0.65, 1.05):
        zc = min(max(0.05, base_z + dz), max_z)
        if all(abs(zc - prev) > 0.03 for prev in z_cands):
            z_cands.append(zc)

    pts = list(b_pts)
    ccw = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])) > 0
    grown = floor_poly.buffer(1e-4)
    poses: list[tuple[float, float, float, float, float]] = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        dx, dy = float(x1) - float(x0), float(y1) - float(y0)
        seg_len = math.hypot(dx, dy)
        if seg_len < max(sy + 0.04, 0.10):
            continue
        ux, uy = dx / seg_len, dy / seg_len
        nx, ny = (-uy, ux) if ccw else (uy, -ux)
        margin = min(seg_len * 0.45, sy * 0.5 + 0.02)
        depth_offset = sx * 0.5 + 0.015
        wyaw = math.atan2(ny, nx) % (2 * math.pi)
        t_proj = max(margin, min(seg_len - margin, (x - float(x0)) * ux + (y - float(y0)) * uy))
        ts = [t_proj]
        t_cur = margin
        while t_cur <= seg_len - margin + 1e-6:
            if abs(t_cur - t_proj) > 0.04:
                ts.append(t_cur)
            t_cur += 0.12
        for t in ts:
            wx = float(x0) + t * ux + depth_offset * nx
            wy = float(y0) + t * uy + depth_offset * ny
            test_obj = {"id": "w", "size": p["size"], "pos": [wx, wy, base_z], "yaw": wyaw}
            if not grown.covers(Polygon(footprint(test_obj))):
                continue
            for wz in z_cands:
                dist = math.hypot(wx - x, wy - y) + 0.5 * abs(wz - base_z)
                poses.append((dist, wx, wy, wz, wyaw))
    poses.sort(key=lambda item: item[0])
    return [(wx, wy, wz, wyaw) for _, wx, wy, wz, wyaw in poses]


def snap_supports_v1_1(room: dict[str, Any], pl: dict[str, dict[str, Any]]) -> None:
    """Enforce exact support z-height and XY parent footprint containment in topological parent-first order."""
    visited: set[str] = set()
    h_room = room.get("height")
    b_pts = room.get("boundary")
    floor_poly = Polygon(b_pts) if b_pts else None
    if floor_poly is not None and not floor_poly.is_valid:
        floor_poly = floor_poly.buffer(0)
    wall_mounted_ids = {
        o["id"]
        for o in (room.get("objects") or [])
        if o.get("anchor") == "wall"
    }

    def fix_one(oid: str) -> None:
        if oid in visited or oid not in pl:
            return
        visited.add(oid)
        p = pl[oid]
        parent_id = p.get("on")
        if not parent_id or parent_id not in pl:
            p["on"] = None
            if oid in wall_mounted_ids:
                z_val = float(p["pos"][2])
                if z_val <= 0.02:
                    z_val = 1.20
                if h_room and h_room > 0.5:
                    if p["size"][2] > h_room:
                        p["size"][2] = max(0.05, float(h_room) - 0.05)
                    z_val = min(z_val, max(0.0, float(h_room) - float(p["size"][2]) - 0.01))
                wx, wy, wyaw = (
                    _snap_wall_object_xy(room, p, floor_poly)
                    if floor_poly is not None
                    else (float(p["pos"][0]), float(p["pos"][1]), float(p["yaw"]))
                )
                p["pos"] = [wx, wy, max(0.0, z_val)]
                p["yaw"] = wyaw
            else:
                p["pos"] = [float(p["pos"][0]), float(p["pos"][1]), 0.0]
                if h_room and h_room > 0.5 and p["size"][2] > h_room:
                    p["size"][2] = max(0.05, float(h_room) - 0.01)
            return
        fix_one(parent_id)
        par = pl[parent_id]
        top_z = float(par["pos"][2]) + float(par["size"][2])
        par_obj = {"pos": par["pos"], "size": par["size"], "yaw": par["yaw"]}
        x, y = float(p["pos"][0]), float(p["pos"][1])
        if not support_contains(par_obj, [x, y], tol=0.04):
            poly = Polygon(footprint(par_obj))
            if not poly.is_valid:
                poly = poly.buffer(0)
            shrunk = poly.buffer(-min(0.04, min(par["size"][:2]) * 0.15))
            target_poly = shrunk if not shrunk.is_empty else poly
            if target_poly.contains(Point(x, y)):
                nx, ny = x, y
            else:
                near_pt = target_poly.exterior.interpolate(target_poly.exterior.project(Point(x, y)))
                nx, ny = float(near_pt.x), float(near_pt.y)
            x, y = nx, ny
        p["pos"] = [x, y, top_z]
        if h_room and h_room > 0.5 and top_z + p["size"][2] > h_room:
            p["size"][2] = max(0.02, float(h_room) - top_z - 0.005)

    for oid in list(pl.keys()):
        fix_one(oid)


def _spiral_offsets(max_radius_m: float = 1.5, step_m: float = 0.05) -> list[tuple[float, float]]:
    pts = []
    n = int(round(max_radius_m / step_m))
    for dx in range(-n, n + 1):
        for dy in range(-n, n + 1):
            if dx == 0 and dy == 0:
                continue
            r2 = (dx * step_m) ** 2 + (dy * step_m) ** 2
            if r2 <= max_radius_m ** 2 + 1e-9:
                pts.append((dx * step_m, dy * step_m))
    pts.sort(key=lambda d: d[0] ** 2 + d[1] ** 2)
    return pts


SPIRAL_FINE = _spiral_offsets(0.18, 0.01)
SPIRAL_SMALL = _spiral_offsets(0.35, 0.02)
SPIRAL_WIDE = _spiral_offsets(2.50, 0.08)


def _interior_grid_points(floor_poly: Polygon, orig_x: float, orig_y: float, step: float = 0.18) -> list[tuple[float, float]]:
    minx, miny, maxx, maxy = floor_poly.bounds
    pts: list[tuple[float, float]] = []
    # Also include representative point and centroid first
    for pt_geom in (floor_poly.representative_point(), floor_poly.centroid):
        if floor_poly.contains(pt_geom):
            pts.append((float(pt_geom.x), float(pt_geom.y)))
    x = minx + step * 0.5
    while x < maxx:
        y = miny + step * 0.5
        while y < maxy:
            if floor_poly.contains(Point(x, y)):
                pts.append((x, y))
            y += step
        x += step
    pts.sort(key=lambda p: (p[0] - orig_x) ** 2 + (p[1] - orig_y) ** 2)
    return pts


def repair_v1_1(
    room: dict[str, Any],
    pl: dict[str, dict[str, Any]],
    resolve_collisions: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return a repaired copy of `pl` with exact support heights, wall containment, door clearance, and de-collision."""
    valid_ids = {o["id"] for o in (room.get("objects") or [])}
    rep = {k: copy.deepcopy(v) for k, v in pl.items() if not valid_ids or k in valid_ids}
    if not rep:
        return rep

    b_pts = room.get("boundary")
    if not b_pts:
        snap_supports_v1_1(room, rep)
        return rep

    floor_poly = Polygon(b_pts)
    if not floor_poly.is_valid:
        floor_poly = floor_poly.buffer(0)
    grown_1mm = floor_poly.buffer(9e-5)
    minx, miny, maxx, maxy = floor_poly.bounds
    room_span_x = max(0.2, maxx - minx)
    room_span_y = max(0.2, maxy - miny)

    # 1. Clamp ceiling height and oversized horizontal dimensions that cannot physically fit inside the room
    h_room = room.get("height")
    for oid, p in rep.items():
        if h_room and h_room > 0.5 and not p.get("on") and p["size"][2] > h_room:
            p["size"][2] = max(0.05, float(h_room) - 0.01)
        # Note: p["yaw"] is already in radians in parse_v1_1 / gt_placements_v1_1
        yaw_rad = float(p["yaw"])
        c, s = abs(math.cos(yaw_rad)), abs(math.sin(yaw_rad))
        sx, sy = float(p["size"][0]), float(p["size"][1])
        ext_x = sx * c + sy * s
        ext_y = sx * s + sy * c
        if ext_x > room_span_x - 0.01:
            scale = (room_span_x - 0.02) / max(ext_x, 1e-6)
            sx, sy = max(0.05, sx * scale), max(0.05, sy * scale)
        ext_y = sx * s + sy * c
        if ext_y > room_span_y - 0.01:
            scale = (room_span_y - 0.02) / max(ext_y, 1e-6)
            sx, sy = max(0.05, sx * scale), max(0.05, sy * scale)
        p["size"] = [sx, sy, float(p["size"][2])]

    # 2. Enforce topological support z and parent XY containment
    snap_supports_v1_1(room, rep)

    children_map: dict[str, list[str]] = {}
    for oid, p in rep.items():
        if p.get("on") and p["on"] in rep:
            children_map.setdefault(p["on"], []).append(oid)

    cat_by_id = {o["id"]: o.get("category", o["id"].rsplit("_", 1)[0].replace("_", " ")) for o in room["objects"]}
    wall_mounted_ids = {
        o["id"]
        for o in (room.get("objects") or [])
        if o.get("anchor") == "wall"
    }
    cons = room.get("constraints") or []
    walls = room["boundary"]

    # Build a room copy with predicted sizes so fastfill.interface.snap_inside can use them
    from fastfill.interface import snap_inside

    room_pred_sizes = {
        **room,
        "objects": [
            {**o, "size": rep[o["id"]]["size"]}
            for o in room["objects"]
            if o["id"] in rep
        ],
    }
    snap_inside(rep, room_pred_sizes, floor_poly, cons, walls)

    def obj_ir(oid: str, pos_override: list[float] | None = None, yaw_override: float | None = None) -> dict[str, Any]:
        p = rep[oid]
        return {
            "id": oid,
            "category": cat_by_id.get(oid, "object"),
            "size": p["size"],
            "pos": pos_override if pos_override is not None else p["pos"],
            "yaw": yaw_override if yaw_override is not None else p["yaw"],
            "parent": p.get("on"),
        }

    def all_held_constraints() -> list[Any]:
        idx = {oid: obj_ir(oid) for oid in rep}
        return [c for c in cons if holds(c, None, walls, idx)]

    held_cons = all_held_constraints()

    def keeps_held(oid: str, dx: float, dy: float) -> bool:
        if not held_cons:
            return True
        subtree = set(_subtree_ids(oid, children_map))
        rel = [c for c in held_cons if subtree & set(c[1:])]
        if not rel:
            return True
        idx = {
            j: obj_ir(j, [rep[j]["pos"][0] + dx, rep[j]["pos"][1] + dy, rep[j]["pos"][2]])
            if j in subtree
            else obj_ir(j)
            for j in rep
        }
        return all(holds(c, None, walls, idx) for c in rel)

    def subtree_inside_1mm(oid: str, dx: float, dy: float) -> bool:
        for ch in _subtree_ids(oid, children_map):
            ch_pos = [rep[ch]["pos"][0] + dx, rep[ch]["pos"][1] + dy, rep[ch]["pos"][2]]
            if not grown_1mm.covers(Polygon(footprint(obj_ir(ch, ch_pos)))):
                return False
        return True

    # Fixed non-window obstacles (doors, columns, stairs)
    fixed_blocking_polys: list[tuple[dict[str, Any], Polygon]] = []
    for f in room.get("fixed") or []:
        fcat = norm_cat(f.get("category") or f["id"].rsplit("_", 1)[0].replace("_", " "))
        if {"window", "windows"}.intersection(fcat.split()):
            continue
        fixed_blocking_polys.append((f, Polygon(footprint(f))))

    def blocks_fixed(cand_o: dict[str, Any], cand_fp: Polygon) -> bool:
        for f, fpoly in fixed_blocking_polys:
            if (
                min(cand_o["pos"][2] + cand_o["size"][2], f["pos"][2] + f["size"][2])
                - max(cand_o["pos"][2], f["pos"][2])
                > 0.05
                and cand_fp.intersection(fpoly).area > 0.01
            ):
                return True
        return False

    # 3. Snap any remaining floor objects (> 0.1 mm outside walls or blocking a door) inside & clear of doors
    floor_ids = [oid for oid, p in rep.items() if not p.get("on") and oid not in wall_mounted_ids]
    for oid in floor_ids:
        o = obj_ir(oid)
        fp = Polygon(footprint(o))
        need_wall = not subtree_inside_1mm(oid, 0.0, 0.0)
        need_door = blocks_fixed(o, fp)
        if not need_wall and not need_door:
            continue
        orig_x, orig_y = float(o["pos"][0]), float(o["pos"][1])
        # If center is outside floor_poly, project onto floor_poly first so spiral searches inside the room
        base_x, base_y = orig_x, orig_y
        if not floor_poly.contains(Point(orig_x, orig_y)):
            near = floor_poly.exterior.interpolate(floor_poly.exterior.project(Point(orig_x, orig_y)))
            base_x, base_y = float(near.x), float(near.y)

        spiral_pts = [(base_x + dx, base_y + dy) for dx, dy in [(0.0, 0.0)] + SPIRAL_SMALL + SPIRAL_WIDE]
        grid_pts = _interior_grid_points(floor_poly, orig_x, orig_y)
        all_target_pts = spiral_pts + grid_pts

        placed = False
        for require_held in (True, False):
            for tx, ty in all_target_pts:
                dx, dy = tx - orig_x, ty - orig_y
                cand_pos = [tx, ty, o["pos"][2]]
                cand_o = obj_ir(oid, cand_pos)
                cand_fp = Polygon(footprint(cand_o))
                if not grown_1mm.covers(cand_fp) or not subtree_inside_1mm(oid, dx, dy):
                    continue
                if blocks_fixed(cand_o, cand_fp):
                    continue
                if require_held and not keeps_held(oid, dx, dy):
                    continue
                _move_with_children(rep, oid, dx, dy, children_map)
                placed = True
                break
            if placed:
                break

    if not resolve_collisions:
        snap_supports_v1_1(room, rep)
        return rep

    # 4a. De-collide sibling children on the same parent surface first (including thin/small tabletop items)
    for par_id, ch_ids in children_map.items():
        if len(ch_ids) < 2 or par_id not in rep:
            continue
        par = rep[par_id]
        par_obj = {"pos": par["pos"], "size": par["size"], "yaw": par["yaw"]}
        par_poly = Polygon(footprint(par_obj))
        if not par_poly.is_valid:
            par_poly = par_poly.buffer(0)
        ch_sorted = sorted(ch_ids, key=lambda cid: -(rep[cid]["size"][0] * rep[cid]["size"][1]))
        placed_ch: list[str] = []
        for cid in ch_sorted:
            co = obj_ir(cid)
            cfp = Polygon(footprint(co))

            def child_collides(cand_co: dict[str, Any], cand_cfp: Polygon) -> bool:
                for ocid in placed_ch:
                    other = obj_ir(ocid)
                    if (
                        min(cand_co["pos"][2] + cand_co["size"][2], other["pos"][2] + other["size"][2])
                        - max(cand_co["pos"][2], other["pos"][2])
                        <= 1e-4
                    ):
                        continue
                    ofp = Polygon(footprint(other))
                    if cand_cfp.intersection(ofp).area > 1e-6:
                        return True
                return False

            if child_collides(co, cfp):
                ox, oy = float(co["pos"][0]), float(co["pos"][1])
                for dx, dy in SPIRAL_FINE + SPIRAL_SMALL + SPIRAL_WIDE:
                    nx, ny = ox + dx, oy + dy
                    if not support_contains(par_obj, [nx, ny], tol=0.04):
                        continue
                    cand_co = obj_ir(cid, [nx, ny, co["pos"][2]])
                    cand_cfp = Polygon(footprint(cand_co))
                    if not grown_1mm.covers(cand_cfp):
                        continue
                    if not child_collides(cand_co, cand_cfp):
                        _move_with_children(rep, cid, dx, dy, children_map)
                        break
            placed_ch.append(cid)

    # 4b. Iterative MTV / spiral + interior-grid de-collision for floor subtrees
    floor_ids_sorted = sorted(
        floor_ids,
        key=lambda oid: -(rep[oid]["size"][0] * rep[oid]["size"][1]),
    )
    placed_tree_ids: list[str] = []
    for oid in floor_ids_sorted:
        subtree = _subtree_ids(oid, children_map)
        o = obj_ir(oid)
        if head(o) in FLAT or o["size"][2] < 0.06:
            placed_tree_ids.extend(subtree)
            continue

        def subtree_has_main_collision(dx: float, dy: float) -> bool:
            for m_id in subtree:
                m_pos = [rep[m_id]["pos"][0] + dx, rep[m_id]["pos"][1] + dy, rep[m_id]["pos"][2]]
                cand_m = obj_ir(m_id, m_pos)
                if head(cand_m) in FLAT or cand_m["size"][2] < 0.06:
                    continue
                cand_fp = Polygon(footprint(cand_m))
                if not m_id or not rep[m_id].get("on"):
                    if blocks_fixed(cand_m, cand_fp):
                        return True
                for other_id in placed_tree_ids:
                    other = obj_ir(other_id)
                    if head(other) in FLAT or other["size"][2] < 0.06:
                        continue
                    if other.get("parent") == m_id or cand_m.get("parent") == other_id:
                        continue
                    if (
                        min(cand_m["pos"][2] + cand_m["size"][2], other["pos"][2] + other["size"][2])
                        - max(cand_m["pos"][2], other["pos"][2])
                        <= 0.05
                    ):
                        continue
                    other_fp = Polygon(footprint(other))
                    if cand_fp.intersection(other_fp).area > 0.01:
                        if collision_kind(cand_m, other, cand_fp, other_fp, tol=0.05) == "main":
                            return True
            return False

        if subtree_has_main_collision(0.0, 0.0):
            orig_x, orig_y = float(o["pos"][0]), float(o["pos"][1])
            spiral_deltas = SPIRAL_SMALL + SPIRAL_WIDE
            grid_deltas = [(gx - orig_x, gy - orig_y) for gx, gy in _interior_grid_points(floor_poly, orig_x, orig_y)]
            all_deltas = spiral_deltas + grid_deltas

            resolved = False
            for require_held in (True, False):
                for dx, dy in all_deltas:
                    if not subtree_inside_1mm(oid, dx, dy):
                        continue
                    if require_held and not keeps_held(oid, dx, dy):
                        continue
                    if not subtree_has_main_collision(dx, dy):
                        _move_with_children(rep, oid, dx, dy, children_map)
                        resolved = True
                        break
                if resolved:
                    break
        placed_tree_ids.extend(subtree)

    # 4c. De-collide wall-mounted objects against placed floor objects, fixed obstacles, and other wall objects
    wall_ids_sorted = sorted(
        [oid for oid in rep if oid in wall_mounted_ids and not rep[oid].get("on")],
        key=lambda oid: -(rep[oid]["size"][0] * rep[oid]["size"][1]),
    )
    for wid in wall_ids_sorted:
        wo = obj_ir(wid)
        if head(wo) in FLAT or wo["size"][2] < 0.06:
            placed_tree_ids.append(wid)
            continue

        def wall_collides(cand_wo: dict[str, Any], cand_fp: Polygon) -> bool:
            if blocks_fixed(cand_wo, cand_fp):
                return True
            for other_id in placed_tree_ids:
                other = obj_ir(other_id)
                if head(other) in FLAT or other["size"][2] < 0.06:
                    continue
                if (
                    min(cand_wo["pos"][2] + cand_wo["size"][2], other["pos"][2] + other["size"][2])
                    - max(cand_wo["pos"][2], other["pos"][2])
                    <= 0.05
                ):
                    continue
                other_fp = Polygon(footprint(other))
                if cand_fp.intersection(other_fp).area > 0.01:
                    if collision_kind(cand_wo, other, cand_fp, other_fp, tol=0.05) == "main":
                        return True
            return False

        wfp = Polygon(footprint(wo))
        if wall_collides(wo, wfp):
            for wx, wy, wz, wyaw in _wall_candidate_poses(room, rep[wid], floor_poly):
                cand_wo = obj_ir(wid, [wx, wy, wz], wyaw)
                cand_fp = Polygon(footprint(cand_wo))
                if not grown_1mm.covers(cand_fp):
                    continue
                if not holds(["against_wall", wid], [cand_wo], walls):
                    continue
                if not wall_collides(cand_wo, cand_fp):
                    rep[wid]["pos"] = [wx, wy, wz]
                    rep[wid]["yaw"] = wyaw
                    break
        placed_tree_ids.append(wid)

    # 5. Final support snap for all children and wall-mounted objects
    snap_supports_v1_1(room, rep)
    return rep


def check_v1_1(room: dict[str, Any]) -> dict[str, Any]:
    """Run `fastfill.validate.check(room)` while treating wall-mounted (`anchor == 'wall'`) objects as wall-supported."""
    from fastfill.validate import check

    c = check(room)
    wall_objs = [
        o
        for o in (room.get("objects") or [])
        if o.get("anchor") == "wall"
    ]
    if not wall_objs:
        return c
    b_pts = room.get("boundary") or []
    h_room = room.get("height")
    ok_wall_ids: set[str] = set()
    bad_wall_ids: set[str] = set()
    for o in wall_objs:
        z_ok = float(o["pos"][2]) >= -0.05 and (not h_room or float(o["pos"][2]) + float(o["size"][2]) <= float(h_room) + 0.05)
        wall_ok = not b_pts or holds(["against_wall", o["id"]], room["objects"], b_pts)
        if z_ok and wall_ok:
            ok_wall_ids.add(o["id"])
        else:
            bad_wall_ids.add(o["id"])
    sf = [oid for oid in c["support_fail"] if oid not in ok_wall_ids]
    for oid in bad_wall_ids:
        if oid not in sf:
            sf.append(oid)
    c["support_fail"] = sf
    c["valid"] = not (c["oob"] or c["support_fail"] or c["ceiling"])
    return c


if __name__ == "__main__":
    from fastfill.validate import check

    test_room = {
        "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
        "boundary_type": "polygon",
        "height": 2.7,
        "fixed": [{"id": "door_1", "category": "door", "size": [0.1, 0.9, 2.1], "pos": [0.05, 0.8, 0], "yaw": 0}],
        "objects": [
            {"id": "bed_1", "category": "bed", "size": [2.0, 1.6, 0.6], "pos": [1.2, 1.5, 0], "yaw": 0},
            {"id": "desk_1", "category": "desk", "size": [1.2, 0.6, 0.75], "pos": [1.3, 1.5, 0], "yaw": 0},
            {"id": "lamp_1", "category": "lamp", "size": [0.2, 0.2, 0.4], "pos": [1.3, 1.5, 0.3], "yaw": 0, "parent": "desk_1"},
        ],
    }
    raw_pl = {
        "bed_1": {"size": [2.0, 1.6, 0.6], "pos": [1.2, 1.5, 0.0], "yaw": 0.0, "on": None},
        "desk_1": {"size": [1.2, 0.6, 0.75], "pos": [1.3, 1.5, 0.0], "yaw": 0.0, "on": None},
        "lamp_1": {"size": [0.2, 0.2, 0.4], "pos": [1.3, 1.5, 0.3], "yaw": 0.0, "on": "desk_1"},
    }
    c_before = check(apply_v1_1(test_room, raw_pl))
    assert c_before["collisions"]["main"] and c_before["support_fail"] == ["lamp_1"]
    fixed_pl = repair_v1_1(test_room, raw_pl)
    c_after = check(apply_v1_1(test_room, fixed_pl))
    assert c_after["valid"] and not c_after["collisions"]["main"] and not c_after["support_fail"], c_after
    print("postprocess.py self-check ok")
