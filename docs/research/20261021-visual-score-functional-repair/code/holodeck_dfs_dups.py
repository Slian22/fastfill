"""Probe: Holodeck-style grid DFS re-placement of EXACT-DUPLICATE objects only (everything else frozen).

Holodeck (floor_objects.py): grid = room extent / 20, rotations {0,90,180,270}, hard = inside room + no collision
(door/window boxes are obstacles), soft = edge 1.0 / near 1.8 (80 cm) / direction 0.5 / alignment 0.5 (5 cm),
greedy per object after the first level. Here the LLM relations are replaced by the raw layout itself: the frozen
twin of a duplicate plays the anchor ("near" + "center aligned" + "face same as"; "edge" if the twin is at a wall).
Small objects on a parent use a grid over the parent top (Holodeck delegates these to THOR RandomSpawn).
Hard constraints mirror our checker (validate.check pair rule), strict R2 (floor envelope minus door keep-outs) and
A4 (AABB overlap > 1 cm, same parent, height ignored). Nothing is dropped: an unplaceable duplicate stays raw.

python3 -I holodeck_dfs_dups.py --repo REPO --rooms rooms.jsonl --requests requests.jsonl --geometry geo.jsonl
        --s1 geo-s1.jsonl --system rooms.jsonl --out DIR [--limit N] [--workers N] [--no-s1]
"""
import argparse, collections, json, math, os, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.affinity import affine_transform

TOL, MIN_AREA, ENV = 0.05, 0.01, 1e-3
CARD = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
W_GLOBAL, W_DIST, W_DIR, W_ALIGN, NEAR, ALIGN_EPS, EDGE = 1.0, 1.8, 0.5, 0.5, 0.8, 0.05, 0.10
STRATA = ((1, 10), (11, 20), (21, 50), (51, 10 ** 9))


def _imports(repo):
    sys.path.insert(0, repo)
    global r, d, g, footprint, head, FLAT, holds, support_contains
    from fastfill.comparison import output_repair as r, downstream as d, geometry as g
    from fastfill.scene import footprint, head
    from fastfill.anchors import FLAT
    from fastfill.validate import holds, support_contains


def dup_pairs(objs):
    seen, out = {}, []
    for o in objs:
        k = (round(o['pos'][0], 3), round(o['pos'][1], 3), round(o['pos'][2], 3), round(o['yaw'], 6))
        if k in seen:
            out.append((o['id'], seen[k]))
        else:
            seen[k] = o['id']
    return out


def corners(xs, ys, yaw, sx, sy):
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = sx / 2, sy / 2
    off = np.array([(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)])
    rot = off @ np.array([[c, s], [-s, c]])          # (4,2): x' = c*dx - s*dy, y' = s*dx + c*dy
    return np.stack([xs, ys], -1)[:, None, :] + rot[None]


def hover(a, b):
    return min(a['pos'][2] + a['size'][2], b['pos'][2] + b['size'][2]) - max(a['pos'][2], b['pos'][2])


def blocks(cand, o):                                  # validate.check pair relevance
    if o.get('parent') == cand['id'] or cand.get('parent') == o['id'] or o['id'] == cand['id']:
        return False
    if hover(cand, o) <= TOL or FLAT.intersection((head(cand), head(o))) or min(cand['size'][2], o['size'][2]) < 0.06:
        return False
    return True


def blocks_fixed(cand, f):
    if {'window', 'windows'} & set(str(f.get('category', '')).lower().split()):
        return False
    return not (head(cand) in FLAT or cand['size'][2] < 0.06 or hover(cand, f) <= TOL)


def aabbs(pts, m):                                     # (N,4,2) corners -> (N,4) AABB in the P2 frame
    a, b, c_, d_, e, f = m
    x = a * pts[..., 0] + b * pts[..., 1] + e
    y = c_ * pts[..., 0] + d_ * pts[..., 1] + f
    return np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)


def a4_free(cbb, obb):                                 # downstream._overlaps with TOL_M on both axes
    if not len(obb):
        return np.ones(len(cbb), bool)
    ox = np.minimum(cbb[:, None, 2], obb[None, :, 2]) - np.maximum(cbb[:, None, 0], obb[None, :, 0])
    oy = np.minimum(cbb[:, None, 3], obb[None, :, 3]) - np.maximum(cbb[:, None, 1], obb[None, :, 1])
    return ~((ox > d.TOL_M) & (oy > d.TOL_M)).any(1)


def place_one(o, twin, by, fixed, ctx):
    """Best Holodeck-scored legal pose for duplicate o, or None."""
    floor, env, m, ext = ctx['floor'], ctx['env'], ctx['m'], ctx['ext']
    parent = by.get(o.get('parent')) or fixed.get(o.get('parent'))
    if o.get('parent') and parent is None:
        return None
    if parent is None:
        minx, miny, maxx, maxy = floor.bounds
        step = min(0.30, max(0.05, max(maxx - minx, maxy - miny) / 20))     # Holodeck grid_density 20
        gx, gy = np.meshgrid(np.arange(minx + step / 2, maxx, step), np.arange(miny + step / 2, maxy, step))
        pts = shapely.points(np.stack([gx.ravel(), gy.ravel()], -1))
        keep = shapely.contains(floor, pts)
        xs, ys, z = gx.ravel()[keep], gy.ravel()[keep], 0.0
        yaws = [o['yaw']] + [c for c in CARD if abs((c - o['yaw'] + math.pi) % math.tau - math.pi) > 1e-6]
    else:
        px, py = parent['pos'][:2]
        c, s = math.cos(parent['yaw']), math.sin(parent['yaw'])
        su, sv = (max(0.02, parent['size'][k] / 10) for k in (0, 1))
        u, v = np.meshgrid(np.arange(-parent['size'][0] / 2, parent['size'][0] / 2 + 1e-9, su),
                           np.arange(-parent['size'][1] / 2, parent['size'][1] / 2 + 1e-9, sv))
        xs, ys = px + c * u.ravel() - s * v.ravel(), py + s * u.ravel() + c * v.ravel()
        z = parent['pos'][2] + parent['size'][2]
        pfp = Polygon(footprint(parent))
        keep = shapely.distance(shapely.points(np.stack([xs, ys], -1)), pfp) <= TOL
        xs, ys = xs[keep], ys[keep]
        yaws = [o['yaw']]
    if not len(xs):
        return None
    probe = {**o, 'pos': [0, 0, z]}
    obs = [p for p in by.values() if blocks(probe, p)]
    obs_polys = [Polygon(footprint(p)) for p in obs] + [Polygon(footprint(f)) for f in fixed.values() if blocks_fixed(probe, f)]
    obs_arr = np.array(obs_polys, dtype=object)
    tree = shapely.STRtree(obs_polys) if obs_polys else None
    sib = [p for p in by.values() if p['id'] != o['id'] and p.get('parent') == o.get('parent')
           and not d.covering(p, by.get(p.get('parent')))]
    sib_bb = np.array([d.aabb(p, m) for p in sib]) if sib and not d.covering(probe, parent) else np.zeros((0, 4))
    tx, ty = twin['pos'][:2]
    twin_edge = parent is None and ext.distance(Polygon(footprint(twin))) <= EDGE
    best = None
    for yaw in yaws:
        pts = corners(xs, ys, yaw, o['size'][0], o['size'][1])
        polys = shapely.polygons(pts)
        ok = shapely.contains(env, polys)
        if tree is not None and ok.any():                 # bbox tree query, exact area only on the hits
            ci, oi = tree.query(polys, predicate='intersects')
            if len(ci):
                area = shapely.area(shapely.intersection(polys[ci], obs_arr[oi]))
                ok[np.unique(ci[area > MIN_AREA])] = False
        if ok.any():
            idx = np.flatnonzero(ok)
            ok[idx] = a4_free(aabbs(pts[idx], m), sib_bb)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        dist = np.hypot(xs[idx] - tx, ys[idx] - ty)
        score = W_DIST * (1 - np.minimum(dist, NEAR) / NEAR)
        score += W_ALIGN * ((np.abs(xs[idx] - tx) < ALIGN_EPS) | (np.abs(ys[idx] - ty) < ALIGN_EPS))
        score += W_DIR * (abs(yaw - o['yaw']) < 1e-9)
        if twin_edge:
            score += W_GLOBAL * (shapely.distance(polys[idx], ext) <= EDGE)
        k = np.lexsort((dist, -score))[0]
        cand = (score[k], -dist[k], [round(float(xs[idx][k]), 3), round(float(ys[idx][k]), 3), round(z, 3)], yaw)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return None if best is None else best[2:]


def replace_dups(room):
    objs = room['objects']
    by = {o['id']: o for o in objs}
    fixed = {f['id']: f for f in room.get('fixed') or []}
    poly = Polygon(room['boundary'])
    strips = [s['strip'] for s in d.door_strips(room)]
    floor = d.keepout_floor(poly, strips)[1]
    if floor.is_empty:
        floor = poly
    env = floor.buffer(ENV)
    shapely.prepare(env)
    ctx = {'floor': floor, 'env': env, 'm': d.frame(room['boundary'])[1], 'ext': poly.exterior}
    moved, unplaced = [], []
    for oid, tid in dup_pairs(objs):
        o = by[oid]
        got = place_one(o, by[tid], by, fixed, ctx)
        if got is None:
            unplaced.append(oid)
            continue
        pos, yaw = got
        dyaw, (ox, oy) = yaw - o['yaw'], o['pos'][:2]
        c, s = math.cos(dyaw), math.sin(dyaw)
        for ch in objs:                                   # the support subtree follows (translate + rotate)
            if ch.get('parent') == oid:
                dx, dy = ch['pos'][0] - ox, ch['pos'][1] - oy
                ch['pos'] = [round(pos[0] + c * dx - s * dy, 3), round(pos[1] + s * dx + c * dy, 3), ch['pos'][2]]
                ch['yaw'] = (ch['yaw'] + dyaw) % math.tau
        o['pos'], o['yaw'] = pos, yaw % math.tau
        moved.append(oid)
    return moved, unplaced


def metrics(room):
    gm = g.geometry(room)
    acc = d.accept(room, room['objects'])
    return {'checker': bool(gm['geometry_success']), 'accepted': bool(acc['accepted']),
            'strict': bool(gm['geometry_success'] and acc['accepted']),
            'violations': int(gm['main_collisions'] + gm['severe_collisions'] + gm['fixed_blocking'] + gm['oob_objects']
                              + gm['support_fail'] + gm['ceiling_fail']),
            'reasons': acc['reasons']}


def one(task):
    room, messages, raw_layout, use_s1 = task
    t0 = time.monotonic()
    pred = r.predicted_room(room, messages, raw_layout)
    raw = {o['id']: o for o in pred['objects']}
    pairs = dup_pairs(pred['objects'])
    work = {**pred, 'objects': [dict(o) for o in pred['objects']]}
    moved, unplaced = replace_dups(work)
    row = {'uid': room['uid'], 'source': room['source'], 'objects': len(raw_layout), 'dups': len(pairs),
           'floor_dups': sum(raw[i].get('parent') is None for i, _ in pairs),
           'child_dups': sum(raw[i].get('parent') is not None for i, _ in pairs),
           'moved': len(moved), 'unplaced': len(unplaced), 'raw': metrics(pred), 'dfs': metrics(work)}
    aw = lambda o, objs: holds(['against_wall', o['id']], objs, work['boundary'])
    wall_raw = [i for i in moved if raw[i].get('parent') is None and aw(raw[i], pred['objects'])]
    cur = {o['id']: o for o in work['objects']}
    row['wall_kept'] = [sum(aw(cur[i], work['objects']) for i in wall_raw), len(wall_raw)]
    row['yaw_changed'] = sum(abs(cur[i]['yaw'] - raw[i]['yaw']) > 1e-6 for i in moved)
    row['displacement_m'] = [round(math.dist(cur[i]['pos'][:2], raw[i]['pos'][:2]), 3) for i in moved]
    if use_s1:
        res = r.repair(work)
        row['dfs_s1'] = {**metrics(res['room']), 'status': res['status'], 'rolled_back': res['diagnostics'].get('rolled_back')}
    row['layout'] = [{k: o[k] for k in ('id', 'category', 'size', 'pos', 'yaw', 'parent')} for o in work['objects']]
    row['runtime_s'] = round(time.monotonic() - t0, 3)
    return row


def stratum(n):
    return next(f'{a}-{b if b < 10 ** 9 else ""}' for a, b in STRATA if a <= n <= b)


def rate(rows, f):
    return round(sum(f(x) for x in rows) / len(rows), 4) if rows else None


def summarize(rows, s1, system):
    def block(sub):
        out = {'rooms': len(sub)}
        for name, fn in (('raw_checker', lambda x: x['raw']['checker']), ('raw_strict', lambda x: x['raw']['strict']),
                         ('dfs_checker', lambda x: x['dfs']['checker']), ('dfs_strict', lambda x: x['dfs']['strict']),
                         ('dfs_s1_checker', lambda x: x.get('dfs_s1', {}).get('checker', False)),
                         ('dfs_s1_strict', lambda x: x.get('dfs_s1', {}).get('strict', False)),
                         ('s1_alone_checker', lambda x: s1.get(x['uid'], {}).get('geometry_success_s1', False)),
                         ('s1_alone_strict', lambda x: s1.get(x['uid'], {}).get('geometry_success_s1', False)
                          and s1.get(x['uid'], {}).get('accepted_s1', False)),
                         ('system_checker', lambda x: system.get(x['uid'], {}).get('checker_pass', False)),
                         ('system_strict', lambda x: system.get(x['uid'], {}).get('checker_pass', False)
                          and system.get(x['uid'], {}).get('accepted', False)),
                         ('all_dups_placed', lambda x: x['unplaced'] == 0)):
            out[name] = rate(sub, fn)
        out['system_stage'] = dict(collections.Counter(system.get(x['uid'], {}).get('stage') for x in sub))
        out['mean_violations_raw'] = round(statistics.mean(x['raw']['violations'] for x in sub), 2) if sub else None
        out['mean_violations_dfs'] = round(statistics.mean(x['dfs']['violations'] for x in sub), 2) if sub else None
        out['dups'] = sum(x['dups'] for x in sub)
        out['unplaced'] = sum(x['unplaced'] for x in sub)
        wk = [x['wall_kept'] for x in sub]
        out['against_wall_kept'] = [sum(a for a, _ in wk), sum(b for _, b in wk)]
        out['yaw_changed'] = sum(x['yaw_changed'] for x in sub)
        disp = [v for x in sub for v in x['displacement_m']]
        out['median_displacement_m'] = round(statistics.median(disp), 3) if disp else None
        out['mean_runtime_s'] = round(statistics.mean(x['runtime_s'] for x in sub), 2) if sub else None
        return out
    kind = lambda x: 'floor_only' if not x['child_dups'] else 'child_only' if not x['floor_dups'] else 'both'
    return {'all': block(rows),
            'by_objects': {stratum(n): block([x for x in rows if stratum(x['objects']) == stratum(n)]) for n in (1, 11, 21, 51)},
            'by_dup_kind': {k: block([x for x in rows if kind(x) == k]) for k in ('floor_only', 'child_only', 'both')}}


def main():
    ap = argparse.ArgumentParser()
    for k in ('repo', 'rooms', 'requests', 'geometry', 's1', 'system', 'out'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument('--no-s1', action='store_true')
    a = ap.parse_args()
    _imports(a.repo)
    load = lambda p: {j['uid']: j for j in map(json.loads, open(p))}
    rooms, reqs = load(a.rooms), load(a.requests)
    tasks = []
    for line in open(a.geometry):
        j = json.loads(line)
        if j['complete'] and dup_pairs(j['raw_layout']):
            tasks.append((rooms[j['uid']], reqs[j['uid']]['messages'], j['raw_layout'], not a.no_s1))
    tasks.sort(key=lambda t: -len(t[2]))
    if a.limit:
        tasks = tasks[::max(1, len(tasks) // a.limit)][:a.limit]
    s1 = {j['uid']: {k: j['s1'][k] for k in ('geometry_success_s1', 'accepted_s1', 'status')} for j in map(json.loads, open(a.s1))}
    system = {j['uid']: {k: j[k] for k in ('checker_pass', 'accepted', 'stage')} for j in map(json.loads, open(a.system))}
    os.makedirs(a.out, exist_ok=True)
    rows, t0 = [], time.monotonic()
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(a.workers, initializer=_imports, initargs=(a.repo,)) as ex, open(os.path.join(a.out, 'rows.jsonl'), 'w') as f:
        for fut in as_completed([ex.submit(one, t) for t in tasks]):
            row = fut.result()
            rows.append(row)
            f.write(json.dumps(row) + '\n')
            if len(rows) % 100 == 0:
                print(len(rows), 'rooms', round(time.monotonic() - t0), 's', flush=True)
    summary = {'rooms': len(rows), 'wall_s': round(time.monotonic() - t0), 'summary': summarize(rows, s1, system)}
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary['summary']['all'], indent=1))


if __name__ == '__main__':
    main()
