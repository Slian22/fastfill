"""Symbolic MoMa executability checks on IR top-down layouts (protocol H5, descriptive).
Thresholds + sources: ../moma-checks/THRESHOLDS.md (fixed before any number). Robot-side meaning of see / reach / open:
../moma-checks/ROBOT_CONTRACT.md. Checks: base_pose (disc base in the door-reached grid within arm reach R of the
footprint), visibility (2.5D rays at camera height), approach/opening clearance (front rectangle; door swing),
goal-relation top_free. Arms: GT, raw, S1 (s2-render/cohort-s1.jsonl), S2 (s2f/cohort/merged-s2.jsonl), S2F
(s2f/cohort/final-s2f.jsonl, s1_layout field).
    python3 moma_checks.py            # cohort -> moma-checks/rows.jsonl
    python3 moma_checks.py --report   # rows.jsonl -> moma-checks/report.md (tables + render-free spot check)
    python3 moma_checks.py --selftest
"""
import json, math, re, sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013')
P = ROOT / 'probes-20261021'
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
sys.path.insert(0, str(P / 'scripts'))
from fastfill.comparison import downstream as d          # noqa: E402
from fastfill.interface import snap_inside                # noqa: E402
from fastfill.scene import footprint, norm_cat            # noqa: E402
import task_reach_headroom as T                           # noqa: E402

DATA = ROOT / 'visual-final-selected-002/inputs/eval-data-v1'
OUT = P / 'moma-checks'
G = d.NAV_GRID_M                                    # 0.25, nav.py:43
RADII = (0.2, 0.25)                                 # nav.py:40; Stretch-like variant
REACH = (0.4, 0.6, 0.8)                             # arm reach from the footprint edge
R0, REACH0 = 0.2, 0.6
CAM_H, D_FRONT, DOOR_DEFAULT, TOP_SQ, TOP_MAX_H = 1.2, 0.6, 0.5, 0.15, 1.22
TOL, COVER_H = d.TOL_M, d.FLOOR_COVERING_H_M

STRONG = {'cabinet', 'cabinetry', 'drawer', 'wardrobe', 'cupboard'}
NONFUNC = {'chair', 'stool', 'sofa', 'sofastool', 'bench', 'ottoman', 'pouf', 'bed', 'lamp'}
FRONT = {'wardrobe', 'armoire', 'dresser', 'nightstand', 'bedside', 'fridge', 'refrigerator', 'oven', 'stove', 'range',
         'cooker', 'dishwasher', 'washing', 'dryer', 'microwave', 'bookshelf', 'bookcase', 'shelf', 'sideboard',
         'console', 'desk', 'vanity', 'pantry', 'counter', 'countertop', 'island', 'cooler', 'media', 'entertainment',
         'tv', 'dressing', 'sink', 'washbasin'} | STRONG
WORDS = FRONT | NONFUNC | {'table'}


def kind(cat):
    """'front' (openable / storage: clearance in front of local +X), 'table' (any side) or None."""
    t = {'shelf' if w == 'shelves' else w[:-1] if w.endswith('s') and w[:-1] in WORDS else w
         for w in re.split(r"[\s_/\-']+", cat.lower()) if w}
    if t & STRONG:
        return 'front'
    if t & NONFUNC:
        return None
    return 'front' if t & FRONT else 'table' if 'table' in t else None


def lrect(o, x0, x1, y0, y1):
    """Rectangle [x0,x1] x [y0,y1] in o's local frame, as a room-frame polygon."""
    c, s = math.cos(o['yaw']), math.sin(o['yaw'])
    x, y = o['pos'][0], o['pos'][1]
    return Polygon([(x + c * u - s * v, y + s * u + c * v) for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])


def realise(room, layout):
    """The layout as serve.py would realise it: downstream.accept's R2 snap, then R4 parents."""
    poly = orient(Polygon(room['boundary']))
    doors = d.door_strips(room)
    _, floor = d.keepout_floor(poly, [s['strip'] for s in doors])
    pl = {o['id']: {'pos': list(o['pos']), 'yaw': o['yaw'], 'on': o.get('parent')} for o in layout}
    if not floor.is_empty:
        snap_inside(pl, {'objects': layout, 'fixed': []}, floor, (), room['boundary'])
    by = {o['id']: {**o, 'pos': pl[o['id']]['pos']} for o in layout}
    theta, m = d.frame(room['boundary'])
    parent = d.r4_parents(by, {i: d.aabb(o, m) for i, o in by.items()})
    return poly, doors, m, by, parent


def grid(fpoly, aabbs, r):
    """downstream.accept's free-cell lattice with base radius r (P2 frame)."""
    w, dd = fpoly.bounds[2:]
    lo = math.ceil(r / G - 0.5)
    ux, uy = (math.floor((e - r) / G - 0.5) for e in (w, dd))
    rects = [d._inflate(b, r) for b in aabbs]
    blocked = set().union(*(d._cells(x, lo, ux, uy) for x in rects))
    inner = prep(fpoly.buffer(-r))
    free = {(i, j) for i in range(lo, ux + 1) for j in range(lo, uy + 1)
            if (i, j) not in blocked and inner.covers(Point((i + 0.5) * G, (j + 0.5) * G))}
    return free, rects, (w, dd), lo, ux, uy


def flood(start, free):
    seen, q = {start}, deque([start])
    while q:
        i, j = q.popleft()
        for n in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if n in free and n not in seen:
                seen.add(n)
                q.append(n)
    return seen


def reached(g, seeds):
    """Union of the cells flooded from every door seed; doorless (seeds None): the largest free component."""
    if seeds is not None:
        return set().union(*(d._reach(g, s)[0] for s in seeds)) if seeds else set()
    best, seen = set(), set()
    for c in sorted(g[0]):
        if c not in seen:
            comp = flood(c, g[0])
            seen |= comp
            best = max(best, comp, key=len)
    return best


def clip(a, b, hx, hy):
    """Liang-Barsky: parameters [t0, t1] of segment a-b inside [-hx,hx] x [-hy,hy], or None."""
    t0, t1 = 0.0, 1.0
    for p0, dp, h in ((a[0], b[0] - a[0], hx), (a[1], b[1] - a[1], hy)):
        if abs(dp) < 1e-12:
            if abs(p0) > h:
                return None
            continue
        u, v = sorted(((-h - p0) / dp, (h - p0) / dp))
        t0, t1 = max(t0, u), min(t1, v)
        if t0 > t1:
            return None
    return t0, t1


def blocked_ray(cam, p, zt, occ):
    """2.5D: the ray (cam, CAM_H) -> (p, zt) passes below the top of some floor-extruded occluder box."""
    for ox, oy, c, s, hx, hy, top in occ:
        a = (c * (cam[0] - ox) + s * (cam[1] - oy), -s * (cam[0] - ox) + c * (cam[1] - oy))
        b = (c * (p[0] - ox) + s * (p[1] - oy), -s * (p[0] - ox) + c * (p[1] - oy))
        t = clip(a, b, hx, hy)
        if t is not None and top > CAM_H + t[1] * (zt - CAM_H):     # lowest ray height on the crossing (zt <= CAM_H)
            return True
    return False


def nearest_visible(cells, dist, tgt, occ, walls):
    """Distance (to the footprint) of the nearest reached cell with an unblocked ray to tgt, or None."""
    zt = min(tgt['pos'][2] + tgt['size'][2], CAM_H)
    cx, cy = tgt['pos'][:2]
    pts = [(cx, cy)] + [(cx + 0.9 * (x - cx), cy + 0.9 * (y - cy)) for x, y in footprint(tgt)]
    occ = [o for o in occ if o[6] > zt]                  # the ray never dips below zt, so lower boxes cannot block
    for k in np.argsort(dist, kind='stable'):
        cam = tuple(cells[k])
        for p in pts:
            if walls is not None and not walls.covers(LineString([cam, p])):
                continue
            if not blocked_ray(cam, p, zt, occ):
                return float(dist[k])
    return None


def free_rect(rect, cover, fps, skip):
    """(clear, blocker ids): rect inside the room and no floor piece overlapping it (shrunk by TOL)."""
    if not cover.covers(rect):
        return False, ['ROOM_BOUNDARY']
    shrunk = rect.buffer(-TOL, join_style=2)
    hits = sorted(i for i, fp in fps.items() if i not in skip and shrunk.intersection(fp).area > 0)
    return not hits, hits


def door_swings(room, poly, fps):
    """Per fixed door: a w x w rectangle inward from the wall (hinge side unknown), clear of floor pieces."""
    pts = list(poly.exterior.coords)
    edges = [(a, b) for a, b in zip(pts, pts[1:]) if math.dist(a, b) > 1e-9]
    out = []
    for f in room.get('fixed') or []:
        if not d.DOOR_WORDS & set(norm_cat(f.get('category')).split()):
            continue
        p = Point(f['pos'][:2])                          # the geometry of downstream.door_strips, depth = width
        gap, (a, b) = min(((LineString(e).distance(p), e) for e in edges), key=lambda t: t[0])
        length = math.dist(a, b)
        u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        n = (-u[1], u[0])
        t = min(max((p.x - a[0]) * u[0] + (p.y - a[1]) * u[1], 0.0), length)
        c, s = math.cos(f['yaw']), math.sin(f['yaw'])
        w = abs(f['size'][0] * (u[0] * c + u[1] * s)) + abs(f['size'][1] * (u[1] * c - u[0] * s))
        r = w if w > 0.05 else DOOR_DEFAULT
        at = lambda along, depth: (a[0] + u[0] * along + n[0] * depth, a[1] + u[1] * along + n[1] * depth)
        lo, hi = max(t - r / 2, 0.0), min(t + r / 2, length)
        sw = Polygon([at(lo, 0), at(hi, 0), at(hi, r), at(lo, r)]).intersection(poly)
        shrunk = sw.buffer(-TOL, join_style=2)
        hits = sorted(i for i, fp in fps.items() if shrunk.intersection(fp).area > 0)
        out.append({'id': f['id'], 'w': round(w, 3), 'ok': not hits, 'hits': hits, 'detached': gap > d.DETACHED_DOOR_M})
    return out


def top_covers(o, by, parent):
    """Ids covering o's top: its children (R4) and anything overlapping it in 2D that rises above the top."""
    top, fo = o['pos'][2] + o['size'][2], Polygon(footprint(o))
    return [j for j, q in by.items() if j != o['id'] and (parent.get(j) == o['id'] or (
        q['pos'][2] < top + TOP_SQ and q['pos'][2] + q['size'][2] > top + TOL and Polygon(footprint(q)).intersects(fo)))]


def top_free(o, by, parent):
    """A TOP_SQ x TOP_SQ square fits on o's top, clear of top_covers."""
    sx, sy = o['size'][:2]
    if sx <= TOP_SQ or sy <= TOP_SQ:
        return False
    h = TOP_SQ / 2
    c, s = math.cos(o['yaw']), math.sin(o['yaw'])
    x, y = o['pos'][0], o['pos'][1]
    to_local = [c, s, -s, c, -(c * x + s * y), s * x - c * y]
    covers = [affine_transform(Polygon(footprint(by[j])), to_local).buffer(h, join_style=2)
              for j in top_covers(o, by, parent)]
    area = box(-sx / 2 + h, -sy / 2 + h, sx / 2 - h, sy / 2 - h)
    return not covers or area.difference(unary_union(covers)).area > 0


def evaluate(room, layout, cats):
    poly, doors, m, by, parent = realise(room, layout)
    fixed = [f for f in room.get('fixed') or []
             if not (d.DOOR_WORDS | {'window', 'windows'}) & set(norm_cat(f.get('category')).split())]
    floor = {i: o for i, o in by.items() if parent[i] is None and o['size'][2] >= COVER_H}
    floor.update({f['id']: f for f in fixed})
    fps = {i: Polygon(footprint(o)) for i, o in floor.items()}
    fpoly = affine_transform(poly, m)
    seeds = [(m[0] * s['seed'][0] + m[1] * s['seed'][1] + m[4], m[2] * s['seed'][0] + m[3] * s['seed'][1] + m[5])
             for s in doors] if doors else None
    c0, s0, x0, y0 = m[0], m[1], -m[4], -m[5]
    cells = {}
    for r in RADII:
        cf = np.array(sorted(reached(grid(fpoly, [d.aabb(o, m) for o in floor.values()], r), seeds)),
                      dtype=float).reshape(-1, 2)
        cf = (cf + 0.5) * G
        cr = np.column_stack([c0 * (cf[:, 0] + x0) - s0 * (cf[:, 1] + y0), s0 * (cf[:, 0] + x0) + c0 * (cf[:, 1] + y0)])
        cells[r] = (cf, cr, shapely.points(cr) if len(cr) else None)

    def chain(i):                                        # i and its ancestors (admit._floor_root walk)
        seen = [i]
        while parent.get(seen[-1]) and parent[seen[-1]] not in seen:
            seen.append(parent[seen[-1]])
        return seen
    anc = {i: chain(i) for i in by}
    occ_all = [(o['pos'][0], o['pos'][1], math.cos(o['yaw']), math.sin(o['yaw']), o['size'][0] / 2, o['size'][1] / 2,
                o['pos'][2] + o['size'][2], i) for i, o in list(by.items()) + [(f['id'], f) for f in fixed]
               if o['pos'][2] + o['size'][2] >= COVER_H]
    cover = prep(poly.buffer(TOL))
    walls = None if d.is_rectangle(poly) else prep(poly.buffer(0.02))
    want = {i for i, o in by.items() if o['category'] in cats} | {i for i in floor if i in by and kind(by[i]['category'])}
    objs = []
    for i in sorted(want):
        o, rt = by[i], anc[i][-1]
        fam = {j for j in by if i in anc[j]} | set(anc[i])
        row = {'id': i, 'cat': o['category'], 'kind': kind(o['category']), 'task': o['category'] in cats,
               'floor': i in floor, 'min_d': {}, 'a3': {}}
        for r in RADII:
            cf, cr, pts = cells[r]
            dist = shapely.distance(pts, Polygon(footprint(by[rt]))) if pts is not None else np.zeros(0)
            row['min_d'][str(r)] = round(float(dist.min()), 4) if len(dist) else None
            b = d._inflate(d.aabb(by[rt], m), r + G)
            row['a3'][str(r)] = bool(len(cf) and ((cf[:, 0] >= b[0]) & (cf[:, 0] <= b[2]) & (cf[:, 1] >= b[1]) &
                                                   (cf[:, 1] <= b[3])).any())
            if r == R0:
                v = nearest_visible(cr, dist, o, [x[:7] for x in occ_all if x[7] not in fam], walls)
                row['vis_d'] = None if v is None else round(v, 4)
        k = row['kind']
        if k and i in floor:
            sx, sy = o['size'][:2]
            sides = [(sx / 2, sx / 2 + D_FRONT, -sy / 2, sy / 2)]
            if k == 'table':
                sides += [(-sx / 2 - D_FRONT, -sx / 2, -sy / 2, sy / 2), (-sx / 2, sx / 2, sy / 2, sy / 2 + D_FRONT),
                          (-sx / 2, sx / 2, -sy / 2 - D_FRONT, -sy / 2)]
            res = [free_rect(lrect(o, *sd), cover, fps, {i}) for sd in sides]
            row['clear'] = any(ok for ok, _ in res)
            row['clear_hits'] = res[0][1]
            if k == 'front':
                sb = affine_transform(lrect(o, sx / 2, 1.5 * sx, -sy / 2, sy / 2), m).bounds
                row['eg_hits'] = sorted(j for j in floor if j != i and d._positive(sb, d.aabb(floor[j], m)))
                row['eg_front'] = not row['eg_hits']
            top = o['pos'][2] + o['size'][2]
            row['top_h'] = round(top, 3)
            row['recept'] = top <= TOP_MAX_H
            if row['recept']:
                row['top_free'] = top_free(o, by, parent)
        objs.append(row)
    return {'objs': objs, 'doors': door_swings(room, poly, fps), 'n_cells': {str(r): len(cells[r][0]) for r in RADII},
            'seed': 'doors' if doors else 'largest_component'}


def base(row, r=R0, R=REACH0):
    v = row['min_d'][str(r)]
    return v is not None and v <= R + 1e-9


def obj_all(row):
    return base(row) and row['vis_d'] is not None and row.get('clear') is not False


def room_level(res, cats):
    task = {c: any(obj_all(o) for o in res['objs'] if o['cat'] == c) for c in cats}
    goal = any(o.get('top_free') and base(o) for o in res['objs'])
    doors_ok = all(x['ok'] for x in res['doors'])
    return {'task': task, 'task_all': all(task.values()), 'goal_feasible': goal, 'doors_ok': doors_ok,
            'moma_all': all(task.values()) and goal and doors_ok}


def one(args):
    room, layouts = args
    cats = T.task_cats(room)
    out = {'uid': room['uid'], 'source': room['source'], 'n': len(room['objects']), 'bin': T.bin_(len(room['objects'])),
           'n_doors': len(d.door_strips(room)), 'cats': cats, 'arms': {}}
    for arm, lay in layouts.items():
        if lay is None:
            out['arms'][arm] = None
            continue
        res = evaluate(room, lay, cats)
        res['room'] = room_level(res, cats)
        res['ds'] = T.scores(room, lay, cats)
        out['arms'][arm] = res
    return out


def load(path, field):
    return {r['uid']: (r[field] if r.get('complete') else None) for r in map(json.loads, open(path))}


def main():
    s1 = [json.loads(l) for l in open(P / 's2-render/cohort-s1.jsonl')]
    s2 = load(P / 's2f/cohort/merged-s2.jsonl', 's1_layout')
    s2f = load(P / 's2f/cohort/final-s2f.jsonl', 's1_layout')
    uids = {r['uid'] for r in s1}
    rooms = {r['uid']: r for r in map(json.loads, open(DATA / 'rooms.jsonl')) if r['uid'] in uids}
    tasks = [(rooms[r['uid']], {'GT': rooms[r['uid']]['objects'],
                                'raw': r['raw_layout'] if r['complete'] else None,
                                'S1': r['s1_layout'] if r['complete'] else None,
                                'S2': s2[r['uid']], 'S2F': s2f[r['uid']]}) for r in s1]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(one, tasks, chunksize=4))
    (OUT / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in res))
    print(len(res), 'rows ->', OUT / 'rows.jsonl')


def selftest():
    assert kind('Bookcase / jewelry Armoire') == kind('tall_pantry_cabinet') == kind('bench_cabinet') == 'front'
    assert kind('Dining Chair') is None and kind('table lamp') is None and kind('King-size Bed') is None
    assert kind('Coffee Table') == kind('dining_table') == 'table' and kind('console_table') == 'front'
    assert clip((-2, 0), (2, 0), 1, 1) == (0.25, 0.75) and clip((-2, 2), (2, 2), 1, 1) is None
    # 4 x 4 room, door on the south wall, a 2.0 m wardrobe on the north wall facing -Y (local +X at yaw -90 deg)
    room = {'uid': 't', 'boundary': [[0, 0], [4, 0], [4, 4], [0, 4]], 'fixed': [
        {'id': 'door', 'category': 'door', 'size': [0.9, 0.1, 2.1], 'pos': [2.0, 0.0, 0], 'yaw': 0.0}]}
    ward = {'id': 'w', 'category': 'wardrobe', 'size': [0.6, 1.2, 2.0], 'pos': [2.0, 3.6, 0], 'yaw': -math.pi / 2,
            'parent': None}
    nst = {'id': 'n', 'category': 'nightstand', 'size': [0.4, 0.4, 0.5], 'pos': [0.4, 2.0, 0], 'yaw': 0.0, 'parent': None}
    r = evaluate(room, [ward, nst], ['wardrobe'])
    w, n = {o['id']: o for o in r['objs']}['w'], {o['id']: o for o in r['objs']}['n']
    assert base(w) and w['vis_d'] is not None and w['clear'] and w['eg_front'] and not w['recept'], w
    assert base(n) and n['clear'] and n['top_free'] and r['doors'][0]['ok'] and r['seed'] == 'doors', (n, r['doors'])
    # a 1.6 m tall screen between the door and the nightstand hides it only from cells behind the screen
    wall = {'id': 's', 'category': 'partition', 'size': [0.1, 4.0, 1.6], 'pos': [1.2, 2.0, 0], 'yaw': 0.0, 'parent': None}
    r = evaluate(room, [ward, nst, wall], ['nightstand'])
    n = {o['id']: o for o in r['objs']}['n']
    assert not base(n, R=0.8) and n['vis_d'] is None, n                     # cut off and unseen from the door side
    # a box in front of the wardrobe blocks its clearance, a box on the door swing blocks the door
    blk = {'id': 'b', 'category': 'box', 'size': [0.4, 0.4, 0.5], 'pos': [2.0, 2.9, 0], 'yaw': 0.0, 'parent': None}
    blk2 = {**blk, 'id': 'b2', 'pos': [2.0, 0.5, 0]}
    r = evaluate(room, [ward, nst, blk, blk2], ['wardrobe'])
    w = {o['id']: o for o in r['objs']}['w']
    assert not w['clear'] and w['clear_hits'] == ['b'] and not r['doors'][0]['ok'] and r['doors'][0]['hits'] == ['b2']
    # a child covering the whole nightstand top leaves no free square
    lamp = {'id': 'l', 'category': 'lamp', 'size': [0.4, 0.4, 0.3], 'pos': [0.4, 2.0, 0.5], 'yaw': 0.0, 'parent': 'n'}
    r = evaluate(room, [ward, nst, lamp], ['wardrobe'])
    assert {o['id']: o for o in r['objs']}['n']['top_free'] is False
    print('selftest ok')


ARMS = ('GT', 'raw', 'S1', 'S2', 'S2F')
BINS = ('1-10', '11-20', '21+', 'all')


def pct(xs):
    xs = list(xs)
    return f'{sum(xs) / len(xs):.3f}' if xs else '—'


def task_classes(x, cats):
    """Class-level task results: {cat: {check: bool}} (a class passes a check if one instance does)."""
    out = {}
    for c in cats:
        ins = [o for o in x['objs'] if o['cat'] == c]
        appl = [o for o in ins if 'clear' in o]
        out[c] = {'b04': any(base(o, R=0.4) for o in ins), 'b06': any(base(o) for o in ins),
                  'b08': any(base(o, R=0.8) for o in ins), 'b06_r25': any(base(o, 0.25) for o in ins),
                  'a3': any(o['a3']['0.2'] for o in ins), 'vis': any(o['vis_d'] is not None for o in ins),
                  'vis_base': any(o['vis_d'] is not None and o['vis_d'] <= REACH0 + 1e-9 for o in ins),
                  'clear': any(o['clear'] for o in appl) if appl else None, 'all': any(obj_all(o) for o in ins),
                  'nocell': all(o['min_d']['0.2'] is None for o in ins)}
    return out


def occupancy(room, lay):
    a = sum(o['size'][0] * o['size'][1] for o in lay if not o.get('parent') and o['size'][2] >= COVER_H)
    return a / Polygon(room['boundary']).area


def why_hidden(room, lay, oid):
    """Blockers of the centre ray from the nearest reached cell (r = 0.2) to oid, for the spot check."""
    poly, doors, m, by, parent = realise(room, lay)
    o = by[oid]
    floor = {i: q for i, q in by.items() if parent[i] is None and q['size'][2] >= COVER_H}
    fpoly = affine_transform(poly, m)
    seeds = [(m[0] * s['seed'][0] + m[1] * s['seed'][1] + m[4], m[2] * s['seed'][0] + m[3] * s['seed'][1] + m[5])
             for s in doors] if doors else None
    cf = (np.array(sorted(reached(grid(fpoly, [d.aabb(q, m) for q in floor.values()], R0), seeds)), dtype=float)
          + 0.5) * G
    c0, s0, x0, y0 = m[0], m[1], -m[4], -m[5]
    cr = np.column_stack([c0 * (cf[:, 0] + x0) - s0 * (cf[:, 1] + y0), s0 * (cf[:, 0] + x0) + c0 * (cf[:, 1] + y0)])
    cam = tuple(cr[np.argmin(shapely.distance(shapely.points(cr), Polygon(footprint(o))))])
    zt = min(o['pos'][2] + o['size'][2], CAM_H)
    hits = [i for i, q in by.items() if i != oid and q['pos'][2] + q['size'][2] > zt and blocked_ray(
        cam, tuple(o['pos'][:2]), zt, [(q['pos'][0], q['pos'][1], math.cos(q['yaw']), math.sin(q['yaw']),
                                        q['size'][0] / 2, q['size'][1] / 2, q['pos'][2] + q['size'][2])])]
    return [round(float(v), 2) for v in cam], hits, by


def describe(by, i):
    o = by[i]
    return (f"`{i}` {o['category']}（pos ({o['pos'][0]:.2f}, {o['pos'][1]:.2f})，yaw {math.degrees(o['yaw']) % 360:.0f}°，"
            f"size {[round(v, 2) for v in o['size']]}）")


def spot_check(rows, rooms, layouts):
    picks = json.load(open(P / 's2f/picks-A.json'))
    img = {p['uid']: f"s2f/compare-A-small/{p['k']:02d}-{p['group']}.png" for p in picks}
    by_uid = {r['uid']: r for r in rows}
    found, used = [], set()
    tests = [   # one failure of each check, five different rooms; base_pose wants a clear-cut (> 1 m) miss
        ('visibility', lambda x: [o for o in x['objs'] if o['task'] and o['vis_d'] is None and base(o, R=10)]),
        ('clearance', lambda x: [o for o in x['objs'] if o.get('clear') is False and o['clear_hits'] != ['ROOM_BOUNDARY']]),
        ('door_swing', lambda x: [dr for dr in x['doors'] if not dr['ok']]),
        ('top_free', lambda x: [o for o in x['objs'] if o.get('top_free') is False]),
        ('base_pose', lambda x: [o for o in x['objs'] if o['task'] and (o['min_d']['0.2'] or 99) > 1.0]),
    ]
    for name, f in tests:
        hit = next(((u, a, xs[0]) for u in img if u not in used for a in ('GT', 'S1', 'S2', 'S2F')
                    for xs in [f(by_uid[u]['arms'][a])] if xs), None)
        if hit:
            used.add(hit[0])
            found.append((name, *hit))
    lines = []
    for name, uid, arm, x in found:
        room, row = rooms[uid], by_uid[uid]
        res = row['arms'][arm]
        lay = layouts[uid][arm]
        _, _, _, by, parent = realise(room, lay)
        head = f"- **{name}** — `{uid}`，arm **{arm}**，图 `{img[uid]}`（列 GT / +S1 / +S2 / +S2-F），" \
               f"门 {row['n_doors']} 扇，可达区 {res['seed']}（{res['n_cells']['0.2']} 格）。"
        if name == 'base_pose':
            body = f"任务物体 {describe(by, x['id'])}：最近可达底座中心距足迹 " + (
                f"{x['min_d']['0.2']:.2f} m > {REACH0} m" if x['min_d']['0.2'] is not None else '无（可达区为空）') + '。'
        elif name == 'visibility':
            cam, hits, _ = why_hidden(room, lay, x['id'])
            body = (f"任务物体 {describe(by, x['id'])} 从任何可达格都看不见（1.2 m 相机，5 条射线）；从最近格 {cam} 到中心的射线"
                    f"被 {', '.join(describe(by, h) for h in hits) or '房间墙角'} 挡住。")
        elif name == 'clearance':
            body = (f"{describe(by, x['id'])} 前方（+X）{D_FRONT} m 矩形被 "
                    f"{', '.join(describe(by, h) if h in by else h for h in x['clear_hits'])} 占用"
                    + ('（桌子类：四边都不空）' if x['kind'] == 'table' else '') + '。')
        elif name == 'door_swing':
            body = (f"门 `{x['id']}`（宽 {x['w']} m）向内 {x['w']} × {x['w']} m 扫掠矩形被 "
                    f"{', '.join(describe(by, h) if h in by else h for h in x['hits'])} 占用。")
        else:
            body = (f"receptacle {describe(by, x['id'])}（顶高 {x['top_h']} m）顶面放不下 {TOP_SQ} × {TOP_SQ} m 方块，覆盖物："
                    f"{', '.join(describe(by, j) for j in top_covers(by[x['id']], by, parent))}；该房间 goal_feasible="
                    f"{res['room']['goal_feasible']}。")
        lines.append(head + body)
    return lines


def report():
    rows = [json.loads(l) for l in open(OUT / 'rows.jsonl')]
    uids = {r['uid'] for r in rows}
    rooms = {r['uid']: r for r in map(json.loads, open(DATA / 'rooms.jsonl')) if r['uid'] in uids}
    s1 = {r['uid']: r for r in map(json.loads, open(P / 's2-render/cohort-s1.jsonl'))}
    s2, s2f = load(P / 's2f/cohort/merged-s2.jsonl', 's1_layout'), load(P / 's2f/cohort/final-s2f.jsonl', 's1_layout')
    layouts = {u: {'GT': rooms[u]['objects'], 'raw': s1[u]['raw_layout'], 'S1': s1[u]['s1_layout'], 'S2': s2[u],
                   'S2F': s2f[u]} for u in uids}
    sel = lambda b: [r for r in rows if b == 'all' or r['bin'] == b]
    L = ['# MoMa 可执行性符号检查（H5，描述性）', '',
         f'视觉 cohort {len(rows)} 个 test 房间（IL3D_3dfront {sum(r["source"] == "IL3D_3dfront" for r in rows)}，'
         f'OptiScene_holodeck {sum(r["source"] == "OptiScene_holodeck" for r in rows)}），五个 arm：GT / raw / S1 / S2 / S2F。'
         '阈值与出处见 THRESHOLDS.md（先于所有数字固定；当时 moma/survey.md 尚不存在，用任务书默认值）；'
         '机器人侧语义见 ROBOT_CONTRACT.md。脚本：`probes-20261021/scripts/moma_checks.py`；逐房间结果：rows.jsonl。', '',
         '**读法**：EmbodiedGen 的任务判定本身不看几何（ROBOT_CONTRACT §0）；下面每项都是比机器人侧更严的几何代理，'
         '只有 `a3_touch` 与 `eg_front` 是对现有规则的镜像。无门房间 '
         f'{sum(r["n_doors"] == 0 for r in rows)}/{len(rows)}（可达区退回最大连通自由分量）。'
         '各 arm 用各自尺寸，raw/S1/S2/S2F 是预测尺寸，所以 GT 与预测 arm 之间的差异混有尺寸差（见占地率一行）。', '']
    rl = lambda a, k, rs=rows: sum(r['arms'][a]['room'][k] for r in rs) / len(rs)
    L += ['## 0. 要点', '',
          '- 房间级 moma_all（全部任务物体可达+可见+有操作空间，且有可放置顶面，且门扇扫掠空闲）：' +
          '，'.join(f"{a} {rl(a, 'moma_all'):.3f}" for a in ARMS) + '；task_all：' +
          '，'.join(f"{a} {rl(a, 'task_all'):.3f}" for a in ARMS) + '。',
          '- 任务类别失败的首因主要是 clearance（前方 0.6 m 被占），其次是底座够不到；可见性几乎从不单独致败（§6）。',
          '- EG 镜像 a3_touch（0.45 m 切比雪夫）与 base r0.2/R0.6 几乎一致；eg_front 与 0.6 m clearance 同量级（§1、§2）。',
          '- 汇总上 S2/S2F 相对 S1 每项都更好（门扇扫掠、clearance、可达），但 GT 在 MoMa 各项上反而最差：'
          'GT 占地率更高（预测尺寸偏小，§4 占地率 0.387 vs 0.350），且部分 GT 自身越界（§2：GT front 类前方矩形出边界 108 个）。'
          '因此这里的 arm 间比较是“各自尺寸下”的描述，不是对 GT 的保真度。',
          f"- 有门房间显著更差（入口种子 + 门扇扫掠更严）：GT 有门 moma_all "
          f"{rl('GT', 'moma_all', [r for r in rows if r['n_doors']]):.3f} vs 无门 "
          f"{rl('GT', 'moma_all', [r for r in rows if not r['n_doors']]):.3f}（§5）。", '']
    # Table 1: task objects, class level
    L += ['## 1. 任务物体（类别级，每房间 2 个哈希选类别；同类任一实例通过即通过）', '',
          '| arm | 类别数 | base r0.2 R0.4 | R0.6 | R0.8 | base r0.25 R0.6 | a3_touch (EG 0.45 m) | vis_any | vis_base (≤0.6 m) | clearance（适用数） | 全部通过 | 无可达格 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        tc = [v for r in rows for v in task_classes(r['arms'][a], r['cats']).values()]
        cl = [v['clear'] for v in tc if v['clear'] is not None]
        L.append(f"| {a} | {len(tc)} | {pct(v['b04'] for v in tc)} | {pct(v['b06'] for v in tc)} | {pct(v['b08'] for v in tc)} | "
                 f"{pct(v['b06_r25'] for v in tc)} | {pct(v['a3'] for v in tc)} | {pct(v['vis'] for v in tc)} | "
                 f"{pct(v['vis_base'] for v in tc)} | {pct(cl)} ({len(cl)}) | {pct(v['all'] for v in tc)} | {pct(v['nocell'] for v in tc)} |")
    L += ['', '「全部通过」= 同一实例 base(r0.2, R0.6) ∧ vis_any ∧ clearance（若适用）。「无可达格」= 可达区为空（入口被堵或房间太窄），所有检查必败。', '']
    # Table 2: functional instances
    L += ['## 2. 功能物体（所有 front/table 类地面实例，逐实例）', '',
          '| arm | 实例 | base r0.2 R0.6 | vis_any | clearance front (n) | clearance table 任一边 (n) | eg_front 镜像 (n) | receptacle ≤1.22 m (n) | top_free |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        fo = [o for r in rows for o in r['arms'][a]['objs'] if o['kind'] and o['floor']]
        fr, tb, eg = [o for o in fo if o['kind'] == 'front'], [o for o in fo if o['kind'] == 'table'], [o for o in fo if 'eg_front' in o]
        rc = [o for o in fo if o.get('recept')]
        L.append(f"| {a} | {len(fo)} | {pct(base(o) for o in fo)} | {pct(o['vis_d'] is not None for o in fo)} | "
                 f"{pct(o['clear'] for o in fr)} ({len(fr)}) | {pct(o['clear'] for o in tb)} ({len(tb)}) | "
                 f"{pct(o['eg_front'] for o in eg)} ({len(eg)}) | {len(rc)} | {pct(o['top_free'] for o in rc)} |")
    fr_gt = [o for r in rows for o in r['arms']['GT']['objs'] if o.get('kind') == 'front' and o['floor'] and o['clear'] is False]
    L += ['', f"GT 中 front 类 clearance 失败 {len(fr_gt)} 个，其中前方矩形出房间边界 "
          f"{sum(o['clear_hits'] == ['ROOM_BOUNDARY'] for o in fr_gt)} 个，其余被地面物体占用。", '']
    # Table 3: doors
    L += ['## 3. 门扇扫掠（门宽 × 门宽，向内）', '', '| arm | 门 | 扫掠空闲 | 有门房间 | 房间内全部门空闲 |', '|---|---:|---:|---:|---:|']
    for a in ARMS:
        ds = [dr for r in rows for dr in r['arms'][a]['doors']]
        dr_rooms = [r for r in rows if r['arms'][a]['doors']]
        L.append(f"| {a} | {len(ds)} | {pct(dr['ok'] for dr in ds)} | {len(dr_rooms)} | {pct(r['arms'][a]['room']['doors_ok'] for r in dr_rooms)} |")
    # Table 4: room level by bin
    L += ['', '## 4. 房间级（按物体数分档）', '',
          '`task_all` = 全部任务类别通过；`goal` = 存在 top_free 且底座可达的 receptacle；`moma_all` = task_all ∧ goal ∧ 全部门扫掠空闲；'
          '`ds strict / ds task` = 下游准入（全类别必需 / 只要求任务物体，`task_reach_headroom.scores`），作背景对照。', '',
          '| 物体数 | 房间 | arm | 占地率 | task_all | goal | moma_all | ds strict | ds task | ds task ∧ moma_all |',
          '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in BINS:
        rs = sel(b)
        for a in ARMS:
            X = [(r, r['arms'][a]) for r in rs]
            occ = sum(occupancy(rooms[r['uid']], layouts[r['uid']][a]) for r in rs) / len(rs)
            L.append(f"| {b} | {len(rs)} | {a} | {occ:.3f} | {pct(x['room']['task_all'] for _, x in X)} | "
                     f"{pct(x['room']['goal_feasible'] for _, x in X)} | {pct(x['room']['moma_all'] for _, x in X)} | "
                     f"{pct(x['ds']['all'] for _, x in X)} | {pct(x['ds']['task'] for _, x in X)} | "
                     f"{pct(x['ds']['task'] and x['room']['moma_all'] for _, x in X)} |")
    # Table 5: door vs doorless
    L += ['', '## 5. 有门 vs 无门房间（task_all / moma_all）', '', '| arm | 有门 task_all | 有门 moma_all | 无门 task_all | 无门 moma_all |', '|---|---:|---:|---:|---:|']
    for a in ARMS:
        dr, nd = [r for r in rows if r['n_doors']], [r for r in rows if not r['n_doors']]
        L.append(f"| {a} | {pct(r['arms'][a]['room']['task_all'] for r in dr)} | {pct(r['arms'][a]['room']['moma_all'] for r in dr)} | "
                 f"{pct(r['arms'][a]['room']['task_all'] for r in nd)} | {pct(r['arms'][a]['room']['moma_all'] for r in nd)} |")
    L += ['', f'有门 {sum(r["n_doors"] > 0 for r in rows)} 间，无门 {sum(r["n_doors"] == 0 for r in rows)} 间。', '']
    # Table 6: first failing check for task classes
    L += ['## 6. 任务类别失败的首个原因（按 base → vis → clearance 顺序，取该类最好的实例）', '',
          '| arm | 失败类别 | 无可达格 | 超出 R0.6 | 看不见 | clearance |', '|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        cnt = {'nocell': 0, 'far': 0, 'vis': 0, 'clear': 0}
        nf = 0
        for r in rows:
            for c, v in task_classes(r['arms'][a], r['cats']).items():
                if v['all']:
                    continue
                nf += 1
                cnt['nocell' if v['nocell'] else 'far' if not v['b06'] else 'vis' if not v['vis'] else 'clear'] += 1
        L.append(f"| {a} | {nf} | {cnt['nocell']} | {cnt['far']} | {cnt['vis']} | {cnt['clear']} |")
    L += ['', '## 7. 5 房间人工核对（不渲染；对照已有图 `probes-20261021/s2f/compare-A-small/`，同一批房间也在 `s2-render/compare-small/`）', '',
          '坐标为房间帧（米，boundary 最小角为原点附近），yaw 逆时针自房间 +X 量到物体前方（IR local +X）。图的俯视方向未标定，请按类别 + 相对位置辨认。', '']
    L += spot_check(rows, rooms, layouts)
    L += ['', '## 8. 简化与局限', '',
          '- 可见性是 2.5D：遮挡物从地面实心拉伸，无视场角/俯仰；只 5 个目标点。门、窗不遮挡。',
          '- clearance 只评估地面物体；有 parent 的功能物体计 n/a。桌子类取四边任一边（桌子无语义前方），偏离任务书的 +X。',
          '- 门扇扫掠用矩形覆盖两种铰链的四分之一圆，偏保守；门宽 ≤ 1 m 时被 A2 保留带包含。',
          '- 无门房间（68%）的可达区是最大连通分量，不是真实入口；有门/无门已分开报告。',
          '- task 物体用 `task_cats`（哈希选 2 类，GT 高度 ≥ 0.3 m），模拟上游 “verb obj1, obj2”；不是上游真实契约物体。',
          '- 预测 arm 用预测尺寸；占地率差异会直接改变 clearance / 可达，GT 不是 “上界”。', '']
    (OUT / 'report.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else report() if '--report' in sys.argv else main()
