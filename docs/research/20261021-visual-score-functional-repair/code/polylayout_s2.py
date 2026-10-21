"""Probe polylayout-s2: PolyLayout (arXiv 2608.10838) Stage-3 solver transferred as S2, a constructive re-placement
candidate run after S1 on the rooms S1 leaves failing. Free, CPU, shapely only; never modifies the repo.

python3 polylayout_s2.py --repo REPO --s1 S1.jsonl --data EVAL_DATA --out OUT [--limit N] [--workers W] [--no-polish]
    [--phi checker|strict]   strict adds downstream A4 (same parent, AABB overlap > 1 cm, height ignored) to Phi and to the bad set
python3 polylayout_s2.py --selfcheck

Paper rule (eq. 1-4) with our invariants (ids/order/category/size/yaw/parent/z frozen, every object placed exactly once):
  good units (no checker / strict-R2-A2 violation) are hard obstacles; bad units are re-placed one at a time, largest
  root footprint first, from their current pose: offsets k*0.1 m along the nearest wall's +-direction and its inward
  normal, k increasing, K = floor(wall length / 0.1) (30 for the normal); then re-anchored flush on the other walls
  (midpoint, +-along). Phi = every member footprint inside floor-minus-door-keepouts (+1 mm), clear of every
  non-window fixed box, zero checker-counted overlap with placed objects. First feasible pose wins, else the minimum
  (outside + overlap) area pose: NEVER unplaced (paper allows unplaced; our contract does not).
  Bad children (collision / off-seat) are re-placed on their seat by a 5 cm grid inside the clamp region, nearest
  first (paper's surface placement, grid instead of largest free rectangle).
Candidate ranking is the pipeline's: S2 is a candidate beside S1; this probe reports S1 -> S2 (-> S1 polish).
"""
import argparse
import collections
import hashlib
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from shapely.affinity import affine_transform, translate
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.prepared import prep

STEP, KCAP, GRID, TOL, MIN_AREA, CLAMP, CLEAR, TIME_LIMIT, CHILD_CAP = 0.10, 30, 0.05, 0.05, 0.01, 1e-3, 0.01, 60.0, 400


def _imports(repo):
    sys.path.insert(0, repo)
    global FLAT, d, g, orp, footprint, head, check, collision_kind, support_contains
    from fastfill.anchors import FLAT
    from fastfill.comparison import downstream as d, geometry as g, output_repair as orp
    from fastfill.scene import footprint, head
    from fastfill.validate import check, collision_kind, support_contains


# ------------------------------------------------------------------ geometry records

def _rec(o, m=None, fixed=False, covering=False):
    p = Polygon(footprint(o))
    return {'id': o['id'], 'obj': o, 'poly': p, 'b': p.bounds, 'z0': o['pos'][2], 'z1': o['pos'][2] + o['size'][2],
            'skip': head(o) in FLAT or o['size'][2] < 0.06, 'fixed': fixed, 'parent': o.get('parent'),
            'aabb': affine_transform(p, m).bounds if m is not None else None, 'covering': covering}


def _a4(m, aabb, o):
    """strict mode: downstream A4_COLLISION (same parent, neither a covering, AABB overlap > TOL_M, height ignored)."""
    if aabb is None or o['fixed'] or m['parent'] != o['parent'] or m['covering'] or o['covering'] \
            or not d._overlaps(aabb, o['aabb']):
        return 0.0
    return (min(aabb[2], o['aabb'][2]) - max(aabb[0], o['aabb'][0])) * (min(aabb[3], o['aabb'][3]) - max(aabb[1], o['aabb'][1]))


def _overlap(m, poly, b, o, aabb=None):
    """Checker-counted overlap area of moving member m (at polygon poly) with obstacle o, 0 when the checker ignores it;
    strict mode adds the A4 AABB overlap."""
    if m['id'] == o['id'] or m['parent'] == o['id'] or o['parent'] == m['id']:
        return 0.0
    a4 = _a4(m, aabb, o)
    if m['skip'] or o['skip'] or min(m['z1'], o['z1']) - max(m['z0'], o['z0']) <= TOL:
        return a4
    ob = o['b']
    if not (b[0] < ob[2] and ob[0] < b[2] and b[1] < ob[3] and ob[1] < b[3]):
        return a4
    area = poly.intersection(o['poly']).area
    if area <= MIN_AREA or (not o['fixed'] and collision_kind(m['obj'], o['obj'], poly, o['poly'], TOL) != 'main'):
        return a4
    return max(area, a4)


def _cost(members, dx, dy, obstacles, env, env_p, m=None):
    out = ovl = 0.0
    for mem in members:
        p = translate(mem['poly'], dx, dy) if (dx or dy) else mem['poly']
        if not env_p.covers(p):
            out += p.difference(env).area
        b, aabb = p.bounds, (affine_transform(p, m).bounds if m is not None else None)
        for o in obstacles:
            ovl += _overlap(mem, p, b, o, aabb)
    return out, ovl


def _edges(boundary):
    pts = list(orient(Polygon(boundary)).exterior.coords)
    out = []
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        if L > 1e-9:
            u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
            out.append({'a': a, 'b': b, 'L': L, 'u': u, 'n': (-u[1], u[0]), 'line': LineString([a, b])})
    return out


def _root_candidates(root, edges):
    """(dx, dy) offsets from the current pose: paper eq. 3 on the nearest wall, then re-anchoring on the other walls."""
    x, y = root['obj']['pos'][:2]
    order = sorted(edges, key=lambda e: e['line'].distance(Point(x, y)))
    ref = order[0]
    yield 0.0, 0.0
    dirs = [(ref['u'], int(ref['L'] / STEP)), ((-ref['u'][0], -ref['u'][1]), int(ref['L'] / STEP)), (ref['n'], KCAP)]
    for k in range(1, max(k for _, k in dirs) + 1):
        for (ux, uy), K in dirs:
            if k <= K:
                yield k * STEP * ux, k * STEP * uy
    corners = root['poly'].exterior.coords[:-1]
    for e in order[1:]:
        proj = [c[0] * e['n'][0] + c[1] * e['n'][1] for c in corners]
        along = [c[0] * e['u'][0] + c[1] * e['u'][1] for c in corners]
        half_n, half_u = (max(proj) - min(proj)) / 2, (max(along) - min(along)) / 2
        mid = ((e['a'][0] + e['b'][0]) / 2, (e['a'][1] + e['b'][1]) / 2)
        tx, ty = mid[0] + e['n'][0] * (half_n + CLEAR), mid[1] + e['n'][1] * (half_n + CLEAR)
        K = int(max(e['L'] / 2 - half_u, 0) / STEP)
        yield tx - x, ty - y
        for k in range(1, K + 1):
            for s in (1, -1):
                yield tx + s * k * STEP * e['u'][0] - x, ty + s * k * STEP * e['u'][1] - y


def _child_candidates(child, seat):
    """(dx, dy) to grid points of the seat's clamp region, nearest first (support_contains holds by construction)."""
    c, s = math.cos(seat['yaw']), math.sin(seat['yaw'])
    hx, hy = max(seat['size'][0] / 2 - CLAMP, 0.0), max(seat['size'][1] / 2 - CLAMP, 0.0)
    x, y = child['obj']['pos'][:2]
    pts = []
    for i in range(-int(hx / GRID), int(hx / GRID) + 1):
        for j in range(-int(hy / GRID), int(hy / GRID) + 1):
            lx, ly = i * GRID, j * GRID
            px, py = seat['pos'][0] + c * lx - s * ly, seat['pos'][1] + s * lx + c * ly
            pts.append(((px - x) ** 2 + (py - y) ** 2, px - x, py - y))
    if not pts:
        pts = [(0.0, seat['pos'][0] - x, seat['pos'][1] - y)]
    for _, dx, dy in sorted(pts)[:CHILD_CAP]:
        yield dx, dy


def _search(members, candidates, obstacles, env, env_p, deadline, m=None):
    best, timed_out = None, False
    for dx, dy in candidates:
        if time.monotonic() > deadline:
            timed_out = True
            break
        out, ovl = _cost(members, dx, dy, obstacles, env, env_p, m)
        if out == 0.0 and ovl == 0.0:
            return (dx, dy), False, timed_out
        key = (out + ovl, abs(dx) + abs(dy))
        if best is None or key < best[0]:
            best = (key, (dx, dy))
    return (best[1] if best else (0.0, 0.0)), True, timed_out


# ------------------------------------------------------------------ bad sets

def _bad(room, keepouts, phi):
    """ids the checker / strict R2+A2 (+A4 in strict mode) blame; of a colliding pair the exact duplicate (later) or
    the smaller one (output_repair.audit is the authoritative audit)."""
    by = {o['id']: o for o in room['objects']}
    order = {i: k for k, i in enumerate(by)}
    L = orp.audit(room, keepouts, [])['lists']
    bad = set(L['oob']) | set(L['support_fail']) | {o for o, _ in L['fixed_blocking']} | set(L['outside_1mm']) \
        | {i for i, _ in L['keepout_1cm']}
    key = lambda o: (tuple(round(v * 1000) for v in o['pos'][:2]), round(o['yaw'], 6))
    area = lambda o: o['size'][0] * o['size'][1]
    for a, b in list(L['main']) + (list(L['a4_aabb']) if phi == 'strict' else []):
        a, b = sorted((a, b), key=order.get)
        oa, ob = by[a], by[b]
        bad.add(b if key(oa) == key(ob) or area(ob) <= area(oa) else a)
    return bad, L


def s2(room, keepouts, deadline, phi='checker'):
    objs = room['objects']
    by = {o['id']: o for o in objs}
    fixed = {f['id']: f for f in room.get('fixed') or []}
    windows = {f['id'] for f in fixed.values() if {'window', 'windows'} & set(d.norm_cat(f.get('category')).split())}
    kids = collections.defaultdict(list)
    for o in objs:
        if o.get('parent') in by:
            kids[o['parent']].append(o['id'])
    roots = [o['id'] for o in objs if o.get('parent') not in by]

    def subtree(i):
        out, todo = [], [i]
        while todo:
            j = todo.pop()
            if j not in out:
                out.append(j)
                todo += kids[j]
        return out

    poly = Polygon(room['boundary'])
    s = orp._strict(room, keepouts)
    env = (s['floor'] if s and not s['floor'].is_empty else poly).buffer(orp.ENVELOPE_TOL)
    env_p = prep(env)
    edges = _edges(room['boundary'])
    m = d.frame(room['boundary'])[1] if phi == 'strict' else None
    cur = {o['id']: dict(o) for o in objs}
    shift = {o['id']: (0.0, 0.0) for o in objs}
    fixed_recs = [_rec({**f}, fixed=True) for f in fixed.values() if f['id'] not in windows]
    stats = {'bad_roots': 0, 'bad_children': 0, 'min_overlap': 0, 'timed_out': False}

    def recs(ids):
        return [_rec(cur[i], m, covering=cur[i].get('parent') not in by and cur[i]['size'][2] < d.FLOOR_COVERING_H_M)
                for i in ids]

    def move(ids, dx, dy):
        for i in ids:
            o = cur[i]
            cur[i] = {**o, 'pos': [round(o['pos'][0] + dx, 6), round(o['pos'][1] + dy, 6), o['pos'][2]]}
            shift[i] = (shift[i][0] + dx, shift[i][1] + dy)

    # pass 1: floor-root units, largest first; good units are hard obstacles
    bad, _ = _bad({**room, 'objects': [cur[o['id']] for o in objs]}, keepouts, phi)
    bad_roots = sorted([r for r in roots if r in bad], key=lambda r: -cur[r]['size'][0] * cur[r]['size'][1])
    stats['bad_roots'] = len(bad_roots)
    placed = [i for r in roots if r not in bad for i in subtree(r)]
    for r in bad_roots:
        if time.monotonic() > deadline:
            stats['timed_out'] = True
            break
        members = recs(subtree(r))
        obstacles = recs(placed) + fixed_recs
        (dx, dy), fallback, to = _search(members, _root_candidates(members[0], edges), obstacles, env, env_p, deadline, m)
        stats['min_overlap'] += fallback
        stats['timed_out'] |= to
        move(subtree(r), dx, dy)
        placed += subtree(r)
    # pass 2: children on their seat, lowest first
    bad, _ = _bad({**room, 'objects': [cur[o['id']] for o in objs]}, keepouts, phi)
    bad_children = sorted([i for i in bad if i not in roots and (cur[i]['parent'] in by or cur[i]['parent'] in fixed)],
                          key=lambda i: (cur[i]['pos'][2], -cur[i]['size'][0] * cur[i]['size'][1]))
    stats['bad_children'] = len(bad_children)
    for i in bad_children:
        if time.monotonic() > deadline:
            stats['timed_out'] = True
            break
        seat = cur.get(cur[i]['parent']) or fixed[cur[i]['parent']]
        tree = subtree(i)
        members = recs(tree)
        obstacles = recs([j for j in cur if j not in tree]) + fixed_recs
        (dx, dy), fallback, to = _search(members, _child_candidates(members[0], seat), obstacles, env, env_p, deadline, m)
        stats['min_overlap'] += fallback
        stats['timed_out'] |= to
        move(tree, dx, dy)
    return [cur[o['id']] for o in objs], shift, stats


# ------------------------------------------------------------------ rows

def _invariant(before, after):
    keys = ('id', 'category', 'size', 'yaw', 'parent')
    return len(before) == len(after) and all(all(a.get(k) == b.get(k) for k in keys) and a['pos'][2] == b['pos'][2]
                                             for a, b in zip(before, after))


def _eval(room, objs):
    r = {**room, 'objects': objs}
    geo, acc = g.geometry(r), d.accept(r, objs)
    return {'checker': geo['geometry_success'], 'strict': acc['accepted'],
            'both': geo['geometry_success'] and acc['accepted'], 'main': geo['main_collisions'],
            'oob': geo['oob_objects'], 'fixed_blocking': geo['fixed_blocking'], 'support_fail': geo['support_fail'],
            'strict_reasons': acc['reasons']}


def probe_row(task):
    row, room_gt, request, polish, phi = task
    started = time.monotonic()
    room = orp.predicted_room(room_gt, request['messages'], row['s1_layout'])
    keepouts = orp.door_keepouts(room)
    before = _eval(room, room['objects'])
    objs, shift, stats = s2(room, keepouts, started + TIME_LIMIT, phi)
    ok = _invariant(room['objects'], objs)
    after = _eval(room, objs)
    out = {'uid': row['uid'], 'source': row['source'], 'objects': row['objects'], 'stratum': row['s1']['stratum'],
           'doors': row['s1']['doors'], 'rectangle': row['s1']['rectangle'], 'before': before, 'after': after,
           'invariant_ok': ok, **stats,
           'moved': sum(1 for v in shift.values() if v != (0.0, 0.0)),
           'displacement_max': max(math.hypot(*v) for v in shift.values()),
           'displacement_moved_median': statistics.median([math.hypot(*v) for v in shift.values() if v != (0.0, 0.0)] or [0.0])}
    if polish:
        res = orp.repair({**room, 'objects': objs})
        out['polish'] = {**_eval(room, res['room']['objects']), 'status': res['status']}
        out['s2_layout'] = res['room']['objects'] if res['status'] != 'unchanged' else objs
    else:
        out['s2_layout'] = objs
    out['runtime_s'] = time.monotonic() - started
    return out


def _rate(rows, stage, key):
    return sum(bool(r[stage][key]) for r in rows) / len(rows) if rows else None


def summarize(rows, polish):
    stages = ['before', 'after'] + (['polish'] if polish else [])
    def agg(rs):
        return {'rooms': len(rs), **{f'{st}_{k}': _rate(rs, st, k) for st in stages for k in ('checker', 'strict', 'both')},
                'invariant_violations': sum(not r['invariant_ok'] for r in rs), 'timed_out': sum(r['timed_out'] for r in rs),
                'min_overlap_rooms': sum(r['min_overlap'] > 0 for r in rs),
                'runtime_mean_s': statistics.mean(r['runtime_s'] for r in rs) if rs else None,
                'runtime_max_s': max((r['runtime_s'] for r in rs), default=None),
                'displacement_moved_median_m': statistics.median([r['displacement_moved_median'] for r in rs if r['moved']] or [0.0])}
    return {'all': agg(rows),
            'by_stratum': {k: agg([r for r in rows if r['stratum'] == k]) for k in ('1-10', '11-30', '31-50', '51+')},
            'by_doors': {k: agg([r for r in rows if (r['doors'] > 0) == (k == 'doors')]) for k in ('no_doors', 'doors')},
            'by_shape': {k: agg([r for r in rows if r['rectangle'] == (k == 'rectangle')]) for k in ('rectangle', 'non_rectangle')},
            'by_source': {k: agg([r for r in rows if r['source'] == k]) for k in sorted({r['source'] for r in rows})}}


def _report(summary, polish):
    f = lambda v: '—' if v is None else f'{v:.3f}'
    cols = ['checker', 'strict', 'both']
    lines = ['# PolyLayout S2 probe (S1-failing rooms of the 27B test)', '',
             '| scope | rooms | ' + ' | '.join(f'{c}: S1 → +S2' + (' → +S1 polish' if polish else '') for c in cols)
             + ' | min-overlap rooms | timed out | runtime mean/max s |', '|---|---:|' + '---:|' * (len(cols) + 3)]
    scopes = [('all', summary['all'])] + [(f'{grp} {k}', v) for grp in ('by_stratum', 'by_doors', 'by_shape', 'by_source')
                                          for k, v in summary[grp].items()]
    for name, a in scopes:
        if not a['rooms']:
            continue
        cells = [f"{f(a['before_' + c])} → {f(a['after_' + c])}" + (f" → {f(a['polish_' + c])}" if polish else '') for c in cols]
        lines.append(f"| {name} | {a['rooms']} | " + ' | '.join(cells) +
                     f" | {a['min_overlap_rooms']} | {a['timed_out']} | {f(a['runtime_mean_s'])} / {f(a['runtime_max_s'])} |")
    lines += ['', f"invariant violations (must be 0): {summary['all']['invariant_violations']}"]
    return '\n'.join(lines) + '\n'


def load(s1, data, limit):
    rooms = {json.loads(l)['uid']: json.loads(l) for l in open(Path(data) / 'rooms.jsonl')}
    requests = {json.loads(l)['uid']: json.loads(l) for l in open(Path(data) / 'placement' / 'requests.jsonl')}
    rows = [json.loads(l) for l in open(s1)]
    rows = [r for r in rows if r['complete'] and not r['s1']['geometry_success_s1']]
    rows.sort(key=lambda r: hashlib.sha256(r['uid'].encode()).hexdigest())
    if limit:
        rows = rows[:limit]
    return [(r, rooms[r['uid']], requests[r['uid']]) for r in rows]


def selfcheck():
    boundary = [[0, 0], [4, 0], [4, 3], [0, 3]]
    w = {'category': 'wardrobe', 'size': [0.6, 1.2, 2.0], 'pos': [0.3, 1.5, 0], 'yaw': 0.0, 'parent': None}
    lamp = {'id': 'l', 'category': 'lamp', 'size': [0.2, 0.2, 0.4], 'pos': [0.3, 1.5, 2.0], 'yaw': 0.0, 'parent': 'w1'}
    lamp2 = {**lamp, 'id': 'l2'}
    room = {'boundary': boundary, 'boundary_type': 'polygon', 'height': 2.8, 'constraints': [],
            'fixed': [{'id': 'd', 'category': 'door', 'size': [0.9, 0.1, 2.1], 'pos': [2.0, 0.05, 0], 'yaw': 0.0}],
            'objects': [{'id': 'w1', **w}, {'id': 'w2', **w}, {'id': 'w3', **w}, lamp, lamp2]}
    keep = orp.door_keepouts(room)
    assert check(room)['collisions']['main'], 'self-check needs a colliding start'
    for phi in ('checker', 'strict'):
        objs, shift, stats = s2(room, keep, time.monotonic() + 30, phi)
        after = {**room, 'objects': objs}
        assert _invariant(room['objects'], objs)
        c = check(after)
        assert not c['collisions']['main'] and not c['oob'] and not c['fixed_blocking'] and not c['support_fail'], c
        assert stats['bad_roots'] == 2 and stats['bad_children'] == 1 and stats['min_overlap'] == 0, stats
        assert phi == 'checker' or d.accept(after, objs)['accepted'], d.accept(after, objs)['reasons']
        print('selfcheck ok', phi, {i: [round(v, 2) for v in s] for i, s in shift.items()}, stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default='/Users/slian/Desktop/3D/Worldedge/OptiScene')
    ap.add_argument('--s1'); ap.add_argument('--data'); ap.add_argument('--out')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--no-polish', action='store_true'); ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--phi', choices=('checker', 'strict'), default='checker')
    a = ap.parse_args()
    _imports(a.repo)
    if a.selfcheck:
        return selfcheck()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    tasks = [(*t, not a.no_polish, a.phi) for t in load(a.s1, a.data, a.limit)]
    if a.workers == 1:
        rows = [probe_row(t) for t in tasks]
    else:
        with ProcessPoolExecutor(a.workers, initializer=_imports, initargs=(a.repo,)) as pool:
            rows = list(pool.map(probe_row, tasks, chunksize=2))
    summary = summarize(rows, not a.no_polish)
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'summary.json').write_text(json.dumps({'params': {'step': STEP, 'kcap': KCAP, 'grid': GRID, 'time_limit': TIME_LIMIT,
                                                             'limit': a.limit, 'polish': not a.no_polish, 'phi': a.phi, 's1': a.s1},
                                                  'summary': summary}, indent=1))
    (out / 'report.md').write_text(_report(summary, not a.no_polish))
    print(_report(summary, not a.no_polish))


if __name__ == '__main__':
    main()
