"""S1: deterministic minimum-displacement repair of predicted layouts; raw is always kept and reported beside it.

python -m fastfill.comparison.output_repair --data EVAL --geometry RUN/joint-geometry.jsonl \
    --requests-arm joint|placement --out NEW_DIR [--workers N] [--max-trials 2000] [--time-limit 60] [--uids FILE]

Only XY translations of whole support subtrees. ids, order, categories, size, yaw, parent, z and the object count
never change; nothing is dropped, merged or resized. Unmoved objects keep their raw pos bit for bit; a moved object
is raw + an integer number of millimetres (serialised at 1e-6), and every candidate is audited exactly as it would be
returned. pose_targets.repair is NOT used (offline teacher: it rounds every object and tries cardinal yaws); only the
frozen primitives are (validate.check / collision_kind / support_ok / holds, geometry.geometry, interface.snap_inside).

Audit vector (COMPONENTS): validate.check counts, support by geometry()'s rule (a fixed parent supports), the strict
downstream stage (downstream.py, every object: R2 footprint inside downstream.keepout_floor(boundary,
keep-outs).buffer(ENVELOPE_TOL), floor coverings and stacked objects included; A2 keep-out AABB rule with
KEEPOUT_SLACK, floor coverings exempt; A4 a4_aabb = pairs of non-covering objects with one parent whose AABBs
overlap by more than d.TOL_M; covering and parent after downstream R4, P2 frame) and held relations (the caller's
hard constraints true on raw, plus against_wall of every object true on raw; sliding along a wall is allowed).
phi = main overlap + fixed-blocking overlap + area outside the 10 cm boundary + area outside the 1 mm floor envelope
+ keep-out intrusion. A candidate is accepted iff no component rises and one falls or phi falls by EPS_PHI (Pareto,
never lexicographic). A3 reachability is not a component: the final guard below covers it.
Stage a (checker): passes over severe, main, fixed_blocking, oob, support_xy problems (most severe first, then ids);
the first accepted candidate of a problem is applied and the pass moves on; a pass without progress ends the stage.
SAT pushes (4 axes, away from the other centre, + CLEAR, + 1 and 2 extents of the other), smaller unit first (sum of
member footprints, ties: larger sha256(id) moves), then the larger, then both by half; co-located pairs (centres
< COLOCATED apart) push along the mover's local +-x/+-y, most free room first. A unit is an object with its support
subtree. Each object is tried alone first (a child moves only on its seat), then the branch it heads below the
common ancestor (different floor roots: its whole tree); a child unit's centre is clamped into its parent footprint
(shrunk 1 mm), as pose_targets._normalize.
Stage b (strict): interface.snap_inside into boundary minus keep-outs as one atomic candidate, then for every strict
violator (lowest z first) the floor push, its axis parts, keep-out AABB pushes + PUSH_MARGIN (every keep-out it
intrudes, or for a footprint outside the floor envelope every keep-out it enters) and SNAP_GRID; here a candidate must
also clear that violator (a partial phi gain cannot make a room acceptable downstream).
support_z, ceiling and parent_missing cannot be fixed by XY moves: flagged, never targeted. Windows are never a target
or a gate (EmbodiedGen has no window keep-out); the window component must only not rise.
The final layout is re-audited with check(), geometry() and strict_audit(); any component above raw, a broken
invariant, or any downstream.accept flag (DOWNSTREAM_FLAGS) true on raw and false after rolls the room back to raw
(status unchanged, diagnostics rolled_back). status ok = every FAILS component clean (accept() itself is reported
beside it). Stages a and b have max_trials trials each (deterministic limit); time_limit (monotonic) is a safety net
and marks the room timed_out (not reproducible). Deployed, S1 must run inside serve.py before its validation.
Keep-outs: the request's, else downstream.door_strips (keepout_source). Rows: <geometry stem>-s1.jsonl = the original
row + s1_layout + s1 (raw and S1 metrics side by side, downstream.accept before and after); repaired_layout is left as
is. Rates use every frozen request as denominator (the geometry file must cover every request); with --uids FILE (a
JSON list of uids, or a cohort JSON with 'uids', e.g. RUN/llm-cohort-v1.json) exactly those uids instead (the geometry
file must cover exactly them). An incomplete row passes through and counts as a failure. Summary strata: object count,
rectangle / not, doorless / with doors, source.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

from shapely.affinity import affine_transform
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points
from shapely.prepared import prep

from fastfill.anchors import FLAT
from fastfill.comparison import downstream as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.interface import SNAP_GRID, snap_inside
from fastfill.scene import footprint, head, norm_cat
from fastfill.validate import SUPPORT_EPS, check, collision_kind, holds, support_contains, support_ok

COMPONENTS = ('oob', 'support_fail', 'ceiling', 'fixed_blocking', 'window', 'main', 'severe',
              'tuck', 'contained', 'outside_1mm', 'keepout_1cm', 'held_broken', 'a4_aabb')
STRATA = ((1, 10), (11, 30), (31, 50), (51, None))
CLEAR, ENVELOPE_TOL, KEEPOUT_SLACK = 0.01, d.ENTRY_TOL_M, d.TOL_M
DOOR_DEPTH, DOOR_MARGIN = d.KEEPOUT_DEPTH_M, d.KEEPOUT_SIDE_MARGIN_M
TOL, OOB_TOL, MIN_AREA, SEVERE = 0.05, 0.10, 0.01, 0.3          # validate.check defaults
EPS_PHI, COLOCATED, PUSH_MARGIN, CLAMP = 1e-4, 1e-3, 0.002, 1e-3
SEVERITY = {'severe': 0, 'main': 1, 'fixed': 2, 'oob': 3, 'support': 4}
WINDOW_WORDS = {'window', 'windows'}
FAILS = ('oob', 'support_fail', 'ceiling', 'fixed_blocking', 'main', 'severe', 'outside_1mm', 'keepout_1cm',
         'held_broken', 'a4_aabb')
DOWNSTREAM_FLAGS = ('endpoint_ok', 'attach_ok', 'admit_ok', 'admit_lenient_ok', 'admit_no_required_ok')


def predicted_room(room, request_messages, raw_layout):
    """geometry.score's pred: request-frame boundary, height and fixed boxes (yaw deg -> rad), constraints []."""
    request = h.strict_json(request_messages[1]['content'])
    fixed = [{**f, 'yaw': math.radians(f['yaw'])} for f in request.get('fixed', [])]
    return {**copy.deepcopy(room), 'objects': copy.deepcopy(raw_layout), 'fixed': fixed,
            'boundary': request['boundary'], 'height': request.get('height'), 'constraints': []}


def door_keepouts(room):
    """downstream.door_strips as keep-out rings in the room frame (a strip clipped into pieces: one ring each)."""
    return [[list(p) for p in part.exterior.coords[:-1]] for s in d.door_strips(room)
            for part in getattr(s['strip'], 'geoms', [s['strip']]) if part.geom_type == 'Polygon' and part.area > 0]


def _strict(room, keepouts):
    """Strict-stage geometry, None unless the boundary is a polygon or hull."""
    if room.get('boundary_type') not in ('polygon', 'hull'):
        return None
    poly = Polygon(room['boundary'])
    theta, m = d.frame(room['boundary'])
    zones = [Polygon(k) for k in keepouts]
    framed = [affine_transform(z, m) for z in zones]
    floor = d.keepout_floor(poly, zones)[1]                 # the endpoint's floor (R2): keep-outs and pockets out
    return {'env': floor.buffer(ENVELOPE_TOL), 'env_p': prep(floor.buffer(ENVELOPE_TOL)), 'theta': theta, 'm': m,
            'zones': framed, 'shrunk': [z.buffer(-KEEPOUT_SLACK, join_style=2) for z in framed], 'floor': floor}


def _strict_row(s, o, fp, parent, bb):
    """(area outside the 1 mm floor envelope or None, [(keep-out index, AABB intrusion area)]); parent after R4,
    bb = the P2-frame AABB."""
    out = None if s['env_p'].covers(fp) else fp.difference(s['env']).area
    if d.covering(o, parent):
        return out, []
    rect = box(*bb)
    hits = [(k, rect.intersection(z).area) for k, z in enumerate(s['shrunk'])]
    return out, [(k, a) for k, a in hits if a > 0]


def _support(objs, boundary):
    """Downstream R4 on a whole layout: (P2 frame matrix, {id: AABB in it}, {id: parent after R4})."""
    m = d.frame(boundary)[1]
    by = {o['id']: o for o in objs}
    bbm = {i: d.aabb(o, m) for i, o in by.items()}
    return m, bbm, d.r4_parents(by, bbm)


def _a4(i, j, cur, r4, bbm):
    """downstream A4_COLLISION: one parent after R4, neither a covering, AABBs overlap beyond d.TOL_M."""
    return r4[i] == r4[j] and d._overlaps(bbm[i], bbm[j]) and not d.covering(cur[i], r4[i]) \
        and not d.covering(cur[j], r4[j])


def strict_audit(room, keepouts):
    """{'applicable', 'outside_1mm': [id], 'keepout_1cm': [[id, keep-out index]]} (downstream R2 and A2 rules)."""
    s = _strict(room, keepouts)
    if s is None:
        return {'applicable': False, 'outside_1mm': [], 'keepout_1cm': []}
    _, bbm, r4 = _support(room['objects'], room['boundary'])
    rows = [(o['id'], _strict_row(s, o, Polygon(footprint(o)), r4[o['id']], bbm[o['id']])) for o in room['objects']]
    return {'applicable': True, 'outside_1mm': [i for i, (out, _) in rows if out is not None],
            'keepout_1cm': [[i, k] for i, (_, hits) in rows for k, _ in hits]}


def audit(room, keepouts, held):
    """Full, authoritative audit from check(), geometry() and strict_audit(): vector, phi and the lists behind it."""
    c, gm, s = check(room), g.geometry(room), _strict(room, keepouts)
    objs = room['objects']
    by = {o['id']: o for o in objs}
    fixed = {f['id']: f for f in room.get('fixed') or []}
    fp = lambda i: Polygon(footprint(by[i]))
    parents = {**by, **fixed}
    support = [o['id'] for o in objs if (o.get('parent') is not None and o['parent'] not in parents)
               or not support_ok(o, parents.get(o.get('parent')))]          # geometry()'s rule, with ids
    _, bbm, r4 = _support(objs, room['boundary'])
    strict = [(o['id'], _strict_row(s, o, fp(o['id']), r4[o['id']], bbm[o['id']])) for o in objs] if s else []
    ids = list(by)
    a4 = [[i, j] for k, i in enumerate(ids) for j in ids[k + 1:] if _a4(i, j, by, r4, bbm)]
    windows = [p for p in c['fixed_collisions'] if p not in c['fixed_blocking']]
    broken = [k for k in held if not holds(k, objs, room['boundary'], by)]
    lists = {'oob': c['oob'], 'support_fail': support, 'ceiling': c['ceiling'], 'fixed_blocking': c['fixed_blocking'],
             'window': windows, **{k: c['collisions'][k] for k in ('main', 'severe', 'tuck', 'contained')},
             'outside_1mm': [i for i, (out, _) in strict if out is not None],
             'keepout_1cm': [[i, k] for i, (_, hits) in strict for k, _ in hits], 'held_broken': broken,
             'a4_aabb': a4}
    if gm['support_fail'] != len(support):
        raise AssertionError('support rule drifted from geometry()')
    inner = Polygon(room['boundary']).buffer(OOB_TOL + 1e-9) if c['oob'] else None
    phi = (sum(fp(a).intersection(fp(b)).area for a, b in c['collisions']['main'])
           + sum(fp(o).intersection(Polygon(footprint(fixed[f]))).area for o, f in c['fixed_blocking'])
           + sum(fp(i).difference(inner).area for i in c['oob'])
           + sum(out or 0 for _, (out, _) in strict) + sum(a for _, (_, hits) in strict for _, a in hits))
    return {'v': tuple(len(lists[k]) for k in COMPONENTS), 'phi': phi, 'lists': lists}


# ------------------------------------------------------------------ incremental state (parity: tests)

def _context(room, keepouts, constraints):
    objs = room['objects']
    ids = [o['id'] for o in objs]
    by = {o['id']: o for o in objs}
    order = {i: k for k, i in enumerate(ids)}
    fixed = {f['id']: f for f in room.get('fixed') or []}
    top = lambda o: o['pos'][2] + o['size'][2]
    bounded = room.get('boundary_type') in ('polygon', 'hull')
    poly = Polygon(room['boundary'])
    parent = {i: by[i].get('parent') for i in ids}
    kids = collections.defaultdict(list)
    for i in ids:
        if parent[i] in by:
            kids[parent[i]].append(i)
    cyclic = set()
    for i in ids:
        path, cur = [], i
        while cur in by and cur not in path:
            path.append(cur)
            cur = parent[cur]
        if cur in path:
            cyclic.update(path[path.index(cur):])
    chain, members = {}, {}
    for i in ids:
        path = [i]
        while parent[path[-1]] in by and parent[path[-1]] not in path and path[-1] not in cyclic:
            path.append(parent[path[-1]])
        chain[i] = path
        tree, todo = [], [i]
        while todo:
            j = todo.pop()
            if j not in tree:
                tree.append(j)
                todo += [k for k in kids[j] if k not in cyclic]
        members[i] = sorted(tree, key=order.get)
    flags = {i: [] for i in ids}
    for i, o in by.items():
        seat = by.get(parent[i]) or fixed.get(parent[i])
        if parent[i] is not None and seat is None:
            flags[i].append('parent_missing')
        elif abs(o['pos'][2] - (top(seat) if seat else 0.0)) > TOL + SUPPORT_EPS:
            flags[i].append('support_z')
        if i in cyclic:
            flags[i].append('support_cycle')
        if room.get('height') and top(o) > room['height'] + TOL:
            flags[i].append('ceiling')
    partners = {i: [] for i in ids}
    for a_index, a in enumerate(objs):
        for b in objs[a_index + 1:]:
            if a.get('parent') == b['id'] or b.get('parent') == a['id'] \
                    or min(top(a), top(b)) - max(a['pos'][2], b['pos'][2]) <= TOL \
                    or FLAT.intersection((head(a), head(b))) or min(a['size'][2], b['size'][2]) < 0.06:
                continue
            partners[a['id']].append(b['id'])
            partners[b['id']].append(a['id'])
    hazards = {i: [(f['id'], Polygon(footprint(f)), bool(WINDOW_WORDS & set(norm_cat(f.get('category')).split())))
                   for f in fixed.values() if min(top(o), top(f)) - max(o['pos'][2], f['pos'][2]) > TOL]
               if head(o) not in FLAT and o['size'][2] >= 0.06 else [] for i, o in by.items()}
    against = [['against_wall', i] for i in ids]
    held = [list(k) for k in list(constraints) + against if holds(k, objs, room['boundary'], by)]
    held_of = {i: [k for k, c in enumerate(held) if i in c[1:]] for i in ids}
    m, _, r4 = _support(objs, room['boundary'])
    return {'room': room, 'ids': ids, 'order': order, 'raw': by, 'fixed': fixed, 'kids': kids, 'm': m,
            'r4_cyclic': {i: r4[i] for i in cyclic},              # pinned: a cycle's R4 parents never change
            'parent': parent, 'chain': chain, 'members': members, 'flags': flags,
            'pinned': {i for i in ids if 'parent_missing' in flags[i] or i in cyclic},
            'area': {i: sum(by[j]['size'][0] * by[j]['size'][1] for j in members[i]) for i in ids},
            'sha': {i: int(hashlib.sha256(i.encode()).hexdigest(), 16) for i in ids},
            'bounded': bounded, 'poly': poly, 'inner': poly.buffer(OOB_TOL + 1e-9),
            'inner_p': prep(poly.buffer(OOB_TOL + 1e-9)), 'strict': _strict(room, keepouts),
            'keepouts': keepouts, 'partners': partners, 'hazards': hazards, 'held': held, 'held_of': held_of,
            'ceiling': sum('ceiling' in f for f in flags.values()),
            'reach': math.dist(poly.bounds[:2], poly.bounds[2:]) + 1.0}


def _placed(ctx, i, shift):
    o = ctx['raw'][i]
    if shift == (0, 0):
        return o
    return {**o, 'pos': [round(o['pos'][0] + shift[0] / 1000, 6), round(o['pos'][1] + shift[1] / 1000, 6), o['pos'][2]]}


def _r4(ctx, bbm, i):
    """downstream.r4_parents for one object: a non-cyclic object's depends only on it and its declared parent."""
    if i in ctx['r4_cyclic']:
        return ctx['r4_cyclic'][i]
    p = ctx['parent'][i]
    return p if p in ctx['raw'] and d._positive(bbm[i], bbm[p]) else None


def _row(ctx, cur, fps, i, parent, bb):
    """Rows of object i alone: (oob area | None, support bad, outside area | None, keep-out hits, fixed hits)."""
    o, fp = cur[i], fps[i]
    oob = fp.difference(ctx['inner']).area if ctx['bounded'] and not ctx['inner_p'].contains(fp) else None
    p = o.get('parent')
    seat = cur.get(p) or ctx['fixed'].get(p)
    support = (p is not None and seat is None) or not support_ok(o, seat)
    out, keep = _strict_row(ctx['strict'], o, fp, parent, bb) if ctx['strict'] else (None, [])
    hits = [(f, a, w) for f, zone, w in ctx['hazards'][i] for a in [fp.intersection(zone).area] if a > MIN_AREA]
    return oob, support, out, keep, hits


def _pair(ctx, cur, fps, bbs, i, j):
    a, b = (i, j) if ctx['order'][i] < ctx['order'][j] else (j, i)
    p, q = bbs[a], bbs[b]
    if not (p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]):
        return None
    area = fps[a].intersection(fps[b]).area
    if area <= MIN_AREA:
        return None
    kind = collision_kind(cur[a], cur[b], fps[a], fps[b], TOL)
    return (a, b), (kind, area, kind == 'main' and area > SEVERE * min(fps[a].area, fps[b].area))


def _state(ctx, shift, base=None, changed=None):
    """Audit state; with a base only the rows touching `changed` are recomputed (O(s n))."""
    changed = ctx['ids'] if base is None else list(dict.fromkeys(  # a child's support follows its seat
        changed + [k for i in changed for k in ctx['kids'].get(i, ())]))
    keys = ('cur', 'fp', 'bb', 'bbm', 'r4', 'row', 'held')
    cur, fps, bbs, bbm, r4, row, held = (dict(base[k]) for k in keys) if base else ({}, {}, {}, {}, {}, {}, {})
    moved, order = set(changed), ctx['order']
    pair = {k: v for k, v in base['pair'].items() if k[0] not in moved and k[1] not in moved} if base else {}
    a4 = {k for k in base['a4'] if k[0] not in moved and k[1] not in moved} if base else set()
    for i in changed:
        cur[i] = _placed(ctx, i, shift[i])
        fps[i] = Polygon(footprint(cur[i]))
        bbs[i] = fps[i].bounds
        bbm[i] = d.aabb(cur[i], ctx['m'])
    for i in changed:
        r4[i] = _r4(ctx, bbm, i)
    for i in changed:
        row[i] = _row(ctx, cur, fps, i, r4[i], bbm[i])
        for j in ctx['partners'][i]:
            if j in moved and order[j] < order[i]:
                continue
            hit = _pair(ctx, cur, fps, bbs, i, j)
            if hit:
                pair[hit[0]] = hit[1]
        a4.update((i, j) if order[i] < order[j] else (j, i) for j in ctx['ids'] if j != i
                  and not (j in moved and order[j] < order[i]) and _a4(i, j, cur, r4, bbm))
    for k in sorted({k for i in changed for k in ctx['held_of'][i]}):
        held[k] = holds(ctx['held'][k], None, ctx['room']['boundary'], cur)
    rows = list(row.values())
    count = collections.Counter(kind for kind, _, _ in pair.values())
    v = (sum(r[0] is not None for r in rows), sum(r[1] for r in rows), ctx['ceiling'],
         sum(not w for r in rows for _, _, w in r[4]), sum(w for r in rows for _, _, w in r[4]),
         count['main'], sum(s for _, _, s in pair.values()), count['tuck'], count['contained'],
         sum(r[2] is not None for r in rows), sum(len(r[3]) for r in rows), sum(not x for x in held.values()),
         len(a4))
    phi = (sum(a for kind, a, _ in pair.values() if kind == 'main') + sum(a for r in rows for _, a, w in r[4] if not w)
           + sum(r[0] or 0 for r in rows) + sum(r[2] or 0 for r in rows) + sum(a for r in rows for _, a in r[3]))
    return {'shift': shift, 'cur': cur, 'fp': fps, 'bb': bbs, 'bbm': bbm, 'r4': r4, 'row': row, 'held': held,
            'pair': pair, 'a4': a4, 'v': v, 'phi': phi}


def audit_delta(ctx, state, moves):
    """state after adding the per-object millimetre translations `moves` {id: (mx, my)}."""
    shift = {**state['shift'], **{i: (state['shift'][i][0] + m[0], state['shift'][i][1] + m[1])
                                  for i, m in moves.items()}}
    return _state(ctx, shift, state, list(moves))


def _better(a, b):
    return all(y <= x for x, y in zip(a['v'], b['v'])) and (b['v'] != a['v'] or b['phi'] <= a['phi'] - EPS_PHI)


# ------------------------------------------------------------------ candidates

def _rot(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def _proj(o, u):
    values = [p[0] * u[0] + p[1] * u[1] for p in footprint(o)]
    return min(values), max(values)


def _along(x, y, u):
    """Push moving x along unit u until clear of y, and y's extent on u."""
    lo, hi = _proj(y, u)
    return hi - _proj(x, u)[0], u, hi - lo


def _sat(x, y):
    """[(push, unit, extent of y)] on the SAT axes of the two footprints, away from y's centre, shortest first."""
    axes = []
    for o in (x, y):
        for u in ((math.cos(o['yaw']), math.sin(o['yaw'])), (-math.sin(o['yaw']), math.cos(o['yaw']))):
            if not any(abs(u[0] * w[1] - u[1] * w[0]) < 1e-9 for w in axes):
                axes.append(u)
    out = []
    for u in axes:
        side = (x['pos'][0] - y['pos'][0]) * u[0] + (x['pos'][1] - y['pos'][1]) * u[1]
        out.append(_along(x, y, u if side >= 0 else (-u[0], -u[1])))
    return sorted(out, key=lambda t: t[0])


def _free_dirs(ctx, x, y):
    """Co-located: x's local +x, +y, -x, -y, most free room (ray to the boundary minus half extent) first."""
    out = []
    for k, (u, half) in enumerate(zip([_rot(1, 0, x['yaw']), _rot(0, 1, x['yaw'])] * 2,
                                      [x['size'][0] / 2, x['size'][1] / 2] * 2)):
        u = u if k < 2 else (-u[0], -u[1])
        ray = LineString([x['pos'][:2], (x['pos'][0] + u[0] * ctx['reach'], x['pos'][1] + u[1] * ctx['reach'])])
        inside = ray.intersection(ctx['poly'])
        parts = getattr(inside, 'geoms', [inside])
        ahead = max((p.length for p in parts if p.geom_type == 'LineString'
                     and p.distance(Point(x['pos'][:2])) < 1e-9), default=0.0)
        out.append((-(ahead - half), k, _along(x, y, u)))
    return [t[2] for t in sorted(out, key=lambda t: t[:2])]


def _clamp(seat, x, y):
    """Centre clamped into the seat footprint shrunk by CLAMP (pose_targets._normalize's rule)."""
    lx, ly = _rot(x - seat['pos'][0], y - seat['pos'][1], -seat['yaw'])
    hx, hy = max(seat['size'][0] / 2 - CLAMP, 0.0), max(seat['size'][1] / 2 - CLAMP, 0.0)
    dx, dy = _rot(min(max(lx, -hx), hx), min(max(ly, -hy), hy), seat['yaw'])
    return seat['pos'][0] + dx, seat['pos'][1] + dy


def _unit_move(ctx, st, unit, dx, dy):
    """{member: (mx, my)} moving `unit` by ~(dx, dy) in whole millimetres, or None (pinned, no-op, off its seat)."""
    if unit in ctx['pinned']:
        return None
    o = st['cur'][unit]
    p = o.get('parent')
    seat = st['cur'].get(p) or ctx['fixed'].get(p)
    x, y = o['pos'][0] + dx, o['pos'][1] + dy
    if seat is not None:
        x, y = _clamp(seat, x, y)
    mm = (round((x - o['pos'][0]) * 1000), round((y - o['pos'][1]) * 1000))
    if mm == (0, 0):
        return None
    if seat is not None:
        s = st['shift'][unit]
        if not support_contains(seat, _placed(ctx, unit, (s[0] + mm[0], s[1] + mm[1]))['pos'], tol=0):
            return None
    return {j: mm for j in ctx['members'][unit]}


def _movers(ctx, a, b):
    """[(unit, its object, the other object)]: each object alone (a child stays on its seat) and the branch it heads
    below the common ancestor (different floor roots: its whole tree); smaller unit first (ties: larger sha256 id)."""
    ca, cb = ctx['chain'][a], ctx['chain'][b]
    if ca[-1] != cb[-1]:
        tops = (ca[-1], cb[-1])
    else:
        common = set(ca) & set(cb)
        below = lambda c: next(c[k - 1] if k else None for k, z in enumerate(c) if z in common)
        tops = (below(ca), below(cb))
    picks = [(unit, x, y) for top, x, y in zip(tops, (a, b), (b, a)) if top is not None
             for unit in dict.fromkeys((x, top)) if unit not in ctx['pinned']]
    return sorted(picks, key=lambda p: (ctx['area'][p[0]], -ctx['sha'][p[0]]))


def _pair_moves(ctx, st, a, b):
    cur = st['cur']
    movers = _movers(ctx, a, b)
    co = math.dist(cur[a]['pos'][:2], cur[b]['pos'][:2]) < COLOCATED
    dirs = {x: _free_dirs(ctx, cur[x], cur[y]) if co else _sat(cur[x], cur[y])
            for x, y in dict.fromkeys((x, y) for _, x, y in movers)}
    plans = [(unit, x, dirs[x]) for unit, x, _ in movers]
    for unit, _, ways in plans:
        for push, u, ext in ways:
            for m in range(3):
                yield _unit_move(ctx, st, unit, *(u[k] * (push + CLEAR + m * ext) for k in (0, 1)))
    other = next((p for p in plans if p[1] != plans[0][1]), None) if plans else None
    if other and plans[0][2]:                  # the smallest unit and the other object's smallest, half each
        push, u, _ = plans[0][2][0]
        half = (push + CLEAR) / 2
        first, second = (_unit_move(ctx, st, unit, u[0] * s * half, u[1] * s * half)
                         for (unit, _, _), s in zip((plans[0], other), (1, -1)))
        if first and second:
            yield {**first, **second}


def _push(fp, region):
    """interface._push: per axis the largest corner move onto `region`; the push and its axis parts."""
    push = [0.0, 0.0]
    for c in fp.exterior.coords[:-1]:
        if not region.covers(Point(c)):
            q = nearest_points(region.boundary, Point(c))[0]
            for k, delta in enumerate((q.x - c[0], q.y - c[1])):
                if abs(delta) > abs(push[k]):
                    push[k] = delta
    return [p for p in (tuple(push), (push[0], 0.0), (0.0, push[1])) if p != (0.0, 0.0)]


def _candidates(ctx, st, kind, ids):
    cur = st['cur']
    if kind == 'pair':
        yield from _pair_moves(ctx, st, *ids)
    elif kind == 'fixed':
        o, f = ids
        for push, u, _ in _sat(cur[o], ctx['fixed'][f]):
            yield _unit_move(ctx, st, ctx['chain'][o][-1], u[0] * (push + CLEAR), u[1] * (push + CLEAR))
    elif kind == 'oob':
        for unit in dict.fromkeys((ids[0], ctx['chain'][ids[0]][-1])):
            for dx, dy in _push(st['fp'][ids[0]], ctx['poly']):
                yield _unit_move(ctx, st, unit, dx, dy)
    elif kind == 'support':
        o = cur[ids[0]]
        seat = cur.get(o['parent']) or ctx['fixed'].get(o['parent'])
        x, y = _clamp(seat, *o['pos'][:2])
        yield _unit_move(ctx, st, ids[0], x - o['pos'][0], y - o['pos'][1])
    elif kind == 'strict':
        s, i = ctx['strict'], ids[0]
        bb, frame = st['bbm'][i], []
        zones = [k for k, _ in st['row'][i][3]]
        if st['row'][i][2] is not None:         # outside the floor envelope: also every keep-out it enters (coverings)
            fpm = affine_transform(st['fp'][i], s['m'])
            zones += [k for k, z in enumerate(s['zones']) if k not in zones and fpm.intersection(z).area > 0]
        for k in zones:                         # clear of each keep-out's AABB (P2 frame), shortest first
            z = s['zones'][k].bounds
            frame += sorted(((z[2] - bb[0] + PUSH_MARGIN, 0.0), (z[0] - bb[2] - PUSH_MARGIN, 0.0),
                             (0.0, z[3] - bb[1] + PUSH_MARGIN), (0.0, z[1] - bb[3] - PUSH_MARGIN)),
                            key=lambda p: abs(p[0]) + abs(p[1]))
        pushes = _push(st['fp'][i], s['floor']) + [_rot(dx, dy, s['theta']) for dx, dy in frame]
        for unit in dict.fromkeys((i, ctx['chain'][i][-1])):
            for dx, dy in pushes + SNAP_GRID:
                yield _unit_move(ctx, st, unit, dx, dy)


def _problems(ctx, st):
    out = [(SEVERITY['severe' if severe else 'main'], ids, 'pair') for ids, (kind, _, severe) in st['pair'].items()
           if kind == 'main']
    for i, r in st['row'].items():
        out += [(SEVERITY['fixed'], (i, f), 'fixed') for f, _, w in r[4] if not w]
        if r[0] is not None:
            out.append((SEVERITY['oob'], (i,), 'oob'))
        if r[1] and not {'support_z', 'parent_missing', 'support_cycle'} & set(ctx['flags'][i]):
            out.append((SEVERITY['support'], (i,), 'support'))
    return sorted(out)


def _still(st, kind, ids):
    if kind == 'pair':
        return st['pair'].get(ids, ('',))[0] == 'main'
    if kind == 'fixed':
        return any(f == ids[1] and not w for f, _, w in st['row'][ids[0]][4])
    if kind == 'strict':
        return st['row'][ids[0]][2] is not None or bool(st['row'][ids[0]][3])
    return st['row'][ids[0]][0] is not None if kind == 'oob' else st['row'][ids[0]][1]


# ------------------------------------------------------------------ repair

def _finite(value, n):
    return isinstance(value, (list, tuple)) and len(value) == n and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)


def _validate(room):
    objs = room.get('objects')
    if not isinstance(objs, list) or not all(isinstance(o, dict) for o in objs):
        raise ValueError('objects: expected a list of dicts')
    ids = [o.get('id') for o in objs]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('ids: nonempty and unique strings required')
    for o in objs + list(room.get('fixed') or []):
        if not (_finite(o.get('size'), 3) and min(o['size']) > 0 and _finite(o.get('pos'), 3)
                and _finite([o.get('yaw')], 1)):
            raise ValueError('object geometry: finite size > 0, pos and yaw required')
    b = room.get('boundary')
    if not (isinstance(b, list) and len(b) >= 3 and all(_finite(p, 2) for p in b)) \
            or not Polygon(b).is_valid or Polygon(b).area <= 0:
        raise ValueError('boundary: expected a valid polygon')


def _invariant(raw, out, shift):
    keys = ('id', 'category', 'size', 'yaw', 'parent')
    return len(raw) == len(out) and all(
        all(a.get(k) == b.get(k) for k in keys) and a['pos'][2] == b['pos'][2]
        and (shift[a['id']] != (0, 0) or a is b) for a, b in zip(raw, out))


def repair(room, *, keepouts=None, constraints=(), max_trials=2000, time_limit=60.0):
    """{'status': ok|partial|unchanged|rejected, 'room', 'diagnostics'}; see the module docstring."""
    started = time.monotonic()
    try:
        _validate(room)
    except ValueError as exc:
        return {'status': 'rejected', 'room': copy.deepcopy(room), 'diagnostics': {'error': str(exc)}}
    source = 'request' if keepouts is not None else ('doors' if d.door_strips(room) else 'none')
    keepouts = door_keepouts(room) if keepouts is None else keepouts
    ctx = _context(room, keepouts, constraints)
    before = audit(room, keepouts, ctx['held'])
    state = _state(ctx, {i: (0, 0) for i in ctx['ids']})
    deadline, trials, steps, timed_out = started + time_limit, {'a': 0, 'b': 0}, [], False

    def attempt(stage, problem, candidates, clears=None):
        nonlocal state, timed_out
        seen = set()
        for moves in candidates:
            key = tuple(sorted(moves.items())) if moves else None
            if key is None or key in seen:
                continue
            seen.add(key)
            if trials[stage] >= max_trials:
                return False
            if time.monotonic() > deadline:
                timed_out = True
                return False
            trials[stage] += 1
            after = audit_delta(ctx, state, moves)
            if _better(state, after) and (clears is None or clears(after)):
                state = after
                steps.append({'stage': stage, 'problem': [problem[0], *problem[1]],
                              'move_mm': {i: list(m) for i, m in moves.items()}})
                return True
        return False

    while not timed_out and trials['a'] < max_trials:                       # stage a: checker level
        progress = False
        for _, ids, kind in _problems(ctx, state):
            if _still(state, kind, ids) and attempt('a', (kind, ids), _candidates(ctx, state, kind, ids)):
                progress = True
            if timed_out or trials['a'] >= max_trials:
                break
        if not progress:
            break
    after_a = state
    # An empty usable floor (keep-outs cover the room; downstream R1 refuses it anyway) has nothing to snap into.
    strict_skipped = 'empty_floor' if ctx['strict'] and ctx['strict']['floor'].is_empty else None
    if ctx['strict'] and not timed_out and not strict_skipped:               # stage b: strict downstream stage
        strict_bad = lambda: [i for i in ctx['ids'] if _still(state, 'strict', (i,))]
        if strict_bad():
            pl = {i: {'pos': list(state['cur'][i]['pos']), 'yaw': state['cur'][i]['yaw'], 'on': ctx['parent'][i]}
                  for i in ctx['ids']}
            snap_inside(pl, {'objects': room['objects'], 'fixed': room.get('fixed') or []}, ctx['strict']['floor'],
                        ctx['held'], room['boundary'])
            snapped = {i: (round((pl[i]['pos'][0] - state['cur'][i]['pos'][0]) * 1000),
                           round((pl[i]['pos'][1] - state['cur'][i]['pos'][1]) * 1000)) for i in ctx['ids']}
            attempt('b', ('snap', ()), [{i: m for i, m in snapped.items() if m != (0, 0)}])
            for i in sorted(strict_bad(), key=lambda i: (state['cur'][i]['pos'][2], i)):
                if _still(state, 'strict', (i,)):
                    attempt('b', ('strict', (i,)), _candidates(ctx, state, 'strict', (i,)),
                            lambda after: not _still(after, 'strict', (i,)))
    objects = [copy.deepcopy(state['cur'][i]) if state['shift'][i] != (0, 0) else ctx['raw'][i] for i in ctx['ids']]
    final_room = {**room, 'objects': objects}
    final = audit(final_room, keepouts, ctx['held'])
    rolled, verdict = None, {'raw': d.accept(room, room['objects'])}
    if not _invariant(room['objects'], objects, state['shift']):
        rolled = 'guard_violation'
    elif any(b > a for a, b in zip(before['v'], final['v'])):
        rolled = 'checker_worse'
    elif steps:
        verdict['s1'] = d.accept(final_room, objects)
        if any(verdict['raw'][k] and not verdict['s1'][k] for k in DOWNSTREAM_FLAGS):
            rolled = 'downstream_worse'
    if rolled or not steps:
        final_room, final, objects = copy.deepcopy(room), before, room['objects']
        verdict['s1'] = verdict['raw']
    budget = trials['a'] >= max_trials or trials['b'] >= max_trials
    lists = final['lists']
    unsolved = collections.defaultdict(list)
    for key in FAILS:
        for item in lists[key]:
            for i in [item] if isinstance(item, str) else [x for x in item if isinstance(x, str) and x in ctx['raw']]:
                if key not in unsolved[i]:
                    unsolved[i].append(key)
    for i in list(unsolved):
        unsolved[i] += ctx['flags'][i] + (['timeout'] if timed_out else []) + (['budget_exhausted'] if budget else [])
    clean = not any(lists[k] for k in FAILS)
    status = 'unchanged' if rolled or not steps else 'ok' if clean else 'partial'
    vector = lambda s: dict(zip(COMPONENTS, s['v'])) | {'phi': s['phi']}
    return {'status': status, 'room': copy.deepcopy(final_room if status != 'unchanged' else room),
            'diagnostics': {'before': vector(before), 'after_a': vector(after_a), 'after_b': vector(state),
                            'final': vector(final), 'audit_parity': rolled is not None or not steps or state['v'] == final['v'],
                            'displacement': [math.dist(a['pos'][:2], b['pos'][:2])
                                             for a, b in zip(room['objects'], final_room['objects'])],
                            'unsolved': dict(unsolved), 'flags': {i: f for i, f in ctx['flags'].items() if f},
                            'before_lists': before['lists'], 'trials': trials, 'steps': steps,
                            'runtime_s': time.monotonic() - started, 'timed_out': timed_out,
                            'budget_exhausted': budget, 'rolled_back': rolled, 'keepout_source': source,
                            'keepouts': keepouts, 'downstream': verdict, 'strict_skipped': strict_skipped}}


# ------------------------------------------------------------------ rows, summary, CLI

RATES = ('geometry_success', 'strict_ok', 'accepted', 'endpoint_ok', 'attach_ok', 'admit_ok', 'admit_lenient_ok',
         'admit_no_required_ok')
COUNTERS = ('contained', 'tuck', 'thin', 'thin_reference_tall', 'flat_head', 'same_category_parent',
            'parents_changed', 'yaw_changed', 'size_changed', 'identity_changed')
BUGS = ('parents_changed', 'yaw_changed', 'size_changed', 'identity_changed')
NOT_XY = ('support_z', 'parent_missing', 'support_cycle', 'ceiling')


def anti_gaming(room, raw, reference=None):
    """Counters 1-9 of `room` against `raw` (and the GT `reference`); parents/yaw/size/identity changes must be 0."""
    c = check(room)
    objs, base = room['objects'], raw['objects']
    by = {o['id']: o for o in objs}
    ref = {o['id']: o for o in (reference or {}).get('objects', [])}
    thin = [o for o in objs if o['size'][2] < 0.06]
    pairs = lambda rows: [(o['id'], o.get('category')) for o in rows]
    return {'contained': len(c['collisions']['contained']), 'tuck': len(c['collisions']['tuck']), 'thin': len(thin),
            'thin_reference_tall': sum(ref[o['id']]['size'][2] >= 0.06 for o in thin if o['id'] in ref),
            'flat_head': sum(head(o) in FLAT for o in objs),
            'same_category_parent': sum(o.get('parent') in by and norm_cat(by[o['parent']].get('category'))
                                        == norm_cat(o.get('category')) for o in objs),
            'parents_changed': sum(a.get('parent') != b.get('parent') for a, b in zip(objs, base)),
            'yaw_changed': sum(a['yaw'] != b['yaw'] for a, b in zip(objs, base)),
            'size_changed': sum(a['size'] != b['size'] for a, b in zip(objs, base)),
            'identity_changed': int(pairs(objs) != pairs(base)),
            'parent_accuracy': statistics.mean(o.get('parent') == ref.get(o['id'], {}).get('parent') for o in objs)
            if ref and objs else None}


def _stratum(n):
    return next(f'{lo}+' if hi is None else f'{lo}-{hi}' for lo, hi in STRATA if n >= lo and (hi is None or n <= hi))


def _displacement(disp):
    """XY displacement over every object, and over the moved ones only (moved_*)."""
    if not disp:
        return None
    moved = [x for x in disp if x > 0]
    return {'objects': len(disp), 'median': statistics.median(disp), 'p95': g._quantile(disp, .95), 'max': max(disp),
            'moved_fraction': len(moved) / len(disp), 'over_0_5m_fraction': sum(x > .5 for x in disp) / len(disp),
            'moved_objects': len(moved), 'moved_median': statistics.median(moved) if moved else None,
            'moved_p95': g._quantile(moved, .95) if moved else None}


def _verdict(v):
    return {k: v[k] for k in ('accepted', 'endpoint_ok', 'attach_ok', 'admit_ok', 'admit_lenient_ok',
                              'admit_no_required_ok', 'reasons', 'violations', 'window_overlap', 'snap_moved')}


def repair_row(task):
    """(geometry_row, gt_room, request[, max_trials, time_limit]) -> the row + s1_layout + s1 (raw beside S1)."""
    row, room, request, *limits = task
    max_trials, time_limit = limits or (2000, 60.0)
    pred = predicted_room(room, request['messages'], row['raw_layout'])
    doors = d.door_strips(pred)                 # strata from the request, incomplete rows included
    s1 = {'stratum': _stratum(row['objects']), 'objects': row['objects'], 'boundary_type': row['boundary_type'],
          'rectangle': d.is_rectangle(Polygon(pred['boundary'])), 'doors': len(doors),
          'detached_doors': sum(x['detached'] for x in doors)}
    if not row['complete']:
        return {**row, 's1_layout': None,
                's1': {**s1, 'status': 'not_complete', **{f'{k}_{a}': False for k in RATES for a in ('raw', 's1')}}}
    result = repair(pred, max_trials=max_trials, time_limit=time_limit)
    if result['status'] == 'rejected':
        raise ValueError('complete row rejected by repair: ' + result['diagnostics']['error'])
    out, diag = result['room'], result['diagnostics']
    rooms = {'raw': pred, 's1': out}
    for arm, layout_room in rooms.items():
        geo, verdict = g.geometry(layout_room), diag['downstream'][arm]       # accept() as repair ran it
        s1 |= {f'geometry_{arm}': geo, f'downstream_{arm}': _verdict(verdict),
               f'geometry_success_{arm}': geo['geometry_success'], f'strict_ok_{arm}': verdict['accepted'],
               f'accepted_{arm}': geo['geometry_success'] and verdict['accepted'],
               **{f'{k}_{arm}': verdict[k] for k in RATES[3:]}}
    disp, flags, lists = diag['displacement'], diag['flags'], diag['before_lists']
    xy = {k: len(lists[k]) for k in ('main', 'severe', 'fixed_blocking', 'oob', 'outside_1mm', 'keepout_1cm',
                                     'a4_aabb')}
    xy['support_xy'] = sum(not set(NOT_XY[:3]) & set(flags.get(i, [])) for i in lists['support_fail'])
    s1 |= {'status': result['status'], 'keepout_source': diag['keepout_source'],
           'displacement_m': disp, 'displacement': _displacement(disp),
           'vectors': {k: diag[k] for k in ('before', 'after_a', 'after_b', 'final')},
           'raw_failures': {'xy_fixable': xy, 'not_xy_fixable': dict(collections.Counter(
               f for fs in flags.values() for f in fs if f in NOT_XY))},
           **{k: diag[k] for k in ('runtime_s', 'timed_out', 'budget_exhausted', 'rolled_back', 'trials', 'unsolved',
                                   'audit_parity')}, 'steps': len(diag['steps']),
           'anti_gaming_raw': anti_gaming(pred, pred, room), 'anti_gaming_s1': anti_gaming(out, pred, room)}
    return {**row, 's1_layout': out['objects'], 's1': s1}


def _aggregate(rows):
    n, s = len(rows), [r['s1'] for r in rows]
    done = [x for x in s if x['status'] != 'not_complete']
    result = {'requests': n, 'complete': len(done), 'status': dict(collections.Counter(x['status'] for x in s))}
    for key in RATES:
        for arm in ('raw', 's1'):
            result[f'{key}_{arm}_rate'] = sum(bool(x[f'{key}_{arm}']) for x in s) / n if n else None
    result['displacement'] = _displacement([v for x in done for v in x['displacement_m']])
    for key in ('timed_out', 'budget_exhausted'):
        result[key] = sum(bool(x[key]) for x in done)
    result['audit_parity_mismatch'] = sum(not x['audit_parity'] for x in done)
    result['rolled_back'] = dict(collections.Counter(x['rolled_back'] for x in done if x['rolled_back']))
    result['runtime_s'] = {'mean': statistics.mean(x['runtime_s'] for x in done),
                           'max': max(x['runtime_s'] for x in done)} if done else None
    result['anti_gaming'] = {arm: {k: sum(x[f'anti_gaming_{arm}'][k] for x in done) for k in COUNTERS}
                             for arm in ('raw', 's1')}
    accuracy = [x['anti_gaming_raw']['parent_accuracy'] for x in done if x['anti_gaming_raw']['parent_accuracy'] is not None]
    result['parent_accuracy'] = statistics.mean(accuracy) if accuracy else None
    result['bugs'] = {k: result['anti_gaming']['s1'][k] for k in BUGS if result['anti_gaming']['s1'][k]}
    result['raw_failures'] = {group: dict(sum((collections.Counter(x['raw_failures'][group]) for x in done),
                                              collections.Counter())) for group in ('xy_fixable', 'not_xy_fixable')}
    return result


def summarize(rows):
    """all / by object-count stratum / room shape / doors / source; raw and S1 side by side, all frozen requests as
    denominator."""
    labels = [_stratum(lo) for lo, _ in STRATA]
    sources = sorted({r['source'] for r in rows})
    return {'all': _aggregate(rows),
            'by_stratum': {k: _aggregate([r for r in rows if r['s1']['stratum'] == k]) for k in labels},
            'by_shape': {k: _aggregate([r for r in rows if r['s1']['rectangle'] == (k == 'rectangle')])
                         for k in ('rectangle', 'non_rectangle')},
            'by_doors': {k: _aggregate([r for r in rows if (r['s1']['doors'] > 0) == (k == 'doors')])
                         for k in ('no_doors', 'doors')},
            'by_source': {k: _aggregate([r for r in rows if r['source'] == k]) for k in sources}}


def load_uids(path):
    """--uids FILE: a JSON list of uids, or a cohort JSON with 'uids'."""
    value = h.strict_json(Path(path).read_text())
    uids = value.get('uids') if isinstance(value, dict) else value
    if not isinstance(uids, list) or not uids or not all(isinstance(u, str) and u for u in uids) \
            or len(set(uids)) != len(uids):
        raise ValueError('uids: expected a nonempty list of unique uid strings')
    return uids


def _report(out, summary, given=None):
    fmt = lambda v: '—' if v is None else f'{v:.3f}'
    denominator = '全部冻结请求' if given is None else f'--uids 给定的全部 {given} 个请求'
    lines = ['# S1 输出修复：raw 与 S1 并列', '',
             'S1 只平移整棵支撑子树；类别、物体集合、id、尺寸、朝向、父物体和 z 都不变，修不好只标记。S1 不替换 raw。', '',
             '几何成功 = geometry_success（冻结检查器）。下游接受 = downstream.accept（EmbodiedGen Path A 规则的 FastFill 移植；'
             '门禁区按后端规则由门的固定框合成，属近似，详见 downstream.py）。两者都过 = 同时满足。'
             f'比率分母为{denominator}，未完成输出与无法修复的房间都计为失败。', '',
             '| 范围 | 请求 | 几何成功 raw → S1 | 下游接受 raw → S1 | 两者都过 raw → S1 | 位移中位 / p95，全部物体 (m) '
             '| 位移中位 / p95，仅被移动物体 (m) | 超时 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    names = {'rectangle': '矩形房间', 'non_rectangle': '非矩形房间', 'no_doors': '无门房间', 'doors': '有门房间'}
    scopes = [('全部', summary['all'])] + [(f'物体数 {k}', v) for k, v in summary['by_stratum'].items()] + \
        [(names[k], v) for group in ('by_shape', 'by_doors') for k, v in summary[group].items()] + \
        list(summary['by_source'].items())
    for name, r in scopes:
        disp = r['displacement'] or {}
        lines.append(f"| {name} | {r['requests']} | " + ' | '.join(
            f"{fmt(r[k + '_raw_rate'])} → {fmt(r[k + '_s1_rate'])}" for k in ('geometry_success', 'strict_ok', 'accepted'))
            + f" | {fmt(disp.get('median'))} / {fmt(disp.get('p95'))}"
            + f" | {fmt(disp.get('moved_median'))} / {fmt(disp.get('moved_p95'))} | {r['timed_out']} |")
    bugs = summary['all']['bugs']
    lines += ['', f"不变量计数（父物体、朝向、尺寸、身份改变，必须为 0）：{bugs or '全部为 0'}。",
              'summary.json 另含宽松入口（P4）、required 为空的下界（P5）、端点 / 包络各分量、反作弊计数和原始失败类型拆分。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def run(data, geometry, requests_arm, out, workers=1, max_trials=2000, time_limit=60.0, uids=None):
    data, geometry, out = Path(data), Path(geometry), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if requests_arm not in h.ARMS:
        raise ValueError('requests arm must be joint or placement')
    manifest = e._verified(data)
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    name = requests_arm + '/requests.jsonl'
    requests = h._by_uid(h._rows(data, name, manifest['files'][name]))
    rows = h._rows(geometry.parent, geometry.name, e._verified(geometry.parent)['files'][geometry.name])
    given = None if uids is None else load_uids(uids)
    if given is None and {x['uid'] for x in rows} != set(requests):   # else the rate denominator silently shrinks
        raise ValueError('geometry rows must cover every frozen request')
    if given is not None and {x['uid'] for x in rows} != set(given):
        raise ValueError('geometry rows must cover exactly the given uids')
    tasks = []
    for row in rows:
        room, request = rooms.get(row['uid']), requests.get(row['uid'])
        if room is None or request is None or any(row[k] != room[k] or request[k] != room[k]
                                                  for k in ('uid', 'source', 'group')):
            raise ValueError('geometry/request/reference identity mismatch')
        if row.get('arm') in h.ARMS and row['arm'] != requests_arm:
            raise ValueError('geometry arm does not match --requests-arm')
        tasks.append((row, room, request, max_trials, time_limit))
    if workers == 1:
        results = [repair_row(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(repair_row, tasks, chunksize=4))
    code = {p.name: h.sha256(p) for p in map(Path, (__file__, d.__file__, g.__file__, check.__code__.co_filename,
                                                    snap_inside.__code__.co_filename))}
    summary = {'status': 'S1_COMPLETE', 'requests_arm': requests_arm, 'rows': len(results),
               'params': {'max_trials': max_trials, 'time_limit_s': time_limit, 'components': COMPONENTS,
                          'clear_m': CLEAR, 'envelope_tol_m': ENVELOPE_TOL, 'keepout_slack_m': KEEPOUT_SLACK,
                          'door_depth_m': DOOR_DEPTH, 'door_margin_m': DOOR_MARGIN, 'eps_phi': EPS_PHI},
               'inputs': {'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'), 'geometry': geometry.name,
                          'geometry_sha256': h.sha256(geometry),
                          'geometry_manifest_sha256': h.sha256(geometry.parent / 'MANIFEST.json')},
               'metric_contract': {'rate_denominator': 'all_frozen_requests', 'raw': 'reported_beside_s1_never_replaced',
                                   'strict_ok': 'downstream.accept_path_a_strict_every_door_entry',
                                   'accepted': 'geometry_success_and_strict_ok', 'windows': 'not_gated'},
               'summary': summarize(results), 'code_sha256': code}
    if given is not None:
        summary['inputs']['uids'] = {'file': Path(uids).name, 'sha256': h.sha256(uids), 'count': len(given)}
        summary['metric_contract']['rate_denominator'] = 'given_uids'
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        (dest / (geometry.stem + '-s1.jsonl')).write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in results))
        h.write_json(dest / 'summary.json', summary)
        _report(dest, summary['summary'], None if given is None else len(given))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest), 'code_sha256': code})
        _publish(dest, out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('data', 'geometry', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--requests-arm', choices=h.ARMS, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-trials', type=int, default=2000)
    parser.add_argument('--time-limit', type=float, default=60.0)
    parser.add_argument('--uids', type=Path)
    args = parser.parse_args()
    result = run(args.data, args.geometry, args.requests_arm, args.out, args.workers, args.max_trials, args.time_limit,
                 args.uids)
    print(json.dumps({'status': result['status'], 'rows': result['rows']}))


if __name__ == '__main__':
    main()
