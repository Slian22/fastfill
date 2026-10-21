"""S2-F-task (protocol H4): a stage after S2-F that makes the room's TASK objects reachable for a mobile robot.

Task objects: task_reach_headroom.task_cats (2 hash-picked floor categories). Task-scoped admission = downstream.accept
with room['objects'] restricted to them (task_nav._task_accept). Only rooms failing it with an A3 code are touched.
State, moves and audit = output_repair (_context / _state / audit_delta, as labbuilder_nav_repair): whole support
subtrees, XY millimetre translations; ids, sizes, yaw, parents, z frozen; every object placed.
Progress key (potential): accept()'s nav grid on the state (AGENT_RADIUS_M, NAV_GRID_M, every door an entry) ->
(A3 proxy = task objects whose footprint inflated by AGENT_RADIUS_M + NAV_GRID_M misses a door's reached region,
summed over doors, + entry-blocked / too-few doors; obstruction = fewest blocked cells between each door's reached
region and each such object). Every move below must keep: no audit component rises (checker, strict R2/A2/A4, S1's
against_wall; as in S2-F, a companion moving alone with its relation kept may leave its wall), every companion pair
held at stage start still holds, and the key falls (lexicographic).
  1. relocate: each unreachable task object's floor-root subtree (an anchor with its held companions as one group; a
     companion alone, else its anchor's whole group) to candidates = S2's wall-anchored candidates
     (s2_protect._root_candidates) + a GRID_M grid over the room, nearest first; a candidate is taken only if the
     object then touches the reached region of every door.
  2. push blockers (labbuilder_nav_repair moves: +-x/+-y of the P2 frame by PUSH_STEPS x NAV_GRID_M, larger steps and
     more rounds): first the roots on the cheapest obstruction paths, then the reached region's frontier.
  2b. if no push helps (BLOCKER_RELOCATE, chosen on dev: +9 of 600 rooms): relocate a path blocker's unit to its
     MAX_BLOCKER_CANDS nearest candidates (same candidate set and gates as 1, no touch rule). 1 / 2 / 2b repeat while
     the key falls (deadline TIME_LIMIT per room).
  3. the room change is kept iff task-scoped admission (the real accept) becomes true, checker and all-category
     strict not worse, invariants hold; else the S2-F layout is returned unchanged.

python3 s2f_task.py --selfcheck
python3 s2f_task.py --run --final S2F_FINAL.jsonl --data DATA --out OUT [--workers 8]
"""
import argparse
import collections
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from shapely.affinity import affine_transform, translate
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient
from shapely.prepared import prep

sys.path.insert(0, str(Path(__file__).parent))
import companions as C  # noqa: E402
import labbuilder_nav_repair as L  # noqa: E402
import s2_protect as S  # noqa: E402
import task_reach_headroom as T  # noqa: E402

REPO = '/Users/slian/Desktop/3D/Worldedge/OptiScene'
S._imports(REPO)
d, g, r = S.d, S.g, S.orp
R, G = d.AGENT_RADIUS_M, d.NAV_GRID_M
A3 = L.A3_FIX
GRID_M, MAX_CANDS = 0.05, 8000                 # relocation grid, candidates per unit
PUSH_STEPS, PUSH_EVALS = (1, 2, 3, 4, 6, 8), 3000
BLOCKER_RELOCATE, MAX_BLOCKER_CANDS = True, 2000   # step 2b (dev ablation), its nearest candidates per unit
TIME_LIMIT = 90.0
BINS = (('1-10', 1, 10), ('11-20', 11, 20), ('21+', 21, 10 ** 9))


def task_accept(room, lay, cats):
    return d.accept({**room, 'objects': [o for o in room['objects'] if o['category'] in cats]}, lay)


# ------------------------------------------------------------------ accept()'s nav grid on an audit state

def nav_ctx(room):
    poly = orient(Polygon(room['boundary']))
    m = d.frame(room['boundary'])[1]
    fpoly = affine_transform(poly, m)
    w, dd = fpoly.bounds[2:]
    lo = math.ceil(R / G - 0.5)
    ux, uy = (math.floor((e - R) / G - 0.5) for e in (w, dd))
    inner = prep(fpoly.buffer(-R))
    cells = {(i, j) for i in range(lo, ux + 1) for j in range(lo, uy + 1) if inner.covers(Point((i + 0.5) * G, (j + 0.5) * G))}
    seeds = [(m[0] * s['seed'][0] + m[1] * s['seed'][1] + m[4], m[2] * s['seed'][0] + m[3] * s['seed'][1] + m[5])
             for s in d.door_strips(room)]
    return {'cells': cells, 'wd': (w, dd), 'lo': lo, 'ux': ux, 'uy': uy, 'seeds': seeds, 'm': m}


def reach_without(nc, st, ids):
    """per door the reached cells with objects `ids` taken out (an upper bound of the reach after any move of them)"""
    lo, ux, uy = nc['lo'], nc['ux'], nc['uy']
    rects = [d._inflate(st['bbm'][i], R) for i, p in st['r4'].items()
             if p is None and not d.covering(st['cur'][i], None) and i not in ids]
    grid = (nc['cells'] - set().union(*(d._cells(x, lo, ux, uy) for x in rects)), rects, nc['wd'], lo, ux, uy)
    return [cells if seeded else set() for cells, seeded in (d._reach(grid, s) for s in nc['seeds'])]


def _bfs01(cells, free, src, goal):
    """fewest blocked grid cells to cross from src to goal (4-connected, inside the eroded room): (cost, path)"""
    dist, prev, dq = {}, {}, collections.deque()
    for c in src:
        dist[c] = 0 if c in free else 1
        dq.append(c)
    while dq:
        c = dq.popleft()
        if c in goal:
            path = [c]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return dist[c], path[::-1]
        for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)):
            if n in cells:
                w = dist[c] + (n not in free)
                if w < dist.get(n, 1e9):
                    dist[n], prev[n] = w, c
                    (dq.appendleft if n in free else dq.append)(n)
    return 1000, []


def potential(nc, st, task):
    """accept()'s A3 on the state's AABBs / R4 parents (no snap_inside), every door an entry: (key, far, blockers).
    key = (A3 proxy: per door the task objects not touching its reached region, +1 entry blocked or too few cells,
    and an unseeded door counts every task object; obstruction: per door and far task object the fewest blocked
    cells between the reached region (or the seed cell) and the object's touch cells). blockers = floor roots whose
    inflated AABB covers a blocked cell of those cheapest paths, in path order."""
    lo, ux, uy = nc['lo'], nc['ux'], nc['uy']
    rects = {i: d._inflate(st['bbm'][i], R) for i, p in st['r4'].items() if p is None and not d.covering(st['cur'][i], None)}
    free = nc['cells'] - set().union(*(d._cells(x, lo, ux, uy) for x in rects.values()))
    grid = (free, list(rects.values()), nc['wd'], lo, ux, uy)

    def root(i):
        seen = {i}
        while st['r4'][i] and st['r4'][i] not in seen:          # admit._floor_root on R4 parents
            i = st['r4'][i]
            seen.add(i)
        return i
    near = {t: d._cells(d._inflate(st['bbm'][root(t)], R + G), lo, ux, uy) for t in task}
    a3 = obst = 0
    far_all, path = set(), []
    for s in nc['seeds']:
        cells, seeded = d._reach(grid, s)
        far = [t for t in task if not (seeded and cells & near[t])]
        far_all.update(far)
        a3 += len(far) + (not seeded) + (seeded and len(cells) < d.MIN_REACHABLE_CELLS)
        src = cells if seeded else {(min(max(int(s[0] / G), lo), ux), min(max(int(s[1] / G), lo), uy))}
        for t in far:
            c, p = _bfs01(nc['cells'], free, src, near[t] & nc['cells'])
            obst += c
            path += [q for q in p if q not in free]
    centre = lambda q: ((q[0] + 0.5) * G, (q[1] + 0.5) * G)
    blockers = list(dict.fromkeys(i for q in path for i, x in rects.items() if d._inside(centre(q), x)))
    return (a3, obst), far_all, blockers


# ------------------------------------------------------------------ the stage

def repair(room, cats, deadline, relocate=True, push=True):
    """room = predicted room with the S2-F layout; returns (layout, info). Pure: room is not modified."""
    ctx = r._context(room, r.door_keepouts(room), ())
    state = r._state(ctx, {i: (0, 0) for i in ctx['ids']})
    layout = lambda st: [st['cur'][i] for i in ctx['ids']]
    nc = nav_ctx(room)
    task = [o['id'] for o in room['objects'] if o['category'] in cats]
    key, far, blockers = potential(nc, state, task)
    key0, moves = key, []
    pairs = C.pairs(room['objects'])
    comps = collections.defaultdict(list)
    for p in pairs:
        comps[p[2]].append(p[1])
    pair_of = {p[1]: p for p in pairs}
    edges = S._edges(room['boundary'])
    env_p = (ctx['strict'] or {}).get('env_p') or prep(Polygon(room['boundary']).buffer(r.ENVELOPE_TOL))
    bx0, by0, bx1, by1 = Polygon(room['boundary']).bounds
    theta = d.frame(room['boundary'])[0]
    stats = {'evals': 0, 'accepts': 0, 'timed_out': False, 'rise': collections.Counter(), 'pair_reject': 0,
             'touch_fail': 0, 'key_fail': 0}

    def units(x):
        """move options for floor root x: its own unit (an anchor with its held companions), then for a companion
        its anchor's group"""
        own = [x] + comps.get(x, [])
        return [own] + ([[pair_of[x][2]] + comps[pair_of[x][2]]] if x in pair_of else [])

    def try_move(st, us, dx, dy):
        mm = (round(dx * 1000), round(dy * 1000))
        if mm == (0, 0) or any(u in ctx['pinned'] for u in us):
            return None
        mv = {j: mm for u in us for j in ctx['members'][u]}
        st2 = r.audit_delta(ctx, st, mv)
        stats['evals'] += 1
        rise = [k for k, a, b in zip(r.COMPONENTS, st['v'], st2['v']) if b > a
                and not (k == 'held_broken' and len(us) == 1 and us[0] in pair_of)]  # S2-F (b): a companion's relation
        if rise:                                                                     # outranks its wall contact
            stats['rise'][rise[0]] += 1
            return None
        if not all(C.holds(p, st2['cur']) for p in pairs if p[1] in mv or p[2] in mv):
            stats['pair_reject'] += 1
            return None
        return st2

    def candidates(st, lead):
        o = st['cur'][lead]
        rec = S._rec(o)
        x, y = o['pos'][:2]
        out = {}
        for dx, dy in S._root_candidates(rec, edges):
            out.setdefault((round(dx * 1000), round(dy * 1000)), (dx, dy))
        for i in range(int((bx1 - bx0) / GRID_M) + 1):
            for j in range(int((by1 - by0) / GRID_M) + 1):
                dx, dy = bx0 + (i + 0.5) * GRID_M - x, by0 + (j + 0.5) * GRID_M - y
                out.setdefault((round(dx * 1000), round(dy * 1000)), (dx, dy))
        cs = sorted((v for k, v in out.items() if k != (0, 0)), key=lambda c: math.hypot(*c))
        return [c for c in cs if env_p.covers(translate(rec['poly'], *c))][:MAX_CANDS]

    def take(st2, res, step, us, dx, dy, obj):
        nonlocal state, key, far, blockers
        state, (key, far, blockers) = st2, res
        stats['accepts'] += 1
        moves.append({'step': step, 'object': obj, 'unit': us, 'd': [round(dx, 3), round(dy, 3)], 'key': list(key)})

    def r4root(i):
        seen = {i}
        while state['r4'][i] and state['r4'][i] not in seen:
            i = state['r4'][i]
            seen.add(i)
        return i

    def relocate_any(ids, step, need_touch):
        """first (object, unit, candidate) in order whose move makes the key fall; True if one was taken. With
        need_touch the object must touch every door's reached region; candidates whose inflated footprint misses the
        reach computed without the unit (a superset of any reach after the move) are skipped before the audit."""
        m = nc['m']
        for t in ids:
            for us in units(ctx['chain'][t][-1]):
                cands = candidates(state, us[0])
                rt = r4root(t)
                if need_touch and rt in [j for u in us for j in ctx['members'][u]]:
                    wo = reach_without(nc, state, {j for u in us for j in ctx['members'][u]})
                    bb = state['bbm'][rt]
                    def can_touch(c):
                        fx, fy = m[0] * c[0] + m[1] * c[1], m[2] * c[0] + m[3] * c[1]
                        near = d._cells(d._inflate((bb[0] + fx, bb[1] + fy, bb[2] + fx, bb[3] + fy), R + G),
                                        nc['lo'], nc['ux'], nc['uy'])
                        return all(near & w for w in wo)
                    cands = [c for c in cands if can_touch(c)]
                elif not need_touch:
                    cands = cands[:MAX_BLOCKER_CANDS]
                for dx, dy in cands:
                    if time.monotonic() > deadline:
                        return False
                    st2 = try_move(state, us, dx, dy)
                    if st2 is None:
                        continue
                    res = potential(nc, st2, task)
                    if need_touch and t in res[1]:
                        stats['touch_fail'] += 1
                    elif res[0] >= key:
                        stats['key_fail'] += 1
                    else:
                        take(st2, res, step, us, dx, dy, t)
                        return True
        return False

    def push_any():
        """labbuilder_nav_repair moves of the path blockers, then of the frontier: first one making the key fall"""
        front = [b for fr, _, _ in L.grid_blockers(room, layout(state))[0] for b in fr]
        for b in dict.fromkeys(blockers + front):
            for us in units(b):
                for step in PUSH_STEPS:
                    for ux_, uy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if stats['push_evals'] >= PUSH_EVALS or time.monotonic() > deadline:
                            return False
                        dx, dy = r._rot(ux_ * step * G, uy_ * step * G, theta)
                        st2 = try_move(state, us, dx, dy)
                        if st2 is None:
                            continue
                        stats['push_evals'] += 1
                        res = potential(nc, st2, task)
                        if res[0] < key:
                            take(st2, res, 'push', us, dx, dy, b)
                            return True
        return False

    # 1. relocate the unreachable task objects themselves (each must then touch every door's reached region);
    # 2. else push blockers; 2b (BLOCKER_RELOCATE) else relocate a path blocker; repeat while the key falls
    stats['push_evals'] = 0
    while key[0] and time.monotonic() < deadline:
        if relocate and relocate_any([t for t in task if t in far], 'relocate', True):
            continue
        if push and push_any():
            continue
        if not (BLOCKER_RELOCATE and push and relocate_any([b for b in blockers if b in ctx['raw']], 'relocate_blocker', False)):
            break
    stats['timed_out'] = time.monotonic() > deadline
    final = layout(state)
    verdict = task_accept(room, final, cats)
    full = {**room, 'objects': final}
    after = {'task': verdict['accepted'], 'checker': g.geometry(full)['geometry_success'], 'strict': d.accept(room, final)['accepted']}
    before = {'checker': g.geometry(room)['geometry_success'], 'strict': d.accept(room, room['objects'])['accepted']}
    kept = bool(moves) and after['task'] and after['checker'] >= before['checker'] and after['strict'] >= before['strict'] \
        and S._invariant(room['objects'], final)
    return (final if kept else room['objects']), {'kept': kept, 'moves': moves, 'key_before': list(key0), 'key_after': list(key),
                                                  'reasons_after': verdict['reasons'],
                                                  'proxy_disagrees': (key[0] == 0) != verdict['accepted'] and not set(verdict['reasons']) - A3,
                                                  'pairs': len(pairs), **stats}


# ------------------------------------------------------------------ rows / run

def work(task):
    room_gt, request, row = task
    t0 = time.monotonic()
    cats = T.task_cats(room_gt)
    out = {'uid': row['uid'], 'n': row['objects'], 'cats': cats, 'complete': row['complete']}
    if not row['complete']:
        return out | {'task': [False, False], 'checker': [False, False], 'strict': [False, False], 'processed': False,
                      'changed': False}, None
    pred = r.predicted_room(room_gt, request['messages'], row['s1_layout'])
    v = task_accept(pred, pred['objects'], cats)
    e = S._eval(pred, pred['objects'])
    out |= {'reasons_before': v['reasons'], 'a3_only': v['reasons'] == ['A3_UNREACHABLE_REQUIRED']}
    lay, info = pred['objects'], None
    if not v['accepted'] and set(v['reasons']) & A3:
        lay, info = repair(pred, cats, t0 + TIME_LIMIT)
    e2 = S._eval(pred, lay) if info else e
    v2 = task_accept(pred, lay, cats) if info else v
    disp = [math.dist(a['pos'][:2], b['pos'][:2]) for a, b in zip(pred['objects'], lay)]
    out |= {'processed': info is not None, 'changed': any(x > 0 for x in disp), 'task': [v['accepted'], v2['accepted']],
            'checker': [e['checker'], e2['checker']], 'strict': [e['strict'], e2['strict']],
            'invariant_ok': S._invariant(pred['objects'], lay), 'retention': C.retention(pred['objects'], lay),
            'disp': [x for x in disp if x > 0], 'runtime_s': time.monotonic() - t0, 'info': info}
    return out, lay


def run(a):
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rooms = {x['uid']: x for x in map(json.loads, open(data / 'rooms.jsonl'))}
    reqs = {x['uid']: x for x in map(json.loads, open(data / 'placement' / 'requests.jsonl'))}
    rows = [json.loads(l) for l in open(a.final)]
    t0 = time.monotonic()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(work, [(rooms[x['uid']], reqs[x['uid']], x) for x in rows], chunksize=2))
    wall = time.monotonic() - t0
    (out / 'rows.jsonl').write_text(''.join(json.dumps(x) + '\n' for x, _ in res))
    with open(out / 'final-s2ftask.jsonl', 'w') as f:
        for row, (x, lay) in zip(rows, res):
            if x.get('changed'):
                row = {**row, 's1_layout': lay, 's1': {**row['s1'], 'geometry_success_s1': x['checker'][1], 'strict_ok_s1': x['strict'][1]}}
            f.write(json.dumps(row) + '\n')
    summary = summarize([x for x, _ in res], wall, a)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    text = report(summary, a)
    (out / 'report.md').write_text(text)
    print(text)


def agg(rs):
    n = len(rs)
    ch = [x for x in rs if x['changed']]
    pr = [x for x in rs if x['processed']]
    disp = [v for x in ch for v in x['disp']]
    rate = lambda k, i: sum(x[k][i] for x in rs) / n if n else None
    return {'rooms': n, 'task_before': rate('task', 0), 'task_after': rate('task', 1),
            'checker_before': rate('checker', 0), 'checker_after': rate('checker', 1),
            'strict_before': rate('strict', 0), 'strict_after': rate('strict', 1),
            'checker_regressions': sum(x['checker'][0] and not x['checker'][1] for x in rs),
            'strict_regressions': sum(x['strict'][0] and not x['strict'][1] for x in rs),
            'processed': len(pr), 'changed': len(ch), 'fixed': sum(x['task'][1] and not x['task'][0] for x in rs),
            'a3_only': sum(x.get('a3_only', False) for x in rs),
            'a3_only_fixed': sum(x.get('a3_only', False) and x['task'][1] for x in rs),
            'companions_kept_changed': [sum(x['retention'][0] for x in ch), sum(x['retention'][1] for x in ch)],
            'objects_moved': len(disp), 'objects_in_changed': sum(x['n'] for x in ch),
            'median_disp_m': statistics.median(disp) if disp else None, 'max_disp_m': max(disp) if disp else None,
            'moves_relocate': sum(m['step'] == 'relocate' for x in ch for m in x['info']['moves']),
            'moves_push': sum(m['step'] == 'push' for x in ch for m in x['info']['moves']),
            'moves_relocate_blocker': sum(m['step'] == 'relocate_blocker' for x in ch for m in x['info']['moves']),
            'runtime_mean_s_processed': statistics.mean(x['runtime_s'] for x in pr) if pr else None,
            'runtime_max_s_processed': max((x['runtime_s'] for x in pr), default=None),
            'timed_out': sum(x['info']['timed_out'] for x in pr),
            'invariant_violations': sum(not x.get('invariant_ok', True) for x in rs)}


def summarize(rs, wall, a):
    return {'final': a.final, 'data': a.data, 'wall_s': wall, 'workers': a.workers,
            'params': {'grid_m': GRID_M, 'max_cands': MAX_CANDS, 'push_steps_m': [k * G for k in PUSH_STEPS],
                       'push_evals': PUSH_EVALS, 'blocker_relocate': BLOCKER_RELOCATE, 'max_blocker_cands': MAX_BLOCKER_CANDS, 'time_limit_s': TIME_LIMIT},
            'all': agg(rs), 'bins': {b: agg([x for x in rs if lo <= x['n'] <= hi]) for b, lo, hi in BINS},
            'reasons_after_not_fixed': collections.Counter(c for x in rs if x['processed'] and not x['task'][1]
                                                           for c in (x['info']['reasons_after'])).most_common()}


def report(s, a):
    f = lambda v: '—' if v is None else f'{v:.3f}'
    lines = [f'# S2-F-task on {a.final}', '', '| bin | rooms | task-scoped admission S2F → +task | checker | all-cat strict | '
             'processed / changed / fixed | A3-only fixed | companions kept (changed rooms) | objects moved / in changed rooms | '
             'median disp m | runtime mean/max s (processed) |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, x in [('all', s['all'])] + list(s['bins'].items()):
        if not x['rooms']:
            continue
        ck = x['companions_kept_changed']
        lines.append(f"| {name} | {x['rooms']} | {f(x['task_before'])} → {f(x['task_after'])} | {f(x['checker_before'])} → "
                     f"{f(x['checker_after'])} | {f(x['strict_before'])} → {f(x['strict_after'])} | {x['processed']} / "
                     f"{x['changed']} / {x['fixed']} | {x['a3_only_fixed']}/{x['a3_only']} | {ck[0]}/{ck[1]} | "
                     f"{x['objects_moved']} / {x['objects_in_changed']} | {f(x['median_disp_m'])} | "
                     f"{f(x['runtime_mean_s_processed'])} / {f(x['runtime_max_s_processed'])} |")
    x = s['all']
    lines += ['', f"checker regressions {x['checker_regressions']}, strict regressions {x['strict_regressions']}, invariant "
              f"violations {x['invariant_violations']}, timed out {x['timed_out']}; moves: relocate {x['moves_relocate']}, push "
              f"{x['moves_push']}, relocate blocker {x['moves_relocate_blocker']}; wall {s['wall_s']:.0f} s on {s['workers']} workers", '',
              f"not fixed, task-scoped reasons after: {s['reasons_after_not_fixed']}", '', f"params: {s['params']}"]
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ self-check

def selfcheck():
    base = {'uid': 'toy', 'boundary': [[0, 0], [4, 0], [4, 3], [0, 3]], 'boundary_type': 'polygon', 'height': 2.8,
            'constraints': [], 'fixed': [{'id': 'door', 'category': 'door', 'size': [0.9, 0.1, 2.1], 'pos': [3.4, 0.05, 0], 'yaw': 0.0}]}
    desk = {'id': 'desk', 'category': 'desk', 'size': [1.2, 0.6, 0.75], 'pos': [0.6, 2.7, 0], 'yaw': 0.0, 'parent': None}
    chair = {'id': 'chair', 'category': 'desk chair', 'size': [0.5, 0.5, 0.9], 'pos': [0.6, 2.1, 0], 'yaw': math.pi / 2, 'parent': None}
    lamp = {'id': 'lamp', 'category': 'desk lamp', 'size': [0.2, 0.2, 0.4], 'pos': [0.3, 2.7, 0.75], 'yaw': 0.0, 'parent': 'desk'}
    ward = {'id': 'ward', 'category': 'wardrobe', 'size': [0.6, 3.0, 2.0], 'pos': [1.6, 1.5, 0], 'yaw': 0.0, 'parent': None}
    cats = {'desk'}
    # toy 1: the wardrobe walls the desk (+ its chair and lamp) off from the door
    room = {**base, 'objects': [desk, chair, lamp, ward]}
    v0 = task_accept(room, room['objects'], cats)
    assert not v0['accepted'] and v0['reasons'] == ['A3_UNREACHABLE_REQUIRED'], v0['reasons']
    assert g.geometry(room)['geometry_success'] and C.pairs(room['objects']) == [('adj', 'chair', 'desk')]
    lay, info = repair(room, cats, time.monotonic() + 60)
    by = {o['id']: o for o in lay}
    assert info['kept'] and task_accept(room, lay, cats)['accepted'], info
    assert [m['step'] for m in info['moves']] == ['relocate'] and by['ward']['pos'] == ward['pos'], info['moves']
    assert by['desk']['pos'][0] > 1.9 and C.retention(room['objects'], lay) == (1, 1), by   # group moved past the wardrobe
    assert abs((by['lamp']['pos'][0] - by['desk']['pos'][0]) - (lamp['pos'][0] - desk['pos'][0])) < 1e-6   # subtree moved
    assert S._invariant(room['objects'], lay) and g.geometry({**room, 'objects': lay})['geometry_success']
    print('selfcheck ok toy1 (relocate)', {i: [round(a - b, 3) for a, b in zip(by[i]['pos'][:2], o['pos'][:2])]
                                           for i, o in ((o['id'], o) for o in room['objects'])}, info['moves'])
    # toy 2 (6 x 4 m): the wardrobe leaves a 1.5 m gap that an armchair closes; with relocation off a push > 1 m opens it
    door2 = {**base['fixed'][0], 'pos': [5.4, 0.05, 0]}
    room = {**base, 'boundary': [[0, 0], [6, 0], [6, 4], [0, 4]], 'fixed': [door2],
            'objects': [{**desk, 'pos': [0.6, 3.7, 0]}, {**lamp, 'pos': [0.3, 3.7, 0.75]},
                        {**ward, 'size': [0.6, 2.5, 2.0], 'pos': [2.3, 2.75, 0]},
                        {'id': 'arm', 'category': 'armchair', 'size': [0.8, 0.8, 0.8], 'pos': [2.3, 0.75, 0], 'yaw': 0.0, 'parent': None}]}
    assert task_accept(room, room['objects'], cats)['reasons'] == ['A3_UNREACHABLE_REQUIRED']
    lay, info = repair(room, cats, time.monotonic() + 60, relocate=False)
    by = {o['id']: o for o in lay}
    assert info['kept'] and {m['step'] for m in info['moves']} == {'push'} and by['desk']['pos'] == [0.6, 3.7, 0], info
    assert max(abs(c) for m in info['moves'] for c in m['d']) > 4 * G - 1e-9, info['moves']        # beyond labbuilder's steps
    print('selfcheck ok toy2 (push)', info['moves'])
    # toy 3: the desk + chair group cannot fit the reachable 1.5 m strip and the wardrobe cannot move: unchanged
    room = {**base, 'objects': [{**desk, 'size': [1.8, 0.8, 0.75], 'pos': [0.9, 2.6, 0]}, {**chair, 'pos': [0.9, 1.95, 0]},
                                {**lamp, 'pos': [0.3, 2.6, 0.75]}, {**ward, 'pos': [2.2, 1.5, 0]}]}
    assert task_accept(room, room['objects'], cats)['reasons'] == ['A3_UNREACHABLE_REQUIRED'] and len(C.pairs(room['objects'])) == 1
    lay, info = repair(room, cats, time.monotonic() + 60)
    assert not info['kept'] and lay == room['objects'], info
    print('selfcheck ok toy3 (unfixable -> unchanged)', info['reasons_after'], info['evals'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selfcheck', action='store_true'); ap.add_argument('--run', action='store_true')
    ap.add_argument('--final'); ap.add_argument('--data'); ap.add_argument('--out'); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if a.run:
        return run(a)
    ap.print_help()


if __name__ == '__main__':
    main()
