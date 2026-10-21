"""Probe: LabBuilder-style navigation-aware refinement (Alg. 2, operator Upsilon) as a stage after S1.

For every S1 row whose only residual strict codes are reachability ones (A3_UNREACHABLE_REQUIRED /
A3_ENTRY_BLOCKED / A3_TOO_FEW_REACHABLE), iterate <= T rounds: find the parentless solid objects on the frontier of
the door-reached region (blame rule: the paper leaves it unspecified), try XY translations of their whole support
subtree along +-x/+-y of the P2 frame by k * NAV_GRID_M, accept a move iff no S1 audit component rises (checker,
strict, held relations, A4) and the A3 violation count falls (LabBuilder's monotone rule, v strictly decreasing).
Yaw/z/parent/size frozen as in S1. Pure CPU, no model, no API.

python labbuilder_nav_repair.py --s1 S1.jsonl --rooms rooms.jsonl --requests requests.jsonl --out DIR [--workers 8]
"""
import argparse, collections, json, math, statistics, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from shapely.affinity import affine_transform
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely.prepared import prep

from fastfill.comparison import downstream as d
from fastfill.comparison import output_repair as r

A3_FIX = {'A3_UNREACHABLE_REQUIRED', 'A3_ENTRY_BLOCKED', 'A3_TOO_FEW_REACHABLE'}
STEPS = (1, 2, 3, 4)            # multiples of NAV_GRID_M (0.25 m): smallest first (S1 spirit; paper keeps the largest)
T, MAX_EVALS, TIME_LIMIT = 3, 400, 60.0


def a3_count(verdict):
    return sum(max(1, len(v['ids'])) for v in verdict['violations'] if v['code'] in A3_FIX)


def grid_blockers(room, lay):
    """Per door: (frontier blocker ids of the door-reached region, unreachable-target roots), accept()'s grid."""
    poly = orient(Polygon(room['boundary']))
    by = {o['id']: o for o in lay}
    theta, m = d.frame(room['boundary'])
    bb = {i: d.aabb(o, m) for i, o in by.items()}
    parent = d.r4_parents(by, bb)
    solid = [i for i in by if not d.covering(by[i], parent[i])]
    fpoly = affine_transform(poly, m)
    w, dd = fpoly.bounds[2:]
    lo = math.ceil(d.AGENT_RADIUS_M / d.NAV_GRID_M - 0.5)
    ux, uy = (math.floor((e - d.AGENT_RADIUS_M) / d.NAV_GRID_M - 0.5) for e in (w, dd))
    roots = [i for i in solid if parent[i] is None]
    cells = {i: d._cells(d._inflate(bb[i], d.AGENT_RADIUS_M), lo, ux, uy) for i in roots}
    blocked = set().union(*cells.values()) if cells else set()
    inner = prep(fpoly.buffer(-d.AGENT_RADIUS_M))
    free = {(i, j) for i in range(lo, ux + 1) for j in range(lo, uy + 1)
            if (i, j) not in blocked and inner.covers(Point((i + 0.5) * d.NAV_GRID_M, (j + 0.5) * d.NAV_GRID_M))}
    grid = (free, [d._inflate(bb[i], d.AGENT_RADIUS_M) for i in roots], (w, dd), lo, ux, uy)
    out = []
    for s in d.door_strips(room):
        seed = (m[0] * s['seed'][0] + m[1] * s['seed'][1] + m[4], m[2] * s['seed'][0] + m[3] * s['seed'][1] + m[5])
        reached, seeded = d._reach(grid, seed)
        if not seeded:                                   # entry blocked: whoever covers the seed
            front = [i for i in roots if d._inside(seed, d._inflate(bb[i], d.AGENT_RADIUS_M))]
        else:
            ring = {(i + di, j + dj) for i, j in reached for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))} - reached
            front = [i for i in roots if cells[i] & ring]
        out.append((front, seed, reached))
    centre = lambda i: ((bb[i][0] + bb[i][2]) / 2, (bb[i][1] + bb[i][3]) / 2)
    return out, theta, centre, parent


def nav_repair(room):
    ctx = r._context(room, r.door_keepouts(room), ())
    state = r._state(ctx, {i: (0, 0) for i in ctx['ids']})
    layout = lambda st: [st['cur'][i] for i in ctx['ids']]
    verdict = d.accept(room, layout(state))
    before, count, evals, moves, started = verdict, a3_count(verdict), 0, [], time.monotonic()
    rounds = 0
    while count and rounds < T and evals < MAX_EVALS and time.monotonic() - started < TIME_LIMIT:
        rounds += 1
        progress = False
        doors, theta, centre, parent = grid_blockers(room, layout(state))
        targets = {v['door']: [i for i in v['ids']] for v in verdict['violations'] if v['code'] == 'A3_UNREACHABLE_REQUIRED'}
        for k, (front, seed, reached) in enumerate(doors):
            door_id = verdict['doors'][k]['id']
            if not any(v['door'] == door_id and v['code'] in A3_FIX for v in verdict['violations']):
                continue
            goal = targets.get(door_id) or []
            root = lambda i: ctx['chain'][i][-1]
            goals = [centre(root(i)) for i in goal if root(i) in ctx['raw']] or [seed]
            near = lambda i: min(math.dist(centre(i), gc) for gc in goals)
            for b in sorted(front, key=near):                 # blockers nearest the unreachable target first
                done = False
                for k_step in STEPS:
                    for ux_, uy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        dx, dy = r._rot(ux_ * k_step * d.NAV_GRID_M, uy_ * k_step * d.NAV_GRID_M, theta)
                        mv = r._unit_move(ctx, state, b, dx, dy)
                        if not mv or evals >= MAX_EVALS or time.monotonic() - started > TIME_LIMIT:
                            continue
                        after = r.audit_delta(ctx, state, mv)
                        if any(y > x for x, y in zip(state['v'], after['v'])):
                            continue
                        evals += 1
                        v2 = d.accept(room, layout(after))
                        c2 = a3_count(v2)
                        if c2 < count:
                            state, verdict, count, progress, done = after, v2, c2, True, True
                            moves.append({'object': b, 'members': list(mv), 'step_m': k_step * d.NAV_GRID_M,
                                          'dir': [ux_, uy_], 'a3': c2})
                            break
                    if done:
                        break
                if done:
                    break                                     # re-derive the frontier after a change
            if progress:
                break
        if not progress:
            break
    final = layout(state)
    raw = room['objects']
    assert all(a['id'] == b['id'] and a['yaw'] == b['yaw'] and a['size'] == b['size'] and a.get('parent') == b.get('parent')
               and a['pos'][2] == b['pos'][2] for a, b in zip(raw, final)), 'invariant broken'
    regress = [f for f in r.DOWNSTREAM_FLAGS if before[f] and not verdict[f]]
    disp = [math.dist(a['pos'][:2], b['pos'][:2]) for a, b in zip(raw, final)]
    return {'a3_before': a3_count(before), 'a3_after': count, 'accepted_before': before['accepted'],
            'accepted_after': verdict['accepted'] and not regress, 'regressions': regress,
            'reasons_after': verdict['reasons'], 'rounds': rounds, 'evals': evals, 'moves': moves,
            'moved_objects': sum(x > 0 for x in disp), 'max_disp_m': max(disp) if disp else 0.0,
            'runtime_s': time.monotonic() - started, 'timed_out': time.monotonic() - started > TIME_LIMIT,
            'budget_exhausted': evals >= MAX_EVALS, 'layout': final}


def work(task):
    uid, room, request, row = task
    pred = r.predicted_room(room, request['messages'], row['s1_layout'])
    out = nav_repair(pred)
    return {'uid': uid, 'stratum': row['s1']['stratum'], 'objects': row['objects'], 'doors': row['s1']['doors'],
            'rectangle': row['s1']['rectangle'], 'reasons_s1': row['s1']['downstream_s1']['reasons'], **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s1', required=True); ap.add_argument('--rooms', required=True); ap.add_argument('--requests', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=1); ap.add_argument('--limit', type=int)
    ap.add_argument('--cohort', default='a3only', choices=('a3only', 'a3plus'))
    a = ap.parse_args()
    rooms = {json.loads(l)['uid']: json.loads(l) for l in open(a.rooms)}
    reqs = {json.loads(l)['uid']: json.loads(l) for l in open(a.requests)}
    tasks = []
    for line in open(a.s1):
        row = json.loads(line); s = row.get('s1') or {}
        if not s or not s.get('geometry_success_s1') or s.get('accepted_s1'):
            continue
        rs = set(s['downstream_s1']['reasons'])
        ok = rs <= A3_FIX if a.cohort == 'a3only' else (bool(rs & A3_FIX) and rs <= A3_FIX | {'A4_COLLISION'})
        if ok:
            tasks.append((row['uid'], rooms[row['uid']], reqs[row['uid']], row))
    if a.limit:
        tasks = tasks[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if a.workers == 1:
        rows = [work(t) for t in tasks]
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            rows = list(pool.map(work, tasks, chunksize=2))
    with open(out / 'rows.jsonl', 'w') as f:
        for x in rows:
            f.write(json.dumps(x) + '\n')
    n = len(rows)
    fixed = [x for x in rows if x['accepted_after']]
    agg = lambda xs: {'rooms': len(xs), 'fixed': sum(x['accepted_after'] for x in xs),
                      'a3_reduced_not_fixed': sum((x['a3_after'] < x['a3_before']) and not x['accepted_after'] for x in xs),
                      'fixed_rate': (sum(x['accepted_after'] for x in xs) / len(xs)) if xs else None}
    summary = {'cohort': a.cohort, 'rooms': n, 'fixed': len(fixed), 'fixed_rate': len(fixed) / n if n else None,
               'a3_reduced_not_fixed': sum((x['a3_after'] < x['a3_before']) and not x['accepted_after'] for x in rows),
               'regressions': sum(bool(x['regressions']) for x in rows),
               'by_stratum': {k: agg([x for x in rows if x['stratum'] == k]) for k in sorted({x['stratum'] for x in rows})},
               'by_doors': {'1': agg([x for x in rows if x['doors'] == 1]), '2+': agg([x for x in rows if x['doors'] >= 2])},
               'by_shape': {'rectangle': agg([x for x in rows if x['rectangle']]), 'non_rectangle': agg([x for x in rows if not x['rectangle']])},
               'moves_per_fixed_room': statistics.mean(len(x['moves']) for x in fixed) if fixed else None,
               'max_disp_fixed_median_m': statistics.median(x['max_disp_m'] for x in fixed) if fixed else None,
               'max_disp_fixed_p95_m': sorted(x['max_disp_m'] for x in fixed)[int(0.95 * (len(fixed) - 1))] if fixed else None,
               'runtime_s_mean': statistics.mean(x['runtime_s'] for x in rows) if rows else None,
               'runtime_s_max': max(x['runtime_s'] for x in rows) if rows else None,
               'timed_out': sum(x['timed_out'] for x in rows), 'budget_exhausted': sum(x['budget_exhausted'] for x in rows),
               'wall_s': time.monotonic() - t0, 'params': {'T': T, 'steps_m': [k * d.NAV_GRID_M for k in STEPS],
                                                           'max_evals': MAX_EVALS, 'time_limit_s': TIME_LIMIT},
               'test_rooms_total': 8647, 'strict_gain_upper_bound_on_test': n / 8647, 'strict_gain_measured_on_test': len(fixed) / 8647}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
