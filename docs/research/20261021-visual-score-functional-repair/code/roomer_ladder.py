"""Probe roomer-ladder (2026-10-21): Roomer (arXiv 2608.01973, Table A11 + Eq. 3) transferred onto the delivered S1.

Deterministic, XY only, contract-safe: ids / order / category / size / yaw / parent / z frozen, whole support
subtrees move (S1's _unit_move). No planner, no SCALE/INSERT/DELETE/REPLACE/ROTATE.
  candidates  s1      = S1's own pushes (SAT / co-located / half-half / boundary push)
              ladder  = S1's pushes first (the "seed"), then Roomer's MOVE fallback: distances 0.05..1.20 m x 8
                        directions x orthogonal offsets {0, +-0.1}, then a room-wide 0.2 m grid (nearest GRID_CAP)
  accept      pareto  = S1's rule (no audit component rises, one falls or phi falls): partial progress allowed
              roomer  = target issue fully resolved + no new hard-violation KEY (collision pair / oob id /
                        fixed-blocking pair / support id) + no audit component rises (Eq. 3 adapted)
  schedule    Roomer: one unblocked issue per round (S1 severity order), blocked set cleared on every commit
  start       s1 (the delivered S1 output, s1_layout) or raw (raw_layout)
Final guard as S1: invariant, checker_worse, downstream_worse -> roll back to the start layout.

PYTHONPATH=REPO python3 roomer_ladder.py --out DIR --start s1 --cand ladder --accept roomer [--limit N --seed 0]
"""
import argparse
import collections
import copy
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import statistics
import time

from shapely.geometry import Point

from fastfill.comparison import downstream as d
from fastfill.comparison import geometry as g
from fastfill.comparison import output_repair as R

BASE = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013')
ROOMS = BASE / 'visual-final-selected-002/inputs/eval-data-v1/rooms.jsonl'
REQUESTS = BASE / 'visual-final-selected-002/inputs/eval-data-v1/placement/requests.jsonl'
S1_ROWS = BASE / 'a026-inputs/s1-022/placement-arm/placement-geometry-s1.jsonl'
LOCAL_D = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20)        # Roomer Table A11, MOVE
DIRS = [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]
ORTHO = (0.0, 0.1, -0.1)
GRID, GRID_CAP = 0.2, 400          # ponytail: Roomer caps at 1,000; 400 nearest, raise if budget_exhausted dominates
HARD = ('oob', 'support_fail', 'ceiling', 'fixed_blocking', 'main', 'severe')


def keys(st):
    """Stable hard-violation keys (Roomer App. F): collision pairs, oob ids, fixed-blocking pairs, support ids."""
    ks = {('main',) + k for k, (kind, _, _) in st['pair'].items() if kind == 'main'}
    for i, r in st['row'].items():
        if r[0] is not None:
            ks.add(('oob', i))
        if r[1]:
            ks.add(('support', i))
        ks.update(('fixed', i, f) for f, _, w in r[4] if not w)
    return ks


def accept_roomer(before, after, kind, ids):
    return not R._still(after, kind, ids) and all(y <= x for x, y in zip(before['v'], after['v'])) \
        and keys(after) <= keys(before)


def accept_pareto(before, after, kind, ids):
    return R._better(before, after)


def grid_points(ctx):
    poly = ctx['poly']
    x0, y0, x1, y1 = poly.bounds
    pts = [(x0 + GRID * i, y0 + GRID * j) for i in range(int((x1 - x0) / GRID) + 1)
           for j in range(int((y1 - y0) / GRID) + 1)]
    return [p for p in pts if poly.contains(Point(p))]


def fallback(ctx, st, unit, grid):
    x0, y0 = st['cur'][unit]['pos'][:2]
    for dist in LOCAL_D:
        for u in DIRS:
            for off in ORTHO:
                yield R._unit_move(ctx, st, unit, u[0] * dist - u[1] * off, u[1] * dist + u[0] * off)
    for gx, gy in sorted(grid, key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2)[:GRID_CAP]:
        yield R._unit_move(ctx, st, unit, gx - x0, gy - y0)


def units(ctx, kind, ids):
    if kind == 'pair':
        return list(dict.fromkeys(u for u, _, _ in R._movers(ctx, *ids)))
    if kind == 'fixed':
        return [ctx['chain'][ids[0]][-1]]
    if kind == 'oob':
        return list(dict.fromkeys((ids[0], ctx['chain'][ids[0]][-1])))
    return []


def candidates(ctx, st, kind, ids, cand, grid):
    yield from R._candidates(ctx, st, kind, ids)
    if cand == 'ladder':
        for u in units(ctx, kind, ids):
            if u not in ctx['pinned']:
                yield from fallback(ctx, st, u, grid)


def solve(pred, cand, accept, max_trials, time_limit):
    started = time.monotonic()
    keepouts = R.door_keepouts(pred)
    ctx = R._context(pred, keepouts, ())
    before = R.audit(pred, keepouts, ctx['held'])
    st = R._state(ctx, {i: (0, 0) for i in ctx['ids']})
    ok = accept_roomer if accept == 'roomer' else accept_pareto
    grid = grid_points(ctx) if cand == 'ladder' else []
    blocked, trials, steps, timed_out, rounds = set(), 0, [], False, 0
    while trials < max_trials and not timed_out:
        probs = [(ids, kind) for _, ids, kind in R._problems(ctx, st)
                 if R._still(st, kind, ids) and (kind, ids) not in blocked]
        if not probs:
            break
        ids, kind = probs[0]
        rounds += 1
        seen, accepted = set(), False
        for mv in candidates(ctx, st, kind, ids, cand, grid):
            if not mv:
                continue
            key = tuple(sorted(mv.items()))
            if key in seen:
                continue
            seen.add(key)
            if trials >= max_trials:
                break
            if time.monotonic() - started > time_limit:
                timed_out = True
                break
            trials += 1
            after = R.audit_delta(ctx, st, mv)
            if ok(st, after, kind, ids):
                st, accepted = after, True
                steps.append({'problem': [kind, *ids], 'move_mm': {i: list(m) for i, m in mv.items()},
                              'trial': trials})
                break
        if accepted:
            blocked.clear()
        else:
            blocked.add((kind, ids))
    objects = [copy.deepcopy(st['cur'][i]) if st['shift'][i] != (0, 0) else ctx['raw'][i] for i in ctx['ids']]
    final_room = {**pred, 'objects': objects}
    final = R.audit(final_room, keepouts, ctx['held'])
    rolled, verdict = None, {'start': d.accept(pred, pred['objects'])}
    if not R._invariant(pred['objects'], objects, st['shift']):
        rolled = 'guard_violation'
    elif any(b > a for a, b in zip(before['v'], final['v'])):
        rolled = 'checker_worse'
    elif steps:
        verdict['final'] = d.accept(final_room, objects)
        if any(verdict['start'][k] and not verdict['final'][k] for k in R.DOWNSTREAM_FLAGS):
            rolled = 'downstream_worse'
    if rolled or not steps:
        final_room, final, verdict['final'] = pred, before, verdict['start']
    vec = lambda s: dict(zip(R.COMPONENTS, s['v'])) | {'phi': s['phi']}
    return {'room': final_room, 'before': vec(before), 'final': vec(final), 'steps': steps, 'rounds': rounds,
            'trials': trials, 'blocked': len(blocked), 'timed_out': timed_out, 'budget_exhausted': trials >= max_trials,
            'rolled_back': rolled, 'runtime_s': time.monotonic() - started,
            'strict_start': verdict['start']['accepted'], 'strict_final': verdict['final']['accepted'],
            'displacement': [math.dist(a['pos'][:2], b['pos'][:2]) for a, b in zip(pred['objects'], final_room['objects'])]}


def one(task):
    row, room, request, args = task
    layout = row['s1_layout'] if args['start'] == 's1' else row['raw_layout']
    pred = R.predicted_room(room, request['messages'], layout)
    res = solve(pred, args['cand'], args['accept'], args['max_trials'], args['time_limit'])
    geo0, geo1 = g.geometry(pred), g.geometry(res['room'])
    kinds = sorted(k for k in HARD if res['before'][k])
    return {'uid': row['uid'], 'source': row['source'], 'objects': row['objects'], 'stratum': R._stratum(row['objects']),
            'doors': len(d.door_strips(pred)), 'rectangle': d.is_rectangle(R.Polygon(pred['boundary'])),
            'start_kinds': kinds, 'geometry_success_start': geo0['geometry_success'],
            'geometry_success_final': geo1['geometry_success'], 'geometry_final': geo1,
            **{k: res[k] for k in ('before', 'final', 'rounds', 'trials', 'blocked', 'timed_out', 'budget_exhausted',
                                   'rolled_back', 'runtime_s', 'strict_start', 'strict_final')},
            'steps': len(res['steps']), 'step_list': res['steps'],
            'moved_objects': sum(x > 0 for x in res['displacement']),
            'moved_median_m': statistics.median([x for x in res['displacement'] if x > 0] or [0.0]),
            'moved_max_m': max(res['displacement'], default=0.0), 'final_layout': res['room']['objects']}


def agg(rows):
    n = len(rows)
    f = lambda k: sum(bool(r[k]) for r in rows)
    out = {'rooms': n, 'checker_fixed': f('geometry_success_final'), 'strict_fixed': f('strict_final'),
           'strict_start': f('strict_start'), 'timed_out': f('timed_out'), 'budget_exhausted': f('budget_exhausted'),
           'rolled_back': dict(collections.Counter(r['rolled_back'] for r in rows if r['rolled_back'])),
           'runtime_mean_s': statistics.mean(r['runtime_s'] for r in rows) if n else None,
           'runtime_max_s': max((r['runtime_s'] for r in rows), default=None),
           'trials_mean': statistics.mean(r['trials'] for r in rows) if n else None,
           'steps_mean': statistics.mean(r['steps'] for r in rows) if n else None,
           'residual_main_mean': statistics.mean(r['final']['main'] for r in rows) if n else None,
           'start_main_mean': statistics.mean(r['before']['main'] for r in rows) if n else None}
    out['checker_fixed_rate'] = out['checker_fixed'] / n if n else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--start', choices=('s1', 'raw'), default='s1')
    ap.add_argument('--cand', choices=('s1', 'ladder'), default='ladder')
    ap.add_argument('--accept', choices=('pareto', 'roomer'), default='roomer')
    ap.add_argument('--limit', type=int, default=0, help='per-stratum cap (0 = every failing room)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--max-trials', type=int, default=20000)
    ap.add_argument('--time-limit', type=float, default=60.0)
    ap.add_argument('--uids', type=Path, help='JSON list of uids (overrides --limit)')
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    key = 'geometry_success_s1' if a.start == 's1' else 'geometry_success_raw'
    rows = [json.loads(l) for l in open(S1_ROWS)]
    rows = [r for r in rows if r['complete'] and not r['s1'][key]]
    if a.uids:
        want = set(json.load(open(a.uids)))
        rows = [r for r in rows if r['uid'] in want]
    elif a.limit:
        rng, by = random.Random(a.seed), collections.defaultdict(list)
        for r in rows:
            by[r['s1']['stratum']].append(r)
        rows = [r for k in sorted(by) for r in rng.sample(by[k], min(a.limit, len(by[k])))]
    uids = {r['uid'] for r in rows}
    rooms = {x['uid']: x for x in map(json.loads, open(ROOMS)) if x['uid'] in uids}
    reqs = {x['uid']: x for x in map(json.loads, open(REQUESTS)) if x['uid'] in uids}
    args = {'start': a.start, 'cand': a.cand, 'accept': a.accept, 'max_trials': a.max_trials, 'time_limit': a.time_limit}
    tasks = [(r, rooms[r['uid']], reqs[r['uid']], args) for r in rows]
    t0 = time.monotonic()
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(one, tasks, chunksize=2))
    a.out.mkdir(parents=True)
    with open(a.out / 'rows.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r, allow_nan=False) + '\n')
    strata = [R._stratum(lo) for lo, _ in R.STRATA]
    summary = {'probe': 'roomer-ladder', 'args': {**args, 'limit': a.limit, 'seed': a.seed, 'grid_m': GRID,
                                                   'grid_cap': GRID_CAP, 'local_distances_m': LOCAL_D},
               'wall_s': time.monotonic() - t0, 'all': agg(results),
               'by_stratum': {k: agg([r for r in results if r['stratum'] == k]) for k in strata},
               'by_kinds': {'+'.join(k): agg([r for r in results if tuple(r['start_kinds']) == k])
                            for k in sorted({tuple(r['start_kinds']) for r in results})},
               'by_doors': {k: agg([r for r in results if (r['doors'] > 0) == (k == 'doors')]) for k in ('no_doors', 'doors')}}
    json.dump(summary, open(a.out / 'summary.json', 'w'), indent=1)
    print(json.dumps({'rooms': len(results), 'all': summary['all'],
                      'by_stratum': {k: (v['rooms'], v['checker_fixed']) for k, v in summary['by_stratum'].items()},
                      'wall_s': round(summary['wall_s'], 1)}))


if __name__ == '__main__':
    main()
