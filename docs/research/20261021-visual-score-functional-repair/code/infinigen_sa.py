"""Probe infinigen-sa (2026-10-21): Infinigen Indoors' object solver (Raistrick et al., CVPR 2024, Sec. 3.3 + App. E;
code: example_solver/annealing.py, propose_continous.py, base_indoors.gin) transferred onto the delivered S1 state.

What is transferred (free, local CPU, shapely only):
  acceptance  annealing.metrop_hastings_with_viol: viol' < viol -> accept; viol' > viol -> reject;
              viol' == viol -> Metropolis on the soft loss, log u < -(phi' - phi) / T.
              viol = sum of the checker FAILS components (oob, support_fail, ceiling, fixed_blocking, main, severe)
              [--viol strict adds outside_1mm, keepout_1cm, held_broken, a4_aabb]; phi = S1's overlap-area loss.
  schedule    T = T0 * r**it, r = (T1/T0)**(1/(steps*(1-finetune))), clipped; last `finetune` of the steps at T1
              (annealing.reset). Paper: T0 .25 -> T1 .001; gin: 3 -> .001, finetune_pct .15.
  translate   propose_continous.propose_translate: N(0, sigma), sigma = max(TRANS_MIN .01, TRANS_MULT 8 * T), in the
              unit's DoF plane (XY of the floor, or its seat footprint: S1's _unit_move clamps and keeps the seat).
  rotate      --yaw: propose_rotate, N(0, pi*T) rounded away from zero to multiples of ANGLE_STEP (pi/2 here, not the
              paper's pi/4: keeps integer-degree yaws and wall alignment); the support subtree rotates about the
              unit's centre (children: pos rotated, yaw += angle). Weights translate 1 : rotate .5 (solve.py).
  staging     --stages 3: greedy large / medium / small (App. E.1): active units = floor roots with footprint area
              >= median, then every floor root, then everything; steps split 300:200:50 (base_indoors.gin).
  filtering   move filtering (Sec. 4.1): with p = --p-viol the moved unit is one that takes part in a violation.
  best state  the paper keeps none (checkpoint_best raises); here the best (viol, phi) state is returned, since the
              final S1 guard must see a valid end state. Deviation, recorded.
Everything else is S1's: ids / order / category / size / parent / z frozen (yaw too unless --yaw), whole support
subtrees move, seat clamp, mm positions, final guard (invariant, no audit component above the start = the GT-relation
guard incl. against_wall held relations, no downstream flag lost -> roll back to the start). --guard checker relaxes the
component guard to the checker FAILS (measures what the against_wall / a4 guard costs).

Rooms: complete 27B test rooms whose S1 output still fails the checker (2,225), start = s1_layout [--start raw].
Ablation of the paper's acceptance: --t0 1e-9 --t1 1e-9 = greedy (same proposals, monotone).

PYTHONPATH=REPO python3 infinigen_sa.py --out DIR [--limit N --seed 0 --workers 10 --max-trials 5000 --time-limit 60
    --t0 .25 --t1 .001 --finetune .15 --stages 3 --yaw --viol checker|strict --guard full|checker --p-viol .8]
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

from shapely.geometry import Polygon

from fastfill.comparison import downstream as d
from fastfill.comparison import geometry as g
from fastfill.comparison import output_repair as R

BASE = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013')
ROOMS = BASE / 'visual-final-selected-002/inputs/eval-data-v1/rooms.jsonl'
REQUESTS = BASE / 'visual-final-selected-002/inputs/eval-data-v1/placement/requests.jsonl'
S1_ROWS = BASE / 'a026-inputs/s1-022/placement-arm/placement-geometry-s1.jsonl'
TRANS_MULT, TRANS_MIN, ROT_MULT, ANGLE_STEP = 8.0, 0.01, math.pi, math.pi / 2     # propose_continous.py (pi/4 there)
STAGE_STEPS = (300, 200, 50)                                                       # base_indoors.gin
W_TRANSLATE, W_ROTATE = 1.0, 0.5                                                   # solve.py
CHECKER = ('oob', 'support_fail', 'ceiling', 'fixed_blocking', 'main', 'severe')
STRICT = CHECKER + ('outside_1mm', 'keepout_1cm', 'held_broken', 'a4_aabb')
IDX = {k: i for i, k in enumerate(R.COMPONENTS)}


# ------------------------------------------------------------------ yaw support: shift = (mm_x, mm_y[, quarter turns])

def _placed(ctx, i, shift):
    o = ctx['raw'][i]
    q = shift[2] if len(shift) > 2 else 0
    if shift[0] == 0 and shift[1] == 0 and q == 0:
        return o
    out = {**o, 'pos': [round(o['pos'][0] + shift[0] / 1000, 6), round(o['pos'][1] + shift[1] / 1000, 6), o['pos'][2]]}
    if q:
        out['yaw'] = (o['yaw'] + q * ANGLE_STEP) % math.tau
    return out


R._placed = _placed                      # S1's _state / _unit_move read shifts through this name


def delta(ctx, st, moves):
    """state after per-object moves {id: (mx, my[, dq])}."""
    shift = dict(st['shift'])
    for i, m in moves.items():
        s = shift[i]
        shift[i] = (s[0] + m[0], s[1] + m[1], (s[2] if len(s) > 2 else 0) + (m[2] if len(m) > 2 else 0))
    return R._state(ctx, shift, st, list(moves))


def rotate_unit(ctx, st, unit, q):
    """Rotate `unit` and its subtree by q quarter turns about the unit's centre: {member: (mx, my, dq)} or None."""
    if unit in ctx['pinned'] or q == 0:
        return None
    c = st['cur'][unit]['pos']
    ang, out = q * ANGLE_STEP, {}
    for j in ctx['members'][unit]:
        p = st['cur'][j]['pos']
        x, y = R._rot(p[0] - c[0], p[1] - c[1], ang)
        out[j] = (round((c[0] + x - p[0]) * 1000), round((c[1] + y - p[1]) * 1000), q)
    return out


# ------------------------------------------------------------------ solver

def viol(st, comps):
    return sum(st['v'][IDX[k]] for k in comps)


def violators(ctx, st):
    ids = set()
    for (a, b), (kind, _, _) in st['pair'].items():
        if kind == 'main':
            ids.update((a, b))
    for i, r in st['row'].items():
        if r[0] is not None or r[1] or r[4]:
            ids.add(i)
    return ids


def roots(ctx):
    return [i for i in ctx['ids'] if ctx['chain'][i][-1] == i]


def stage_sets(ctx, n):
    allu = [i for i in ctx['ids'] if i not in ctx['pinned']]
    if n == 1:
        return [allu]
    rs = [i for i in roots(ctx) if i not in ctx['pinned']]
    area = {i: ctx['raw'][i]['size'][0] * ctx['raw'][i]['size'][1] for i in rs}
    med = statistics.median(area.values()) if area else 0.0
    big = [i for i in rs if area[i] >= med] or rs
    return [big, rs, allu]


def anneal(ctx, st, active, steps, a, rng, log):
    """Infinigen SA over `active` units for `steps` proposals; returns (state, best state)."""
    comps = STRICT if a['viol'] == 'strict' else CHECKER
    best, cur_v, best_v = st, (viol(st, comps), st['phi']), (viol(st, comps), st['phi'])
    cool = max(1, int(steps * (1 - a['finetune'])))
    rate = (a['t1'] / a['t0']) ** (1 / cool)
    for it in range(steps):
        if time.monotonic() > a['deadline']:
            log['timed_out'] = True
            break
        temp = min(max(a['t0'] * rate ** it, a['t1']), a['t0'])
        bad = violators(ctx, st)
        pool = [u for u in active if u in bad or any(m in bad for m in ctx['members'][u])] if rng.random() < a['p_viol'] else []
        unit = rng.choice(pool or active)
        if a['yaw'] and rng.random() < W_ROTATE / (W_TRANSLATE + W_ROTATE):
            ang = rng.gauss(0, max(1e-9, ROT_MULT * temp)) / ANGLE_STEP
            q = int(math.ceil(ang) if ang > 0 else math.floor(ang)) % 4
            mv = rotate_unit(ctx, st, unit, q)
        else:
            sigma = max(TRANS_MIN, TRANS_MULT * temp)
            mv = R._unit_move(ctx, st, unit, rng.gauss(0, sigma), rng.gauss(0, sigma))
        log['proposals'] += 1
        if not mv:
            log['invalid'] += 1
            continue
        nxt = delta(ctx, st, mv)
        nv = (viol(nxt, comps), nxt['phi'])
        if nv[0] < cur_v[0]:
            ok = True
        elif nv[0] > cur_v[0]:
            ok = False
        else:
            ok = math.log(rng.random() or 1e-300) < -(nv[1] - cur_v[1]) / temp
        if ok:
            st, cur_v = nxt, nv
            log['accepted'] += 1
            if nv < best_v:
                best, best_v = st, nv
        if best_v[0] == 0 and a['viol'] == 'checker' and best_v[1] <= 0:
            break
    return st, best


def solve(pred, a, rng):
    started = time.monotonic()
    a = {**a, 'deadline': started + a['time_limit']}
    keepouts = R.door_keepouts(pred)
    ctx = R._context(pred, keepouts, ())
    before = R.audit(pred, keepouts, ctx['held'])
    st = R._state(ctx, {i: (0, 0, 0) for i in ctx['ids']})
    log = {'proposals': 0, 'accepted': 0, 'invalid': 0, 'timed_out': False}
    sets = stage_sets(ctx, a['stages'])
    share = STAGE_STEPS[:len(sets)] if len(sets) == 3 else (1,)
    best_all = st
    for active, w in zip(sets, share):
        if not active:
            continue
        st, best = anneal(ctx, st, active, max(1, round(a['max_trials'] * w / sum(share))), a, rng, log)
        if (viol(best, STRICT), best['phi']) < (viol(best_all, STRICT), best_all['phi']):
            best_all = best
        st = best                                   # next stage starts from this stage's best (deviation, see doc)
    st = best_all
    objects = [copy.deepcopy(st['cur'][i]) if any(st['shift'][i]) else ctx['raw'][i] for i in ctx['ids']]
    final_room = {**pred, 'objects': objects}
    final = R.audit(final_room, keepouts, ctx['held'])
    moved = any(any(s) for s in st['shift'].values())
    rolled, verdict = None, {'start': d.accept(pred, pred['objects'])}
    keys = ('id', 'category', 'size', 'parent') + (() if a['yaw'] else ('yaw',))
    guard = R.COMPONENTS if a['guard'] == 'full' else CHECKER
    if not all(all(x.get(k) == y.get(k) for k in keys) and x['pos'][2] == y['pos'][2]
               for x, y in zip(pred['objects'], objects)):
        rolled = 'guard_violation'
    elif any(final['v'][IDX[k]] > before['v'][IDX[k]] for k in guard):
        rolled = 'checker_worse'
    elif moved:
        verdict['final'] = d.accept(final_room, objects)
        if any(verdict['start'][k] and not verdict['final'][k] for k in R.DOWNSTREAM_FLAGS):
            rolled = 'downstream_worse'
    if rolled or not moved:
        final_room, final, verdict['final'] = pred, before, verdict['start']
    vec = lambda s: dict(zip(R.COMPONENTS, s['v'])) | {'phi': s['phi']}
    return {'room': final_room, 'before': vec(before), 'final': vec(final), 'rolled_back': rolled, **log,
            'runtime_s': time.monotonic() - started, 'strict_start': verdict['start']['accepted'],
            'strict_final': verdict['final']['accepted'],
            'displacement': [math.dist(x['pos'][:2], y['pos'][:2]) for x, y in zip(pred['objects'], final_room['objects'])],
            'yaw_changed': sum(x['yaw'] != y['yaw'] for x, y in zip(pred['objects'], final_room['objects']))}


def bin3(n):
    return '<=20' if n <= 20 else '21-50' if n <= 50 else '>50'


def one(task):
    row, room, request, a = task
    layout = row['s1_layout'] if a['start'] == 's1' else row['raw_layout']
    pred = R.predicted_room(room, request['messages'], layout)
    res = solve(pred, a, random.Random(f"{a['seed']}:{row['uid']}"))
    geo0, geo1 = g.geometry(pred), g.geometry(res['room'])
    return {'uid': row['uid'], 'source': row['source'], 'objects': row['objects'], 'stratum': R._stratum(row['objects']),
            'bin3': bin3(row['objects']), 'doors': len(d.door_strips(pred)),
            'rectangle': d.is_rectangle(Polygon(pred['boundary'])),
            'start_kinds': sorted(k for k in CHECKER if res['before'][k]),
            'geometry_success_start': geo0['geometry_success'], 'geometry_success_final': geo1['geometry_success'],
            **{k: res[k] for k in ('before', 'final', 'rolled_back', 'proposals', 'accepted', 'invalid', 'timed_out',
                                   'runtime_s', 'strict_start', 'strict_final', 'yaw_changed')},
            'moved_objects': sum(x > 0 for x in res['displacement']),
            'moved_median_m': statistics.median([x for x in res['displacement'] if x > 0] or [0.0]),
            'moved_max_m': max(res['displacement'], default=0.0), 'final_layout': res['room']['objects']}


def agg(rows):
    n = len(rows)
    f = lambda k: sum(bool(r[k]) for r in rows)
    m = lambda k: statistics.mean(r[k] for r in rows) if n else None
    out = {'rooms': n, 'checker_fixed': f('geometry_success_final'), 'strict_fixed': f('strict_final'),
           'strict_start': f('strict_start'), 'timed_out': f('timed_out'),
           'rolled_back': dict(collections.Counter(r['rolled_back'] for r in rows if r['rolled_back'])),
           'runtime_mean_s': m('runtime_s'), 'runtime_max_s': max((r['runtime_s'] for r in rows), default=None),
           'proposals_mean': m('proposals'), 'accepted_mean': m('accepted'),
           'start_main_mean': statistics.mean(r['before']['main'] for r in rows) if n else None,
           'final_main_mean': statistics.mean(r['final']['main'] for r in rows) if n else None,
           'final_viol_mean': statistics.mean(sum(r['final'][k] for k in CHECKER) for r in rows) if n else None,
           'start_viol_mean': statistics.mean(sum(r['before'][k] for k in CHECKER) for r in rows) if n else None,
           'held_broken_delta_mean': statistics.mean(r['final']['held_broken'] - r['before']['held_broken'] for r in rows) if n else None,
           'yaw_changed_mean': m('yaw_changed'), 'moved_objects_mean': m('moved_objects'),
           'moved_median_m_mean': m('moved_median_m')}
    out['checker_fixed_rate'] = out['checker_fixed'] / n if n else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--start', choices=('s1', 'raw'), default='s1')
    ap.add_argument('--limit', type=int, default=0, help='per-stratum cap (0 = every failing room)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--max-trials', type=int, default=5000)
    ap.add_argument('--time-limit', type=float, default=60.0)
    ap.add_argument('--t0', type=float, default=0.25)
    ap.add_argument('--t1', type=float, default=0.001)
    ap.add_argument('--finetune', type=float, default=0.15)
    ap.add_argument('--stages', type=int, choices=(1, 3), default=3)
    ap.add_argument('--yaw', action='store_true')
    ap.add_argument('--viol', choices=('checker', 'strict'), default='checker')
    ap.add_argument('--guard', choices=('full', 'checker'), default='full')
    ap.add_argument('--p-viol', type=float, default=0.8)
    ap.add_argument('--uids', type=Path)
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
    args = {k: getattr(a, k) for k in ('start', 'seed', 'max_trials', 'time_limit', 't0', 't1', 'finetune', 'stages',
                                       'yaw', 'viol', 'guard', 'p_viol')}
    tasks = [(r, rooms[r['uid']], reqs[r['uid']], args) for r in rows]
    t0 = time.monotonic()
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(one, tasks, chunksize=1))
    a.out.mkdir(parents=True)
    with open(a.out / 'rows.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r, allow_nan=False) + '\n')
    strata = [R._stratum(lo) for lo, _ in R.STRATA]
    summary = {'probe': 'infinigen-sa', 'args': {**args, 'limit': a.limit, 'trans_mult': TRANS_MULT,
                                                  'trans_min': TRANS_MIN, 'angle_step_deg': 90, 'stage_steps': STAGE_STEPS},
               'wall_s': time.monotonic() - t0, 'all': agg(results),
               'by_stratum': {k: agg([r for r in results if r['stratum'] == k]) for k in strata},
               'by_bin3': {k: agg([r for r in results if r['bin3'] == k]) for k in ('<=20', '21-50', '>50')},
               'by_kinds': {'+'.join(k): agg([r for r in results if tuple(r['start_kinds']) == k])
                            for k in sorted({tuple(r['start_kinds']) for r in results})},
               'by_doors': {k: agg([r for r in results if (r['doors'] > 0) == (k == 'doors')]) for k in ('no_doors', 'doors')}}
    json.dump(summary, open(a.out / 'summary.json', 'w'), indent=1)
    print(json.dumps({'rooms': len(results), 'all': summary['all'],
                      'by_stratum': {k: (v['rooms'], v['checker_fixed']) for k, v in summary['by_stratum'].items()},
                      'wall_s': round(summary['wall_s'], 1)}))


if __name__ == '__main__':
    main()
