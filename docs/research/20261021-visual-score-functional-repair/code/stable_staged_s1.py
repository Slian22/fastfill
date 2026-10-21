"""STABLE transfer probe (free, local CPU): staged S1 = build the 27B raw layout tier by tier and run S1 after each
tier (STABLE Algorithm 1 with S1 as the corrector), versus one-shot S1 with the same total budget (3x trials/time).
Tiers use only deployment-visible info: tier 1 = floor roots with footprint area >= median root area ("anchors"),
tier 2 = the other floor roots, tier 3 = supported objects (the model's own `on`), shifted with their subtree root.
Nothing else changes: ids/size/yaw/z/parent frozen by S1 itself. Stored S1 (1x budget) is the shipped baseline.

python3 stable_staged_s1.py --s1 S1.jsonl --rooms ROOMS.jsonl --requests REQ.jsonl --out DIR
    [--n-fail 100 --n-pass 40 --min-objects 21 --workers 10 --max-trials 2000 --time-limit 60 --uids FILE]
"""
import argparse, collections, copy, hashlib, json, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = '/Users/slian/Desktop/3D/Worldedge/OptiScene'
sys.path.insert(0, REPO)
from fastfill.comparison import downstream as d          # noqa: E402
from fastfill.comparison import geometry as g            # noqa: E402
from fastfill.comparison.output_repair import predicted_room, repair   # noqa: E402

STRATA = ((1, 10), (11, 20), (21, 50), (51, 10**9))


def stratum(n):
    return next(f'{lo}-{hi}' if hi < 10**9 else f'{lo}+' for lo, hi in STRATA if lo <= n <= hi)


def tiers(layout):
    by = {o['id']: o for o in layout}
    def root_of(i):
        seen = set()
        while by[i].get('parent') in by and i not in seen:
            seen.add(i); i = by[i]['parent']
        return i
    roots = [o['id'] for o in layout if root_of(o['id']) == o['id']]
    areas = {i: by[i]['size'][0] * by[i]['size'][1] for i in roots}
    med = statistics.median(areas.values()) if areas else 0
    t1 = {i for i in roots if areas[i] >= med} if len(roots) > 2 else set(roots)
    t2 = set(roots) - t1
    t3 = {o['id'] for o in layout} - set(roots)
    return [t1, t2, t3], {o['id']: root_of(o['id']) for o in layout}


def metrics(room):
    geo = g.geometry(room)
    return {'success': bool(geo['geometry_success']), 'accepted': bool(geo['geometry_success'] and
            d.accept(room, room['objects'])['accepted']), 'main': geo['main_collisions'],
            'severe': geo['severe_collisions'], 'oob': geo['oob_objects'], 'fixed_blocking': geo['fixed_blocking'],
            'support_fail': geo['support_fail']}


def invariant(raw, out):
    return len(raw) == len(out) and all(a['id'] == b['id'] and a['size'] == b['size'] and a['yaw'] == b['yaw']
                                        and a.get('parent') == b.get('parent') and a['pos'][2] == b['pos'][2]
                                        for a, b in zip(raw, out))


def staged(pred, max_trials, time_limit):
    raw = pred['objects']
    tier, root = tiers(raw)
    rawby = {o['id']: o for o in raw}
    corrected, stages, t0 = {}, [], time.monotonic()
    active = set()
    for k, t in enumerate(tier, 1):
        active |= t
        if not t:
            stages.append({'stage': k, 'added': 0, 'skipped': True}); continue
        objs = []
        for o in raw:
            i = o['id']
            if i not in active:
                continue
            if i in corrected:
                objs.append(copy.deepcopy(corrected[i])); continue
            c = copy.deepcopy(o)
            r = root[i]
            if r in corrected and r != i:                       # a child enters at raw pose + its root's S1 shift
                dx = corrected[r]['pos'][0] - rawby[r]['pos'][0]; dy = corrected[r]['pos'][1] - rawby[r]['pos'][1]
                c['pos'] = [c['pos'][0] + dx, c['pos'][1] + dy, c['pos'][2]]
            objs.append(c)
        res = repair({**pred, 'objects': objs}, max_trials=max_trials, time_limit=time_limit)
        if res['status'] == 'rejected':
            raise ValueError(res['diagnostics']['error'])
        corrected = {o['id']: o for o in res['room']['objects']}
        stages.append({'stage': k, 'added': len(t), 'objects': len(objs), 'status': res['status'],
                       'trials': res['diagnostics']['trials'], 'runtime_s': res['diagnostics']['runtime_s'],
                       'final': res['diagnostics']['final']})
    final = {**pred, 'objects': [corrected[o['id']] for o in raw]}
    return {**metrics(final), 'invariant': invariant(raw, final['objects']), 'runtime_s': time.monotonic() - t0,
            'stages': stages, 'tiers': [len(t) for t in tier]}, final['objects']


def control(pred, max_trials, time_limit):
    t0 = time.monotonic()
    res = repair(pred, max_trials=max_trials, time_limit=time_limit)
    return {**metrics(res['room']), 'status': res['status'], 'trials': res['diagnostics']['trials'],
            'runtime_s': time.monotonic() - t0}, res['room']['objects']


def one(task):
    uid, room, request, raw, stored, max_trials, time_limit = task
    pred = predicted_room(room, request['messages'], raw)
    st, st_layout = staged(pred, max_trials, time_limit)
    ct, _ = control(pred, 3 * max_trials, 3 * time_limit)
    return {'uid': uid, 'objects': len(raw), 'stratum': stratum(len(raw)), 's1_stored': stored, 'staged': st,
            'control3x': ct, 'staged_layout': st_layout}


def main():
    ap = argparse.ArgumentParser()
    for k in ('s1', 'rooms', 'requests', 'out'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--n-fail', type=int, default=100); ap.add_argument('--n-pass', type=int, default=40)
    ap.add_argument('--min-objects', type=int, default=21); ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--max-objects', type=int, default=120)
    ap.add_argument('--max-trials', type=int, default=2000); ap.add_argument('--time-limit', type=float, default=60.0)
    ap.add_argument('--uids')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cand = []
    for line in open(a.s1):
        r = json.loads(line)
        if r.get('complete') and a.min_objects <= len(r['raw_layout']) <= a.max_objects:
            s = r['s1']
            cand.append((hashlib.sha256(('stable-staged:' + r['uid']).encode()).hexdigest(), r['uid'], r['raw_layout'],
                         {'success': bool(s['geometry_success_s1']), 'accepted': bool(s.get('accepted_s1')),
                          'status': s['status']}))
    cand.sort()
    if a.uids:
        want = set(json.load(open(a.uids))); sel = [c for c in cand if c[1] in want]
    else:
        fails = [c for c in cand if not c[3]['success']][:a.n_fail]
        passes = [c for c in cand if c[3]['success']][:a.n_pass]
        sel = fails + passes
    uids = {c[1] for c in sel}
    rooms = {r['uid']: r for r in map(json.loads, open(a.rooms)) if r['uid'] in uids}
    reqs = {r['uid']: r for r in map(json.loads, open(a.requests)) if r['uid'] in uids}
    tasks = [(u, rooms[u], reqs[u], raw, stored, a.max_trials, a.time_limit) for _, u, raw, stored in sel]
    json.dump({'uids': [t[0] for t in tasks], 'rule': 'sha256(stable-staged:uid) order; first n_fail S1-failing + '
               'n_pass S1-passing rooms with min_objects <= objects <= max_objects', 'max_trials': a.max_trials,
               'time_limit': a.time_limit}, open(out / 'selection.json', 'w'), indent=1)
    rows = []
    with ProcessPoolExecutor(a.workers) as pool, open(out / 'rows.jsonl', 'w') as f:
        for fut in as_completed([pool.submit(one, t) for t in tasks]):      # rows land as they finish (unordered)
            row = fut.result()
            rows.append(row); f.write(json.dumps(row) + '\n'); f.flush()
            print(f"{row['uid']} n={row['objects']} stored={row['s1_stored']['success']} staged={row['staged']['success']}"
                  f" ctrl3x={row['control3x']['success']} t={row['staged']['runtime_s']:.0f}/{row['control3x']['runtime_s']:.0f}s",
                  flush=True)
    summarize(rows, out)


def summarize(rows, out):
    lines = []
    def rate(rs, arm, k='success'):
        return f"{sum(r[arm][k] for r in rs)}/{len(rs)} = {sum(r[arm][k] for r in rs) / len(rs):.3f}" if rs else '-'
    for name, rs in [('all', rows)] + [(f'stratum {s}', [r for r in rows if r['stratum'] == s]) for s in ('11-20', '21-50', '51+')] \
            + [(f'S1-stored {k}', [r for r in rows if r['s1_stored']['success'] == v]) for k, v in (('fail', False), ('pass', True))]:
        if rs:
            lines.append(f"{name} (n={len(rs)}): checker stored-S1 {rate(rs, 's1_stored')} | staged {rate(rs, 'staged')} | "
                         f"one-shot 3x {rate(rs, 'control3x')} || strict stored {rate(rs, 's1_stored', 'accepted')} | "
                         f"staged {rate(rs, 'staged', 'accepted')} | 3x {rate(rs, 'control3x', 'accepted')}")
    both = sum(r['staged']['success'] and r['control3x']['success'] for r in rows)
    so = sum(r['staged']['success'] and not r['control3x']['success'] for r in rows)
    co = sum(r['control3x']['success'] and not r['staged']['success'] for r in rows)
    lines.append(f"paired: both {both}, staged-only {so}, 3x-only {co}, neither {len(rows) - both - so - co}")
    bad = [r['uid'] for r in rows if not r['staged']['invariant']]
    lines.append(f"invariant violations (staged): {len(bad)} {bad[:5]}")
    mm = lambda arm: statistics.mean(r[arm]['main'] + r[arm]['severe'] for r in rows)
    lines.append(f"mean residual main+severe collisions: staged {mm('staged'):.2f} | 3x {mm('control3x'):.2f}")
    rt = lambda arm: statistics.median(r[arm]['runtime_s'] for r in rows)
    lines.append(f"median runtime s: staged {rt('staged'):.1f} | 3x {rt('control3x'):.1f}")
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
