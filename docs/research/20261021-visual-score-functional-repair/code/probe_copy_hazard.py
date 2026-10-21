"""Probe copy-hazard (free, CPU): does the 27B's exact-copy failure depend on OUTPUT POSITION (then a FuncRoom-style
chunked/segment decode that restarts the answer every k records can remove it) or only on ROOM SIZE (then it cannot)?

Reads the 27B test raw rows, the S1 rows and the frozen requests; writes JSON + Markdown under --out.
  A. copy hazard h(k | N-bin): P(record k is an exact copy of an earlier record), by output position k and room size N
  B. rooms failing after S1 that contain a copy (what chunking could touch) vs none (what it cannot)
  C. de-dup ceiling on a seeded sample of S1-failing copy rooms: drop later copies, re-run checker and S1
     (an upper bound: the dropped objects still need a place) and the empirical reference pass_s1 | no-copy per bin

python3 probe_copy_hazard.py --raw RAW.jsonl --s1 S1.jsonl --requests REQ.jsonl --rooms ROOMS.jsonl --out DIR
       [--sample 200] [--workers 8] [--chunk 8]
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')  # ponytail: hard-coded repo; PYTHONPATH if moved
from fastfill.comparison import geometry as g            # noqa: E402
from fastfill.comparison import output_repair as r      # noqa: E402

BINS = ((1, 10), (11, 20), (21, 50), (51, 10 ** 9))
KBINS = ((1, 4), (5, 8), (9, 16), (17, 24), (25, 40), (41, 10 ** 9))


def nbin(n):
    return next(f'{a}-{b}' if b < 10 ** 9 else f'{a}+' for a, b in BINS if a <= n <= b)


def kbin(k):
    return next(f'{a}-{b}' if b < 10 ** 9 else f'{a}+' for a, b in KBINS if a <= k <= b)


def key(o):
    return (tuple(round(v, 3) for v in o['pos']), round(o['yaw'], 4))


def copies(layout):
    """index -> True if an exact copy (mm position and yaw) of an earlier record."""
    seen, out = set(), []
    for o in layout:
        k = key(o)
        out.append(k in seen)
        seen.add(k)
    return out


def load(path, keep=None):
    rows = {}
    for line in open(path):
        row = json.loads(line)
        if keep is None or keep(row):
            rows[row['uid']] = row
    return rows


def room_for(gt, request, layout):
    return r.predicted_room(gt, request['messages'], layout)


def dedup_task(task):
    uid, gt, request, layout = task
    flags = copies(layout)
    dropped, kept = set(), []
    for o, c in zip(layout, flags):           # a copy goes with its whole support subtree
        if c or o.get('parent') in dropped:
            dropped.add(o['id'])
        else:
            kept.append(o)
    room = room_for(gt, request, kept)
    raw = g.geometry(room)['geometry_success']
    res = r.repair(room, max_trials=2000, time_limit=60.0)
    s1 = g.geometry(res['room'])['geometry_success']
    return {'uid': uid, 'objects': len(layout), 'dropped': len(layout) - len(kept), 'dedup_raw_pass': bool(raw),
            'dedup_s1_pass': bool(s1), 's1_status': res['status'], 'runtime_s': res['diagnostics']['runtime_s']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--s1', required=True)
    ap.add_argument('--requests', required=True)
    ap.add_argument('--rooms', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample', type=int, default=200)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = load(a.raw, lambda x: x.get('complete') and x.get('raw_layout'))
    s1 = {u: x['s1'] for u, x in load(a.s1).items()}

    # ---- A: hazard by position and size
    hz = collections.defaultdict(lambda: [0, 0])          # (N-bin, k-bin) -> [copies, records]
    rel = collections.defaultdict(lambda: [0, 0])         # (N-bin, decile of k/N) -> [copies, records]
    first = collections.defaultdict(lambda: [0, 0])       # N-bin -> [copies in k<=chunk, records in k<=chunk]
    later = collections.defaultdict(lambda: [0, 0])
    rooms_copy = collections.Counter()
    rooms_n = collections.Counter()
    cf_expect = collections.defaultdict(float)            # N-bin -> sum over rooms of P(room has a copy | chunk restart)
    per_room = {}
    for uid, row in raw.items():
        lay = row['raw_layout']
        n = len(lay)
        c = copies(lay)
        b = nbin(n)
        rooms_n[b] += 1
        rooms_copy[b] += any(c)
        per_room[uid] = (n, sum(c), b)
        for k, flag in enumerate(c, 1):
            hz[(b, kbin(k))][0] += flag
            hz[(b, kbin(k))][1] += 1
            d = min(9, int(10 * (k - 1) / n))
            rel[(b, d)][0] += flag
            rel[(b, d)][1] += 1
            tgt = first if k <= a.chunk else later
            tgt[b][0] += flag
            tgt[b][1] += 1
    for uid, (n, _, b) in per_room.items():
        h1 = first[b][0] / max(1, first[b][1])
        cf_expect[b] += 1 - (1 - h1) ** n   # independence approximation: every chunk behaves like the first
    A = {'hazard_by_position': {f'{b}|k{kb}': {'copies': v[0], 'records': v[1], 'h': v[0] / v[1]}
                                for (b, kb), v in sorted(hz.items())},
         'hazard_by_relative_decile': {f'{b}|d{d}': {'copies': v[0], 'records': v[1], 'h': v[0] / v[1]}
                                       for (b, d), v in sorted(rel.items())},
         'first_chunk_vs_later': {b: {'h_first': first[b][0] / max(1, first[b][1]), 'n_first': first[b][1],
                                      'h_later': later[b][0] / max(1, later[b][1]), 'n_later': later[b][1],
                                      'rooms': rooms_n[b], 'rooms_with_copy': rooms_copy[b],
                                      'observed_room_copy_rate': rooms_copy[b] / rooms_n[b],
                                      'chunk_restart_expected_room_copy_rate': cf_expect[b] / rooms_n[b]}
                                  for b in rooms_n}}

    # ---- B: S1-failing rooms with / without a copy
    B = collections.defaultdict(collections.Counter)
    for uid, (n, ncopy, b) in per_room.items():
        s = s1.get(uid)
        if s is None or s.get('status') == 'not_complete':
            continue
        fail = not s['geometry_success_s1']
        B[b]['complete'] += 1
        B[b]['fail_s1'] += fail
        B[b]['fail_s1_with_copy'] += fail and ncopy > 0
        B[b]['fail_s1_no_copy'] += fail and ncopy == 0
        B[b]['pass_s1_no_copy'] += (not fail) and ncopy == 0
        B[b]['no_copy'] += ncopy == 0
        B[b]['pass_s1_with_copy'] += (not fail) and ncopy > 0
        B[b]['with_copy'] += ncopy > 0
    for b in B:
        B[b]['pass_s1_rate_no_copy'] = B[b]['pass_s1_no_copy'] / max(1, B[b]['no_copy'])
        B[b]['pass_s1_rate_with_copy'] = B[b]['pass_s1_with_copy'] / max(1, B[b]['with_copy'])

    # ---- C: de-dup ceiling on a seeded sample of S1-failing copy rooms
    cand = sorted(uid for uid, (n, ncopy, b) in per_room.items()
                  if ncopy > 0 and uid in s1 and s1[uid].get('status') != 'not_complete'
                  and not s1[uid]['geometry_success_s1'])
    random.Random('probe-copy-hazard-20261021').shuffle(cand)
    cand = cand[:a.sample]
    want = set(cand)
    reqs = load(a.requests, lambda x: x['uid'] in want)
    gts = load(a.rooms, lambda x: x['uid'] in want)
    tasks = [(u, gts[u], reqs[u], raw[u]['raw_layout']) for u in cand]
    with ProcessPoolExecutor(a.workers) as pool:
        C_rows = list(pool.map(dedup_task, tasks, chunksize=2))
    C = collections.defaultdict(collections.Counter)
    for x in C_rows:
        b = nbin(x['objects'])
        C[b]['rooms'] += 1
        C[b]['dedup_raw_pass'] += x['dedup_raw_pass']
        C[b]['dedup_s1_pass'] += x['dedup_s1_pass']
        C[b]['dropped_objects'] += x['dropped']
        C[b]['objects'] += x['objects']
    C_all = {'rooms': len(C_rows), 'dedup_raw_pass': sum(x['dedup_raw_pass'] for x in C_rows),
             'dedup_s1_pass': sum(x['dedup_s1_pass'] for x in C_rows)}

    result = {'inputs': {k: getattr(a, k) for k in ('raw', 's1', 'requests', 'rooms', 'sample', 'chunk')},
              'rooms_complete': len(raw), 'A': A, 'B': {b: dict(v) for b, v in B.items()},
              'C': {'by_bin': {b: dict(v) for b, v in C.items()}, 'all': C_all, 'rows': C_rows}}
    (out / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False))
    lines = ['# copy-hazard probe (27B test raw outputs)', '',
             f'complete rooms {len(raw)}; chunk size for the restart counterfactual = {a.chunk}', '',
             '## A. copy hazard h(k | N) = P(record k is an exact copy of an earlier record)', '',
             '| N-bin | ' + ' | '.join(f'k {a_}-{b_}' if b_ < 10 ** 9 else f'k {a_}+' for a_, b_ in KBINS) + ' |',
             '|---|' + '---:|' * len(KBINS)]
    for b in sorted(rooms_n, key=lambda x: int(x.split('-')[0].rstrip('+'))):
        cells = []
        for a_, b_ in KBINS:
            kb = f'{a_}-{b_}' if b_ < 10 ** 9 else f'{a_}+'
            v = hz.get((b, kb))
            cells.append(f'{v[0] / v[1]:.3f} ({v[1]})' if v and v[1] else '-')
        lines.append(f'| {b} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| N-bin | h first chunk | h later | rooms | observed room copy rate | chunk-restart expected |',
              '|---|---:|---:|---:|---:|---:|']
    for b, v in sorted(A['first_chunk_vs_later'].items(), key=lambda x: int(x[0].split('-')[0].rstrip('+'))):
        lines.append(f"| {b} | {v['h_first']:.3f} | {v['h_later']:.3f} | {v['rooms']} | "
                     f"{v['observed_room_copy_rate']:.3f} | {v['chunk_restart_expected_room_copy_rate']:.3f} |")
    lines += ['', '## B. rooms failing after S1', '',
              '| N-bin | complete | fail S1 | fail with copy | fail no copy | pass_s1 rate no-copy | pass_s1 rate with-copy |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for b, v in sorted(B.items(), key=lambda x: int(x[0].split('-')[0].rstrip('+'))):
        lines.append(f"| {b} | {v['complete']} | {v['fail_s1']} | {v['fail_s1_with_copy']} | {v['fail_s1_no_copy']} | "
                     f"{v['pass_s1_rate_no_copy']:.3f} | {v['pass_s1_rate_with_copy']:.3f} |")
    lines += ['', f'## C. de-dup ceiling (sample {len(C_rows)} S1-failing copy rooms; drop later copies; upper bound)', '',
              '| N-bin | rooms | dropped/objects | raw pass after de-dup | S1 pass after de-dup |', '|---|---:|---:|---:|---:|']
    for b, v in sorted(C.items(), key=lambda x: int(x[0].split('-')[0].rstrip('+'))):
        lines.append(f"| {b} | {v['rooms']} | {v['dropped_objects']}/{v['objects']} | {v['dedup_raw_pass']} | {v['dedup_s1_pass']} |")
    lines.append(f"| all | {C_all['rooms']} | | {C_all['dedup_raw_pass']} | {C_all['dedup_s1_pass']} |")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
