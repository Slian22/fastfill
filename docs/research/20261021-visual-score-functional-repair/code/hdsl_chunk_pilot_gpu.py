"""GPU pilot (DESIGNED HERE, UNTESTED - needs the delivered 27B adapter on one GH200; not run in this workflow).

HDSL-lite chunked inference with the UNCHANGED delivered placer (no retraining): every room is answered in
ceil(n/C) greedy calls; call k lists chunk k (area-descending) and injects earlier chunks as fixed records
(hdsl_chunk_requests). Rounds run in lockstep over all rooms so vLLM batches them. The merged request-order
output is scored by pipeline._score (S1 + frozen checker + strict) next to the single-call answer of the same
placer on the same rooms. Primary read-outs: exact-duplicate records per room (same pos mm + yaw) and checker
pass after S1, by object stratum; secondary: strict admission, support links, chunk parse failures, tokens.

PYTHONPATH=<repo> python3 hdsl_chunk_pilot_gpu.py --requests DEV/placement/requests.jsonl --model BASE --adapter ADAPTER \
    --out OUT [--C 10] [--workers 8] [--uids FILE]
Suggested rooms: dev-select-v1 (600 rooms; a025-inputs/dev-select-v1/placement/requests.jsonl), or the test 21-50 and
51+ strata if the pilot is explicitly exploratory. Single GH200, ~25 min for 600 rooms is the expectation, not a measurement.
"""
import argparse
import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hdsl_chunk_requests import chunk_plan, chunk_request, merge  # noqa: E402
from fastfill.comparison import contract  # noqa: E402
from fastfill.system import pipeline as P  # noqa: E402


def dups(layout):
    seen, n = set(), 0
    for o in layout or []:
        key = (tuple(round(v, 3) for v in o['pos']), round(o['yaw'], 4))
        n += key in seen
        seen.add(key)
    return n


def chunk_answers(records, placer, C):
    """{uid: merged raw JSON text or None}; lockstep rounds over rooms."""
    users = {r['uid']: json.loads(r['messages'][1]['content']) for r in records}
    plans = {u: chunk_plan(users[u], C) for u in users}
    placed, answers, failed = collections.defaultdict(dict), collections.defaultdict(list), set()
    for k in range(max(len(p) for p in plans.values())):
        todo = [r for r in records if len(plans[r['uid']]) > k and r['uid'] not in failed]
        if not todo:
            break
        reqs, maps = [], {}
        for r in todo:
            cu, m = chunk_request(users[r['uid']], plans[r['uid']], k, placed[r['uid']])
            maps[r['uid']] = m
            reqs.append({**r, 'uid': f"{r['uid']}#{k}", 'messages': [r['messages'][0],
                                                                    {'role': 'user', 'content': contract._json(cu)}]})
        raw = placer.generate(reqs)
        for r in todo:
            u = r['uid']
            text = (raw.get(f'{u}#{k}') or {}).get('raw')
            want = [f'obj_{i:04d}' for i in range(1, len(plans[u][k]) + 1)]
            try:
                out = json.loads(text)
                if [p['id'] for p in out['placements']] != want or any(not isinstance(p['yaw'], int) for p in out['placements']):
                    raise ValueError('chunk ids/yaw')
            except (TypeError, ValueError, KeyError):
                failed.add(u)
                continue
            answers[u].append((out, maps[u]))
            for p in out['placements']:
                oid = maps[u]['obj'][p['id']]
                placed[u][oid] = {'pos': p['pos'], 'yaw': p['yaw'], 'on': p.get('on')}
    return {r['uid']: (None if r['uid'] in failed else contract._json(merge(users[r['uid']], plans[r['uid']], answers[r['uid']])))
            for r in records}, {u: len(p) for u, p in plans.items()}, failed


def main():
    ap = argparse.ArgumentParser()
    for f in ('requests', 'model', 'adapter', 'out'):
        ap.add_argument('--' + f, required=True)
    ap.add_argument('--C', type=int, default=10)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--uids')
    a = ap.parse_args()
    records = [json.loads(l) for l in open(a.requests) if l.strip()]
    if a.uids:
        keep = set(json.load(open(a.uids)))
        records = [r for r in records if r['uid'] in keep]
    rooms = {r['uid']: P.request_room(r) for r in records}
    placer = P.vllm_placer(a.model, a.adapter)
    single = placer.generate(records)
    chunked, nchunks, failed = chunk_answers(records, placer, a.C)
    tasks = [('single', rooms[r['uid']], r, single[r['uid']]['raw'], single[r['uid']]['error'], single[r['uid']]['cap_hit'])
             for r in records] + \
            [('chunk', rooms[r['uid']], r, chunked[r['uid']], None if chunked[r['uid']] else 'CHUNK_FAILED', False)
             for r in records]
    scored = P._score_all(tasks, a.workers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r, s, c in zip(records, scored[:len(records)], scored[len(records):]):
        n = len(json.loads(r['messages'][1]['content'])['objects'])
        rows.append({'uid': r['uid'], 'objects': n, 'chunks': nchunks[r['uid']],
                     'single': {k: s[k] for k in ('complete', 'checker_pass_raw', 'checker_pass', 'accepted', 'violations')}
                     | {'dups': dups(s['layout'])},
                     'chunk': {k: c[k] for k in ('complete', 'checker_pass_raw', 'checker_pass', 'accepted', 'violations')}
                     | {'dups': dups(c['layout'])}})
    with open(out / 'rooms.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    strata = (('1-10', 1, 10), ('11-20', 11, 20), ('21-50', 21, 50), ('51+', 51, 10 ** 9))
    summary = {'C': a.C, 'rooms': len(rows), 'chunk_failed_rooms': len(failed), 'placer': placer.identity}
    for name, lo, hi in strata:
        rs = [x for x in rows if lo <= x['objects'] <= hi]
        if rs:
            summary[name] = {arm: {'rooms': len(rs),
                                   'checker_pass_raw': sum(bool(x[arm]['checker_pass_raw']) for x in rs) / len(rs),
                                   'checker_pass_s1': sum(bool(x[arm]['checker_pass']) for x in rs) / len(rs),
                                   'accepted': sum(bool(x[arm]['accepted']) for x in rs) / len(rs),
                                   'dups_per_room': sum(x[arm]['dups'] for x in rs) / len(rs)} for arm in ('single', 'chunk')}
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
