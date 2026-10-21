"""API probe runner (DESIGNED HERE, UNTESTED: runs on Isambard as a CPU sbatch job with ~/.fastfill_api.env).

  api:   python3 hdsl_local_regen_run.py api REQUESTS_JSONL OUT_DIR [--env-file ~/.fastfill_api.env] [--max-calls 100]
         one gpt-6.1-sol call per row (llm_agent.call_api, reasoning medium, SUFFIX appended, no repair round),
         receipts appended to OUT_DIR/receipts.jsonl before the next call; resumable; hard cap on calls.
  score: python3 hdsl_local_regen_run.py score REQUESTS_JSONL RECEIPTS_JSONL TEST_REQUESTS_JSONL OUT_DIR [--workers 8]
         merges each answer into the full room (locked objects back to movable), scores with pipeline._score
         (S1 + frozen checker + strict), compares with the S1 baseline (all rows fail the checker by construction).
Needs PYTHONPATH=<repo> (uses fastfill.comparison.llm_agent / contract and fastfill.system.pipeline).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fastfill.comparison import contract  # noqa: E402
from fastfill.comparison import llm_agent as L  # noqa: E402

MAX_OUTPUT_TOKENS, TIMEOUT = 16384, 900


def _opt(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def api(reqp, outd):
    outd.mkdir(parents=True, exist_ok=True)
    cap = _opt('--max-calls', 100, int)
    config = L.load_config(Path(_opt('--env-file', '~/.fastfill_api.env')).expanduser())
    if config.model != L.EXPECTED_MODEL:
        raise SystemExit('configured_model_mismatch')
    rec = outd / 'receipts.jsonl'
    done = {json.loads(l)['uid'] for l in open(rec)} if rec.exists() else set()
    calls = len(done)
    for line in open(reqp):
        row = json.loads(line)
        if row['uid'] in done:
            continue
        if calls >= cap:
            print('cap reached', calls)
            break
        calls += 1
        msgs = [*row['messages'], {'role': 'user', 'content': L.SUFFIX}]
        try:
            reply = L.call_api(config, msgs, MAX_OUTPUT_TOKENS, TIMEOUT)
            out = {'uid': row['uid'], 'raw': reply['text'], 'usage': reply.get('usage'), 'error': None}
        except L.ApiError as exc:
            out = {'uid': row['uid'], 'raw': None, 'usage': None, 'error': exc.code}
        with open(rec, 'a') as f:
            f.write(json.dumps(out) + '\n')
            f.flush()
    print(json.dumps({'calls': calls}))


def score(reqp, recp, testp, outd):
    from fastfill.system import pipeline as P
    outd.mkdir(parents=True, exist_ok=True)
    rows = {r['uid']: r for r in map(json.loads, open(reqp))}
    receipts = {r['uid']: r for r in map(json.loads, open(recp))}
    tests = {r['uid']: r for r in map(json.loads, open(testp)) if r['uid'] in rows}
    tasks, keys = [], []
    for uid, row in rows.items():
        rc = receipts.get(uid)
        raw = None
        if rc and rc['raw']:
            try:
                out = json.loads(rc['raw'])
                want = [f'obj_{k:04d}' for k in range(1, len(row['plan'][1]) + 1)]
                if [p['id'] for p in out['placements']] != want:
                    raise ValueError('ids')
                back = {**row['maps']['fixed'], **row['maps']['obj']}
                placed = {}
                for p in out['placements']:
                    item = {'id': back[p['id']], 'pos': p['pos'], 'yaw': p['yaw']}
                    if 'on' in p:
                        item['on'] = back.get(p['on'], p['on'])      # injected fixed -> movable; real fixed stays
                    placed[item['id']] = item
                for i, l in row['locked'].items():                   # locked objects: S1 pose and S1 support
                    placed[i] = {'id': i, 'pos': l['pos'], 'yaw': l['yaw'], **({'on': l['on']} if l['on'] else {})}
                user = json.loads(tests[uid]['messages'][1]['content'])
                raw = contract._json({'placements': [placed[o['id']] for o in user['objects']]})
            except (ValueError, KeyError, TypeError):
                raw = None
        room = P.request_room(tests[uid])
        tasks.append(('local_regen', room, tests[uid], raw, None if raw else 'INVALID', False))
        keys.append(uid)
    results = P._score_all(tasks, _opt('--workers', 8, int))
    out_rows = [{'uid': uid, 'stratum': rows[uid]['stratum'], 'regen': rows[uid]['regen'], 'objects': rows[uid]['objects'],
                 'complete': r['complete'], 'checker_pass_raw': r['checker_pass_raw'], 'checker_pass': r['checker_pass'],
                 'accepted': r['accepted'], 'violations': r['violations'],
                 'usage': (receipts.get(uid) or {}).get('usage')} for uid, r in zip(keys, results)]
    with open(outd / 'rooms.jsonl', 'w') as f:
        for r in out_rows:
            f.write(json.dumps(r) + '\n')
    summ = {}
    for s in sorted({r['stratum'] for r in out_rows}) + ['all']:
        rs = [r for r in out_rows if s == 'all' or r['stratum'] == s]
        summ[s] = {'rooms': len(rs), 'complete': sum(r['complete'] for r in rs),
                   'checker_pass_raw': sum(bool(r['checker_pass_raw']) for r in rs) / len(rs),
                   'checker_pass_after_s1': sum(bool(r['checker_pass']) for r in rs) / len(rs),
                   'accepted': sum(bool(r['accepted']) for r in rs) / len(rs),
                   'baseline_checker_pass_s1': 0.0}
    json.dump(summ, open(outd / 'summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    cmd, a = sys.argv[1], [Path(x) for x in sys.argv[2:] if not x.startswith('--') and not x.lstrip('-').isdigit()]
    if cmd == 'api':
        api(a[0], a[1])
    elif cmd == 'score':
        score(a[0], a[1], a[2], a[3])
