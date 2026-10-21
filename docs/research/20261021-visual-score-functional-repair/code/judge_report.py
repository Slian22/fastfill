import json, math, sys, hashlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from judge_lib import *
focus = lambda n: 'reference' if n.startswith('ref') else 'S2F' if n.startswith('S2F') else 'S2'
eye = {'better': [2, 3, 11, 14], 'worse': [1, 5, 6, 9, 12, 15, 17, 18, 19]}
picks = [p['uid'] for p in json.load(open(P / 's2-render/picks.json'))]
lab = {picks[i - 1]: 'better' for i in eye['better']} | {picks[i - 1]: 'worse' for i in eye['worse']}
def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
out = ['# Blind pairwise VLM judge (gpt-6.1-sol, medium)', '']
ok, busy = counts(); tin = tout = 0
for l in LOG.read_text().splitlines():
    u = json.loads(l).get('usage') or {}; tin += u.get('prompt_tokens', 0); tout += u.get('completion_tokens', 0)
out += [f'Successful responses {ok}/200 (incl. 2 probes), busy/error-logged {busy}/300; tokens prompt {tin} completion {tout}', '',
        '| comparison | focus | n | win | tie | loss | win rate excl. ties (Wilson 95%) | chose A | by-eye agreement |', '|---|---|---|---|---|---|---|---|---|']
for f in sorted(J.glob('rows-*.jsonl')):
    n = f.stem[5:]; rows = [json.loads(l) for l in f.read_text().splitlines()]; fo = focus(n)
    if n == 'S1-vs-S2-flipped': fo = 'S2'
    w = sum(r['winner'] == fo for r in rows); t = sum(r['winner'] in ('tie', 'invalid') for r in rows); l_ = len(rows) - w - t
    lo, hi = wilson(w, w + l_); A = sum(r['winner'] == r['first'] or False for r in rows)
    chose_a = sum(1 for r in rows if r['winner'] == r['first'])
    ag = ''
    if n.startswith('S1-vs-S2'):
        k = m = 0
        for r in rows:
            e = lab.get(r['uid'])
            if e: m += 1; k += (r['winner'] == 'S2') if e == 'better' else (r['winner'] == 'S1')
        ag = f'{k}/{m}'
    out.append(f'| {n} | {fo} | {len(rows)} | {w} | {t} | {l_} | {w/max(w+l_,1):.2f} [{lo:.2f},{hi:.2f}] | {chose_a/len(rows):.2f} | {ag} |')
fl = J / 'rows-S1-vs-S2-flipped.jsonl'; base = J / 'rows-S1-vs-S2.jsonl'
if fl.exists() and base.exists():
    b = {json.loads(l)['uid']: json.loads(l)['winner'] for l in base.read_text().splitlines()}
    rr = [json.loads(l) for l in fl.read_text().splitlines()]
    c = sum(b.get(r['uid']) == r['winner'] for r in rr)
    out += ['', f'Position-bias check: {c}/{len(rr)} flipped pairs gave the same winner as the original order.']
(J / 'report.md').write_text('\n'.join(out) + '\n'); print('\n'.join(out))
