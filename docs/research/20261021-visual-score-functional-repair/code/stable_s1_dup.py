"""On the S1 rows of the 27B test: does S1 rescue duplicate rooms? what remains after S1 by object count? and does the
number of fixed boxes in the request hurt raw success at equal object count (proxy for 'placed objects as fixed')?
Usage: python3 -I stable_s1_dup.py S1.jsonl REQUESTS.jsonl OUT_DIR"""
import json, sys, collections
from pathlib import Path
s1p, reqp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
nfixed, fixedcats = {}, collections.Counter()
for line in reqp.open():
    r = json.loads(line)
    u = json.loads(r['messages'][1]['content'])
    nfixed[r['uid']] = len(u.get('fixed', []))
    for f in u.get('fixed', []):
        fixedcats[f['category']] += 1
def b(n): return '<=10' if n <= 10 else '11-20' if n <= 20 else '21-50' if n <= 50 else '>50'
def bf(k): return '0' if k == 0 else '1' if k == 1 else '2-3' if k <= 3 else '4+'
tab = collections.defaultdict(lambda: collections.Counter())
fin = collections.defaultdict(lambda: collections.Counter())
ftab = collections.defaultdict(lambda: collections.Counter())
for line in s1p.open():
    r = json.loads(line)
    if not r.get('complete'):
        continue
    lay = r['raw_layout']; seen = set(); dup = 0
    for o in lay:
        k = (round(o['pos'][0], 3), round(o['pos'][1], 3), round(o['pos'][2], 3), round(o['yaw'], 4))
        dup += k in seen; seen.add(k)
    s = r['s1']; key = (b(len(lay)), 'dup' if dup else 'nodup')
    tab[key]['n'] += 1; tab[key]['raw'] += bool(r['geometry_success']); tab[key]['s1'] += bool(s['geometry_success_s1'])
    tab[key]['strict_s1'] += bool(s.get('accepted_s1'))
    if not s['geometry_success_s1']:
        g = s['geometry_s1']
        for comp in ('main_collisions', 'severe_collisions', 'oob_objects', 'fixed_blocking', 'support_fail', 'ceiling_fail'):
            fin[key[0]][comp] += bool(g.get(comp))
        fin[key[0]]['n_fail'] += 1
        fin[key[0]]['main_sum'] += g.get('main_collisions', 0)
    fk = (b(len(lay)), bf(nfixed.get(r['uid'], 0)))
    ftab[fk]['n'] += 1; ftab[fk]['raw'] += bool(r['geometry_success']); ftab[fk]['s1'] += bool(s['geometry_success_s1'])
lines = ['fixed categories in test requests: ' + json.dumps(fixedcats.most_common(12))]
lines.append('count x dup: rooms | raw | S1 | S1+strict')
for nb in ['<=10', '11-20', '21-50', '>50']:
    for dk in ['nodup', 'dup']:
        c = tab[(nb, dk)]
        if c['n']:
            lines.append(f"  n {nb:>5} {dk:>5}: {c['n']:5d} | raw {c['raw']/c['n']:.3f} | S1 {c['s1']/c['n']:.3f} | strict {c['strict_s1']/c['n']:.3f}")
lines.append('after S1 still failing, share of failing rooms with each component (and mean main collisions):')
for nb in ['<=10', '11-20', '21-50', '>50']:
    c = fin[nb]
    if c['n_fail']:
        lines.append(f"  n {nb:>5}: fail {c['n_fail']:5d} | " + ' '.join(f"{k} {c[k]/c['n_fail']:.2f}" for k in ('main_collisions', 'severe_collisions', 'oob_objects', 'fixed_blocking', 'support_fail', 'ceiling_fail')) + f" | mean main {c['main_sum']/c['n_fail']:.1f}")
lines.append('count x number of fixed boxes in request: rooms | raw | S1')
for nb in ['<=10', '11-20', '21-50', '>50']:
    for fb in ['0', '1', '2-3', '4+']:
        c = ftab[(nb, fb)]
        if c['n']:
            lines.append(f"  n {nb:>5} fixed {fb:>3}: {c['n']:5d} | raw {c['raw']/c['n']:.3f} | S1 {c['s1']/c['n']:.3f}")
(out / 'summary-s1.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
