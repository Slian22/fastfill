import sys, time, json, hashlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from judge_lib import *
R20 = ['s2-render/render-0', 's2-render/render-1']
picks = [p['uid'] for p in json.load(open(P / 's2-render/picks.json'))]
def step(name, *a, **k):
    if (J / 'STOP').exists(): raise SystemExit('stopped')
    run(name, *a, **k); print(time.strftime('%H:%M:%S'), 'done', name, counts(), flush=True)
step('ref-vs-raw', R20, 'reference', 'raw', picks)
step('S1-vs-S2', R20, 'S1', 'S2', picks)
flip10 = sorted(picks, key=lambda u: hashlib.sha256(('flip' + u).encode()).hexdigest())[:10]
step('S1-vs-S2-flipped', R20, 'S1', 'S2', flip10, flip=True)
done = set()
t0 = time.time()
while len(done) < 2 and time.time() - t0 < 6 * 3600:
    for s in 'AB':
        L = [f's2f/render-{s}-0', f's2f/render-{s}-1']
        if s in done or not all((P / l / 'ledger.json').exists() for l in L): continue
        cs = load_cases(L)
        if not cs or any(c.get('status') == 'pending' for c in json.load(open(P / L[0] / 'ledger.json'))['cases']): continue
        uids = list(cs)
        step(f'S2F-vs-S1-{s}', L, 'S2F', 'S1', uids); step(f'S2F-vs-S2-{s}', L, 'S2F', 'S2', uids)
        if s == 'B': step('ref-vs-S1-B', L, 'reference', 'S1', uids)
        done.add(s)
    if len(done) < 2: time.sleep(600)
