"""Per-tier object counts (what a staged placer would see per call) and where exact duplicates sit, on the 27B test raw layouts."""
import json, sys, statistics, collections
sys.path.insert(0, '/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/852ae94c-d667-446b-a2a1-ce6deaf82ed6/scratchpad/probes')
from stable_staged_s1 import tiers
geo = sys.argv[1]
by_b = collections.defaultdict(list); dup_tier = collections.Counter(); objs_tier = collections.Counter()
def b(n): return '<=10' if n <= 10 else '11-20' if n <= 20 else '21-50' if n <= 50 else '>50'
for line in open(geo):
    r = json.loads(line)
    if not r.get('complete'): continue
    lay = r['raw_layout']; t, root = tiers(lay); sizes = [len(x) for x in t]
    by_b[b(len(lay))].append(sizes)
    tier_of = {i: k for k, s in enumerate(t, 1) for i in s}
    seen = set()
    for o in lay:
        k = (round(o['pos'][0], 3), round(o['pos'][1], 3), round(o['pos'][2], 3), round(o['yaw'], 4))
        objs_tier[(b(len(lay)), tier_of[o['id']])] += 1
        if k in seen: dup_tier[(b(len(lay)), tier_of[o['id']])] += 1
        seen.add(k)
for nb in ['<=10', '11-20', '21-50', '>50']:
    rows = by_b[nb]
    med = [statistics.median(x[k] for x in rows) for k in range(3)]
    p90 = [sorted(x[k] for x in rows)[int(0.9 * len(rows))] for k in range(3)]
    dup = [f"{dup_tier[(nb, k)] / max(1, objs_tier[(nb, k)]):.3f}" for k in (1, 2, 3)]
    print(f"n {nb:>5}: rooms {len(rows):5d} | median objects per tier {med} | p90 {p90} | duplicate share of objects by tier {dup}")
