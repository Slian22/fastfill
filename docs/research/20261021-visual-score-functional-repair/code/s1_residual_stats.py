"""What remains after S1 on the 27B test outputs: counts by stratum, residual failure components, not-XY-fixable flags,
co-located (exact duplicate) pairs among S1-failing rooms. Read-only."""
import collections, json, math, sys
p = '/Volumes/harddisk/fastfillout/unknownsize-20261013/a026-inputs/s1-022/placement-arm/placement-geometry-s1.jsonl'
strat = lambda n: '<=10' if n <= 10 else '11-20' if n <= 20 else '21-50' if n <= 50 else '>50'
tot = collections.Counter(); fail = collections.Counter(); comp = collections.Counter(); notxy = collections.Counter()
dup_rooms = collections.Counter(); strict_fail = collections.Counter(); strict_reasons = collections.Counter()
only_notxy = 0; timed = 0; budget = 0; rt = []
for line in open(p):
    r = json.loads(line); s = r['s1']; k = strat(r['objects']); tot[k] += 1
    if s['status'] == 'not_complete': fail[k] += 1; comp['not_complete'] += 1; continue
    if not s['geometry_success_s1']:
        fail[k] += 1
        fin = s['vectors']['final']
        for c in ('oob','support_fail','ceiling','fixed_blocking','main','severe'):
            if fin[c]: comp[c] += 1
        nx = s['raw_failures']['not_xy_fixable']
        for c, v in nx.items(): notxy[c] += 1
        if nx and not any(fin[c] for c in ('oob','fixed_blocking','main','severe')): only_notxy += 1
        timed += s['timed_out']; budget += s['budget_exhausted']; rt.append(s['runtime_s'])
        lay = s['s1_layout'] if False else r['s1_layout']
        seen = set(); dup = False
        for o in lay:
            key = (round(o['pos'][0],3), round(o['pos'][1],3), round(o['yaw'],4))
            if key in seen: dup = True; break
            seen.add(key)
        dup_rooms[k] += dup
    if s['geometry_success_s1'] and not s['strict_ok_s1']:
        strict_fail[k] += 1
        for reason in s['downstream_s1']['reasons']: strict_reasons[reason] += 1
print('total by stratum', dict(tot))
print('S1 checker-fail by stratum', dict(fail), 'sum', sum(fail.values()))
print('residual components among S1-fail rooms (room counts)', dict(comp))
print('not_xy_fixable flags among S1-fail rooms', dict(notxy), 'rooms failing ONLY for not-xy reasons', only_notxy)
print('S1-fail rooms with an exact co-located pair', dict(dup_rooms))
print('timed_out', timed, 'budget_exhausted', budget, 'runtime mean', sum(rt)/len(rt), 'p95', sorted(rt)[int(.95*len(rt))])
print('checker-pass but strict-fail by stratum', dict(strict_fail), 'reasons', dict(strict_reasons))
