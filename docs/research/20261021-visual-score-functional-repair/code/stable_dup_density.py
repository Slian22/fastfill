"""Does the 27B duplicate/collision failure scale with object COUNT (sequence length) or with room DENSITY?
If count dominates at fixed density, STABLE-style staged generation (fewer objects per call) has a basis.
Usage: python3 -I stable_dup_density.py GEOMETRY.jsonl ROOMS.jsonl OUT_DIR
"""
import json, sys, math, collections
from pathlib import Path
from shapely.geometry import Polygon

geo, rooms_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
area = {}
for line in rooms_p.open():
    r = json.loads(line)
    try:
        area[r['uid']] = Polygon(r['boundary']).area
    except Exception:
        pass

def fp_area(o):
    return o['size'][0] * o['size'][1]

rows = []
for line in geo.open():
    r = json.loads(line)
    if not r.get('complete') or r['uid'] not in area:
        continue
    lay = r['raw_layout']
    n = len(lay)
    roots = [o for o in lay if not o.get('parent')]
    dens = sum(fp_area(o) for o in roots) / area[r['uid']]
    seen, dup, samecat = set(), 0, 0
    key_cat = {}
    for o in lay:
        k = (round(o['pos'][0], 3), round(o['pos'][1], 3), round(o['pos'][2], 3), round(o['yaw'], 4))
        if k in seen:
            dup += 1
            if key_cat.get(k) == o['category']:
                samecat += 1
        seen.add(k); key_cat.setdefault(k, o['category'])
    rows.append({'uid': r['uid'], 'n': n, 'area': area[r['uid']], 'density': dens, 'dup': dup,
                 'dup_room': int(dup > 0), 'success': int(bool(r['geometry_success'])),
                 'main': r.get('main_collisions'), 'oob': r.get('oob_objects'),
                 'fixed_blocking': r.get('fixed_blocking'), 'n_roots': len(roots)})

def bucket_n(n):
    return '<=10' if n <= 10 else '11-20' if n <= 20 else '21-50' if n <= 50 else '>50'
def bucket_d(d):
    return '<0.2' if d < 0.2 else '0.2-0.35' if d < 0.35 else '0.35-0.5' if d < 0.5 else '>=0.5'
def bucket_a(a):
    return '<12' if a < 12 else '12-20' if a < 20 else '20-35' if a < 35 else '>=35'

tab = collections.defaultdict(list)
for r in rows:
    tab[(bucket_n(r['n']), bucket_d(r['density']))].append(r)
lines = ['count x density (root footprint area / room area): rooms | dup-room share | raw checker success']
for nb in ['<=10', '11-20', '21-50', '>50']:
    for db in ['<0.2', '0.2-0.35', '0.35-0.5', '>=0.5']:
        rs = tab.get((nb, db), [])
        if rs:
            lines.append(f'  n {nb:>5} dens {db:>8}: {len(rs):5d} | dup {sum(r["dup_room"] for r in rs)/len(rs):.3f} | succ {sum(r["success"] for r in rs)/len(rs):.3f}')
tab2 = collections.defaultdict(list)
for r in rows:
    tab2[(bucket_n(r['n']), bucket_a(r['area']))].append(r)
lines.append('count x room area (m2): rooms | dup-room share | raw checker success')
for nb in ['<=10', '11-20', '21-50', '>50']:
    for ab in ['<12', '12-20', '20-35', '>=35']:
        rs = tab2.get((nb, ab), [])
        if rs:
            lines.append(f'  n {nb:>5} area {ab:>5}: {len(rs):5d} | dup {sum(r["dup_room"] for r in rs)/len(rs):.3f} | succ {sum(r["success"] for r in rs)/len(rs):.3f}')
# simple logistic-ish: within 21-50, split density at median
mid = [r for r in rows if 21 <= r['n'] <= 50]
md = sorted(r['density'] for r in mid)[len(mid)//2]
lo = [r for r in mid if r['density'] < md]; hi = [r for r in mid if r['density'] >= md]
lines.append(f'21-50 objects: density median {md:.3f}; low-dens dup {sum(r["dup_room"] for r in lo)/len(lo):.3f} succ {sum(r["success"] for r in lo)/len(lo):.3f} (n={len(lo)}); high-dens dup {sum(r["dup_room"] for r in hi)/len(hi):.3f} succ {sum(r["success"] for r in hi)/len(hi):.3f} (n={len(hi)})')
# position of first duplicate in the sequence (index / n)
pos = []
for line in geo.open():
    r = json.loads(line)
    if not r.get('complete'): continue
    seen = set()
    for k_i, o in enumerate(r['raw_layout']):
        k = (round(o['pos'][0], 3), round(o['pos'][1], 3), round(o['pos'][2], 3), round(o['yaw'], 4))
        if k in seen:
            pos.append(k_i); break
        seen.add(k)
if pos:
    import statistics
    lines.append(f'first duplicate index: median {statistics.median(pos)}, p10 {sorted(pos)[len(pos)//10]}, p90 {sorted(pos)[9*len(pos)//10]} (rooms with dup: {len(pos)})')
# collisions per room vs count of unique positions: in failing dup rooms, are collisions mostly duplicate pairs?
(out / 'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
(out / 'summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
