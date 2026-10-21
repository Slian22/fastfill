"""Probe gt-prior-size: is the room still feasible at the GT arrangement once sizes are the train prior?
GT poses/yaw/parents kept; sizes replaced by eval-data size-estimates (the delivered prior); children's z re-seated
on the parent's new top. Checker = geometry.geometry (frozen). Also footprint density (sum root footprint / area)."""
import collections, copy, json, sys
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from fastfill.comparison import geometry as g

E = '/Volumes/harddisk/fastfillout/unknownsize-20261013/visual-final-selected-002/inputs/eval-data-v1'
OUT = '/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021/gt-prior-size'

def bin_(n): return '1-10' if n <= 10 else '11-20' if n <= 20 else '21-50' if n <= 50 else '51+'

def reseat(room):
    by = {o['id']: o for o in room['objects'] + (room.get('fixed') or [])}
    done = set()
    def top(i):
        o = by[i]
        if o.get('parent') in by and i not in done and o in room['objects']:
            p = by[o['parent']]; top(o['parent'])
            o['pos'] = [o['pos'][0], o['pos'][1], p['pos'][2] + p['size'][2]]
        done.add(i)
    for o in room['objects']: top(o['id'])

def density(room):
    a = Polygon(room['boundary']).area
    return sum(o['size'][0] * o['size'][1] for o in room['objects'] if not o.get('parent')) / a if a else 0

def one(args):
    room, est = args
    gt = g.geometry(room)
    pr = copy.deepcopy(room)
    for o in pr['objects']: o['size'] = list(est[o['id']])
    reseat(pr)
    p = g.geometry(pr)
    return {'uid': room['uid'], 'n': len(room['objects']), 'gt_pass': gt['geometry_success'], 'prior_pass': p['geometry_success'],
            'gt_main': gt['main_collisions'], 'prior_main': p['main_collisions'], 'prior_oob': p['oob_objects'],
            'dens_gt': density(room), 'dens_prior': density(pr)}

if __name__ == '__main__':
    est = {r['uid']: r['estimated_sizes'] for r in map(json.loads, open(f'{E}/size-estimates.jsonl'))}
    rooms = [r for r in map(json.loads, open(f'{E}/rooms.jsonl')) if r['uid'] in est]
    with ProcessPoolExecutor(8) as ex: rows = list(ex.map(one, [(r, est[r['uid']]) for r in rooms], chunksize=32))
    with open(f'{OUT}/rows.jsonl', 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for b in (bin_(r['n']), 'all'):
            c = agg[b]; c['rooms'] += 1; c['gt'] += r['gt_pass']; c['prior'] += r['prior_pass']
            c['gt_and_prior'] += r['gt_pass'] and r['prior_pass']; c['dg'] += r['dens_gt']; c['dp'] += r['dens_prior']
    lines = ['| bin | rooms | GT sizes checker | prior sizes checker (same GT poses) | GT-pass rooms still passing | root density GT -> prior |', '|---|---:|---:|---:|---:|---:|']
    for b in ('1-10', '11-20', '21-50', '51+', 'all'):
        c = agg[b]; n = c['rooms']
        lines.append(f"| {b} | {n} | {c['gt']/n:.3f} | {c['prior']/n:.3f} | {c['gt_and_prior']/max(1,c['gt']):.3f} | {c['dg']/n:.2f} -> {c['dp']/n:.2f} |")
    open(f'{OUT}/report.md', 'w').write('# GT poses with prior sizes (27B test rooms)\n\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines))
