"""Probe g1-layoutvlm: LayoutVLM-style differentiable joint repair of the delivered 27B test outputs (free, CPU).

LayoutVLM's solver (sunfanyunn/LayoutVLM grad_solver.py): Adam over every asset's pose, overlap loss on oriented boxes,
boundary projection every 100 steps, best-state tracking, 400 steps. Here the same machinery is used as a REPAIR
operator with FastFill's invariants: XY of whole support units (children ride on their root), z/parent/size frozen,
yaw frozen (or --yaw: childless floor units only), losses = SAT penetration of the pairs the frozen checker counts
(+ fixed non-window boxes), corner distance outside the polygon, optional strict terms (A4 AABB in the P2 frame incl.
tucked pairs, door keep-out strips), and a weak anchor to the raw pose (minimum displacement, relation preservation).
Co-located exact duplicates get a deterministic 2 cm jitter (zero gradient otherwise). Output rounded to mm / integer
degrees, audited by geometry.geometry (checker), downstream.accept (strict), held against_wall relations, then S1
(output_repair.repair) is run on top (G1+S1) as the pipeline would. S1-alone verdicts come from the S1 rows.

python -I g1_layoutvlm_repair.py --out DIR [--limit N] [--mode checker|strict] [--yaw] [--workers W] [--uids FILE]
"""
import argparse
import collections
import hashlib
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
import torch                                                        # noqa: E402
from shapely.geometry import LineString, Point, Polygon             # noqa: E402

from fastfill.anchors import FLAT                                   # noqa: E402
from fastfill.comparison import downstream as d                     # noqa: E402
from fastfill.comparison import geometry as g                       # noqa: E402
from fastfill.comparison import output_repair as s1mod              # noqa: E402
from fastfill.comparison.best_of_n import VIOLATIONS                # noqa: E402
from fastfill.scene import footprint, head                          # noqa: E402
from fastfill.validate import collision_kind, holds                 # noqa: E402

BASE = '/Volumes/harddisk/fastfillout/unknownsize-20261013'
S1_ROWS = BASE + '/a026-inputs/s1-022/placement-arm/placement-geometry-s1.jsonl'
REQUESTS = BASE + '/visual-final-selected-002/inputs/eval-data-v1/placement/requests.jsonl'
ROOMS = BASE + '/visual-final-selected-002/inputs/eval-data-v1/rooms.jsonl'
TOL, CLEAR, MIN_H = 0.05, 0.01, 0.06                                # validate.check tolerances, S1 CLEAR
HP = {'steps': 400, 'lr': 0.02, 'lr_end': 0.001, 'w_pen': 20.0, 'w_oob': 20.0, 'w_wall': 20.0, 'w_anchor': 0.1,
      'w_yaw': 0.5, 'jitter': 0.02, 'snapshot': 20}


def _corners(xy, yaw, size):
    """[n,4,2] oriented footprint corners from [n,2] centres, [n] yaw, [n,2] size (fastfill.scene.footprint order)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    hx, hy = size[:, 0] / 2, size[:, 1] / 2
    signs = torch.tensor([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
    dx, dy = signs[None, :, 0] * hx[:, None], signs[None, :, 1] * hy[:, None]
    return torch.stack([xy[:, 0:1] + c[:, None] * dx - s[:, None] * dy,
                        xy[:, 1:2] + s[:, None] * dx + c[:, None] * dy], dim=-1)


def _penetration(ca, cb, axes):
    """[P] SAT penetration depth (min over the given axes of the projection overlap) of corner sets ca, cb [P,4,2]."""
    pa, pb = torch.einsum('pkd,pad->pka', ca, axes), torch.einsum('pkd,pad->pka', cb, axes)
    overlap = torch.minimum(pa.max(1).values, pb.max(1).values) - torch.maximum(pa.min(1).values, pb.min(1).values)
    return overlap.min(1).values


def _seg_dist(points, segs):
    """[N] distance from points [N,2] to the nearest of segments [M,2,2]."""
    a, b = segs[None, :, 0], segs[None, :, 1]
    ab = b - a
    t = (((points[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return ((a + t[..., None] * ab - points[:, None]).norm(dim=-1)).min(1).values


def _inside(points, segs):
    """[N] even-odd point-in-polygon test of points [N,2] against the ring edges segs [E,2,2] (no gradient)."""
    with torch.no_grad():
        p, a, b = points[:, None], segs[None, :, 0], segs[None, :, 1]
        cross = (a[..., 1] > p[..., 1]) != (b[..., 1] > p[..., 1])
        x = (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) / (b[..., 1] - a[..., 1] + 1e-12) + a[..., 0]
        return ((cross & (p[..., 0] < x)).sum(1) % 2) == 1


class Problem:
    def __init__(self, room, mode, yaw, held_ids=()):
        objs = room['objects']
        self.room, self.objs, self.mode = room, objs, mode
        by = {o['id']: o for o in objs}
        ids = [o['id'] for o in objs]
        self.index = {i: k for k, i in enumerate(ids)}
        root = {}
        for o in objs:
            path, cur = [o['id']], o.get('parent')
            while cur in by and cur not in path:
                path.append(cur)
                cur = by[cur].get('parent')
            root[o['id']] = path[-1]
        self.root = root
        kids = collections.Counter(root.values())
        # a unit is movable iff its root stands on the floor (parent None); a root on a fixed/unknown parent is frozen
        self.units = [i for i in ids if root[i] == i and by[i].get('parent') is None]
        self.unit_of = {i: self.units.index(root[i]) if root[i] in self.units else -1 for i in ids}
        # held against_wall (true on raw, S1's hard component): the unit may slide along its wall, not leave it
        ring = list(Polygon(room['boundary']).exterior.coords)
        edges = [(ring[k], ring[k + 1]) for k in range(len(ring) - 1) if math.dist(ring[k], ring[k + 1]) > 1e-9]
        wall = {}
        for i in held_ids:
            c = Point(by[i]['pos'][:2])
            a, b = min(edges, key=lambda e: c.distance(LineString(e)))
            length = math.dist(a, b)
            wall.setdefault(self.unit_of[i], []).append(((a[1] - b[1]) / length, (b[0] - a[0]) / length))
        self.wall_units = torch.tensor(sorted(wall), dtype=torch.long)
        self.wall_normals = torch.tensor([wall[u][0] for u in sorted(wall)]).reshape(-1, 2)
        self.yaw_units = [u for u in self.units if yaw and kids[u] == 1 and self.units.index(u) not in wall]
        n = len(objs)
        self.xy0 = torch.tensor([o['pos'][:2] for o in objs])
        self.yaw0 = torch.tensor([float(o['yaw']) for o in objs])
        self.size = torch.tensor([o['size'][:2] for o in objs])
        self.member = torch.tensor([self.unit_of[i] for i in ids])
        fps = [Polygon(footprint(o)) for o in objs]
        top = lambda o: o['pos'][2] + o['size'][2]
        solid = lambda o: head(o) not in FLAT and o['size'][2] >= MIN_H
        pairs, strict_pairs = [], []
        for a in range(n):
            for b in range(a + 1, n):
                oa, ob = objs[a], objs[b]
                if self.unit_of[ids[a]] == self.unit_of[ids[b]]:
                    continue                                    # rigid: a unit translation cannot change the pair
                if mode == 'strict' and oa.get('parent') == ob.get('parent') \
                        and not d.covering(oa, oa.get('parent')) and not d.covering(ob, ob.get('parent')):
                    strict_pairs.append((a, b))                 # A4: same parent, AABB overlap > 1 cm, height ignored
                if min(top(oa), top(ob)) - max(oa['pos'][2], ob['pos'][2]) <= TOL or not solid(oa) or not solid(ob):
                    continue
                kind = collision_kind(oa, ob, fps[a], fps[b], TOL)
                if kind == 'tuck' or (kind == 'contained' and fps[a].intersects(fps[b])):
                    continue                                    # tolerated by the checker on raw; S1 does the same
                pairs.append((a, b))
        self.pairs = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
        self.strict_pairs = torch.tensor(strict_pairs, dtype=torch.long).reshape(-1, 2)
        fixed = [f for f in room.get('fixed') or []
                 if not (s1mod.WINDOW_WORDS & set(s1mod.norm_cat(f.get('category')).split()))]
        hz = [(k, f) for k in range(n) for f in fixed if solid(objs[k])
              and min(top(objs[k]), top(f)) - max(objs[k]['pos'][2], f['pos'][2]) > TOL]
        self.hz_obj = torch.tensor([k for k, _ in hz], dtype=torch.long)
        self.hz_corners = torch.tensor([footprint(f) for _, f in hz]).reshape(-1, 4, 2)
        self.hz_yaw = torch.tensor([float(f['yaw']) for _, f in hz])
        self.segs = torch.tensor([[ring[k], ring[k + 1]] for k in range(len(ring) - 1)])
        self.bounded = room.get('boundary_type') in ('polygon', 'hull')
        strips = [part for s in d.door_strips(room) for part in getattr(s['strip'], 'geoms', [s['strip']])
                  if part.geom_type == 'Polygon' and part.area > 0] if mode == 'strict' else []
        self.strip_segs = [torch.tensor([[c[k], c[k + 1]] for c in [list(s.exterior.coords)]
                                         for k in range(len(c) - 1)]) for s in strips]
        theta = d.frame(room['boundary'])[0]
        self.frame_axes = torch.tensor([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])

    def jitter(self):
        """Deterministic 2 cm symmetry breaking for exactly co-located, non-rigid pairs (SAT has no gradient there)."""
        shift = torch.zeros(len(self.units), 2)
        for a, b in self.pairs.tolist():
            if math.dist(self.objs[a]['pos'][:2], self.objs[b]['pos'][:2]) < 1e-3:
                u = self.unit_of[self.objs[b]['id']]
                ang = int(hashlib.sha256(self.objs[b]['id'].encode()).hexdigest(), 16) % 360
                shift[u] += HP['jitter'] * torch.tensor([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        return shift

    def losses(self, shift, dyaw):
        """(penetration-type violation count, violation sum, anchor, loss) at unit shifts [U,2] and unit yaw deltas [U]."""
        xy = self.xy0 + shift[self.member]
        yaw = self.yaw0 + dyaw[self.member]
        corners = _corners(xy, yaw, self.size)
        zero = torch.zeros(())
        viol, total, count = [], zero, 0
        if len(self.pairs):
            a, b = self.pairs[:, 0], self.pairs[:, 1]
            axes = torch.stack([torch.stack([torch.cos(y), torch.sin(y)], -1) for y in (yaw[a], yaw[b])] +
                               [torch.stack([-torch.sin(y), torch.cos(y)], -1) for y in (yaw[a], yaw[b])], dim=1)
            pen = torch.relu(_penetration(corners[a], corners[b], axes) + CLEAR)
            total, count = total + HP['w_pen'] * pen.sum(), count + int((pen > CLEAR).sum())
        if len(self.strict_pairs):
            a, b = self.strict_pairs[:, 0], self.strict_pairs[:, 1]
            axes = self.frame_axes[None].expand(len(a), 2, 2)
            pen = torch.relu(_penetration(corners[a], corners[b], axes) - d.TOL_M + CLEAR)
            total, count = total + HP['w_pen'] * pen.sum(), count + int((pen > CLEAR).sum())
        if len(self.hz_obj):
            k = self.hz_obj
            axes = torch.stack([torch.stack([torch.cos(y), torch.sin(y)], -1) for y in (yaw[k], self.hz_yaw)] +
                               [torch.stack([-torch.sin(y), torch.cos(y)], -1) for y in (yaw[k], self.hz_yaw)], dim=1)
            pen = torch.relu(_penetration(corners[k], self.hz_corners, axes) + CLEAR)
            total, count = total + HP['w_pen'] * pen.sum(), count + int((pen > CLEAR).sum())
        if self.bounded:
            pts = corners.reshape(-1, 2)
            outside = ~_inside(pts, self.segs)
            if outside.any():
                dist = _seg_dist(pts[outside], self.segs)
                total, count = total + HP['w_oob'] * dist.sum(), count + int(outside.sum())
            for segs in self.strip_segs:
                inside = _inside(pts, segs)
                if inside.any():
                    dist = _seg_dist(pts[inside], segs) + CLEAR
                    total, count = total + HP['w_oob'] * dist.sum(), count + int(inside.sum())
        if len(self.wall_units):
            total = total + HP['w_wall'] * (shift[self.wall_units] * self.wall_normals).sum(-1).abs().sum()
        anchor = HP['w_anchor'] * (shift[self.member] ** 2).sum() + HP['w_yaw'] * (1 - torch.cos(dyaw[self.member])).sum()
        return count, total, anchor, total + anchor

    def solve(self):
        if not self.units:
            return {o['id']: (0.0, 0.0, 0.0) for o in self.objs}, {'steps': 0, 'best_step': 0}
        shift = torch.nn.Parameter(self.jitter())
        dyaw = torch.nn.Parameter(torch.zeros(len(self.units)))
        yaw_mask = torch.tensor([u in self.yaw_units for u in self.units], dtype=torch.float32)
        params = [shift] + ([dyaw] if self.yaw_units else [])
        opt = torch.optim.Adam(params, lr=HP['lr'])
        gamma = (HP['lr_end'] / HP['lr']) ** (1 / HP['steps'])
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
        best = None
        for step in range(HP['steps'] + 1):
            count, total, anchor, loss = self.losses(shift, dyaw * yaw_mask)
            if step % HP['snapshot'] == 0 or step == HP['steps']:
                key = (count, float(total.detach()), float(anchor.detach()))
                if best is None or key < best[0]:
                    best = (key, shift.detach().clone(), dyaw.detach().clone() * yaw_mask, step)
                if count == 0 and float(total) == 0.0:
                    break
            if step == HP['steps']:
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        _, s, y, at = best
        out = {}
        for o in self.objs:
            u = self.unit_of[o['id']]
            out[o['id']] = (0.0, 0.0, 0.0) if u < 0 else (float(s[u, 0]), float(s[u, 1]), float(y[u]))
        return out, {'steps': step, 'best_step': at, 'best_key': list(best[0])}


def apply(objs, moves):
    """Raw + whole millimetres (S1's serialisation rule); yaw to integer degrees only when it changed."""
    out = []
    for o in objs:
        dx, dy, dyaw = moves[o['id']]
        mx, my = round(dx * 1000), round(dy * 1000)
        deg = round(math.degrees((o['yaw'] + dyaw) % math.tau)) % 360 if dyaw else None
        yaw = o['yaw'] if deg is None or deg == round(math.degrees(o['yaw'] % math.tau)) % 360 else math.radians(deg)
        if (mx, my) == (0, 0) and yaw == o['yaw']:
            out.append(o)
            continue
        out.append({**o, 'pos': [round(o['pos'][0] + mx / 1000, 6), round(o['pos'][1] + my / 1000, 6), o['pos'][2]],
                    'yaw': yaw})
    return out


def audit(room, objs, held):
    lay = {**room, 'objects': objs}
    geo = g.geometry(lay)
    verdict = d.accept(lay, objs)
    by = {o['id']: o for o in objs}
    return {'checker': geo['geometry_success'], 'violations': sum(geo[k] for k in VIOLATIONS),
            'strict': verdict['accepted'], 'strict_reasons': verdict['reasons'],
            'held_broken': sum(not holds(k, objs, room['boundary'], by) for k in held),
            'geometry': {k: geo[k] for k in VIOLATIONS}}


def one(task):
    row, request, room, mode, yaw = task
    pred = s1mod.predicted_room(room, request['messages'], row['raw_layout'])
    by = {o['id']: o for o in pred['objects']}
    held = [['against_wall', i] for i in by if holds(['against_wall', i], pred['objects'], pred['boundary'], by)]
    t0 = time.monotonic()
    problem = Problem(pred, mode, yaw, [k[1] for k in held])
    moves, info = problem.solve()
    g1 = apply(pred['objects'], moves)
    t1 = time.monotonic()
    a_g1 = audit(pred, g1, held)
    res = s1mod.repair({**pred, 'objects': g1}, max_trials=2000, time_limit=60.0)
    g1s1 = res['room']['objects']
    t2 = time.monotonic()
    a_g1s1 = audit(pred, g1s1, held)
    disp = [math.dist(a['pos'][:2], b['pos'][:2]) for a, b in zip(pred['objects'], g1s1)]
    s1 = row['s1']
    return {'uid': row['uid'], 'source': row['source'], 'objects': row['objects'], 'stratum': s1['stratum'],
            'doors': s1['doors'], 'rectangle': s1['rectangle'], 'mode': mode, 'yaw': yaw,
            'raw': {'checker': row['geometry_success'], 'strict': s1['strict_ok_raw']},
            's1': {'checker': s1['geometry_success_s1'], 'strict': s1['strict_ok_s1'], 'runtime_s': s1['runtime_s'],
                   'violations': sum(s1['geometry_s1'][k] for k in VIOLATIONS)},
            'g1': {**a_g1, 'runtime_s': t1 - t0, **info},
            'g1s1': {**a_g1s1, 'runtime_s': t2 - t1, 's1_status': res['status'],
                     'rolled_back': res['diagnostics'].get('rolled_back')},
            'units': len(problem.units), 'pairs': int(len(problem.pairs)), 'strict_pairs': int(len(problem.strict_pairs)),
            'yaw_changed': sum(a['yaw'] != b['yaw'] for a, b in zip(pred['objects'], g1s1)),
            'displacement': {'moved_fraction': sum(x > 0 for x in disp) / len(disp), 'median': statistics.median(disp),
                             'p95': g._quantile(disp, .95), 'max': max(disp)},
            'g1s1_layout': g1s1}


def rate(rows, key, arm):
    return sum(bool(r[arm][key]) for r in rows) / len(rows) if rows else None


def summarize(rows):
    def block(rs):
        best = [max((r['s1'], r['g1s1']), key=lambda a: (a['checker'], a['strict'], -a['violations'])) for r in rs]
        return {'rooms': len(rs),
                **{f'{k}_{arm}': rate(rs, k, arm) for k in ('checker', 'strict') for arm in ('raw', 's1', 'g1', 'g1s1')},
                'checker_best_of_s1_g1s1': sum(b['checker'] for b in best) / len(rs) if rs else None,
                'strict_best_of_s1_g1s1': sum(b['strict'] for b in best) / len(rs) if rs else None,
                'held_broken_g1s1_rooms': sum(r['g1s1']['held_broken'] > 0 for r in rs),
                'yaw_changed_objects': sum(r['yaw_changed'] for r in rs),
                'runtime_s_mean': {'s1': statistics.mean(r['s1']['runtime_s'] for r in rs) if rs else None,
                                   'g1': statistics.mean(r['g1']['runtime_s'] for r in rs) if rs else None,
                                   'g1s1': statistics.mean(r['g1s1']['runtime_s'] for r in rs) if rs else None},
                'displacement_median_mean': statistics.mean(r['displacement']['median'] for r in rs) if rs else None,
                'displacement_p95_mean': statistics.mean(r['displacement']['p95'] for r in rs) if rs else None}
    strata = ['1-10', '11-30', '31-50', '51+']
    return {'all': block(rows), 'by_stratum': {k: block([r for r in rows if r['stratum'] == k]) for k in strata},
            'by_doors': {k: block([r for r in rows if (r['doors'] > 0) == (k == 'doors')]) for k in ('no_doors', 'doors')},
            'by_shape': {k: block([r for r in rows if r['rectangle'] == (k == 'rectangle')])
                         for k in ('rectangle', 'non_rectangle')},
            's1_fail_only': block([r for r in rows if not r['s1']['checker']])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='first N raw-failing rooms in sha256(uid) order (0 = all)')
    ap.add_argument('--mode', choices=('checker', 'strict'), default='checker')
    ap.add_argument('--yaw', action='store_true')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--uids', type=Path)
    ap.add_argument('--select', choices=('raw_fail', 's1_fail'), default='raw_fail')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = {}
    for line in open(S1_ROWS):
        r = json.loads(line)
        if r['complete'] and not r['geometry_success'] and (args.select == 'raw_fail' or not r['s1']['geometry_success_s1']):
            rows[r['uid']] = r
    uids = sorted(rows, key=lambda u: hashlib.sha256(u.encode()).hexdigest())
    if args.uids:
        given = json.loads(args.uids.read_text())
        uids = [u for u in (given.get('uids') if isinstance(given, dict) else given) if u in rows]
    if args.limit:
        uids = uids[:args.limit]
    keep = set(uids)
    requests = {r['uid']: r for r in map(json.loads, open(REQUESTS)) if r['uid'] in keep}
    rooms = {r['uid']: r for r in map(json.loads, open(ROOMS)) if r['uid'] in keep}
    tasks = [(rows[u], requests[u], rooms[u], args.mode, args.yaw) for u in uids]
    t0 = time.monotonic()
    if args.workers == 1:
        results = [one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(one, tasks, chunksize=2))
    with open(args.out / 'rows.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    summary = {'probe': 'g1-layoutvlm-repair', 'mode': args.mode, 'yaw': args.yaw, 'select': args.select,
               'hyper': HP, 'rooms': len(results), 'wall_s': time.monotonic() - t0,
               'inputs': {'s1_rows': S1_ROWS, 'requests': REQUESTS, 'rooms': ROOMS}, 'summary': summarize(results)}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['summary']['all'], indent=1))


if __name__ == '__main__':
    main()
