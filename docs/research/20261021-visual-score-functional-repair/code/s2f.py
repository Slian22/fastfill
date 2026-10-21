"""S2-F: S2 (s2_protect.py: PolyLayout re-placement, wall protection, optional A4 strict target) plus companion
protection. Companion pairs (companions.py) are read from the layout being repaired, never from GT.

  (a) a bad root anchor moves with its good held companions as one unit (same S2 candidates, every member checked);
      if no feasible unit pose exists the anchor moves alone (S2) and each good companion left broken tries
      relation-keeping candidates (feasible only, else it stays: counted 'left_behind').
  (b) a bad companion first takes relation-keeping candidates around its anchor's current pose (adjacency: 5 cm grid
      within the 0.30 m ring, nightstand: head half and no footprint overlap with its bed; facing: 10 cm grid in the
      +-45 deg / 5 m cone; inside the envelope and clear of the anchor), nearest first, a walled companion's on-wall
      ones before its off-wall ones; only if none is feasible the S2 candidates (wall-filtered as S2; counted
      'fallbacks' when the relation ends up broken); if neither has a feasible pose, the min-cost keep pose.
  (c) in a colliding pair (checker main or strict A4) exactly one of which is in a held pair, the other one is blamed.
  Bad companions are placed right after their anchor. Children pass, wall protection, Phi, invariants: as S2.

python3 s2f.py --selfcheck
python3 s2f.py --pipeline --s1 S1.jsonl --data DATA --out OUT [--workers W]
    S1 -> S2F(checker, protect, rows failing checker) -> merge -> S2F(strict, protect, rows failing strict) -> merge;
    merge = adopt the polished S2F layout iff checker and strict are both not worse (as the S2 dev/render merges).
"""
import argparse
import collections
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from shapely.affinity import affine_transform, translate
from shapely.geometry import Polygon
from shapely.prepared import prep

sys.path.insert(0, str(Path(__file__).parent))
import companions as C  # noqa: E402
import s2_protect as S  # noqa: E402

KEEP_GRID, FACE_GRID, KEEP_CAP = 0.05, 0.10, 400
REPO = '/Users/slian/Desktop/3D/Worldedge/OptiScene'


def _imports(repo=REPO):
    S._imports(repo)


def _bad(room, keepouts, phi, in_pair):
    """S._bad with rule (c): of a colliding pair exactly one of which is in a held companion pair, blame the other."""
    by = {o['id']: o for o in room['objects']}
    order = {i: k for k, i in enumerate(by)}
    L = S.orp.audit(room, keepouts, [])['lists']
    bad = set(L['oob']) | set(L['support_fail']) | {o for o, _ in L['fixed_blocking']} | set(L['outside_1mm']) \
        | {i for i, _ in L['keepout_1cm']}
    key = lambda o: (tuple(round(v * 1000) for v in o['pos'][:2]), round(o['yaw'], 6))
    area = lambda o: o['size'][0] * o['size'][1]
    for a, b in list(L['main']) + (list(L['a4_aabb']) if phi == 'strict' else []):
        a, b = sorted((a, b), key=order.get)
        oa, ob = by[a], by[b]
        if key(oa) == key(ob):
            bad.add(b)
        elif (a in in_pair) != (b in in_pair):
            bad.add(b if a in in_pair else a)
        else:
            bad.add(b if area(ob) <= area(oa) else a)
    return bad


def _keep_candidates(cr, ar, pair, env, env_p, mtx):
    """(dx, dy) moving companion record cr to grid points where its pair with anchor record ar (current pose) holds,
    inside the envelope and clear of the anchor under the stage's cost (nightstand: no footprint overlap either),
    nearest first, at most KEEP_CAP."""
    c, a, kind = cr['obj'], ar['obj'], pair[0]
    x0, y0 = c['pos'][:2]
    if kind == 'face':
        R, step = C.FACE_DIST, FACE_GRID
        bx0, by0, bx1, by1 = a['pos'][0] - R, a['pos'][1] - R, a['pos'][0] + R, a['pos'][1] + R
    else:
        R, step = C.GAP + math.hypot(c['size'][0], c['size'][1]) / 2, KEEP_GRID
        bx0, by0, bx1, by1 = ar['poly'].buffer(R).bounds
    eb = env.bounds
    bx0, by0, bx1, by1 = max(bx0, eb[0]), max(by0, eb[1]), min(bx1, eb[2]), min(by1, eb[3])
    pts = []
    for i in range(int((bx1 - bx0) / step) + 1):
        for j in range(int((by1 - by0) / step) + 1):
            px, py = bx0 + i * step, by0 + j * step
            if not C.holds(pair, {c['id']: {**c, 'pos': [px, py, c['pos'][2]]}, a['id']: a}):
                continue
            p = translate(cr['poly'], px - x0, py - y0)
            if not env_p.covers(p):
                continue
            if S._overlap(cr, p, p.bounds, ar, affine_transform(p, mtx).bounds if mtx is not None else None) > 0:
                continue
            if kind == 'adj_head' and p.intersection(ar['poly']).area > S.MIN_AREA:
                continue
            pts.append(((px - x0) ** 2 + (py - y0) ** 2, px - x0, py - y0))
    for _, dx, dy in sorted(pts)[:KEEP_CAP]:
        yield dx, dy


def s2f(room, keepouts, deadline, phi='checker', protect=False):
    objs = room['objects']
    by = {o['id']: o for o in objs}
    fixed = {f['id']: f for f in room.get('fixed') or []}
    windows = {f['id'] for f in fixed.values() if {'window', 'windows'} & set(S.d.norm_cat(f.get('category')).split())}
    kids = collections.defaultdict(list)
    for o in objs:
        if o.get('parent') in by:
            kids[o['parent']].append(o['id'])
    roots = [o['id'] for o in objs if o.get('parent') not in by]

    def subtree(i):
        out, todo = [], [i]
        while todo:
            j = todo.pop()
            if j not in out:
                out.append(j)
                todo += kids[j]
        return out

    poly = Polygon(room['boundary'])
    s = S.orp._strict(room, keepouts)
    env = (s['floor'] if s and not s['floor'].is_empty else poly).buffer(S.orp.ENVELOPE_TOL)
    env_p = prep(env)
    edges = S._edges(room['boundary'])
    m = S.d.frame(room['boundary'])[1] if phi == 'strict' else None
    cur = {o['id']: dict(o) for o in objs}
    shift = {o['id']: (0.0, 0.0) for o in objs}
    fixed_recs = [S._rec({**f}, fixed=True) for f in fixed.values() if f['id'] not in windows]
    stats = {'bad_roots': 0, 'bad_children': 0, 'min_overlap': 0, 'timed_out': False,
             'group_moves': 0, 'group_failed': 0, 'keep_moves': 0, 'fallbacks': 0, 'rejoined': 0, 'left_behind': 0}

    def recs(ids):
        return [S._rec(cur[i], m, covering=cur[i].get('parent') not in by and cur[i]['size'][2] < S.d.FLOOR_COVERING_H_M)
                for i in ids]

    def move(ids, dx, dy):
        for i in ids:
            o = cur[i]
            cur[i] = {**o, 'pos': [round(o['pos'][0] + dx, 6), round(o['pos'][1] + dy, 6), o['pos'][2]]}
            shift[i] = (shift[i][0] + dx, shift[i][1] + dy)

    # companion pairs held in the layout being repaired (roots only); a companion has one anchor, anchors are no companions
    pairs = C.pairs(objs)
    pair_of = {p[1]: p for p in pairs}
    comps = collections.defaultdict(list)
    for p in pairs:
        comps[p[2]].append(p[1])
    in_pair = set(pair_of) | set(comps)
    zone = Polygon(room['boundary']).exterior.buffer(0.1 + 1e-9)
    walled = {r for r in roots if protect and S._on_wall(Polygon(S.footprint(cur[r])), zone)}

    def wall_ok(ids):
        return lambda c: all(S._on_wall(translate(Polygon(S.footprint(cur[i])), *c), zone) for i in ids if i in walled)

    def keep(c):
        """relation-keeping candidates; a walled companion takes on-wall ones first, then off-wall ones (the relation
        outranks wall contact for a companion; S2 candidates stay wall-filtered)"""
        p = pair_of[c]
        cr, ar = recs([c, p[2]])
        ks, ok = list(_keep_candidates(cr, ar, p, env, env_p, m)), wall_ok([c])
        return [k for k in ks if ok(k)] + [k for k in ks if not ok(k)]

    bad = _bad({**room, 'objects': [cur[o['id']] for o in objs]}, keepouts, phi, in_pair)
    area = lambda r: cur[r]['size'][0] * cur[r]['size'][1]
    # largest first (S2); a bad companion right after its bad anchor
    lead = lambda r: pair_of[r][2] if r in pair_of and pair_of[r][2] in bad else r
    bad_roots = sorted([r for r in roots if r in bad], key=lambda r: (-area(lead(r)), lead(r), r != lead(r)))
    stats['bad_roots'] = len(bad_roots)
    placed = [i for r in roots if r not in bad for i in subtree(r)]
    for r in bad_roots:
        if time.monotonic() > deadline:
            stats['timed_out'] = True
            break
        group = [c for c in comps.get(r, []) if c not in bad]
        if group:                                                            # (a) anchor + good companions as one unit
            ids = subtree(r) + [i for c in group for i in subtree(c)]
            members = recs(ids)
            obstacles = recs([i for i in placed if i not in ids]) + fixed_recs
            cands = filter(wall_ok([r] + group), S._root_candidates(members[0], edges))
            (dx, dy), infeasible, to = S._search(members, cands, obstacles, env, env_p, deadline, m)
            stats['timed_out'] |= to
            if not infeasible:
                move(ids, dx, dy)
                placed = [i for i in placed if i not in ids] + ids
                stats['group_moves'] += 1
                continue
            stats['group_failed'] += 1
        members = recs(subtree(r))
        obstacles = recs(placed) + fixed_recs
        cands = filter(wall_ok([r]), S._root_candidates(members[0], edges))
        if r in pair_of:                     # (b) relation-keeping candidates first; S2's only if none is feasible, and
            ks = ([(0.0, 0.0)] if C.holds(pair_of[r], cur) else []) + keep(r)      # if neither is, the min-cost keep pose
            (dx, dy), fallback, to = S._search(members, ks, obstacles, env, env_p, deadline, m)
            if fallback:
                (ux, uy), fallback, to2 = S._search(members, cands, obstacles, env, env_p, deadline, m)
                to |= to2
                if not fallback or not ks:
                    dx, dy = ux, uy
        else:
            (dx, dy), fallback, to = S._search(members, cands, obstacles, env, env_p, deadline, m)
        stats['min_overlap'] += fallback
        stats['timed_out'] |= to
        move(subtree(r), dx, dy)
        placed += subtree(r)
        if r in pair_of:
            held = C.holds(pair_of[r], cur)
            stats['keep_moves'] += held
            stats['fallbacks'] += not held
        for c in group:                                                      # (a) companions left behind try to rejoin
            if C.holds(pair_of[c], cur):
                continue
            tree = subtree(c)
            members = recs(tree)
            obstacles = recs([i for i in placed if i not in tree]) + fixed_recs
            (dx, dy), infeasible, to = S._search(members, keep(c), obstacles, env, env_p, deadline, m)
            stats['timed_out'] |= to
            if infeasible:
                stats['left_behind'] += 1
            else:
                move(tree, dx, dy)
                stats['rejoined'] += 1
    # pass 2: children on their seat, lowest first (S2 unchanged)
    bad = _bad({**room, 'objects': [cur[o['id']] for o in objs]}, keepouts, phi, in_pair)
    bad_children = sorted([i for i in bad if i not in roots and (cur[i]['parent'] in by or cur[i]['parent'] in fixed)],
                          key=lambda i: (cur[i]['pos'][2], -cur[i]['size'][0] * cur[i]['size'][1]))
    stats['bad_children'] = len(bad_children)
    for i in bad_children:
        if time.monotonic() > deadline:
            stats['timed_out'] = True
            break
        seat = cur.get(cur[i]['parent']) or fixed[cur[i]['parent']]
        tree = subtree(i)
        members = recs(tree)
        obstacles = recs([j for j in cur if j not in tree]) + fixed_recs
        (dx, dy), fallback, to = S._search(members, S._child_candidates(members[0], seat), obstacles, env, env_p, deadline, m)
        stats['min_overlap'] += fallback
        stats['timed_out'] |= to
        move(tree, dx, dy)
    stats['pairs'] = len(pairs)
    return [cur[o['id']] for o in objs], shift, stats


# ------------------------------------------------------------------ rows / pipeline

def probe_row(task):
    """S.probe_row with s2f; adds companion retention of the stage input -> stage output."""
    row, room_gt, request, phi, protect = task
    rel = S.gt_relations(room_gt)
    held = lambda objs: sum(S.holds(c, None, room_gt['boundary'], {o['id']: o for o in objs}) for c in rel)
    started = time.monotonic()
    room = S.orp.predicted_room(room_gt, request['messages'], row['s1_layout'])
    keepouts = S.orp.door_keepouts(room)
    before = S._eval(room, room['objects'])
    objs, shift, stats = s2f(room, keepouts, started + S.TIME_LIMIT, phi, protect)
    ok = S._invariant(room['objects'], objs)
    after = S._eval(room, objs)
    res = S.orp.repair({**room, 'objects': objs})
    out = {'uid': row['uid'], 'source': row['source'], 'objects': row['objects'], 'stratum': row['s1']['stratum'],
           'doors': row['s1']['doors'], 'rectangle': row['s1']['rectangle'], 'before': before, 'after': after,
           'invariant_ok': ok, **stats,
           'moved': sum(1 for v in shift.values() if v != (0.0, 0.0)),
           'displacement_max': max(math.hypot(*v) for v in shift.values()),
           'displacement_moved_median': statistics.median([math.hypot(*v) for v in shift.values() if v != (0.0, 0.0)] or [0.0]),
           'polish': {**S._eval(room, res['room']['objects']), 'status': res['status']}}
    out['s2_layout'] = res['room']['objects'] if res['status'] != 'unchanged' else objs
    out['invariant_ok'] &= S._invariant(room['objects'], out['s2_layout'])
    out['relations'] = len(rel)
    out['held'] = {'before': held(room['objects']), 'after': held(objs), 'polish': held(out['s2_layout'])}
    out['retention'] = {'after': C.retention(room['objects'], objs), 'polish': C.retention(room['objects'], out['s2_layout'])}
    out['runtime_s'] = time.monotonic() - started
    return out


def merge(rows, stage):
    """Adopt the polished stage layout iff checker and strict are both not worse (reproduces after-s2-protect.jsonl)."""
    by = {r['uid']: r for r in stage}
    out = []
    for r in rows:
        s = by.get(r['uid'])
        if s is not None:
            b, p = s['before'], s['polish']
            assert b['checker'] == r['s1']['geometry_success_s1'] and b['strict'] == r['s1']['strict_ok_s1'], r['uid']
            if p['checker'] >= b['checker'] and p['strict'] >= b['strict']:
                r = {**r, 's1_layout': s['s2_layout'],
                     's1': {**r['s1'], 'geometry_success_s1': p['checker'], 'strict_ok_s1': p['strict']}}
        out.append(r)
    return out


def _stage(rows, rooms, requests, phi, select, workers, out):
    key = 'geometry_success_s1' if select == 'checker' else 'strict_ok_s1'
    tasks = [(r, rooms[r['uid']], requests[r['uid']], phi, True) for r in rows if r['complete'] and not r['s1'][key]]
    with ProcessPoolExecutor(workers, initializer=_imports) as pool:
        res = list(pool.map(probe_row, tasks, chunksize=2))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in res))
    summary = S.summarize(res, True)
    keys = ('group_moves', 'group_failed', 'keep_moves', 'fallbacks', 'rejoined', 'left_behind', 'pairs')
    counts = {k: sum(r[k] for r in res) for k in keys}
    (out / 'summary.json').write_text(json.dumps({'params': {'phi': phi, 'select': select, 'protect': True,
                                                             'keep_grid': KEEP_GRID, 'face_grid': FACE_GRID, 'keep_cap': KEEP_CAP,
                                                             'gap': C.GAP, 'cone_deg': 45, 'face_dist': C.FACE_DIST},
                                                  'counts': counts, 'summary': summary}, indent=1))
    (out / 'report.md').write_text(S._report(summary, True) + '\n' + json.dumps(counts) + '\n')
    print(S._report(summary, True), counts, flush=True)
    return res


def pipeline(a):
    _imports()
    rooms = {json.loads(l)['uid']: json.loads(l) for l in open(Path(a.data) / 'rooms.jsonl')}
    requests = {json.loads(l)['uid']: json.loads(l) for l in open(Path(a.data) / 'placement' / 'requests.jsonl')}
    rows = [json.loads(l) for l in open(a.s1)]
    out = Path(a.out)
    st1 = _stage(rows, rooms, requests, 'checker', 'checker', a.workers, out / 's2f-checker')
    rows = merge(rows, st1)
    (out / 'after-s2f-checker.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    st2 = _stage(rows, rooms, requests, 'strict', 'strict', a.workers, out / 's2f-strict')
    rows = merge(rows, st2)
    (out / 'final-s2f.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


# ------------------------------------------------------------------ self-check

def selfcheck():
    _imports()
    boundary = [[0, 0], [5, 0], [5, 5], [0, 5]]
    bed = {'id': 'bed', 'category': 'King-size Bed', 'size': [2.0, 1.6, 0.5], 'pos': [1.0, 2.0, 0], 'yaw': 0.0, 'parent': None}
    n1 = {'id': 'n1', 'category': 'nightstand', 'size': [0.4, 0.4, 0.5], 'pos': [0.25, 0.95, 0], 'yaw': 0.0, 'parent': None}
    n2 = {**n1, 'id': 'n2', 'pos': [0.25, 3.05, 0]}
    ward = {'id': 'w', 'category': 'wardrobe', 'size': [0.6, 1.2, 2.0], 'pos': [0.9, 0.6, 0], 'yaw': math.pi / 2, 'parent': None}
    column = {'id': 'col', 'category': 'column', 'size': [0.3, 0.3, 2.8], 'pos': [1.5, 2.0, 0], 'yaw': 0.0}
    door = {'id': 'd', 'category': 'door', 'size': [0.9, 0.1, 2.1], 'pos': [4.0, 0.05, 0], 'yaw': 0.0}
    base = {'boundary': boundary, 'boundary_type': 'polygon', 'height': 2.8, 'constraints': []}
    # toy 1: column through the bed (bed must move), wardrobe colliding with a nightstand (rule c blames the wardrobe)
    room = {**base, 'fixed': [door, column], 'objects': [bed, n1, n2, ward]}
    c0 = S.check(room)
    assert c0['collisions']['main'] and c0['fixed_blocking'] and C.retention(room['objects'], room['objects']) == (2, 2), c0
    keep = S.orp.door_keepouts(room)
    blamed = _bad(room, keep, 'checker', {'bed', 'n1', 'n2'})
    assert 'w' in blamed and 'n1' not in blamed and 'bed' in blamed, blamed          # rule (c); S2 blames n1
    for phi in ('checker', 'strict'):
        objs, shift, stats = s2f(room, keep, time.monotonic() + 30, phi, protect=True)
        c = S.check({**room, 'objects': objs})
        assert S._invariant(room['objects'], objs)
        assert not c['collisions']['main'] and not c['oob'] and not c['fixed_blocking'] and not c['support_fail'], c
        assert C.retention(room['objects'], objs) == (2, 2), ('companions separated', phi, shift)
        assert stats['group_moves'] == 1 and stats['fallbacks'] == 0, stats
        print('selfcheck ok toy1', phi, {i: [round(v, 2) for v in s] for i, s in shift.items()}, stats)
    plain = S.s2(room, keep, time.monotonic() + 30, 'checker', True)[0]
    print('  plain S2 on toy1 keeps', C.retention(room['objects'], plain), 'pairs')
    # toy 2: column through a free-standing nightstand only (bed good): it must take a relation-keeping candidate
    n1 = {**n1, 'pos': [0.35, 0.95, 0]}
    room = {**base, 'fixed': [door, {**column, 'pos': [0.35, 0.95, 0]}], 'objects': [bed, n1, n2]}
    keep = S.orp.door_keepouts(room)
    objs, shift, stats = s2f(room, keep, time.monotonic() + 30, 'checker', protect=True)
    c = S.check({**room, 'objects': objs})
    assert not c['collisions']['main'] and not c['fixed_blocking'] and not c['oob'], c
    assert C.retention(room['objects'], objs) == (2, 2) and stats['keep_moves'] == 1 and stats['fallbacks'] == 0, (shift, stats)
    assert shift['bed'] == (0.0, 0.0) and shift['n2'] == (0.0, 0.0)
    print('selfcheck ok toy2', {i: [round(v, 2) for v in s] for i, s in shift.items()}, stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selfcheck', action='store_true'); ap.add_argument('--pipeline', action='store_true')
    ap.add_argument('--s1'); ap.add_argument('--data'); ap.add_argument('--out')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if a.pipeline:
        return pipeline(a)
    ap.print_help()


if __name__ == '__main__':
    main()
