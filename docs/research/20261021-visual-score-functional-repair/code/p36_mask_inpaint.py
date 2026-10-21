"""Probe p36 (paper 36, Steerable Scene Generation): mask-and-inpaint transfer, CPU-only part.

For every 27B test room that is complete but still fails the frozen checker after S1:
  1. mask = every object in a violating pair/flag (paper's mask generator: both members of a penetrating pair);
     M2 = mask closed over whole floor trees (no `on: fixed_` exists in our data, so a clamped parent cannot be
     referenced; a masked tree moves as a unit).
  2. remainder (clamped objects only) must pass the checker (the 'welded' base of the paper).
  3. dumb inpainting = the paper with B = infinity and a UNIFORM proposal: masked objects re-placed one by one with
     random XY / cardinal yaw, accepted iff no new checker violation (reward = feasible count). Lower bound of what a
     learned proposal (27B placer or gpt-6.1-sol on the rewritten request) could give; semantics NOT measured.
  4. contract round trip: clamped movables -> fixed_NNNN, masked movables -> obj_0001.., messages()[:2], a synthetic
     answer through validate_output and back to the original ids. Writes the <= 100 API records for the paid probe.

python -I p36_mask_inpaint.py --out DIR [--sample N] [--tries 300] [--workers 10] [--api-n 100] [--seed 0]
"""
import argparse
import collections
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from shapely.geometry import Point, Polygon  # noqa: E402

from fastfill.anchors import FLAT  # noqa: E402
from fastfill.comparison import contract, downstream as d, geometry as g, output_repair  # noqa: E402
from fastfill.scene import footprint, head, norm_cat  # noqa: E402
from fastfill.validate import check, collision_kind  # noqa: E402

ROOT = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013')
S1_ROWS = ROOT / 'a026-inputs/s1-022/placement-arm/placement-geometry-s1.jsonl'
ROOMS = ROOT / 'visual-final-selected-002/inputs/eval-data-v1/rooms.jsonl'
REQUESTS = ROOT / 'visual-final-selected-002/inputs/eval-data-v1/placement/requests.jsonl'
TOL, MIN_AREA = 0.05, 0.01
STRATA = ((1, 10), (11, 20), (21, 50), (51, 10 ** 9))


def stratum(n):
    return next(f'{lo}-{hi if hi < 10 ** 9 else "inf"}' for lo, hi in STRATA if lo <= n <= hi)


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def mask_ids(c):
    m = set()
    for kind in ('main', 'severe'):
        for a, b in c['collisions'][kind]:
            m.update((a, b))
    m.update(c['oob']); m.update(c['support_fail']); m.update(c['ceiling'])
    m.update(o for o, _ in c['fixed_blocking'])
    return m


def trees(objs):
    """{id: floor-root id} and {root: [ids of the whole tree]}."""
    parent = {o['id']: o.get('parent') for o in objs}
    root = {}
    for i in parent:
        cur, seen = i, set()
        while parent.get(cur) is not None and cur not in seen:
            seen.add(cur); cur = parent[cur]
        root[i] = cur
    members = collections.defaultdict(list)
    for i, r in root.items():
        members[r].append(i)
    return root, members


def _violates(o, fp, placed, fps, fixed, fixed_fps, inner, height):
    """One object against the current scene with the checker's pair rules (main collisions, oob, fixed_blocking,
    ceiling). tuck/contained pairs are not failures, as in validate.check."""
    if not inner.contains(fp):
        return True
    if height and o['pos'][2] + o['size'][2] > height + TOL:
        return True
    flat_or_short = head(o) in FLAT or o['size'][2] < 0.06
    for p, pfp in zip(placed, fps):
        if p['id'] == o['id'] or o.get('parent') == p['id'] or p.get('parent') == o['id']:
            continue
        if min(o['pos'][2] + o['size'][2], p['pos'][2] + p['size'][2]) - max(o['pos'][2], p['pos'][2]) <= TOL:
            continue
        if FLAT.intersection((head(o), head(p))) or min(o['size'][2], p['size'][2]) < 0.06:
            continue
        if fp.intersection(pfp).area > MIN_AREA and collision_kind(o, p, fp, pfp, TOL) == 'main':
            return True
    if not flat_or_short:
        for f, ffp in zip(fixed, fixed_fps):
            if {'window', 'windows'}.intersection(norm_cat(f.get('category')).split()):
                continue
            if min(o['pos'][2] + o['size'][2], f['pos'][2] + f['size'][2]) - max(o['pos'][2], f['pos'][2]) <= TOL:
                continue
            if fp.intersection(ffp).area > MIN_AREA:
                return True
    return False


def dumb_inpaint(room, masked, tries, rng, time_cap=30.0):
    """Greedy random re-placement of the masked objects (parents before children). Returns (layout, placed_count)."""
    objs = [dict(o) for o in room['objects']]
    by = {o['id']: o for o in objs}
    poly = Polygon(room['boundary'])
    inner = poly.buffer(1e-9)                       # strictly inside the floor: feasible for checker and strict
    bx0, by0, bx1, by1 = poly.bounds
    fixed = room.get('fixed') or []
    fixed_fps = [Polygon(footprint(f)) for f in fixed]
    depth = {}

    def dep(i):
        if i not in depth:
            p = by[i].get('parent')
            depth[i] = 0 if p is None or p not in by else 1 + dep(p)
        return depth[i]
    order = sorted(masked, key=lambda i: (dep(i), i))
    clamped = [o for o in objs if o['id'] not in masked]
    placed, fps = list(clamped), [Polygon(footprint(o)) for o in clamped]
    count, start = 0, time.monotonic()
    for i in order:
        o = by[i]
        ok = False
        if time.monotonic() - start < time_cap:
            parent = by.get(o.get('parent'))
            pfp = Polygon(footprint(parent)) if parent is not None else None
            for _ in range(tries):
                yaw = rng.choice((0.0, math.pi / 2, math.pi, 3 * math.pi / 2))
                if parent is None:
                    x, y, z = rng.uniform(bx0, bx1), rng.uniform(by0, by1), 0.0
                else:
                    px0, py0, px1, py1 = pfp.bounds
                    x, y, z = rng.uniform(px0, px1), rng.uniform(py0, py1), parent['pos'][2] + parent['size'][2]
                    if not pfp.contains(Point(x, y)):
                        continue
                cand = {**o, 'pos': [x, y, z], 'yaw': yaw}
                fp = Polygon(footprint(cand))
                if not _violates(cand, fp, placed, fps, fixed, fixed_fps, inner, room.get('height')):
                    o.update(cand); ok = True; count += 1
                    break
        placed.append(o); fps.append(Polygon(footprint(o)))
    return objs, count


def rewrite(room_ir, request, layout, masked):
    """Clamped movables -> fixed_ records after the request's fixed; masked movables -> obj_0001.. in request order.
    Returns (bound room for contract.messages, sizes, back maps)."""
    req = json.loads(request['messages'][1]['content'])
    fixed = [{**f, 'yaw': math.radians(f['yaw'])} for f in req.get('fixed', [])]
    new_fixed, fixed_map = list(fixed), {}
    for o in layout:
        if o['id'] not in masked:
            fid = f'fixed_{len(new_fixed) + 1:04d}'
            fixed_map[o['id']] = fid
            new_fixed.append({'id': fid, 'category': o['category'], 'size': list(o['size']), 'pos': list(o['pos']),
                              'yaw': o['yaw']})
    new_objs, obj_map = [], {}
    for o in layout:
        if o['id'] in masked:
            oid = f'obj_{len(new_objs) + 1:04d}'
            obj_map[o['id']] = oid
            parent = o.get('parent')
            new_objs.append({'id': oid, 'category': o['category'], 'size': list(o['size']), 'pos': list(o['pos']),
                             'yaw': o['yaw'], 'parent': obj_map.get(parent) if parent is not None else None})
    bound = {**room_ir, 'boundary': req['boundary'], 'height': req.get('height'), 'fixed': new_fixed,
             'objects': new_objs, contract._MARKER: contract.PROTOCOL}
    bound.pop('constraints', None)
    sizes = {o['id']: o['size'] for o in new_objs}
    return bound, sizes, {v: k for k, v in obj_map.items()}, {v: k for k, v in fixed_map.items()}


def map_back(answer, back_obj, back_fixed, layout):
    """Model answer on the rewritten request -> full layout in original ids (clamped objects unchanged)."""
    by = {o['id']: dict(o) for o in layout}
    for p in answer['placements']:
        o = by[back_obj[p['id']]]
        on = p.get('on')
        o['pos'], o['yaw'] = list(p['pos']), math.radians(p['yaw'])
        o['parent'] = None if on is None else back_obj.get(on, back_fixed.get(on))
    return [by[o['id']] for o in layout]


def one(task):
    row, room_ir, request, tries, seed = task
    rng = random.Random(f'{seed}:{row["uid"]}')
    t0 = time.monotonic()
    layout = row['s1_layout']
    room = output_repair.predicted_room(room_ir, request['messages'], layout)
    c = check(room)
    m1 = mask_ids(c)
    root, members = trees(room['objects'])
    m2 = {j for i in m1 for j in members[root[i]]}
    n = len(layout)
    remainder = {**room, 'objects': [o for o in room['objects'] if o['id'] not in m2]}
    rem_pass = g.geometry(remainder)['geometry_success'] if remainder['objects'] else True
    new_layout, placed = dumb_inpaint(room, m2, tries, rng)
    after = g.geometry({**room, 'objects': new_layout})
    strict = None
    try:
        strict = d.accept(room, new_layout)
        strict = strict.get('admit_ok') if isinstance(strict, dict) else None
    except Exception:
        strict = None
    # contract round trip with a synthetic answer (the new layout itself)
    bound, sizes, back_obj, back_fixed = rewrite(room_ir, request, layout, m2)
    rt_ok, prompt_chars = False, None
    if new_objs := bound['objects']:
        msgs = contract.messages(bound, 'placement', sizes)
        prompt_chars = len(msgs[1]['content'])
        obj_map = {v: k for k, v in back_obj.items()}
        nb = {o['id']: o for o in new_layout}
        ans = {'placements': [{'id': o['id'], 'pos': [round(v, 2) for v in nb[back_obj[o['id']]]['pos']],
                               'yaw': round(math.degrees(nb[back_obj[o['id']]]['yaw'] % math.tau)) % 360,
                               **({'on': obj_map[p]} if (p := nb[back_obj[o['id']]].get('parent')) in obj_map else {})}
                              for o in new_objs]}
        try:
            out = contract.validate_output(json.dumps(ans), bound, 'placement')
            back = map_back(out, back_obj, back_fixed, layout)
            rt_ok = [o['id'] for o in back] == [o['id'] for o in layout]
        except ValueError:
            rt_ok = False
    return {'uid': row['uid'], 'source': row['source'], 'n': n, 'stratum': stratum(n), 'mask': len(m1),
            'mask_tree': len(m2), 'whole_room': len(m2) == n, 'remainder_pass': rem_pass, 'placed': placed,
            'pass_after': after['geometry_success'], 'violations_after': sum(after[k] for k in (
                'main_collisions', 'severe_collisions', 'fixed_blocking', 'oob_objects', 'support_fail', 'ceiling_fail')),
            'strict_after': strict, 'roundtrip_ok': rt_ok, 'prompt_chars': prompt_chars,
            'orig_prompt_chars': len(request['messages'][1]['content']), 'seconds': time.monotonic() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sample', type=int)
    ap.add_argument('--tries', type=int, default=300)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--api-n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--api-only', action='store_true', help='reuse OUT/rows.jsonl; only rewrite the API records')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rooms = {r['uid']: r for r in jsonl(ROOMS)}
    requests = {r['uid']: r for r in jsonl(REQUESTS)}
    rows = [r for r in jsonl(S1_ROWS) if r['complete'] and not r['s1']['geometry_success_s1']]
    rng = random.Random(a.seed)
    if a.sample:
        rows = rng.sample(rows, min(a.sample, len(rows)))
    # OOD check: furniture-like fixed boxes in the (test) rooms
    fixed_total = fixed_furn = rooms_furn = 0
    for r in rooms.values():
        furn = [f for f in r.get('fixed') or [] if not {'door', 'doors', 'window', 'windows', 'doorframe'}
                .intersection(norm_cat(f.get('category') or '').replace('(', ' ').split())]
        fixed_total += len(r.get('fixed') or []); fixed_furn += len(furn); rooms_furn += bool(furn)
    if a.api_only:
        results = jsonl(a.out / 'rows.jsonl')
        rows = [r for r in rows if r['uid'] in {x['uid'] for x in results}]
    else:
        tasks = [(r, rooms[r['uid']], requests[r['uid']], a.tries, a.seed) for r in rows]
        with ProcessPoolExecutor(a.workers) as pool:
            results = list(pool.map(one, tasks, chunksize=4))
        with open(a.out / 'rows.jsonl', 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
    by = collections.defaultdict(list)
    for r in results:
        by[r['stratum']].append(r); by['all'].append(r)

    def agg(rs):
        k = len(rs)
        return {'rooms': k, 'mask_frac_mean': sum(r['mask'] / r['n'] for r in rs) / k,
                'mask_tree_frac_mean': sum(r['mask_tree'] / r['n'] for r in rs) / k,
                'whole_room_masked': sum(r['whole_room'] for r in rs) / k,
                'remainder_pass': sum(r['remainder_pass'] for r in rs) / k,
                'dumb_inpaint_pass': sum(r['pass_after'] for r in rs) / k,
                'dumb_inpaint_strict': sum(bool(r['strict_after']) for r in rs) / k,
                'placed_frac_mean': sum(r['placed'] / max(1, r['mask_tree']) for r in rs) / k,
                'roundtrip_ok': sum(r['roundtrip_ok'] for r in rs) / k,
                'prompt_chars_ratio_mean': sum((r['prompt_chars'] or 0) / r['orig_prompt_chars'] for r in rs) / k,
                'seconds_mean': sum(r['seconds'] for r in rs) / k}
    summary = {'s1_failed_complete_rooms': len(rows), 'tries': a.tries, 'seed': a.seed,
               'fixed_boxes_total': fixed_total, 'fixed_furniture_like': fixed_furn,
               'rooms_with_furniture_like_fixed': rooms_furn, 'rooms_total': len(rooms),
               'by_stratum': {k: agg(v) for k, v in sorted(by.items())}}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    # API records (<= api_n): stratified among rooms where inpainting is partial (not the whole room)
    cands = collections.defaultdict(list)
    for r in results:
        if not r['whole_room'] and r['roundtrip_ok']:
            cands[r['stratum']].append(r['uid'])
    per = max(1, a.api_n // len(STRATA))
    chosen = [u for k in sorted(cands) for u in rng.sample(sorted(cands[k]), min(per, len(cands[k])))][:a.api_n]
    rows_by = {r['uid']: r for r in rows}
    with open(a.out / 'api-requests.jsonl', 'w') as f:
        for uid in chosen:
            row, req = rows_by[uid], requests[uid]
            room = output_repair.predicted_room(rooms[uid], req['messages'], row['s1_layout'])
            m1 = mask_ids(check(room)); root, members = trees(room['objects'])
            m2 = {j for i in m1 for j in members[root[i]]}
            bound, sizes, back_obj, back_fixed = rewrite(rooms[uid], req, row['s1_layout'], m2)
            msgs = contract.messages(bound, 'placement', sizes)[:2]
            f.write(json.dumps({**{k: req.get(k) for k in ('uid', 'source', 'group', 'component', 'split', 'room_type')},
                                'messages': msgs, 'p36': {'masked': sorted(m2), 'back_obj': back_obj,
                                                          'back_fixed': back_fixed}}) + '\n')
    print(json.dumps(summary, indent=1))
    print('api records', len(chosen), '->', a.out / 'api-requests.jsonl')


if __name__ == '__main__':
    main()
