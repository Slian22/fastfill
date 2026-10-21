#!/usr/bin/env python3
"""LayoutDSL (arXiv 2608.07547) transfer probe for FastFill, 2026-10-21.

A LayoutDSL-style generator/interpreter adapted to the FastFill contract (meters, Z up, local +X = front, yaw deg CCW
from +X, pos = bottom centre, prior sizes given, support 'on' kept via a new action 'put').

Statement:  <action> <id> orient:<deg> <anchor> <align>:<delta> distance:<d>
  action  place (floor, z=0) | put (on top of the anchor: z = anchor top, output on=anchor)
  anchor  wall<k> (boundary edge k: vertex k -> k+1, CCW) | obj_NNNN / fixed_NNNN (already placed or fixed) | room0
  align   wall anchor:   wall_center:<delta>   delta = signed offset of the centre along the wall from its midpoint
                         distance = gap between the wall line and the nearest face of the object (>= 0 inside)
          object anchor: front_center|back_center|left_center|right_center:<delta>  (side in the anchor's local frame)
                         delta = lateral offset of the centre from the side-face centre, distance = face-to-face gap
          put:           on_center:<delta>   distance = +X offset, delta = +Y offset in the anchor's local frame
          room0:         room_center:<delta>  distance = +X offset, delta = +Y offset from the boundary centroid
The interpreter is the exact inverse (objects are decoded in statement order; a forward anchor reference is a decode
failure), then the placements are re-sorted into request order so contract.validate_output is unchanged.

Sub-commands
  gt-stats  --rooms ROOMS.jsonl --out DIR [--limit N]        free, local: encode GT, decode at cm precision, anchor
            kinds / chain depth / drift by object-count stratum, checker pass of GT after DSL round trip vs direct cm
            rounding (needs PYTHONPATH=<repo> for fastfill.validate).
  select    --geometry placement-geometry.jsonl --requests requests.jsonl --out DIR [--per-stratum 25]
            deterministic 50 hard rooms (complete, raw checker fail, doors, polygon; 25 of 21-50 and 25 of >50 objects).
  api       --requests requests.jsonl --uids DIR/uids.json --out DIR/api --env-file ~/.fastfill_api.env
            Isambard only: 2 calls per room (JSON-coordinate arm = P request verbatim; DSL arm = same user JSON with the
            DSL system prompt), 1 round each, <= 100 calls, through llm_agent's ledger; then S1 and the checker.
  report    --api DIR/api --out DIR                          paired raw / S1 checker pass, duplicates, oob, decode rate.
"""
import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon

WALL_TOL, SIDE_TOL, OBJ_TOL = 0.15, 0.05, 1.0
BACK_COS = math.cos(math.radians(15))
SIDES = {'front': 0.0, 'left': 90.0, 'back': 180.0, 'right': 270.0}
STRATA = ((1, 10), (11, 20), (21, 50), (51, None))
SALT = 'layoutdsl-probe-20261021:'
STATEMENT = re.compile(r'^(place|put)\s+(\S+)\s+orient:(-?\d+(?:\.\d+)?)\s+(\S+)\s+([a-z_]+):(-?\d+(?:\.\d+)?)'
                       r'\s+distance:(-?\d+(?:\.\d+)?)\s*$')


def stratum(n):
    for lo, hi in STRATA:
        if n >= lo and (hi is None or n <= hi):
            return f'{lo}-{hi}' if hi else f'{lo}+'


def axes(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c, s), (-s, c)


def half_extent(o, n):
    ex, ey = axes(o['yaw'])
    return abs(o['size'][0] / 2 * (ex[0] * n[0] + ex[1] * n[1])) + abs(o['size'][1] / 2 * (ey[0] * n[0] + ey[1] * n[1]))


def footprint(o):
    ex, ey = axes(o['yaw'])
    hx, hy, (x, y) = o['size'][0] / 2, o['size'][1] / 2, o['pos'][:2]
    return Polygon([(x + a * ex[0] + b * ey[0], y + a * ex[1] + b * ey[1])
                    for a, b in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))])


def walls(boundary):
    pts = [tuple(p) for p in boundary]
    out = []
    for k, (a, b) in enumerate(zip(pts, pts[1:] + pts[:1])):
        length = math.dist(a, b)
        if length < 1e-9:
            out.append(None)
            continue
        u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        out.append({'a': a, 'u': u, 'n': (-u[1], u[0]), 'mid': ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                    'len': length, 'line': LineString([a, b]), 'id': f'wall{k}'})
    return out


def side_frame(anchor, side):
    ang = anchor['yaw'] + math.radians(SIDES[side])
    n = (math.cos(ang), math.sin(ang))
    return n, (-n[1], n[0])


# ------------------------------------------------------------------ generator (GT / repaired layout -> statements)

def order(objects):
    """Floor objects by footprint area (desc), then supported objects by support depth then area; ties by id."""
    by = {o['id']: o for o in objects}
    def depth(o, seen=()):
        p = o.get('parent')
        return 0 if p is None or p not in by or p in seen else 1 + depth(by[p], seen + (o['id'],))
    return sorted(objects, key=lambda o: (depth(o), -o['size'][0] * o['size'][1], o['id']))


def select_wall(o, ws, fp):
    c = o['pos'][:2]
    ex = axes(o['yaw'])[0]
    best = None
    for pref in (True, False):
        for w in ws:
            if w is None:
                continue
            along = (c[0] - w['a'][0]) * w['u'][0] + (c[1] - w['a'][1]) * w['u'][1]
            if not 0 <= along <= w['len']:
                continue
            gap = fp.distance(w['line'])
            back = ex[0] * w['n'][0] + ex[1] * w['n'][1] >= BACK_COS       # back face toward the wall
            if (pref and back and gap <= WALL_TOL) or (not pref and gap <= SIDE_TOL):
                if best is None or gap < best[0]:
                    best = (gap, w, 'back' if back else 'side')
        if best:
            return best
    return None


def encode(room, objects=None, rnd=2):
    """[(statement, meta)] in generator order; meta has anchor kind, depth, exact (unrounded) values."""
    objects = order(room['objects'] if objects is None else objects)
    ws = walls(room['boundary'])
    poly = Polygon(room['boundary'])
    centroid = (poly.centroid.x, poly.centroid.y)
    fixed = {f['id']: f for f in room.get('fixed') or []}
    placed, depth, out = {}, {}, []
    for o in objects:
        c, yaw_deg = o['pos'][:2], round(math.degrees(o['yaw'] % math.tau)) % 360
        fp = footprint(o)
        parent = o.get('parent')
        if parent is not None and (parent in placed or parent in fixed):
            p = placed.get(parent) or fixed[parent]
            ex, ey = axes(p['yaw'])
            r = (c[0] - p['pos'][0], c[1] - p['pos'][1])
            d, delta = r[0] * ex[0] + r[1] * ex[1], r[0] * ey[0] + r[1] * ey[1]
            kind, anchor, align, dep = 'put', parent, 'on_center', (depth.get(parent, 0) + 1)
            action = 'put'
        else:
            action = 'place'
            hit = select_wall(o, ws, fp)
            if hit:
                gap, w, how = hit
                delta = (c[0] - w['mid'][0]) * w['u'][0] + (c[1] - w['mid'][1]) * w['u'][1]
                t = (c[0] - w['a'][0]) * w['n'][0] + (c[1] - w['a'][1]) * w['n'][1]
                d = t - half_extent(o, w['n'])
                kind, anchor, align, dep = f'wall_{how}', w['id'], 'wall_center', 1
            else:
                cands = [(footprint(a).distance(fp), a['id']) for a in placed.values() if a.get('parent') is None]
                cands = [x for x in cands if x[0] <= OBJ_TOL]
                if cands:
                    _, aid = min(cands)
                    a = placed[aid]
                    r = (c[0] - a['pos'][0], c[1] - a['pos'][1])
                    side = max(SIDES, key=lambda s: r[0] * side_frame(a, s)[0][0] + r[1] * side_frame(a, s)[0][1])
                    n, tg = side_frame(a, side)
                    t = r[0] * n[0] + r[1] * n[1]
                    d = t - half_extent(a, n) - half_extent(o, n)
                    delta = r[0] * tg[0] + r[1] * tg[1]
                    kind, anchor, align, dep = 'object', aid, f'{side}_center', depth[aid] + 1
                else:
                    d, delta = c[0] - centroid[0], c[1] - centroid[1]
                    kind, anchor, align, dep = 'room', 'room0', 'room_center', 1
        text = f'{action} {o["id"]} orient:{yaw_deg} {anchor} {align}:{round(delta, rnd):.{rnd}f} distance:{round(d, rnd):.{rnd}f}'
        out.append((text, {'id': o['id'], 'kind': kind, 'depth': dep, 'anchor': anchor}))
        placed[o['id']] = o
        depth[o['id']] = dep
    return out


# ------------------------------------------------------------------ interpreter (statements -> placements)

def decode(statements, room, sizes):
    """room: boundary + fixed (IR, yaw rad); sizes: {id: [sx,sy,sz]}. Returns {id: {pos, yaw(deg), on?}} or raises."""
    ws = walls(room['boundary'])
    poly = Polygon(room['boundary'])
    centroid = (poly.centroid.x, poly.centroid.y)
    fixed = {f['id']: f for f in room.get('fixed') or []}
    placed, out = {}, {}
    for line in statements:
        m = STATEMENT.match(line.strip())
        if not m:
            raise ValueError('syntax: ' + line[:80])
        action, oid, deg, anchor, align, delta, d = m.groups()
        if oid not in sizes or oid in out:
            raise ValueError('unknown or repeated id ' + oid)
        deg, delta, d = float(deg), float(delta), float(d)
        yaw_deg = round(deg) % 360
        o = {'id': oid, 'size': sizes[oid], 'yaw': math.radians(yaw_deg), 'pos': [0.0, 0.0, 0.0]}
        z, on = 0.0, None
        if action == 'put':
            p = placed.get(anchor) or fixed.get(anchor)
            if p is None or align != 'on_center':
                raise ValueError('put needs a placed/fixed anchor with on_center')
            ex, ey = axes(p['yaw'])
            x, y = p['pos'][0] + d * ex[0] + delta * ey[0], p['pos'][1] + d * ex[1] + delta * ey[1]
            z, on = p['pos'][2] + p['size'][2], anchor
        elif anchor.startswith('wall') and align == 'wall_center':
            k = int(anchor[4:]) if anchor[4:].isdigit() else -1
            if not 0 <= k < len(ws) or ws[k] is None:
                raise ValueError('unknown wall ' + anchor)
            w = ws[k]
            t = d + half_extent(o, w['n'])
            x, y = w['mid'][0] + delta * w['u'][0] + t * w['n'][0], w['mid'][1] + delta * w['u'][1] + t * w['n'][1]
        elif anchor == 'room0' and align == 'room_center':
            x, y = centroid[0] + d, centroid[1] + delta
        elif align.endswith('_center') and align[:-7] in SIDES:
            a = placed.get(anchor) or fixed.get(anchor)
            if a is None:
                raise ValueError('unknown or forward anchor ' + anchor)
            n, tg = side_frame(a, align[:-7])
            t = d + half_extent(a, n) + half_extent(o, n)
            x, y = a['pos'][0] + t * n[0] + delta * tg[0], a['pos'][1] + t * n[1] + delta * tg[1]
        else:
            raise ValueError('bad anchor/align ' + anchor + ' ' + align)
        o['pos'] = [x, y, z]
        placed[oid] = o
        out[oid] = {'pos': [x, y, z], 'yaw': yaw_deg, **({'on': on} if on else {})}
    if set(out) != set(sizes):
        raise ValueError('missing ids: %d of %d' % (len(sizes) - len(out), len(sizes)))
    return out


def to_placements(decoded, request_objects):
    return {'placements': [{'id': o['id'], 'pos': [round(v, 6) for v in decoded[o['id']]['pos']],
                            'yaw': decoded[o['id']]['yaw'], **({'on': decoded[o['id']]['on']} if 'on' in decoded[o['id']] else {})}
                           for o in request_objects]}


# ------------------------------------------------------------------ gt-stats

def _check(room):
    from fastfill.validate import check
    c = check(room)
    parents = {o['id']: o for o in room['objects'] + (room.get('fixed') or [])}
    from fastfill.validate import support_ok
    sf = [o['id'] for o in room['objects'] if (o.get('parent') is not None and o['parent'] not in parents)
          or not support_ok(o, parents.get(o.get('parent')))]
    return bool(not c['oob'] and not sf and not c['ceiling'] and not c['collisions']['main']
                and not c['collisions']['severe'] and not c['fixed_blocking'])


def gt_stats(args):
    rows = [json.loads(l) for l in Path(args.rooms).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    kinds = collections.defaultdict(collections.Counter)
    depths = collections.defaultdict(list)
    maxdepth = collections.defaultdict(list)
    drift = collections.defaultdict(list)
    bydepth = collections.defaultdict(list)
    passes = collections.defaultdict(lambda: collections.Counter())
    fails = 0
    for r in rows:
        if r.get('boundary_type') not in ('polygon', 'hull') or len(r['objects']) > 400:
            continue
        s = stratum(len(r['objects']))
        stmts = encode(r)
        sizes = {o['id']: o['size'] for o in r['objects']}
        try:
            dec = decode([t for t, _ in stmts], r, sizes)
        except ValueError as exc:
            fails += 1
            continue
        gt = {o['id']: o for o in r['objects']}
        md = 0
        for text, meta in stmts:
            kinds[s][meta['kind']] += 1
            depths[s].append(meta['depth'])
            md = max(md, meta['depth'])
            e = math.dist(dec[meta['id']]['pos'][:2], gt[meta['id']]['pos'][:2])
            drift[s].append(e)
            bydepth[min(meta['depth'], 6)].append(e)
        maxdepth[s].append(md)
        if not args.no_checker:
            # GT after DSL round trip at cm precision vs GT with pos rounded to cm directly (the JSON path).
            dsl_room = {**r, 'objects': [{**o, 'pos': dec[o['id']]['pos'], 'yaw': math.radians(dec[o['id']]['yaw'])}
                                         for o in r['objects']]}
            json_room = {**r, 'objects': [{**o, 'pos': [round(v, 2) for v in o['pos']],
                                           'yaw': math.radians(round(math.degrees(o['yaw'] % math.tau)) % 360)}
                                          for o in r['objects']]}
            passes[s]['rooms'] += 1
            passes[s]['gt_raw'] += _check(r)
            passes[s]['gt_json_cm'] += _check(json_room)
            passes[s]['gt_dsl_cm'] += _check(dsl_room)
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else None
    report = {'rooms': len(rows), 'decode_failures': fails, 'strata': {}}
    for s in sorted(kinds, key=lambda k: int(k.split('-')[0].rstrip('+'))):
        n = sum(kinds[s].values())
        report['strata'][s] = {
            'objects': n, 'rooms': len(maxdepth[s]),
            'anchor_share': {k: round(v / n, 4) for k, v in sorted(kinds[s].items())},
            'depth_mean': round(statistics.mean(depths[s]), 3), 'depth_p95': q(depths[s], .95),
            'room_max_depth_mean': round(statistics.mean(maxdepth[s]), 3), 'room_max_depth_p95': q(maxdepth[s], .95),
            'drift_cm_mean': round(100 * statistics.mean(drift[s]), 3), 'drift_cm_p95': round(100 * q(drift[s], .95), 3),
            'drift_cm_max': round(100 * max(drift[s]), 3),
            'drift_gt_5cm_share': round(sum(e > .05 for e in drift[s]) / n, 5),
            'checker': {k: (round(v / passes[s]['rooms'], 4) if k != 'rooms' else v) for k, v in passes[s].items()}}
    report['drift_cm_by_depth'] = {str(k): {'n': len(v), 'mean': round(100 * statistics.mean(v), 3),
                                            'p95': round(100 * q(v, .95), 3)} for k, v in sorted(bydepth.items())}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'gt-stats.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# ------------------------------------------------------------------ select (deterministic hard rooms)

def select(args):
    req = {}
    for l in Path(args.requests).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            req[r['uid']] = r
    rank = lambda uid: hashlib.sha256((SALT + uid).encode()).hexdigest()
    buckets = collections.defaultdict(list)
    for l in Path(args.geometry).read_text().splitlines():
        if not l.strip():
            continue
        g = json.loads(l)
        if not g['complete'] or g['geometry_success'] or g['objects'] < 21 or g['boundary_type'] != 'polygon':
            continue
        user = json.loads(req[g['uid']]['messages'][1]['content'])
        if not any('door' in (f.get('category') or '').lower() for f in user.get('fixed', [])):
            continue
        buckets['21-50' if g['objects'] <= 50 else '51+'].append(g['uid'])
    chosen = {s: sorted(v, key=rank)[:args.per_stratum] for s, v in buckets.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    value = {'salt': SALT, 'rule': 'complete & raw checker fail & >=21 objects & polygon & has door; sha256 order',
             'pool': {s: len(v) for s, v in buckets.items()}, 'uids': chosen,
             'calls_upper_bound': 2 * sum(len(v) for v in chosen.values())}
    (out / 'uids.json').write_text(json.dumps(value, indent=1))
    print(json.dumps({k: v for k, v in value.items() if k != 'uids'}))


# ------------------------------------------------------------------ api (Isambard) and report

DSL_SYSTEM_TAIL = (
    'Use the supplied object sizes. Instead of coordinates, output one placement statement per object, in any order '
    'you like (place large floor furniture first, then dependent objects, then small/decorative ones), each object '
    'exactly once:\n'
    '  <action> <id> orient:<deg> <anchor> <align>:<delta> distance:<d>\n'
    'action: place (on the floor) or put (on top of the anchor object; the anchor must be a fixed item or an object '
    'already placed). orient: yaw in integer degrees as defined above. anchor: wall<k> (the boundary edge from '
    'vertex k to vertex k+1, counter-clockwise, k from 0), a fixed id, an already placed object id, or room0.\n'
    'align/distance: wall anchor -> wall_center:<delta> distance:<d>, delta = signed offset of the object centre '
    'along the wall from the wall midpoint (in the counter-clockwise direction), d = gap in metres between the wall '
    'and the nearest face of the object (0 = touching the wall). Object anchor -> front_center|back_center|'
    'left_center|right_center:<delta> distance:<d>: the side is taken in the anchor\'s own frame (+X front, +Y left), '
    'd = face-to-face gap along that side\'s normal (0 = touching), delta = sideways offset of the centre from the '
    'side-face centre (positive = counter-clockwise of the side normal). put -> on_center:<delta> distance:<d>: '
    'd and delta are the +X and +Y offsets of the object centre in the anchor\'s frame. room0 -> room_center:<delta> '
    'distance:<d>: d and delta are the +X and +Y offsets from the room centroid.\n'
    'All objects must stay inside the room and must not overlap each other, the fixed items, or the 1 m clearance '
    'in front of doors. Output only JSON {"layout_dsl":["<statement>", ...]}.')


def dsl_request(record):
    from fastfill.comparison import contract
    return {**record, 'messages': [{'role': 'system', 'content': contract._COMMON + DSL_SYSTEM_TAIL},
                                   record['messages'][1]]}


def api(args):
    from types import SimpleNamespace
    from fastfill.comparison import llm_agent as L
    from fastfill.comparison import train_pair_helpers as h
    from fastfill.system import pipeline
    uids = json.loads(Path(args.uids).read_text())['uids']
    todo = [u for v in uids.values() for u in v]
    by = {}
    for l in Path(args.requests).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r['uid'] in todo:
                by[r['uid']] = r
    config = pipeline._api_config(args.env_file)
    out = Path(args.out)
    for arm, make in (('json', lambda r: r), ('dsl', dsl_request)):
        records = [make(by[u]) for u in todo]
        suffix = L.SUFFIX if arm == 'json' else 'Return only the JSON object {"layout_dsl":[...]} with nothing before or after it.'
        identity = {'probe': 'layoutdsl-transfer', 'arm': arm, 'model': L.EXPECTED_MODEL, 'reasoning_effort': L.EFFORT,
                    'rounds': 1, 'suffix': suffix, 'max_output_tokens': pipeline.API['max_output_tokens']}
        run = L._Run(out / arm, h.sha256_text(json.dumps(identity, sort_keys=True)) if hasattr(h, 'sha256_text')
                     else hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), config,
                     SimpleNamespace(max_calls=args.max_calls, arm='placement', max_attempts=1,
                                     max_output_tokens=pipeline.API['max_output_tokens'], timeout=pipeline.API['timeout']),
                     L.call_api, todo)
        (out / arm / 'receipts').mkdir(parents=True, exist_ok=True)
        for r in records:
            if run.path(r['uid']).exists():
                continue
            messages = [*r['messages'], {'role': 'user', 'content': suffix}]
            try:
                reply, failed = L._answer(run, r['uid'], 1, 0, messages)
                run.receipt(r, {'attempt': 1, 'error': None, 'answers': [{'raw': reply['text'], 'usage': reply['usage']}],
                                'http_errors': failed})
            except (L._Stop, L._Failed) as exc:
                run.receipt(r, {'attempt': 1, 'error': type(exc).__name__, 'answers': [], 'http_errors': []})
                break
        (out / arm / 'identity.json').write_text(json.dumps(identity, indent=1))
    print('calls ledger:', sum(1 for arm in ('json', 'dsl') for _ in (out / arm / 'calls.jsonl').open()
                               if (out / arm / 'calls.jsonl').exists()))


def report(args):
    from fastfill.comparison import geometry as g
    from fastfill.comparison import llm_agent as L
    from fastfill.comparison import output_repair
    from fastfill.system import pipeline
    out, api_dir = Path(args.out), Path(args.api)
    uids = json.loads(Path(args.uids).read_text())['uids']
    todo = [u for v in uids.values() for u in v]
    by = {}
    for l in Path(args.requests).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r['uid'] in todo:
                by[r['uid']] = r
    rows = []
    for uid in todo:
        record, room = by[uid], pipeline.request_room(by[uid])
        user = json.loads(record['messages'][1]['content'])
        sizes = {o['id']: o['size'] for o in user['objects']}
        row = {'uid': uid, 'objects': len(sizes), 'stratum': '21-50' if len(sizes) <= 50 else '51+'}
        for arm in ('json', 'dsl'):
            p = api_dir / arm / 'receipts' / f'{uid}.json'
            raw, err = None, 'NOT_ANSWERED'
            if p.exists():
                rec = json.loads(p.read_text())
                if rec['answers']:
                    text = rec['answers'][0]['raw']
                    if arm == 'json':
                        raw, err = L.enforce_sizes(text, record)[0], None
                    else:
                        try:
                            value = json.loads(text.strip().strip('`').removeprefix('json'))
                            dec = decode(value['layout_dsl'], room, sizes)
                            raw, err = json.dumps(to_placements(dec, user['objects'])), None
                        except (ValueError, KeyError, TypeError) as exc:
                            err = 'DECODE_FAIL: ' + str(exc)[:120]
            sc = g.score(room, record['messages'], raw, 'placement', err)
            s1 = output_repair.repair_row((sc, room, record, 2000, 60.0))['s1']
            dup = 0
            if sc['complete']:
                seen = collections.Counter((tuple(round(v, 3) for v in o['pos']), round(o['yaw'], 6)) for o in sc['raw_layout'])
                dup = sum(v - 1 for v in seen.values() if v > 1)
            row[arm] = {'complete': sc['complete'], 'error': sc['error'], 'raw_pass': sc['geometry_success'],
                        's1_pass': bool(s1.get('geometry_success_s1')), 'strict_s1': bool(s1.get('accepted_s1')),
                        'main': sc.get('main_collisions'), 'oob': sc.get('oob_objects'),
                        'fixed_blocking': sc.get('fixed_blocking'), 'duplicates': dup}
        rows.append(row)
    def rate(arm, key, sub=None):
        v = [r for r in rows if sub is None or r['stratum'] == sub]
        return round(sum(bool(r[arm][key]) for r in v) / len(v), 3) if v else None
    summary = {'rooms': len(rows)}
    for sub in (None, '21-50', '51+'):
        for key in ('complete', 'raw_pass', 's1_pass', 'strict_s1'):
            summary[f'{sub or "all"}/{key}'] = {arm: rate(arm, key, sub) for arm in ('json', 'dsl')}
    import random
    diffs = [int(bool(r['dsl']['s1_pass'])) - int(bool(r['json']['s1_pass'])) for r in rows]
    rng, boots = random.Random(42), []
    for _ in range(2000):
        boots.append(statistics.mean(rng.choice(diffs) for _ in diffs))
    boots.sort()
    summary['dsl_minus_json_s1_pass'] = {'estimate': round(statistics.mean(diffs), 3),
                                         'ci95': [round(boots[50], 3), round(boots[1949], 3)]}
    summary['duplicates_per_room'] = {arm: round(statistics.mean(r[arm]['duplicates'] for r in rows), 2) for arm in ('json', 'dsl')}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def selftest():
    room = {'boundary': [[0, 0], [4, 0], [4, 3], [0, 3]], 'fixed': [
        {'id': 'fixed_0001', 'category': 'door', 'size': [0.1, 0.9, 2.0], 'pos': [4, 1.5, 0], 'yaw': math.pi / 2}],
        'objects': [
            {'id': 'obj_0001', 'size': [0.4, 0.4, 0.5], 'pos': [1.3, 0.47, 0.5], 'yaw': math.radians(17), 'parent': 'obj_0002'},
            {'id': 'obj_0002', 'size': [2.0, 1.6, 0.5], 'pos': [1.0, 1.0, 0], 'yaw': math.pi / 2, 'parent': None},
            {'id': 'obj_0003', 'size': [0.5, 0.5, 0.6], 'pos': [2.3, 0.9, 0], 'yaw': math.radians(63), 'parent': None},
            {'id': 'obj_0004', 'size': [0.3, 0.3, 0.3], 'pos': [3.5, 2.6, 0], 'yaw': 0, 'parent': None}]}
    stmts = encode(room, rnd=9)
    dec = decode([t for t, _ in stmts], room, {o['id']: o['size'] for o in room['objects']})
    for o in room['objects']:
        assert math.dist(dec[o['id']]['pos'][:2], o['pos'][:2]) < 1e-6, (o['id'], dec[o['id']], o['pos'])
    kinds = {m['id']: m['kind'] for _, m in stmts}
    assert kinds['obj_0002'].startswith('wall') and kinds['obj_0001'] == 'put' and kinds['obj_0003'] == 'object', kinds
    assert dec['obj_0001']['on'] == 'obj_0002' and abs(dec['obj_0001']['pos'][2] - 0.5) < 1e-9
    try:
        decode(['place obj_0003 orient:0 obj_0009 front_center:0 distance:0'], room, {'obj_0003': [1, 1, 1]})
        raise AssertionError('forward anchor accepted')
    except ValueError:
        pass
    print('selftest ok', [t for t, _ in stmts])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('gt-stats'); s.add_argument('--rooms', required=True); s.add_argument('--out', required=True)
    s.add_argument('--limit', type=int); s.add_argument('--no-checker', action='store_true')
    s = sub.add_parser('select'); s.add_argument('--geometry', required=True); s.add_argument('--requests', required=True)
    s.add_argument('--out', required=True); s.add_argument('--per-stratum', type=int, default=25)
    s = sub.add_parser('api'); s.add_argument('--requests', required=True); s.add_argument('--uids', required=True)
    s.add_argument('--out', required=True); s.add_argument('--env-file', required=True); s.add_argument('--max-calls', type=int, default=100)
    s = sub.add_parser('report'); s.add_argument('--api', required=True); s.add_argument('--uids', required=True)
    s.add_argument('--requests', required=True); s.add_argument('--out', required=True)
    sub.add_parser('selftest')
    a = p.parse_args(argv)
    {'gt-stats': gt_stats, 'select': select, 'api': api, 'report': report, 'selftest': lambda _: selftest()}[a.cmd](a)


if __name__ == '__main__':
    main()
