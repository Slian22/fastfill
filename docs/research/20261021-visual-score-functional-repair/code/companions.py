"""Companion relations of a layout, by keyword rules on normalised free-text categories (floor roots only).

  adjacency  nightstand/bedside table -> bed (and beside the bed's head half: bed local front = +X, head = -X half,
             verified on GT: >= 95% of nightstands within 0.30 m of a bed sit in its -X half);
             desk/office/task chair -> desk; dining chair -> dining table;
             armchair/lounge chair/ottoman/footstool/coffee/side/end table -> nearest sofa.
             The anchor is the nearest anchor-category root by footprint gap; held if the gap <= 0.30 m.
  facing     TV / TV stand / TV cabinet / media console -> the nearest bed or sofa whose centre lies within +-45 deg
             of the companion's front axis (+X) and within 5 m (centre distance); held if that still holds.
pairs(objs) lists the pairs HELD in a layout as (kind, companion id, anchor id); holds(pair, index) re-tests one
pair on another layout of the same ids. retention(a, b) = held-in-a pairs still held in b (read from the layout
being repaired); gt_score(gt, b) uses GT pairs and is for evaluation only.

python3 companions.py --selfcheck
python3 companions.py --report OUT.json  ROWS.jsonl ROOMS.jsonl [ROWS ROOMS ...]   pairs per room in GT / raw / S1
"""
import json
import math
import sys

from shapely.geometry import Polygon

sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from fastfill.scene import footprint, norm_cat  # noqa: E402

GAP, CONE, FACE_DIST = 0.30, math.radians(45), 5.0
PROPS = ('book', 'lamp', 'light', 'remote', 'cup', 'mug', 'vase', 'candle', 'plant', 'bottle', 'tray', 'organizer', 'toy')
SOFA = {'sofa', 'couch', 'loveseat', 'sectional', 'settee', 'sofas'}
SOFA_COMP = ('armchair', 'lounge chair', 'accent chair', 'ottoman', 'footstool', 'pouf', 'coffee table', 'coffeetable',
             'side table', 'sidetable', 'end table', 'tea table', 'teatable')
DESK_CHAIR = ('desk chair', 'office chair', 'task chair', 'computer chair', 'swivel chair', 'executive chair',
              'work chair', 'studentchair', 'gaming chair', 'game chair')
# companion role -> (kind, anchor roles)
PAIRS = {'nightstand': ('adj', ('bed',)), 'desk_chair': ('adj', ('desk',)), 'dining_chair': ('adj', ('dining_table',)),
         'sofa_comp': ('adj', ('sofa',)), 'tv': ('face', ('bed', 'sofa'))}


def role(o):
    s = norm_cat(o.get('category'))
    w = s.split()
    h = w[-1] if w else ''
    if any(p in s for p in PROPS):
        return None
    if 'nightstand' in s or 'night stand' in s or 'night table' in s or 'bedside' in s:
        return 'nightstand'
    if 'tv' in w or 'television' in w or 'media console' in s or 'media cabinet' in s or 'entertainment' in s:
        return 'tv'
    if h.endswith('bed') or s == 'bed frame':
        return 'bed'
    if h in SOFA:
        return 'sofa'
    if 'dining chair' in s or 'diningchair' in s:
        return 'dining_chair'
    if 'dining table' in s or 'diningtable' in s:
        return 'dining_table'
    if any(k in s for k in DESK_CHAIR):
        return 'desk_chair'
    if h in ('desk', 'studentdesk'):
        return 'desk'
    if any(k in s for k in SOFA_COMP):
        return 'sofa_comp'
    return None


def _gap(a, b):
    return Polygon(footprint(a)).distance(Polygon(footprint(b)))


def _local_x(o, ref):
    return (o['pos'][0] - ref['pos'][0]) * math.cos(ref['yaw']) + (o['pos'][1] - ref['pos'][1]) * math.sin(ref['yaw'])


def _in_cone(c, a):
    dx, dy = a['pos'][0] - c['pos'][0], a['pos'][1] - c['pos'][1]
    ang = abs((math.atan2(dy, dx) - c['yaw'] + math.pi) % (2 * math.pi) - math.pi)
    return ang <= CONE + 1e-9 and math.hypot(dx, dy) <= FACE_DIST + 1e-9


def holds(p, index):
    kind, ci, ai = p
    c, a = index[ci], index[ai]
    if kind == 'face':
        return _in_cone(c, a)
    return _gap(c, a) <= GAP + 1e-9 and (kind != 'adj_head' or _local_x(c, a) < 0)


def roots(objs):
    objs = objs or []
    ids = {o["id"] for o in objs}
    return [o for o in objs if o.get('parent') not in ids]


def pairs(objs):
    """Held pairs of a layout: (kind, companion, anchor); kind adj | adj_head (nightstand) | face."""
    rs = roots(objs)
    roles = {o['id']: role(o) for o in rs}
    out = []
    for c in rs:
        spec = PAIRS.get(roles[c['id']])
        if not spec:
            continue
        kind, want = spec
        anchors = [a for a in rs if roles[a['id']] in want and a['id'] != c['id']]
        if kind == 'face':
            anchors = [a for a in anchors if _in_cone(c, a)]
            if anchors:
                a = min(anchors, key=lambda a: math.dist(c['pos'][:2], a['pos'][:2]))
                out.append(('face', c['id'], a['id']))
            continue
        if not anchors:
            continue
        a = min(anchors, key=lambda a: _gap(c, a))
        p = ('adj_head' if roles[c['id']] == 'nightstand' else 'adj', c['id'], a['id'])
        if holds(p, {c['id']: c, a['id']: a}):
            out.append(p)
    return out


def retention(src, out):
    """(held in both, held in src) for the pairs read from layout src, re-tested on layout out (same ids)."""
    ps = pairs(src)
    if not out:                       # an incomplete row holds none
        return 0, len(ps)
    idx = {o['id']: o for o in out}
    return sum(holds(p, idx) for p in ps), len(ps)


gt_score = retention   # retention(gt_objects, layout): GT pairs held in layout; evaluation only, never for repair


def report(out, args):
    res = {}
    for rows_f, rooms_f in zip(args[::2], args[1::2]):
        rooms = {json.loads(l)['uid']: json.loads(l) for l in open(rooms_f)}
        acc = {k: [] for k in ('gt', 'raw', 's1')}
        kinds = {k: {} for k in acc}
        hold = {'raw': [0, 0], 's1': [0, 0]}
        for l in open(rows_f):
            r = json.loads(l)
            gt = rooms[r['uid']]['objects']
            for k, objs in (('gt', gt), ('raw', r['raw_layout']), ('s1', r['s1_layout'])):
                ps = pairs(objs)
                acc[k].append(len(ps))
                for p in ps:
                    kinds[k][p[0]] = kinds[k].get(p[0], 0) + 1
            for k in ('raw', 's1'):
                a, b = gt_score(gt, r[k + '_layout'])
                hold[k][0] += a
                hold[k][1] += b
        res[rows_f] = {'rooms': len(acc['gt']),
                       **{k: {'pairs_mean': sum(v) / len(v), 'rooms_with_pairs': sum(x > 0 for x in v), 'pairs_total': sum(v),
                              'by_kind': kinds[k]} for k, v in acc.items()},
                       'gt_pairs_held_in': {k: v[0] / max(v[1], 1) for k, v in hold.items()}}
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


def selfcheck():
    bed = {'id': 'b', 'category': 'King-size Bed', 'size': [2.1, 1.8, 0.5], 'pos': [1.05, 1.5, 0], 'yaw': 0.0, 'parent': None}
    n1 = {'id': 'n1', 'category': 'Nightstand', 'size': [0.4, 0.4, 0.5], 'pos': [0.2, 0.35, 0], 'yaw': 0.0, 'parent': None}
    n2 = {**n1, 'id': 'n2', 'pos': [1.9, 0.35, 0]}                       # beside the foot half: not a nightstand pair
    tv = {'id': 't', 'category': 'TV Stand', 'size': [1.2, 0.4, 0.5], 'pos': [3.8, 1.5, 0], 'yaw': math.pi, 'parent': None}
    book = {'id': 'k', 'category': 'nightstandbook', 'size': [0.2, 0.1, 0.03], 'pos': [0.2, 0.35, 0.5], 'yaw': 0.0, 'parent': None}
    assert role(bed) == 'bed' and role(n1) == 'nightstand' and role(tv) == 'tv' and role(book) is None
    assert role({'category': 'Bookcase / jewelry Armoire'}) is None and role({'category': 'bedside table'}) == 'nightstand'
    assert role({'category': 'dining chair'}) == 'dining_chair' and role({'category': 'Lounge Chair / Cafe Chair'}) == 'sofa_comp'
    assert role({'category': 'L-shaped Sofa'}) == 'sofa' and role({'category': 'tv remote control'}) is None
    layout = [bed, n1, n2, tv, book]
    assert sorted(pairs(layout)) == [('adj_head', 'n1', 'b'), ('face', 't', 'b')], pairs(layout)
    moved = [bed, {**n1, 'pos': [0.2, 2.65, 0]}, n2, {**tv, 'yaw': math.pi / 2}, book]   # other head side; TV turned away
    assert retention(layout, moved) == (1, 2), retention(layout, moved)
    assert retention(layout, [bed, {**n1, 'pos': [3.0, 0.3, 0]}, n2, tv, book]) == (1, 2)
    print('companions selfcheck ok')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--selfcheck']:
        selfcheck()
    elif sys.argv[1:2] == ['--report']:
        report(sys.argv[2], sys.argv[3:])
