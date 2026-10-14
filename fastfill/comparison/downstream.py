"""Downstream acceptance, FastFill side: the EmbodiedGen growth backend's rules ported to IR rooms, and the
RoomGenBench export. EmbodiedGen is read only; rules cite embodied_gen/world/<file>:<line> at the time of the port.

    from fastfill.comparison import downstream
    verdict = downstream.accept(room, layout)    # room: IR (boundary, boundary_type, height, fixed, objects = request)
    condition, layout_v2, scene = downstream.to_roomgenbench(room, layout, names=sidecar['bound_to_original'])

No CLI: output_repair.py calls accept() per room. Metres, Z up, pos = bottom centre, yaw radians CCW about +Z,
footprint = fastfill.scene.footprint. accept() is pure (inputs are not modified).

Path A (deployed): FastFillBackend.realize -> fastfill_request -> serve.py -> ff_to_ours -> assert_attachable ->
admit.validate. accepted = endpoint_ok (R1, R2) and attach_ok (R5 as P1) and admit_ok (A0, A2, A3, A4, every door
an entry; A6 of an entry strip is implied by A2). Reported next to it, never instead of it: admit_lenient_ok (P4:
the best single door is the entry, every other door strip is A6 only) and admit_no_required_ok (P5 lower bound,
required = {}). Window overlap is a non-blocking metric: EmbodiedGen has no window keep-out at all
(AttachmentSpec has no window field, spec.py:264; portals.cut_window, portals.py:110, is never tested against
furniture), and validate.check only lists windows in fixed_collisions (validate.py:108-110).

Exact ports:
  R1  keepout_not_representable (interface.py:117-135, WALL_SNAP opening included; the cm-rounding refusal of
      clean_boundary, interface.py:139-143, is not ported)
  R2  serve.py's interface.snap_inside (fastfill/interface.py:255, moves <= 0.10 m) into keepout_floor (the floor
      minus every keep-out and minus the pockets under POCKET_M2 they cut off, interface.py:137), then check().valid
      (validate.py:72) and every footprint, stacked ones and floor coverings included, covered by that
      floor.buffer(1e-3) (interface.py:384-385); serve.py has no fixed geometry on this path (backend.py:1849-1859).
      Deployed, a FastFill-side repair must run inside serve.py before that validation (else the endpoint refuses
      the raw answer first)
  R4  support_parent becomes the parent only on a positive-area AABB overlap, no slack (backend.py:1786-1806,
      413-416); otherwise no parent (also a fixed id, gap G3)
  A0  per-category counts of the request (admit.py:303-381)
  floor covering: no parent after R4 and height < FLOOR_COVERING_H_M, at any z (backend.py:142, 247-254)
  A2  any non-covering instance, height and parent ignored, AABB overlap > TOL_M on both axes (admit.py:484-521,
      245-253; footprint_xy backend.py:229-244)
  A3  nav.reachable_cells (nav.py:81-130) seeded NAV_GRID_M inside the door midpoint (admit.py:256-278), blockers =
      parentless non-covering AABBs, codes and the 0.45 m touch rule of admit.py:524-617 (floor root, 281-298)
  A4  FLOATING and COLLISION (same parent, neither covering, height ignored, so a tucked chair counts) on AABBs
      with TOL_M (admit.py:620-702); MISSING_PARENT cannot fire after R4
Approximations [APPROX]:
  P1  R5 (_shift_ban, backend.py:257-297, 335-342) is footprint inside polygon.buffer(ENTRY_TOL_M): equal on
      axis-aligned rectangles (`rectangle`); the backend itself only builds the envelope rectangle (backend.py:1896)
  P2  AABB tests run in the frame of the minimum rotated rectangle (rectangles) or of the longest edge
  P3  door strips (door_strips) from fixed boxes with a door word: nearest boundary edge (v2/direct_layout.py _shell),
      along-wall width |sx (u.x_d)| + |sy (u.y_d)|, door +- KEEPOUT_SIDE_MARGIN_M, KEEPOUT_DEPTH_M deep, clipped to
      the polygon (growth_planner.py:197-235); a door over DETACHED_DOOR_M from every wall is flagged, strip kept
  P4  strict: every door is the entry (A2 + A3 from each door); lenient: as above
  P5  required = every requested category with its count; required = {} is the lower bound
  P6  A3 on polygons: envelope = bbox in the P2 frame, a cell is free only if its centre is in the polygon eroded by
      AGENT_RADIUS_M (redundant on rectangles)
  P7  R2 uses the room's own height, not WALL_HEIGHT_M = 3.0 (config.py:85)
Not computed (NOT_COMPUTED): A1 (only world-side arms are live), world-side A5, A7 (no planned doors), relations and
goals (none in our input; with them every room is refused today, gap G1), A3 of doorless rooms (no entry to seed).
Sizes are the layout's own: Path A always sends catalogue sizes and drops J's (gap G2); the caller chooses.
Doorless rooms: R2, P1, A0 and A4 only. Hull rooms: hull edges are not walls, so their strips are approximate.
"""
import math
import warnings
from collections import Counter, deque

from shapely.affinity import affine_transform
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.prepared import prep

from fastfill.interface import KEEPOUT_PAD, POCKET_M2, WALL_SNAP, snap_inside
from fastfill.scene import footprint, norm_cat
from fastfill.validate import check

ENTRY_TOL_M = 1e-3                                  # backend.py:129 (assert_attachable / _shift_ban)
TOL_M = 0.01                                        # admit.py:139
FLOOR_COVERING_H_M = 0.1                            # backend.py:142
AGENT_RADIUS_M, NAV_GRID_M = 0.2, 0.25              # nav.py:40, 43
MIN_REACHABLE_CELLS = 5                             # admit.py:131
MIN_DOORWAY_M = 2 * AGENT_RADIUS_M                  # admit.py:135
KEEPOUT_DEPTH_M, KEEPOUT_SIDE_MARGIN_M = 1.0, 0.15  # growth_planner.py:70-71
DETACHED_DOOR_M = 0.3                               # [APPROX P3]
DOOR_WORDS = {'door', 'doors', 'doorway', 'doorways'}
NOT_COMPUTED = ('A1', 'A5_world', 'A7', 'relations_goals')


def frame(boundary):
    """[APPROX P2] (theta, shapely affine matrix) mapping the room into its AABB frame, bbox minimum at (0, 0)."""
    poly = orient(Polygon(boundary))
    theta = 0.0
    if poly.envelope.area - poly.area > 1e-9 * poly.area:         # not an axis-aligned rectangle
        with warnings.catch_warnings():                            # GEOS divides by zero on axis-aligned edges
            warnings.simplefilter('ignore', RuntimeWarning)
            mrr = poly.minimum_rotated_rectangle
        ring = mrr if mrr.area - poly.area <= 1e-6 * poly.area else poly
        pts = list(ring.exterior.coords)
        a, b = max(zip(pts, pts[1:]), key=lambda e: math.dist(*e))
        theta = (math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi / 4) % (math.pi / 2) - math.pi / 4
        theta = 0.0 if abs(theta) < 1e-12 else theta
    c, s = math.cos(theta), math.sin(theta)
    x0, y0, _, _ = affine_transform(poly, [c, s, -s, c, 0, 0]).bounds
    return theta, [c, s, -s, c, -x0, -y0]


def aabb(o, m):
    """footprint_xy (backend.py:229-244) in the frame of matrix m: the AABB of the rotated box."""
    return affine_transform(Polygon(footprint(o)), m).bounds


def _overlaps(a, b, slack=TOL_M):
    """admit.py:245-253: shared area beyond `slack` on both axes."""
    return a[0] < b[2] - slack and b[0] < a[2] - slack and a[1] < b[3] - slack and b[1] < a[3] - slack


def _positive(a, b):
    """backend.py:413-416: any positive-area overlap, no slack."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _inside(p, r):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]          # nav.py:54-55, closed


def _inflate(r, d):
    return (r[0] - d, r[1] - d, r[2] + d, r[3] + d)                 # nav.py:49-51


def _crosses(start, end, rect):
    """nav.py:58-78: does the closed segment start-end meet the closed rect."""
    enter, leave = 0.0, 1.0
    for axis in (0, 1):
        delta = end[axis] - start[axis]
        if delta == 0.0:
            if not rect[axis] <= start[axis] <= rect[axis + 2]:
                return False
            continue
        near, far = sorted(((rect[axis] - start[axis]) / delta, (rect[axis + 2] - start[axis]) / delta))
        enter, leave = max(enter, near), min(leave, far)
        if enter > leave:
            return False
    return True


def door_strips(room):
    """[APPROX P3] [{'id', 'strip' (shapely, room frame), 'seed' (room frame), 'width', 'detached'}] per door."""
    poly = orient(Polygon(room['boundary']))
    pts = list(poly.exterior.coords)
    edges = [(a, b) for a, b in zip(pts, pts[1:]) if math.dist(a, b) > 1e-9]
    out = []
    for f in room.get('fixed') or []:
        if not DOOR_WORDS & set(norm_cat(f.get('category')).split()):
            continue
        p = Point(f['pos'][:2])
        gap, (a, b) = min(((LineString(e).distance(p), e) for e in edges), key=lambda t: t[0])
        length = math.dist(a, b)
        u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        n = (-u[1], u[0])                                # inward normal of a counter-clockwise ring
        t = min(max((p.x - a[0]) * u[0] + (p.y - a[1]) * u[1], 0.0), length)
        c, s = math.cos(f['yaw']), math.sin(f['yaw'])
        width = abs(f['size'][0] * (u[0] * c + u[1] * s)) + abs(f['size'][1] * (u[1] * c - u[0] * s))
        half = width / 2 + KEEPOUT_SIDE_MARGIN_M
        lo, hi = max(t - half, 0.0), min(t + half, length)
        at = lambda along, depth: (a[0] + u[0] * along + n[0] * depth, a[1] + u[1] * along + n[1] * depth)
        strip = Polygon([at(lo, 0), at(hi, 0), at(hi, KEEPOUT_DEPTH_M), at(lo, KEEPOUT_DEPTH_M)]).intersection(poly)
        out.append({'id': f['id'], 'strip': strip, 'seed': at(t, NAV_GRID_M), 'width': width,
                    'detached': gap > DETACHED_DOOR_M})
    return out


def _cells(rect, lo, ux, uy):
    """Grid cells (nav.py:96-104 lattice) whose centre is inside the closed rect."""
    span = lambda a, b, top: range(max(lo, math.floor(a / NAV_GRID_M - 0.5)), min(top, math.ceil(b / NAV_GRID_M - 0.5)) + 1)
    centre = lambda i, j: ((i + 0.5) * NAV_GRID_M, (j + 0.5) * NAV_GRID_M)
    return {(i, j) for i in span(rect[0], rect[2], ux) for j in span(rect[1], rect[3], uy) if _inside(centre(i, j), rect)}


def _reach(grid, seed):
    """nav.reachable_cells (nav.py:81-130) on a precomputed free-cell set: (cells, seeded)."""
    free, solid, (w, d), lo, ux, uy = grid
    if not _inside(seed, (0.0, 0.0, w, d)):
        return set(), False
    if lo > min(ux, uy):
        return set(), _inside(seed, _inflate((0.0, 0.0, w, d), -AGENT_RADIUS_M)) and \
            not any(_inside(seed, r) for r in solid)
    start = (min(max(int(seed[0] / NAV_GRID_M), lo), ux), min(max(int(seed[1] / NAV_GRID_M), lo), uy))
    if any(_crosses(seed, ((start[0] + 0.5) * NAV_GRID_M, (start[1] + 0.5) * NAV_GRID_M), r) for r in solid):
        return set(), False
    seen, queue = {start}, deque([start])
    while queue:
        i, j = queue.popleft()
        for nxt in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if nxt not in seen and nxt in free:
                seen.add(nxt)
                queue.append(nxt)
    return seen, True


def keepout_floor(poly, zones):
    """R1 and R2's floor, interface.py:117-137: (representable, floor). The padded keep-outs, after the WALL_SNAP
    opening, must leave one floor without an island; the floor is the polygon minus the keep-outs and minus the
    pockets under POCKET_M2 the padded keep-outs cut off."""
    if not zones:
        return True, poly
    cut = unary_union(zones)
    seen = poly.difference(cut.buffer(KEEPOUT_PAD, join_style=2))
    snap = seen.intersection(seen.buffer(-WALL_SNAP / 2, join_style=2).buffer(WALL_SNAP / 2, join_style=2))
    big = lambda g: sum(q.area >= POCKET_M2 for q in getattr(g, 'geoms', [g]))
    seen = seen if big(snap) > 1 and big(seen) == 1 else snap
    parts = sorted(getattr(seen, 'geoms', [seen]), key=lambda q: -q.area)
    pockets = [q for q in parts[1:] if q.area < POCKET_M2]
    ok = not (seen.is_empty or len(parts) - len(pockets) != 1 or len(getattr(parts[0], 'interiors', ())))
    return ok, poly.difference(unary_union([cut, *pockets]))


def r4_parents(by, bb):
    """R4, ff_to_ours.one (backend.py:1786-1806): {id: parent or None}. support_parent survives only on a
    positive-area overlap of the AABBs `bb` (one frame); a fixed or unknown id, or a cycle closing on the path, is
    None (the first object of a cycle reached keeps its parent)."""
    parent = {}

    def resolve(i, path):
        if i in parent:
            return
        sup = by[i].get('parent')
        if sup in by and sup not in path:
            resolve(sup, (*path, sup))
            parent[i] = sup if _positive(bb[i], bb[sup]) else None
        parent[i] = parent.get(i)
    for i in by:
        resolve(i, (i,))
    return parent


def covering(o, parent):
    """Floor covering (backend.py:142, 247-254): no parent after R4 and lower than FLOOR_COVERING_H_M, at any z."""
    return parent is None and o['size'][2] < FLOOR_COVERING_H_M


def is_rectangle(poly):
    return abs(poly.envelope.area - poly.area) <= 1e-9 * poly.area


def accept(room, layout):
    """Path A acceptance of one layout (IR objects) in its room; see the module docstring for every rule."""
    poly = orient(Polygon(room['boundary']))
    doors = door_strips(room)
    strips = [s['strip'] for s in doors]
    v = []                                     # (code, ids, door index or None, depends on the required set)
    representable, floor = keepout_floor(poly, strips)
    if not representable:
        v.append(('R1_KEEPOUT_NOT_REPRESENTABLE', [], None, False))
    pl = {o['id']: {'pos': list(o['pos']), 'yaw': o['yaw'], 'on': o.get('parent')} for o in layout}
    if not floor.is_empty:  # The endpoint refuses an unrepresentable keep-out (R1) before it would snap.
        snap_inside(pl, {'objects': layout, 'fixed': []}, floor, (), room['boundary'])
    lay = [{**o, 'pos': pl[o['id']]['pos']} for o in layout]
    c = check({**room, 'objects': lay})
    if not c['valid']:
        v.append(('R2_CHECK', sorted(set(c['oob'] + c['support_fail'] + c['ceiling'])), None, False))
    grown = prep(floor.buffer(ENTRY_TOL_M))
    v.append(('R2_OUTSIDE_1MM', [o['id'] for o in lay if not grown.covers(Polygon(footprint(o)))], None, False))
    env = prep(poly.buffer(ENTRY_TOL_M))
    v.append(('R5_OUTSIDE_ENVELOPE', [o['id'] for o in lay if not env.covers(Polygon(footprint(o)))], None, False))

    by = {o['id']: o for o in lay}
    theta, m = frame(room['boundary'])
    bb = {i: aabb(o, m) for i, o in by.items()}
    parent = r4_parents(by, bb)

    need, have = Counter(o['category'] for o in room['objects']), Counter(o['category'] for o in lay)
    v.append(('A0_REQUIRED_MISSING', sorted(k for k in need if have[k] < need[k]), None, True))
    v.append(('A4_FLOATING', [i for i, p in parent.items() if p and not _overlaps(bb[i], bb[p])], None, True))
    solid = [i for i in by if not covering(by[i], parent[i])]
    v += [('A4_COLLISION', [i, j], None, True) for k, i in enumerate(solid) for j in solid[k + 1:]
          if parent[i] == parent[j] and _overlaps(bb[i], bb[j])]

    fpoly = affine_transform(poly, m)
    w, d = fpoly.bounds[2:]
    lo = math.ceil(AGENT_RADIUS_M / NAV_GRID_M - 0.5)
    ux, uy = (math.floor((e - AGENT_RADIUS_M) / NAV_GRID_M - 0.5) for e in (w, d))
    rects = [_inflate(bb[i], AGENT_RADIUS_M) for i in solid if parent[i] is None]
    blocked = set().union(*(_cells(r, lo, ux, uy) for r in rects))
    inner = prep(fpoly.buffer(-AGENT_RADIUS_M))
    free = {(i, j) for i in range(lo, ux + 1) for j in range(lo, uy + 1)
            if (i, j) not in blocked and inner.covers(Point((i + 0.5) * NAV_GRID_M, (j + 0.5) * NAV_GRID_M))}
    grid = (free, rects, (w, d), lo, ux, uy)

    def root(i):                               # admit._floor_root, admit.py:281-298
        seen = {i}
        while parent[i] and parent[i] not in seen:
            i = parent[i]
            seen.add(i)
        return i
    lenient = []                               # A6 hits per door, lenient variant only
    for k, s in enumerate(doors):
        shrunk = affine_transform(s['strip'], m).buffer(-TOL_M, join_style=2)
        hit = [i for i in solid if box(*bb[i]).intersection(shrunk).area > 0]
        v.append(('A2_KEEPOUT_BLOCKED', hit, k, False))
        lenient.append([i for i in hit if parent[i] is None])
        if s['width'] <= MIN_DOORWAY_M:
            v.append(('A3_DOORWAY_TOO_NARROW', [], k, False))
        reached, seeded = _reach(grid, (m[0] * s['seed'][0] + m[1] * s['seed'][1] + m[4],
                                        m[2] * s['seed'][0] + m[3] * s['seed'][1] + m[5]))
        if not seeded:
            v.append(('A3_ENTRY_BLOCKED', [], k, False))
            continue
        if len(reached) < MIN_REACHABLE_CELLS:
            v.append(('A3_TOO_FEW_REACHABLE', [], k, False))
        for cls in sorted(need):
            members = [i for i in by if by[i]['category'] == cls]
            far = [i for i in members if not reached & _cells(_inflate(bb[root(i)], AGENT_RADIUS_M + NAV_GRID_M),
                                                               lo, ux, uy)]
            if len(members) - len(far) < need[cls]:
                v.append(('A3_UNREACHABLE_REQUIRED', far, k, True))

    v = [x for x in v if x[1] or not x[0].startswith(('R2_OUTSIDE', 'R5_', 'A0_', 'A4_FLOATING', 'A2_'))]
    base = [x for x in v if x[2] is None and x[0].startswith('A')]
    at = lambda k, required=True: [x for x in v if x[2] == k and (required or not x[3])]
    endpoint_ok = not [x for x in v if x[0].startswith(('R1', 'R2'))]
    attach_ok = not [x for x in v if x[0].startswith('R5')]
    admit_ok = not base and all(not at(k) for k in range(len(doors)))
    objects = {}
    for code, ids, _, _ in v:
        for i in ids:
            objects.setdefault(i, []).append(code)
    return {'accepted': endpoint_ok and attach_ok and admit_ok, 'endpoint_ok': endpoint_ok, 'attach_ok': attach_ok,
            'admit_ok': admit_ok,
            'admit_lenient_ok': not base and (not doors or any(not at(e) and not any(
                lenient[k] for k in range(len(doors)) if k != e) for e in range(len(doors)))),
            'admit_no_required_ok': not [x for x in base if not x[3]] and all(
                not at(k, False) for k in range(len(doors))),
            'reasons': sorted({x[0] for x in v}),
            'violations': [{'code': x[0], 'ids': x[1], 'door': None if x[2] is None else doors[x[2]]['id']} for x in v],
            'objects': {i: sorted(set(codes)) for i, codes in objects.items()},
            'doors': [{'id': s['id'], 'width': s['width'], 'detached': s['detached'], 'a6': lenient[k]}
                      for k, s in enumerate(doors)],
            'parents': parent, 'snap_moved': [o['id'] for o, q in zip(layout, lay) if o['pos'] != q['pos']],
            'window_overlap': [list(p) for p in c['fixed_collisions'] if p not in c['fixed_blocking']],
            'rectangle': is_rectangle(poly),
            'boundary_type': room.get('boundary_type'), 'theta': theta,
            'not_computed': list(NOT_COMPUTED) + ([] if doors else ['A3'])}


def to_roomgenbench(room, layout, names=None):
    """Section 4: v2 condition + layout -> fastfill.v2.direct_layout.layout_to_roomgenbench (reused, not copied).

    names: bound -> original ids (contract sidecar `bound_to_original`). description = category [APPROX: our
    input has none]. support_parent is the layout's own parent; without one it is left out, so infer_support
    derives floor / on_object / wall from the boxes (never invented), and a None place_id is refused as
    roomgenbench._handoff(require_placement=True) does (roomgenbench.py:46-55).
    """
    from fastfill.v2 import SCHEMA_VERSION                     # lazy: fastfill.v2 imports torch
    from fastfill.v2.direct_layout import layout_to_roomgenbench
    from fastfill.v2.geometry import wrap_yaw
    name = lambda i: (names or {}).get(i, i)
    fixed = [{'id': name(f['id']), 'category': f.get('category') or 'object', 'size_local_m': list(f['size']),
              'bottom_center_m': list(f['pos']), 'yaw_rad': f['yaw']} for f in room.get('fixed') or []]
    condition = {'schema_version': SCHEMA_VERSION, 'constraints': [],
                 'room': {'frame': 'right_handed_z_up', 'floor_polygon_xy_m': [list(p) for p in room['boundary']],
                          'floor_z_m': 0.0, 'height_m': room.get('height'), 'room_type': room.get('room_type'),
                          'fixed_objects': fixed},
                 'objects': [{'id': name(o['id']), 'category': o.get('category') or 'object',
                              'description': o.get('category') or 'object',
                              **({'support_parent': name(o['parent'])} if o.get('parent') else {})} for o in layout]}
    result = {'schema_version': SCHEMA_VERSION,
              'objects': [{'id': name(o['id']), 'target_size_local_m': list(o['size']),
                           'bottom_center_m': list(o['pos']), 'yaw_rad': wrap_yaw(o['yaw'])} for o in layout]}
    scene = layout_to_roomgenbench(condition, result)
    if any(o['place_id'] is None for o in scene['objects']):
        raise ValueError('every requested object needs an explicit placement')
    return condition, result, scene
