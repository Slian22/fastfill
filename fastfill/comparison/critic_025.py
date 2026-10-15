"""Amendment 025: SceneSmith-style critic -> constrained repair on the 27B placer + S1 (dev selection, local logic).

Spec: docs/research/20261013-unknown-size-comparison/amendment-025-critic-repair-20261015.md. Nothing here calls an
API or a server: the critic (gpt-6.1-sol, effort medium) and the judge run on the server through the parent; this
module builds their inputs and reads their answers. Outputs are write-once directories with MANIFEST.json.

    H=/Volumes/harddisk/fastfillout/unknownsize-20261013; IN=$H/a025-inputs; OUT=$H/a025
    python -m fastfill.comparison.critic_025 rows --data $IN/dev-select-v1 --arm $IN/a021-dev-select-arm \
        --out $OUT/round-0 [--workers 8]
    python -m fastfill.comparison.critic_025 round --prev $OUT/round-0 --answers ANSWERS-1 --out $OUT/round-1
    python -m fastfill.comparison.critic_025 round --prev $OUT/round-1 --answers ANSWERS-2 --out $OUT/round-2
    python -m fastfill.comparison.critic_025 evaluate --b $OUT/round-0 --c $OUT/round-2 [--judge JUDGE_OUT] --out $OUT/eval

rows      Arm B. The 27B vLLM greedy dev-select records scored as a021_select.load_run scores them (evaluate_pair._score,
          placement; report_v2.cap_rule: a cap hit without EOS is incomplete) of the a021 27B run (receipt config.model
          .../a021/27b, vLLM LoRA); Addendum A selection: per amendment-015 family (room_families.family of the raw
          room_type) at most PER_FAMILY complete rooms with the lowest sha256('a025:' + uid), families in FAMILIES
          order (count and per-family counts in summary.json before any critic call); S1 = output_repair.repair_row
          at the amendment-012 settings (S1). -> round-0 state.
round     Round k = prev round + 1 (at most MAX_ROUNDS). Reads the round-k critic answers (ANSWERS: any *.json below it
          holding {"id": case id, "raw_final_text": answer}: critic_batch_025's server_api_judge .result.json records;
          each must carry the expected_model CRITIC_MODEL and the content_sha256 of its case in the previous state's
          critic-manifest.json, else the round is refused), applies each active room's answer (_step) and writes the
          round-k state; the rooms that go on get round-(k+1) cases. A missing answer, an empty one or a record that is
          not status completed from CRITIC_MODEL is a failed call: the room stays unedited and stops (reported, never
          retried), even when the failed record carries answer text.
evaluate  B (round-0 state) vs C (the last round's state) on the same rooms, metrics recomputed from the layouts:
          frozen checker, strict admission, required = {} admission, OOR, displacement of C vs B (median / p95 of
          moved objects), share of rooms edited, invariant counts, critic call / edit accounting and, given --judge (a
          judge_batch output with arms B and C), the judge endpoint (functionality + layout) / 2 per room. Paired C - B
          contrasts: report_v2.contrasts (house-cluster bootstrap, REPEATS draws, seed 42). The judge run must be on
          judge_bundle_025 dev's bundle of exactly these B and C (summary.json geometry_manifest_sha256 =
          judge_lineage of their MANIFEST hashes). Adoption (section 2): judge
          CI lower bound > 0, strict admission C - B >= -0.005, checker C - B >= -0.005; without --judge: pending.
          -> results.json + report.md (Chinese).

State directory (round-0 = B, round-k): rows.jsonl (per room: uid, source, group, the request-frame room geometry
'room' = output_repair.predicted_room without objects, the current 'layout', its 'metrics', 'active' = gets a
next-round case, and the per-round 'history'), summary.json, and for every active room the next round's critic case:
KEY-rK-top.png, KEY-rK-oblique.png, KEY-rK-prompt.txt (KEY = sha256(uid)[:16]) listed in critic-manifest.json
(server_api_judge manifest schema, expected_model CRITIC_MODEL) and critic-mapping.json {case id: uid}.

Views (render_views): matplotlib Agg, VIEW_INCHES x VIEW_DPI = 1024 px, no PNG metadata (deterministic). Top view and
one oblique view (elevation 35, azimuth -60): the boundary, fixed boxes (door green, window cyan, other grey), every
movable object as its box with its id and a black front arrow, coloured by checker flag (first that applies: main
collision red (incl. severe), fixed_blocking brown, oob orange, support purple, ok grey-blue). Flags come from
output_repair.audit (check() and geometry()'s support rule).
Front convention: an object's front is its local +X axis; yaw (radians in the IR, degrees in the prompt and in edits)
runs counter-clockwise from room +X to the front (scene.py IR doc, validate.holds 'faces', the P request prompt);
the back is local -X (inspect_ir's wall test). sx is the extent along the front, sy across it.

Critic answer (parse_edits, strict, never repaired): exactly {"scores": {"functionality": int 0-10, "layout": int
0-10}, "edits": [...]}, else the whole answer is invalid (codes invalid_json, invalid_keys, invalid_scores,
invalid_edits). Each edit has exactly the fields of its op (OPS + id, op, reason): an edit past MAX_EDITS
(over_limit), not an object (not_object), another op (forbidden_op: add, remove, resize, relabel, reparent, ...),
extra or missing fields (unexpected_field, missing_field), an unknown id (unknown_id), a non-string reason
(invalid_reason), a non-finite number or one beyond +-BOUND (invalid_number), an unknown or own target
(unknown_target, self_target) is rejected alone; the valid edits are applied in order as one edit set.
Ops (apply_edits): move sets (x, y); rotate sets yaw; against_wall moves the object to touch the nearest boundary
edge (to its centre; ties: ring order of the counter-clockwise boundary) with its back, front along the inward normal,
centred on the projection of its centre and clamped so its width fits on that edge when it can; face turns its front
to the target's centre (a movable or fixed id; a coincident centre: degenerate_target, edit skipped). Every op moves
the object's support subtree rigidly (XY and yaw; z, sizes, categories, ids and parents never change). Then S1 (same
settings) and the re-check: the candidate replaces the previous layout only if the frozen-checker key KEY (valid
first, then counts) is not lexicographically worse, strict admission does not go pass -> fail and the invariant counts
are 0; otherwise the previous layout stays. An accepted set that changed nothing is recorded as 'unchanged' and still
counts as accepted (section 1: those two conditions only).
Stop (no round k+1) when the call failed, the answer is invalid, no edit is valid, the edit set was not accepted, both
scores are >= STOP_SCORE (this round's accepted edits are still kept) or k = MAX_ROUNDS.
"""
import argparse
import collections
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient

from fastfill.comparison import a021_select as A
from fastfill.comparison import downstream as d
from fastfill.comparison import dpo_stages as dps
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import judge_batch
from fastfill.comparison import output_repair as rep
from fastfill.comparison import render_assets as render
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.room_families import FAMILIES, family
from fastfill.scene import footprint, norm_cat

ARM, SALT, PER_FAMILY, MODEL_SUFFIX = 'placement', 'a025:', 31, '/a021/27b'   # Addendum A
S1 = {'max_trials': 2000, 'time_limit': 60.0}               # amendment 012 (= fastfill.system.pipeline.S1)
CRITIC_MODEL, MAX_EDITS, MAX_ROUNDS, STOP_SCORE = 'gpt-6.1-sol', 5, 2, 8
BOUND = 1e6                       # |edit number| above this is invalid (shapely fails near 1e300)
OPS = {'move': ('x', 'y'), 'rotate': ('yaw_deg',), 'against_wall': (), 'face': ('target',)}
BASE = ('id', 'op', 'reason')
KEY = ('invalid', 'fixed_blocking', 'severe', 'main', 'oob', 'support_fail', 'ceiling')
FIELDS = ('geometry_success', 'accepted', 'admitted_no_required', 'oor')
REPEATS, SEED, SLACK = 2000, 42, -0.005
FLAGS = ('main', 'fixed_blocking', 'oob', 'support')
COLOURS = {'main': '#d62728', 'fixed_blocking': '#8c564b', 'oob': '#ff7f0e', 'support': '#9467bd', 'ok': '#9fb7cf'}
VIEW_INCHES, VIEW_DPI = 8, 128
TAU = 2 * math.pi
CODES = ('Codes: R1/R2 = refused by the endpoint (keep-out not representable, invalid layout, footprint outside the '
         'usable floor by > 1 mm); R5 = outside the room envelope; A0 = a requested object missing; A2_KEEPOUT_BLOCKED = '
         'an object in the 1 m deep keep-out strip of a door; A3 = a door entry blocked or required objects unreachable '
         'from it (0.2 m agent radius); A4_COLLISION = two boxes on the same support overlap (axis-aligned, 1 cm); '
         'A4_FLOATING = a supported object off its support.')


# ------------------------------------------------------------------ selection, metrics

def rank(uid):
    return hashlib.sha256((SALT + uid).encode('utf-8')).hexdigest()


def select(complete_uids, room_type):
    """Addendum A: per family (FAMILIES order) the PER_FAMILY complete rooms with the lowest sha256('a025:' + uid).
    room_type: {uid: raw room_type}. -> (uids, {family: count})."""
    pools = collections.defaultdict(list)
    for u in set(complete_uids):
        pools[family(room_type[u])].append(u)
    picked = {f: sorted(pools[f], key=lambda u: (rank(u), u))[:PER_FAMILY] for f in FAMILIES}
    return [u for f in FAMILIES for u in picked[f]], {f: len(picked[f]) for f in FAMILIES}


def metrics(room, layout):
    """Frozen checker (geometry.geometry), its lexicographic key and downstream.accept on one layout."""
    pred = {**room, 'objects': layout}
    geo, v = g.geometry(pred), d.accept(pred, layout)
    key = [int(not geo['geometry_valid']), geo['fixed_blocking'], geo['severe_collisions'], geo['main_collisions'],
           geo['oob_objects'], geo['support_fail'], geo['ceiling_fail']]
    return {'geometry': geo, 'key': key, 'accepted': v['accepted'],
            'admitted_no_required': v['endpoint_ok'] and v['attach_ok'] and v['admit_no_required_ok'],
            'reasons': v['reasons'], 'violations': v['violations']}


def _metrics(task):
    return metrics(*task)


def verdict(before, after):
    """Re-check reasons (empty = accept): checker key lexicographically worse, strict admission pass -> fail."""
    return (['checker_worse'] if tuple(after['key']) > tuple(before['key']) else []) + \
        (['admission_worse'] if before['accepted'] and not after['accepted'] else [])


def invariants(before, after):
    """Changes that must never happen (all 0): object count, ids/order, categories, sizes, parents."""
    pairs = list(zip(before, after))
    return {'count': int(len(before) != len(after)), **{k: sum(a.get(k) != b.get(k) for a, b in pairs)
                                                         for k in ('id', 'category', 'size', 'parent')}}


# ------------------------------------------------------------------ critic answers

def _edit_error(edit, index, ids, targets):
    if index >= MAX_EDITS:
        return 'over_limit'
    if not isinstance(edit, dict):
        return 'not_object'
    op = edit.get('op')
    if not isinstance(op, str) or op not in OPS:
        return 'forbidden_op'
    fields = set(BASE + OPS[op])
    if set(edit) - fields:
        return 'unexpected_field'
    if fields - set(edit):
        return 'missing_field'
    if not isinstance(edit['id'], str) or edit['id'] not in ids:
        return 'unknown_id'
    if not isinstance(edit['reason'], str):
        return 'invalid_reason'
    if any(type(edit[f]) not in (int, float) or not math.isfinite(edit[f]) or abs(edit[f]) > BOUND
           for f in OPS[op] if f != 'target'):
        return 'invalid_number'
    if op == 'face':
        if not isinstance(edit['target'], str) or edit['target'] not in targets:
            return 'unknown_target'
        if edit['target'] == edit['id']:
            return 'self_target'
    return None


def parse_edits(text, ids, targets):
    """Strict critic answer -> {'error', 'scores', 'edits' (valid, in order), 'rejected' [{index, code, edit}]}."""
    out = {'error': None, 'scores': None, 'edits': [], 'rejected': []}
    try:
        value = h.strict_json(text)
    except (ValueError, TypeError, RecursionError):
        return {**out, 'error': 'invalid_json'}
    if not isinstance(value, dict) or set(value) != {'scores', 'edits'}:
        return {**out, 'error': 'invalid_keys'}
    scores = value['scores']
    if not isinstance(scores, dict) or set(scores) != {'functionality', 'layout'} or \
            any(type(v) is not int or not 0 <= v <= 10 for v in scores.values()):
        return {**out, 'error': 'invalid_scores'}
    if not isinstance(value['edits'], list):
        return {**out, 'error': 'invalid_edits'}
    out['scores'] = scores
    for index, edit in enumerate(value['edits']):
        code = _edit_error(edit, index, ids, targets)
        if code:
            out['rejected'].append({'index': index, 'code': code, 'edit': edit})
        else:
            out['edits'].append(edit)
    return out


# ------------------------------------------------------------------ edits

def _subtree(layout, i):
    kids = collections.defaultdict(list)
    for o in layout:
        kids[o.get('parent')].append(o['id'])
    out, todo = [], [i]
    while todo:
        j = todo.pop()
        if j not in out:
            out.append(j)
            todo += kids[j]
    return set(out)


def _rigid(layout, i, x, y, yaw):
    """Pose object i at (x, y, yaw); its support subtree follows rigidly (XY rotation about i's centre, yaw)."""
    o = next(q for q in layout if q['id'] == i)
    x0, y0, turn = o['pos'][0], o['pos'][1], yaw - o['yaw']
    c, s = math.cos(turn), math.sin(turn)
    moved, out = _subtree(layout, i), []
    for q in layout:
        if q['id'] in moved:
            rx, ry = q['pos'][0] - x0, q['pos'][1] - y0
            q = {**q, 'pos': [x + c * rx - s * ry, y + s * rx + c * ry, q['pos'][2]],
                 'yaw': (q['yaw'] + turn) % TAU if turn else q['yaw']}
        out.append(q)
    return out


def against_wall(boundary, o):
    """(x, y, yaw): back (local -X) on the nearest boundary edge, front along its inward normal."""
    pts = list(orient(Polygon(boundary)).exterior.coords)
    edges = [(a, b) for a, b in zip(pts, pts[1:]) if math.dist(a, b) > 1e-9]
    p = Point(o['pos'][:2])
    a, b = min(edges, key=lambda ab: LineString(ab).distance(p))      # min keeps the first edge on ties
    length = math.dist(a, b)
    u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
    n = (-u[1], u[0])                                                   # inward normal of a counter-clockwise ring
    half = o['size'][1] / 2
    t = (p.x - a[0]) * u[0] + (p.y - a[1]) * u[1]
    t = min(max(t, half), length - half) if length >= 2 * half else length / 2
    depth = o['size'][0] / 2
    return a[0] + u[0] * t + n[0] * depth, a[1] + u[1] * t + n[1] * depth, math.atan2(n[1], n[0]) % TAU


def apply_edit(room, layout, edit):
    """One validated edit -> (layout, None) or (layout unchanged, skip code)."""
    by = {o['id']: o for o in layout}
    o, op = by[edit['id']], edit['op']
    x, y, yaw = o['pos'][0], o['pos'][1], o['yaw']
    if op == 'move':
        x, y = edit['x'], edit['y']
    elif op == 'rotate':
        yaw = math.radians(edit['yaw_deg']) % TAU
    elif op == 'face':
        t = by.get(edit['target']) or next(f for f in room['fixed'] if f['id'] == edit['target'])
        dx, dy = t['pos'][0] - x, t['pos'][1] - y
        if math.hypot(dx, dy) < 1e-9:
            return layout, 'degenerate_target'
        yaw = math.atan2(dy, dx) % TAU
    else:
        x, y, yaw = against_wall(room['boundary'], o)
    return _rigid(layout, edit['id'], x, y, yaw), None


def apply_edits(room, layout, edits, settings=S1):
    """Apply one edit set, S1 (sizes frozen), re-check against the previous layout; see the module docstring."""
    cur, skipped = copy.deepcopy(layout), []
    for index, edit in enumerate(edits):
        cur, code = apply_edit(room, cur, edit)
        if code:
            skipped.append({'index': index, 'code': code})
    result = rep.repair({**room, 'objects': cur}, **settings)
    candidate = result['room']['objects'] if result['status'] != 'rejected' else cur
    before, after = metrics(room, layout), metrics(room, candidate)
    inv = invariants(layout, candidate)
    reasons = verdict(before, after) + (['invariant'] if any(inv.values()) else [])
    disp = [math.dist(a['pos'][:2], b['pos'][:2]) for a, b in zip(layout, candidate)]
    turned = sum(a['yaw'] != b['yaw'] for a, b in zip(layout, candidate))
    accepted = not reasons
    return {'accepted': accepted, 'reasons': reasons, 'unchanged': accepted and not turned and not any(disp),
            'layout': candidate if accepted else layout,
            'metrics': after if accepted else before, 'before_key': before['key'], 'candidate_key': after['key'],
            'candidate_accepted': after['accepted'], 'invariants': inv, 'skipped': skipped,
            'displacement': rep._displacement(disp), 'yaw_changed': turned,
            's1': {'status': result['status'], **{k: result['diagnostics'].get(k) for k in ('timed_out', 'rolled_back')}}}


# ------------------------------------------------------------------ views and prompt

def flags(room, layout):
    """({id: checker flag}, audit lists) on the frozen checker's lists (output_repair.audit)."""
    lists = rep.audit({**room, 'objects': layout}, [], [])['lists']
    bad = {'main': {i for pair in lists['main'] + lists['severe'] for i in pair},
           'fixed_blocking': {pair[0] for pair in lists['fixed_blocking']}, 'oob': set(lists['oob']),
           'support': set(lists['support_fail'])}
    return {o['id']: next((f for f in FLAGS if o['id'] in bad[f]), 'ok') for o in layout}, lists


def _fixed_colour(f):
    words = set(norm_cat(f.get('category')).split())
    return '#2ca02c' if words & d.DOOR_WORDS else '#17becf' if words & {'window', 'windows'} else '#7f7f7f'


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=VIEW_DPI, metadata={'Software': None})
    return buffer.getvalue()


def _arrow(o):
    """Centre and tip of the front arrow (local +X), 0.45 of the depth, at least 0.15 m."""
    length = max(0.45 * o['size'][0], 0.15)
    x, y = o['pos'][:2]
    return (x, y), (x + length * math.cos(o['yaw']), y + length * math.sin(o['yaw']))


def render_views(room, layout, flag, title, flag_legend=True):
    """(top PNG bytes, oblique PNG bytes), VIEW_INCHES x VIEW_DPI px each; flag_legend=False drops the checker-flag
    legend entries (the judge's neutral views, all boxes 'ok')."""
    order = sorted(layout, key=lambda o: (o['pos'][2], o['id']))
    xs, ys = zip(*room['boundary'])
    ring = list(room['boundary']) + [room['boundary'][0]]
    legend = [Patch(color=COLOURS[k], label=k) for k in (*FLAGS, 'ok') if flag_legend] + \
        [Patch(color=c, label=k) for k, c in (('door', '#2ca02c'), ('window', '#17becf'), ('fixed', '#7f7f7f'))]

    fig = Figure(figsize=(VIEW_INCHES, VIEW_INCHES), dpi=VIEW_DPI)
    ax = fig.add_axes([0.07, 0.1, 0.9, 0.83])
    ax.plot(*zip(*ring), color='black', lw=2)
    for f in room['fixed']:
        ax.add_patch(MplPolygon(footprint(f), closed=True, fc=_fixed_colour(f), ec='black', alpha=0.6, lw=0.8))
        ax.text(*f['pos'][:2], f"{f['id']}\n{f.get('category')}", fontsize=5, ha='center', va='center')
    for o in order:
        ax.add_patch(MplPolygon(footprint(o), closed=True, fc=COLOURS[flag[o['id']]], ec='#222222', alpha=0.55,
                                lw=0.8))
        start, tip = _arrow(o)
        ax.annotate('', xy=tip, xytext=start, arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1.0})
        ax.text(*start, o['id'], fontsize=6, ha='center', va='top' if o.get('parent') else 'bottom')
    pad = 0.3
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect('equal')
    ax.grid(True, ls=':', lw=0.5)
    fig.legend(handles=legend, fontsize=7, loc='lower center', ncol=len(legend))
    ax.set_title(f'{title} - top view (arrow = front, local +X)', fontsize=9)
    top = _png(fig)

    fig = Figure(figsize=(VIEW_INCHES, VIEW_INCHES), dpi=VIEW_DPI)
    ax = fig.add_subplot(projection='3d', computed_zorder=False)   # painter's order below; labels drawn last
    ax.plot(*zip(*ring), [0.0] * len(ring), color='black', lw=2)
    zmax, (elev, azim) = 0.5, (35, -60)
    eye = (math.cos(math.radians(azim)), math.sin(math.radians(azim)))
    boxes = [(f, _fixed_colour(f), 0.25, None) for f in room['fixed']] + \
        [(o, COLOURS[flag[o['id']]], 0.5, o['id']) for o in order]
    for box, colour, alpha, label in sorted(boxes, key=lambda b: (b[0]['pos'][0] * eye[0] + b[0]['pos'][1] * eye[1],
                                                                  b[0]['pos'][2], b[0]['id'])):
        z0, z1 = box['pos'][2], box['pos'][2] + box['size'][2]
        zmax = max(zmax, z1)
        q = footprint(box)
        faces = [[(*p, z0) for p in q], [(*p, z1) for p in q]] + \
            [[(*q[k], z0), (*q[(k + 1) % 4], z0), (*q[(k + 1) % 4], z1), (*q[k], z1)] for k in range(4)]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=colour, edgecolors='#222222', linewidths=0.3,
                                             alpha=alpha))
        if label:
            start, tip = _arrow(box)
            ax.plot([start[0], tip[0]], [start[1], tip[1]], [z1, z1], color='black', lw=1.2, zorder=5)
            ax.scatter([tip[0]], [tip[1]], [z1], color='black', s=6, zorder=5)
            ax.text(*start, z1 + 0.05, label, fontsize=5, zorder=6)
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((max(xs) - min(xs), max(ys) - min(ys), zmax))
    ax.view_init(elev=elev, azim=azim)
    fig.legend(handles=legend, fontsize=7, loc='lower center', ncol=len(legend))
    ax.set_title(f'{title} - oblique view (line + dot = front)', fontsize=9)
    return top, _png(fig)


def _r(values, n=3):
    return [round(v, n) for v in values]


def prompt(row, k, lists):
    """The critic case text of one room in round k (rubric, schema, room/object JSON, checker, admission)."""
    room, layout, m = row['room'], row['layout'], row['metrics']
    deg = lambda yaw: round(math.degrees(yaw) % 360, 1)
    fixed = [{'id': f['id'], 'category': f.get('category'), 'size': _r(f['size']), 'pos': _r(f['pos']),
              'yaw': deg(f['yaw'])} for f in room['fixed']]
    objects = [{'id': o['id'], 'category': o['category'], 'size': _r(o['size']), 'pos': _r(o['pos']),
                'yaw': deg(o['yaw']), 'parent': o.get('parent')} for o in layout]
    geo = m['geometry']
    checker = {'geometry_success': geo['geometry_success'], 'valid': geo['geometry_valid'],
               'main_collisions': lists['main'], 'severe_collisions': lists['severe'],
               'fixed_blocking': lists['fixed_blocking'], 'out_of_room': lists['oob'],
               'support_fail': lists['support_fail'], 'ceiling': lists['ceiling']}
    admission = {'accepted': m['accepted'], 'accepted_without_required_counts': m['admitted_no_required'],
                 'failure_reasons': m['reasons'], 'violations': m['violations']}
    dump = lambda v: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
    return '\n'.join([
        f'You are the critic in a furniture-layout repair loop, round {k} of at most {MAX_ROUNDS}. A placement model '
        'laid out the room below and a deterministic solver then removed what collisions it could. Judge whether the '
        f'furniture is usable and sensibly arranged and propose at most {MAX_EDITS} edits.',
        '',
        'Conventions: metres, Z up, floor z = 0. pos = [x, y, z] is the bottom centre of a box; size = [sx, sy, sz] are '
        "its extents along the object's local X, Y, Z. The FRONT of an object is its local +X axis (its back is local "
        '-X); yaw is in degrees, counter-clockwise from room +X to the front. The boundary is a counter-clockwise floor '
        'polygon (boundary_type polygon = walls, hull = scanned floor outline). parent = the object it stands on.',
        '',
        'Images: (1) top view, (2) oblique view. Every object is drawn as its exact box (there are no meshes: '
        'overlapping boxes really collide), labelled with its id, with a black arrow pointing to its front. Colours: '
        'red = collision, brown = blocks a door or another fixed item, orange = outside the room, purple = support '
        'problem, grey-blue = ok; fixed items: green = door, cyan = window, grey = other.',
        'The checker and admission results below are computed exactly from the geometry; they take precedence over '
        'anything you read from the images.',
        '',
        'Scores, integers 0-10: functionality = can every object be used as intended (reachable, facing the right way, '
        'not blocked); layout = is the arrangement sensible (walls, alignment, grouping, circulation).',
        'Checklist:',
        "- a bed's headboard (its back) is against a wall;",
        '- chairs face their desk or table; a sofa faces the TV;',
        '- storage (wardrobes, cabinets, shelves, dressers) stands against a wall with its front reachable;',
        '- there is a clear path from each door into the room;',
        '- nothing blocks a door or a window.',
        '',
        'Edits (objects standing on the edited object move with it rigidly):',
        '- {"id": ID, "op": "move", "x": X, "y": Y, "reason": TEXT}: put the centre at (x, y);',
        '- {"id": ID, "op": "rotate", "yaw_deg": DEG, "reason": TEXT}: set the yaw (front direction), degrees;',
        '- {"id": ID, "op": "against_wall", "reason": TEXT}: move it to touch the nearest wall with its back, front '
        'facing into the room;',
        '- {"id": ID, "op": "face", "target": TARGET_ID, "reason": TEXT}: turn it so its front faces the centre of '
        'the target (a movable or fixed id).',
        'Use exactly these fields for each op. Objects cannot be added, removed, resized, relabelled or re-parented; '
        'such edits and unknown ids are rejected. After your edits the solver removes collisions, and the edit set is '
        'kept only if the checker and admission results do not get worse; otherwise the layout stays as it is. Return '
        'an empty edits list if no edit would help.',
        '',
        'Return only one JSON object, no markdown:',
        '{"scores": {"functionality": INT, "layout": INT}, "edits": [EDIT, ...]}',
        '',
        'Room: ' + dump({'room_type': room.get('room_type'), 'boundary_type': room['boundary_type'],
                         'boundary': room['boundary'], 'height': room.get('height'), 'fixed': fixed}),
        '',
        'Objects: ' + dump(objects),
        '',
        'Frozen checker: ' + dump(checker),
        '',
        'Strict admission (downstream backend rules): ' + dump(admission),
        CODES, ''])


def key_of(uid):
    return hashlib.sha256(uid.encode('utf-8')).hexdigest()[:16]


def case_id(uid, k):
    return f'a025:r{k}:{uid}'


def case_files(task):
    """(state row, round k) -> ({file name: bytes}, manifest case) of its round-k critic case."""
    row, k = task
    flag, lists = flags(row['room'], row['layout'])
    stem = f"{key_of(row['uid'])}-r{k}"
    top, oblique = render_views(row['room'], row['layout'], flag, f"{row['uid'][-48:]} r{k}")
    text = prompt(row, k, lists).encode('utf-8')
    files = {f'{stem}-prompt.txt': text, f'{stem}-top.png': top, f'{stem}-oblique.png': oblique}
    digest = lambda b: hashlib.sha256(b).hexdigest()
    case = {'id': case_id(row['uid'], k), 'prompt_file': f'{stem}-prompt.txt', 'prompt_sha256': digest(text),
            'images': [{'path': name, 'sha256': digest(files[name])} for name in (f'{stem}-top.png',
                                                                                  f'{stem}-oblique.png')]}
    return files, case


# ------------------------------------------------------------------ rounds

def _step(task):
    """(state row, answer record or None, round k) -> the row after round k (see the module docstring)."""
    row, record, k = task
    room, layout, m = row['room'], row['layout'], row['metrics']
    record = record or {}
    entry = {'round': k, 'case_id': case_id(row['uid'], k), 'returned_model': record.get('returned_model'),
             'record_status': record.get('status'), 'record_error': record.get('error')}
    text, stop = record.get('raw_final_text'), None
    if record.get('status') != 'completed' or record.get('returned_model') != CRITIC_MODEL \
            or not isinstance(text, str) or not text.strip():
        entry['call'], stop = 'failed', 'call_failed'
    else:
        ids = {o['id'] for o in layout}
        parsed = parse_edits(text, ids, ids | {f['id'] for f in room['fixed']})
        entry |= {'call': 'answered', 'error': parsed['error'], 'scores': parsed['scores'],
                  'edits': parsed['edits'], 'rejected': parsed['rejected']}
        if parsed['error']:
            stop = 'invalid_answer'
        elif not parsed['edits']:
            stop = 'no_valid_edits'
        else:
            result = apply_edits(room, layout, parsed['edits'])
            entry |= {key: result[key] for key in ('accepted', 'reasons', 'unchanged', 'before_key', 'candidate_key',
                                                   'candidate_accepted', 'invariants', 'skipped', 'displacement',
                                                   'yaw_changed', 's1')}
            layout, m = result['layout'], result['metrics']
            stop = None if result['accepted'] else 'edit_set_rejected'
        if stop is None and min(parsed['scores'].values()) >= STOP_SCORE:
            stop = 'scores_ge_8'
    if stop is None and k >= MAX_ROUNDS:
        stop = 'max_rounds'
    return {**row, 'round': k, 'layout': layout, 'metrics': m, 'active': stop is None,
            'history': row['history'] + [{**entry, 'stop': stop}]}


def load_answers(root):
    """{case id: record} from every *.json below root that holds 'raw_final_text' (duplicates refused)."""
    out = {}
    for path in sorted(Path(root).rglob('*.json')):
        value = h.strict_json(path.read_text())
        if isinstance(value, dict) and 'raw_final_text' in value:
            if value.get('id') in out:
                raise ValueError(f'duplicate answer for {value.get("id")}')
            out[value.get('id')] = value
    return out


def _round_stats(rows, k):
    entries = [x for r in rows for x in r['history'] if x['round'] == k]
    count = lambda key: dict(sorted(collections.Counter(key(x) for x in entries if key(x) is not None).items()))
    return {'round': k, 'cases': len(entries), 'calls': count(lambda x: x['call']),
            'answer_errors': count(lambda x: x.get('error')),
            'edits_valid': sum(len(x.get('edits') or []) for x in entries),
            'edits_rejected': dict(sorted(collections.Counter(j['code'] for x in entries
                                                               for j in x.get('rejected') or []).items())),
            'edit_sets': sum('accepted' in x for x in entries), 'edit_sets_accepted': sum(bool(x.get('accepted'))
                                                                                        for x in entries),
            'edit_sets_unchanged': sum(bool(x.get('unchanged')) for x in entries),
            'set_reject_reasons': dict(sorted(collections.Counter(c for x in entries
                                                                  for c in x.get('reasons') or []).items())),
            's1_timed_out': sum(bool((x.get('s1') or {}).get('timed_out')) for x in entries),
            'stops': count(lambda x: x['stop']), 'next_round_rooms': sum(x['stop'] is None for x in entries)}


def _publish_state(out, rows, summary, workers):
    k = summary['round'] + 1
    built = r2._map(case_files, [(r, k) for r in rows if r['active']], workers)
    files = {name: data for f, _ in built for name, data in f.items()}
    if built:
        files['critic-manifest.json'] = _json({'schema_version': 1, 'expected_model': CRITIC_MODEL,
                                               'cases': [c for _, c in built]})
        files['critic-mapping.json'] = _json({c['id']: r['uid'] for (_, c), r in
                                              zip(built, [r for r in rows if r['active']])})
    files['rows.jsonl'] = dps._jsonl(rows)
    files['summary.json'] = _json({**summary, 'next_round_cases': len(built), 'code_sha256': _code()})
    dps._publish_dir(out, files)


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _code():
    return {Path(p).name: h.sha256(p) for p in (__file__, rep.__file__, d.__file__, g.__file__)}


def _state(root):
    """(rows, summary, MANIFEST sha256) of a verified state directory."""
    root = Path(root)
    dps._verified_dir(root)
    rows = [h.strict_json(line) for line in (root / 'rows.jsonl').read_text().splitlines()]
    return rows, h.strict_json((root / 'summary.json').read_text()), h.sha256(root / 'MANIFEST.json')


def _initial(task):
    """(repair_row output, GT room, request) -> the round-0 state row (arm B), checked against S1's own verdict."""
    x, gt, request = task
    pred = rep.predicted_room(gt, request['messages'], [])
    room = {k: pred[k] for k in ('uid', 'source', 'group', 'room_type', 'boundary_type', 'boundary', 'height', 'fixed')}
    m = metrics(room, x['s1_layout'])
    if m['geometry'] != x['s1']['geometry_s1'] or m['accepted'] != x['s1']['strict_ok_s1']:
        raise AssertionError(f"B metrics differ from S1's own for {x['uid']}")
    return {'uid': x['uid'], 'source': x['source'], 'group': x['group'], 'objects': x['objects'], 'room': room,
            'layout': x['s1_layout'], 'metrics': m, 'round': 0, 'active': True, 'history': [],
            'b': {'raw_geometry_success': x['geometry_success'], 's1_status': x['s1']['status'],
                  's1_timed_out': x['s1']['timed_out']}}


def rows(data, arm, out, workers=1):
    data, arm, out = Path(data), Path(arm), Path(out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    expected, rooms, requests = A._slice(data)
    records, receipt = e._inference(arm, data, expected)
    config = receipt['config']
    if config.get('arm') != ARM or not str(config.get('model')).endswith(MODEL_SUFFIX) \
            or config.get('lora') != A.VLLM_LORA:
        raise ValueError(f'the arm run is not the {ARM} vLLM run of ...{MODEL_SUFFIX}')
    scored = r2._map(e._score, [(rooms[x['uid']], requests[x['uid']], x, ARM) for x in records], workers)
    scored, flipped = r2.cap_rule(scored, records)
    chosen, per_family = select([s['uid'] for s in scored if s['complete']],
                                {u: r.get('room_type') for u, r in rooms.items()})
    by = {s['uid']: s for s in scored}
    repaired = r2._map(rep.repair_row, [(by[u], rooms[u], requests[u], S1['max_trials'], S1['time_limit'])
                                       for u in chosen], workers)
    state = r2._map(_initial, [(x, rooms[x['uid']], requests[x['uid']]) for x in repaired], workers)
    summary = {'status': 'A025_B_COMPLETE', 'round': 0, 'arm': 'B = 27B vLLM greedy + S1',
               'selection': {'salt': SALT, 'per_family_cap': PER_FAMILY, 'rooms': len(chosen), 'per_family': per_family,
                             'requests': len(records), 'complete': sum(s['complete'] for s in scored),
                             'cap_failures': flipped, 'uids': chosen},
               'inputs': {'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'), 'run_sha256': receipt['run_sha256'],
                          'model': config['model'], 'lora': config['lora'],
                          'run_manifest_sha256': h.sha256(arm / 'MANIFEST.json')},
               's1': S1, 'b_manifest_sha256': None, 'prev_manifest_sha256': None,
               'b_status': dict(collections.Counter(r['b']['s1_status'] for r in state))}
    _publish_state(out, state, summary, workers)
    return {'status': summary['status'], 'rooms': len(state), 'complete': summary['selection']['complete']}


def round_(prev, answers, out, workers=1):
    prev, out = Path(prev), Path(out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    rows_, summary, prev_sha = _state(prev)
    k = summary['round'] + 1
    expected = {case_id(r['uid'], k): r['uid'] for r in rows_ if r['active']}
    if not expected:
        raise ValueError('no room continues to round %d' % k)
    if h.strict_json((prev / 'critic-mapping.json').read_text()) != expected:
        raise ValueError('critic-mapping.json differs from the active rooms')
    got = load_answers(answers)
    unknown = sorted(map(str, set(got) - set(expected)))
    if unknown:
        raise ValueError(f'answers for cases outside round {k}: {unknown[:3]}')
    content = {c['id']: judge_batch.content_hash(c)
               for c in h.strict_json((prev / 'critic-manifest.json').read_text())['cases']}
    stale = sorted(i for i, x in got.items() if x.get('content_sha256') != content.get(i)
                   or x.get('expected_model') != CRITIC_MODEL)
    if stale:
        raise ValueError(f'answers not made on this round\'s cases or model: {stale[:3]}')
    stepped = r2._map(_step, [(r, got.get(case_id(r['uid'], k)), k) for r in rows_ if r['active']], workers)
    by = {r['uid']: r for r in stepped}
    state = [by.get(r['uid'], {**r, 'round': k}) for r in rows_]
    stats = _round_stats(state, k)
    _publish_state(out, state, {'status': 'A025_ROUND_COMPLETE', 'round': k, 's1': S1, 'stats': stats,
                                'answers_found': len(got), 'prev_manifest_sha256': prev_sha,
                                'b_manifest_sha256': summary['b_manifest_sha256'] or prev_sha}, workers)
    return {'status': 'A025_ROUND_COMPLETE', 'round': k, **{x: stats[x] for x in ('cases', 'edit_sets_accepted',
                                                                                 'next_round_rooms')}}


# ------------------------------------------------------------------ evaluation

def judge_lineage(b_sha, c_sha):
    """The judge bundle's geometry_manifest_sha256 for states B and C (as judge_bundle_025.dev; path-free)."""
    return render.json_sha256({'B': b_sha, 'C': c_sha})


def judge_values(root, uids):
    """{(uid, arm): (functionality + layout) / 2 or None} from a judge_batch output with arms B and C."""
    records = [h.strict_json(x) for x in (Path(root) / 'records.jsonl').read_text().splitlines()]
    if {r['arm'] for r in records} != {'B', 'C'} or not {r['uid'] for r in records} <= set(uids):
        raise ValueError('judge records must cover arms B and C on the amendment-025 rooms only')
    out = {}
    for r in records:
        s = r.get('scores') or {}
        ok = r['status'] == 'completed' and all(type(s.get(f'{m}_score')) is int for m in ('functionality', 'layout'))
        out[r['uid'], r['arm']] = (s['functionality_score'] + s['layout_score']) / 2 if ok else None
    return out


COVERAGE = 0.9                     # addendum B: >= 90% of the rooms with both arms judged
LEGEND_MARK = 'The boxes in the images are labelled by id'   # addendum B formal prompt (judge_bundle_025._legend)


def adoption(contrast, rooms=None):
    """Section 2 rule + addendum B coverage; the judge part is pending (None) without judge rows and fails closed on an
    undefined CI."""
    judge = contrast.get('judge')
    rules = {'judge_ci_lower_gt_0': None if judge is None else bool(judge['ci95'] and judge['ci95'][0] > 0),
             **({'judge_coverage_ge_0.9': judge['n'] >= COVERAGE * rooms} if judge is not None and rooms else {}),
             'strict_admission_ge_-0.005': contrast['accepted']['estimate'] >= SLACK,
             'checker_ge_-0.005': contrast['geometry_success']['estimate'] >= SLACK}
    return {'rules': rules, 'adopted': None if rules['judge_ci_lower_gt_0'] is None else all(rules.values())}


def _quantiles(values):
    return {'n': len(values), 'median': statistics.median(values) if values else None,
            'p95': g._quantile(values, .95) if values else None}


def evaluate(b, c, out, judge=None, workers=1):
    out = Path(out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    b_rows, b_summary, b_sha = _state(b)
    c_rows, c_summary, c_sha = _state(c)
    if b_summary['round'] != 0 or c_summary['round'] < 1 or c_summary['b_manifest_sha256'] != b_sha \
            or [r['uid'] for r in b_rows] != [r['uid'] for r in c_rows]:
        raise ValueError('C must be a round state built on this B')
    if any(r['active'] for r in c_rows):
        raise ValueError('C is not final: some rooms go on to the next round')
    uids = [r['uid'] for r in b_rows]
    if judge:
        if h.strict_json((Path(judge) / 'summary.json').read_text()).get('geometry_manifest_sha256') != \
                judge_lineage(b_sha, c_sha):
            raise ValueError('the judge run was not made on judge_bundle_025 dev of these B and C states')
        prompts = sorted((Path(judge) / 'artifacts').glob('*.txt'))
        if not prompts or any(LEGEND_MARK not in x.read_text(encoding='utf-8') for x in prompts):
            raise ValueError('the judge run did not use the addendum-B prompt (id -> category list)')
    js = judge_values(judge, uids) if judge else {}
    arms = {}
    for name, rows_ in (('B', b_rows), ('C', c_rows)):
        ms = r2._map(_metrics, [(r['room'], r['layout']) for r in rows_], workers)
        arms[name] = [{'uid': r['uid'], 'source': r['source'], 'group': r['group'], 'complete': True,
                       'geometry_success': m['geometry']['geometry_success'], 'accepted': m['accepted'],
                       'admitted_no_required': m['admitted_no_required'], 'oor': m['geometry']['oor'],
                       **({'judge': js.get((r['uid'], name))} if judge else {})} for r, m in zip(rows_, ms)]
    fields = FIELDS + (('judge',) if judge else ())
    contrast = r2.contrasts(arms['C'], arms['B'], fields, REPEATS, SEED)
    mean = lambda v: statistics.mean(v) if v else None
    rates = {name: {f: mean([float(x[f]) for x in rows_ if x[f] is not None]) for f in fields}
             for name, rows_ in arms.items()}
    moved, turned, edited, inv = [], 0, 0, collections.Counter()
    for rb, rc in zip(b_rows, c_rows):
        disp = [math.dist(x['pos'][:2], y['pos'][:2]) for x, y in zip(rb['layout'], rc['layout'])]
        yaw = sum(x['yaw'] != y['yaw'] for x, y in zip(rb['layout'], rc['layout']))
        moved += [v for v in disp if v > 0]
        turned += yaw
        edited += bool(yaw or any(disp))
        inv.update(invariants(rb['layout'], rc['layout']))
    rounds = sorted({x['round'] for r in c_rows for x in r['history']})
    value = {'protocol': 'amendment-025-dev', 'rooms': len(uids), 'per_family': b_summary['selection']['per_family'],
             'b_manifest_sha256': b_sha,
             'c_manifest_sha256': c_sha, 'c_round': c_summary['round'], 'rates': rates,
             'contrast_c_minus_b': contrast, 'bootstrap': {'draws': REPEATS, 'seed': SEED, 'cluster': 'house group'},
             'displacement_moved_objects_m': _quantiles(moved), 'moved_objects': len(moved), 'yaw_changed_objects': turned,
             'rooms_edited': edited, 'share_rooms_edited': edited / len(uids),
             'invariants': {k: inv[k] for k in ('count', 'id', 'category', 'size', 'parent')},
             'critic': {k: _round_stats(c_rows, k) for k in rounds},
             'judge': None if not judge else {'records_sha256': h.sha256(Path(judge) / 'records.jsonl'),
                                              'both_scored': contrast['judge']['n'],
                                              'summary_sha256': h.sha256(Path(judge) / 'summary.json'),
                                              'image_source': 'box_views'},
             'adoption': adoption(contrast, len(uids)), 'code_sha256': _code()}
    dps._publish_dir(out, {'results.json': _json(value), 'report.md': _report(value).encode('utf-8')})
    return {'status': 'A025_EVALUATED', 'adopted': value['adoption']['adopted'],
            **{f: contrast[f]['estimate'] for f in fields}}


def _report(v):
    num = lambda x, n=3: '—' if x is None else f'{x:.{n}f}'
    ci = lambda c: '—' if not c['ci95'] else f"[{num(c['ci95'][0])}, {num(c['ci95'][1])}]"
    names = {'geometry_success': '冻结检查器成功率', 'accepted': '严格准入率', 'admitted_no_required': 'required = {} 准入率',
             'oor': 'OOR（越小越好）', 'judge': 'GPT judge (功能 + 布局) / 2'}
    lines = ['# 修订 025：critic → 受限修复（dev 选择，探索性）', '',
             f"B = 27B + S1，C = 27B + S1 + critic 修复（最多 {MAX_ROUNDS} 轮，最终状态为第 {v['c_round']} 轮）。"
             f"{v['rooms']} 间 dev 房间，指标都由布局重新计算；C − B 为配对差，房屋连通组 bootstrap "
             f"{v['bootstrap']['draws']} 次，seed {v['bootstrap']['seed']}。", '',
             '房间选择（Addendum A）：每个修订 015 房间族最多 31 间，族内取 sha256(\'a025:\' + uid) 最小者。', '',
             '| 房间族 | 房间数 |', '|---|---:|', *[f'| {f} | {n} |' for f, n in v['per_family'].items()], '',
             '| 指标 | B | C | C − B | 95% CI | n |', '|---|---:|---:|---:|---:|---:|']
    for f, c in v['contrast_c_minus_b'].items():
        lines.append(f"| {names[f]} | {num(v['rates']['B'][f])} | {num(v['rates']['C'][f])} | {num(c['estimate'], 4)} "
                     f"| {ci(c)} | {c['n']} |")
    dq = v['displacement_moved_objects_m']
    rule = v['adoption']
    verdict_ = {None: '待定（尚无 judge 结果）', True: '采纳', False: '不采纳'}[rule['adopted']]
    lines += ['', f"被编辑房间：{v['rooms_edited']} / {v['rooms']}（{num(v['share_rooms_edited'])}）；"
              f"被移动物体 {v['moved_objects']} 个，位移（C 相对 B）中位 {num(dq['median'])} m、p95 {num(dq['p95'])} m；"
              f"朝向改变的物体 {v['yaw_changed_objects']} 个。",
              f"不变量计数（物体数、id、类别、尺寸、父物体改变，必须全为 0）：{v['invariants']}。", '',
              '| 轮 | case | 调用结果 | 有效编辑 | 被拒编辑（代码） | 编辑集 接受 / 总数 | 编辑集被拒原因 | 停止原因 | 进入下一轮 |',
              '|---:|---:|---|---:|---|---:|---|---|---:|']
    for k, s in v['critic'].items():
        lines.append(f"| {k} | {s['cases']} | {s['calls']} | {s['edits_valid']} | {s['edits_rejected'] or '无'} | "
                     f"{s['edit_sets_accepted']} / {s['edit_sets']}（其中无变化 {s['edit_sets_unchanged']}） | "
                     f"{s['set_reject_reasons'] or '无'} | {s['stops']} | "
                     f"{s['next_round_rooms']} |")
    lines += ['', f'采纳规则（修订 025 §2，三条都要满足）：{rule["rules"]}。结论：**{verdict_}**。', '',
              '失败或无效的 critic 调用不重试、不替换，房间保持未编辑并计入上表。按构造，C 的检查器和严格准入在每个房间都不劣于 B'
              '（编辑集只有在字典序不变差、严格准入不由过变不过时才被接受）。这是 dev 上的探索性结果，不改变任何已报告结果。']
    if v['judge']:
        lines += ['', '披露：dev 房间无法接入 run-002 的 Blender 管线，judge 看的是 critic 的同款盒子视图（judge_bundle_025，'
                  'image_source = box_views），物体按冻结检查器标记着色；按构造 C 的检查器结果不劣于 B，'
                  '这种着色可能使 judge 端点偏向 C。']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('rows')
    for flag in ('data', 'arm', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step = sub.add_parser('round')
    for flag in ('prev', 'answers', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step = sub.add_parser('evaluate')
    for flag in ('b', 'c', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--judge', type=Path)
    for step in sub.choices.values():
        step.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == 'rows':
        result = rows(args.data, args.arm, args.out, args.workers)
    elif args.command == 'round':
        result = round_(args.prev, args.answers, args.out, args.workers)
    else:
        result = evaluate(args.b, args.c, args.out, args.judge, args.workers)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
