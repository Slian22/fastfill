"""Amendment 016 §1 (scale-pass curves) and §6 (checker validity): EXPLORATORY and descriptive; sizes are intervened
on, nothing else, and no result or decision of amendments 005-015 changes.

python -m fastfill.comparison.scale_curve --data RUN/eval-data-v1 --geometry RUN/geometry-v5 \
    --cohort RUN/llm-cohort-v1.json --out NEW_DIR \
    [--joint-run RUN/infer-joint-002 --placement-run RUN/infer-placement-005] \
    [--agent NAME=RUN/llm-eval-NAME-selected-v1 ...] [--extra NAME=ROWS.jsonl[::KEY=VALUE,...] ...] \
    [--s1 SYSTEM=RUN/s1-012-test/DIR ...] [--workers N] [--repeats 2000]

Systems: GT = report_v2._gt (the GT answer scored on the J request, as reference_rows scores it); J and P = the
--geometry rows (evaluate_pair, MANIFEST-verified; summary GEOMETRY_COMPLETE on --data with the running geometry.py),
with --joint-run/--placement-run the amendment-005 cap rule (report_v2.cap_rule on those inference records, which
--geometry must score); every agent's llm_harness and llm_raw rows (NAME-harness, NAME-raw; report_v2._agent: complete,
on --data, paired with the J/P runs --geometry scores, rows = the cohort; size-free only: an amendment-016 §2
size-fixed evaluation (request_arm placement) is refused; the cap rule per row where rows carry the flags, as
report_v2); each --extra a geometry.score-style rows file listed in its directory's MANIFEST (uid, source, group,
complete, raw_layout) keeping the rows whose KEY equals VALUE, e.g.
RUN/optiscene-014/score/e2-rows.jsonl::arm=dpo2-e2-sft,mode=prior. Rooms come from --data, contexts from its requests
(P: placement, everyone else: joint; boundary, height and fixed boxes are the same in both).
Scopes: all-test (GT, J, P on every test room), cohort (GT, J, P and every agent on the cohort uids), extra-K (GT, J,
P and the extras sharing the K-th distinct extra uid set).
Intervention, s in GRID: each object's x/y (variant xy) or x/y/z (variant xyz) size times s about its own bottom
centre; yaw and the parent link are kept, and so are the x/y of unsupported objects. Departure from "positions kept"
(review 2026-10-14): a supported child keeps its place on its parent, i.e. its offset from the parent's centre scales
with the parent (else shrinking a table drops a cup near its edge off the 5 cm support tolerance and the curves count
that artefact), and it rises or falls with the parent's top (xyz; support gaps kept, addendum B §3); a fixed parent
never moves and s = 1 is the identity (resized). Each layout is re-scored in its request context
(output_repair.predicted_room, as geometry.score builds it) with the frozen checker (geometry.geometry,
geometry_success) and the downstream port (downstream.accept, 'accepted'); s = 1 must reproduce each row's
geometry_success and checker failure counts. An incomplete (or cap-failed) row fails at every s; the denominator is
every request.
Size substitution (addendum A §3, the primary §1 evidence): on the same layout, every object's x/y (variant xy) or
x/y/z (variant xyz) size replaced by the train prior (the P request's sizes) or by GT (the eval room's), with the same
support rule (resized); per system pass(own) - pass(prior) and pass(own) - pass(GT), paired (same rooms and draws),
next to the footprint bias B (report_v2.size_bias, mean over the system's complete rooms). The curves only show
sensitivity: GT is drawn on every panel as the reference and "steep" is never read as gaming on its own.
Per scope, subset (all; object-count bins <= 20, 21-50, > 50; SOURCE_KIND real / synthetic; room family as
room_families.scopes: paper / other families, then each family) and system: the rate at each s with a house-cluster
bootstrap band (geometry.paired_intervals' design, single-arm: components = house groups, strata = source
signatures, 2,000 draws, seed 42, room micro), the OLS slope over s in WINDOW, s_match = the s nearest 1 at which
the curve meets GT's rate at s = 1 in the same subset (linear between grid points; None when it never does), the
substitution effects and B, with intervals from the same draws (GT paired). House rule (report_v2._design): no
interval below NO_CI house components; flags 'few clusters' (k < 30) and 'small n' (n < 50).
§6 at s = 1: the checker at OOB tolerance 1 mm / 1 cm / 10 cm (geometry.geometry with validate.check's oob_tol;
10 cm = frozen), the checker with tucked seats counted as collisions (validate.check 'tuck' pairs fail, as downstream
A4 counts them), downstream acceptance and the share with an A4_COLLISION; checker and downstream failure reasons per
source; the rule-level table (addendum A §4): per subset and system the share of rooms failing each checker rule
(CHECKER_REASONS) and each downstream.accept reason code, raw and, for every --s1 SYSTEM=DIR (output_repair of that
system's rows file at the amendment-012 settings, test_apply_012._system), after S1 (rows the system fails stay
incomplete). A downstream variant that exempts tucked seats is not computed: accept() has no such switch.
Outputs (write-once): results.json, curves.csv, summary.csv, validity.csv, reasons.csv, rules.csv, rows.jsonl (per
room and system), report.md (Chinese), curves.png (matplotlib Agg; skipped with a note when matplotlib is missing),
MANIFEST.json.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import random
import tempfile
from unittest import mock

import numpy as np

from fastfill.comparison import downstream as dn
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import output_repair as r
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import room_families as rf
from fastfill.comparison import test_apply_012 as t12
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.validate import check

GRID = tuple(round(.70 + .05 * k, 2) for k in range(9))           # 0.70, 0.75, ..., 1.10
ONE = GRID.index(1.0)
VARIANTS = ('xy', 'xyz')
SOURCES = ('prior', 'gt')                                         # size substitution (addendum A §3)
MEASURES = ('checker', 'downstream')
WINDOW = (.80, 1.00)
OOB = (('1mm', .001), ('1cm', .01), ('10cm', .10))                # validate.check oob_tol; 0.10 = the frozen checker
VALIDITY = ('complete', 'checker_oob_1mm', 'checker_oob_1cm', 'checker_oob_10cm', 'checker_tuck_counted',
            'downstream', 'downstream_a4_collision')
CHECKER_REASONS = ('oob_objects', 'support_fail', 'ceiling_fail', 'main_collisions', 'severe_collisions',
                   'fixed_blocking')
BINS = (('objects<=20', 1, 20), ('objects21-50', 21, 50), ('objects>50', 51, None))
SEED = 42
LABEL = 'amendment-016 §1/§6 (EXPLORATORY, descriptive): amendments 005-015 results and decisions unchanged'
# Amendment 016 §5: real = scanned or designer-authored layouts, synthetic = generated ones (the rest). Origins as
# recorded in fastfill/README.md. The amendment's list is binding, IL3D_3dfront included (see its note).
SOURCE_KIND = {
    'InternScenes_3rscan': ('real', '3RScan RGB-D scans of real rooms (InternScenes annotation)'),
    'InternScenes_arkit': ('real', 'ARKitScenes real scans (InternScenes annotation)'),
    'InternScenes_mp3d': ('real', 'Matterport3D real scans (InternScenes annotation)'),
    'InternScenes_scannet': ('real', 'ScanNet real scans (InternScenes annotation)'),
    'MultiScan': ('real', 'real RGB-D scans with annotated fronts'),
    'Scan2CAD': ('real', 'ScanNet scans with aligned CAD models'),
    'SpatialLM': ('real', 'designer floor plans with official wall polygons; human layouts, not a scan'),
    'HSSD200': ('real', 'artist-authored homes with official region polygons; human layouts, not a scan'),
    'InteriorGS': ('real', 'professionally designed interiors with official room profiles; not a scan'),
    'Structured3D': ('real', 'professional house designs with official room polygons and heights; not a scan'),
    'IL3D_3dfront': ('synthetic', 'not in the amendment-016 real list; NOTE: 3D-FRONT rooms are designer-authored, '
                                  'like Structured3D, so by layout origin they would count as real'),
    'IL3D_synthetic': ('synthetic', 'procedurally synthesised rooms on HSSD assets'),
    'InternScenes_gen': ('synthetic', 'Infinigen procedural generation'),
    'MansionWorld': ('synthetic', 'procedurally generated buildings'),
    'OptiScene_holodeck': ('synthetic', 'Holodeck (LLM) generated rooms'),
    'SAGE-10k': ('synthetic', 'agent-generated single rooms'),
    'SceneSmith': ('synthetic', 'held-out scene generator (evaluation only)'),
    'SpatialGen': ('synthetic', 'held-out scene generator (evaluation only)'),
}


def kind(source):
    if source not in SOURCE_KIND:
        raise ValueError(f'source {source!r} has no real/synthetic entry in SOURCE_KIND')
    return SOURCE_KIND[source][0]


def resized(objects, sizes):
    """The objects with these sizes (in order), each about its own bottom centre; yaw and parent kept. A supported
    child keeps its place on its parent: its offset from the parent's centre, in the parent's frame, scales with the
    parent's x/y size, and it rises or falls with the parent's top (support gaps kept). Unsupported objects keep their
    pos, a fixed parent never moves, unchanged sizes give the same layout. Complete rows are acyclic and sizes
    positive (contract.validate_output)."""
    by, moved = {o['id']: (o, n) for o, n in zip(objects, sizes, strict=True)}, {}

    def shift(i):
        if i not in moved:
            o = by[i][0]
            if o.get('parent') in by:
                (q, n), (ax, ay, az) = by[o['parent']], shift(o['parent'])
                c, s = math.cos(q['yaw']), math.sin(q['yaw'])
                dx, dy = o['pos'][0] - q['pos'][0], o['pos'][1] - q['pos'][1]
                u = (n[0] / q['size'][0] - 1) * (c * dx + s * dy)            # parent-frame offset change
                v = (n[1] / q['size'][1] - 1) * (c * dy - s * dx)
                moved[i] = (ax + c * u - s * v, ay + s * u + c * v, az + n[2] - q['size'][2])
            else:
                moved[i] = (0.0, 0.0, 0.0)
        return moved[i]
    return [{**o, 'size': list(n), 'pos': [x + d for x, d in zip(o['pos'], shift(o['id']))]}
            for o, n in zip(objects, sizes)]


def scaled(objects, s, variant):
    """Every object's x/y (xyz: and z) size times s (resized)."""
    tall = variant == 'xyz'
    return resized(objects, [[o['size'][0] * s, o['size'][1] * s, o['size'][2] * (s if tall else 1)] for o in objects])


def substituted(objects, given, variant):
    """Every object's x/y (xyz: and z) size replaced by given[id] (resized)."""
    return resized(objects, [given[o['id']][:2] + (given[o['id']][2:] if variant == 'xyz' else o['size'][2:])
                             for o in objects])


def _probe(pred, tol):
    """geometry.geometry with validate.check's oob_tol = tol (0.10 = the frozen checker), and that check() result."""
    seen = []
    with mock.patch.object(g, 'check', lambda room: seen.append(check(room, oob_tol=tol)) or seen[0]):
        return g.geometry(pred), seen[0]


def evaluate_row(task):
    """(system, scored row or None for GT, room, its request, the P request) -> checker / downstream per variant and
    s and after size substitution, B, the §6 flags and the failure reasons at s = 1."""
    system, row, room, request, placement = task
    row = row or r2._gt((room, request))
    out = {'system': system, 'uid': room['uid'], 'complete': bool(row['complete'])}
    if not row['complete']:
        fail = {v: [False] * len(GRID) for v in VARIANTS}
        swap = {m: {v: dict.fromkeys(SOURCES, False) for v in VARIANTS} for m in MEASURES}
        return {**out, 'checker': fail, 'downstream': fail, 'substitution': swap, 'bias_fp': None,
                'validity': dict.fromkeys(VALIDITY, False), 'checker_reasons': ['incomplete'],
                'downstream_reasons': ['incomplete']}
    base = r.predicted_room(room, request['messages'], row['raw_layout'])

    def score(objects):
        pred = {**base, 'objects': objects}
        geo, verdict = g.geometry(pred), dn.accept(pred, objects)
        return {'checker': geo['geometry_success'], 'downstream': verdict['accepted'], 'geo': geo, 'verdict': verdict}
    scores = {}
    for v in VARIANTS:
        for s in GRID:
            if (s, v if s != 1 else 'xy') not in scores:              # s = 1 is the same layout in both variants
                scores[s, v if s != 1 else 'xy'] = score(scaled(base['objects'], s, v))
    geo, verdict = scores[1.0, 'xy']['geo'], scores[1.0, 'xy']['verdict']
    if any(geo[k] != row[k] for k in ('geometry_success', *CHECKER_REASONS)):
        raise ValueError(f"{system} {room['uid']}: s = 1 does not reproduce the scored checker result")
    given = {'prior': {o['id']: o['size'] for o in h.strict_json(placement['messages'][1]['content'])['objects']},
             'gt': {o['id']: o['size'] for o in room['objects']}}
    swap = {(src, v): score(substituted(base['objects'], given[src], v)) for src in SOURCES for v in VARIANTS}
    tol = {label: _probe(base, t) for label, t in OOB}
    validity = {'complete': True, **{f'checker_oob_{k}': x[0]['geometry_success'] for k, x in tol.items()},
                'checker_tuck_counted': geo['geometry_success'] and not tol['10cm'][1]['collisions']['tuck'],
                'downstream': verdict['accepted'], 'downstream_a4_collision': 'A4_COLLISION' in verdict['reasons']}
    return {**out, **{m: {v: [scores[s, v if s != 1 else 'xy'][m] for s in GRID] for v in VARIANTS} for m in MEASURES},
            'substitution': {m: {v: {src: swap[src, v][m] for src in SOURCES} for v in VARIANTS} for m in MEASURES},
            'bias_fp': r2.size_bias(room['objects'], row['raw_layout'])[0], 'validity': validity,
            'checker_reasons': [k for k in CHECKER_REASONS if geo[k]], 'downstream_reasons': verdict['reasons']}


# ------------------------------------------------------------------ aggregation (numpy over the bootstrap draws)

def design(rooms, repeats=2000, seed=SEED):
    """geometry._paired_design single-arm, and its draws as component multiplicities [repeats x components], drawn as
    geometry._metric_intervals draws (one rng.choice per component slot of each stratum, in order)."""
    t = [{k: x[k] for k in ('uid', 'source', 'group')} for x in rooms]
    components, strata = g._paired_design(t, t)
    index = {a['uid']: k for k, pairs in enumerate(components) for a, _ in pairs}
    rng = random.Random(seed)
    draws = np.array([np.bincount([rng.choice(s) for s in strata for _ in s], minlength=len(components))
                      for _ in range(repeats)], dtype=float)
    return {'index': np.array([index[x['uid']] for x in rooms]), 'draws': draws, 'components': len(components),
            'singletons': sum(len(s) == 1 for s in strata), 'repeats': repeats}


CURVES = len(MEASURES) * len(VARIANTS) * len(GRID)                # _vector: curves, substitutions, validity
SWAPS = [(m, v, src) for m in MEASURES for v in VARIANTS for src in SOURCES]


def _vector(x):
    return [x[m][v][k] for m in MEASURES for v in VARIANTS for k in range(len(GRID))] + \
        [x['substitution'][m][v][src] for m, v, src in SWAPS] + [x['validity'][k] for k in VALIDITY]


def _col(measure, variant):
    start = (MEASURES.index(measure) * len(VARIANTS) + VARIANTS.index(variant)) * len(GRID)
    return slice(start, start + len(GRID))


def rates(d, matrix, mask=None):
    """Room-micro means of a [rooms x columns] matrix over the rooms in mask (default all): (point [columns],
    bootstrap [repeats x columns]); nan where no room counts."""
    mask = np.ones(len(matrix)) if mask is None else mask
    sums, n = np.zeros((d['components'], matrix.shape[1])), np.bincount(d['index'], mask, d['components'])
    np.add.at(sums, d['index'], matrix * mask[:, None])
    with np.errstate(invalid='ignore', divide='ignore'):
        return sums.sum(0) / n.sum(), (d['draws'] @ sums) / (d['draws'] @ n)[:, None]


def rule_shares(reasons):
    """[(checker reasons, downstream reasons)] per room -> {rule: share of rooms failing it} ('incomplete' once)."""
    count = collections.Counter(k for c, dr in reasons for k in {'incomplete' if x == 'incomplete' else f'{kind}:{x}'
                                                                 for kind, xs in (('checker', c), ('downstream', dr))
                                                                 for x in xs})
    return {k: n / len(reasons) for k, n in sorted(count.items())}


def slope(curve):
    """OLS slope of rate on s over the grid points in WINDOW (per unit s); curve [..., len(GRID)]."""
    k = [i for i, s in enumerate(GRID) if WINDOW[0] - 1e-9 <= s <= WINDOW[1] + 1e-9]
    x = np.array([GRID[i] for i in k])
    x = x - x.mean()
    y = np.asarray(curve)[..., k]
    return (y - y.mean(-1, keepdims=True)) @ x / (x @ x)


def s_match(curve, target):
    """The s nearest 1 (ties: the smaller) where the curve meets target, linear between grid points; None if never."""
    d = [float(c) - float(target) for c in curve]
    hits = [s for s, a in zip(GRID, d) if a == 0]
    hits += [s0 + (s1 - s0) * a / (a - b) for s0, s1, a, b in zip(GRID, GRID[1:], d, d[1:]) if a * b < 0]
    return min(hits, key=lambda s: (abs(s - 1), s)) if hits else None


def _iv(estimate, draws, d):
    """Interval over the draws (undefined: None or nan); no interval below report_v2.NO_CI house components."""
    clean = lambda x: None if x is None or not math.isfinite(x) else float(x)
    iv = g._interval(clean(estimate), [clean(x) for x in draws], d['repeats'], d['singletons'])
    if d['components'] < r2.NO_CI:
        iv['ci95'] = None
    return {'estimate': iv.pop('estimate_joint_minus_placement'), **iv}


def summarize(rooms, systems, results, repeats=2000, after=None):
    """{system: curves, slope, s_match, substitution, B, validity, rule shares (raw; after S1 where `after` has the
    system)} on these rooms (GT first: s_match targets are its draws)."""
    d, after = design(rooms, repeats), after or {}
    est = {s: rates(d, np.array([_vector(results[s, x['uid']]) for x in rooms], dtype=float)) for s in systems}
    gt_point, gt_boot = est['GT']
    out = {}
    for system, (point, boot) in est.items():
        entry = {'rooms': len(rooms), 'curves': {}, 'slope': {}, 's_match': {}, 'substitution': {}}
        for m in MEASURES:
            target, targets = gt_point[_col(m, 'xy')][ONE], gt_boot[:, _col(m, 'xy')][:, ONE]
            entry['curves'][m], entry['slope'][m], entry['s_match'][m] = {}, {}, {}
            for v in VARIANTS:
                c, cb = point[_col(m, v)], boot[:, _col(m, v)]
                entry['curves'][m][v] = [{'s': s, **_iv(c[k], cb[:, k], d)} for k, s in enumerate(GRID)]
                entry['slope'][m][v] = _iv(slope(c), slope(cb), d)
                side = None if s_match(c, target) is not None else (
                    'above_gt_everywhere' if all(c > target) else 'below_gt_everywhere')
                entry['s_match'][m][v] = {**_iv(s_match(c, target), [s_match(b, t) for b, t in zip(cb, targets)], d),
                                          'gt_target': float(target), 'unmatched': side}
        for k, (m, v, src) in enumerate(SWAPS):
            own, sub = point[_col(m, 'xy')][ONE], point[CURVES + k]
            owns, subs = boot[:, _col(m, 'xy')][:, ONE], boot[:, CURVES + k]
            entry['substitution'].setdefault(m, {}).setdefault(v, {})[src] = {
                'pass': _iv(sub, subs, d), 'own_minus': _iv(own - sub, owns - subs, d)}
        offset = CURVES + len(SWAPS)
        entry['validity'] = {k: _iv(point[offset + i], boot[:, offset + i], d) for i, k in enumerate(VALIDITY)}
        rows = [results[system, x['uid']] for x in rooms]
        done = np.array([x['complete'] for x in rows], dtype=float)          # B: mean over complete rooms
        b, bb = rates(d, np.array([[x['bias_fp'] or 0.0] for x in rows]), done)
        entry['bias_fp'] = _iv(b[0], bb[:, 0], d)
        entry['rules'] = {'raw': rule_shares([(x['checker_reasons'], x['downstream_reasons']) for x in rows]),
                          **({'s1': rule_shares([after[system][x['uid']] for x in rooms])} if system in after else {})}
        out[system] = entry
    return {'rooms': len(rooms), 'components': d['components'], 'flags': r2._design(rooms)['flags'],
            'singleton_strata': d['singletons'], 'systems': out}


def subsets(rooms):
    """'all', the object-count bins, real / synthetic and the room families (room_families.scopes but 'all':
    family:paper_families, family:other_families, family:NAME) -> rooms (empty subsets left out)."""
    out = {'all': rooms}
    for name, lo, hi in BINS:
        out[name] = [x for x in rooms if lo <= len(x['objects']) and (hi is None or len(x['objects']) <= hi)]
    for k in ('real', 'synthetic'):
        out[k] = [x for x in rooms if kind(x['source']) == k]
    families = rf.scopes([{**x, 'family': rf.family(x.get('room_type'))} for x in rooms])
    out |= {'family:' + k: v for k, v in families.items() if k != 'all'}
    return {k: v for k, v in out.items() if v}


def reasons(rooms, systems, results):
    """{system: {source: rooms, kind, checker / downstream pass at s = 1, failure reason counts}}."""
    out = {}
    for system in systems:
        by = collections.defaultdict(list)
        for x in rooms:
            by[x['source']].append(results[system, x['uid']])
        out[system] = {source: {'kind': kind(source), 'rooms': len(rs),
                                'checker_rate': sum(x['checker']['xy'][ONE] for x in rs) / len(rs),
                                'downstream_rate': sum(x['downstream']['xy'][ONE] for x in rs) / len(rs),
                                'checker_reasons': dict(collections.Counter(
                                    k for x in rs for k in x['checker_reasons'])),
                                'downstream_reasons': dict(collections.Counter(
                                    k for x in rs for k in x['downstream_reasons']))}
                       for source, rs in sorted(by.items())}
    return out


# ------------------------------------------------------------------ inputs

def _pairs(values, what, reserved=('GT', 'J', 'P')):
    out = {}
    for value in values or ():
        name, sep, path = value.partition('=')
        if not sep or not name or not path or name in out or name in reserved:
            raise ValueError(f'{what}: expected unique NAME=PATH (NAME not {"/".join(reserved)}), got {value!r}')
        out[name] = path
    return out


def _extra(spec, rooms):
    """ROWS.jsonl[::KEY=VALUE,...], a file of its directory's verified MANIFEST -> (rows keyed by uid, input record)."""
    path, _, flt = spec.partition('::')
    path = Path(path)
    if path.name not in e._verified(path.parent)['files']:
        raise ValueError(f'{spec}: not listed in {path.parent}/MANIFEST.json')
    keep = dict(kv.split('=', 1) for kv in flt.split(',')) if flt else {}
    rows = [h.strict_json(line) for line in path.read_text().splitlines() if line.strip()]
    rows = h._by_uid([x for x in rows if all(str(x.get(k)) == v for k, v in keep.items())])
    if not rows or any(u not in rooms or (x['source'], x['group']) != (rooms[u]['source'], rooms[u]['group'])
                       for u, x in rows.items()):
        raise ValueError(f'{spec}: no rows, or rows outside the test rooms')
    return rows, {'file': str(path), 'sha256': h.sha256(path), 'filter': keep, 'rows': len(rows),
                  'manifest_sha256': h.sha256(path.parent / 'MANIFEST.json')}


def _agent(root, cohort, data_sha, run_hashes):
    """report_v2._agent (complete, this data, these J/P runs, its cohort), size-free and exactly this cohort ->
    ({'harness'|'raw': rows by uid} with the amendment-006 per-row cap rule (report_v2), manifest sha, flagged)."""
    root = Path(root)
    rows, summary, uids = r2._agent(root, data_sha, run_hashes)
    if summary['llm']['config'].get('request_arm', 'joint') != 'joint':
        raise ValueError(f'{root}: a size-fixed (P input) evaluation; the curves take size-free agents only')
    if set(uids) != set(cohort):
        raise ValueError(f'{root}: agent rows do not cover exactly the cohort')
    flagged = any('reached_token_cap' in x and 'stopped_on_eos' in x for v in rows.values() for x in v)
    capped = lambda x: flagged and x.get('reached_token_cap', False) and not x.get('stopped_on_eos', True)
    return ({v.removeprefix('llm_'): {x['uid']: r2.cap_failure(x) if capped(x) else x for x in vs}
             for v, vs in rows.items()}, h.sha256(root / 'MANIFEST.json'), flagged)


def _s1(root, source, arm, rows, data_sha):
    """output_repair of this system's rows file at the amendment-012 settings -> {uid: (checker reasons, downstream
    reasons) after S1}; a row the system fails (incomplete, cap rule) stays incomplete."""
    repaired, info = t12._system(Path(root), arm, source.name, data_sha)
    if info['inputs']['geometry_sha256'] != h.sha256(source) or {x['uid'] for x in repaired} != set(rows):
        raise ValueError(f'{root}: not S1 of {source} on its rooms')
    return {x['uid']: (['incomplete'], ['incomplete']) if not rows[x['uid']]['complete'] else
            ([k for k in CHECKER_REASONS if x['s1']['geometry_s1'][k]], x['s1']['downstream_s1']['reasons'])
            for x in repaired}


def run(data, geometry, cohort, out, runs=None, agents=None, extras=None, workers=1, repeats=2000, s1=None):
    data, geometry, cohort, out = map(Path, (data, geometry, cohort, out))
    if out.exists():
        raise FileExistsError(out)
    runs, agents, extras, s1 = runs or {}, agents or {}, extras or {}, s1 or {}
    if set(runs) not in (set(), set(h.ARMS)):
        raise ValueError('pass both J and P inference runs, or neither')
    manifest = e._verified(data)
    expected = manifest['expected_uids']
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    requests = {arm: h._by_uid(h._rows(data, arm + '/requests.jsonl', manifest['files'][arm + '/requests.jsonl']))
                for arm in h.ARMS}
    if set(rooms) != set(expected):
        raise ValueError('reference inventory mismatch')
    for x in rooms.values():
        kind(x['source'])
    gm = e._verified(geometry)
    ours = {arm: h._rows(geometry, arm + '-geometry.jsonl', gm['files'][arm + '-geometry.jsonl']) for arm in h.ARMS}
    if any([x['uid'] for x in rows] != expected for rows in ours.values()):
        raise ValueError('geometry rows must cover every test request in inventory order')
    inputs = {'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
              'geometry_manifest_sha256': h.sha256(geometry / 'MANIFEST.json')}
    gsum = h.strict_json((geometry / 'summary.json').read_text())
    if gsum.get('status') != 'GEOMETRY_COMPLETE' or gsum.get('data_manifest_sha256') != inputs[
            'data_manifest_sha256']:
        raise ValueError('geometry is not complete for this evaluation data')
    r2._same_geometry(gsum, 'geometry')
    bound = gsum['inference_run_sha256']
    cap = dict.fromkeys(('J', 'P'), 'not applied: no inference runs given')
    if runs:
        for arm in h.ARMS:
            records, receipt = e._inference(Path(runs[arm]), data, expected)
            if bound[arm] != receipt['run_sha256']:
                raise ValueError(f'{geometry} does not score {runs[arm]}')
            ours[arm], flipped = r2.cap_rule(ours[arm], records)
            cap['J' if arm == 'joint' else 'P'] = f'applied: {flipped} complete rows failed'
            inputs[f'{arm}_run_sha256'] = receipt['run_sha256']
    given = r.load_uids(cohort)
    if not set(given) <= set(expected):
        raise ValueError('cohort uids outside the test rooms')
    inputs['cohort_sha256'] = h.sha256(cohort)
    system_rows = {'J': {x['uid']: x for x in ours['joint']}, 'P': {x['uid']: x for x in ours['placement']}}
    scopes = {'all-test': (expected, ['GT', 'J', 'P']), 'cohort': (given, ['GT', 'J', 'P'])}
    files = {'J': geometry / 'joint-geometry.jsonl', 'P': geometry / 'placement-geometry.jsonl'}
    for name, root in agents.items():
        variants, inputs[f'agent:{name}'], flagged = _agent(root, given, inputs['data_manifest_sha256'], bound)
        for v, rows in variants.items():
            system_rows[f'{name}-{v}'] = rows
            files[f'{name}-{v}'] = Path(root) / f'llm_{v}-geometry.jsonl'
            scopes['cohort'][1].append(f'{name}-{v}')
            cap[f'{name}-{v}'] = 'per_row' if flagged else 'no_flags'
    after = {}
    for name, root in s1.items():
        if name not in files:
            raise ValueError(f'--s1 {name}: not J, P or an agent system')
        after[name] = _s1(root, files[name], 'placement' if name == 'P' else 'joint', system_rows[name],
                          inputs['data_manifest_sha256'])
        inputs[f's1:{name}'] = h.sha256(Path(root) / 'MANIFEST.json')
    for name, spec in extras.items():
        if name in system_rows:
            raise ValueError(f'extra {name}: name already used')
        system_rows[name], inputs[f'extra:{name}'] = _extra(spec, rooms)
        uids = [u for u in expected if u in system_rows[name]]
        scope = next((k for k, (u, _) in scopes.items() if k.startswith('extra-') and u == uids), None)
        scope = scope or f"extra-{sum(k.startswith('extra-') for k in scopes) + 1}"
        scopes.setdefault(scope, (uids, ['GT', 'J', 'P']))[1].append(name)
    tasks = [('GT', None, rooms[u], requests['joint'][u], requests['placement'][u]) for u in expected]
    tasks += [(s, x, rooms[u], requests['placement' if s == 'P' else 'joint'][u], requests['placement'][u])
              for s, rows in system_rows.items() for u, x in rows.items()]
    for system, x, room, *_ in tasks:
        if x is not None and (x['source'], x['group']) != (room['source'], room['group']):
            raise ValueError(f'{system} {room["uid"]}: row / room identity mismatch')
    if workers == 1:
        evaluated = [evaluate_row(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as pool:
            evaluated = list(pool.map(evaluate_row, tasks, chunksize=4))
    results = {(x['system'], x['uid']): x for x in evaluated}
    summary = {}
    for scope, (uids, systems) in scopes.items():
        scope_rooms = [rooms[u] for u in uids]
        summary[scope] = {'rooms': len(uids), 'systems': systems,
                          'subsets': {k: summarize(v, systems, results, repeats, after)
                                      for k, v in subsets(scope_rooms).items()},
                          'reasons': reasons(scope_rooms, systems, results)}
    code = {p.name: h.sha256(p) for p in map(Path, (__file__, g.__file__, dn.__file__, r.__file__, r2.__file__,
                                                    rf.__file__, check.__code__.co_filename))}
    value = {'status': 'SCALE_CURVE_COMPLETE', 'label': LABEL, 'grid': GRID, 'variants': VARIANTS,
             'substitution_sources': SOURCES, 's1_systems': sorted(after),
             'slope_window': WINDOW, 'oob_tolerances_m': dict(OOB), 'source_kind': SOURCE_KIND, 'bins': BINS,
             'bootstrap': {'repeats': repeats, 'seed': SEED,
                           'design': 'geometry._paired_design single-arm, room micro'},
             'cap_rule': cap, 'inputs': inputs, 'scopes': summary,
             'code_sha256': code}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        value['plot'] = _plot(dest / 'curves.png', summary)
        (dest / 'rows.jsonl').write_text(''.join(json.dumps(x, allow_nan=False) + '\n' for x in evaluated))
        h.write_json(dest / 'results.json', value)
        _tables(dest, summary)
        (dest / 'report.md').write_text(_report(value))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest), 'code_sha256': code})
        _publish(dest, out)
    return value


# ------------------------------------------------------------------ outputs

def _ci(iv, k):
    return None if iv['ci95'] is None else iv['ci95'][k]


def _tables(dest, summary):
    curves, slopes, validity, causes, rules = [], [], [], [], []
    for scope, s in summary.items():
        for subset, sub in s['subsets'].items():
            for system, x in sub['systems'].items():
                key = {'scope': scope, 'subset': subset, 'system': system, 'rooms': x['rooms'],
                       'components': sub['components'], 'flags': sub['flags']}
                for m in MEASURES:
                    for v in VARIANTS:
                        curves += [{**key, 'measure': m, 'variant': v, 's': p['s'], 'rate': p['estimate'],
                                    'ci_low': _ci(p, 0), 'ci_high': _ci(p, 1)} for p in x['curves'][m][v]]
                        a, b = x['slope'][m][v], x['s_match'][m][v]
                        swap = {}
                        for src in SOURCES:
                            for k in ('pass', 'own_minus'):
                                iv = x['substitution'][m][v][src][k]
                                swap |= {f'{k}_{src}': iv['estimate'], f'{k}_{src}_ci_low': _ci(iv, 0),
                                         f'{k}_{src}_ci_high': _ci(iv, 1)}
                        slopes.append({**key, 'measure': m, 'variant': v, 'rate_s1': x['curves'][m][v][ONE]['estimate'],
                                       'slope': a['estimate'], 'slope_ci_low': _ci(a, 0), 'slope_ci_high': _ci(a, 1),
                                       's_match': b['estimate'], 's_match_ci_low': _ci(b, 0),
                                       's_match_ci_high': _ci(b, 1), 's_match_valid_draws': b['valid_bootstrap_draws'],
                                       'unmatched': b['unmatched'], 'gt_target': b['gt_target'], **swap,
                                       'bias_fp': x['bias_fp']['estimate'], 'bias_fp_ci_low': _ci(x['bias_fp'], 0),
                                       'bias_fp_ci_high': _ci(x['bias_fp'], 1)})
                validity += [{**key, 'metric': k, 'rate': iv['estimate'], 'ci_low': _ci(iv, 0), 'ci_high': _ci(iv, 1)}
                             for k, iv in x['validity'].items()]
                rules += [{**key, 'stage': stage, 'rule': k, 'share': share}
                          for stage, shares in x['rules'].items() for k, share in shares.items()]
        for system, sources in s['reasons'].items():
            causes += [{'scope': scope, 'system': system, 'source': source, **{k: v for k, v in x.items()
                        if not k.endswith('_reasons')}, **{k: ';'.join(f'{a}:{n}' for a, n in sorted(x[k].items()))
                                                            for k in ('checker_reasons', 'downstream_reasons')}}
                       for source, x in sources.items()]
    for name, rows in (('curves.csv', curves), ('summary.csv', slopes), ('validity.csv', validity),
                       ('reasons.csv', causes), ('rules.csv', rules)):
        with (dest / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _plot(path, summary):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'matplotlib missing: curves.png not written'
    panels = [(m, v) for m in MEASURES for v in VARIANTS]
    fig, axes = plt.subplots(len(summary), len(panels), figsize=(5 * len(panels), 4 * len(summary)), squeeze=False)
    for row, (scope, s) in zip(axes, summary.items()):
        for ax, (m, v) in zip(row, panels):
            for k, (system, x) in enumerate(s['subsets']['all']['systems'].items()):
                points = x['curves'][m][v]
                y = [p['estimate'] for p in points]
                line, = ax.plot(GRID, y, marker='.', lw=2.4 if system == 'GT' else 1.2, label=system,
                                color='black' if system == 'GT' else plt.cm.tab20(k % 20))
                ax.fill_between(GRID, [_ci(p, 0) if p['ci95'] else p['estimate'] for p in points],
                                [_ci(p, 1) if p['ci95'] else p['estimate'] for p in points],
                                color=line.get_color(), alpha=.15, lw=0)
            ax.axvline(1.0, color='grey', ls=':', lw=1)
            ax.set(title=f'{scope}: {m} ({v})', xlabel='size scale s', ylabel='pass rate', ylim=(0, 1))
        row[-1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, .5))
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return 'curves.png'


def _f(value, sign=False):
    return '—' if value is None else (f'{value:+.3f}' if sign else f'{value:.3f}')


def _with_ci(iv, sign=False):
    return _f(iv['estimate'], sign) + ('' if iv['ci95'] is None else
                                       f" [{_f(iv['ci95'][0], sign)}, {_f(iv['ci95'][1], sign)}]")


def _match(iv):
    if iv['estimate'] is None:
        return {'above_gt_everywhere': '全程高于 GT', 'below_gt_everywhere': '全程低于 GT'}[iv['unmatched']]
    return _with_ci(iv)


def _report(value):
    names = {'checker': '检查器成功', 'downstream': '下游接受'}
    lines = [f"# 尺寸缩放–通过率曲线、尺寸替换与检查器有效性（{LABEL}）", '',
             '干预只改尺寸：每件物体的 x/y（xy 变体）或 x/y/z（xyz 变体）尺寸乘以 s，围绕自身底面中心缩放；朝向和支撑父物体不变，'
             '无支撑物体的位置不变。与“位置不变”的偏离（2026-10-14 复核）：被支撑的子物体保持在父物体上的相对位置——相对父物体中心'
             '的偏移随父物体一起缩放（否则缩小桌子会把桌边的杯子挤出 5 cm 支撑容差，曲线会把这种人为的支撑失败算进去），并随父物体'
             '顶面升降（xyz 变体，保持原有支撑间隙，附录 B §3）；固定件父物体不动，s = 1 即原布局。每个布局在其请求上下文中用冻结'
             '检查器（geometry.geometry）和下游接受移植（downstream.accept）重新评分；s = 1 必须复现原行的 geometry_success 与各项'
             '检查器失败计数。未完成（含 amendment-005 截断规则判失败）的行在每个 s 都计为失败，分母为该范围全部请求。', '',
             f"s 网格：{', '.join(f'{s:.2f}' for s in GRID)}。斜率 = s 在 {WINDOW[0]:.2f}–{WINDOW[1]:.2f} 上的最小二乘斜率"
             '（每单位 s 的通过率变化）。陡峭只说明通过率对尺寸敏感，必须对照同图的 GT 曲线解读，不能单凭陡峭读成“靠缩尺寸取巧”'
             '（附录 A §3）。s_match = 曲线与同一子集 GT 在 s = 1 的通过率相交的 s（网格点间线性插值，取最接近 1 者）。', '',
             '尺寸替换（附录 A §3，§1 的主要证据）：同一布局（位置、朝向、支撑不变，支撑规则同上）把每件物体的 x/y（xy）或 x/y/z'
             '（xyz）尺寸换成训练集先验（P 请求中的尺寸）或 GT 尺寸，报告 通过率(自身尺寸) − 通过率(先验尺寸) 与 通过率(自身) − '
             '通过率(GT 尺寸)，配对（同房间、同一批抽样），旁列足迹偏差 B（report_v2.size_bias，系统自身完整房间的平均 '
             'log(预测/GT x·y)）。', '',
             '区间：房屋连通分量自举（geometry.paired_intervals 的设计的单臂版：按房屋分组、按来源签名分层，'
             f"{value['bootstrap']['repeats']} 次，种子 {SEED}，房间微平均），所有量用同一批抽样（GT 配对）。房屋连通组 k < "
             f'{r2.NO_CI} 时不给区间；k < 30 标 few clusters，房间数 < 50 标 small n（report_v2._design）。仅作描述，无决策规则。', '']
    for scope, s in value['scopes'].items():
        everything = s['subsets']['all']
        lines += [f"## 范围 {scope}（{s['rooms']} 个房间，k = {everything['components']}）", '']
        for v in VARIANTS:
            lines += [f'### 曲线摘要，{v} 变体，全部房间', '',
                      '| 系统 | ' + ' | '.join(f'{names[m]} s={s_:.2f}' for m in MEASURES for s_ in (.8, 1.0, 1.1))
                      + ' | ' + ' | '.join(f'{names[m]} 斜率 | {names[m]} s_match' for m in MEASURES) + ' |',
                      '|---|' + '---:|' * 10]
            for system, x in everything['systems'].items():
                cells = [_with_ci(x['curves'][m][v][GRID.index(s_)]) for m in MEASURES for s_ in (.8, 1.0, 1.1)]
                cells += [c for m in MEASURES for c in (_with_ci(x['slope'][m][v], True), _match(x['s_match'][m][v]))]
                lines.append(f'| {system} | ' + ' | '.join(cells) + ' |')
            lines.append('')
        for m in MEASURES:
            lines += [f'### 尺寸替换：{names[m]}，全部房间', '',
                      '| 系统 | 足迹偏差 B | 自身尺寸 s=1 | 自身−先验（xy） | 自身−先验（xyz） | 自身−GT（xy） | 自身−GT（xyz） |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for system, x in everything['systems'].items():
                lines.append(f"| {system} | {_with_ci(x['bias_fp'], True)} | {_with_ci(x['curves'][m]['xy'][ONE])} | "
                             + ' | '.join(_with_ci(x['substitution'][m][v][src]['own_minus'], True)
                                          for src in SOURCES for v in VARIANTS) + ' |')
            lines.append('')
        lines += ['### 分层（xy 变体）：物体数分箱、真实 / 合成来源、房型大类（修订 015）', '',
                  '| 子集 | 房间 | k | 标记 | 系统 | B | 检查器 s=1 | 检查器斜率 | 下游 s=1 | 下游斜率 | 下游 自身−先验 | '
                  '下游 自身−GT |', '|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for subset, sub in s['subsets'].items():
            if subset == 'all':
                continue
            for system, x in sub['systems'].items():
                swap = x['substitution']['downstream']['xy']
                lines.append(f"| {subset} | {sub['rooms']} | {sub['components']} | {sub['flags'] or ''} | {system} | "
                             + ' | '.join([_with_ci(x['bias_fp'], True)] + [
                                 c for m in MEASURES for c in (_with_ci(x['curves'][m]['xy'][ONE]),
                                                               _with_ci(x['slope'][m]['xy'], True))]
                                 + [_with_ci(swap[src]['own_minus'], True) for src in SOURCES]) + ' |')
        lines += ['', '分层的 s_match、xyz 变体、检查器的替换效应与区间见 summary.csv。', '',
                  '### 检查器有效性（s = 1，全部房间）', '',
                  '| 系统 | 完整率 | 越界容差 1 mm | 1 cm | 10 cm（冻结） | 塞入座椅计为碰撞 | 下游接受 | 下游 A4 碰撞占比 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for system, x in everything['systems'].items():
            lines.append(f'| {system} | ' + ' | '.join(_with_ci(x['validity'][k]) for k in VALIDITY) + ' |')
        rules = sorted({k for x in everything['systems'].values() for st in x['rules'].values() for k in st},
                       key=lambda k: (k != 'incomplete', not k.startswith('checker:'), k))
        lines += ['', '### 逐规则失败占比（附录 A §4，s = 1，全部房间；+S1 = 修订 012 S1 修复后）', '',
                  '| 系统 | ' + ' | '.join(rules) + ' |', '|---|' + '---:|' * len(rules)]
        for system, x in everything['systems'].items():
            for stage, shares in x['rules'].items():
                lines.append(f"| {system}{' +S1' if stage == 's1' else ''} | "
                             + ' | '.join(_f(shares.get(k, 0.0)) for k in rules) + ' |')
        lines += ['', '### GT 按来源的通过率与失败原因（s = 1）', '',
                  '| 来源 | 类别 | 房间 | 检查器成功 | 下游接受 | 检查器失败原因（房间数） | 下游原因（房间数） |',
                  '|---|---|---:|---:|---:|---|---|']
        for source, x in s['reasons']['GT'].items():
            fmt = lambda c: '；'.join(f'{a} {n}' for a, n in sorted(c.items(), key=lambda t: -t[1])) or '—'
            lines.append(f"| {source} | {x['kind']} | {x['rooms']} | {_f(x['checker_rate'])} | "
                         f"{_f(x['downstream_rate'])} | {fmt(x['checker_reasons'])} | {fmt(x['downstream_reasons'])} |")
        lines.append('')
    extras = {k[6:]: x for k, x in value['inputs'].items() if k.startswith('extra:')}
    if extras:
        lines += ['## extras（只含下列筛选出的行；同一文件中未被筛选的其他臂不计入）', '', '| 名称 | 文件 | 筛选 | 房间 |',
                  '|---|---|---|---:|']
        lines += [f"| {k} | {x['file']} | {x['filter']} | {x['rows']} |" for k, x in extras.items()] + ['']
    lines += ['## 来源划分（amendment 016 §5）', '', '| 来源 | 类别 | 依据 |', '|---|---|---|']
    lines += [f'| {k} | {a} | {b} |' for k, (a, b) in SOURCE_KIND.items()]
    lines += ['', '“真实”按修正案列表 = 真实扫描或设计师绘制的布局；其中 SpatialLM、HSSD200、InteriorGS、Structured3D 不是扫描。'
              'IL3D_3dfront（3D-FRONT，设计师绘制）不在修正案的真实列表里，按“其余为合成”归为合成，与 Structured3D 的归类不一致，'
              '解读真实 / 合成分层时需注意。', '',
              '塞入座椅：冻结检查器把座椅–桌子、床–床头柜的重叠归为 tuck 不算碰撞；下游 A4 把同一父物体下 AABB 重叠超过 1 cm 的'
              '都算碰撞（包括塞入的椅子）。下游没有豁免 tuck 的开关，故只报告 A4 碰撞占比，不另算豁免后的下游接受。', '',
              f"截断规则：{value['cap_rule']}。S1 系统：{', '.join(value['s1_systems']) or '无'}。图：{value['plot']}（GT 为黑色粗线，"
              '作为参照）。逐房间结果见 rows.jsonl，表格见 curves.csv、summary.csv、validity.csv、reasons.csv、rules.csv。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('data', 'geometry', 'cohort', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--joint-run', type=Path)
    parser.add_argument('--placement-run', type=Path)
    parser.add_argument('--agent', action='append', help='NAME=RUN/llm-eval-NAME-selected-v1')
    parser.add_argument('--extra', action='append', help='NAME=ROWS.jsonl[::KEY=VALUE,...]')
    parser.add_argument('--s1', action='append', help='SYSTEM=RUN/s1-012-test/DIR (SYSTEM: J, P or NAME-harness/raw)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2000)
    args = parser.parse_args()
    runs = {arm: getattr(args, arm + '_run') for arm in h.ARMS if getattr(args, arm + '_run') is not None}
    value = run(args.data, args.geometry, args.cohort, args.out, runs, _pairs(args.agent, '--agent'),
                _pairs(args.extra, '--extra'), args.workers, args.repeats, _pairs(args.s1, '--s1', ('GT',)))
    print(json.dumps({'status': value['status'], 'scopes': {k: v['rooms'] for k, v in value['scopes'].items()},
                      'plot': value['plot']}))


if __name__ == '__main__':
    main()
