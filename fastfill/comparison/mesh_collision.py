"""Amendment 019: mesh-level collisions on the rendered rooms (EXPLORATORY, descriptive; no decision rule).

python -m fastfill.comparison.mesh_collision --run VISUAL_RUN --out NEW_DIR [--workers 4] [--limit N] [--repeats 2000]

Spec: docs/research/20261013-unknown-size-comparison/amendment-019-mesh-collisions-20261014.md (all numbers there).
VISUAL_RUN is the amendment-011 run (visual-final-selected-002), read only: visual/chunks/*/input.json are the cases the
renderer consumed (every system's layout), visual/cases.jsonl the frozen cohort (cohort rooms absent from the chunks are
missing rooms), visual/J-geometry.jsonl the house groups, visual/attachment-ledger.jsonl why a system layout is absent.
--limit N analyses an evenly spaced subset of N rooms (timing runs).

Placement: render_assets.plan_case itself (validation, asset mapping, geometry contract), only its OBJ inspection is
swapped for a parser that keeps the faces (the visual_cases._planner injection). Every mesh, mapped back into its own
object frame, must span its layout box to 1e-6 m or the room fails. The voxeliser uses the algebraically identical
split transform = rotation(yaw) . (render transform at yaw 0, pos 0) + pos, with pos = whole voxels + a sub-voxel
offset, so one voxel solid serves every placement of the same (asset, size, native scale, yaw, sub-voxel offset).
Metric: candidates = distinct movable objects whose placed-mesh world AABBs intersect (closed boxes), minus parent-child
pairs (the layout's own parent links); fixed context objects are not compared (as in validate.check's pairs). Each
mesh's triangles are sampled <= 0.5 cm apart, samples are rounded onto the shared world grid of 2 cm voxels (centres at
multiples of 2 cm) and scipy.ndimage.binary_fill_holes fills each object; closed = the fill added a voxel, else the mesh
stays a shell (a voxel-level, size-dependent verdict, not mesh watertightness: the assets are rarely watertight).
Interpenetration = shared voxels x 8 cm3; a pair collides at tau when it exceeds tau (125/500/2000 cm3, 500 primary); a
room is collision-free at tau when no pair collides.
Joined per pair: the frozen checker (validate.check on the same layout and room context: severe > main > tuck >
contained, else below_threshold and its first skip rule), the A4 box verdict (downstream.accept on the case's request
context) and, for every A4 pair, the amendment-017 evidence class (gt-admission-anatomy-20261014.md order: tucked, box
artefact, small overlap, contained, genuine). Diagnostic outside the amendment: deep_cm3 = shared voxels that are not
surface voxels of both objects (two flush faces share one 2 cm layer, ~2 cm x contact area).
Outputs (write-once directory, MANIFEST.json last): rows.jsonl (room x system), pairs.jsonl, summary.json, report.md
(Chinese). CIs: house-cluster bootstrap with report_v2.contrasts (geometry.paired_intervals' design and draws).
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import FunctionType

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from fastfill.anchors import FLAT
from fastfill.comparison import downstream as dn
from fastfill.comparison import render_assets as render
from fastfill.comparison import report_v2
from fastfill.comparison import visual_cases
from fastfill.comparison.corpus import _publish
from fastfill.interface import snap_inside
from fastfill.scene import footprint, head
from fastfill.validate import check, collision_kind

PITCH = 0.02                       # m, amendment 019
VOXEL_CM3 = (PITCH * 100) ** 3     # 8 cm3
# Triangle sample spacing. The assets are open meshes: the voxel shell seals their narrow holes only when the voxels a
# hole rim merely grazes get a sample. At PITCH / 2 the fill leaked through such rims (Holodeck c6d23e49: 16,224 solid
# voxels vs 28,366 at PITCH / 4 and 28,558 at PITCH / 16; sampling just its hole rims densely gives 28,182).
SAMPLE_M = PITCH / 4
TAUS = (125, 500, 2000)            # cm3
PRIMARY = 500
FIELDS = [f'mesh_free_{t}' for t in TAUS]
BBOX_TOL_M = 1e-6
ROT_TOL = math.radians(0.5)        # gt-admission-anatomy: rotated = yaw off the A4 frame by > 0.5 degrees
SYSTEMS = ('GT', 'J', 'P', 'L', 'O1a', 'O1b', 'O2', 'O3', 'O4', 'O6', 'O7')
AMENDMENT_SYSTEMS = ('GT', 'J', 'P', 'L', 'O2', 'O3', 'O4', 'O6', 'O7')
CLASSES = ('tucked', 'box_artefact', 'small_overlap', 'contained', 'genuine')
CLASS_ZH = {'tucked': '塞入（tucked）', 'box_artefact': '旋转盒伪影（box artefact）',
            'small_overlap': '小重叠（small overlap）', 'contained': '包含（contained）', 'genuine': '真实（genuine）'}
SPEC = 'docs/research/20261013-unknown-size-comparison/amendment-019-mesh-collisions-20261014.md'
ROOT = Path(__file__).resolve().parents[2]
CODE = ('fastfill/comparison/mesh_collision.py', 'fastfill/comparison/render_assets.py', 'fastfill/validate.py',
        'fastfill/comparison/downstream.py', 'fastfill/interface.py', 'fastfill/comparison/report_v2.py',
        'fastfill/comparison/geometry.py')
_MESHES = {}                       # per room (cleared by analyse_case): (mesh path, asset id) -> parsed mesh


def load_obj(path, asset_id):
    """render_assets.inspect_asset's vertex bounds (same tokeniser) plus the fan-triangulated faces and the sha256."""
    content = Path(path).read_bytes()
    vertices, faces = [], []
    for line in content.decode('utf-8').splitlines():
        key, tail = render.statement(line)
        if key == 'v':
            vertices.append([float(v) for v in tail.split()[:3]])
        elif key == 'f':
            idx = [int(t.split('/')[0]) for t in tail.split()]
            idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
            faces.extend((idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1))
    vertices = np.array(vertices, dtype=float).reshape(-1, 3)
    faces = np.array(faces, dtype=np.int64).reshape(-1, 3)
    if not len(vertices) or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise render.RenderReject(f'invalid_mesh: {path}')
    return {'mesh_path': str(path), 'asset_id': asset_id, 'key': (str(path), asset_id),
            'bounds': (tuple(vertices.min(0).tolist()), tuple(vertices.max(0).tolist())),
            'vertices': vertices, 'faces': faces, 'sha256': hashlib.sha256(content).hexdigest()}


def _inspect(path, asset_id):
    if (path, asset_id) not in _MESHES:
        _MESHES[path, asset_id] = load_obj(path, asset_id)
    return _MESHES[path, asset_id]


def _planner():
    """render_assets.plan_case with inspect_asset -> _inspect (no copied planner code, no global monkeypatch)."""
    mapped, plan = render.mapped_object, render.plan_case
    mapped = FunctionType(mapped.__code__, {**mapped.__globals__, 'inspect_asset': _inspect})
    return FunctionType(plan.__code__, {**plan.__globals__, 'mapped_object': mapped}, plan.__name__,
                        plan.__defaults__, plan.__closure__)


PLAN = _planner()


def surface_samples(vertices, faces, spacing=SAMPLE_M):
    """Points on every triangle, <= spacing apart along its shortest edge and the next edge from the same vertex; grid
    points beyond the third edge are pulled onto it along the ray from that vertex. Edge lengths are rounded (1e-9)
    before choosing the vertex and the counts, so float noise in the placement does not change the sample grid."""
    t = vertices[faces]
    lengths = np.round(np.linalg.norm(t[:, [1, 2, 0]] - t, axis=2) / spacing, 9)   # edge k: vertex k -> k + 1
    s = lengths.argmin(1)
    t = np.take_along_axis(t, ((s[:, None] + np.arange(3)) % 3)[:, :, None], 1)
    a, e1, e2 = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    pick = lambda shift: np.take_along_axis(lengths, ((s + shift) % 3)[:, None], 1)[:, 0]
    nu, nv = (np.maximum(1, np.ceil(pick(shift))).astype(np.int64) for shift in (0, 2))
    counts = (nu + 1) * (nv + 1)
    tri = np.repeat(np.arange(len(t)), counts)
    k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    u, v = (k % (nu + 1)[tri]) / nu[tri], (k // (nu + 1)[tri]) / nv[tri]
    over = np.maximum(u + v, 1.0)
    return a[tri] + (u / over)[:, None] * e1[tri] + (v / over)[:, None] * e2[tri]


def voxelise(vertices, faces, spacing, chunk=1024):
    """Mesh in voxel units -> (lowest index, shell, filled solid); the shell is every voxel holding a surface sample.
    Triangles go in chunks so memory stays bounded (one wardrobe takes 2e7 samples at PITCH / 4). Two voxels of padding
    beyond the rounded vertex bounds: an empty border for the fill, and room for a sample that float rounding puts
    past a vertex lying on a voxel face."""
    corners = vertices[faces].reshape(-1, 3)
    lo = np.floor(corners.min(0) + 0.5).astype(np.int64) - 2
    shell = np.zeros(np.floor(corners.max(0) + 0.5).astype(np.int64) - lo + 3, bool)
    for k in range(0, len(faces), chunk):
        idx = np.floor(surface_samples(vertices, faces[k:k + chunk], spacing) + 0.5).astype(np.int64)
        shell[tuple((idx - lo).T)] = True
    return lo, shell, ndimage.binary_fill_holes(shell)


def _affine(points, linear, offset=(0, 0, 0)):
    """points @ linear.T + offset as column products ((n, 3) @ (3, 3) through BLAS is slow on a loaded CPU)."""
    linear = np.asarray(linear, dtype=float)
    return points[:, :1] * linear[:, 0] + points[:, 1:2] * linear[:, 1] + points[:, 2:] * linear[:, 2] + offset


def place(obj, solids):
    """One object as render_assets placed it: world AABB, voxel solid on the shared grid, bbox error."""
    res, tf = obj['resource'], obj['transform']
    world = _affine(res['vertices'], tf.linear, np.array(tf.offset))
    c, s = math.cos(obj['yaw']), math.sin(obj['yaw'])
    rot = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    pos, size = np.array(obj['pos'], dtype=float), np.array(obj['size'], dtype=float)
    local = _affine(world - pos, rot.T)
    half = np.array([size[0] / 2, size[1] / 2, 0])
    err = max(np.abs(local.min(0) + half).max(), np.abs(local.max(0) - half - [0, 0, size[2]]).max())
    d = np.floor(pos / PITCH + 0.5)
    frac = np.round(pos - d * PITCH, 9)            # sub-voxel offset, snapped to 1 nm so the solid cache is exact
    key = (res['key'], obj['source'], tuple(obj['size']), tuple(obj['native_scale']), obj['yaw'], tuple(frac))
    if key not in solids:
        t0 = render.build_transform(res['bounds'], obj['source'], obj['size'], (0, 0, 0), 0, obj['native_scale'])
        grid = _affine(_affine(res['vertices'], t0.linear, np.array(t0.offset)), rot, frac) / PITCH
        solids[key] = voxelise(grid, res['faces'], SAMPLE_M / PITCH)
    lo, shell, solid = solids[key]
    return {'aabb': (world.min(0), world.max(0)), 'lo': lo + d.astype(np.int64), 'shell': shell, 'solid': solid,
            'closed': bool(solid.sum() > shell.sum()), 'bbox_err': float(err)}


def overlap(a, b):
    """(shared solid voxels, voxels in both shells) of two placed objects."""
    lo = np.maximum(a['lo'], b['lo'])
    hi = np.minimum(a['lo'] + a['solid'].shape, b['lo'] + b['solid'].shape)
    if (hi <= lo).any():
        return 0, 0
    sa, sb = (tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, x['lo'])) for x in (a, b))
    return int((a['solid'][sa] & b['solid'][sb]).sum()), int((a['shell'][sa] & b['shell'][sb]).sum())


def checker_skip(a, b, tol=0.05):
    """The first validate.check guard (validate.py:84-92, default tol/min_area) that drops an unbucketed pair."""
    if a.get('parent') == b['id'] or b.get('parent') == a['id']:
        return 'parent_child'
    if min(a['pos'][2] + a['size'][2], b['pos'][2] + b['size'][2]) - max(a['pos'][2], b['pos'][2]) <= tol:
        return 'z_overlap_le_5cm'
    if FLAT.intersection((head(a), head(b))) or min(a['size'][2], b['size'][2]) < 0.06:
        return 'flat_or_thin'
    return 'area_le_100cm2'


def snapped(room, layout):
    """downstream.accept's R2 snap (downstream.py:239-249), for the box-artefact class."""
    floor = dn.keepout_floor(orient(Polygon(room['boundary'])), [s['strip'] for s in dn.door_strips(room)])[1]
    pl = {o['id']: {'pos': list(o['pos']), 'yaw': o['yaw'], 'on': o.get('parent')} for o in layout}
    if not floor.is_empty:
        snap_inside(pl, {'objects': layout, 'fixed': []}, floor, (), room['boundary'])
    return {o['id']: {**o, 'pos': pl[o['id']]['pos']} for o in layout}


def a4_class(a, b, sa, sb, theta, checker):
    """Amendment-017 evidence class of an A4 pair (gt-admission-anatomy ladder: first that fits)."""
    if collision_kind(a, b, Polygon(footprint(a)), Polygon(footprint(b))) == 'tuck':
        return 'tucked'
    rotated = any(abs((o['yaw'] - theta + math.pi / 4) % (math.pi / 2) - math.pi / 4) > ROT_TOL for o in (sa, sb))
    if rotated and Polygon(footprint(sa)).intersection(Polygon(footprint(sb))).area <= 1e-9:
        return 'box_artefact'
    return {'main': 'genuine', 'severe': 'genuine', 'contained': 'contained'}.get(checker, 'small_overlap')


def analyse_layout(room, layout, planned, solids):
    """(room-level fields, pair records) of one system's layout."""
    objs = [place(o, solids) for o in planned]
    worst = max(o['bbox_err'] for o in objs)
    if worst > BBOX_TOL_M:
        raise ValueError(f'placed mesh does not span its layout box: {worst:.3g} m')
    verdict = {}
    found = check({**room, 'objects': layout})['collisions']
    for kind in ('contained', 'tuck', 'main', 'severe'):                    # later wins: severe over main
        verdict.update({frozenset(p): kind for p in found[kind]})
    accepted = dn.accept(room, layout)
    a4 = {frozenset(x['ids']) for x in accepted['violations'] if x['code'] == 'A4_COLLISION'}
    snap = snapped(room, layout) if a4 else {}
    pairs = []
    for (i, a), (j, b) in combinations(enumerate(layout), 2):
        key = frozenset((a['id'], b['id']))
        parent_child = a.get('parent') == b['id'] or b.get('parent') == a['id']
        (alo, ahi), (blo, bhi) = objs[i]['aabb'], objs[j]['aabb']
        candidate = not parent_child and bool((alo <= bhi).all() and (blo <= ahi).all())
        if not (candidate or key in verdict or key in a4):
            continue
        shared, contact = overlap(objs[i], objs[j]) if candidate else (0, 0)
        kind = verdict.get(key, 'below_threshold')
        pairs.append({'ids': [a['id'], b['id']], 'categories': [a.get('category'), b.get('category')],
                      'candidate': candidate, 'parent_child': parent_child,
                      'volume_cm3': shared * VOXEL_CM3, 'deep_cm3': (shared - contact) * VOXEL_CM3,
                      'closed': [objs[i]['closed'], objs[j]['closed']],
                      **{f'collides_{t}': shared * VOXEL_CM3 > t for t in TAUS},
                      'checker': kind, 'checker_skip': checker_skip(a, b) if kind == 'below_threshold' else None,
                      'a4': key in a4, 'a4_class': a4_class(a, b, snap[a['id']], snap[b['id']], accepted['theta'], kind)
                      if key in a4 else None})
    volumes = [p['volume_cm3'] for p in pairs if p['candidate']]
    row = {'objects': len(layout), 'candidates': len(volumes), 'open_meshes': sum(not o['closed'] for o in objs),
           'max_volume_cm3': max(volumes, default=0.0), 'bbox_max_err_m': worst,
           **{f'colliding_{t}': sum(v > t for v in volumes) for t in TAUS},
           **{f'mesh_free_{t}': not any(v > t for v in volumes) for t in TAUS},
           f'mesh_free_deep_{PRIMARY}': not any(p['deep_cm3'] > PRIMARY for p in pairs),
           'checker_collision_free': not (found['main'] or found['severe']), 'a4_free': not a4}
    return row, pairs


def analyse_case(case):
    """Every system's layout of one rendered case -> {uid, rows, pairs, error, seconds, meshes, cache}."""
    start = time.perf_counter()
    _MESHES.clear()
    solids, rows, pairs = {}, [], []
    try:
        plan = PLAN(case)
        room = {**{k: case.get(k) for k in ('boundary', 'boundary_type', 'height', 'fixed')},
                'objects': case['layouts']['reference']}
        for label, planned in plan['layouts'].items():
            system = 'GT' if label == 'reference' else label
            row, found = analyse_layout(room, case['layouts'][label], planned, solids)
            rows.append({'uid': case['uid'], 'source': case['source'], 'stratum': case.get('stratum'),
                         'system': system, **row})
            pairs += [{'uid': case['uid'], 'system': system, **p} for p in found]
        error = None
    except Exception as exc:  # Room isolation: a failed room is reported missing, never filled in.
        rows, pairs, error = [], [], f'{type(exc).__name__}: {exc}'
    placements = sum(len(v) for v in case.get('layouts', {}).values())
    return {'uid': case['uid'], 'rows': rows, 'pairs': pairs, 'error': error,
            'seconds': time.perf_counter() - start,
            'meshes': {m['asset_id']: m['sha256'] for m in _MESHES.values()},
            'cache': {'placements': placements, 'voxelised': len(solids)}}


# ---------------------------------------------------------------------------------------------- aggregation

def _rate(k, n):
    return {'k': k, 'n': n, 'rate': k / n if n else None}


def summarize(results, groups, cohort, ledger, repeats, seed=report_v2.SEED):
    rows = [{**r, 'group': groups[r['uid']]} for x in results for r in x['rows']]
    pairs = [p for x in results for p in x['pairs']]
    present = [s for s in SYSTEMS if any(r['system'] == s for r in rows)]
    by = {s: sorted((r for r in rows if r['system'] == s), key=lambda r: r['uid']) for s in present}
    fields = FIELDS
    analysed = {x['uid'] for x in results if x['error'] is None}
    systems = {}
    for s in present:
        mine = by[s]
        levels = report_v2.contrasts(mine, [{**r, **dict.fromkeys(fields + ['checker_collision_free'], 0.0)}
                                            for r in mine], fields + ['checker_collision_free'], repeats, seed)
        sp = [p for p in pairs if p['system'] == s]
        per_tau = {}
        for t in TAUS:
            hit = [p for p in sp if p[f'collides_{t}']]
            ms = [p for p in sp if p['checker'] in ('main', 'severe')]
            a4 = [p for p in sp if p['a4']]
            per_tau[t] = {'checker_main_severe_confirmed': _rate(sum(p[f'collides_{t}'] for p in ms), len(ms)),
                          'a4_confirmed': _rate(sum(p[f'collides_{t}'] for p in a4), len(a4)),
                          'mesh_colliding_pairs': len(hit),
                          'checker_missed': _rate(sum(p['checker'] not in ('main', 'severe') for p in hit), len(hit)),
                          'missed_by_checker_verdict': dict(Counter(p['checker_skip'] or p['checker'] for p in hit
                                                                    if p['checker'] not in ('main', 'severe'))),
                          'contact_dominated': _rate(sum(p['deep_cm3'] <= t for p in hit), len(hit))}
        have = {r['uid'] for r in mine}
        missing = [u for u in analysed if u not in have]
        systems[s] = {'rooms': len(mine), 'levels': levels, 'per_tau': per_tau,
                      'a4_free': _rate(sum(r['a4_free'] for r in mine), len(mine)),
                      f'mesh_free_deep_{PRIMARY}': _rate(sum(r[f'mesh_free_deep_{PRIMARY}'] for r in mine), len(mine)),
                      'placements': sum(r['objects'] for r in mine), 'open_meshes': sum(r['open_meshes'] for r in mine),
                      'missing_rooms': len(missing), 'missing_reasons': dict(Counter(
                          str((ledger.get(u, {}).get('errors') or {}).get(s, 'not in case'))[:80] for u in missing))}
    paired = {}
    if 'P' in by:
        p_rows = {r['uid']: r for r in by['P']}
        for s in present:
            if s == 'P':
                continue
            common = [r for r in by[s] if r['uid'] in p_rows]
            paired[s] = report_v2.contrasts(common, [p_rows[r['uid']] for r in common], fields, repeats, seed)
    gt = [p for p in pairs if p['system'] == 'GT' and p['a4']]
    classes = {c: {'pairs': sum(p['a4_class'] == c for p in gt),
                   **{f'collides_{t}': sum(p['a4_class'] == c and p[f'collides_{t}'] for p in gt) for t in TAUS},
                   'median_volume_cm3': statistics.median([p['volume_cm3'] for p in gt if p['a4_class'] == c] or [0])}
               for c in CLASSES}
    failed = {x['uid']: x['error'] for x in results if x['error']}
    done = {x['uid'] for x in results}
    seconds = [x['seconds'] for x in results]
    cache = Counter()
    for x in results:
        cache.update(x['cache'])
    return {'amendment': '019', 'status': 'EXPLORATORY_DESCRIPTIVE', 'pitch_m': PITCH, 'voxel_cm3': VOXEL_CM3,
            'sample_spacing_m': SAMPLE_M, 'taus_cm3': list(TAUS), 'primary_tau_cm3': PRIMARY,
            'repeats': repeats, 'seed': seed, 'systems_present': present,
            'systems_beyond_amendment_list': [s for s in present if s not in AMENDMENT_SYSTEMS],
            'cohort_rooms': len(cohort), 'analysed_rooms': len(analysed),
            'missing_rooms': {**{u: 'not rendered: ' + str(c.get('input_error', 'absent from render chunks'))
                                 for u, c in cohort.items() if u not in done},
                              **failed},
            'bbox_max_err_m': max((r['bbox_max_err_m'] for r in rows), default=None),
            'systems': systems, 'paired_vs_P': paired, 'gt_a4_classes': classes,
            'gt_a4_pairs': len(gt), 'cache': dict(cache),
            'runtime': {'rooms': len(seconds), 'room_seconds_sum': sum(seconds),
                        'room_seconds_mean': statistics.mean(seconds) if seconds else None,
                        'room_seconds_max': max(seconds, default=None)}}


def _ci(x):
    if not x or x.get('estimate') is None:
        return '—'
    ci = x.get('ci95')
    return f"{x['estimate']:.3f} " + (f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else '[CI —]')


def _diff(x):
    if not x or x.get('estimate') is None:
        return '—'
    ci = x.get('ci95')
    return f"{x['estimate']:+.3f} " + (f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else '[CI —]')


def _share(x):
    return '—' if not x['n'] else f"{x['rate']:.3f} ({x['k']}/{x['n']})"


def report(summary, wall_seconds, workers, command):
    s, lines = summary, []
    add = lines.append
    add('# 修订 019：渲染房间的网格级碰撞（探索性、描述性，无决策规则）\n')
    add(f"方案：`{SPEC}`（锁定于任何输出之前）。体素边长 {PITCH * 100:.0f} cm（每体素 {VOXEL_CM3:.0f} cm³），"
        f"三角形采样间距 ≤ {SAMPLE_M * 100:g} cm，阈值 τ ∈ {{{', '.join(map(str, TAUS))}}} cm³，主阈值 τ = {PRIMARY} cm³；"
        f"CI 为房屋连通组聚类自举（report_v2.contrasts，{s['repeats']} 次，种子 {s['seed']}）。\n")
    add('## 覆盖与缺失\n')
    add(f"- 冻结队列 {s['cohort_rooms']} 个房间，分析成功 {s['analysed_rooms']} 个；缺失 {len(s['missing_rooms'])} 个（原因见下，从不补齐）。")
    for u, why in sorted(s['missing_rooms'].items()):
        add(f'  - `{u}`：{why}')
    add(f"- 系统：{', '.join(s['systems_present'])}。修订列出的系统为 {', '.join(AMENDMENT_SYSTEMS)}；"
        f"运行中另外渲染了 {', '.join(s['systems_beyond_amendment_list']) or '无'}，按“该运行渲染的每个系统”一并报告。")
    bbox_error = '—' if s['bbox_max_err_m'] is None else f"{s['bbox_max_err_m']:.2e}"
    add(f"- 放置核验：每个网格反变换回物体坐标系后与布局包围盒的最大误差 {bbox_error} m（阈值 {BBOX_TOL_M:g} m）。")
    add('- 仅比较可移动物体之间的对；固定上下文物体（门窗等）不参与，与 validate.check 的物体对一致。\n')
    add('## 1. 各系统网格无碰撞房间比例（括号为 95% CI）\n')
    add('| 系统 | 房间 | 缺失 | 无碰撞 τ=125 | 无碰撞 τ=500（主） | 无碰撞 τ=2000 | 冻结检查器无 main/severe | A4 无碰撞 | 仅外壳（填充未增加体素） |')
    add('|---|---|---|---|---|---|---|---|---|')
    for name, x in s['systems'].items():
        lv = x['levels']
        add(f"| {name} | {x['rooms']} | {x['missing_rooms']} | {_ci(lv['mesh_free_125'])} | {_ci(lv['mesh_free_500'])} | "
            f"{_ci(lv['mesh_free_2000'])} | {_ci(lv['checker_collision_free'])} | {_share(x['a4_free'])} | "
            f"{x['open_meshes']}/{x['placements']} |")
    add('\n缺失 = 已分析房间中该系统没有布局（输出失败）；原因见 summary.json `missing_reasons`。'
        '“仅外壳”按放置后的物体计：填充没有增加任何体素。这是体素层面的判定，随系统预测的尺寸变化，'
        '不是网格水密性（资产网格几乎都不水密）。\n')
    add('## 2. 盒级碰撞对被网格确认的比例、检查器漏报\n')
    add('确认率 = 该类盒级碰撞对中网格干涉体积 > τ 的比例；漏报 = 网格碰撞对中冻结检查器未判 main/severe 的比例。\n')
    for t in TAUS:
        add(f'### τ = {t} cm³' + ('（主）' if t == PRIMARY else '') + '\n')
        add('| 系统 | 检查器 main/severe 对确认 | A4 对确认 | 网格碰撞对 | 检查器漏报 | 漏报构成 |')
        add('|---|---|---|---|---|---|')
        for name, x in s['systems'].items():
            q = x['per_tau'][t]
            mix = ', '.join(f'{k} {v}' for k, v in sorted(q['missed_by_checker_verdict'].items(), key=lambda kv: -kv[1]))
            add(f"| {name} | {_share(q['checker_main_severe_confirmed'])} | {_share(q['a4_confirmed'])} | "
                f"{q['mesh_colliding_pairs']} | {_share(q['checker_missed'])} | {mix or '—'} |")
        add('')
    add('## 3. GT：A4 证据类别 × 网格判定\n')
    add(f"GT 的 A4 盒级碰撞对共 {s['gt_a4_pairs']} 个，类别按修订 017 证据（gt-admission-anatomy）的顺序取第一个符合者。\n")
    add('| 类别 | 对数 | 网格碰撞 τ=125 | τ=500 | τ=2000 | 干涉体积中位数 cm³ |')
    add('|---|---|---|---|---|---|')
    for c, x in s['gt_a4_classes'].items():
        n = x['pairs']
        add(f"| {CLASS_ZH[c]} | {n} | " + ' | '.join(
            f"{x[f'collides_{t}']} ({x[f'collides_{t}'] / n:.2f})" if n else '—' for t in TAUS)
            + f" | {x['median_volume_cm3']:.0f} |")
    add('\n## 4. 与 P 的配对差（系统 − P，同房间，95% CI）\n')
    add('| 系统 | 配对房间 | Δ无碰撞 τ=125 | Δ τ=500（主） | Δ τ=2000 |')
    add('|---|---|---|---|---|')
    for name, x in s['paired_vs_P'].items():
        add(f"| {name} | {x['mesh_free_500']['n']} | {_diff(x['mesh_free_125'])} | {_diff(x['mesh_free_500'])} | "
            f"{_diff(x['mesh_free_2000'])} |")
    add('\n## 5. 诊断（不属于修订 019 的预设指标）\n')
    add('两物体平齐贴靠时会共享一层 2 cm 体素（约 2 cm × 接触面积），可超过 τ。deep_cm3 = 去掉“两物体表面体素重合”后的共享体积。\n')
    add('| 系统 | τ=500 网格碰撞对中仅接触层主导（deep ≤ τ）| 以 deep_cm3 计的无碰撞房间（τ=500） |')
    add('|---|---|---|')
    for name, x in s['systems'].items():
        add(f"| {name} | {_share(x['per_tau'][PRIMARY]['contact_dominated'])} | {_share(x[f'mesh_free_deep_{PRIMARY}'])} |")
    rt, c = s['runtime'], s['cache']
    add('\n## 运行\n')
    add(f"- 命令：`{command}`")
    add(f"- 墙钟 {wall_seconds:.0f} s，{workers} 个进程；单房间（全部系统）平均 {rt['room_seconds_mean'] or 0:.1f} s，"
        f"最长 {rt['room_seconds_max'] or 0:.1f} s，合计 {rt['room_seconds_sum']:.0f} CPU·s。")
    add(f"- 体素实体缓存（键：资产×尺寸×原生缩放×偏航角×亚体素偏移）：放置 {c.get('placements', 0)} 次，"
        f"实际体素化 {c.get('voxelised', 0)} 次。")
    add('- 地位：描述性、探索性；不改变修订 005–018 的任何结论或决策。')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------------------------- CLI

def _sha(path):
    return render.file_sha256(path)


def _jsonl(records):
    return ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in records)


def run(run_dir, out, workers=4, limit=None, repeats=2000, command=''):
    out = Path(out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    visual = Path(run_dir).resolve() / 'visual'
    cohort = visual_cases._index(visual_cases._read_rows(visual / 'cases.jsonl'))
    chunks = sorted(visual.glob('chunks/*/input.json'))
    rendered = visual_cases._index([c for p in chunks for c in json.loads(p.read_text())['cases']])
    if rendered.keys() - cohort.keys() or any(c != cohort[u] for u, c in rendered.items()):
        raise ValueError('render chunk inputs differ from visual/cases.jsonl')
    groups = {r['uid']: r['group'] for r in visual_cases._read_rows(visual / 'J-geometry.jsonl')}
    ledger = visual_cases._index(visual_cases._read_rows(visual / 'attachment-ledger.jsonl'))
    order = [u for u in cohort if u in rendered]
    if limit:
        count = min(limit, len(order))
        order = [order[round(i * len(order) / count)] for i in range(count)]
        cohort = {u: cohort[u] for u in order}
    inputs = {str(p.relative_to(visual.parent)): _sha(p) for p in
              [visual / 'cases.jsonl', visual / 'J-geometry.jsonl', visual / 'attachment-ledger.jsonl', *chunks]}
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(workers) as pool:
        mapper = pool.map if workers > 1 else map
        for k, x in enumerate(mapper(analyse_case, (rendered[u] for u in order)), 1):
            results.append(x)
            if k % 25 == 0 or k == len(order):
                print(f'{k}/{len(order)} rooms, {time.perf_counter() - start:.0f} s', file=sys.stderr, flush=True)
    wall = time.perf_counter() - start
    summary = {**summarize(results, groups, cohort, ledger, repeats), 'wall_seconds': wall, 'workers': workers,
               'limit': limit, 'command': command}
    meshes = {}
    for x in results:
        meshes.update(x['meshes'])
    if any(_sha(p) != h for p, h in ((visual.parent / q, h) for q, h in inputs.items())):
        raise ValueError('an input changed during the analysis')
    files = {'rows.jsonl': _jsonl({**r, 'group': groups[r['uid']]} for x in results for r in x['rows']),
             'pairs.jsonl': _jsonl(p for x in results for p in x['pairs']),
             'summary.json': json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
             'report.md': report(summary, wall, workers, command)}
    manifest = {'amendment': '019', 'run': str(Path(run_dir).resolve()), 'inputs_sha256': inputs,
                'spec_sha256': _sha(ROOT / SPEC) if (ROOT / SPEC).exists() else None,
                'code_sha256': {c: _sha(ROOT / c) for c in CODE}, 'mesh_sha256': dict(sorted(meshes.items())),
                'config': {'pitch_m': PITCH, 'sample_spacing_m': SAMPLE_M, 'taus_cm3': list(TAUS),
                           'primary_tau_cm3': PRIMARY, 'workers': workers, 'limit': limit, 'repeats': repeats},
                'command': command}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temp:
        for name, text in files.items():
            (Path(temp) / name).write_text(text, encoding='utf-8')
        manifest['files'] = {name: _sha(Path(temp) / name) for name in files}
        (Path(temp) / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        _publish(temp, out)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--repeats', type=int, default=2000)
    args = parser.parse_args(argv)
    command = ' '.join(['python -m fastfill.comparison.mesh_collision', *(argv if argv is not None else sys.argv[1:])])
    summary = run(args.run, args.out, args.workers, args.limit, args.repeats, command)
    print(json.dumps({'out': str(args.out), 'analysed_rooms': summary['analysed_rooms'],
                      'missing_rooms': len(summary['missing_rooms']), 'wall_seconds': round(summary['wall_seconds'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
