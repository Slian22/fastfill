"""Raw geometry and dimension metrics for the frozen unknown-size comparison.

OOR means pairwise axis-aligned object overlap / summed object footprint area;
it is NOT out-of-room rate. Failed requests keep null conditional metrics.
The separate <=10 cm boundary repair uses predicted sizes and never resizes.
"""
import collections
import copy
import math
import random
import statistics

from shapely.geometry import Polygon

from fastfill.comparison.contract import validate_output
from fastfill.comparison.train_pair_helpers import strict_json
from fastfill.interface import snap_inside
from fastfill.validate import check, support_ok

SCALARS = ('oor', 'size_iou', 'size_log_mae', 'parent_accuracy', 'oor_repaired')
RATES = ('complete', 'geometry_success', 'geometry_success_repaired')


def dimensions(reference, predicted):
    ious, errors = [], []
    for gt, pred in zip(reference, predicted):
        a, b = gt['size'], pred['size']
        overlap = math.prod(min(x, y) for x, y in zip(a, b))
        ious.append(overlap / (math.prod(a) + math.prod(b) - overlap))
        errors.append(statistics.mean(abs(math.log(x / y)) for x, y in zip(a, b)))
    return statistics.mean(ious), statistics.mean(errors)


def geometry(room):
    c = check(room)
    parents = {o['id']: o for o in room['objects'] + (room.get('fixed') or [])}
    support_fail = [o['id'] for o in room['objects']
                    if (o.get('parent') is not None and o['parent'] not in parents)
                    or not support_ok(o, parents.get(o.get('parent')))]
    valid = not (c['oob'] or support_fail or c['ceiling'])
    return {'oor': c['oor'], 'geometry_valid': valid,
            'geometry_success': bool(valid and not c['collisions']['main'] and
                                     not c['collisions']['severe'] and not c['fixed_blocking']),
            'oob_objects': len(c['oob']), 'support_fail': len(support_fail),
            'ceiling_fail': len(c['ceiling']), 'main_collisions': len(c['collisions']['main']),
            'severe_collisions': len(c['collisions']['severe']),
            'fixed_blocking': len(c['fixed_blocking'])}


def layout(room, placements):
    return [{**copy.deepcopy(o), 'size': copy.deepcopy(p['size']),
             'pos': copy.deepcopy(p['pos']), 'yaw': math.radians(p['yaw']),
             'parent': p.get('on')} for o, p in zip(room['objects'], placements)]


def small_repair(predicted):
    """Conservatively roll back a repair that breaks an existing fixed support."""
    placements = {o['id']: {'pos': list(o['pos']), 'yaw': o['yaw'], 'on': o.get('parent')}
                  for o in predicted['objects']}
    snap_inside(placements, predicted, Polygon(predicted['boundary']), (), predicted['boundary'])
    objects = [{**copy.deepcopy(o), 'pos': placements[o['id']]['pos']} for o in predicted['objects']]
    if any(a['size'] != b['size'] for a, b in zip(objects, predicted['objects'])):
        raise ValueError('repair changed predicted dimensions')
    fixed = {o['id']: o for o in predicted.get('fixed') or []}
    for before, after in zip(predicted['objects'], objects):
        parent = fixed.get(before.get('parent'))
        if parent is not None and support_ok(before, parent) and not support_ok(after, parent):
            return copy.deepcopy(predicted)
    return {**copy.deepcopy(predicted), 'objects': objects}


def _id_coverage(raw, expected):
    try:
        value = strict_json(raw)
        ids = {o.get('id') for o in value['placements'] if isinstance(o, dict) and isinstance(o.get('id'), str)}
        return len(ids & expected) / len(expected)
    except (ValueError, KeyError, TypeError):
        return 0.0


def score(room, request_messages, raw, arm, error=None):
    """Score a complete strict output; never inherit GT movable geometry/support.

    Context/fixed geometry is projected from the actual frozen model request.
    GT boxes are used only for dimension/support-label diagnostics.
    """
    row = {'uid': room['uid'], 'source': room['source'], 'group': room['group'], 'arm': arm,
           'room_type': room.get('room_type'), 'boundary_type': room['boundary_type'],
           'height_known': room.get('height') is not None, 'objects': len(room['objects']),
           'complete': False, 'geometry_success': False, 'geometry_success_repaired': False,
           'error': error, 'raw_layout': None, 'repaired_layout': None,
           'id_coverage': _id_coverage(raw, {o['id'] for o in room['objects']}) if raw else 0.0,
           **{k: None for k in SCALARS}}
    if error or raw is None:
        return {**row, 'error': error or 'missing_output'}
    try:
        if len(request_messages) != 2 or [m['role'] for m in request_messages] != ['system', 'user']:
            raise ValueError('expected exactly two input messages')
        target = validate_output(raw, room, arm)['placements']
        request = strict_json(request_messages[1]['content'])
        if [o['id'] for o in request['objects']] != [o['id'] for o in room['objects']]:
            raise ValueError('request/reference identity mismatch')
        if arm == 'placement':
            sizes = {o['id']: o['size'] for o in request['objects']}
            target = [{**p, 'size': sizes[p['id']]} for p in target]
        fixed = [{**f, 'yaw': math.radians(f['yaw'])} for f in request.get('fixed', [])]
        pred = {**copy.deepcopy(room), 'objects': layout(room, target), 'fixed': fixed,
                'boundary': request['boundary'], 'height': request.get('height'), 'constraints': []}
        size_iou, log_mae = dimensions(room['objects'], pred['objects'])
        repaired = small_repair(pred)
        raw_metrics, repaired_metrics = geometry(pred), geometry(repaired)
        return {**row, **raw_metrics, 'complete': True, 'error': None, 'size_iou': size_iou,
                'size_log_mae': log_mae, 'raw_layout': pred['objects'],
                'repaired_layout': repaired['objects'], 'oor_repaired': repaired_metrics['oor'],
                'geometry_success_repaired': repaired_metrics['geometry_success'],
                'repaired_metrics': repaired_metrics,
                'parent_accuracy': statistics.mean(a.get('parent') == b.get('parent')
                                                   for a, b in zip(room['objects'], pred['objects']))}
    except (ValueError, KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
        return {**row, 'error': type(exc).__name__ + ': invalid output/request/geometry'}


def aggregate(rows):
    n = len(rows)
    result = {'requests': n, 'complete_count': sum(r['complete'] for r in rows)}
    for key in RATES:
        result[key + '_rate'] = sum(bool(r[key]) for r in rows) / n if n else None
    result['mean_id_coverage'] = statistics.mean(r['id_coverage'] for r in rows) if n else None
    for key in SCALARS:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = statistics.mean(values) if values else None
        result[key + '_rooms'] = len(values)
    return result


def summarize(rows):
    sources = collections.defaultdict(list)
    for row in rows:
        sources[row['source']].append(row)
    by_source = {s: aggregate(r) for s, r in sorted(sources.items())}
    macro = {'sources': len(by_source)}
    for key in SCALARS + tuple(k + '_rate' for k in RATES):
        values = [r[key] for r in by_source.values() if r[key] is not None]
        macro[key] = statistics.mean(values) if values else None
        macro[key + '_sources'] = len(values)
    return {'all': aggregate(rows), 'by_source': by_source, 'source_macro': macro}


def _quantile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def _paired_design(joint, placement):
    """Freeze membership/signatures using every request, before metric filtering."""
    maps = [{r['uid']: r for r in rows} for rows in (joint, placement)]
    if any(len(m) != len(rows) for m, rows in zip(maps, (joint, placement))) or maps[0].keys() != maps[1].keys():
        raise ValueError('paired UID coverage mismatch or duplicates')
    groups = collections.defaultdict(list)
    for uid in sorted(maps[0]):
        a, b = maps[0][uid], maps[1][uid]
        if (a['source'], a['group']) != (b['source'], b['group']):
            raise ValueError('paired component/source mismatch')
        groups[a['group']].append((a, b))
    components = tuple(tuple(groups[k]) for k in sorted(groups))
    strata = collections.defaultdict(list)
    for index, pairs in enumerate(components):
        signature = tuple(sorted({a['source'] for a, _ in pairs}))
        strata[signature].append(index)
    return components, tuple(tuple(strata[k]) for k in sorted(strata))


def _metric_components(components, field):
    result = []
    for pairs in components:
        values = collections.defaultdict(list)
        for a, b in pairs:
            if field in SCALARS and not (a['complete'] and b['complete']):
                continue
            if a[field] is not None and b[field] is not None:
                delta = float(a[field]) - float(b[field])
                if not math.isfinite(delta):
                    raise ValueError('nonfinite paired metric')
                values[a['source']].append(delta)
        result.append({s: (sum(v), len(v)) for s, v in values.items()})
    return tuple(result)  # Includes zero-observation components as empty dicts.


def _source_totals(components):
    totals = {}
    for component in components:
        for source, (total, count) in component.items():
            previous, n = totals.get(source, (0., 0))
            totals[source] = (previous + total, n + count)
    return totals


def _paired_estimates(totals, macro_sources):
    count = sum(n for _, n in totals.values())
    micro = sum(d for d, _ in totals.values()) / count if count else None
    macro = (statistics.mean(totals[s][0] / totals[s][1] for s in macro_sources)
             if macro_sources and all(totals.get(s, (0, 0))[1] for s in macro_sources) else None)
    return micro, macro


def _interval(estimate, draws, repeats, singletons):
    defined = [v for v in draws if v is not None]
    undefined = repeats - len(defined)
    enough = len(defined) >= max(100, math.ceil(.95 * repeats))
    warnings = (['singleton_source_signature_strata'] if singletons else [])
    warnings += ['undefined_bootstrap_draws'] if undefined else []
    warnings += ['too_few_valid_bootstrap_draws'] if not enough else []
    return {'estimate_joint_minus_placement': estimate,
            'ci95': [_quantile(defined, .025), _quantile(defined, .975)] if enough else None,
            'valid_bootstrap_draws': len(defined), 'undefined_bootstrap_draws': undefined,
            'ci_conditioning': 'defined_draws', 'warnings': warnings}


def _metric_intervals(components, strata, sources, field, repeats, seed):
    parts = _metric_components(components, field)
    totals = _source_totals(parts)
    qualified = sorted(totals)
    estimates = _paired_estimates(totals, qualified)
    singletons = sum(len(v) == 1 for v in strata)
    rng, draws = random.Random(seed), []
    for _ in range(repeats):
        sampled = (parts[rng.choice(group)] for group in strata for _ in group)
        draws.append(_paired_estimates(_source_totals(sampled), qualified))
    micro = _interval(estimates[0], [d[0] for d in draws], repeats, singletons)
    macro = _interval(estimates[1], [d[1] for d in draws], repeats, singletons)
    return {**micro, 'paired_rooms': sum(n for _, n in totals.values()),
            'components': len(parts), 'data_qualified_components': sum(bool(p) for p in parts),
            'source_universe': list(sources), 'data_qualified_sources': len(qualified),
            'paired_rooms_by_source': {s: totals.get(s, (0, 0))[1] for s in sources},
            'singleton_source_signature_strata': singletons, 'replicates': repeats, 'seed': seed,
            'source_macro': {**macro, 'source_universe': qualified,
                             'data_qualified_sources': len(qualified)}}


def paired_intervals(joint, placement, repeats=2000, seed=42):
    """Paired room-micro and nested source_macro J-P deltas, sharing cluster draws.

    Freeze global components/source signatures on ALL requests, including components
    with no available metric. Scalars use common-complete, non-null pairs; rates
    retain every request. Macro averages the sources with eligible original pairs
    equally. That data-qualified source universe is fixed across draws: a draw
    missing any of those sources is undefined, never silently reweighted.

    CI endpoints condition on defined draws. Fewer than 100 defined draws or <95%
    of requested draws suppresses the CI and warns (a numerical guard, not a
    coverage guarantee). Singleton signature strata warn that between-component
    uncertainty is unidentified even when a numeric partial-uncertainty CI exists.
    Empty/all-missing inputs retain design counts and undefined-draw accounting.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError('at least two bootstrap replicates required')
    components, strata = _paired_design(joint, placement)
    sources = sorted({a['source'] for pairs in components for a, _ in pairs})
    return {field: _metric_intervals(components, strata, sources, field, repeats, seed)
            for field in SCALARS + RATES}
