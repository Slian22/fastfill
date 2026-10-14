"""Amendment-005 report and delivery decision for the unknown-size J/P comparison; never imputes.

python -m fastfill.comparison.report_v2 --data EVAL --geometry GEOMETRY_V2 --joint MERGED_J \
    --placement MERGED_P --reference REFERENCE.json --out NEW_DIR [--agent NAME=EVAL_LLM_DIR ...] \
    [--fid FID.json] [--judge JUDGE_DIR] [--corrections UIDS.json] [--repeats 2000] [--workers 1]

Every input is verified fail-closed: merged J/P inference dirs (evaluate_pair._inference), the
geometry-v2 evaluate_pair output that must bind those runs and this data, the frozen model-free
reference (reference_rows.py) that must bind this data and the same geometry.py, evaluate_llm agent
outputs on one shared cohort that paired these J/P runs, and FID/GPT outputs carrying this
geometry's lineage. geometry-v2 and every agent evaluation must have used the running geometry.py.
FID/GPT come from either the J/P visual run or the N-system one (fid systems mode with KID, judge
arms = visual_pipeline.judged_systems, e.g. J/P/L/O6, with summary.endpoint); the N-system lineage
is visual_pipeline.systems_lineage over J/P = GEOMETRY_V2 and each rendered agent label, which must
be passed as --agent LABEL=DIR.
Rules (amendments 005/006):
- reached_token_cap without stopped_on_eos is a failure in every all-request denominator (J/P per
  row; agents per row where their rows carry the flags, i.e. open models, else a count + warning);
- B_arm = mean over J/P common-complete rooms of the room mean of log(pred/GT footprint), also volume;
- (a') rescales each arm's predicted x/y about the object centre by exp(-B_arm/2), keeps pos/yaw
  and re-scores with geometry.geometry on the request context, exactly as geometry.score builds it;
- deliver J iff the (b) J-P CI lies below zero and the (a') J-P CI lower bound >= -0.02, else P.
Intervals re-use geometry.paired_intervals' design and draws (house-component bootstrap, seed 42,
room micro); p = 2 min(P(d<=0), P(d>=0)); Holm only across the co-primaries (a), (b). Agent
contrasts are estimation only. --corrections is a JSON list of UIDs changed by the final
source-correction candidate (separate sensitivity row). Outputs: report.md (Chinese), one CSV per
table, results.json and MANIFEST.json.
Audit 2026-10-14 reporting: results.json records the geometry summary's stage (absent = sft); the
delivery section is binding only at dpo2 (amendment 008), else an informational application of the
rule. T2/T3/S1 carry house components k: report.md prints no CI when k < 10 and flags k < 30 and
rooms < 50. T5 adds each system's own B on the cohort's all-system common-complete rooms and its
own-B (a') via _derive (EXPLORATORY). report.md renders report- and interval-level warnings.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import random
import statistics
import tempfile

from shapely.geometry import Polygon

from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import judge_batch
from fastfill.comparison import llm_agent
from fastfill.comparison import render_assets as render
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison import visual_pipeline
from fastfill.comparison.corpus import _publish
from fastfill.comparison.eval_data import PAPER_STRATA
from fastfill.comparison.fid import WEIGHTS_SHA256
from fastfill.comparison.final_tables import PAPER
from fastfill.scene import footprint, norm_room_type

KIND = 'amendment-005-reference-rows'
SEED = 42
MARGIN = -0.02
CAP = 'TOKEN_CAP_WITHOUT_EOS'
GLOBAL_FALLBACK = [0.4, 0.5, 0.65]
RUNTIME_FAILURES = ('BATCH_GENERATION_FAILED', 'INPUT_TOO_LONG', 'TOKENIZATION_FAILED', 'API_FAILED')
GPT_KEYS = ('functionality_score', 'layout_score', 'aesthetics_score')
LEVELS = ('bias_fp', 'bias_vol', 'size_log_mae')  # single-arm levels on J/P common-complete rooms
JP_FIELDS = ('geometry_success', 'size_log_mae', 'debiased', *(k + s for k in LEVELS for s in ('_J', '_P')))
P_RULE = ('category × room-type median, then category median, with constant fallback (0.40, 0.50, 0.65 m) '
          'when neither bucket has ≥ 5 training observations')  # Amendment 006, verbatim.
P_LABEL = f'P（{P_RULE}）'
NO_CI, FEW_CLUSTERS, SMALL_N = 10, 30, 50  # Audit 2026-10-14 F7: house components k and rooms per scope.
INTERVAL_WARNINGS = {'singleton_source_signature_strata': '某来源签名层只有一个房屋连通组，组间不确定性无法识别',
                     'undefined_bootstrap_draws': '部分自举抽样无定义（区间以有定义的抽样为条件）',
                     'too_few_valid_bootstrap_draws': '有定义的自举抽样不足，不给区间'}
ROLE = {'sft': 'SFT 阶段：(a)、(b) 的 J−P 为方案锁定的主要（确认性）比较（protocol.md），两终点间做 Holm 校正；其余均为描述或估计。',
        'dpo1': 'DPO1 阶段：全部 J−P 均为描述性估计，Holm p 只作记录、不作确认性陈述；'
                '主要（确认性）比较为 SFT J0/P0（protocol.md），交付比较为 DPO2（修订 008）。',
        'dpo2': 'DPO2 阶段：(a)、(b) 及交付规则所用的 (a′) 的 J−P 为次要的交付比较（修订 008），Holm 只在本阶段 (a)、(b) 内校正，不作确认性陈述；'
                '主要（确认性）比较为 SFT J0/P0（protocol.md）。'}


def stratum(row):
    key = (row['source'], norm_room_type(row.get('room_type')))
    return '/'.join(key) if key in PAPER_STRATA else None


def scopes(rows, low_reuse):
    """Named subsets; reference_rows uses the same predicates, so names line up across files."""
    named = {'all': rows}
    for source in sorted({r['source'] for r in rows}):
        named['source/' + source] = [r for r in rows if r['source'] == source]
    for source, kind in PAPER_STRATA:
        named[f'stratum/{source}/{kind}'] = [r for r in rows if stratum(r) == f'{source}/{kind}']
    named['strata_pooled'] = [r for r in rows if stratum(r)]
    named['IL3D_3dfront/living room+living dining room'] = [
        r for r in rows if r['source'] == 'IL3D_3dfront'
        and norm_room_type(r.get('room_type')) in ('living room', 'living dining room')]
    named['reuse/low'] = [r for r in rows if r['source'] in low_reuse]
    named['reuse/high'] = [r for r in rows if r['source'] not in low_reuse]
    return named


def rotated_oor(objects):
    """validate.check's OOR on the rotated footprints it already builds: pair overlap / total area."""
    shapes = [Polygon(footprint(o)) for o in objects]
    total = sum(s.area for s in shapes)
    overlap = sum(a.intersection(b).area for i, a in enumerate(shapes) for b in shapes[i + 1:])
    return overlap / total if total > 0 else 0.0


def size_bias(reference, predicted):
    """Room means over objects of signed log(pred/GT) footprint (x*y) and volume."""
    pairs = list(zip(reference, predicted))
    return (statistics.mean(math.log(p['size'][0] * p['size'][1] / (r['size'][0] * r['size'][1])) for r, p in pairs),
            statistics.mean(math.log(math.prod(p['size']) / math.prod(r['size'])) for r, p in pairs))


def cap_failure(row):
    return {**row, 'complete': False, 'geometry_success': False, 'geometry_success_repaired': False,
            'error': CAP, 'raw_layout': None, 'repaired_layout': None, **{k: None for k in g.SCALARS}}


def _with_bias(row, room):
    fp, vol = size_bias(room['objects'], row['raw_layout']) if row['complete'] else (None, None)
    return {**row, 'bias_fp': fp, 'bias_vol': vol}


def _derive(task):
    """Rotated OOR and, with a factor, the size-debiased success of one scored row."""
    row, room, request, factor = task
    debiased = None if factor is None else False
    if not row['complete']:
        return {'oor_rotated': None, 'success_debiased': debiased}
    if factor is not None:
        req = h.strict_json(request['messages'][1]['content'])
        objects = [{**o, 'size': [o['size'][0] * factor, o['size'][1] * factor, o['size'][2]]}
                   for o in row['raw_layout']]
        pred = {**room, 'objects': objects, 'fixed': [{**f, 'yaw': math.radians(f['yaw'])} for f in req.get('fixed', [])],
                'boundary': req['boundary'], 'height': req.get('height'), 'constraints': []}
        debiased = g.geometry(pred)['geometry_success']
    return {'oor_rotated': rotated_oor(row['raw_layout']), 'success_debiased': debiased}


def _map(function, tasks, workers):
    if workers == 1:
        return [function(t) for t in tasks]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(function, tasks, chunksize=16))


def contrasts(first, second, fields, repeats, seed=SEED):
    """Room-micro first-minus-second per field with paired_intervals' own design and draws."""
    components, strata = g._paired_design(first, second)
    parts = {f: g._metric_components(components, f) for f in fields}
    qualified = {f: sorted(g._source_totals(p)) for f, p in parts.items()}
    rng, draws = random.Random(seed), {f: [] for f in fields}
    for _ in range(repeats):
        picks = [rng.choice(group) for group in strata for _ in group]  # Same rng use as geometry.
        for f in fields:
            draws[f].append(g._paired_estimates(g._source_totals(parts[f][i] for i in picks), qualified[f])[0])
    singletons, result = sum(len(s) == 1 for s in strata), {}
    for f in fields:
        totals = g._source_totals(parts[f])
        estimate = g._paired_estimates(totals, qualified[f])[0]
        interval = g._interval(estimate, draws[f], repeats, singletons)
        defined = [d for d in draws[f] if d is not None]
        p = (min(1., 2 * min(sum(d <= 0 for d in defined), sum(d >= 0 for d in defined)) / len(defined))
             if defined and estimate is not None else None)
        result[f] = {'estimate': estimate, 'ci95': interval['ci95'], 'p': p,
                     'n': sum(n for _, n in totals.values()), 'warnings': interval['warnings']}
    return result


def holm(pvalues):
    if any(p is None for p in pvalues):
        return [None] * len(pvalues)
    adjusted, running = [None] * len(pvalues), 0.
    for rank, i in enumerate(sorted(range(len(pvalues)), key=lambda k: pvalues[k])):
        running = max(running, min(1., (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def delivery(size_mae, debiased):
    """Amendment 005 section 1; any undefined interval fails closed to P."""
    first = (size_mae['estimate'] is not None and size_mae['ci95'] is not None
             and size_mae['estimate'] < 0 and size_mae['ci95'][1] < 0)
    second = debiased['ci95'] is not None and debiased['ci95'][0] >= MARGIN
    return {'delivered': 'J' if first and second else 'P', 'condition_i_size_log_mae': first,
            'condition_ii_debiased_success': second, 'margin': MARGIN,
            'inputs': {'b_size_log_mae_joint_minus_placement': size_mae,
                       'a_prime_debiased_success_joint_minus_placement': debiased}}


def _rate(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return sum(map(bool, values)) / len(values) if values else None


def _pair_rows(joint, placement):
    """J/P rows; single-arm level fields ride on J rows (P copies carry 0.0), so J-minus-P = level."""
    first, second = [], []
    for a, b in zip(joint, placement):
        if a['uid'] != b['uid']:
            raise ValueError('J/P rows are not aligned')
        both = a['complete'] and b['complete']
        levels = {k + s: (r[k] if both else None) for k in LEVELS for s, r in (('_J', a), ('_P', b))}
        first.append({**a, **levels, 'debiased': a['success_debiased']})
        second.append({**b, **{k: 0. for k in levels}, 'debiased': b['success_debiased']})
    return first, second


def jp_scope(joint, placement, repeats):
    first, second = _pair_rows(joint, placement)
    return {'requests': len(joint), 'J_a': _rate(joint, 'geometry_success'), 'P_a': _rate(placement, 'geometry_success'),
            'J_a1': _rate(joint, 'success_debiased'), 'P_a1': _rate(placement, 'success_debiased'),
            'common_complete': sum(a['complete'] and b['complete'] for a, b in zip(joint, placement)),
            **contrasts(first, second, JP_FIELDS, repeats)}


def _design(rows):
    """House components k (the bootstrap clusters) of a scope and its small-sample flags."""
    k = len({r['group'] for r in rows})
    flags = [f for f, hit in (('few clusters', k < FEW_CLUSTERS), ('small n', len(rows) < SMALL_N)) if hit]
    return {'components': k, 'flags': '; '.join(flags) or None}


def own_bias(rows, keep, repeats):
    """One system's B (footprint, volume) on the kept rooms, house-cluster CI as a level against 0.0."""
    levels = [{**r, **{k: r[k] if c else None for k in ('bias_fp', 'bias_vol')}} for r, c in zip(rows, keep, strict=True)]
    return contrasts(levels, [{**r, 'bias_fp': 0., 'bias_vol': 0.} for r in rows], ('bias_fp', 'bias_vol'), repeats)


def _json(path):
    return h.strict_json(Path(path).read_text())


def _reference(path, data, expected, rooms):
    ref = _json(path)
    if ref.get('kind') != KIND or ref.get('data_manifest_sha256') != h.sha256(data / 'MANIFEST.json'):
        raise ValueError('reference rows do not bind this evaluation data')
    gt_pass, levels = ref['gt_pass']['uids'], ref['prior_levels']
    if (llm_agent._digest(json.dumps(gt_pass)) != ref['gt_pass']['sha256'] or not set(gt_pass) <= set(expected)
            or not set(ref['known_defect_uids']) <= set(expected) or set(levels) != set(expected)
            or any(len(levels[u]) != len(rooms[u]['objects']) for u in expected)):
        raise ValueError('reference rows do not match the frozen inventory')
    return ref


def _same_geometry(summary, what):
    if summary.get('code_sha256', {}).get('geometry.py') != h.sha256(g.__file__):
        raise ValueError(f'{what} used a geometry.py other than the running one')


def _geometry(root, data, expected, receipts, ref):
    manifest = e._verified(root)
    summary = _json(root / 'summary.json')
    if summary.get('status') != 'GEOMETRY_COMPLETE' or summary.get('data_manifest_sha256') != h.sha256(data / 'MANIFEST.json'):
        raise ValueError('geometry is not complete for this evaluation data')
    if summary.get('inference_run_sha256') != {arm: r['run_sha256'] for arm, r in receipts.items()}:
        raise ValueError('geometry does not bind the merged inference runs')
    if summary['code_sha256'].get('geometry.py') != ref['code_sha256']['geometry.py']:
        raise ValueError('reference and geometry used different geometry.py')
    _same_geometry(summary, 'geometry-v2')
    rows = {}
    for arm in h.ARMS:
        name = arm + '-geometry.jsonl'
        rows[arm] = h._rows(root, name, manifest['files'][name])
        if [r['uid'] for r in rows[arm]] != expected or any(r['arm'] != arm for r in rows[arm]):
            raise ValueError('geometry rows do not match the frozen inventory')
    stage = summary.get('stage', 'sft')  # Amendment 009 label; pre-009 SFT summaries carry none.
    if stage not in ROLE:
        raise ValueError(f'unknown geometry stage {stage!r}')
    return rows, stage


def _agent(root, data_sha, run_hashes):
    manifest = e._verified(root)
    summary = _json(root / 'summary.json')
    if summary.get('status') != 'LLM_COMPARISON_COMPLETE' or summary.get('data_manifest_sha256') != data_sha:
        raise ValueError('agent evaluation is incomplete or binds other data')
    if summary.get('inference_run_sha256') and summary['inference_run_sha256'] != run_hashes:
        raise ValueError('agent evaluation paired other J/P runs')
    _same_geometry(summary, 'agent evaluation ' + root.name)
    rows = {v: h._rows(root, v + '-geometry.jsonl', manifest['files'][v + '-geometry.jsonl'])
            for v in llm_agent.VARIANTS}
    uids = [r['uid'] for r in rows['llm_raw']]
    if ([r['uid'] for r in rows['llm_harness']] != uids
            or llm_agent._digest(json.dumps(uids)) != summary['cohort']['uids_sha256']):
        raise ValueError('agent rows do not match their cohort')
    return rows, summary, uids


def _lineage(geometry, agents, labels):
    """geometry_manifest_sha256 a visual run must carry: the geometry MANIFEST hash (J/P run, labels
    None) or json_sha256(systems_lineage) over J/P = geometry and each agent label's evaluation dir."""
    if labels is None:
        return h.sha256(geometry / 'MANIFEST.json')
    dirs = {label: geometry if label in ('J', 'P') else (agents or {}).get(label) for label in labels}
    if None in dirs.values():
        raise ValueError('visual systems need --agent LABEL=DIR for every rendered agent label')
    systems = visual_pipeline._systems(dirs)
    if list(systems) != list(labels):
        raise ValueError('visual system labels are not in protocol order')
    return render.json_sha256(visual_pipeline.systems_lineage(systems))


def _fid(path, geometry, agents):
    fid = _json(path)
    if (fid.get('extractor') != 'pytorch_fid:0.3.0:pool3' or fid.get('feature_dimension') != 2048
            or fid.get('weights_sha256') != WEIGHTS_SHA256
            or fid.get('provenance', {}).get('provenance', {}).get('geometry_manifest_sha256')
            != _lineage(geometry, agents, fid.get('systems'))):
        raise ValueError('FID extractor/weights/lineage mismatch')
    return fid


def _judge(root, geometry, agents, labels):
    """J/P judge, or the N-system judge (arms = judged_systems of the rendered labels)."""
    summary = _json(root / 'summary.json')
    records = [h.strict_json(line) for line in (root / 'records.jsonl').read_text().splitlines()]
    judge_batch.validate_summary(summary)
    arms = summary.get('arms')
    if arms is not None:
        labels = labels or ['J', 'P', *(s for s in visual_pipeline.SYSTEMS[2:] if s in (agents or {}))]
        if arms != visual_pipeline.judged_systems(labels):
            raise ValueError('judged system set mismatch')
    lineage = _lineage(geometry, agents, None if arms is None else labels)
    if summary.get('model') != 'gpt-6.1-sol' or summary.get('provenance', {}).get('geometry_manifest_sha256') != lineage:
        raise ValueError('judge model/lineage mismatch')
    again = judge_batch.summarize(records, summary['model'], summary['counts']['runnable'], arms or judge_batch.ARMS)
    if again['by_stratum'] != summary['by_stratum'] or again.get('endpoint') != summary.get('endpoint'):
        raise ValueError('judge records disagree with their summary')
    return records, summary


def _gpt(root, geometry, agents, labels, repeats):
    records, summary = _judge(root, geometry, agents, labels)
    if 'arms' not in summary:
        return gpt_endpoint(records, repeats)
    e = summary['endpoint']  # N-system run: the judge's own stratified-bootstrap endpoint is canonical.
    return {**gpt_endpoint(records, repeats, tuple(summary['arms'])),
            'pooled': {'estimate': e['mean_difference'], 'ci95': e['ci95'], 'n': e['both_complete'],
                       'source': 'judge_batch summary.endpoint', 'bootstrap': e['bootstrap']}}


def gpt_endpoint(records, repeats, arms=judge_batch.ARMS, seed=SEED):
    """Section 5: room mean of three sub-scores; J-P on both-complete rooms, resampled within stratum."""
    def mean3(r):
        return statistics.mean(r['scores'][k] for k in GPT_KEYS)
    by = {(r['uid'], r['arm']): r for r in records}
    diffs = collections.defaultdict(list)
    for uid, name in sorted({(r['uid'], r['stratum']) for r in records}):
        a, b = by.get((uid, arms[0])), by.get((uid, arms[1]))
        if a and b and a['status'] == b['status'] == 'completed':
            diffs[name].append(mean3(a) - mean3(b))
    pooled = [d for s in sorted(diffs) for d in diffs[s]]
    rng = random.Random(seed)
    draws = [statistics.mean([rng.choice(diffs[s]) for s in sorted(diffs) for _ in diffs[s]])
             for _ in range(repeats)] if pooled else []
    by_arm = {(r['stratum'], r['arm']): [] for r in records}
    for r in records:
        if r['status'] == 'completed':
            by_arm[r['stratum'], r['arm']].append(mean3(r))
    return {'arms': list(arms), 'pooled': {'estimate': statistics.mean(pooled) if pooled else None, 'n': len(pooled),
                       'ci95': [g._quantile(draws, .025), g._quantile(draws, .975)] if draws else None},
            'difference_by_stratum': {s: {'n': len(v), 'mean_difference': statistics.mean(v)} for s, v in diffs.items()},
            'arm_by_stratum': {f'{s}|{a}': {'success': len(v), 'requested': sum(r['stratum'] == s and r['arm'] == a
                               for r in records), 'mean': statistics.fmean(v) if v else None} for (s, a), v in by_arm.items()}}


def _parses(raw):
    try:
        h.strict_json(raw)
        return True
    except (ValueError, RecursionError, TypeError):
        return False


def taxonomy(rows, records=None, cap_known=None):
    """Failure kinds; the cap kind is listed (0 included) only where per-row cap flags exist."""
    cap = ('token_cap_without_eos',) if (records is not None if cap_known is None else cap_known) else ()
    kinds = cap + (('invalid_json', 'invalid_schema_or_geometry') if records is not None
                   else ('invalid_output',)) + ('generation_or_api_failure', 'missing_output')
    counts = collections.Counter({'requests': len(rows), 'complete': 0, **dict.fromkeys(kinds, 0)})
    for i, row in enumerate(rows):
        error = row.get('error') or 'missing_output'
        if row['complete']:
            kind = 'complete'
        elif error == CAP:
            kind = 'token_cap_without_eos'
        elif error.split(':')[0] in RUNTIME_FAILURES:
            kind = 'generation_or_api_failure'
        elif error.endswith('invalid output/request/geometry'):
            kind = ('invalid_output' if records is None else
                    'invalid_schema_or_geometry' if _parses(records[i]['raw']) else 'invalid_json')
        else:
            kind = 'missing_output'
        counts[kind] += 1
    return {**counts, 'incomplete': len(rows) - counts['complete']}


def _object_levels(joint, placement, rooms, levels):
    """Object-micro size errors by prior level on J/P common-complete rooms."""
    acc = collections.defaultdict(lambda: {'objects': 0, 'J': [], 'P': []})
    for a, b in zip(joint, placement):
        if a['complete'] and b['complete']:
            for level, gt, pj, pp in zip(levels[a['uid']], rooms[a['uid']]['objects'], a['raw_layout'], b['raw_layout']):
                acc[level]['objects'] += 1
                acc[level]['J'].append(g.dimensions([gt], [pj]))
                acc[level]['P'].append(g.dimensions([gt], [pp]))
    return {level: {'objects': v['objects'], **{f'{arm}_{k}': statistics.mean(x[i] for x in v[arm])
                    for arm in 'JP' for i, k in enumerate(('size_iou', 'size_log_mae'))}} for level, v in sorted(acc.items())}


def _cell(value, components=None):
    if value is None:
        return '—'
    if isinstance(value, dict):
        if value.get('estimate') is None:
            return '—'
        if components is not None and components < NO_CI:
            return f"{value['estimate']:+.4f} — ({components} components)"
        ci = value.get('ci95')
        return f"{value['estimate']:+.4f} " + (f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else '[CI —]')
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def _md(rows, columns):
    lines = ['| ' + ' | '.join(title for _, title in columns) + ' |', '|' + '---|' * len(columns)]
    return lines + ['| ' + ' | '.join(_cell(r.get(key), r.get('components')) for key, _ in columns) + ' |'
                    for r in rows] + ['']


def _flat(row):
    out = {}
    for key, value in row.items():
        if isinstance(value, dict) and 'estimate' in value:
            ci = value.get('ci95') or [None, None]
            out.update({key: value['estimate'], key + '_ci_low': ci[0], key + '_ci_high': ci[1],
                        key + '_n': value.get('n'), key + '_p': value.get('p')})
        elif not isinstance(value, (dict, list)):
            out[key] = value
    return out


def _csv(path, rows):
    flat = [_flat(r) for r in rows]
    fields = list(dict.fromkeys(k for r in flat for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


DESIGN_COLUMNS = [('components', '房屋连通组 k'), ('flags', '标记')]
JP_COLUMNS = [('scope', '范围'), ('requests', '请求'), *DESIGN_COLUMNS, ('gt_success', 'GT成功 原始'),
              ('gt_success_snapped', 'GT成功 贴边'),
              ('J_a', 'J (a)'), ('P_a', 'P (a)'), ('geometry_success', 'Δ(a) J−P [95% CI]'),
              ('size_log_mae_J', 'J log MAE'), ('size_log_mae_P', 'P log MAE'), ('size_log_mae', 'Δ(b) J−P'),
              ('bias_fp_J', 'B_J 足迹'), ('bias_fp_P', 'B_P 足迹'), ('debiased', "Δ(a′) J−P"),
              ('prior_size_log_mae', '先验 log MAE（全部房间）')]


def _jp_row(name, joint, placement, ref_scope, repeats):
    ref_scope = ref_scope or {}
    return {'scope': name, **jp_scope(joint, placement, repeats), **_design(joint),
            'gt_success': ref_scope.get('gt', {}).get('gt_success'),
            'gt_success_snapped': ref_scope.get('gt', {}).get('gt_success_snapped'),
            'prior_size_log_mae': ref_scope.get('prior', {}).get('prior_size_log_mae')}


def _aggregate(rows):
    agg = g.aggregate(rows)
    values = [r['oor_rotated'] for r in rows if r['oor_rotated'] is not None]
    return {**agg, 'oor_rotated': statistics.mean(values) if values else None}


def build(data, geometry, runs, reference, out, agents=None, fid=None, judge=None, corrections=None,
          repeats=2000, workers=1):
    data, geometry, out = Path(data), Path(geometry), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = e._verified(data)
    expected, data_sha = manifest['expected_uids'], h.sha256(data / 'MANIFEST.json')
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    requests = {arm: h._by_uid(h._rows(data, arm + '/requests.jsonl', manifest['files'][arm + '/requests.jsonl']))
                for arm in h.ARMS}
    records, receipts = {}, {}
    for arm in h.ARMS:
        records[arm], receipts[arm] = e._inference(Path(runs[arm]), data, expected)
    run_hashes = {arm: r['run_sha256'] for arm, r in receipts.items()}
    ref = _reference(reference, data, expected, rooms)
    (scored, stage), geometry_sha = _geometry(geometry, data, expected, receipts, ref), h.sha256(geometry / 'MANIFEST.json')
    rows, cap_complete = {}, {}
    for arm in h.ARMS:
        caps = [r['reached_token_cap'] and not r['stopped_on_eos'] for r in records[arm]]
        cap_complete[arm] = sum(c and r['complete'] for c, r in zip(caps, scored[arm]))
        rows[arm] = [_with_bias(cap_failure(r) if c else r, rooms[r['uid']]) for r, c in zip(scored[arm], caps)]
    both = [a['complete'] and b['complete'] for a, b in zip(rows['joint'], rows['placement'])]
    bias = {arm: statistics.mean(r['bias_fp'] for r, c in zip(rows[arm], both) if c) if any(both) else None
            for arm in h.ARMS}
    factor = {arm: None if b is None else math.exp(-b / 2) for arm, b in bias.items()}
    for arm in h.ARMS:
        tasks = [(r, rooms[r['uid']], requests[arm][r['uid']], factor[arm]) for r in rows[arm]]
        rows[arm] = [{**r, **d} for r, d in zip(rows[arm], _map(_derive, tasks, workers))]
    systems, agent_meta, cohort, row_cap, flipped = {}, {}, None, {}, {}
    for name, root in (agents or {}).items():
        variants, summary, uids = _agent(Path(root), data_sha, run_hashes)
        if cohort is not None and uids != cohort:
            raise ValueError('agents were scored on different cohorts')
        cohort, agent_meta[name] = uids, summary
        # Amendment 006: the cap rule per row where the rows carry the flags (open models).
        # Rooms without an answer (e.g. INPUT_TOO_LONG) carry no flags, so any flagged row enables the rule.
        row_cap[name] = any('reached_token_cap' in r and 'stopped_on_eos' in r for v in variants.values() for r in v)
        for variant, vrows in variants.items():
            label = f"{name}-{variant.removeprefix('llm_')}"
            caps = [row_cap[name] and r.get('reached_token_cap', False) and not r.get('stopped_on_eos', True)
                    for r in vrows]
            flipped[label] = sum(c and r['complete'] for c, r in zip(caps, vrows)) if row_cap[name] else None
            vrows = [_with_bias(cap_failure(r) if c else r, rooms[r['uid']]) for r, c in zip(vrows, caps)]
            derived = _map(_derive, [(r, rooms[r['uid']], None, None) for r in vrows], workers)
            systems[label] = [{**r, **d} for r, d in zip(vrows, derived)]
    visual_pipeline.check_labels(agent_meta)
    fid_result = _fid(fid, geometry, agents) if fid else None
    gpt = _gpt(Path(judge), geometry, agents, (fid_result or {}).get('systems'), repeats) if judge else None
    corrected = set(_json(corrections)) if corrections else None
    if corrected is not None and not corrected <= set(expected):
        raise ValueError('correction UIDs outside the frozen inventory')

    J, P, low = rows['joint'], rows['placement'], ref['low_reuse_sources']
    sj, sp, rs = scopes(J, low), scopes(P, low), ref['scopes']
    tables = {}
    # T1: co-primaries, bias guard, debiased success and the delivery decision.
    t1 = _jp_row('all', J, P, rs['all'], repeats)
    p_holm = holm([t1['geometry_success']['p'], t1['size_log_mae']['p']])
    decision = {**delivery(t1['size_log_mae'], t1['debiased']), 'stage': stage, 'binding': stage == 'dpo2',  # Amendment 008.
                'raw_a_joint_minus_placement': t1['geometry_success'], 'B_footprint': {'J': t1['bias_fp_J'], 'P': t1['bias_fp_P']},
                'scale_factor': factor, 'gt_reference_success': t1['gt_success'],
                'a_prime': {'J': t1['J_a1'], 'P': t1['P_a1']}, 'raw_a': {'J': t1['J_a'], 'P': t1['P_a']}}
    tables['T1'] = [
        {'endpoint': '(a) 全请求原始几何成功率', 'J': t1['J_a'], 'P': t1['P_a'], 'delta': t1['geometry_success'],
         'n': t1['geometry_success']['n'], 'p': t1['geometry_success']['p'], 'p_holm': p_holm[0]},
        {'endpoint': '(b) 尺寸 log MAE（共同完整）', 'J': t1['size_log_mae_J'], 'P': t1['size_log_mae_P'],
         'delta': t1['size_log_mae'], 'n': t1['size_log_mae']['n'], 'p': t1['size_log_mae']['p'], 'p_holm': p_holm[1]},
        {'endpoint': "(a′) 尺寸去偏几何成功率", 'J': t1['J_a1'], 'P': t1['P_a1'], 'delta': t1['debiased'],
         'n': t1['debiased']['n']}]
    prior_all = rs['all']['prior']
    tables['T1_bias'] = [
        {'system': 'J', 'footprint': t1['bias_fp_J'], 'volume': t1['bias_vol_J'], 'scale_factor': factor['joint']},
        {'system': 'P', 'footprint': t1['bias_fp_P'], 'volume': t1['bias_vol_P'], 'scale_factor': factor['placement']},
        {'system': 'P 先验（全部房间，参考）', 'footprint': prior_all['prior_bias_fp'], 'volume': prior_all['prior_bias_vol']}]
    secondary = g.paired_intervals(J, P, repeats=repeats)
    tables['T1b'] = [{'field': f, **{k: {'estimate': v['estimate_joint_minus_placement'], 'ci95': v['ci95'], 'n': n}
                      for k, v, n in (('micro', secondary[f], secondary[f]['paired_rooms']),
                                      ('macro', secondary[f]['source_macro'], secondary[f]['source_macro']['data_qualified_sources']))}}
                     for f in ('complete', 'geometry_success', 'geometry_success_repaired', 'oor', 'oor_repaired', 'size_iou', 'size_log_mae')]
    # T2: model-free reference rows.
    tables['T2'] = [{'scope': name, 'requests': v['requests'], **_design(sj[name]), **v['gt'], **v['prior']}
                    for name, v in rs.items()]
    # T3: per source and paper stratum, GT reference beside each arm.
    t3_names = ['all', *(n for n in sj if n.startswith(('source/', 'stratum/'))), 'strata_pooled']
    tables['T3'] = [t1, *(_jp_row(n, sj[n], sp[n], rs.get(n), repeats) for n in t3_names[1:])]
    # T4: paper-style six strata. Visual keys: joint/placement (J/P run) or J/P/agent label (N-system
    # run, which renders and judges each agent's harness answer).
    everyone = {'J': J, 'P': P, **systems}
    fid_systems = 'systems' in (fid_result or {})
    judge_systems = (gpt or {}).get('arms', list(judge_batch.ARMS)) != list(judge_batch.ARMS)

    def visual_key(label, n_mode):
        if label in ('J', 'P'):
            return label if n_mode else {'J': 'joint', 'P': 'placement'}[label]
        return label.removesuffix('-harness') if n_mode and label.endswith('-harness') else None
    tables['T4'] = []
    for source, kind in PAPER_STRATA:
        name = f'{source}/{kind}'
        visual = (fid_result or {}).get('strata', {}).get(name, {})
        for label, srows in everyone.items():
            agg = _aggregate([r for r in srows if stratum(r) == name])
            arm, judge_arm = visual_key(label, fid_systems), visual_key(label, judge_systems)
            judged = (gpt or {}).get('arm_by_stratum', {}).get(f'{name}|{judge_arm}', {})
            shared = visual.get('counts', {}).get('shared_success')
            kid = ((visual.get('kid') or {}).get(arm) or {}) if arm else {}
            tables['T4'].append({
                'stratum': name, 'system': label, 'requests': agg['requests'], 'complete': agg['complete_count'],
                'oor': agg['oor'], 'oor_rotated': agg['oor_rotated'],
                'gt_oor': rs.get('stratum/' + name, {}).get('gt', {}).get('gt_oor'),
                'gt_oor_rotated': rs.get('stratum/' + name, {}).get('gt', {}).get('gt_oor_rotated'),
                'fid': visual.get(arm + '_vs_reference') if arm else None, 'fid_n': shared if arm else None,
                'fid_flag': ('N<50' if shared is not None and shared < 50 else None) if arm else None,
                'kid': kid.get('kid'), 'kid_se': kid.get('se'),
                'gpt': judged.get('mean'), 'gpt_success': judged.get('success'), 'gpt_requested': judged.get('requested')})
    tables['T4_gpt'] = ([{'scope': 'pooled', 'delta': gpt['pooled'], 'n': gpt['pooled']['n']}]
                        + [{'scope': s, 'mean_difference': v['mean_difference'], 'n': v['n']}
                           for s, v in sorted(gpt['difference_by_stratum'].items())]) if gpt else []
    tables['T4_paper'] = [{'stratum': f'{s}/{k}', **dict(zip(('FID', 'OOR', 'functionality', 'layout', 'aesthetics'), v))}
                          for (s, k), v in zip(PAPER_STRATA, PAPER)]
    # T5: agent benchmark on the shared cohort.
    tables['T5'], tables['T5_contrasts'], tables['T5_cost'] = [], [], []
    if cohort is not None:
        index = {arm: {r['uid']: r for r in rows[arm]} for arm in h.ARMS}
        bench = {'J': [index['joint'][u] for u in cohort], 'P': [index['placement'][u] for u in cohort], **systems}
        # EXPLORATORY (audit 2026-10-14): each system's own B on the rooms where J, P and that system are all
        # complete (so one weak agent cannot shrink every other system's room set), and (a') rescaled by it
        # through _derive; agents answer the J request (evaluate_llm).
        base = [j['complete'] and p['complete'] for j, p in zip(bench['J'], bench['P'], strict=True)]
        common = {label: [c and r['complete'] for c, r in zip(base, brows, strict=True)] for label, brows in bench.items()}
        own_factor = {}
        for label, brows in bench.items():
            b = (statistics.mean(r['bias_fp'] for r, c in zip(brows, common[label]) if c)
                 if any(common[label]) else None)
            own_factor[label] = None if b is None else math.exp(-b / 2)
            arm = 'placement' if label == 'P' else 'joint'
            tasks = [(r, rooms[r['uid']], requests[arm][r['uid']], own_factor[label]) for r in brows]
            bench[label] = [{**r, 'own_debiased': d['success_debiased']} for r, d in zip(brows, _map(_derive, tasks, workers))]
        for scope_name, keep in (('cohort', lambda r: True), ('strata_pooled', stratum)):
            subset = {k: [r for r in v if keep(r)] for k, v in bench.items()}
            for label, srows in subset.items():
                mask = [c for c, r in zip(common[label], bench[label], strict=True) if keep(r)]
                agg = g.aggregate(srows)
                tables['T5'].append({'scope': scope_name, 'system': label, **{k: agg[k] for k in (
                    'requests', 'complete_rate', 'geometry_success_rate', 'size_log_mae', 'size_iou', 'oor')},
                    **own_bias(srows, mask, repeats), 'scale_factor': own_factor[label],
                    'success_debiased_rate': _rate(srows, 'own_debiased')})
            pairs = [('J', 'P'), *((a, s) for s in systems for a in ('J', 'P'))]
            for a, b in pairs:
                c = contrasts(subset[a], subset[b], ('geometry_success', 'size_log_mae'), repeats)
                tables['T5_contrasts'].append({'scope': scope_name, 'comparison': f'{a}−{b}', **c})
        cohort_set = set(cohort)
        for arm, label in (('joint', 'J'), ('placement', 'P')):
            cohort_records = [r for r in records[arm] if r['uid'] in cohort_set]
            batches = {r.get('batch_id'): r.get('duration_seconds') for r in records[arm]}
            tables['T5_cost'].append({'system': label, 'model': 'Qwen3-8B + LoRA',
                'gpu_hours_full_test_generation': (sum(batches.values()) / 3600 if None not in batches and None not in batches.values() else None),
                'output_tokens': sum(r.get('generated_tokens') or 0 for r in cohort_records),
                'input_tokens': sum(r.get('input_tokens') or 0 for r in cohort_records)})
        for name, summary in agent_meta.items():
            result, config = summary['llm']['result'], summary['llm']['config']
            cost = summary['llm'].get('cost') or {}  # Open models: the run's cost.json via evaluate_llm.
            tables['T5_cost'].append({'system': name, 'model': config.get('model'), 'reasoning': config.get('reasoning_effort'),
                'calls': result.get('calls_ledger'), 'gpu_hours': cost.get('gpu_hours'),
                'input_tokens': (result.get('tokens') or {}).get('input_tokens'),
                'output_tokens': (result.get('tokens') or {}).get('output_tokens'),
                'reasoning_tokens': (result.get('tokens') or {}).get('reasoning_tokens'),
                'latency_p50_s': (result.get('latency_seconds') or {}).get('p50'),
                'answers_reaching_cap': result.get('reached_token_cap')})
    # T6: failure taxonomy (J/P on the full test set, agents on the cohort).
    tables['T6'] = [{'system': 'J', **taxonomy(J, records['joint']), 'cap_rule_flipped_complete': cap_complete['joint']},
                    {'system': 'P', **taxonomy(P, records['placement']), 'cap_rule_flipped_complete': cap_complete['placement']},
                    *({'system': k, **taxonomy(v, cap_known=flipped[k] is not None), 'cap_rule_flipped_complete': flipped[k],
                       'answers_reaching_cap': agent_meta[k.rsplit('-', 1)[0]]['llm']['result'].get('reached_token_cap')}
                      for k, v in systems.items())]
    # Sensitivity.
    gt_pass, known, levels = set(ref['gt_pass']['uids']), set(ref['known_defect_uids']), ref['prior_levels']
    filters = {f"GT 通过冻结几何门（{len(gt_pass)}）": lambda r: r['uid'] in gt_pass,
               f"排除声明的已知缺陷房间（{len(known)}）": lambda r: r['uid'] not in known}
    if corrected is not None:
        filters[f'排除源修正候选变化房间（{len(corrected)}）'] = lambda r: r['uid'] not in corrected
    filters['含全局回退物体的房间'] = lambda r: 'G' in levels[r['uid']]
    filters['不含全局回退物体的房间'] = lambda r: 'G' not in levels[r['uid']]
    tables['S1'] = [_jp_row(n, [r for r in J if f(r)], [r for r in P if f(r)], None, repeats) for n, f in filters.items()]
    tables['S1'] += [_jp_row(n, sj[n], sp[n], rs.get(n), repeats)
                     for n in ('reuse/low', 'reuse/high', 'IL3D_3dfront/living room+living dining room')]
    reference_levels = ref['prior_level_objects']
    tables['S2'] = [{'level': level, **v, **{'prior_' + k: x for k, x in reference_levels.get(level, {}).items()}}
                    for level, v in _object_levels(J, P, rooms, levels).items()]

    warnings = []
    if ref['prior_level_method'] != 'exact_frozen_prior':
        warnings.append('prior_level_pair_category_unresolved_without_frozen_size_priors')
    if any((s['llm']['result'].get('reached_token_cap') or 0) and not row_cap[k] for k, s in agent_meta.items()):
        warnings.append('agent_answers_reached_output_cap_not_row_applied')
    if corrected is None:
        warnings.append('source_correction_sensitivity_not_provided')
    result = {'status': 'REPORT_V2_COMPLETE', 'amendment': '006', 'stage': stage, 'delivery': decision, 'tables': tables,
              'holm_co_primaries': {'a': p_holm[0], 'b': p_holm[1]}, 'warnings': warnings,
              'agents': {k: {'model': v['llm']['config'].get('model'), 'cohort_sha256': v['cohort']['sha256'],
                             'cap_rule_per_row': row_cap[k]} for k, v in agent_meta.items()},
              'visual_mode': {'fid': 'systems' if fid_systems else 'J/P' if fid else None,
                              'judge': 'systems' if judge_systems else 'J/P' if judge else None},
              'cohort_rooms': len(cohort) if cohort else None, 'repeats': repeats, 'seed': SEED,
              'input_sha256': {'data_MANIFEST': data_sha, 'geometry_MANIFEST': geometry_sha,
                               'reference': h.sha256(reference), 'inference_run_sha256': run_hashes,
                               **{'agent_' + k + '_MANIFEST': h.sha256(Path(v) / 'MANIFEST.json') for k, v in (agents or {}).items()},
                               **({'FID': h.sha256(fid)} if fid else {}),
                               **({'judge_summary': h.sha256(Path(judge) / 'summary.json'),
                                   'judge_records': h.sha256(Path(judge) / 'records.jsonl')} if judge else {}),
                               **({'corrections': h.sha256(corrections)} if corrections else {})},
              'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(g.__file__), Path(e.__file__))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'results.json', result)
        for name, table in tables.items():
            if table:
                _csv(dest / (name + '.csv'), table)
        (dest / 'report.md').write_text(_report(result, agent_meta))
        h.write_json(dest / 'MANIFEST.json', {'input_sha256': result['input_sha256'], 'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return result


def _report(result, agent_meta):
    t, d, stage = result['tables'], result['delivery'], result['stage']
    decision = (['## 交付决定（修正案 005 §1；修订 008：DPO2 为正式交付比较）', '',
                 f"**交付：{d['delivered']}**。这是为用户管线选择的产物，不是科学上的优劣结论。", '']
                if d['binding'] else
                [f'## 修正案 005 §1 交付规则的参考性应用（{stage} 阶段，非交付决定；binding decision at DPO2, amendment 008）', '',
                 f"规则结果：{d['delivered']}（仅供参考；正式交付决定只在 DPO2 阶段作出，修订 008）。", ''])
    lines = [f"# 未知尺寸 J／P 对比（{stage} 阶段）：修正案 005 报告" + ('与交付决定' if d['binding'] else ''), '',
             f'J：联合预测尺寸、位置和朝向。{P_LABEL}：尺寸来自训练集先验（不是预测），再生成位置和朝向。'
             f"两者为同一 Qwen3-8B 基座、单一训练种子；区间只覆盖测试房间抽样（房屋连通组自举 {result['repeats']} 次，"
             f'种子 {SEED}；协议设定为 2,000 次），不覆盖训练随机性。历史测试集曾用于诊断，不是新盲测。缺失值一律显示为“—”，从不补值。', '',
             *decision,
             f"- 条件 (i) J 的尺寸 log MAE 低于 P 且配对 95% CI 不含 0：{_cell(d['condition_i_size_log_mae'])}；"
             f"Δ(b) J−P = {_cell(d['inputs']['b_size_log_mae_joint_minus_placement'])}",
             f"- 条件 (ii) (a′) J−P 的 95% CI 下界 ≥ −0.02（−2.0 个百分点）：{_cell(d['condition_ii_debiased_success'])}；"
             f"Δ(a′) J−P = {_cell(d['inputs']['a_prime_debiased_success_joint_minus_placement'])}",
             f"- 同时列出：原始 (a) Δ = {_cell(d['raw_a_joint_minus_placement'])}；B_J = {_cell(d['B_footprint']['J'])}，"
             f"B_P = {_cell(d['B_footprint']['P'])}；缩放系数 exp(−B/2)：J {_cell(d['scale_factor']['joint'])}，"
             f"P {_cell(d['scale_factor']['placement'])}；GT 参考几何成功率 {_cell(d['gt_reference_success'])}。", '',
             '任一区间无法计算时，规则按失败处理并交付 P。', '',
             f"## T1 (a)、(b) 终点（全部测试请求，配对 J−P；{ {'sft': 'SFT：主要确认性比较', 'dpo1': 'DPO1：描述性', 'dpo2': 'DPO2：次要交付比较'}[stage] }）", '']
    lines += _md(t['T1'], [('endpoint', '终点'), ('J', 'J'), ('P', 'P'), ('delta', 'Δ J−P [95% CI]'), ('n', '配对 n'),
                           ('p', '自举双侧 p'), ('p_holm', 'Holm p（本阶段 (a)、(b) 内）')])
    lines += ['(a) 以全部请求为分母，失败/不完整输出计 0；达到生成上限且未输出 EOS 的请求即使 JSON 可解析也计为失败。'
              '(b) 只在 J、P 共同完整的房间上计算，房间内按物体平均，再按房间平均。'
              "(a′) 将各臂预测的 x/y 尺寸绕物体中心乘以 exp(−B/2)，位置与朝向不变，用冻结几何代码重新评分。", '',
              '## T1-B 带符号尺寸偏差 B（共同完整房间，log(预测/GT)）', '']
    lines += _md(t['T1_bias'], [('system', '系统'), ('footprint', '足迹偏差 [95% CI]'), ('volume', '体积偏差 [95% CI]'),
                                ('scale_factor', '去偏缩放 exp(−B/2)')])
    lines += ['## T1b 次要配对终点（J−P；房间微平均与来源宏平均）', '']
    lines += _md(t['T1b'], [('field', '指标'), ('micro', '房间微平均 [95% CI]'), ('macro', '来源宏平均 [95% CI]')])
    lines += ['“repaired” 指“贴边（≤10 cm）；可能产生碰撞”：它不单调，在 GT 上反而降低成功率。原始结果为主。', '',
              '## T2 模型无关参考行（冻结）', '']
    lines += _md(t['T2'], [('scope', '范围'), ('requests', '房间'), *DESIGN_COLUMNS, ('gt_success', 'GT成功 原始'),
                           ('gt_success_snapped', 'GT成功 贴边'), ('gt_oor', 'GT OOR 轴对齐'), ('gt_oor_snapped', 'GT OOR 贴边'),
                           ('gt_oor_rotated', 'GT OOR 旋转'), ('prior_size_iou', '先验 尺寸IoU'),
                           ('prior_size_log_mae', '先验 log MAE'), ('prior_bias_fp', '先验 足迹偏差'),
                           ('global_fallback_object_rate', '全局回退 物体比例'),
                           ('rooms_with_global_fallback_rate', '含回退 房间比例')])
    lines += ['GT 是参考，不是上限。先验行使用全部房间；J／P 行只在完成的输出上计算尺寸指标。'
              f'房屋连通组 k = 该范围内自举重抽样的聚类数（T2、T3、敏感性表同）：k < {NO_CI} 时不给区间，'
              f'显示“— (k components)”；k < {FEW_CLUSTERS} 标记 few clusters（百分位聚类自举可能覆盖不足）；'
              f'房间 < {SMALL_N} 标记 small n。', '',
              '## T3 按来源与论文六层（GT 参考并列）', '']
    lines += _md(t['T3'], JP_COLUMNS)
    lines += ['## T4 论文式六层', '']
    lines += _md(t['T4'], [('stratum', '来源/房型'), ('system', '系统'), ('complete', '完整'), ('requests', '请求'),
                           ('oor', 'OOR 轴对齐↓'), ('oor_rotated', 'OOR 旋转↓'), ('gt_oor', 'GT OOR 轴对齐'),
                           ('gt_oor_rotated', 'GT OOR 旋转'), ('fid', 'FID↓'), ('kid', 'KID↓'), ('kid_se', 'KID SE'),
                           ('fid_n', 'FID 共同 N'), ('fid_flag', '标记'),
                           ('gpt', 'GPT 三项均值↑'), ('gpt_success', 'GPT 成功'), ('gpt_requested', 'GPT 选定')])
    fid_note = ('FID 与 KID 只在参考及全部渲染系统都成功的房间交集上计算（N 为该交集）；KID 为配对 U 统计量，'
                '不能与文献 KID 比较；代理只渲染 harness 答案。' if result['visual_mode']['fid'] == 'systems' else
                'FID 只在参考、J、P 共同成功渲染的交集上计算；此 J/P 视觉运行不产出 KID，记为未测。')
    lines += ['轴对齐 OOR 为论文定义（忽略朝向）；旋转 OOR 使用检查器已构建的旋转足迹。' + fid_note + 'N<50 标记。'
              'GPT 为 `gpt-6.1-sol` 同一裁判的功能/布局/美观三项房间均值，不能与论文 GPT-4o 分数互换。', '']
    if t['T4_gpt']:
        lines += ['GPT 主终点：J−P 两臂均完成房间的配对均值差，按层内重抽样自举'
                  + ('（N 系统裁判运行自身的 summary.endpoint）。' if result['visual_mode']['judge'] == 'systems' else '。'), '']
        lines += _md(t['T4_gpt'], [('scope', '范围'), ('delta', 'Δ J−P [95% CI]'), ('mean_difference', '分层描述性均值差'), ('n', 'n')])
    lines += ['论文数值（独立协议，仅作背景，不能据此宣称胜负）：', '']
    lines += _md(t['T4_paper'], [('stratum', '来源/房型'), ('FID', 'FID'), ('OOR', 'OOR'), ('functionality', '功能'),
                                 ('layout', '布局'), ('aesthetics', '美观')])
    lines += ['## T5 代理基准（同一冻结队列，同一输入）', '']
    if t['T5']:
        lines += _md(t['T5'], [('scope', '范围'), ('system', '系统'), ('requests', '请求'), ('complete_rate', '完整率'),
                               ('geometry_success_rate', '(a) 几何成功率'), ('size_log_mae', '(b) log MAE（完整输出）'),
                               ('size_iou', '尺寸 IoU'), ('oor', '原始 OOR'),
                               ('bias_fp', 'B 足迹 [95% CI]（EXPLORATORY）'), ('bias_vol', 'B 体积 [95% CI]（EXPLORATORY）'),
                               ('scale_factor', 'exp(−B/2)（全队列）'),
                               ('success_debiased_rate', "(a′) 自身 B 去偏成功率（EXPLORATORY）")])
        lines += ['(b) log MAE、尺寸 IoU 与 OOR 为各系统在自身完整输出上的均值（各系统房间集合不同）；'
                  '配对 Δ(b) 只用两系统共同完整的房间。'
                  'B 为各系统在本范围内全部系统共同完整房间上的 log(预测/GT)（足迹 x·y、体积），房屋连通组自举区间。'
                  "(a′) 用全队列共同完整房间上的自身 B 将 x/y 尺寸乘以 exp(−B/2)，位置与朝向不变，经同一 _derive 用冻结几何重新评分，"
                  '以全部请求为分母。代理的 B 与 (a′) 为 EXPLORATORY（修正案 005 §4 未预先登记）；'
                  'J／P 此处用队列内自身 B，不同于 T1 预先登记的全测试 B，同属 EXPLORATORY。', '',
                  '配对差只给估计与区间（仅全队列与六层合并），不作“显著”判断；harness 为主比较对象，raw 为单次回答敏感性。', '']
        lines += _md(t['T5_contrasts'], [('scope', '范围'), ('comparison', '比较'), ('geometry_success', 'Δ(a) [95% CI]'),
                                         ('size_log_mae', 'Δ(b) [95% CI]')])
        lines += _md(t['T5_cost'], [('system', '系统'), ('model', '模型'), ('reasoning', '推理强度'), ('calls', '调用／生成请求'),
                                    ('gpu_hours', 'GPU 时（cost.json）'), ('gpu_hours_full_test_generation', 'GPU 时（全测试生成）'),
                                    ('input_tokens', '输入 tokens（队列）'), ('output_tokens', '输出 tokens（队列）'),
                                    ('reasoning_tokens', '推理 tokens'), ('latency_p50_s', '中位延迟 s')])
        lines += ['API 失败计入分母。代理的模型名称：' + '；'.join(f"{k} = {v['llm']['config'].get('model')}"
                                                     for k, v in agent_meta.items()), '']
    else:
        lines += ['未提供代理评测输出：本表未测。', '']
    lines += ['## T6 失败分类', '']
    lines += _md(t['T6'], [('system', '系统'), ('requests', '请求'), ('complete', '完整'), ('incomplete', '未完成'),
                           ('token_cap_without_eos', '达上限无 EOS'), ('cap_rule_flipped_complete', '其中原可评分'),
                           ('invalid_json', 'JSON 无效'), ('invalid_schema_or_geometry', '格式/几何无效'),
                           ('invalid_output', '无效（代理未分 JSON/格式）'), ('generation_or_api_failure', '生成/API 失败'),
                           ('missing_output', '无输出'), ('answers_reaching_cap', '代理达上限回答数')])
    lines += ['## 敏感性分析', '']
    lines += _md(t['S1'], JP_COLUMNS)
    lines += ['按先验层级的物体级尺寸误差（共同完整房间，物体微平均；先验列为全部房间）。'
              'G = 常数全局回退；P = 类别×房型；C = 类别；N = 类别×房型或类别（缺少冻结先验文件，未区分）。', '']
    lines += _md(t['S2'], [('level', '层级'), ('objects', '物体'), ('J_size_log_mae', 'J log MAE'),
                           ('P_size_log_mae', 'P log MAE'), ('prior_size_log_mae', '先验 log MAE（全部房间）'),
                           ('J_size_iou', 'J IoU'), ('P_size_iou', 'P IoU'), ('prior_objects', '先验 物体数')])
    lines += ['低复用来源 = 真实扫描、SceneSmith、SpatialGen；高复用 = 资产尺寸在各切分间重复的合成来源。', '',
              '## 说明', '',
              ROLE[stage] + '自举 p = 2·min(P(d≤0), P(d≥0))。', '',
              '## 警告（results.json）', '',
              f"- 报告级：{', '.join(result['warnings']) or '无'}。"]
    for warning, where in _interval_warnings(t).items():
        lines.append(f'- 区间级 `{warning}`（{INTERVAL_WARNINGS.get(warning, warning)}）：'
                     + '；'.join(f"{name} {', '.join(labels)}" for name, labels in where.items()) + '。')
    lines += ['', 'results.json 保存全部数值、区间和输入哈希；MANIFEST.json 固定本报告的文件哈希。']
    return '\n'.join(lines) + '\n'


def _interval_warnings(tables):
    """Interval warnings results.json carries inside table cells: warning -> table -> row labels."""
    found = collections.defaultdict(dict)
    for name, rows in tables.items():
        for r in rows:
            label = ' '.join(str(r[k]) for k in ('scope', 'endpoint', 'comparison', 'system') if k in r)
            for w in (w for v in r.values() if isinstance(v, dict) for w in v.get('warnings', [])):
                found[w].setdefault(name, {})[label] = None
    return {w: {name: list(labels) for name, labels in where.items()} for w, where in sorted(found.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('data', 'geometry', 'joint', 'placement', 'reference', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ('fid', 'judge', 'corrections'):
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--agent', action='append', default=[], help='NAME=EVAL_LLM_DIR (repeatable)')
    parser.add_argument('--repeats', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    agents = dict(item.split('=', 1) for item in args.agent)
    if len(agents) != len(args.agent) or any(not k or '-' in k for k in agents):
        raise ValueError('agents need unique NAME=DIR with no "-" in NAME')
    result = build(args.data, args.geometry, {arm: getattr(args, arm) for arm in h.ARMS}, args.reference, args.out,
                   agents, args.fid, args.judge, args.corrections, args.repeats, args.workers)
    print(json.dumps({'status': result['status'], 'stage': result['stage'], 'delivered': result['delivery']['delivered'],
                      'binding': result['delivery']['binding']}))


if __name__ == '__main__':
    main()
