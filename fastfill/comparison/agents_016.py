"""Amendment 016 §2 (size-fixed regeneration; EXPLORATORY): the pilot subsample (addendum A6 / B1) and the ranking of
P and the open agents with their own sizes (J input) vs P's sizes (P input), addendum A1. Amendments 005-015 results
and decisions are unchanged.

python -m fastfill.comparison.agents_016 pilot --data RUN/eval-data-v1 --cohort RUN/llm-cohort-v1.json \
    --out RUN/a016-pilot-v1.json
python -m fastfill.comparison.agents_016 compare --data RUN/eval-data-v1 --cohort RUN/llm-cohort-v1.json \
    --uids RUN/a016-pilot-v1.json|RUN/llm-cohort-v1.json --s1-012 RUN/s1-012-test \
    --fixed O2=RUN/s1-016-size-fixed/agent-O2-P [--fixed NAME=PREFIX ...] [--extra O2-PGT=PREFIX ...] \
    [--optiscene S1_DIR] --out NEW_DIR [--repeats 2000]

pilot: every cohort uid in a scale_curve.BINS object-count bin (count = objects in its joint request); PILOT_SIZE
rooms allocated to the bins in proportion to their cohort counts (largest remainder, ties in bin order; PROTOCOL GAP:
the amendment says "stratified" without the allocation, so proportional is this tool's reading and must be locked in
the protocol before any output), each bin's quota taken by ascending sha256('a016-pilot:' + uid) (uid tie-break);
written once in cohort order with its counts and hashes, and an existing file must equal its rebuild. Addendum B1:
it is also L's size-fixed subsample. llm_open / llm_agent / evaluate_llm / output_repair take it as --uids.

compare: every input is an output_repair (S1) dir at the amendment-012 settings (test_apply_012._system: S1_COMPLETE on
this data, max_trials 2000, time_limit 60), all with the same S1 params and code. P = S1-012/placement-rpo-dpo1
(requests arm placement). Per NAME in --fixed and variant V in (harness, raw): size-free = S1-012/agent-NAME-V
(amendment-012 test application: J input, own sizes; requests arm joint), size-fixed = PREFIX-V (output_repair on
llm-eval-NAME-P-v1/llm_V-geometry.jsonl, requests arm placement). --extra NAME=PREFIX (size-fixed, e.g. O2 with GT
sizes) are levels only (not ranked). Rows are restricted to the --uids scope (llm_agent.subset; the cohort file itself
= the whole cohort) and must cover it. Per scope subset (scale_curve.subsets: all, object-count bins, real / synthetic,
room families), variant, system and stage (raw, s1): complete / checker / downstream / accepted rates
(test_apply_012._views) with B; size-free minus size-fixed per NAME; the rankings by downstream acceptance (RANK) in
each setting; Kendall tau-b between the size-free and size-fixed rankings of {P, every --fixed NAME}; the pre-specified
flips P - O2 and P - L (each setting) and, with --optiscene (S1 of OptiScene E2 in prior-size mode, requests arm
placement), OptiScene - P on its rooms in the scope. Intervals: scale_curve.design / rates / _iv = geometry.
paired_intervals' house-cluster design and draws (2,000, seed 42, room micro, every request as denominator); tau and
every difference are recomputed in each draw; p two-sided as report_v2.contrasts, unadjusted. Addendum B1: tau on
the pilot scope including L, and on the cohort without L (two invocations). Outputs (write-once): results.json,
levels.csv, contrasts.csv, report.md (Chinese), MANIFEST.json.
"""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import infer
from fastfill.comparison import llm_agent
from fastfill.comparison import scale_curve as sc
from fastfill.comparison import test_apply_012 as t12
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

PILOT_KEY, PILOT_SIZE = 'a016-pilot:', 300
FIELDS, STAGES, VARIANTS = t12.FIELDS, t12.STAGES, t12.VARIANTS
RANK = 'downstream'
FLIPS = ('O2', 'L')   # P minus each, in both settings (addendum A1)
LABEL = 'amendment-016 §2 (EXPLORATORY): amendments 005-015 results and decisions unchanged'


# ---- pilot -----------------------------------------------------------------------------------------------------

def build_pilot(data, cohort):
    rows, identity = infer.load_eval(data, 'joint')
    if cohort != llm_agent.build_cohort(data):
        raise ValueError('cohort file differs from its deterministic rebuild')
    count = {r['uid']: len(h.request(infer._schema_record(r))['objects']) for r in rows}
    bins = {name: [u for u in cohort['uids'] if lo <= count[u] and (hi is None or count[u] <= hi)]
            for name, lo, hi in sc.BINS}
    if sum(map(len, bins.values())) != len(cohort['uids']):
        raise ValueError('a cohort room falls outside the object-count bins')
    exact = {k: PILOT_SIZE * len(v) / len(cohort['uids']) for k, v in bins.items()}
    quota = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: -(exact[k] - quota[k]))[:PILOT_SIZE - sum(quota.values())]:
        quota[k] += 1   # Largest remainder; sorted() is stable, so ties keep bin order.
    chosen = {u for k, v in bins.items() for u in sorted(v, key=lambda u: (llm_agent._digest(PILOT_KEY + u), u))[:quota[k]]}
    uids = [u for u in cohort['uids'] if u in chosen]
    return {'version': 1, 'label': LABEL, 'data_manifest_sha256': identity['manifest_sha256'],
            'cohort_uids_sha256': cohort['uids_sha256'],
            'algorithm': f"scale_curve.BINS on the joint request's object count; {PILOT_SIZE} rooms allocated in "
                         'proportion to the cohort bin counts (largest remainder, ties in bin order); per bin ascending '
                         f"sha256('{PILOT_KEY}'+uid), uid tie-break; cohort order",
            'counts': {k: {'cohort': len(v), 'pilot': quota[k]} for k, v in bins.items()},
            'uids_sha256': llm_agent._digest(json.dumps(uids)), 'uids': uids}


def pilot(data, cohort_path, out):
    value = build_pilot(Path(data), h.strict_json(Path(cohort_path).read_text()))
    out = Path(out)
    if out.exists():
        if h.strict_json(out.read_text()) != value:
            raise ValueError(f'{out} differs from its deterministic rebuild')
    else:
        infer.write_once(out, value)
    return value


# ---- compare ---------------------------------------------------------------------------------------------------

def tau(a, b):
    """Kendall tau-b between the last axes of a and b; nan when undefined (one side all tied)."""
    i, j = np.triu_indices(a.shape[-1], 1)
    x, y = np.sign(a[..., i] - a[..., j]), np.sign(b[..., i] - b[..., j])
    with np.errstate(invalid='ignore', divide='ignore'):
        return (x * y).sum(-1) / np.sqrt(np.count_nonzero(x, -1) * np.count_nonzero(y, -1))


def _diff(point, boot, d, a, b):
    draws = boot[:, a] - boot[:, b]
    defined = draws[np.isfinite(draws)]
    p = min(1., 2 * min((defined <= 0).sum(), (defined >= 0).sum()) / len(defined)) if len(defined) else None
    return {**sc._iv(point[a] - point[b], draws, d), 'p': p}


def _inputs(s1_012, fixed, extra, optiscene, data_sha):
    """{label: (S1 dir, setting, requests arm, geometry name)}; setting P / free / fixed / extra / optiscene."""
    out = {'P': (s1_012 / 'placement-rpo-dpo1', 'P', 'placement', 'placement-geometry.jsonl')}
    for name, prefix in fixed.items():
        for v in VARIANTS:
            out[f'{name}-{v}'] = (s1_012 / f'agent-{name}-{v}', 'free', 'joint', f'llm_{v}-geometry.jsonl')
            out[f'{name}-P-{v}'] = (Path(f'{prefix}-{v}'), 'fixed', 'placement', f'llm_{v}-geometry.jsonl')
    for name, prefix in extra.items():
        for v in VARIANTS:
            out[f'{name}-{v}'] = (Path(f'{prefix}-{v}'), 'extra', 'placement', f'llm_{v}-geometry.jsonl')
    if optiscene:
        geometry = h.strict_json((optiscene / 'summary.json').read_text())['inputs']['geometry']
        out['OptiScene'] = (optiscene, 'optiscene', 'placement', geometry)
    return out


def _estimate(rooms, views, labels, repeats):
    """(design, point, boot, column index) of every label x stage x field on these rooms."""
    cols = [(label, stage, f) for label in labels for stage in STAGES for f in FIELDS]
    matrix = np.array([[views[label][stage][x['uid']][f] for label, stage, f in cols] for x in rooms], dtype=float)
    d = sc.design(rooms, repeats)
    point, boot = sc.rates(d, matrix)
    return d, point, boot, {c: k for k, c in enumerate(cols)}


def _variant(rooms, views, fixed, extra, v, repeats):
    names = list(fixed)
    free, sized = ['P', *(f'{n}-{v}' for n in names)], ['P', *(f'{n}-P-{v}' for n in names)]
    labels = list(dict.fromkeys([*free, *sized, *(f'{n}-{v}' for n in extra)]))
    d, point, boot, col = _estimate(rooms, views, labels, repeats)
    level = {label: {stage: {f: sc._iv(point[col[label, stage, f]], boot[:, col[label, stage, f]], d) for f in FIELDS}
                     | {'bias_fp': t12._level([views[label][stage][x['uid']] for x in rooms])['bias_fp']}
                     for stage in STAGES} for label in labels}
    out = {'levels': level, 'size_free_minus_fixed': {}, 'rankings': {}, 'kendall_tau': {}, 'flips': []}
    for n in names:
        out['size_free_minus_fixed'][n] = {stage: {f: _diff(point, boot, d, col[f'{n}-{v}', stage, f],
                                                            col[f'{n}-P-{v}', stage, f]) for f in FIELDS}
                                           for stage in STAGES}
    for stage in STAGES:
        a, b = ([col[label, stage, RANK] for label in side] for side in (free, sized))
        out['rankings'][stage] = {setting: sorted(([label.split('-')[0], float(point[k])] for label, k in zip(side, ks)),
                                                  key=lambda x: -x[1])
                                  for setting, side, ks in (('size_free', free, a), ('size_fixed', sized, b))}
        out['kendall_tau'][stage] = {**sc._iv(float(tau(point[a], point[b])), tau(boot[:, a], boot[:, b]), d),
                                     'systems': ['P', *names]}
    for n in (n for n in FLIPS if n in names):
        for setting, label in (('size_free', f'{n}-{v}'), ('size_fixed', f'{n}-P-{v}')):
            for stage in STAGES:
                out['flips'].append({'comparison': f'P - {n}', 'setting': setting, 'stage': stage,
                                     'fields': {f: _diff(point, boot, d, col['P', stage, f], col[label, stage, f])
                                                for f in FIELDS}})
    return out


def _optiscene(rooms, views, repeats):
    """OptiScene - P on the OptiScene rooms of `rooms` (prior sizes: size-fixed by construction)."""
    mine = [x for x in rooms if x['uid'] in views['OptiScene']['raw']]
    if not mine:
        return None
    d, point, boot, col = _estimate(mine, views, ['OptiScene', 'P'], repeats)
    return {'rooms': len(mine), 'components': d['components'],
            'flips': [{'comparison': 'OptiScene - P', 'setting': 'size_fixed', 'stage': stage,
                       'fields': {f: _diff(point, boot, d, col['OptiScene', stage, f], col['P', stage, f])
                                  for f in FIELDS}} for stage in STAGES]}


def compare(data, cohort_path, uids_path, s1_012, fixed, out, extra=None, optiscene=None, repeats=2000):
    data, cohort_path, uids_path, s1_012, out = map(Path, (data, cohort_path, uids_path, s1_012, out))
    extra, optiscene = extra or {}, optiscene and Path(optiscene)
    if out.exists():
        raise FileExistsError(out)
    if not fixed:
        raise ValueError('rank at least one --fixed agent')
    manifest = e._verified(data)
    data_sha = h.sha256(data / 'MANIFEST.json')
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    cohort = h.strict_json(cohort_path.read_text())
    if (cohort.get('data_manifest_sha256') != data_sha or cohort.get('uids_sha256') != llm_agent._digest(json.dumps(cohort['uids']))
            or not set(cohort['uids']) <= set(manifest['expected_uids'])):
        raise ValueError('cohort does not bind this test data or its uid hash')
    scope = llm_agent.subset(cohort, uids_path)
    keep, inputs, infos, views = set(scope), _inputs(s1_012, fixed, extra, optiscene, data_sha), {}, {}
    for label, (root, setting, arm, geometry) in inputs.items():
        rows, infos[label] = t12._system(root, arm, geometry, data_sha)
        kept = [x for x in rows if x['uid'] in keep]
        if setting != 'optiscene' and sorted(x['uid'] for x in kept) != sorted(scope):
            raise ValueError(f'{label}: S1 rows do not cover the --uids scope')
        views[label] = {stage: {x['uid']: x for x in vs} for stage, vs in t12._views(kept, rooms).items()}
    if any(i['params'] != infos['P']['params'] or i['code_sha256'] != infos['P']['code_sha256'] for i in infos.values()):
        raise ValueError('S1 inputs ran with different parameters or code')
    subsets = {}
    for name, members in sc.subsets([rooms[u] for u in scope]).items():
        subsets[name] = {'rooms': len(members), 'components': len({x['group'] for x in members}),
                         **{v: _variant(members, views, fixed, extra, v, repeats) for v in VARIANTS},
                         **({'optiscene': _optiscene(members, views, repeats)} if optiscene else {})}
    result = {'status': 'A016_AGENTS_COMPLETE', 'label': LABEL, 'protocol': 'amendment-016 §2, addenda A1/A6/B1',
              'scope': {'rooms': len(scope), 'cohort_rooms': len(cohort['uids']), 'uids_sha256': llm_agent._digest(json.dumps(scope)),
                        'uids_file_sha256': h.sha256(uids_path), 'is_cohort': scope == cohort['uids']},
              'systems': {'P': 'rpo-dpo1 (prior sizes in both settings)', 'ranked': ['P', *fixed],
                          'fixed': {n: str(p) for n, p in fixed.items()}, 'extra': {n: str(p) for n, p in extra.items()},
                          'optiscene': str(optiscene) if optiscene else None},
              'settings': {'rank_by': RANK, 'repeats': repeats, 'seed': sc.SEED, 'variants': VARIANTS, 'stages': STAGES},
              'subsets': subsets,
              'metric_contract': {
                  'rates': 'test_apply_012._views of the S1 rows: every request of the scope as denominator; downstream '
                           '= s1 strict_ok (downstream.accept path A); accepted = checker and downstream',
                  'intervals': 'scale_curve.design/rates/_iv: geometry.paired_intervals house-cluster design and draws; '
                               'no CI below report_v2.NO_CI components',
                  'kendall_tau': 'tau-b of the size-free vs size-fixed rate vectors of the ranked systems, per draw; '
                                 'P has the same rows in both settings',
                  'p': 'two-sided bootstrap sign share as report_v2.contrasts, unadjusted (exploratory)'},
              'inputs': {label: {'dir': str(root), 'MANIFEST': h.sha256(root / 'MANIFEST.json'), 'setting': setting,
                                 'requests_arm': arm, 'geometry': geometry,
                                 'geometry_MANIFEST': infos[label]['inputs']['geometry_manifest_sha256']}
                         for label, (root, setting, arm, geometry) in inputs.items()},
              's1_params': infos['P']['params'], 's1_code_sha256': infos['P']['code_sha256'],
              'code_sha256': {p.name: h.sha256(p) for p in map(Path, (__file__, t12.__file__, sc.__file__,
                                                                      llm_agent.__file__))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'results.json', result)
        _csv(dest, result)
        (dest / 'report.md').write_text(_report(result))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return result


# ---- output ----------------------------------------------------------------------------------------------------

def _ci(x, k):
    return None if x['ci95'] is None else x['ci95'][k]


def _flips(s):
    """(variant, flip) of one subset: P - O2 / P - L per variant, then OptiScene - P."""
    yield from ((v, flip) for v in VARIANTS for flip in s[v]['flips'])
    yield from (('', flip) for flip in (s.get('optiscene') or {}).get('flips', []))


def _contrasts(result):
    for subset, s in result['subsets'].items():
        for v in VARIANTS:
            for n, stages in s[v]['size_free_minus_fixed'].items():
                for stage, fields in stages.items():
                    yield subset, v, f'{n} size-free - size-fixed', '', stage, fields
        for v, flip in _flips(s):
            yield subset, v, flip['comparison'], flip['setting'], flip['stage'], flip['fields']


def _csv(dest, result):
    with (dest / 'levels.csv').open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['label', 'subset', 'variant', 'system', 'stage', 'field', 'rate', 'ci_low', 'ci_high', 'bias_fp'])
        for subset, s in result['subsets'].items():
            for v in VARIANTS:
                for system, stages in s[v]['levels'].items():
                    for stage, fields in stages.items():
                        for f in FIELDS:
                            w.writerow([LABEL, subset, v, system, stage, f, fields[f]['estimate'], _ci(fields[f], 0),
                                        _ci(fields[f], 1), fields['bias_fp']])
    with (dest / 'contrasts.csv').open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['label', 'subset', 'variant', 'comparison', 'setting', 'stage', 'field', 'estimate', 'ci_low',
                    'ci_high', 'p_unadjusted', 'warnings'])
        for subset, v, name, setting, stage, fields in _contrasts(result):
            for f, x in fields.items():
                w.writerow([LABEL, subset, v, name, setting, stage, f, x['estimate'], _ci(x, 0), _ci(x, 1), x['p'],
                            ';'.join(x['warnings'])])


def _f(value, signed=False):
    return '—' if value is None else (f'{value:+.3f}' if signed else f'{value:.3f}')


def _iv_text(x):
    return _f(x['estimate'], True) + (f" [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" if x['ci95'] else ' [CI —]')


SETTING = {'size_free': '自选尺寸（J 输入）', 'size_fixed': '固定尺寸（P 输入）', '': ''}


def _report(result):
    sc_ = result['scope']
    scope = (f"全队列 {sc_['rooms']} 间" if sc_['is_cohort'] else
             f"队列子集 {sc_['rooms']} 间（取自 {sc_['cohort_rooms']} 间队列；如修订 016 试点子样本：试点数字只是早期读数，"
             '不替代全队列结果）')
    lines = ['# 修订 016 §2：固定尺寸重生成下的排名（EXPLORATORY）', '',
             '探索性结果；修订 005–015 的结果与决定都不变。同一组房间上比较两种设置：自选尺寸 = 代理收到 J 的输入并自己给尺寸'
             '（修订 012 测试集应用的既有输出）；固定尺寸 = 代理收到 P 的输入（训练集先验尺寸），只摆放，评分前回答里的尺寸'
             '被请求尺寸覆盖（附录 B2）。P（rpo-dpo1）在两种设置下是同一组输出。raw 与 S1（修订 012 的输出修复，同一设定）都报告。',
             '',
             f"范围：{scope}。排名指标 = 下游接受（downstream.accept path A，分母为全部请求）。Kendall τ-b 比较两种设置下"
             f"{'、'.join(result['systems']['ranked'])} 的排名；区间为房屋连通组自举（{result['settings']['repeats']} 次，"
             f"seed {result['settings']['seed']}，与 geometry.paired_intervals 相同的设计与抽样），τ 与每个差值都在每次抽样中重算；"
             'p 值未校正。附录 B1：τ 在试点 300 间（含 L）与全队列 1,454 间（不含 L）各算一次，是两次调用。', '']
    s = result['subsets']['all']
    for v in VARIANTS:
        lines += [f'## 排名与 Kendall τ（{v}）', '',
                  '| 阶段 | 自选尺寸排名（下游接受） | 固定尺寸排名（下游接受） | τ-b [95% CI] |', '|---|---|---|---|']
        for stage in STAGES:
            r = s[v]['rankings'][stage]
            fmt = lambda side: ' > '.join(f'{n} {_f(x)}' for n, x in r[side])
            lines.append(f"| {stage} | {fmt('size_free')} | {fmt('size_fixed')} | {_iv_text(s[v]['kendall_tau'][stage])} |")
        lines.append('')
    lines += ['## 每个代理：自选尺寸 − 固定尺寸（全部范围，95% CI）', '',
              '| 代理 | 变体 | 阶段 | Δ下游接受 | Δ两者都过 | Δ检查器成功 | Δ完整率 | B 自选 / 固定 |', '|---|---|---|---|---|---|---|---|']
    for v in VARIANTS:
        for n, stages in s[v]['size_free_minus_fixed'].items():
            for stage, x in stages.items():
                b = (s[v]['levels'][f'{n}-{v}'][stage]['bias_fp'], s[v]['levels'][f'{n}-P-{v}'][stage]['bias_fp'])
                lines.append(f'| {n} | {v} | {stage} | ' + ' | '.join(_iv_text(x[f]) for f in ('downstream', 'accepted',
                                                                                              'checker', 'complete'))
                             + f' | {_f(b[0], True)} / {_f(b[1], True)} |')
    lines += ['', '## 预先指定的翻转（前者减后者，95% CI）', '',
              '| 比较 | 变体 | 设置 | 阶段 | Δ下游接受 | Δ两者都过 | Δ检查器成功 |', '|---|---|---|---|---|---|---|']
    for v, flip in _flips(s):
        lines.append(f"| {flip['comparison']} | {v or '—'} | {SETTING[flip['setting']]} | {flip['stage']} | "
                     + ' | '.join(_iv_text(flip['fields'][f]) for f in ('downstream', 'accepted', 'checker')) + ' |')
    os_ = s.get('optiscene')
    lines += ['', ('OptiScene − P 只在 OptiScene E2（先验尺寸）房间与本范围的交集上计算（'
                   f"{os_['rooms']} 间）。" if os_ else '未提供 OptiScene E2 的 S1 输出：OptiScene − P 未计算。')
              + 'L 只在试点子样本上重跑（附录 A2/B1），因此 P − L 只出现在试点范围。', '',
              '## 分层（附录 A §5：物体数分箱、真实／合成、房型族；描述性）', '',
              '| 分层 | 房间 | 房屋组 | 变体 | 阶段 | τ-b | ' + ' | '.join(f'{n} Δ下游（自选−固定）' for n in result['systems']['fixed'])
              + ' |', '|---|---:|---:|---|---|---|' + '---|' * len(result['systems']['fixed'])]
    for subset, x in result['subsets'].items():
        for v in VARIANTS:
            for stage in STAGES:
                lines.append(f"| {subset} | {x['rooms']} | {x['components']} | {v} | {stage} | "
                             f"{_iv_text(x[v]['kendall_tau'][stage])} | "
                             + ' | '.join(_iv_text(x[v]['size_free_minus_fixed'][n][stage][RANK])
                                          for n in result['systems']['fixed']) + ' |')
    lines += ['', '各系统各阶段的比率与区间见 levels.csv；全部差值（含各分层）见 contrasts.csv。']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    def pairs(items):
        out = {}
        for item in items or ():
            name, _, path = item.partition('=')
            if not name or not path or name in out:
                raise argparse.ArgumentTypeError(f'need unique NAME=PATH: {item}')
            out[name] = Path(path)
        return out
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pilot')
    for flag in ('data', 'cohort', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    c = sub.add_parser('compare')
    for flag in ('data', 'cohort', 'uids', 's1-012', 'out'):
        c.add_argument('--' + flag, type=Path, required=True)
    c.add_argument('--fixed', action='append', required=True)
    c.add_argument('--extra', action='append')
    c.add_argument('--optiscene', type=Path)
    c.add_argument('--repeats', type=int, default=2000)
    a = parser.parse_args(argv)
    if a.command == 'pilot':
        value = pilot(a.data, a.cohort, a.out)
        print(json.dumps({'rooms': len(value['uids']), 'counts': value['counts'], 'uids_sha256': value['uids_sha256']}))
        return 0
    result = compare(a.data, a.cohort, a.uids, a.s1_012, pairs(a.fixed), a.out, pairs(a.extra), a.optiscene, a.repeats)
    print(json.dumps({'status': result['status'], 'scope': result['scope'],
                      'kendall_tau': {v: {k: x['estimate'] for k, x in result['subsets']['all'][v]['kendall_tau'].items()}
                                      for v in VARIANTS}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
