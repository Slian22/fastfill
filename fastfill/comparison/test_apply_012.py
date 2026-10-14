"""Amendment 012 §4 test application of S1: raw vs S1 on test. Secondary, EXPLORATORY; confirmatory and delivery
results are unchanged. ("test" = the test split; this module is not a unit test.)

python -m fastfill.comparison.test_apply_012 summarize --data RUN/eval-data-v1 --s1-root RUN/s1-012-test \
    --cohort RUN/llm-cohort-v1.json --decision RUN/decision-012 --out NEW_DIR [--workers N] [--repeats 2000]

Inputs (each MANIFEST-verified, S1_COMPLETE on this data at max_trials 2000 / time_limit 60): S1-ROOT/joint-sft and
S1-ROOT/placement-rpo-dpo1 (output_repair on the J and P rows of one geometry file, every test request, requests
arm joint / placement), and S1-ROOT/agent-NAME-{harness,raw} for every NAME in AGENTS (output_repair --uids
COHORT on the agent's llm_harness / llm_raw geometry rows, requests arm joint: agents received J's input; rows
exactly the cohort uids); all inputs must share S1 params and code hashes. The decision dir must adopt S1 and
neither S2. GT reference = decide_012._gt on every test room (downstream.accept and geometry.geometry on the GT layout).
Per system and stage (raw, S1), every request of the scope as denominator: complete, checker (geometry_success,
frozen checker), downstream (strict_ok = downstream.accept path A), accepted (both); footprint bias B = mean over
the system's own complete rooms (not report-v2's J/P room sets) of report_v2.size_bias (room mean log pred/GT x*y;
S1 never resizes, GT = 0); for S1 rows output_repair._aggregate (status, displacement, rollbacks, timeouts,
anti-gaming counters, invariant bugs).
Scopes: test (J, P, GT) and cohort (everyone, the cohort uids), each overall, per source and per paper stratum
(eval_data.PAPER_STRATA). Contrasts, first minus second, with report_v2.contrasts (geometry.paired_intervals' house-
component design and draws, 2,000, seed 42, room micro, every request): test P+S1 - J+S1; cohort P+S1 - X+S1 and
J+S1 - X+S1 for every agent system X; S1 - raw for every system on each of its scopes; each overall and per paper
stratum; p unadjusted. The amendment-005 cap rule is not re-applied (S1 ran on the scored rows); where rows carry
the flags the number of complete rows it would fail is reported. Outputs (write-once): results.json, levels.csv, contrasts.csv,
report.md (Chinese), MANIFEST.json.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import statistics
import tempfile

from fastfill.comparison import decide_012 as d12
from fastfill.comparison import downstream as dn
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import llm_agent
from fastfill.comparison import output_repair as r
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.comparison.eval_data import PAPER_STRATA
from fastfill.scene import norm_room_type

OURS = {'J': ('joint-sft', 'joint', 'joint-geometry.jsonl'),
        'P': ('placement-rpo-dpo1', 'placement', 'placement-geometry.jsonl')}
AGENTS = ('L', 'O1a', 'O1b', 'O2', 'O3', 'O4', 'O6', 'O7')   # = submit_test_012.sh
VARIANTS = ('harness', 'raw')
FIELDS = ('complete', 'checker', 'downstream', 'accepted')
STAGES = ('raw', 's1')
MAX_TRIALS, TIME_LIMIT, SEED = 2000, 60.0, 42
LABEL = 'amendment-012 §4 test application: secondary, EXPLORATORY; confirmatory and delivery results unchanged'
TAG = 'amendment-012 test application (EXPLORATORY)'  # every CSV row


def _system(root, arm, geometry, data_sha):
    manifest, info = e._verified(root), h.strict_json((root / 'summary.json').read_text())
    if (info.get('status') != 'S1_COMPLETE' or info['inputs']['data_manifest_sha256'] != data_sha
            or info['requests_arm'] != arm or info['inputs']['geometry'] != geometry
            or (info['params']['max_trials'], info['params']['time_limit_s']) != (MAX_TRIALS, TIME_LIMIT)):
        raise ValueError(f'{root}: not S1 of {geometry} (requests arm {arm}) on this data at the protocol settings')
    name = Path(geometry).stem + '-s1.jsonl'
    return h._rows(root, name, manifest['files'][name]), info


def _paper(room):
    key = (room['source'], norm_room_type(room.get('room_type')))
    return '/'.join(key) if key in PAPER_STRATA else None


def _subsets(items, rooms):
    """'all', 'paper:SOURCE/TYPE' and 'source:SOURCE' -> the items in each."""
    out = collections.defaultdict(list)
    for x in items:
        room = rooms[x['uid']]
        paper = _paper(room)
        for key in ('all', 'source:' + room['source'], *(['paper:' + paper] if paper else [])):
            out[key].append(x)
    return {k: out[k] for k in sorted(out, key=lambda k: (k != 'all', k))}


def _views(rows, rooms):
    """{stage: per-room views}; B from the raw layout (S1 never resizes)."""
    out = {stage: [] for stage in STAGES}
    for row in rows:
        room, s1, done = rooms[row['uid']], row['s1'], bool(row['complete'])
        if (row['source'], row['group']) != (room['source'], room['group']):
            raise ValueError('S1 row / test room identity mismatch')
        base = {'uid': row['uid'], 'source': room['source'], 'group': room['group'], 'complete': done,
                'bias_fp': r2.size_bias(room['objects'], row['raw_layout'])[0] if done else None}
        for stage in STAGES:
            out[stage].append({**base, 'checker': bool(s1[f'geometry_success_{stage}']),
                               'downstream': bool(s1[f'strict_ok_{stage}']), 'accepted': bool(s1[f'accepted_{stage}'])})
    return out


def _gt_views(rooms, workers):
    return [{'uid': room['uid'], 'source': room['source'], 'group': room['group'], 'complete': True,
             'checker': x['checker'], 'downstream': x['strict_ok'], 'accepted': x['checker'] and x['strict_ok'],
             'bias_fp': 0.0} for room, x in zip(rooms, r2._map(d12._gt, rooms, workers), strict=True)]


def _level(views):
    n, done = len(views), [v for v in views if v['complete']]
    return {'requests': n, 'complete_rooms': len(done), **{f + '_rate': sum(v[f] for v in views) / n for f in FIELDS},
            'bias_fp': statistics.fmean(v['bias_fp'] for v in done) if done else None}


def _levels(views, rows, rooms, keep):
    """{subset: {stage: level[, 'repair': output_repair._aggregate]}} on the uids in keep."""
    split = {stage: _subsets([v for v in vs if v['uid'] in keep], rooms) for stage, vs in views.items()}
    repaired = _subsets([x for x in rows if x['uid'] in keep], rooms) if rows else {}
    return {k: {**{stage: _level(s[k]) for stage, s in split.items()},
                **({'repair': r._aggregate(repaired[k])} if rows else {})} for k in split['raw']}


def _cap(rows):
    """Complete rows the amendment-005 cap rule would fail (cap hit without EOS); None when no row carries the flags."""
    flagged = [x for x in rows if 'reached_token_cap' in x and 'stopped_on_eos' in x]
    return sum(bool(x['complete'] and x['reached_token_cap'] and not x['stopped_on_eos']) for x in flagged) \
        if flagged else None


def _contrast(task):
    first, second, repeats = task
    return r2.contrasts(first, second, FIELDS, repeats, SEED)


def summarize(data, s1_root, cohort_path, decision, out, workers=1, repeats=2000):
    data, s1_root, cohort_path, decision, out = map(Path, (data, s1_root, cohort_path, decision, out))
    if out.exists():
        raise FileExistsError(out)
    manifest = e._verified(data)
    data_sha, expected = h.sha256(data / 'MANIFEST.json'), manifest['expected_uids']
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    e._verified(decision)
    verdict = h.strict_json((decision / 'decision.json').read_text())
    decided = verdict.get('decisions') or {}
    if verdict.get('status') != 'DECISION_012_COMPLETE' or decided.get('s1_adopted') is not True \
            or any(decided[f's2_{arm}']['passed'] for arm in h.ARMS):
        raise ValueError('the dev decision must adopt S1 and neither S2 (amendment 012 §4)')
    cohort = h.strict_json(cohort_path.read_text())
    uids = cohort['uids']
    if (cohort.get('data_manifest_sha256') != data_sha or llm_agent._digest(json.dumps(uids)) != cohort.get('uids_sha256')
            or len(set(uids)) != len(uids) or not set(uids) <= set(expected)):
        raise ValueError('cohort does not bind this test data or its uid hash')
    systems, infos, caps = {}, {}, {}
    for label, (name, arm, geometry) in OURS.items():
        rows, infos[label] = _system(s1_root / name, arm, geometry, data_sha)
        if len(rows) != len(expected) or {x['uid'] for x in rows} != set(expected):
            raise ValueError(f'{label}: S1 rows must cover every test request')
        systems[label] = (_views(rows, rooms), rows)
    if infos['J']['inputs']['geometry_manifest_sha256'] != infos['P']['inputs']['geometry_manifest_sha256']:
        raise ValueError('J and P S1 outputs repaired different geometry files')
    names = list(AGENTS)
    agents = [f'{n}-{v}' for n in names for v in VARIANTS]
    for name in names:
        for variant in VARIANTS:
            label = f'{name}-{variant}'
            rows, infos[label] = _system(s1_root / f'agent-{label}', 'joint', f'llm_{variant}-geometry.jsonl', data_sha)
            if len(rows) != len(uids) or {x['uid'] for x in rows} != set(uids):
                raise ValueError(f'{label}: S1 rows must be exactly the cohort uids')
            systems[label] = (_views(rows, rooms), rows)
        if infos[f'{name}-harness']['inputs']['geometry_manifest_sha256'] != \
                infos[f'{name}-raw']['inputs']['geometry_manifest_sha256']:
            raise ValueError(f'{name}: harness and raw S1 outputs repaired different agent evaluations')
    if any(i['params'] != infos['J']['params'] or i['code_sha256'] != infos['J']['code_sha256'] for i in infos.values()):
        raise ValueError('S1 inputs ran with different parameters or code')
    for label, (_, rows) in systems.items():
        caps[label] = _cap(rows)
    systems['GT'] = ({'raw': _gt_views([rooms[u] for u in expected], workers)}, None)
    scopes = {'test': (expected, ['J', 'P', 'GT']), 'cohort': (uids, ['J', 'P', 'GT', *agents])}
    levels = {scope: {label: _levels(*systems[label], rooms, set(keep)) for label in labels}
              for scope, (keep, labels) in scopes.items()}
    specs = [('test', 'P+S1 - J+S1', ('P', 's1'), ('J', 's1'))]
    specs += [('cohort', f'{a}+S1 - {x}+S1', (a, 's1'), (x, 's1')) for x in agents for a in ('P', 'J')]
    specs += [(scope, f'{s}+S1 - {s}', (s, 's1'), (s, 'raw'))
              for scope, labels in (('test', ('J', 'P')), ('cohort', ('J', 'P', *agents))) for s in labels]
    tasks, meta = [], []
    for scope, name, (a, sa), (b, sb) in specs:
        keep = set(scopes[scope][0])
        first, second = (_subsets([v for v in systems[x][0][s] if v['uid'] in keep], rooms) for x, s in ((a, sa), (b, sb)))
        for subset in (k for k in first if k == 'all' or k.startswith('paper:')):
            tasks.append((first[subset], second[subset], repeats))
            meta.append({'scope': scope, 'comparison': name, 'subset': subset, 'rooms': len(first[subset]),
                         'components': len({v['group'] for v in first[subset]}),
                         'bias_fp': [_level(first[subset])['bias_fp'], _level(second[subset])['bias_fp']]})
    if workers == 1:
        fields = [_contrast(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as pool:
            fields = list(pool.map(_contrast, tasks))
    contrasts = [{**m, 'fields': f} for m, f in zip(meta, fields)]
    result = {'status': 'APPLY_012_TEST_COMPLETE', 'label': LABEL, 'protocol': 'amendment-012 §4',
              'settings': {'max_trials': MAX_TRIALS, 'time_limit_s': TIME_LIMIT, 'best_of_n': 'not applied',
                           'repeats': repeats, 'seed': SEED},
              'decision': {'s1_adopted': True, 's2_passed': {arm: decided[f's2_{arm}']['passed'] for arm in h.ARMS}},
              'agents': names, 'scopes': {k: {'rooms': len(v[0]), 'systems': v[1]} for k, v in scopes.items()},
              'levels': levels, 'contrasts': contrasts, 'cap_rule_would_fail_complete': caps,
              'metric_contract': {
                  'rate_denominator': 'every request of the scope (incomplete = failure)',
                  'checker': 'geometry_success (frozen checker)', 'downstream': 's1 strict_ok = downstream.accept path A',
                  'accepted': 'checker and downstream', 'bias_fp': "mean over the system's own complete rooms of the "
                  'subset (n = complete_rooms) of report_v2.size_bias footprint (log pred/GT x*y); S1 never resizes; '
                  'GT = 0. Not the report-v2 room sets (T1_bias: J and P complete; T5: J, P and system complete), so '
                  'values differ slightly from the published B',
                  'contrasts': 'report_v2.contrasts = geometry.paired_intervals design and draws; first minus second',
                  'p': 'unadjusted for the number of contrasts (exploratory)',
                  'cap_rule': 'not re-applied; counts reported where rows carry the flags (J and P carry them in the '
                              'inference records, not in geometry-v5: report-v2 selected T6 cap_rule_flipped_complete '
                              '= 0 for both on test)'},
              'inputs': {'data_MANIFEST': data_sha, 'cohort': h.sha256(cohort_path),
                         'decision_MANIFEST': h.sha256(decision / 'MANIFEST.json'),
                         **{label: {'MANIFEST': h.sha256(s1_root / (OURS[label][0] if label in OURS else f'agent-{label}')
                                                         / 'MANIFEST.json'),
                                    'geometry': i['inputs']['geometry'],
                                    'geometry_MANIFEST': i['inputs']['geometry_manifest_sha256']}
                            for label, i in infos.items()}},
              'code_sha256': {p.name: h.sha256(p) for p in map(Path, (__file__, r.__file__, d12.__file__, dn.__file__,
                                                                      g.__file__, r2.__file__))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'results.json', result)
        _csv(dest, result)
        (dest / 'report.md').write_text(_report(result))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return result


# ---- output ---------------------------------------------------------------------------------------------------

def _csv(dest, result):
    with (dest / 'levels.csv').open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['label', 'scope', 'subset', 'system', 'stage', 'requests', 'complete_rooms',
                    *(f + '_rate' for f in FIELDS), 'bias_fp', 'moved_median_m', 'moved_p95_m', 'moved_fraction', 'over_0_5m_fraction', 'timed_out',
                    'rolled_back'])
        for scope, systems in result['levels'].items():
            for system, subsets in systems.items():
                for subset, v in subsets.items():
                    for stage in (s for s in STAGES if s in v):
                        rep = v.get('repair') if stage == 's1' else None
                        disp = (rep or {}).get('displacement') or {}
                        w.writerow([TAG, scope, subset, system, stage, v[stage]['requests'],
                                    v[stage]['complete_rooms'], *(v[stage][f + '_rate'] for f in FIELDS),
                                    v[stage]['bias_fp'],
                                    *(disp.get(k) for k in ('moved_median', 'moved_p95', 'moved_fraction',
                                                            'over_0_5m_fraction')),
                                    rep and rep['timed_out'], rep and sum(rep['rolled_back'].values())])
    with (dest / 'contrasts.csv').open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['label', 'scope', 'comparison', 'subset', 'rooms', 'components', 'bias_fp_first',
                    'bias_fp_second', 'field', 'estimate', 'ci_low', 'ci_high', 'p_unadjusted', 'n', 'warnings'])
        for c in result['contrasts']:
            for field, x in c['fields'].items():
                w.writerow([TAG, c['scope'], c['comparison'], c['subset'], c['rooms'], c['components'],
                            *c['bias_fp'], field, x['estimate'], *(x['ci95'] or (None, None)), x['p'], x['n'], ';'.join(x['warnings'])])


def _f(value, signed=False):
    return '—' if value is None else (f'{value:+.3f}' if signed else f'{value:.3f}')


def _ci(x):
    return _f(x['estimate'], True) + (f" [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" if x['ci95'] else ' [CI —]')


def _arrow(v, key):
    return f"{_f(v['raw'][key])} → {_f(v['s1'][key])}" if 's1' in v else _f(v['raw'][key])


SCOPE = {'test': '全测试集', 'cohort': '代理队列'}


def _report(result):
    s = result['scopes']
    caps = {k: v for k, v in result['cap_rule_would_fail_complete'].items() if v is not None}
    lines = ['# 修订 012 §4 测试集应用：S1 输出修复（次要结果 / EXPLORATORY）', '',
             '本报告是修订 012 §4 在测试集上的一次性应用，属次要、探索性（EXPLORATORY）结果；修订 005–011 的确认性结果与'
             '交付结果都不变。dev 决定（decision-012）：S1 采纳，S2（最佳 N 选一）不采纳；因此测试集上只应用 S1，设定同 dev'
             f"（max_trials {result['settings']['max_trials']}，time_limit {result['settings']['time_limit_s']:g} s），"
             '不做最佳 N 选一。', '',
             f"J = SFT，P = rpo-dpo1，各自在全部 {s['test']['rooms']} 个测试房间上；每个代理（"
             f"{'、'.join(result['agents']) or '无'}）在其 {s['cohort']['rooms']} 房间队列上的既有输出（harness 与 raw）"
             '都经过同一个 S1。代理收到的是 J 的输入，因此按 J 的请求修复；开放模型与 API 代理得到相同的修复。', '',
             '完整率；检查器成功 = geometry_success（冻结检查器）；下游接受 = downstream.accept（EmbodiedGen path A 严格规则'
             '的移植，每个门入口）；两者都过 = 同时满足。比率分母为该范围的全部请求，未完成输出计为失败。', '',
             'B = 足迹尺寸偏差：该系统自身的完整房间上（n = levels.csv 的 complete_rooms）每个房间 log(预测 / GT 的 x·y 面积)'
             ' 的均值，再对房间平均。房间集合不同于 report-v2 的 T1_bias（J 与 P 都完整）与 T5（J、P 与该系统都完整），所以'
             '数值与已发表的 B 略有差异。S1 不改尺寸，所以 raw 与 S1 的 B 相同。B < 0 表示物体偏小（代理普遍偏小）；偏小的物体更容易通过碰撞与边界检查，所以每个接受率都要与 B'
             ' 一起读。GT = 测试房间的真值布局本身（同一检查器与 downstream.accept），只作参考，B = 0。', '',
             f"配对差为前者减后者；房屋连通组自举（geometry.paired_intervals 的设计与抽样，{result['settings']['repeats']} "
             f"次，seed {result['settings']['seed']}，房间微平均，全部请求）。修订 005 的截断规则未在此重新应用；行内带标志时，"
             f"它会判为失败的完整行数：{caps or '无（没有行带标志）'}。J 与 P 的标志在推理记录中、不在 geometry-v5 行内，"
             '因此此处不计；report-v2（selected）T6 中 J 与 P 在测试集上的 cap_rule_flipped_complete 均为 0。p 值未按比较数'
             '校正（contrasts.csv 的 p_unadjusted）。', '']
    for k, (scope, name) in enumerate(SCOPE.items(), 1):
        lines += [f'## 表 {k}：{name}（{s[scope]["rooms"]} 个房间）', '',
                  '| 系统 | 请求 | 完整率 | 检查器成功 raw → S1 | 下游接受 raw → S1 | 两者都过 raw → S1 | B | '
                  '被移动物体位移中位 / p95 (m) | 被移动物体占比 | 超时 | 回滚 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for system, subsets in result['levels'][scope].items():
            v = subsets['all']
            rep = v.get('repair') or {}
            disp = rep.get('displacement') or {}
            lines.append(f"| {system} | {v['raw']['requests']} | {_f(v['raw']['complete_rate'])} | "
                         + ' | '.join(_arrow(v, f + '_rate') for f in FIELDS[1:]) + f" | {_f(v['raw']['bias_fp'], True)} | "
                         f"{_f(disp.get('moved_median'))} / {_f(disp.get('moved_p95'))} | {_f(disp.get('moved_fraction'))} | "
                         f"{rep.get('timed_out', '—')} | {sum(rep['rolled_back'].values()) if rep else '—'} |")
        lines.append('')
    lines += ['## 表 3：配对差（前者减后者，95% CI）', '',
              '| 比较 | 范围 | 房间 | 房屋组 | Δ下游接受 | Δ两者都过 | Δ检查器成功 | Δ完整率 | B 前者 / 后者 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in (c for c in result['contrasts'] if c['subset'] == 'all'):
        x = c['fields']
        lines.append(f"| {c['comparison']} | {SCOPE[c['scope']]} | {c['rooms']} | {c['components']} | "
                     + ' | '.join(_ci(x[f]) for f in ('downstream', 'accepted', 'checker', 'complete'))
                     + f" | {_f(c['bias_fp'][0], True)} / {_f(c['bias_fp'][1], True)} |")
    lines += ['', '各论文分层的配对差见 contrasts.csv（subset = paper:来源/房型）。', '']
    for k, (prefix, title) in enumerate((('paper:', '论文分层'), ('source:', '来源')), 4):
        lines += [f'## 表 {k}：按{title}', '',
                  f'| 范围 | {title} | 系统 | 请求 | 下游接受 raw → S1 | 两者都过 raw → S1 | 检查器成功 raw → S1 | B |',
                  '|---|---|---|---:|---:|---:|---:|---:|']
        for scope, systems in result['levels'].items():
            for subset in (x for x in systems['GT'] if x.startswith(prefix)):
                for system, subsets in systems.items():
                    v = subsets.get(subset)
                    if v:
                        lines.append(f"| {SCOPE[scope]} | {subset.removeprefix(prefix)} | {system} | {v['raw']['requests']} | "
                                     + ' | '.join(_arrow(v, f + '_rate') for f in ('downstream', 'accepted', 'checker'))
                                     + f" | {_f(v['raw']['bias_fp'], True)} |")
        lines.append('')
    shown = ('contained', 'tuck', 'thin', 'flat_head', 'same_category_parent')
    lines += ['## 表 6：S1 状态与反作弊计数（各系统自身的范围，全部房间）', '',
              '| 系统 | 范围 | 状态 | 回滚 | 超时 / 预算耗尽 | ' + ' | '.join(f'{k} raw → S1' for k in shown)
              + ' | 不变量改变（须为 0） |', '|---|---|---|---|---:|' + '---:|' * len(shown) + '---|']
    for system in ['J', 'P'] + [x for x in result['levels']['cohort'] if x not in ('J', 'P', 'GT')]:
        scope = 'test' if system in result['levels']['test'] else 'cohort'
        rep = result['levels'][scope][system]['all']['repair']
        lines.append(f"| {system} | {SCOPE[scope]} | {rep['status']} | {rep['rolled_back'] or 0} | "
                     f"{rep['timed_out']} / {rep['budget_exhausted']} | "
                     + ' | '.join(f"{rep['anti_gaming']['raw'][k]} → {rep['anti_gaming']['s1'][k]}" for k in shown)
                     + f" | {rep['bugs'] or '全部为 0'} |")
    lines += ['', '完整数字（含每个分层与来源、位移分布、P4／P5 宽松入口与原始失败类型拆分）见 results.json、levels.csv 与 '
              'contrasts.csv。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('summarize')
    for flag in ('data', 's1-root', 'cohort', 'decision', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--repeats', type=int, default=2000)
    a = parser.parse_args()
    result = summarize(a.data, a.s1_root, a.cohort, a.decision, a.out, a.workers, a.repeats)
    print(json.dumps({'status': result['status'], 'agents': result['agents'], 'scopes': result['scopes']}))


if __name__ == '__main__':
    main()
