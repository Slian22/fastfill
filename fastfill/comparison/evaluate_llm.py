"""Score the LLM-agent arm L on its frozen cohort and pair it with J and P (amendment 002).

python -m fastfill.comparison.evaluate_llm --data EVAL --cohort COHORT.json --llm LLM_RUN \
    [--joint MERGED_J --placement MERGED_P] --out NEW_DIR [--repeats 2000] [--workers 1]

Both L variants (llm_raw, llm_harness) are scored by evaluate_pair._score with
joint semantics (predicted sizes) on the J request the model received. J/P come
from verified merged inference dirs (evaluate_pair._inference), restricted to
the cohort. Per-arm tables use geometry.summarize, the six paper strata use the
evaluate_pair rule, and paired differences J-L and P-L (both L variants, pooled
and per stratum) use geometry.paired_intervals (house-cluster bootstrap). In
paired output, estimate_joint_minus_placement means FIRST minus SECOND arm of
the comparison key. Without --joint/--placement only L tables are written.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile

from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer
from fastfill.comparison import llm_agent
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.comparison.eval_data import PAPER_STRATA
from fastfill.scene import norm_room_type

ORDER = ('joint', 'placement', *llm_agent.VARIANTS)
LABELS = {'joint': 'J', 'placement': 'P（尺寸为训练集先验）', 'llm_raw': 'L-raw', 'llm_harness': 'L-harness'}
PAIRED_METRICS = ('complete', 'geometry_success', 'oor', 'size_log_mae', 'size_iou')


def _llm(root, data, cohort, cohort_sha256):
    manifest = e._verified(root)
    if manifest.get('expected_uids') != cohort['uids']:
        raise ValueError('LLM run does not cover exactly the frozen cohort')
    result = h.strict_json((root / 'result.json').read_text())
    if result.get('status') != 'LLM_COMPLETE':
        raise ValueError('LLM run is not complete')
    receipt = h.strict_json((root / 'run_receipt.json').read_text())
    if infer._hash({k: v for k, v in receipt.items() if k != 'run_sha256'}) != receipt.get('run_sha256'):
        raise ValueError('LLM run identity hash mismatch')
    if receipt['data']['manifest_sha256'] != h.sha256(data / 'MANIFEST.json') or receipt['cohort']['sha256'] != cohort_sha256:
        raise ValueError('LLM run data/cohort identity mismatch')
    records = {}
    for variant in llm_agent.VARIANTS:
        name = variant + '/records.jsonl'
        rows = h._rows(root, name, manifest['files'][name])
        if ([r['uid'] for r in rows] != cohort['uids']
                or any(r['run_sha256'] != receipt['run_sha256'] or r['arm'] != variant for r in rows)):
            raise ValueError('LLM records do not match run identity/cohort')
        records[variant] = rows
    return records, receipt, result


def _score_llm(task):
    room, request, record = task
    return {**e._score((room, request, {**record, 'arm': 'joint'}, 'joint')), 'arm': record['arm']}


def _ci(item):
    value = item['estimate_joint_minus_placement']
    if value is None:
        return '—'
    ci = item['ci95']
    return f'{value:+.4f}' + (f' [{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else ' [CI —]')


def _report(summary):
    lines = ['# 未知尺寸：LLM 代理 L 与 J／P 同输入对比', '',
             'L 收到与 J 完全相同的 system 与 user 消息，另加一条对所有请求相同的“只输出 JSON”后缀，由 J 的解析器判定。'
             'L-raw 是第一次回答；L-harness 是至多两轮修复后的答案，修复消息只含解析器错误文本。P 的尺寸来自训练集先验，不是预测。', '',
             f"队列：{summary['cohort']['rooms']} 个测试房间（论文六层全部，加每个来源至多 30 个哈希抽样）。"
             '完整率与几何成功率以队列全部请求为分母；OOR 与尺寸指标只在完整输出上计算。单次运行，区间只覆盖房间抽样。', '',
             '## 表 1：各路线指标', '',
             '| 范围 | 路线 | 请求数 | 完整输出 | 完整率↑ | OOR↓ | 尺寸 IoU↑ | log MAE↓ | 几何成功率↑ | 修复后几何成功率↑ |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    scopes = [('全队列', {n: a['all'] for n, a in summary['arms'].items()}), *summary['paper_strata'].items()]
    for scope, arms in scopes:
        for name, m in arms.items():
            lines.append(f"| {scope} | {LABELS[name]} | {m['requests']} | {m['complete_count']} | " + ' | '.join(
                e._fmt(m[k]) for k in ('complete_rate', 'oor', 'size_iou', 'size_log_mae',
                                       'geometry_success_rate', 'geometry_success_repaired_rate')) + ' |')
    if summary['paired']:
        lines += ['', '## 表 2：配对差（前者减后者；房屋连通组自举 95% CI）', '',
                  '| 比较 | 范围 | 配对房间 | Δ完整率 | Δ几何成功率 | ΔOOR | Δlog MAE | Δ尺寸 IoU |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for key, scopes in summary['paired'].items():
            first, second = key.split('_minus_')
            for scope, fields in scopes.items():
                lines.append(f"| {LABELS[first]} − {LABELS[second]} | {'全队列' if scope == 'all' else scope} | "
                             f"{fields['complete']['paired_rooms']} | " + ' | '.join(_ci(fields[k]) for k in PAIRED_METRICS) + ' |')
    r = summary['llm']['result']
    latency = (r['latency_seconds'] or {}).get('p50')
    lines += ['', f"## 表 3：L 调用统计（{summary['llm']['config']['model']}，推理强度 {summary['llm']['config']['reasoning_effort']}）", '',
              '| 房间 | API 失败 | 首答合法 | 最终合法 | 回答轮数分布 | 调用（含失败重试） | 输入 tokens | 输出 tokens | 推理 tokens | 中位延迟 s |',
              '|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|',
              f"| {r['rooms']} | {r['api_failed_rooms']} | {r['valid_first']} | {r['valid_final']} | "
              + ', '.join(f'{k}轮:{v}' for k, v in r['rounds_histogram'].items())
              + f" | {r['calls_ledger']} | {r['tokens']['input_tokens']} | {r['tokens']['output_tokens']} | "
              f"{r['tokens']['reasoning_tokens']} | {e._fmt(latency)} |", '',
              '几何成功与 OOR 的定义同 evaluate_pair 的 report.md；这是本项目的系统对比，不是论文复现，也不是新盲测。']
    return '\n'.join(lines) + '\n'


def evaluate(data, cohort_path, llm, out, runs=None, repeats=2000, workers=1):
    data, out, cohort_path = Path(data), Path(out), Path(cohort_path)
    runs = runs or {}
    if out.exists():
        raise FileExistsError(out)
    if set(runs) not in (set(), set(h.ARMS)):
        raise ValueError('pass both J and P inference runs, or neither')
    manifest = e._verified(data)
    cohort = h.strict_json(cohort_path.read_text())
    if cohort != llm_agent.build_cohort(data):
        raise ValueError('cohort file differs from its deterministic rebuild')
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    requests = {arm: h._by_uid(h._rows(data, arm + '/requests.jsonl', manifest['files'][arm + '/requests.jsonl']))
                for arm in h.ARMS}
    records, llm_receipt, llm_result = _llm(Path(llm), data, cohort, h.sha256(cohort_path))
    jobs = {v: (_score_llm, [(rooms[r['uid']], requests['joint'][r['uid']], r) for r in rows])
            for v, rows in records.items()}
    receipts = {}
    for arm, root in runs.items():
        rows, receipt = e._inference(Path(root), data, manifest['expected_uids'])
        if receipt['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
            raise ValueError('training dataset does not match frozen evaluation prior')
        if receipt['config']['arm'] != arm:
            raise ValueError('run arm mismatch')
        receipts[arm] = receipt['run_sha256']
        indexed = {r['uid']: r for r in rows}
        jobs[arm] = (e._score, [(rooms[uid], requests[arm][uid], indexed[uid], arm) for uid in cohort['uids']])
    scored = {}
    for name in (n for n in ORDER if n in jobs):
        function, tasks = jobs[name]
        if workers == 1:
            scored[name] = [function(task) for task in tasks]
        else:
            with ProcessPoolExecutor(workers) as pool:
                scored[name] = list(pool.map(function, tasks, chunksize=16))
    strata = {source + '/' + kind: {n: [r for r in rows if r['source'] == source
                                        and norm_room_type(r.get('room_type')) == kind] for n, rows in scored.items()}
              for source, kind in PAPER_STRATA}
    if sum(len(v['llm_raw']) for v in strata.values()) != cohort['counts']['paper_strata_total']:
        raise ValueError('cohort strata disagree with the scored reference room types')
    paired = {f'{arm}_minus_{variant}': {
                  'all': g.paired_intervals(scored[arm], scored[variant], repeats=repeats),
                  **{s: g.paired_intervals(v[arm], v[variant], repeats=repeats) for s, v in strata.items()}}
              for arm in h.ARMS if arm in scored for variant in llm_agent.VARIANTS}
    summary = {'status': 'LLM_COMPARISON_COMPLETE',
               'cohort': {'rooms': len(cohort['uids']), 'sha256': h.sha256(cohort_path),
                          'uids_sha256': cohort['uids_sha256'], 'counts': cohort['counts']},
               'arms': {n: g.summarize(rows) for n, rows in scored.items()},
               'paper_strata': {s: {n: g.aggregate(rows) for n, rows in v.items()} for s, v in strata.items()},
               'paired': paired,
               'llm': {'run_sha256': llm_receipt['run_sha256'], 'config': llm_receipt['config'], 'result': llm_result},
               'inference_run_sha256': receipts, 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
               'metric_contract': {'llm_scoring': 'evaluate_pair._score_joint_semantics_on_the_J_request',
                   'paired': 'geometry.paired_intervals_on_cohort; estimate_joint_minus_placement = first minus second arm in key',
                   'placement_size': 'train_only_prior_not_predicted', 'rate_denominator': 'all_cohort_requests',
                   'scalar_missingness': 'conditional_on_complete_output_no_imputation'},
               'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(e.__file__), Path(g.__file__),
                                                              Path(llm_agent.__file__))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        for name, rows in scored.items():
            (dest / (name + '-geometry.jsonl')).write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
        h.write_json(dest / 'summary.json', summary)
        (dest / 'report.md').write_text(_report(summary))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('data', 'cohort', 'llm', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for arm in h.ARMS:
        parser.add_argument('--' + arm, type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2000)
    args = parser.parse_args()
    runs = {arm: getattr(args, arm) for arm in h.ARMS if getattr(args, arm) is not None}
    result = evaluate(args.data, args.cohort, args.llm, args.out, runs, args.repeats, args.workers)
    print(json.dumps({'status': result['status'], 'rooms': result['cohort']['rooms'], 'arms': list(result['arms'])}))


if __name__ == '__main__':
    main()
