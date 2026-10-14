"""Score an LLM agent (API agent L or open model O*) on its frozen cohort and pair it with J and P (amendment 002).

python -m fastfill.comparison.evaluate_llm --data EVAL --cohort COHORT.json --llm LLM_RUN \
    [--joint MERGED_J --placement MERGED_P] --out NEW_DIR [--repeats 2000] [--workers 1] [--uids FILE]

Both L variants (llm_raw, llm_harness) are scored by evaluate_pair._score with
joint semantics (predicted sizes) on the J request the model received. J/P come
from verified merged inference dirs (evaluate_pair._inference), restricted to
the cohort. Per-arm tables use geometry.summarize, the six paper strata use the
evaluate_pair rule, and paired differences J-L and P-L (both L variants, pooled
and per stratum) use geometry.paired_intervals (house-cluster bootstrap). In
paired output, estimate_joint_minus_placement means FIRST minus SECOND arm of
the comparison key. Without --joint/--placement only L tables are written.
Carried through unchanged for report_v2 (not applied here): the scored answer's
reached_token_cap/stopped_on_eos on each L geometry row where the receipts carry
them (open models), and the run's cost.json (GPU-hours) as summary.llm.cost.
Rows and titles carry the agent's system id and model name from its run config (L when the config names no
system; audit minor 17) and summary.paired_stage records the J/P policies paired with it (the stage each
inference receipt records, sft when absent, plus its run and lineage hashes).
A run made with --arm placement (its config records request_arm 'placement'; amendment 016 section 2) is scored
with placement semantics on the P request it received (request_sizes 'gt': llm_agent.gt_request on rooms.jsonl),
after addendum B2's llm_agent.enforce_sizes on each answer (returned sizes replaced by the request's, supported z
recomputed; summary.size_enforcement counts the answers / objects whose returned sizes differ from the request by more
than llm_agent.SIZE_TOL, the "unauthorised size changes"). Its rows are labelled SYSTEM-P (SYSTEM-PGT). --uids FILE
scores a subset of the cohort (llm_agent.subset, e.g. the amendment-016 pilot): the run must cover exactly it, J/P
are restricted to it and summary.cohort.subset records it. Joint runs without --uids are scored and reported unchanged.
"""
import argparse
import collections
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


def labels(system, arm='joint', sizes=None):
    tag = system + ('-P' if arm == 'placement' else '') + ('GT' if sizes == 'gt' else '')
    return {**LABELS, 'llm_raw': f'{tag}-raw', 'llm_harness': f'{tag}-harness'}
PAIRED_METRICS = ('complete', 'geometry_success', 'oor', 'size_log_mae', 'size_iou')
CAP_FLAGS = ('reached_token_cap', 'stopped_on_eos')


def _cap_flags(receipt, variant):
    """Cap flags of the answer a variant row scores (llm_agent._finalize), where the receipt has them."""
    answers = receipt['answers'][:1] if variant == 'llm_raw' else receipt['answers'][-1:]
    return {k: answers[0][k] for k in CAP_FLAGS if answers and k in answers[0]}


def _llm(root, data, cohort, cohort_sha256, uids=None):
    uids = cohort['uids'] if uids is None else uids
    manifest = e._verified(root)
    if manifest.get('expected_uids') != uids:
        raise ValueError('LLM run does not cover exactly the frozen cohort' + (' subset' if uids != cohort['uids'] else ''))
    result = h.strict_json((root / 'result.json').read_text())
    if result.get('status') != 'LLM_COMPLETE':
        raise ValueError('LLM run is not complete')
    receipt = h.strict_json((root / 'run_receipt.json').read_text())
    if infer._hash({k: v for k, v in receipt.items() if k != 'run_sha256'}) != receipt.get('run_sha256'):
        raise ValueError('LLM run identity hash mismatch')
    if receipt['data']['manifest_sha256'] != h.sha256(data / 'MANIFEST.json') or receipt['cohort']['sha256'] != cohort_sha256:
        raise ValueError('LLM run data/cohort identity mismatch')
    receipts = {}
    for uid in uids:
        name = f'receipts/{llm_agent._digest(uid)}.json'  # Hash-checked by _verified.
        stored = h.strict_json((root / name).read_text()) if name in manifest['files'] else {}
        if stored.get('uid') != uid or stored.get('run_sha256') != receipt['run_sha256']:
            raise ValueError('LLM receipts do not match run identity/cohort')
        receipts[uid] = stored
    records = {}
    for variant in llm_agent.VARIANTS:
        name = variant + '/records.jsonl'
        rows = h._rows(root, name, manifest['files'][name])
        if ([r['uid'] for r in rows] != uids
                or any(r['run_sha256'] != receipt['run_sha256'] or r['arm'] != variant for r in rows)):
            raise ValueError('LLM records do not match run identity/cohort')
        records[variant] = [{**r, **_cap_flags(receipts[r['uid']], variant)} for r in rows]
    cost = h.strict_json((root / 'cost.json').read_text()) if 'cost.json' in manifest['files'] else None
    return records, receipt, result, cost


def _enforced(rows, requests):
    """Addendum B2 on every answer of one variant: (rows carrying the enforced answer, counts over the answers that
    are a placements object)."""
    out, tally = [], collections.Counter()
    for row in rows:
        text, counts = llm_agent.enforce_sizes(row['raw'], requests[row['uid']]) if row['raw'] else (row['raw'], None)
        if counts is not None:
            tally.update(counts, answers=1, **{'answers_' + k: counts[k] > 0 for k in counts if k != 'objects'})
        out.append({**row, 'raw': text})
    keys = ('answers', 'answers_size_returned', 'answers_size_changed', 'answers_z_changed',
            'objects', 'size_returned', 'size_changed', 'z_changed')
    return out, {k: tally[k] for k in keys}


def _score_llm(task):
    room, request, record, arm = task
    return {**e._score((room, request, {**record, 'arm': arm}, arm)), 'arm': record['arm'],
            **{k: record[k] for k in CAP_FLAGS if k in record}}


def _ci(item):
    value = item['estimate_joint_minus_placement']
    if value is None:
        return '—'
    ci = item['ci95']
    return f'{value:+.4f}' + (f' [{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else ' [CI —]')


def _report(summary):
    s, config = summary['system'], summary['llm']['config']
    fixed, gt = config.get('request_arm') == 'placement', config.get('request_sizes') == 'gt'
    names = labels(s, config.get('request_arm', 'joint'), config.get('request_sizes'))
    stages = summary.get('paired_stage') or {}
    paired = ('配对的 J／P 策略（推理收据记录的阶段）：' + '；'.join(
        f"{LABELS[arm][0]} = {v['stage']}（推理 run {v['run_sha256'][:12]}"
        + (f"，谱系 {v['lineage_sha256'][:12]}" if v.get('lineage_sha256') else '') + '）' for arm, v in stages.items())
        + '。' if stages else '未配对 J／P，只有代理自身的表。')
    title, intro = (('固定尺寸重生成（P 的输入' + ('，GT 尺寸' if gt else '') + '）与 J／P 对比',
                     (f'{s} 收到 P 的 system 与 user 消息，但尺寸换成 GT（修订 016 附录 A1 的补充图：只 O2、只试点子样本）'
                      if gt else f'{s} 收到与 P 完全相同的 system 与 user 消息（尺寸为训练集先验）')
                     + '，另加一条对所有请求相同的“只输出 JSON”后缀。按附录 B2，判定与评分前回答里的任何尺寸都被请求尺寸覆盖，'
                     '每个有支撑的物体的 z 重算为父物体在请求尺寸下的顶面（落地物体保持原 z），只差尺寸的回答不算不合法；'
                     '之后由 P 的解析器判定，所以只比较摆放（P 自己的输出不做这一步）。')
                    if fixed else ('与 J／P 同输入对比', f'{s} 收到与 J 完全相同的 system 与 user 消息，另加一条对所有请求相同的'
                                   '“只输出 JSON”后缀，由 J 的解析器判定。'))
    lines = [f"# 未知尺寸：LLM 代理 {s}（{config['model']}）{title}", '',
             intro + f"{names['llm_raw']} 是第一次回答；{names['llm_harness']} 是至多两轮重试后的答案（共至多 3 次尝试），重试消息只含异常类名"
             '（通用重试，不含解析器错误文本）。P 的尺寸来自训练集先验，不是预测。' + paired, '',
             (f"队列子集：{summary['cohort']['subset']['rooms']} 个房间（取自 {summary['cohort']['rooms']} 间的冻结队列；"
              f"--uids 文件 sha256 {summary['cohort']['subset']['sha256'][:12]}，例如修订 016 的试点子样本：试点数字只是早期读数，"
              '不替代全队列结果）。' if 'subset' in summary['cohort'] else
              f"队列：{summary['cohort']['rooms']} 个测试房间（论文六层全部，加每个来源至多 30 个哈希抽样）。")
             + '完整率与几何成功率以队列全部请求为分母；OOR 与尺寸指标只在完整输出上计算。单次运行，区间只覆盖房间抽样。', '',
             '## 表 1：各路线指标', '',
             '| 范围 | 路线 | 请求数 | 完整输出 | 完整率↑ | OOR↓ | 尺寸 IoU↑ | log MAE↓ | 几何成功率↑ | 修复后几何成功率↑ |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    scopes = [('全队列', {n: a['all'] for n, a in summary['arms'].items()}), *summary['paper_strata'].items()]
    for scope, arms in scopes:
        for name, m in arms.items():
            lines.append(f"| {scope} | {names[name]} | {m['requests']} | {m['complete_count']} | " + ' | '.join(
                e._fmt(m[k]) for k in ('complete_rate', 'oor', 'size_iou', 'size_log_mae',
                                       'geometry_success_rate', 'geometry_success_repaired_rate')) + ' |')
    if summary['paired']:
        lines += ['', '## 表 2：配对差（前者减后者；房屋连通组自举 95% CI）', '',
                  '| 比较 | 范围 | 配对房间 | Δ完整率 | Δ几何成功率 | ΔOOR | Δlog MAE | Δ尺寸 IoU |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for key, scopes in summary['paired'].items():
            first, second = key.split('_minus_')
            for scope, fields in scopes.items():
                lines.append(f"| {names[first]} − {names[second]} | {'全队列' if scope == 'all' else scope} | "
                             f"{fields['complete']['paired_rooms']} | " + ' | '.join(_ci(fields[k]) for k in PAIRED_METRICS) + ' |')
    r = summary['llm']['result']
    latency = (r['latency_seconds'] or {}).get('p50')
    lines += ['', f"## 表 3：{s} 调用统计（{config['model']}，推理强度 {config.get('reasoning_effort', '未记录')}）", '',
              '| 房间 | 调用失败房间 | 首答合法 | 最终合法 | 回答轮数分布 | 调用（含失败重试） | 输入 tokens | 输出 tokens | 推理 tokens | 中位延迟 s |',
              '|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|',
              f"| {r['rooms']} | {r['api_failed_rooms']} | {r['valid_first']} | {r['valid_final']} | "
              + ', '.join(f'{k}轮:{v}' for k, v in r['rounds_histogram'].items())
              + f" | {r['calls_ledger']} | {r['tokens']['input_tokens']} | {r['tokens']['output_tokens']} | "
              f"{r['tokens']['reasoning_tokens']} | {e._fmt(latency)} |", '']
    if summary.get('size_enforcement'):
        lines += ['## 表 4：未授权的尺寸改动（附录 B2；与请求尺寸相差超过 1 mm 即算改动）', '',
                  '| 路线 | 可解析回答 | 带 size 的回答 | 尺寸被改的回答（占比） | 物体 | 尺寸被改的物体（占比） | z 被重算的物体 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for v, c in summary['size_enforcement'].items():
            share = (lambda k, n: f"{c[k]}（{e._fmt(c[k] / c[n]) if c[n] else '—'}）")
            lines.append(f"| {names[v]} | {c['answers']} | {c['answers_size_returned']} | "
                         f"{share('answers_size_changed', 'answers')} | {c['objects']} | {share('size_changed', 'objects')} | "
                         f"{c['z_changed']} |")
        lines.append('')
    lines.append('几何成功与 OOR 的定义同 evaluate_pair 的 report.md；这是本项目的系统对比，不是论文复现，也不是新盲测。')
    return '\n'.join(lines) + '\n'


def evaluate(data, cohort_path, llm, out, runs=None, repeats=2000, workers=1, uids_path=None):
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
    uids = cohort['uids'] if uids_path is None else llm_agent.subset(cohort, uids_path)
    records, llm_receipt, llm_result, llm_cost = _llm(Path(llm), data, cohort, h.sha256(cohort_path), uids)
    config = llm_receipt['config']
    request_arm = config.get('request_arm', 'joint')  # 'placement': amendment 016 section 2.
    asked = {uid: requests[request_arm][uid] for uid in uids}
    if config.get('request_sizes') == 'gt':  # Addendum A1 supplementary.
        asked = {uid: llm_agent.gt_request(record, rooms[uid]) for uid, record in asked.items()}
    enforcement = {}
    if request_arm == 'placement':  # Addendum B2.
        for variant in list(records):
            records[variant], enforcement[variant] = _enforced(records[variant], asked)
    jobs = {v: (_score_llm, [(rooms[r['uid']], asked[r['uid']], r, request_arm) for r in rows])
            for v, rows in records.items()}
    receipts, stages = {}, {}
    for arm, root in runs.items():
        rows, receipt = e._inference(Path(root), data, manifest['expected_uids'])
        if receipt['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
            raise ValueError('training dataset does not match frozen evaluation prior')
        if receipt['config']['arm'] != arm:
            raise ValueError('run arm mismatch')
        receipts[arm] = receipt['run_sha256']
        stages[arm] = {'stage': receipt.get('stage', 'sft'), 'run_sha256': receipt['run_sha256'],
                       'lineage_sha256': receipt.get('lineage_sha256'), 'lora': receipt['config'].get('lora')}
        indexed = {r['uid']: r for r in rows}
        jobs[arm] = (e._score, [(rooms[uid], requests[arm][uid], indexed[uid], arm) for uid in uids])
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
    expected = (cohort['counts']['paper_strata_total'] if uids_path is None
                else sum(llm_agent._stratum(requests['joint'][uid]) is not None for uid in uids))
    if sum(len(v['llm_raw']) for v in strata.values()) != expected:
        raise ValueError('cohort strata disagree with the scored reference room types')
    paired = {f'{arm}_minus_{variant}': {
                  'all': g.paired_intervals(scored[arm], scored[variant], repeats=repeats),
                  **{s: g.paired_intervals(v[arm], v[variant], repeats=repeats) for s, v in strata.items()}}
              for arm in h.ARMS if arm in scored for variant in llm_agent.VARIANTS}
    summary = {'status': 'LLM_COMPARISON_COMPLETE', 'system': llm_receipt['config'].get('system', 'L'),
               'model': llm_receipt['config']['model'], 'paired_stage': stages,
               'cohort': {'rooms': len(cohort['uids']), 'sha256': h.sha256(cohort_path),
                          'uids_sha256': cohort['uids_sha256'], 'counts': cohort['counts'],
                          **({'subset': {'rooms': len(uids), 'sha256': h.sha256(uids_path),
                                         'uids_sha256': llm_agent._digest(json.dumps(uids))}} if uids_path else {})},
               'arms': {n: g.summarize(rows) for n, rows in scored.items()},
               'paper_strata': {s: {n: g.aggregate(rows) for n, rows in v.items()} for s, v in strata.items()},
               'paired': paired, **({'size_enforcement': enforcement} if enforcement else {}),
               'llm': {'run_sha256': llm_receipt['run_sha256'], 'config': llm_receipt['config'], 'result': llm_result,
                       'cost': llm_cost},
               'inference_run_sha256': receipts, 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
               'metric_contract': {
                   'llm_scoring': f'evaluate_pair._score_{request_arm}_semantics_on_the_{request_arm[0].upper()}_request'
                                  + ('_with_GT_sizes' if config.get('request_sizes') == 'gt' else ''),
                   **({'size_enforcement': 'addendum B2: llm_agent.enforce_sizes before scoring; counts over answers that '
                                           'are a placements object; changed = returned size off the request by more '
                                           f'than {llm_agent.SIZE_TOL} m (or malformed); z_changed = supported z moved '
                                           f'by more than {llm_agent.SIZE_TOL} m'} if enforcement else {}),
                   'paired': 'geometry.paired_intervals_on_cohort; estimate_joint_minus_placement = first minus second arm in key',
                   'placement_size': 'train_only_prior_not_predicted', 'rate_denominator': 'all_cohort_requests',
                   'scalar_missingness': 'conditional_on_complete_output_no_imputation',
                   'cap_flags': 'copied_per_row_where_receipts_carry_them_not_applied_here'},
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
    parser.add_argument('--uids', type=Path)  # A subset of the cohort (amendment 016 pilot).
    args = parser.parse_args()
    runs = {arm: getattr(args, arm) for arm in h.ARMS if getattr(args, arm) is not None}
    result = evaluate(args.data, args.cohort, args.llm, args.out, runs, args.repeats, args.workers, args.uids)
    print(json.dumps({'status': result['status'], 'rooms': result['cohort']['rooms'], 'arms': list(result['arms'])}))


if __name__ == '__main__':
    main()
