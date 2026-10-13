"""Verify both completed inference runs and publish reproducible geometry tables."""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import tempfile

from fastfill.comparison import geometry as g
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.comparison.eval_data import PAPER_STRATA
from fastfill.scene import norm_room_type

DECODE_KEYS = ('max_new_tokens', 'max_input_tokens', 'context_budget', 'batch_size',
               'logits_to_keep', 'do_sample', 'num_beams', 'dtype', 'attention')


def _verified(root):
    manifest = h.strict_json((root / 'MANIFEST.json').read_text())
    h.verify_artifacts(root, manifest['files'])
    return manifest


def _inference(root, data, expected):
    manifest = _verified(root)
    if manifest.get('expected_uids') != expected:
        raise ValueError('inference UID inventory mismatch')
    status = h.strict_json((root / 'result.json').read_text())
    if status.get('status') != 'INFERENCE_COMPLETE' or status.get('pending_count') != 0:
        raise ValueError('inference not complete; partial results are not a formal comparison')
    receipt = h.strict_json((root / 'run_receipt.json').read_text())
    identity = {k: v for k, v in receipt.items() if k != 'run_sha256'}
    computed = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if computed != receipt.get('run_sha256'):
        raise ValueError('inference run identity hash mismatch')
    if receipt.get('context_plan_sha256') is not None:
        if h.sha256(root / 'context_plan.json') != receipt['context_plan_sha256']:
            raise ValueError('context plan receipt hash mismatch')
    if receipt['data']['manifest_sha256'] != h.sha256(data / 'MANIFEST.json'):
        raise ValueError('inference dataset identity mismatch')
    records = h._rows(root, 'records.jsonl', manifest['files']['records.jsonl'])
    if [r['uid'] for r in records] != expected or any(r['run_sha256'] != receipt['run_sha256'] for r in records):
        raise ValueError('inference records do not match run identity/inventory')
    return records, receipt


def _score(task):
    room, request, record, arm = task
    if any(record[k] != room[k] or request[k] != room[k] for k in ('uid', 'source', 'group')):
        raise ValueError('prediction/request/reference identity mismatch')
    if record['arm'] != arm:
        raise ValueError('prediction arm mismatch')
    return g.score(room, request['messages'], record.get('raw'), arm, record.get('error'))


def _fmt(value):
    return '—' if value is None else f'{value:.4f}'


def _write_tables(out, summary):
    columns = ['scope', 'arm', 'requests', 'complete_count', 'complete_rate', 'oor', 'size_iou',
               'size_log_mae', 'geometry_success_rate', 'oor_repaired', 'geometry_success_repaired_rate']
    entries = []
    for arm, result in summary['arms'].items():
        entries.append({'scope': 'all_sources_room_micro', 'arm': arm, **result['all']})
        entries.extend({'scope': source, 'arm': arm, **metrics} for source, metrics in result['by_source'].items())
    for stratum, arms in summary['paper_strata'].items():
        entries.extend({'scope': stratum, 'arm': arm, **metrics} for arm, metrics in arms.items())
    with (out / 'geometry.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(entries)
    lines = ['# 未知尺寸：联合生成与估计尺寸后布局', '',
        'J：联合预测尺寸、坐标和朝向。P：训练集估计尺寸，再预测坐标和朝向。两者使用相同通用基座和共同训练样本。', '',
        '**这是本项目改造后的系统对比，不是作者原始权重，也不是论文原始测试集的复现。**', '',
        f"模型阶段（stage）：{summary['stage']}（sft / dpo1 / dpo2）。", '',
        'OOR 是物体重叠率：忽略朝向和高度的两两 XY 交叠面积之和，除以物体面积之和。它不是越界率，也不一定小于 1。', '',
        '尺寸 IoU 和 log MAE 均先在房间内按物体平均，再按房间平均；只对完整合法输出计算。完整率及几何成功率以全部请求为分母。', '',
        '| 数据范围 | 路线 | 请求数 | 完整输出 | OOR↓ | 尺寸 IoU↑ | log MAE↓ | 几何成功率↑ |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in entries:
        lines.append(f"| {row['scope']} | {row['arm']} | {row['requests']} | {row['complete_count']} | "
                     + ' | '.join(_fmt(row[k]) for k in ('oor', 'size_iou', 'size_log_mae', 'geometry_success_rate')) + ' |')
    lines += ['', '几何成功要求边界、支撑及已知层高检查通过，并且无主要碰撞、严重碰撞或固定障碍阻挡。'
              '这些是带容差及类别豁免的包围盒检查，不能代替网格级物理验证。扫描 hull、近似 proxy 与 polygon 的边界证据强度不同。', '',
              '原始输出与只移动位置的边界修复结果分别保存在 geometry.csv 和逐房间记录中；修复使用预测尺寸，不缩放家具。', '',
              'FID、GPT 三维评分和真人 UR 不由此脚本生成；没有完成的视觉或真人评测不填数字。', '',
              'summary.json 保存来源宏平均、共同成功样本上的配对差异、房屋连通分组自举区间及适用限制。'
              '历史测试集已经用于过去的诊断，不能称作新盲测。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def evaluate(data, runs, out, repeats=2000, workers=1):
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if set(runs) != set(h.ARMS):
        raise ValueError('both arms required')
    manifest = _verified(data)
    expected = manifest['expected_uids']
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    if set(rooms) != set(expected):
        raise ValueError('reference inventory mismatch')
    output_rows, receipts = {}, {}
    for arm in h.ARMS:
        records, receipt = _inference(Path(runs[arm]), data, expected)
        if receipt['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
            raise ValueError('training dataset does not match frozen evaluation prior')
        if receipt['config']['arm'] != arm:
            raise ValueError('run arm mismatch')
        receipts[arm] = receipt
        name = arm + '/requests.jsonl'
        requests = h._by_uid(h._rows(data, name, manifest['files'][name]))
        tasks = [(rooms[r['uid']], requests[r['uid']], r, arm) for r in records]
        if workers == 1:
            output_rows[arm] = [_score(task) for task in tasks]
        else:
            with ProcessPoolExecutor(workers) as pool:
                output_rows[arm] = list(pool.map(_score, tasks, chunksize=16))
    first, second = (receipts[arm] for arm in h.ARMS)
    if first['base_sha256'] != second['base_sha256']:  # Generic Qwen3-8B lineage root, also for merged stages.
        raise ValueError('base checkpoint mismatch between arms')
    if first.get('stage') != second.get('stage'):  # Amendment 008: SFT (absent), dpo1 or dpo2.
        raise ValueError('model stage mismatch between arms')
    if first.get('context_plan_sha256') != second.get('context_plan_sha256'):
        raise ValueError('paired context budget plan mismatch')
    if any(first['config'].get(k) != second['config'].get(k) for k in DECODE_KEYS):
        raise ValueError('decoding protocol mismatch between arms')
    strata = {source + '/' + kind: {arm: g.aggregate([r for r in rows if r['source'] == source
                and norm_room_type(r.get('room_type')) == kind]) for arm, rows in output_rows.items()}
              for source, kind in PAPER_STRATA}
    summary = {'status': 'GEOMETRY_COMPLETE', 'stage': first.get('stage', 'sft'),  # Amendment 009 label.
               'expected_rooms_per_arm': len(expected),
               'arms': {arm: g.summarize(rows) for arm, rows in output_rows.items()}, 'paper_strata': strata,
               'paired': g.paired_intervals(output_rows['joint'], output_rows['placement'], repeats=repeats),
               'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
               'inference_run_sha256': {arm: r['run_sha256'] for arm, r in receipts.items()},
               'metric_contract': {'oor': 'axis_aligned_XY_pair_overlap_over_total_area',
                   'scalar_missingness': 'conditional_on_complete_output_no_imputation',
                   'rate_denominator': 'all_frozen_requests', 'repair': 'predicted_size_frozen_boundary_snap_10cm',
                   'FID': 'separate_visual_evaluation', 'GPT': 'separate_API_evaluation', 'UR': 'not_measured'},
               'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(g.__file__))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        for arm, rows in output_rows.items():
            (dest / (arm + '-geometry.jsonl')).write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
        h.write_json(dest / 'summary.json', summary)
        _write_tables(dest, summary)
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('data', 'joint', 'placement', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2000)
    args = parser.parse_args()
    result = evaluate(args.data, {arm: getattr(args, arm) for arm in h.ARMS}, args.out, args.repeats, args.workers)
    print(json.dumps({'status': result['status'], 'rooms_per_arm': result['expected_rooms_per_arm']}))


if __name__ == '__main__':
    main()
