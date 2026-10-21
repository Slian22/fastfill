"""Publish measured geometry/FID/GPT tables; never impute unmeasured results."""
import argparse
import csv
import json
import math
from pathlib import Path
import tempfile

from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.comparison.eval_data import PAPER_STRATA
from fastfill.comparison.fid import WEIGHTS_SHA256
from fastfill.comparison.judge_batch import validate_summary

ARMS = ('joint', 'placement')
SCORES = ('functionality_score', 'layout_score', 'aesthetics_score', 'overall_score')
PAPER = ((33.56, .03, 8.6, 9.1, 9.3), (37.68, .08, 8.8, 8.7, 9.0),
         (26.45, .01, 8.7, 9.0, 9.2), (31.25, .02, 8.9, 9.2, 9.1),
         (41.73, .05, 8.4, 8.3, 8.8), (35.29, .02, 8.7, 9.4, 9.0))


def _read(path):
    return h.strict_json(Path(path).read_text())


def _fmt(value):
    return '未测' if value is None else f'{value:.4f}'


def _verify(geometry, fid_path, judge_path):
    manifest = _read(geometry / 'MANIFEST.json')
    h.verify_artifacts(geometry, manifest['files'])
    summary, fid, judge = _read(geometry / 'summary.json'), _read(fid_path), _read(judge_path)
    if summary.get('status') != 'GEOMETRY_COMPLETE':
        raise ValueError('geometry must be complete')
    if (fid.get('extractor') != 'pytorch_fid:0.3.0:pool3' or fid.get('feature_dimension') != 2048
            or fid.get('weights_sha256') != WEIGHTS_SHA256):
        raise ValueError('FID extractor/weights mismatch')
    if fid.get('provenance', {}).get('provenance', {}).get('geometry_manifest_sha256') != h.sha256(geometry / 'MANIFEST.json'):
        raise ValueError('FID geometry lineage mismatch')
    if judge.get('model') != 'gpt-6.1-sol':
        raise ValueError('judge model mismatch')
    if judge.get('provenance', {}).get('geometry_manifest_sha256') != h.sha256(geometry / 'MANIFEST.json'):
        raise ValueError('judge lineage mismatch')
    for arms in judge.get('by_stratum', {}).values():
        for scores in arms.values():
            requested, success = scores.get('requested'), scores.get('success')
            if type(requested) is not int or type(success) is not int or not 0 <= success <= requested:
                raise ValueError('invalid judge observations')
            for key in SCORES:
                value = scores.get(key)
                if (not success and value is not None) or (value is not None and
                        (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10)):
                    raise ValueError('judge score inconsistent with observations')
    validate_summary(judge)
    expected_strata = {source + '/' + kind for source, kind in PAPER_STRATA
                       if any(summary['paper_strata'].get(source + '/' + kind, {})
                              .get(arm, {}).get('requests', 0) > 0 for arm in ARMS)}
    if set(judge['by_stratum']) != expected_strata:
        raise ValueError('judge stratum coverage mismatch')
    return summary, fid, judge


def _rows(summary, fid, judge):
    rows = []
    for source, room_type in PAPER_STRATA:
        stratum = source + '/' + room_type
        visual = fid['strata'].get(stratum, {})
        for arm in ARMS:
            geometry = summary['paper_strata'].get(stratum, {}).get(arm, {})
            judged = judge.get('by_stratum', {}).get(stratum, {}).get(arm, {})
            rows.append({'stratum': stratum, 'arm': arm,
                'requested': geometry.get('requests', 0), 'OOR_complete_n': geometry.get('complete_count', 0),
                'OOR': geometry.get('oor'), 'FID': visual.get(arm + '_vs_reference'),
                'FID_shared_n': visual.get('counts', {}).get('shared_success', 0),
                'GPT_requested': judged.get('requested', 0), 'GPT_success': judged.get('success', 0),
                **{score: judged.get(score) for score in SCORES}, 'UR': None})
    return rows


def _csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(summary, rows, fid, judge):
    lines = ['# 未知尺寸联合生成：实测结果', '',
        '**本实验不是论文原始测试集或作者原始权重的复现。** J 为联合尺寸与布局生成；'
        'P 为训练集类别×房间类型尺寸估计，再生成布局。两者使用相同通用 Qwen3-8B 基座、共同训练样本和训练预算。', '',
        '主任务：输入不提供待放家具的真实尺寸，输出尺寸、位置和朝向。P 的尺寸先验用训练集连通组五折交叉拟合；'
        '测试使用完整训练先验。先验拟合池大于解码器共同合格池，此额外训练数据暴露已记录。', '',
        '## 表 1：论文同类六组房间', '',
        '| 来源/房型 | 路线 | 完整/请求 | OOR↓ | FID↓ | FID共同样本 |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['stratum']} | {r['arm']} | {r['OOR_complete_n']}/{r['requested']} | "
                     f"{_fmt(r['OOR'])} | {_fmt(r['FID'])} | {r['FID_shared_n']} |")
    lines += ['', 'OOR 为物体重叠率，不是越界率：忽略朝向和高度，物体两两 XY 交叠面积之和除以总物体占地面积。'
              'OOR 在本行全部合法完整输出上计算；FID 只在参考、J、P 三方共同成功的渲染上计算，两者分母不同。', '',
              '## 表 2：同一 GPT 裁判的视觉评分', '',
              '| 来源/房型 | 路线 | 成功/选定 | 功能↑ | 布局↑ | 美观↑ |',
              '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['stratum']} | {r['arm']} | {r['GPT_success']}/{r['GPT_requested']} | "
                     + ' | '.join(_fmt(r[k]) for k in SCORES[:3]) + ' |')
    lines += ['', '裁判为 `gpt-6.1-sol`，输入同一房间的透视图和顶视图，每个视图 512×512。'
              '裁判只获得类别清单及房间固定条件。此评分不能与论文 GPT-4o 分数直接等同。真人 UR 未测。', '',
              '## 表 3：全部来源几何与尺寸', '',
              '| 范围 | 路线 | 完整/请求 | 尺寸 IoU↑ | log MAE↓ | OOR↓ | 几何成功率↑ |',
              '|---|---|---:|---:|---:|---:|---:|']
    for arm, result in summary['arms'].items():
        entries = {'全部房间微平均': result['all'], **result['by_source']}
        for name, r in entries.items():
            lines.append(f"| {name} | {arm} | {r['complete_count']}/{r['requests']} | "
                         + ' | '.join(_fmt(r[k]) for k in ('size_iou', 'size_log_mae', 'oor', 'geometry_success_rate')) + ' |')
    lines += ['', '尺寸指标先按房间内物体平均，再按房间平均；只对合法完整输出计算。完整率和几何成功率以全部请求为分母。'
              '几何成功是带容差和类别豁免的包围盒检查，不代表网格物理有效性。'
              '原始结果与仅移动位置的 10 cm 边界修复结果分开记录；修复不改变预测尺寸。', '',
              '## 论文数值参考（独立协议，不能据此宣称胜负）', '',
              '| 来源/房型 | 论文 FID↓ | 论文 OOR↓ | 功能↑ | 布局↑ | 美观↑ |',
              '|---|---:|---:|---:|---:|---:|']
    for (source, kind), values in zip(PAPER_STRATA, PAPER):
        lines.append('| ' + source + '/' + kind + ' | ' + ' | '.join(_fmt(v) for v in values) + ' |')
    lines += ['', '参考：[OptiScene 论文](https://arxiv.org/abs/2506.07570)表 1、2。论文输入已检索的家具尺寸、使用其数据切分与渲染流程；'
              '本实验使用历史多来源测试集、真实资产辅助渲染和不同裁判。数值并排仅展示评测维度。', '',
              '## 适用限制与证据', '',
              '历史测试集过去已用于诊断，不能称作新盲测。只训练一个随机种子；房屋连通组配对自举区间不覆盖训练随机性。'
              '来源宏平均、配对差值和区间保存在 results.json 的 geometry 字段中；缺失分数不补零。', '',
              'FID 使用固定真实资产、相同相机和渲染器，将资产拟合至各自预测尺寸。'
              '不是作者的检索/Isaac 渲染流程；小样本 FID 有偏且不稳定，不将六组 FID 平均成总分。', '',
              f"FID 计数：`{json.dumps(fid.get('counts', {}), ensure_ascii=False)}`。",
              f"GPT 计数：`{json.dumps(judge.get('counts', {}), ensure_ascii=False)}`。", '',
              '逐房间失败、渲染哈希、请求和 API 回执保留在同级运行目录；MANIFEST.json 固定本报告的数据来源和文件哈希。']
    return '\n'.join(lines) + '\n'


def build(geometry, fid, judge, out):
    geometry, fid, judge, out = map(Path, (geometry, fid, judge, out))
    if out.exists():
        raise FileExistsError(out)
    summary, visual, judged = _verify(geometry, fid, judge)
    rows = _rows(summary, visual, judged)
    result = {'status': 'MEASURED_TABLES_PUBLISHED', 'paper_style_rows': rows,
        'geometry': summary, 'fid': visual, 'judge': judged,
        'human_UR': 'not_measured', 'quality_claim': 'no_automatic_winner',
        'input_sha256': {'geometry_MANIFEST': h.sha256(geometry / 'MANIFEST.json'),
                         'FID': h.sha256(fid), 'GPT': h.sha256(judge)},
        'code_sha256': h.sha256(__file__)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temporary:
        dest = Path(temporary)
        h.write_json(dest / 'results.json', result)
        _csv(dest / 'paper-style.csv', rows)
        (dest / 'report.md').write_text(_report(summary, rows, visual, judged))
        h.write_json(dest / 'MANIFEST.json', {'input_sha256': result['input_sha256'], 'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('geometry', 'fid', 'judge', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    result = build(**vars(parser.parse_args()))
    print(json.dumps({'status': result['status']}))


if __name__ == '__main__':
    main()
