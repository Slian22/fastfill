"""Publish measured geometry/FID/GPT tables; never impute unmeasured results.

J/P: --geometry GEO --fid FID --judge JUDGE --out OUT.
N systems (amendment 005 section 5): --system J=GEO --system P=GEO --system L=LLM
[--system O5=...] --fid FID --judge JUDGE --out OUT, inputs from
visual_pipeline prepare-systems, fid (systems mode) and judge_batch --arms.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import tempfile

from fastfill.comparison import render_assets as render
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison import visual_pipeline
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
    _check_visual(fid, judge, h.sha256(geometry / 'MANIFEST.json'))
    expected_strata = {source + '/' + kind for source, kind in PAPER_STRATA
                       if any(summary['paper_strata'].get(source + '/' + kind, {})
                              .get(arm, {}).get('requests', 0) > 0 for arm in ARMS)}
    if set(judge['by_stratum']) != expected_strata:
        raise ValueError('judge stratum coverage mismatch')
    return summary, fid, judge


def _check_visual(fid, judge, lineage):
    if (fid.get('extractor') != 'pytorch_fid:0.3.0:pool3' or fid.get('feature_dimension') != 2048
            or fid.get('weights_sha256') != WEIGHTS_SHA256):
        raise ValueError('FID extractor/weights mismatch')
    if fid.get('provenance', {}).get('provenance', {}).get('geometry_manifest_sha256') != lineage:
        raise ValueError('FID geometry lineage mismatch')
    if judge.get('model') != 'gpt-6.1-sol':
        raise ValueError('judge model mismatch')
    if judge.get('provenance', {}).get('geometry_manifest_sha256') != lineage:
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


def _verify_systems(systems, fid_path, judge_path):
    """Each record dir verified and complete on ONE evaluation manifest; lineage = all dirs."""
    systems = visual_pipeline._systems(systems)
    summaries = {}
    for label, (root, _) in systems.items():
        h.verify_artifacts(root, _read(root / 'MANIFEST.json')['files'])
        summaries[label] = _read(root / 'summary.json')
        if not str(summaries[label].get('status')).endswith('_COMPLETE'):
            raise ValueError(f'{label}: evaluation must be complete')
    if len({s.get('data_manifest_sha256') for s in summaries.values()}) != 1:
        raise ValueError('system evaluation manifest mismatch')
    fid, judge = _read(fid_path), _read(judge_path)
    _check_visual(fid, judge, render.json_sha256(visual_pipeline.systems_lineage(systems)))
    if fid.get('systems') != list(systems):
        raise ValueError('FID system set mismatch')
    if judge.get('arms') != visual_pipeline.judged_systems(systems):
        raise ValueError('judged system set mismatch')
    expected_strata = {source + '/' + kind for source, kind in PAPER_STRATA
                       if any(summaries[label]['paper_strata'].get(source + '/' + kind, {})
                              .get(arm, {}).get('requests', 0) > 0 for label, (_, arm) in systems.items())}
    if set(judge['by_stratum']) != expected_strata:
        raise ValueError('judge stratum coverage mismatch')
    return systems, summaries, fid, judge


def _system_rows(systems, summaries, fid, judge):
    rows = []
    for source, room_type in PAPER_STRATA:
        stratum = source + '/' + room_type
        visual = fid['strata'].get(stratum, {})
        for label, (_, arm) in systems.items():
            geometry = summaries[label]['paper_strata'].get(stratum, {}).get(arm, {})
            kid = (visual.get('kid') or {}).get(label) or {}
            judged = judge['by_stratum'].get(stratum, {}).get(label, {})
            means = [judged.get(k) for k in SCORES[:3]]
            rows.append({'stratum': stratum, 'system': label, 'requested': geometry.get('requests', 0),
                'OOR_complete_n': geometry.get('complete_count', 0), 'OOR': geometry.get('oor'),
                'FID': visual.get(label + '_vs_reference'), 'KID': kid.get('kid'), 'KID_SE': kid.get('se'),
                'FID_shared_n': visual.get('counts', {}).get('shared_success', 0),
                'FID_small_n': visual.get('small_n'),
                'GPT_requested': judged.get('requested', 0), 'GPT_success': judged.get('success', 0),
                **{score: judged.get(score) for score in SCORES},
                'GPT_room_mean': None if None in means else math.fsum(means) / 3, 'UR': None})
    return rows


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


def _systems_report(rows, fid, judge):
    e = judge['endpoint']
    estimate = '未测' if e['mean_difference'] is None else f"{e['mean_difference']:+.4f}"
    interval = '—' if e['ci95'] is None else f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
    lines = ['# 未知尺寸：多系统视觉对比实测结果', '',
        '**本实验不是论文原始测试集或作者原始权重的复现。** J：联合预测尺寸与布局；'
        'P：类别×房型中位数尺寸（常数回退 0.40、0.50、0.65 m）后布局；L：API 代理 gpt-6.1-sol（推理 medium，修复后答案）；'
        'O*：开源模型代理（修复后答案）。各系统输入相同；代理只是对照，交付模型只在 J、P 中选择。', '',
        '## 表 1：六组房间 OOR、FID、KID', '',
        '| 来源/房型 | 系统 | 完整/请求 | OOR↓ | FID↓ | KID↓ (SE) | 共同样本 N |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        kid = '未测' if r['KID'] is None else f"{r['KID']:.4f} ({r['KID_SE']:.4f})"
        flag = '（N<50）' if r['FID_small_n'] else ''
        lines.append(f"| {r['stratum']} | {r['system']} | {r['OOR_complete_n']}/{r['requested']} | "
                     f"{_fmt(r['OOR'])} | {_fmt(r['FID'])} | {kid} | {r['FID_shared_n']}{flag} |")
    lines += ['', 'OOR 为物体两两 XY 交叠面积之和除以总占地面积（不看朝向和高度），在各系统自身全部完整输出上计算。'
              'FID 与 KID 只在参考布局及全部系统都成功渲染的房间交集上计算，N 为该交集；N<50 的层已标记，结果不稳定。'
              'KID 为三次多项式核的配对 U 统计量，括号内为一阶标准误。各层分别报告，不合并、不平均。', '',
              '## 表 2：同一 GPT 裁判的视觉评分', '',
              '| 来源/房型 | 系统 | 成功/选定 | 功能↑ | 布局↑ | 美观↑ | 三项均值↑ |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['GPT_requested']:
            lines.append(f"| {r['stratum']} | {r['system']} | {r['GPT_success']}/{r['GPT_requested']} | "
                         + ' | '.join(_fmt(r[k]) for k in (*SCORES[:3], 'GPT_room_mean')) + ' |')
    lines += ['', f"GPT 终点：{e['direction'].replace('_minus_', '−')} 每房间三项均值配对差 {estimate}，95% CI {interval}；"
              f"双方均成功 {e['both_complete']}/{e['selected_pairs']} 个房间；按层内重抽样房间的自举 "
              f"{e['bootstrap']['draws']} 次，种子 {e['bootstrap']['seed']}。分项分数只作描述。", '',
              '裁判为 `gpt-6.1-sol`，输入同一房间的透视图和顶视图（各 512×512），只获得类别清单及房间固定条件；'
              '失败计入分母、不补分。此评分不能与论文 GPT-4o 分数等同。真人 UR 未测。', '',
              '视觉使用固定真实资产、相同相机和渲染器，把资产拟合到各系统预测尺寸；不是作者的检索/Isaac 渲染流程。'
              '单一训练种子；区间只覆盖房间抽样。', '',
              f"FID 计数：`{json.dumps(fid.get('counts', {}), ensure_ascii=False)}`。",
              f"GPT 计数：`{json.dumps(judge.get('counts', {}), ensure_ascii=False)}`。", '',
              '逐房间失败、渲染哈希、请求和 API 回执保留在同级运行目录；MANIFEST.json 固定本报告的数据来源和文件哈希。']
    return '\n'.join(lines) + '\n'


def _publish_tables(out, result, csv_name, rows, report):
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temporary:
        dest = Path(temporary)
        h.write_json(dest / 'results.json', result)
        _csv(dest / csv_name, rows)
        (dest / 'report.md').write_text(report)
        h.write_json(dest / 'MANIFEST.json', {'input_sha256': result['input_sha256'], 'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return result


def build_systems(systems, fid, judge, out):
    """N-system visual tables; systems maps a visual_pipeline.SYSTEMS label -> record dir."""
    fid, judge, out = map(Path, (fid, judge, out))
    if out.exists():
        raise FileExistsError(out)
    systems, summaries, visual, judged = _verify_systems(systems, fid, judge)
    rows = _system_rows(systems, summaries, visual, judged)
    lineage = visual_pipeline.systems_lineage(systems)
    result = {'status': 'MEASURED_SYSTEM_TABLES_PUBLISHED', 'system_rows': rows, 'systems': lineage,
              'gpt_endpoint': judged['endpoint'], 'fid': visual, 'judge': judged,
              'human_UR': 'not_measured', 'quality_claim': 'no_automatic_winner',
              'input_sha256': {'systems_lineage': render.json_sha256(lineage),
                               'FID': h.sha256(fid), 'GPT': h.sha256(judge)},
              'code_sha256': h.sha256(__file__)}
    return _publish_tables(out, result, 'system-style.csv', rows, _systems_report(rows, visual, judged))


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
    return _publish_tables(out, result, 'paper-style.csv', rows, _report(summary, rows, visual, judged))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('fid', 'judge', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--geometry', type=Path)
    mode.add_argument('--system', action='append', metavar='LABEL=DIR')
    args = parser.parse_args(argv)
    if args.system:
        pairs = [value.split('=', 1) for value in args.system]
        if any(len(p) != 2 for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
            parser.error('--system must be unique LABEL=DIR')
        result = build_systems(dict(pairs), args.fid, args.judge, args.out)
    else:
        result = build(args.geometry, args.fid, args.judge, args.out)
    print(json.dumps({'status': result['status']}))


if __name__ == '__main__':
    main()
