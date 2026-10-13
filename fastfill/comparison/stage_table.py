"""Amendments 008/009: paper-Table-6-style stage ablation SFT -> DPO1 -> DPO2 for J and P; never imputes.

python -m fastfill.comparison.stage_table --run-root RUN --report sft=REPORT_V2_DIR \
    [--report dpo1=DIR] [--report dpo2=DIR] --out NEW_DIR

Reads only small JSON evidence under the run-root layout of the DPO chain (submit_dpo_stages.sh):
- each stage's report_v2 output dir (whole MANIFEST verified; results.json);
- RUN/geometry-v{2,3,4}/summary.json, hash-checked against its MANIFEST, whose sha256 must be the
  report's geometry_MANIFEST; status GEOMETRY_COMPLETE and stage label (absent = sft, as written by
  the pre-amendment-009 evaluate_pair);
- RUN/infer-{joint,placement}-00{2,3,4}/run_receipt.json: self-hash = the report's run hash, stage
  label, arm, LoRA mode (SFT: runtime GPU merge; DPO: exact policy, amendment 009), one shared
  context plan, decode settings, base and training data across every stage;
- RUN/dpo-001/{arm}/{sft,dpo1,dpo2}-merged.json lineage receipts (chain hashes, merge_diagnostic) bound
  to the evaluated inference (SFT adapter / DPO adapter, training base and lineage hashes);
- RUN/dpo-001/stage{1,2}-pairs/summary.json (MANIFEST-checked) bound to each DPO receipt's pairs.
Every stage report must share data, reference, repeats, seed, report_v2.py and geometry.py.
A stage without a report is 'not run'; a value the inputs do not carry is '未测' with its reason.
Outputs: stage_table.md (Chinese), one CSV per table, stage_table.json, MANIFEST.json.
"""
import argparse
import json
from pathlib import Path
import tempfile

from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import infer as i
from fastfill.comparison import report_v2 as R
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

STAGES = d.STAGES
TAG = {'sft': '002', 'dpo1': '003', 'dpo2': '004'}
GEOMETRY = {'sft': 'geometry-v2', 'dpo1': 'geometry-v3', 'dpo2': 'geometry-v4'}
DPO_ROOT = 'dpo-001'
LORA = {'sft': 'merge_and_unload_bf16', 'dpo': 'unmerged_fp32_stage_adapter_on_bf16_training_base'}
POLICY = {'sft': 'SFT LoRA 运行时 GPU 合并（merge_and_unload_bf16）',
          'dpo': '精确训练策略：未合并 FP32 适配器 + BF16 训练基座（修订 009）'}
DECODE = tuple(k for k in e.DECODE_KEYS if k != 'lora')
LABELS = {'joint': 'J', 'placement': 'P'}
ENDPOINTS = ('(a)', '(b)', '(a′)')
T1B = ('complete', 'geometry_success_repaired', 'oor', 'oor_repaired', 'size_iou')
NOT_RUN = '未运行：未提供该阶段的 report_v2 输出'
REASONS = {'oor_rotated': '未测：report_v2 只按论文六层输出旋转 OOR（见六层合并列）',
           'automatic_acceptance': '未测：report_v2 未输出自动接受率字段'}
COLUMNS = [('stage', '阶段'), ('arm', '系统'), ('requests', '请求'), ('complete_rate', '完整率'),
           ('geometry_success', '(a) 原始几何成功率'), ('debiased_success', "(a′) 去偏成功率"),
           ('size_iou', '尺寸 IoU¹'), ('size_log_mae', '(b) 尺寸 log MAE [95% CI]²'),
           ('oor_axis', 'OOR 轴对齐¹'), ('oor_rotated', 'OOR 旋转（全部）'),
           ('oor_axis_strata', 'OOR 轴对齐（六层）³'), ('oor_rotated_strata', 'OOR 旋转（六层）³'),
           ('automatic_acceptance', '自动接受率'), ('bias_fp', '足迹偏差 B [95% CI]²'),
           ('bias_vol', '体积偏差 [95% CI]²'), ('cap_rule_flipped_complete', '上限规则翻转行')]
VALUES = [k for k, _ in COLUMNS[2:]]


def _read(path):
    return h.strict_json(Path(path).read_text())


def _bound(root, name):
    """One MANIFEST-listed file (large siblings stay unread); returns (value, MANIFEST sha256)."""
    root = Path(root)
    h.verify_artifacts(root, {name: _read(root / 'MANIFEST.json')['files'][name]})
    return _read(root / name), h.sha256(root / 'MANIFEST.json')


def load_stage(run, stage, report):
    """A stage's report_v2 output bound to its geometry summary and both inference run receipts."""
    run, report = Path(run), Path(report)
    e._verified(report)
    results = _read(report / 'results.json')
    if results.get('status') != 'REPORT_V2_COMPLETE':
        raise ValueError(f'{stage}: report_v2 output is not complete')
    sha = results['input_sha256']
    summary, geometry_sha = _bound(run / GEOMETRY[stage], 'summary.json')
    if geometry_sha != sha['geometry_MANIFEST']:
        raise ValueError(f'{stage}: report was not built on {GEOMETRY[stage]}')
    if (summary.get('status') != 'GEOMETRY_COMPLETE' or summary.get('stage', 'sft') != stage
            or summary.get('data_manifest_sha256') != sha['data_MANIFEST']
            or summary.get('inference_run_sha256') != sha['inference_run_sha256']):
        raise ValueError(f'{stage}: {GEOMETRY[stage]} is not this stage on this data and these runs')
    receipts = {}
    for arm in h.ARMS:
        path = run / f'infer-{arm}-{TAG[stage]}' / 'run_receipt.json'
        receipt = _read(path)
        identity = {k: v for k, v in receipt.items() if k != 'run_sha256'}
        if (i._hash(identity) != receipt.get('run_sha256') or receipt['run_sha256'] != sha['inference_run_sha256'][arm]
                or receipt.get('stage', 'sft') != stage or receipt['config'].get('arm') != arm
                or receipt['config'].get('lora') != LORA['sft' if stage == 'sft' else 'dpo']
                or receipt['data']['manifest_sha256'] != sha['data_MANIFEST']):
            raise ValueError(f'{stage}/{arm}: {path.parent.name} does not bind this report and stage')
        receipts[arm] = receipt
    return {'stage': stage, 'report': report, 'results': results, 'geometry': summary, 'receipts': receipts}


def check_shared(stages):
    """Same evaluation data, reference, bootstrap, scoring code, plan, decoding, base and training data."""
    first = stages[0]
    anchor = first['receipts']['joint']
    keys = ('data_MANIFEST', 'reference')
    for s in stages:
        r, f = s['results'], first['results']
        if (any(r['input_sha256'][k] != f['input_sha256'][k] for k in keys)
                or (r['repeats'], r['seed']) != (f['repeats'], f['seed'])
                or any(r['code_sha256'][k] != f['code_sha256'][k] for k in ('report_v2.py', 'geometry.py'))):
            raise ValueError(f"{s['stage']}: report differs from {first['stage']} in data, reference, bootstrap or code")
        for arm, receipt in s['receipts'].items():
            if (receipt['context_plan_sha256'] != anchor['context_plan_sha256']
                    or any(receipt['config'].get(k) != anchor['config'].get(k) for k in DECODE)
                    or receipt['base_sha256'] != anchor['base_sha256'] or receipt['training_data'] != anchor['training_data']):
                raise ValueError(f"{s['stage']}/{arm}: other plan, decoding, base or training data than {first['stage']}")


def lineage(run, arm, stages):
    """Merged-model receipts sft..max(stages), chained by hash; receipts only, no weights."""
    root, chain = Path(run) / DPO_ROOT / arm, {}
    for stage in STAGES[:max(STAGES.index(s) for s in stages) + 1]:
        path = root / f'{stage}-merged.json'
        receipt = _read(path)
        if (receipt.get('protocol'), receipt.get('arm'), receipt.get('stage')) != (d.MERGE_PROTOCOL, arm, stage):
            raise ValueError(f'lineage receipt is not {arm}/{stage}: {path}')
        if stage != 'sft':
            parent = chain[STAGES[STAGES.index(stage) - 1]]
            if (receipt['parent'].get('lineage_sha256') != parent['sha256']
                    or any(receipt[k] != parent['receipt'][k] for k in ('base', 'base_sha256', 'training_data'))):
                raise ValueError(f'{arm}/{stage}: lineage parent hash, base or training data mismatch')
        chain[stage] = {'receipt': receipt, 'sha256': h.sha256(path), 'path': str(path)}
    return chain


def bind(stage, chain, receipts):
    """The evaluated inference used exactly the receipt's adapter (SFT) or adapter on training base (DPO)."""
    for arm, receipt in receipts.items():
        link = chain[arm][stage]['receipt']
        adapter = i._hash(link['adapter']['files'])
        if receipt['base_sha256'] != link['base_sha256'] or receipt['training_data'] != link['training_data']:
            raise ValueError(f'{stage}/{arm}: inference base/training data differ from the lineage')
        if stage == 'sft':
            ok = (receipt.get('training_run_sha256') == link['training'].get('training_run_sha256')
                  and receipt.get('adapter_sha256') == adapter)
        else:
            policy = receipt.get('policy') or {}
            ok = (receipt.get('lineage_sha256') == chain[arm][stage]['sha256'] and policy.get('adapter_sha256') == adapter
                  and policy.get('training_base_sha256') == chain[arm][STAGES[STAGES.index(stage) - 1]]['receipt']['model_sha256'])
        if not ok:
            raise ValueError(f'{stage}/{arm}: evaluated policy is not the lineage receipt\'s model')


def pairs(run, stage, chain):
    summary, manifest = _bound(Path(run) / DPO_ROOT / d.PAIRS[stage], 'summary.json')
    for arm in h.ARMS:
        got = chain[arm][stage]['receipt']['training']['pairs']
        if got.get('manifest_sha256') != manifest or got.get('count') != summary['kept_both_arms']:
            raise ValueError(f'{stage}/{arm}: trained pairs are not {d.PAIRS[stage]}')
    return summary, manifest


def merge_fraction(receipt):
    """(overall changed fraction, per-module fractions, reason when unmeasured)."""
    diag = receipt.get('merge_diagnostic')
    if not isinstance(diag, dict) or 'modules' not in diag:
        reason = diag.get('error') if isinstance(diag, dict) else None
        return None, None, '未测：合并诊断失败（' + reason + '）' if reason else '未测：lineage 收据无 merge_diagnostic'
    modules = diag['modules']
    total = sum(m['total'] for m in modules.values())
    return (sum(m['changed'] for m in modules.values()) / total if total else None,
            {k: m['fraction'] for k, m in modules.items()}, None if total else '未测：诊断未比较任何权重')


def _pooled(rows, key):
    """Exact pooled mean of per-stratum means; each mean is over the stratum's complete rows."""
    kept = [r for r in rows if r[key] is not None and r['complete']]
    n = sum(r['complete'] for r in kept)
    return sum(r[key] * r['complete'] for r in kept) / n if n else None


def _finish(row, default):
    missing = row.pop('missing', {})
    row.update({k: row.get(k) for k in VALUES})
    row['missing'] = {k: missing.get(k, default) for k in VALUES if row[k] is None}
    return row


def arm_rows(stage):
    t = stage['results']['tables']
    t1 = {r['endpoint'].split()[0]: r for r in t['T1']}
    bias, t6 = ({r['system']: r for r in t[name]} for name in ('T1_bias', 'T6'))
    rows = []
    for arm, label in LABELS.items():
        overall = stage['geometry']['arms'][arm]['all']
        strata = [r for r in t['T4'] if r['system'] == label]
        rows.append(_finish({
            'stage': stage['stage'], 'arm': label, 'requests': t6[label]['requests'],
            'complete_rate': t6[label]['complete'] / t6[label]['requests'] if t6[label]['requests'] else None,
            'geometry_success': t1['(a)'][label], 'debiased_success': t1['(a′)'][label],
            'size_iou': overall['size_iou'], 'size_log_mae': t1['(b)'][label], 'oor_axis': overall['oor'],
            'oor_rotated': None, 'oor_axis_strata': _pooled(strata, 'oor'),
            'oor_rotated_strata': _pooled(strata, 'oor_rotated'), 'automatic_acceptance': None,
            'bias_fp': bias[label]['footprint'], 'bias_vol': bias[label]['volume'],
            'cap_rule_flipped_complete': t6[label]['cap_rule_flipped_complete'], 'missing': dict(REASONS)},
            '未测：输入中无此值'))
    return rows


def reference_rows(results):
    """Model-free rows from the frozen amendment-005 reference carried by every report (T2)."""
    t2 = {r['scope']: r for r in results['tables']['T2']}
    every, strata = t2['all'], t2['strata_pooled']
    gt = {'stage': '参考', 'arm': 'GT 布局（无模型）', 'requests': every['requests'],
          'geometry_success': every['gt_success'], 'oor_axis': every['gt_oor'], 'oor_rotated': every['gt_oor_rotated'],
          'oor_axis_strata': strata['gt_oor'], 'oor_rotated_strata': strata['gt_oor_rotated']}
    prior = {'stage': '参考', 'arm': 'P 先验尺寸（全部房间）', 'requests': every['requests'],
             'size_iou': every['prior_size_iou'], 'size_log_mae': every['prior_size_log_mae'],
             'bias_fp': every['prior_bias_fp'], 'bias_vol': every['prior_bias_vol']}
    return [_finish(gt, '不适用：GT 参考行不含该量'), _finish(prior, '不适用：先验行只含尺寸量')]


def differences(stage):
    t = stage['results']['tables']
    rows = [{'stage': stage['stage'], 'endpoint': r['endpoint'], 'delta': r['delta'], 'n': r['n'],
             'p': r.get('p'), 'p_holm': r.get('p_holm')} for r in t['T1']]
    rows += [{'stage': stage['stage'], 'endpoint': f"{r['field']}（房间微平均）", 'delta': r['micro'], 'n': r['micro']['n']}
             for r in t['T1b'] if r['field'] in T1B]
    return [{**r, 'missing': {k: '不适用：只对 (a)、(b) 计算 p 与 Holm' for k in ('p', 'p_holm') if r.get(k) is None}}
            for r in rows]


def lineage_rows(stage, chain, pair_summary):
    rows = []
    for arm, label in LABELS.items():
        receipt = chain[arm][stage]['receipt']
        fraction, modules, reason = merge_fraction(receipt)
        training = receipt['training']
        recipe = training['recipe']
        if isinstance(recipe, dict):
            recipe = (f"lr {recipe['lr']}, {recipe['epochs']} epochs, LoRA r{recipe['lora_r']}/α{recipe['lora_alpha']}, "
                      f"dropout {recipe.get('lora_dropout')}, β {recipe['beta']}, 有效 batch {recipe.get('effective_batch')}")
        completion = training.get('completion') or {}
        rows.append({'stage': stage, 'arm': label, 'evaluated_policy': POLICY['sft' if stage == 'sft' else 'dpo'],
                     'recipe': recipe, 'pairs': (training.get('pairs') or {}).get('count'),
                     'pairs_kept_both_arms': pair_summary['kept_both_arms'] if pair_summary else None,
                     'admitted_pairs': completion.get('admitted_pairs'), 'steps': completion.get('steps'),
                     'merge_changed_fraction': fraction, 'merge_modules': modules,
                     'merge_note': reason, 'model_sha256': receipt['model_sha256'],
                     'lineage_receipt_sha256': chain[arm][stage]['sha256'],
                     'missing': {**({k: '不适用：SFT 无偏好对' for k in ('pairs', 'admitted_pairs', 'steps',
                                                                         'pairs_kept_both_arms')} if stage == 'sft' else {}),
                                 **({'merge_changed_fraction': reason} if reason else {}),
                                 'merge_note': '不适用：诊断成功'}})
    return rows


def build(run, reports, out):
    run, out = Path(run), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if 'sft' not in reports or not set(reports) <= set(STAGES):
        raise ValueError('reports must map sft (required), dpo1, dpo2 to report_v2 dirs')
    stages = [load_stage(run, s, reports[s]) for s in STAGES if s in reports]
    check_shared(stages)
    chain = {arm: lineage(run, arm, list(reports)) for arm in h.ARMS}
    tables = {'stages': [], 'differences': [], 'decisions': [], 'lineage': [], 'pairs': []}
    tables['reference'] = reference_rows(stages[0]['results'])
    inputs, done = {}, {s['stage']: s for s in stages}
    for name in STAGES:
        if name not in done:
            tables['stages'] += [_finish({'stage': name, 'arm': label}, NOT_RUN) for label in LABELS.values()]
            tables['decisions'].append({'stage': name, 'note': NOT_RUN, 'missing': dict.fromkeys(
                ('rule_outcome', 'condition_i_size_log_mae', 'condition_ii_debiased_success'), NOT_RUN)})
            continue
        stage = done[name]
        bind(name, chain, stage['receipts'])
        summary = None
        if name != 'sft':
            summary, manifest = pairs(run, name, chain)
            tables['pairs'].append({'stage': name, 'pairs_dir': d.PAIRS[name], 'manifest_sha256': manifest,
                                    **{k: summary[k] for k in ('positives', 'kept_both_arms', 'verified_per_arm',
                                                               'dropped_by_other_arm_only', 'kept_by_source')}})
        tables['stages'] += arm_rows(stage)
        tables['differences'] += differences(stage)
        rule = stage['results']['delivery']
        tables['decisions'].append({'stage': name, 'rule_outcome': rule['delivered'],
                                    'condition_i_size_log_mae': rule['condition_i_size_log_mae'],
                                    'condition_ii_debiased_success': rule['condition_ii_debiased_success'],
                                    'note': '正式交付决定（修订 008）' if name == 'dpo2' else '仅供参考，非交付决定'})
        tables['lineage'] += lineage_rows(name, chain, summary)
        inputs[name] = {'report': str(stage['report']), 'report_MANIFEST': h.sha256(stage['report'] / 'MANIFEST.json'),
                        'results': h.sha256(stage['report'] / 'results.json'),
                        'geometry_MANIFEST': stage['results']['input_sha256']['geometry_MANIFEST'],
                        'inference_run_sha256': stage['results']['input_sha256']['inference_run_sha256'],
                        'lineage_receipts': {arm: chain[arm][name]['sha256'] for arm in h.ARMS}}
    warnings = sorted({w for s in stages for w in s['results'].get('warnings', [])})
    result = {'status': 'STAGE_TABLE_COMPLETE', 'stages_run': [s['stage'] for s in stages],
              'not_run': {s: NOT_RUN for s in STAGES if s not in done}, 'tables': tables,
              'context_plan_sha256': stages[0]['receipts']['joint']['context_plan_sha256'],
              'data_manifest_sha256': stages[0]['results']['input_sha256']['data_MANIFEST'],
              'repeats': stages[0]['results']['repeats'], 'seed': stages[0]['results']['seed'],
              'report_warnings': warnings, 'inputs': inputs,
              'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(R.__file__))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'stage_table.json', result)
        for name, rows in tables.items():
            if rows:
                R._csv(dest / f'stage_{name}.csv', [{**r, 'missing': '; '.join(f'{k}: {v}' for k, v in r.get('missing', {}).items())}
                                                    for r in rows])
        (dest / 'stage_table.md').write_text(_report(result))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return result


def _cell(row, key):
    value = row.get(key)
    estimate = isinstance(value, dict) and 'estimate' in value
    if value is None or (estimate and value['estimate'] is None):
        return (row.get('missing') or {}).get(key, '未测').split('：')[0]
    if estimate and key == 'delta':
        return R._cell(value).replace('[CI —]', '[CI 未测]')
    if estimate:
        ci = value.get('ci95')
        return f"{value['estimate']:.4f} " + (f'[{ci[0]:.4f}, {ci[1]:.4f}]' if ci else '[CI 未测]')
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return R._cell(value)


def _md(rows, columns):
    lines = ['| ' + ' | '.join(t for _, t in columns) + ' |', '|' + '---|' * len(columns)]
    return lines + ['| ' + ' | '.join(_cell(r, k) for k, _ in columns) + ' |' for r in rows] + ['']


def _report(result):
    t = result['tables']
    lines = ['# 未知尺寸 J／P：OptiScene 阶段消融（论文 Table 6 式）SFT → DPO1 → DPO2', '',
             f'J：联合预测尺寸、位置和朝向。{R.P_LABEL}：尺寸来自训练集先验，再生成位置和朝向。'
             '两臂同一 Qwen3-8B 基座、单一训练种子；全部阶段使用同一测试请求、同一共享推理计划'
             f"（context plan sha256 {result['context_plan_sha256'][:12]}…）和冻结几何代码。"
             f"区间为房屋连通组自举 {result['repeats']} 次（种子 {result['seed']}），不覆盖训练随机性。"
             '历史测试集曾用于诊断，不是新盲测。真人 UR 未测。', '',
             '**评测策略（修订 009）**：SFT 行来自 infer-v2-002，SFT LoRA 在 GPU 上运行时合并（merge_and_unload_bf16）；'
             'DPO1／DPO2 行评测精确训练策略——阶段适配器不合并、按训练时的 FP32 作用于 BF16 训练基座（上一阶段合并模型）。'
             '各阶段合并后的 BF16 模型（下一阶段基座、交付产物）改变的权重比例见“训练谱系”表。', '']
    if result['not_run']:
        lines += ['未运行阶段：' + '、'.join(result['not_run']) + '（' + NOT_RUN + '）。', '']
    lines += ['## 阶段消融主表', '']
    lines += _md(t['stages'], COLUMNS)
    lines += ['¹ 来自该阶段 geometry 汇总（全部完整输出的房间均值），未应用“达上限无 EOS 即失败”规则；'
              '受影响行数见“上限规则翻转行”，为 0 时与规则后数值一致。'
              '² 只在该阶段 J、P 共同完整的房间上计算（应用上限规则），区间为单臂水平值的自举区间。'
              '³ 论文六层合并：各层均值按该层完整输出数加权（精确合并，非插补），应用上限规则。'
              '(a)、(a′)、完整率以全部请求为分母。', '',
              '## 模型无关参考行（冻结，修订 005 §2）', '']
    lines += _md(t['reference'], COLUMNS[1:])
    lines += ['GT 是参考，不是上限。先验行为全部房间上的 P 先验尺寸误差。', '',
              '## 各阶段 J−P 配对差（房屋连通组自举）', '']
    lines += _md(t['differences'], [('stage', '阶段'), ('endpoint', '终点'), ('delta', 'Δ J−P [95% CI]'),
                                     ('n', '配对 n'), ('p', '自举双侧 p'), ('p_holm', 'Holm p（两共同主要终点）')])
    lines += ['只有 DPO2 的 (a)、(b) J−P 为确认性比较（Holm）；其它阶段与次要终点为描述性估计。', '',
              '## 修订 005 交付规则在各阶段的结果', '']
    lines += _md(t['decisions'], [('stage', '阶段'), ('rule_outcome', '规则结果'),
                                  ('condition_i_size_log_mae', '条件 (i)'), ('condition_ii_debiased_success', '条件 (ii)'),
                                  ('note', '说明')])
    lines += ['## 训练谱系：配方、偏好对、步数、合并诊断（lineage 收据）', '']
    lines += _md(t['lineage'], [('stage', '阶段'), ('arm', '系统'), ('evaluated_policy', '评测策略'), ('recipe', '配方'),
                                ('pairs', '训练偏好对'), ('admitted_pairs', '准入对'), ('steps', '优化步数'),
                                ('merge_changed_fraction', '合并改变的权重比例'), ('merge_note', '诊断说明')])
    lines += ['SFT 行的合并比例来自 sft-merged（DPO1 的训练基座）；SFT 评测本身用运行时 GPU 合并，未读取该文件。'
              'DPO 正样本为程序化几何验证的选择，不是论文的一致人类认可。', '']
    if t['pairs']:
        lines += ['## 偏好对汇总', '']
        lines += _md(t['pairs'], [('stage', '阶段'), ('positives', '正样本'), ('kept_both_arms', '两臂均保留'),
                                  ('verified_per_arm', '各臂验证通过'), ('dropped_by_other_arm_only', '仅因另一臂丢弃'),
                                  ('kept_by_source', '按来源')])
    reasons = sorted({(k, v) for r in t['stages'] for k, v in r['missing'].items()})
    lines += ['## 缺失值原因', ''] + [f'- {k}：{v}' for k, v in reasons] + ['- 参考行的“不适用”：该参考不含此量。', 
        '', f"report_v2 警告：{', '.join(result['report_warnings']) or '无'}。",
        'stage_table.json 保存全部数值、区间与输入哈希；MANIFEST.json 固定本输出的文件哈希。']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--report', action='append', required=True, metavar='STAGE=DIR')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    reports = dict(item.split('=', 1) for item in args.report)
    if len(reports) != len(args.report):
        raise ValueError('one --report per stage')
    result = build(args.run_root, reports, args.out)
    print(json.dumps({'status': result['status'], 'stages_run': result['stages_run']}))


if __name__ == '__main__':
    main()
