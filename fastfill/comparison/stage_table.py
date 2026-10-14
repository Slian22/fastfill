"""Amendments 008/009/010: paper-Table-6-style stage ablation SFT -> DPO1 -> DPO2 for J and P, the amendment
010 dev selection and the dev-selected models' test row; never imputes.

python -m fastfill.comparison.stage_table --run-root RUN --report sft=REPORT_V2_DIR \
    [--report dpo1=DIR] [--report dpo2=DIR] [--report selected=DIR] \
    [--selection RUN/dev-eval-010/selection.json] --out NEW_DIR

Reads only small JSON evidence under the run-root layout of the DPO chain (submit_dpo_stages.sh) and the
amendment 010 chain (submit_dev_selection.sh):
- each stage's report_v2 output dir (whole MANIFEST verified; results.json);
- RUN/geometry-v{2,3,4,5}/summary.json, hash-checked against its MANIFEST, whose sha256 must be the
  report's geometry_MANIFEST; status GEOMETRY_COMPLETE and stage label (absent = sft, as written by
  the pre-amendment-009 evaluate_pair);
- RUN/infer-{joint,placement}-00{2,3,4}/run_receipt.json: self-hash = the report's run hash, stage
  label, arm, LoRA mode (SFT: runtime GPU merge; DPO: exact policy, amendment 009), one shared
  context plan, decode settings, base and training data across every stage; for 'selected' each arm's
  dev-selected test run (dev_select.selected_dirs: infer-ARM-002 for SFT, infer-ARM-005 for a DPO
  candidate), also bound to the candidate's model exactly as dev_select.score binds dev runs (_bind);
- RUN/dpo-001/{arm}/{sft,dpo1,dpo2}-merged.json lineage receipts (chain hashes, merge_diagnostic) bound
  to the evaluated inference (SFT adapter / DPO adapter, training base and lineage hashes); a selected
  candidate's DPO stages come from its own run dir (dpo_stages.CANDIDATES: e1 dpo-002, rpo dpo-003),
  trained on dpo-001's SFT merge;
- RUN/dpo-001/stage{1,2}-pairs/summary.json (MANIFEST-checked) bound to each DPO receipt's pairs;
- --selection: dev_select's selection.json, bound to its scores dir (MANIFEST sha256 =
  scores_manifest_sha256) and reproduced by dev_select.choose; it renders the dev table (every candidate
  plus the report-only a009 rows) and is required for the selected row; --report-only-scores (amendment
  010 addendum 2) gives the separately scored a009 rows, bound to the same dev data, plan and SFT rows.
Every stage report must share data, reference, repeats, seed, geometry.py and report_v2.py (or two
report_v2.py versions in SAME_NUMBERS, audited to compute identical numbers).
A stage without a report is 'not run' (the amendment-009 DPO2 test run: stopped, amendment 010
addendum); a value the inputs do not carry is '未测' with its reason.
Outputs: stage_table.md (Chinese), one CSV per table, stage_table.json, MANIFEST.json.
"""
import argparse
import json
from pathlib import Path
import tempfile

from fastfill.comparison import dev_select as ds
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import infer as i
from fastfill.comparison import report_v2 as R
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

STAGES = d.STAGES  # Lineage chain stages.
ROWS = (*STAGES, 'selected')  # Table rows; selected = each arm's dev-selected model (amendment 010).
TAG = {'sft': '002', 'dpo1': '003', 'dpo2': '004'}
GEOMETRY = {'sft': 'geometry-v2', 'dpo1': 'geometry-v3', 'dpo2': 'geometry-v4', 'selected': ds.GEOMETRY}
DPO_ROOT = 'dpo-001'
LORA = {'sft': 'merge_and_unload_bf16', 'dpo': 'unmerged_fp32_stage_adapter_on_bf16_training_base'}
POLICY = {'sft': 'SFT LoRA 运行时 GPU 合并（merge_and_unload_bf16）',
          'dpo': '精确训练策略：未合并 FP32 适配器 + BF16 训练基座（修订 009）'}
DECODE = tuple(k for k in e.DECODE_KEYS if k != 'lora')
LABELS = {'joint': 'J', 'placement': 'P'}
ENDPOINTS = ('(a)', '(b)', '(a′)')
T1B = ('complete', 'geometry_success_repaired', 'oor', 'oor_repaired', 'size_iou')
NOT_RUN = '未运行：未提供该阶段的 report_v2 输出'
STOPPED = '已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析'
REPORT_ONLY = '仅报告：修订 010 补充，修订 009 阶段只在 dev 上评分，不可选'
UNSCORED = '未测：修订 010 补充 2，仅报告行单独评分，本表未提供其评分'
# protocol.md: matched SFT J0/P0 stays the primary endpoint; the dev-selected models decide delivery (amendment
# 010), secondary, never confirmatory; DPO1/DPO2 are descriptive. Holm in report_v2 runs within one stage's (a), (b).
ROLE = {'sft': (('(a)', '(b)'), '主要（确认性，protocol.md）'), 'selected': (ENDPOINTS, '次要（交付比较，修订 010）')}
# report_v2.py versions audited to compute identical numbers: the amendment-009 SFT/DPO1 reports' (snapshot
# main-281257f25) and the amendment-010 one; git diff 281257f25 between them changes report text only (T5 note,
# stage roles, the selected stage). A new report_v2.py must be re-audited before it joins this set.
SAME_NUMBERS = {'181bb2fe3928026e8f2241ed109e835e70f0d9a0c29328cbfa925e2674c254e4',
                '9bc9e0837d120ccf28325b897947c4fdfeb2c81e13d05adc30d47b9770ec3105'}
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
DEV_COLUMNS = [('arm', '系统'), ('candidate', '候选'), ('requests', 'dev 请求'), ('complete_rate', '完整率'),
               ('geometry_success', '(a) 原始几何成功率'), ('size_log_mae', 'log MAE（候选）'),
               ('sft_size_log_mae', 'log MAE（SFT，同一批房间）'), ('common_with_sft', '与 SFT 共同完整房间'),
               ('footprint_bias', '自身足迹偏差 B'), ('a_prime', "(a′) 自身 B 去偏成功率"), ('eligible', '合格'),
               ('selected', '选定')]


def _read(path):
    return h.strict_json(Path(path).read_text())


def _bound(root, name):
    """One MANIFEST-listed file (large siblings stay unread); returns (value, MANIFEST sha256)."""
    root = Path(root)
    h.verify_artifacts(root, {name: _read(root / 'MANIFEST.json')['files'][name]})
    return _read(root / name), h.sha256(root / 'MANIFEST.json')


def candidate_stage(candidate):
    """sft -> sft; RECIPE-dpoN (amendment 010) -> dpoN."""
    return candidate.rsplit('-', 1)[-1]


def load_selection(path, report_only=None):
    """dev_select's selection.json, bound to its scores dir and reproduced by dev_select.choose; report_only
    (amendment 010 addendum 2) = the separately scored a009 rows, bound to the same dev data, plan and SFT rows."""
    path = Path(path)
    value, scores_dir = _read(path), path.parent / 'scores'
    d._verified_dir(scores_dir)
    scores = _read(scores_dir / 'scores.json')
    if (value.get('protocol') != 'amendment-010-dev-selection' or value.get('rule') != ds.RULE
            or value.get('scores_manifest_sha256') != h.sha256(scores_dir / 'MANIFEST.json')
            or any(value.get(k) != scores[k] for k in ('data_manifest_sha256', 'uids_sha256'))):
        raise ValueError(f'{path} is not bound to its dev scores')
    for arm in h.ARMS:
        again, arm_value = ds.choose({c: scores['candidates'][arm][c] for c in ds.CANDIDATES}), value['arms'][arm]
        if ({k: arm_value[k] for k in again} != again or arm_value['candidates'] != scores['candidates'][arm]
                or value['selected'][arm] != again['selected'] or value['test'][arm]['candidate'] != again['selected']):
            raise ValueError(f'{arm}: selection.json does not reproduce from its dev scores')
    extra = {}
    if report_only:
        d._verified_dir(Path(report_only))
        other = _read(Path(report_only) / 'scores.json')
        if (other.get('only') != 'report-only' or other.get('protocol') != scores.get('protocol')
                or any(other.get(k) != scores.get(k) for k in ('data_manifest_sha256', 'uids_sha256', 'plan_sha256'))
                or any(other['candidates'][arm] != {'sft': scores['candidates'][arm]['sft']} for arm in h.ARMS)):
            raise ValueError(f'{report_only} is not scored on the same dev data, plan and SFT rows')
        extra = {'report_only': other['report_only'], 'report_only_sha256': h.sha256(Path(report_only) / 'MANIFEST.json')}
    return {'path': path, 'sha256': h.sha256(path), 'value': value, 'scores': scores, **extra}


def dev_rows(selection):
    """Every candidate's dev metrics as scored, eligibility and selection as dev_select.choose gave them."""
    rows = []
    for arm, label in LABELS.items():
        chosen = selection['value']['arms'][arm]
        for name in ds.CANDIDATES + ds.REPORT_ONLY:
            selectable = name in ds.CANDIDATES
            m = (selection['scores']['candidates'] if selectable else
                 selection.get('report_only') or selection['scores'].get('report_only', {}))[arm].get(name)
            if m is None:  # Amendment 010 addendum 2: report-only rows scored separately; not given = 未测.
                keys = ('requests', 'complete_rate', 'geometry_success', 'size_log_mae', 'sft_size_log_mae',
                        'common_with_sft', 'footprint_bias', 'a_prime', 'eligible', 'selected')
                rows.append({'arm': label, 'candidate': name, **dict.fromkeys(keys), 'missing': dict.fromkeys(keys, UNSCORED)})
                continue
            row = {'arm': label, 'candidate': name, 'requests': m['requests'], 'complete_rate': m['complete_rate'],
                   'geometry_success': m['geometry_success_rate'], 'size_log_mae': m['size_log_mae_common_with_sft'],
                   'sft_size_log_mae': m['sft_size_log_mae_common'], 'common_with_sft': m['common_with_sft'],
                   'footprint_bias': m['footprint_bias'], 'a_prime': m['a_prime'],
                   'eligible': name in chosen['eligible'] if selectable else None,
                   'selected': name == chosen['selected'] if selectable else None}
            rows.append({**row, 'missing': {k: REPORT_ONLY if k in ('eligible', 'selected') else '未测：无（与 SFT 共同）完整输出'
                                            for k, v in row.items() if v is None}})
    return rows


def load_stage(run, stage, report, selection=None):
    """A stage's report_v2 output bound to its geometry summary and both inference run receipts; for 'selected'
    (amendment 010, needs load_selection) each arm's dev-selected test run, bound to its candidate model."""
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
    picked = selection['value']['selected'] if stage == 'selected' and selection else None
    if stage == 'selected' and (not picked or sha.get('selection') != selection['sha256']
                                or summary.get('stages') != {arm: candidate_stage(picked[arm]) for arm in h.ARMS}):
        raise ValueError('selected: report and geometry do not bind this dev selection')
    dirs = ds.selected_dirs(run, picked) if picked else {arm: run / f'infer-{arm}-{TAG[stage]}' for arm in h.ARMS}
    receipts = {}
    for arm in h.ARMS:
        path, model = dirs[arm] / 'run_receipt.json', candidate_stage(picked[arm]) if picked else stage
        receipt = _read(path)
        identity = {k: v for k, v in receipt.items() if k != 'run_sha256'}
        if (i._hash(identity) != receipt.get('run_sha256') or receipt['run_sha256'] != sha['inference_run_sha256'][arm]
                or receipt.get('stage', 'sft') != model or receipt['config'].get('arm') != arm
                or receipt['config'].get('lora') != LORA['sft' if model == 'sft' else 'dpo']
                or receipt['data']['manifest_sha256'] != sha['data_MANIFEST']):
            raise ValueError(f'{stage}/{arm}: {path.parent.name} does not bind this report and stage')
        if picked:
            ds._bind(receipt, run, arm, picked[arm])  # The candidate's model, as for its dev run.
        receipts[arm] = receipt
    return {'stage': stage, 'report': report, 'results': results, 'geometry': summary, 'receipts': receipts,
            **({'candidates': picked} if picked else {})}


def check_shared(stages):
    """Same evaluation data, reference, bootstrap, scoring code, plan, decoding, base and training data."""
    first = stages[0]
    anchor = first['receipts']['joint']
    keys = ('data_MANIFEST', 'reference')
    for s in stages:
        r, f = s['results'], first['results']
        code = {r['code_sha256']['report_v2.py'], f['code_sha256']['report_v2.py']}
        if (any(r['input_sha256'][k] != f['input_sha256'][k] for k in keys)
                or (r['repeats'], r['seed']) != (f['repeats'], f['seed'])
                or r['code_sha256']['geometry.py'] != f['code_sha256']['geometry.py']
                or not (len(code) == 1 or code <= SAME_NUMBERS)):
            raise ValueError(f"{s['stage']}: report differs from {first['stage']} in data, reference, bootstrap or code")
        for arm, receipt in s['receipts'].items():
            if (receipt['context_plan_sha256'] != anchor['context_plan_sha256']
                    or any(receipt['config'].get(k) != anchor['config'].get(k) for k in DECODE)
                    or receipt['base_sha256'] != anchor['base_sha256'] or receipt['training_data'] != anchor['training_data']):
                raise ValueError(f"{s['stage']}/{arm}: other plan, decoding, base or training data than {first['stage']}")


def lineage(run, arm, stages, recipe=None):
    """Merged-model receipts sft..max(stages), chained by hash; receipts only, no weights. recipe (amendment 010
    candidate e1/rpo): its DPO stages come from its own run dir, trained on dpo-001's SFT merge."""
    chain = {}
    for stage in STAGES[:max(STAGES.index(s) for s in stages) + 1]:
        path = Path(run) / (d.CANDIDATES[recipe][0] if recipe and stage != 'sft' else DPO_ROOT) / arm / f'{stage}-merged.json'
        receipt = _read(path)
        if ((receipt.get('protocol'), receipt.get('arm'), receipt.get('stage')) != (d.MERGE_PROTOCOL, arm, stage)
                or (stage != 'sft' and receipt['training'].get('candidate') != recipe)):
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
    for arm in chain:
        got = chain[arm][stage]['receipt']['training']['pairs']
        if got.get('manifest_sha256') != manifest or got.get('count') != summary['kept_both_arms']:
            raise ValueError(f'{stage}/{arm}: trained pairs are not {d.PAIRS[stage]}')
    return summary, manifest


def selected_lineage(run, stage):
    """Amendment 010: each arm's candidate chain (dpo-001 SFT merge, then the candidate's own DPO stages) bound to
    its evaluated policy and its DPO stages' pairs; -> (chain, {arm: {dpo stage: pair summary}})."""
    chain, kept = {}, {}
    for arm, candidate in stage['candidates'].items():
        name = candidate_stage(candidate)
        chain[arm] = lineage(run, arm, [name], None if name == 'sft' else candidate.split('-')[0])
        bind(name, {arm: chain[arm]}, {arm: stage['receipts'][arm]})
        kept[arm] = {s: pairs(run, s, {arm: chain[arm]})[0] for s in STAGES[1:STAGES.index(name) + 1]}
    return chain, kept


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


def _arm_stage(stage, arm):
    """The model stage evaluated for this arm: the row's stage, or the dev-selected candidate's (amendment 010)."""
    return candidate_stage(stage['candidates'][arm]) if 'candidates' in stage else stage['stage']


def _label(stage, arm):
    return f"selected（{stage['candidates'][arm]}）" if 'candidates' in stage else stage['stage']


def arm_rows(stage):
    t = stage['results']['tables']
    t1 = {r['endpoint'].split()[0]: r for r in t['T1']}
    bias, t6 = ({r['system']: r for r in t[name]} for name in ('T1_bias', 'T6'))
    rows = []
    for arm, label in LABELS.items():
        overall = stage['geometry']['arms'][arm]['all']
        strata = [r for r in t['T4'] if r['system'] == label]
        rows.append(_finish({
            'stage': _label(stage, arm), 'arm': label, 'requests': t6[label]['requests'],
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
    scope, role = ROLE.get(stage['stage'], ((), None))
    return [{**r, 'role': role if r['endpoint'].split(' ')[0] in scope else '描述性',
             'missing': {k: '不适用：只对 (a)、(b) 计算 p 与 Holm' for k in ('p', 'p_holm') if r.get(k) is None}}
            for r in rows]


def lineage_rows(stage, chain, pair_summaries):
    """stage: a loaded stage; chain and pair_summaries ({arm: summary or None}) per arm."""
    rows = []
    for arm, label in LABELS.items():
        name, pair_summary = _arm_stage(stage, arm), pair_summaries[arm]
        receipt = chain[arm][name]['receipt']
        fraction, modules, reason = merge_fraction(receipt)
        training = receipt['training']
        recipe = training['recipe']
        if isinstance(recipe, dict):
            recipe = (f"lr {recipe['lr']}, {recipe['epochs']} epochs, LoRA r{recipe['lora_r']}/α{recipe['lora_alpha']}, "
                      f"dropout {recipe.get('lora_dropout')}, β {recipe['beta']}, 有效 batch {recipe.get('effective_batch')}")
        completion = training.get('completion') or {}
        rows.append({'stage': _label(stage, arm), 'arm': label, 'evaluated_policy': POLICY['sft' if name == 'sft' else 'dpo'],
                     'recipe': recipe, 'pairs': (training.get('pairs') or {}).get('count'),
                     'pairs_kept_both_arms': pair_summary['kept_both_arms'] if pair_summary else None,
                     'admitted_pairs': completion.get('admitted_pairs'), 'steps': completion.get('steps'),
                     'merge_changed_fraction': fraction, 'merge_modules': modules,
                     'merge_note': reason, 'model_sha256': receipt['model_sha256'],
                     'lineage_receipt_sha256': chain[arm][name]['sha256'],
                     'missing': {**({k: '不适用：SFT 无偏好对' for k in ('pairs', 'admitted_pairs', 'steps',
                                                                         'pairs_kept_both_arms')} if name == 'sft' else {}),
                                 **({'merge_changed_fraction': reason} if reason else {}),
                                 'merge_note': '不适用：诊断成功'}})
    return rows


def build(run, reports, out, selection=None, report_only=None):
    run, out = Path(run), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if 'sft' not in reports or not set(reports) <= set(ROWS):
        raise ValueError('reports must map sft (required), dpo1, dpo2, selected to report_v2 dirs')
    if 'selected' in reports and selection is None:
        raise ValueError('the selected row needs --selection (amendment 010)')
    picked = load_selection(selection, report_only) if selection else None
    stages = [load_stage(run, s, reports[s], picked) for s in ROWS if s in reports]
    check_shared(stages)
    chain = {arm: lineage(run, arm, [s for s in reports if s in STAGES]) for arm in h.ARMS}
    tables = {'stages': [], 'differences': [], 'decisions': [], 'lineage': [], 'pairs': [],
              'dev': dev_rows(picked) if picked else []}
    tables['reference'] = reference_rows(stages[0]['results'])
    inputs, done = {}, {s['stage']: s for s in stages}
    not_run = {name: STOPPED if name == 'dpo2' else NOT_RUN for name in ROWS if name not in done}
    for name in ROWS:
        if name not in done:
            tables['stages'] += [_finish({'stage': name, 'arm': label}, not_run[name]) for label in LABELS.values()]
            tables['decisions'].append({'stage': name, 'note': not_run[name], 'missing': dict.fromkeys(
                ('rule_outcome', 'condition_i_size_log_mae', 'condition_ii_debiased_success'), not_run[name])})
            continue
        stage = done[name]
        if name == 'selected':
            links, kept = selected_lineage(run, stage)
            summaries = {arm: kept[arm].get(_arm_stage(stage, arm)) for arm in h.ARMS}
        else:
            links, summary = chain, None
            bind(name, chain, stage['receipts'])
            if name != 'sft':
                summary, manifest = pairs(run, name, chain)
                tables['pairs'].append({'stage': name, 'pairs_dir': d.PAIRS[name], 'manifest_sha256': manifest,
                                        **{k: summary[k] for k in ('positives', 'kept_both_arms', 'verified_per_arm',
                                                                   'dropped_by_other_arm_only', 'kept_by_source')}})
            summaries = dict.fromkeys(h.ARMS, summary)
        tables['stages'] += arm_rows(stage)
        tables['differences'] += differences(stage)
        rule = stage['results']['delivery']
        tables['decisions'].append({'stage': name, 'rule_outcome': rule['delivered'],
                                    'condition_i_size_log_mae': rule['condition_i_size_log_mae'],
                                    'condition_ii_debiased_success': rule['condition_ii_debiased_success'],
                                    'note': '正式交付决定（修订 010：两臂 dev 选定模型）' if name == 'selected' else '仅供参考，非交付决定'})
        tables['lineage'] += lineage_rows(stage, links, summaries)
        inputs[name] = {'report': str(stage['report']), 'report_MANIFEST': h.sha256(stage['report'] / 'MANIFEST.json'),
                        'results': h.sha256(stage['report'] / 'results.json'),
                        'geometry_MANIFEST': stage['results']['input_sha256']['geometry_MANIFEST'],
                        'inference_run_sha256': stage['results']['input_sha256']['inference_run_sha256'],
                        'lineage_receipts': {arm: links[arm][_arm_stage(stage, arm)]['sha256'] for arm in h.ARMS},
                        **({'candidates': stage['candidates']} if 'candidates' in stage else {})}
    warnings = sorted({w for s in stages for w in s['results'].get('warnings', [])})
    dev = picked and {'selection': str(picked['path']), 'selection_sha256': picked['sha256'], 'rule': ds.RULE,
                      'rooms': picked['scores']['candidates']['joint']['sft']['requests'], 'selected': picked['value']['selected'],
                      **{k: picked['value'][k] for k in ('scores_manifest_sha256', 'data_manifest_sha256', 'uids_sha256')},
                      **({'report_only_scores_manifest_sha256': picked['report_only_sha256']} if 'report_only_sha256' in picked else {})}
    result = {'status': 'STAGE_TABLE_COMPLETE', 'stages_run': [s['stage'] for s in stages],
              'not_run': not_run, 'tables': tables, **({'dev_selection': dev} if dev else {}),
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
        return (row.get('missing') or {}).get(key, '未测').rsplit('：', 1)[0]
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
    lines = ['# 未知尺寸 J／P：OptiScene 阶段消融（论文 Table 6 式）SFT → DPO1 → DPO2，及修订 010 dev 选定模型', '',
             f'J：联合预测尺寸、位置和朝向。{R.P_LABEL}：尺寸来自训练集先验，再生成位置和朝向。'
             '两臂同一 Qwen3-8B 基座、单一训练种子；全部阶段使用同一测试请求、同一共享推理计划'
             f"（context plan sha256 {result['context_plan_sha256'][:12]}…）和冻结几何代码。"
             f"区间为房屋连通组自举 {result['repeats']} 次（种子 {result['seed']}），不覆盖训练随机性。"
             '历史测试集曾用于诊断，不是新盲测。真人 UR 未测。', '',
             '**评测策略（修订 009）**：SFT 行来自 infer-v2-002，SFT LoRA 在 GPU 上运行时合并（merge_and_unload_bf16）；'
             'DPO1／DPO2 行评测精确训练策略——阶段适配器不合并、按训练时的 FP32 作用于 BF16 训练基座（上一阶段合并模型）。'
             '各阶段合并后的 BF16 模型（下一阶段基座、交付产物）改变的权重比例见“训练谱系”表。'
             'DPO1／DPO2 行为修订 009（官方超参，dpo-001）。selected 行为两臂各自在 dev 上选定的模型（修订 010，行标签带候选名）：'
             '选中 SFT 时即 infer-v2-002 的 SFT 测试推理；选中 DPO 候选（e1 = dpo-002，rpo = dpo-003）时为其精确训练策略的测试推理 infer-v2-005。', '']
    if result['not_run']:
        lines += ['未运行或已停止的阶段：' + '；'.join(f'{k}（{v}）' for k, v in result['not_run'].items()) + '。', '']
    lines += ['## 阶段消融主表', '']
    lines += _md(t['stages'], COLUMNS)
    lines += ['¹ 来自该阶段 geometry 汇总（全部完整输出的房间均值），未应用“达上限无 EOS 即失败”规则；'
              '受影响行数见“上限规则翻转行”，为 0 时与规则后数值一致。'
              '² 只在该阶段 J、P 共同完整的房间上计算（应用上限规则），区间为单臂水平值的自举区间。'
              '³ 论文六层合并：各层均值按该层完整输出数加权（精确合并，非插补），应用上限规则。'
              '(a)、(a′)、完整率以全部请求为分母。', '',
              '## 模型无关参考行（冻结，修订 005 §2）', '']
    lines += _md(t['reference'], COLUMNS[1:])
    lines += ['GT 是参考，不是上限。先验行为全部房间上的 P 先验尺寸误差。', '']
    if 'dev_selection' in result:
        v = result['dev_selection']
        lines += ['## 修订 010 dev 选择（只用 dev，不用测试数据）', '',
                  f"dev 集 {v['rooms']} 间房（uids sha256 {v['uids_sha256'][:12]}…）。selection.json sha256 "
                  f"{v['selection_sha256'][:12]}…，绑定 dev 评分 MANIFEST {v['scores_manifest_sha256'][:12]}…，"
                  '并由 dev_select.choose 在生成本表时重算一致。', '', f"选择规则（dev_select.RULE）：{v['rule']}", '']
        lines += _md(t['dev'], DEV_COLUMNS)
        lines += ['两列 log MAE 在候选与 SFT 均完整的同一批房间上计算（修订 005 的共同完整配对）；B 为候选自身完整房间上的足迹偏差；'
                  "(a)、(a′) 与完整率以全部 dev 请求为分母。a009-dpo1／a009-dpo2 为修订 009 阶段，按修订 010 补充只在 dev 上评分、仅作报告，不参与选择。"
                  '选定：' + '；'.join(f'{LABELS[arm]} = {c}' for arm, c in v['selected'].items()) + '。', '']
    lines += ['## 各阶段 J−P 配对差（房屋连通组自举）', '']
    lines += _md(t['differences'], [('stage', '阶段'), ('endpoint', '终点'), ('role', '比较地位'),
                                     ('delta', 'Δ J−P [95% CI]'), ('n', '配对 n'), ('p', '自举双侧 p'),
                                     ('p_holm', 'Holm p（同阶段 (a)、(b) 内）')])
    lines += ['SFT 阶段 (a)、(b) J−P 为方案保留的主要（确认性）比较（Holm，protocol.md）；'
              'selected（两臂各自 dev 选定的模型）J−P 决定交付（修订 010；规则用 (b) 与 (a′)），属次要比较；'
              'DPO1、DPO2（修订 009）与其它终点为描述性估计。'
              'Holm 只在同一阶段的 (a)、(b) 两终点内校正。', '',
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
    reasons = sorted({(k, v) for r in t['stages'] + t['dev'] for k, v in r['missing'].items()})
    lines += ['## 缺失值原因', ''] + [f'- {k}：{v}' for k, v in reasons] + ['- 参考行的“不适用”：该参考不含此量。', 
        '', f"report_v2 警告：{', '.join(result['report_warnings']) or '无'}。",
        'stage_table.json 保存全部数值、区间与输入哈希；MANIFEST.json 固定本输出的文件哈希。']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--report', action='append', required=True, metavar='STAGE=DIR')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path)  # Amendment 010: RUN/dev-eval-010/selection.json.
    parser.add_argument('--report-only-scores', type=Path)  # Addendum 2: RUN/dev-eval-010/scores-report-only.
    args = parser.parse_args(argv)
    reports = dict(item.split('=', 1) for item in args.report)
    if len(reports) != len(args.report):
        raise ValueError('one --report per stage')
    result = build(args.run_root, reports, args.out, args.selection, args.report_only_scores)
    print(json.dumps({'status': result['status'], 'stages_run': result['stages_run']}))


if __name__ == '__main__':
    main()
