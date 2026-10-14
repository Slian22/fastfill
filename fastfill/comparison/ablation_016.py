"""Amendment 016 section 3: DPO2 attribution ablation on dev (single-shot greedy, no repair; EXPLORATORY).

python -m fastfill.comparison.ablation_016 STEP ...   (sbatch: isambard_016.sbatch; chain: submit_016_ablation.sh)

train  --dpo-root RUN/dpo-005 --candidate rft  |  --dpo-root RUN/dpo-006 --candidate pert   (4 GPUs; a rerun resumes)
    Both start from rpo-dpo1's exact policy exactly as onpolicy_dpo train: training base dpo-001/placement/sft-merged
    + --init_adapter dpo-003/placement/dpo1/final (dpo_stages.exact_policy rehashes both), LoRA r32/a16, lr 5e-6,
    effective batch 4, seed 42, onpolicy_dpo.RECIPE's 2 epochs.
      rft:  train_dpo --sft_only (chosen mean token NLL; no DPO term, no reference pass) on
            dpo-004/stage2-pairs/placement.jsonl verbatim (the N pairs op-dpo2 trained on: same rows, order and repeats,
            so the DistributedSampler (seed 42, same N) replays op-dpo2's batches and (b) differs from (d) only by the
            DPO term; N must equal the op-dpo2 receipt's pair count, so 2 x ceil(N/4) = S steps).
      pert: DPO + RPO alpha 1.0 (onpolicy_dpo.RECIPE) on dpo-001/stage2-pairs/placement.jsonl (amendment 009: GT chosen,
            synthetic collision_shift / wall_oob rejected); one epoch = 4 x S/2 rows: the unique pairs in ascending
            sha256('ablation-016-v1:pert:' + key) order, cycled (a pool larger than an epoch is a hash-ranked subset).
    Steps: op-dpo2 e2's optimiser steps S, read from the dpo-004/placement/dpo2-merged lineage receipt (missing = raise:
    the ablation runs after amendment-013 training); 2 epochs are exactly S steps. The rows are published new-only as
    dpo-00X/train-rows (a rerun must rebuild the same bytes). Lineage: dpo-00X/placement/dpo2-merged receipt as
    onpolicy_dpo's (parent dpo-003 dpo1-merged, training_base dpo-001 sft-merged with is_dpo1_merge false, init adapter
    sha256, steps_matched_to), so infer_shard --exact-policy loads training base + the final adapter.
score  --data RUN/dev-select-v1 --run-root RUN --out RUN/dev-eval-016/ablation.json [--workers 30]
    (a) rpo-dpo1 and (d) op-dpo2-e2 rescored by onpolicy_dpo from dev-eval-010 / dev-eval-013 (dev-eval-010's and
    dev-eval-013's scores of record must reproduce); (b) rft and (c) pert from dev-eval-016/CAND/infer-placement (exact
    policy of dpo-00X/placement/dpo2-merged), all on the shared dev-eval-010 plan; (b), (c) and (d) must have the same
    optimiser steps. Per candidate onpolicy_dpo._finish: complete, a', downstream acceptance over all 600 rooms
    (incomplete = not accepted), checker success, size log MAE, dev halves. Paired house-cluster bootstrap
    (report_v2.contrasts, 2,000 draws, seed 42), first minus second: b-a, c-a, d-a, d-b, d-c, c-b. INTERPRETATION is
    applied as fixed in amendment 016. Writes ablation.json (with compact per-room rows) and ablation-report.md
    (Chinese) next to it, new-only. Test data is never touched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from fastfill.comparison import dev_select as ds
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h

ARM, STAGE, REFERENCE, OP = op.ARM, op.STAGE, op.REFERENCE, 'op-dpo2-e2'
RUNS = {'rft': 'dpo-005', 'pert': 'dpo-006'}
SOURCES = {'rft': (op.RUN, ('prompt', 'chosen')), 'pert': (d.SOURCE_RUN, ('prompt', 'chosen', 'rejected'))}
RECIPES = {'rft': {**{k: v for k, v in op.RECIPE.items() if k != 'rpo_alpha'}, 'sft_only': True}, 'pert': op.RECIPE}
LOSS = {'rft': ("continued SFT: mean token NLL of the chosen response only (train_dpo --sft_only; no DPO term, no "
                "reference pass, rejected unused); the policy LoRA starts as rpo-dpo1's adapter (--init_adapter) on its "
                "training base"),
        'pert': op.LOSS + '; RPO alpha 1.0; pairs: amendment-009 GT chosen vs synthetic perturbation rejected'}
SALT, EVAL, REPORT = 'ablation-016-v1:', 'dev-eval-016', 'ablation-report.md'
ORDER = (REFERENCE, 'rft', 'pert', OP)
LABELS = {REFERENCE: '(a) rpo-dpo1', 'rft': '(b) RFT', 'pert': '(c) pert-DPO2', OP: '(d) op-dpo2 e2'}
PAIRS = (('rft', REFERENCE), ('pert', REFERENCE), (OP, REFERENCE), (OP, 'rft'), (OP, 'pert'), ('pert', 'rft'))
FIELDS = (*op.FIELDS, 'size_log_mae')
ROW_KEYS = ('uid', 'source', 'group', 'complete', 'geometry_success', 'success_debiased', 'accepted', 'size_log_mae')
PROTOCOL = 'amendment-016-dpo2-attribution-ablation'
INTERPRETATION = (
    "fixed in advance (amendment 016 section 3), on a' (success_debiased) and downstream acceptance (accepted): "
    "(d) - (b) with both 95% CIs above 0 -> the preference construction has value beyond its repaired data; "
    "(d) ~ (b), operationalised as both CIs of (d) - (b) including 0 (not an equivalence test) -> the contribution is "
    "the repair-based data; (c) ~ (d), both CIs of (d) - (c) including 0 -> real failures add nothing over synthetic "
    "ones; any other pattern has no pre-registered reading and is reported as measured")
READINGS = {'preference_beyond_repaired_data': '偏好构造在其修复数据之外仍有价值',
            'contribution_is_repair_data': '贡献来自基于修复的数据（(d) 与 (b) 无法区分）',
            'real_failures_add_nothing_over_synthetic': '真实失败相对合成扰动没有额外贡献（(c) 与 (d) 无法区分）',
            'not_pre_registered': '不属于预先固定的情形，按测得结果报告，不作解读'}
SIDES = {'above_zero': 'CI 在 0 之上', 'below_zero': 'CI 在 0 之下', 'includes_zero': 'CI 含 0', 'undefined': 'CI 未定义'}


def model_path(run_root, name):
    return Path(run_root) / RUNS[name] / ARM / f'{STAGE}-merged'


# ---- train ---------------------------------------------------------------------------------------

def matched_steps(run_root):
    """op-dpo2 e2's optimiser steps from its dpo-004 lineage receipt; missing = amendment-013 training not done."""
    path = d.receipt_path((Path(run_root) / op.RUN / ARM / op.MODELS[OP]).resolve())
    if not path.exists():
        raise ValueError(f'no op-dpo2 training receipt {path}: the ablation starts after amendment-013 training')
    training = d._lineage(path, ARM, STAGE)['training']
    completion = Path(run_root) / op.RUN / ARM / STAGE / 'final' / 'completion.json'
    if (training.get('candidate') != op.CANDIDATE or 'epoch' in training
            or training['recipe']['epochs'] != op.RECIPE['epochs'] or training['completion']['steps'] % op.RECIPE['epochs']
            or h.sha256(completion) != training['completion']['sha256']):
        raise ValueError(f'{path} is not the completed op-dpo2 epoch-2 training')
    return {'run': op.RUN, 'candidate': OP, 'steps': training['completion']['steps'], 'epochs': op.RECIPE['epochs'],
            'lineage_sha256': h.sha256(path), 'pairs_manifest_sha256': training['pairs']['manifest_sha256'],
            'pairs_count': training['pairs']['count']}


def rank(name, key):
    return hashlib.sha256((SALT + name + ':' + key).encode('utf-8')).hexdigest()


def _key(pair, name):
    return json.dumps([pair[k] for k in SOURCES[name][1]], ensure_ascii=False, sort_keys=True)


def take(pairs, name, count):
    """Unique items (first occurrence per key) in hash order, cycled to COUNT rows -> (rows, info); pure (pert only)."""
    unique = {}
    for pair in pairs:
        unique.setdefault(_key(pair, name), pair)
    if not unique or count < 1:
        raise ValueError(f'{name}: nothing to train on')
    ordered = [unique[k] for k in sorted(unique, key=lambda k: (rank(name, k), k))]
    return [ordered[k % len(ordered)] for k in range(count)], {
        'source_rows': len(pairs), 'unique': len(ordered), 'rows_per_epoch': count,
        'full_cycles': count // len(ordered), 'extra_rows': count % len(ordered)}


def _source(run_root, name, parent, matched):
    """The verified source pairs of NAME -> (rows, receipt)."""
    pairs_dir = Path(run_root) / SOURCES[name][0] / 'stage2-pairs'
    files = d._verified_dir(pairs_dir)
    path = (pairs_dir / f'{ARM}.jsonl').resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != h._expected_hash(files[path.name]):
        raise ValueError(f'{path} changed since its manifest')
    rows = [h.strict_json(line) for line in payload.decode('utf-8').splitlines()]  # op pairs repeat UIDs.
    manifest = h.sha256(pairs_dir / 'MANIFEST.json')
    if name == 'rft':
        summary = h.strict_json((pairs_dir / 'summary.json').read_text())
        if (summary.get('status') != 'OP_PAIRS_COMPLETE' or manifest != matched['pairs_manifest_sha256']
                or summary['inputs']['reference_lineage_sha256'] != h.sha256(d.receipt_path(parent))):
            raise ValueError(f"{pairs_dir} are not the pairs op-dpo2 was trained on")
    elif any(r.get('reject_kind') not in ('collision_shift', 'wall_oob') for r in rows):
        raise ValueError(f'{pairs_dir} are not amendment-009 synthetic perturbation pairs')
    return rows, {'path': str(path), 'sha256': h._expected_hash(files[path.name]), 'manifest_sha256': manifest}


def train(args):
    root, name = Path(args.dpo_root), args.candidate
    if root.name != RUNS[name]:
        raise ValueError(f'candidate {name} trains in {RUNS[name]}, not {root.name}')
    run_root = root.parent
    parent = (run_root / op.PARENT_RUN / ARM / 'dpo1-merged').resolve()
    out, merged = root / ARM / STAGE, model_path(run_root, name)
    if d.receipt_path(merged).exists():
        return d.verify_merged(merged, ARM, STAGE)
    lineage, policy = d.exact_policy(parent, ARM)  # Rehashes rpo-dpo1's training base and adapter.
    if lineage['stage'] != 'dpo1' or lineage['training'].get('candidate') != 'rpo':
        raise ValueError('the ablation start must be the dev-selected rpo-dpo1 (dpo-003)')
    base, init = Path(policy['training_base']), Path(policy['adapter'])
    matched, recipe = matched_steps(run_root), RECIPES[name]
    source, source_info = _source(run_root, name, parent, matched)
    if name == 'rft':  # op-dpo2's pairs verbatim: the same N rows in file order, so only the DPO term differs.
        count, rows = len(source), source
        info = {'source_rows': count, 'unique': len({_key(r, name) for r in source}), 'rows_per_epoch': count,
                'verbatim': True}
        order = 'the op-dpo2 pairs file verbatim (file order; train_dpo DistributedSampler, seed 42, as op-dpo2)'
        if count != matched['pairs_count']:
            raise ValueError(f"rft: {count} pairs, op-dpo2 was trained on {matched['pairs_count']}")
    else:
        count = d.WORLD_SIZE * recipe['bs'] * recipe['grad_accum'] * (matched['steps'] // recipe['epochs'])
        rows, info = take(source, name, count)
        order = f"ascending sha256('{SALT}' + candidate + ':' + json {list(SOURCES[name][1])}), cycled"
    payload, rows_dir = d._jsonl(rows), root / 'train-rows'
    if name == 'rft' and hashlib.sha256(payload).hexdigest() != source_info['sha256']:
        raise ValueError(f"rft rows are not {source_info['path']} byte for byte")
    if not os.path.lexists(rows_dir):
        selection = {'protocol': 'amendment-016-ablation-rows-v1', 'candidate': name, 'source': source_info, **info,
                     'order': order, 'steps_matched_to': matched}
        d._publish_dir(rows_dir, {f'{ARM}.jsonl': payload, 'selection.json': op._json_bytes(selection)})
    files = d._verified_dir(rows_dir)
    path = (rows_dir / f'{ARM}.jsonl').resolve()
    if path.read_bytes() != payload:
        raise ValueError(f'{rows_dir} holds other rows than this rule rebuilds')
    from fastfill import merge_lora
    from fastfill.v1_1 import train_dpo, training_guard
    settings = {**recipe, **d.TRAIN_FIXED, 'loss': LOSS[name],
                'init_adapter': {'path': str(init), 'sha256': policy['adapter_sha256'], 'of': 'rpo-dpo1'}}
    training_guard.guard_training(base, path, out, settings, [Path(train_dpo.__file__), Path(d.__file__), Path(op.__file__),
                                                              Path(__file__), Path(merge_lora.__file__)])
    if not (out / 'final' / 'completion.json').exists():
        d.preflight(base, path, count)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={d.WORLD_SIZE}',
                   '-m', 'fastfill.v1_1.train_dpo', '--model', str(base), '--pairs', str(path), '--out', str(out),
                   '--init_adapter', str(init)]
        subprocess.run(command + d._flags(recipe), check=True)
    completion = d._completion(out, count, recipe)
    if completion['steps'] != matched['steps']:
        raise ValueError(f"{name} ran {completion['steps']} optimiser steps, op-dpo2 e2 {matched['steps']}")
    d._merge(base, out / 'final', merged, {
        'arm': ARM, 'stage': STAGE,
        'parent': {'stage': 'dpo1', 'path': str(parent), 'lineage_sha256': h.sha256(d.receipt_path(parent))},
        'training_base': {'stage': 'sft', 'path': str(base), 'model_sha256': policy['training_base_sha256'],
                          'lineage_sha256': h.sha256(d.receipt_path(base)), 'is_dpo1_merge': False},
        'merged_model': op.MERGED_MODEL, **{k: lineage[k] for k in ('base', 'base_sha256', 'training_data')},
        'training': {'pairs': {'path': str(path), 'count': count, 'sha256': h._expected_hash(files[path.name]),
                               'manifest_sha256': h.sha256(rows_dir / 'MANIFEST.json')},
                     'rows': {'source': source_info, **info}, 'steps_matched_to': matched,
                     'guard_sha256': h.sha256(out / 'training_guard.json'), 'completion': completion, 'recipe': settings,
                     'candidate': name, 'ablation_016_sha256': h.sha256(__file__)}})
    return d.verify_merged(merged, ARM, STAGE)


# ---- score ---------------------------------------------------------------------------------------

def training(run_root):
    """(b), (c) and (d) lineage receipts (receipts only, no weights): the same optimiser steps, from rpo-dpo1."""
    matched, result = matched_steps(run_root), {}
    for name in RUNS:
        path = d.receipt_path(model_path(run_root, name).resolve())
        t = d._lineage(path, ARM, STAGE)['training']
        if t.get('candidate') != name or t['completion']['steps'] != matched['steps']:
            raise ValueError(f'{path} is not a step-matched {name} run')
        result[name] = {'lineage_sha256': h.sha256(path), 'steps': t['completion']['steps'], 'rows': t['rows'],
                        'loss': t['recipe']['loss'], 'init_adapter': t['recipe']['init_adapter']}
    return {**result, OP: matched}


def _scored(run_root, name, data, manifest, rooms, requests, workers):
    """onpolicy_dpo._scored for the ablation runs: dev-eval-016/NAME, exact policy of dpo-00X/placement/dpo2-merged."""
    path = Path(run_root) / EVAL / name / f'infer-{ARM}'
    records, receipt = e._inference(path, data, manifest['expected_uids'])
    config = receipt['config']
    if (receipt['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']
            or config.get('lora') != ds.EXACT or receipt.get('stage') != STAGE or config.get('arm') != ARM
            or config.get('model') != str(model_path(run_root, name).resolve())):
        raise ValueError(f'dev inference is not {ARM}/{name}')
    rows, caps = op._scored_rows(records, rooms, requests, workers)
    return records, receipt, rows, {'inference': {
        'path': str(path.resolve()), 'run_sha256': receipt['run_sha256'], 'manifest_sha256': h.sha256(path / 'MANIFEST.json'),
        'stage': receipt['stage'], 'lora': config['lora'], 'policy': receipt.get('policy')}, 'token_cap_failures': caps}


def _side(contrast):
    ci = contrast['ci95']
    return 'undefined' if ci is None else 'above_zero' if ci[0] > 0 else 'below_zero' if ci[1] < 0 else 'includes_zero'


def interpret(contrasts):
    """INTERPRETATION on the (d) - (b) and (d) - (c) contrasts."""
    sides = {key: {f: _side(contrasts[key][f]) for f in ('success_debiased', 'accepted')}
             for key in (f'{OP} - rft', f'{OP} - pert')}
    db, dc = (set(v.values()) for v in sides.values())
    return {'sides': sides,
            'd_vs_b': ('preference_beyond_repaired_data' if db == {'above_zero'} else
                       'contribution_is_repair_data' if db == {'includes_zero'} else 'not_pre_registered'),
            'c_vs_d': 'real_failures_add_nothing_over_synthetic' if dc == {'includes_zero'} else 'not_pre_registered'}


def _num(value, sign=False):
    return '—' if value is None else f'{value:+.3f}' if sign else f'{value:.3f}'


def _cell(contrast):
    ci = contrast['ci95']
    return f"{_num(contrast['estimate'], True)} [{_num(ci[0], True)}, {_num(ci[1], True)}]" if ci else (
        f"{_num(contrast['estimate'], True)} [CI 未定义]")


def report(value):
    c, t = value['candidates'], value['training']
    lines = ['# 修订 016 §3：DPO2 归因消融（dev，单次贪心，无修复；探索性）', '',
             f"dev-select-v1 的 {c[REFERENCE]['requests']} 个房间，HF 精确策略（训练基座 + 未合并的 FP32 adapter），"
             '与修订 010/013 相同的共享推理计划。四个候选均从 rpo-dpo1 的精确策略出发；(b)(c) 与 (d) 的优化器步数相同'
             f"（S = {t[OP]['steps']}，{t[OP]['epochs']} 个 epoch，有效批量 4）。测试集不受此消融影响。", '',
             '| 候选 | 训练 | 每 epoch 行数（唯一条目） |', '|---|---|---|',
             f"| {LABELS[REFERENCE]} | 起点（修订 010） | — |",
             *(f"| {LABELS[n]} | {'仅 chosen 的 NLL（--sft_only）' if n == 'rft' else 'DPO + RPO α 1.0'} | "
               f"{t[n]['rows']['rows_per_epoch']}（{t[n]['rows']['unique']}） |" for n in RUNS),
             f"| {LABELS[OP]} | 在线策略 DPO2（修订 013） | — |", '',
             '| 候选 | 完整率 | a′ | 下游接受率 | 两者均通过 | 检查器成功率 | 尺寸 log MAE | a′ 论文源半 | a′ 其他源半 |',
             '|---|---|---|---|---|---|---|---|---|']
    for n in ORDER:
        m = c[n]
        lines.append(f"| {LABELS[n]} | {_num(m['complete_rate'])} | {_num(m['a_prime'])} | {_num(m['accepted_rate'])} | "
                     f"{_num(m['both_pass_rate'])} | {_num(m['geometry_success_rate'])} | {_num(m['size_log_mae_own'])} | "
                     f"{_num(m['halves']['paper_sources']['a_prime'])} | {_num(m['halves']['other_sources']['a_prime'])} |")
    lines += ['', '## 配对差异（前者减后者；房屋连通组 bootstrap，2,000 次，seed 42，95% CI）', '',
              '| 比较 | 完整率 | a′ | 下游接受率 | 检查器成功率 | 尺寸 log MAE |', '|---|---|---|---|---|---|']
    for key, x in value['contrasts'].items():
        first, second = key.split(' - ')
        lines.append(f"| {LABELS[first]} − {LABELS[second]} | {_cell(x['complete'])} | {_cell(x['success_debiased'])} | "
                     f"{_cell(x['accepted'])} | {_cell(x['geometry_success'])} | {_cell(x['size_log_mae'])} |")
    reading = value['interpretation']

    def says(name, key):  # The reading, then each metric's CI side and the CI itself (to set against the ±0.01 bar).
        x, side = value['contrasts'][f'{OP} - {name}'], reading['sides'][f'{OP} - {name}']
        return (f"{READINGS[reading[key]]}（a′ {SIDES[side['success_debiased']]} {_cell(x['success_debiased'])}，"
                f"下游接受率 {SIDES[side['accepted']]} {_cell(x['accepted'])}）。")
    lines += ['', '## 预先固定的解读（修订 016 §3）', '',
              f"- (d) 对 (b)：{says('rft', 'd_vs_b')}", f"- (c) 对 (d)：{says('pert', 'c_vs_d')}", '',
              '说明：', '- “≈”按预先约定操作化为 a′ 与下游接受率两项差异的 95% CI 均包含 0；这不是等价检验，'
              '600 个房间上成功率差异的标准误约 0.012（CI 半宽约 0.025）。CI 端点超出修订 013 的 ±0.01 采纳门槛时，'
              '“≈”只表示未分辨出差异，不表示差异小于该门槛。',
              '- P 的尺寸取自请求先验（placement 合同无尺寸字段），尺寸 log MAE 的差别只来自完整房间集合不同。',
              '- (b) 的训练行即 op-dpo2 所训的 pair 文件原样（同样的行、行序与重复次数，DistributedSampler seed 42 同序），'
              '--sft_only 不读取 rejected，因此 (b) 与 (d) 只差 DPO 项；(c) 的 chosen 是 GT，rejected 是合成扰动。',
              '- 本消融为探索性分析，不改变修订 005–015 的任何结果或决定。']
    return '\n'.join(lines) + '\n'


def _write_new(path, payload):
    """Publish bytes at PATH without ever replacing an existing file (os.link fails if it exists)."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()


def score(args):
    data, run_root, out = Path(args.data), Path(args.run_root), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    manifest, rooms, requests, record, runs, finished = op._reference(data, run_root / 'dev-eval-010', args.workers)
    ref = run_root / 'dev-eval-013' / 'scores'
    d._verified_dir(ref)
    scores = h.strict_json((ref / 'scores.json').read_text())
    if scores.get('protocol') != op.SCORES_PROTOCOL or scores['data_manifest_sha256'] != h.sha256(data / 'MANIFEST.json'):
        raise ValueError(f'{ref} is not the amendment-013 dev score on this dev data')
    lineage = training(run_root)
    runs[OP] = op._scored(run_root / 'dev-eval-013', OP, data, manifest, rooms, requests[ARM], args.workers)
    runs.update({n: _scored(run_root, n, data, manifest, rooms, requests[ARM], args.workers) for n in RUNS})
    if {run[1]['context_plan_sha256'] for run in runs.values()} | {scores['plan_sha256']} != {record['plan_sha256']}:
        raise ValueError('ablation runs are not on the amendment-010 shared dev plan')
    result, rows = {}, {}
    for n in ORDER:
        value, rows[n] = finished[n] if n in finished else op._finish(runs[n][2], runs[op.SFT][2], rooms, requests[ARM],
                                                                      args.workers)
        result[n] = {**runs[n][3], **value}
    for n in (REFERENCE, OP):
        entry = scores['candidates'][n]
        if any(result[n][k] != entry[k] for k in ('a_prime', 'complete_rate', 'accepted_rate')) or (
                result[n]['inference']['run_sha256'] != entry['inference']['run_sha256']):
            raise ValueError(f'{n} rescoring differs from its amendment-013 dev score')
    contrasts = {f'{a} - {b}': r2.contrasts(rows[a], rows[b], FIELDS, op.REPEATS, r2.SEED) for a, b in PAIRS}
    value = {'protocol': PROTOCOL, 'status': 'EXPLORATORY', 'arm': ARM, 'labels': LABELS,
             'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'), 'uids_sha256': manifest['uids_sha256'],
             'plan_sha256': record['plan_sha256'], 'candidates': result, 'training': lineage, 'contrasts': contrasts,
             'rule': INTERPRETATION, 'interpretation': interpret(contrasts),
             'reference_scores': {name: h.sha256(run_root / name / 'scores' / 'MANIFEST.json')
                                  for name in ('dev-eval-010', 'dev-eval-013')},
             'scope': ("contrasts: first minus second, paired house-cluster bootstrap on the dev rooms (report_v2.contrasts, "
                       "2,000 draws, seed 42); rates over every request (incomplete = failure), size log MAE on rooms "
                       "complete in both; dev only, reported only, never used to select or to touch test"),
             'rows': {n: [{k: r[k] for k in ROW_KEYS} for r in rows[n]] for n in ORDER},
             'code_sha256': {**op._code(), Path(__file__).name: h.sha256(__file__)}}
    payload, path = report(value).encode('utf-8'), out.with_name(REPORT)
    if not (path.exists() and path.read_bytes() == payload):  # A crash between the two writes leaves the same report.
        _write_new(path, payload)
    _write_new(out, op._json_bytes(value))
    return {'status': 'ABLATION_016_COMPLETE', 'interpretation': {k: v for k, v in value['interpretation'].items()
                                                                   if k != 'sides'},
            **{n: {k: round(result[n][k], 4) for k in ('a_prime', 'accepted_rate', 'complete_rate')} for n in ORDER}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('train')
    step.add_argument('--dpo-root', type=Path, required=True)
    step.add_argument('--candidate', choices=tuple(RUNS), required=True)
    step = sub.add_parser('score')
    for flag in ('data', 'run-root', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    result = {'train': train, 'score': score}[args.command](args)
    if args.command == 'train':
        result = {'status': 'MERGED', 'stage': result['stage'], 'model_sha256': result['model_sha256']}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
