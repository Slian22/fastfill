"""Amendment 016 section 3: DPO2 attribution ablation on dev (single-shot greedy, no repair; EXPLORATORY).

python -m fastfill.comparison.ablation_016 STEP ...   (sbatch: isambard_016.sbatch; chain: submit_016_ablation.sh)

train  --dpo-root RUN/RUNS[CAND] --candidate CAND   (4 GPUs; a rerun resumes); CAND -> run (training seed):
       rft -> dpo-005 (42), rft-s43 -> dpo-005-s43 (43), pert -> dpo-006 (42), op-dpo2-s43 -> dpo-007-s43 (43)
    All start from rpo-dpo1's exact policy exactly as onpolicy_dpo train: training base dpo-001/placement/sft-merged
    + --init_adapter dpo-003/placement/dpo1/final (dpo_stages.exact_policy rehashes both), LoRA r32/a16, lr 5e-6,
    effective batch 4, onpolicy_dpo.RECIPE's 2 epochs; the seed is train_dpo's own --seed (default 42), recorded as
    recipe.seed in the guard and the lineage receipt (SEED_EFFECT: it sets the data order, padding rows included, and
    the dropout RNG; the LoRA starting weights are rpo-dpo1's adapter for every arm, so they do not depend on it).
      rft:  train_dpo --sft_only (chosen mean token NLL; no DPO term, no reference pass) on
            dpo-004/stage2-pairs/placement.jsonl verbatim (the N pairs op-dpo2 trained on: same rows, order and repeats,
            so the DistributedSampler (same seed, same N) replays op-dpo2's batches; the loss differs from (d)'s only by
            the DPO term, but the dropout RNG stream does not match: DPO runs two policy forwards per batch, --sft_only
            one; N must equal the op-dpo2 receipt's pair count, so 2 x ceil(N/4) = S steps).
      pert: DPO + RPO alpha 1.0 (onpolicy_dpo.RECIPE) on dpo-001/stage2-pairs/placement.jsonl (amendment 009: GT chosen,
            synthetic collision_shift / wall_oob rejected); one epoch = 4 x S/2 rows: the unique pairs in ascending
            sha256('ablation-016-v1:pert:' + key) order, cycled (a pool larger than an epoch is a hash-ranked subset).
      op-dpo2-s43: onpolicy_dpo train's recipe and inputs (amendment 013) with seed 43: --pairs is op-dpo2's own
            dpo-004/stage2-pairs/placement.jsonl (same checks: OP_PAIRS_COMPLETE, sampled from rpo-dpo1, N = the
            op-dpo2 receipt's count), the same --model, --init_adapter (= the frozen reference) and flags; train_dpo's
            arguments differ from op-dpo2's only in --seed and --out. No epoch-1 merge (only e2 is compared).
    Steps: op-dpo2 e2's optimiser steps S, read from the dpo-004/placement/dpo2-merged lineage receipt (missing = raise:
    the ablation runs after amendment-013 training); 2 epochs are exactly S steps. rft / pert rows are published
    new-only as dpo-00X/train-rows (a rerun must rebuild the same bytes). Lineage: RUN/placement/dpo2-merged receipt as
    onpolicy_dpo's (parent dpo-003 dpo1-merged, training_base dpo-001 sft-merged with is_dpo1_merge false, init adapter
    sha256, steps_matched_to), so infer_shard --exact-policy loads training base + the final adapter.
score  --data RUN/dev-select-v1 --run-root RUN --out RUN/dev-eval-016/ablation.json [--workers 30]
    (a) rpo-dpo1 and (d) op-dpo2-e2 seed 42 rescored by onpolicy_dpo from dev-eval-010 / dev-eval-013 (dev-eval-010's
    and dev-eval-013's scores of record must reproduce); every RUNS candidate from dev-eval-016/CAND/infer-placement
    (exact policy of RUNS[CAND]/placement/dpo2-merged), all on the shared dev-eval-010 plan; every trained arm has
    op-dpo2 e2's optimiser steps and its own seed. Per candidate onpolicy_dpo._finish: complete, a', downstream
    acceptance over all 600 rooms (incomplete = not accepted), checker success, size log MAE, footprint bias B (P sizes
    are the train prior, so B is the prior's bias on the candidate's complete rooms), dev halves; the same metrics per
    section-5 stratum (scale_curve.subsets: object-count bins, real / synthetic, room families; descriptive), per seed
    and as the two-seed mean of (b) and (d). Paired
    house-cluster bootstrap (report_v2.contrasts, 2,000 draws, seed 42), first minus second (PAIRS). INTERPRETATION is
    applied per seed ((b), (d) of the same seed; (c) single seed) and a reading is claimed only when both seeds give
    it (addenda A5, C8). Writes ablation.json (with compact per-room rows) and ablation-report.md (Chinese) next to it,
    new-only. Test data is never touched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from fastfill.comparison import dev_select as ds
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import scale_curve as sc
from fastfill.comparison import train_pair_helpers as h

ARM, STAGE, REFERENCE, OP = op.ARM, op.STAGE, op.REFERENCE, 'op-dpo2-e2'
RUNS = {'rft': 'dpo-005', 'rft-s43': 'dpo-005-s43', 'pert': 'dpo-006', 'op-dpo2-s43': 'dpo-007-s43'}
KINDS = {'rft': 'rft', 'rft-s43': 'rft', 'pert': 'pert', 'op-dpo2-s43': 'op'}  # Each run's pairs, recipe and loss.
SEEDS = {'rft': d.SEED, 'rft-s43': 43, 'pert': d.SEED, 'op-dpo2-s43': 43}  # train_dpo --seed (addenda A5, C8).
SOURCES = {'rft': (op.RUN, ('prompt', 'chosen')), 'pert': (d.SOURCE_RUN, ('prompt', 'chosen', 'rejected')),
           'op': (op.RUN, ('prompt', 'chosen', 'rejected'))}
RECIPES = {'rft': {**{k: v for k, v in op.RECIPE.items() if k != 'rpo_alpha'}, 'sft_only': True}, 'pert': op.RECIPE,
           'op': op.RECIPE}
LOSS = {'rft': ("continued SFT: mean token NLL of the chosen response only (train_dpo --sft_only; no DPO term, no "
                "reference pass, rejected unused); the policy LoRA starts as rpo-dpo1's adapter (--init_adapter) on its "
                "training base"),
        'pert': op.LOSS + '; RPO alpha 1.0; pairs: amendment-009 GT chosen vs synthetic perturbation rejected',
        'op': op.LOSS}
SEED_EFFECT = ("train_dpo --seed sets transformers.set_seed (python, numpy, torch RNG: LoRA dropout masks) and the "
               "DistributedSampler seed (per-epoch data order, including the <= 3 rows repeated from the head of the "
               "shuffled order to pad an epoch to a multiple of 4, so when N is not a multiple of 4 an epoch's row "
               "multiset differs between seeds by those rows); every arm's LoRA starts as rpo-dpo1's adapter "
               "(--init_adapter), so the starting weights do not depend on the seed")
SALT, EVAL, REPORT = 'ablation-016-v1:', 'dev-eval-016', 'ablation-report.md'
ORDER = (REFERENCE, 'rft', 'rft-s43', 'pert', OP, 'op-dpo2-s43')
SEED_ARMS = {42: ('rft', OP), 43: ('rft-s43', 'op-dpo2-s43')}  # (b), (d) of one training seed; (a), (c) single seed.
SINGLE_SEED = (REFERENCE, 'pert')
MEANS = {'rft-mean': ('rft', 'rft-s43'), 'op-dpo2-mean': (OP, 'op-dpo2-s43')}  # Two-seed means (reported only).
SHOWN = (REFERENCE, 'rft', 'rft-s43', 'rft-mean', 'pert', OP, 'op-dpo2-s43', 'op-dpo2-mean')
LABELS = {REFERENCE: '(a) rpo-dpo1', 'rft': '(b) RFT s42', 'rft-s43': '(b) RFT s43', 'rft-mean': '(b) RFT 两种子均值',
          'pert': '(c) pert-DPO2', OP: '(d) op-dpo2 e2 s42', 'op-dpo2-s43': '(d) op-dpo2 e2 s43',
          'op-dpo2-mean': '(d) op-dpo2 e2 两种子均值'}
SEED_TEXT = {REFERENCE: '42（单种子）', 'rft': '42', 'rft-s43': '43', 'rft-mean': '42、43 均值', 'pert': '42（单种子）',
             OP: '42', 'op-dpo2-s43': '43', 'op-dpo2-mean': '42、43 均值'}
PAIRS = (('rft', REFERENCE), ('rft-s43', REFERENCE), ('pert', REFERENCE), (OP, REFERENCE), ('op-dpo2-s43', REFERENCE),
         (OP, 'rft'), ('op-dpo2-s43', 'rft-s43'), (OP, 'pert'), ('op-dpo2-s43', 'pert'), ('pert', 'rft'), ('pert', 'rft-s43'))
FIELDS = (*op.FIELDS, 'size_log_mae')
ROW_KEYS = ('uid', 'source', 'group', 'objects', 'complete', 'geometry_success', 'success_debiased', 'accepted',
            'size_log_mae', 'bias_fp')
METRICS = ('complete_rate', 'a_prime', 'accepted_rate', 'both_pass_rate', 'geometry_success_rate', 'size_log_mae_own',
           'footprint_bias')
PROTOCOL = 'amendment-016-dpo2-attribution-ablation'
INTERPRETATION = (
    "fixed in advance (amendment 016 section 3), on a' (success_debiased) and downstream acceptance (accepted): "
    "(d) - (b) with both 95% CIs above 0 -> the preference construction has value beyond its repaired data; "
    "(d) ~ (b), operationalised as both CIs of (d) - (b) including 0 (not an equivalence test) -> the contribution is "
    "the repair-based data; (c) ~ (d), both CIs of (d) - (c) including 0 -> real failures add nothing over synthetic "
    "ones; any other pattern has no pre-registered reading and is reported as measured. Evaluated per training seed "
    "((b), (d) of seed 42 and of seed 43, each against the single-seed (c)); a reading is claimed only when both seeds "
    "give it, else seeds_disagree (addenda A5, C8)")
READINGS = {'preference_beyond_repaired_data': '偏好构造在其修复数据之外仍有价值',
            'contribution_is_repair_data': '贡献来自基于修复的数据（(d) 与 (b) 无法区分）',
            'real_failures_add_nothing_over_synthetic': '真实失败相对合成扰动没有额外贡献（(c) 与 (d) 无法区分）',
            'not_pre_registered': '不属于预先固定的情形，按测得结果报告，不作解读',
            'seeds_disagree': '两个种子不一致（seeds disagree），不下结论'}
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


def _source(run_root, kind, parent, matched):
    """The verified source pairs of KIND -> (rows, receipt)."""
    pairs_dir = Path(run_root) / SOURCES[kind][0] / 'stage2-pairs'
    files = d._verified_dir(pairs_dir)
    path = (pairs_dir / f'{ARM}.jsonl').resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != h._expected_hash(files[path.name]):
        raise ValueError(f'{path} changed since its manifest')
    rows = [h.strict_json(line) for line in payload.decode('utf-8').splitlines()]  # op pairs repeat UIDs.
    manifest = h.sha256(pairs_dir / 'MANIFEST.json')
    if kind != 'pert':  # rft and op: op-dpo2's own pairs (onpolicy_dpo train's checks plus its receipt's manifest).
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
    kind, matched = KINDS[name], matched_steps(run_root)
    recipe = {**RECIPES[kind], 'seed': SEEDS[name]}  # Seed 42: RECIPES[kind] itself, same keys in the same order.
    source, source_info = _source(run_root, kind, parent, matched)
    if kind != 'pert':  # op-dpo2's pairs verbatim: the same N rows in file order (rft: same batches, no DPO term).
        count, rows = len(source), source
        info = {'source_rows': count, 'unique': len({_key(r, kind) for r in source}), 'rows_per_epoch': count,
                'verbatim': True}
        order = ('the op-dpo2 pairs file verbatim (file order; train_dpo DistributedSampler, '
                 f"seed {SEEDS[name]}, as op-dpo2{'' if SEEDS[name] == d.SEED else f'-s{SEEDS[name]}'})")
        if count != matched['pairs_count']:
            raise ValueError(f"{name}: {count} pairs, op-dpo2 was trained on {matched['pairs_count']}")
    else:
        count = d.WORLD_SIZE * recipe['bs'] * recipe['grad_accum'] * (matched['steps'] // recipe['epochs'])
        rows, info = take(source, kind, count)
        order = f"ascending sha256('{SALT}' + candidate + ':' + json {list(SOURCES[kind][1])}), cycled"
    payload = d._jsonl(rows)
    if kind != 'pert' and hashlib.sha256(payload).hexdigest() != source_info['sha256']:
        raise ValueError(f"{name} rows are not {source_info['path']} byte for byte")
    # op: onpolicy_dpo train's own --pairs file (dpo-004/stage2-pairs); rft, pert: new-only train-rows of this run.
    rows_dir = Path(source_info['path']).parent if kind == 'op' else root / 'train-rows'
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
    settings = {**recipe, **d.TRAIN_FIXED, 'loss': LOSS[kind],
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
    """(b), (c) and (d) lineage receipts (receipts only, no weights): the same optimiser steps from rpo-dpo1, each run
    with its own seed (recipe.seed; op-dpo2 e2 from its dpo-004 receipt)."""
    matched, result = matched_steps(run_root), {}
    for name in RUNS:
        path = d.receipt_path(model_path(run_root, name).resolve())
        t = d._lineage(path, ARM, STAGE)['training']
        if (t.get('candidate') != name or t['completion']['steps'] != matched['steps']
                or t['recipe']['seed'] != SEEDS[name]):
            raise ValueError(f'{path} is not a step-matched {name} run with seed {SEEDS[name]}')
        result[name] = {'lineage_sha256': h.sha256(path), 'steps': t['completion']['steps'], 'rows': t['rows'],
                        'loss': t['recipe']['loss'], 'init_adapter': t['recipe']['init_adapter'],
                        'seed': t['recipe']['seed']}
    path = d.receipt_path((Path(run_root) / op.RUN / ARM / op.MODELS[OP]).resolve())
    return {**result, OP: {**matched, 'seed': d._lineage(path, ARM, STAGE)['training']['recipe']['seed']}}


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
    """INTERPRETATION per training seed on its (d) - (b) and (d) - (c) contrasts; a reading holds only when both seeds
    give it, else seeds_disagree."""
    per_seed = {}
    for seed, (b, dd) in SEED_ARMS.items():
        sides = {key: {f: _side(contrasts[key][f]) for f in ('success_debiased', 'accepted')}
                 for key in (f'{dd} - {b}', f'{dd} - pert')}
        db, dc = (set(v.values()) for v in sides.values())
        per_seed[str(seed)] = {
            'sides': sides,
            'd_vs_b': ('preference_beyond_repaired_data' if db == {'above_zero'} else
                       'contribution_is_repair_data' if db == {'includes_zero'} else 'not_pre_registered'),
            'c_vs_d': 'real_failures_add_nothing_over_synthetic' if dc == {'includes_zero'} else 'not_pre_registered'}
    first, second = per_seed.values()
    return {'per_seed': per_seed,
            **{k: first[k] if first[k] == second[k] else 'seeds_disagree' for k in ('d_vs_b', 'c_vs_d')}}


def levels(rows):
    """METRICS of one candidate on ROWS (per-room rows): rates over every row, size log MAE and B over complete ones."""
    done = [r for r in rows if r['complete']]

    def rate(test):
        return sum(bool(test(r)) for r in rows) / len(rows) if rows else None

    def mean(key):
        return statistics.mean(r[key] for r in done) if done else None
    return {'complete_rate': rate(lambda r: r['complete']), 'a_prime': rate(lambda r: r['success_debiased']),
            'accepted_rate': rate(lambda r: r['accepted']),
            'both_pass_rate': rate(lambda r: r['geometry_success'] and r['accepted']),
            'geometry_success_rate': rate(lambda r: r['geometry_success']),
            'size_log_mae_own': mean('size_log_mae'), 'footprint_bias': mean('bias_fp')}


def with_means(table):
    """{name: metrics} plus MEANS: per metric the arithmetic mean of the two seeds (None if either is None)."""
    def mean(x, y):
        return {k: None if x[k] is None or y[k] is None else (x[k] + y[k]) / 2 for k in x}
    return {**table, **{m: mean(table[x], table[y]) for m, (x, y) in MEANS.items()}}


def tables(result, rows, rooms):
    """The reported numbers: 'all' (onpolicy_dpo._finish's metrics and dev-half a') and each section-5 stratum
    (scale_curve.subsets on the dev rooms: object-count bins, real / synthetic, room families; levels), every one per
    seed and with the two-seed means."""
    out = {'all': {'rooms': result[REFERENCE]['requests'], 'candidates': with_means({n: {
        **{k: result[n][k] for k in METRICS}, **{f'a_prime_{x}': result[n]['halves'][x]['a_prime'] for x in op.HALVES}}
        for n in ORDER})}}
    for name, members in sc.subsets([rooms[r['uid']] for r in rows[REFERENCE]]).items():
        if name != 'all':
            keep = {x['uid'] for x in members}
            part = {n: [r for r in rows[n] if r['uid'] in keep] for n in ORDER}
            out[name] = {'rooms': len(part[REFERENCE]), 'candidates': with_means({n: levels(v) for n, v in part.items()})}
    return out


def _num(value, sign=False):
    return '—' if value is None else f'{value:+.3f}' if sign else f'{value:.3f}'


def _cell(contrast):
    ci = contrast['ci95']
    return f"{_num(contrast['estimate'], True)} [{_num(ci[0], True)}, {_num(ci[1], True)}]" if ci else (
        f"{_num(contrast['estimate'], True)} [CI 未定义]")


COLUMNS = (('complete_rate', '完整率'), ('a_prime', 'a′'), ('accepted_rate', '下游接受率'), ('both_pass_rate', '两者均通过'),
           ('geometry_success_rate', '检查器成功率'), ('size_log_mae_own', '尺寸 log MAE'), ('footprint_bias', '足迹偏差 B'))
HALF_COLUMNS = (('a_prime_paper_sources', 'a′ 论文源半'), ('a_prime_other_sources', 'a′ 其他源半'))


def _table(candidates, columns):
    lines = ['| 候选 | 种子 | ' + ' | '.join(t for _, t in columns) + ' |', '|---|---|' + '---|' * len(columns)]
    return lines + [f"| {LABELS[n]} | {SEED_TEXT[n]} | "
                    + ' | '.join(_num(candidates[n][k], k == 'footprint_bias') for k, _ in columns) + ' |' for n in SHOWN]


def report(value):
    t, tab = value['training'], value['tables']
    count = t['rft']['rows']['rows_per_epoch']  # N: op-dpo2's pair count, which rft trains on verbatim.
    lines = ['# 修订 016 §3：DPO2 归因消融（dev，单次贪心，无修复；探索性）', '',
             f"dev-select-v1 的 {tab['all']['rooms']} 个房间，HF 精确策略（训练基座 + 未合并的 FP32 adapter），"
             '与修订 010/013 相同的共享推理计划。所有候选均从 rpo-dpo1 的精确策略出发；(b)(c)(d) 的优化器步数相同'
             f"（S = {t[OP]['steps']}，{t[OP]['epochs']} 个 epoch，有效批量 4）。(b) RFT 与 (d) op-dpo2 各训练两个种子"
             '（42、43，逐种子报告并给出两种子均值）；(a) rpo-dpo1 与 (c) pert-DPO2 为单种子。测试集不受此消融影响。', '',
             '种子的作用：train_dpo --seed 设定 transformers.set_seed（LoRA dropout 掩码等随机数）与 DistributedSampler 的'
             '每 epoch 数据顺序；所有候选的 LoRA 都以 rpo-dpo1 的 adapter 为初值（--init_adapter），初始权重不随种子变化，'
             '因此附录 C8 所说的“LoRA 初始化”在本设置下不起作用。N 不是 4 的倍数时，DistributedSampler 每个 epoch 用打乱后'
             '序列开头的 ≤3 行重复样本补齐，这几行也随种子变化，所以两个种子每个 epoch 实际训练的样本多重集可以差这几行'
             f"（(b)(d) 的 N = {count}，每 epoch 补齐 {-count % d.WORLD_SIZE} 行；(c) 每 epoch 行数是 4 的倍数，不补齐）。"
             '训练文件、步数、初始 adapter、训练基座与参考策略在两个种子间相同。', '',
             '| 候选 | 种子（谱系收据） | 训练 | 每 epoch 行数（唯一条目） |', '|---|---|---|---|',
             f"| {LABELS[REFERENCE]} | {SEED_TEXT[REFERENCE]} | 起点（修订 010） | — |",
             *(f"| {LABELS[n]} | {t[n]['seed']}{'（单种子）' if n in SINGLE_SEED else ''} | "
               + ({'rft': '仅 chosen 的 NLL（--sft_only）', 'pert': 'DPO + RPO α 1.0（合成扰动 pair）',
                   'op': '在线策略 DPO2（修订 013 配方，op-dpo2 的 pair 文件）'}[KINDS[n]]
                  + f" | {t[n]['rows']['rows_per_epoch']}（{t[n]['rows']['unique']}） |" if n in RUNS else
                  '在线策略 DPO2（修订 013） | — |') for n in ORDER[1:]), '',
             f"## 全部 dev 房间（{tab['all']['rooms']} 个）", '', *_table(tab['all']['candidates'], COLUMNS + HALF_COLUMNS),
             '', '## 分层（修订 016 §5：物体数分箱、真实／合成来源、房型族；scale_curve.subsets；描述性，无 CI，不设决策规则）',
             '', 'real = §5 列出的真实或策划来源（附录 C6：SpatialLM、HSSD200、InteriorGS、Structured3D 为设计师制作而非扫描），'
             'synthetic = 其余来源（含 IL3D_3dfront）；family:* 为修订 015 的房型族。房间数为 0 的分层不列出。']
    for name in (k for k in tab if k != 'all'):
        lines += ['', f"### {name}（{tab[name]['rooms']} 个房间）", '', *_table(tab[name]['candidates'], COLUMNS)]
    lines += ['', '## 配对差异（前者减后者；房屋连通组 bootstrap，2,000 次，seed 42，95% CI）', '',
              '| 比较 | 完整率 | a′ | 下游接受率 | 检查器成功率 | 尺寸 log MAE |', '|---|---|---|---|---|---|']
    for key, x in value['contrasts'].items():
        first, second = key.split(' - ')
        lines.append(f"| {LABELS[first]} − {LABELS[second]} | {_cell(x['complete'])} | {_cell(x['success_debiased'])} | "
                     f"{_cell(x['accepted'])} | {_cell(x['geometry_success'])} | {_cell(x['size_log_mae'])} |")
    reading = value['interpretation']

    def says(seed, other, key):  # One seed's reading, then each metric's CI side and the CI (against the ±0.01 bar).
        b, dd = SEED_ARMS[seed]
        pair, own = f"{dd} - {other or b}", reading['per_seed'][str(seed)]
        x, side = value['contrasts'][pair], own['sides'][pair]
        return (f"  - seed {seed}（{LABELS[dd]} − {LABELS[other or b]}）：{READINGS[own[key]]}"
                f"（a′ {SIDES[side['success_debiased']]} {_cell(x['success_debiased'])}，"
                f"下游接受率 {SIDES[side['accepted']]} {_cell(x['accepted'])}）。")
    lines += ['', '## 预先固定的解读（修订 016 §3；逐种子判定，两个种子一致才下结论，附录 A5、C8）', '',
              f"- (d) 对 (b)：{READINGS[reading['d_vs_b']]}", *(says(s, None, 'd_vs_b') for s in SEED_ARMS),
              f"- (c) 对 (d)：{READINGS[reading['c_vs_d']]}", *(says(s, 'pert', 'c_vs_d') for s in SEED_ARMS), '',
              '说明：', '- “≈”按预先约定操作化为 a′ 与下游接受率两项差异的 95% CI 均包含 0；这不是等价检验，'
              '600 个房间上成功率差异的标准误约 0.012（CI 半宽约 0.025）。CI 端点超出修订 013 的 ±0.01 采纳门槛时，'
              '“≈”只表示未分辨出差异，不表示差异小于该门槛。',
              '- 足迹偏差 B：P 的尺寸取自请求中的训练先验（placement 合同无尺寸字段），因此每个候选的 B 是训练先验在该候选'
              '完整房间上的偏差，不是模型选择的尺寸；各候选 B 与尺寸 log MAE 的差别只来自完整房间集合不同。',
              '- (b) 的训练行即 op-dpo2 所训的 pair 文件原样（同样的行、行序与重复次数），--sft_only 不读取 rejected。同种子下'
              ' DistributedSampler 的数据顺序相同；损失只差 DPO 项，但 LoRA dropout 掩码的随机数流不同（DPO 路径每批多一次'
              '策略前向），所以同种子的 (b) 与 (d) 不只差 DPO 项。(c) 的 chosen 是 GT，rejected 是合成扰动。',
              '- 两种子均值是两个种子数值的算术平均，只作报告；解读只看逐种子的配对 CI。(a)(c) 为单种子，'
              '涉及它们的 CI 不反映训练种子带来的波动。',
              '- 分层表为描述性（修订 016 §5），房间数少的分层波动大。',
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
        value, scored = finished[n] if n in finished else op._finish(runs[n][2], runs[op.SFT][2], rooms, requests[ARM],
                                                                     args.workers)
        result[n], rows[n] = {**runs[n][3], **value}, [{**r, 'objects': len(rooms[r['uid']]['objects'])} for r in scored]
    for n in (REFERENCE, OP):
        entry = scores['candidates'][n]
        if any(result[n][k] != entry[k] for k in ('a_prime', 'complete_rate', 'accepted_rate')) or (
                result[n]['inference']['run_sha256'] != entry['inference']['run_sha256']):
            raise ValueError(f'{n} rescoring differs from its amendment-013 dev score')
    contrasts = {f'{a} - {b}': r2.contrasts(rows[a], rows[b], FIELDS, op.REPEATS, r2.SEED) for a, b in PAIRS}
    value = {'protocol': PROTOCOL, 'status': 'EXPLORATORY', 'arm': ARM, 'labels': LABELS,
             'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'), 'uids_sha256': manifest['uids_sha256'],
             'plan_sha256': record['plan_sha256'], 'candidates': result, 'training': lineage, 'contrasts': contrasts,
             'rule': INTERPRETATION, 'interpretation': interpret(contrasts), 'tables': tables(result, rows, rooms),
             'seeds': {'by_seed': {str(s): list(v) for s, v in SEED_ARMS.items()}, 'single_seed': list(SINGLE_SEED),
                       'means': {m: list(v) for m, v in MEANS.items()}, 'effect': SEED_EFFECT},
             'reference_scores': {name: h.sha256(run_root / name / 'scores' / 'MANIFEST.json')
                                  for name in ('dev-eval-010', 'dev-eval-013')},
             'scope': ("contrasts: first minus second, paired house-cluster bootstrap on the dev rooms (report_v2.contrasts, "
                       "2,000 draws, seed 42); rates over every request (incomplete = failure), size log MAE on rooms "
                       "complete in both; footprint_bias B: P sizes are the train prior from the request (the placement "
                       "contract has no size field), so B is the prior's footprint bias on the candidate's complete rooms; "
                       "tables: per seed, two-seed arithmetic means of (b) and (d), section-5 strata (scale_curve.subsets: "
                       "object-count bins, real / synthetic, room families) descriptive without CIs; dev only, reported "
                       "only, never used to select or to touch test"),
             'rows': {n: [{k: r[k] for k in ROW_KEYS} for r in rows[n]] for n in ORDER},
             'code_sha256': {**op._code(), Path(__file__).name: h.sha256(__file__)}}
    payload, path = report(value).encode('utf-8'), out.with_name(REPORT)
    if not (path.exists() and path.read_bytes() == payload):  # A crash between the two writes leaves the same report.
        _write_new(path, payload)
    _write_new(out, op._json_bytes(value))
    reading = value['interpretation']
    return {'status': 'ABLATION_016_COMPLETE', 'interpretation': {
                'd_vs_b': reading['d_vs_b'], 'c_vs_d': reading['c_vs_d'],
                'per_seed': {s: {k: v[k] for k in ('d_vs_b', 'c_vs_d')} for s, v in reading['per_seed'].items()}},
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
