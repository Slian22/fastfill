"""Amendment 018 sections 2-4: arm P distilled from the free open teacher O2 on top of rpo-dpo1 (SFT-D -> DPO1-D ->
DPO2-D, run RUN/distill-018), each stage's final adapter a dev candidate against rpo-dpo1; test only if adopted.

python -m fastfill.comparison.distill_018_chain STEP ...   (sbatch: isambard_018.sbatch; chain: submit_018_chain.sh)

Section 1 (the T-A/T-B/T-C plan RUN/distill-018/plan, the O2 teacher run and its filter RUN/distill-018/teacher) is
distill_018's; its loaders (load_plan, load_targets) are imported lazily, so this module also imports without it.
Every model is "training base dpo-001/placement/sft-merged (BF16) + one FP32 adapter". Each stage gets a dpo_stages
lineage receipt as onpolicy_dpo / ablation_016 write them: stage 'dpo2' (the structural label of a stage after rpo-dpo1
on rpo-dpo1's training base), parent dpo-003/placement/dpo1-merged, training_base dpo-001/placement/sft-merged, so
dpo_stages.exact_policy / inference_identity / sample_vllm / infer_shard --exact-policy load training base + the stage
adapter. training.init binds the adapter the stage started from (path, sha256, its receipt) and training.chain names
the real lineage SFT -> rpo-dpo1 -> SFT-D -> DPO1-D -> DPO2-D.

tc-data    --plan RUN/distill-018/plan --out RUN/distill-018/t-c
    T-C as its own data dir (sample_vllm samples, and onpolicy_dpo pairs binds, every room of a data dir): the plan's
    request / room / prior rows of uids/tc.json verbatim, in the onpolicy_dpo subset layout with MANIFEST scope TC_SCOPE,
    split train, the plan's training manifest and the T-C sft_prompt_tokens (distill_018.load_plan verifies the plan).
sft-data   --data MAIN --sft-runs SFT_J SFT_P --exclude RUN/dpo-001/positives --teacher RUN/distill-018/teacher
           --base-model RUN/dpo-003/placement/dpo1-merged --dev DEV... --test TEST... --out RUN/distill-018/sft-d-rows
           [--workers N]
    SFT-D rows: every teacher target of T-A and T-B (distill_018.load_targets: complete teacher filter; sorted by uid)
    as {uid, source, group, set, prompt, chosen = target, rejected = target} (train_dpo --sft_only reads chosen only).
    Raises unless every room is in the SFT length-admitted train pool minus the
    DPO1 positives and shares no UID or house group with dev/test (onpolicy_dpo._held_out), its prompt is P's SFT
    training prompt (main data) and tokenizes, with train_dpo's tokenizer (rpo-dpo1's training base), to the SFT run's
    prompt token count, and its target is canonical P JSON (contract._json) that geometry.score finds complete and
    passing. Rows that do not fit train_dpo's 8192 tokens are dropped and listed. dpo_stages.preflight admits the rest.
pairs-dpo1 --data RUN/dpo-004/train-subset --samples RUN/dpo-004/samples --base-model RUN/dpo-003/placement/dpo1-merged
           --targets RUN/distill-018/sft-d-rows --out RUN/distill-018/dpo1-d-pairs [--workers N]
    On T-A (the amendment-013 subset; samples bound as onpolicy_dpo pairs: rpo-dpo1's exact policy, k order, prompt
    tokens = the sampled input tokens). Per T-A room with a teacher target: chosen = that target (it must pass the
    frozen checker), rejected = up to REJECTED_PER_ROOM of rpo-dpo1's own complete samples that fail the checker or the
    strict admission (onpolicy_dpo._sample / _passes), in the amendment-013 tier order (colocated duplicates, main
    collisions, other, then k); identical texts (or the chosen text) collapse; over-length pairs are dropped. The
    rejected samples are off-policy by one stage (OFF_POLICY, stated in the summary and the receipt).
pairs-dpo2 --data RUN/distill-018/t-c --samples RUN/distill-018/dpo2-d-samples --base-model
           RUN/distill-018/placement/dpo1-d-merged --targets RUN/distill-018/sft-d-rows --t-a RUN/dpo-004/train-subset
           --out RUN/distill-018/dpo2-d-pairs [--workers N]
    onpolicy_dpo.build_pairs (PAIR_RULE, its floors and preflight) on T-C (scope TC_SCOPE) with the sampled policy =
    DPO1-D (its receipt: stage dpo2, candidate dpo1-d). T-C must share no room with T-A or the SFT-D rows. The samples
    are sample_vllm run / finalize of DPO1-D's exact policy on T-C (k = 0 greedy, k = 1..8 T 0.7 top_p 0.95).
    Below the amendment-013 floors (1,500 pairs, 600 rooms) DPO2-D is not trained: the out dir then holds only the
    ledger and a summary with status NOT_TRAINED and the counts, and the step succeeds; train-dpo2, DPO2-D's dev
    shards and dev merge are then no-ops, and dev-score / select drop DPO2-D as a candidate (RULE).
train-sft | train-dpo1 | train-dpo2   --run-root RUN   (4 GPUs; a rerun resumes)
    torchrun train_dpo on the training base with --init_adapter = the previous stage's FP32 adapter (sft-d: rpo-dpo1's,
    dpo1-d: SFT-D's, dpo2-d: DPO1-D's; the frozen copy of it is the DPO reference), training_guard, preflight,
    completion, then the BF16 merge RUN/distill-018/placement/CAND-merged (receipt anchor; never the evaluated policy).
    sft-d: --sft_only, lr 1e-5, 2 epochs on sft-d-rows. dpo1-d / dpo2-d: onpolicy_dpo.RECIPE (sigmoid DPO, beta 0.1,
    RPO alpha 1.0, lr 5e-6, 2 epochs) on dpo1-d-pairs (bound to the SFT-D rows and rpo-dpo1's samples) / dpo2-d-pairs
    (sampled by DPO1-D). LoRA r32/a16, effective batch 4, seed 42.
dev-score  --data RUN/dev-select-v1 --root RUN/dev-eval-018 --ref-root RUN/dev-eval-010 --out RUN/dev-eval-018/scores
    rpo-dpo1 (and sft, for dev_select.metrics) from dev-eval-010 (onpolicy_dpo._reference: their scores of record must
    reproduce), the candidates from ROOT (exact policy of RUN/distill-018/placement/CAND-merged), one shared plan;
    DPO2-D is left out (scores.json not_trained) only when distill-018/dpo2-d-pairs is the verified NOT_TRAINED record.
    Per candidate (_finish): complete, frozen checker success over all 600 requests (incomplete = fail), strict
    admission and the required = {} variant (endpoint_ok and attach_ok and admit_no_required_ok; incomplete = refused),
    a', OOR, size log MAE and B (P sizes are the prior: unchanged by construction), both dev halves; paired house-cluster
    bootstrap CAND - rpo-dpo1 (report_v2.contrasts, 2,000 draws, seed 42). S1 is not applied here.
select     --scores RUN/dev-eval-018/scores --out RUN/dev-eval-018/selection.json --code-root CODE [--shards 6]
           [--no-submit]
    RULE (write-once; a rerun must reproduce it, code hashes aside) plus selection-report.md (Chinese). Adopted and no
    --no-submit: infer-v2-002/plan is copied to infer-v2-007/plan and the test chain is submitted with held roots
    (test shards -> infer-placement-007 -> geometry-v7 (J-SFT vs P) -> S1 on every test request
    (s1-018-test/placement-distill, the cohort comparison input) and test-report-018; merged-weight dev shards ->
    check-merged).
check-merged --data RUN/dev-select-v1 --root RUN/dev-eval-018 --ref-root RUN/dev-eval-010 --out RUN/dev-eval-018/merged-check
    Amendment 011 section 5 (dev_select.MERGED_RULE) for the adopted candidate: its BF16 merge as is vs its exact policy.
test-report --data RUN/eval-data-v1 --selection RUN/dev-eval-018/selection.json --p18 RUN/infer-placement-007
           --ref RUN/infer-placement-005 --out RUN/test-report-018
    Adopted candidate vs rpo-dpo1 on every test room: _finish metrics and paired contrasts (EXPLORATORY, reported only).
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import types

from fastfill import scene
from fastfill.comparison import contract
from fastfill.comparison import dev_select as ds
from fastfill.comparison import downstream as dn
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer as i
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import output_repair as rep
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

ARM, STAGE, REFERENCE, SFT = op.ARM, op.STAGE, op.REFERENCE, op.SFT
RUN, EVAL, TC_SCOPE = 'distill-018', 'dev-eval-018', 'a018_dpo2_train_subset'
CANDIDATES = ('sft-d', 'dpo1-d', 'dpo2-d')  # Stage order: the later stage wins checker ties.
CHAIN = (SFT, REFERENCE, *CANDIDATES)
INIT = {'sft-d': REFERENCE, 'dpo1-d': 'sft-d', 'dpo2-d': 'dpo1-d'}
ROWS = {'sft-d': 'sft-d-rows', 'dpo1-d': 'dpo1-d-pairs', 'dpo2-d': 'dpo2-d-pairs'}
STATUS = {'sft-d': 'A018_SFT_ROWS_COMPLETE', 'dpo1-d': 'A018_DPO1_PAIRS_COMPLETE', 'dpo2-d': 'OP_PAIRS_COMPLETE'}
SFT_RECIPE = {**{k: v for k, v in op.RECIPE.items() if k != 'rpo_alpha'}, 'lr': 1e-5, 'sft_only': True}
RECIPES = {'sft-d': SFT_RECIPE, 'dpo1-d': op.RECIPE, 'dpo2-d': op.RECIPE}
LOSS = {'sft-d': ("SFT-D: mean token NLL of the teacher target only (train_dpo --sft_only; no DPO term, no reference "
                  "pass, rejected unused); the policy LoRA starts as rpo-dpo1's adapter (--init_adapter) on its training base"),
        'dpo1-d': ("DPO1-D: sigmoid DPO + RPO alpha 1.0 vs the frozen SFT-D policy (training base + SFT-D's FP32 adapter, "
                   "dropout off); the policy LoRA starts as that adapter; chosen = teacher target, rejected = rpo-dpo1 failures"),
        'dpo2-d': ("DPO2-D: the amendment-013 recipe and PAIR_RULE with the sampling policy DPO1-D: sigmoid DPO + RPO alpha "
                   "1.0 vs the frozen DPO1-D policy; the policy LoRA starts as DPO1-D's adapter")}
OFF_POLICY = ("DPO1-D's rejected samples are rpo-dpo1's own (RUN/dpo-004/samples), not SFT-D's: they are off-policy by "
              "one stage")
NOT_TRAINED = 'A018_DPO2_NOT_TRAINED'
REJECTED_PER_ROOM = 2
COMPLETE_MARGIN, GAIN, ADMISSION_MARGIN, TIE, EPS = 0.005, 0.02, 0.01, 0.005, 1e-12
FIELDS = ('complete', 'geometry_success', 'accepted', 'admitted_no_required', 'success_debiased')
HALF_FIELDS = (('geometry_success_rate', 'geometry_success'), ('a_prime', 'success_debiased'), ('accepted_rate', 'accepted'))
TEST_TAG, GEOMETRY, REPORT, S1_TEST, MERGED_SHARDS = '007', 'geometry-v7', 'test-report-018', 's1-018-test', 2
GPU = ['--gpus-per-node=1', '--cpus-per-task=8', '--mem=96G']
TEST_SHARD, DEV_SHARD = [*GPU, '--time=12:00:00'], [*GPU, '--time=04:00:00']
S1_CPU = ['--cpus-per-task=72', '--mem=200G', '--time=04:00:00']
PROTOCOL, SCORES_PROTOCOL = 'amendment-018-dev-selection', 'amendment-018-dev-scores'
RULE = ("P = rpo-dpo1 unless a distilled checkpoint c in sft-d, dpo1-d, dpo2-d (each stage's final adapter) is eligible "
        "on the 600 dev-select rooms against R = rpo-dpo1: complete(c) >= complete(R) - 0.005; frozen checker success "
        "(all 600 requests, incomplete = fail) checker(c) >= checker(R) + 0.02 and the paired house-cluster bootstrap 95% "
        "CI of checker(c) - checker(R) has its lower bound > 0; strict admission(c) >= admission(R) - 0.01; "
        "checker(c) - checker(R) >= 0 in both dev halves (300 OptiScene_holodeck + IL3D_3dfront rooms, 300 other rooms); "
        "bounds inclusive except the CI. Adopted: the eligible c with the highest checker success; within 0.005 the "
        "later stage. None eligible: P stays rpo-dpo1 and nothing runs on test. A stage that was not trained is not a "
        "candidate: DPO2-D when its amendment-013 PAIR_RULE pairs on T-C miss the 1,500-pair / 600-room floors "
        f"(distill-018/dpo2-d-pairs records {NOT_TRAINED})")


def _code():
    return {**op._code(), Path(__file__).name: h.sha256(__file__)}


def model_path(run_root, name):
    return Path(run_root) / RUN / ARM / f'{name}-merged'


def _rpo(model):
    """rpo-dpo1's lineage and exact policy (dpo_stages.exact_policy rehashes its training base and adapter)."""
    lineage, policy = d.exact_policy(model, ARM)
    if lineage['stage'] != 'dpo1' or lineage['training'].get('candidate') != 'rpo':
        raise ValueError('the amendment-018 start must be the dev-selected rpo-dpo1 (dpo-003)')
    return lineage, policy


def _lines(path):
    return [h.strict_json(x) for x in Path(path).read_text().splitlines()]


def _fits(tok, prompt, ids, chosen, rejected):
    """train_dpo.DPOPairDataset's admission (as onpolicy_dpo pairs): both sequences <= 8192 tokens, prompt prefix stable."""
    c, r = tok([prompt + chosen + tok.eos_token, prompt + rejected + tok.eos_token], add_special_tokens=False)['input_ids']
    return all(len(ids) < len(x) <= d.MAX_LEN and x[:len(ids)] == ids for x in (c, r))


def _publish_rows(out, base, rows, extra):
    """dpo_stages.preflight (train_dpo admits every row) on the training base, then publish placement.jsonl + EXTRA
    ({name: bytes})."""
    payload = d._jsonl(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, prefix='.rows-', suffix='.jsonl') as stream:
        stream.write(payload)
        stream.flush()
        d.preflight(base, Path(stream.name), len(rows))
    d._publish_dir(out, {f'{ARM}.jsonl': payload, **extra})


# ---- T-C data and SFT-D rows ---------------------------------------------------------------------

TC_FILES = ('joint/requests.jsonl', 'placement/requests.jsonl', 'rooms.jsonl', 'size-estimates.jsonl')


def tc_data(args):
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    from fastfill.comparison import distill_018  # Section 1 (plan / teacher), imported lazily.
    plan = distill_018.load_plan(args.plan)
    root, manifest, uids = plan['root'], plan['manifest'], sorted(plan['sets']['T-C'])
    keep = set(uids)
    files = {n: [r for r in h._rows(root, n, manifest['files'][n]) if r['uid'] in keep] for n in TC_FILES}
    if any([r['uid'] for r in rows] != uids for rows in files.values()):
        raise ValueError('plan rows do not cover T-C in expected_uids order')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        for name, rows in files.items():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            (dest / name).write_bytes(d._jsonl(rows))
        h.write_json(dest / 'MANIFEST.json', {
            'version': 1, 'scope': TC_SCOPE, 'split': 'train', 'expected_uids': uids, 'uids_sha256': d._digest(uids),
            'counts': {'train': len(uids)}, 'training_manifest_sha256': manifest['training_manifest_sha256'],
            'main_data_manifest_sha256': manifest['main_data_manifest_sha256'],
            'sft_prompt_tokens': {u: manifest['sft_prompt_tokens'][u] for u in uids},
            'input_policy': manifest['input_policy'],
            'selection': {'plan': {'path': str(Path(root).resolve()), 'manifest_sha256': h.sha256(Path(root) / 'MANIFEST.json'),
                                   'uids_sha256': h.sha256(Path(root) / distill_018.UID_FILES['T-C'])},
                          'T-C': manifest['selection']['T-C']},
            'code_sha256': {Path(__file__).name: h.sha256(__file__)}, 'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return {'status': 'A018_TC_DATA_COMPLETE', 'rooms': len(uids), 'uids_sha256': d._digest(uids)}


def _target_ok(task):
    room, prompt, target = task
    try:
        canonical = contract._json(h.strict_json(target)) == target
    except ValueError:
        return False
    row = g.score(room, prompt, target, ARM)
    return canonical and row['complete'] and row['geometry_success']


def sft_data(args):
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    pairs, data = h.load_pair(args.data)
    if data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('SFT-D rows require the qualified main data')
    admitted, prompt_tokens, _ = d._admitted(args.sft_runs, data, 'train')
    train = h._by_uid(pairs['train'][ARM])
    rooms = h._by_uid(h._rows(Path(args.data), 'train_rooms.jsonl', data['files']['train_rooms.jsonl']))
    positives, _, _, positives_sha = d.load_positives(args.exclude)
    teacher = Path(args.teacher)
    from fastfill.comparison import distill_018  # Section 1 (plan / teacher), imported lazily.
    targets = sorted(distill_018.load_targets(teacher)[0], key=lambda t: t['uid'])
    uids, pool = [t['uid'] for t in targets], set(admitted) - set(positives['uids'])
    outside = [u for u in uids if u not in pool]
    if outside:
        raise ValueError(f'{len(outside)} teacher rooms outside the length-admitted train pool minus the DPO1 positives: '
                         f'{outside[:5]}')
    held = [op._held_out(roots) for roots in (args.dev, args.test)]
    leaked = [u for u in uids if any(u in x or train[u]['group'] in groups for x, groups, _ in held)]
    if leaked:
        raise ValueError(f'{len(leaked)} teacher rooms share a UID or house group with dev/test: {leaked[:5]}')
    foreign = [t['uid'] for t in targets if t['prompt'] != train[t['uid']]['messages'][:2]]
    if foreign:
        raise ValueError(f"{len(foreign)} teacher prompts are not P's SFT training prompt: {foreign[:5]}")
    failing = [u for u, ok in zip(uids, r2._map(_target_ok, [(rooms[t['uid']], t['prompt'], t['target']) for t in targets],
                                                 args.workers)) if not ok]
    if failing:
        raise ValueError(f'{len(failing)} teacher targets are not canonical, complete frozen-checker passes: {failing[:5]}')
    _, policy = _rpo(Path(args.base_model))
    tok = d._tokenizer(policy['training_base'])  # train_dpo's --model: rpo-dpo1's training base.
    prompts = [scene.prompt_text(tok, t['prompt']) for t in targets]
    prompt_ids = tok(prompts, add_special_tokens=False)['input_ids']
    drift = [u for u, p in zip(uids, prompt_ids) if len(p) != prompt_tokens[u][ARM]]
    if drift:
        raise ValueError(f'{len(drift)} prompts tokenize differently from SFT training: {drift[:5]}')
    rows, overlength = [], []
    for t, prompt, ids in zip(targets, prompts, prompt_ids):
        if not _fits(tok, prompt, ids, t['target'], t['target']):
            overlength.append(t['uid'])
            continue
        meta = {k: train[t['uid']][k] for k in ('uid', 'source', 'group')}
        rows.append({**meta, 'set': t['set'], 'prompt': t['prompt'], 'chosen': t['target'], 'rejected': t['target']})
    if not rows:
        raise ValueError('no SFT-D rows')
    summary = {'status': STATUS['sft-d'], 'rows': len(rows), 'by_set': dict(sorted(Counter(r['set'] for r in rows).items())),
               'by_source': dict(sorted(Counter(r['source'] for r in rows).items())), 'overlength_dropped': overlength,
               'rule': ('every teacher target of T-A and T-B as chosen (rejected = chosen, unused by --sft_only); sorted by '
                        'uid; rows over 8192 tokens dropped'),
               'inputs': {'teacher_manifest_sha256': h.sha256(teacher / 'MANIFEST.json'),
                          'main_data_manifest_sha256': data['manifest_sha256'],
                          'excluded_positives_manifest_sha256': positives_sha,
                          'held_out': [receipts for _, _, receipts in held], 'tokenizer_model': policy['training_base']},
               'code_sha256': _code()}
    _publish_rows(out, policy['training_base'], rows, {'summary.json': op._json_bytes(summary)})
    return {'status': summary['status'], 'rows': len(rows), 'by_set': summary['by_set'], 'overlength': len(overlength)}


# ---- DPO1-D pairs (T-A) --------------------------------------------------------------------------

def _dpo1_room(task):
    """One T-A room: chosen = the teacher target, rejected = rpo-dpo1's complete failing samples in tier order."""
    room, request, prior, samples, chosen = task
    if any(s[k] != room[k] or request[k] != room[k] for s in samples for k in ('uid', 'source', 'group')):
        raise ValueError('sample/request/reference identity mismatch')
    teacher, _, verdict = op._sample(room, request, prior, {'k': None, 'raw': chosen, 'error': None, 'finish_reason': 'stop'})
    if not teacher['geometry_success']:
        raise ValueError(f"teacher target fails the frozen checker: {room['uid']}")
    scored = [op._sample(room, request, prior, s) for s in samples]
    ledger = [{'k': s['k'], 'finish_reason': s['finish_reason'], 'tier': None,
               'class': 'incomplete' if not row['complete'] else 'pass' if op._passes(row, v) else 'reject'}
              for s, (row, _, v) in zip(samples, scored)]
    rejected = [k for k, x in enumerate(ledger) if x['class'] == 'reject']
    for k in rejected:
        ledger[k]['tier'] = op.tier(scored[k][0])
    rejected.sort(key=lambda k: (op.TIERS.index(ledger[k]['tier']), k))
    pairs, seen = [], {chosen}
    for k in rejected:
        raw, (row, _, v) = samples[k]['raw'], scored[k]
        if raw in seen or len(pairs) == REJECTED_PER_ROOM:
            ledger[k]['drop'] = 'duplicate_text' if raw in seen else 'room_cap'
            continue
        seen.add(raw)
        pairs.append({'uid': room['uid'], 'source': room['source'], 'group': room['group'], 'reject_kind': 'rpo_dpo1_failure',
                      'prompt': request['messages'], 'chosen': chosen, 'rejected': raw, 'chosen_from': 'teacher_target',
                      'rejected_k': k, 'rejected_tier': ledger[k]['tier'],
                      'violations': {'chosen': {x: teacher[x] for x in op.bo.VIOLATIONS},
                                     'rejected': {x: row[x] for x in op.bo.VIOLATIONS}},
                      'downstream': {'chosen': verdict['reasons'], 'rejected': v['reasons']}})
    return {'pairs': pairs, 'samples': ledger, 'teacher_accepted': verdict['accepted']}


def pairs_dpo1(args):
    data, samples, out, targets = Path(args.data), Path(args.samples), Path(args.out), Path(args.targets)
    if os.path.lexists(out):
        raise FileExistsError(out)
    manifest = e._verified(data)
    if manifest.get('scope') != op.SUBSET_SCOPE:
        raise ValueError('DPO1-D pairs are built on T-A (the amendment-013 training subset) only')

    def rows(name):
        return h._by_uid(h._rows(data, name, manifest['files'][name]))
    rooms, requests, priors = rows('rooms.jsonl'), rows(f'{ARM}/requests.jsonl'), rows('size-estimates.jsonl')
    receipt, grouped = op._sampling(data, samples, manifest)
    model = Path(args.base_model).resolve()
    _, policy = _rpo(model)
    if (receipt['model'].get('policy') != policy
            or receipt['model'].get('lineage_sha256') != h.sha256(d.receipt_path(model))):
        raise ValueError("samples are not of rpo-dpo1's exact policy")
    d._verified_dir(targets)
    if h.strict_json((targets / 'summary.json').read_text()).get('status') != STATUS['sft-d']:
        raise ValueError(f'{targets} are not the SFT-D rows')
    chosen = h._by_uid(_lines(targets / f'{ARM}.jsonl'))
    uids = [u for u in manifest['expected_uids'] if u in chosen]
    if any(chosen[u]['prompt'] != requests[u]['messages'] for u in uids):
        raise ValueError('a teacher prompt differs from the T-A request')
    tok = d._tokenizer(policy['training_base'])
    prompts = [scene.prompt_text(tok, requests[u]['messages']) for u in uids]
    prompt_ids = tok(prompts, add_special_tokens=False)['input_ids']
    drift = [u for u, p in zip(uids, prompt_ids) if len(p) != grouped[u][0]['input_tokens']]
    if drift:
        raise ValueError(f'{len(drift)} prompts tokenize differently from the sampled prompts (template drift): {drift[:5]}')
    results = r2._map(_dpo1_room, [(rooms[u], requests[u], priors[u]['estimated_sizes'], grouped[u], chosen[u]['chosen'])
                                   for u in uids], args.workers)
    kept, ledger, overlength = [], [], 0
    for uid, prompt, ids, result in zip(uids, prompts, prompt_ids, results):
        used = [p for p in result['pairs'] if _fits(tok, prompt, ids, p['chosen'], p['rejected'])]
        overlength += len(result['pairs']) - len(used)
        kept += used
        ledger.append({'uid': uid, 'source': rooms[uid]['source'], 'bin': op.bo.bin_name(len(rooms[uid]['objects'])),
                       'samples': result['samples'], 'teacher_accepted': result['teacher_accepted'], 'pairs': len(used)})
    if not kept:
        raise ValueError('no DPO1-D pairs')
    summary = {'status': STATUS['dpo1-d'], 'note': OFF_POLICY, 'rooms_t_a': len(manifest['expected_uids']),
               'rooms_with_target': len(uids), 'rooms_with_pair': sum(x['pairs'] > 0 for x in ledger), 'pairs': len(kept),
               'pairs_per_room': dict(sorted(Counter(x['pairs'] for x in ledger).items())),
               'by_tier': dict(sorted(Counter(p['rejected_tier'] for p in kept).items())),
               'by_source': dict(sorted(Counter(p['source'] for p in kept).items())),
               'chosen_not_admitted_pairs': sum(x['pairs'] for x in ledger if not x['teacher_accepted']),
               'overlength_dropped': overlength,
               'samples': dict(sorted(Counter(s['class'] for x in ledger for s in x['samples']).items())),
               'rule': (f'per T-A room with a teacher target: chosen = the teacher target; rejected = up to {REJECTED_PER_ROOM} '
                        "of rpo-dpo1's complete samples failing the frozen checker or the strict admission, tiers "
                        'colocated_duplicate, main_collision, other, then k; identical texts collapse'),
               'inputs': {'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
                          'samples_manifest_sha256': h.sha256(samples / 'MANIFEST.json'),
                          'samples_run_sha256': receipt['run_sha256'], 'reference_model': str(model),
                          'reference_lineage_sha256': h.sha256(d.receipt_path(model)),
                          'targets_manifest_sha256': h.sha256(targets / 'MANIFEST.json')},
               'code_sha256': _code()}
    _publish_rows(out, policy['training_base'], kept, {'ledger.jsonl': d._jsonl(ledger),
                                                       'summary.json': op._json_bytes(summary)})
    return {'status': summary['status'], 'pairs': len(kept), 'rooms_with_pair': summary['rooms_with_pair']}


# ---- DPO2-D pairs (T-C, on-policy) ---------------------------------------------------------------

def pairs_dpo2(args):
    data = Path(args.data)
    manifest = e._verified(data)
    if manifest.get('scope') != TC_SCOPE:
        raise ValueError('DPO2-D pairs are built on T-C only')
    d._verified_dir(args.targets)
    trained = set(e._verified(Path(args.t_a))['expected_uids']) | {r['uid'] for r in _lines(Path(args.targets) / f'{ARM}.jsonl')}
    overlap = sorted(trained & set(manifest['expected_uids']))
    if overlap:
        raise ValueError(f'{len(overlap)} T-C rooms are T-A or SFT-D training rooms: {overlap[:5]}')
    out = Path(args.out)

    def short(summary, ledger):  # Below the floors: record why DPO2-D is not trained; nothing trainable is published.
        summary = {**summary, 'status': NOT_TRAINED,
                   'note': 'below the amendment-013 floors: DPO2-D is not trained and is not a dev candidate'}
        d._publish_dir(out, {'ledger.jsonl': d._jsonl(ledger), 'summary.json': op._json_bytes(summary)})
        return {'status': NOT_TRAINED, 'pairs': summary['pairs'], 'rooms_with_pair': summary['rooms_with_pair']}
    return op.build_pairs(types.SimpleNamespace(data=data, samples=args.samples, base_model=args.base_model, out=out,
                                                workers=args.workers, dry_run=None),
                          scope=TC_SCOPE, sampler=(STAGE, INIT['dpo2-d']), short=short)


def not_trained(run_root):
    """True iff RUN_ROOT/distill-018/dpo2-d-pairs is a verified NOT_TRAINED record (pairs-dpo2 below the floors)."""
    path = Path(run_root) / RUN / ROWS['dpo2-d']
    if not path.exists():
        return False
    d._verified_dir(path)
    return h.strict_json((path / 'summary.json').read_text()).get('status') == NOT_TRAINED


# ---- train (SFT-D, DPO1-D, DPO2-D) ---------------------------------------------------------------

def train(args):
    name, run_root = args.candidate, Path(args.run_root)
    if name == 'dpo2-d' and not_trained(run_root):
        return {'status': NOT_TRAINED, 'candidate': name}
    root = run_root / RUN
    out, merged = root / ARM / name, model_path(run_root, name)
    if d.receipt_path(merged).exists():
        return d.verify_merged(merged, ARM, STAGE)
    rpo = (run_root / op.PARENT_RUN / ARM / 'dpo1-merged').resolve()
    lineage, start = _rpo(rpo)
    base, previous = Path(start['training_base']), INIT[name]
    if previous == REFERENCE:
        before, init = None, {'of': REFERENCE, 'path': start['adapter'], 'sha256': start['adapter_sha256'],
                              'lineage_sha256': h.sha256(d.receipt_path(rpo))}
    else:
        source = model_path(run_root, previous).resolve()
        before, policy = d.exact_policy(source, ARM)
        if before['training'].get('candidate') != previous or policy['training_base'] != str(base):
            raise ValueError(f'{name} must start from {previous} on the rpo-dpo1 training base')
        init = {'of': previous, 'path': policy['adapter'], 'sha256': policy['adapter_sha256'],
                'lineage_sha256': h.sha256(d.receipt_path(source))}
    rows_dir = root / ROWS[name]
    files = d._verified_dir(rows_dir)
    summary = h.strict_json((rows_dir / 'summary.json').read_text())
    sampled = {'dpo1-d': h.sha256(d.receipt_path(rpo)), 'dpo2-d': init['lineage_sha256']}
    if (summary.get('status') != STATUS[name]
            or (name in sampled and summary['inputs']['reference_lineage_sha256'] != sampled[name])
            or (name == 'dpo1-d' and summary['inputs']['targets_manifest_sha256']
                != before['training']['pairs']['manifest_sha256'])):
        raise ValueError(f'{rows_dir} is not the {name} training data of this chain')
    rows = (rows_dir / f'{ARM}.jsonl').resolve()
    count = len(rows.read_bytes().splitlines())  # Up to 4 pairs per room: h._rows would refuse repeated UIDs.
    recipe = RECIPES[name]
    from fastfill import merge_lora
    from fastfill.v1_1 import train_dpo, training_guard
    settings = {**recipe, **d.TRAIN_FIXED, 'loss': LOSS[name],
                'init_adapter': {'path': init['path'], 'sha256': init['sha256'], 'of': init['of']}}
    training_guard.guard_training(base, rows, out, settings, [Path(train_dpo.__file__), Path(d.__file__), Path(op.__file__),
                                                              Path(__file__), Path(merge_lora.__file__)])
    if not (out / 'final' / 'completion.json').exists():
        d.preflight(base, rows, count)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={d.WORLD_SIZE}',
                   '-m', 'fastfill.v1_1.train_dpo', '--model', str(base), '--pairs', str(rows), '--out', str(out),
                   '--init_adapter', init['path']]
        subprocess.run(command + d._flags(recipe), check=True)
    completion = d._completion(out, count, recipe)
    d._merge(base, out / 'final', merged, {
        'arm': ARM, 'stage': STAGE,
        'parent': {'stage': 'dpo1', 'path': str(rpo), 'lineage_sha256': h.sha256(d.receipt_path(rpo))},
        'training_base': {'stage': 'sft', 'path': str(base), 'model_sha256': start['training_base_sha256'],
                          'lineage_sha256': h.sha256(d.receipt_path(base)), 'is_dpo1_merge': False},
        'merged_model': op.MERGED_MODEL, **{k: lineage[k] for k in ('base', 'base_sha256', 'training_data')},
        'training': {'pairs': {'path': str(rows), 'count': count, 'sha256': h._expected_hash(files[rows.name]),
                               'manifest_sha256': h.sha256(rows_dir / 'MANIFEST.json')},
                     'guard_sha256': h.sha256(out / 'training_guard.json'), 'completion': completion, 'recipe': settings,
                     'candidate': name, 'init': init, 'chain': list(CHAIN[:CHAIN.index(name) + 1]),
                     **({'note': OFF_POLICY} if name == 'dpo1-d' else {}),
                     'distill_018_sha256': h.sha256(__file__)}})
    return d.verify_merged(merged, ARM, STAGE)


# ---- dev score and select ------------------------------------------------------------------------

def _bind(receipt, run_root, candidate):
    """A dev/test inference must come from this candidate: exact policy, or (-merged) its BF16 merge as is."""
    config, name = receipt['config'], candidate.removesuffix('-merged')
    loaded = ((config.get('lora') == ds.PREMERGED and 'model_sha256' in receipt and 'policy' not in receipt)
              if name != candidate else config.get('lora') == ds.EXACT)
    if not (name in CANDIDATES and loaded and receipt.get('stage') == STAGE and config.get('arm') == ARM
            and config.get('model') == str(model_path(run_root, name).resolve())):
        raise ValueError(f'inference is not {ARM}/{candidate}')


def _admission(task):
    """(strict admission, required = {} variant) of one scored row; an incomplete row is refused by both."""
    row, room, request = task
    if not row['complete']:
        return False, False
    v = dn.accept(rep.predicted_room(room, request['messages'], row['raw_layout']), row['raw_layout'])
    return v['accepted'], v['endpoint_ok'] and v['attach_ok'] and v['admit_no_required_ok']


def _finish(rows, sft_rows, rooms, requests, workers):
    """dev_select._finished (metrics, own-bias a') plus both admission rules, OOR and the per-half rates."""
    value, payload = ds._finished(rows, sft_rows, rooms, {ARM: requests}, ARM, workers)
    verdicts = r2._map(_admission, [(r, rooms[r['uid']], requests[r['uid']]) for r in rows], workers)
    rows = [{**h.strict_json(line), 'accepted': a, 'admitted_no_required': b}
            for line, (a, b) in zip(payload.decode().splitlines(), verdicts, strict=True)]
    oor = [r['oor'] for r in rows if r['complete'] and r['oor'] is not None]
    value.update({'accepted_rate': op._rate(rows, 'accepted'), 'no_required_rate': op._rate(rows, 'admitted_no_required'),
                  'oor_mean': statistics.mean(oor) if oor else None,
                  'halves': {name: {'rooms': len(part), **{k: op._rate(part, f) for k, f in HALF_FIELDS}}
                             for name, part in op._halves(rows).items()}})
    return value, rows


def dev_score(args):
    data, root, ref, out = Path(args.data), Path(args.root), Path(args.ref_root), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    manifest, rooms, requests, record, runs, _ = op._reference(data, ref, args.workers)
    untrained = {'dpo2-d': h.sha256(root.parent / RUN / ROWS['dpo2-d'] / 'MANIFEST.json')} if not_trained(root.parent) else {}
    scored = [c for c in CANDIDATES if c not in untrained]
    runs.update({c: op._scored(root, c, data, manifest, rooms, requests[ARM], args.workers, bind=_bind) for c in scored})
    if {run[1]['context_plan_sha256'] for run in runs.values()} != {record['plan_sha256']}:
        raise ValueError('candidates were not run on the amendment-010 shared dev plan')
    result, rows = {}, {}
    for c in (REFERENCE, *scored):
        value, rows[c] = _finish(runs[c][2], runs[SFT][2], rooms, requests[ARM], args.workers)
        result[c] = {**runs[c][3], **value}
    for c in scored:
        result[c]['model'] = str(model_path(root.parent, c).resolve())
    value = {'protocol': SCORES_PROTOCOL, 'arm': ARM, 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
             'uids_sha256': manifest['uids_sha256'], 'plan_sha256': record['plan_sha256'],
             'reference_scores': {'path': str((ref / 'scores').resolve()),
                                  'manifest_sha256': h.sha256(ref / 'scores' / 'MANIFEST.json')},
             'candidates': result, 'not_trained': untrained, 'note': OFF_POLICY,
             'contrasts_vs_rpo_dpo1': {c: r2.contrasts(rows[c], rows[REFERENCE], FIELDS, op.REPEATS, r2.SEED)
                                       for c in scored},
             'scope': ('rates over all 600 requests (incomplete = failure); geometry_success = frozen checker success '
                       '(primary, amendment 017); accepted = strict admission port; admitted_no_required = endpoint and '
                       'attach and admit with required = {}; contrasts: paired house-cluster bootstrap CAND - rpo-dpo1; P '
                       "sizes are the request prior, so size log MAE and B are not changed by any candidate's training"),
             'code_sha256': _code()}
    d._publish_dir(out, {'scores.json': op._json_bytes(value),
                         **{f'{ARM}-{c}-geometry.jsonl': d._jsonl(rows[c]) for c in rows}})
    return {'status': 'A018_DEV_SCORES_COMPLETE', **{c: {k: round(result[c][k], 4) for k in (
        'geometry_success_rate', 'accepted_rate', 'complete_rate')} for c in result}}


def checks(candidates, contrasts, name):
    """RULE's four eligibility tests of NAME against rpo-dpo1."""
    m, ref, ci = candidates[name], candidates[REFERENCE], contrasts[name]['geometry_success']['ci95']
    return {'complete': m['complete_rate'] >= ref['complete_rate'] - COMPLETE_MARGIN - EPS,
            'checker_gain': (m['geometry_success_rate'] >= ref['geometry_success_rate'] + GAIN - EPS
                             and ci is not None and ci[0] > 0),
            'strict_admission': m['accepted_rate'] >= ref['accepted_rate'] - ADMISSION_MARGIN - EPS,
            'both_halves': all(m['halves'][k]['geometry_success_rate'] >= ref['halves'][k]['geometry_success_rate'] - EPS
                               for k in op.HALVES)}


def choose(candidates, contrasts, untrained=()):
    """RULE on {name: metrics} and {name: contrasts vs rpo-dpo1}, rpo-dpo1 and every trained candidate present
    (untrained: dev-score's NOT_TRAINED record names, DPO2-D at most)."""
    untrained = sorted(untrained)
    scored = [c for c in CANDIDATES if c not in untrained]
    if (not set(untrained) <= {'dpo2-d'} or set(untrained) & set(candidates) or not {REFERENCE, *scored} <= set(candidates)
            or not set(scored) <= set(contrasts)):
        raise ValueError('rpo-dpo1 and every trained amendment-018 candidate must be scored')
    tests = {c: checks(candidates, contrasts, c) for c in scored}
    names = [c for c in scored if all(tests[c].values())]
    if not names:
        return {'selected': REFERENCE, 'adopted': False, 'eligible': [], 'tied': [], 'best_checker': None, 'checks': tests,
                'not_trained': untrained}
    best = max(candidates[c]['geometry_success_rate'] for c in names)
    tied = [c for c in names if candidates[c]['geometry_success_rate'] >= best - TIE - EPS]
    return {'selected': tied[-1], 'adopted': True, 'eligible': names, 'tied': tied, 'best_checker': best,
            'checks': tests, 'not_trained': untrained}  # The later stage within 0.005.


def load_selection(path):
    """selection.json bound to its scores dir and reproduced by choose -> (value, scores)."""
    path = Path(path)
    value, scores_dir = h.strict_json(path.read_text()), path.parent / 'scores'
    d._verified_dir(scores_dir)
    scores = h.strict_json((scores_dir / 'scores.json').read_text())
    again = choose(scores['candidates'], scores['contrasts_vs_rpo_dpo1'], scores['not_trained'])
    if (value.get('protocol') != PROTOCOL or value.get('rule') != RULE or scores.get('protocol') != SCORES_PROTOCOL
            or value.get('scores_manifest_sha256') != h.sha256(scores_dir / 'MANIFEST.json')
            or {k: value.get(k) for k in again} != again or value.get('candidates') != scores['candidates']):
        raise ValueError(f'{path} is not bound to its dev scores')
    return value, scores


def _adopted(path):
    value, scores = load_selection(path)
    if not value['adopted']:
        raise ValueError('no amendment-018 candidate was adopted: P stays rpo-dpo1 and nothing runs on test')
    return value, scores


def _num(value):
    return '—' if value is None else f'{value:.3f}'


def report(value, contrasts):
    """selection-report.md (Chinese)."""
    scored = [n for n in CANDIDATES if n not in value['not_trained']]
    c, names = value['candidates'], (REFERENCE, *scored)
    labels = {REFERENCE: 'R = rpo-dpo1', 'sft-d': 'SFT-D', 'dpo1-d': 'DPO1-D', 'dpo2-d': 'DPO2-D'}
    cols = (('complete_rate', '完整率'), ('geometry_success_rate', '检查器成功率'), ('accepted_rate', '严格准入'),
            ('no_required_rate', 'required = {} 准入'), ('a_prime', 'a′'), ('oor_mean', 'OOR'))
    lines = ['# 修订 018 dev 选择（dev-select-v1，600 个房间，HF 精确策略）', '',
             '| 候选 | ' + ' | '.join(t for _, t in cols) + ' | 检查器 论文源半 | 检查器 其他源半 |',
             '|---|' + '---|' * (len(cols) + 2)]
    lines += [f"| {labels[n]} | " + ' | '.join(_num(c[n][k]) for k, _ in cols) + ' | '
              + ' | '.join(_num(c[n]['halves'][x]['geometry_success_rate']) for x in op.HALVES) + ' |' for n in names]
    lines += ['', '## 检查器成功率差（候选 − R；房屋连通组配对 bootstrap，2,000 次，seed 42，95% CI）', '']
    for n in scored:
        x = contrasts[n]['geometry_success']
        ci = '未定义' if x['ci95'] is None else f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
        ok = '、'.join(k for k, v in value['checks'][n].items() if not v) or '全部满足'
        lines.append(f"- {labels[n]}：{x['estimate']:+.3f} {ci}；未满足的条件：{ok}")
    decision = (f"采纳 {labels[value['selected']]}（交付形式：训练基座 + 该候选的 FP32 adapter；谱系 SFT → rpo-dpo1 → SFT-D → "
                'DPO1-D → DPO2-D 中到该阶段为止）' if value['adopted'] else 'P 保持 rpo-dpo1，测试集不运行任何修订 018 推理')
    lines += ['', f'## 决定：{decision}', '',
              '规则（修订 018 §3）：完整率 ≥ R − 0.005；检查器成功率 ≥ R + 0.02 且配对 CI 下界 > 0；严格准入 ≥ R − 0.01；'
              '两个 dev 半区的检查器增益 ≥ 0。合格者中取检查器成功率最高者，0.005 以内取较后阶段。', '',
              '说明：', '- DPO1-D 的 rejected 是 rpo-dpo1 自己的样本（RUN/dpo-004/samples），相对 SFT-D 落后一个阶段（off-policy）。',
              '- P 的尺寸取自请求中的训练先验，尺寸 log MAE 与足迹偏差 B 不随训练改变。',
              '- 600 个房间上成功率差的标准误约 0.012；本表只用于 dev 选择，测试结果另行报告（探索性）。']
    if 'dpo2-d' in scored:
        lines.append("- DPO2-D 的对复用 013 的 PAIR_RULE 代码，dpo2-d-pairs/summary.json 的 rule 文字沿用 013 写法"
                     "（“rpo-dpo1 的样本”）；实际采样策略是 DPO1-D（见其 inputs.reference_model）。")
    else:
        lines.append('- DPO2-D 未训练：T-C 上按 013 PAIR_RULE 建出的对低于下限（1,500 对 / 600 间），'
                     f'distill-018/dpo2-d-pairs 记为 {NOT_TRAINED}（含计数），因此不是候选。')
    return '\n'.join(lines) + '\n'


def chain_jobs(code_root, run_root, candidate, shards):
    """sbatch jobs (label, afterok labels, argv) of the adopted candidate's test run, S1 and merged-weight check."""
    script = str(Path(code_root) / 'fastfill/comparison/isambard_018.sbatch')
    merge, code, run_root = str(Path(code_root) / 'fastfill/comparison/isambard_infer_merge.sbatch'), str(code_root), Path(run_root)
    tests, checks_ = [f'test shard {k}' for k in range(shards)], [f'merged dev shard {k}' for k in range(MERGED_SHARDS)]
    jobs = [{'label': label, 'after': [], 'argv': [*TEST_SHARD, script, code, 'shard', 'test', candidate, str(k), str(shards)]}
            for k, label in enumerate(tests)]
    jobs += [{'label': 'test merge', 'after': tests,
              'argv': [merge, code, ARM, str(shards), str(run_root / f'infer-v2-{TEST_TAG}'), str(run_root / f'infer-{ARM}-{TEST_TAG}')]},
             {'label': 'test geometry', 'after': ['test merge'], 'argv': [script, code, 'test-geometry']},
             {'label': 'test report', 'after': ['test merge'], 'argv': ['--time=06:00:00', script, code, 'test-report']},
             {'label': 'test s1', 'after': ['test geometry'], 'argv': [*S1_CPU, script, code, 's1-test']}]
    jobs += [{'label': label, 'after': [], 'argv': [*DEV_SHARD, script, code, 'shard', 'dev', candidate + '-merged', str(k),
                                                    str(MERGED_SHARDS)]} for k, label in enumerate(checks_)]
    jobs += [{'label': 'merged dev merge', 'after': checks_,
              'argv': [script, code, 'dev-merge', candidate + '-merged', str(MERGED_SHARDS)]},
             {'label': 'check-merged', 'after': ['merged dev merge'], 'argv': [script, code, 'check-merged']}]
    return jobs


def select(args, run=subprocess.run, which=shutil.which):
    scores_dir, out = Path(args.scores), Path(args.out)
    d._verified_dir(scores_dir)
    scores = h.strict_json((scores_dir / 'scores.json').read_text())
    if scores.get('protocol') != SCORES_PROTOCOL:
        raise ValueError(f'{scores_dir} is not an amendment-018 dev score')
    run_root = scores_dir.parent.parent
    choice = choose(scores['candidates'], scores['contrasts_vs_rpo_dpo1'], scores['not_trained'])
    test = {'candidate': choice['selected'], 'model': str(model_path(run_root, choice['selected']).resolve()),
            'inference': str(run_root / f'infer-{ARM}-{TEST_TAG}'), 'joint': str(run_root / f'infer-joint-{ds.SFT_TEST_TAG}'),
            'geometry': str(run_root / GEOMETRY), 'report': str(run_root / REPORT),
            's1': str(run_root / S1_TEST / f'{ARM}-distill')} if choice['adopted'] else None
    value = {'protocol': PROTOCOL, 'rule': RULE, 'scores_manifest_sha256': h.sha256(scores_dir / 'MANIFEST.json'),
             'data_manifest_sha256': scores['data_manifest_sha256'], 'uids_sha256': scores['uids_sha256'], 'arm': ARM,
             'reference': REFERENCE, **choice, 'candidates': scores['candidates'], 'test': test,
             'code_sha256': {Path(__file__).name: h.sha256(__file__)}}
    if out.exists():  # A rerun (submit_018_chain.sh test, maybe from a newer snapshot) must reproduce it, code hashes aside.
        if {**h.strict_json(out.read_text()), 'code_sha256': None} != {**value, 'code_sha256': None}:
            raise ValueError(f'{out} exists with another selection')
    else:
        i.write_once(out, value)
    path = out.with_name('selection-report.md')
    if not path.exists():  # Derived from selection.json; a rerun keeps the first one.
        from fastfill.comparison.ablation_016 import _write_new
        _write_new(path, report(value, scores['contrasts_vs_rpo_dpo1']).encode('utf-8'))
    if not choice['adopted']:
        return {'status': 'NOT_ADOPTED', 'selected': REFERENCE, 'note': 'P stays rpo-dpo1; nothing runs on test'}
    receipt = out.with_name('submission.json')
    if receipt.exists():
        raise FileExistsError(f'test chain already submitted: {receipt}')
    if getattr(args, 'no_submit', False):
        return {'status': 'ADOPTED', 'selected': choice['selected']}
    fresh = [run_root / f'infer-v2-{TEST_TAG}' / ARM, run_root / f'infer-{ARM}-{TEST_TAG}', run_root / GEOMETRY,
             run_root / REPORT, run_root / S1_TEST, out.parent / (choice['selected'] + '-merged'), out.parent / 'merged-check']
    taken = [str(p) for p in fresh if os.path.lexists(p)]
    if taken:
        raise FileExistsError(f'refusing existing test outputs: {taken}')
    from fastfill.comparison import infer_shard
    src, plan = run_root / f'infer-v2-{ds.SFT_TEST_TAG}/plan', run_root / f'infer-v2-{TEST_TAG}/plan'
    if not plan.exists():
        d.copy_plan(types.SimpleNamespace(src=src, out=[plan]))
    elif infer_shard.load_plan(plan)[1] != infer_shard.load_plan(src)[1]:
        raise ValueError(f'{plan} is not the infer-v2-{ds.SFT_TEST_TAG} plan')
    jobs = chain_jobs(args.code_root, run_root, choice['selected'], args.shards)
    ids = ds.submit(jobs, run, which)
    i.write_once(receipt, {'selection_sha256': h.sha256(out), 'jobs': [{**j, 'id': ids[j['label']]} for j in jobs]})
    return {'status': 'TEST_CHAIN_SUBMITTED', 'selected': choice['selected'], 'jobs': ids}


# ---- merged-weight check and test report (only after adoption) ------------------------------------

def check_merged(args):
    data, root, ref, out = Path(args.data), Path(args.root), Path(args.ref_root), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    selection = root / 'selection.json'
    value, scores = _adopted(selection)
    candidate = value['selected']
    merged = candidate + '-merged'
    manifest, rooms, requests = ds._dev(data)
    sft = ds._scored(ref, SFT, ARM, data, manifest, rooms, requests, args.workers)
    runs = {c: op._scored(root, c, data, manifest, rooms, requests[ARM], args.workers, bind=_bind) for c in (candidate, merged)}
    if {run[1]['context_plan_sha256'] for run in (sft, *runs.values())} != {scores['plan_sha256']}:
        raise ValueError('merged check not on the shared dev plan')
    same = [{k: v for k, v in runs[c][1]['config'].items() if k not in ('lora', 'shard_count')} for c in runs]
    if same[0] != same[1]:
        raise ValueError('merged and unmerged dev runs differ beyond the weight loading')
    result, files = {}, {}
    for c, (_, _, rows, info) in runs.items():
        metrics, rows = _finish(rows, sft[2], rooms, requests[ARM], args.workers)
        result[c], files[f'{ARM}-{c}-geometry.jsonl'] = {**info, **metrics}, d._jsonl(rows)
    entry = scores['candidates'][candidate]
    if any(result[candidate][k] != entry[k] for k in ('a_prime', 'complete_rate', 'geometry_success_rate', 'accepted_rate')) or (
            result[candidate]['inference']['run_sha256'] != entry['inference']['run_sha256']):
        raise ValueError(f'unmerged {candidate} rescoring differs from its dev score')
    delta = {k: result[merged][k] - result[candidate][k] for k in ('complete_rate', 'geometry_success_rate', 'a_prime',
                                                                    'accepted_rate')}
    listed = abs(delta['a_prime']) <= ds.MERGED_A_PRIME + EPS and abs(delta['complete_rate']) <= ds.MERGED_COMPLETE + EPS
    identical = sum(a['raw'] is not None and a['raw'] == b['raw'] for a, b in zip(runs[candidate][0], runs[merged][0], strict=True))
    check = {'protocol': 'amendment-018-merged-weight-check', 'arm': ARM, 'candidate': candidate, 'rule': ds.MERGED_RULE,
             'thresholds': {'a_prime': ds.MERGED_A_PRIME, 'complete_rate': ds.MERGED_COMPLETE},
             'selection_sha256': h.sha256(selection), 'rooms': len(runs[merged][0]), 'unmerged': result[candidate],
             'merged': result[merged], 'delta_merged_minus_unmerged': delta,
             'byte_identical': {'rooms': identical, 'fraction': identical / len(runs[merged][0])},
             'merged_listed_as_convenience': listed,
             'decision': 'merged_convenience_artifact' if listed else 'adapter_and_training_base_only', 'code_sha256': _code()}
    d._publish_dir(out, {'check.json': op._json_bytes(check), **files})
    return {'status': 'MERGED_CHECK_COMPLETE', 'decision': check['decision'], 'delta': {k: round(v, 4) for k, v in delta.items()}}


def test_report(args):
    data, selection, out = Path(args.data), Path(args.selection), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    value, _ = _adopted(selection)
    run_root = selection.resolve().parent.parent
    manifest = e._verified(data)
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    requests = h._by_uid(h._rows(data, f'{ARM}/requests.jsonl', manifest['files'][f'{ARM}/requests.jsonl']))
    runs = {}
    for name, path in ((value['selected'], Path(args.p18)), (REFERENCE, Path(args.ref))):
        records, receipt = e._inference(path, data, manifest['expected_uids'])
        if receipt['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
            raise ValueError(f'{path}: trained on another training manifest')
        if name == REFERENCE:
            ds._bind(receipt, run_root, ARM, REFERENCE)
        else:
            _bind(receipt, run_root, name)
        rows, caps = op._scored_rows(records, rooms, requests, args.workers)
        runs[name] = (receipt, rows, {'inference': {'path': str(path.resolve()), 'run_sha256': receipt['run_sha256'],
                                                    'manifest_sha256': h.sha256(path / 'MANIFEST.json')},
                                      'token_cap_failures': caps})
    if len({r[0]['context_plan_sha256'] for r in runs.values()}) != 1:
        raise ValueError('the two test runs are not on one shared plan')
    result, rows = {}, {}
    for name, (_, scored, info) in runs.items():
        metrics, rows[name] = _finish(scored, runs[REFERENCE][1], rooms, requests, args.workers)
        result[name] = {**info, **metrics}
    report_ = {'protocol': 'amendment-018-test-report', 'status': 'EXPLORATORY', 'note': OFF_POLICY, 'arm': ARM,
               'selected': value['selected'],
               'selection_sha256': h.sha256(selection), 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
               'rooms': len(manifest['expected_uids']), 'runs': result,
               'contrast_selected_minus_rpo_dpo1': r2.contrasts(rows[value['selected']], rows[REFERENCE], FIELDS,
                                                                op.REPEATS, r2.SEED),
               'scope': ("reported only, never used to re-select; a' uses each run's own footprint bias on test; the "
                         "*_with_sft / sft_* metric fields compare with rpo-dpo1 here"),
               'code_sha256': _code()}
    d._publish_dir(out, {'report.json': op._json_bytes(report_),
                         **{f'{ARM}-{c}-geometry.jsonl': d._jsonl(v) for c, v in rows.items()}})
    return {'status': 'A018_TEST_REPORT_COMPLETE', **{c: round(result[c]['geometry_success_rate'], 4) for c in result}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('tc-data')
    for flag in ('plan', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step = sub.add_parser('sft-data')
    for flag in ('data', 'exclude', 'teacher', 'base-model', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--sft-runs', type=Path, nargs=2, required=True, metavar=('SFT_JOINT', 'SFT_PLACEMENT'))
    for flag in ('dev', 'test'):
        step.add_argument('--' + flag, type=Path, nargs='+', required=True)
    step.add_argument('--workers', type=int, default=1)
    for name, extra in (('pairs-dpo1', ()), ('pairs-dpo2', ('t-a',))):
        step = sub.add_parser(name)
        for flag in ('data', 'samples', 'base-model', 'targets', 'out', *extra):
            step.add_argument('--' + flag, type=Path, required=True)
        step.add_argument('--workers', type=int, default=1)
    for name, candidate in (('train-sft', 'sft-d'), ('train-dpo1', 'dpo1-d'), ('train-dpo2', 'dpo2-d')):
        step = sub.add_parser(name)
        step.add_argument('--run-root', type=Path, required=True)
        step.set_defaults(candidate=candidate)
    for name in ('dev-score', 'check-merged'):
        step = sub.add_parser(name)
        for flag in ('data', 'root', 'ref-root', 'out'):
            step.add_argument('--' + flag, type=Path, required=True)
        step.add_argument('--workers', type=int, default=1)
    step = sub.add_parser('select')
    for flag in ('scores', 'out', 'code-root'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--shards', type=int, default=6)
    step.add_argument('--no-submit', action='store_true')
    step = sub.add_parser('test-report')
    for flag in ('data', 'selection', 'p18', 'ref', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    steps = {'tc-data': tc_data, 'sft-data': sft_data, 'pairs-dpo1': pairs_dpo1, 'pairs-dpo2': pairs_dpo2, 'train-sft': train,
             'train-dpo1': train, 'train-dpo2': train, 'dev-score': dev_score, 'select': select,
             'check-merged': check_merged, 'test-report': test_report}
    result = steps[args.command](args)
    if args.command.startswith('train-') and result.get('status') != NOT_TRAINED:
        result = {'status': 'MERGED', 'candidate': args.candidate, 'model_sha256': result['model_sha256']}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
