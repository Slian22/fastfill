"""Amendment 013: on-policy DPO2 for arm P on top of rpo-dpo1 (candidate op, run RUN/dpo-004), selected on dev.

python -m fastfill.comparison.onpolicy_dpo STEP ...   (sbatch: isambard_013.sbatch; chain: submit_013.sh)

recheck      --data RUN/dev-select-v1 --ref-root RUN/dev-eval-010 [--workers 30]
    dev-score's first check, run before any GPU work: this code rescoring dev-eval-010's sft and rpo-dpo1 dev runs
    reproduces their a', complete rate and run in dev-eval-010/scores (else raise). Writes nothing.
subset       --data MAIN --sft-runs SFT_J SFT_P --exclude RUN/dpo-001/positives --dev RUN/dev-select-v1 RUN/dev-full-v1
             --test RUN/eval-data-v1 --out RUN/dpo-004/train-subset [--rooms 3000]
    Pool: the SFT common length-admitted train UIDs (dpo_stages._admitted, the 8192-token cap train_dpo uses) minus
    the 2,200 dpo-001 positives (rpo-dpo1 was trained on those prompts). A pool UID or house group found in any dev
    or test expected_uids / rooms.jsonl raises. One cell per (best_of_n.bin_name(objects), source); each cell's
    quota is its share of the pool by largest remainders (ties by cell name), rooms by ascending
    sha256('op-dpo2-v1:' + uid). Proportional quotas never exceed their cell, so the only possible shortfall is a
    pool smaller than --rooms (recorded). Written new-only in the dev-full-v1 layout (split train): per arm
    requests.jsonl, rooms.jsonl, size-estimates.jsonl (the P request sizes) and MANIFEST.json with sorted
    expected_uids, training_manifest_sha256 = MAIN, sft_prompt_tokens (each UID's P prompt token count in the SFT run,
    as dpo_stages.verify_pair checks) and the selection block. infer.load_eval / sample_vllm read it.
(sampling)   sample_vllm run / finalize, unchanged: rpo-dpo1's exact policy (dpo-003/placement/dpo1-merged
             --exact-policy) on the subset -> RUN/dpo-004/samples; k = 0 greedy, k = 1..8 T 0.7 top_p 0.95 's2-bo-v1' seeds.
pairs        --data RUN/dpo-004/train-subset --samples RUN/dpo-004/samples --base-model RUN/dpo-003/placement/dpo1-merged
             --out RUN/dpo-004/stage2-pairs [--workers 72] [--dry-run ROOMS]
    Binding (raises): samples SAMPLING_COMPLETE and hash-verified on this data (as best_of_n.evaluate), sampled by
    rpo-dpo1's exact policy (receipt model.policy == dpo_stages.exact_policy(BASE_MODEL), stage dpo1, candidate rpo),
    tokenizer = train_dpo's (dpo_stages._tokenizer of rpo-dpo1's training base, EOS im_end) and every prompt's token
    count equals the sampled input_tokens (template drift) and the subset's sft_prompt_tokens (verify_pair's SFT
    binding). Per sample:
    geometry.score (+ report_v2.cap_failure on 'length'), best_of_n violations / exploits and downstream.accept on the
    raw layout. pass = geometry_success and accepted; reject = complete and not pass (cap hits and invalid JSON never
    count). Rejected, ordered by tier then k: colocated_duplicate (a main/severe/contained pair of one normalized
    category on one parent), main_collision (main + severe > 0), other; duplicate texts collapse. Chosen per rejected r: its own S1 repair
    (output_repair.repair_row, 2000 trials, 60 s; status ok) rebuilt as perturb does (only moved objects' x, y as
    scene.num cm values, yaw / on / key order kept, contract._json; r must be canonical JSON) and re-scored from
    scratch (must still pass: cm rounding can undo a mm repair), else the best passing sample of the room (lowest
    k, greedy wins; alternatives admissible vs greedy as best_of_n), else no pair (never the GT). Guards (raise):
    validate_output, no size key, request sizes, only pos differs from r. Pairs: drop V(chosen) >= V(rejected)
    (V = checker violations + downstream violations), chosen == rejected, > 8192 tokens (verify_pair's rule); at
    most 4 per room. Floors: >= 1,500 pairs and >= 600 rooms, and dpo_stages.preflight admits every pair; else
    nothing is published (the summary is printed). Writes placement.jsonl, ledger.jsonl, summary.json, MANIFEST.json.
    --dry-run ROOMS (the pre-sampling yield check, isambard_013.sbatch yield): the same rule on the ROOMS lowest-rank
    UIDs of any finalized rpo-dpo1 P sampling run, i.e. amendment 012's dev-full-v1 samples (same policy, same K);
    no scope or SFT binding, no floors, no pairs file. Writes ledger.jsonl and summary.json with the yield projected
    to the 3,000-room subset (meets_floors gates submit_013.sh). Dev rooms, unlike the subset's, were not SFT training
    rooms, so the projection is indicative only.
train        --dpo-root RUN/dpo-004   (4 GPUs; a rerun resumes)
    DPO2 starts from rpo-dpo1's evaluated exact policy, not its BF16 merge (amendment 011 section 5: the merge loses
    the DPO1 gain): dpo_stages.exact_policy(dpo-003/placement/dpo1-merged) gives (and rehashes) rpo-dpo1's training
    base (dpo-001/placement/sft-merged, BF16) and adapter (dpo-003/placement/dpo1/final, FP32). The same torchrun
    train_dpo entry point, recipe {**RECIPE, epochs 2, rpo_alpha 1.0, save_every_epoch}, training_guard, preflight
    and completion run on that training base with --init_adapter = that adapter: the policy LoRA starts as it and
    a frozen copy of it is the DPO reference (LOSS), on dpo-004/stage2-pairs/placement.jsonl. Each saved adapter is
    again one FP32 adapter on the training base. Before the final merge each epoch of EPOCH_MERGES is merged from
    out/epoch-0E (epoch.json must be {epoch: E, step: E x steps per epoch}) into dpo2-epoch0E-merged, so an existing
    dpo2-merged receipt means every merge is done. The merges (training base + adapter, BF16) are receipt anchors
    and convenience artifacts (MERGED_MODEL), not the evaluated policy. Lineage: dpo2[-epoch01]-merged -> parent
    dpo-003/placement/dpo1-merged -> dpo-001/placement/sft-merged, plus training_base = dpo-001/placement/sft-merged
    (path, model and receipt sha256, is_dpo1_merge false), so dpo_stages.exact_policy / inference_identity
    (--exact-policy) load training base + the DPO2 adapter.
dev-score    --data RUN/dev-select-v1 --root RUN/dev-eval-013 --ref-root RUN/dev-eval-010 --out RUN/dev-eval-013/scores
    sft and rpo-dpo1 from dev-eval-010 (dev_select._scored; both must reproduce their a', complete rate and run
    from dev-eval-010/scores), op-dpo2-e1 / op-dpo2-e2 from ROOT (own binding: stage dpo2, exact policy, model path),
    one shared plan (= dev-eval-010's). Per candidate dev_select._finished (metrics, own-bias a'), downstream
    acceptance over all 600 rooms (incomplete = not accepted), both-pass, both dev halves; paired house-cluster
    bootstrap op - rpo-dpo1 (report_v2.contrasts, 2,000 draws, seed 42; reported only). P sizes are the request prior
    by construction (geometry.score takes them from the request; the placement contract has no size field), so no
    size check is run or cited.
select       --scores RUN/dev-eval-013/scores --out RUN/dev-eval-013/selection.json --code-root CODE [--shards 6]
             [--no-submit]
    RULE; selection.json is write-once (a rerun must reproduce it; code hashes aside). Not adopted: P stays rpo-dpo1,
    nothing runs on test. Adopted: infer-v2-002/plan is copied to infer-v2-006/plan and the test chain (test shards ->
    infer-placement-006 -> geometry-v6 with J-SFT and test-report; merged-weight dev shards -> check-merged) is
    submitted with held roots (dev_select.submit). The chain passes --no-submit; the test chain is submitted by hand
    (submit_013.sh CODE test) once the delivery tooling takes the amendment-013 selection.
check-merged --data RUN/dev-select-v1 --root RUN/dev-eval-013 --ref-root RUN/dev-eval-010 --out RUN/dev-eval-013/merged-check
    Amendment 011 section 5 for the adopted checkpoint: its BF16 merge loaded as is vs its exact policy on dev.
test-report  --data RUN/eval-data-v1 --selection RUN/dev-eval-013/selection.json --op RUN/infer-placement-006
             --ref RUN/infer-placement-005 --out RUN/test-report-013
    P-op vs P-rpo-dpo1 on every test room: complete, (a), a', downstream acceptance, paired bootstrap; reported only.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

from fastfill import interface
from fastfill import scene
from fastfill import validate
from fastfill.comparison import best_of_n as bo
from fastfill.comparison import contract
from fastfill.comparison import dev_select as ds
from fastfill.comparison import downstream as dn
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer as i
from fastfill.comparison import output_repair as rep
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

ARMS, ARM, STAGE = h.ARMS, 'placement', 'dpo2'
RUN, PARENT_RUN, CANDIDATE = 'dpo-004', 'dpo-003', 'op'
SALT, ROOMS, SUBSET_SCOPE = 'op-dpo2-v1:', 3000, 'op_dpo2_train_subset'
RECIPE = {**d.RECIPE, 'epochs': 2, 'rpo_alpha': 1.0, 'save_every_epoch': True}  # lr 5e-6, beta 0.1, r32/a16, seed 42.
EPOCH_MERGES = (1,)
LOSS = ("sigmoid DPO vs the frozen rpo-dpo1 exact policy (its BF16 training base + its original FP32 adapter, dropout "
        "off); the policy LoRA starts as that adapter (train_dpo --init_adapter), so DPO2 starts from rpo-dpo1 exactly")
MERGED_MODEL = ("convenience BF16 merge of training_base + adapter, unevaluated as trained: the evaluated policy is the "
                "exact policy (adapter unmerged in FP32 on training_base); only check-merged loads this merge")
MODELS = {'op-dpo2-e1': 'dpo2-epoch01-merged', 'op-dpo2-e2': 'dpo2-merged'}
CANDIDATES, SFT, REFERENCE = tuple(MODELS), 'sft', 'rpo-dpo1'
TIERS = ('colocated_duplicate', 'main_collision', 'other')
PER_ROOM, MIN_PAIRS, MIN_ROOMS = 4, 1500, 600
MAX_TRIALS, TIME_LIMIT = 2000, 60.0  # Amendment 012 S1 settings.
COMPLETE_MARGIN, GAIN, TIE, EPS = 0.005, 0.01, 0.005, 1e-12
HALVES = tuple(ds.TARGETS)  # paper_sources (OptiScene_holodeck + IL3D_3dfront), other_sources.
FIELDS, REPEATS = ('complete', 'geometry_success', 'success_debiased', 'accepted'), 2000
TEST_TAG, GEOMETRY, REPORT, MERGED_SHARDS = '006', 'geometry-v6', 'test-report-013', 2
SHARD = ['--gpus-per-node=1', '--cpus-per-task=8', '--mem=96G', '--time=14:00:00']  # As isambard_010_infer_shard.
PROTOCOL, SCORES_PROTOCOL = 'amendment-013-dev-selection', 'amendment-013-dev-scores'
PAIR_RULE = ("per train room, samples k = 0..8 of rpo-dpo1 scored by geometry.score (+ cap rule) and downstream.accept; "
             "pass = geometry_success and accepted; rejected = complete and not pass, tiers colocated_duplicate, "
             "main_collision, other, then k; chosen = the rejected sample's own S1 repair (status ok, moved x/y as cm "
             "values, re-scored and still passing, admissible vs greedy), else the best passing admissible sample "
             "(lowest k), else no pair (never the GT); drop V(chosen) >= V(rejected), chosen == rejected, > 8192 tokens; "
             "at most 4 pairs per room")
RULE = ("P = rpo-dpo1 unless an on-policy DPO2 checkpoint c in op-dpo2-e1, op-dpo2-e2 (dpo-004 epoch 1, epoch 2) is "
        "eligible on the 600 dev-select rooms against R = rpo-dpo1 (bounds inclusive): complete(c) >= complete(R) - 0.005; "
        "a'(c) >= a'(R) + 0.01; downstream acceptance(c) >= acceptance(R) + 0.01 (all 600 rooms, incomplete = not "
        "accepted); a'(c) - a'(R) >= 0 in both dev halves (300 OptiScene_holodeck + IL3D_3dfront rooms, 300 other "
        "rooms). Adopted: the eligible c with the highest a'; within 0.005 the earlier epoch. None eligible: P stays "
        "rpo-dpo1 and nothing runs on test")


def _json_bytes(value):
    return (json.dumps(value, indent=1, allow_nan=False) + '\n').encode()


def _code():
    return {p.name: h.sha256(p) for p in map(Path, (__file__, d.__file__, bo.__file__, rep.__file__, dn.__file__,
                                                     g.__file__, r2.__file__, ds.__file__, validate.__file__,
                                                     scene.__file__, contract.__file__, interface.__file__))}


# ---- subset --------------------------------------------------------------------------------------

def rank(uid):
    return hashlib.sha256((SALT + uid).encode('utf-8')).hexdigest()


def pick(pool, rooms):
    """pool {uid: cell} -> (sorted uids, info); proportional largest-remainder quotas; pure and order independent."""
    sizes, total = Counter(pool.values()), len(pool)
    n = min(rooms, total)
    quotas = {c: n * s // total for c, s in sizes.items()}
    for c in sorted(sizes, key=lambda c: (-(n * sizes[c] % total), c))[:n - sum(quotas.values())]:
        quotas[c] += 1  # A quota never exceeds its cell (n <= total), so no in-cell shortfall can occur.
    ordered = sorted(pool, key=lambda u: (rank(u), u))
    uids = sorted(u for c, q in quotas.items() for u in [x for x in ordered if pool[x] == c][:q])
    return uids, {'cells': dict(sorted(sizes.items())), 'quotas': dict(sorted(quotas.items())), 'shortfall': rooms - n}


def _held_out(roots):
    """(UIDs, house groups, receipts) of verified eval-layout dirs."""
    uids, groups, receipts = set(), set(), []
    for root in map(Path, roots):
        manifest = e._verified(root)
        uids |= set(manifest['expected_uids'])
        groups |= {r['group'] for r in h._rows(root, 'rooms.jsonl', manifest['files']['rooms.jsonl'])}
        receipts.append({'path': str(root.resolve()), 'manifest_sha256': h.sha256(root / 'MANIFEST.json')})
    return uids, groups, receipts


def subset(args):
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    pairs, data = h.load_pair(args.data)
    if data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('the training subset requires the qualified main data')
    admitted, prompt_tokens, lengths = d._admitted(args.sft_runs, data, 'train')
    rows = {arm: h._by_uid(pairs['train'][arm]) for arm in ARMS}
    rooms = h._by_uid(h._rows(Path(args.data), 'train_rooms.jsonl', data['files']['train_rooms.jsonl']))
    if not set(admitted) <= rows['joint'].keys() & rooms.keys():
        raise ValueError('length-admitted UID outside the verified train rows/rooms')
    positives, _, _, positives_sha = d.load_positives(args.exclude)
    excluded = set(positives['uids'])
    pool = {u: f"{bo.bin_name(len(rooms[u]['objects']))}|{rows['joint'][u]['source']}" for u in admitted
            if u not in excluded}
    if not pool:
        raise ValueError('empty training pool')
    held = {name: _held_out(roots) for name, roots in (('dev', args.dev), ('test', args.test))}
    leaked = sorted(u for u in pool if any(u in uids or rows['joint'][u]['group'] in groups for uids, groups, _ in held.values()))
    if leaked:
        raise ValueError(f'{len(leaked)} training rooms share a UID or house group with dev/test: {leaked[:5]}')
    uids, info = pick(pool, args.rooms)
    files = {f'{arm}/requests.jsonl': [] for arm in ARMS} | {'rooms.jsonl': [], 'size-estimates.jsonl': []}
    for uid in uids:  # sample_vllm.build's layout, split train.
        component, room = rows['joint'][uid]['group'], rooms[uid]
        if room['source'] != rows['joint'][uid]['source'] or any(rows[a][uid]['group'] != component for a in ARMS):
            raise ValueError(f'train row/room identity mismatch: {uid}')
        meta = {'uid': uid, 'source': room['source'], 'group': component, 'component': component, 'split': 'train',
                **({'room_type': room['room_type']} if room.get('room_type') is not None else {})}
        for arm in ARMS:
            files[f'{arm}/requests.jsonl'].append(i.validate_request({**meta, 'messages': rows[arm][uid]['messages'][:2]}, arm))
        files['rooms.jsonl'].append({**room, 'group': component, 'component': component, 'split': 'train'})
        request = h.strict_json(rows['placement'][uid]['messages'][1]['content'])
        files['size-estimates.jsonl'].append({'uid': uid, 'estimated_sizes': {o['id']: o['size'] for o in request['objects']}})
    selection = {'protocol': 'amendment-013-op-subset-v1', 'salt': SALT, 'rooms_requested': args.rooms,
                 'pool': {'length_admitted_train': len(admitted), 'excluded_positives': len(excluded & set(admitted)),
                          'eligible': len(pool)}, **info,
                 'by_source': dict(sorted(Counter(rooms[u]['source'] for u in uids).items())),
                 'excluded_positives_manifest_sha256': positives_sha,
                 'held_out': {name: receipts for name, (_, _, receipts) in held.items()}, 'sft_lengths_sha256': lengths}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        for arm in ARMS:
            (dest / arm).mkdir()
        for name, values in files.items():
            (dest / name).write_bytes(d._jsonl(values))
        h.write_json(dest / 'MANIFEST.json', {
            'version': 1, 'scope': SUBSET_SCOPE, 'split': 'train', 'expected_uids': uids, 'uids_sha256': d._digest(uids),
            'counts': {'train': len(uids)}, 'training_manifest_sha256': data['manifest_sha256'],
            'main_data_manifest_sha256': data['manifest_sha256'],
            'sft_prompt_tokens': {u: prompt_tokens[u][ARM] for u in uids},
            'input_policy': 'system_user_only_labels_and_inventory_separate', 'selection': selection,
            'code_sha256': {'onpolicy_dpo.py': h.sha256(__file__)}, 'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return {'status': 'OP_SUBSET_COMPLETE', 'rooms': len(uids), 'uids_sha256': d._digest(uids),
            'shortfall': info['shortfall']}


# ---- pairs ---------------------------------------------------------------------------------------

def _sample(room, request, prior, sample):
    """best_of_n.score_room per sample (cap rule, violations, guard inputs) plus downstream.accept on the raw layout."""
    row = g.score(room, request['messages'], sample['raw'], ARM, sample['error'])
    if sample['finish_reason'] == 'length':
        row = r2.cap_failure(row)
    done = row['complete']
    reference = [{'size': prior[o['id']]} for o in room['objects']]
    compact = {'k': sample['k'], 'complete': done, 'violations': bo.violations(row),
               'gate_fail': bo.size_gate(row['raw_layout'], prior) if done else None, **bo.exploits(row, room, prior),
               'bias_prior': r2.size_bias(reference, row['raw_layout'])[0] if done else None}
    verdict = dn.accept(rep.predicted_room(room, request['messages'], row['raw_layout']), row['raw_layout']) if done else None
    return row, compact, verdict


def _passes(row, verdict):
    return row['complete'] and row['geometry_success'] and verdict['accepted']


def _violations(row, verdict):
    return bo.violations(row) + len(verdict['violations'])


def tier(row):
    layout = {o['id']: o for o in row['raw_layout']}
    pairs = validate.check({'objects': row['raw_layout']})['collisions']
    if any(scene.norm_cat(layout[a]['category']) == scene.norm_cat(layout[b]['category'])
           and layout[a].get('parent') == layout[b].get('parent') for kind in ('main', 'severe', 'contained')
           for a, b in pairs[kind]):
        return TIERS[0]
    return TIERS[1] if row['main_collisions'] + row['severe_collisions'] else TIERS[2]


def guard(text, raw, room, request):
    """The S1 chosen is the sample itself with positions moved: anything else is a bug, never a pair."""
    contract.validate_output(text, room, ARM)
    value, before = h.strict_json(text)['placements'], h.strict_json(raw)['placements']
    parsed = h.parse_prediction(text, i._schema_record(request), ARM)
    sizes = [o['size'] for o in h.strict_json(request['messages'][1]['content'])['objects']]
    if any('size' in p for p in value) or not parsed['valid'] or [p['size'] for p in parsed['placements']] != sizes:
        raise ValueError(f"S1 chosen carries sizes other than the request prior: {room['uid']}")
    if len(value) != len(before) or any(list(a) != list(b) or {k: v for k, v in a.items() if k != 'pos'} !=
                                        {k: v for k, v in b.items() if k != 'pos'} for a, b in zip(value, before)):
        raise ValueError(f"S1 chosen differs from its sample beyond positions: {room['uid']}")


def s1_chosen(room, request, prior, sample, row, repaired, greedy):
    """(text, scored) of the sample's own S1 repair at cm precision, or (None, reason)."""
    status = repaired['s1']['status']
    if status != 'ok':
        return None, 's1_' + status
    raw, value = sample['raw'], h.strict_json(sample['raw'])
    if contract._json(value) != raw:
        return None, 'non_canonical'
    for p, before, after in zip(value['placements'], row['raw_layout'], repaired['s1_layout'], strict=True):
        if not p['id'] == before['id'] == after['id']:
            raise ValueError('S1 layout order differs from the raw placements')
        if after['pos'][:2] != before['pos'][:2]:  # S1 keeps unmoved objects bit for bit.
            p['pos'][0], p['pos'][1] = scene.num(after['pos'][0]), scene.num(after['pos'][1])
    text = contract._json(value)
    guard(text, raw, room, request)
    scored = _sample(room, request, prior, {**sample, 'raw': text, 'error': None, 'finish_reason': 'stop'})
    if not _passes(scored[0], scored[2]):
        return None, 'rounding_broke'
    if not bo.admissible(scored[1], greedy):
        return None, 's1_inadmissible'
    return text, scored


def _room(task):
    """One room's samples -> ordered candidate pairs (no token rule yet) and the per-sample ledger."""
    room, request, prior, samples = task
    if any(s[k] != room[k] or request[k] != room[k] for s in samples for k in ('uid', 'source', 'group')):
        raise ValueError('sample/request/reference identity mismatch')
    scored = [_sample(room, request, prior, s) for s in samples]
    greedy = scored[0][1]
    good = [k for k, (row, _, verdict) in enumerate(scored) if row['complete'] and _passes(row, verdict)]
    best = next((k for k in good if k == 0 or bo.admissible(scored[k][1], greedy)), None)  # Greedy wins ties.
    ledger = [{'k': s['k'], 'finish_reason': s['finish_reason'],
               'class': 'pass' if k in good else 'reject' if row['complete'] else 'incomplete', 'tier': None, 's1': None}
              for k, (s, (row, *_)) in enumerate(zip(samples, scored))]
    rejected = [k for k, x in enumerate(ledger) if x['class'] == 'reject']
    for k in rejected:
        ledger[k]['tier'] = tier(scored[k][0])
    rejected.sort(key=lambda k: (TIERS.index(ledger[k]['tier']), k))
    candidates, seen = [], set()
    for k in rejected:
        row, _, verdict = scored[k]
        raw = samples[k]['raw']
        if raw in seen:
            candidates.append({'rejected_k': k, 'drop': 'duplicate_rejected_text'})
            continue
        seen.add(raw)
        repaired = rep.repair_row((row, room, request, MAX_TRIALS, TIME_LIMIT))
        s1 = repaired['s1']
        ledger[k]['s1'] = {'status': s1['status'], 'timed_out': s1['timed_out']}
        text, chosen = s1_chosen(room, request, prior, samples[k], row, repaired, greedy)
        entry = {'rejected_k': k, **({'s1_rejected': chosen} if text is None else {})}
        source, chosen_k = 's1_same_sample', k
        if text is None and best is not None:
            text, chosen, source, chosen_k = samples[best]['raw'], scored[best], 'best_passing_sample', best
        if text is None:
            candidates.append({**entry, 'drop': 'no_chosen'})
        elif _violations(chosen[0], chosen[2]) >= _violations(row, verdict):
            candidates.append({**entry, 'drop': 'not_fewer_violations'})
        elif text == raw:
            candidates.append({**entry, 'drop': 'chosen_equals_rejected'})
        else:
            moved = s1['displacement'] or {}
            candidates.append({**entry, 'pair': {
                'uid': room['uid'], 'source': room['source'], 'group': room['group'], 'reject_kind': 'onpolicy',
                'prompt': request['messages'], 'chosen': text, 'rejected': raw, 'chosen_from': source,
                'chosen_k': chosen_k, 'rejected_k': k, 'rejected_tier': ledger[k]['tier'],
                'violations': {'chosen': {v: chosen[0][v] for v in bo.VIOLATIONS}, 'rejected': {v: row[v] for v in bo.VIOLATIONS}},
                'downstream': {'chosen': chosen[2]['reasons'], 'rejected': verdict['reasons']},
                's1': {'status': s1['status'], 'moved': moved.get('moved_objects'), 'moved_median_m': moved.get('moved_median')}
                if source == 's1_same_sample' else None}})
    return {'candidates': candidates, 'samples': ledger}


def _sampling(data, samples, manifest):
    """best_of_n.evaluate's binding of a finalized sample_vllm run to this data -> (receipt, {uid: samples k order})."""
    sampled = e._verified(samples)
    receipt = h.strict_json((samples / 'run_receipt.json').read_text())
    run_hash, n = receipt['run_sha256'], receipt['sampling']['n_samples']
    if i._hash({k: v for k, v in receipt.items() if k != 'run_sha256'}) != run_hash:
        raise ValueError('sampling run identity hash mismatch')
    expected = manifest['expected_uids']
    if (h.strict_json((samples / 'result.json').read_text()).get('status') != 'SAMPLING_COMPLETE' or receipt['arm'] != ARM
            or sampled.get('expected_uids') != expected or 'samples.jsonl' not in sampled['files']
            or receipt['data']['manifest_sha256'] != h.sha256(data / 'MANIFEST.json')):
        raise ValueError('samples are not a complete sampling run of arm P on this data')
    if receipt['model']['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
        raise ValueError('sampled model was trained on another training manifest')
    grouped = {}
    for line in (samples / 'samples.jsonl').read_text().splitlines():
        sample = h.strict_json(line)
        grouped.setdefault(sample['uid'], []).append(sample)
    if list(grouped) != expected or any([s['k'] for s in v] != list(range(n + 1)) or
                                        any(s['run_sha256'] != run_hash or s['arm'] != ARM for s in v)
                                        for v in grouped.values()):
        raise ValueError(f'samples.jsonl must hold k = 0..{n} of this run for every expected room, in order')
    return receipt, grouped


def build_pairs(args, scope=SUBSET_SCOPE, sampler=('dpo1', 'rpo'), short=None):
    """scope / sampler (stage, candidate of the sampled policy's lineage receipt): amendment 018 runs this PAIR_RULE code
    path on its own rooms and policy; short(summary, ledger), when given, is returned instead of raising below the
    floors (nothing trainable is published either way). The defaults are amendment 013's."""
    data, samples, out, dry = Path(args.data), Path(args.samples), Path(args.out), args.dry_run
    if os.path.lexists(out):
        raise FileExistsError(out)
    manifest = e._verified(data)
    if dry is None and manifest.get('scope') != scope:
        raise ValueError('pairs are built on the amendment-013 training subset only')
    expected = manifest['expected_uids']
    def rows(name):
        return h._by_uid(h._rows(data, name, manifest['files'][name]))
    rooms, requests, priors = rows('rooms.jsonl'), rows(f'{ARM}/requests.jsonl'), rows('size-estimates.jsonl')
    receipt, grouped = _sampling(data, samples, manifest)
    model = Path(args.base_model).resolve()
    lineage, policy = d.exact_policy(model, ARM)
    if (lineage['stage'] != sampler[0] or lineage['training'].get('candidate') != sampler[1]
            or receipt['model'].get('policy') != policy
            or receipt['model'].get('lineage_sha256') != h.sha256(d.receipt_path(model))):
        raise ValueError("samples are not of rpo-dpo1's exact policy (the DPO2 reference model)")
    uids = expected if dry is None else sorted(sorted(expected, key=lambda u: (rank(u), u))[:dry])
    tok = d._tokenizer(policy['training_base'])  # train_dpo's: the tokenizer of --model, rpo-dpo1's training base.
    prompts = [scene.prompt_text(tok, requests[u]['messages']) for u in uids]
    prompt_ids = tok(prompts, add_special_tokens=False)['input_ids']
    drift = [u for u, p in zip(uids, prompt_ids) if len(p) != grouped[u][0]['input_tokens']]
    if drift:
        raise ValueError(f'{len(drift)} prompts tokenize differently from the sampled prompts (template drift): {drift[:5]}')
    if dry is None:  # dpo_stages.verify_pair's binding: the DPO prompt is the SFT-trained prompt, token for token.
        drift = [u for u, p in zip(uids, prompt_ids) if len(p) != manifest['sft_prompt_tokens'][u]]
        if drift:
            raise ValueError(f'{len(drift)} prompts tokenize differently from SFT training: {drift[:5]}')
    results = r2._map(_room, [(rooms[u], requests[u], priors[u]['estimated_sizes'], grouped[u]) for u in uids],
                      args.workers)
    kept, ledger, drops = [], [], Counter()
    for uid, prompt, p, result in zip(uids, prompts, prompt_ids, results):
        used, outcomes = [], []
        for entry in result['candidates']:
            pair, outcome = entry.get('pair'), entry.get('drop')
            if pair is not None and len(used) == PER_ROOM:
                outcome = 'room_cap'
            elif pair is not None:  # verify_pair's length rule = train_dpo.DPOPairDataset's admission.
                c, r = tok([prompt + pair['chosen'] + tok.eos_token, prompt + pair['rejected'] + tok.eos_token],
                           add_special_tokens=False)['input_ids']
                if all(len(p) < len(x) <= d.MAX_LEN and x[:len(p)] == p for x in (c, r)):
                    used.append({**pair, 'tokens': {'prompt': len(p), 'chosen': len(c) - len(p), 'rejected': len(r) - len(p)}})
                    outcome = 'pair'
                else:
                    outcome = 'overlength'
            if outcome != 'pair':
                drops[outcome] += 1
            outcomes.append({k: v for k, v in entry.items() if k not in ('pair', 'drop')} | {'outcome': outcome} | (
                {'chosen_from': pair['chosen_from'], 'chosen_k': pair['chosen_k']} if pair else {}))
        kept += used
        ledger.append({'uid': uid, 'source': rooms[uid]['source'], 'bin': bo.bin_name(len(rooms[uid]['objects'])),
                       'samples': result['samples'], 'candidates': outcomes, 'pairs': len(used)})
    tried = [s for x in ledger for s in x['samples'] if s['s1'] is not None]
    summary = {'status': 'OP_PAIRS_COMPLETE', 'rule': PAIR_RULE, 'rooms': len(uids),
               'rooms_with_pair': sum(x['pairs'] > 0 for x in ledger), 'pairs': len(kept),
               'by_source': dict(sorted(Counter(p['source'] for p in kept).items())),
               'by_bin': dict(sorted(Counter(x['bin'] for x in ledger for _ in range(x['pairs'])).items())),
               'by_chosen_from': dict(sorted(Counter(p['chosen_from'] for p in kept).items())),
               'by_tier': dict(sorted(Counter(p['rejected_tier'] for p in kept).items())),
               'drop_reasons': dict(sorted(drops.items())),
               's1_rejected': dict(sorted(Counter(c['s1_rejected'] for x in ledger for c in x['candidates']
                                                   if 's1_rejected' in c).items())),
               'pairs_per_room': dict(sorted(Counter(x['pairs'] for x in ledger).items())),
               'samples': dict(sorted(Counter(s['class'] for x in ledger for s in x['samples']).items())),
               's1': {'repaired_samples': len(tried), 'status': dict(sorted(Counter(s['s1']['status'] for s in tried).items())),
                      'timed_out': sum(s['s1']['timed_out'] for s in tried),
                      'note': 'S1 time limits are wall clock: a timed-out repair is not reproducible'},
               'limits': {'pairs_per_room': PER_ROOM, 'max_len': d.MAX_LEN, 's1_max_trials': MAX_TRIALS,
                          's1_time_limit_s': TIME_LIMIT, 'min_pairs': MIN_PAIRS, 'min_rooms': MIN_ROOMS},
               'inputs': {'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
                          'samples_manifest_sha256': h.sha256(samples / 'MANIFEST.json'),
                          'samples_run_sha256': receipt['run_sha256'], 'reference_model': str(model),
                          'reference_policy': policy, 'reference_lineage_sha256': h.sha256(d.receipt_path(model))},
               'code_sha256': _code()}
    if dry is not None:  # Yield only: projected linearly to the ROOMS-room subset; the floors gate submit_013.sh.
        n, with_pair = len(uids), summary['rooms_with_pair']
        summary.update({'status': 'OP_YIELD_DRY_RUN', 'projection': {
            'rooms': ROOMS, 'pairs': len(kept) * ROOMS / n, 'rooms_with_pair': with_pair * ROOMS / n,
            'meets_floors': len(kept) * ROOMS >= MIN_PAIRS * n and with_pair * ROOMS >= MIN_ROOMS * n},
            'scope': (f'{n} of {len(expected)} rooms (lowest rank) of this sampling run; these rooms were not SFT '
                      'training rooms, unlike the subset, so the projection is indicative only')})
        d._publish_dir(out, {'ledger.jsonl': d._jsonl(ledger), 'summary.json': _json_bytes(summary)})
        return {'status': summary['status'], 'rooms': n, 'pairs': len(kept), 'rooms_with_pair': with_pair,
                'projection': summary['projection']}
    if len(kept) < MIN_PAIRS or summary['rooms_with_pair'] < MIN_ROOMS:
        if short is not None:
            return short(summary, ledger)
        print(json.dumps({k: v for k, v in summary.items() if k != 'rule'}), file=sys.stderr)
        raise ValueError(f"too few on-policy pairs: {len(kept)} pairs in {summary['rooms_with_pair']} rooms "
                         f'(floors {MIN_PAIRS} and {MIN_ROOMS}); nothing published')
    payload = d._jsonl(kept)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, prefix='.pairs-', suffix='.jsonl') as stream:
        stream.write(payload)
        stream.flush()
        d.preflight(policy['training_base'], Path(stream.name), len(kept))  # train_dpo must admit every pair.
    d._publish_dir(out, {f'{ARM}.jsonl': payload, 'ledger.jsonl': d._jsonl(ledger), 'summary.json': _json_bytes(summary)})
    return {'status': summary['status'], 'pairs': len(kept), 'rooms_with_pair': summary['rooms_with_pair'],
            'by_chosen_from': summary['by_chosen_from']}


# ---- train ---------------------------------------------------------------------------------------

def train(args):
    root = Path(args.dpo_root)
    if root.name != RUN:
        raise ValueError(f'candidate {CANDIDATE} trains in {RUN}, not {root.name}')
    parent = (root.parent / PARENT_RUN / ARM / 'dpo1-merged').resolve()  # rpo-dpo1's receipt; its BF16 bytes are unused.
    out, merged = root / ARM / STAGE, root / ARM / f'{STAGE}-merged'
    if d.receipt_path(merged).exists():  # Epoch merges come first: the final receipt means every merge is done.
        return d.verify_merged(merged, ARM, STAGE)
    lineage, policy = d.exact_policy(parent, ARM)  # Rehashes rpo-dpo1's training base and adapter.
    if lineage['stage'] != 'dpo1' or lineage['training'].get('candidate') != 'rpo':
        raise ValueError('the DPO2 start must be the dev-selected rpo-dpo1 (dpo-003)')
    base, init = Path(policy['training_base']), Path(policy['adapter'])
    pairs_dir = root / 'stage2-pairs'
    files = d._verified_dir(pairs_dir)
    summary = h.strict_json((pairs_dir / 'summary.json').read_text())
    if (summary.get('status') != 'OP_PAIRS_COMPLETE'
            or summary['inputs']['reference_lineage_sha256'] != h.sha256(d.receipt_path(parent))):
        raise ValueError('on-policy pairs are incomplete or were not sampled from this DPO2 base')
    pairs = (pairs_dir / f'{ARM}.jsonl').resolve()
    count = len(pairs.read_bytes().splitlines())  # Up to 4 pairs per room: h._rows would refuse repeated UIDs.
    from fastfill import merge_lora
    from fastfill.v1_1 import train_dpo, training_guard
    settings = {**RECIPE, **d.TRAIN_FIXED, 'loss': LOSS,
                'init_adapter': {'path': str(init), 'sha256': policy['adapter_sha256'], 'of': 'rpo-dpo1'}}
    training_guard.guard_training(base, pairs, out, settings, [Path(train_dpo.__file__), Path(d.__file__),
                                                               Path(__file__), Path(merge_lora.__file__)])
    if not (out / 'final' / 'completion.json').exists():
        d.preflight(base, pairs, count)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={d.WORLD_SIZE}',
                   '-m', 'fastfill.v1_1.train_dpo', '--model', str(base), '--pairs', str(pairs), '--out', str(out),
                   '--init_adapter', str(init)]
        subprocess.run(command + d._flags(RECIPE), check=True)
    completion = d._completion(out, count, RECIPE)
    per_epoch = completion['steps'] // RECIPE['epochs']
    fields = {'arm': ARM, 'stage': STAGE,
              'parent': {'stage': 'dpo1', 'path': str(parent), 'lineage_sha256': h.sha256(d.receipt_path(parent))},
              'training_base': {'stage': 'sft', 'path': str(base), 'model_sha256': policy['training_base_sha256'],
                                'lineage_sha256': h.sha256(d.receipt_path(base)), 'is_dpo1_merge': False},
              'merged_model': MERGED_MODEL, **{k: lineage[k] for k in ('base', 'base_sha256', 'training_data')}}
    training = {'pairs': {'path': str(pairs), 'count': count, 'sha256': h._expected_hash(files[pairs.name]),
                          'manifest_sha256': h.sha256(pairs_dir / 'MANIFEST.json')},
                'guard_sha256': h.sha256(out / 'training_guard.json'), 'completion': completion, 'recipe': settings,
                'candidate': CANDIDATE, 'onpolicy_dpo_sha256': h.sha256(__file__)}
    for epoch in EPOCH_MERGES:
        target, adapter = root / ARM / f'{STAGE}-epoch{epoch:02d}-merged', out / f'epoch-{epoch:02d}'
        if not d.receipt_path(target).exists():
            state = h.strict_json((adapter / 'epoch.json').read_text())
            if state != {'epoch': epoch, 'step': epoch * per_epoch}:
                raise ValueError(f'{adapter} is not the end of epoch {epoch}: {state}')
            d._merge(base, adapter, target, {**fields, 'training': {**training, 'epoch': epoch}})
        d.verify_merged(target, ARM, STAGE)
    d._merge(base, out / 'final', merged, {**fields, 'training': training})
    return d.verify_merged(merged, ARM, STAGE)


# ---- dev score and select ------------------------------------------------------------------------

def model_path(run_root, candidate):
    return Path(run_root) / RUN / ARM / MODELS[candidate.removesuffix('-merged')]


def _bind(receipt, run_root, candidate):
    """An op dev/test inference must come from this checkpoint: exact policy, or (-merged) its BF16 merge as is."""
    config = receipt['config']
    loaded = ((config.get('lora') == ds.PREMERGED and 'model_sha256' in receipt and 'policy' not in receipt)
              if candidate.endswith('-merged') else config.get('lora') == ds.EXACT)
    if not (loaded and receipt.get('stage') == STAGE and config.get('arm') == ARM
            and config.get('model') == str(model_path(run_root, candidate).resolve())):
        raise ValueError(f'inference is not {ARM}/{candidate}')


def _scored_rows(records, rooms, requests, workers):
    """Frozen geometry per record, token cap without EOS = failure (amendment 005), own footprint bias."""
    scored = r2._map(e._score, [(rooms[r['uid']], requests[r['uid']], r, ARM) for r in records], workers)
    caps = [r['reached_token_cap'] and not r['stopped_on_eos'] for r in records]
    return [r2._with_bias(r2.cap_failure(s) if c else s, rooms[s['uid']]) for s, c in zip(scored, caps)], sum(caps)


def _scored(root, candidate, data, manifest, rooms, requests, workers, bind=_bind):
    """dev_select._scored with this amendment's binding (requests: placement rows by uid; bind: amendment 018's own)."""
    path = root / candidate / f'infer-{ARM}'
    records, receipt = e._inference(path, data, manifest['expected_uids'])
    if receipt['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
        raise ValueError('dev inference used other training data')
    bind(receipt, root.parent, candidate)
    rows, caps = _scored_rows(records, rooms, requests, workers)
    info = {'inference': {'path': str(path.resolve()), 'run_sha256': receipt['run_sha256'],
                          'manifest_sha256': h.sha256(path / 'MANIFEST.json'), 'stage': receipt['stage'],
                          'lora': receipt['config']['lora']}, 'token_cap_failures': caps}
    return records, receipt, rows, info


def _accepted(task):
    row, room, request = task
    return bool(row['complete'] and dn.accept(rep.predicted_room(room, request['messages'], row['raw_layout']),
                                              row['raw_layout'])['accepted'])


def _halves(rows):
    return {name: [r for r in rows if (r['source'] in ds.PAPER_SOURCES) == (name == HALVES[0])] for name in HALVES}


def _rate(rows, key):
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else None


def _finish(rows, sft_rows, rooms, requests, workers):
    """dev_select._finished (metrics, own-bias a') plus downstream acceptance over every request and per dev half."""
    value, payload = ds._finished(rows, sft_rows, rooms, {ARM: requests}, ARM, workers)
    accepted = r2._map(_accepted, [(r, rooms[r['uid']], requests[r['uid']]) for r in rows], workers)
    rows = [{**h.strict_json(line), 'accepted': a} for line, a in zip(payload.decode().splitlines(), accepted, strict=True)]
    value.update({'accepted_rate': _rate(rows, 'accepted'),
                  'both_pass_rate': sum(r['geometry_success'] and r['accepted'] for r in rows) / len(rows),
                  'halves': {name: {'rooms': len(part), 'a_prime': _rate(part, 'success_debiased'),
                                    'accepted_rate': _rate(part, 'accepted')} for name, part in _halves(rows).items()}})
    return value, rows


def _reference(data, ref, workers):
    """dev-eval-010's sft and rpo-dpo1 dev runs rescored by this code; their a', complete rate and run of record must
    reproduce -> (manifest, rooms, requests, record, {c: _scored tuple}, {c: (finished metrics, rows)})."""
    manifest, rooms, requests = ds._dev(data)
    if ds._digest(manifest['expected_uids']) != manifest['uids_sha256']:
        raise ValueError('dev UIDs differ from the frozen dev selection')
    d._verified_dir(ref / 'scores')
    record = h.strict_json((ref / 'scores' / 'scores.json').read_text())
    if record['data_manifest_sha256'] != h.sha256(data / 'MANIFEST.json'):
        raise ValueError(f'{ref / "scores"} is not scored on this dev data')
    runs = {c: ds._scored(ref, c, ARM, data, manifest, rooms, requests, workers) for c in (SFT, REFERENCE)}
    finished = {c: _finish(runs[c][2], runs[SFT][2], rooms, requests[ARM], workers) for c in runs}
    for c, (value, _) in finished.items():
        entry = record['candidates'][ARM][c]
        if any(value[k] != entry[k] for k in ('a_prime', 'complete_rate')) or (
                runs[c][3]['inference']['run_sha256'] != entry['inference']['run_sha256']):
            raise ValueError(f'{c} rescoring differs from its dev score in {ref / "scores"}')
    return manifest, rooms, requests, record, runs, finished


def recheck(args):
    """dev-score's reference check before any GPU work (scoring code drift since dev-eval-010 raises here)."""
    finished = _reference(Path(args.data), Path(args.ref_root), args.workers)[-1]
    return {'status': 'OP_RECHECK_COMPLETE', **{c: {k: round(value[k], 4) for k in ('a_prime', 'complete_rate')}
                                                for c, (value, _) in finished.items()}}


def dev_score(args):
    data, root, ref, out = Path(args.data), Path(args.root), Path(args.ref_root), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    manifest, rooms, requests, record, runs, finished = _reference(data, ref, args.workers)
    runs.update({c: _scored(root, c, data, manifest, rooms, requests[ARM], args.workers) for c in CANDIDATES})
    if {run[1]['context_plan_sha256'] for run in runs.values()} != {record['plan_sha256']}:
        raise ValueError('candidates were not run on the amendment-010 shared dev plan')
    result, rows = {}, {}
    for c, (_, _, scored, info) in runs.items():
        value, rows[c] = finished[c] if c in finished else _finish(scored, runs[SFT][2], rooms, requests[ARM], args.workers)
        result[c] = {**info, **value}
    for c in CANDIDATES:
        result[c]['model'] = str(model_path(root.parent, c).resolve())
    value = {'protocol': SCORES_PROTOCOL, 'arm': ARM, 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
             'uids_sha256': manifest['uids_sha256'], 'plan_sha256': record['plan_sha256'],
             'reference_scores': {'path': str((ref / 'scores').resolve()),
                                  'manifest_sha256': h.sha256(ref / 'scores' / 'MANIFEST.json')},
             'candidates': result,
             'contrasts_vs_rpo_dpo1': {c: r2.contrasts(rows[c], rows[REFERENCE], FIELDS, REPEATS, r2.SEED)
                                       for c in CANDIDATES},
             'scope': ('contrasts: paired house-cluster bootstrap op - rpo-dpo1 on the 600 dev rooms, reported only; '
                       'on 600 rooms the SE of a success delta is about 0.012, so the +0.01 bars are about 1 SE'),
             'code_sha256': _code()}
    d._publish_dir(out, {'scores.json': _json_bytes(value), **{f'{ARM}-{c}-geometry.jsonl': d._jsonl(rows[c]) for c in rows}})
    return {'status': 'OP_DEV_SCORES_COMPLETE', **{c: {k: round(result[c][k], 4) for k in ('a_prime', 'accepted_rate',
                                                                                          'complete_rate')} for c in result}}


def choose(candidates):
    """RULE on {name: metrics} with rpo-dpo1 and both op checkpoints present."""
    if not {REFERENCE, *CANDIDATES} <= set(candidates):
        raise ValueError('rpo-dpo1 and every op checkpoint must be scored')
    ref = candidates[REFERENCE]

    def eligible(name):
        m = candidates[name]
        return (m['complete_rate'] >= ref['complete_rate'] - COMPLETE_MARGIN - EPS
                and m['a_prime'] >= ref['a_prime'] + GAIN - EPS
                and m['accepted_rate'] >= ref['accepted_rate'] + GAIN - EPS
                and all(m['halves'][k]['a_prime'] >= ref['halves'][k]['a_prime'] - EPS for k in HALVES))
    names = [c for c in CANDIDATES if eligible(c)]
    if not names:
        return {'selected': REFERENCE, 'adopted': False, 'eligible': [], 'tied': [], 'best_a_prime': None}
    best = max(candidates[c]['a_prime'] for c in names)
    tied = [c for c in names if candidates[c]['a_prime'] >= best - TIE - EPS]
    return {'selected': tied[0], 'adopted': True, 'eligible': names, 'tied': tied, 'best_a_prime': best}  # Earlier epoch.


def load_selection(path):
    """selection.json bound to its scores dir and reproduced by choose -> (value, scores)."""
    path = Path(path)
    value, scores_dir = h.strict_json(path.read_text()), path.parent / 'scores'
    d._verified_dir(scores_dir)
    scores = h.strict_json((scores_dir / 'scores.json').read_text())
    again = choose(scores['candidates'])
    if (value.get('protocol') != PROTOCOL or value.get('rule') != RULE or scores.get('protocol') != SCORES_PROTOCOL
            or value.get('scores_manifest_sha256') != h.sha256(scores_dir / 'MANIFEST.json')
            or {k: value.get(k) for k in again} != again or value.get('candidates') != scores['candidates']):
        raise ValueError(f'{path} is not bound to its dev scores')
    return value, scores


def chain_jobs(code_root, run_root, candidate, shards):
    """sbatch jobs (label, afterok labels, argv) of the adopted checkpoint's test run and merged-weight check."""
    script = str(Path(code_root) / 'fastfill/comparison/isambard_013.sbatch')
    merge, code, run_root = str(Path(code_root) / 'fastfill/comparison/isambard_infer_merge.sbatch'), str(code_root), Path(run_root)
    tests, checks = [f'test shard {k}' for k in range(shards)], [f'merged dev shard {k}' for k in range(MERGED_SHARDS)]
    jobs = [{'label': label, 'after': [], 'argv': [*SHARD, script, code, 'shard', 'test', candidate, str(k), str(shards)]}
            for k, label in enumerate(tests)]
    jobs += [{'label': 'test merge', 'after': tests,
              'argv': [merge, code, ARM, str(shards), str(run_root / f'infer-v2-{TEST_TAG}'), str(run_root / f'infer-{ARM}-{TEST_TAG}')]},
             {'label': 'test geometry', 'after': ['test merge'], 'argv': [script, code, 'test-geometry']},
             {'label': 'test report', 'after': ['test merge'], 'argv': ['--time=06:00:00', script, code, 'test-report']}]
    jobs += [{'label': label, 'after': [], 'argv': [*SHARD, script, code, 'shard', 'dev', candidate + '-merged', str(k),
                                                    str(MERGED_SHARDS)]} for k, label in enumerate(checks)]
    jobs += [{'label': 'merged dev merge', 'after': checks,
              'argv': [script, code, 'dev-merge', candidate + '-merged', str(MERGED_SHARDS)]},
             {'label': 'check-merged', 'after': ['merged dev merge'], 'argv': [script, code, 'check-merged']}]
    return jobs


def select(args, run=subprocess.run, which=shutil.which):
    scores_dir, out = Path(args.scores), Path(args.out)
    d._verified_dir(scores_dir)
    scores = h.strict_json((scores_dir / 'scores.json').read_text())
    if scores.get('protocol') != SCORES_PROTOCOL:
        raise ValueError(f'{scores_dir} is not an amendment-013 dev score')
    run_root = scores_dir.parent.parent
    choice = choose(scores['candidates'])
    test = {'candidate': choice['selected'], 'model': str(model_path(run_root, choice['selected']).resolve()),
            'inference': str(run_root / f'infer-{ARM}-{TEST_TAG}'), 'joint': str(run_root / f'infer-joint-{ds.SFT_TEST_TAG}'),
            'geometry': str(run_root / GEOMETRY), 'report': str(run_root / REPORT)} if choice['adopted'] else None
    value = {'protocol': PROTOCOL, 'rule': RULE, 'scores_manifest_sha256': h.sha256(scores_dir / 'MANIFEST.json'),
             'data_manifest_sha256': scores['data_manifest_sha256'], 'uids_sha256': scores['uids_sha256'], 'arm': ARM,
             'reference': REFERENCE, **choice, 'candidates': scores['candidates'], 'test': test,
             'code_sha256': {'onpolicy_dpo.py': h.sha256(__file__)}}
    if out.exists():  # A rerun (submit_013.sh test, maybe from a newer snapshot) must reproduce it, code hashes aside.
        if {**h.strict_json(out.read_text()), 'code_sha256': None} != {**value, 'code_sha256': None}:
            raise ValueError(f'{out} exists with another selection')
    else:
        i.write_once(out, value)
    if not choice['adopted']:
        return {'status': 'NOT_ADOPTED', 'selected': REFERENCE,
                'note': 'P stays rpo-dpo1; op-dpo2 is a dev-only stage-table row; nothing runs on test'}
    receipt = out.with_name('submission.json')
    if receipt.exists():
        raise FileExistsError(f'test chain already submitted: {receipt}')
    if getattr(args, 'no_submit', False):
        return {'status': 'ADOPTED', 'selected': choice['selected']}
    fresh = [run_root / f'infer-v2-{TEST_TAG}' / ARM, run_root / f'infer-{ARM}-{TEST_TAG}', run_root / GEOMETRY,
             run_root / REPORT, out.parent / (choice['selected'] + '-merged'), out.parent / 'merged-check']
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

def _adopted(path):
    value, scores = load_selection(path)
    if not value['adopted']:
        raise ValueError('op-dpo2 was not adopted: P stays rpo-dpo1 and nothing runs on test')
    return value, scores


def check_merged(args):
    data, root, ref, out = Path(args.data), Path(args.root), Path(args.ref_root), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    selection = root / 'selection.json'
    value, scores = _adopted(selection)
    candidate = value['selected']
    merged = candidate + '-merged'
    manifest, rooms, requests = ds._dev(data)
    if ds._digest(manifest['expected_uids']) != manifest['uids_sha256']:
        raise ValueError('dev UIDs differ from the frozen dev selection')
    sft = ds._scored(ref, SFT, ARM, data, manifest, rooms, requests, args.workers)
    runs = {c: _scored(root, c, data, manifest, rooms, requests[ARM], args.workers) for c in (candidate, merged)}
    plans = {run[1]['context_plan_sha256'] for run in (sft, *runs.values())}
    if len(plans) != 1 or None in plans:
        raise ValueError('merged check not on the shared dev plan')
    same = [{k: v for k, v in runs[c][1]['config'].items() if k not in ('lora', 'shard_count')} for c in runs]
    if same[0] != same[1]:
        raise ValueError('merged and unmerged dev runs differ beyond the weight loading')
    result, files = {}, {}
    for c, (_, _, rows, info) in runs.items():
        metrics, rows = _finish(rows, sft[2], rooms, requests[ARM], args.workers)
        result[c], files[f'{ARM}-{c}-geometry.jsonl'] = {**info, **metrics}, d._jsonl(rows)
    entry = scores['candidates'][candidate]
    if any(result[candidate][k] != entry[k] for k in ('a_prime', 'complete_rate', 'accepted_rate')) or (
            result[candidate]['inference']['run_sha256'] != entry['inference']['run_sha256']):
        raise ValueError(f'unmerged {candidate} rescoring differs from its dev score')
    receipt = runs[merged][1]
    result[merged]['model'] = {'path': receipt['config']['model'], 'model_sha256': receipt['model_sha256'],
                               'lineage_sha256': receipt.get('lineage_sha256')}
    identical = sum(a['raw'] is not None and a['raw'] == b['raw'] for a, b in zip(runs[candidate][0], runs[merged][0], strict=True))
    delta = {k: result[merged][k] - result[candidate][k] for k in ('complete_rate', 'geometry_success_rate', 'a_prime',
                                                                    'accepted_rate')}
    listed = abs(delta['a_prime']) <= ds.MERGED_A_PRIME + EPS and abs(delta['complete_rate']) <= ds.MERGED_COMPLETE + EPS
    check = {'protocol': 'amendment-013-merged-weight-check', 'arm': ARM, 'candidate': candidate, 'rule': ds.MERGED_RULE,
             'thresholds': {'a_prime': ds.MERGED_A_PRIME, 'complete_rate': ds.MERGED_COMPLETE},
             'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'), 'uids_sha256': manifest['uids_sha256'],
             'plan_sha256': plans.pop(), 'selection_sha256': h.sha256(selection), 'rooms': len(runs[merged][0]),
             'unmerged': result[candidate], 'merged': result[merged], 'delta_merged_minus_unmerged': delta,
             'byte_identical': {'rooms': identical, 'fraction': identical / len(runs[merged][0])},
             'merged_listed_as_convenience': listed,
             'decision': 'merged_convenience_artifact' if listed else 'adapter_and_training_base_only',
             'code_sha256': _code()}
    d._publish_dir(out, {'check.json': _json_bytes(check), **files})
    return {'status': 'MERGED_CHECK_COMPLETE', 'decision': check['decision'],
            'delta': {k: round(v, 4) for k, v in delta.items()}}


def test_report(args):
    data, selection, out = Path(args.data), Path(args.selection), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    value, _ = _adopted(selection)
    run_root = selection.resolve().parent.parent
    manifest = e._verified(data)
    expected = manifest['expected_uids']
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    requests = h._by_uid(h._rows(data, f'{ARM}/requests.jsonl', manifest['files'][f'{ARM}/requests.jsonl']))
    runs = {}
    for name, path in ((value['selected'], args.op), (REFERENCE, args.ref)):
        records, receipt = e._inference(Path(path), data, expected)
        if receipt['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
            raise ValueError(f'{path}: trained on another training manifest')
        if name == REFERENCE:
            ds._bind(receipt, run_root, ARM, REFERENCE)
        else:
            _bind(receipt, run_root, name)
        rows, caps = _scored_rows(records, rooms, requests, args.workers)
        runs[name] = (receipt, rows, {'inference': {'path': str(Path(path).resolve()), 'run_sha256': receipt['run_sha256'],
                                                    'manifest_sha256': h.sha256(Path(path) / 'MANIFEST.json')},
                                      'token_cap_failures': caps})
    if len({r[0]['context_plan_sha256'] for r in runs.values()}) != 1:
        raise ValueError('P-op and P-rpo-dpo1 test runs are not on one shared plan')
    result, rows = {}, {}
    for name, (_, scored, info) in runs.items():
        metrics, rows[name] = _finish(scored, runs[REFERENCE][1], rooms, requests, args.workers)
        result[name] = {**info, **metrics}
    report = {'protocol': 'amendment-013-test-report', 'arm': ARM, 'selected': value['selected'],
              'selection_sha256': h.sha256(selection), 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
              'rooms': len(expected), 'runs': result,
              'contrast_op_minus_rpo_dpo1': r2.contrasts(rows[value['selected']], rows[REFERENCE], FIELDS, REPEATS, r2.SEED),
              'scope': ("reported only, never used to re-select; a' uses each run's own footprint bias on test; the "
                        "*_with_sft / sft_* metric fields compare with rpo-dpo1 here"),
              'code_sha256': _code()}
    d._publish_dir(out, {'report.json': _json_bytes(report),
                         **{f'{ARM}-{c}-geometry.jsonl': d._jsonl(v) for c, v in rows.items()}})
    return {'status': 'OP_TEST_REPORT_COMPLETE', **{c: round(result[c]['a_prime'], 4) for c in result}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('subset')
    for flag in ('data', 'exclude', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--sft-runs', type=Path, nargs=2, required=True, metavar=('SFT_JOINT', 'SFT_PLACEMENT'))
    for flag in ('dev', 'test'):
        step.add_argument('--' + flag, type=Path, nargs='+', required=True)
    step.add_argument('--rooms', type=int, default=ROOMS)
    step = sub.add_parser('pairs')
    for flag in ('data', 'samples', 'base-model', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    step.add_argument('--dry-run', type=int, metavar='ROOMS')
    step = sub.add_parser('recheck')
    for flag in ('data', 'ref-root'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    step = sub.add_parser('train')
    step.add_argument('--dpo-root', type=Path, required=True)
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
    for flag in ('data', 'selection', 'op', 'ref', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    steps = {'recheck': recheck, 'subset': subset, 'pairs': build_pairs, 'train': train, 'dev-score': dev_score,
             'select': select, 'check-merged': check_merged, 'test-report': test_report}
    result = steps[args.command](args)
    if args.command == 'train':
        result = {'status': 'MERGED', 'stage': result['stage'], 'model_sha256': result['model_sha256']}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
