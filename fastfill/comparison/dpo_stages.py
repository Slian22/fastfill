"""Amendments 008/009: OptiScene recipe SFT -> DPO-stage1 -> DPO-stage2 for both arms; final model = DPO2.

python -m fastfill.comparison.dpo_stages STEP ...   (sbatch: isambard_dpo_*.sbatch; chain: submit_dpo_stages.sh)

positives    --data MAIN --sft-runs SFT_J SFT_P --out DPO/positives
    2,200 shared train UIDs: 1,100 OptiScene_holodeck + 1,100 IL3D_3dfront from the SFT common
    length-admitted pool (both completed runs' lengths.json) whose GT (= J target) and P target pass
    the frozen geometry gate (geometry.score geometry_success), ranked by sha256('dpo-positives-v1:'
    + uid). A source shortfall is filled with the next gate-passing rooms of any source in the same
    order and recorded. Programmatic geometry-verified selection, NOT unanimous human approval.
merge-sft    --arm A --base QWEN3 --sft-run SFT_A --dpo-root DPO
    fastfill.merge_lora of A's verified SFT adapter into BF16 Qwen3-8B -> DPO/A/sft-merged.
generate     --arm A --dpo-root DPO   (vLLM env, 1 GPU; a rerun resumes from per-UID receipts)
    v1.1 stage-1 sampling (T 0.85, top-p 0.95, n 4, seed 42 + positive index) of A's merged SFT
    model on A's positive prompts: SFT chat template with thinking off, token ids passed to vLLM,
    prompt token counts must equal the SFT run's; max_tokens = 8192 - prompt tokens.
stage1-pairs --dpo-root DPO --base QWEN3
    chosen = A's SFT training answer; rejected = A's first sample that stopped on EOS, is
    schema-valid and differs from it; pairs verified (verify_pair); kept only for UIDs with a
    verified pair in BOTH arms.
stage2-pairs --dpo-root DPO --base QWEN3
    rejected = chosen with positions perturbed (v1.1 collision_shift / wall_oob; supported objects
    move with their support; size, yaw and support untouched); one schedule per UID seeded by
    'dpo-stage2-v1:' + uid, first step verified (targeted gate failure) in BOTH arms, used for both.
train        --arm A --stage dpo1|dpo2 --dpo-root DPO   (4 GPUs, torchrun DDP; a rerun resumes)
    fastfill.v1_1.train_dpo (sigmoid DPO, the TRL default loss, without trl) on the merged previous
    stage with the official OptiScene README hyper-parameters (amendment 009: lr 5e-6, 10 epochs,
    LoRA r32 a16, effective batch 4 = 1 per device x 4 GPUs x accumulation 1), bound by
    v1_1.training_guard, then merged -> DPO/A/STAGE-merged (next stage's base, delivered artifact).
train        --arm A --stage dpo1|dpo2 --dpo-root RUN/dpo-002|dpo-003 --candidate e1|rpo   (amendment 010)
    regularised candidates: e1 = the amendment-009 recipe for 1 epoch (dpo-002); rpo = 3 epochs plus
    rpo_alpha 1.0 x the chosen mean token NLL (dpo-003); both save the adapter every epoch. They reuse the
    hash-verified dpo-001 stage1/stage2 pairs and dpo-001/A/sft-merged by reference (lineage receipts bind
    their paths and hashes); dpo1 trains on dpo-001/A/sft-merged, dpo2 on the candidate's own dpo1-merged.
copy-plan    --src RUN/infer-v2-002/plan --out RUN/infer-v2-003/plan RUN/infer-v2-004/plan
    byte-identical shared amendment-002 plan for the stage inference (infer_shard shard --model).

Every merged model has a sibling lineage receipt MODEL.json binding its bytes, adapter, training
evidence and parent receipt, plus the per-module fraction of weights the merge changed (amendment
009); base_sha256 stays the generic Qwen3-8B root through all stages. A DPO stage is evaluated as
its exact trained policy (infer_shard shard --model STAGE-merged --exact-policy): the stage adapter
unmerged, in FP32 as trained, on its BF16 training base (the merged previous stage).
Outputs are new-only (published directories or write-once receipts); nothing is overwritten.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types

from fastfill.comparison import infer as i
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

ARMS = h.ARMS
STAGES = ('sft', 'dpo1', 'dpo2')
PAIRS = {'dpo1': 'stage1-pairs', 'dpo2': 'stage2-pairs'}
SALT, SCHEDULE_SALT = 'dpo-positives-v1:', 'dpo-stage2-v1:'
TARGETS = {'OptiScene_holodeck': 1100, 'IL3D_3dfront': 1100}
MAX_LEN = 8192  # The SFT --max-len: every DPO sequence fits whole; nothing is truncated.
EOS = '<|im_end|>'  # SFT supervises it; train_dpo appends tok.eos_token.
SEED = 42
SAMPLING = {'temperature': 0.85, 'top_p': 0.95, 'n': 4}  # v1.1 stage 1.
RECIPE = {'beta': 0.1, 'lr': 5e-6, 'epochs': 10, 'bs': 1, 'grad_accum': 1, 'max_len': MAX_LEN,
          'lora_r': 32, 'lora_alpha': 16, 'save_steps': 100, 'seed': SEED}  # train_dpo flags (amendment 009).
WORLD_SIZE = 4  # torchrun ranks: effective batch = bs x WORLD_SIZE x grad_accum = 4, the README's 1 x accumulation 4.
TRAIN_FIXED = {'world_size': WORLD_SIZE, 'effective_batch': WORLD_SIZE, 'launcher': 'torchrun --standalone (DDP)',
               'lora_dropout': 0.05, 'dtype': 'bfloat16', 'gradient_checkpointing': True,
               'target_modules': 'q,k,v,o,gate,up,down_proj', 'scheduler': 'cosine, 5% warmup',
               'sampler': 'DistributedSampler: each epoch padded to a multiple of 4 by repeating pairs (as v1.1)',
               'loss': 'sigmoid DPO vs the frozen previous stage (adapter disabled)'}
CANDIDATES = {'e1': ('dpo-002', {**RECIPE, 'epochs': 1, 'save_every_epoch': True}),  # Amendment 010.
              'rpo': ('dpo-003', {**RECIPE, 'epochs': 3, 'rpo_alpha': 1.0, 'save_every_epoch': True})}
SOURCE_RUN = 'dpo-001'  # Candidates' pairs and SFT merges, referenced with lineage, never copied.
MERGE_PROTOCOL = 'amendment-008-merge-v1'
CODE = ('comparison/dpo_stages.py', 'comparison/geometry.py', 'comparison/contract.py',
        'comparison/infer.py', 'comparison/train_pair_helpers.py', 'merge_lora.py',
        'v1_1/train_dpo.py', 'v1_1/training_guard.py', 'validate.py', 'scene.py')


def _code():
    package = Path(__file__).resolve().parents[1]
    return {name: h.sha256(package / name) for name in CODE}


def _digest(value):
    return hashlib.sha256(json.dumps(value).encode('utf-8')).hexdigest()


def _map(function, tasks, workers):
    if workers == 1:
        return [function(task) for task in tasks]
    # Spawned workers receive only their tasks; forked ones would copy the whole pool on GC.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        return list(pool.map(function, tasks, chunksize=64))


def _publish_dir(out, files):
    """Stage {name: bytes} plus MANIFEST.json and publish the directory without replacement."""
    out = Path(out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        for name, payload in files.items():
            (dest / name).write_bytes(payload)
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)


def _jsonl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows).encode('utf-8')


def _verified_dir(root):
    root = Path(root)
    manifest = h.strict_json((root / 'MANIFEST.json').read_text())
    h.verify_artifacts(root, manifest['files'])
    return manifest['files']


# ---- positives ---------------------------------------------------------------------------------

def rank(uid):
    return hashlib.sha256((SALT + uid).encode('utf-8')).hexdigest()


def select(entries):
    """Pure and input-order independent: entries are {uid, source, gate: {arm: bool}}."""
    passing = sorted((e for e in entries if all(e['gate'].values())), key=lambda e: (rank(e['uid']), e['uid']))
    chosen, shortfall = [], {}
    for source, target in TARGETS.items():
        picked = [e['uid'] for e in passing if e['source'] == source][:target]
        shortfall[source] = target - len(picked)
        chosen += picked
    taken = set(chosen)
    fill = [e['uid'] for e in passing if e['uid'] not in taken][:sum(shortfall.values())]
    if len(chosen) + len(fill) != sum(TARGETS.values()):
        raise ValueError('too few gate-passing length-admitted train rooms for the positives')
    source = {e['uid']: e['source'] for e in entries}
    uids = sorted(chosen + fill)
    return {'uids': uids, 'uids_sha256': _digest(uids),
            'by_source': dict(sorted(Counter(source[u] for u in uids).items())),
            'shortfall': shortfall, 'fill_uids': sorted(fill),
            'pool': {'length_admitted': len(entries), 'gate_pass_both_arms': len(passing),
                     'gate_fail': {arm: sum(not e['gate'][arm] for e in entries) for arm in ARMS},
                     'gate_pass_by_source': dict(sorted(Counter(e['source'] for e in passing).items()))}}


def _gate(task):
    from fastfill.comparison import geometry as g
    room, rows = task
    return {arm: g.score(room, rows[arm]['messages'][:2], rows[arm]['messages'][2]['content'], arm)['geometry_success']
            for arm in ARMS}


def _admitted(runs, data, split='train'):
    """The SFT common length admission, read from both completed runs (identical by construction)."""
    reports, digests = [], {}
    for arm, run in zip(ARMS, runs):
        run = Path(run)
        inventory = h.strict_json((run / 'SHA256.json').read_text())
        names = ('lengths.json', 'run_receipt.json', 'status.json')
        if not set(names) <= inventory.keys():
            raise ValueError(f'{arm} SFT inventory lacks lengths/receipt/status')
        h.verify_artifacts(run, {n: inventory[n] for n in names})
        status = h.strict_json((run / 'status.json').read_text())
        receipt = h.strict_json((run / 'run_receipt.json').read_text())
        if status.get('status') != 'TRAINING_COMPLETE' or arm not in status.get('completed_arms', []):
            raise ValueError(f'{arm} SFT run is not TRAINING_COMPLETE')
        if receipt['data']['manifest_sha256'] != data['manifest_sha256'] or receipt['config']['max_len'] != MAX_LEN:
            raise ValueError(f'{arm} SFT run used other data or another max length')
        reports.append(h.strict_json((run / 'lengths.json').read_text())[split])
        digests[arm] = inventory['lengths.json']
    if reports[0] != reports[1]:
        raise ValueError('SFT runs disagree on the common length admission')
    prompts = {e['uid']: {a: e['lengths'][a] - e['answer_tokens'][a] for a in ARMS} for e in reports[0]['lengths']}
    return reports[0]['accepted_uids'], prompts, digests


def positives(args):
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    pairs, data = h.load_pair(args.data)
    if data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('positives require the qualified main target policy')
    admitted, prompt_tokens, lengths = _admitted(args.sft_runs, data)
    rows = {arm: h._by_uid(pairs['train'][arm]) for arm in ARMS}
    rooms = h._by_uid(h._rows(Path(args.data), 'train_rooms.jsonl', data['files']['train_rooms.jsonl']))
    if not set(admitted) <= rows['joint'].keys() & rooms.keys():
        raise ValueError('length-admitted UID outside the verified train rows/rooms')
    gates = _map(_gate, [(rooms[u], {a: rows[a][u] for a in ARMS}) for u in admitted], args.workers)
    selection = select([{'uid': u, 'source': rows['joint'][u]['source'], 'gate': gate}
                        for u, gate in zip(admitted, gates)])
    uids = selection['uids']
    value = {'protocol': 'amendment-008-positives-v1',
             'label': 'programmatic geometry-verified selection, not unanimous human approval',
             'rule': ("SFT common length-admitted train pool; GT (J target) and P target pass geometry.score "
                      "geometry_success; per source ascending sha256('dpo-positives-v1:'+uid); shortfall filled "
                      "with the next gate-passing rooms of any source in the same order"),
             'targets': TARGETS, **selection, 'prompt_tokens': {u: prompt_tokens[u] for u in uids},
             'inputs': {'main_data_manifest_sha256': data['manifest_sha256'], 'sft_lengths_sha256': lengths},
             'code_sha256': _code()}
    _publish_dir(out, {'positives.json': (json.dumps(value, indent=1) + '\n').encode('utf-8'),
                       **{f'{a}.jsonl': _jsonl(rows[a][u] for u in uids) for a in ARMS},
                       'rooms.jsonl': _jsonl(rooms[u] for u in uids)})
    return {'status': 'POSITIVES_COMPLETE', 'positives': len(uids), 'by_source': selection['by_source'],
            'shortfall': selection['shortfall']}


def load_positives(root):
    root = Path(root)
    files = _verified_dir(root)
    value = h.strict_json((root / 'positives.json').read_text())
    records = {a: h._rows(root, f'{a}.jsonl', files[f'{a}.jsonl']) for a in ARMS}
    rooms = h._by_uid(h._rows(root, 'rooms.jsonl', files['rooms.jsonl']))
    if any([r['uid'] for r in records[a]] != value['uids'] for a in ARMS) or list(rooms) != value['uids']:
        raise ValueError('positive records do not match positives.json')
    return value, records, rooms, h.sha256(root / 'MANIFEST.json')


# ---- merged-model lineage ----------------------------------------------------------------------

def receipt_path(model):
    model = Path(model)
    return model.with_name(model.name + '.json')


def _merge(base, adapter, out, fields):
    from fastfill.merge_lora import apply_lora
    if os.path.lexists(out):
        raise FileExistsError(f'{out} exists without a lineage receipt; inspect it before rerunning')
    apply_lora(str(base), str(out), str(adapter))
    files = h.artifact_manifest(out)
    i.write_once(receipt_path(out), {'protocol': MERGE_PROTOCOL, **fields,
        'adapter': {'path': str(Path(adapter).resolve()), 'files': h.artifact_manifest(adapter)},
        'model': {'path': str(Path(out).resolve()), 'files': files}, 'model_sha256': i._hash(files),
        'merge': 'fastfill.merge_lora: BF16 PeftModel.merge_and_unload',
        'merge_diagnostic': _safe_diagnostic(base, out), 'code_sha256': _code()})


def _safe_diagnostic(parent, merged):
    """A diagnostic failure must not strand a published merge without its lineage receipt."""
    try:
        return merge_diagnostic(parent, merged)
    except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
        return {'error': f'{type(exc).__name__}: {exc}'[:500]}


def merge_diagnostic(parent, merged):
    """Amendment 009: per module type, how many stored (BF16) weights the merge changed, exact compare."""
    import contextlib
    from safetensors import safe_open
    with contextlib.ExitStack() as stack:
        def index(model):
            names = {}
            for path in sorted(Path(model).glob('*.safetensors')):
                handle = stack.enter_context(safe_open(str(path), framework='pt'))
                names.update(dict.fromkeys(handle.keys(), handle))
            return names
        old, new = index(parent), index(merged)
        modules, dtypes = {}, set()
        for name in sorted(old.keys() & new.keys()):
            before, after = old[name].get_tensor(name), new[name].get_tensor(name)
            dtypes |= {str(before.dtype), str(after.dtype)}
            entry = modules.setdefault(next((p for p in name.split('.') if p.endswith('_proj')), 'other'),
                                       {'changed': 0, 'total': 0})
            entry['changed'] += int((before != after).sum())
            entry['total'] += before.numel()
    for entry in modules.values():
        entry['fraction'] = entry['changed'] / entry['total'] if entry['total'] else None
    return {'modules': dict(sorted(modules.items())), 'dtypes': sorted(dtypes),
            'tensors_only_in_parent': len(old.keys() - new.keys()), 'tensors_only_in_merged': len(new.keys() - old.keys())}


def _lineage(path, arm, stage):
    """Receipt chain down to the generic base; only receipts are read, no weights."""
    receipt = h.strict_json(Path(path).read_text())
    if (receipt.get('protocol'), receipt.get('arm'), receipt.get('stage')) != (MERGE_PROTOCOL, arm, stage):
        raise ValueError(f'lineage receipt is not {arm}/{stage}: {path}')
    if receipt['base_sha256'] != i._hash(receipt['base']['files']):
        raise ValueError('lineage root hash mismatch')
    if stage != 'sft':
        parent = receipt['parent']
        upstream_path = receipt_path(parent['path'])
        if h.sha256(upstream_path) != parent['lineage_sha256']:
            raise ValueError('parent lineage receipt changed')
        upstream = _lineage(upstream_path, arm, STAGES[STAGES.index(stage) - 1])
        if any(upstream[k] != receipt[k] for k in ('base', 'base_sha256', 'training_data')):
            raise ValueError('lineage root or training data differs from the parent stage')
    return receipt


def verify_merged(model, arm, stage=None):
    """Rehash the merged model against its receipt and walk the receipt chain."""
    model = Path(model).resolve()
    path = receipt_path(model)
    stage = stage or h.strict_json(path.read_text()).get('stage')
    if stage not in STAGES:
        raise ValueError('unknown stage')
    receipt = _lineage(path, arm, stage)
    if receipt['model']['path'] != str(model) or h.artifact_manifest(model) != receipt['model']['files']:
        raise ValueError(f'merged model bytes differ from its lineage receipt: {model}')
    return receipt


def exact_policy(model, arm):
    """Amendment 009: a DPO stage's trained policy is its adapter, unmerged, on its BF16 training base.

    Walks the stage's receipt chain and rehashes what is loaded (training base and adapter), not the
    merged stage model, which is only the next stage's base and the delivered artifact."""
    path = receipt_path(Path(model).resolve())
    receipt = _lineage(path, arm, h.strict_json(path.read_text()).get('stage'))
    if receipt['stage'] not in PAIRS:
        raise ValueError('exact-policy evaluation is for DPO stages only')
    parent = verify_merged(receipt['parent']['path'], arm, receipt['parent']['stage'])
    adapter = Path(receipt['adapter']['path'])
    if h.artifact_manifest(adapter) != receipt['adapter']['files']:
        raise ValueError(f'stage adapter bytes differ from its lineage receipt: {adapter}')
    return receipt, {'mode': 'unmerged_fp32_adapter_on_bf16_training_base',
                     'training_base': receipt['parent']['path'], 'training_base_sha256': parent['model_sha256'],
                     'adapter': str(adapter), 'adapter_sha256': i._hash(receipt['adapter']['files'])}


def inference_identity(run):
    """infer_shard shard --model: base_sha256 is the generic Qwen3-8B lineage root, not this model.

    With run.exact_policy the identity binds the loaded training base and adapter (policy) instead
    of the merged model's bytes (model_sha256)."""
    if getattr(run, 'exact_policy', False):
        receipt, loaded = exact_policy(run.model, run.arm)
        loaded = {'policy': loaded}
    else:
        receipt = verify_merged(run.model, run.arm)
        loaded = {'model_sha256': receipt['model_sha256']}
    base = h.base_receipt(run.base)
    if any(base[k] != receipt['base'][k] for k in ('revision', 'files', 'receipt_hashes')):
        raise ValueError('generic base differs from the merged model lineage root')
    return {'base': base, 'base_sha256': receipt['base_sha256'], 'training_data': receipt['training_data'],
            'stage': receipt['stage'], **loaded, 'lineage_sha256': h.sha256(receipt_path(Path(run.model).resolve()))}


def merge_sft(args):
    out = Path(args.dpo_root) / args.arm / 'sft-merged'
    if receipt_path(out).exists():
        return verify_merged(out, args.arm, 'sft')
    run = types.SimpleNamespace(base=Path(args.base), adapter=Path(args.sft_run) / args.arm / 'adapter',
                                training_run=Path(args.sft_run), arm=args.arm, context_budget=0,
                                max_input_tokens=MAX_LEN, max_new_tokens=1)
    identity = i.training_identity(run)  # TRAINING_COMPLETE inventory, adapter and generic base hashes.
    out.parent.mkdir(parents=True, exist_ok=True)
    _merge(args.base, run.adapter, out, {'arm': args.arm, 'stage': 'sft',
        'parent': {'stage': 'generic_base', 'path': identity['base']['path']},
        'training': {'run': str(run.training_run.resolve()), 'training_run_sha256': identity['training_run_sha256'],
                     'recipe': 'SFT as trained: LoRA r32 a64, lr 5e-5, 1 epoch'},
        **{k: identity[k] for k in ('base', 'base_sha256', 'training_data')}})
    return verify_merged(out, args.arm, 'sft')


# ---- stage-1 generation --------------------------------------------------------------------------

def _load_vllm(engine):
    from vllm import LLM, SamplingParams
    return LLM(**engine), SamplingParams


def _versions(*names):
    found = {}
    for name in names:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = 'not_installed'
    return found


def _model_tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)


def generate(args, load=_load_vllm):
    root, arm = Path(args.dpo_root), args.arm
    value, records, _, positives_sha = load_positives(root / 'positives')
    model = (root / arm / 'sft-merged').resolve()
    lineage = verify_merged(model, arm, 'sft')
    out = root / arm / 'stage1-generations'
    engine = {'model': str(model), 'tensor_parallel_size': 1, 'max_model_len': MAX_LEN, 'seed': SEED, 'dtype': 'bfloat16'}
    config = {'arm': arm, 'engine': engine, 'sampling': {**SAMPLING, 'seed': 'SEED + positive index',
              'max_tokens': 'MAX_LEN - prompt tokens'}, 'max_len': MAX_LEN, 'seed': SEED,
              'prompt': 'SFT chat template, enable_thinking=False, token ids', 'versions': _versions('vllm', 'torch', 'transformers')}
    identity = {'protocol': 'amendment-008-stage1-generation-v1', 'config': config, 'positives_sha256': positives_sha,
                'model_sha256': lineage['model_sha256'], 'lineage_sha256': h.sha256(receipt_path(model)),
                'code_sha256': _code()}
    run_hash = i._hash(identity)
    (out / 'receipts').mkdir(parents=True, exist_ok=True)
    if (out / 'run_receipt.json').exists():
        if h.strict_json((out / 'run_receipt.json').read_text()) != {**identity, 'run_sha256': run_hash}:
            raise ValueError('resume identity mismatch')
    else:
        i.write_once(out / 'run_receipt.json', {**identity, 'run_sha256': run_hash})
    if (out / 'MANIFEST.json').exists():
        return h.strict_json((out / 'result.json').read_text())
    tok = _model_tokenizer(model)
    prompts = []
    for row in records[arm]:
        ids = i.prompt_ids(tok, {'messages': row['messages'][:2]})
        if len(ids) != value['prompt_tokens'][row['uid']][arm]:
            raise ValueError(f'prompt rendering differs from SFT training: {row["uid"]}')
        prompts.append(ids)
    def path(k):
        return out / 'receipts' / f'{k:05d}.json'
    pending = [k for k in range(len(prompts)) if not path(k).exists()]
    if pending:
        llm, params = load(engine)
        for start in range(0, len(pending), args.chunk):
            chunk = pending[start:start + args.chunk]
            outputs = llm.generate([{'prompt_token_ids': prompts[k]} for k in chunk],
                                   [params(**SAMPLING, seed=SEED + k, max_tokens=MAX_LEN - len(prompts[k])) for k in chunk],
                                   use_tqdm=False)
            if len(outputs) != len(chunk):
                raise ValueError('generation batch cardinality mismatch')
            for k, output in zip(chunk, outputs):
                i.write_once(path(k), {'uid': records[arm][k]['uid'], 'run_sha256': run_hash,
                                       'prompt_tokens': len(prompts[k]),
                                       'samples': [{'text': c.text, 'finish_reason': c.finish_reason,
                                                    'tokens': len(c.token_ids)} for c in output.outputs]})
            h.write_json(out / 'status.json', {'status': 'RUNNING', 'expected': len(prompts),
                                               'receipts': sum(path(k).exists() for k in range(len(prompts)))})
    rows = []
    for k, row in enumerate(records[arm]):
        receipt = h.strict_json(path(k).read_text())
        if (receipt['uid'], receipt['run_sha256'], len(receipt['samples'])) != (row['uid'], run_hash, SAMPLING['n']):
            raise ValueError(f'generation receipt identity mismatch: {path(k).name}')
        rows.append(receipt)
    (out / 'generations.jsonl').write_bytes(_jsonl(rows))
    samples = [s for r in rows for s in r['samples']]
    result = {'status': 'GENERATION_COMPLETE', 'arm': arm, 'prompts': len(rows), 'samples': len(samples),
              'finish_reasons': dict(sorted(Counter(s['finish_reason'] for s in samples).items())), 'run_sha256': run_hash}
    h.write_json(out / 'result.json', result)
    h.write_json(out / 'status.json', result)
    i.write_once(out / 'MANIFEST.json', {'files': h.artifact_manifest(out)})
    return result


def _generations(root, arm, positives_sha, uids):
    out = root / arm / 'stage1-generations'
    files = _verified_dir(out)
    receipt = h.strict_json((out / 'run_receipt.json').read_text())
    model = (root / arm / 'sft-merged').resolve()
    if (i._hash({k: v for k, v in receipt.items() if k != 'run_sha256'}) != receipt['run_sha256'] or
            receipt['positives_sha256'] != positives_sha or receipt['config']['arm'] != arm or
            receipt['lineage_sha256'] != h.sha256(receipt_path(model))):
        raise ValueError(f'{arm} generations are not bound to these positives and this SFT model')
    rows = h._rows(out, 'generations.jsonl', files['generations.jsonl'])
    if [r['uid'] for r in rows] != uids or any(r['run_sha256'] != receipt['run_sha256'] for r in rows):
        raise ValueError(f'{arm} generations do not cover the positives')
    return rows, receipt


# ---- pairs ---------------------------------------------------------------------------------------

def _tokenizer(base):
    tok = _model_tokenizer(base)
    if tok.eos_token != EOS:
        raise ValueError('DPO EOS must be the SFT-supervised <|im_end|>')
    return tok


def verify_pair(pair, row, room, arm, tok, prompt_tokens, stage):
    """v1.1 audit_pair on the J/P contract: identity errors raise, label failures return a reason."""
    from fastfill.comparison import geometry as g
    from fastfill.scene import prompt_text
    if ((pair.get('uid'), pair.get('source')) != (row['uid'], row['source']) or pair.get('prompt') != row['messages'][:2]
            or pair.get('chosen') != row['messages'][2]['content'] or not isinstance(pair.get('rejected'), str)):
        raise ValueError(f'pair identity differs from the training reference: {row["uid"]}')
    i.validate_request({**{k: row[k] for k in ('uid', 'source', 'group')}, 'messages': pair['prompt']}, arm)
    if not g.score(room, pair['prompt'], pair['chosen'], arm)['geometry_success']:
        raise ValueError(f'positive fails the frozen geometry gate: {row["uid"]}')
    chosen, rejected = (h.parse_prediction(pair[k], row, arm) for k in ('chosen', 'rejected'))
    if not rejected['valid']:
        return 'rejected_schema_invalid'
    if rejected['placements'] == chosen['placements']:
        return 'rejected_equals_positive'
    if stage == 'stage2':
        def unmoved(text):
            return [{k: v for k, v in o.items() if k != 'pos'} for o in h.strict_json(text)['placements']]
        if unmoved(pair['chosen']) != unmoved(pair['rejected']):
            return 'changed_beyond_positions'
        score = g.score(room, pair['prompt'], pair['rejected'], arm)
        targeted = (score['oob_objects'] if pair['reject_kind'] == 'wall_oob'
                    else score['main_collisions'] + score['severe_collisions'])
        if score['geometry_success'] or not targeted:
            return 'no_targeted_violation'
    prompt = prompt_text(tok, pair['prompt'])  # Exactly train_dpo.DPOPairDataset's admission.
    p, c, r = tok([prompt, prompt + pair['chosen'] + tok.eos_token, prompt + pair['rejected'] + tok.eos_token],
                  add_special_tokens=False)['input_ids']
    if len(p) != prompt_tokens:
        raise ValueError(f'prompt tokens differ from SFT training: {row["uid"]}')
    if not all(len(p) < len(x) <= MAX_LEN and x[:len(p)] == p for x in (c, r)):
        return 'overlength'
    return None


def _pair(row, rejected, kind, **extra):
    return {'uid': row['uid'], 'source': row['source'], 'reject_kind': kind, 'prompt': row['messages'][:2],
            'chosen': row['messages'][2]['content'], 'rejected': rejected, **extra}


def _stage1(task):
    from fastfill.comparison import geometry as g
    row, room, generation, arm, tok, prompt_tokens = task
    reasons = []
    for k, sample in enumerate(generation['samples']):
        if sample['finish_reason'] != 'stop':
            reasons.append('not_stopped_on_eos')
            continue
        pair = _pair(row, sample['text'], 'sft_generation', sample=k)
        reason = verify_pair(pair, row, room, arm, tok, prompt_tokens, 'stage1')
        if reason is None:
            pair['rejected_geometry_success'] = g.score(room, pair['prompt'], pair['rejected'], arm)['geometry_success']
            return pair, reasons
        reasons.append(reason)
    return None, reasons


def _publish_pairs(out, value, pairs, ledger, inputs):
    kept = [u for u in value['uids'] if all(pairs[a].get(u) for a in ARMS)]
    for entry in ledger:
        entry['kept'] = entry['uid'] in kept
    summary = {'positives': len(value['uids']), 'kept_both_arms': len(kept),
               'verified_per_arm': {a: len(pairs[a]) for a in ARMS},
               'dropped_by_other_arm_only': {a: sum(u in pairs[a] for u in value['uids']) - len(kept) for a in ARMS},
               'kinds': {a: dict(sorted(Counter(pairs[a][u]['reject_kind'] for u in kept).items())) for a in ARMS},
               'kept_by_source': dict(sorted(Counter(pairs['joint'][u]['source'] for u in kept).items())),
               'inputs': inputs, 'max_len': MAX_LEN, 'code_sha256': _code()}
    _publish_dir(out, {**{f'{a}.jsonl': _jsonl(pairs[a][u] for u in kept) for a in ARMS},
                       'ledger.jsonl': _jsonl(ledger), 'summary.json': (json.dumps(summary, indent=1) + '\n').encode()})
    return {'status': 'PAIRS_COMPLETE', **{k: summary[k] for k in ('positives', 'kept_both_arms', 'verified_per_arm')}}


def stage1_pairs(args):
    root = Path(args.dpo_root)
    value, records, rooms, positives_sha = load_positives(root / 'positives')
    tok = _tokenizer(args.base)
    generations, receipts = {}, {}
    for arm in ARMS:
        generations[arm], receipts[arm] = _generations(root, arm, positives_sha, value['uids'])
    shared = [{**receipts[a]['config'], 'arm': None, 'engine': {**receipts[a]['config']['engine'], 'model': None}}
              for a in ARMS]
    if shared[0] != shared[1]:
        raise ValueError('arms were not sampled with identical settings')
    pairs, reasons = {a: {} for a in ARMS}, {}
    for arm in ARMS:
        tasks = [(row, rooms[row['uid']], gen, arm, tok, value['prompt_tokens'][row['uid']][arm])
                 for row, gen in zip(records[arm], generations[arm])]
        for row, (pair, why) in zip(records[arm], map(_stage1, tasks)):
            reasons[arm, row['uid']] = why
            if pair:
                pairs[arm][row['uid']] = pair
    ledger = [{'uid': u, **{a: {'verified': u in pairs[a], 'rejected_samples': reasons[a, u]} for a in ARMS}}
              for u in value['uids']]
    inputs = {'positives_sha256': positives_sha,
              'generations_manifest_sha256': {a: h.sha256(root / a / 'stage1-generations/MANIFEST.json') for a in ARMS}}
    return _publish_pairs(root / 'stage1-pairs', value, pairs, ledger, inputs)


def schedule(uid, floor):
    """v1.1 _make_hard_negative ops restricted to positions; identical for both arms (IDs, not geometry)."""
    rng = random.Random(SCHEDULE_SALT + uid)
    ops = ['collision_shift', 'collision_shift', 'wall_oob'] if len(floor) >= 2 else ['wall_oob'] * bool(floor)
    rng.shuffle(ops)
    steps = []
    for op in ops:
        if op == 'collision_shift':
            move, onto = rng.sample(floor, 2)
            steps.append({'op': op, 'move': move, 'onto': onto,
                          'offset': [rng.uniform(-.08, .08), rng.uniform(-.08, .08)]})
        else:
            steps.append({'op': op, 'move': rng.choice(floor), 'side': 'x_max' if rng.random() < .5 else 'y_min'})
    return steps


def perturb(answer, request, step):
    """Move one floor object (and everything it transitively supports) in XY; returns (JSON, moved IDs)."""
    from fastfill.comparison.contract import _json
    from fastfill.scene import num
    value = h.strict_json(answer)
    if _json(value) != answer:
        raise ValueError('positive is not canonical contract JSON')
    placements = {o['id']: o for o in value['placements']}
    old = placements[step['move']]['pos']
    if step['op'] == 'collision_shift':
        target = placements[step['onto']]['pos']
        new = [target[0] + step['offset'][0], target[1] + step['offset'][1]]
    else:
        xs, ys = zip(*request['boundary'])
        new = [max(xs) + .35, old[1]] if step['side'] == 'x_max' else [old[0], min(ys) - .35]
    dx, dy = new[0] - old[0], new[1] - old[1]
    moving, grown = {step['move']}, True
    while grown:
        extra = {o['id'] for o in value['placements'] if o.get('on') in moving} - moving
        moving, grown = moving | extra, bool(extra)
    for oid in moving:
        pos = placements[oid]['pos']
        pos[0], pos[1] = num(pos[0] + dx), num(pos[1] + dy)
    return _json(value), sorted(moving)


def _floor(row):
    return [o['id'] for o in h.strict_json(row['messages'][2]['content'])['placements'] if 'on' not in o]


def _stage2(task):
    rows, room, tok, prompt_tokens = task
    uid = rows['joint']['uid']
    floor = _floor(rows['joint'])
    if floor != _floor(rows['placement']):
        raise ValueError(f'arms disagree on the support structure: {uid}')
    tried = []
    for step in schedule(uid, floor):
        pairs, reasons = {}, {}
        for arm in ARMS:
            row = rows[arm]
            rejected, moved = perturb(row['messages'][2]['content'], h.strict_json(row['messages'][1]['content']), step)
            pairs[arm] = _pair(row, rejected, step['op'], perturbation={**step, 'moved': moved})
            reasons[arm] = verify_pair(pairs[arm], row, room, arm, tok, prompt_tokens[arm], 'stage2')
        tried.append({'step': step, 'reasons': reasons})
        if not any(reasons.values()):
            return pairs, tried
    return None, tried


def stage2_pairs(args):
    root = Path(args.dpo_root)
    value, records, rooms, positives_sha = load_positives(root / 'positives')
    tok = _tokenizer(args.base)
    rows = {a: h._by_uid(records[a]) for a in ARMS}
    tasks = [({a: rows[a][u] for a in ARMS}, rooms[u], tok, value['prompt_tokens'][u]) for u in value['uids']]
    pairs, ledger = {a: {} for a in ARMS}, []
    for uid, (found, tried) in zip(value['uids'], map(_stage2, tasks)):
        ledger.append({'uid': uid, 'schedule_steps': len(tried), 'tried': tried})
        for arm in ARMS:
            if found:
                pairs[arm][uid] = found[arm]
    return _publish_pairs(root / 'stage2-pairs', value, pairs, ledger,
                          {'positives_sha256': positives_sha, 'schedule_seed': SCHEDULE_SALT + 'uid'})


# ---- DPO training ------------------------------------------------------------------------------

def preflight(model, pairs, count):
    """train_dpo skips over-length/prefix-unstable pairs silently; refuse before any GPU work."""
    from fastfill.v1_1.train_dpo import DPOPairDataset
    tok = _tokenizer(model)
    if len(DPOPairDataset(tok, pairs, max_len=MAX_LEN)) != count:
        raise ValueError('train_dpo would drop pairs; refusing a partial-data DPO run')


def _completion(out, count, recipe=RECIPE):
    from fastfill.v1_1.evaluation_protocol import file_identity
    final = Path(out) / 'final'
    value = h.strict_json((final / 'completion.json').read_text())
    batches = math.ceil(count / (recipe['bs'] * WORLD_SIZE))  # Per rank; DistributedSampler pads the last.
    steps = recipe['epochs'] * math.ceil(batches / recipe['grad_accum'])
    expected = {'kind': 'fastfill_v1.1_dpo_completion', 'admitted_pairs': count, 'loader_batches': batches,
                'world_size': WORLD_SIZE, 'epochs': recipe['epochs'], 'expected_total_steps': steps,
                'position': {'epoch': recipe['epochs'] - 1, 'it': batches - 1, 'step': steps},
                'training_guard': file_identity(Path(out) / 'training_guard.json')}
    if (any(value.get(k) != v for k, v in expected.items()) or
            any(file_identity(final / n) != v for n, v in value['adapter_files'].items())):
        raise ValueError('DPO completion does not prove the full guarded run over every pair')
    return {'sha256': h.sha256(final / 'completion.json'), 'steps': steps, 'admitted_pairs': count}


def _flags(recipe):
    """train_dpo arguments; switches (True) go last, after every --key value pair."""
    values = [x for k, v in recipe.items() if not isinstance(v, bool) for x in ('--' + k, str(v))]
    return values + ['--' + k for k, v in recipe.items() if v is True]


def train(args):
    root, arm, stage = Path(args.dpo_root), args.arm, args.stage
    candidate = getattr(args, 'candidate', None)
    recipe, inputs = RECIPE, root
    if candidate is not None:  # Amendment 010: fixed run directory per candidate, inputs from dpo-001.
        run, recipe = CANDIDATES[candidate]
        if root.name != run:
            raise ValueError(f'candidate {candidate} trains in {run}, not {root.name}')
        inputs = root.parent / SOURCE_RUN
    previous = STAGES[STAGES.index(stage) - 1]
    parent = ((inputs if previous == 'sft' else root) / arm / f'{previous}-merged').resolve()
    out, merged = root / arm / stage, root / arm / f'{stage}-merged'
    if receipt_path(merged).exists():
        return verify_merged(merged, arm, stage)
    lineage = verify_merged(parent, arm, previous)
    pairs_dir = inputs / PAIRS[stage]
    files = _verified_dir(pairs_dir)
    pairs = (pairs_dir / f'{arm}.jsonl').resolve()
    count = len(h._rows(pairs_dir, pairs.name, files[pairs.name]))
    from fastfill import merge_lora
    from fastfill.v1_1 import train_dpo, training_guard
    settings = {**recipe, **TRAIN_FIXED}
    training_guard.guard_training(parent, pairs, out, settings,
                                  [Path(train_dpo.__file__), Path(__file__), Path(merge_lora.__file__)])
    if not (out / 'final' / 'completion.json').exists():
        preflight(parent, pairs, count)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={WORLD_SIZE}',
                   '-m', 'fastfill.v1_1.train_dpo', '--model', str(parent), '--pairs', str(pairs), '--out', str(out)]
        subprocess.run(command + _flags(recipe), check=True)
    completion = _completion(out, count, recipe)
    _merge(parent, out / 'final', merged, {'arm': arm, 'stage': stage,
        'parent': {'stage': previous, 'path': str(parent), 'lineage_sha256': h.sha256(receipt_path(parent))},
        'training': {'pairs': {'path': str(pairs), 'count': count, 'sha256': h._expected_hash(files[pairs.name]),
                               'manifest_sha256': h.sha256(pairs_dir / 'MANIFEST.json')},
                     'guard_sha256': h.sha256(out / 'training_guard.json'), 'completion': completion,
                     'recipe': settings, **({'candidate': candidate} if candidate else {})},
        **{k: lineage[k] for k in ('base', 'base_sha256', 'training_data')}})
    return verify_merged(merged, arm, stage)


# ---- shared inference plan -----------------------------------------------------------------------

def copy_plan(args):
    from fastfill.comparison import infer_shard
    _, digest, payload = infer_shard.load_plan(args.src)
    manifest = (Path(args.src) / 'MANIFEST.json').read_bytes()
    for out in args.out:
        if os.path.lexists(out):
            raise FileExistsError(out)
    for out in args.out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
            (Path(tmp) / 'plan.json').write_bytes(payload)
            (Path(tmp) / 'MANIFEST.json').write_bytes(manifest)
            _publish(Path(tmp), out)
        if infer_shard.load_plan(out)[1] != digest:
            raise ValueError('copied plan differs')
    return {'status': 'PLAN_COPIED', 'plan_sha256': digest, 'out': [str(p) for p in args.out]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('positives')
    step.add_argument('--data', type=Path, required=True)
    step.add_argument('--sft-runs', type=Path, nargs=2, required=True, metavar=('SFT_JOINT', 'SFT_PLACEMENT'))
    step.add_argument('--out', type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    step = sub.add_parser('merge-sft')
    for flag in ('base', 'sft-run', 'dpo-root'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--arm', choices=ARMS, required=True)
    step = sub.add_parser('generate')
    step.add_argument('--dpo-root', type=Path, required=True)
    step.add_argument('--arm', choices=ARMS, required=True)
    step.add_argument('--chunk', type=int, default=256)
    for name in ('stage1-pairs', 'stage2-pairs'):
        step = sub.add_parser(name)
        step.add_argument('--dpo-root', type=Path, required=True)
        step.add_argument('--base', type=Path, required=True)
    step = sub.add_parser('train')
    step.add_argument('--dpo-root', type=Path, required=True)
    step.add_argument('--arm', choices=ARMS, required=True)
    step.add_argument('--stage', choices=('dpo1', 'dpo2'), required=True)
    step.add_argument('--candidate', choices=tuple(CANDIDATES))  # Amendment 010; omitted = amendment 009 recipe.
    step = sub.add_parser('copy-plan')
    step.add_argument('--src', type=Path, required=True)
    step.add_argument('--out', type=Path, nargs='+', required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    steps = {'positives': positives, 'merge-sft': merge_sft, 'generate': generate, 'stage1-pairs': stage1_pairs,
             'stage2-pairs': stage2_pairs, 'train': train, 'copy-plan': copy_plan}
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    result = steps[args.command](args)
    if args.command in ('merge-sft', 'train'):
        result = {'status': 'MERGED', 'stage': result['stage'], 'model_sha256': result['model_sha256']}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
