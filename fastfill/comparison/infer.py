"""Bounded, offline, one-GPU HF BF16/SDPA LoRA inference; fresh output only.

python -m fastfill.comparison.infer --data EVAL_ROOT --base PINNED_QWEN3 \
    --adapter RUN/joint/adapter --training-run RUN --arm joint --out NEW_DIR

EVAL_ROOT/MANIFEST.json requires ordered, unique nonempty expected_uids and
files mapping joint/requests.jsonl, placement/requests.jsonl, rooms.jsonl and
size-estimates.jsonl to SHA256 strings (or {sha256, lines}). Sidecars are only
streamed for hashing, NEVER parsed. Requests contain uid/source/group/messages
and optional split; messages are exactly system/user, without labels. Model
inputs come only from requests. No flag filtering, repair, retry or resume.

Completed records.jsonl is in expected_uids order: uid/source/group/arm/raw,
error (runtime only), schema {valid,error,placements}, generated_tokens,
reached_token_cap, stopped_on_eos, generated, input_tokens, batch_id and timing.
Generated token counts INCLUDE first EOS, EXCLUDE subsequent batch padding.
duration_seconds is shared batch wall time, not independent per-item latency.
Immutable receipts plus fsynced ledger.jsonl survive partial runs; status.json
has partial accounting. Only a fully accounted run gets INFERENCE_COMPLETE,
even with zero valid outputs. This is never a quality claim. Parent owns jobs,
walltime, frozen test selection and statistics. No GT sidecar reaches parsing.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

from fastfill.comparison import train_pair_helpers as h


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_once(path, value):
    """Atomically publish JSON without replacing an existing immutable receipt."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()


def _schema_record(record):
    # h.request requires three nonempty turns. This contains NO target and is
    # used ONLY by validation, never by tokenization or generation.
    return {**record, 'messages': [*record['messages'], {'role': 'assistant', 'content': '{}'}]}


def validate_request(record, arm):
    if (not isinstance(record, dict) or set(record) - {'uid', 'source', 'group', 'messages', 'split', 'component', 'room_type'} or
            any(not isinstance(record.get(k), str) or not record[k].strip() for k in ('uid', 'source', 'group'))):
        raise ValueError('request requires UID/source/group and no labels or extra metadata')
    if record.get('component', record['group']) != record['group']:
        raise ValueError('request component/group mismatch')
    if 'room_type' in record and not isinstance(record['room_type'], str):
        raise ValueError('request room_type metadata must be text')
    messages = record.get('messages')
    if (not isinstance(messages, list) or len(messages) != 2 or
            any(not isinstance(m, dict) or set(m) != {'role', 'content'} or
                not isinstance(m['content'], str) or not m['content'] for m in messages) or
            [m['role'] for m in messages] != ['system', 'user']):
        raise ValueError('request must contain exactly nonempty system/user messages')
    req = h.request(_schema_record(record))
    if arm not in h.ARMS or set(req) - {'room_type', 'boundary_type', 'boundary', 'height', 'fixed', 'objects'}:
        raise ValueError('unexpected request fields or arm')
    fields = {'id', 'category'} | ({'size'} if arm == 'placement' else set())
    for obj in req['objects']:
        if set(obj) != fields or not isinstance(obj['category'], str) or not obj['category']:
            raise ValueError('movable input contains labels or lacks category/size')
        if arm == 'placement':
            h._vector(obj['size'], positive=True)
    for obj in req.get('fixed', []):
        if set(obj) != {'id', 'category', 'size', 'pos', 'yaw'}:
            raise ValueError('unexpected fixed fields')
    return record


def parse_prediction(raw, record, arm):
    return h.parse_prediction(raw, _schema_record(record), arm)


def load_eval(root, arm):
    root = Path(root)
    payload = (root / 'MANIFEST.json').read_bytes()
    manifest = h.strict_json(payload)
    files, uids = manifest.get('files', {}), manifest.get('expected_uids')
    required = {'joint/requests.jsonl', 'placement/requests.jsonl', 'rooms.jsonl', 'size-estimates.jsonl'}
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError('eval manifest missing required files')
    if (not isinstance(uids, list) or not uids or any(not isinstance(u, str) or not u for u in uids)
            or len(set(uids)) != len(uids)):
        raise ValueError('expected_uids must be ordered unique nonempty strings')
    h.verify_artifacts(root, files)  # Opaque streaming digests, including sidecars.
    name = f'{arm}/requests.jsonl'
    rows = h._rows(root, name, files[name])
    if [r['uid'] for r in rows] != uids:
        raise ValueError('request UID order/coverage differs from frozen expected_uids')
    for record in rows:
        validate_request(record, arm)
    return rows, {'manifest_sha256': hashlib.sha256(payload).hexdigest(), 'expected_uids': uids,
                  'files': {n: h._expected_hash(v) for n, v in files.items()}}


def training_identity(args):
    root, arm_dir = args.training_run.resolve(), args.training_run.resolve() / args.arm
    if args.adapter.resolve() != (arm_dir / 'adapter').resolve():
        raise ValueError('adapter must belong to requested training run/arm')
    inventory = h.strict_json((root / 'SHA256.json').read_text())
    needed = {'run_receipt.json', 'status.json', f'{args.arm}/result.json'}
    if not needed <= inventory.keys():
        raise ValueError('training inventory lacks terminal evidence')
    h.verify_artifacts(root, inventory)
    status = h.strict_json((root / 'status.json').read_text())
    result = h.strict_json((arm_dir / 'result.json').read_text())
    if (status.get('status') != 'TRAINING_COMPLETE' or not status.get('training_executed') or
            args.arm not in status.get('completed_arms', []) or result.get('status') != 'TRAINING_COMPLETE' or
            result.get('arm') != args.arm or result.get('finite_loss') is not True or
            result.get('clean_load') is not True or result.get('global_step', 0) <= 0):
        raise ValueError('requested arm lacks successful terminal training result')
    arm_files = h.strict_json((arm_dir / 'SHA256.json').read_text())
    h.verify_artifacts(arm_dir, arm_files)
    adapter_files = {n.removeprefix('adapter/'): h._expected_hash(v)
                     for n, v in arm_files.items() if n.startswith('adapter/')}
    if (not {'adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json'} <= adapter_files.keys()
            or h.artifact_manifest(args.adapter) != adapter_files or
            any(inventory.get(f'{args.arm}/adapter/{n}') != v for n, v in adapter_files.items())):
        raise ValueError('adapter inventory mismatch')
    receipt = h.strict_json((root / 'run_receipt.json').read_text())
    base = h.base_receipt(args.base)
    if any(base[k] != receipt['base'].get(k) for k in ('revision', 'files', 'receipt_hashes')):
        raise ValueError('base differs from training receipt')
    if not receipt['data'].get('files'):
        raise ValueError('training data hashes missing')
    h._expected_hash(receipt['data']['manifest_sha256'])
    for digest in receipt['data']['files'].values():
        h._expected_hash(digest)
    budget = getattr(args, 'context_budget', 0)
    if budget and not 1 < budget <= base['max_position_embeddings']:
        raise ValueError('context budget outside base context')
    if budget and args.max_input_tokens >= budget:
        raise ValueError('context budget must leave room for generation')
    if not budget and args.max_input_tokens + args.max_new_tokens > base['max_position_embeddings']:
        raise ValueError('input plus generation budgets exceed base context')
    return {'training_run_sha256': h.sha256(root / 'SHA256.json'), 'base': base,
            'base_sha256': _hash(base['files']),
            'training_data': receipt['data'], 'adapter_files': adapter_files,
            'adapter_sha256': _hash(adapter_files)}


def decode_output(tok, sequence, prompt_width, eos, cap):
    tokens = list(sequence[prompt_width:])
    stopped = eos in tokens
    used = tokens[:tokens.index(eos) + 1] if stopped else tokens
    if not used or len(used) > cap:
        raise ValueError('generation length outside configured budget')
    raw = tok.decode(used[:-1] if stopped else used, skip_special_tokens=False)
    return raw, {'generated_tokens': len(used), 'reached_token_cap': len(used) >= cap,
                 'stopped_on_eos': stopped}


def prompt_ids(tok, record):
    prompt = tok.apply_chat_template(record['messages'], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    return tok(prompt, add_special_tokens=False, truncation=False)['input_ids']


def context_plan(paired, tok, batch_size, budget, max_input, max_new):
    """Identical per-batch output allowance from both arms' input tokens only."""
    uids = [[r['uid'] for r in paired[arm]] for arm in h.ARMS]
    if uids[0] != uids[1]:
        raise ValueError('paired context plan UID mismatch')
    lengths = {}
    for arm in h.ARMS:
        lengths[arm] = [len(prompt_ids(tok, record)) for record in paired[arm]]
    plan = []
    for start in range(0, len(uids[0]), batch_size):
        subset = {arm: values[start:start + batch_size] for arm, values in lengths.items()}
        eligible = [n for values in subset.values() for n in values if 0 < n <= max_input and n < budget]
        width = max(eligible, default=0)
        plan.append({'uids': uids[0][start:start + batch_size], 'input_tokens': subset,
                     'shared_padded_width': width, 'max_new_tokens': min(max_new, budget - width)})
    return plan


def process_batch(records, arm, tok, generate, eos, max_input, cap, batch_id, context_budget=0,
                  shared_max_new_tokens=None):
    start = time.monotonic()
    results, eligible = [], []
    for record in records:
        base = {k: record[k] for k in ('uid', 'source', 'group')}
        result = {**base, 'arm': arm, 'raw': None, 'error': None, 'generated': False,
                  'schema': {'valid': False, 'error': 'NOT_GENERATED', 'placements': None},
                  'generated_tokens': 0, 'reached_token_cap': False, 'stopped_on_eos': False,
                  'input_tokens': None, 'batch_id': batch_id}
        try:
            ids = prompt_ids(tok, record)
            if not ids:
                raise ValueError('empty prompt')
            result = {**result, 'input_tokens': len(ids)}
            if len(ids) > max_input or (context_budget and len(ids) >= context_budget):
                result = {**result, 'error': 'INPUT_TOO_LONG'}
            else:
                eligible.append((len(results), record, ids))
        except Exception as exc:
            result = {**result, 'error': 'TOKENIZATION_FAILED:' + type(exc).__name__}
        results.append(result)
    if eligible:
        effective_cap = min(cap, context_budget - max(len(e[2]) for e in eligible)) if context_budget else cap
        if shared_max_new_tokens is not None:
            effective_cap = min(effective_cap, shared_max_new_tokens)
        try:
            inputs = [entry[2] for entry in eligible]
            sequences, width = generate(inputs, effective_cap) if shared_max_new_tokens is not None else generate(inputs)
            if len(sequences) != len(eligible):
                raise ValueError('generation batch cardinality mismatch')
            decoded = [decode_output(tok, seq, width, eos, effective_cap) for seq in sequences]
            updates = {i: {**results[i], **info, 'raw': raw, 'generated': True,
                           'effective_max_new_tokens': effective_cap,
                           'schema': parse_prediction(raw, record, arm)}
                       for (i, record, _), (raw, info) in zip(eligible, decoded)}
        except Exception as exc:
            updates = {i: {**results[i], 'effective_max_new_tokens': effective_cap,
                           'error': 'BATCH_GENERATION_FAILED:' + type(exc).__name__}
                       for i, _, _ in eligible}
        results = [updates.get(i, result) for i, result in enumerate(results)]
    duration = time.monotonic() - start
    return [{**result, 'duration_seconds': duration, 'duration_scope': 'shared_batch'} for result in results]


def accounting(expected, records):
    if len(records) > len(expected) or [r['uid'] for r in records] != expected[:len(records)]:
        raise ValueError('ledger is not an exact expected UID prefix')
    return {'expected_count': len(expected), 'recorded_count': len(records),
            'pending_count': len(expected) - len(records),
            'generated_count': sum(r['generated'] for r in records),
            'valid_count': sum(r['schema']['valid'] for r in records),
            'error_count': sum(r['error'] is not None for r in records)}


def ledger_entry(out, path, record):
    return {**{k: record[k] for k in ('uid', 'source', 'group', 'arm', 'generated', 'error')},
            'schema': {'valid': record['schema']['valid']},
            'receipt': str(path.relative_to(out)), 'sha256': h.sha256(path)}


def ledger_append(out, path, record):
    """Bind one immutable receipt with an fsynced ledger line; returns the entry."""
    entry = ledger_entry(out, path, record)
    with (out / 'ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(entry, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return entry


def _runtime(args, merge=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('requires one process and exactly one BF16 GPU')
    torch.cuda.set_device(0)
    merged = getattr(args, 'model', None)  # Amendment 008: an already merged BF16 stage model, no adapter.
    # Amendment 009: a DPO stage's exact trained policy keeps the adapter unmerged and in FP32, as train_dpo's
    # get_peft_model created it (PEFT autocast); BF16-cast LoRA weights would already differ from training.
    exact = getattr(args, 'exact_policy', False)
    if exact and (merge or merged):
        raise ValueError('the exact policy is an unmerged adapter on its training base')
    tok = AutoTokenizer.from_pretrained(str(merged or args.adapter), local_files_only=True, trust_remote_code=False)
    tok.padding_side = 'left'
    eos = tok.get_vocab().get('<|im_end|>')
    if eos is None:
        raise ValueError('Qwen3 tokenizer missing im_end')
    if tok.pad_token_id is None:
        tok.pad_token_id = eos
    base = AutoModelForCausalLM.from_pretrained(str(merged or args.base), local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0})
    model = base if merged else PeftModel.from_pretrained(base, str(args.adapter), local_files_only=True,
                                                         is_trainable=False, autocast_adapter_dtype=exact)
    if merge and not merged:  # Amendment 002: LoRA folded into the BF16 base before generation.
        model = model.merge_and_unload(safe_merge=True)
    model.eval()
    def generate(ids, shared_cap=None):
        inputs = tok.pad({'input_ids': ids, 'attention_mask': [[1] * len(x) for x in ids]},
                         padding=True, return_tensors='pt').to('cuda:0')
        budget = getattr(args, 'context_budget', 0)
        cap = min(args.max_new_tokens, budget - inputs['input_ids'].shape[1]) if budget else args.max_new_tokens
        if shared_cap is not None:
            if type(shared_cap) is not int or not 0 < shared_cap <= cap:
                raise ValueError('shared generation cap exceeds local context')
            cap = shared_cap
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=cap,
                                  eos_token_id=eos, pad_token_id=tok.pad_token_id, use_cache=True)
        with torch.inference_mode():
            outputs = model.generate(**inputs, generation_config=config, logits_to_keep=1)
        return outputs.cpu().tolist(), inputs['input_ids'].shape[1]
    return tok, generate, eos


def _versions():
    versions = {}
    for name in ('torch', 'transformers', 'peft'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not_installed'
    return versions


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    ledger, expected = [], []
    try:
        rows, data = load_eval(args.data, args.arm)
        expected = data['expected_uids']
        config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config = {**config, 'dtype': 'bfloat16', 'attention': 'sdpa', 'padding_side': 'left',
                  'do_sample': False, 'num_beams': 1, 'logits_to_keep': 1,
                  'enable_thinking': False, 'resume_supported': False}
        identity = {'config': config, 'config_sha256': _hash(config), 'data': data,
                    **training_identity(args), 'versions': _versions(),
                    'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(h.__file__))}}
        h.write_json(args.out / 'status.json', {'status': 'LOADING', **accounting(expected, ledger)})
        tok, generate, eos = _runtime(args)
        plan = None
        if getattr(args, 'context_budget', 0):
            other_arm = next(arm for arm in h.ARMS if arm != args.arm)
            other_rows, other_data = load_eval(args.data, other_arm)
            if other_data != data:
                raise ValueError('paired context plan dataset mismatch')
            plan = context_plan({args.arm: rows, other_arm: other_rows}, tok, args.batch_size,
                                args.context_budget, args.max_input_tokens, args.max_new_tokens)
            write_once(args.out / 'context_plan.json', plan)
            identity = {**identity, 'context_plan_sha256': h.sha256(args.out / 'context_plan.json')}
        run_hash = _hash(identity)
        write_once(args.out / 'run_receipt.json', {**identity, 'run_sha256': run_hash})
        (args.out / 'receipts').mkdir()
        h.write_json(args.out / 'status.json', {'status': 'RUNNING', **accounting(expected, ledger)})
        for offset in range(0, len(rows), args.batch_size):
            results = process_batch(rows[offset:offset + args.batch_size], args.arm, tok, generate,
                                    eos, args.max_input_tokens, args.max_new_tokens, offset // args.batch_size,
                                    context_budget=getattr(args, 'context_budget', 0),
                                    shared_max_new_tokens=plan[offset // args.batch_size]['max_new_tokens'] if plan else None)
            for index, record in enumerate(results, offset):
                path = args.out / 'receipts' / f'{index:08d}.json'
                write_once(path, {**record, 'run_sha256': run_hash})
                ledger.append(ledger_append(args.out, path, record))
                h.write_json(args.out / 'status.json', {'status': 'RUNNING', **accounting(expected, ledger)})
        temporary = args.out / 'records.jsonl.tmp'
        with temporary.open('x') as stream:
            for entry in ledger:
                h.verify_artifacts(args.out, {entry['receipt']: entry['sha256']})
                stream.write((args.out / entry['receipt']).read_text())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.out / 'records.jsonl')
        temporary.unlink()
        result = {'status': 'INFERENCE_COMPLETE', **accounting(expected, ledger),
                  'quality_claim': False, 'resume_supported': False, 'run_sha256': run_hash}
        write_once(args.out / 'result.json', result)
        h.write_json(args.out / 'status.json', result)
        write_once(args.out / 'MANIFEST.json', {'files': h.artifact_manifest(args.out), 'expected_uids': expected})
        return result
    except BaseException as exc:
        h.write_json(args.out / 'status.json', {'status': 'FAIL', 'error_type': type(exc).__name__,
                     **accounting(expected, ledger), 'quality_claim': False, 'resume_supported': False})
        raise


def parse_args(argv=None):
    def positive(text):
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError('must be positive')
        return value
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('data', 'base', 'adapter', 'training-run', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--arm', choices=h.ARMS, required=True)
    for flag, default in (('batch-size', 4), ('max-new-tokens', 8192), ('max-input-tokens', 8192)):
        parser.add_argument('--' + flag, type=positive, default=default)
    parser.add_argument('--context-budget', type=positive, default=0,
                        help='If set, generation uses the remaining budget after padded input width; never truncates inputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
