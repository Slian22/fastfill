"""Inference v2 (amendment 002): one shared input-ordered plan, round-robin shards, verified merge.

plan   --data EVAL --base QWEN3 --out ROOT/plan
       Pinned base tokenizer (CPU; adapters may not exist yet). Order: movable
       objects desc, larger of both arms' prompt tokens desc, UID asc; batches
       are consecutive entries. Per batch cap = min(budget - W, 1024 + 128 *
       max_objects, max_new), W = shared padded width of both arms' eligible
       inputs. Only request uid/messages are read; no target, GT or sidecar.
shard  --data --base --adapter --training-run --arm --plan ROOT/plan
       --shard-index I --shard-count N --out ROOT/ARM/shard-I [--resume]
       Plan batches b % N == I, LoRA merged into the BF16 base, greedy decoding
       through infer.process_batch. Receipts plus fsynced ledger as infer.py;
       or --model MERGED instead of --adapter/--training-run (amendment 008): a
       lineage-verified merged stage model loaded as is; base_sha256 stays the
       generic Qwen3-8B lineage root, so evaluate_pair's cross-arm check holds;
       --resume continues from a verified ledger under the identical run
       identity (shard index and out are deliberately not part of it).
merge  --data --plan ROOT/plan --arm ARM --shards S0 .. SN-1 --out INFER_DIR
       Verifies every shard, then publishes the infer.py layout that the
       unchanged evaluate_pair accepts: records.jsonl in expected_uids order,
       INFERENCE_COMPLETE, run_receipt.json, context_plan.json, MANIFEST.json.
probe  --base --adapter --training-run --arm --out ROOT/ARM/probe
       Amendment 005 section 6 gate: the shard's merged BF16 runtime regenerates
       the arm's 2 recorded training-time dev probes (each at batch 1, then both
       repeated into one left-padded batch of 8) at the training probe's cap.
       Write-once probe.json records exact match, schema validity (training
       parser + contract) and token counts; exits 1 iff a prompt whose recorded
       unmerged output was schema-valid becomes schema-invalid.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import types

from fastfill.comparison import infer as i
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

CAP_BASE, CAP_PER_OBJECT = 1024, 128
ORDER = 'movable_objects_desc,max_arm_prompt_tokens_desc,uid_asc'
TOKENIZER_FILES = ('tokenizer', 'special_tokens', 'chat_template', 'vocab', 'merges')
FIXED = {'dtype': 'bfloat16', 'attention': 'sdpa', 'padding_side': 'left', 'do_sample': False,
         'num_beams': 1, 'logits_to_keep': 1, 'enable_thinking': False,
         'lora': 'merge_and_unload_bf16', 'resume_supported': True}
OFFLINE = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY')


def entries(paired, tok):
    """Movable object count (user JSON) and both arms' prompt tokens per UID."""
    if [r['uid'] for r in paired['joint']] != [r['uid'] for r in paired['placement']]:
        raise ValueError('paired plan UID mismatch')
    result = []
    for pair in zip(*(paired[arm] for arm in h.ARMS)):
        objects = {len(h.request(i._schema_record(r))['objects']) for r in pair}
        if len(objects) != 1:
            raise ValueError('paired movable object count mismatch')
        result.append({'uid': pair[0]['uid'], 'objects': objects.pop(),
                       'input_tokens': {arm: len(i.prompt_ids(tok, r)) for arm, r in zip(h.ARMS, pair)}})
    return result


def batches(items, decode):
    items = sorted(items, key=lambda e: (-e['objects'], -max(e['input_tokens'].values()), e['uid']))
    budget, result = decode['context_budget'], []
    for start in range(0, len(items), decode['batch_size']):
        chunk = items[start:start + decode['batch_size']]
        width = max((n for e in chunk for n in e['input_tokens'].values()
                     if 0 < n <= decode['max_input_tokens'] and n < budget), default=0)
        most = max(e['objects'] for e in chunk)
        result.append({'uids': [e['uid'] for e in chunk], 'objects': [e['objects'] for e in chunk],
                       'input_tokens': {arm: [e['input_tokens'][arm] for e in chunk] for arm in h.ARMS},
                       'shared_padded_width': width, 'max_objects': most,
                       'max_new_tokens': min(decode['max_new_tokens'], budget - width,
                                             decode['cap_base'] + decode['cap_per_object'] * most)})
    return result


def load_plan(root):
    """Hash-verified plan whose order and caps are re-derived from its own inputs."""
    root = Path(root)
    manifest = h.strict_json((root / 'MANIFEST.json').read_text())
    if not isinstance(manifest.get('files'), dict) or set(manifest['files']) != {'plan.json'}:
        raise ValueError('plan manifest must bind exactly plan.json')
    payload = (root / 'plan.json').read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != h._expected_hash(manifest['files']['plan.json']):
        raise ValueError('plan hash mismatch')
    plan = h.strict_json(payload)
    decode = plan['decode']
    if (decode['cap_base'], decode['cap_per_object'], plan['order']) != (CAP_BASE, CAP_PER_OBJECT, ORDER):
        raise ValueError('plan cap/order differs from amendment 002')
    flat = [{'uid': uid, 'objects': b['objects'][k], 'input_tokens': {a: b['input_tokens'][a][k] for a in h.ARMS}}
            for b in plan['batches'] for k, uid in enumerate(b['uids'])]
    if batches(flat, decode) != plan['batches']:
        raise ValueError('plan order or caps do not follow amendment 002')
    uids = [e['uid'] for e in flat]
    if len(set(uids)) != len(uids) or set(uids) != set(plan['data']['expected_uids']):
        raise ValueError('plan must cover expected UIDs exactly once')
    return plan, digest, payload


def _plan_tokenizer(base):
    from transformers import AutoTokenizer
    receipt = h.base_receipt(base)
    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    info = {'source': 'pinned_base', 'path': receipt['path'], 'revision': receipt['revision'],
            'files': {n: v for n, v in receipt['files'].items() if n.startswith(TOKENIZER_FILES)},
            'chat_template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest()}
    return tok, info, i._hash(receipt['files'])


def plan(args):
    if args.max_input_tokens >= args.context_budget:
        raise ValueError('context budget must leave room for generation')
    args.out.mkdir(parents=True, exist_ok=False)
    for key in OFFLINE:
        os.environ[key] = '1'
    loaded = {arm: i.load_eval(args.data, arm) for arm in h.ARMS}
    data = loaded['joint'][1]
    if loaded['placement'][1] != data:
        raise ValueError('paired plan dataset mismatch')
    decode = {'batch_size': args.batch_size, 'context_budget': args.context_budget,
              'max_input_tokens': args.max_input_tokens, 'max_new_tokens': args.max_new_tokens,
              'cap_base': CAP_BASE, 'cap_per_object': CAP_PER_OBJECT}
    tok, tokenizer, base_sha256 = _plan_tokenizer(args.base)
    value = {'protocol': 'amendment-002-inference-v2', 'order': ORDER, 'decode': decode, 'data': data,
             'tokenizer': tokenizer, 'base_sha256': base_sha256,
             'batches': batches(entries({arm: rows for arm, (rows, _) in loaded.items()}, tok), decode)}
    i.write_once(args.out / 'plan.json', value)
    i.write_once(args.out / 'MANIFEST.json', {'files': {'plan.json': h.sha256(args.out / 'plan.json')}})
    return {'status': 'PLAN_COMPLETE', 'plan_sha256': h.sha256(args.out / 'plan.json'),
            'requests': len(data['expected_uids']), 'batches': len(value['batches'])}


def _mine(plan, index, count):
    if not 0 <= index < count:
        raise ValueError('shard index outside shard count')
    return [(b, plan['batches'][b]) for b in range(index, len(plan['batches']), count)]


def _receipt(out, index, uid, run_hash):
    path = out / 'receipts' / f'{index:08d}.json'
    payload = path.read_bytes()
    record = h.strict_json(payload)
    if (payload.count(b'\n') != 1 or not payload.endswith(b'\n') or
            record.get('uid') != uid or record.get('run_sha256') != run_hash):
        raise ValueError(f'receipt identity mismatch: {path.name}')
    return path, payload, record


def _ledger(out, expected, run_hash, adopt=False):
    """Verified exact ledger prefix; a resume adopts the single receipt a kill can orphan."""
    path = out / 'ledger.jsonl'
    text = path.read_text() if path.exists() else ''
    if text and not text.endswith('\n'):
        raise ValueError('torn ledger line')
    ledger, receipts = [h.strict_json(line) for line in text.splitlines()], []
    i.accounting(expected, ledger)
    for index, entry in enumerate(ledger):
        path, payload, record = _receipt(out, index, entry['uid'], run_hash)
        if entry != i.ledger_entry(out, path, record) or entry['sha256'] != hashlib.sha256(payload).hexdigest():
            raise ValueError(f'ledger/receipt mismatch: {path.name}')
        receipts.append((payload, record))
    names = {p.name for p in (out / 'receipts').iterdir() if p.suffix == '.json'}
    orphan = f'{len(ledger):08d}.json'
    if names - {f'{n:08d}.json' for n in range(len(ledger))} - ({orphan} if adopt else set()):
        raise ValueError('unledgered receipts')
    if orphan in names and adopt:
        if len(ledger) == len(expected):
            raise ValueError('unledgered receipt beyond shard')
        path, payload, record = _receipt(out, len(ledger), expected[len(ledger)], run_hash)
        ledger.append(i.ledger_append(out, path, record))
        receipts.append((payload, record))
    return ledger, receipts


def _verified_shard(root, plan, plan_sha, arm):
    manifest = h.strict_json((root / 'MANIFEST.json').read_text())
    if {k: v for k, v in h.artifact_manifest(root).items() if k != 'MANIFEST.json'} != manifest.get('files'):
        raise ValueError(f'shard artifact inventory/hash mismatch: {root}')
    receipt, marker, result = (h.strict_json((root / n).read_text())
                               for n in ('run_receipt.json', 'shard.json', 'result.json'))
    run_hash = receipt.get('run_sha256')
    if i._hash({k: v for k, v in receipt.items() if k != 'run_sha256'}) != run_hash:
        raise ValueError('shard run identity hash mismatch')
    if receipt['context_plan_sha256'] != plan_sha or receipt['config']['arm'] != arm:
        raise ValueError('shard plan or arm mismatch')
    index, count = marker['shard_index'], receipt['config']['shard_count']
    mine = _mine(plan, index, count)
    expected = [u for _, b in mine for u in b['uids']]
    if (marker != {'shard_index': index, 'shard_count': count, 'run_sha256': run_hash,
                   'batches': [b for b, _ in mine], 'expected_uids': expected} or
            manifest.get('expected_uids') != expected or result.get('status') != 'SHARD_COMPLETE' or
            result.get('pending_count') != 0 or result.get('run_sha256') != run_hash):
        raise ValueError(f'shard incomplete or inconsistent with plan: {root}')
    ledger, receipts = _ledger(root, expected, run_hash)
    if (len(ledger) != len(expected) or any(r['arm'] != arm for _, r in receipts) or
            [r['batch_id'] for _, r in receipts] != [b for b, batch in mine for _ in batch['uids']]):
        raise ValueError(f'shard records inconsistent with plan: {root}')
    return receipt, marker, result, receipts


def shard(args):
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    elif not (args.out / 'run_receipt.json').is_file():
        raise FileNotFoundError('resume requires an existing shard run receipt')
    for key in OFFLINE:
        os.environ[key] = '1'
    ledger, expected = [], []
    try:
        plan_value, plan_sha, _ = load_plan(args.plan)
        rows, data = i.load_eval(args.data, args.arm)
        if plan_value['data'] != data:
            raise ValueError('plan dataset mismatch')
        mine = _mine(plan_value, args.shard_index, args.shard_count)
        expected = [u for _, b in mine for u in b['uids']]
        decode = plan_value['decode']
        merged = getattr(args, 'model', None) is not None
        if merged == (args.adapter is not None or args.training_run is not None) or (
                not merged and None in (args.adapter, args.training_run)):
            raise ValueError('pass --model (amendment 008 merged stage model) or --adapter with --training-run')
        sources = ('model',) if merged else ('adapter', 'training_run')
        config = {**{k: str(getattr(args, k).resolve()) for k in ('data', 'base', *sources, 'plan')},
                  'arm': args.arm, 'shard_count': args.shard_count, **decode, **FIXED,
                  **({'lora': 'premerged_bf16_stage_model'} if merged else {})}
        run = types.SimpleNamespace(**{k: getattr(args, k) for k in ('base', *sources, 'arm')}, **decode)
        if merged:
            from fastfill.comparison import dpo_stages
        identity = {'config': config, 'config_sha256': i._hash(config), 'data': data,
                    **(dpo_stages.inference_identity(run) if merged else i.training_identity(run)),
                    'versions': i._versions(), 'context_plan_sha256': plan_sha,
                    'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(i.__file__), Path(h.__file__))}}
        if identity['base_sha256'] != plan_value['base_sha256']:
            raise ValueError('base differs from plan tokenizer base')
        run_hash = i._hash(identity)
        receipt = {**identity, 'run_sha256': run_hash}
        marker = {'shard_index': args.shard_index, 'shard_count': args.shard_count, 'run_sha256': run_hash,
                  'batches': [b for b, _ in mine], 'expected_uids': expected}
        if args.resume:
            if (h.strict_json((args.out / 'run_receipt.json').read_text()) != receipt or
                    h.strict_json((args.out / 'shard.json').read_text()) != marker):
                raise ValueError('resume run identity mismatch')
            if (args.out / 'result.json').exists():
                return _verified_shard(args.out, plan_value, plan_sha, args.arm)[2]
        else:
            i.write_once(args.out / 'run_receipt.json', receipt)
            i.write_once(args.out / 'shard.json', marker)
        (args.out / 'receipts').mkdir(exist_ok=True)
        ledger, _ = _ledger(args.out, expected, run_hash, adopt=True)
        h.write_json(args.out / 'status.json', {'status': 'LOADING', **i.accounting(expected, ledger)})
        tok, generate, eos = i._runtime(run, merge=True)
        by_uid = {r['uid']: r for r in rows}
        for _, batch in mine:
            if [len(i.prompt_ids(tok, by_uid[u])) for u in batch['uids']] != batch['input_tokens'][args.arm]:
                raise ValueError('adapter tokenizer differs from plan tokenizer')
        h.write_json(args.out / 'status.json', {'status': 'RUNNING', **i.accounting(expected, ledger)})
        start = 0
        for b, batch in mine:
            end = start + len(batch['uids'])
            if end > len(ledger):  # A resumed partial batch is regenerated; its recorded rows are kept.
                results = i.process_batch([by_uid[u] for u in batch['uids']], args.arm, tok, generate, eos,
                                          decode['max_input_tokens'], decode['max_new_tokens'], b,
                                          context_budget=decode['context_budget'],
                                          shared_max_new_tokens=batch['max_new_tokens'])
                for index, record in enumerate(results, start):
                    if index < len(ledger):
                        continue
                    path = args.out / 'receipts' / f'{index:08d}.json'
                    i.write_once(path, {**record, 'run_sha256': run_hash})
                    ledger.append(i.ledger_append(args.out, path, record))
                    h.write_json(args.out / 'status.json', {'status': 'RUNNING', **i.accounting(expected, ledger)})
            start = end
        result = {'status': 'SHARD_COMPLETE', **i.accounting(expected, ledger), 'quality_claim': False,
                  'run_sha256': run_hash, 'shard_index': args.shard_index, 'shard_count': args.shard_count}
        i.write_once(args.out / 'result.json', result)
        h.write_json(args.out / 'status.json', result)
        i.write_once(args.out / 'MANIFEST.json', {'files': h.artifact_manifest(args.out), 'expected_uids': expected})
        return result
    except BaseException as exc:
        if not (args.out / 'result.json').exists():  # Never touch a completed, manifested shard.
            h.write_json(args.out / 'status.json', {'status': 'FAIL', 'error_type': type(exc).__name__,
                         **i.accounting(expected, ledger), 'quality_claim': False, 'resume_supported': True})
        raise


def merge(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    _, data = i.load_eval(args.data, args.arm)
    expected = data['expected_uids']
    plan_value, plan_sha, plan_payload = load_plan(args.plan)
    if plan_value['data'] != data:
        raise ValueError('plan dataset mismatch')
    shards = [_verified_shard(Path(root), plan_value, plan_sha, args.arm) for root in args.shards]
    receipt = shards[0][0]
    if any(s[0] != receipt for s in shards) or receipt['data'] != data:
        raise ValueError('shard run identities differ')
    if sorted(s[1]['shard_index'] for s in shards) != list(range(receipt['config']['shard_count'])):
        raise ValueError('missing or duplicate shard')
    found = {}
    for *_, receipts in shards:
        for payload, record in receipts:
            if record['uid'] in found:
                raise ValueError('duplicate UID across shards')
            found[record['uid']] = (payload, record)
    if found.keys() != set(expected):
        raise ValueError('merged shards do not cover expected UIDs')
    records = [found[u][1] for u in expected]
    result = {'status': 'INFERENCE_COMPLETE', **i.accounting(expected, records), 'quality_claim': False,
              'resume_supported': True, 'run_sha256': receipt['run_sha256'], 'context_plan_sha256': plan_sha,
              'shards': [{'shard_index': s[1]['shard_index'], 'path': str(Path(root).resolve()),
                          'manifest_sha256': h.sha256(Path(root) / 'MANIFEST.json')}
                         for root, s in zip(args.shards, shards)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + args.out.name + '-', dir=args.out.parent) as tmp:
        dest = Path(tmp)
        i.write_once(dest / 'run_receipt.json', receipt)
        (dest / 'context_plan.json').write_bytes(plan_payload)
        (dest / 'records.jsonl').write_bytes(b''.join(found[u][0] for u in expected))
        i.write_once(dest / 'result.json', result)
        h.write_json(dest / 'status.json', result)
        i.write_once(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest), 'expected_uids': expected})
        _publish(dest, args.out)
    return result


def probe(args):
    from fastfill.comparison.contract import validate_output
    args.out.mkdir(parents=True, exist_ok=False)
    for key in OFFLINE:
        os.environ[key] = '1'
    root = args.training_run.resolve()
    receipt = h.strict_json((root / 'run_receipt.json').read_text())
    config, files = receipt['config'], receipt['data']['files']
    cap = config['max_new_tokens']  # The recorded probe's own decoding cap.
    run = types.SimpleNamespace(**{k: getattr(args, k) for k in ('base', 'adapter', 'training_run', 'arm')},
                                context_budget=0, max_input_tokens=config['max_len'], max_new_tokens=cap)
    identity = i.training_identity(run)  # Hash-verifies the whole training inventory.
    if not {'dev_selection.json', f'{args.arm}/dev_generations.json'} <= h.strict_json(
            (root / 'SHA256.json').read_text()).keys():
        raise ValueError('training inventory lacks dev probe evidence')
    uids = h.strict_json((root / 'dev_selection.json').read_text())['uids']
    recorded = h._by_uid(h.strict_json((root / args.arm / 'dev_generations.json').read_text()))
    if len(uids) != 2 or list(recorded) != uids:
        raise ValueError('recorded dev probe does not match dev selection')
    rows = h._by_uid(h._rows(Path(config['data']), f'{args.arm}/dev.jsonl', files[f'{args.arm}/dev.jsonl']))
    rooms = h._by_uid(h._rows(Path(config['data']), 'dev_rooms.jsonl', files['dev_rooms.jsonl']))
    prompts = [{**rows[u], 'messages': rows[u]['messages'][:-1]} for u in uids]  # Target turn dropped.
    tok, generate, eos = i._runtime(run, merge=True)
    outputs = []
    for batch_id, batch in enumerate((prompts[:1], prompts[1:], prompts * 4)):
        results = i.process_batch(batch, args.arm, tok, generate, eos, config['max_len'], cap, batch_id,
                                  shared_max_new_tokens=cap)
        for position, r in enumerate(results):
            valid, before = r['schema']['valid'], recorded[r['uid']]
            try:
                if valid:  # The contract check the training probe applied after parsing.
                    validate_output(r['raw'], rooms[r['uid']], args.arm)
            except ValueError:
                valid = False
            outputs.append({'batch_id': batch_id, 'batch_size': len(batch), 'position': position, 'uid': r['uid'],
                            'exact_match': r['raw'] == before['raw'], 'schema_valid': valid,
                            'recorded_schema_valid': before['schema']['valid'],
                            'input_tokens': r['input_tokens'], 'generated_tokens': r['generated_tokens'],
                            'recorded_generated_tokens': before['generated_tokens'],
                            'stopped_on_eos': r['stopped_on_eos'], 'reached_token_cap': r['reached_token_cap'],
                            'error': r['error'], 'raw': r['raw']})
    regressions = sum(o['recorded_schema_valid'] and not o['schema_valid'] for o in outputs)
    result = {'status': 'PROBE_FAIL' if regressions else 'PROBE_PASS', 'protocol': 'amendment-005-dev-probe-gate',
              'arm': args.arm, 'max_new_tokens': cap, 'lora': FIXED['lora'], 'quality_claim': False,
              'exact_match_count': sum(o['exact_match'] for o in outputs), 'output_count': len(outputs),
              'schema_regressions': regressions, 'outputs': outputs, 'identity': identity,
              'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(i.__file__), Path(h.__file__))}}
    i.write_once(args.out / 'probe.json', result)
    return result


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)  # Slurm/timeout TERM: record FAIL status, keep receipts for --resume.


def parse_args(argv=None):
    def count(minimum):
        def parse(text):
            value = int(text)
            if value < minimum:
                raise argparse.ArgumentTypeError(f'must be >= {minimum}')
            return value
        return parse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    planner = sub.add_parser('plan')
    for flag in ('data', 'base', 'out'):
        planner.add_argument('--' + flag, type=Path, required=True)
    for flag, default in (('batch-size', 8), ('context-budget', 40960), ('max-input-tokens', 40959),
                          ('max-new-tokens', 40959)):
        planner.add_argument('--' + flag, type=count(1), default=default)
    worker = sub.add_parser('shard')
    for flag in ('data', 'base', 'plan', 'out'):
        worker.add_argument('--' + flag, type=Path, required=True)
    for flag in ('adapter', 'training-run', 'model'):  # --model: amendment 008 merged stage model.
        worker.add_argument('--' + flag, type=Path)
    worker.add_argument('--arm', choices=h.ARMS, required=True)
    worker.add_argument('--shard-index', type=count(0), required=True)
    worker.add_argument('--shard-count', type=count(1), required=True)
    worker.add_argument('--resume', action='store_true')
    merger = sub.add_parser('merge')
    for flag in ('data', 'plan', 'out'):
        merger.add_argument('--' + flag, type=Path, required=True)
    merger.add_argument('--arm', choices=h.ARMS, required=True)
    merger.add_argument('--shards', type=Path, nargs='+', required=True)
    prober = sub.add_parser('probe')
    for flag in ('base', 'adapter', 'training-run', 'out'):
        prober.add_argument('--' + flag, type=Path, required=True)
    prober.add_argument('--arm', choices=h.ARMS, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'shard':
        signal.signal(signal.SIGTERM, _terminate)
    result = {'plan': plan, 'shard': shard, 'merge': merge, 'probe': probe}[args.command](args)
    print(json.dumps({k: v for k, v in result.items() if k not in ('shards', 'outputs', 'identity')}))
    if result['status'] == 'PROBE_FAIL':
        raise SystemExit(1)  # Amendment 005 section 6: fail the inference chain closed.


if __name__ == '__main__':
    main()
