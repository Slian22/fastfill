"""S2 (amendment-011 draft): best-of-N samples of the dev-selected J/P models on offline vLLM; resumable.

python -m fastfill.comparison.sample_vllm build --data RUN/main-data-v1 --out RUN/dev-full-v1
python -m fastfill.comparison.sample_vllm run --data EVAL --base QWEN3 --arm joint|placement --out OUT
    (--adapter RUN/sft-joint-001/joint/adapter --training-run RUN/sft-joint-001
     | --model RUN/dpo-003/placement/dpo1-merged --exact-policy)
    [--shard-index I --shard-count K] [--limit M] [--chunk-tokens 2000000]
python -m fastfill.comparison.sample_vllm finalize --data EVAL --out OUT
(sbatch: isambard_sample_vllm.sbatch; chain: submit_sample_vllm.sh; selection: best_of_n.py)

build     Every dev room of main-data-v1 (all 8,167: no length admission, no pick) in the eval-data-v1 layout,
          written exactly as dev_select.build writes dev-select-v1: per arm requests.jsonl (system + user turns, no
          target), rooms.jsonl (GT, scoring only), size-estimates.jsonl (the P request sizes = train prior) and
          MANIFEST.json with sorted expected_uids. S1 reads the same directory.
run       Per room k = 0 is greedy (temperature 0) and k = 1..N are sampled (T, top_p, seed = int(sha256(SEED_SALT +
          uid + ':' + k)[:8], 16)): one n=1 request per (uid, k), so the sample set does not depend on chunking,
          sharding or resume (numerics may still vary with batch composition); a room's N+1 requests share one
          LLM.generate call, where prefix caching shares the prompt. Prompt = infer.prompt_ids with the HF tokenizer
          of the loaded adapter dir (as infer._runtime), sent as token ids; max_tokens = min(40960 - prompt,
          1024 + 128 x movable objects) (llm_open.output_cap); no room left is INPUT_TOO_LONG (a receipt with N+1
          ungenerated samples). raw = infer.decode_output of the generated ids (HF decode up to the first <|im_end|>);
          finish_reason 'length' is a failure downstream (amendment 005). ignore_eos with stop_token_ids [<|im_end|>]:
          generation stops on <|im_end|> only, as the HF runs (eos_token_id = im_end); without it vLLM also stops on
          the base generation_config EOS ids (Qwen3: <|endoftext|>). The SamplingParams repr (vLLM defaults: top_k,
          min_p, repetition_penalty, ...) is logged per invocation and lands in result.json. The model must be trained
          on the data's training_manifest_sha256. Model: an SFT adapter on the generic base
          (infer.training_identity), or a DPO stage's exact policy (dpo_stages.inference_identity): its adapter on
          its BF16 training base, or (amendment 021) --model RUN/a021/27b|8b --exact-policy (a021_train: the adapter on
          its pinned base; the 27B adds MULTIMODAL_ENGINE, image/video inputs 0); all rehash what vLLM loads. vLLM
          applies LoRA in BF16, so outputs are NOT
          bit-identical to the HF runs of amendment 009/010: the k = 0 pass is the S2 baseline, and best_of_n
          --reference measures its agreement with the HF records on dev-select-v1.
          Rooms sort by (-movable objects, uid); a shard takes [index::count], then --limit (the smoke: the largest).
          Chunks (one LLM.generate call each) take rooms in that order while (N + 1) x their caps sum to at most
          --chunk-tokens (at least one room): a chunk of the largest rooms stays a bounded amount of work, and a
          timeout loses at most the chunk in flight (receipts are written per chunk).
          One write-once receipt per room (OUT/receipts/sha256(uid).json); a rerun skips rooms with a receipt. The
          run identity (model, data, sampling/seed/cap/engine/prompt rules, versions, code) excludes shard, limit and
          out, so shards and the smoke share OUT and run_receipt.json (first writer creates it, the rest compare).
          invocations.jsonl (fsynced) logs load and chunk wall time and tokens, for GPU-hours.
finalize  CPU: every expected room has exactly one receipt of this run with samples k = 0..N; writes samples.jsonl
          (one row per (uid, k), expected_uids then k order), result.json (SAMPLING_COMPLETE, finish reasons, cap
          hits, tokens, gpu_hours, the SamplingParams repr; never a quality claim) and MANIFEST.json, in that order.
          Refuses existing outputs.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from fastfill.comparison import dpo_stages as d
from fastfill.comparison import infer
from fastfill.comparison import llm_open
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.comparison.llm_open import CAP_BASE, CAP_PER_OBJECT, _append, output_cap

N, TEMPERATURE, TOP_P, SEED_SALT, MAX_MODEL_LEN, CHUNK_TOKENS = 8, 0.7, 0.95, 's2-bo-v1:', 40960, 2_000_000
ENGINE = {'enable_lora': True, 'max_lora_rank': 32, 'max_loras': 1, 'max_model_len': MAX_MODEL_LEN,
          'dtype': 'bfloat16', 'seed': 0, 'tensor_parallel_size': 1}
MULTIMODAL_ENGINE = {'limit_mm_per_prompt': {'image': 0, 'video': 0}}  # Amendment 021: the pinned 27B is multimodal.
PROTOCOL = 's2-best-of-n-sampling-v1'
EOS = '<|im_end|>'
PROMPT = ('infer.prompt_ids: the HF tokenizer of the loaded adapter dir, its chat template with add_generation_prompt '
          'and enable_thinking=False, no added special tokens; sent to vLLM as prompt_token_ids')
OUTPUTS = ('samples.jsonl', 'result.json', 'MANIFEST.json')


def sample_seed(uid, k):
    return int(hashlib.sha256(f'{SEED_SALT}{uid}:{k}'.encode('utf-8')).hexdigest()[:8], 16)


def objects(record):
    return len(h.request(infer._schema_record(record))['objects'])


def mine(rows, index, count):
    if not 0 <= index < count:
        raise ValueError('shard index outside shard count')
    return sorted(rows, key=lambda r: (-objects(r), r['uid']))[index::count]


def model_identity(args):
    """(identity, base, adapter): what vLLM loads, rehashed against its training run or stage lineage."""
    sft = None not in (args.adapter, args.training_run) and args.model is None and not args.exact_policy
    dpo = args.model is not None and args.exact_policy and args.adapter is None and args.training_run is None
    if sft == dpo:
        raise ValueError('pass --adapter with --training-run, or --model STAGE-merged --exact-policy')
    if sft:
        run = SimpleNamespace(base=Path(args.base), adapter=Path(args.adapter), training_run=Path(args.training_run),
                              arm=args.arm, context_budget=MAX_MODEL_LEN, max_input_tokens=MAX_MODEL_LEN - 1,
                              max_new_tokens=MAX_MODEL_LEN - 1)
        return infer.training_identity(run), Path(args.base).resolve(), Path(args.adapter).resolve()
    identity = d.inference_identity(SimpleNamespace(base=Path(args.base), model=Path(args.model), arm=args.arm,
                                                    exact_policy=True))
    return identity, Path(identity['policy']['training_base']), Path(identity['policy']['adapter'])


def params(cap, uid, k, eos):
    common = {'n': 1, 'max_tokens': cap, 'skip_special_tokens': False, 'stop_token_ids': [eos], 'ignore_eos': True}
    if k == 0:
        return {**common, 'temperature': 0.0}
    return {**common, 'temperature': TEMPERATURE, 'top_p': TOP_P, 'seed': sample_seed(uid, k)}


def load_vllm(engine):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    return LLM(**engine), SamplingParams, LoRARequest


def chunks(jobs, budget):
    """Consecutive (record, ids, cap) jobs per LLM.generate call: (N + 1) x caps <= budget, at least one room."""
    groups, used = [], 0
    for job in jobs:
        if not groups or used + (N + 1) * job[2] > budget:
            groups.append([])
            used = 0
        groups[-1].append(job)
        used += (N + 1) * job[2]
    return groups


def _path(out, uid):
    return Path(out) / 'receipts' / (hashlib.sha256(uid.encode('utf-8')).hexdigest() + '.json')


def _sample(p, k, raw=None, finish=None, stop=None, info=None):
    return {'k': k, 'seed': p.get('seed'), 'temperature': p['temperature'], 'top_p': p.get('top_p', 1.0),
            'raw': raw, 'finish_reason': finish, 'stop_reason': stop,
            **(info or {'generated_tokens': 0, 'reached_token_cap': False, 'stopped_on_eos': False})}


def build(args):
    """dev_select.build over every dev room (no length admission, no pick)."""
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    pairs, data = h.load_pair(args.data)
    if data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('full dev requires the qualified main data')
    rows = {arm: h._by_uid(pairs['dev'][arm]) for arm in h.ARMS}
    rooms = h._by_uid(h._rows(Path(args.data), 'dev_rooms.jsonl', data['files']['dev_rooms.jsonl']))
    uids = sorted(rows['joint'])
    if set(rows['placement']) != set(uids) or set(rooms) != set(uids):
        raise ValueError('dev rows and rooms cover different UIDs')
    files = {f'{arm}/requests.jsonl': [] for arm in h.ARMS}
    files.update({'rooms.jsonl': [], 'size-estimates.jsonl': []})
    for uid in uids:
        component, room = rows['joint'][uid]['group'], rooms[uid]
        if room['source'] != rows['joint'][uid]['source'] or any(rows[a][uid]['group'] != component for a in h.ARMS):
            raise ValueError(f'dev row/room identity mismatch: {uid}')
        meta = {'uid': uid, 'source': room['source'], 'group': component, 'component': component, 'split': 'dev',
                **({'room_type': room['room_type']} if room.get('room_type') is not None else {})}
        for arm in h.ARMS:
            files[f'{arm}/requests.jsonl'].append(infer.validate_request({**meta, 'messages': rows[arm][uid]['messages'][:2]}, arm))
        files['rooms.jsonl'].append({**room, 'group': component, 'component': component, 'split': 'dev'})
        request = h.strict_json(rows['placement'][uid]['messages'][1]['content'])
        files['size-estimates.jsonl'].append({'uid': uid, 'estimated_sizes': {o['id']: o['size'] for o in request['objects']}})
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        for arm in h.ARMS:
            (dest / arm).mkdir()
        for name, values in files.items():
            (dest / name).write_bytes(d._jsonl(values))
        h.write_json(dest / 'MANIFEST.json', {
            'version': 1, 'scope': 's2_full_dev', 'split': 'dev', 'expected_uids': uids, 'uids_sha256': d._digest(uids),
            'counts': {'dev': len(uids)}, 'training_manifest_sha256': data['manifest_sha256'],
            'main_data_manifest_sha256': data['manifest_sha256'],
            'input_policy': 'system_user_only_labels_and_inventory_separate',
            'code_sha256': {'sample_vllm.py': h.sha256(__file__)}, 'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return {'status': 'DEV_FULL_COMPLETE', 'rooms': len(uids), 'uids_sha256': d._digest(uids)}


def run(args, load=load_vllm):
    started, out = time.monotonic(), Path(args.out)
    rows, data = infer.load_eval(args.data, args.arm)
    model, base, adapter = model_identity(args)
    if model['training_data']['manifest_sha256'] != h.strict_json(
            (Path(args.data) / 'MANIFEST.json').read_text()).get('training_manifest_sha256'):
        raise ValueError('model training data differs from the data training_manifest_sha256')
    # Amendment 021: an a021 policy on the pinned 27B (dpo_stages.inference_identity -> a021_train) runs text only.
    engine = {**ENGINE, **(MULTIMODAL_ENGINE if model.get('base', {}).get('model_type') == 'qwen3_5' else {}),
              'model': str(base)}
    identity = {'protocol': PROTOCOL, 'arm': args.arm, 'model': model,
                'loaded': {'base': str(base), 'adapter': str(adapter), 'lora_request': [args.arm, 1, str(adapter)],
                           'lora_dtype': 'engine dtype (bfloat16)'},
                'data': {'manifest_sha256': data['manifest_sha256'],
                         'requests_sha256': data['files'][f'{args.arm}/requests.jsonl']},
                'sampling': {'n_samples': N, 'temperature': TEMPERATURE, 'top_p': TOP_P, 'greedy': 'k = 0, temperature 0',
                             'seed': f"k >= 1: int(sha256('{SEED_SALT}' + uid + ':' + str(k))[:8], 16)",
                             'requests': 'n = 1 per (uid, k)', 'skip_special_tokens': False, 'stop_token': EOS,
                             'ignore_eos': 'stop on stop_token_ids [stop_token] only (HF eos_token_id = im_end)',
                             'raw': 'infer.decode_output (HF decode up to the first stop token)'},
                'cap': {'base': CAP_BASE, 'per_object': CAP_PER_OBJECT, 'context': MAX_MODEL_LEN,
                        'rule': 'min(context - prompt tokens, base + per_object x movable objects); < 1: INPUT_TOO_LONG'},
                'engine': engine, 'prompt': PROMPT, 'versions': d._versions('vllm', 'torch', 'transformers'),
                'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(infer.__file__), Path(d.__file__),
                                                              Path(h.__file__), Path(llm_open.__file__))}}
    run_hash = infer._hash(identity)
    receipt = {**identity, 'run_sha256': run_hash}
    (out / 'receipts').mkdir(parents=True, exist_ok=True)
    try:  # Shards and the smoke share one identity: the first writer creates it, the rest must match.
        infer.write_once(out / 'run_receipt.json', receipt)
    except FileExistsError:
        if h.strict_json((out / 'run_receipt.json').read_text()) != receipt:
            raise ValueError('run identity mismatch with the existing run_receipt.json') from None
    if (out / 'MANIFEST.json').exists():
        return h.strict_json((out / 'result.json').read_text())
    assigned = mine(rows, args.shard_index, args.shard_count)[:args.limit]
    pending = [r for r in assigned if not _path(out, r['uid']).exists()]
    invocation = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.getpid()}:{time.time_ns()}"
    def log(entry):
        _append(out / 'invocations.jsonl', {'invocation': invocation, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                                            'elapsed_seconds': time.monotonic() - started, **entry})
    log({'event': 'start', 'shard_index': args.shard_index, 'shard_count': args.shard_count, 'limit': args.limit,
         'assigned': len(assigned), 'pending': len(pending)})
    def write(record, ids, cap, error, samples):
        infer.write_once(_path(out, record['uid']), {
            **{k: record[k] for k in ('uid', 'source', 'group')}, 'arm': args.arm, 'run_sha256': run_hash,
            'input_tokens': len(ids), 'objects': objects(record), 'max_tokens': None if error else cap,
            'error': error, 'samples': samples})
    if pending:
        tok = d._model_tokenizer(adapter)  # As infer._runtime: the adapter dir's tokenizer.
        eos = tok.get_vocab().get(EOS)
        if eos is None:
            raise ValueError('Qwen3 tokenizer missing im_end')
        jobs = []
        for record in pending:
            ids = infer.prompt_ids(tok, record)
            if not ids:
                raise ValueError('empty prompt')
            cap = min(MAX_MODEL_LEN - len(ids), output_cap(record, {'thinking': False}))
            if cap < 1:
                write(record, ids, cap, 'INPUT_TOO_LONG',
                      [_sample(params(0, record['uid'], k, eos), k) for k in range(N + 1)])
            else:
                jobs.append((record, ids, cap))
        if jobs:
            llm, sampling, lora_request = load(engine)
            lora = lora_request(args.arm, 1, str(adapter))
            log({'event': 'loaded', 'sampling_params': [repr(sampling(**params(1, 'uid', k, eos))) for k in (0, 1)]})
            for chunk in chunks(jobs, args.chunk_tokens):
                requests = [(record, ids, cap, k) for record, ids, cap in chunk for k in range(N + 1)]
                values = [params(cap, record['uid'], k, eos) for record, _, cap, k in requests]
                begin = time.monotonic()
                outputs = llm.generate([{'prompt_token_ids': ids} for _, ids, _, _ in requests],
                                       [sampling(**p) for p in values], lora_request=lora, use_tqdm=False)
                wall = time.monotonic() - begin
                if len(outputs) != len(requests):
                    raise ValueError('generation batch cardinality mismatch')
                samples = {}
                for (record, ids, cap, k), p, output in zip(requests, values, outputs):
                    if list(output.prompt_token_ids) != ids or len(output.outputs) != 1:
                        raise ValueError('vLLM output does not match its request')
                    completion = output.outputs[0]
                    raw, info = infer.decode_output(tok, list(completion.token_ids), 0, eos, cap)
                    samples.setdefault(record['uid'], []).append(
                        _sample(p, k, raw, completion.finish_reason, completion.stop_reason, info))
                for record, ids, cap in chunk:
                    write(record, ids, cap, None, samples[record['uid']])
                log({'event': 'chunk', 'rooms': len(chunk), 'requests': len(requests), 'wall_seconds': wall,
                     'cap_tokens': sum(cap for _, _, cap, _ in requests),
                     'input_tokens': sum(len(ids) for _, ids, _, _ in requests),
                     'generated_tokens': sum(s['generated_tokens'] for v in samples.values() for s in v)})
    log({'event': 'end'})
    return {'status': 'SHARD_DONE', 'run_sha256': run_hash, 'shard_index': args.shard_index,
            'shard_count': args.shard_count, 'assigned': len(assigned), 'generated_rooms': len(pending)}


def finalize(args):
    out = Path(args.out)
    taken = [n for n in OUTPUTS if os.path.lexists(out / n)]
    if taken:
        raise FileExistsError(f'refusing existing outputs: {taken}')
    receipt = h.strict_json((out / 'run_receipt.json').read_text())
    run_hash, n = receipt['run_sha256'], receipt['sampling']['n_samples']
    if infer._hash({k: v for k, v in receipt.items() if k != 'run_sha256'}) != run_hash:
        raise ValueError('run identity hash mismatch')
    rows, data = infer.load_eval(args.data, receipt['arm'])
    if data['manifest_sha256'] != receipt['data']['manifest_sha256']:
        raise ValueError('sampling run used other data')
    expected = data['expected_uids']
    if {p.name for p in (out / 'receipts').iterdir() if p.suffix == '.json'} != {_path(out, u).name for u in expected}:
        raise ValueError('receipts do not cover exactly the expected rooms (missing or extra receipts)')
    receipts, lines = [], []
    for record in rows:
        value = h.strict_json(_path(out, record['uid']).read_text())
        if (value['uid'], value['run_sha256'], value['arm']) != (record['uid'], run_hash, receipt['arm']):
            raise ValueError(f"receipt identity mismatch: {record['uid']}")
        if [s['k'] for s in value['samples']] != list(range(n + 1)):
            raise ValueError(f"receipt must hold samples k = 0..{n}: {record['uid']}")
        receipts.append(value)
        meta = {k: value[k] for k in ('uid', 'source', 'group', 'arm', 'input_tokens', 'objects', 'max_tokens', 'error')}
        lines += [{**meta, **s, 'run_sha256': run_hash} for s in value['samples']]
    temporary = out / 'samples.jsonl.tmp'
    with temporary.open('xb') as stream:
        stream.write(d._jsonl(lines))
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, out / 'samples.jsonl')
    temporary.unlink()
    events = [h.strict_json(x) for x in (out / 'invocations.jsonl').read_text().splitlines()] \
        if (out / 'invocations.jsonl').exists() else []
    spans = {}
    for event in events:
        spans[event['invocation']] = max(spans.get(event['invocation'], 0), event['elapsed_seconds'])
    samples = [s for v in receipts for s in v['samples']]
    result = {'status': 'SAMPLING_COMPLETE', 'arm': receipt['arm'], 'rooms': len(receipts), 'samples': len(samples),
              'n_samples': n, 'input_too_long_rooms': sum(v['error'] == 'INPUT_TOO_LONG' for v in receipts),
              'finish_reasons': dict(sorted(Counter(str(s['finish_reason']) for s in samples).items())),
              'cap_hits': {'all': sum(s['finish_reason'] == 'length' for s in samples),
                           'greedy': sum(s['finish_reason'] == 'length' for s in samples if s['k'] == 0)},
              'tokens': {'input': sum(v['input_tokens'] * (n + 1) for v in receipts if v['error'] is None),
                         'generated': sum(s['generated_tokens'] for s in samples)},
              'gpu_hours': sum(spans.values()) / 3600, 'invocations': len(spans),
              'sampling_params_repr': [list(x) for x in sorted({tuple(e['sampling_params']) for e in events
                                                                if e['event'] == 'loaded'})],
              'slurm_job_ids': sorted({e['slurm_job_id'] for e in events if e['slurm_job_id']}),
              'quality_claim': False, 'run_sha256': run_hash,
              'scope': {'gpu_hours': 'process wall time x 1 GPU per invocation, model load included; a killed invocation '
                                     'loses the time after its last logged event (sacct over slurm_job_ids is authoritative)',
                        'tokens.input': 'prompt tokens x (N + 1) requests per generated room, before prefix caching'}}
    infer.write_once(out / 'result.json', result)
    infer.write_once(out / 'MANIFEST.json', {'files': h.artifact_manifest(out), 'expected_uids': expected})
    return result


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
    builder, runner, finisher = (sub.add_parser(name) for name in ('build', 'run', 'finalize'))
    for step in (builder, runner, finisher):
        for flag in ('data', 'out'):
            step.add_argument('--' + flag, type=Path, required=True)
    runner.add_argument('--base', type=Path, required=True)
    for flag in ('adapter', 'training-run', 'model'):
        runner.add_argument('--' + flag, type=Path)
    runner.add_argument('--exact-policy', action='store_true')
    runner.add_argument('--arm', choices=h.ARMS, required=True)
    runner.add_argument('--shard-index', type=count(0), default=0)
    runner.add_argument('--shard-count', type=count(1), default=1)
    runner.add_argument('--limit', type=count(1))
    runner.add_argument('--chunk-tokens', type=count(1), default=CHUNK_TOKENS)
    return parser.parse_args(argv)


def main(argv=None, load=load_vllm):
    args = parse_args(argv)
    result = run(args, load) if args.command == 'run' else {'build': build, 'finalize': finalize}[args.command](args)
    print(json.dumps({k: v for k, v in result.items() if k not in ('scope', 'finish_reasons')}))


if __name__ == '__main__':
    main()
