"""Open-weight agents O1a-O7 (amendments 003/004/006/007): the API agent's input and harness on offline vLLM.

  python -m fastfill.comparison.llm_open run --system O4 --data EVAL --cohort COHORT.json \
      --model-root MODELS --out RUN [--limit 2 --smoke-gate] [--chunk 256] [--arm placement [--sizes gt]] \
      [--uids FILE] [--all-rooms]
  python -m fastfill.comparison.llm_open slurm --system O4 --mode smoke|full   (sbatch resource flags)
  python -m fastfill.comparison.llm_open status --run-root ROOT    (writes ROOT/llm-open-status.json)

Same as llm_agent: frozen cohort (rebuilt and compared), the exact J system+user messages plus
llm_agent.SUFFIX, infer.parse_prediction(answer, request, 'joint'), up to llm_agent.REPAIRS repair
rounds that append the answer as the assistant turn and a user turn holding only the parser error
text, variants llm_raw/llm_harness, and the output layout written by llm_agent._finalize, which
evaluate_llm consumes unchanged.

Generation: one vLLM LLM.chat call per round per chunk of rooms (chunks follow an input-derived
order: movable objects desc, prompt characters desc, UID; outputs are written in cohort order),
greedy (temperature 0, seed 0), the model's own chat template with the system's fixed
chat_template_kwargs, max_model_len 40960. Per-answer cap = min(1024 + 128 x movable objects (+16384
for thinking systems), 40960 - prompt tokens), the prompt counted as LLM.chat renders it (the
tokenizer's chat template, encoded without added special tokens; actual counts are recorded too).
A prompt with no room left is not generated: INPUT_TOO_LONG as the room error in round 0 (a failure
in every denominator), repair_stopped in a repair round.

Reasoning is removed before parsing from text decoded with special tokens kept: the answer is the
text after the last </think> (Qwen3, Qwen3.8, DeepSeek-V4, Muse-Glimmer), after the last <channel|> (Gemma 4), or
the gpt-oss harmony final channel up to <|return|>/<|end|>/<|call|>. vLLM's reasoning parsers are
not used: they serve OpenAI requests, and the gpt-oss one does not extract offline. An unclosed
reasoning block in a thinking system is all reasoning unless generation stopped by itself.
reasoning_tokens = generated - answer tokens (so it includes channel markers and the stop token).
Latency is llm.chat wall time / requests in the call (amortized).

Receipts (write-once, one per UID) make reruns resume; invocations.jsonl (fsynced) logs every
llm.chat call and the process wall time for GPU-hours; cost.json summarises them. The run identity
binds data, cohort, code, the system config (model path, model file hashes, engine and sampling
settings, chat_template_kwargs) and the vllm/torch/transformers versions.

--arm placement (amendment 016 section 2, size-fixed regeneration): the P request (P's system prompt, user
JSON with the train-prior sizes) replaces the J request, parsed by llm_agent.parse (addendum B2: returned sizes are
replaced by the request's and supported z recomputed by llm_agent.enforce_sizes, then
infer.parse_prediction(..., 'placement')), so sizes come only from the request and a size-only difference is not
invalid; everything else above is unchanged. --sizes gt (addendum A1 supplementary: O2 on the pilot) sends P's
request with GT sizes instead (llm_agent.gt_request on rooms.jsonl). --uids FILE runs that subset of the cohort
(llm_agent.subset; the pilot of addendum A6/B1). The identity records request_arm, request_sizes, the rooms hash
and the subset only when given: the default (joint) identity is unchanged.

--all-rooms (amendment 018 teacher, distill_018): the cohort is every request of --data in its expected_uids order
(all_rooms_cohort; the teacher plan is a train set run whole in --uids shards) instead of L.build_cohort's test sample;
recorded as config cohort_rule 'all_rooms' only when given.

--system O2nt (amendment 021 section 1.4, reference rows only): O2 (same model files, engine, TP, harness, sampling and
repairs) with enable_thinking False and so the non-thinking cap. It must reason 0 tokens: an answer with any reasoning
raises before its receipt is written (config reasoning_tokens 'zero_else_failure'). O2nt is in SPECS, not SYSTEMS: the
amendment 003-007 status table (llm-open-status.json), scripts and every existing identity are unchanged.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import train_pair_helpers as h

MAX_MODEL_LEN = 40960
SEED = 0
CAP_BASE, CAP_PER_OBJECT, THINKING_ALLOWANCE = 1024, 128, 16384
SAMPLING = {'temperature': 0.0, 'seed': SEED, 'skip_special_tokens': False, 'spaces_between_special_tokens': False}
WEIGHTS = ('.safetensors', '.bin', '.pt', '.pth', '.gguf')
# hours = full-run Slurm limit; smoke runs get SMOKE_MINUTES. Modes are fixed by amendments 006/007 (written
# out explicitly where the template reads them); thinking systems get THINKING_ALLOWANCE in their cap.
# 'reasoning_effort' is the template variable vLLM's server fills from a request's reasoning_effort.
SYSTEMS = {
    'O1a': {'name': 'qwen3-8b-nothink', 'model': 'Qwen3-8B', 'tp': 1, 'hours': 3, 'split': 'think',
            'thinking': False, 'template': {'enable_thinking': False}, 'effort': 'thinking off', 'engine': {}},
    'O1b': {'name': 'qwen3-8b-think', 'model': 'Qwen3-8B', 'tp': 1, 'hours': 6, 'split': 'think',
            'thinking': True, 'template': {'enable_thinking': True}, 'effort': 'thinking on', 'engine': {}},
    'O2': {'name': 'qwen3.8-27b', 'model': 'Qwen3.8-27B', 'tp': 2, 'hours': 8, 'split': 'think',
           'thinking': True, 'template': {'enable_thinking': True}, 'effort': 'thinking on (model default)',
           'engine': {'limit_mm_per_prompt': {'image': 0, 'video': 0}}},
    'O3': {'name': 'gemma-4-31b-it', 'model': 'gemma-4-31b-it', 'tp': 2, 'hours': 4, 'split': 'gemma',
           'thinking': False, 'template': {'enable_thinking': False}, 'effort': 'thinking off (model default)',
           'engine': {'limit_mm_per_prompt': {'image': 0, 'audio': 0}}},
    'O4': {'name': 'gpt-oss-120b', 'model': 'gpt-oss-120b', 'tp': 2, 'hours': 6, 'split': 'harmony',
           'thinking': True, 'template': {'reasoning_effort': 'medium'}, 'effort': 'medium', 'engine': {}},
    'O5': {'name': 'deepseek-v4-flash-0731', 'model': 'DeepSeek-V4-Flash-0731', 'tp': 4, 'hours': 6, 'split': 'think',
           'thinking': True, 'template': {'thinking': True, 'reasoning_effort': 'high'},
           'effort': 'thinking on, high (amendment 007; the model offers low/high/max)',
           # vLLM recipe: built-in DeepSeek-V4 encoding (no Jinja template), EP MoE, FP8 KV, block 256. The Hopper
           # (H200) recipe uses the Marlin MoE backend; DeepGEMM JIT cannot build here (nvcc/CCCL header mismatch),
           # so the sbatch disables DeepGEMM for O5.
           'engine': {'tokenizer_mode': 'deepseek_v4', 'enable_expert_parallel': True, 'kv_cache_dtype': 'fp8',
                      'block_size': 256, 'trust_remote_code': True, 'moe_backend': 'marlin'}},
    'O6': {'name': 'qwen3.8-flash-next-fp8', 'model': 'Qwen3.8-Flash-Next-FP8', 'tp': 4, 'hours': 6, 'split': 'think',
           'thinking': True, 'template': {'enable_thinking': True, 'reasoning_effort': 'medium'},
           'effort': 'thinking on, medium (amendment 007; the model offers xhigh/medium/low)',
           # vLLM recipe on Hopper: TP4 + expert parallel (plain TP conflicts with its 128-wide FP8 blocks), triton MoE.
           'engine': {'enable_expert_parallel': True, 'moe_backend': 'triton',
                      'limit_mm_per_prompt': {'image': 0, 'video': 0}}},
    # ponytail: no cached Muse-Glimmer chat template; '</think>' is assumed. If it closes reasoning otherwise,
    # stopped answers carry no reasoning and the smoke fails closed ('thinking system produced no reasoning').
    'O7': {'name': 'muse-glimmer-30b', 'model': 'Muse-Glimmer-30B', 'tp': 2, 'hours': 8, 'split': 'muse',
           'thinking': True, 'template': {}, 'effort': 'model default (no "Reasoning strength:" system line)',
           'engine': {}},
}
# Amendment 021 section 1.4: run / slurm accept these too; status, scripts and SYSTEMS identities ignore them.
EXTRA_SYSTEMS = {
    'O2nt': {**SYSTEMS['O2'], 'name': 'qwen3.8-27b-nothink', 'hours': 3, 'thinking': False,
             'template': {'enable_thinking': False}, 'effort': 'thinking off (amendment 021)', 'no_reasoning': True},
}
SPECS = {**SYSTEMS, **EXTRA_SYSTEMS}
_TOO_LARGE = 'not run (exceeds one 4xGH200 node in a Hopper precision)'
_NOT_SELECTED = 'not run (fits; not selected: lower adoption/scores or a custom vLLM build)'
NOT_RUN = {  # Amendment 007.
    **dict.fromkeys(('qwen3.8-2.4t-a95b', 'deepseek-v4-pro', 'kimi-k3', 'kimi-k2.6', 'minimax-m3', 'mimo-v2.6-pro',
                     'mistral-large-3', 'nemotron-3-ultra', 'llama-4-maverick'), _TOO_LARGE),
    'deepseek-v4.1-flash': 'not run (exceeds one node: 510 GB)',
    'glm-5.2': 'not run (exceeds one node: 755 GB FP8)', 'glm-5.3': 'not run (exceeds one node: 755 GB FP8)',
    'glm-5.3-flash': 'not run (FP8 328 GB leaves no KV-cache headroom on one node)',
    'llama-4-scout': 'not run (gated without access: HTTP 403)',
    **dict.fromkeys(('mimo-v2.6-flash', 'minimax-m2.7', 'step-3.7-flash', 'hy3', 'mistral-small-4',
                     'mistral-medium-3.5', 'nemotron-3-super'), _NOT_SELECTED),
    'phi': 'not run (context too short)',
    'qwen3.6-27b': 'not run (superseded by Qwen3.8-27B)',
}
SMOKE_MINUTES = 90
HARMONY_FINAL = re.compile(r'<\|channel\|>\s*final\b[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>')
HARMONY_END = re.compile(r'<\|(?:return|end|call)\|>')
CLOSE = {'think': '</think>', 'gemma': '<channel|>'}
MUSE_ANSWER = 'to=user<|message|>'


def split_reasoning(text, style, thinking, stopped):
    """(reasoning, answer) from text decoded with special tokens kept."""
    if style == 'harmony':
        finals = list(HARMONY_FINAL.finditer(text))
        if not finals:
            return text, ''
        end = finals[-1].end()
        return text[:end], HARMONY_END.split(text[end:], 1)[0]
    if style == 'muse':  # ATEM: reasoning on 'assistant to=self' (<|eom|>), answer on 'to=user' (<|eot|>)
        head, close, tail = text.rpartition(MUSE_ANSWER)
        if close:  # reasoning only counts when a 'to=self' message was actually produced
            return (head + close if 'to=self' in head else ''), tail.split('<|eot|>', 1)[0]
        return (text, '') if thinking and (not stopped or 'to=self' in text) else ('', text)
    head, close, tail = text.rpartition(CLOSE[style])
    if close:
        return head + close, tail
    return (text, '') if thinking and not stopped else ('', text)


def output_cap(record, spec):
    objects = len(h.request(infer._schema_record(record))['objects'])
    return CAP_BASE + CAP_PER_OBJECT * objects + (THINKING_ALLOWANCE if spec['thinking'] else 0)


def out_name(system, mode):
    return f'llm-open-{system}-smoke-001' if mode == 'smoke' else f'llm-open-{system}-001'


def slurm_flags(system, mode):
    spec = SPECS[system]
    minutes = SMOKE_MINUTES if mode == 'smoke' else 60 * spec['hours']
    return (f"--gpus-per-node={spec['tp']} --cpus-per-task={16 * spec['tp']} --mem={100 * spec['tp']}G "
            f'--time={minutes // 60:02d}:{minutes % 60:02d}:00')


def all_rooms_cohort(data):
    """--all-rooms: every request uid of DATA in expected_uids order, in L.build_cohort's format."""
    rows, identity = infer.load_eval(data, 'joint')
    uids = [r['uid'] for r in rows]
    return {'version': 1, 'data_manifest_sha256': identity['manifest_sha256'], 'fields_used': ['uid'],
            'algorithm': 'every request uid of the data, expected_uids order (llm_open --all-rooms, amendment 018)',
            'counts': {'total': len(uids)}, 'uids_sha256': L._digest(json.dumps(uids)), 'uids': uids}


def load_vllm(engine):
    from vllm import LLM, SamplingParams
    return LLM(**engine), SamplingParams


def _versions():
    versions = {}
    for name in ('vllm', 'torch', 'transformers'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not_installed'
    return versions


def _append(path, entry):
    with path.open('a') as stream:
        stream.write(json.dumps(entry, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _prompt_tokens(tok, conversation, kwargs):
    prompt = tok.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, **kwargs)
    return len(prompt) if isinstance(prompt, list) else len(tok.encode(prompt, add_special_tokens=False))


def _answer(output, cap, precount, spec, tok, record, latency, arm):
    completion = output.outputs[0]
    stopped, capped = completion.finish_reason == 'stop', completion.finish_reason == 'length'
    reasoning, text = split_reasoning(completion.text, spec['split'], spec['thinking'], stopped)
    prompt, generated = len(output.prompt_token_ids), len(completion.token_ids)
    answer_tokens = len(tok.encode(text, add_special_tokens=False)) if text else 0
    return {'raw': text, 'schema': L.parse(text, record, arm), 'reasoning': reasoning,
            'finish_reason': completion.finish_reason, 'status': 'incomplete' if capped else 'completed',
            'incomplete_reason': 'max_output_tokens' if capped else None,
            'reached_token_cap': capped, 'stopped_on_eos': stopped,
            'max_tokens': cap, 'prompt_tokens_precount': precount, 'answer_tokens': answer_tokens,
            'usage': {'input_tokens': prompt, 'output_tokens': generated, 'total_tokens': prompt + generated,
                      'reasoning_tokens': max(0, generated - answer_tokens) if reasoning else 0},
            'latency_seconds': latency}


def _chunk(llm, params, tok, spec, records, log, arm):
    """All rounds for one chunk; returns {uid: receipt body}."""
    by_uid = {r['uid']: r for r in records}
    conversations = {uid: [*r['messages'], {'role': 'user', 'content': L.SUFFIX}] for uid, r in by_uid.items()}
    answers, stopped, active = {uid: [] for uid in by_uid}, {}, list(by_uid)
    for round_ in range(L.REPAIRS + 1):
        jobs = []
        for uid in active:
            precount = _prompt_tokens(tok, conversations[uid], spec['template'])
            cap = min(output_cap(by_uid[uid], spec), MAX_MODEL_LEN - precount)
            if cap < 1:
                stopped[uid] = 'INPUT_TOO_LONG'
            else:
                jobs.append((uid, precount, cap))
        if not jobs:
            break
        start = time.monotonic()
        outputs = llm.chat([conversations[uid] for uid, _, _ in jobs],
                           [params(max_tokens=cap, **SAMPLING) for _, _, cap in jobs],
                           use_tqdm=False, chat_template_kwargs=spec['template'] or None)
        wall = time.monotonic() - start
        if len(outputs) != len(jobs):
            raise ValueError('generation batch cardinality mismatch')
        log({'event': 'call', 'round': round_, 'requests': len(jobs), 'wall_seconds': wall})
        active = []
        for (uid, precount, cap), output in zip(jobs, outputs):
            answer = _answer(output, cap, precount, spec, tok, by_uid[uid], wall / len(jobs), arm)
            violated = bool(spec.get('no_reasoning') and (answer['reasoning'] or answer['usage']['reasoning_tokens']))
            if violated:  # Amendment 021 addendum B: a no-reasoning answer that reasons is a failure, never repaired.
                answer = {**answer, 'schema': {**answer['schema'], 'valid': False,
                                               'error': 'reasoning_in_no_reasoning_system'},
                          'no_reasoning_violation': True}
            answers[uid].append(answer)
            if not answer['schema']['valid'] and round_ < L.REPAIRS and not violated:
                conversations[uid] = [*conversations[uid], {'role': 'assistant', 'content': answer['raw']},
                                      {'role': 'user', 'content': answer['schema']['error']}]
                active.append(uid)
    return {uid: {'attempt': 1, 'error': None if values else stopped[uid], 'answers': values, 'http_errors': [],
                  'repair_stopped': stopped.get(uid) if values else None, 'rounds': len(values), 'calls': len(values),
                  'valid_first': bool(values) and values[0]['schema']['valid'],
                  'valid_final': bool(values) and values[-1]['schema']['valid']}
            for uid, values in answers.items()}


def _cost(out, system, identity, receipts):
    events = L._jsonl(out / 'invocations.jsonl')
    spans = {}
    for event in events:
        spans[event['invocation']] = max(spans.get(event['invocation'], 0), event['elapsed_seconds'])
    calls = [e for e in events if e['event'] == 'call']
    answers = [a for r in receipts.values() for a in r['answers']]
    gpus = SPECS[system]['tp']
    return {'system': system, 'model': identity['config']['model'], 'model_path': identity['config']['engine']['model'],
            'config_sha256': identity['config_sha256'], 'versions': identity['config']['versions'], 'gpus': gpus,
            'invocations': len(spans), 'slurm_job_ids': sorted({e['slurm_job_id'] for e in events if e['slurm_job_id']}),
            'wall_seconds': sum(spans.values()), 'gpu_hours': sum(spans.values()) * gpus / 3600,
            'generation_seconds': sum(e['wall_seconds'] for e in calls),
            'generated_requests': sum(e['requests'] for e in calls),
            'answers_with_content': sum(bool(a['raw']) for a in answers),
            'input_too_long_rooms': sum(r['error'] == 'INPUT_TOO_LONG' for r in receipts.values()),
            'repair_stopped_input_too_long': sum(r['repair_stopped'] == 'INPUT_TOO_LONG' for r in receipts.values()),
            'prompt_token_precount_mismatches': sum(a['usage']['input_tokens'] != a['prompt_tokens_precount']
                                                    for a in answers),
            'scope': {'gpu_hours': 'process wall time x GPUs, model load included; a killed invocation loses the time '
                                   'after its last ledger event (sacct over slurm_job_ids is authoritative)',
                      'latency_seconds': 'llm.chat wall time / requests in that call (amortized, not per request)',
                      'generated_requests': 'every generated answer, including chunks lost to a kill'}}


def run(args, load=load_vllm):
    started = time.monotonic()
    spec = SPECS[args.system]
    rows, data = infer.load_eval(args.data, args.arm)
    cohort = h.strict_json(Path(args.cohort).read_text())
    if cohort != (all_rooms_cohort(args.data) if args.all_rooms else L.build_cohort(args.data)):
        raise ValueError('cohort file differs from its deterministic rebuild')
    uids = (L.subset(cohort, args.uids) if args.uids else cohort['uids'])[:args.limit]
    by_uid = h._by_uid(rows)
    records = [by_uid[uid] for uid in uids]
    gt = args.sizes == 'gt'
    if gt:
        if args.arm != 'placement':
            raise ValueError('--sizes gt needs --arm placement')
        rooms = h._by_uid(h._rows(Path(args.data), 'rooms.jsonl', data['files']['rooms.jsonl']))
        records = [L.gt_request(record, rooms[record['uid']]) for record in records]
    model = (Path(args.model_root) / spec['model']).resolve()
    files = {p.name: h.sha256(p) for p in sorted(model.iterdir()) if p.is_file() and not p.name.endswith(WEIGHTS)}
    if 'config.json' not in files:
        raise ValueError('model directory lacks config.json')
    config = {'system': args.system, 'model': spec['name'], 'reasoning_effort': spec['effort'],
              'model_files_sha256': files, 'chat_template_kwargs': spec['template'], 'thinking': spec['thinking'],
              'engine': {'model': str(model), 'tensor_parallel_size': spec['tp'], 'max_model_len': MAX_MODEL_LEN,
                         'seed': SEED, **spec['engine']},
              'sampling': SAMPLING, 'cap': {'base': CAP_BASE, 'per_object': CAP_PER_OBJECT,
                                            'thinking_allowance': THINKING_ALLOWANCE if spec['thinking'] else 0},
              'reasoning_split': spec['split'], 'repair_rounds': L.REPAIRS,
              'suffix': L.SUFFIX, 'input': f'{args.arm}_system_user_verbatim_then_suffix_user',
              'parser': ('infer.parse_prediction(answer_after_reasoning, joint_request, joint)' if args.arm == 'joint'
                         else 'infer.parse_prediction(enforce_sizes(answer_after_reasoning), placement_request, placement)'),
              'repair_message': 'parser_error_text_only', 'limit': args.limit, 'versions': _versions(),
              **({'request_arm': 'placement'} if args.arm == 'placement' else {}),  # Joint identity unchanged.
              **({'request_sizes': 'gt'} if gt else {}),
              **({'cohort_rule': 'all_rooms'} if args.all_rooms else {}),  # Amendment 018 teacher plan.
              **({'reasoning_tokens': 'zero_else_failure'} if spec.get('no_reasoning') else {})}  # Amendment 021 O2nt.
    identity = {'arm': 'llm_open', 'variants': list(L.VARIANTS), 'config': config, 'config_sha256': infer._hash(config),
                'data': {'manifest_sha256': data['manifest_sha256'],
                         f'{args.arm}_requests_sha256': data['files'][f'{args.arm}/requests.jsonl'],
                         **({'rooms_sha256': data['files']['rooms.jsonl']} if gt else {})},
                'cohort': {'sha256': h.sha256(args.cohort), 'uids_sha256': cohort['uids_sha256'], 'rooms': len(uids),
                           **({'subset': {'sha256': h.sha256(args.uids), 'uids_sha256': L._digest(json.dumps(uids))}}
                              if args.uids else {})},
                'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(L.__file__), Path(infer.__file__),
                                                              Path(h.__file__))}}
    run_hash = infer._hash(identity)
    out = Path(args.out)
    (out / 'receipts').mkdir(parents=True, exist_ok=True)
    if (out / 'run_receipt.json').exists():
        if h.strict_json((out / 'run_receipt.json').read_text()) != {**identity, 'run_sha256': run_hash}:
            raise ValueError('resume identity mismatch')
    else:
        infer.write_once(out / 'run_receipt.json', {**identity, 'run_sha256': run_hash})
    if (out / 'MANIFEST.json').exists():
        return h.strict_json((out / 'result.json').read_text())
    def path(uid):
        return out / 'receipts' / (L._digest(uid) + '.json')
    invocation = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.getpid()}:{time.time_ns()}"
    def log(entry):
        _append(out / 'invocations.jsonl', {'invocation': invocation, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                                            'elapsed_seconds': time.monotonic() - started, **entry})
    pending = sorted((r for r in records if not path(r['uid']).exists()),
                     key=lambda r: (-output_cap(r, spec), -len(r['messages'][1]['content']), r['uid']))
    log({'event': 'start', 'pending': len(pending)})
    if pending:
        llm, params = load(config['engine'])
        tok = llm.get_tokenizer()
        log({'event': 'loaded'})
        for start in range(0, len(pending), args.chunk):
            chunk = pending[start:start + args.chunk]
            bodies = _chunk(llm, params, tok, spec, chunk, log, args.arm)
            for record in chunk:
                infer.write_once(path(record['uid']), {**{k: record[k] for k in ('uid', 'source', 'group')},
                                                       'run_sha256': run_hash, **bodies[record['uid']]})
            h.write_json(out / 'status.json', {'status': 'RUNNING', 'expected_count': len(uids),
                                               'receipt_count': sum(path(uid).exists() for uid in uids)})
    receipts = {}
    for uid in uids:
        receipt = h.strict_json(path(uid).read_text())
        if receipt['uid'] != uid or receipt['run_sha256'] != run_hash:
            raise ValueError('receipt identity mismatch')
        receipts[uid] = receipt
    log({'event': 'finalize'})
    calls = sum(e['requests'] for e in L._jsonl(out / 'invocations.jsonl') if e['event'] == 'call')
    h.write_json(out / 'cost.json', _cost(out, args.system, identity, receipts))
    return L._finalize(out, records, receipts, run_hash, calls)


def _read(path):
    return h.strict_json(path.read_text()) if path.exists() else None


def _reason(status):
    return ': '.join(str(status[k]) for k in ('status', 'error_type', 'error') if status.get(k))


def smoke_failure(out):
    """None when the smoke passed, else the reason the system is 'not run'.

    Amendment 006: 'not run' only when the system cannot load or execute. A smoke that loads and
    generates but yields only cap-hit reasoning (or invalid answers) is a measured failure and passes.
    """
    status = _read(out / 'status.json')
    if status is None:
        return 'smoke output missing (job not started or failed before its first write; see the Slurm log)'
    if status.get('status') != 'LLM_COMPLETE':
        return _reason(status)
    if not _read(out / 'cost.json')['generated_requests']:
        return 'smoke generated no answer'
    receipt = _read(out / 'run_receipt.json') or {}
    if (receipt.get('config') or {}).get('thinking') and not (status.get('tokens') or {}).get('reasoning_tokens'):
        return 'thinking system produced no reasoning (mode not applied)'
    return None


def write_status(root):
    root, rows = Path(root), {}
    for system, spec in SYSTEMS.items():
        full = root / out_name(system, 'full')
        status = _read(full / 'status.json') or {}
        failure = smoke_failure(root / out_name(system, 'smoke'))
        if status.get('status') == 'LLM_COMPLETE':
            state, reason = 'complete', None
        elif failure:
            state, reason = 'not run', failure
        else:
            state, reason = 'incomplete', _reason(status) or 'full run not started'
        rows[system] = {'model': spec['name'], 'state': state, 'reason': reason, 'dir': full.name,
                        'result': status if state == 'complete' else None, 'cost': _read(full / 'cost.json')}
    rows.update({name: {'model': name, 'state': 'not run', 'reason': reason} for name, reason in NOT_RUN.items()})
    h.write_json(root / 'llm-open-status.json', rows)
    return rows


def main(argv=None, load=load_vllm):
    def positive(text):
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError('must be positive')
        return value
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run_parser, slurm, status = (commands.add_parser(name) for name in ('run', 'slurm', 'status'))
    for flag in ('data', 'cohort', 'model-root', 'out'):
        run_parser.add_argument('--' + flag, type=Path, required=True)
    for sub in (run_parser, slurm):
        sub.add_argument('--system', choices=SPECS, required=True)
    run_parser.add_argument('--limit', type=positive)
    run_parser.add_argument('--chunk', type=positive, default=256)
    run_parser.add_argument('--smoke-gate', action='store_true')
    run_parser.add_argument('--arm', choices=h.ARMS, default='joint')  # placement: amendment 016 section 2.
    run_parser.add_argument('--sizes', choices=('prior', 'gt'), default='prior')  # gt: addendum A1 supplementary.
    run_parser.add_argument('--uids', type=Path)  # A subset of the cohort (amendment 016 pilot).
    run_parser.add_argument('--all-rooms', action='store_true')  # Cohort = every request (amendment 018 teacher plan).
    slurm.add_argument('--mode', choices=('smoke', 'full'), required=True)
    status.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'slurm':
        print(slurm_flags(args.system, args.mode))
        return 0
    if args.command == 'status':
        print(json.dumps({name: row['state'] for name, row in write_status(args.run_root).items()}))
        return 0
    try:
        result = run(args, load)
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        if not (args.out / 'MANIFEST.json').exists():  # Never touch a published run.
            h.write_json(args.out / 'status.json', {'status': 'FAIL', 'error_type': type(exc).__name__,
                                                    'error': str(exc)[:2000]})
        raise
    failure = smoke_failure(args.out) if args.smoke_gate else None
    print(json.dumps({**{k: result.get(k) for k in ('status', 'rooms', 'valid_first', 'valid_final',
                                                    'reached_token_cap', 'tokens')}, 'smoke_failure': failure}))
    return 0 if result['status'] == 'LLM_COMPLETE' and not failure else 1


if __name__ == '__main__':
    sys.exit(main())
