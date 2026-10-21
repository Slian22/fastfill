"""Amendment 022 §1, the delivered FastFill system: prior sizes -> placer P -> S1 -> escalation. CLI: cli.py.

Input, one room per JSONL line, in either form:
  eval-data joint request  {uid, source, group, [component, split, room_type], messages: [J system, J user]}, as in
                           eval-data-v1 joint/requests.jsonl (bound ids obj_NNNN / fixed_NNNN; output ids = those);
  simple room              {"uid": "r1", "room_type": "bedroom", "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
                            "height": 2.8, "fixed": [{"id": "door", "category": "door", "size": [0.1, 0.9, 2.0],
                            "pos": [4, 1.5, 0], "yaw": 1.5708}], "furniture": {"bed": 1, "nightstand": 2}}
                           metres, Z up, boundary counter-clockwise, pos = bottom centre, yaw radians CCW from +X.
                           "objects": [{"id", "category"}] may replace "furniture" (ids then {category}_{k}).
                           Optional: source ("user"), group (uid), room_type, boundary_type ("polygon"), height,
                           fixed. Bound by contract.bind_room and serialised by contract.messages as eval_data does
                           (movable geometry is a placeholder only the dropped label turn reads); output ids and
                           RoomGenBench names are the given ids (the bind sidecar).
Sizes: size_prior.predict_sizes on the pinned train prior (training size-priors.json, its full_prior, or a bare prior;
sha256 recorded and optionally pinned), rounded as eval_data. P's request = the J request with those sizes and P's
system prompt: byte-identical to eval-data-v1 placement/requests.jsonl (verified on all 8,647 rooms, see the tests).
Opt-in size_fallback='api': use eligible category/room or category prior buckets; resolve other objects through
the size-only API before placement (sizing.py). A missing unpinned prior may be loaded with allow_missing=True.
size_api_cap and size_api_max_rooms are separate from layout escalation limits. Failures abort before placement,
with API receipts retained. Missing-size dry runs only write a plan, without model/API execution.

Stages (amendment 022 §1), every candidate scored by geometry.score (placement semantics, the frozen checker) and
repaired by S1 (output_repair.repair_row, amendment 012: max_trials 2000, time_limit 60 s):
  placer    the delivered P: HF exact policy (infer._runtime, adapter unmerged on its training base, infer_shard batches
            and caps) or vLLM greedy (a021_select.greedy: sample_vllm k = 0, 021's engine, (-objects, uid) order in
            LLM.generate calls of <= 400k cap tokens); a token-cap hit without EOS is incomplete (amendment 005). Both
            run in one spawned child process (HF: it sees only the first visible GPU) whose exit frees the GPU for O2.
  gpt       rooms incomplete or failing the checker after S1: gpt-6.1-sol on P's request + llm_agent.SUFFIX (llm_agent
            client, call ledger and redacted receipts), one answer + at most one repair round, B2 enforce_sizes.
  o2        rooms still failing: O2 (Qwen3.8-27B, TP 2 in this process) size-fixed with llm_open's engine and harness,
            scored as amendment 016 §2 / evaluate_llm (no token-cap rule).
  fallback  only if no candidate is complete: every object on the floor at one interior point, then S1.
The answer is the best candidate by (checker pass, complete, fewer checker violations), the earlier stage on ties, so
a passing stage ends the chain and no room is ever dropped or returned empty. The API ledger (OUT/api/calls.jsonl)
counts every HTTP request, retries included; at api_cap the remaining rooms skip the API (receipt error
CALL_CAP_REACHED) and go on to O2; more than api_max_rooms rooms needing the API stops the run before any call.
The env file, the O2 model dir and (real engine) 2 visible GPUs are checked before the placer runs. S1 workers are
forkserver children (never forks of a process holding a vLLM engine). dry_run runs placer + S1 only and lists the
rooms that would escalate.
Outputs (OUT must not exist): rooms.jsonl (per room: the final IR layout id/category/size/pos/yaw/parent, its stage,
every candidate's scores, the placer identity), roomgenbench.jsonl (downstream.to_roomgenbench), result.json (identities,
stage contributions, API calls and tokens) and report.md (Chinese).
"""
import collections
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import functools
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Callable

from shapely.geometry import Polygon

from fastfill.comparison import a021_select as A
from fastfill.comparison import contract, dpo_stages, infer, infer_shard, llm_open, output_repair, sample_vllm
from fastfill.comparison import downstream as d
from fastfill.comparison import geometry as g
from fastfill.comparison import llm_agent as L
from fastfill.comparison import size_prior
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.best_of_n import VIOLATIONS
from fastfill.comparison.report_v2 import cap_failure
from fastfill.system import sizing

S1 = {'max_trials': 2000, 'time_limit': 60.0}                       # amendment 012
TRIGGERS = ('checker', 'strict')                                   # amendment 026 addendum B
API_ROUNDS = 2                                                      # one answer + at most one repair round
API = {'max_output_tokens': 65536, 'timeout': 900, 'concurrency': 4}    # llm_agent run defaults
HF_BUDGET = 40960                                                   # infer_shard plans: budget 40960, tokens 40959
ESCALATE = ('none', 'o2', 'l+o2')
PLACER_KEYS = ('engine', 'model', 'adapter', 'adapter_sha256')


# ------------------------------------------------------------------ requests

def load_prior(path, sha256=None, *, allow_missing=False):
    """(prior table, {'path', 'sha256'}); a training size-priors.json yields its full_prior, as eval_data uses."""
    if path is None:
        if not allow_missing or sha256 is not None:
            raise ValueError('prior is required unless size_fallback=api without a pinned hash')
        return None, {'path': None, 'sha256': None, 'status': 'not_provided'}
    try:
        digest = h.sha256(path)
    except FileNotFoundError:
        if not allow_missing or sha256 is not None:
            raise
        return None, {'path': str(Path(path).resolve()), 'sha256': None, 'status': 'missing'}
    if sha256 is not None and digest != sha256:
        raise ValueError('prior table sha256 differs from the pinned value')
    value = h.strict_json(Path(path).read_text())
    table = value.get('full_prior', value) if isinstance(value, dict) else None
    if not isinstance(table, dict):
        raise ValueError('prior table must be a JSON object')
    return table, {'path': str(Path(path).resolve()), 'sha256': digest}


def joint_request(room, seed=contract.DEFAULT_SEED):
    """Simple room -> (J request record, bind sidecar), bound and serialised exactly as eval_data."""
    objects = room.get('objects') or [{'id': f'{c}_{k}', 'category': c}
                                      for c, n in room.get('furniture', {}).items() for k in range(1, n + 1)]
    prepared = {'uid': room['uid'], 'source': room.get('source', 'user'), 'room_type': room.get('room_type'),
                'boundary_type': room.get('boundary_type', 'polygon'), 'boundary': room['boundary'],
                'height': room.get('height'), 'fixed': room.get('fixed', []),
                'objects': [{'id': o['id'], 'category': o['category'], 'size': [1, 1, 1], 'pos': [0, 0, 0], 'yaw': 0}
                            for o in objects]}
    bound, sidecar = contract.bind_room(prepared, seed)
    meta = {'uid': prepared['uid'], 'source': prepared['source'], 'group': room.get('group', prepared['uid']),
            **({'room_type': prepared['room_type']} if prepared['room_type'] is not None else {})}
    return {**meta, 'messages': contract.messages(bound, 'joint')[:2]}, sidecar


def load_requests(path):
    """[(J record, sidecar or None)] from a JSONL file of eval-data joint requests and/or simple rooms."""
    rows = [h.strict_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [(row, None) if 'messages' in row else joint_request(row) for row in rows]


def prior_sizes(record, prior):
    """eval_data's P sizes: predict_sizes on (room type, id, category), centimetres, at least 1 cm."""
    user = h.strict_json(record['messages'][1]['content'])
    features = {'room_type': record.get('room_type'), 'objects': user['objects']}
    return {oid: [max(.01, round(v, 2)) for v in values]
            for oid, values in size_prior.predict_sizes(features, prior).items()}


def placement_request(record, sizes):
    """P's request: the J request with sizes and P's system prompt, as contract.messages(room, 'placement')."""
    infer.validate_request(record, 'joint')
    if record['messages'][0]['content'] != contract._SYSTEM['joint']:
        raise ValueError('expected an eval-data joint request')
    user = h.strict_json(record['messages'][1]['content'])
    user = {**user, 'objects': [{**o, 'size': sizes[o['id']]} for o in user['objects']]}
    return {**record, 'messages': [{'role': 'system', 'content': contract._SYSTEM['placement']},
                                   {'role': 'user', 'content': contract._json(user)}]}


def prepare(requests, prior, size_by_uid=None):
    """(P records, {uid: output-id names}, {uid: request room}) of load_requests rows under load_prior's table."""
    records = [placement_request(r, prior_sizes(r, prior[0]) if size_by_uid is None else size_by_uid[r['uid']])
               for r, _ in requests]
    if size_by_uid is not None:
        for record in records:
            infer.validate_request(record, 'placement')
    if len({r['uid'] for r in records}) != len(records):
        raise ValueError('duplicate request uid')
    names = {r['uid']: (s or {}).get('bound_to_original', {}) for r, (_, s) in zip(records, requests)}
    return records, names, {r['uid']: request_room(r) for r in records}


def request_room(request):
    """The label-free IR room geometry.score and S1 read: P's request, fixed yaw in radians. No GT exists here, so
    movable objects carry only id, category and the request size (score's size / parent diagnostics are void)."""
    user = h.strict_json(request['messages'][1]['content'])
    return {**{k: request[k] for k in ('uid', 'source', 'group')}, 'room_type': request.get('room_type'),
            'boundary_type': user['boundary_type'], 'boundary': user['boundary'], 'height': user.get('height'),
            'fixed': [{**f, 'yaw': math.radians(f['yaw'])} for f in user.get('fixed', [])],
            'objects': user['objects'], 'constraints': [], contract._MARKER: contract.PROTOCOL}


# ------------------------------------------------------------------ candidates

def _score(task):
    """(stage, room, request, raw, error, cap_hit) -> scores after S1 and the S1 layout (None if incomplete)."""
    stage, room, request, raw, error, cap_hit = task
    row = g.score(room, request['messages'], raw, 'placement', error)
    if cap_hit:
        row = cap_failure(row)
    out = output_repair.repair_row((row, room, request, S1['max_trials'], S1['time_limit']))
    s1, geo = out['s1'], out['s1'].get('geometry_s1')
    v = s1.get('downstream_s1') or {}
    return {'stage': stage, 'complete': row['complete'], 'error': row['error'],
            'checker_pass_raw': row['geometry_success'], 'checker_pass': s1['geometry_success_s1'],
            'violations': sum(geo[k] for k in VIOLATIONS) if geo else None,
            'accepted': s1['accepted_s1'], 's1_status': s1['status'], 'layout': out['s1_layout'],
            # required={} admission (amendment 017); '_' keys are never written, so default outputs are unchanged.
            '_strict': bool(s1.get('strict_ok_s1')),
            '_noreq': bool(v.get('endpoint_ok') and v.get('attach_ok') and v.get('admit_no_required_ok'))}


def _score_all(tasks, workers):
    if workers == 1:
        return [_score(t) for t in tasks]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('forkserver')) as pool:   # O2 may be live
        return list(pool.map(_score, tasks, chunksize=4))


def best(candidates, trigger='checker'):
    """Checker pass > complete > fewer checker violations; max keeps the first (earlier stage) on ties.
    trigger 'strict' (amendment 026 addendum B): checker pass > strict admission (alone, as frontier_026 ranks) >
    required={} > complete > fewer."""
    tail = lambda c: (c['complete'], -math.inf if c['violations'] is None else -c['violations'])
    if trigger == 'strict':
        return max(candidates, key=lambda c: (c['checker_pass'], c['_strict'], c['_noreq'], *tail(c)))
    return max(candidates, key=lambda c: (c['checker_pass'], *tail(c)))


def fallback_answer(request):
    """Last resort: every object on the floor at one interior point; S1 then pushes them apart."""
    # ponytail: one pile + S1; a grid packer if fallback rooms turn out to be common.
    user = h.strict_json(request['messages'][1]['content'])
    p = Polygon(user['boundary']).representative_point()
    return json.dumps({'placements': [{'id': o['id'], 'pos': [p.x, p.y, 0], 'yaw': 0} for o in user['objects']]})


def _llm_task(stage, room, request, body):
    """The final (harness) answer after B2 enforce_sizes; no cap rule, as amendment 016 §2 / evaluate_llm score O2."""
    answers = body['answers']
    if not answers:
        return (stage, room, request, None, body['error'] or 'NOT_ANSWERED', False), None
    text, counts = L.enforce_sizes(answers[-1]['raw'], request)
    return (stage, room, request, text, None, False), counts


def _place(records, rooms, placer, workers):
    """The placer stage: {uid: [placer candidate after S1]} in request order."""
    answers = placer.generate(records)
    missing = {'raw': None, 'error': 'NOT_GENERATED', 'cap_hit': False}
    tasks = [('placer', rooms[r['uid']], r, *(answers.get(r['uid'], missing)[k] for k in ('raw', 'error', 'cap_hit')))
             for r in records]
    return {r['uid']: [c] for r, c in zip(records, _score_all(tasks, workers))}


def _failing(records, cands, trigger='checker'):
    """The rooms the system escalates: no candidate passes the frozen checker ('strict': the checker and strict
    admission, i.e. 'accepted'; amendment 026 addendum B)."""
    key = 'accepted' if trigger == 'strict' else 'checker_pass'
    return [r for r in records if not any(c[key] for c in cands[r['uid']])]


def _escalate(cands, stage, records, rooms, bodies, workers):
    tasks, counts = zip(*(_llm_task(stage, rooms[r['uid']], r, bodies[r['uid']]) for r in records))
    for r, c, n in zip(records, _score_all(tasks, workers), counts):
        cands[r['uid']].append({**c, 'size_enforcement': n})


# ------------------------------------------------------------------ placers

@dataclass(frozen=True)
class Placer:
    """identity: what is loaded (in every output); generate([P records]) -> {uid: {'raw', 'error', 'cap_hit'}}."""
    identity: dict
    generate: Callable


def _config_files(model):
    return {p.name: h.sha256(p) for p in sorted(Path(model).iterdir())
            if p.is_file() and not p.name.endswith(llm_open.WEIGHTS)}


def model_identity(engine, model, adapter, **settings):
    """Paths, non-weight base files and the adapter files; adapter_sha256 as infer.training_identity hashes them."""
    model, adapter = Path(model).resolve(), Path(adapter).resolve()
    files = h.artifact_manifest(adapter)
    return {'engine': engine, 'model': str(model), 'adapter': str(adapter), 'adapter_sha256': infer._hash(files),
            'adapter_files_sha256': files, 'model_files_sha256': _config_files(model), **settings}


def _isolated(generate, one_gpu, records):
    """generate(records) in one fresh spawned child (one_gpu: it sees only the first visible GPU, as infer._runtime
    requires). The placer's CUDA context and GPU memory end with the child, so O2 (TP 2) then loads in this process."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn')) as pool:
        try:
            if one_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = ('0' if visible is None else visible).split(',')[0]
            future = pool.submit(generate, records)         # a spawn pool starts its child here, with this env
        finally:
            if visible is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = visible
        return future.result()


def _hf_generate(model, adapter, decode, records):
    """hf_placer's pass (in its child): infer.process_batch over infer_shard.batches (order, shared caps)."""
    for key in infer_shard.OFFLINE:
        os.environ[key] = '1'
    tok, gen, eos = infer._runtime(SimpleNamespace(base=Path(model), adapter=Path(adapter), exact_policy=True,
                                                   **decode))
    by, result = {r['uid']: r for r in records}, {}
    items = [{'uid': r['uid'], 'objects': sample_vllm.objects(r),
              'input_tokens': dict.fromkeys(h.ARMS, len(infer.prompt_ids(tok, r)))} for r in records]
    for b, batch in enumerate(infer_shard.batches(items, decode)):
        for row in infer.process_batch([by[u] for u in batch['uids']], 'placement', tok, gen, eos,
                                       decode['max_input_tokens'], decode['max_new_tokens'], b,
                                       context_budget=HF_BUDGET, shared_max_new_tokens=batch['max_new_tokens']):
            result[row['uid']] = {'raw': row['raw'], 'error': row['error'],
                                  'cap_hit': row['reached_token_cap'] and not row['stopped_on_eos']}
    return result


def hf_placer(model, adapter, batch_size=8):
    """HF exact policy on one BF16 GPU (the first visible one, in a child process)."""
    decode = {'batch_size': batch_size, 'context_budget': HF_BUDGET, 'max_input_tokens': HF_BUDGET - 1,
              'max_new_tokens': HF_BUDGET - 1, 'cap_base': infer_shard.CAP_BASE,
              'cap_per_object': infer_shard.CAP_PER_OBJECT}
    generate = functools.partial(_hf_generate, str(model), str(adapter), decode)
    return Placer(model_identity('hf_exact_policy', model, adapter, decode=decode, sampling='greedy',
                                 lora='unmerged adapter (FP32 as trained) on its BF16 training base',
                                 process='one spawned child, first visible GPU only'),
                  functools.partial(_isolated, generate, True))


def _vllm_generate(adapter, engine, records, load=sample_vllm.load_vllm):
    """vllm_placer's pass (in its child), as a021_select.greedy: rooms in (-objects, uid) order, LLM.generate calls of
    <= 400k cap tokens, each output checked against its prompt ids."""
    tok = dpo_stages._model_tokenizer(adapter)
    eos = tok.get_vocab()[sample_vllm.EOS]
    result, jobs = {}, []
    for r in A._order(records, 'objects'):
        ids = infer.prompt_ids(tok, r)
        cap = min(sample_vllm.MAX_MODEL_LEN - len(ids), llm_open.output_cap(r, {'thinking': False}))
        if cap < 1:
            result[r['uid']] = {'raw': None, 'error': 'INPUT_TOO_LONG', 'cap_hit': False}
        else:
            jobs.append((r, ids, cap))
    if jobs:
        llm, sampling, lora_request = load(engine)
        lora = lora_request('placement', 1, adapter)
        for chunk in A._chunks(jobs, A.BATCHINGS['objects']):
            outputs = llm.generate([{'prompt_token_ids': ids} for _, ids, _ in chunk],
                                   [sampling(**sample_vllm.params(cap, r['uid'], 0, eos)) for r, _, cap in chunk],
                                   lora_request=lora, use_tqdm=False)
            for (r, ids, cap), output in zip(chunk, outputs, strict=True):
                if list(output.prompt_token_ids) != ids or len(output.outputs) != 1:
                    raise ValueError('vLLM output does not match its request')
                completion = output.outputs[0]
                try:
                    raw, error = infer.decode_output(tok, list(completion.token_ids), 0, eos, cap)[0], None
                except ValueError:
                    raw, error = None, 'DECODE_FAILED'
                result[r['uid']] = {'raw': raw, 'error': error, 'cap_hit': completion.finish_reason == 'length'}
    return result


def vllm_placer(model, adapter, load=sample_vllm.load_vllm):
    """a021_select.greedy's engine and pass (k = 0): BF16 LoRA, adapter tokenizer, cap min(40960 - prompt, 1024 + 128 x
    objects); a base config with a vision_config (Qwen3.8-27B) gets image/video inputs 0. Runs in a child process."""
    multimodal = 'vision_config' in h.strict_json((Path(model) / 'config.json').read_text())
    engine = {**sample_vllm.ENGINE, 'model': str(Path(model).resolve()),
              **(sample_vllm.MULTIMODAL_ENGINE if multimodal else {})}
    generate = functools.partial(_vllm_generate, str(Path(adapter).resolve()), engine, load=load)
    return Placer(model_identity('vllm_greedy_bf16_lora', model, adapter, vllm_engine=engine,
                                 sampling='sample_vllm.params k = 0 (temperature 0)',
                                 cap='min(40960 - prompt tokens, 1024 + 128 x movable objects)',
                                 batching='objects', chunk_cap_tokens=A.BATCHINGS['objects'],
                                 process='one spawned child'),
                  functools.partial(_isolated, generate, False))


# ------------------------------------------------------------------ escalation

def _api_solve(run, record):
    """llm_agent._solve with API_ROUNDS rounds and placement parsing (B2); the cap or an outage ends the room."""
    uid, answers, errors = record['uid'], [], []
    messages = [*record['messages'], {'role': 'user', 'content': L.SUFFIX}]
    try:
        for round_ in range(API_ROUNDS):
            reply, failed = L._answer(run, uid, 1, round_, messages)
            errors += failed
            schema = L.parse(reply['text'], record, 'placement')
            answers.append({'raw': reply['text'], 'schema': schema, **{k: reply[k] for k in (
                'status', 'incomplete_reason', 'usage', 'latency_seconds', 'tries')}})
            if schema['valid']:
                break
            messages = [*messages, {'role': 'assistant', 'content': reply['text']},
                        {'role': 'user', 'content': schema['error']}]
    except L._Stop:
        pass
    except L._Failed as exc:     # fail closed, as llm_agent: no further API calls in this run
        run.halt('API_FAILURE', {'uid': uid, 'attempt': 1, 'round': len(answers), 'errors': exc.errors})
    run.receipt(record, {'attempt': 1, 'error': None if answers else run.reason or 'NOT_ANSWERED',
                         'answers': answers, 'http_errors': errors})


def _api_config(env_file):
    config = L.load_config(env_file)
    if config.model != L.EXPECTED_MODEL:
        raise L.ApiError('configured_model_mismatch')
    return config


def _visible_gpus():
    import torch
    return torch.cuda.device_count()


def api_answers(records, out, cap, env_file, post=L.call_api, concurrency=API['concurrency']):
    """gpt-6.1-sol on P's requests through llm_agent's ledger: {uid: receipt}, identity + calls + tokens."""
    config = _api_config(env_file)
    identity = {'model': L.EXPECTED_MODEL, 'reasoning_effort': L.EFFORT, 'rounds': API_ROUNDS, 'suffix': L.SUFFIX,
                'input': 'placement_system_user_verbatim_then_suffix_user', 'max_calls': cap,
                'parser': 'infer.parse_prediction(enforce_sizes(raw), placement_request, placement)',
                **{k: API[k] for k in ('max_output_tokens', 'timeout')}}
    run_hash = infer._hash(identity)
    (out / 'receipts').mkdir(parents=True)
    args = SimpleNamespace(max_calls=cap, arm='placement', max_attempts=1, **{k: API[k] for k in ('max_output_tokens',
                                                                                                   'timeout')})
    run = L._Run(out, run_hash, config, args, post, [r['uid'] for r in records])
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(lambda r: _api_solve(run, r), records))
    receipts = {r['uid']: h.strict_json(run.path(r['uid']).read_text()) for r in records}
    usage = [a['usage'] for v in receipts.values() for a in v['answers']]
    return receipts, {**identity, 'run_sha256': run_hash, 'rooms': len(records), 'calls_ledger': run.count,
                      'stopped': run.reason, 'tokens': {k: sum(u.get(k, 0) for u in usage)
                                                        for k in ('input_tokens', 'output_tokens', 'reasoning_tokens')}}


def _o2_setup(model_dir):
    """(spec, vLLM engine, identity) of O2 size-fixed with llm_open's engine and harness."""
    spec, model = llm_open.SPECS['O2'], Path(model_dir).resolve()
    engine = {'model': str(model), 'tensor_parallel_size': spec['tp'], 'max_model_len': llm_open.MAX_MODEL_LEN,
              'seed': llm_open.SEED, **spec['engine']}
    identity = {'system': 'O2', 'model': spec['name'], 'model_files_sha256': _config_files(model), 'engine': engine,
                'sampling': llm_open.SAMPLING, 'chat_template_kwargs': spec['template'],
                'repair_rounds': L.REPAIRS, 'suffix': L.SUFFIX, 'request_arm': 'placement'}
    return spec, engine, identity


def _o2_chunks(records, spec, chunk):
    """llm_open's chunk order (cap desc, prompt characters desc, uid), cut into chunks of `chunk` rooms."""
    pending = sorted(records, key=lambda r: (-llm_open.output_cap(r, spec), -len(r['messages'][1]['content']),
                                             r['uid']))
    return [pending[start:start + chunk] for start in range(0, len(pending), chunk)]


def o2_answers(records, out, model_dir, load=llm_open.load_vllm, chunk=256):
    """O2 size-fixed with llm_open's engine, chunk order and harness (llm_open._chunk): {uid: body}, identity."""
    spec, engine, identity = _o2_setup(model_dir)
    out.mkdir(parents=True)
    started = time.monotonic()
    log = lambda e: llm_open._append(out / 'invocations.jsonl', {'elapsed_seconds': time.monotonic() - started, **e})
    llm, params = load(engine)
    tok = llm.get_tokenizer()
    bodies = {}
    for part in _o2_chunks(records, spec, chunk):
        bodies.update(llm_open._chunk(llm, params, tok, spec, part, log, 'placement'))
    (out / 'receipts.jsonl').write_text(''.join(json.dumps({'uid': r['uid'], **bodies[r['uid']]}, allow_nan=False)
                                                + '\n' for r in records))
    return bodies, {**identity, 'rooms': len(records)}


# ------------------------------------------------------------------ sharded O2 (fallback for one long O2 job)
# plan (CPU): run()'s placer stage once -> the escalated set; o2_shard (GPU, one per shard, resumable per chunk);
# run(placed=plan candidates, o2=o2_merge of the finished shards) writes the outputs a normal run writes.

SHARD_RULE = "int(sha256('a022-o2-shard:' + uid), 16) mod N"
PLAN_FLAGS = {'_strict': 'strict_s1', '_noreq': 'noreq_s1'}        # best()'s strict keys, public in a strict plan
PLAN_FLAGS_BACK = {v: k for k, v in PLAN_FLAGS.items()}


def shard_of(uid, shards):
    return int(L._digest('a022-o2-shard:' + uid), 16) % shards


def plan(requests, prior, placer, out, shards, workers=1, trigger='checker'):
    """run()'s placer stage (placer + S1) once, persisted for the shards and the finalize: OUT/placer.jsonl (every
    room's placer candidate, request order) then OUT/plan.json (written last: the escalated uids, N, the rule).
    trigger 'strict' (amendment 027): escalated = _failing(..., 'strict'), plan.json escalation_trigger, and
    placer.jsonl keeps best()'s strict / required={} flags as PLAN_FLAGS (load_plan restores them); 'checker' writes
    neither."""
    if shards < 1 or trigger not in TRIGGERS:
        raise ValueError(f'shards must be >= 1 and trigger one of {TRIGGERS}')
    records, _, rooms = prepare(requests, prior)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cands = _place(records, rooms, placer, workers)
    flags = lambda c: {v: c[k] for k, v in PLAN_FLAGS.items()} if trigger == 'strict' else {}
    _jsonl(out / 'placer.jsonl', [{'uid': r['uid'], **_public(cands[r['uid']][0]), **flags(cands[r['uid']][0])}
                                  for r in records])
    escalated = [r['uid'] for r in _failing(records, cands, trigger)]
    value = {'status': 'PLAN_COMPLETE', 'rooms': len(records), 'records_sha256': infer._hash(records),
             'placer': placer.identity, 'prior': prior[1], 's1': S1, 'shards': shards, 'shard_rule': SHARD_RULE,
             'escalated': escalated, 'shard_rooms': [sum(shard_of(u, shards) == k for u in escalated)
                                                     for k in range(shards)],
             'placer_sha256': h.sha256(out / 'placer.jsonl'), 'code_sha256': {'pipeline.py': h.sha256(__file__)},
             **({'escalation_trigger': trigger} if trigger != 'checker' else {})}
    h.write_json(out / 'plan.json', value)
    return value


def load_plan(out):
    """(plan.json, {uid: placer candidate}) of a complete plan; a strict plan's PLAN_FLAGS become best()'s keys."""
    out = Path(out)
    value = h.strict_json((out / 'plan.json').read_text())
    if value.get('status') != 'PLAN_COMPLETE' or h.sha256(out / 'placer.jsonl') != value['placer_sha256']:
        raise ValueError(f'{out}: not a complete plan')
    placed = {}
    for line in (out / 'placer.jsonl').read_text().splitlines():
        row = h.strict_json(line)
        uid = row.pop('uid')
        placed[uid] = {PLAN_FLAGS_BACK.get(k, k): v for k, v in row.items()}
    return value, placed


def _shard_identity(records, model_dir, index, shards, chunk):
    """(spec, engine, chunks, identity): o2_answers' identity + the shard (index, N, rule, chunk size, chunk uids)."""
    spec, engine, identity = _o2_setup(model_dir)
    parts = _o2_chunks(records, spec, chunk)
    return spec, engine, parts, {**identity, 'rooms': len(records), 'sharding': {
        'shards': shards, 'index': index, 'rule': SHARD_RULE, 'chunk_size': chunk,
        'chunks': [[r['uid'] for r in part] for part in parts]}}


def _shard_bodies(out, identity):
    """{uid: body tagged with shard and chunk} from every chunk file the shard identity lists."""
    s, bodies = identity['sharding'], {}
    for k, uids in enumerate(s['chunks']):
        value = h.strict_json((Path(out) / 'chunks' / f'{k:04d}.json').read_text())
        if (value['shard'], value['chunk'], [x['uid'] for x in value['receipts']]) != (s['index'], k, uids):
            raise ValueError(f'{out}: chunk {k} is not the one the shard identity lists')
        for x in value['receipts']:
            bodies[x.pop('uid')] = {'shard': s['index'], 'chunk': k, **x}
    return bodies


def o2_shard(records, out, model_dir, index, shards, load=llm_open.load_vllm, chunk=256):
    """o2_answers on one shard's rooms, resumable: same engine, sampling, template, harness and chunking (size, order
    over the shard's rooms). Each finished chunk is persisted at once (chunks/NNNN.json, atomic, write-once), so a rerun
    redoes only the unfinished chunks, each whole. identity.json must match on resume; COMPLETE.json marks the end."""
    spec, engine, parts, identity = _shard_identity(records, model_dir, index, shards, chunk)
    out = Path(out)
    (out / 'chunks').mkdir(parents=True, exist_ok=True)
    if not (out / 'identity.json').exists():
        infer.write_once(out / 'identity.json', identity)
    elif h.strict_json((out / 'identity.json').read_text()) != identity:
        raise ValueError(f'{out}: resume identity mismatch')
    path = lambda k: out / 'chunks' / f'{k:04d}.json'
    todo = [k for k in range(len(parts)) if not path(k).exists()]
    started, job = time.monotonic(), os.environ.get('SLURM_JOB_ID')
    log = lambda e: llm_open._append(out / 'invocations.jsonl', {
        'slurm_job_id': job, 'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started, **e})
    log({'event': 'start', 'pending_chunks': todo})
    if todo:
        llm, params = load(engine)
        tok = llm.get_tokenizer()
        for k in todo:
            bodies = llm_open._chunk(llm, params, tok, spec, parts[k], log, 'placement')
            infer.write_once(path(k), {'shard': index, 'chunk': k,
                                       'receipts': [{'uid': r['uid'], **bodies[r['uid']]} for r in parts[k]]})
            log({'event': 'chunk_done', 'chunk': k})
    bodies = _shard_bodies(out, identity)
    if not (out / 'COMPLETE.json').exists():
        infer.write_once(out / 'COMPLETE.json', {'status': 'SHARD_COMPLETE', 'index': index, 'shards': shards,
                                                 'rooms': len(records), 'chunks': len(parts),
                                                 'identity_sha256': infer._hash(identity)})
    return bodies, identity


def o2_merge(records, out, shard_dirs, model_dir, chunk=256):
    """run()'s O2 stage from finished o2_shard dirs (shard k = shard_dirs[k], N = their count): each must carry
    COMPLETE.json and exactly the identity o2_shard builds for its rooms of `records` under the rule, so every room is
    answered once, with o2_answers' settings. Writes OUT/receipts.jsonl as o2_answers does (lines tagged with shard and
    chunk); the identity is o2_answers' plus 'sharding' (N, rule, chunk size, per-shard rooms / chunks / identity)."""
    shards, bodies, detail = len(shard_dirs), {}, []
    for k, d in enumerate(map(Path, shard_dirs)):
        mine = [r for r in records if shard_of(r['uid'], shards) == k]
        identity = _shard_identity(mine, model_dir, k, shards, chunk)[3]
        if not (d / 'COMPLETE.json').exists() or h.strict_json((d / 'identity.json').read_text()) != identity:
            raise ValueError(f'{d}: not the finished shard {k} of {shards} for these rooms and settings')
        bodies.update(_shard_bodies(d, identity))
        detail.append({'index': k, 'dir': str(d.resolve()), 'rooms': len(mine),
                       'chunks': len(identity['sharding']['chunks']), 'identity_sha256': infer._hash(identity)})
    if sorted(bodies) != sorted(r['uid'] for r in records):
        raise ValueError('the shards do not answer exactly the escalated rooms')
    out.mkdir(parents=True)
    (out / 'receipts.jsonl').write_text(''.join(json.dumps({'uid': r['uid'], **bodies[r['uid']]}, allow_nan=False)
                                                + '\n' for r in records))
    base = _o2_setup(model_dir)[2]
    return bodies, {**base, 'rooms': len(records), 'sharding': {'shards': shards, 'rule': SHARD_RULE,
                                                                'chunk_size': chunk, 'detail': detail}}


# ------------------------------------------------------------------ system

def _layout(candidate, names):
    name = lambda i: names.get(i, i)
    return [{'id': name(o['id']), 'category': o['category'], 'size': o['size'], 'pos': o['pos'], 'yaw': o['yaw'],
             'parent': None if o.get('parent') is None else name(o['parent'])} for o in candidate['layout']]


def _export(room, candidate, names):
    try:
        condition, layout, scene = d.to_roomgenbench(room, candidate['layout'], names)
        return {'condition': condition, 'layout': layout, 'scene': scene, 'error': None}
    except ValueError as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}


def _public(candidate):
    return {k: v for k, v in candidate.items() if not k.startswith('_')}


def _jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def run(requests, prior, placer, out, *, escalate='none', api_cap=None, api_max_rooms=None, env_file=None,
        o2_model=None, dry_run=False, workers=1, post=L.call_api, o2_load=llm_open.load_vllm,
        api_concurrency=API['concurrency'], placed=None, o2=None, trigger='checker',
        size_fallback='prior', size_api_cap=None, size_api_max_rooms=None):
    """requests = load_requests(...); prior = load_prior(...); placer = hf_placer / vllm_placer / any Placer.
    Sharded O2 only: placed = load_plan's placer candidates (the placer stage is not rerun; trigger 'strict' needs a
    strict plan's, which carry best()'s flags); o2(todo, OUT/o2) -> (bodies, identity) replaces o2_answers (o2_merge;
    no GPU here)."""
    if escalate not in ESCALATE:
        raise ValueError(f'escalate must be one of {ESCALATE}')
    if trigger not in TRIGGERS:
        raise ValueError(f'trigger must be one of {TRIGGERS}')
    if trigger == 'strict' and placed is not None and any(PLAN_FLAGS.keys() - c.keys() for c in placed.values()):
        raise ValueError('trigger strict: the placed candidates lack the strict / required={} flags (no strict plan)')
    if size_fallback not in sizing.POLICIES:
        raise ValueError(f'size_fallback must be one of {sizing.POLICIES}')
    if size_fallback == 'prior' and prior[0] is None:
        raise ValueError('a prior is required unless size_fallback=api')
    if size_fallback == 'api' and (placed is not None or o2 is not None):
        raise ValueError('API sizing cannot reuse precomputed placed candidates or O2 callbacks')
    size_plan = sizing.plan(requests, prior[0]) if size_fallback == 'api' else None
    if size_plan is not None:
        sizing.preflight(size_plan, size_api_cap, size_api_max_rooms, dry_run=dry_run)
        if dry_run and size_plan['requests']:
            names = {r['uid']: (s or {}).get('bound_to_original', {}) for r, s in requests}
            return sizing.dry_report(size_plan, prior, out, size_api_cap, size_api_max_rooms, names)
    size_config = None
    if size_plan is not None and size_plan['requests']:
        if env_file is None:
            raise ValueError('size API needs env_file')
        size_config = _api_config(env_file)
    if not dry_run and ((escalate == 'l+o2' and None in (api_cap, env_file)) or (escalate != 'none' and not o2_model)):
        raise ValueError('escalation needs api_cap and env_file (l+o2) and o2_model (o2, l+o2)')
    if not dry_run and escalate == 'l+o2':
        _api_config(env_file)                       # fail now, not after the placer and S1
    if not dry_run and escalate != 'none':
        _config_files(o2_model)
        if o2 is None and o2_load is llm_open.load_vllm and _visible_gpus() < llm_open.SPECS['O2']['tp']:
            raise RuntimeError(f"O2 needs {llm_open.SPECS['O2']['tp']} visible GPUs (tensor parallel)")
    started, out = time.monotonic(), Path(out)
    size_api = None
    if size_plan is not None:
        out.mkdir(parents=True, exist_ok=False)
        if size_plan['requests']:
            size_plan, size_api = sizing.estimate(size_plan, out / 'size-api', size_api_cap,
                                                   size_config, post, api_concurrency)
    records, names, rooms = prepare(requests, prior, size_plan['sizes'] if size_plan is not None else None)
    if placed is not None and placed.keys() != rooms.keys():
        raise ValueError('the placed candidates are not these requests')
    if size_plan is None:
        out.mkdir(parents=True, exist_ok=False)
    else:
        _jsonl(out / 'size-estimates.jsonl', [
            {'uid': r['uid'], 'estimated_sizes': {names[r['uid']].get(k, k): v
                                                for k, v in size_plan['sizes'][r['uid']].items()},
             'sources': {names[r['uid']].get(k, k): v for k, v in size_plan['sources'][r['uid']].items()},
             'placement_request_sha256': infer._hash(r)} for r in records])
    cands = _place(records, rooms, placer, workers) if placed is None else {r['uid']: [placed[r['uid']]]
                                                                             for r in records}
    failing = lambda: _failing(records, cands, trigger)
    needs = [r['uid'] for r in failing()]
    info = {}
    if escalate == 'l+o2' and not dry_run and (todo := failing()):
        if api_max_rooms is not None and len(todo) > api_max_rooms:
            scope = 'layout API' if size_plan is not None else 'API'
            raise RuntimeError(f'{len(todo)} rooms need the API > api_max_rooms {api_max_rooms}; no {scope} call was made')
        receipts, info['api'] = api_answers(todo, out / 'api', api_cap, env_file, post, api_concurrency)
        _escalate(cands, 'gpt', todo, rooms, receipts, workers)
    if escalate != 'none' and not dry_run and (todo := failing()):
        bodies, info['o2'] = o2_answers(todo, out / 'o2', o2_model, o2_load) if o2 is None else o2(todo, out / 'o2')
        _escalate(cands, 'o2', todo, rooms, bodies, workers)
    empty = [r for r in records if not any(c['complete'] for c in cands[r['uid']])]
    for r, c in zip(empty, _score_all([('fallback', rooms[r['uid']], r, fallback_answer(r), None, False)
                                       for r in empty], workers)):
        cands[r['uid']].append(c)
    final = {uid: best(c, trigger) for uid, c in cands.items()}
    if any(c['layout'] is None for c in final.values()):
        raise RuntimeError('a room has no complete candidate after the fallback')
    tag = {k: placer.identity.get(k) for k in PLACER_KEYS}
    _jsonl(out / 'rooms.jsonl', [
        {**{k: r[k] for k in ('uid', 'source', 'group')}, 'room_type': r.get('room_type'),
         **{k: final[r['uid']][k] for k in ('stage', 'complete', 'checker_pass', 'accepted', 'violations')},
         'needs_escalation': r['uid'] in needs, 'layout': _layout(final[r['uid']], names[r['uid']]),
         **({'size_sources': {names[r['uid']].get(k, k): v for k, v in size_plan['sources'][r['uid']].items()}}
            if size_plan is not None else {}),
         'candidates': [{k: v for k, v in _public(c).items() if k != 'layout'} for c in cands[r['uid']]], 'placer': tag}
        for r in records])
    _jsonl(out / 'roomgenbench.jsonl', [{'uid': r['uid'], 'stage': final[r['uid']]['stage'], 'placer': tag,
                                         **_export(rooms[r['uid']], final[r['uid']], names[r['uid']])}
                                        for r in records])
    first = [c[0] for c in cands.values()]
    count = lambda values: sum(bool(v) for v in values)
    result = {
        'status': 'DRY_RUN' if dry_run else 'SYSTEM_COMPLETE', 'rooms': len(records), 'escalate': escalate,
        'dry_run': dry_run, 'api_cap': api_cap, 'api_max_rooms': api_max_rooms, 'api_rounds': API_ROUNDS, 's1': S1,
        'placer': placer.identity, **({'escalation_trigger': trigger} if trigger != 'checker' else {}),
        'prior': prior[1], 'stages': {
            'placer_raw_checker_pass': count(c['checker_pass_raw'] for c in first),
            'placer_s1_checker_pass': count(c['checker_pass'] for c in first),
            'needs_escalation': len(needs),
            'final_checker_pass': count(c['checker_pass'] for c in final.values()),
            'final_complete': count(c['complete'] for c in final.values()),
            'final_accepted': count(c['accepted'] for c in final.values()),
            'final_stage': dict(collections.Counter(c['stage'] for c in final.values()))},
        'escalation': info, 'would_escalate': needs if dry_run and escalate != 'none' else [],
        'api_calls_upper_bound': len(needs) * API_ROUNDS * (1 + L.RETRIES) if dry_run and escalate == 'l+o2' else None,
        'code_sha256': {Path(m.__file__).name: h.sha256(m.__file__) for m in (
            contract, size_prior, g, output_repair, d, L, llm_open, infer, infer_shard, sample_vllm, A)}
        | {'pipeline.py': h.sha256(__file__)},
        'runtime_seconds': time.monotonic() - started}
    if size_plan is not None:
        result = {**result, 'sizing': sizing.summary(size_plan, size_api), 'size_api_cap': size_api_cap,
                  'size_api_max_rooms': size_api_max_rooms,
                  'api_calls_total': (size_api['calls_ledger'] if size_api else 0)
                                     + info.get('api', {}).get('calls_ledger', 0),
                  'code_sha256': {**result['code_sha256'], 'sizing.py': h.sha256(sizing.__file__),
                                  'sizers_020.py': h.sha256(sizing.S.__file__)}}
    h.write_json(out / 'result.json', result)
    _report(out, result)
    return result


def _report(out, s):
    n, st = s['rooms'], s['stages']
    rate = lambda k: f"{st[k]} / {n}（{st[k] / n:.3f}）" if n else '—'
    api = s['escalation'].get('api')
    size_text = (f"训练集先验表 sha256 {s['prior']['sha256'][:12]}" if s['prior'].get('sha256') else '未提供先验表')
    if 'sizing' in s:
        size_text += f"；尺寸来源 {s['sizing']['objects_by_source']}"
    lines = ['# FastFill 交付系统运行报告（amendment 022 §1）', '',
             f"放置模型：{s['placer'].get('engine')}，adapter sha256 {str(s['placer'].get('adapter_sha256'))[:12]}；"
             f"尺寸：{size_text}；S1：max_trials {S1['max_trials']}，"
             f"每房间 {S1['time_limit']:.0f} 秒。" + ('**试运行（dry run）：未调用 API 与 O2。**' if s['dry_run'] else ''), '',
             '| 阶段 | 冻结检查器通过 |', '|---|---:|',
             f"| 放置模型原始输出 | {rate('placer_raw_checker_pass')} |",
             f"| + S1 修复 | {rate('placer_s1_checker_pass')} |",
             f"| + 升级（{s['escalate']}），最终 | {rate('final_checker_pass')} |", '',
             f"S1 后仍需升级的房间：{st['needs_escalation']}。最终结果完整：{rate('final_complete')}；"
             f"下游接受（检查器且 downstream.accept）：{rate('final_accepted')}。最终来源：{st['final_stage']}。",
             f"API 调用（账本，含重试）：{api['calls_ledger'] if api else 0}，上限 {s['api_cap']}"
             + (f"，因 {api['stopped']} 停止" if api and api['stopped'] else '') + '。', '',
             '升级结果属于"带 API / 额外模型的系统"，不能当作单一模型与单次生成的智能体直接比较。']
    if 'sizing' in s:
        size_api = s['sizing']['api']
        lines += ['', f"尺寸 API 调用（含重试）：{size_api['calls_ledger'] if size_api else 0}，"
                      f"独立上限 {s['size_api_cap']}；全流程 API 调用：{s['api_calls_total']}。",
                  'API 尺寸是估计值；尺寸来源及最终尺寸见 size-estimates.jsonl，未测量该模式的质量。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
