"""Amendment 020 stage 1 (EXPLORATORY): LLM sizers give real-world sizes only; size quality vs GT (table A).

  python -m fastfill.comparison.sizers_020 rooms --data RUN/eval-data-v1 --e2 RUN/optiscene-014/e2 \
      --e3 RUN/optiscene-015/e3 --out RUN/a020-rooms-v1.json
  python -m fastfill.comparison.sizers_020 cap --run-root RUN --rooms N            (API cap preflight, read-only)
  python -m fastfill.comparison.sizers_020 run-api --data EVAL --rooms ROOMS --run-root RUN --out RUN/sizes-020-L \
      [--env-file ~/.fastfill_api.env] [--concurrency 4] [--limit N]
  python -m fastfill.comparison.sizers_020 run-open --data EVAL --rooms ROOMS --model-root MODELS \
      --out RUN/sizes-020-O2 --shard-index I --shard-count N [--chunk 256] [--limit N]
  python -m fastfill.comparison.sizers_020 finalize --rooms ROOMS --out RUN/sizes-020-O2
  python -m fastfill.comparison.sizers_020 report --data EVAL --rooms ROOMS --run-root RUN --l RUN/sizes-020-L \
      --o2 RUN/sizes-020-O2 --l-joint RUN/llm-eval-L-selected-v1 --out RUN/sizes-020-report-v1 [--repeats 2000]

Rooms: E2 = the uids of the amendment-014 plan-e2 requests, E3 = those of the amendment-015 plan-e3 requests (disjoint),
in eval-data expected_uids order, each with room_families.family of its raw room type. Written once (cohort format);
an existing file must equal its rebuild.

Sizer request: J's request (joint/requests.jsonl) with J's user JSON kept as is (room type, boundary, height, fixed
boxes with their known geometry, movable id/category: J's request has no movable placement field) and J's system
prompt replaced by SYSTEM: J's frame and size definition verbatim (SIZE_DEFINITION, same axes as J's output), the
real-world size of each object id asked for, no placement, fit or shrink instruction. Answer = JSON {id: [sx, sy, sz]}
in J's axes as the amendment requires (local +X is the front, so [sx, sy, sz] = [depth, width, height]; the amendment's
"[w, d, h]" names the extents, not this order). Then llm_agent.SUFFIX as in the agent runs. parse_sizes: strict JSON, exactly the
requested ids, each 3 positive finite numbers < 1000; else invalid, with a fixed error text (never model text) as
the repair turn. One answer + repair rounds (L-size: L_REPAIRS = 1, addendum A; O2-size: llm_agent.REPAIRS); a room
without a valid final answer is a failure.

L-size (run-api): gpt-6.1-sol through llm_agent's own client, ledger and receipts (load_config, call_api, _Run,
_answer: amendment-006 settings, every HTTP request fsynced to calls.jsonl before it is sent, transient retries, fail
closed on outages, write-once receipts so a rerun resumes). Cap preflight before anything else: calls used = per API
run directly under --run-root (run_receipt arm llm_agent, or another sizer_020 L run) the larger of its calls.jsonl
lines and its receipts' calls; refuses (status.json CAP_REFUSED, no call, exit 1) when used + rooms x 3 > 4,800 (rooms =
the whole rooms file, --limit or not) or when a counted run has no MANIFEST.json (it could still resume and spend), and
its own ledger is capped at its rooms x 3 across resumes, so the total never passes 4,800.
O2-size (run-open): Qwen3.8-27B with llm_open's amendment-007 O2 engine, chat template kwargs, sampling, output cap
and reasoning split; shard I of N = rooms[I::N]; shards share one identity and receipts directory and run side by
side; finalize merges once every room has its receipt.

Outputs (RUN/sizes-020-{L,O2}): run_receipt.json, receipts/, estimates.jsonl ({uid, estimated_sizes {id: [sx, sy,
sz]}}: the size-estimates.jsonl format, valid rooms only), result.json (failed rooms with reasons), MANIFEST.json.

report: per size source (GT, prior = P's input, L-joint = sizes in gpt-6.1-sol's amendment-016 J-input run on cohort
rooms (llm_harness rows; API rows carry no cap flags), L-size, O2-size) and scope (all, E2, E3, each family): completeness, size log MAE and
centred-box IoU (geometry.dimensions), footprint bias B (report_v2.size_bias), as levels and paired differences
(report_v2.contrasts = geometry.paired_intervals' house-component bootstrap, seed 42; no CI below report_v2.NO_CI
components). Writes sizes.json, report.md (Chinese), levels.csv, contrasts.csv, MANIFEST.json, once. Refuses unless
O2-size is complete; L-size may be reported as not run (CAP_REFUSED or no output directory).
"""
import argparse
import collections
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

from fastfill.comparison import contract
from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import train_pair_helpers as h

PROTOCOL = 'amendment-020'
# Amendment 020 addendum A: the LLM-agent call cap (amendment 002's 4,800) is raised to 6,500 and L-size gets one answer
# + at most one repair round per room (O2-size, free, keeps llm_agent.REPAIRS rounds).
CAP, L_REPAIRS = 6500, 1
CALLS_PER_ROOM = 1 + L_REPAIRS
SEED = 42
SIZE_DEFINITION = 'size=[sx,sy,sz] denotes positive extents along local X,Y,Z; local +X is the front.'
FRAME = ('Frame: meters, right-handed, Z up; floor z=0. boundary is a counter-clockwise floor polygon; '
         'boundary_type polygon denotes walls, hull denotes a scanned floor hull, proxy an approximate boundary. '
         'height is the ceiling height; a missing field means unknown.\n')
POSE = 'pos=[x,y,z] is the bottom center. yaw is integer degrees in [0,360), counter-clockwise from room +X.'
SYSTEM = ('You are a furniture size estimator. Give the real-world dimensions of every listed movable object.\n'
          + FRAME + 'Objects have opaque instance IDs and explicit categories. Fixed items have explicit categories '
          'and known size, pos and yaw. ' + POSE + '\n' + SIZE_DEFINITION + '\n'
          'Output only JSON {"<id>":[sx,sy,sz],...} with every listed object ID exactly once and nothing else.')
REQUEST_KEYS = ('room_type', 'boundary_type', 'boundary', 'height', 'fixed', 'objects')  # J's user JSON order.
FIXED_KEYS = ('id', 'pos', 'yaw', 'size', 'category')
NOT_JSON = 'invalid size output: not one strict JSON value'
NOT_OBJECT = 'invalid size output: expected one JSON object {"<id>":[sx,sy,sz],...}'
WRONG_IDS = 'invalid size output: every listed object ID exactly once and no other key'
BAD_SIZE = 'invalid size output: each size is [sx,sy,sz], three positive finite numbers below 1000'
FIELDS = ('complete', 'size_log_mae', 'size_iou', 'bias_fp')
SOURCES = ('GT', 'prior', 'L-joint', 'L-size', 'O2-size')
PAIRS = (('L-size', 'prior'), ('O2-size', 'prior'), ('L-size', 'O2-size'), ('L-size', 'L-joint'))


# ---- rooms and requests ----------------------------------------------------------------------------------------

def build_rooms(data, e2, e3):
    """The amendment-020 room file: E2 and E3 plan uids in expected_uids order with population and family."""
    from fastfill.comparison import optiscene_eval as oe
    from fastfill.comparison.room_families import family
    rows, identity = infer.load_eval(data, 'joint')
    plans, population = {}, {}
    for name, root in (('e2', e2), ('e3', e3)):
        meta = oe._verified(root)
        if meta.get('plan') != name or meta.get('eval_data_manifest_sha256') != identity['manifest_sha256']:
            raise ValueError(f'{root} is not a plan-{name} output of this evaluation data')
        uids = {r['uid'] for r in oe._jsonl(Path(root) / 'requests.jsonl')}
        if uids & set(population):
            raise ValueError('E2 and E3 rooms overlap')
        population.update(dict.fromkeys(uids, name))
        plans[name] = {'manifest_sha256': oe._sha(Path(root) / 'MANIFEST.json'), 'rooms': len(uids)}
    if not set(population) <= set(identity['expected_uids']):
        raise ValueError('plan rooms outside the evaluation data')
    picked = [r for r in rows if r['uid'] in population]
    uids = [r['uid'] for r in picked]
    families = {r['uid']: family(r.get('room_type')) for r in picked}
    return {'version': 1, 'protocol': PROTOCOL, 'data_manifest_sha256': identity['manifest_sha256'], 'plans': plans,
            'fields_used': ['uid', 'room_type'],
            'algorithm': 'E2 = uids of the amendment-014 plan-e2 requests, E3 = uids of the amendment-015 plan-e3 '
                         'requests; union in eval-data expected_uids order; family = room_families.family(room_type)',
            'counts': {'total': len(uids), 'e2': plans['e2']['rooms'], 'e3': plans['e3']['rooms'],
                       'by_family': dict(sorted(collections.Counter(families.values()).items()))},
            'uids_sha256': L._digest(json.dumps(uids)), 'uids': uids,
            'population': {u: population[u] for u in uids}, 'family': families}


def load_rooms(path, data):
    rooms = h.strict_json(Path(path).read_text())
    uids = rooms.get('uids')
    if (rooms.get('protocol') != PROTOCOL or rooms.get('data_manifest_sha256') != data['manifest_sha256']
            or not isinstance(uids, list) or L._digest(json.dumps(uids)) != rooms.get('uids_sha256')
            or len(set(uids)) != len(uids) or not set(uids) <= set(data['expected_uids'])
            or set(rooms.get('population', ())) != set(uids) or set(rooms.get('family', ())) != set(uids)):
        raise ValueError('rooms file does not bind this evaluation data')
    return rooms


def sizer_request(record):
    """J's request -> the sizer's (same metadata, J's user JSON projected to REQUEST_KEYS, SYSTEM)."""
    value = h.request(infer._schema_record(record))
    user = {k: value[k] for k in REQUEST_KEYS if k in value}
    user['objects'] = [{'id': o['id'], 'category': o['category']} for o in value['objects']]
    if 'fixed' in user:
        user['fixed'] = [{k: f[k] for k in FIXED_KEYS} for f in value['fixed']]
    return {**{k: record[k] for k in ('uid', 'source', 'group')},
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': contract._json(user)}]}


def object_ids(record):
    return [o['id'] for o in h.strict_json(record['messages'][1]['content'])['objects']]


def parse_sizes(text, ids):
    """{'valid', 'error', 'sizes'}; the error is a fixed text (it becomes the repair turn)."""
    def invalid(error):
        return {'valid': False, 'error': error, 'sizes': None}
    try:
        value = h.strict_json(text)
    except (ValueError, TypeError, RecursionError):
        return invalid(NOT_JSON)
    if not isinstance(value, dict):
        return invalid(NOT_OBJECT)
    if set(value) != set(ids):
        return invalid(WRONG_IDS)
    try:
        for size in value.values():
            h._vector(size, positive=True)
    except (ValueError, TypeError):
        return invalid(BAD_SIZE)
    return {'valid': True, 'error': None, 'sizes': {i: [float(v) for v in value[i]] for i in ids}}


def _records(args):
    rows, data = infer.load_eval(args.data, 'joint')
    rooms = load_rooms(args.rooms, data)
    by_uid = h._by_uid(rows)
    return [sizer_request(by_uid[u]) for u in rooms['uids'][:args.limit]], data, rooms


def _identity(sizer, config, data, rooms_path, rooms, n, modules=()):
    return {'arm': 'sizer_020', 'protocol': PROTOCOL, 'sizer': sizer,
            'config': {**config, 'system_prompt': SYSTEM, 'suffix': L.SUFFIX,
                       'repair_rounds': L_REPAIRS if sizer == 'L' else L.REPAIRS,
                       'parser': 'sizers_020.parse_sizes', 'repair_message': 'parser_error_text_only'},
            'data': {'manifest_sha256': data['manifest_sha256'],
                     'joint_requests_sha256': data['files']['joint/requests.jsonl']},
            'rooms': {'sha256': h.sha256(rooms_path), 'uids_sha256': rooms['uids_sha256'], 'rooms': n},
            'code_sha256': {Path(p).name: h.sha256(p) for p in (__file__, L.__file__, infer.__file__, h.__file__,
                                                                 contract.__file__, *modules)}}


def _bind(out, identity):
    """run_sha256; the first writer (any shard) publishes run_receipt.json, every later one must match it."""
    run_hash, out = infer._hash(identity), Path(out)
    (out / 'receipts').mkdir(parents=True, exist_ok=True)
    try:
        infer.write_once(out / 'run_receipt.json', {**identity, 'run_sha256': run_hash})
    except FileExistsError:
        pass
    if h.strict_json((out / 'run_receipt.json').read_text()) != {**identity, 'run_sha256': run_hash}:
        raise ValueError('resume identity mismatch')
    return run_hash


def _receipt_path(out, uid):
    return Path(out) / 'receipts' / (L._digest(uid) + '.json')


# ---- L-size: the API agent's client, ledger and receipts ---------------------------------------------------------

def calls_used(run_root, exclude=None):
    """{run dir: calls} for API runs directly under run_root: max(calls.jsonl lines, sum of receipt calls)."""
    used = {}
    for path in sorted(Path(run_root).glob('*/run_receipt.json')):
        root, receipt = path.parent, h.strict_json(path.read_text())
        if exclude is not None and root.resolve() == Path(exclude).resolve():
            continue
        if receipt.get('arm') == 'llm_agent' or (receipt.get('arm') == 'sizer_020' and receipt.get('sizer') == 'L'):
            recorded = sum(h.strict_json(p.read_text()).get('calls', 0) for p in (root / 'receipts').glob('*.json'))
            used[root.name] = max(len(L._jsonl(root / 'calls.jsonl')), recorded)
    return used


def preflight(run_root, rooms, exclude=None):
    used = calls_used(run_root, exclude)
    total, needed = sum(used.values()), rooms * CALLS_PER_ROOM
    unfinished = [name for name in used if not (Path(run_root) / name / 'MANIFEST.json').exists()]  # fail closed
    return {'cap': CAP, 'used': total, 'by_run': used, 'unfinished': unfinished, 'rooms': rooms,
            'calls_per_room': CALLS_PER_ROOM, 'needed': needed, 'ok': not unfinished and total + needed <= CAP}


def _solve(run, record, attempt):
    uid, ids = record['uid'], object_ids(record)
    messages = [*record['messages'], {'role': 'user', 'content': L.SUFFIX}]
    answers, errors = [], []
    for round_ in range(L_REPAIRS + 1):
        try:
            reply, failed = L._answer(run, uid, attempt, round_, messages)
        except L._Failed as exc:
            run.halt('API_FAILURE', {'uid': uid, 'attempt': attempt, 'round': round_, 'errors': exc.errors})
            return
        errors += failed
        schema = parse_sizes(reply['text'], ids)
        answers.append({'raw': reply['text'], 'schema': schema,
                        **{k: reply[k] for k in ('status', 'incomplete_reason', 'usage', 'latency_seconds', 'tries')}})
        if schema['valid']:
            break
        messages = [*messages, {'role': 'assistant', 'content': reply['text']},
                    {'role': 'user', 'content': schema['error']}]
    run.receipt(record, {'attempt': attempt, 'error': None, 'answers': answers, 'http_errors': errors})


def _work(run, record):
    """llm_agent._work with the sizer's _solve."""
    uid = record['uid']
    try:
        if run.reason:
            return
        if run.failures[uid] >= run.args.max_attempts:
            run.receipt(record, {'attempt': run.attempts.get(uid, 0), 'error': 'API_FAILED', 'answers': [],
                                 'http_errors': []})
            return
        _solve(run, record, run.attempts.get(uid, 0) + 1)
    except L._Stop:
        return
    except BaseException:
        run.halt('INTERNAL_ERROR')
        raise


def run_api(args, post=L.call_api):
    out = Path(args.out)
    if (out / 'MANIFEST.json').exists():
        return h.strict_json((out / 'result.json').read_text())
    records, data, rooms = _records(args)
    check = preflight(args.run_root, len(rooms['uids']), exclude=out)  # the locked rule: all rooms, --limit or not
    if not check['ok']:
        out.mkdir(parents=True, exist_ok=True)
        h.write_json(out / 'status.json', {'status': 'CAP_REFUSED', **check})
        return {'status': 'CAP_REFUSED', **check}
    config = L.load_config(args.env_file)
    if config.model != args.expected_model:
        raise L.ApiError('configured_model_mismatch')
    identity = _identity('L', {'model': args.expected_model, 'reasoning_effort': L.EFFORT,
                               'max_output_tokens': args.max_output_tokens, 'limit': args.limit,
                               'call_cap': {'cap': CAP, 'calls_per_room': CALLS_PER_ROOM}},
                         data, args.rooms, rooms, len(records))
    run_hash = _bind(out, identity)
    settings = SimpleNamespace(max_calls=len(records) * CALLS_PER_ROOM, max_attempts=args.max_attempts,
                               max_output_tokens=args.max_output_tokens, timeout=args.timeout)
    state = L._Run(out, run_hash, config, settings, post, [r['uid'] for r in records])
    pending = [r for r in records if not state.path(r['uid']).exists()]
    with ThreadPoolExecutor(args.concurrency) as pool:
        for _ in pool.map(lambda record: _work(state, record), pending):
            pass
    missing = sum(not state.path(r['uid']).exists() for r in records)
    if missing:
        status = {'status': state.reason or 'INCOMPLETE', 'expected_count': len(records),
                  'receipt_count': len(records) - missing, 'calls_ledger': state.count}
        h.write_json(out / 'status.json', status)
        return status
    return finalize(out, args.rooms)


# ---- O2-size: llm_open's amendment-007 O2 engine ----------------------------------------------------------------

def _open_chunk(O, llm, params, tok, spec, records, log):
    """llm_open._chunk with parse_sizes: all rounds for one chunk -> {uid: receipt body}."""
    by_uid = {r['uid']: r for r in records}
    ids = {uid: object_ids(r) for uid, r in by_uid.items()}
    conversations = {uid: [*r['messages'], {'role': 'user', 'content': L.SUFFIX}] for uid, r in by_uid.items()}
    answers, stopped, active = {uid: [] for uid in by_uid}, {}, list(by_uid)
    for round_ in range(L.REPAIRS + 1):
        jobs = []
        for uid in active:
            precount = O._prompt_tokens(tok, conversations[uid], spec['template'])
            cap = min(O.output_cap(by_uid[uid], spec), O.MAX_MODEL_LEN - precount)
            if cap < 1:
                stopped[uid] = 'INPUT_TOO_LONG'
            else:
                jobs.append((uid, precount, cap))
        if not jobs:
            break
        start = time.monotonic()
        outputs = llm.chat([conversations[uid] for uid, _, _ in jobs],
                           [params(max_tokens=cap, **O.SAMPLING) for _, _, cap in jobs],
                           use_tqdm=False, chat_template_kwargs=spec['template'] or None)
        wall = time.monotonic() - start
        if len(outputs) != len(jobs):
            raise ValueError('generation batch cardinality mismatch')
        log({'event': 'call', 'round': round_, 'requests': len(jobs), 'wall_seconds': wall})
        active = []
        for (uid, precount, cap), output in zip(jobs, outputs):
            completion = output.outputs[0]
            eos, capped = completion.finish_reason == 'stop', completion.finish_reason == 'length'
            reasoning, text = O.split_reasoning(completion.text, spec['split'], spec['thinking'], eos)
            prompt, generated = len(output.prompt_token_ids), len(completion.token_ids)
            answer_tokens = len(tok.encode(text, add_special_tokens=False)) if text else 0
            answer = {'raw': text, 'schema': parse_sizes(text, ids[uid]), 'reasoning': reasoning,
                      'finish_reason': completion.finish_reason, 'status': 'incomplete' if capped else 'completed',
                      'incomplete_reason': 'max_output_tokens' if capped else None, 'reached_token_cap': capped,
                      'stopped_on_eos': eos, 'max_tokens': cap, 'prompt_tokens_precount': precount,
                      'answer_tokens': answer_tokens,
                      'usage': {'input_tokens': prompt, 'output_tokens': generated, 'total_tokens': prompt + generated,
                                'reasoning_tokens': max(0, generated - answer_tokens) if reasoning else 0},
                      'latency_seconds': wall / len(jobs)}
            answers[uid].append(answer)
            if not answer['schema']['valid'] and round_ < L.REPAIRS:
                conversations[uid] = [*conversations[uid], {'role': 'assistant', 'content': text},
                                      {'role': 'user', 'content': answer['schema']['error']}]
                active.append(uid)
    return {uid: {'attempt': 1, 'error': None if values else stopped[uid], 'answers': values, 'http_errors': [],
                  'repair_stopped': stopped.get(uid) if values else None, 'rounds': len(values), 'calls': len(values),
                  'valid_first': bool(values) and values[0]['schema']['valid'],
                  'valid_final': bool(values) and values[-1]['schema']['valid']}
            for uid, values in answers.items()}


def run_open(args, load=None):
    from fastfill.comparison import llm_open as O
    started = time.monotonic()
    spec = O.SYSTEMS['O2']
    records, data, rooms = _records(args)
    model = (Path(args.model_root) / spec['model']).resolve()
    files = {p.name: h.sha256(p) for p in sorted(model.iterdir()) if p.is_file() and not p.name.endswith(O.WEIGHTS)}
    if 'config.json' not in files:
        raise ValueError('model directory lacks config.json')
    config = {'system': 'O2', 'model': spec['name'], 'reasoning_effort': spec['effort'], 'model_files_sha256': files,
              'chat_template_kwargs': spec['template'], 'thinking': spec['thinking'],
              'engine': {'model': str(model), 'tensor_parallel_size': spec['tp'], 'max_model_len': O.MAX_MODEL_LEN,
                         'seed': O.SEED, **spec['engine']},
              'sampling': O.SAMPLING, 'cap': {'base': O.CAP_BASE, 'per_object': O.CAP_PER_OBJECT,
                                              'thinking_allowance': O.THINKING_ALLOWANCE if spec['thinking'] else 0},
              'reasoning_split': spec['split'], 'limit': args.limit, 'versions': O._versions()}
    out = Path(args.out)
    run_hash = _bind(out, _identity('O2', config, data, args.rooms, rooms, len(records), (O.__file__,)))
    if (out / 'MANIFEST.json').exists():
        return h.strict_json((out / 'result.json').read_text())
    shard = records[args.shard_index::args.shard_count]
    pending = sorted((r for r in shard if not _receipt_path(out, r['uid']).exists()),
                     key=lambda r: (-O.output_cap(r, spec), -len(r['messages'][1]['content']), r['uid']))
    invocation = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.getpid()}:{time.time_ns()}"
    ledger = out / f'invocations-s{args.shard_index:02d}-of-{args.shard_count:02d}.jsonl'

    def log(entry):
        O._append(ledger, {'invocation': invocation, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                           'elapsed_seconds': time.monotonic() - started, **entry})
    log({'event': 'start', 'pending': len(pending)})
    if pending:
        llm, params = (load or O.load_vllm)(config['engine'])
        tok = llm.get_tokenizer()
        log({'event': 'loaded'})
        for start in range(0, len(pending), args.chunk):
            chunk = pending[start:start + args.chunk]
            bodies = _open_chunk(O, llm, params, tok, spec, chunk, log)
            for record in chunk:
                infer.write_once(_receipt_path(out, record['uid']), {
                    **{k: record[k] for k in ('uid', 'source', 'group')}, 'run_sha256': run_hash,
                    **bodies[record['uid']]})
    log({'event': 'done'})
    return {'status': 'SHARD_COMPLETE', 'shard': [args.shard_index, args.shard_count], 'rooms': len(shard),
            'generated_rooms': len(pending)}


# ---- finalize ---------------------------------------------------------------------------------------------------

def _cost(out, receipt):
    if receipt['sizer'] == 'L':
        return {'calls_ledger': len(L._jsonl(out / 'calls.jsonl'))}
    spans = {}
    for path in sorted(out.glob('invocations-*.jsonl')):
        for event in L._jsonl(path):
            spans[event['invocation']] = max(spans.get(event['invocation'], 0), event['elapsed_seconds'])
    return {'invocations': len(spans), 'gpu_hours': sum(spans.values())
            * receipt['config']['engine']['tensor_parallel_size'] / 3600,
            'gpu_hours_scope': 'process wall time x GPUs per invocation, model load included'}


def finalize(out, rooms_path):
    """estimates.jsonl, result.json and MANIFEST.json once every room has a receipt; else status INCOMPLETE."""
    out = Path(out)
    if (out / 'MANIFEST.json').exists():
        return h.strict_json((out / 'result.json').read_text())
    receipt = h.strict_json((out / 'run_receipt.json').read_text())
    if receipt['rooms']['sha256'] != h.sha256(rooms_path):
        raise ValueError('rooms file differs from the run\'s')
    uids = h.strict_json(Path(rooms_path).read_text())['uids'][:receipt['config']['limit']]
    receipts = {}
    for uid in uids:
        path = _receipt_path(out, uid)
        if not path.exists():
            status = {'status': 'INCOMPLETE', 'expected_count': len(uids), 'receipt_count': len(receipts)}
            h.write_json(out / 'status.json', status)
            return status
        value = h.strict_json(path.read_text())
        if value['uid'] != uid or value['run_sha256'] != receipt['run_sha256']:
            raise ValueError('receipt identity mismatch')
        receipts[uid] = value
    final = {u: (r['answers'][-1]['schema'] if r['answers'] else None) for u, r in receipts.items()}
    estimates = [{'uid': u, 'estimated_sizes': s['sizes']} for u, s in final.items() if s and s['valid']]
    answers = [a for r in receipts.values() for a in r['answers']]
    tokens = {k: sum(a['usage'].get(k, 0) for a in answers)
              for k in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens')}
    result = {'status': 'SIZES_COMPLETE', 'protocol': PROTOCOL, 'sizer': receipt['sizer'],
              'run_sha256': receipt['run_sha256'], 'rooms': len(uids),
              'valid_first': sum(r['valid_first'] for r in receipts.values()), 'valid_final': len(estimates),
              'failed': {u: r['error'] or (final[u] or {}).get('error') or 'NO_ANSWER'
                         for u, r in receipts.items() if not (final[u] and final[u]['valid'])},
              'rounds_histogram': {str(k): v for k, v in sorted(collections.Counter(
                  r['rounds'] for r in receipts.values()).items())},
              'answers': len(answers), 'reached_token_cap': sum(a['incomplete_reason'] == 'max_output_tokens'
                                                                for a in answers),
              'tokens': tokens, **_cost(out, receipt), 'quality_claim': False}
    if receipt['config'].get('thinking') and not tokens['reasoning_tokens']:
        result['mode_check'] = 'FAIL: thinking system produced no reasoning (mode not applied)'
    (out / 'estimates.jsonl').write_text(''.join(json.dumps(e, allow_nan=False) + '\n' for e in estimates))
    h.write_json(out / 'result.json', result)
    h.write_json(out / 'status.json', result)
    infer.write_once(out / 'MANIFEST.json', {'files': h.artifact_manifest(out), 'expected_uids': uids})
    return result


# ---- size quality report (table A) ------------------------------------------------------------------------------

def _view(room, rooms, sizes):
    from fastfill.comparison import geometry as g
    from fastfill.comparison import report_v2 as r2
    uid = room['uid']
    view = {'uid': uid, 'source': room['source'], 'group': room['group'], 'family': rooms['family'][uid],
            'population': rooms['population'][uid], 'complete': sizes is not None}
    if sizes is None:
        return {**view, 'size_log_mae': None, 'size_iou': None, 'bias_fp': None}
    pred = [{'size': sizes[o['id']]} for o in room['objects']]
    iou, mae = g.dimensions(room['objects'], pred)
    return {**view, 'size_log_mae': mae, 'size_iou': iou, 'bias_fp': r2.size_bias(room['objects'], pred)[0]}


def _sizer(root, rooms_sha):
    """(estimates {uid: sizes} | None, status line, result | None) of one sizer output directory."""
    from fastfill.comparison import evaluate_pair as e
    root = Path(root)
    if not root.exists():
        return None, '未提交（无输出目录）', None
    if not (root / 'MANIFEST.json').exists():
        status = h.strict_json((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        if status.get('status') == 'CAP_REFUSED':
            why = (f"未完成的 API 运行 {', '.join(status['unfinished'])} 仍可能继续调用，从严拒绝" if status.get('unfinished')
                   else f"已用 {status['used']:,} + {status['rooms']:,} 间 × {status['calls_per_room']} = "
                        f"{status['used'] + status['needed']:,} > {status['cap']:,}")
            return None, f'未运行：API 上限预检拒绝（{why}）', None
        raise ValueError(f"{root}: sizer run not complete ({status.get('status')}); finish or finalize it first")
    manifest = e._verified(root)
    receipt = h.strict_json((root / 'run_receipt.json').read_text())
    result = h.strict_json((root / 'result.json').read_text())
    if (result.get('status') != 'SIZES_COMPLETE' or receipt['rooms']['sha256'] != rooms_sha
            or receipt['config']['limit'] is not None or 'estimates.jsonl' not in manifest['files']):
        raise ValueError(f'{root}: not a complete full-room sizer run on these rooms')
    rows = [h.strict_json(line) for line in (root / 'estimates.jsonl').read_text().splitlines()]
    return {r['uid']: r['estimated_sizes'] for r in rows}, '完成', result


def _l_joint(root, data_sha, uids):
    """{uid: sizes | None} of gpt-6.1-sol's J-input run (llm_harness rows) on the rooms it covers."""
    from fastfill.comparison import evaluate_pair as e
    root = Path(root)
    manifest = e._verified(root)
    summary = h.strict_json((root / 'summary.json').read_text())
    if (summary.get('status') != 'LLM_COMPARISON_COMPLETE' or summary.get('data_manifest_sha256') != data_sha
            or summary.get('system') != 'L' or summary['llm']['config'].get('request_arm', 'joint') != 'joint'):
        raise ValueError(f'{root}: not a complete J-input evaluation of L on this data')
    keep = set(uids)
    return {r['uid']: ({o['id']: o['size'] for o in r['raw_layout']} if r['complete'] else None)
            for r in h._rows(root, 'llm_harness-geometry.jsonl', manifest['files']['llm_harness-geometry.jsonl'])
            if r['uid'] in keep}


def _scopes(views):
    from fastfill.comparison.room_families import FAMILIES
    named = {'all': views, 'E2': [v for v in views if v['population'] == 'e2'],
             'E3': [v for v in views if v['population'] == 'e3']}
    return {**named, **{f: [v for v in views if v['family'] == f] for f in FAMILIES}}


def _interval(first, second, fields, repeats):
    from fastfill.comparison import report_v2 as r2
    design = {'rooms': len(first), **r2._design(first)}
    if not first:
        return {**design, 'fields': {}}
    result = r2.contrasts(first, second, fields, repeats, SEED)
    if design['components'] < r2.NO_CI:
        result = {f: {**v, 'ci95': None, 'p': None} for f, v in result.items()}
    return {**design, 'fields': result}


def level(views, repeats):
    """Each field's level with a house-cluster CI (a contrast against zero, as report_v2.own_bias)."""
    rate = _interval(views, [{**v, 'complete': False} for v in views], FIELDS[:1], repeats)
    scalars = _interval(views, [{**v, 'complete': True, **dict.fromkeys(FIELDS[1:], 0.)} for v in views],
                        FIELDS[1:], repeats)
    return {**rate, 'fields': {**rate['fields'], **scalars['fields']}}


def build(views, repeats):
    """views: {source: {uid: view}} -> levels and paired contrasts per scope."""
    levels = {scope: {} for scope in _scopes([])}
    for name, by_uid in views.items():
        for scope, rows in _scopes(list(by_uid.values())).items():
            if rows:
                levels[scope][name] = level(rows, repeats)
    contrasts = {}
    for a, b in PAIRS:
        if a in views and b in views:
            common = [u for u in views[a] if u in views[b]]
            table = {}
            for scope, rows in _scopes([views[a][u] for u in common]).items():
                if rows:
                    table[scope] = _interval(rows, [views[b][v['uid']] for v in rows], FIELDS, repeats)
            contrasts[f'{a} − {b}'] = table
    return {'levels': {k: v for k, v in levels.items() if v}, 'contrasts': contrasts}


def _f(x, signed=False):
    return '—' if x is None else (f'{x:+.3f}' if signed else f'{x:.3f}')


def _cell(item, signed=False):
    if not item or item['estimate'] is None:
        return '—'
    ci = item['ci95']
    return _f(item['estimate'], signed) + (f' [{_f(ci[0], signed)}, {_f(ci[1], signed)}]' if ci else '')


NAMES = {'complete': '完整率', 'size_log_mae': '尺寸 log MAE', 'size_iou': '中心对齐 IoU', 'bias_fp': '足迹偏差 B'}


def report_md(result):
    from fastfill.comparison import report_v2 as r2
    p, counts = result['provenance'], result['rooms']['counts']
    lines = ['# 修订 020 第一阶段：尺寸估计质量（表 A，EXPLORATORY）', '',
             f"房间：E2（修订 014 的六个论文分层，{counts['e2']:,} 间）∪ E3（修订 015 的 9 个非论文房型族，"
             f"{counts['e3']:,} 间），共 {counts['total']:,} 间；房型族 = room_families.family。", '',
             '尺寸来源：GT（真值，参照）；prior（训练集先验 = P 的输入）；L-joint（gpt-6.1-sol 在修订 016 J 输入运行中自己给出的尺寸，'
             'llm_harness 行，只有队列房间）；L-size（gpt-6.1-sol，只估尺寸）；O2-size（Qwen3.8-27B，修订 007 引擎设置，只估尺寸）。'
             '尺寸估计器的请求 = J 的请求去掉一切摆放指令（房型、边界、层高、固定物体、物体 id/类别），尺寸定义沿用 J 的原文：'
             f"“{SIZE_DEFINITION}”；没有“放得下”或“缩小”的指令。一次回答 + 解析修复（L-size 最多 {L_REPAIRS} 轮，O2-size 最多 {L.REPAIRS} 轮；附录 A）；缺任何物体即该房间失败。", '',
             '指标：完整率（全部请求为分母）；尺寸 log MAE（每个物体三轴 |log(预测/GT)| 的平均，再按房间平均）；中心对齐 IoU'
             '（两个以原点为中心的盒子的 IoU）；足迹偏差 B（房间内物体 log(预测 xy 面积 / GT xy 面积) 的平均，负 = 偏小）。'
             '后三者只在完整房间上平均。', '',
             f"区间：房屋连通组自举（report_v2.contrasts = geometry.paired_intervals 的设计与抽样，{result['repeats']:,} 次，"
             f"seed {SEED}）；水平值的区间是对 0 的对比；差值为前者减后者，标量只在两者都完整的房间上配对。"
             f"房屋组 k < {r2.NO_CI} 不给区间；k < {r2.FEW_CLUSTERS} 标记 few clusters，房间 < {r2.SMALL_N} 标记 small n。"
             '各族之间不做多重比较校正；全部为描述性（EXPLORATORY）。', '',
             '## 尺寸估计器状态', '', '| 来源 | 状态 | 房间 | 首答有效 | 最终有效 | 失败 | 达到输出上限 | 成本 |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for name in ('L-size', 'O2-size'):
        s = p['sizers'][name]
        r = s['result'] or {}
        cost = (f"{r['calls_ledger']:,} 次 API 调用" if 'calls_ledger' in r else
                f"{r['gpu_hours']:.2f} GPU-h" if 'gpu_hours' in r else '—')
        lines.append(f"| {name} | {s['status']}{'；' + r['mode_check'] if r.get('mode_check') else ''} | "
                     f"{r.get('rooms', '—')} | {r.get('valid_first', '—')} | {r.get('valid_final', '—')} | "
                     f"{len(r['failed']) if r else '—'} | {r.get('reached_token_cap', '—')} | {cost} |")
    cap = p['api_cap_at_report']
    lines += ['', f"API 调用上限（修订 002）：{cap['cap']:,}；报告时已用 {cap['used']:,}（{cap['by_run']}）。", '',
              '## 表 A：尺寸来源 × 范围', '',
              '| 范围 | 来源 | 房间 | 房屋组 k | ' + ' | '.join(NAMES[f] for f in FIELDS) + ' | 标记 |',
              '|---|---|---:|---:|' + '---:|' * len(FIELDS) + '---|']
    for scope, table in result['levels'].items():
        for name, v in table.items():
            lines.append(f"| {scope} | {name} | {v['rooms']} | {v['components']} | "
                         + ' | '.join(_cell(v['fields'].get(f), f == 'bias_fp') for f in FIELDS)
                         + f" | {v['flags'] or ''} |")
    lines += ['', '## 配对差值（前者 − 后者，EXPLORATORY）', '',
              'L-size − L-joint 只在队列房间上（L-joint 只有这些房间）。', '']
    for name, table in result['contrasts'].items():
        lines += [f'### {name}', '', '| 范围 | 房间 | k | ' + ' | '.join('Δ' + NAMES[f] for f in FIELDS) + ' | 标记 |',
                  '|---|---:|---:|' + '---:|' * len(FIELDS) + '---|']
        lines += [f"| {scope} | {v['rooms']} | {v['components']} | "
                  + ' | '.join(_cell(v['fields'].get(f), True) for f in FIELDS) + f" | {v['flags'] or ''} |"
                  for scope, v in table.items()] + ['']
    return '\n'.join(lines) + '\n'


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['empty'])
        writer.writeheader()
        writer.writerows(rows)


def _flat(kind, name, scope, v):
    return [{'table': kind, 'name': name, 'scope': scope, 'rooms': v['rooms'], 'components': v['components'],
             'flags': v['flags'], 'field': f, 'estimate': x['estimate'],
             'ci_low': x['ci95'][0] if x['ci95'] else None, 'ci_high': x['ci95'][1] if x['ci95'] else None,
             'n': x['n'], 'warnings': ';'.join(x['warnings'])} for f, x in v['fields'].items()]


def report(data, rooms_path, run_root, sizer_dirs, l_joint, out, repeats=2000):
    from fastfill.comparison.corpus import _publish
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    _, ident = infer.load_eval(data, 'joint')
    rooms = load_rooms(rooms_path, ident)
    data_sha = h.sha256(Path(data) / 'MANIFEST.json')
    labels = h._by_uid(h._rows(Path(data), 'rooms.jsonl', ident['files']['rooms.jsonl']))
    prior = h._by_uid(h._rows(Path(data), 'size-estimates.jsonl', ident['files']['size-estimates.jsonl']))
    uids, rooms_sha = rooms['uids'], h.sha256(rooms_path)
    sources = {'GT': {u: {o['id']: o['size'] for o in labels[u]['objects']} for u in uids},
               'prior': {u: prior[u]['estimated_sizes'] for u in uids},
               'L-joint': _l_joint(l_joint, data_sha, uids)}
    sizers, inputs = {}, {'data_MANIFEST': data_sha, 'rooms': rooms_sha,
                          'L-joint_MANIFEST': h.sha256(Path(l_joint) / 'MANIFEST.json')}
    for name, root in sizer_dirs.items():
        estimates, status, result = _sizer(root, rooms_sha)
        sizers[name] = {'dir': str(root), 'status': status, 'result': result}
        if estimates is not None:
            sources[name] = {u: estimates.get(u) for u in uids}
            inputs[name + '_MANIFEST'] = h.sha256(Path(root) / 'MANIFEST.json')
    if 'O2-size' not in sources:
        raise ValueError(f"O2-size has no complete output ({sizers['O2-size']['status']}); the report needs it")
    views = {name: {u: _view(labels[u], rooms, sizes) for u, sizes in by_uid.items()}
             for name, by_uid in sources.items()}
    result = {'status': 'SIZES_020_REPORT_COMPLETE', 'label': 'EXPLORATORY', 'protocol': PROTOCOL,
              **build({k: views[k] for k in SOURCES if k in views}, repeats),
              'rooms': {k: rooms[k] for k in ('counts', 'uids_sha256', 'plans')}, 'repeats': repeats, 'seed': SEED,
              'provenance': {'inputs': inputs, 'sizers': sizers, 'api_cap_at_report': preflight(run_root, 0),
                             'l_joint_rooms': len(sources['L-joint'])},
              'code_sha256': {Path(p).name: h.sha256(p) for p in (__file__,)}}
    flat = [r for scope, t in result['levels'].items() for name, v in t.items() for r in _flat('level', name, scope, v)]
    flat_c = [r for name, t in result['contrasts'].items() for scope, v in t.items()
              for r in _flat('contrast', name, scope, v)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'sizes.json', result)
        (dest / 'report.md').write_text(report_md(result))
        _csv(dest / 'levels.csv', flat)
        _csv(dest / 'contrasts.csv', flat_c)
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return result


# ---- CLI --------------------------------------------------------------------------------------------------------

def main(argv=None, post=L.call_api, load=None):
    def positive(text):
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError('must be positive')
        return value
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    rooms = sub.add_parser('rooms')
    for flag in ('data', 'e2', 'e3', 'out'):
        rooms.add_argument('--' + flag, type=Path, required=True)
    cap = sub.add_parser('cap')
    cap.add_argument('--run-root', type=Path, required=True)
    cap.add_argument('--rooms', type=int, required=True)
    api, open_, fin, rep = (sub.add_parser(n) for n in ('run-api', 'run-open', 'finalize', 'report'))
    for p in (api, open_, rep):
        p.add_argument('--data', type=Path, required=True)
    for p in (api, open_, fin, rep):
        p.add_argument('--rooms', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
    for p in (api, open_):
        p.add_argument('--limit', type=positive)
    for p in (api, rep):
        p.add_argument('--run-root', type=Path, required=True)
    api.add_argument('--env-file', type=Path, default=Path(os.environ.get('FASTFILL_API_ENV', '~/.fastfill_api.env')))
    api.add_argument('--expected-model', default=L.EXPECTED_MODEL)
    for flag, default in (('concurrency', 4), ('max-output-tokens', 65536), ('timeout', 900), ('max-attempts', 2)):
        api.add_argument('--' + flag, type=positive, default=default)
    open_.add_argument('--model-root', type=Path, required=True)
    open_.add_argument('--shard-index', type=int, required=True)
    open_.add_argument('--shard-count', type=positive, required=True)
    open_.add_argument('--chunk', type=positive, default=256)
    rep.add_argument('--l', type=Path, required=True)
    rep.add_argument('--o2', type=Path, required=True)
    rep.add_argument('--l-joint', type=Path, required=True)
    rep.add_argument('--repeats', type=positive, default=2000)
    args = parser.parse_args(argv)
    if args.command == 'rooms':
        value = build_rooms(args.data, args.e2, args.e3)
        if args.out.exists():
            if h.strict_json(args.out.read_text()) != value:
                raise ValueError(f'{args.out} differs from its deterministic rebuild')
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            infer.write_once(args.out, value)
        print(json.dumps(value['counts']))
        return 0
    if args.command == 'cap':
        check = preflight(args.run_root, args.rooms)
        print(json.dumps(check))
        return 0 if check['ok'] else 1
    if args.command == 'run-open' and not 0 <= args.shard_index < args.shard_count:
        parser.error('--shard-index must be in [0, --shard-count)')
    if args.command == 'run-api':
        try:
            result = run_api(args, post)
        except L.ApiError as exc:
            print('sizers_020_error: ' + exc.code, file=sys.stderr)
            return 1
    elif args.command == 'run-open':
        result = run_open(args, load)
    elif args.command == 'finalize':
        result = finalize(args.out, args.rooms)
    else:
        result = report(args.data, args.rooms, args.run_root, {'L-size': args.l, 'O2-size': args.o2}, args.l_joint,
                        args.out, args.repeats)
        print(json.dumps({'status': result['status'], 'sizers': {k: v['status'] for k, v in
                                                                 result['provenance']['sizers'].items()}},
                         ensure_ascii=False))
        return 0
    print(json.dumps({k: v for k, v in result.items() if k != 'failed'}))
    status = result['status']
    return 0 if status in ('SIZES_COMPLETE', 'SHARD_COMPLETE') else 2 if status == 'API_FAILURE' else 1


if __name__ == '__main__':
    sys.exit(main())
