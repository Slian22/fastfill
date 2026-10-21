"""Opt-in API estimates for objects not covered by the train prior.

Resolve all sizes before any placer runs. Reuse the amendment-020 size-only
prompt/parser and the API client's retry-counted ledger, with a separate cap.
This is a runtime feature, not an amendment to frozen research measurements.
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from types import SimpleNamespace

from fastfill.comparison import contract, infer, size_prior
from fastfill.comparison import llm_agent as L
from fastfill.comparison import sizers_020 as S
from fastfill.comparison import train_pair_helpers as h

POLICIES = ('prior', 'api')
ROUNDS = 2
API = {'max_output_tokens': 65536, 'timeout': 900}
INCOMPLETE = 'invalid size output: API response was not completed'


def rounded(size):
    return [max(.01, round(v, 2)) for v in size]


def _source(prior, category, room_type):
    category, room_type = size_prior._norm_cat(category), size_prior._norm_room_type(room_type) or ''
    pair = prior['category_room_type'].get(category, {}).get(room_type)
    category_bucket = prior['category'].get(category)
    for source, bucket in (('prior_category_room_type', pair), ('prior_category', category_bucket)):
        if bucket is not None:
            if not isinstance(bucket, dict) or type(bucket.get('count')) is not int or bucket['count'] < 0:
                raise ValueError('invalid prior bucket count')
            if bucket['count'] >= prior['min_count']:
                return source
    return None


def _validated_user(record):
    infer.validate_request(record, 'joint')
    if record['messages'][0]['content'] != contract._SYSTEM['joint']:
        raise ValueError('expected an eval-data joint request')
    user = h.strict_json(record['messages'][1]['content'])
    if size_prior._norm_room_type(user.get('room_type')) != size_prior._norm_room_type(record.get('room_type')):
        raise ValueError('room_type metadata differs from the model input')
    for fixed in user.get('fixed', []):
        if type(fixed['yaw']) is not int or not 0 <= fixed['yaw'] < 360:
            raise ValueError('fixed yaw must be integer degrees in [0,360)')
    # Validate the full input geometry; dummy movable labels never enter any model request.
    room = {**user, contract._MARKER: contract.PROTOCOL,
            'fixed': [{**f, 'yaw': math.radians(f['yaw'])} for f in user.get('fixed', [])],
            'objects': [{**o, 'size': [1, 1, 1], 'pos': [0, 0, 0], 'yaw': 0} for o in user['objects']]}
    contract._validate_room(room)
    contract._bound_records(room)
    return user


def plan(requests, prior):
    """Validate the whole batch; return known sizes, sources and size-only API requests."""
    sizes, sources, pending = {}, {}, []
    for record, _ in requests:
        user = _validated_user(record)
        uid = record['uid']
        if uid in sizes:
            raise ValueError('duplicate request uid')
        features = {'room_type': record.get('room_type'), 'objects': user['objects']}
        estimates = size_prior.predict_sizes(features, prior) if prior is not None else {}
        levels = {o['id']: _source(prior, o['category'], record.get('room_type')) if prior is not None else None
                  for o in user['objects']}
        known = {oid: rounded(estimates[oid]) for oid, level in levels.items() if level}
        # Validate before calls, including the sizes that remain untouched by the API.
        for size in known.values():
            h._vector(size, positive=True)
        sizes[uid], sources[uid] = known, {oid: level for oid, level in levels.items() if level}
        missing = [o for o in user['objects'] if levels[o['id']] is None]
        if missing:
            request = S.sizer_request(record)
            pending.append({**request, 'messages': [request['messages'][0],
                {'role': 'user', 'content': contract._json({**user, 'objects': missing})}]})
    return {'sizes': sizes, 'sources': sources, 'requests': pending}


def preflight(plan, cap, max_rooms, *, dry_run=False):
    for name, value in (('size_api_cap', cap), ('size_api_max_rooms', max_rooms)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f'{name} must be an integer >= 0')
    rooms = len(plan['requests'])
    if rooms and not dry_run:
        if max_rooms is not None and rooms > max_rooms:
            raise ValueError(f'{rooms} rooms need size estimation > size_api_max_rooms {max_rooms}; no API calls')
        if cap is None or cap < rooms:
            raise ValueError(f'size_api_cap must allow at least one call per room needing sizes ({rooms})')


def _solve(run, record):
    uid, answers, errors = record['uid'], [], []
    messages = [*record['messages'], {'role': 'user', 'content': L.SUFFIX}]
    try:
        for round_ in range(ROUNDS):
            reply, failed = L._answer(run, uid, 1, round_, messages)
            errors += failed
            schema = S.parse_sizes(reply['text'], S.object_ids(record))
            if schema['valid']:
                # A valid raw dimension can round to the exclusive 1000 m bound.
                value = {oid: rounded(size) for oid, size in schema['sizes'].items()}
                schema = S.parse_sizes(contract._json(value), S.object_ids(record))
            if reply['status'] != 'completed' or reply['incomplete_reason'] is not None:
                schema = {'valid': False, 'error': INCOMPLETE, 'sizes': None}
            answers.append({'raw': reply['text'], 'schema': schema, **{k: reply[k] for k in (
                'status', 'incomplete_reason', 'usage', 'latency_seconds', 'tries')}})
            if schema['valid']:
                break
            messages = [*messages, {'role': 'assistant', 'content': reply['text']},
                        {'role': 'user', 'content': schema['error']}]
    except L._Stop:
        pass
    except L._Failed as exc:
        errors += exc.errors
        run.halt('API_FAILURE', {'uid': uid, 'attempt': 1, 'round': len(answers), 'errors': exc.errors})
    except Exception:
        # Unexpected client/response failures must stop spending without logging arbitrary exception text.
        run.halt('INVALID_API_RESPONSE', {'uid': uid, 'errors': ['invalid_api_response']})
    run.receipt(record, {'attempt': 1, 'answers': answers, 'http_errors': errors,
                         'error': None if answers and answers[-1]['schema']['valid'] else
                                  run.reason or 'INVALID_SIZE_RESPONSE'})


def estimate(plan, out, cap, config, post, concurrency=4):
    """Return merged estimates + observed API usage, or persist failure and raise before placement."""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError('size API concurrency must be an integer >= 1')
    out = Path(out)
    records = plan['requests']
    identity = {'protocol': 'runtime-size-api-v1', 'model': L.EXPECTED_MODEL, 'reasoning_effort': L.EFFORT,
                'system_prompt': S.SYSTEM, 'suffix': L.SUFFIX, 'rounds': ROUNDS, 'max_calls': cap,
                'requests_sha256': infer._hash(records), 'parser': 'sizers_020.parse_sizes',
                'code_sha256': {Path(m.__file__).name: h.sha256(m.__file__) for m in (S, L)},
                'sizing_code_sha256': h.sha256(__file__), **API}
    run_hash = infer._hash(identity)
    (out / 'receipts').mkdir(parents=True, exist_ok=False)
    h.write_json(out / 'run_receipt.json', {**identity, 'run_sha256': run_hash})
    args = SimpleNamespace(max_calls=cap, max_attempts=1, arm='size', **API)
    run = L._Run(out, run_hash, config, args, post, [r['uid'] for r in records])
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(lambda record: _solve(run, record), records))
    receipts = {r['uid']: h.strict_json(run.path(r['uid']).read_text()) for r in records}
    failed = [uid for uid, receipt in receipts.items() if not receipt['valid_final']]
    usage = [a['usage'] for receipt in receipts.values() for a in receipt['answers']]
    info = {**identity, 'run_sha256': run_hash, 'rooms': len(records), 'calls_ledger': run.count,
            'stopped': run.reason, 'failed_uids': failed,
            'status': 'SIZE_ESTIMATION_FAILED' if failed else 'SIZE_ESTIMATION_COMPLETE',
            'tokens': {k: sum(u.get(k, 0) for u in usage) for k in (
                'input_tokens', 'output_tokens', 'reasoning_tokens')},
            'usage_scope': 'returned responses only; failed HTTP attempts may have unreported usage'}
    h.write_json(out / 'result.json', info)
    h.write_json(out / 'status.json', {k: info[k] for k in ('status', 'calls_ledger', 'failed_uids', 'stopped')})
    if failed:
        raise RuntimeError(f'size estimation failed for {len(failed)} room(s); see {out / "result.json"}')
    estimates = {uid: {oid: rounded(size) for oid, size in receipt['answers'][-1]['schema']['sizes'].items()}
                 for uid, receipt in receipts.items()}
    sizes = {uid: {**known, **estimates.get(uid, {})} for uid, known in plan['sizes'].items()}
    sources = {uid: {**known, **dict.fromkeys(estimates.get(uid, {}), 'api')}
               for uid, known in plan['sources'].items()}
    return {**plan, 'sizes': sizes, 'sources': sources}, info


def summary(plan, api=None):
    return {'policy': 'api_fallback', 'objects_by_source': dict(collections.Counter(
        source for sources in plan['sources'].values() for source in sources.values())),
        'rooms_needing_api': len(plan['requests']), 'api': api}


def dry_report(plan, prior, out, cap, max_rooms, names):
    """Missing-size dry runs cannot measure layout quality; do not load models or call the API."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pending = {r['uid']: [names[r['uid']].get(oid, oid) for oid in S.object_ids(r)] for r in plan['requests']}
    result = {'status': 'DRY_RUN_SIZE_ESTIMATION_REQUIRED', 'dry_run': True, 'rooms': len(plan['sizes']),
              'prior': prior[1], 'stages': {}, 'would_escalate': None, 'would_estimate_sizes': pending,
              'sizing': summary(plan), 'size_api_cap': cap, 'size_api_max_rooms': max_rooms,
              'size_api_calls_upper_bound': len(pending) * ROUNDS * (1 + L.RETRIES), 'api_calls_total': 0}
    h.write_json(out / 'result.json', result)
    (out / 'report.md').write_text('# FastFill 尺寸估计预检\n\n'
        f'{len(pending)} 个房间需要尺寸 API；未加载模型、未调用 API。尺寸尚未确定，摆放与升级量未测。\n')
    return result
