"""LLM-agent arm L (amendment 002, section 3): the exact J request, answered by an API model.

  python -m fastfill.comparison.llm_agent cohort --data EVAL --out COHORT.json
  python -m fastfill.comparison.llm_agent run --data EVAL --cohort COHORT.json --out RUN \
      [--env-file ~/.fastfill_api.env] [--concurrency 4] [--max-calls 4800] [--limit N] \
      [--max-attempts 2] [--require-no-api-failed]

Cohort: every paper six-strata test room (eval_data.PAPER_STRATA on
scene.norm_room_type, as evaluate_pair) plus, per source, up to 30 remaining test
UIDs ranked by sha256('llm-cohort-v1:'+uid). Only request uid/source/room_type is
read, never rooms.jsonl or any target. Written once in expected_uids order; run
and the evaluator recompute it and refuse any difference.

Input per UID: the J system+user messages from joint/requests.jsonl, unchanged,
then one fixed user message SUFFIX (identical for every request). No size tables,
priors, statistics or per-request extras. The model is OPENAI_MODEL from the
private 0600 env file and must equal --expected-model; reasoning effort medium.
Answers are parsed by infer.parse_prediction(raw, request, 'joint'), exactly as J.
An invalid answer gets up to 2 repair rounds: the answer is appended as the
assistant turn, followed by a user turn holding only the parser error text.
llm_raw = first answer; llm_harness = final answer.

Every HTTP request, including transient retries, is appended and fsynced to
calls.jsonl BEFORE it is sent; --max-calls bounds that ledger across resumes.
Transient failures (timeout, transport, 408/409/429/5xx, broken stream) are
retried twice per call. A call that still fails ends the UID's attempt without a
receipt, logs fixed error codes to api-errors.jsonl, stops scheduling (fail
closed) and exits 2. Rerunning the same command resumes: UIDs with receipts are
never called again. Only request-specific rejections (HTTP 400/413/422) count
per UID; after --max-attempts of them the UID gets a final API_FAILED receipt
without a call; outages never finalize a UID (--require-no-api-failed, the smoke
gate, exits 1 if any UID ended API_FAILED). Completion publishes
llm_raw/records.jsonl and llm_harness/records.jsonl (inference records shape),
result.json and MANIFEST.json.
The API key is never logged or printed; stored receipts are redacted.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shlex
import stat
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from fastfill.comparison import infer
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.eval_data import PAPER_STRATA
from fastfill.scene import norm_room_type

VARIANTS = ('llm_raw', 'llm_harness')
COHORT_KEY = 'llm-cohort-v1:'
PER_SOURCE = 30
EXPECTED_MODEL = 'gpt-6.1-sol'
EFFORT = 'medium'
REPAIRS = 2
RETRIES = 2
BACKOFF_SECONDS = 10
SUFFIX = ('Return only the JSON object specified above, with nothing before or after it: '
          'no prose, no markdown and no code fences.')
TRANSIENT_HTTP = {408, 409, 429, 500, 502, 503, 504}
REJECTED = {'http_status_400', 'http_status_413', 'http_status_422'}  # Request-specific, counted per UID.
USER_AGENT = 'fastfill-evaluation/1.0'  # The API proxy refused urllib's default agent (403).
MAX_STREAM_BYTES = 16 * 1024 * 1024
CONFIG_KEYS = ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL')
NOT_GENERATED = {'valid': False, 'error': 'NOT_GENERATED', 'placements': None}


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _stratum(row):
    key = (row['source'], norm_room_type(row.get('room_type')))
    return '/'.join(key) if key in PAPER_STRATA else None


def build_cohort(data):
    """Paper strata plus <=30 hash-ranked other UIDs per source; uid/source/room_type only."""
    rows, data_identity = infer.load_eval(data, 'joint')
    meta = [{k: r.get(k) for k in ('uid', 'source', 'room_type')} for r in rows]
    strata = {m['uid']: _stratum(m) for m in meta if _stratum(m)}
    sampled = {}
    for source in sorted({m['source'] for m in meta}):
        pool = sorted((m['uid'] for m in meta if m['source'] == source and m['uid'] not in strata),
                      key=lambda uid: (_digest(COHORT_KEY + uid), uid))
        sampled[source] = pool[:PER_SOURCE]
    chosen = set(strata).union(*sampled.values())
    uids = [m['uid'] for m in meta if m['uid'] in chosen]
    return {'version': 1, 'data_manifest_sha256': data_identity['manifest_sha256'],
            'fields_used': ['uid', 'source', 'room_type'],
            'algorithm': "eval_data.PAPER_STRATA on scene.norm_room_type, plus per source up to 30 remaining "
                         "UIDs ranked by sha256('llm-cohort-v1:'+uid) with uid tie-break; expected_uids order",
            'counts': {'total': len(uids), 'paper_strata_total': len(strata),
                       'paper_strata': {s + '/' + k: sum(v == s + '/' + k for v in strata.values())
                                        for s, k in PAPER_STRATA},
                       'sampled_by_source': {s: len(v) for s, v in sampled.items()},
                       'by_source': dict(sorted(Counter(m['source'] for m in meta if m['uid'] in chosen).items()))},
            'uids_sha256': _digest(json.dumps(uids)), 'uids': uids}


class ApiError(Exception):
    """Carries only a fixed, non-sensitive code."""
    def __init__(self, code, transient=False):
        super().__init__(code)
        self.code, self.transient = code, transient


@dataclass(frozen=True)
class Config:
    api_key: str = field(repr=False)
    base_url: str
    model: str


def load_config(path):
    """Same rules as server_api_judge.load_config: owner-only 0600 file, no shell, no ambient env."""
    try:
        descriptor = os.open(Path(path).expanduser(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, encoding='utf-8') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.getuid()):
                raise ApiError('config_permissions')
            text = stream.read(65537)
    except (OSError, UnicodeError, ValueError):
        raise ApiError('config_read_error') from None
    values = {}
    for line in text.splitlines():
        line = line.strip()
        line = line[7:].lstrip() if line.startswith('export ') else line
        name, separator, value = line.partition('=')
        name = name.strip()
        if line.startswith('#') or name not in CONFIG_KEYS:
            continue
        try:
            parts = shlex.split(value, comments=False, posix=True)
        except ValueError:
            raise ApiError('invalid_config') from None
        if (not separator or name in values or not parts or not parts[0]
                or (len(parts) > 1 and not parts[1].startswith('#'))):
            raise ApiError('invalid_config')
        values[name] = parts[0]
    if (len(text) > 65536 or set(values) != set(CONFIG_KEYS)
            or any(ord(c) < 33 or ord(c) > 126 for v in values.values() for c in v)):
        raise ApiError('invalid_config')
    try:
        url = urllib.parse.urlsplit(values['OPENAI_BASE_URL'])
        if (url.scheme != 'https' or not url.hostname or url.username is not None or url.password is not None
                or url.query or url.fragment or (url.port is not None and not 1 <= url.port <= 65535)):
            raise ValueError()
    except ValueError:
        raise ApiError('invalid_base_url') from None
    return Config(values['OPENAI_API_KEY'], values['OPENAI_BASE_URL'].rstrip('/'), values['OPENAI_MODEL'])


def redact(value, secret):
    if isinstance(value, str):
        for variant in sorted({secret, json.dumps(secret)[1:-1]}, key=len, reverse=True):
            value = value.replace(variant, '[REDACTED]')
        return re.sub(r'(?i)(\bbearer\s+|\bOPENAI_API_KEY\s*=\s*)[A-Za-z0-9._~+/=-]+', r'\1[REDACTED]', value)
    if isinstance(value, dict):
        return {k: redact(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ApiError('redirect_refused')


def _open(request, timeout):
    """Exactly one POST: no environment proxies, no redirects."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout)


def request_body(model, messages, max_output_tokens):
    return {'model': model, 'input': messages, 'reasoning': {'effort': EFFORT},
            'max_output_tokens': max_output_tokens, 'store': False, 'stream': True}


def _output_text(response):
    parts = [p.get('text') for item in response.get('output') or [] if isinstance(item, dict)
             and item.get('type') == 'message' for p in item.get('content') or []
             if isinstance(p, dict) and p.get('type') == 'output_text']
    return ''.join(p for p in parts if isinstance(p, str)) if parts else None


def _dict(value):
    return value if isinstance(value, dict) else {}


def read_sse(stream, model, deadline):
    """Bounded SSE up to response.completed/incomplete; incomplete text is still the answer."""
    total, lines, deltas = 0, [], ''
    while True:
        if time.monotonic() > deadline:
            raise ApiError('timeout', True)
        raw = stream.readline(MAX_STREAM_BYTES + 1 - total)
        total += len(raw)
        if total > MAX_STREAM_BYTES:
            raise ApiError('stream_too_large')
        if not raw:
            raise ApiError('missing_completed_event', True)
        try:
            line = raw.decode('utf-8').rstrip('\r\n')
        except UnicodeError:
            raise ApiError('invalid_sse_utf8', True) from None
        if line.startswith('data:'):
            lines.append(line[5:].removeprefix(' '))
            continue
        if line or not lines:
            continue
        data, lines = '\n'.join(lines), []
        if data == '[DONE]':
            raise ApiError('missing_completed_event', True)
        try:
            event = h.strict_json(data)
        except (ValueError, RecursionError):
            raise ApiError('invalid_sse_json', True) from None
        kind = _dict(event).get('type')
        if kind == 'response.output_text.delta' and isinstance(event.get('delta'), str):
            deltas += event['delta']
        elif kind in ('error', 'response.failed'):
            raise ApiError('remote_response_failed', True)
        elif kind in ('response.completed', 'response.incomplete'):
            response = event.get('response')
            if not isinstance(response, dict):
                raise ApiError('invalid_completed_event', True)
            if response.get('model') != model:
                raise ApiError('model_mismatch')
            usage = _dict(response.get('usage'))
            counts = {k: usage[k] for k in ('input_tokens', 'output_tokens', 'total_tokens') if type(usage.get(k)) is int}
            reasoning = _dict(usage.get('output_tokens_details')).get('reasoning_tokens')
            if type(reasoning) is int:
                counts['reasoning_tokens'] = reasoning
            reason = _dict(response.get('incomplete_details')).get('reason')
            text = _output_text(response)
            return {'text': deltas if text is None else text, 'status': kind.removeprefix('response.'),
                    'incomplete_reason': reason if isinstance(reason, str) else None, 'usage': counts}


def call_api(config, messages, max_output_tokens, timeout):
    """One streamed POST to {base}/responses; raises ApiError with a fixed code."""
    body = json.dumps(request_body(config.model, messages, max_output_tokens), allow_nan=False).encode('utf-8')
    request = urllib.request.Request(config.base_url + '/responses', data=body, method='POST', headers={
        'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json',
        'Accept': 'text/event-stream', 'User-Agent': USER_AGENT})
    try:
        with _open(request, timeout) as response:
            if response.status != 200:
                raise ApiError('unexpected_http_status', True)
            if response.headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'text/event-stream':
                raise ApiError('unexpected_content_type')
            return read_sse(response, config.model, time.monotonic() + timeout)
    except ApiError:
        raise
    except urllib.error.HTTPError as exc:
        code = exc.code if type(exc.code) is int else 0
        exc.close()
        raise ApiError(f'http_status_{code}', code in TRANSIENT_HTTP) from None
    except urllib.error.URLError as exc:
        raise ApiError('timeout' if isinstance(exc.reason, TimeoutError) else 'transport_error', True) from None
    except TimeoutError:
        raise ApiError('timeout', True) from None
    except (OSError, http.client.HTTPException, ValueError):
        raise ApiError('transport_error', True) from None


def _jsonl(path):
    return [h.strict_json(line) for line in path.read_text().splitlines()] if path.exists() else []


class _Stop(Exception):
    """Scheduling stopped: call cap, API failure or another worker's error."""


class _Failed(Exception):
    def __init__(self, errors):
        super().__init__('api_failure')
        self.errors = errors


class _Run:
    def __init__(self, out, run_hash, config, args, post, expected):
        self.out, self.run_hash, self.config, self.args, self.post = out, run_hash, config, args, post
        self.lock, self.reason, self.expected = threading.Lock(), None, expected
        self.done = sum(self.path(uid).exists() for uid in expected)
        calls = _jsonl(out / 'calls.jsonl')
        attempts = defaultdict(set)
        for entry in calls:
            attempts[entry['uid']].add(entry['attempt'])
        self.count, self.attempts = len(calls), {uid: len(v) for uid, v in attempts.items()}
        self.failures = Counter(e['uid'] for e in _jsonl(out / 'api-errors.jsonl') if e['errors'][-1] in REJECTED)

    def path(self, uid):
        return self.out / 'receipts' / (_digest(uid) + '.json')

    def _append(self, name, entry):
        with (self.out / name).open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def reserve(self, entry):
        with self.lock:
            if self.reason:
                raise _Stop()
            if self.count >= self.args.max_calls:
                self.reason = 'CALL_CAP_REACHED'
                raise _Stop()
            self._append('calls.jsonl', entry)
            self.count += 1

    def halt(self, reason, entry=None):
        with self.lock:
            if entry is not None:
                self._append('api-errors.jsonl', entry)
            self.reason = self.reason or reason

    def receipt(self, record, body):
        answers = body['answers']
        value = {**{k: record[k] for k in ('uid', 'source', 'group')}, 'run_sha256': self.run_hash, **body,
                 'rounds': len(answers), 'calls': sum(a['tries'] for a in answers),
                 'valid_first': bool(answers) and answers[0]['schema']['valid'],
                 'valid_final': bool(answers) and answers[-1]['schema']['valid']}
        infer.write_once(self.path(record['uid']), redact(value, self.config.api_key))
        with self.lock:
            self.done += 1
            h.write_json(self.out / 'status.json', {'status': 'RUNNING', 'expected_count': len(self.expected),
                                                    'receipt_count': self.done, 'calls_ledger': self.count})


def _answer(run, uid, attempt, round_, messages):
    errors = []
    for retry in range(RETRIES + 1):
        run.reserve({'uid': uid, 'attempt': attempt, 'round': round_, 'try': retry})
        start = time.monotonic()
        try:
            reply = run.post(run.config, messages, run.args.max_output_tokens, run.args.timeout)
            return {**reply, 'latency_seconds': time.monotonic() - start, 'tries': retry + 1}, errors
        except ApiError as exc:
            errors.append(exc.code)
            if not exc.transient or retry == RETRIES:
                raise _Failed(errors) from None
            time.sleep(BACKOFF_SECONDS * (retry + 1))


def _solve(run, record, attempt):
    uid = record['uid']
    messages = [*record['messages'], {'role': 'user', 'content': SUFFIX}]
    answers, errors = [], []
    for round_ in range(REPAIRS + 1):
        try:
            reply, failed = _answer(run, uid, attempt, round_, messages)
        except _Failed as exc:
            run.halt('API_FAILURE', {'uid': uid, 'attempt': attempt, 'round': round_, 'errors': exc.errors})
            return
        errors += failed
        schema = infer.parse_prediction(reply['text'], record, 'joint')
        answers.append({'raw': reply['text'], 'schema': schema,
                        **{k: reply[k] for k in ('status', 'incomplete_reason', 'usage', 'latency_seconds', 'tries')}})
        if schema['valid']:
            break
        messages = [*messages, {'role': 'assistant', 'content': reply['text']},
                    {'role': 'user', 'content': schema['error']}]
    run.receipt(record, {'attempt': attempt, 'error': None, 'answers': answers, 'http_errors': errors})


def _work(run, record):
    uid = record['uid']
    try:
        if run.reason:
            return
        if run.failures[uid] >= run.args.max_attempts:
            run.receipt(record, {'attempt': run.attempts.get(uid, 0), 'error': 'API_FAILED',
                                 'answers': [], 'http_errors': []})
            return
        _solve(run, record, run.attempts.get(uid, 0) + 1)
    except _Stop:
        return
    except BaseException:
        run.halt('INTERNAL_ERROR')
        raise


def _finalize(out, records, receipts, run_hash, calls):
    rows = {v: [] for v in VARIANTS}
    for record in records:
        receipt = receipts[record['uid']]
        for variant, answer in zip(VARIANTS, (receipt['answers'][:1], receipt['answers'][-1:])):
            answer = answer[0] if answer else None
            rows[variant].append({**{k: record[k] for k in ('uid', 'source', 'group')}, 'arm': variant,
                                  'raw': answer['raw'] if answer else None, 'error': receipt['error'],
                                  'generated': answer is not None,
                                  'schema': answer['schema'] if answer else NOT_GENERATED,
                                  'rounds': min(receipt['rounds'], 1) if variant == 'llm_raw' else receipt['rounds'],
                                  'run_sha256': run_hash})
    for variant, values in rows.items():
        path = out / variant / 'records.jsonl'
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in values))
        temporary.replace(path)
    answers = [a for r in receipts.values() for a in r['answers']]
    latency = sorted(a['latency_seconds'] for a in answers)
    result = {'status': 'LLM_COMPLETE', 'run_sha256': run_hash, 'rooms': len(records),
              'api_failed_rooms': sum(r['error'] is not None for r in receipts.values()),
              'valid_first': sum(r['valid_first'] for r in receipts.values()),
              'valid_final': sum(r['valid_final'] for r in receipts.values()),
              'rounds_histogram': {str(k): v for k, v in sorted(Counter(r['rounds'] for r in receipts.values()).items())},
              'answers': len(answers), 'calls_ledger': calls,
              'transient_errors': sum(len(r['http_errors']) for r in receipts.values()),
              'reached_token_cap': sum(a['incomplete_reason'] == 'max_output_tokens' for a in answers),
              'tokens': {k: sum(a['usage'].get(k, 0) for a in answers)
                         for k in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens')},
              'latency_seconds': {'mean': statistics.mean(latency), 'p50': latency[len(latency) // 2],
                                  'p95': latency[int(.95 * (len(latency) - 1))], 'max': latency[-1]} if latency else None,
              'quality_claim': False}
    h.write_json(out / 'status.json', result)
    if not (out / 'result.json').exists():
        infer.write_once(out / 'result.json', result)
    infer.write_once(out / 'MANIFEST.json', {'files': h.artifact_manifest(out), 'expected_uids': [r['uid'] for r in records]})
    return result


def run(args, post=call_api):
    rows, data = infer.load_eval(args.data, 'joint')
    cohort = h.strict_json(Path(args.cohort).read_text())
    if cohort != build_cohort(args.data):
        raise ValueError('cohort file differs from its deterministic rebuild')
    uids = cohort['uids'][:args.limit]
    by_uid = h._by_uid(rows)
    records = [by_uid[uid] for uid in uids]
    config = load_config(args.env_file)
    if config.model != args.expected_model:
        raise ApiError('configured_model_mismatch')
    identity = {'arm': 'llm_agent', 'variants': list(VARIANTS),
                'config': {'model': args.expected_model, 'reasoning_effort': EFFORT,
                           'max_output_tokens': args.max_output_tokens, 'repair_rounds': REPAIRS,
                           'suffix': SUFFIX, 'input': 'joint_system_user_verbatim_then_suffix_user',
                           'parser': 'infer.parse_prediction(raw, joint_request, joint)',
                           'repair_message': 'parser_error_text_only', 'limit': args.limit},
                'data': {'manifest_sha256': data['manifest_sha256'],
                         'joint_requests_sha256': data['files']['joint/requests.jsonl']},
                'cohort': {'sha256': h.sha256(args.cohort), 'uids_sha256': cohort['uids_sha256'], 'rooms': len(uids)},
                'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(infer.__file__), Path(h.__file__))}}
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
    state = _Run(out, run_hash, config, args, post, uids)
    pending = [r for r in records if not state.path(r['uid']).exists()]
    with ThreadPoolExecutor(args.concurrency) as pool:
        for _ in pool.map(lambda record: _work(state, record), pending):
            pass
    receipts = {}
    for uid in uids:
        if state.path(uid).exists():
            receipt = h.strict_json(state.path(uid).read_text())
            if receipt['uid'] != uid or receipt['run_sha256'] != run_hash:
                raise ValueError('receipt identity mismatch')
            receipts[uid] = receipt
    if len(receipts) < len(uids):
        status = {'status': state.reason or 'INCOMPLETE', 'expected_count': len(uids),
                  'receipt_count': len(receipts), 'calls_ledger': state.count}
        h.write_json(out / 'status.json', status)
        return status
    return _finalize(out, records, receipts, run_hash, state.count)


def main(argv=None):
    def positive(text):
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError('must be positive')
        return value
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    cohort = commands.add_parser('cohort')
    run_parser = commands.add_parser('run')
    for sub, flags in ((cohort, ('data', 'out')), (run_parser, ('data', 'cohort', 'out'))):
        for flag in flags:
            sub.add_argument('--' + flag, type=Path, required=True)
    run_parser.add_argument('--env-file', type=Path,
                            default=Path(os.environ.get('FASTFILL_API_ENV', '~/.fastfill_api.env')))
    run_parser.add_argument('--expected-model', default=EXPECTED_MODEL)
    for flag, default in (('concurrency', 4), ('max-calls', 4800), ('max-output-tokens', 65536),
                          ('timeout', 900), ('max-attempts', 2)):
        run_parser.add_argument('--' + flag, type=positive, default=default)
    run_parser.add_argument('--limit', type=positive)
    run_parser.add_argument('--require-no-api-failed', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'cohort':
        value = build_cohort(args.data)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        infer.write_once(args.out, value)
        print(json.dumps({'rooms': value['counts']['total'], 'uids_sha256': value['uids_sha256']}))
        return 0
    try:
        result = run(args)
    except ApiError as exc:
        print('llm_agent_error: ' + exc.code, file=sys.stderr)
        return 1
    print(json.dumps({k: result.get(k) for k in ('status', 'rooms', 'expected_count', 'receipt_count',
                                                 'calls_ledger', 'api_failed_rooms', 'valid_first', 'valid_final')}))
    if args.require_no_api_failed and result.get('api_failed_rooms'):
        return 1
    return 0 if result['status'] == 'LLM_COMPLETE' else 2 if result['status'] == 'API_FAILURE' else 1


if __name__ == '__main__':
    sys.exit(main())
