#!/usr/bin/env python3
"""Amendment 025 critic batch: bounded one-case critic_client_025 subprocesses over a critic_025 bundle.

Run only on the credential-owning server as a CPU job (submit_025.sh):
  python -m fastfill.comparison.critic_batch_025 --bundle BUNDLE --output NEW --env-file ENV \
      [--cap 500] [--limit N] [--workers 4] [--timeout 600]
BUNDLE is a critic_025 state directory: its critic-manifest.json uses server_api_judge's manifest schema,
expected_model gpt-6.1-sol and exactly two PNG views per case (top, oblique). The first --limit cases (default
all) are selected; selecting more than --cap (at most 500) is refused before anything is written. No retries and
no replacement: each case gets one reservation and at most one API call, and a failed call leaves that room
unedited. Only the child client reads the env file; no ambient OPENAI_* variables are forwarded. NEW must not
exist; the client code must not change during the run (checked before every launch).

NEW holds artifacts/ (frozen prompts and views), snapshot.json, cases/<sha256(id)>/ (reservation.json,
manifest.json and the client's immutable results/ attempt and result files), records.jsonl in manifest order
({id, content_sha256, status, critic | error, returned_model, usage, launched, attempted}; attempted = an API
call may have been made), answers/<sha256(id)>.json per case ({id, status, error, returned_model, raw_final_text},
raw_final_text null unless the call completed), summary.json (counts, error codes, token totals) and MANIFEST.json
(sha256 of every other file under NEW), written last. NEW/answers is the critic_025 `round --answers`: a failed call
(even one whose stored result keeps a complete text) reaches the round as call_failed and the room stays unedited.
Passing NEW itself is refused by the round (the result files duplicate the answer ids).
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from fastfill.comparison import critic_client_025 as critic, judge_batch
from fastfill.comparison import server_api_judge as judge

CAP, MANIFEST = 500, 'critic-manifest.json'
CODE_ROOT = Path(__file__).resolve().parents[2]
CODE = ('fastfill/comparison/critic_client_025.py', 'fastfill/comparison/server_api_judge.py')
USAGE = ('input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens')
ERROR = re.compile(r'[a-z0-9_]{1,64}\Z')


class BatchError(ValueError):
    """Fixed, non-sensitive orchestration error."""


def _code():
    return {name: judge.sha256((CODE_ROOT / name).read_bytes()) for name in CODE}


def _result(case, results, returncode):
    stem = judge.sha256(case['id'].encode())
    identity = {'schema_version': 1, 'id': case['id'], 'expected_model': critic.MODEL,
                'content_sha256': judge_batch.content_hash(case)}
    attempt = judge.read_record(results / (stem + '.attempt.json'), identity)
    record = judge.read_record(results / (stem + '.result.json'), identity)
    if (attempt.get('status') != 'attempted' or not judge.valid_hash(attempt.get('request_sha256'))
            or record.get('request_sha256') != attempt['request_sha256']):
        raise BatchError('invalid_attempt_identity')
    meta = {'returned_model': record.get('returned_model'), 'usage': record.get('usage')}
    if record.get('status') == 'failed':
        error = record.get('error')
        return {'status': 'failed', 'error': error if isinstance(error, str) and ERROR.match(error)
                else 'client_evaluation_failed', **meta}
    if (returncode != 0 or record.get('status') != 'completed' or 'error' in record
            or record.get('returned_model') != critic.MODEL):
        raise BatchError('invalid_completion')
    parsed = critic.parse_critic(record.get('raw_final_text'))
    if critic.parse_critic(judge.canonical(record.get('scores'))) != parsed:
        raise BatchError('critic_text_mismatch')
    return {'status': 'completed', 'critic': parsed, 'raw_final_text': record['raw_final_text'], **meta}


def _run_case(case, output, env_file, timeout, code):
    directory = output / 'cases' / judge.sha256(case['id'].encode())
    directory.mkdir(parents=True)
    judge_batch._write(directory / 'reservation.json', {'id': case['id'], 'attempt_limit': 1,
                                                        'content_sha256': judge_batch.content_hash(case)})
    manifest = directory / 'manifest.json'
    judge_batch._write(manifest, {'schema_version': 1, 'expected_model': critic.MODEL, 'cases': [case]})
    launched = False
    try:
        if _code() != code:
            raise BatchError('code_changed')
        command = [sys.executable, '-B', '-m', 'fastfill.comparison.critic_client_025', '--manifest', str(manifest),
                   '--results-dir', str(directory / 'results'), '--env-file', str(env_file),
                   '--limit', '1', '--timeout', str(timeout)]
        environment = {k: os.environ[k] for k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL') if k in os.environ}
        launched = True
        process = subprocess.run(command, cwd=CODE_ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                                 timeout=timeout + 60)
        result = _result(case, directory / 'results', process.returncode)
    except subprocess.TimeoutExpired:
        result = {'status': 'failed', 'error': 'client_timeout'}
    except (OSError, ValueError, judge.JudgeError):
        result = {'status': 'failed', 'error': 'client_or_artifact_failure'}
    stem = judge.sha256(case['id'].encode())
    attempted = (directory / 'results' / (stem + '.attempt.json')).exists()
    judge_batch._write(output / 'answers' / (stem + '.json'), {
        'id': case['id'], 'status': result['status'], 'error': result.get('error'),
        'returned_model': result.get('returned_model'), 'raw_final_text': result.pop('raw_final_text', None)})
    return {**result, 'launched': launched, 'attempted': attempted}


def evaluate(bundle, output, env_file, cap=CAP, limit=None, workers=4, timeout=600):
    if (type(cap) is not int or not 1 <= cap <= CAP or (limit is not None and (type(limit) is not int or limit < 1))
            or type(workers) is not int or not 1 <= workers <= 16
            or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600):
        raise BatchError('invalid_bounds')
    output = Path(output).absolute()
    if os.path.lexists(output):
        raise BatchError('output_already_exists')
    root = Path(bundle).resolve()
    manifest = judge.load_manifest(root / MANIFEST)
    if manifest['expected_model'] != critic.MODEL or any(len(c['images']) != 2 for c in manifest['cases']):
        raise BatchError('invalid_critic_bundle')
    cases = manifest['cases'][:limit]
    if len(cases) > cap:
        raise BatchError('cap_exceeded')
    if output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve()):
        raise BatchError('overlapping_output_bundle')
    code = _code()
    output.mkdir(mode=0o700, parents=True)
    (output / 'artifacts').mkdir()
    (output / 'answers').mkdir()
    frozen = [judge_batch._freeze_case(judge, root, c, output / 'artifacts') for c in cases]
    judge_batch._write(output / 'snapshot.json', {
        'schema_version': 1, 'amendment': '025', 'model': critic.MODEL, 'reasoning_effort': critic.EFFORT,
        'instructions_sha256': judge.sha256(critic.INSTRUCTIONS.encode()), 'cap': cap, 'limit': limit,
        'selected': len(frozen), 'retry': False, 'replacement': False, 'code_sha256': code,
        'bundle_manifest_sha256': judge.sha256((root / MANIFEST).read_bytes()), 'cases': frozen})
    env_file = Path(env_file).expanduser()
    records = []
    pool = ThreadPoolExecutor(workers)
    try:
        with (output / 'records.jsonl').open('xb') as stream:
            for case, result in zip(frozen, pool.map(lambda c: _run_case(c, output, env_file, timeout, code), frozen)):
                record = {'id': case['id'], 'content_sha256': judge_batch.content_hash(case), **result}
                records.append(record)
                stream.write(judge.canonical(record) + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        pool.shutdown(cancel_futures=True)  # An unexpected error launches no further cases.
    done = [r['critic'] for r in records if r['status'] == 'completed']
    summary = {'schema_version': 1, 'amendment': '025', 'model': critic.MODEL, 'reasoning_effort': critic.EFFORT,
               'status': 'completed' if len(done) == len(records) else 'partial',
               'counts': {'selected': len(records), 'launched': sum(r['launched'] for r in records),
                          'attempted': sum(r['attempted'] for r in records), 'completed': len(done),
                          'failed': len(records) - len(done), 'with_edits': sum(bool(c['edits']) for c in done),
                          'edits': sum(len(c['edits']) for c in done)},
               'errors': dict(sorted(Counter(r['error'] for r in records if 'error' in r).items())),
               'usage': {k: sum((r.get('usage') or {}).get(k, 0) for r in records) for k in USAGE},
               'retry': False, 'replacement': False}
    judge_batch._write(output / 'summary.json', summary)
    files = {p.relative_to(output).as_posix(): judge.sha256(p.read_bytes()) for p in sorted(output.rglob('*'))
             if p.is_file()}
    judge_batch._write(output / 'MANIFEST.json', {'files': files})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--env-file', type=Path, default=Path('~/.fastfill_api.env'))
    parser.add_argument('--cap', type=int, default=CAP)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args(argv)
    try:
        summary = evaluate(args.bundle, args.output, args.env_file, args.cap, args.limit, args.workers, args.timeout)
    except (BatchError, judge_batch.BatchError, judge.JudgeError) as exc:
        print('critic_batch_error: ' + str(exc), file=sys.stderr)
        return 1
    except Exception:
        print('critic_batch_error: unexpected_error', file=sys.stderr)
        return 1
    print(json.dumps({'status': summary['status'], 'counts': summary['counts']}))
    return int(bool(summary['counts']['failed']))


if __name__ == '__main__':
    sys.exit(main())
