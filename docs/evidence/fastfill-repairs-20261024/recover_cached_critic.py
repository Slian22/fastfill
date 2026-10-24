"""One-off offline recovery of the four verified a025 answer bundles.

No network client is invoked. All source manifests, raw result identities and
answer text are verified before creating a separate destination. Historical
directories are never written. Run from the reviewed repair snapshot with
--historical RUN/unknownsize-20261013 --output NEW/critic-answers.
"""
import argparse
import hashlib
import json
from pathlib import Path

from fastfill.comparison import critic_025 as consumer
from fastfill.comparison import critic_batch_025 as batch
from fastfill.comparison import judge_batch
from fastfill.comparison import train_pair_helpers as h


def recover(historical, output):
    historical, output = Path(historical).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    resolved = output.resolve()
    if resolved.is_relative_to(historical) or historical.is_relative_to(resolved):
        raise ValueError('recovery output must be separate from historical inputs')
    pending, receipts = {}, {}
    for name in ('round-0', 'round-1', 'test-round-0', 'test-round-1'):
        state = historical / 'a025' / name
        source = state.with_name(name + '.critic')
        h.verify_artifacts(source, h.strict_json((source / 'MANIFEST.json').read_text())['files'])
        rows, summary, state_sha = consumer._state(state)
        round_number = summary['round'] + 1
        active = {consumer.case_id(r['uid'], round_number) for r in rows if r['active']}
        cases = h.strict_json((state / 'critic-manifest.json').read_text())['cases']
        content = {c['id']: judge_batch.content_hash(c) for c in cases}
        frozen = h.strict_json((source / 'snapshot.json').read_text())['cases']
        if {c['id']: judge_batch.content_hash(c) for c in frozen} != content or set(content) != active:
            raise ValueError('cached cases differ from active round identity')
        answers = consumer.load_answers(source / 'answers')
        if set(answers) != active:
            raise ValueError('cached answer inventory differs from active round')
        recovered = {}
        for case in frozen:
            stem = hashlib.sha256(case['id'].encode()).hexdigest()
            result = batch._result(case, source / 'cases' / stem / 'results', 0)
            previous = answers[case['id']]
            # This recovery deliberately supports only the observed all-complete
            # batches: failed/ambiguous calls require separate investigation.
            if result['status'] != 'completed' or previous.get('status') != 'completed':
                raise ValueError('only verified completed cached answers may be recovered')
            if any(previous.get(k) != result.get(k) for k in ('raw_final_text', 'returned_model', 'error')):
                raise ValueError('answer differs from the verified raw client result')
            recovered[stem + '.json'] = {**previous, 'expected_model': consumer.CRITIC_MODEL,
                                        'content_sha256': content[case['id']]}
        pending[name] = recovered
        receipts[name] = {'answers': len(recovered), 'state_manifest_sha256': state_sha,
                          'batch_manifest_sha256': h.sha256(source / 'MANIFEST.json'),
                          'round': round_number, 'status': 'IDENTITY_VERIFIED'}
    output.mkdir(parents=True, mode=0o700)
    for name, answers in pending.items():
        target = output / name / 'answers'
        target.mkdir(parents=True)
        for filename, answer in answers.items():
            with (target / filename).open('x') as stream:
                stream.write(json.dumps(answer, ensure_ascii=False, allow_nan=False) + '\n')
        loaded = consumer.load_answers(target)
        if {v['id']: v for v in answers.values()} != loaded:
            raise AssertionError('published answers differ from verified recovery')
    report = {'status': 'CACHED_ANSWERS_RECOVERED', 'historical': str(historical),
              'output': str(output), 'api_calls': 0, 'new_geometry_rounds': 0,
              'answers': sum(v['answers'] for v in receipts.values()), 'batches': receipts,
              'script_sha256': h.sha256(__file__),
              'code_sha256': {Path(p).name: h.sha256(p) for p in (batch.__file__, consumer.__file__)}}
    with (output / 'summary.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    h.write_json(output / 'MANIFEST.json', {'files': {
        p.relative_to(output).as_posix(): h.sha256(p)
        for p in sorted(output.rglob('*')) if p.is_file()}})
    h.verify_artifacts(output, h.strict_json((output / 'MANIFEST.json').read_text())['files'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(args.historical, args.output), indent=2))
