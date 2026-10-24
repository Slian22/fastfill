"""CPU only; reuse the repository's fake SSE transport, never calls any API."""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from pytest import MonkeyPatch

from fastfill.comparison import critic_025 as c, critic_batch_025 as b
from fastfill.comparison import dpo_stages as d, judge_agreement_023 as a
from fastfill.comparison.tests import test_critic_client_025 as fixtures


def critic_handoff(root):
    bundle = fixtures.critic_bundle(root, 1)
    manifest_path = bundle / 'critic-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['cases'][0]['id'] = c.case_id('audit-u1', 1)
    manifest_path.write_text(json.dumps(manifest))
    prev = root / 'prev'
    encode = lambda value: (json.dumps(value) + '\n').encode()
    d._publish_dir(prev, {
        'rows.jsonl': encode({'uid': 'audit-u1', 'active': True}),
        'summary.json': encode({'round': 0}),
        'critic-manifest.json': encode(manifest),
        'critic-mapping.json': encode({c.case_id('audit-u1', 1): 'audit-u1'}),
    })
    out = root / 'batch'
    with MonkeyPatch.context() as patch:
        seen, calls = [], []
        fixtures.transport(patch, fixtures.judge, seen, {'room-0': json.dumps(fixtures.GOOD)})
        fixtures.in_process(patch, calls)
        with redirect_stdout(io.StringIO()):
            summary = b.evaluate(bundle, out, fixtures.env_file(root), workers=1, timeout=30)
    answers = c.load_answers(out / 'answers')
    assert summary['counts']['completed'] == 1, summary
    error = None
    try:
        c.round_(prev, out / 'answers', root / 'next')
    except ValueError as exc:
        error = str(exc)
    assert error and 'not made on this round' in error, error
    return {'fake_api_completed': summary['counts']['completed'],
            'answer_keys': sorted(next(iter(answers.values()))),
            'round_error': error,
            'next_round_published': (root / 'next').exists()}


def optional_overall(root):
    rows = [{'id': f'case-{i}', 'uid': f'room-{i // 2}', 'arm': ('P', 'L')[i % 2], 'status': 'completed',
             'scores': {'functionality_score': 6+i, 'layout_score': 5+i, 'aesthetics_score': 7+i}}
            for i in range(4)]
    paths = [root / 'judge-a.jsonl', root / 'judge-b.jsonl']
    for path in paths:
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    error = None
    try:
        a.compare(*paths, ['judge-a', 'judge-b'])
    except ValueError as exc:
        error = str(exc)
    assert error == 'not enough values to unpack (expected 2, got 0)', error
    return {'completed_per_judge': 4, 'per_case_scores_valid': True,
            'optional_overall_omitted': True, 'compare_error': error}


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='fastfill-root-audit-') as temp:
        root = Path(temp)
        results = {'critic_batch_round_contract': critic_handoff(root),
                   'judge_optional_overall': optional_overall(root)}
        print(json.dumps(results, indent=2))
