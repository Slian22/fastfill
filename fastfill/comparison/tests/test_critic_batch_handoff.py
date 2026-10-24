"""The documented batch answers directory must feed the actual next-round API."""
import json

import pytest

from fastfill.comparison import critic_025 as critic, critic_batch_025 as batch
from fastfill.comparison.tests import test_critic_client_025 as fixtures
from fastfill.comparison.tests.test_critic_025 import bedroom, state


def test_batch_answers_feed_next_round_and_reject_stale_identity(tmp_path, monkeypatch):
    previous = tmp_path / 'round0'
    critic._publish_state(previous, [state('u1', bedroom()), state('u2', bedroom())],
                          {'round': 0, 'b_manifest_sha256': None}, 1)
    responses = iter([json.dumps({'scores': {'functionality': 6, 'layout': 6}, 'edits': []}), 'bad JSON'])
    monkeypatch.setattr(fixtures.judge, 'open_response',
                        lambda request, timeout: fixtures.Response(fixtures.sse(next(responses))))
    calls = []
    fixtures.in_process(monkeypatch, calls)
    output = tmp_path / 'batch'
    summary = batch.evaluate(previous, output, fixtures.env_file(tmp_path), workers=1)
    assert summary['counts']['completed'] == summary['counts']['failed'] == 1
    result = critic.round_(previous, output / 'answers', tmp_path / 'round1')
    assert result['status'] == 'A025_ROUND_COMPLETE'
    rows = [json.loads(line) for line in (tmp_path / 'round1/rows.jsonl').read_text().splitlines()]
    assert [r['history'][-1]['stop'] for r in rows] == ['no_valid_edits', 'call_failed']
    assert [r['layout'] for r in rows] == [bedroom(), bedroom()]
    stale = tmp_path / 'stale'
    stale.mkdir()
    record = json.loads(next((output / 'answers').glob('*.json')).read_text())
    (stale / 'answer.json').write_text(json.dumps({**record, 'content_sha256': '0' * 64}))
    with pytest.raises(ValueError, match="not made on this round"):
        critic.round_(previous, stale, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()
