"""Amendment-025 critic client and batch, fully offline: fake HTTPS transport and in-process subprocesses."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastfill.comparison import critic_batch_025 as batch, critic_client_025 as critic
from fastfill.comparison import server_api_judge as judge
from fastfill.comparison.tests.test_judge_batch import PNG, digest

SECRET = 'sk-test-SECRET-0123456789abcdef'
GOOD = {'scores': {'functionality': 6, 'layout': 5},
        'edits': [{'id': 'obj_0001', 'op': 'move', 'x': 1.2, 'y': -0.4, 'reason': 'clear the door'},
                  {'id': 'obj_0002', 'op': 'rotate', 'yaw_deg': 90, 'reason': 'face the desk'},
                  {'id': 'obj_0003', 'op': 'against_wall', 'reason': 'storage to the wall'},
                  {'id': 'obj_0004', 'op': 'face', 'target': 'obj_0005', 'reason': 'face the TV'}]}

def sse(text, model='gpt-6.1-sol', complete=True):
    usage = {'input_tokens': 900, 'output_tokens': 300, 'total_tokens': 1200,
             'output_tokens_details': {'reasoning_tokens': 250}}
    events = [{'type': 'response.created', 'response': {'model': model, 'status': 'in_progress'}},
              {'type': 'response.output_text.delta', 'delta': text},
              {'type': 'response.completed', 'response': {'model': model, 'status': 'completed', 'usage': usage,
               'output': [{'type': 'message', 'role': 'assistant',
                           'content': [{'type': 'output_text', 'text': text}]}]}}]
    return ''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events[:3 if complete else 2]).encode()


class Response(io.BytesIO):
    status = 200
    headers = {'Content-Type': 'text/event-stream; charset=utf-8'}


def transport(monkeypatch, module, seen, texts):
    """module.open_response replaced: no network. texts maps a case id to the model's final text (or raw SSE bytes)."""
    def fake(request, timeout):
        body = json.loads(request.data)
        seen.append((dict(request.header_items()), body))
        prompt = body['input'][0]['content'][0]['text']
        text = next(t for cid, t in texts.items() if cid in prompt)
        return Response(text if isinstance(text, bytes) else sse(text))
    monkeypatch.setattr(module, 'open_response', fake)


def env_file(tmp_path):
    path = tmp_path / 'api.env'
    path.write_text(f'OPENAI_API_KEY={SECRET}\nOPENAI_BASE_URL=https://api.example.invalid/v1\n'
                    'OPENAI_MODEL=gpt-6.1-sol\n')
    path.chmod(0o600)
    return path


def critic_bundle(tmp_path, n=3):
    root = tmp_path / 'bundle'
    root.mkdir()
    cases = []
    for i in range(n):
        (root / f'room-{i}.txt').write_text(f'Room room-{i}. Objects: obj_0001 bed ...')
        (root / f'room-{i}-top.png').write_bytes(PNG)
        (root / f'room-{i}-oblique.png').write_bytes(PNG)
        cases.append({'id': f'room-{i}', 'prompt_file': f'room-{i}.txt',
                      'prompt_sha256': digest((root / f'room-{i}.txt').read_bytes()),
                      'images': [{'path': f'room-{i}-{v}.png', 'sha256': digest(PNG)} for v in ('top', 'oblique')]})
    (root / 'critic-manifest.json').write_text(json.dumps({'schema_version': 1, 'expected_model': 'gpt-6.1-sol',
                                                           'cases': cases}))
    return root


def files_text(root):
    return [p.read_bytes() for p in Path(root).rglob('*') if p.is_file()]


ANSWERS = [json.dumps(GOOD), '{"scores": {"functionality": 9, "layout": 10}, "edits": []}',
           json.dumps({**GOOD, 'edits': GOOD['edits'] * 2}),                              # extras: over_limit there
           json.dumps({**GOOD, 'edits': [{'op': 'add', 'id': 'obj_0100'}, 'x']}),         # edits rejected alone there
           json.dumps({**GOOD, 'comments': 'x'}), json.dumps({'scores': GOOD['scores']}), '[]', '"text"',
           json.dumps({**GOOD, 'scores': {'functionality': 6, 'layout': 11}}),
           json.dumps({**GOOD, 'scores': {'functionality': 6, 'layout': True}}),
           json.dumps({**GOOD, 'scores': {'functionality': 6, 'layout': 5.0}}),
           json.dumps({**GOOD, 'scores': {'functionality': 6, 'layout': 5, 'aesthetics': 5}}),
           json.dumps({**GOOD, 'edits': {'id': 'obj_0001'}}), json.dumps({**GOOD, 'edits': None}),
           '```json\n' + json.dumps(GOOD) + '\n```', '{"scores": 1, "scores": 2, "edits": []}',
           '{"scores": {"functionality": NaN, "layout": 1}, "edits": []}', 'not json', '']
INF = json.dumps(GOOD).replace('1.2', '1e400')        # parses to inf: unstorable, so the client fails it whole


def outcome(text):
    try:
        critic.parse_critic(text)
        return None
    except judge.JudgeError as exc:
        return str(exc)


def test_parser_is_critic_025_whole_answer_rule():
    """Edits stay opaque; an answer is completed here exactly when critic_025.parse_edits finds no answer error."""
    from fastfill.comparison import critic_025
    ids = {'obj_0001', 'obj_0002', 'obj_0003', 'obj_0004', 'obj_0005'}
    assert critic.parse_critic(json.dumps(GOOD)) == GOOD
    for text in ANSWERS:
        assert outcome(text) == critic_025.parse_edits(text, ids, ids)['error'], text
    assert sum(critic_025.parse_edits(t, ids, ids)['error'] is None for t in ANSWERS) == 4
    # The one documented exception: parse_edits rejects only the inf edit; the client fails the whole answer.
    parsed = critic_025.parse_edits(INF, ids, ids)
    assert parsed['error'] is None and [r['code'] for r in parsed['rejected']] == ['invalid_number']
    assert outcome(INF) == 'invalid_number'


def test_client_payload_record_and_redaction(tmp_path, monkeypatch, capsys):
    root = critic_bundle(tmp_path, 1)
    leaky = {**GOOD, 'edits': [{**GOOD['edits'][0], 'reason': 'key ' + SECRET}, *GOOD['edits'][1:]]}
    seen = []
    transport(monkeypatch, judge, seen, {'room-0': json.dumps(leaky)})
    rc = critic.main(['--manifest', str(root / 'critic-manifest.json'), '--results-dir', str(tmp_path / 'r'),
                      '--env-file', str(env_file(tmp_path)), '--timeout', '30'])
    assert rc == 0 and SECRET not in capsys.readouterr().out
    (headers, body), = seen
    assert headers['Authorization'] == 'Bearer ' + SECRET
    assert body['model'] == 'gpt-6.1-sol' and body['reasoning'] == {'effort': 'medium'}
    assert body['instructions'] == critic.INSTRUCTIONS and body['max_output_tokens'] == 32768
    assert [c['type'] for c in body['input'][0]['content']] == ['input_text', 'input_image', 'input_image']
    result, = [json.loads(p.read_text()) for p in (tmp_path / 'r').glob('*.result.json')]
    assert result['status'] == 'completed' and result['returned_model'] == 'gpt-6.1-sol'
    assert result['usage'] == {'input_tokens': 900, 'output_tokens': 300, 'total_tokens': 1200,
                               'reasoning_tokens': 250}
    assert result['scores']['edits'][1:] == GOOD['edits'][1:]
    assert result['scores']['edits'][0]['reason'] == 'key [REDACTED]'
    assert critic.parse_critic(result['raw_final_text']) == result['scores']
    assert not any(SECRET.encode() in data for data in files_text(tmp_path / 'r'))


def test_client_rejects_wrong_bundle_model(tmp_path, monkeypatch):
    root = critic_bundle(tmp_path, 1)
    manifest = json.loads((root / 'critic-manifest.json').read_text())
    (root / 'critic-manifest.json').write_text(json.dumps({**manifest, 'expected_model': 'gpt-6-astra'}))
    seen = []
    transport(monkeypatch, judge, seen, {'room-0': json.dumps(GOOD)})
    assert critic.main(['--manifest', str(root / 'critic-manifest.json'), '--results-dir', str(tmp_path / 'r'),
                        '--env-file', str(env_file(tmp_path))]) == 1
    assert seen == []


def in_process(monkeypatch, calls):
    """subprocess.run replaced by the client's main in this process (deadline needs a main thread)."""
    monkeypatch.setattr(judge, 'deadline', lambda _seconds: nullcontext())

    def run(command, **kwargs):
        assert command[1:4] == ['-B', '-m', 'fastfill.comparison.critic_client_025']
        assert kwargs['cwd'] == batch.CODE_ROOT and not any('OPENAI' in k for k in kwargs['env'])
        calls.append(command)
        return SimpleNamespace(returncode=critic.main(command[4:]))
    monkeypatch.setattr(batch.subprocess, 'run', run)


def test_batch_offline_end_to_end(tmp_path, monkeypatch):
    root = critic_bundle(tmp_path, 3)
    monkeypatch.setenv('OPENAI_API_KEY', 'ambient-should-not-pass')
    seen, calls = [], []
    transport(monkeypatch, judge, seen, {'room-0': json.dumps(GOOD), 'room-1': '{"scores": {}, "edits": []}',
                                         'room-2': json.dumps(GOOD)})
    in_process(monkeypatch, calls)
    out = tmp_path / 'out'
    summary = batch.evaluate(root, out, env_file(tmp_path), cap=2, limit=2, workers=2, timeout=30)
    assert len(calls) == len(seen) == 2
    rows = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows] == ['room-0', 'room-1']
    assert rows[0]['status'] == 'completed' and rows[0]['critic'] == GOOD and rows[0]['attempted']
    assert rows[1] == {**rows[1], 'status': 'failed', 'error': 'invalid_scores', 'attempted': True}
    assert summary['counts'] == {'selected': 2, 'launched': 2, 'attempted': 2, 'completed': 1, 'failed': 1,
                                 'with_edits': 1, 'edits': 4}
    assert summary['usage'] == {'input_tokens': 1800, 'output_tokens': 600, 'reasoning_tokens': 500,
                                'total_tokens': 2400}
    assert summary['errors'] == {'invalid_scores': 1}
    manifest = json.loads((out / 'MANIFEST.json').read_text())['files']
    assert all(manifest[n] == digest((out / n).read_bytes()) for n in manifest)
    assert set(manifest) == {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()} - {'MANIFEST.json'}
    assert sum(n.endswith('.result.json') for n in manifest) == sum(n.startswith('answers/') for n in manifest) == 2
    snapshot = json.loads((out / 'snapshot.json').read_text())
    assert snapshot['code_sha256'] == batch._code() and snapshot['retry'] is False
    assert not any(SECRET.encode() in data or b'ambient-should' in data for data in files_text(out))
    with pytest.raises(batch.BatchError, match='output_already_exists'):
        batch.evaluate(root, out, env_file(tmp_path), cap=2, limit=2)


@pytest.mark.parametrize('options, error', [({'cap': 2}, 'cap_exceeded'), ({'cap': 501}, 'invalid_bounds'),
                                            ({'cap': 0}, 'invalid_bounds'), ({'limit': 0}, 'invalid_bounds')])
def test_batch_cap_refused_before_any_call(tmp_path, monkeypatch, options, error):
    root = critic_bundle(tmp_path, 3)
    calls = []
    in_process(monkeypatch, calls)
    with pytest.raises(batch.BatchError, match=error):
        batch.evaluate(root, tmp_path / 'out', env_file(tmp_path), **options)
    assert calls == [] and not (tmp_path / 'out').exists()


def test_batch_rejects_non_critic_bundle(tmp_path, monkeypatch):
    root = critic_bundle(tmp_path, 1)
    manifest = json.loads((root / 'critic-manifest.json').read_text())
    manifest['cases'][0]['images'] = manifest['cases'][0]['images'][:1]
    (root / 'critic-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(batch.BatchError, match='invalid_critic_bundle'):
        batch.evaluate(root, tmp_path / 'out', env_file(tmp_path))
    assert not (tmp_path / 'out').exists()


def test_batch_failed_call_leaves_the_room_unedited_in_the_round(tmp_path, monkeypatch):
    """A failed record can keep a complete text (here: no completed event); only OUT/answers reaches the round."""
    from fastfill.comparison import critic_025
    from fastfill.comparison.tests.test_critic_025 import bedroom, state
    root = critic_bundle(tmp_path, 3)
    seen, calls = [], []
    transport(monkeypatch, judge, seen, {'room-0': sse(json.dumps(GOOD), complete=False), 'room-1': INF,
                                         'room-2': json.dumps(GOOD)})
    in_process(monkeypatch, calls)
    out = tmp_path / 'out'
    summary = batch.evaluate(root, out, env_file(tmp_path), cap=3, workers=1, timeout=30)
    assert len(calls) == len(seen) == 3
    assert summary['errors'] == {'invalid_number': 1, 'missing_completed_event': 1}
    stored = {r['id']: r for r in map(json.loads, map(Path.read_text, out.glob('cases/*/results/*.result.json')))}
    assert stored['room-0']['raw_final_text'] == json.dumps(GOOD) and stored['room-0']['status'] == 'failed'
    assert '1e400' in stored['room-1']['raw_final_text']           # the paid answer is kept, not lost
    answers = critic_025.load_answers(out / 'answers')
    assert [(a['status'], a['error'], a['raw_final_text']) for a in map(answers.get, ('room-0', 'room-1', 'room-2'))] \
        == [('failed', 'missing_completed_event', None), ('failed', 'invalid_number', None),
            ('completed', None, json.dumps(GOOD))]
    row = state('u1', bedroom())
    step = critic_025._step((row, answers['room-0'], 1))
    assert step['layout'] == row['layout'] and not step['active']
    last = step['history'][-1]
    assert [last[k] for k in ('call', 'stop', 'record_status', 'record_error')] == \
        ['failed', 'call_failed', 'failed', 'missing_completed_event']
    with pytest.raises(ValueError, match='duplicate answer'):          # OUT itself is refused, never half-used
        critic_025.load_answers(out)
