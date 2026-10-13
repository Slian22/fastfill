"""LLM-agent arm and its evaluator; every HTTP exchange is mocked (no network)."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import urllib.error

import pytest

from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages

SECRET = 'sk-test-SECRET-0123456789'


def _room(uid, source, room_type, group, width):
    room = {'uid': uid, 'source': source, 'group': group, 'boundary_type': 'polygon',
            'boundary': [[0, 0], [width, 0], [width, 5], [0, 5]], 'height': 3,
            'objects': [{'id': 'old-a', 'category': 'chair', 'size': [1, 1, 1], 'pos': [1, 1, 0], 'yaw': 0},
                        {'id': 'old-b', 'category': 'table', 'size': [1, 2, 1], 'pos': [3, 3, 0], 'yaw': 90}],
            'fixed': []}
    if room_type is not None:
        room['room_type'] = room_type
    return bind_room(room)[0]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def eval_data(root, mutate=None):
    """Two paper strata, a non-stratum 3D-FRONT room, a 33-room source and one without room type."""
    specs = ([(f'il-bed-{i}', 'IL3D_3dfront', 'bedroom') for i in range(3)]
             + [('il-ld-0', 'IL3D_3dfront', 'living dining room'), ('holo-k-0', 'OptiScene_holodeck', 'kitchen')]
             + [(f'x-{i:02d}', 'X', 'office') for i in range(33)] + [('y-0', 'Y', None)])
    rooms = [_room(uid, source, kind, 'g-' + uid[:4], 5 + i / 100) for i, (uid, source, kind) in enumerate(sorted(specs))]
    if mutate:
        rooms = [mutate(copy.deepcopy(r)) for r in rooms]
    data = {'rooms.jsonl': [], 'joint/requests.jsonl': [], 'placement/requests.jsonl': [], 'size-estimates.jsonl': []}
    for room in rooms:
        sizes = {o['id']: [round(v + .1, 2) for v in o['size']] for o in room['objects']}
        meta = {k: room[k] for k in ('uid', 'source', 'group')}
        meta = {**meta, **({'room_type': room['room_type']} if room.get('room_type') else {})}
        data['rooms.jsonl'].append(room)
        data['joint/requests.jsonl'].append({**meta, 'messages': messages(room, 'joint')[:2]})
        data['placement/requests.jsonl'].append({**meta, 'messages': messages(room, 'placement', sizes)[:2]})
        data['size-estimates.jsonl'].append({'uid': room['uid'], 'estimated_sizes': sizes})
    for name, rows in data.items():
        write_rows(root / name, rows)
    h.write_json(root / 'MANIFEST.json', {'expected_uids': [r['uid'] for r in rooms], 'training_manifest_sha256': 'a' * 64,
                                          'files': h.artifact_manifest(root)})
    return root


def target(record, arm='joint', room=None):
    """GT answer for the request (fixtures only)."""
    return messages(room, arm, None if arm == 'joint' else {o['id']: o['size'] for o in room['objects']})[-1]['content']


def env_file(root, mode=0o600, model=L.EXPECTED_MODEL):
    path = root / 'api.env'
    path.write_text(f'OPENAI_API_KEY={SECRET}\nexport OPENAI_BASE_URL=https://api.example.invalid/v1\nOPENAI_MODEL={model}\n')
    path.chmod(mode)
    return path


def args(root, **extra):
    values = {'data': root / 'data', 'cohort': root / 'cohort.json', 'out': root / 'run', 'env_file': root / 'api.env',
              'expected_model': L.EXPECTED_MODEL, 'concurrency': 2, 'max_calls': 4800, 'max_output_tokens': 65536,
              'timeout': 900, 'max_attempts': 2, 'limit': None}
    return argparse.Namespace(**{**values, **extra})


class FakeApi:
    """Answers each request from a per-UID script of texts or ApiErrors; logs every exchange."""
    def __init__(self, rooms, script=None, default='valid'):
        self.rooms, self.script, self.default, self.calls = rooms, script or {}, default, []

    def uid(self, messages):
        matches = [uid for uid, room in self.rooms.items() if messages[:2] == room['_request']['messages']]
        assert len(matches) == 1
        return matches[0]

    def __call__(self, config, messages, max_output_tokens, timeout):
        uid = self.uid(messages)
        self.calls.append((uid, copy.deepcopy(messages)))
        queue = self.script.setdefault(uid, [])
        item = queue.pop(0) if queue else self.default
        if isinstance(item, Exception):
            raise item
        text = target(None, 'joint', self.rooms[uid]) if item == 'valid' else item
        return {'text': text, 'status': 'completed', 'incomplete_reason': None,
                'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120, 'reasoning_tokens': 5}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(L, 'BACKOFF_SECONDS', 0)
    data = eval_data(tmp_path / 'data')
    env_file(tmp_path)
    L.main(['cohort', '--data', str(data), '--out', str(tmp_path / 'cohort.json')])
    rooms = {r['uid']: r for r in map(json.loads, (data / 'rooms.jsonl').read_text().splitlines())}
    for request in map(json.loads, (data / 'joint/requests.jsonl').read_text().splitlines()):
        rooms[request['uid']]['_request'] = request
    return tmp_path, rooms


def test_cohort_strata_cap_order_and_write_once(world):
    root, _ = world
    cohort = json.loads((root / 'cohort.json').read_text())
    expected = json.loads((root / 'data/MANIFEST.json').read_text())['expected_uids']
    assert cohort['uids'] == [u for u in expected if u in set(cohort['uids'])]
    assert cohort['counts']['paper_strata']['IL3D_3dfront/bedroom'] == 3
    assert cohort['counts']['paper_strata']['OptiScene_holodeck/kitchen'] == 1
    assert cohort['counts']['paper_strata_total'] == 4
    assert cohort['counts']['sampled_by_source'] == {'IL3D_3dfront': 1, 'OptiScene_holodeck': 0, 'X': 30, 'Y': 1}
    assert len(cohort['uids']) == 36 and cohort['counts']['total'] == 36
    ranked = sorted((f'x-{i:02d}' for i in range(33)), key=lambda u: (L._digest('llm-cohort-v1:' + u), u))
    assert set(ranked[:30]) <= set(cohort['uids']) and not set(ranked[30:]) & set(cohort['uids'])
    assert cohort['uids_sha256'] == L._digest(json.dumps(cohort['uids']))
    with pytest.raises(FileExistsError):
        L.main(['cohort', '--data', str(root / 'data'), '--out', str(root / 'cohort.json')])


def test_cohort_ignores_targets(tmp_path, world):
    root, _ = world
    def shuffle(room):
        for obj in room['objects']:
            obj['size'] = [v * 2 for v in obj['size']]
            obj['pos'] = [obj['pos'][0] + .5, obj['pos'][1], 0]
        return room
    changed = eval_data(tmp_path / 'changed', shuffle)
    assert (changed / 'joint/requests.jsonl').read_bytes() == (root / 'data/joint/requests.jsonl').read_bytes()
    assert (changed / 'rooms.jsonl').read_bytes() != (root / 'data/rooms.jsonl').read_bytes()
    base, other = L.build_cohort(root / 'data'), L.build_cohort(changed)
    assert base['uids'] == other['uids'] and base['counts'] == other['counts']


def test_payload_is_exact_j_messages_plus_fixed_suffix(world, monkeypatch):
    root, rooms = world
    seen = []

    class Response(io.BytesIO):
        status, headers = 200, {'Content-Type': 'text/event-stream; charset=utf-8'}

    def fake_open(request, timeout):
        seen.append(request)
        done = {'type': 'response.completed', 'response': {
            'model': L.EXPECTED_MODEL, 'status': 'completed', 'usage': {
                'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10, 'output_tokens_details': {'reasoning_tokens': 2}},
            'output': [{'type': 'reasoning'}, {'type': 'message', 'role': 'assistant',
                                               'content': [{'type': 'output_text', 'text': '{"placements":[]}'}]}]}}
        events = [{'type': 'response.output_text.delta', 'delta': '{"placements":[]}'}, done]
        return Response(b''.join(b'event: x\ndata: ' + json.dumps(e).encode() + b'\n\n' for e in events))

    monkeypatch.setattr(L, '_open', fake_open)
    config = L.load_config(root / 'api.env')
    bodies = []
    for uid in ('il-bed-0', 'x-00'):
        request = rooms[uid]['_request']
        reply = L.call_api(config, [*request['messages'], {'role': 'user', 'content': L.SUFFIX}], 512, 30)
        assert reply == {'text': '{"placements":[]}', 'status': 'completed', 'incomplete_reason': None,
                         'usage': {'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10, 'reasoning_tokens': 2}}
        body = json.loads(seen[-1].data)
        assert set(body) == {'model', 'input', 'reasoning', 'max_output_tokens', 'store', 'stream'}
        assert body['input'][:2] == request['messages'] and body['input'][2:] == [{'role': 'user', 'content': L.SUFFIX}]
        assert all(set(m) == {'role', 'content'} for m in body['input'])
        assert body['reasoning'] == {'effort': 'medium'} and body['model'] == L.EXPECTED_MODEL
        bodies.append({**body, 'input': body['input'][2:]})
        assert seen[-1].full_url == 'https://api.example.invalid/v1/responses'
        assert seen[-1].get_header('User-agent') == L.USER_AGENT
        assert seen[-1].get_header('Authorization') == 'Bearer ' + SECRET
    assert bodies[0] == bodies[1]  # Only the J messages differ between requests.


def test_sse_and_http_errors_map_to_fixed_codes(world, monkeypatch):
    root, _ = world
    config = L.load_config(root / 'api.env')
    def stream(*events):
        return io.BytesIO(b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events))
    incomplete = {'type': 'response.incomplete', 'response': {
        'model': L.EXPECTED_MODEL, 'incomplete_details': {'reason': 'max_output_tokens'}, 'output': []}}
    reply = L.read_sse(stream({'type': 'response.output_text.delta', 'delta': '{"pla'}, incomplete),
                       L.EXPECTED_MODEL, float('inf'))
    assert reply['text'] == '{"pla' and reply['incomplete_reason'] == 'max_output_tokens'
    for events, code, transient in (((), 'missing_completed_event', True),
                                    (({'type': 'response.failed'},), 'remote_response_failed', True),
                                    (({**incomplete, 'response': {'model': 'other'}},), 'model_mismatch', False)):
        with pytest.raises(L.ApiError) as caught:
            L.read_sse(stream(*events), L.EXPECTED_MODEL, float('inf'))
        assert (caught.value.code, caught.value.transient) == (code, transient)
    for status, transient in ((429, True), (503, True), (400, False), (401, False)):
        def fail(request, timeout, status=status):
            raise urllib.error.HTTPError(request.full_url, status, 'secret ' + SECRET, {}, io.BytesIO(SECRET.encode()))
        monkeypatch.setattr(L, '_open', fail)
        with pytest.raises(L.ApiError) as caught:
            L.call_api(config, [{'role': 'user', 'content': 'x'}], 10, 5)
        assert (caught.value.code, caught.value.transient) == (f'http_status_{status}', transient)
        assert SECRET not in str(caught.value) and caught.value.__cause__ is None


def test_config_requires_private_file_and_matching_model(world):
    root, rooms = world
    assert SECRET not in repr(L.load_config(root / 'api.env'))
    os.chmod(root / 'api.env', 0o644)
    with pytest.raises(L.ApiError, match='config_permissions'):
        L.load_config(root / 'api.env')
    env_file(root, model='other-model')
    with pytest.raises(L.ApiError, match='configured_model_mismatch'):
        L.run(args(root), FakeApi(rooms))
    assert not (root / 'run' / 'calls.jsonl').exists()


def test_full_run_records_resume_and_redaction(world, capsys):
    root, rooms = world
    api = FakeApi(rooms, {'il-bed-1': ['{"placements": "' + SECRET + '"}', 'valid'],
                          'y-0': ['bad', 'worse', 'still bad']})
    result = L.run(args(root), api)
    cohort = json.loads((root / 'cohort.json').read_text())
    assert result['status'] == 'LLM_COMPLETE' and result['rooms'] == 36
    assert result['valid_first'] == 34 and result['valid_final'] == 35
    assert result['rounds_histogram'] == {'1': 34, '2': 1, '3': 1}
    assert result['calls_ledger'] == len(api.calls) == 34 + 2 + 3
    out = root / 'run'
    manifest = json.loads((out / 'MANIFEST.json').read_text())
    h.verify_artifacts(out, manifest['files'])
    assert manifest['expected_uids'] == cohort['uids']
    for variant in L.VARIANTS:
        rows = [json.loads(x) for x in (out / variant / 'records.jsonl').read_text().splitlines()]
        assert [r['uid'] for r in rows] == cohort['uids']
        assert set(rows[0]) == {'uid', 'source', 'group', 'arm', 'raw', 'error', 'generated', 'schema', 'rounds', 'run_sha256'}
        by = {r['uid']: r for r in rows}
        assert by['il-bed-1']['schema']['valid'] is (variant == 'llm_harness')
        assert by['y-0']['raw'] == ('bad' if variant == 'llm_raw' else 'still bad') and not by['y-0']['schema']['valid']
        assert by['il-bed-0']['schema'] == infer.parse_prediction(by['il-bed-0']['raw'], rooms['il-bed-0']['_request'], 'joint')
    # Repair turns: assistant answer, then only the parser's error text.
    y = [m for uid, m in api.calls if uid == 'y-0']
    error = infer.parse_prediction('bad', rooms['y-0']['_request'], 'joint')['error']
    assert y[1] == [*rooms['y-0']['_request']['messages'], {'role': 'user', 'content': L.SUFFIX},
                    {'role': 'assistant', 'content': 'bad'}, {'role': 'user', 'content': error}]
    assert len(y[2]) == 7
    for path in out.rglob('*'):
        assert not path.is_file() or SECRET not in path.read_text()
    assert '[REDACTED]' in (out / 'llm_raw' / 'records.jsonl').read_text()
    before = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
    again = FakeApi(rooms)
    assert L.run(args(root), again)['status'] == 'LLM_COMPLETE' and again.calls == []
    assert before == {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='resume identity'):
        L.run(args(root, max_output_tokens=1024), again)
    assert SECRET not in capsys.readouterr().out


def test_api_failure_stops_then_resume_calls_only_pending(world):
    root, rooms = world
    first = FakeApi(rooms, {'il-bed-1': [L.ApiError('http_status_400')]})
    status = L.run(args(root, concurrency=1), first)
    assert status['status'] == 'API_FAILURE' and status['receipt_count'] < 36
    done = {uid for uid, _ in first.calls} - {'il-bed-1'}
    errors = [json.loads(x) for x in (root / 'run/api-errors.jsonl').read_text().splitlines()]
    assert errors == [{'uid': 'il-bed-1', 'attempt': 1, 'round': 0, 'errors': ['http_status_400']}]
    second = FakeApi(rooms)
    assert L.run(args(root), second)['status'] == 'LLM_COMPLETE'
    assert not done & {uid for uid, _ in second.calls} and 'il-bed-1' in {uid for uid, _ in second.calls}
    receipt = json.loads((root / 'run/receipts' / (L._digest('il-bed-1') + '.json')).read_text())
    assert receipt['attempt'] == 2 and receipt['valid_first']


def test_transient_retry_counted_and_attempt_cap_finalizes_without_calls(world):
    root, rooms = world
    transient = L.ApiError('timeout', True)
    api = FakeApi(rooms, {'il-bed-0': [transient, 'valid'], 'x-01': [transient] * 3})
    status = L.run(args(root, concurrency=1), api)
    assert status['status'] == 'API_FAILURE'
    receipt = json.loads((root / 'run/receipts' / (L._digest('il-bed-0') + '.json')).read_text())
    assert receipt['http_errors'] == ['timeout'] and receipt['calls'] == 2 and receipt['valid_first']
    # The exhausted timeout (an outage) does not count; two request rejections do.
    rejected = [FakeApi(rooms, {'x-01': [L.ApiError('http_status_400')]}) for _ in range(2)]
    for api_n in rejected:
        assert L.run(args(root, concurrency=1), api_n)['status'] == 'API_FAILURE'
        assert 'x-01' in {uid for uid, _ in api_n.calls}
    last = FakeApi(rooms)
    result = L.run(args(root), last)
    assert result['status'] == 'LLM_COMPLETE' and result['api_failed_rooms'] == 1
    assert 'x-01' not in {uid for uid, _ in last.calls}
    raw = {json.loads(x)['uid']: json.loads(x) for x in (root / 'run/llm_raw/records.jsonl').read_text().splitlines()}
    assert raw['x-01']['error'] == 'API_FAILED' and raw['x-01']['raw'] is None and not raw['x-01']['generated']
    ledger = (root / 'run/calls.jsonl').read_text().splitlines()
    assert result['calls_ledger'] == len(ledger) == sum(len(a.calls) for a in (api, *rejected, last))


def test_hard_call_cap_and_smoke_limit(world):
    root, rooms = world
    api = FakeApi(rooms, default='bad')
    status = L.run(args(root, limit=2, max_calls=4, concurrency=1), api)
    assert status['status'] == 'CALL_CAP_REACHED' and len(api.calls) == 4
    assert len((root / 'run/calls.jsonl').read_text().splitlines()) == 4
    result = L.run(args(root, limit=2, max_calls=7), FakeApi(rooms, default='bad'))
    assert result['status'] == 'LLM_COMPLETE' and result['rooms'] == 2 and result['calls_ledger'] == 7
    assert result['rounds_histogram'] == {'3': 2}


# Evaluator ---------------------------------------------------------------

def _inference(root, data, arm, rooms, bad=()):
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    run = root / ('infer-' + arm)
    run.mkdir()
    receipt = {'data': {'manifest_sha256': h.sha256(data / 'MANIFEST.json')},
               'training_data': {'manifest_sha256': 'a' * 64}, 'base_sha256': 'base', 'config': {'arm': arm}}
    receipt['run_sha256'] = infer._hash(receipt)
    h.write_json(run / 'run_receipt.json', receipt)
    write_rows(run / 'records.jsonl', [
        {'uid': u, 'source': rooms[u]['source'], 'group': rooms[u]['group'], 'arm': arm,
         'raw': '{}' if u in bad else target(None, arm, rooms[u]), 'error': None, 'run_sha256': receipt['run_sha256']}
        for u in manifest['expected_uids']])
    h.write_json(run / 'result.json', {'status': 'INFERENCE_COMPLETE', 'pending_count': 0})
    h.write_json(run / 'MANIFEST.json', {'expected_uids': manifest['expected_uids'], 'files': h.artifact_manifest(run)})
    return run


def test_evaluator_pairs_j_p_with_both_l_variants(world):
    root, rooms = world
    L.run(args(root), FakeApi(rooms, {'y-0': ['bad', 'worse', 'still bad'], 'il-bed-1': ['bad', 'valid']}))
    runs = {'joint': _inference(root, root / 'data', 'joint', rooms, bad={'il-bed-0', 'y-0'}),
            'placement': _inference(root, root / 'data', 'placement', rooms)}
    summary = ev.evaluate(root / 'data', root / 'cohort.json', root / 'run', root / 'eval', runs, repeats=200)
    assert summary['status'] == 'LLM_COMPARISON_COMPLETE' and list(summary['arms']) == list(ev.ORDER)
    arms = summary['arms']
    assert {a['all']['requests'] for a in arms.values()} == {36}
    assert arms['llm_raw']['all']['complete_count'] == 34 and arms['llm_harness']['all']['complete_count'] == 35
    assert arms['joint']['all']['complete_count'] == 34 and arms['llm_harness']['all']['size_iou'] == 1
    assert set(summary['paired']) == {'joint_minus_llm_raw', 'joint_minus_llm_harness',
                                      'placement_minus_llm_raw', 'placement_minus_llm_harness'}
    pooled = summary['paired']['joint_minus_llm_harness']['all']
    assert pooled['complete']['paired_rooms'] == 36
    assert pooled['complete']['estimate_joint_minus_placement'] == pytest.approx((34 - 35) / 36)
    assert set(summary['paired']['joint_minus_llm_raw']) == {'all', *summary['paper_strata']}
    assert summary['paper_strata']['IL3D_3dfront/bedroom']['llm_raw']['requests'] == 3
    assert summary['paper_strata']['IL3D_3dfront/living room']['joint']['requests'] == 0
    report = (root / 'eval/report.md').read_text()
    assert 'L-harness' in report and 'J − L-raw' in report and '表 3' in report
    h.verify_artifacts(root / 'eval', json.loads((root / 'eval/MANIFEST.json').read_text())['files'])
    with pytest.raises(FileExistsError):
        ev.evaluate(root / 'data', root / 'cohort.json', root / 'run', root / 'eval', runs, repeats=200)


def test_evaluator_llm_only_and_refuses_smoke_or_tampering(world):
    root, rooms = world
    L.run(args(root, out=root / 'smoke', limit=2), FakeApi(rooms))
    with pytest.raises(ValueError, match='cohort'):
        ev.evaluate(root / 'data', root / 'cohort.json', root / 'smoke', root / 'eval-smoke', repeats=20)
    L.run(args(root), FakeApi(rooms))
    summary = ev.evaluate(root / 'data', root / 'cohort.json', root / 'run', root / 'eval-l', repeats=20)
    assert list(summary['arms']) == list(L.VARIANTS) and summary['paired'] == {}
    path = root / 'run/llm_raw/records.jsonl'
    path.write_text(path.read_text().replace('"generated": true', '"generated": false', 1))
    with pytest.raises(ValueError, match='hash'):
        ev.evaluate(root / 'data', root / 'cohort.json', root / 'run', root / 'eval-x', repeats=20)
