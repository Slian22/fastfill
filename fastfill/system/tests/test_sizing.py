"""Missing-size API fallback: real CLI, validation, S1 and export; no GPU or network."""
import copy
import functools
import json

import pytest

from fastfill.comparison import llm_agent as L
from fastfill.comparison import sizers_020 as S
from fastfill.system import cli
from fastfill.system import pipeline as p
from fastfill.system.tests.test_pipeline import PRIOR, STUB, answer, simple, world


def requests(uid='room', categories=('bed', 'sculpture')):
    return [p.joint_request({**simple(uid), 'furniture': dict.fromkeys(categories, 1)})]


class Placer:
    identity = STUB

    def __init__(self, kind='good'):
        self.seen = []
        self.kind = kind

    def generate(self, records):
        self.seen.extend(copy.deepcopy(records))
        return {r['uid']: {'raw': answer(r, self.kind), 'error': None, 'cap_hit': False} for r in records}


class Api:
    def __init__(self, script=()):
        self.script = iter(script)
        self.seen = []

    def __call__(self, config, messages, max_output_tokens, timeout):
        self.seen.append(copy.deepcopy(messages))
        value = next(self.script, 'valid')
        if isinstance(value, Exception):
            raise value
        user = json.loads(messages[1]['content'])
        if value == 'valid':
            value = json.dumps({o['id']: [.61, .43, .82] for o in user['objects']})
        return {'text': value, 'status': 'completed', 'incomplete_reason': None,
                'usage': {'input_tokens': 10, 'output_tokens': 5, 'reasoning_tokens': 2}}


def run(world, *, prior=PRIOR, rooms=None, api=None, placer=None, **kwargs):
    api, placer = api or Api(), placer or Placer()
    result = p.run(rooms or requests(), (prior, {'path': 'prior.json' if prior else None,
                                               'sha256': '0' * 64 if prior else None}),
                   placer, world / 'out', size_fallback='api', size_api_cap=kwargs.pop('size_api_cap', 20),
                   env_file=world / 'api.env', post=api, api_concurrency=1, **kwargs)
    return result, placer, api


def test_only_unknown_objects_use_api_and_sizes_reach_layout_and_export(world):
    result, placer, api = run(world)
    sent = json.loads(api.seen[0][1]['content'])
    assert [o['category'] for o in sent['objects']] == ['sculpture']
    assert api.seen[0][0]['content'] == S.SYSTEM
    user = json.loads(placer.seen[0]['messages'][1]['content'])
    assert {o['category']: o['size'] for o in user['objects']} == {
        'bed': [2., 1.6, .5], 'sculpture': [.61, .43, .82]}
    row = json.loads((world / 'out/rooms.jsonl').read_text())
    assert row['size_sources'] == {'bed_1': 'prior_category_room_type', 'sculpture_1': 'api'}
    assert {o['category']: o['size'] for o in row['layout']} == {
        'bed': [2., 1.6, .5], 'sculpture': [.61, .43, .82]}
    exported = json.loads((world / 'out/roomgenbench.jsonl').read_text())
    assert exported['error'] is None
    obj = next(o for o in exported['layout']['objects'] if o['id'] == 'sculpture_1')
    assert obj['target_size_local_m'] == [.61, .43, .82]
    assert result['sizing']['api']['calls_ledger'] == result['api_calls_total'] == 1
    assert result['sizing']['api']['tokens']['input_tokens'] == 10
    assert '尺寸 API' in (world / 'out/report.md').read_text()
    assert 'sk-test' not in ''.join(f.read_text() for f in (world / 'out').rglob('*') if f.is_file())


def test_all_sizes_can_come_from_api_without_a_prior(world):
    result, placer, api = run(world, prior=None)
    assert len(json.loads(api.seen[0][1]['content'])['objects']) == 2
    assert all(o['size'] == [.61, .43, .82]
               for o in json.loads(placer.seen[0]['messages'][1]['content'])['objects'])
    assert result['sizing']['objects_by_source'] == {'api': 2}
    assert '训练集先验表 sha256 None' not in (world / 'out/report.md').read_text()


def test_category_bucket_is_used_and_never_inferred_from_equal_sizes(world):
    prior = {**PRIOR, 'category_room_type': {},
             'global': {'count': 1, 'size': PRIOR['category']['bed']['size']}}
    result, _, api = run(world, prior=prior, rooms=requests(categories=('bed', 'nightstand')),
                         size_api_cap=None)
    assert not api.seen
    assert result['sizing']['objects_by_source'] == {'prior_category': 2}
    assert result['api_calls_total'] == 0


def test_insufficient_category_samples_use_api(world):
    prior = {**PRIOR, 'category_room_type': {},
             'category': {**PRIOR['category'], 'bed': {'count': 4, 'size': [2, 1.6, .5]}}}
    _, _, api = run(world, prior=prior, rooms=requests(categories=('bed',)))
    assert [o['category'] for o in json.loads(api.seen[0][1]['content'])['objects']] == ['bed']


@pytest.mark.parametrize('bad', ['{}', '{"fake":[1,1,1]}', 'not JSON'])
def test_invalid_size_response_repairs_once_and_preserves_prior(world, bad):
    result, _, api = run(world, api=Api([bad, 'valid']))
    assert len(api.seen) == result['sizing']['api']['calls_ledger'] == 2
    assert api.seen[1][-1]['content'].startswith('invalid size output:')


def test_exhausted_repair_stops_before_placer_and_retains_failure_receipts(world):
    api, placer = Api(['{}', '{}']), Placer()
    with pytest.raises(RuntimeError, match='size estimation failed'):
        run(world, api=api, placer=placer)
    assert len(api.seen) == 2 and not placer.seen
    result = json.loads((world / 'out/size-api/result.json').read_text())
    assert result['status'] == 'SIZE_ESTIMATION_FAILED' and result['failed_uids'] == ['room']
    assert not (world / 'out/rooms.jsonl').exists()


def test_hard_size_cap_counts_transport_retries(world, monkeypatch):
    monkeypatch.setattr(L, 'BACKOFF_SECONDS', 0)
    api, placer = Api([L.ApiError('http_status_503', True), 'valid']), Placer()
    with pytest.raises(RuntimeError, match='size estimation failed'):
        run(world, api=api, placer=placer, size_api_cap=1)
    assert len(api.seen) == 1 and not placer.seen
    summary = json.loads((world / 'out/size-api/result.json').read_text())
    assert summary['calls_ledger'] == 1 and summary['stopped'] == 'CALL_CAP_REACHED'


@pytest.mark.parametrize('kwargs,match', [({'size_api_cap': 0}, 'size_api_cap'),
                                        ({'size_api_cap': None}, 'size_api_cap'),
                                        ({'size_api_max_rooms': 0}, 'size_api_max_rooms')])
def test_size_budget_preflight_makes_no_calls(world, kwargs, match):
    api, placer = Api(), Placer()
    with pytest.raises((RuntimeError, ValueError), match=match):
        run(world, api=api, placer=placer, **kwargs)
    assert not api.seen and not placer.seen


def test_dry_run_without_sizes_only_plans_and_never_calls_placer_or_api(world):
    result, placer, api = run(world, prior=None, dry_run=True, size_api_cap=None)
    assert result['status'] == 'DRY_RUN_SIZE_ESTIMATION_REQUIRED'
    assert {u: sorted(ids) for u, ids in result['would_estimate_sizes'].items()} == {
        'room': ['bed_1', 'sculpture_1']}
    assert not placer.seen and not api.seen and not (world / 'out/rooms.jsonl').exists()


@pytest.mark.parametrize('kind', ['duplicate_uid', 'bad_system', 'label_leak'])
def test_invalid_inputs_rejected_before_paid_request(world, kind):
    rows = requests()
    record, sidecar = rows[0]
    if kind == 'duplicate_uid':
        rows = rows + rows
    elif kind == 'bad_system':
        rows = [({**record, 'messages': [{'role': 'system', 'content': 'untrusted'}, record['messages'][1]]}, sidecar)]
    else:
        rows = [({**record, 'target': 'ground truth'}, sidecar)]
    api, placer = Api(), Placer()
    with pytest.raises(ValueError):
        run(world, rooms=rows, api=api, placer=placer)
    assert not api.seen and not placer.seen


def test_missing_prior_is_explicit_and_does_not_hide_bad_files(world):
    missing = world / 'missing.json'
    with pytest.raises(FileNotFoundError):
        p.load_prior(missing)
    assert p.load_prior(missing, allow_missing=True)[0] is None
    assert p.load_prior(None, allow_missing=True)[0] is None
    with pytest.raises((FileNotFoundError, ValueError)):
        p.load_prior(missing, '0' * 64, allow_missing=True)
    corrupt = world / 'bad.json'
    corrupt.write_text('bad JSON')
    with pytest.raises(ValueError):
        p.load_prior(corrupt, allow_missing=True)


@pytest.mark.parametrize('missing_path', [False, True])
def test_cli_can_run_without_prior_file(world, monkeypatch, missing_path):
    path = world / 'rooms.jsonl'
    path.write_text(json.dumps(simple('cli')) + '\n')
    placer, api = Placer(), Api()
    monkeypatch.setattr(p, 'hf_placer', lambda *a: placer)
    monkeypatch.setattr(p, 'run', functools.partial(p.run, post=api))
    argv = ['run', '--requests', str(path), '--placer', 'hf', '--model', 'base', '--adapter', 'adapter',
            '--escalate', 'none', '--size-fallback', 'api', '--size-api-cap', '2',
            '--env-file', str(world / 'api.env'), '--out', str(world / 'out')]
    if missing_path:
        argv += ['--prior', str(world / 'missing.json')]
    assert cli.main(argv) == 0
    assert len(api.seen) == len(placer.seen) == 1
    result = json.loads((world / 'out/result.json').read_text())
    assert result['status'] == 'SYSTEM_COMPLETE' and result['prior']['sha256'] is None


def test_original_mode_still_requires_prior(world, monkeypatch):
    monkeypatch.setattr(p, 'hf_placer', lambda *a: pytest.fail('model loaded before argument validation'))
    with pytest.raises(SystemExit):
        cli.main(['run', '--requests', 'r', '--placer', 'hf', '--model', 'm', '--adapter', 'a',
                  '--escalate', 'none', '--out', str(world / 'out')])


def test_size_and_layout_api_have_separate_caps_and_same_frozen_sizes(world):
    size_api, layout_inputs = Api(), []

    def post(config, messages, max_output_tokens, timeout):
        if messages[0]['content'] == S.SYSTEM:
            return size_api(config, messages, max_output_tokens, timeout)
        layout_inputs.append(json.loads(messages[1]['content']))
        return {'text': answer({'messages': messages}, 'good'), 'status': 'completed',
                'incomplete_reason': None, 'usage': {'input_tokens': 20, 'output_tokens': 10}}

    result, placer, _ = run(world, api=post, placer=Placer('bad'), size_api_cap=1,
                            escalate='l+o2', api_cap=1, o2_model=world / 'o2',
                            o2_load=lambda _: pytest.fail('O2 should not be needed'))
    assert len(layout_inputs) == 1 and len(size_api.seen) == 1
    assert layout_inputs[0] == json.loads(placer.seen[0]['messages'][1]['content'])
    assert result['api_calls_total'] == 2
    assert result['sizing']['api']['calls_ledger'] == result['escalation']['api']['calls_ledger'] == 1


def test_missing_size_cli_dry_run_needs_no_model_or_api_config(world, monkeypatch):
    path = world / 'rooms.jsonl'
    path.write_text(json.dumps(simple('dry')) + '\n')
    forbidden = lambda *a, **kw: pytest.fail('dry-run must not load models, config or GPUs')
    monkeypatch.setattr(p, 'hf_placer', forbidden)
    monkeypatch.setattr(p, '_api_config', forbidden)
    monkeypatch.setattr(p, '_visible_gpus', forbidden)
    assert cli.main(['run', '--requests', str(path), '--placer', 'hf', '--model', 'missing', '--adapter', 'missing',
                     '--escalate', 'l+o2', '--size-fallback', 'api', '--dry-run', '--out', str(world / 'out')]) == 0
    assert json.loads((world / 'out/result.json').read_text())['would_escalate'] is None


@pytest.mark.parametrize('bad_size', [[0, 1, 1], [1, -1, 1], [True, 1, 1], [1, 1], [1000, 1, 1]])
def test_invalid_dimensions_are_repaired_before_placement(world, bad_size):
    good, calls = Api(), []

    def post(config, messages, max_output_tokens, timeout):
        value = good(config, messages, max_output_tokens, timeout)
        calls.append(1)
        if len(calls) == 1:
            oid = json.loads(messages[1]['content'])['objects'][0]['id']
            return {**value, 'text': json.dumps({oid: bad_size})}
        return value

    result, _, _ = run(world, api=post)
    assert result['sizing']['api']['calls_ledger'] == 2


def test_incomplete_http_response_is_not_treated_as_valid_sizes(world):
    api = Api()

    def post(*a):
        return {**api(*a), 'status': 'incomplete', 'incomplete_reason': 'max_output_tokens'}

    placer = Placer()
    with pytest.raises(RuntimeError, match='size estimation failed'):
        run(world, api=post, placer=placer)
    assert len(api.seen) == 2 and not placer.seen


def test_api_outage_writes_terminal_status_and_redacted_receipts(world):
    api, placer = Api([L.ApiError('http_status_401', False)]), Placer()
    with pytest.raises(RuntimeError, match='size estimation failed'):
        run(world, api=api, placer=placer)
    status = json.loads((world / 'out/size-api/status.json').read_text())
    assert status['status'] == 'SIZE_ESTIMATION_FAILED' and status['stopped'] == 'API_FAILURE'
    assert status['calls_ledger'] == 1 and not placer.seen


def test_unexpected_api_error_stops_sizing_with_terminal_evidence(world):
    def post(*a):
        raise ValueError('do not expose sk-test')

    placer = Placer()
    with pytest.raises(RuntimeError, match='size estimation failed'):
        run(world, api=post, placer=placer)
    status = json.loads((world / 'out/size-api/status.json').read_text())
    assert status['status'] == 'SIZE_ESTIMATION_FAILED' and status['stopped'] == 'INVALID_API_RESPONSE'
    assert not placer.seen
    assert 'sk-test' not in ''.join(f.read_text() for f in (world / 'out').rglob('*') if f.is_file())


def test_precomputed_layouts_cannot_bypass_size_binding(world):
    api = Api()
    with pytest.raises(ValueError, match='precomputed'):
        run(world, api=api, placed={'room': {}})
    assert not api.seen


def test_concurrent_size_requests_cannot_exceed_cap(world, monkeypatch):
    monkeypatch.setattr(L, 'BACKOFF_SECONDS', 0)
    api, placer = Api(['{}'] * 20), Placer()
    with pytest.raises(RuntimeError, match='size estimation failed'):
        p.run(requests('a') + requests('b'), (PRIOR, {'path': 'prior', 'sha256': '0' * 64}),
              placer, world / 'out', size_fallback='api', size_api_cap=2,
              env_file=world / 'api.env', post=api, api_concurrency=4)
    assert len(api.seen) == 2 and not placer.seen
    summary = json.loads((world / 'out/size-api/result.json').read_text())
    assert summary['calls_ledger'] == 2 and set(summary['failed_uids']) == {'a', 'b'}


@pytest.mark.parametrize('kind', ['boundary', 'height', 'fixed_size', 'fixed_yaw', 'metadata'])
def test_malformed_room_geometry_in_later_request_is_rejected_before_any_call(world, kind):
    record, sidecar = requests('bad')[0]
    user = json.loads(record['messages'][1]['content'])
    if kind == 'boundary':
        user = {**user, 'boundary': [[0, 0], [1, 1], [2, 2]]}
    elif kind == 'height':
        user = {**user, 'height': -1}
    elif kind == 'metadata':
        user = {**user, 'room_type': 'kitchen'}
    else:
        key, value = ('size', [0, 1, 1]) if kind == 'fixed_size' else ('yaw', 400)
        user = {**user, 'fixed': [{**user['fixed'][0], key: value}]}
    record = {**record, 'messages': [record['messages'][0], {'role': 'user', 'content': json.dumps(user)}]}
    api, placer = Api(), Placer()
    with pytest.raises(ValueError):
        run(world, rooms=requests('good') + [(record, sidecar)], api=api, placer=placer)
    assert not api.seen and not placer.seen


def test_old_o2_shards_cannot_be_reused_with_new_sizes(world):
    api = Api()
    with pytest.raises(ValueError, match='precomputed'):
        run(world, api=api, o2=lambda *a: pytest.fail('stale O2 callback must not run'))
    assert not api.seen


@pytest.mark.parametrize('table', [None, [], {'full_prior': None}, {'full_prior': []}])
def test_existing_malformed_table_never_becomes_api_fallback(world, table):
    path = world / 'prior.json'
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match='prior.*object'):
        p.load_prior(path, p.h.sha256(path), allow_missing=True)


def test_dimensions_that_round_out_of_range_are_repaired(world):
    api, calls = Api(), []

    def post(*args):
        reply = api(*args)
        calls.append(1)
        if len(calls) == 1:
            oid = json.loads(args[1][1]['content'])['objects'][0]['id']
            return {**reply, 'text': json.dumps({oid: [999.999, 1, 1]})}
        return reply

    result, placer, _ = run(world, api=post)
    assert result['sizing']['api']['calls_ledger'] == 2 and len(placer.seen) == 1
    assert api.seen[1][-1]['content'] == S.BAD_SIZE
