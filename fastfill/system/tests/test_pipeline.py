"""Amendment-022 delivered system with a stub placer, a fake API and a fake O2 (no GPU, network or API key).

FASTFILL_EVAL_DATA=.../eval-data-v1 [FASTFILL_PRIOR=.../size-priors.json] also checks the real P requests.
"""
import functools
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastfill.comparison import eval_data
from fastfill.comparison import llm_agent as L
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.size_prior import fit_prior
from fastfill.system import cli
from fastfill.system import pipeline as p

RECT = [[0, 0], [5, 0], [5, 4], [0, 4]]
DOOR = {'id': 'door', 'category': 'door', 'size': [0.1, 0.9, 2.0], 'pos': [5, 2, 0], 'yaw': math.pi / 2}
SIZES = {'bed': [2.0, 1.6, 0.5], 'nightstand': [0.5, 0.4, 0.5]}
PRIOR = fit_prior([{'uid': f't{i}', 'room_type': 'bedroom',
                    'objects': [{'id': c, 'category': c, 'size': s} for c, s in SIZES.items()]} for i in range(5)])
SPOTS = [(1.2, 1.0), (1.2, 3.0)]
STUB = {'engine': 'stub', 'model': 'base', 'adapter': 'adapter', 'adapter_sha256': 'f' * 64}


def simple(uid, i=0):
    """Rooms differ only by height, so a request's user turn identifies its room."""
    return {'uid': uid, 'room_type': 'bedroom', 'boundary': RECT, 'height': 2.8 + i / 100, 'fixed': [DOOR],
            'furniture': {'bed': 1, 'nightstand': 1}}


def answer(request, kind):
    """good: apart on the floor; bad: the first object floats 1 m up (a support failure S1 cannot fix); worse: both."""
    if kind == 'invalid':
        return 'no layout'
    lift = {'good': (0, 0), 'bad': (1.0, 0), 'worse': (1.0, 1.0)}[kind]
    objects = h.strict_json(request['messages'][1]['content'])['objects']
    return json.dumps({'placements': [{'id': o['id'], 'pos': [*SPOTS[k], lift[k]], 'yaw': 0}
                                      for k, o in enumerate(objects)]})


class FakeO2:
    """llm_open's vLLM LLM and tokenizer stand-in; answers each conversation with its room's scripted kind."""
    def __init__(self, script, by_user):
        self.script, self.by_user, self.calls = script, by_user, []

    def get_tokenizer(self):
        return self

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False, **kwargs):
        return 'p' * 10

    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)

    def chat(self, conversations, params, use_tqdm=False, chat_template_kwargs=None):
        outputs = []
        for conversation in conversations:
            uid = self.by_user[conversation[1]['content']]
            self.calls.append(uid)
            text = answer({'messages': conversation}, self.script[uid])
            outputs.append(SimpleNamespace(prompt_token_ids=[1] * 10, outputs=[
                SimpleNamespace(text=text, token_ids=[1] * (len(text) + 1), finish_reason='stop')]))
        return outputs


@pytest.fixture
def world(tmp_path):
    env = tmp_path / 'api.env'
    env.write_text(f'OPENAI_API_KEY=sk-test\nOPENAI_BASE_URL=https://api.example.invalid/v1\n'
                   f'OPENAI_MODEL={L.EXPECTED_MODEL}\n')
    env.chmod(0o600)
    (tmp_path / 'o2').mkdir()
    (tmp_path / 'o2' / 'config.json').write_text('{}')
    return tmp_path


def inputs(plan):
    """(requests, {P user turn: uid}, a stub placer answering each room with its scripted kind)."""
    requests = [p.joint_request(simple(uid, i)) for i, uid in enumerate(plan)]
    by_user = {p.placement_request(r, p.prior_sizes(r, PRIOR))['messages'][1]['content']: r['uid'] for r, _ in requests}
    placer = p.Placer(STUB, lambda records: {r['uid']: {'raw': answer(r, plan[r['uid']][0]), 'error': None,
                                                        'cap_hit': False} for r in records})
    return requests, by_user, placer


def system(root, plan, escalate='l+o2', api_cap=100, **kwargs):
    """plan: {uid: (placer kind, [API kinds per round], O2 kind)} -> (result, rows by uid, API uids, O2 uids)."""
    requests, by_user, placer = inputs(plan)
    script, api_calls, o2 = {u: list(v[1]) for u, v in plan.items()}, [], FakeO2({u: v[2] for u, v in plan.items()},
                                                                                  by_user)

    def post(config, messages, max_output_tokens, timeout):
        assert config.model == L.EXPECTED_MODEL and messages[2] == {'role': 'user', 'content': L.SUFFIX}
        uid = by_user[messages[1]['content']]
        api_calls.append(uid)
        queue = script[uid]
        return {'text': answer({'messages': messages}, queue.pop(0) if len(queue) > 1 else queue[0]),
                'status': 'completed', 'incomplete_reason': None, 'usage': {'input_tokens': 10, 'output_tokens': 5}}
    result = p.run(requests, (PRIOR, {'path': 'prior.json', 'sha256': '0' * 64}), placer, root / 'out',
                   escalate=escalate, api_cap=api_cap, env_file=root / 'api.env', o2_model=root / 'o2', post=post,
                   o2_load=lambda engine: (o2, dict), api_concurrency=1, **kwargs)
    rows = {r['uid']: r for r in map(json.loads, (root / 'out/rooms.jsonl').read_text().splitlines())}
    return result, rows, api_calls, o2.calls


def gt_room(uid, room_type, categories):
    return {'uid': uid, 'source': 'S', 'split': 'test', 'boundary_type': 'polygon', 'boundary': RECT, 'height': 2.8,
            'room_type': room_type, 'fixed': [DOOR],
            'objects': [{'id': f'{uid}:{k}', 'category': c, 'size': [1, 1, 1], 'pos': [1 + k / 2, 1, 0], 'yaw': .3,
                         'parent': None} for k, c in enumerate(categories)]}


def test_placement_request_is_eval_datas_byte_for_byte(tmp_path):
    rooms = [gt_room('r1', 'Bedroom', ['bed', 'Nightstand', 'lamp (white)']), gt_room('r2', None, ['bed', 'sofá'])]
    for arm in ('joint', 'placement'):
        (tmp_path / arm).mkdir()
    eval_data._write_rows(tmp_path, rooms, {r['uid']: {'component': 'c:' + r['uid']} for r in rooms}, PRIOR,
                          {'written_rows': 0, 'failed_rows': 0, 'failures': []})
    joint, placement = ((tmp_path / arm / 'requests.jsonl').read_text().splitlines() for arm in ('joint', 'placement'))
    for room, j, line in zip(rooms, joint, placement, strict=True):
        record = json.loads(j)
        assert json.dumps(p.placement_request(record, p.prior_sizes(record, PRIOR)), ensure_ascii=False,
                          allow_nan=False) == line
        given = {**{k: room[k] for k in ('uid', 'source', 'boundary', 'height', 'room_type', 'fixed')},
                 'objects': [{'id': o['id'], 'category': o['category']} for o in room['objects']]}
        simple_record, sidecar = p.joint_request(given)        # no GT: same model turns, ids map back
        assert simple_record['messages'] == record['messages']
        assert set(sidecar['bound_to_original'].values()) == {'door'} | {o['id'] for o in room['objects']}


@pytest.mark.skipif(not os.environ.get('FASTFILL_EVAL_DATA'), reason='needs FASTFILL_EVAL_DATA=.../eval-data-v1')
def test_eval_data_v1_placement_requests_byte_for_byte():
    root, prior = Path(os.environ['FASTFILL_EVAL_DATA']), os.environ.get('FASTFILL_PRIOR')
    prior = p.load_prior(prior)[0] if prior else None
    names = ('joint/requests.jsonl', 'placement/requests.jsonl', 'size-estimates.jsonl')
    for j, line, estimate in zip(*((root / n).read_text().splitlines() for n in names), strict=True):
        record, sizes = json.loads(j), json.loads(estimate)['estimated_sizes']
        if prior is not None:
            assert p.prior_sizes(record, prior) == sizes
        assert json.dumps(p.placement_request(record, sizes), ensure_ascii=False, allow_nan=False) == line


def test_best_candidate_rule():
    c = lambda stage, ok, complete, v: {'stage': stage, 'checker_pass': ok, 'complete': complete, 'violations': v}
    assert p.best([c('placer', False, True, 0), c('gpt', True, True, 0)])['stage'] == 'gpt'
    assert p.best([c('placer', False, False, None), c('gpt', False, True, 9)])['stage'] == 'gpt'
    assert p.best([c('placer', False, True, 2), c('gpt', False, True, 1), c('o2', False, True, 1)])['stage'] == 'gpt'
    assert p.best([c('placer', True, True, 0), c('gpt', True, True, 0)])['stage'] == 'placer'


def test_stage_order_escalation_only_on_failures(world):
    plan = {'a': ('good', ['good'], 'good'), 'b': ('bad', ['good'], 'good'), 'c': ('bad', ['bad'], 'good'),
            'd': ('invalid', ['invalid'], 'bad'), 'e': ('bad', ['invalid'], 'worse')}
    result, rows, api_calls, o2_calls = system(world, plan)
    assert sorted(api_calls) == ['b', 'c', 'd', 'd', 'e', 'e']          # one answer + one repair round at most
    assert sorted(o2_calls) == ['c', 'd', 'e']                          # gpt passed b
    assert {u: r['stage'] for u, r in rows.items()} == {'a': 'placer', 'b': 'gpt', 'c': 'o2', 'd': 'o2', 'e': 'placer'}
    assert [c['stage'] for c in rows['a']['candidates']] == ['placer']
    assert [c['stage'] for c in rows['c']['candidates']] == ['placer', 'gpt', 'o2']
    assert [u for u, r in rows.items() if r['needs_escalation']] == ['b', 'c', 'd', 'e']
    assert (rows['d']['complete'], rows['d']['violations'], rows['e']['violations']) == (True, 1, 1)
    assert result['stages'] | {'final_accepted': None} == {
        'placer_raw_checker_pass': 1, 'placer_s1_checker_pass': 1, 'needs_escalation': 4, 'final_checker_pass': 3,
        'final_complete': 5, 'final_accepted': None, 'final_stage': {'placer': 2, 'gpt': 1, 'o2': 2}}
    api = result['escalation']['api']
    assert (api['calls_ledger'], api['stopped'], api['rooms']) == (6, None, 4)
    assert len((world / 'out/api/calls.jsonl').read_text().splitlines()) == 6
    assert 'sk-test' not in ''.join(p.read_text() for p in (world / 'out').rglob('*') if p.is_file())


def test_never_empty_fallback(world):
    result, rows, api_calls, o2_calls = system(world, {'a': ('invalid', ['good'], 'good')}, escalate='none')
    row = rows['a']
    assert (row['stage'], row['complete'], api_calls, o2_calls) == ('fallback', True, [], [])
    assert [c['stage'] for c in row['candidates']] == ['placer', 'fallback']
    assert sorted(o['id'] for o in row['layout']) == ['bed_1', 'nightstand_1']
    assert all(o['pos'][2] == 0 and o['parent'] is None for o in row['layout'])


def test_api_cap_is_hard(world):
    plan = {'b': ('bad', ['invalid'], 'good'), 'c': ('bad', ['good'], 'good')}
    result, rows, api_calls, o2_calls = system(world, plan, api_cap=1)
    assert api_calls == ['b'] and sorted(o2_calls) == ['b', 'c']
    assert len((world / 'out/api/calls.jsonl').read_text().splitlines()) == 1
    assert (result['escalation']['api']['calls_ledger'], result['escalation']['api']['stopped']) == (1, 'CALL_CAP_REACHED')
    gpt = {u: r['candidates'][1] for u, r in rows.items()}
    assert gpt['c']['error'] == 'CALL_CAP_REACHED' and not gpt['b']['complete']
    assert {u: r['stage'] for u, r in rows.items()} == {'b': 'o2', 'c': 'o2'}


def test_dry_run_reports_escalation_without_calls(world):
    result, rows, api_calls, o2_calls = system(world, {'a': ('good', ['good'], 'good'), 'b': ('bad', ['good'], 'good')},
                                               dry_run=True, workers=2)          # S1 in forkserver workers
    assert (result['status'], result['would_escalate'], api_calls, o2_calls) == ('DRY_RUN', ['b'], [], [])
    assert result['api_calls_upper_bound'] == 2 * (1 + L.RETRIES)
    assert rows['b']['stage'] == 'placer' and not (world / 'out/api').exists()


def test_provenance_and_roomgenbench_export(world):
    result, rows, _, _ = system(world, {'a': ('good', ['good'], 'good'), 'b': ('bad', ['good'], 'good')})
    assert result['placer'] == STUB and result['prior']['sha256'] == '0' * 64 and 'pipeline.py' in result['code_sha256']
    assert result['s1'] == {'max_trials': 2000, 'time_limit': 60.0} and result['api_rounds'] == 2
    for row in rows.values():
        assert row['placer'] == STUB
        assert all({'stage', 'complete', 'checker_pass_raw', 'checker_pass', 'violations', 'accepted', 's1_status'}
                   <= set(c) for c in row['candidates'])
        assert all(set(o) == {'id', 'category', 'size', 'pos', 'yaw', 'parent'} for o in row['layout'])
    assert rows['b']['candidates'][1]['size_enforcement']['size_changed'] == 0
    exports = {r['uid']: r for r in map(json.loads, (world / 'out/roomgenbench.jsonl').read_text().splitlines())}
    assert set(exports) == {'a', 'b'} and all(e['placer'] == STUB for e in exports.values())
    scene = exports['a']['scene']
    assert exports['a']['error'] is None and sorted(o['id'] for o in scene['objects']) == ['bed_1', 'nightstand_1']
    assert [f['id'] for f in scene['fixed_objects']] == ['door']
    assert (world / 'out/report.md').read_text().startswith('# FastFill 交付系统运行报告')


class CharTok:
    """Adapter-tokenizer stand-in: the chat prompt is the user turn, one token per character, <|im_end|> = 0."""
    pad_token_id = 0

    def get_vocab(self):
        return {'<|im_end|>': 0}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[1]['content']

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {'input_ids': [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(map(chr, ids))


def test_placer_backends_with_fake_engines(world, monkeypatch):
    model, adapter = world / 'base', world / 'adapter'
    for path, name in ((model, 'config.json'), (adapter, 'adapter_config.json')):
        path.mkdir()
        (path / name).write_text('{}')
    records = [p.placement_request(r, p.prior_sizes(r, PRIOR)) for r, _ in (p.joint_request(simple(u, i))
                                                                            for i, u in enumerate('ab'))]
    by_user = {r['messages'][1]['content']: r for r in records}
    reply = lambda ids: [ord(c) for c in answer(by_user[CharTok().decode(ids)], 'good')] + [0]

    def hf_generate(batch, cap):                                 # infer._runtime's generate: left-padded batch
        width = max(map(len, batch))
        return [[0] * (width - len(ids)) + ids + reply(ids) for ids in batch], width
    monkeypatch.setattr(p.infer, '_runtime', lambda run: (CharTok(), hf_generate, 0) if run.exact_policy else None)
    hf = p.hf_placer(model, adapter)
    assert (hf.generate.func, hf.generate.args[1]) == (p._isolated, True)      # a child that sees one GPU
    assert hf.generate.args[0](records) == {r['uid']: {'raw': answer(r, 'good'), 'error': None, 'cap_hit': False}
                                            for r in records}
    calls = []

    class FakeVllm:
        def generate(self, prompts, params, lora_request=None, use_tqdm=True):
            calls.append(dict(params=params, lora=lora_request))
            return [SimpleNamespace(prompt_token_ids=x['prompt_token_ids'] + bad, outputs=[SimpleNamespace(
                token_ids=reply(x['prompt_token_ids']), finish_reason='length' if k else 'stop')])
                for k, x in enumerate(prompts)]
    bad = []
    monkeypatch.setattr(p.dpo_stages, '_model_tokenizer', lambda path: CharTok())
    vllm = p.vllm_placer(model, adapter, load=lambda engine: (FakeVllm(), dict, lambda *a: a))
    assert (vllm.generate.func, vllm.generate.args[1]) == (p._isolated, False)
    assert {u: a['cap_hit'] for u, a in vllm.generate.args[0](records).items()} == {'a': False, 'b': True}
    assert [len(c['params']) for c in calls] == [2]                    # one LLM.generate call within 400k cap tokens
    assert all(x['temperature'] == 0 and x['stop_token_ids'] == [0] for x in calls[0]['params'])
    assert calls[0]['lora'] == ('placement', 1, str(adapter.resolve()))
    monkeypatch.setitem(p.A.BATCHINGS, 'objects', 1)                   # a021 chunking: one room per call here
    calls.clear()
    vllm.generate.args[0](records)
    assert [len(c['params']) for c in calls] == [1, 1] and vllm.identity['batching'] == 'objects'
    bad.append(7)
    with pytest.raises(ValueError, match='does not match its request'):
        vllm.generate.args[0](records)
    for placer, engine in ((hf, 'hf_exact_policy'), (vllm, 'vllm_greedy_bf16_lora')):
        assert placer.identity['engine'] == engine and placer.identity['adapter_sha256'] == p.infer._hash(
            {'adapter_config.json': h.sha256(adapter / 'adapter_config.json')})


def test_cli_accepts_both_request_forms(world, monkeypatch):
    requests = world / 'rooms.jsonl'
    requests.write_text(json.dumps(p.joint_request(simple('j', 1))[0]) + '\n' + json.dumps(simple('s', 2)) + '\n')
    prior = world / 'prior.json'
    prior.write_text(json.dumps({'full_prior': PRIOR}))
    monkeypatch.setattr(p, 'hf_placer', lambda model, adapter: p.Placer(STUB, lambda records: {
        r['uid']: {'raw': answer(r, 'good'), 'error': None, 'cap_hit': False} for r in records}))
    argv = ['run', '--requests', str(requests), '--prior', str(prior), '--placer', 'hf', '--model', 'm',
            '--adapter', 'a', '--escalate', 'none', '--out', str(world / 'cli')]
    assert cli.main(argv) == 0
    result = json.loads((world / 'cli/result.json').read_text())
    assert result['prior']['sha256'] == h.sha256(prior) and result['stages']['final_checker_pass'] == 2
    rows = [json.loads(x) for x in (world / 'cli/rooms.jsonl').read_text().splitlines()]
    assert [sorted(o['id'] for o in r['layout']) for r in rows] == [['obj_0001', 'obj_0002'], ['bed_1', 'nightstand_1']]
    with pytest.raises(ValueError, match='pinned'):
        cli.main(argv[:-1] + [str(world / 'cli2'), '--prior-sha256', '0' * 64])


def test_placer_child_sees_only_the_first_gpu(monkeypatch):
    visible = functools.partial(os.getenv, 'CUDA_VISIBLE_DEVICES')        # fn(records) -> the child's value
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,5')
    assert (p._isolated(visible, True, None), p._isolated(visible, False, None)) == ('3', '3,5')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3,5'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert p._isolated(visible, True, None) == '0' and 'CUDA_VISIBLE_DEVICES' not in os.environ


def test_api_room_cap_stops_before_any_call(world):
    plan = {'b': ('bad', ['good'], 'good'), 'c': ('bad', ['good'], 'good')}
    with pytest.raises(RuntimeError, match='api_max_rooms 1; no API call'):
        system(world, plan, api_max_rooms=1)
    assert not (world / 'out/api').exists()
    (world / 'out').rename(world / 'refused')
    result, _, api_calls, _ = system(world, plan, api_max_rooms=2)
    assert sorted(api_calls) == ['b', 'c'] and result['api_max_rooms'] == 2


def fingerprint(out, *local):
    """Digests of a pipeline run's outputs without code hashes, wall times and the given machine-local strings."""
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    result = json.loads((out / 'result.json').read_text())
    text = json.dumps({k: v for k, v in result.items() if k not in ('code_sha256', 'runtime_seconds')}, sort_keys=True)
    for s in local:
        text = text.replace(s, '<local>')
    receipts = [json.loads(x) for x in (out / 'o2/receipts.jsonl').read_text().splitlines()]
    for a in (a for x in receipts for a in x['answers']):
        a.pop('latency_seconds')
    return {'files': sorted(str(f.relative_to(out)) for f in out.rglob('*') if f.is_file() and f.parent.name != 'receipts'),
            'result': sha(text), 'rooms': sha((out / 'rooms.jsonl').read_text()),
            'roomgenbench': sha((out / 'roomgenbench.jsonl').read_text()), 'o2': sha(json.dumps(receipts, sort_keys=True))}


DEFAULT_PLAN = {'a': ('good', ['good'], 'good'), 'b': ('bad', ['good'], 'good'), 'c': ('bad', ['bad'], 'good'),
                'd': ('invalid', ['invalid'], 'bad'), 'e': ('bad', ['invalid'], 'worse'),
                'f': ('invalid', ['invalid'], 'invalid')}
FILES = ['o2/invocations.jsonl', 'o2/receipts.jsonl', 'report.md', 'result.json', 'roomgenbench.jsonl', 'rooms.jsonl']
GOLDEN = {      # captured with the code before the sharded O2 fallback existed (commit 23a51728d)
    'o2': {'files': FILES, 'result': 'f9cac59ff768ac0b2b3980fc193e3ac16877c181f261666be907d16578b793a6',
           'rooms': 'c75ac81ff5e5eb12f40cd34bda812d1b0591f918d081dce3e550be77a0e109f1',
           'roomgenbench': '69accf962a3b2f64a64599363c640722821cc9f6045a0b1d62783ac8658af8e8',
           'o2': '105112843717bb46ad2f41f12c8c32f712cfc15693f64ecbd3ada132284bfcc0'},
    'l+o2': {'files': ['api/calls.jsonl', 'api/status.json', *FILES],
             'result': '1348e6278b48f7ac4d3c7ef505b320e07ae391c7078dadd705dc6bf714a9d273',
             'rooms': '7f71d694fee6295aa893da07312362fb09f087b7f7134243e71cb863a927b0ae',
             'roomgenbench': '4b3cb2a01149bd870d1ea6ee1d0744cfea1bc6e868b8252c7a88321dd903e408',
             'o2': '639001bc00240e0549ac4c25d092deb45693d8701031558acaca8a7b49628a68'}}


@pytest.mark.parametrize('escalate', ['o2', 'l+o2'])
def test_default_outputs_are_unchanged(world, escalate):
    system(world, DEFAULT_PLAN, escalate=escalate)
    assert fingerprint(world / 'out', str(world.resolve()), str(world)) == GOLDEN[escalate]


def test_configuration_is_checked_before_the_placer(world, monkeypatch):
    plan = {'a': ('good', ['good'], 'good')}
    (world / 'api.env').chmod(0o644)
    with pytest.raises(L.ApiError, match='config_permissions'):
        system(world, plan)
    (world / 'o2/config.json').unlink()
    (world / 'o2').rmdir()
    with pytest.raises(FileNotFoundError):
        system(world, plan, escalate='o2')
    monkeypatch.setattr(p, '_visible_gpus', lambda: 1)
    requests = [p.joint_request(simple('a'))]
    with pytest.raises(RuntimeError, match='2 visible GPUs'):
        p.run(requests, (PRIOR, {}), p.Placer(STUB, None), world / 'out', escalate='o2', o2_model=world)
    assert not (world / 'out').exists()                                 # no output dir: nothing ran


# ---- sharded O2 fallback ----------------------------------------------------------------------------

DUMMY_PRIOR = (PRIOR, {'path': 'prior.json', 'sha256': '0' * 64})


class Killed(Exception):
    pass


class KilledO2(FakeO2):
    """Dies, as a Slurm time limit would, on the first generation call that reaches a room of `kill`."""
    def __init__(self, script, by_user, kill):
        super().__init__(script, by_user)
        self.kill = set(kill)

    def chat(self, conversations, *args, **kwargs):
        if self.kill & {self.by_user[c[1]['content']] for c in conversations}:
            raise Killed
        return super().chat(conversations, *args, **kwargs)


def no_latency(bodies):
    return {u: {**b, 'answers': [{k: v for k, v in a.items() if k != 'latency_seconds'} for a in b['answers']]}
            for u, b in bodies.items()}


def test_plan_escalates_as_run_and_the_rule_partitions_it(world):
    requests, _, placer = inputs(DEFAULT_PLAN)
    value = p.plan(requests, DUMMY_PRIOR, placer, world / 'plan', 3)
    _, rows, _, _ = system(world, DEFAULT_PLAN, escalate='o2')
    assert value['escalated'] == [u for u, r in rows.items() if r['needs_escalation']] == list('bcdef')
    for n in range(1, 7):
        parts = [[u for u in value['escalated'] if p.shard_of(u, n) == k] for k in range(n)]
        assert sorted(sum(parts, [])) == value['escalated']                  # each room in exactly one shard
    assert p.shard_of('b', 3) == int(hashlib.sha256(b'a022-o2-shard:b').hexdigest(), 16) % 3
    assert value['shard_rooms'] == [sum(p.shard_of(u, 3) == k for u in value['escalated']) for k in range(3)]
    loaded, placed = p.load_plan(world / 'plan')
    assert loaded == value and list(placed) == list(DEFAULT_PLAN)
    assert [placed[u] | {'layout': None} for u in placed] == [c | {'layout': None} for r in rows.values()
                                                              for c in r['candidates'][:1]]
    with pytest.raises(FileExistsError):
        p.plan(requests, DUMMY_PRIOR, placer, world / 'plan', 3)


def test_shard_persists_each_chunk_and_resumes_after_a_kill(world):
    requests, by_user, _ = inputs(DEFAULT_PLAN)
    records = [r for r in p.prepare(requests, DUMMY_PRIOR)[0] if r['uid'] != 'a']
    script = {u: v[2] for u, v in DEFAULT_PLAN.items()}
    shard = lambda out, o2, chunk=2: p.o2_shard(records, world / out, world / 'o2', 0, 1, load=lambda e: (o2, dict),
                                                 chunk=chunk)
    whole, identity = shard('whole', FakeO2(script, by_user))
    chunks = identity['sharding']['chunks']
    assert [len(c) for c in chunks] == [2, 2, 1] and sorted(sum(chunks, [])) == sorted(r['uid'] for r in records)
    assert identity['sharding'] | {'chunks': None} == {'shards': 1, 'index': 0, 'rule': p.SHARD_RULE, 'chunk_size': 2,
                                                       'chunks': None}
    with pytest.raises(Killed):                                     # killed inside chunk 1: chunk 0 is kept
        shard('shard', KilledO2(script, by_user, chunks[1]))
    assert [f.name for f in (world / 'shard/chunks').iterdir()] == ['0000.json']
    assert not (world / 'shard/COMPLETE.json').exists()
    again = FakeO2(script, by_user)
    resumed, same = shard('shard', again)
    assert same == identity and set(again.calls) == set(chunks[1] + chunks[2])     # chunk 1 redone whole, 0 not
    assert no_latency(resumed) == no_latency(whole)
    assert all((resumed[u]['shard'], resumed[u]['chunk']) == (0, k) for k, c in enumerate(chunks) for u in c)
    complete = json.loads((world / 'shard/COMPLETE.json').read_text())
    assert complete['identity_sha256'] == p.infer._hash(identity) and complete['chunks'] == 3
    p.o2_shard(records, world / 'shard', world / 'o2', 0, 1, load=lambda e: pytest.fail('engine loaded'), chunk=2)
    with pytest.raises(ValueError, match='resume identity mismatch'):
        shard('shard', FakeO2(script, by_user), chunk=3)


def test_sharded_run_equals_the_unsharded_run(world):
    system(world, DEFAULT_PLAN, escalate='o2')                                  # -> world/out
    requests, by_user, placer = inputs(DEFAULT_PLAN)
    value = p.plan(requests, DUMMY_PRIOR, placer, world / 'plan', 3)
    _, placed = p.load_plan(world / 'plan')
    records = p.prepare(requests, DUMMY_PRIOR)[0]
    o2 = FakeO2({u: v[2] for u, v in DEFAULT_PLAN.items()}, by_user)
    dirs = [world / f'shard-{k}' for k in range(3)]
    for k, d in enumerate(dirs):
        p.o2_shard([r for r in records if r['uid'] in value['escalated'] and p.shard_of(r['uid'], 3) == k], d,
                   world / 'o2', k, 3, load=lambda e: (o2, dict))
    with pytest.raises(ValueError, match='not the finished shard'):            # N = 2 is another assignment
        p.o2_merge([r for r in records if r['uid'] in value['escalated']], world / 'm', dirs[:2], world / 'o2')
    rerun = p.Placer(STUB, lambda records: pytest.fail('the placer stage reran'))
    p.run(requests, DUMMY_PRIOR, rerun, world / 'sharded', escalate='o2', api_cap=100, o2_model=world / 'o2', placed=placed,
          o2=lambda todo, out: p.o2_merge(todo, out, dirs, world / 'o2'))      # no GPU check, no engine
    for name in ('rooms.jsonl', 'roomgenbench.jsonl'):
        assert (world / 'sharded' / name).read_text() == (world / 'out' / name).read_text()
    plain, sharded = (json.loads((world / d / 'result.json').read_text()) for d in ('out', 'sharded'))
    sharding = sharded['escalation']['o2'].pop('sharding')
    assert (sharding['shards'], sharding['rule'], sharding['chunk_size']) == (3, p.SHARD_RULE, 256)
    assert [x['rooms'] for x in sharding['detail']] == value['shard_rooms']
    drop = lambda r: {k: v for k, v in r.items() if k not in ('code_sha256', 'runtime_seconds')}
    assert drop(plain) == drop(sharded)
    receipts = lambda d: {x.pop('uid'): x for x in map(json.loads, (world / d / 'o2/receipts.jsonl').read_text().splitlines())}
    tagged = no_latency(receipts('sharded'))
    assert list(tagged) == value['escalated'] and all(tagged[u]['shard'] == p.shard_of(u, 3) for u in tagged)
    assert {u: {k: v for k, v in b.items() if k not in ('shard', 'chunk')} for u, b in tagged.items()} == no_latency(
        receipts('out'))
