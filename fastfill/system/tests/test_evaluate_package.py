"""Amendment 022 §2/§3: evaluate.py and package.py on a fixture run root (real S1 and a real pipeline run with a
replayed placer, a fake gpt-6.1-sol and a fake O2; no GPU, network or API key), plus the Isambard scripts' guards."""
import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fastfill.comparison import dev_select as ds
from fastfill.comparison import geometry as g
from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import output_repair as rep
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import messages
from fastfill.comparison.size_prior import fit_prior
from fastfill.comparison.tests import test_test_apply_012 as T12
from fastfill.system.tests.test_pipeline import Killed, fingerprint
from fastfill.system import cli
from fastfill.system import evaluate as ev
from fastfill.system import package as pk
from fastfill.system import pipeline as pl

CODE = Path(__file__).resolve().parents[3]
ROOMS = {'a': ('IL3D_3dfront', 'h1', 'bedroom', 'good'), 'b': ('IL3D_3dfront', 'h1', 'bedroom', 'overlap'),
         'c': ('MultiScan', 'h2', 'office', 'float'), 'd': ('MultiScan', 'h3', 'office', 'invalid')}
COHORT = ['a', 'b', 'c']


def answer(bound_objects, kind):
    """good: apart; overlap: both wardrobes on one spot (S1 separates them); float: the bed 1 m up (S1 cannot fix)."""
    if kind == 'invalid':
        return 'no layout'
    out, k = [], 0
    for o in bound_objects:
        if o['category'] == 'wardrobe':
            pos, k = ([4.6, 2, 0] if kind == 'overlap' else [4.6, 1 + 2 * k, 0]), k + 1
        else:
            pos = [1.2, 2, 1.0 if kind == 'float' else 0]
        out.append({'id': o['id'], 'pos': pos, 'yaw': 0})
    return json.dumps({'placements': out})


@pytest.fixture
def world(tmp_path):
    run = tmp_path / 'run'
    rooms = [{**T12._room(u, s, grp, t), 'height': 3 + i / 100} for i, (u, (s, grp, t, _)) in enumerate(ROOMS.items())]
    # (distinct heights: a request's user turn identifies its room for the fake API and O2; a raw GT category the
    # request normalises, 'Bed' -> 'bed', as most eval-data rooms have)
    for x in rooms:
        x['objects'] = [{**o, 'category': 'Bed'} if o['category'] == 'bed' else o for o in x['objects']]
    prior = fit_prior([{'uid': f't{i}', 'room_type': 'bedroom',
                        'objects': [{'id': o['id'], 'category': o['category'], 'size': o['size']}
                                    for o in rooms[0]['objects']]} for i in range(5)])
    (run / 'main-data-v1').mkdir(parents=True)
    prior_path = run / 'main-data-v1' / 'size-priors.json'
    prior_path.write_text(json.dumps({'full_prior': prior}))
    joint = [{'uid': x['uid'], 'source': x['source'], 'group': x['group'], 'room_type': x['room_type'],
              'messages': messages(x, 'joint')[:2]} for x in rooms]
    placement = [pl.placement_request(r, pl.prior_sizes(r, prior)) for r in joint]
    data = run / 'eval-data-v1'
    T12._publish(data, {'rooms.jsonl': T12._jsonl(rooms), 'joint/requests.jsonl': T12._jsonl(joint),
                        'placement/requests.jsonl': T12._jsonl(placement)})
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    h.write_json(data / 'MANIFEST.json', {**manifest, 'expected_uids': list(ROOMS),
                                          'input_consumed_sha256': {'training': {'size-priors.json': h.sha256(prior_path)}}})
    data_sha = h.sha256(data / 'MANIFEST.json')
    # The placer's test inference (its receipt, as infer writes it) and the adapter it names.
    adapter, base = tmp_path / 'adapter', tmp_path / 'base'      # base: a private merged stage model (sft-merged)
    adapter.mkdir()
    base.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'lora')
    (adapter / 'adapter_config.json').write_text('{}')
    (base / 'model.safetensors').write_bytes(b'base')
    (base / 'config.json').write_text('{}')
    model = run / 'dpo-003' / 'placement' / 'dpo1-merged'
    identity = {'config': {'model': str(model.resolve()), 'lora': ds.EXACT, 'arm': 'placement'},
                'policy': {'training_base': str(base), 'training_base_sha256': infer._hash(h.artifact_manifest(base)),
                           'adapter': str(adapter), 'adapter_sha256': infer._hash(h.artifact_manifest(adapter))}}
    run_sha = infer._hash(identity)
    (run / 'infer-placement-005').mkdir()
    h.write_json(run / 'infer-placement-005' / 'run_receipt.json', {**identity, 'run_sha256': run_sha})
    records = [{'uid': x['uid'], 'raw': answer(x['objects'], ROOMS[x['uid']][3]), 'error': None,
                'reached_token_cap': False, 'stopped_on_eos': True} for x in rooms]
    geo = run / 'geometry-v5'
    rows = [g.score(x, q['messages'], r['raw'], 'placement') for x, q, r in zip(rooms, placement, records)]
    T12._publish(geo, {'placement-geometry.jsonl': T12._jsonl(rows), 'summary.json': json.dumps(
        {'status': 'GEOMETRY_COMPLETE', 'data_manifest_sha256': data_sha, 'inference_run_sha256': {'placement': run_sha}})})
    rep.run(data, geo / 'placement-geometry.jsonl', 'placement', run / 's1-012-test' / 'placement-rpo-dpo1')
    (run / 'llm-cohort-v1.json').write_text(json.dumps({'version': 1, 'data_manifest_sha256': data_sha,
                                                        'uids_sha256': L._digest(json.dumps(COHORT)), 'uids': COHORT}))
    # One agent row set (L single-shot, J input, its own sizes halved), repaired by S1 as amendment 012 did.
    agent = run / 'llm-eval-L'
    T12._publish(agent, {'llm_raw-geometry.jsonl': T12._jsonl(
        {**g.score(x, q['messages'], T12._answer(x, 'joint', .5), 'joint'), 'arm': 'llm_raw'}
        for x, q in zip(rooms, joint) if x['uid'] in COHORT)})
    rep.run(data, agent / 'llm_raw-geometry.jsonl', 'joint', run / 's1-012-test' / 'agent-L-raw',
            uids=run / 'llm-cohort-v1.json')
    env = tmp_path / 'api.env'
    env.write_text(f'OPENAI_API_KEY=sk-test\nOPENAI_BASE_URL=https://api.example.invalid/v1\nOPENAI_MODEL={L.EXPECTED_MODEL}\n')
    env.chmod(0o600)
    by_user = {q['messages'][1]['content']: x for x, q in zip(rooms, placement)}
    return SimpleNamespace(run=run, data=data, rooms=rooms, placement=h._by_uid(placement), records=records,
                           run_sha=run_sha, env=env, by_user=by_user, api=[], o2=[], identity=identity, base=base)


class FakeO2:
    """llm_open's vLLM LLM / tokenizer stand-in: a good layout for every conversation."""
    def __init__(self, w):
        self.w = w

    def get_tokenizer(self):
        return self

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False, **kwargs):
        return 'p' * 10

    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)

    def chat(self, conversations, params, use_tqdm=False, chat_template_kwargs=None):
        out = []
        for c in conversations:
            room = self.w.by_user[c[1]['content']]
            self.w.o2.append(room['uid'])
            text = answer(room['objects'], 'good')
            out.append(SimpleNamespace(prompt_token_ids=[1] * 10, outputs=[
                SimpleNamespace(text=text, token_ids=[1] * (len(text) + 1), finish_reason='stop')]))
        return out


@pytest.fixture
def system(world, monkeypatch):
    """step_system with the replayed placer, a fake API and a fake O2 behind the real pipeline.run."""
    def post(config, msgs, max_output_tokens, timeout):
        room = world.by_user[msgs[1]['content']]
        world.api.append(room['uid'])
        return {'text': answer(room['objects'], 'good'), 'status': 'completed', 'incomplete_reason': None,
                'usage': {'input_tokens': 100, 'output_tokens': 40, 'reasoning_tokens': 30}}
    run = pl.run
    monkeypatch.setattr(ev.pl, 'run', lambda *a, **k: run(*a, **k, post=post, o2_load=lambda e: (FakeO2(world), dict),
                                                          api_concurrency=1))
    identity = {'engine': 'replay:hf', 'model': 'base', 'adapter': 'adapter', 'adapter_sha256': 'f' * 64,
                'run_sha256': world.run_sha}
    monkeypatch.setattr(ev, 'replay', lambda p, data, run_sha: ev.replay_placer(world.records, world.placement, identity))
    (world.run / 'models' / 'Qwen3.8-27B').mkdir(parents=True)
    monkeypatch.setattr(ev.A, 'models_root', lambda run: world.run / 'models')
    (world.run / 'models' / 'Qwen3.8-27B' / 'config.json').write_text('{}')
    return world


def test_bound_chain_and_failing_set(world):
    p = ev.placer(world.run, 'interim')
    rows, info, run_sha = ev.bound(p, world.data)
    assert run_sha == world.run_sha and info['params']['max_trials'] == 2000
    fail = ev.failing(rows, COHORT)
    assert 'c' in fail and 'a' not in fail                       # the floating bed survives S1
    views = ev.stage_views(rows[0], world.rooms[0])
    assert views['raw']['checker'] and views['s1']['checker'] and views['raw']['complete']
    request = world.placement[rows[2]['uid']]   # final_view reproduces the S1 row's verdicts on its own layout
    final = ev.final_view((world.rooms[2], request, rows[2]['s1_layout']))
    assert [final[k] for k in ('checker', 'strict', 'noreq')] == [ev.stage_views(rows[2], world.rooms[2])['s1'][k]
                                                                  for k in ('checker', 'strict', 'noreq')]
    assert final['size_changed'] == 0
    (p['geometry'] / 'MANIFEST.json').write_text((p['geometry'] / 'MANIFEST.json').read_text() + ' ')
    with pytest.raises(ValueError, match='S1 of other geometry'):
        ev.bound(p, world.data)


def test_cap_rule_makes_a_row_incomplete(world):
    rows = ev.bound(ev.placer(world.run, 'interim'), world.data)[0]
    good = rows[0]
    assert good['complete'] and ev.stage_views(good, world.rooms[0])['s1']['checker'] and not ev.failing(rows, ['a'])
    capped = {**good, 'reached_token_cap': True, 'stopped_on_eos': False}   # amendment 005: incomplete, as the pipeline
    views = ev.stage_views(capped, world.rooms[0], cap_rule=True)
    assert not any(views[s][k] for s in ('raw', 's1') for k in ('complete', 'checker', 'strict', 'noreq'))
    assert ev.failing([capped], ['a']) == ['a']
    assert ev.stage_views({**good, 'reached_token_cap': True, 'stopped_on_eos': True}, world.rooms[0],
                          cap_rule=True)['s1']['checker']
    assert ev.stage_views(capped, world.rooms[0])['s1']['checker']    # agent rows: their protocols had no cap rule


def test_policy_of_hf_and_vllm_receipts(world, tmp_path):
    """HF (infer_shard) receipts carry the policy at the top level, a021 vLLM greedy receipts under 'model'."""
    policy = world.identity['policy']
    assert ev.receipt_policy(world.identity) == policy
    vllm = {'config': world.identity['config'], 'model': {'policy': policy}}
    assert ev.receipt_policy(vllm) == policy and ev.receipt_policy({'config': {}}) == {}
    inference = tmp_path / 'vllm-test'
    inference.mkdir()
    h.write_json(inference / 'run_receipt.json', {**vllm, 'run_sha256': infer._hash(vllm)})
    p = {**ev.placer(world.run, 'interim'), 'inference': inference}
    assert pk._policy(p, infer._hash(vllm))[1] == policy
    assert pk.training_base(world.run, policy) == h.artifact_manifest(world.base)
    public = ev.A.models_root(world.run) / 'Qwen3.8-27B'   # a public base under MODELS: listed, not copied
    assert pk.training_base(world.run, {**policy, 'training_base': str(public)}) is None
    with pytest.raises(ValueError, match='training base bytes'):
        pk.training_base(world.run, {**policy, 'training_base_sha256': 'b' * 64})


def test_replay_answers_only_the_recorded_request(world):
    placer = ev.replay_placer(world.records, world.placement, {'run_sha256': world.run_sha})
    request = world.placement['a']
    assert placer.generate([request])['a'] == {'raw': world.records[0]['raw'], 'error': None, 'cap_hit': False}
    other = {**request, 'messages': [request['messages'][0], {'role': 'user', 'content': '{}'}]}
    with pytest.raises(ValueError, match='another P request'):
        placer.generate([other])


def test_gate_refuses_before_any_call(system, monkeypatch):
    monkeypatch.setitem(ev.SCOPES['cohort'], 'api_rooms', 0)
    with pytest.raises(SystemExit, match='GATE REFUSED'):
        ev.step_system(system.run, 'interim', 'cohort', system.env)
    gate = json.loads(ev.paths(system.run, 'interim', 'cohort')['gate'].read_text())
    assert not gate['ok'] and gate['would_escalate'] >= 1 and system.api == []
    assert not ev.paths(system.run, 'interim', 'cohort')['system'].exists()


def test_gate_counts_earlier_attempts_and_the_global_cap(world, monkeypatch):
    rows = ev.bound(ev.placer(world.run, 'interim'), world.data)[0]
    ledger = world.run / 'system-022' / 'final' / 'cohort' / 'system.failed-1' / 'api' / 'calls.jsonl'
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"uid": "x"}\n' * 1000)
    verdict = ev.gate(world.run, rows, COHORT)
    assert verdict['ok'] and verdict['spent_022'] == 1000 and verdict['api_cap'] == 200
    monkeypatch.setattr(ev.sizers_020, 'calls_used', lambda run: {'llm-agent-001': 6600})
    assert not ev.gate(world.run, rows, COHORT)['ok']


def test_cohort_system_and_summary(system):
    result = ev.step_system(system.run, 'interim', 'cohort', system.env)
    out = ev.paths(system.run, 'interim', 'cohort')
    gate = json.loads(out['gate'].read_text())
    assert result['status'] == 'SYSTEM_COMPLETE' and gate['ok'] and gate['api_cap'] == 1200
    assert sorted(set(system.api)) == sorted(ev.failing(ev.bound(ev.placer(system.run, 'interim'), system.data)[0], COHORT))
    value = ev.step_summarize(system.run, 'interim', 'cohort', repeats=20)
    stages = value['levels']['all']['system']
    assert stages['raw']['checker_rate'] <= stages['s1']['checker_rate'] <= stages['system']['checker_rate'] == 1.0
    esc = value['escalation']
    assert esc['escalated'] == esc['api']['rooms'] == esc['vs_s1_rows']['s1_failing'] == len(set(system.api))
    assert esc['api']['tokens']['input_tokens'] == 100 * esc['api']['calls'] and esc['api']['cost_usd'] == 'not measured'
    stages = {json.loads(x)['uid']: json.loads(x)['stage'] for x in (out['system'] / 'rooms.jsonl').read_text().splitlines()}
    assert value['final_equals_s1_rows'] == sum(s == 'placer' for s in stages.values()) >= 2   # 'Bed' vs 'bed'
    assert value['size_changed_objects'] == 0
    assert 'L-raw' in value['agents'] and 'O2-P-raw' in value['agents_missing']
    assert any(c['comparison'] == 'system - L-raw [s1]' for c in value['contrasts'])
    assert {'objects<=20', 'family:bedroom', 'real'} <= set(value['levels'])
    text = (out['eval'] / 'report.md').read_text()
    assert '含 API 访问' in text and '网格级碰撞（修订 019）：未测量' in text
    h.verify_artifacts(out['eval'], json.loads((out['eval'] / 'MANIFEST.json').read_text())['files'])
    for name, key in (('levels.csv', lambda r: r['system'] == 'system' and r['stage'] == 'system'),
                      ('contrasts.csv', lambda r: r['comparison'].startswith('system'))):
        with (out['eval'] / name).open() as stream:
            rows = list(csv.DictReader(stream))
        assert {r['label'] == ev.LABEL['cohort'] for r in rows} == {True, False}
        assert all((r['label'] == ev.LABEL['cohort']) == key(r) for r in rows), name
    identity, rows = ev.load_pipeline(out['system'])   # fail closed above the caps
    with pytest.raises(ValueError, match='fail closed'):
        ev.evaluate({x['uid']: x for x in system.rooms if x['uid'] in COHORT}, system.placement,
                    [x for x in ev.bound(ev.placer(system.run, 'interim'), system.data)[0] if x['uid'] in COHORT],
                    ({**identity, 'api_calls': 1201}, rows), 'cohort', repeats=2)
    with pytest.raises(FileExistsError):
        ev.step_summarize(system.run, 'interim', 'cohort', repeats=20)


def test_test_scope_escalates_to_o2_only(system):
    ev.step_system(system.run, 'interim', 'test')
    value = ev.step_summarize(system.run, 'interim', 'test', repeats=20)
    esc = value['escalation']
    assert system.api == [] and esc['api']['calls'] == 0 and esc['steps']['o2']['rooms'] == esc['escalated'] >= 2
    assert set(system.o2) >= {'c', 'd'} and value['levels']['all']['system']['system']['complete_rate'] == 1.0
    assert value['agents'] == {} and '仅 O2' in value['label_zh']


TEST_GOLDEN = {      # step_system test scope, captured before the sharded O2 fallback existed (commit 23a51728d)
    'files': ['o2/invocations.jsonl', 'o2/receipts.jsonl', 'report.md', 'result.json', 'roomgenbench.jsonl',
              'rooms.jsonl'], 'result': 'cf922868f6dda7643efff38cfef23741bf465186e62a03ad70b4de5531300d25',
    'rooms': '2151372350d1d51d99c84b45708a8ece78fa845d1f299eb99c73f76608e979e5',
    'roomgenbench': 'fbe279e8632fa45caada1077a3b7236689f649ed08c6f6e9cc2ab96d989a1a49',
    'o2': '1ef68aa2258d978522655ae28f3cd3f542824786bac2a37c75b4875283e338e0'}
TEST_REQUESTS = '9d874f32d64be5e41dec5944c36eec242bb449a59244382067800463713849f7'


def test_default_test_scope_outputs_are_unchanged(system):
    ev.step_system(system.run, 'interim', 'test')
    out = ev.paths(system.run, 'interim', 'test')
    root = system.run.parent
    assert fingerprint(out['system'], str(root.resolve()), str(root), system.run_sha) == TEST_GOLDEN
    assert h.sha256(out['requests']) == TEST_REQUESTS


def test_sharded_test_scope_fallback(system, monkeypatch):
    """plan -> shards (one killed, resumed) -> finalize -> summarize --sharded = the unsharded test run, in test-sharded."""
    ev.step_system(system.run, 'interim', 'test')
    plain = ev.step_summarize(system.run, 'interim', 'test', repeats=20)
    a, b = ev.paths(system.run, 'interim', 'test'), ev.paths(system.run, 'interim', 'test', sharded=True)
    before = {f: f.read_bytes() for f in a['system'].parent.rglob('*') if f.is_file()}     # .../test
    fake = {'o2': FakeO2(system)}
    shard = pl.o2_shard
    monkeypatch.setattr(ev.pl, 'o2_shard', lambda *x, **k: shard(*x, **k, load=lambda e: (fake['o2'], dict)))
    value = ev.step_plan(system.run, 'interim', 3)
    rows = [json.loads(x) for x in (a['system'] / 'rooms.jsonl').read_text().splitlines()]
    assert value['escalated'] == [x['uid'] for x in rows if x['needs_escalation']] and len(value['escalated']) >= 2
    assert ev.step_plan(system.run, 'interim', 3)['status'] == 'PLAN_REUSED'
    with pytest.raises(ValueError, match='another N'):
        ev.step_plan(system.run, 'interim', 2)
    busy = next(k for k in range(3) if value['shard_rooms'][k])

    class Dying(FakeO2):
        def chat(self, *x, **k):
            raise Killed
    fake['o2'] = Dying(system)
    with pytest.raises(Killed):
        ev.step_shard(system.run, 'interim', busy)
    fake['o2'] = FakeO2(system)
    for k in range(3):
        if k != busy:
            ev.step_shard(system.run, 'interim', k)
    with pytest.raises(FileNotFoundError, match='unfinished shards'):
        ev.step_finalize(system.run, 'interim')
    assert not b['system'].exists()
    ev.step_shard(system.run, 'interim', busy)                         # the resubmitted shard resumes
    ev.step_finalize(system.run, 'interim')
    for name in ('rooms.jsonl', 'roomgenbench.jsonl'):
        assert (a['system'] / name).read_text() == (b['system'] / name).read_text()
    assert a['requests'].read_text() == b['requests'].read_text()
    x, y = (json.loads((d['system'] / 'result.json').read_text()) for d in (a, b))
    sharding = y['escalation']['o2'].pop('sharding')
    assert (sharding['shards'], sharding['rule'], sharding['plan_sha256']) == (3, pl.SHARD_RULE,
                                                                               h.sha256(b['plan'] / 'plan.json'))
    drop = lambda r: {k: v for k, v in r.items() if k not in ('code_sha256', 'runtime_seconds')}
    assert drop(x) == drop(y)
    value = ev.step_summarize(system.run, 'interim', 'test', repeats=20, sharded=True)
    assert value['levels'] == plain['levels'] and value['contrasts'] == plain['contrasts']
    steps = lambda r: {s: {k: v for k, v in x.items() if k != 'latency_s'} for s, x in r['escalation']['steps'].items()}
    assert steps(value) == steps(plain) and value['escalation']['final_stage'] == plain['escalation']['final_stage']
    assert value['inputs']['system']['o2_sharding'] == {**sharding, 'plan_sha256': h.sha256(b['plan'] / 'plan.json')}
    assert 'o2_sharding' not in plain['inputs']['system'] and (b['eval'] / 'MANIFEST.json').exists()
    assert {f: f.read_bytes() for f in a['system'].parent.rglob('*') if f.is_file()} == before
    with pytest.raises(FileExistsError):
        ev.step_shard(system.run, 'interim', 0)
    with pytest.raises(FileExistsError):
        ev.step_finalize(system.run, 'interim')
    parse = lambda *words: ev.parse_args(['--run-root', 'r', '--mode', 'final', *words])
    assert (parse('plan', '--shards', '4').shards, parse('shard', '--index', '3').index) == (4, 3)
    assert parse('summarize', '--scope', 'test', '--sharded').sharded
    for words in (('plan',), ('shard',), ('finalize', '--index', '1'), ('summarize', '--scope', 'cohort', '--sharded'),
                  ('system', '--scope', 'test', '--sharded'), ('plan', '--shards', '2', '--scope', 'cohort')):
        with pytest.raises(SystemExit):
            parse(*words)


def test_strict_sharded_test_scope(system, monkeypatch):
    """Amendment 027: plan / shard / finalize / summarize --trigger strict under test-strict-sharded, paired with the
    checker-trigger test-sharded run; never touches test-sharded."""
    shard = pl.o2_shard
    monkeypatch.setattr(ev.pl, 'o2_shard', lambda *x, **k: shard(*x, **k, load=lambda e: (FakeO2(system), dict)))
    checker = ev.step_plan(system.run, 'interim', 2)
    for k in range(2):
        ev.step_shard(system.run, 'interim', k)
    ev.step_finalize(system.run, 'interim')
    a, b = ev.paths(system.run, 'interim', 'test', True), ev.paths(system.run, 'interim', 'test', True, 'strict')
    assert (b['plan'].parent.name, ev.paths(system.run, 'interim', 'test')['eval'].parent.name) == (
        'test-strict-sharded', 'test')
    before = {f: f.read_bytes() for f in a['plan'].parent.rglob('*') if f.is_file()}
    value = ev.step_plan(system.run, 'interim', 2, trigger='strict')
    s1_rows = ev.bound(ev.placer(system.run, 'interim'), system.data)[0]
    assert value['escalation_trigger'] == 'strict' and value['escalated'] == ev.failing(s1_rows, list(ROOMS), 'strict')
    assert set(checker['escalated']) <= set(value['escalated'])     # here equal (a and b pass strict admission too)
    assert ev.step_plan(system.run, 'interim', 2, trigger='strict')['status'] == 'PLAN_REUSED'
    with pytest.raises(FileNotFoundError, match='unfinished shards'):
        ev.step_finalize(system.run, 'interim', trigger='strict')
    for k in range(2):
        ev.step_shard(system.run, 'interim', k, trigger='strict')
    result = ev.step_finalize(system.run, 'interim', trigger='strict')
    rows = [json.loads(x) for x in (b['system'] / 'rooms.jsonl').read_text().splitlines()]
    assert result['escalation_trigger'] == 'strict'
    assert [x['uid'] for x in rows if x['needs_escalation']] == value['escalated']
    a['system'].rename(a['root'] / 'aside')                         # §3 pairs with the checker run: required
    with pytest.raises(FileNotFoundError, match='checker-trigger run'):
        ev.step_summarize(system.run, 'interim', 'test', repeats=20, sharded=True, trigger='strict')
    (a['root'] / 'aside').rename(a['system'])
    summary = ev.step_summarize(system.run, 'interim', 'test', repeats=20, sharded=True, trigger='strict')
    o2 = summary['o2_run']                                          # §2: rounds, process time, GPU-h
    assert (o2['rooms'], o2['rounds'], o2['invocations'], len(o2['shard_seconds'])) == (
        len(value['escalated']), summary['escalation']['steps']['o2']['calls'], 2, 2)
    assert o2['wall_seconds'] == max(o2['shard_seconds']) and o2['gpu_hours'] == sum(o2['shard_seconds']) * 2 / 3600
    assert o2['unanswered_rooms'] == 0 and 0 <= o2['repaired_rooms'] <= o2['rooms']
    assert summary['escalation_trigger'] == 'strict' and summary['inputs']['system']['escalation_trigger'] == 'strict'
    assert '严格准入失败' in summary['label_zh'] and 'strict-admission' in summary['label']
    vs = summary['escalation']['vs_s1_rows']
    assert (vs['s1_failing'], vs['escalated_not_failing'], vs['failing_not_escalated']) == (
        len(value['escalated']), 0, 0)
    paired = {c['subset']: c for c in summary['contrasts'] if c['comparison'] == f'system - {ev.CHECKER_RUN}'}
    assert paired['all']['rooms'] == len(ROOMS) and set(paired) <= set(ev.CONTRAST_SUBSETS)
    assert paired['all']['fields']['strict']['estimate'] == 0          # same escalated rooms and answers here
    assert summary['inputs']['checker_trigger_system']['rooms_sha256'] == h.sha256(a['system'] / 'rooms.jsonl')
    pred = summary['replay_prediction']
    m = pred['measured']
    assert pred['predicted'] == ev.PREDICTED and m['rooms'] == len(COHORT) == summary['inputs']['cohort_rooms']
    assert m['escalated_share'] == sum(x['needs_escalation'] for x in rows if x['uid'] in COHORT) / len(COHORT)
    text = (b['eval'] / 'report.md').read_text()
    assert '严格触发' in text and f'system - {ev.CHECKER_RUN}' in text and '离线回放预测 vs 实测' in text
    assert f"轮次 {o2['rounds']}" in text
    assert {f: f.read_bytes() for f in a['plan'].parent.rglob('*') if f.is_file()} == before
    parse = lambda *words: ev.parse_args(['--run-root', 'r', '--mode', 'final', *words])
    assert parse('plan', '--shards', '8', '--trigger', 'strict').trigger == 'strict'
    assert parse('summarize', '--scope', 'test', '--sharded', '--trigger', 'strict').trigger == 'strict'
    for words in (('summarize', '--scope', 'test', '--trigger', 'strict'), ('plan', '--shards', '2', '--trigger', 'x'),
                  ('system', '--scope', 'test', '--trigger', 'strict')):
        with pytest.raises(SystemExit):
            parse(*words)


def test_interim_package_and_assembly(world, tmp_path):
    value = pk.build(world.run, 'interim', files=lambda: ['fastfill/system/package.py'])
    out = ev.paths(world.run, 'interim')['package']
    card = (out / 'MODEL_CARD.md').read_text()
    assert value['release'] == 'interim' and 'interim' in card and 'not measured' in card and '## English' in card
    assert value['placer']['adapter']['sha256'] == infer._hash(h.artifact_manifest(world.run.parent / 'adapter'))
    assert (out / 'SHA256SUMS').read_text().count('\n') == len(value['files_to_copy'])
    assert value['measured'] == {'cohort': None, 'test': None, 'apply_012': None}
    receipt = pk.delivery.assemble(out, tmp_path / 'assembled')
    assert receipt['status'] == 'DELIVERY_PACKAGE_ASSEMBLED'
    assert (tmp_path / 'assembled' / 'placer' / 'adapter' / 'adapter_model.safetensors').read_bytes() == b'lora'
    assert (tmp_path / 'assembled' / 'placer' / 'training-base' / 'model.safetensors').read_bytes() == b'base'
    assert value['placer']['training_base']['copied'] and '--model placer/training-base' in value['cli']
    assert 'evaluated test rows from hf' in card and '评估所用测试行的引擎：hf' in card
    with pytest.raises(FileExistsError):
        pk.build(world.run, 'interim', files=lambda: ['fastfill/system/package.py'])


def test_package_card_takes_measured_numbers(system):
    ev.step_system(system.run, 'interim', 'cohort', system.env)
    result = ev.step_summarize(system.run, 'interim', 'cohort', repeats=20)
    value = pk.build(system.run, 'interim', files=lambda: ['fastfill/system/package.py'])
    m = value['measured']['cohort']
    assert m['stages']['system']['checker_rate'] == result['levels']['all']['system']['system']['checker_rate']
    card = (ev.paths(system.run, 'interim')['package'] / 'MODEL_CARD.md').read_text()
    assert pk.delivery._fmt(m['stages']['raw']['checker_rate']) in card
    assert any(c['dest'] == 'evaluation/cohort/results.json' for c in value['files_to_copy'])


def test_usage_parses_with_the_cli(world, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.pipeline, 'hf_placer', lambda model, adapter: seen.update(model=model, adapter=adapter))
    monkeypatch.setattr(cli.pipeline, 'load_requests', lambda path: [])
    monkeypatch.setattr(cli.pipeline, 'load_prior', lambda path, sha: (path, sha))
    monkeypatch.setattr(cli.pipeline, 'run', lambda *a, **k: seen.update(k) or dict.fromkeys(
        ('status', 'rooms', 'stages', 'would_escalate')))
    words = shlex.split(pk.usage('hf', 'placer/training-base', ev.prior_table(world.run), Path('/m/Qwen3.8-27B')))
    assert words[:4] == ['PYTHONPATH=code', 'python', '-m', 'fastfill.system.cli']
    assert cli.main(words[4:]) == 0
    assert (seen['escalate'], seen['api_cap'], seen['api_max_rooms']) == ('l+o2', 1200, 600)
    assert seen['model'] == Path('placer/training-base') and seen['adapter'] == Path('placer/adapter')


CLOSURE_RUN = """
import json, sys
from pathlib import Path
from fastfill.comparison.size_prior import fit_prior
from fastfill.system import pipeline as pl
out, room = Path(sys.argv[1]), {'uid': 'r1', 'room_type': 'bedroom', 'boundary': [[0, 0], [4, 0], [4, 3], [0, 3]],
    'height': 2.8, 'furniture': {'bed': 1, 'nightstand': 1}}
prior = fit_prior([{'uid': f't{i}', 'room_type': 'bedroom', 'objects': [
    {'id': 'b', 'category': 'bed', 'size': [2, 1.6, .5]}, {'id': 'n', 'category': 'nightstand', 'size': [.5, .4, .5]}]}
    for i in range(5)])
(out.parent / 'rooms.jsonl').write_text(json.dumps(room) + chr(10))
def generate(records):
    user = json.loads(records[0]['messages'][1]['content'])
    return {'r1': {'raw': json.dumps({'placements': [{'id': o['id'], 'pos': [1 + 2 * k, 1.5, 0], 'yaw': 0}
            for k, o in enumerate(user['objects'])]}), 'error': None, 'cap_hit': False}}
pl.run(pl.load_requests(out.parent / 'rooms.jsonl'), (prior, {'path': 'p', 'sha256': '0' * 64}), pl.Placer({}, generate),
       out)
print((out / 'roomgenbench.jsonl').read_text())
"""


def test_code_closure_runs_the_system(tmp_path):
    """The packaged code alone (a copy of the closure, nothing else importable) runs placer -> S1 -> RoomGenBench."""
    files = pk.code_files()
    assert {'fastfill/system/pipeline.py', 'fastfill/system/cli.py', 'fastfill/comparison/output_repair.py',
            'fastfill/comparison/llm_agent.py', 'fastfill/v2/direct_layout.py',
            'fastfill/v2/validation.py'} <= set(files)
    code = tmp_path / 'code'
    for name in files:
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE / name, code / name)
    env = {**os.environ, 'PYTHONPATH': str(code)}
    result = subprocess.run([sys.executable, '-B', '-c', CLOSURE_RUN, str(tmp_path / 'out')], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    row = json.loads(result.stdout)
    assert row['error'] is None and row['stage'] == 'placer' and row['scene']


def test_scripts_refuse_bad_arguments(tmp_path):
    sbatch, submit = CODE / 'fastfill/system/isambard_022.sbatch', CODE / 'fastfill/system/submit_022.sh'
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM')}
    call = lambda *args: subprocess.run(['bash', *map(str, args)], capture_output=True, text=True, env=env)
    for args in (('system', 'final', 'cohort', '0'), ('s1', 'interim'), ('summarize', 'final', 'both'),
                 ('package', 'final', 'x'), ('system', 'draft', 'test', '6')):
        assert call(sbatch, CODE, *args).returncode == 2, args
    assert 'SLURM_JOB_ID' in call(sbatch, CODE, 'summarize', 'final', 'cohort').stderr
    assert call(submit).returncode == 2 and call(submit, CODE, 'everything').returncode == 2
    result = call(submit, tmp_path, 'cohort')
    assert result.returncode == 1 and 'nothing submitted' in result.stderr
    for args in (('plan', 'final'), ('plan', 'final', '0'), ('shard', 'final', '1'), ('shard', 'final', 'x', '12'),
                 ('shard', 'final', '1', '25'), ('finalize', 'final', '1'), ('summarize', 'final', 'test-shard')):
        assert call(sbatch, CODE, *args).returncode == 2, args
    for args in (('plan', 'final', '4'), ('shard', 'final', '3', '12'), ('finalize', 'final'),
                 ('summarize', 'final', 'test-sharded')):
        assert 'SLURM_JOB_ID' in call(sbatch, CODE, *args).stderr, args
    for args in (('test-sharded', '0'), ('test-sharded', 'x'), ('test-sharded', '17'), ('test', '4')):
        assert call(submit, CODE, *args).returncode == 2, args
    result = call(submit, tmp_path, 'test-sharded', '4')
    assert result.returncode == 1 and 'nothing submitted' in result.stderr
    for args in (('plan', 'final', '8', 'loose'), ('plan', 'final', '8', 'strict', 'x'),
                 ('shard', 'final', '1', '12', 'x'), ('finalize', 'final', 'x'), ('summarize', 'final', 'test-strict')):
        assert call(sbatch, CODE, *args).returncode == 2, args
    for args in (('plan', 'final', '8', 'strict'), ('shard', 'final', '7', '12', 'strict'),
                 ('finalize', 'final', 'strict'), ('summarize', 'final', 'test-strict-sharded')):
        assert 'SLURM_JOB_ID' in call(sbatch, CODE, *args).stderr, args
    for args in (('test-strict-sharded', '0'), ('test-strict-sharded', 'x'), ('test-strict-sharded', '17')):
        assert call(submit, CODE, *args).returncode == 2, args


def test_submit_strict_sharded_chain(tmp_path):
    """submit_022.sh test-strict-sharded with fake sbatch / squeue / scontrol (a copy with PROJECT_ROOT under tmp)."""
    code, project, bin_ = tmp_path / 'code', tmp_path / 'project', tmp_path / 'bin'
    for name in ('pipeline.py', 'cli.py', 'evaluate.py', 'package.py', 'isambard_022.sbatch',
                 '../../docs/research/20261013-unknown-size-comparison/amendment-022-delivered-system-20261014.md'):
        (code / 'fastfill/system' / name).parent.mkdir(parents=True, exist_ok=True)
        (code / 'fastfill/system' / name).write_text('')
    text = (CODE / 'fastfill/system/submit_022.sh').read_text()
    root = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill'
    assert text.count(root) == 1
    (code / 'fastfill/system/submit_022.sh').write_text(text.replace(root, f'PROJECT_ROOT={project}'))
    run, s1 = project / 'runs/unknownsize-20261013', tmp_path / 's1'
    for name in ('eval-data-v1/MANIFEST.json', 'main-data-v1/size-priors.json', 'llm-cohort-v1.json',
                 'a021/final-selection.json', '../../../s1/MANIFEST.json'):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text('{}')
    log, queue = tmp_path / 'sbatch.log', tmp_path / 'queue'
    fakes = {project / 'env/bin/python': f'case "$*" in *ev.placer*) echo {s1}; exit 0 ;; esac\n'
                                          f'exec {sys.executable} "$@"',
             bin_ / 'sbatch': f'n=$(($(cat {tmp_path}/n 2>/dev/null || echo 100) + 1)); echo $n > {tmp_path}/n\n'
                              f'echo "$*" >> {log}; echo $n',
             bin_ / 'scontrol': f'echo "scontrol $*" >> {log}',
             bin_ / 'squeue': 'name=""; while test $# -gt 0; do test "$1" != -n || name="$2"; shift; done\n'
                              f'while read -r id job; do if test -z "$name"; then echo "$id $job"; '
                              f'elif test "$job" = "$name"; then echo "$id"; fi; done < {queue}; exit 0'}
    for path, body in fakes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/bash\n' + body + '\n')
        path.chmod(0o755)
    queue.write_text('')
    env = {**{k: v for k, v in os.environ.items() if not k.startswith('SLURM')}, 'USER': 'u',
           'PATH': f"{bin_}:{os.environ['PATH']}"}
    submit = lambda *args: subprocess.run(['bash', str(code / 'fastfill/system/submit_022.sh'), str(code), *args],
                                          capture_output=True, text=True, env=env)
    result = submit('test-strict-sharded')
    assert result.returncode == 0, result.stderr
    jobs = log.read_text().splitlines()
    name = lambda line: line.split('--job-name=')[1].split()[0]
    assert [name(x) for x in jobs[:-1]] == ['unknownsize-022-test-strict-sharded-' + s for s in (
        'plan', *(f'shard{k}' for k in range(8)), 'finalize', 'summarize')] and jobs[-1] == 'scontrol release 101'
    assert '--hold' in jobs[0] and jobs[0].endswith('plan final 8 strict')
    assert all('afterok:101 ' in x and x.endswith(f'shard final {k} 12 strict') for k, x in enumerate(jobs[1:9]))
    assert 'afterok:101:' + ':'.join(map(str, range(102, 110))) in jobs[9] and jobs[9].endswith('finalize final strict')
    assert 'afterok:110 ' in jobs[10] and jobs[10].endswith('summarize final test-strict-sharded')
    assert '8 shards x 2 GPUs x 12 h = 192 GPU-h' in result.stdout
    log.write_text('')                                                        # the checker chain is as before
    assert submit('test-sharded').returncode == 0
    jobs = log.read_text().splitlines()
    assert [name(x) for x in jobs[:-1]] == ['unknownsize-022-test-sharded-' + s for s in (
        'plan', *(f'shard{k}' for k in range(4)), 'finalize', 'summarize')]
    assert [x.rsplit('isambard_022.sbatch ', 1)[1].split(' ', 1)[1] for x in jobs[:-1]] == [
        'plan final 4', *(f'shard final {k} 12' for k in range(4)), 'finalize final', 'summarize final test-sharded']
    t = run / 'system-022/final/test-strict-sharded'                         # resubmit: only unfinished shards
    (t / 'plan').mkdir(parents=True)
    (t / 'plan/plan.json').write_text(json.dumps({'shards': 8}))
    for k in (0, 3):
        (t / f'shards/shard-{k}').mkdir(parents=True)
        (t / f'shards/shard-{k}/COMPLETE.json').write_text('{}')
    log.write_text('')
    assert submit('test-strict-sharded').returncode == 0
    assert [name(x) for x in log.read_text().splitlines() if 'shard final' in x] == [
        f'unknownsize-022-test-strict-sharded-shard{k}' for k in (1, 2, 4, 5, 6, 7)]
    log.write_text('')
    for args, queued, reason in ((('test-strict-sharded', '4'), '', 'made for 8 shards'),
                                 (('test-strict-sharded',), '7 unknownsize-022-test-strict-sharded-shard1\n',
                                  'already in Slurm (scancel 7)')):
        queue.write_text(queued)
        result = submit(*args)
        assert result.returncode == 1 and reason in result.stderr and 'nothing submitted' in result.stderr, args
    assert log.read_text() == ''
    queue.write_text('8 unknownsize-022-test-sharded-shard1\n')                # the checker chain does not block
    assert submit('test-strict-sharded').returncode == 0
    (t / 'system').mkdir()
    result = submit('test-strict-sharded')
    assert result.returncode == 1 and 'test-strict-sharded/system exists' in result.stderr


def test_package_uses_sharded_test_eval_only_when_unsharded_is_absent(tmp_path):
    from fastfill.system import package as pk
    from fastfill.system import evaluate as ev
    plain, sharded = ev.paths(tmp_path, 'final', 'test')['eval'], ev.paths(tmp_path, 'final', 'test', sharded=True)['eval']
    assert pk._eval_root(tmp_path, 'final', 'test') == plain
    sharded.mkdir(parents=True)
    (sharded / 'MANIFEST.json').write_text('{}')
    assert pk._eval_root(tmp_path, 'final', 'test') == sharded
    assert pk._eval_root(tmp_path, 'final', 'cohort') == ev.paths(tmp_path, 'final', 'cohort')['eval']
    plain.mkdir(parents=True)
    (plain / 'MANIFEST.json').write_text('{}')
    assert pk._eval_root(tmp_path, 'final', 'test') == plain
