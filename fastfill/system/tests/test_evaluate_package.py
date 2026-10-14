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
