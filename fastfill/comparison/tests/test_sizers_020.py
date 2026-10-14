"""Amendment 020 sizers: requests, parser, rooms, API cap, resume, O2 shards, estimates format, report, submit chain."""
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fastfill.comparison import contract
from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import llm_open as O
from fastfill.comparison import optiscene_eval as oe
from fastfill.comparison import sizers_020 as S
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_llm_agent as base

E2 = ['holo-k-0', 'il-bed-0', 'il-bed-1', 'il-bed-2']
E3 = [f'x-{i:02d}' for i in range(10)]


def _door(room):
    room['fixed'] = [{'id': 'fixed_0001', 'category': 'door', 'size': [0.1, 0.9, 2.0], 'pos': [0.05, 1.0, 0.0],
                      'yaw': 0.0, 'parent': None}]
    return room


def _plan(root, name, uids, data):
    rows = [{'uid': u, 'population': name, 'mode': m, 'key': f'{u}|{m}'} for u in uids for m in ('gt', 'prior')]
    oe._publish_files(root, {'requests.jsonl': oe._jsonl_bytes(rows)},
                      {'plan': name, 'eval_data_manifest_sha256': oe._sha(data / 'MANIFEST.json')})
    return root


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(L, 'BACKOFF_SECONDS', 0)
    data = base.eval_data(tmp_path / 'data', mutate=_door)
    base.env_file(tmp_path)
    run = tmp_path / 'run'
    run.mkdir()
    rooms = run / 'a020-rooms-v1.json'
    assert S.main(['rooms', '--data', str(data), '--e2', str(_plan(tmp_path / 'e2', 'e2', E2, data)),
                   '--e3', str(_plan(tmp_path / 'e3', 'e3', E3, data)), '--out', str(rooms)]) == 0
    labels = {r['uid']: r for r in map(json.loads, (data / 'rooms.jsonl').read_text().splitlines())}
    joint = {r['uid']: r for r in map(json.loads, (data / 'joint/requests.jsonl').read_text().splitlines())}
    return SimpleNamespace(tmp=tmp_path, data=data, run=run, rooms=rooms, labels=labels, joint=joint)


def _answer(world, uid, scale=1.1):
    return json.dumps({o['id']: [round(v * scale, 3) for v in o['size']] for o in world.labels[uid]['objects']})


class FakeApi:
    """Per-UID scripted texts or ApiErrors; 'valid' = GT sizes x 1.1."""
    def __init__(self, world, script=None):
        self.world, self.script, self.calls = world, script or {}, []
        self.requests = {u: S.sizer_request(r)['messages'] for u, r in world.joint.items()}

    def __call__(self, config, messages, max_output_tokens, timeout):
        uid = next(u for u, m in self.requests.items() if messages[:2] == m)
        self.calls.append((uid, messages))
        queue = self.script.setdefault(uid, [])
        item = queue.pop(0) if queue else 'valid'
        if isinstance(item, Exception):
            raise item
        return {'text': _answer(self.world, uid) if item == 'valid' else item, 'status': 'completed',
                'incomplete_reason': None, 'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}


def run_api(world, fake, out='sizes-020-L', *extra):
    return S.main(['run-api', '--data', str(world.data), '--rooms', str(world.rooms), '--run-root', str(world.run),
                   '--out', str(world.run / out), '--env-file', str(world.tmp / 'api.env'), '--concurrency', '1',
                   *extra], post=fake)


def test_request_keeps_the_room_drops_placement_and_quotes_j_size_text(world):
    j_system = contract._SYSTEM['joint']
    for text in (S.SIZE_DEFINITION, S.FRAME, S.POSE):
        assert text in j_system and text in S.SYSTEM  # verbatim J wording: same axis convention
    for word in ('Place', 'placements', 'on is the supporting', 'fit', 'shrink', 'Predict'):
        assert word not in S.SYSTEM
    for uid, record in world.joint.items():
        assert record['messages'][0]['content'] == j_system
        request = S.sizer_request(record)
        user = json.loads(request['messages'][1]['content'])
        assert request['messages'][0]['content'] == S.SYSTEM
        assert request['messages'][1]['content'] == record['messages'][1]['content']  # J's user JSON, unchanged
        assert all(set(o) == {'id', 'category'} for o in user['objects'])
        assert user['fixed'][0]['category'] == 'door' and set(user) <= set(S.REQUEST_KEYS)
        assert S.object_ids(request) == [o['id'] for o in world.labels[uid]['objects']]


def test_parse_sizes_validation():
    ids = ['obj_0001', 'obj_0002']
    ok = S.parse_sizes('{"obj_0002":[1,2,3],"obj_0001":[0.5,0.5,0.5]}', ids)
    assert ok == {'valid': True, 'error': None, 'sizes': {'obj_0001': [.5, .5, .5], 'obj_0002': [1., 2., 3.]}}
    cases = {'': S.NOT_JSON, 'sizes: {}': S.NOT_JSON, '{"obj_0001":[1,1,1],"obj_0001":[1,1,1]}': S.NOT_JSON,
             '{"obj_0001":[1,1,1],"obj_0002":[1,1,NaN]}': S.NOT_JSON, '[1,2,3]': S.NOT_OBJECT,
             '{"obj_0001":[1,1,1]}': S.WRONG_IDS, '{"sizes":{"obj_0001":[1,1,1],"obj_0002":[1,1,1]}}': S.WRONG_IDS,
             '{"obj_0001":[1,1,1],"obj_0002":[1,1,1],"obj_0003":[1,1,1]}': S.WRONG_IDS}
    for value in ('[0,1,1]', '[-1,1,1]', '[1,1]', '[1,1,1,1]', '[true,1,1]', '["1",1,1]', '[1e999,1,1]', '[1000,1,1]',
                  '{"w":1}'):
        cases['{"obj_0001":[1,1,1],"obj_0002":' + value + '}'] = S.BAD_SIZE
    for text, error in cases.items():
        assert S.parse_sizes(text, ids) == {'valid': False, 'error': error, 'sizes': None}, text
    assert S.parse_sizes(None, ids)['error'] == S.NOT_JSON


def test_rooms_file_union_families_rebuild_and_overlap(world):
    rooms = json.loads(world.rooms.read_text())
    expected = json.loads((world.data / 'MANIFEST.json').read_text())['expected_uids']
    assert rooms['uids'] == [u for u in expected if u in set(E2 + E3)]
    assert rooms['counts'] == {'total': 14, 'e2': 4, 'e3': 10, 'by_family': {'bedroom': 3, 'kitchen': 1,
                                                                             'study_office': 10}}
    assert rooms['population']['x-03'] == 'e3' and rooms['family']['il-bed-1'] == 'bedroom'
    argv = ['rooms', '--data', str(world.data), '--e2', str(world.tmp / 'e2'), '--e3', str(world.tmp / 'e3'),
            '--out', str(world.rooms)]
    assert S.main(argv) == 0  # an existing file equal to its rebuild is accepted
    with pytest.raises(ValueError, match='overlap'):
        S.build_rooms(world.data, world.tmp / 'e2', _plan(world.tmp / 'e3b', 'e3', E3 + E2[:1], world.data))
    with pytest.raises(ValueError, match='plan-e3'):
        S.build_rooms(world.data, world.tmp / 'e2', world.tmp / 'e2')


def _api_run(root, receipt, ledger=0, receipt_calls=(), finished=True):
    (root / 'receipts').mkdir(parents=True)
    (root / 'run_receipt.json').write_text(json.dumps(receipt))
    (root / 'calls.jsonl').write_text('{"uid":"u","attempt":1}\n' * ledger)
    for k, calls in enumerate(receipt_calls):
        (root / 'receipts' / f'{k}.json').write_text(json.dumps({'calls': calls}))
    if finished:
        (root / 'MANIFEST.json').write_text('{}')


def test_cap_preflight_counts_api_runs_and_refuses_before_any_call(world):
    _api_run(world.run / 'llm-agent-001', {'arm': 'llm_agent'}, ledger=4000, receipt_calls=(1, 2))
    _api_run(world.run / 'llm-agent-P-pilot-001', {'arm': 'llm_agent'}, ledger=1, receipt_calls=(3, 3))  # receipts win
    _api_run(world.run / 'sizes-020-L-smoke', {'arm': 'sizer_020', 'sizer': 'L'}, ledger=50)
    _api_run(world.run / 'sizes-020-O2', {'arm': 'sizer_020', 'sizer': 'O2'}, ledger=999)  # GPU: not API
    _api_run(world.run / 'judge-001', {'arm': 'judge'}, ledger=999)
    check = S.preflight(world.run, 14)
    assert check['by_run'] == {'llm-agent-001': 4000, 'llm-agent-P-pilot-001': 6, 'sizes-020-L-smoke': 50}
    assert check['used'] == 4056 and check['needed'] == 28 and check['ok']  # 14 rooms x (1 + 1 repair), addendum A
    k = S.CALLS_PER_ROOM
    assert S.preflight(world.run, (S.CAP - 4056) // k)['ok'] and not S.preflight(world.run, (S.CAP - 4056) // k + 1)['ok']
    assert S.main(['cap', '--run-root', str(world.run), '--rooms', '1877']) == 1  # 4,056 + 1,877 x 2 = 7,810 > 6,500
    _api_run(world.run / 'llm-agent-002', {'arm': 'llm_agent'}, ledger=S.CAP - 4056 - 27)
    fake = FakeApi(world)
    assert run_api(world, fake) == 1
    status = json.loads((world.run / 'sizes-020-L/status.json').read_text())
    assert status['status'] == 'CAP_REFUSED' and status['used'] + status['needed'] == S.CAP + 1
    assert fake.calls == [] and not (world.run / 'sizes-020-L/run_receipt.json').exists()


def test_cap_preflight_ignores_limit_and_refuses_unfinished_api_runs(world):
    _api_run(world.run / 'llm-agent-001', {'arm': 'llm_agent'}, ledger=S.CAP - 10)  # 14 x 2 > 10 >= 2 x 2
    fake = FakeApi(world)
    assert run_api(world, fake, 'sizes-020-L', '--limit', '2') == 1 and fake.calls == []
    status = json.loads((world.run / 'sizes-020-L/status.json').read_text())
    assert status['status'] == 'CAP_REFUSED' and status['rooms'] == 14 and status['unfinished'] == []
    _api_run(world.run / 'llm-agent-002', {'arm': 'llm_agent'}, ledger=1, finished=False)  # could resume and spend
    (world.run / 'llm-agent-001/calls.jsonl').write_text('')
    check = S.preflight(world.run, 14)
    assert check['used'] + check['needed'] <= S.CAP and not check['ok'] and check['unfinished'] == ['llm-agent-002']
    assert run_api(world, fake, 'sizes-020-L2') == 1 and fake.calls == []
    assert S._sizer(world.run / 'sizes-020-L2', None)[1].startswith('未运行：API 上限预检拒绝（未完成的 API 运行 llm-agent-002')
    (world.run / 'llm-agent-002/MANIFEST.json').write_text('{}')
    assert run_api(world, fake, 'sizes-020-L3', '--limit', '2') == 0 and len(fake.calls) == 2
    status = json.loads((world.run / 'sizes-020-L3/run_receipt.json').read_text())
    assert status['config']['limit'] == 2 and status['rooms']['rooms'] == 2


def test_api_sizer_repairs_failures_and_resume(world):
    order = json.loads(world.rooms.read_text())['uids']
    broken, repaired, hopeless = order[2], order[3], order[4]
    transient = L.ApiError('http_status_503', True)
    first = FakeApi(world, {broken: [transient] * (L.RETRIES + 1), repaired: ['{"sizes":1}', 'valid'],
                            hopeless: ['{}'] * 3})
    assert run_api(world, first) == 2  # outage: fail closed, no receipt for the broken room
    out = world.run / 'sizes-020-L'
    assert json.loads((out / 'status.json').read_text())['status'] == 'API_FAILURE'
    assert {u for u, _ in first.calls} == set(order[:3])
    second = FakeApi(world, {repaired: ['{"sizes":1}', 'valid'], hopeless: ['{}'] * 3})
    assert run_api(world, second) == 0
    assert {u for u, _ in second.calls} == set(order[2:])  # resumed: rooms with receipts are never called again
    repair_turns = [m for u, m in second.calls if u == repaired][1]
    assert repair_turns[2:] == [{'role': 'user', 'content': L.SUFFIX}, {'role': 'assistant', 'content': '{"sizes":1}'},
                                {'role': 'user', 'content': S.WRONG_IDS}]
    result = json.loads((out / 'result.json').read_text())
    assert result['status'] == 'SIZES_COMPLETE' and result['rooms'] == 14 and result['valid_final'] == 13
    assert result['failed'] == {hopeless: S.WRONG_IDS} and result['rounds_histogram'] == {'1': 12, '2': 2}  # L-size: 1 repair round (addendum A)
    assert result['calls_ledger'] == len(first.calls) + len(second.calls) == 19  # hopeless stops after 2 calls
    estimates = [json.loads(line) for line in (out / 'estimates.jsonl').read_text().splitlines()]
    assert [e['uid'] for e in estimates] == [u for u in order if u != hopeless]
    assert estimates[0]['estimated_sizes'] == json.loads(_answer(world, order[0]))
    assert 'run_sha256' in json.loads((out / 'run_receipt.json').read_text())
    assert run_api(world, FakeApi(world)) == 0  # published: no further call


class FakeTok:
    def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=True, **kwargs):
        assert add_generation_prompt and not tokenize
        return 'p' * 100

    def encode(self, text, add_special_tokens=True):
        return [0] * len(text)


class FakeLLM:
    def __init__(self, world, script=None):
        self.world, self.script, self.calls, self.engine = world, script or {}, [], None
        self.requests = {u: S.sizer_request(r)['messages'] for u, r in world.joint.items()}

    def get_tokenizer(self):
        return FakeTok()

    def chat(self, messages, params, use_tqdm=True, chat_template_kwargs=None):
        self.calls.append({'params': params, 'kwargs': chat_template_kwargs})
        outputs = []
        for conversation in messages:
            uid = next(u for u, m in self.requests.items() if conversation[:2] == m)
            queue = self.script.setdefault(uid, [])
            text = '<think>sizes</think>' + (queue.pop(0) if queue else _answer(self.world, uid, 0.9))
            outputs.append(SimpleNamespace(prompt_token_ids=[1] * 100, outputs=[SimpleNamespace(
                text=text, token_ids=[1] * len(text), finish_reason='stop')]))
        return outputs


def run_open(world, fake, index, count=2):
    def load(engine):
        fake.engine = engine
        return fake, lambda **kw: kw
    model = world.tmp / 'models/Qwen3.8-27B'
    model.mkdir(parents=True, exist_ok=True)
    (model / 'config.json').write_text('{}')
    return S.main(['run-open', '--data', str(world.data), '--rooms', str(world.rooms), '--model-root',
                   str(world.tmp / 'models'), '--out', str(world.run / 'sizes-020-O2'), '--shard-index', str(index),
                   '--shard-count', str(count), '--chunk', '3'], load=load)


def finalize(world, name='sizes-020-O2'):
    return S.main(['finalize', '--rooms', str(world.rooms), '--out', str(world.run / name)])


def test_open_sizer_shards_use_the_o2_engine_resume_and_finalize(world):
    order = json.loads(world.rooms.read_text())['uids']
    fake = FakeLLM(world, {order[2]: ['not json']})  # shard 0 = order[0::2]
    assert run_open(world, fake, 0) == 0
    out = world.run / 'sizes-020-O2'
    spec = O.SYSTEMS['O2']
    assert fake.engine == {'model': str((world.tmp / 'models/Qwen3.8-27B').resolve()), 'tensor_parallel_size': spec['tp'],
                           'max_model_len': O.MAX_MODEL_LEN, 'seed': O.SEED, **spec['engine']}
    assert all(c['kwargs'] == spec['template'] == {'enable_thinking': True} for c in fake.calls)
    assert all(p == {'max_tokens': p['max_tokens'], **O.SAMPLING} for c in fake.calls for p in c['params'])
    assert len(list((out / 'receipts').glob('*.json'))) == 7
    assert finalize(world) == 1 and json.loads((out / 'status.json').read_text())['status'] == 'INCOMPLETE'
    again = FakeLLM(world)
    assert run_open(world, again, 0) == 0 and again.calls == [] and again.engine is None  # resumed: nothing to load
    assert run_open(world, FakeLLM(world), 1) == 0
    assert finalize(world) == 0
    result = json.loads((out / 'result.json').read_text())
    assert result['valid_final'] == 14 and result['rounds_histogram'] == {'1': 13, '2': 1}
    assert result['tokens']['reasoning_tokens'] > 0 and 'mode_check' not in result and result['invocations'] == 3
    receipt = json.loads((out / 'receipts' / (L._digest(order[2]) + '.json')).read_text())
    assert receipt['answers'][0]['reasoning'] == '<think>sizes</think>' and receipt['answers'][0]['raw'] == 'not json'


def test_estimates_feed_the_optiscene_plan_code_and_p_requests(world):
    assert run_open(world, FakeLLM(world), 0, 1) == 0 and finalize(world) == 0
    sizes = {r['uid']: r['estimated_sizes'] for r in oe._jsonl(world.run / 'sizes-020-O2/estimates.jsonl')}
    assert set(sizes) == set(E2 + E3)
    for uid, estimate in sizes.items():
        room = world.labels[uid]
        rows = oe.room_requests(room, estimate, 'INSTR\n', 'e2', 'x')
        mine = next(r for r in rows if r['mode'] == 'prior' and r['format'] == 'optiscene')
        assert mine['boxes'] == [[oe._box(estimate[i][2]), oe._box(estimate[i][1]), oe._box(estimate[i][0])]
                                 for i in mine['ids']]
        record = {**{k: room[k] for k in ('uid', 'source', 'group')},
                  'messages': contract.messages(room, 'placement', estimated_sizes=estimate)[:2]}
        infer.validate_request(record, 'placement')


def _l_joint(world, uids):
    root = world.run / 'llm-eval-L-selected-v1'
    root.mkdir()
    rows = [{'uid': u, 'complete': u != uids[-1],
             'raw_layout': [{'id': o['id'], 'size': [v * 0.8 for v in o['size']]} for o in world.labels[u]['objects']]}
            for u in uids]
    (root / 'llm_harness-geometry.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    h.write_json(root / 'summary.json', {'status': 'LLM_COMPARISON_COMPLETE', 'llm': {'config': {'model': S.L_MODEL}},
                                         'data_manifest_sha256': h.sha256(world.data / 'MANIFEST.json')})
    h.write_json(root / 'MANIFEST.json', {'files': h.artifact_manifest(root)})
    return root


def report(world, out):
    return S.main(['report', '--data', str(world.data), '--rooms', str(world.rooms), '--run-root', str(world.run),
                   '--l', str(world.run / 'sizes-020-L'), '--o2', str(world.run / 'sizes-020-O2'),
                   '--l-joint', str(world.run / 'llm-eval-L-selected-v1'), '--out', str(world.run / out),
                   '--repeats', '50'])


def test_report_levels_contrasts_and_not_run_sizers(world):
    assert run_api(world, FakeApi(world)) == 0
    assert run_open(world, FakeLLM(world), 0, 1) == 0 and finalize(world) == 0
    _l_joint(world, ['il-bed-0', 'il-bed-1', 'x-00', 'x-01'])
    assert report(world, 'report-v1') == 0
    result = json.loads((world.run / 'report-v1/sizes.json').read_text())
    est = lambda scope, src, f: result['levels'][scope][src]['fields'][f]['estimate']
    assert est('all', 'GT', 'size_log_mae') == 0 and est('all', 'GT', 'size_iou') == pytest.approx(1)
    assert est('all', 'L-size', 'size_log_mae') == pytest.approx(math.log(1.1), abs=1e-3)
    assert est('all', 'L-size', 'bias_fp') == pytest.approx(2 * math.log(1.1), abs=2e-3)
    assert est('all', 'O2-size', 'bias_fp') == pytest.approx(2 * math.log(0.9), abs=2e-3)
    assert est('all', 'L-joint', 'complete') == pytest.approx(0.75) and result['levels']['all']['L-joint']['rooms'] == 4
    assert est('all', 'prior', 'size_log_mae') > 0 and set(result['levels']) == {'all', 'E2', 'E3', 'bedroom', 'kitchen',
                                                                                  'study_office'}
    assert result['levels']['E3']['prior']['rooms'] == 10
    pair = result['contrasts']['L-size − O2-size']['all']['fields']['bias_fp']['estimate']
    assert pair == pytest.approx(2 * math.log(1.1 / 0.9), abs=3e-3)
    assert result['contrasts']['L-size − L-joint']['all']['rooms'] == 4
    assert result['levels']['all']['GT']['fields']['size_log_mae']['ci95'] is None  # k < NO_CI components
    text = (world.run / 'report-v1/report.md').read_text()
    assert '表 A' in text and S.SIZE_DEFINITION in text
    # A refused L-size is reported as not run; an unfinished sizer blocks the report.
    for name in ('estimates.jsonl', 'result.json', 'MANIFEST.json'):
        (world.run / 'sizes-020-L' / name).unlink()
    h.write_json(world.run / 'sizes-020-L/status.json',
                 {'status': 'CAP_REFUSED', **S.preflight(world.run, 1877, exclude=world.run / 'sizes-020-L')})
    assert report(world, 'report-v2') == 0
    assert 'L-size' not in json.loads((world.run / 'report-v2/sizes.json').read_text())['levels']['all']
    assert '未运行：API 上限预检拒绝（已用 ' in (world.run / 'report-v2/report.md').read_text()
    h.write_json(world.run / 'sizes-020-L/status.json', {'status': 'INCOMPLETE'})
    with pytest.raises(ValueError, match='not complete'):
        report(world, 'report-v3')


def test_report_refuses_without_a_complete_o2_size(world):
    assert run_api(world, FakeApi(world)) == 0
    _l_joint(world, ['il-bed-0', 'x-00'])
    with pytest.raises(ValueError, match='O2-size has no complete output'):
        report(world, 'report-v1')  # all shards failed before any receipt: no write-once report is spent
    assert not (world.run / 'report-v1').exists()


def _submit(tmp_path, squeue='', rate=5.0):
    """submit_020_sizers.sh with PROJECT_ROOT = tmp_path and stub sbatch/squeue/scontrol."""
    code = Path(S.__file__).resolve().parents[2]
    pin = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    text = (code / 'fastfill/comparison/submit_020_sizers.sh').read_text()
    assert text.count(pin) == 1
    bin_ = tmp_path / 'bin'
    bin_.mkdir(parents=True)
    script = bin_ / 'submit.sh'
    script.write_text(text.replace(pin, f'PROJECT_ROOT={tmp_path}\n'))
    python = tmp_path / 'env/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    (bin_ / 'sbatch').write_text('#!/bin/sh\necho "$@" >> "$SBATCH_LOG"\necho "$((100 + $(wc -l < "$SBATCH_LOG")))"\n')
    (bin_ / 'scontrol').write_text('#!/bin/sh\necho "$@" >> "$SCONTROL_LOG"\n')
    (bin_ / 'squeue').write_text(f'#!/bin/sh\ntest "$*" = "-h -u me -o %i %j" || exit 3\nprintf "{squeue}"\n')
    for path in (python, bin_ / 'sbatch', bin_ / 'scontrol', bin_ / 'squeue'):
        path.chmod(0o755)
    run = tmp_path / 'runs/unknownsize-20261013'
    for name in ('eval-data-v1/MANIFEST.json', 'optiscene-014/e2/MANIFEST.json', 'optiscene-015/e3/MANIFEST.json',
                 'llm-eval-L-selected-v1/MANIFEST.json', 'llm-open-O2-001/MANIFEST.json'):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text('{}')
    h.write_json(run / 'llm-open-O2-001/cost.json', {'system': 'O2', 'generation_seconds': 1454 * rate,
                                                      'wall_seconds': 1454 * rate + 300})
    h.write_json(run / 'llm-open-O2-001/result.json', {'status': 'LLM_COMPLETE', 'rooms': 1454})
    (tmp_path / 'models/Qwen3.8-27B').mkdir(parents=True)
    (tmp_path / 'models/Qwen3.8-27B/config.json').write_text('{}')
    env = {'PATH': f'{bin_}:/usr/bin:/bin', 'USER': 'me', 'SBATCH_LOG': str(bin_ / 'log'),
           'SCONTROL_LOG': str(bin_ / 'scontrol.log')}
    done = subprocess.run(['bash', str(script), str(code)], env=env, capture_output=True, text=True)
    log = (bin_ / 'log').read_text().splitlines() if (bin_ / 'log').exists() else []
    return done, log, (bin_ / 'scontrol.log').read_text() if (bin_ / 'scontrol.log').exists() else ''


def test_submit_chain_dependencies_gpu_ceiling_and_api_guard(tmp_path):
    done, log, released = _submit(tmp_path / 'a')
    assert done.returncode == 0, done.stderr
    assert len(log) == 1 + 1 + 6 + 1 and '--hold' in log[0] and log[0].endswith(' rooms')
    assert 'afterok:101,singleton' in log[1] and 'unknownsize-020-sizer-L' in log[1] and log[1].endswith(' l-size')
    minutes = math.ceil(1.5 * (300 + 5.0 * math.ceil(1877 / 6)) / 60)
    for k, line in enumerate(log[2:8]):
        assert '--dependency=afterok:101 ' in line and '--gpus-per-node=2' in line and 'singleton' not in line
        assert f'--time={minutes // 60:02d}:{minutes % 60:02d}:00' in line and line.endswith(f'o2-size {k} 6')
    assert '--dependency=afterany:102:103:104:105:106:107:108' in log[8] and log[8].endswith(' report')
    assert released.strip() == 'release 101' and '"ok": true' in done.stdout  # 1,877 x 2 <= 6,500 (addendum A)
    header = [line for line in (Path(S.__file__).parent / 'isambard_020.sbatch').read_text().splitlines()
              if line.startswith('#SBATCH')]
    assert '#SBATCH --exclude=nid010216,nid011181,nid010569' in header and not any('--gres' in x for x in header)
    done, log, _ = _submit(tmp_path / 'b', squeue='7 unknownsize-llm-agent\\n')
    assert done.returncode == 0 and len(log) == 8 and 'skip L-size' in done.stdout
    assert all('l-size' not in line for line in log) and 'afterany:102:' in log[-1]
    done, log, _ = _submit(tmp_path / 'c', squeue='9 unknownsize-020-sizer-O2\\n')
    assert done.returncode == 1 and log == [] and 'already in Slurm' in done.stderr
    done, log, _ = _submit(tmp_path / 'd', rate=20.0)
    assert done.returncode == 1 and log == [] and 'GPU ceiling' in done.stderr


def test_l_joint_accepts_summaries_without_a_system_key_and_checks_the_model(world):
    root = _l_joint(world, ['il-bed-0', 'x-00'])  # The real llm-eval-L-selected-v1 summary has no 'system' key.
    data_sha = h.sha256(world.data / 'MANIFEST.json')
    assert set(S._l_joint(root, data_sha, ['il-bed-0', 'x-00'])) == {'il-bed-0', 'x-00'}
    summary = json.loads((root / 'summary.json').read_text())
    summary['llm']['config']['model'] = 'other-model'
    h.write_json(root / 'summary.json', summary)
    (root / 'MANIFEST.json').unlink()  # Re-seal: only the model differs.
    h.write_json(root / 'MANIFEST.json', {'files': h.artifact_manifest(root)})
    with pytest.raises(ValueError, match='not a complete J-input evaluation of L'):
        S._l_joint(root, data_sha, ['il-bed-0'])
