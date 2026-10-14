"""Amendment 021 sections 1.4, 3, 4 on CPU: fake vLLM, fake inference dirs, patched model identities, stub Slurm."""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace as ns

import pytest

from fastfill.comparison import a021_select as A
from fastfill.comparison import dev_select as ds
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import infer
from fastfill.comparison import llm_open as O
from fastfill.comparison import sample_vllm as sv
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_dpo_stages as D
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests import test_llm_open as TO
from fastfill.comparison.tests import test_sample_vllm as SV
from fastfill.comparison.tests.test_llm_open import world  # noqa: F401  The llm_open fixture (O2nt tests).

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
DOC = 'docs/research/20261013-unknown-size-comparison/amendment-021-diagnostics-and-one-more-arm-20261014.md'


# ---- confirm slice -------------------------------------------------------------------------------

def dev_room(uid, source, n, group=None):
    objects = [D.obj(f'o{j}', 'cabinet', 0.5 + (j % 5), 0.5 + (j // 5), size=(0.3, 0.3, 0.5)) for j in range(n)]
    room = D.raw_room(uid, source, objects)
    return {**room, 'group': group or room['group']}


PAPER = [dev_room(f'd{k}', 'OptiScene_holodeck', 3) for k in range(4)]
OTHER = [dev_room(f'd{k}', 'SpatialGen', 3) for k in range(4, 7)]
SMALL = [dev_room(f's{k}', 'Structured3D', 2) for k in range(8)]
BIG = [dev_room(f'b{k}', 'Structured3D', 22) for k in range(5)]
SHARED = dev_room('cx', 'Structured3D', 2, group='g-d0')  # Shares a house with a dev-select room.
LONG = dev_room('cn', 'Structured3D', 2)  # Not length-admitted.
ADMITTED = sorted(r['uid'] for r in PAPER + OTHER + SMALL + BIG + [SHARED])


@pytest.fixture
def main(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, 'TARGETS', {'paper_sources': 4, 'other_sources': 3})
    data = D.build_main(tmp_path, dev=PAPER + OTHER + SMALL + BIG + [SHARED, LONG])
    runs = D.build_sft(tmp_path, data)
    for run in runs:
        lengths = h.strict_json((run / 'lengths.json').read_text())
        h.write_json(run / 'lengths.json', {**lengths, 'dev': {'accepted_uids': ADMITTED, 'lengths': []}})
        (run / 'SHA256.json').unlink()
        h.write_json(run / 'SHA256.json', h.artifact_manifest(run))
    ds.build(ns(data=data, sft_runs=runs, out=tmp_path / 'dev-select-v1'))
    return ns(root=tmp_path, data=data, runs=runs)


def confirm_args(w, dev=None):
    return A.parse_args(['confirm', '--data', str(w.data), '--dev', str(dev or w.root / 'dev-select-v1'),
                         '--run-root', str(w.root), '--sft-runs', *map(str, w.runs)])


def test_confirm_slice_is_disjoint_bin_stratified_hash_ranked_and_dev_select_like(main, monkeypatch):
    monkeypatch.setattr(A, 'CONFIRM_ROOMS', 5)
    monkeypatch.setattr(A, 'ENGINE_ROOMS', 2)
    dev = h.strict_json((main.root / 'dev-select-v1/MANIFEST.json').read_text())['expected_uids']
    dev_groups = {r['group'] for r in PAPER + OTHER + SMALL + BIG + [SHARED] if r['uid'] in dev}
    pool = {r['uid']: len(r['objects']) for r in PAPER + OTHER + SMALL + BIG + [SHARED]
            if r['uid'] not in dev and r['group'] not in dev_groups}
    assert 'cx' not in pool and 'cn' not in pool and not set(dev) & set(pool)
    result = A.confirm(confirm_args(main))
    root = main.root / 'a021'
    manifest = h.strict_json((root / 'confirm-v1/MANIFEST.json').read_text())
    order = manifest['selection']['hash_order']
    small = sorted((u for u, n in pool.items() if n <= 20), key=A.rank)
    big = sorted((u for u, n in pool.items() if 21 <= n <= 50), key=A.rank)
    quota = {'objects<=20': 5 * len(small) // len(pool), 'objects21-50': 5 * len(big) // len(pool), 'objects>50': 0}
    quota[max(('objects<=20', 'objects21-50'), key=lambda b: 5 * len(small if b == 'objects<=20' else big) % len(pool))] += 1
    assert manifest['selection']['quotas'] == quota and result['quotas'] == quota and sum(quota.values()) == 5
    assert sorted(order) == sorted(small[:quota['objects<=20']] + big[:quota['objects21-50']])  # Lowest ranks per bin.
    assert order == sorted(order, key=A.rank) and manifest['expected_uids'] == sorted(order)
    assert manifest['selection']['salt'] == 'a021-confirm:' and A.rank('u') == h.hashlib.sha256(b'a021-confirm:u').hexdigest()
    engine = h.strict_json((root / 'engine-v1/MANIFEST.json').read_text())
    assert engine['expected_uids'] == sorted(order[:2]) and engine['selection']['hash_order'] == order[:2]
    for out in (root / 'confirm-v1', root / 'engine-v1'):
        rooms = {r['uid']: r for r in h._rows(out, 'rooms.jsonl', h.strict_json((out / 'MANIFEST.json').read_text())['files']['rooms.jsonl'])}
        assert not set(rooms) & set(dev) and not {r['group'] for r in rooms.values()} & dev_groups
        for arm in h.ARMS:  # The unchanged eval-data loader (infer_shard plan, llm_open, sample_vllm read it).
            rows, info = infer.load_eval(out, arm)
            assert info['expected_uids'] == sorted(rooms) and all(len(r['messages']) == 2 for r in rows)
        assert h.strict_json((out / 'MANIFEST.json').read_text())['training_manifest_sha256'] == h.sha256(main.data / 'MANIFEST.json')
    for name in ('dev-select', 'confirm'):
        cohort = h.strict_json((root / 'cohorts' / f'{name}.json').read_text())
        assert cohort == O.all_rooms_cohort(A.data_dir(main.root, name))
    with pytest.raises(FileExistsError):
        A.confirm(confirm_args(main))
    shutil.rmtree(root)
    with pytest.raises(ValueError, match='--dev must be'):
        A.confirm(confirm_args(main, main.data))
    monkeypatch.setattr(A, 'CONFIRM_ROOMS', len(pool) + 1)
    with pytest.raises(ValueError, match='confirm pool has'):
        A.confirm(confirm_args(main))


def test_pick_is_pure_order_free_and_proportional():
    pool = {f'u{k}': (2 if k % 3 else 30) for k in range(30)}
    assert A.pick(pool, 9) == A.pick(dict(reversed(list(pool.items()))), 9)
    chosen, info = A.pick(pool, 9)
    assert info['quotas'] == {'objects<=20': 6, 'objects21-50': 3, 'objects>50': 0} and len(chosen) == 9
    with pytest.raises(ValueError, match='without movable objects'):
        A.pick({'z': 0}, 1)


# ---- vLLM greedy ---------------------------------------------------------------------------------

def greedy_argv(root, data, out, exact=False, *extra):
    source = (['--model', str(root / 'dpo-003/placement/dpo1-merged'), '--exact-policy'] if exact else
              ['--adapter', str(root / 'sft/placement/adapter'), '--training-run', str(root / 'sft')])
    return ['greedy', '--data', str(data), '--base', str(root / 'base'), '--arm', 'placement', '--out', str(root / out),
            *source, *extra]


@pytest.fixture
def vworld(tmp_path, monkeypatch):
    w = SV.patched(tmp_path, monkeypatch, SV.make_data(tmp_path))
    (tmp_path / 'base').mkdir()
    (tmp_path / 'base/config.json').write_text('{"model_type": "qwen3"}')
    (tmp_path / 'dpo-003/placement/sft-merged').mkdir(parents=True)
    (tmp_path / 'dpo-003/placement/sft-merged/config.json').write_text('{"model_type": "qwen3_5", "vision_config": {}}')
    return w


def test_greedy_is_sample_vllms_k0_request_once_per_room_resumable_and_evaluable(vworld, monkeypatch):
    root, data = vworld.root, vworld.data
    monkeypatch.setattr(A, 'BATCHINGS', {'objects': 2 * SV.CAPS[3] + SV.CAPS[2], 'hash': SV.CAPS[3]})
    fake, engines = SV.FakeLLM(data), []
    A.main(greedy_argv(root, data, 'g'), SV.loader(fake, engines))
    assert engines == [{**sv.ENGINE, 'model': str((root / 'base').resolve())}]  # A text base: no multimodal limits.
    records = {r['uid']: r for r in infer.load_eval(data, 'placement')[0]}
    expected = sorted(records, key=lambda u: (-sv.objects(records[u]), u))
    assert [u for c in fake.calls for u, _ in c['uids']] == expected and all(k == 0 for c in fake.calls for _, k in c['uids'])
    assert [len(c['prompts']) for c in fake.calls] == [3, 3, 1]  # Rooms by size 3, 3, 2 | 2, 2, 1 | 1.
    for call in fake.calls:
        for (uid, _), prompt, p in zip(call['uids'], call['prompts'], call['params']):
            ids = infer.prompt_ids(SV.Tok(), records[uid])
            assert prompt == {'prompt_token_ids': ids}
            assert p == sv.params(min(40960 - len(ids), 1024 + 128 * sv.objects(records[uid])), uid, 0, T.EOS)
    manifest = h.strict_json((data / 'MANIFEST.json').read_text())
    rows, receipt = e._inference(root / 'g', data, manifest['expected_uids'])
    assert [r['uid'] for r in rows] == manifest['expected_uids'] and all(r['stopped_on_eos'] for r in rows)
    A.bind(receipt, {'adapter': str(root / 'sft/placement/adapter'), 'training_run': str(root / 'sft')}, 'vllm')
    with pytest.raises(ValueError, match='inference is not'):
        A.bind(receipt, {'adapter': str(root / 'sft/placement/adapter'), 'training_run': str(root / 'sft')}, 'hf')
    assert receipt['config']['batching'] == 'objects' and receipt['config']['lora'] == A.VLLM_LORA
    calls = len(fake.calls)
    A.main(greedy_argv(root, data, 'g'), SV.loader(fake))  # Finished: no generation, same result.
    assert len(fake.calls) == calls
    # The hash batching (R's second run) changes the batch composition, not the requests.
    hashed = SV.FakeLLM(data)
    A.main(greedy_argv(root, data, 'h', False, '--batching', 'hash'), SV.loader(hashed))
    assert [len(c['prompts']) for c in hashed.calls] == [1] * 7
    assert [u for c in hashed.calls for u, _ in c['uids']] == sorted(records, key=lambda u: A.rank(u, 'a021-batch:'))
    assert [r['raw'] for r in e._inference(root / 'h', data, manifest['expected_uids'])[0]] == [r['raw'] for r in rows]
    with pytest.raises(ValueError, match='run identity mismatch'):
        A.main(greedy_argv(root, data, 'h'), SV.loader(SV.FakeLLM(data)))


def test_greedy_resumes_after_a_crash_and_marks_input_too_long(vworld, monkeypatch):
    root, data = vworld.root, vworld.data
    records = infer.load_eval(data, 'placement')[0]
    long = records[0]['messages'][1]['content']
    monkeypatch.setattr(sv.d, '_model_tokenizer', lambda path: SV.Tok(long=[long]))
    monkeypatch.setattr(A, 'BATCHINGS', {'objects': SV.CAPS[3], 'hash': 1})
    fake, engines = SV.FakeLLM(data, fail_on_call=3), []
    with pytest.raises(RuntimeError, match='simulated'):
        A.main(greedy_argv(root, data, 'x', True), SV.loader(fake, engines))
    assert engines[0] == {**sv.ENGINE, 'model': str(root / 'dpo-003/placement/sft-merged'), **sv.MULTIMODAL_ENGINE}  # 27B-like.
    assert not (root / 'x/MANIFEST.json').exists()
    again = SV.FakeLLM(data)
    A.main(greedy_argv(root, data, 'x', True), SV.loader(again))
    done = sum(len(c['uids']) for c in fake.calls[:2])
    assert sum(len(c['uids']) for c in again.calls) == len(records) - 1 - done  # Only the pending rooms, once.
    rows = {r['uid']: r for r in h._rows(root / 'x', 'records.jsonl', h.sha256(root / 'x/records.jsonl'))}
    assert rows[records[0]['uid']]['error'] == 'INPUT_TOO_LONG' and rows[records[0]['uid']]['raw'] is None
    receipt = h.strict_json((root / 'x/run_receipt.json').read_text())
    A.bind(receipt, {'model': str(root / 'dpo-003/placement/dpo1-merged')}, 'vllm')


# ---- O2nt (llm_open) ------------------------------------------------------------------------------

def test_o2nt_is_o2_without_thinking_asserts_zero_reasoning_and_leaves_the_table_alone(world):  # noqa: F811
    root, rooms = world
    assert 'O2nt' not in O.SYSTEMS and O.SPECS['O2nt']['model'] == O.SYSTEMS['O2']['model']
    assert O.slurm_flags('O2nt', 'full') == '--gpus-per-node=2 --cpus-per-task=32 --mem=200G --time=03:00:00'
    assert O.slurm_flags('O2', 'full') == '--gpus-per-node=2 --cpus-per-task=32 --mem=200G --time=08:00:00'
    think = TO.FakeLLM(rooms, default='<think>plan</think>{valid}', arm='placement')
    assert O.main(TO.argv(root, 'O2', 'o2', '--arm', 'placement', '--limit', '2'), TO.loader(think)) == 0
    fake, engines = TO.FakeLLM(rooms, default='{valid}', arm='placement'), []
    assert O.main(TO.argv(root, 'O2nt', 'o2nt', '--arm', 'placement', '--limit', '2'), TO.loader(fake, engines)) == 0
    a, b = (h.strict_json((root / out / 'run_receipt.json').read_text())['config'] for out in ('o2', 'o2nt'))
    assert engines[0]['tensor_parallel_size'] == 2 and a['engine'] == b['engine'] and 'reasoning_tokens' not in a
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {
        'system', 'model', 'reasoning_effort', 'chat_template_kwargs', 'thinking', 'cap', 'reasoning_tokens'}
    assert (b['chat_template_kwargs'], b['thinking'], b['cap']['thinking_allowance'], b['reasoning_tokens']) == (
        {'enable_thinking': False}, False, 0, 'asserted_zero')
    assert fake.calls[0]['kwargs'] == {'enable_thinking': False}
    assert all(p['max_tokens'] == 1024 + 128 * 2 for p in fake.calls[0]['params'])
    result = h.strict_json((root / 'o2nt/result.json').read_text())
    assert result['tokens']['reasoning_tokens'] == 0 and result['valid_final'] == 2
    # Any reasoning in an O2nt answer stops the run before its receipt is written.
    leak = TO.FakeLLM(rooms, default='<think>x</think>{valid}', arm='placement')
    with pytest.raises(ValueError, match='reasoning tokens in a no-reasoning system'):
        O.main(TO.argv(root, 'O2nt', 'leak', '--arm', 'placement', '--limit', '2'), TO.loader(leak))
    assert not list((root / 'leak/receipts').iterdir())
    assert h.strict_json((root / 'leak/status.json').read_text())['status'] == 'FAIL'
    assert list(O.write_status(root)) == [*O.SYSTEMS, *O.NOT_RUN]  # The amendment 003-007 status table is unchanged.


# ---- rules ---------------------------------------------------------------------------------------

def m(checker, complete=1.0, oor=0.1, accepted=0.5, holds=0.8, big=0.3):
    return {'checker': checker, 'complete_rate': complete, 'oor_mean': oor, 'accepted_rate': accepted,
            'holds_rate': holds, 'checker_21plus': big}


def test_engine_rule_edges():
    assert A.common_engine(0.015) and A.common_engine(-0.015) and A.common_engine(0.47 - 0.455)
    assert not A.common_engine(0.0151) and not A.common_engine(-0.016)
    assert A.engines(True) == {'8B': 'vllm', '27B': 'vllm'} and A.engines(False) == {'8B': 'hf', '27B': 'vllm'}


def test_guards_inclusive_bounds_and_missing_values():
    ref = m(0.5)
    edge = m(0.6, complete=1.0 - 0.005, oor=0.1 + 0.01, accepted=0.5 - 0.01, holds=0.8 - 0.03, big=0.3 - 0.02)
    assert all(A.guards(edge, ref).values())
    for key, value in (('complete_rate', 0.994), ('oor_mean', 0.111), ('accepted_rate', 0.489), ('holds_rate', 0.769),
                       ('checker_21plus', 0.279)):
        assert A.guards({**edge, key: value}, ref) == {**dict.fromkeys(A.guards(edge, ref), True), key: False}
    assert A.guards(m(0.6, big=None), m(0.5, big=None))['checker_21plus']  # No 21+ room in the slice: nothing to guard.
    assert not A.guards(m(0.6, big=None), ref)['checker_21plus'] and not A.guards(m(0.6), m(0.5, oor=None))['oor_mean']


def test_choose_tie_band_prefers_8b_then_r_and_respects_guards():
    serving = {'r': 'vllm', 'op13': 'vllm', 'arm': 'vllm'}
    sizes = {'r': '8B', 'op13': '8B', 'arm': '27B'}
    refs = {'vllm': m(0.50)}
    pick = lambda dev, spread=0.0: A.choose({'r': refs['vllm'], **dev}, refs, sizes, serving, spread)  # noqa: E731
    assert pick({'arm': m(0.52)})['selected'] == 'arm'                       # Beyond the 0.01 band.
    assert pick({'arm': m(0.51)})['selected'] == 'r'                         # Within it: the cheaper model.
    assert pick({'arm': m(0.52)}, 0.03) == {**pick({'arm': m(0.52)}, 0.03), 'selected': 'r', 'tie_band': 0.03}
    assert pick({'arm': m(0.52)}, 0.005)['tie_band'] == 0.01
    assert pick({'op13': m(0.53), 'arm': m(0.535)})['selected'] == 'op13'   # 8B before 27B within the band.
    assert pick({'op13': m(0.505), 'arm': m(0.52)})['selected'] == 'arm'    # R and op13 below the band of arm.
    assert pick({'op13': m(0.515), 'arm': m(0.52)}, 0.02)['selected'] == 'r'  # R, op13, arm tied: R among the 8B.
    choice = pick({'op13': m(0.60, accepted=0.40), 'arm': m(0.55)})
    assert choice['selected'] == 'arm' and choice['eligible'] == ['r', 'arm'] and not choice['guards']['op13']['accepted_rate']
    split = A.choose({'r': m(0.50), 'arm': m(0.56, oor=0.3)}, {'hf': m(0.50), 'vllm': m(0.45, oor=0.3)},
                     sizes, {'r': 'hf', 'arm': 'vllm'}, 0.0)
    assert split['selected'] == 'arm' and all(split['guards']['arm'].values())  # Guarded against R on the same engine.


def rows_of(success):
    return [{'uid': f'u{k}', 'source': 'S', 'group': f'g{k}', 'geometry_success': s} for k, s in enumerate(success)]


def test_confirmation_needs_a_house_cluster_ci_lower_bound_above_zero():
    assert A.confirmation(rows_of([True] * 30), rows_of([k % 2 == 0 for k in range(30)]))['confirmed']
    same = A.confirmation(rows_of([k % 2 == 0 for k in range(30)]), rows_of([k % 2 == 0 for k in range(30)]))
    assert same['estimate'] == 0 and not same['confirmed']
    noisy = A.confirmation(rows_of([True] * 3 + [False] * 9), rows_of([False] * 3 + [True] * 2 + [False] * 7))
    assert noisy['estimate'] > 0 and noisy['ci95'][0] <= 0 and not noisy['confirmed']  # A gain the CI does not support.


# ---- selection end to end ------------------------------------------------------------------------

def answer(request, good):
    objects = json.loads(request['messages'][1]['content'])['objects']
    return json.dumps({'placements': [{'id': o['id'], 'pos': ([0.6 + 1.2 * (j % 5), 0.6 + 1.2 * (j // 5), 0] if good
                                                              else [1, 1, 0]), 'yaw': 0} for j, o in enumerate(objects)]})


def infer_dir(path, data, good, config):
    """An infer.py-layout run (what infer_shard merge and greedy publish); good(uid) -> a passing or colliding answer."""
    rows, info = infer.load_eval(data, 'placement')
    identity = {'config': {'arm': 'placement', **config}, 'data': {'manifest_sha256': info['manifest_sha256']}}
    run = infer._hash(identity)
    path.mkdir(parents=True)
    (path / 'records.jsonl').write_bytes(d._jsonl({**{k: r[k] for k in ('uid', 'source', 'group')}, 'arm': 'placement',
                                                    'raw': answer(r, good(r['uid'])), 'error': None, 'run_sha256': run,
                                                    'reached_token_cap': False, 'stopped_on_eos': True} for r in rows))
    h.write_json(path / 'result.json', {'status': 'INFERENCE_COMPLETE', 'pending_count': 0})
    h.write_json(path / 'run_receipt.json', {**identity, 'run_sha256': run})
    h.write_json(path / 'MANIFEST.json', {'files': h.artifact_manifest(path), 'expected_uids': info['expected_uids']})


def slice_data(run, name, counts):
    tmp = run.parent / f'.{name.replace("/", "-")}'
    SV.make_data(tmp, counts)
    (run / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp / 'data').rename(run / name)


def selection_world(tmp_path, monkeypatch, common):
    run = tmp_path / 'project/runs/unknownsize-20261013'
    slice_data(run, 'dev-select-v1', [2] * 10 + [22, 22])
    slice_data(run, 'a021/confirm-v1', [2] * 30)
    slice_data(run, 'eval-data-v1', [2] * 8)
    d._publish_dir(run / 'a021/engine-check', {'engine.json': A._json({
        'protocol': 'amendment-021-engine-check', 'common_vllm': common, 'engines': A.engines(common)})})
    for rel in ('dev-eval-013/selection.json', 'dev-eval-018/selection.json'):
        (run / rel).parent.mkdir(parents=True)
        (run / rel).write_text('{}')
    monkeypatch.setattr(A.op, 'load_selection', lambda path: ({'adopted': False}, None))
    monkeypatch.setattr(A.ch, 'load_selection', lambda path: ({'adopted': True, 'selected': 'sft-d'}, None))
    arm = run / 'a021/27b'
    arm.mkdir(parents=True)
    (arm / 'a021-policy.json').write_text('{"stage": "sft"}')
    return run, arm


def run_jobs(run, jobs, quality):
    """Each job line's run, as a fake infer dir on its engine (quality(cand, slice) -> good(uid))."""
    specs = A.specs_of(run)
    for job in jobs:
        kind, slice_, cand = job[:3]
        if kind == 'hf':
            continue
        config = {'lora': A.VLLM_LORA if kind == 'vllm' else ds.EXACT, **({'model': str(Path(specs[cand]['model']).resolve())}
                                                                         if 'model' in specs[cand] else {})}
        path = A.vllm_dir(run, slice_, cand, job[3]) if kind == 'vllm' else A.hf_dir(run, slice_, cand, specs)
        infer_dir(path, A.data_dir(run, slice_), quality(cand, slice_), config)


def half(uid):
    return int(uid[1:]) % 2 == 0


def test_common_vllm_selection_confirms_the_27b_arm_and_writes_its_test_rows(tmp_path, monkeypatch):
    run, arm = selection_world(tmp_path, monkeypatch, common=True)
    jobs = A.job_list(run, 'candidates', 'A')
    cands = h.strict_json((run / 'a021/candidates.json').read_text())
    assert cands['serving_engine'] == {'r': 'vllm', 'd18': 'vllm', 'arm': 'vllm'} and cands['not_adopted'] == {
        'op13': h.sha256(run / 'dev-eval-013/selection.json')}
    assert cands['candidates']['d18']['model'] == str(run / 'distill-018/placement/sft-d-merged')
    assert cands['candidates']['arm'] == {'size': '27B', 'kind': 'A', 'exact_policy': True, 'model': str(arm),
                                          'base': str(tmp_path / 'project/models/Qwen3.8-27B'),
                                          'policy_sha256': h.sha256(arm / 'a021-policy.json'), 'hf': {}}
    assert jobs == [('vllm', 'dev-select', 'r', 'objects', 1), ('vllm', 'dev-select', 'r', 'hash', 1),
                    ('vllm', 'confirm', 'r', 'objects', 2), ('vllm', 'dev-select', 'd18', 'objects', 1),
                    ('vllm', 'confirm', 'd18', 'objects', 2), ('vllm', 'dev-select', 'arm', 'objects', 2),
                    ('vllm', 'confirm', 'arm', 'objects', 4)] and A.ceiling(jobs) == 13
    assert A.argv(run, 'vllm', 'confirm', 'arm', 'objects') == [
        'fastfill.comparison.a021_select', 'greedy', '--data', str(run / 'a021/confirm-v1'),
        '--base', str(tmp_path / 'project/models/Qwen3.8-27B'), '--model', str(arm), '--exact-policy', '--arm', 'placement',
        '--out', str(run / 'a021/vllm/confirm/arm'), '--batching', 'objects']
    with pytest.raises(ValueError, match='no new HF run'):
        A.argv(run, 'hf', 'confirm', 'arm', '0', '6')
    (run / 'a021/8b').mkdir()
    (run / 'a021/8b/a021-policy.json').write_text('{}')
    with pytest.raises(ValueError, match='Arm B is the fallback only'):
        A.job_list(run, 'candidates', 'B')
    run_jobs(run, jobs, lambda cand, slice_: (lambda uid: True) if cand == 'arm' else half)
    assert A.job_list(run, 'candidates', 'A') == []  # Finished runs are left out.
    result = A.select(ns(run_root=run, workers=1))
    assert result == {'status': 'A021_FINAL_SELECTED', 'dev_winner': 'arm', 'delivered': 'arm', 'engine': 'vllm'}
    dev = h.strict_json((run / 'a021/dev-selection.json').read_text())
    assert dev['dev']['r']['checker'] == 0.5 and dev['dev']['arm']['checker'] == 1.0 and dev['spread']['abs'] == 0
    assert dev['tie_band'] == 0.01 and dev['dev']['arm']['rooms_21plus'] == 2 and dev['eligible'] == ['r', 'd18', 'arm']
    final = A.load_final(run)
    x = final['confirmation']['contrast_winner_minus_r']
    assert x['confirmed'] and x['estimate'] == 0.5 and x['ci95'][0] > 0 and set(final['confirm_descriptive']) == {'d18'}
    report = (run / 'a021/final-report.md').read_text()
    assert '决定：交付 021 臂（A，27B）' in report and '确认通过' in report
    assert A.select(ns(run_root=run, workers=1))['delivered'] == 'arm'  # A rerun reproduces the write-once files.
    with pytest.raises(ValueError, match='no HF confirm runs'):
        A.job_list(run, 'confirm')
    test_jobs = A.job_list(run, 'test')
    assert test_jobs == [('vllm', 'test', 'arm', 'objects', 8), ('vllm', 'test', 'r', 'objects', 3)]
    run_jobs(run, test_jobs, lambda cand, slice_: (lambda uid: True) if cand == 'arm' else half)
    assert A.test_geometry(ns(run_root=run, workers=1)) == {'status': 'A021_TEST_ROWS_COMPLETE', 'arm': 1.0, 'r': 0.5}
    out = run / 'a021/test-geometry-arm'
    e._verified(out)
    rows = [json.loads(x) for x in (out / 'placement-geometry.jsonl').read_text().splitlines()]
    assert len(rows) == 8 and all({'reached_token_cap', 'stopped_on_eos', 'raw_layout'} <= r.keys() for r in rows)
    summary = h.strict_json((out / 'summary.json').read_text())
    assert summary['contrast_winner_minus_r']['geometry_success']['estimate'] == 0.5
    with pytest.raises(FileExistsError):
        A.test_geometry(ns(run_root=run, workers=1))


def test_a_dev_winner_that_fails_confirmation_leaves_r_delivered(tmp_path, monkeypatch):
    run, arm = selection_world(tmp_path, monkeypatch, common=True)
    jobs = A.job_list(run, 'candidates', 'A')
    run_jobs(run, jobs, lambda cand, slice_: (lambda uid: True) if cand == 'arm' and slice_ == 'dev-select' else half)
    assert A.select(ns(run_root=run, workers=1))['delivered'] == 'r'
    final = A.load_final(run)
    assert final['dev_winner'] == 'arm' and not final['confirmation']['contrast_winner_minus_r']['confirmed']
    assert '确认失败' in (run / 'a021/final-report.md').read_text()
    with pytest.raises(ValueError, match='R stays delivered'):
        A.job_list(run, 'test')


def test_split_engines_reuse_hf_dev_runs_and_confirm_an_hf_winner_after_its_runs(tmp_path, monkeypatch):
    run, arm = selection_world(tmp_path, monkeypatch, common=False)
    specs = A.fixed_specs(run)
    for rel, model in (('dev-eval-010/rpo-dpo1/infer-placement', specs['r']['model']),
                       ('dev-eval-018/sft-d/infer-placement', str(run / 'distill-018/placement/sft-d-merged'))):
        infer_dir(run / rel, run / 'dev-select-v1', half if 'rpo' in rel else (lambda uid: True),
                  {'lora': ds.EXACT, 'model': str(Path(model).resolve())})
    jobs = A.job_list(run, 'candidates', 'A')
    assert h.strict_json((run / 'a021/candidates.json').read_text())['serving_engine'] == {
        'r': 'hf', 'd18': 'hf', 'arm': 'vllm'}
    assert jobs == [('vllm', 'dev-select', 'r', 'objects', 1), ('vllm', 'dev-select', 'r', 'hash', 1),
                    ('vllm', 'dev-select', 'arm', 'objects', 2), ('vllm', 'confirm', 'arm', 'objects', 4),
                    ('vllm', 'confirm', 'r', 'objects', 2)]  # HF dev runs of record reused; no HF confirm yet.
    run_jobs(run, jobs, lambda cand, slice_: (lambda uid: True) if cand == 'arm' else half)
    result = A.select(ns(run_root=run, workers=1))
    assert result == {'status': 'A021_NEEDS_HF_CONFIRM', 'selected': 'd18', 'missing': ['d18', 'r']}  # 8B before 27B.
    assert not (run / 'a021/final-selection.json').exists()
    confirm = A.job_list(run, 'confirm')
    assert confirm[:6] == [('hf', 'confirm', 'd18', str(k), '6', 6, '') for k in range(6)]
    assert confirm[6] == ('hf-merge', 'confirm', 'd18', '6') and confirm[13] == ('hf-merge', 'confirm', 'r', '6')
    assert A.ceiling(confirm) == 72
    assert A.argv(run, 'hf', 'confirm', 'd18', '2', '6') == [
        'fastfill.comparison.infer_shard', 'shard', '--data', str(run / 'a021/confirm-v1'),
        '--base', str(tmp_path / 'project/models/Qwen3-8B'), '--arm', 'placement',
        '--model', str(run / 'distill-018/placement/sft-d-merged'), '--exact-policy',
        '--plan', str(run / 'a021/hf/confirm/plan'), '--shard-index', '2', '--shard-count', '6',
        '--out', str(run / 'a021/hf/confirm/d18/placement/shard-02')]
    assert A.argv(run, 'hf-merge', 'confirm', 'r', '2')[-3:] == ['--shards', *(
        str(run / f'a021/hf/confirm/r/placement/shard-0{k}') for k in range(2))][-3:]
    (run / 'a021/hf/confirm/r/placement/shard-03').mkdir(parents=True)
    assert A.job_list(run, 'confirm')[10] == ('hf', 'confirm', 'r', '3', '6', 6, '--resume')
    run_jobs(run, [('hf-merge', 'confirm', c) for c in ('d18', 'r')], lambda cand, slice_: (
        (lambda uid: True) if cand == 'd18' else half))
    assert A.select(ns(run_root=run, workers=1)) == {'status': 'A021_FINAL_SELECTED', 'dev_winner': 'd18',
                                                     'delivered': 'd18', 'engine': 'hf'}
    final = A.load_final(run)
    assert final['size'] == '8B' and set(final['confirm_descriptive']) == {'arm'}
    assert A.specs_of(run)['d18']['hf']['test'] == str(run / 'infer-placement-007')
    with pytest.raises(FileNotFoundError, match='run of record is missing'):
        A.job_list(run, 'test')  # 018's own HF test run, never a second one.
    model = {'lora': ds.EXACT, 'model': str((run / 'distill-018/placement/sft-d-merged').resolve())}
    infer_dir(run / 'infer-placement-007', run / 'eval-data-v1', lambda uid: True, model)
    infer_dir(run / 'infer-placement-005', run / 'eval-data-v1', half,
              {'lora': ds.EXACT, 'model': str(Path(specs['r']['model']).resolve())})
    assert A.job_list(run, 'test') == []  # Both HF test runs of record stand.
    assert A.test_geometry(ns(run_root=run, workers=1)) == {'status': 'A021_TEST_ROWS_COMPLETE', 'd18': 1.0, 'r': 0.5}


def test_select_refuses_unbound_runs_and_a_drifted_engine_check(tmp_path, monkeypatch):
    run, arm = selection_world(tmp_path, monkeypatch, common=True)
    jobs = A.job_list(run, 'candidates', 'A')
    run_jobs(run, jobs, lambda cand, slice_: half)
    shutil.rmtree(A.vllm_dir(run, 'dev-select', 'd18'))
    infer_dir(A.vllm_dir(run, 'dev-select', 'd18'), run / 'dev-select-v1', half,
              {'lora': A.VLLM_LORA, 'model': str(arm.resolve())})  # Another model's output under d18's name.
    with pytest.raises(ValueError, match='inference is not placement'):
        A.select(ns(run_root=run, workers=1))
    shutil.rmtree(run / 'a021/engine-check')
    d._publish_dir(run / 'a021/engine-check', {'engine.json': A._json({
        'protocol': 'amendment-021-engine-check', 'common_vllm': False, 'engines': A.engines(False)})})
    with pytest.raises(ValueError, match='not bound to the engine check'):
        A.select(ns(run_root=run, workers=1))


def test_engine_check_rule_on_hf_and_vllm_runs(tmp_path, monkeypatch):
    run, _ = selection_world(tmp_path, monkeypatch, common=True)
    shutil.rmtree(run / 'a021/engine-check')
    slice_data(run, 'a021/engine-v1', [2] * 20)
    specs = A.fixed_specs(run)
    exact = {'model': str(Path(specs['r']['model']).resolve())}
    sft = {'adapter': str(Path(specs['psft']['adapter']).resolve()), 'training_run': str(Path(specs['psft']['training_run']).resolve())}
    data = run / 'a021/engine-v1'
    infer_dir(A.hf_dir(run, 'engine', 'r', specs), data, lambda uid: int(uid[1:]) < 10, {'lora': ds.EXACT, **exact})
    infer_dir(A.vllm_dir(run, 'engine', 'r'), data, lambda uid: int(uid[1:]) < 10, {'lora': A.VLLM_LORA, **exact})
    infer_dir(A.hf_dir(run, 'engine', 'psft', specs), data, lambda uid: int(uid[1:]) < 10,
              {'lora': 'merge_and_unload_bf16', **sft})
    infer_dir(A.vllm_dir(run, 'engine', 'psft'), data, lambda uid: int(uid[1:]) < 9, {'lora': A.VLLM_LORA, **sft})
    assert A.engine_check(ns(run_root=run, workers=1)) == {'status': 'A021_ENGINE_CHECKED', 'common_vllm': True,
                                                           'gap': {'r': 0.0, 'psft': 0.05}}
    value, _ = A.load_engine(run)
    assert value['identical_raw_fraction'] == {'r': 1.0, 'psft': 0.95} and value['engines'] == A.engines(True)
    assert A.job_list(run, 'engine') == []


# ---- O2nt scoring --------------------------------------------------------------------------------

def test_o2nt_score_scores_placement_runs_and_rechecks_zero_reasoning(tmp_path, monkeypatch, world):  # noqa: F811
    root, rooms = world
    run = tmp_path / 'project/runs/unknownsize-20261013'
    run.mkdir(parents=True)
    for name in ('dev-select-v1', 'eval-data-v1', 'a021/confirm-v1'):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / 'data', run / name)
    shutil.copy(root / 'cohort.json', run / 'llm-cohort-v1.json')
    d._publish_dir(run / 'a021/cohorts', {f'{s}.json': A._json(O.all_rooms_cohort(A.data_dir(run, s)))
                                          for s in ('dev-select', 'confirm')})
    for slice_ in A.O2NT:
        items = A.argv(run, 'o2nt', slice_)
        assert items[:3] == ['fastfill.comparison.llm_open', 'run', '--system'] and ('--all-rooms' in items) == (slice_ != 'cohort')
        argv = [x if x != str(A.models_root(run)) else str(root / 'models') for x in items[1:]]
        fake = TO.FakeLLM(rooms, default='{valid}', arm='placement')
        assert O.main(argv, TO.loader(fake)) == 0
    result = A.o2nt_score(ns(run_root=run, workers=1))
    value = h.strict_json((run / 'a021/o2nt-scores/scores.json').read_text())
    cohort = h.strict_json((run / 'llm-cohort-v1.json').read_text())
    assert value['slices']['cohort']['rooms'] == len(cohort['uids']) and value['slices']['confirm']['rooms'] == 39
    harness = value['slices']['dev-select']['variants']['llm_harness']
    assert harness['complete_rate'] == 1.0 and result['dev-select'] == harness['checker']
    assert all(v['reasoning_tokens'] == 0 for v in value['slices'].values())
    assert 'O2nt 参考行' in (run / 'a021/o2nt-scores/report.md').read_text()
    status = run / 'a021/o2nt/cohort/result.json'  # A run whose result admits reasoning is refused.
    shutil.rmtree(run / 'a021/o2nt-scores')
    payload = h.strict_json(status.read_text())
    payload['tokens']['reasoning_tokens'] = 3
    h.write_json(status, payload)
    manifest = h.strict_json((run / 'a021/o2nt/cohort/MANIFEST.json').read_text())
    h.write_json(run / 'a021/o2nt/cohort/MANIFEST.json', {**manifest, 'files': {**manifest['files'],
                                                                                 'result.json': h.sha256(status)}})
    with pytest.raises(ValueError, match='O2nt reasoned on cohort'):
        A.o2nt_score(ns(run_root=run, workers=1))


# ---- Slurm scripts -------------------------------------------------------------------------------

@pytest.mark.parametrize('argv, ok', [
    (['confirm'], True), (['confirm', 'x'], False), (['plan', 'engine'], True), (['plan', 'test'], False),
    (['vllm', 'dev-select', 'r', 'objects', '1'], True), (['vllm', 'confirm', 'arm', 'hash', '4'], True),
    (['vllm', 'confirm', 'rpo', 'objects', '1'], False), (['vllm', 'confirm', 'r', 'greedy', '1'], False),
    (['vllm', 'confirm', 'r', 'objects', '0'], False), (['vllm', 'cohort', 'r', 'objects', '1'], False),
    (['hf', 'confirm', 'd18', '5', '6', '6'], True), (['hf', 'confirm', 'd18', '5', '6', '6', '--resume'], True),
    (['hf', 'confirm', 'd18', '6', '6', '6'], False), (['hf', 'confirm', 'd18', '0', '6', '6', '--again'], False),
    (['hf-merge', 'confirm', 'r', '6'], True), (['hf-merge', 'confirm', 'r'], False), (['o2nt', 'cohort', '2'], True),
    (['o2nt', 'engine', '2'], False), (['o2nt-score'], True), (['engine-check'], True), (['select'], True),
    (['test-geometry'], True), (['select', 'x'], False), (['bogus'], False)])
def test_sbatch_modes_validate_arguments_before_the_allocation(tmp_path, argv, ok):
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    done = subprocess.run(['bash', str(SCRIPTS / 'isambard_021_eval.sbatch'), str(CODE), *argv], env=env,
                          capture_output=True, text=True)
    assert (done.returncode, 'Slurm allocation required' in done.stderr) == ((1, True) if ok else (2, False))


def submit_world(tmp_path):
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    for path in ('main-data-v1/MANIFEST.json', 'sft-joint-001/SHA256.json', 'sft-placement-001/SHA256.json',
                 'dev-select-v1/MANIFEST.json', 'llm-cohort-v1.json', 'context-capacity-001/SHA256.json',
                 'dpo-003/placement/dpo1-merged.json', 'dev-eval-013/selection.json', 'dev-eval-018/selection.json',
                 'a021/27b/a021-policy.json'):
        (run / path).parent.mkdir(parents=True, exist_ok=True)
        (run / path).write_text('{}')
    for name in ('Qwen3-8B', 'Qwen3.8-27B'):
        (project / 'models' / name).mkdir(parents=True)
    python = project / 'env/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\necho "$@" >> "$PYLOG"\nprintf "%s\\n" "$JOBS_OUT"\n')
    python.chmod(0o755)
    (tmp_path / 'squeue').write_text('#!/bin/sh\ncase "$*" in\n'
                                     '"-h -u me -n unknownsize-021-eval -o %i") printf "%s" "$SQUEUE_OUT" ;;\n'
                                     '"-h -u me -o %i %j") printf "%s" "$SQUEUE_ALL" ;;\n*) exit 3 ;;\nesac\n')
    (tmp_path / 'sacct').write_text('#!/bin/sh\nF=JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES\ncase "$*" in\n'
                                    '"-n -X -P -u me -S 2026-10-14 -o $F") printf "%s" "$SACCT_OUT" ;;\n'
                                    '"-n -X -P -j $SACCT_IDS -o $F") printf "%s" "$SACCT_BY_ID" ;;\n*) exit 3 ;;\nesac\n')
    for name in ('squeue', 'sacct'):
        (tmp_path / name).chmod(0o755)
    script = tmp_path / 'submit_021_eval.sh'
    original = (SCRIPTS / 'submit_021_eval.sh').read_text()
    line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    assert original.count(line) == 1
    script.write_text(original.replace(line, f'PROJECT_ROOT={project}\n'))
    code = tmp_path / 'main-abc'
    code.mkdir()
    (code / 'fastfill').symlink_to(CODE / 'fastfill')
    (code / DOC).parent.mkdir(parents=True)
    (code / DOC).write_text('# Amendment 021\n')
    return script, run, code


def test_submit_021_stages_dependencies_gpu_cap_and_refusals(tmp_path):
    for name in ('isambard_021_eval.sbatch', 'submit_021_eval.sh'):
        subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me', 'PYLOG': str(tmp_path / 'py.log')}
    script, run, code = submit_world(tmp_path)
    sent = []

    def submit(*argv, **extra):
        before = len(sent)
        done = subprocess.run(['bash', str(script), str(code), *argv], env={**env, **extra}, capture_output=True, text=True)
        log = tmp_path / 'log'
        sent[:] = log.read_text().splitlines() if log.exists() else []
        return done, [x.replace(f'{code}/fastfill/comparison/', '').replace(f' {code} ', ' CODE ') for x in sent[before:]]
    kill, J = '--parsable --kill-on-invalid-dep=yes --dependency=afterok:', 'isambard_021_eval.sbatch CODE'
    done, log = submit('confirm', SACCT_OUT='1|unknownsize-021-train|COMPLETED|36000|720|cpu=288,gres/gpu=4,node=1|gres/gpu=4\n'
                                            '2|unknownsize-018|COMPLETED|36000|720|gres/gpu=4|gres/gpu=4\n')
    assert done.returncode == 0, done.stderr
    assert 'GPU-h: used 40.00 + ceiling 0 <= 200' in done.stdout  # Only unknownsize-021* jobs count.
    assert log == [f'--parsable --hold {J} confirm', f'{kill}101 {J} plan engine', f'{kill}101 {J} plan confirm']
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101\n'
    a21 = run / 'a021'
    (a21 / 'confirm-v1').mkdir(parents=True)
    done, log = submit('confirm')
    assert log == [] and done.returncode == 1 and f'refusing to reuse existing {a21 / "confirm-v1"}' in done.stderr
    for path in ('confirm-v1/MANIFEST.json', 'cohorts/MANIFEST.json', 'engine-v1/MANIFEST.json',
                 'hf/engine/plan/MANIFEST.json', 'engine-check/MANIFEST.json', 'hf/confirm/plan/MANIFEST.json'):
        (a21 / path).parent.mkdir(parents=True, exist_ok=True)
        (a21 / path).write_text('{}')
    o2nt = 'o2nt dev-select 1\no2nt confirm 3\no2nt cohort 2\nceiling 12'
    done, log = submit('o2nt', JOBS_OUT=o2nt)
    assert done.returncode == 0, done.stderr
    o2 = '--gpus-per-node=2 --cpus-per-task=32 --mem=200G'
    assert log == [f'--parsable --hold {o2} --time=01:00:00 {J} o2nt dev-select 1',
                   f'--parsable --hold {o2} --time=03:00:00 {J} o2nt confirm 3',
                   f'--parsable --hold {o2} --time=02:00:00 {J} o2nt cohort 2', f'{kill}104:105:106 {J} o2nt-score']
    assert (tmp_path / 'log.scontrol').read_text().splitlines()[-1] == 'scontrol release 104,105,106'
    assert (tmp_path / 'py.log').read_text().split()[-4:] == ['jobs', 'o2nt', '--run-root', str(run)]
    engine = ('hf engine r 0 2 3\nhf engine r 1 2 3 --resume\nhf-merge engine r 2\nvllm engine r objects 1\n'
              'vllm engine psft objects 1\nceiling 8')
    shard, gpu1 = '--gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=03:00:00', '--gpus-per-node=1 --cpus-per-task=16 --mem=100G'
    done, log = submit('engine', JOBS_OUT=engine)
    assert done.returncode == 1 and f'refusing to reuse existing {a21 / "engine-check"}' in done.stderr and log == []
    shutil.rmtree(a21 / 'engine-check')
    done, log = submit('engine', JOBS_OUT=engine)
    assert done.returncode == 0, done.stderr
    assert log == [f'--parsable --hold {shard} {J} hf engine r 0 2 3', f'--parsable --hold {shard} {J} hf engine r 1 2 3 --resume',
                   f'{kill}108:109 {J} hf-merge engine r 2', f'--parsable --hold {gpu1} --time=01:00:00 {J} vllm engine r objects 1',
                   f'--parsable --hold {gpu1} --time=01:00:00 {J} vllm engine psft objects 1', f'{kill}110:111:112 {J} engine-check']
    (a21 / 'engine-check').mkdir()
    (a21 / 'engine-check/MANIFEST.json').write_text('{}')
    cands = 'vllm dev-select r objects 1\nvllm dev-select arm objects 2\nceiling 3'
    done, log = submit('candidates', 'A', JOBS_OUT=cands,
                       SACCT_OUT='3|unknownsize-021-train|CANCELLED by 1|709200|4320|gres/gpu=1|gres/gpu=1\n')  # 197 + 3 = 200.
    assert done.returncode == 0 and 'used 197.00 + ceiling 3 <= 200' in done.stdout, done.stderr
    assert log[-1] == f'{kill}114:115 {J} select'
    assert (tmp_path / 'py.log').read_text().split()[-6:] == ['jobs', 'candidates', '--run-root', str(run), '--arm-kind', 'A']
    done, log = submit('candidates', 'A', JOBS_OUT=cands,
                       SACCT_OUT='3|unknownsize-021-train|COMPLETED|712800|4320|gres/gpu=1|gres/gpu=1\n')  # 198 + 3 > 200.
    assert log == [] and done.returncode == 1 and 'GPU cap: used 198.00 + ceiling 3 > 200' in done.stderr
    running = '5|unknownsize-021-train|RUNNING|3600|1500|gres/gpu=4|gres/gpu=4\n'  # Charged its 25 h limit: 100.
    done, log = submit('candidates', 'A', JOBS_OUT=cands, SACCT_OUT=running, SQUEUE_ALL='5 unknownsize-021-train\n6 '
                       'unknownsize-021-diag\n7 unknownsize-018-x\n', SACCT_IDS='5,6',  # 6 waits on a dependency: by id.
                       SACCT_BY_ID=running + '6|unknownsize-021-diag|PENDING|0|1500||gres/gpu=4\n')
    assert log == [] and done.returncode == 1 and 'GPU cap: used 200.00 + ceiling 3 > 200' in done.stderr, done.stderr
    done, log = submit('candidates', 'B', JOBS_OUT=cands)
    assert log == [] and done.returncode == 1 and f"missing input {run / 'a021/8b/a021-policy.json'}" in done.stderr
    done, log = submit('candidates', 'A', JOBS_OUT=cands, SQUEUE_OUT='7400001\n')
    assert log == [] and done.returncode == 1 and 'scancel 7400001' in done.stderr
    done, log = submit('candidates', 'C', JOBS_OUT=cands)
    assert done.returncode == 2 and log == []
    done, log = submit('candidates', 'A', JOBS_OUT='ceiling 0')
    assert done.returncode == 0 and log == [f'--parsable --hold {J} select']  # Nothing left to run: select alone.
    done, log = submit('test', JOBS_OUT='ceiling 0')
    assert done.returncode == 1 and f'missing input {a21 / "final-selection.json"}' in done.stderr
    (a21 / 'final-selection.json').write_text('{}')
    done, log = submit('test', JOBS_OUT='vllm test arm objects 8\nvllm test r objects 3\nceiling 11')
    assert done.returncode == 0 and log[-1] == f'{kill}118:119 --time=06:00:00 {J} test-geometry', done.stderr
    done, log = submit('test', JOBS_OUT='bogus line\nceiling 0')
    assert done.returncode == 1 and "unknown job line 'bogus line'" in done.stderr
    done, log = submit('select', JOBS_OUT='ceiling 0')
    assert done.returncode == 1 and 'missing input' in done.stderr
    assert submit('bogus')[0].returncode == 2 and submit('test', 'x')[0].returncode == 2


def test_new_module_leaves_existing_defaults_alone():
    assert d.receipt_path(Path('/x/m')) == Path('/x/m.json')
    assert (sv.N, sv.ENGINE['max_lora_rank'], O.SYSTEMS['O2']['template']) == (8, 32, {'enable_thinking': True})
    assert list(O.SYSTEMS) == list(TO.JOINT_IDENTITY_SHAPE)  # test_llm_open pins every default identity's shape.
