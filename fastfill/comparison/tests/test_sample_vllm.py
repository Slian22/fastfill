"""S2 sampling on CPU: a fake vLLM engine, a one-token-per-character tokenizer and patched model identities."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from fastfill.comparison import infer
from fastfill.comparison import infer_shard
from fastfill.comparison import llm_open
from fastfill.comparison import sample_vllm as sv
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.tests import test_dpo_stages as D
from fastfill.comparison.tests import test_infer_shard as T

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
CAPS = {1: 1152, 2: 1280, 3: 1408}  # 1024 + 128 x objects for the rooms of make_data (short prompts).
IDENTITY = {'training_run_sha256': 'c' * 64, 'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': 'a' * 64},
            'adapter_files': {}, 'adapter_sha256': 'e' * 64}


CATEGORIES = ('cabinet', 'sofa', 'wardrobe', 'bookshelf')


def make_data(root, counts=T.OBJECTS, sources=('S0', 'S1')):
    """eval-data-v1 layout; room k has counts[k] objects on a 1.5 m grid, its own width (distinct prompts) and the
    GT sizes as the size prior."""
    data, rooms = root / 'data', []
    for k, count in enumerate(counts):
        objects = [{'id': f'o{j}', 'category': CATEGORIES[j % 4], 'size': [0.6, 0.6, 0.8],
                    'pos': [1 + 1.5 * (j % 4), 1 + 1.5 * (j // 4), 0], 'yaw': 0} for j in range(count)]
        width = 7 + k / 100
        rooms.append(bind_room({'uid': f'r{k}', 'source': sources[k % len(sources)], 'group': f'g{k}',
                                'room_type': 'bedroom', 'boundary_type': 'polygon',
                                'boundary': [[0, 0], [width, 0], [width, 7], [0, 7]], 'height': 3,
                                'objects': objects, 'fixed': []})[0])
    files = {'rooms.jsonl': rooms, 'size-estimates.jsonl': [], 'joint/requests.jsonl': [], 'placement/requests.jsonl': []}
    for room in rooms:
        sizes = {o['id']: o['size'] for o in room['objects']}
        files['size-estimates.jsonl'].append({'uid': room['uid'], 'estimated_sizes': sizes})
        for arm in h.ARMS:
            files[f'{arm}/requests.jsonl'].append({**{k: room[k] for k in ('uid', 'source', 'group')},
                                                   'messages': messages(room, arm, sizes if arm == 'placement' else None)[:2]})
    for name, rows in files.items():
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    h.write_json(data / 'MANIFEST.json', {'expected_uids': [r['uid'] for r in rooms], 'training_manifest_sha256': 'a' * 64,
                                          'files': h.artifact_manifest(data)})
    return data


class Tok(T.Tok):
    """T.Tok plus the vocabulary lookup; prompts in `long` are padded past the context."""
    def __init__(self, long=()):
        self.long = set(long)

    def apply_chat_template(self, messages, **kwargs):
        text = super().apply_chat_template(messages, **kwargs)
        return text + 'x' * sv.MAX_MODEL_LEN if text in self.long else text

    def get_vocab(self):
        return {sv.EOS: T.EOS}


def spaced(request, uid, k):
    """Valid spaced layout, shifted by k cm, so every (uid, k) has its own text."""
    return json.dumps({'placements': [
        {'id': o['id'], **({} if 'size' in o else {'size': [0.5, 0.5, 0.5]}), 'pos': [0.6 + j, 0.6 + k / 100, 0], 'yaw': 0}
        for j, o in enumerate(request['objects'])]}), 'stop'


class FakeLLM:
    """vLLM stand-in: reads the request from the prompt ids, recovers k from the sampling params, answers script()."""
    def __init__(self, data, script=spaced, fail_on_call=None):
        self.uids = {r['messages'][1]['content']: r['uid'] for arm in h.ARMS for r in infer.load_eval(data, arm)[0]}
        self.script, self.fail_on_call, self.calls = script, fail_on_call, []

    def generate(self, prompts, params, lora_request=None, use_tqdm=True):
        assert use_tqdm is False and len(prompts) == len(params)
        self.calls.append({'prompts': prompts, 'params': params, 'lora': lora_request, 'uids': []})
        if self.fail_on_call == len(self.calls):
            raise RuntimeError('CUDA error: simulated')
        outputs = []
        for prompt, p in zip(prompts, params):
            ids = prompt['prompt_token_ids']
            content = ''.join(map(chr, ids))
            uid = self.uids[content]
            k = 0 if p['temperature'] == 0 else next(k for k in range(1, sv.N + 1) if sv.sample_seed(uid, k) == p['seed'])
            self.calls[-1]['uids'].append((uid, k))
            text, finish = self.script(json.loads(content), uid, k)
            tokens = [ord(c) for c in text][:p['max_tokens']] + ([T.EOS] if finish == 'stop' else [])
            outputs.append(SimpleNamespace(prompt_token_ids=list(ids), outputs=[SimpleNamespace(
                token_ids=tokens, finish_reason=finish, stop_reason=T.EOS if finish == 'stop' else None)]))
        return outputs


def loader(fake, engines=None):
    def load(engine):
        if engines is not None:
            engines.append(engine)
        return fake, dict, lambda name, index, path: (name, index, path)
    return load


def argv(root, data, arm='joint', out='samples', *extra):
    source = (['--adapter', str(root / 'sft' / arm / 'adapter'), '--training-run', str(root / 'sft')] if arm == 'joint'
              else ['--model', str(root / 'dpo-003/placement/dpo1-merged'), '--exact-policy'])
    return ['run', '--data', str(data), '--base', str(root / 'base'), '--arm', arm, '--out', str(root / out), *source, *extra]


def finalize(root, data, out='samples'):
    return sv.finalize(sv.parse_args(['finalize', '--data', str(data), '--out', str(root / out)]))


def patched(tmp_path, monkeypatch, data):
    """Model identities and the tokenizer patched; returns the world namespace."""
    runs = []
    def sft(run):
        runs.append(run)
        return IDENTITY
    monkeypatch.setattr(sv.infer, 'training_identity', sft)
    monkeypatch.setattr(sv.d, 'inference_identity', lambda run: {
        'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': 'a' * 64}, 'stage': 'dpo1',
        'lineage_sha256': '0' * 64, 'policy': {'mode': 'unmerged_fp32_adapter_on_bf16_training_base',
                                               'training_base': str(tmp_path / 'dpo-003/placement/sft-merged'),
                                               'adapter': str(tmp_path / 'dpo-003/placement/dpo1-adapter')}})
    monkeypatch.setattr(sv.d, '_model_tokenizer', lambda path: Tok())
    return SimpleNamespace(root=tmp_path, data=data, runs=runs)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return patched(tmp_path, monkeypatch, make_data(tmp_path))


def test_params_cap_prompt_engine_and_one_lora_request(world):
    root, data = world.root, world.data
    common = {'n': 1, 'max_tokens': 100, 'skip_special_tokens': False, 'stop_token_ids': [7], 'ignore_eos': True}
    assert sv.params(100, 'u', 0, 7) == {**common, 'temperature': 0.0}  # Stop on im_end only, as HF.
    seed = int(hashlib.sha256(b's2-bo-v1:u:3').hexdigest()[:8], 16)
    assert sv.params(100, 'u', 3, 7) == {**common, 'temperature': 0.7, 'top_p': 0.95, 'seed': seed}
    assert len({sv.sample_seed(u, k) for u in ('u', 'v') for k in range(1, 9)}) == 16
    assert (sv.N, sv.TEMPERATURE, sv.TOP_P, sv.MAX_MODEL_LEN) == (8, 0.7, 0.95, 40960)
    assert (llm_open.CAP_BASE, llm_open.CAP_PER_OBJECT) == (infer_shard.CAP_BASE, infer_shard.CAP_PER_OBJECT) == (1024, 128)
    fake, engines = FakeLLM(data), []
    budget = 9 * (2 * CAPS[3] + CAPS[2])  # Rooms by size: 3, 3, 2 | 2, 2, 1 | 1 objects.
    sv.main(argv(root, data, 'joint', 'samples', '--chunk-tokens', str(budget)), loader(fake, engines))
    run = world.runs[0]
    assert (run.context_budget, run.max_input_tokens, run.max_new_tokens, run.arm) == (40960, 40959, 40959, 'joint')
    assert engines == [{**sv.ENGINE, 'model': str((root / 'base').resolve())}]
    adapter = str((root / 'sft/joint/adapter').resolve())
    assert {c['lora'] for c in fake.calls} == {('joint', 1, adapter)} and [len(c['prompts']) for c in fake.calls] == [27, 27, 9]
    assert [sum(p['max_tokens'] for p in c['params']) for c in fake.calls] == [budget, 9 * (2 * CAPS[2] + CAPS[1]), 9 * CAPS[1]]
    records = {r['uid']: r for r in infer.load_eval(data, 'joint')[0]}
    for call in fake.calls:
        assert [k for _, k in call['uids']] == list(range(9)) * (len(call['uids']) // 9)  # A room's 9 requests together.
        for (uid, k), prompt, p in zip(call['uids'], call['prompts'], call['params']):
            ids = infer.prompt_ids(Tok(), records[uid])
            assert prompt == {'prompt_token_ids': ids}
            objects = len(json.loads(records[uid]['messages'][1]['content'])['objects'])
            assert p == sv.params(min(40960 - len(ids), 1024 + 128 * objects), uid, k, T.EOS)
    receipt = h.strict_json((root / 'samples/run_receipt.json').read_text())
    assert receipt['model'] == IDENTITY and receipt['loaded']['lora_request'] == ['joint', 1, adapter]
    assert receipt['run_sha256'] == infer._hash({k: v for k, v in receipt.items() if k != 'run_sha256'})
    assert {'sample_vllm.py', 'infer.py', 'llm_open.py'} <= set(receipt['code_sha256'])
    loaded = [json.loads(x) for x in (root / 'samples/invocations.jsonl').read_text().splitlines()][1]
    assert loaded['event'] == 'loaded' and loaded['sampling_params'][0] == repr(sv.params(1, 'uid', 0, T.EOS))
    assert finalize(root, data)['sampling_params_repr'] == [loaded['sampling_params']]


def test_chunks_cut_by_cap_tokens_with_at_least_one_room():
    jobs = [(None, None, cap) for cap in (10, 10, 5, 100, 1)]
    assert [[j[2] for j in c] for c in sv.chunks(jobs, 9 * 20)] == [[10, 10], [5], [100], [1]]
    assert [len(c) for c in sv.chunks(jobs, 10 ** 9)] == [5] and sv.CHUNK_TOKENS == 2_000_000


def test_shards_in_any_order_plus_smoke_finalize_byte_identical(world):
    root, data = world.root, world.data
    first = FakeLLM(data)
    sv.main(argv(root, data, 'joint', 'a', '--limit', '2'), loader(first))  # Smoke: the two largest rooms.
    assert {u for u, _ in first.calls[0]['uids']} == {'r1', 'r4'}
    for index in (1, 0):
        sv.main(argv(root, data, 'joint', 'a', '--shard-index', str(index), '--shard-count', '2'), loader(first))
    for index in (0, 1):
        sv.main(argv(root, data, 'joint', 'b', '--shard-index', str(index), '--shard-count', '2'), loader(FakeLLM(data)))
    generated = [u for c in first.calls for u, k in c['uids'] if k == 0]
    assert sorted(generated) == sorted(r['uid'] for r in infer.load_eval(data, 'joint')[0])  # Each room once.
    results = [finalize(root, data, out) for out in ('a', 'b')]
    assert (root / 'a/samples.jsonl').read_bytes() == (root / 'b/samples.jsonl').read_bytes()
    assert results[0]['status'] == 'SAMPLING_COMPLETE' and results[0]['samples'] == 7 * 9 and not results[0]['quality_claim']
    assert results[0]['run_sha256'] == results[1]['run_sha256'] and results[0]['invocations'] == 3
    rows = [json.loads(x) for x in (root / 'a/samples.jsonl').read_text().splitlines()]
    expected = h.strict_json((data / 'MANIFEST.json').read_text())['expected_uids']
    assert [(r['uid'], r['k']) for r in rows] == [(u, k) for u in expected for k in range(9)]
    assert all(r['finish_reason'] == 'stop' and r['stopped_on_eos'] and r['raw'] == spaced(
        json.loads(infer.load_eval(data, 'joint')[0][expected.index(r['uid'])]['messages'][1]['content']), r['uid'], r['k'])[0]
        for r in rows)
    manifest = h.strict_json((root / 'a/MANIFEST.json').read_text())
    assert manifest['expected_uids'] == expected and {'samples.jsonl', 'result.json', 'invocations.jsonl'} <= set(manifest['files'])
    # The DPO arm loads its exact policy: the training base with the unmerged stage adapter.
    fake, engines = FakeLLM(data), []
    sv.main(argv(root, data, 'placement', 'p'), loader(fake, engines))
    assert engines[0]['model'] == str(root / 'dpo-003/placement/sft-merged')
    assert fake.calls[0]['lora'] == ('placement', 1, str(root / 'dpo-003/placement/dpo1-adapter'))
    assert finalize(root, data, 'p')['arm'] == 'placement'


def test_failed_chunk_resumes_pending_rooms_only_and_identity_is_binding(world, monkeypatch):
    root, data = world.root, world.data
    with pytest.raises(RuntimeError, match='simulated'):
        sv.main(argv(root, data, 'joint', 'samples', '--chunk-tokens', str(9 * 2 * CAPS[3])),
                loader(FakeLLM(data, fail_on_call=2)))
    assert len(list((root / 'samples/receipts').iterdir())) == 2
    fake = FakeLLM(data)
    sv.main(argv(root, data, 'joint', 'samples', '--chunk-tokens', str(9 * 2 * CAPS[3])), loader(fake))
    done = {u for c in fake.calls for u, _ in c['uids']}
    assert len(done) == 5 and not done & {'r1', 'r4'}  # The first chunk (largest rooms) is never regenerated.
    monkeypatch.setattr(sv, 'TEMPERATURE', 0.8)
    with pytest.raises(ValueError, match='identity mismatch'):
        sv.main(argv(root, data, 'joint', 'samples'), loader(FakeLLM(data)))
    monkeypatch.setattr(sv, 'TEMPERATURE', 0.7)
    with pytest.raises(ValueError, match='pass --adapter'):
        sv.main([*argv(root, data, 'joint', 'samples'), '--exact-policy'], loader(FakeLLM(data)))
    monkeypatch.setattr(sv.infer, 'training_identity', lambda run: {**IDENTITY, 'training_data': {'manifest_sha256': 'f' * 64}})
    with pytest.raises(ValueError, match='training data'):  # As dev_select._scored: the model must match the data.
        sv.main(argv(root, data, 'joint', 'other'), loader(FakeLLM(data)))


def test_prompt_without_room_left_is_an_input_too_long_receipt(world, monkeypatch):
    root, data = world.root, world.data
    record = infer.load_eval(data, 'joint')[0][0]
    monkeypatch.setattr(sv.d, '_model_tokenizer', lambda path: Tok(long=[record['messages'][1]['content']]))
    fake = FakeLLM(data)
    sv.main(argv(root, data), loader(fake))
    assert record['uid'] not in {u for c in fake.calls for u, _ in c['uids']}
    receipt = h.strict_json(sv._path(root / 'samples', record['uid']).read_text())
    assert receipt['error'] == 'INPUT_TOO_LONG' and receipt['input_tokens'] >= 40960 and receipt['max_tokens'] is None
    assert [s['k'] for s in receipt['samples']] == list(range(9)) and {s['raw'] for s in receipt['samples']} == {None}
    assert finalize(root, data)['input_too_long_rooms'] == 1


def test_finalize_refuses_missing_tampered_short_receipts_and_existing_output(world):
    root, data = world.root, world.data
    sv.main(argv(root, data, 'joint', 'samples', '--limit', '3'), loader(FakeLLM(data)))
    with pytest.raises(ValueError, match='missing or extra receipts'):
        finalize(root, data)
    sv.main(argv(root, data), loader(FakeLLM(data)))
    path = sv._path(root / 'samples', 'r3')
    original = path.read_text()
    for edit, match in ((lambda v: {**v, 'run_sha256': 'f' * 64}, 'receipt identity mismatch'),
                        (lambda v: {**v, 'samples': v['samples'][:-1]}, r'k = 0\.\.8')):
        path.write_text(json.dumps(edit(json.loads(original))) + '\n')
        with pytest.raises(ValueError, match=match):
            finalize(root, data)
    path.write_text(original)
    assert finalize(root, data)['status'] == 'SAMPLING_COMPLETE'
    with pytest.raises(FileExistsError):
        finalize(root, data)
    fake = FakeLLM(data)
    assert sv.run(sv.parse_args(argv(root, data)), loader(fake))['status'] == 'SAMPLING_COMPLETE'
    assert fake.calls == []  # A finalized run is read-only.


def test_build_writes_every_dev_room_in_the_eval_layout(tmp_path):
    dev = [D.spaced(uid, source) for uid, source in (('d2', 'IL3D_3dfront'), ('d0', 'Structured3D'), ('d1', 'X'))]
    main = D.build_main(tmp_path, dev=dev)
    out = tmp_path / 'dev-full-v1'
    assert sv.main(['build', '--data', str(main), '--out', str(out)]) is None
    manifest = h.strict_json((out / 'MANIFEST.json').read_text())
    assert manifest['expected_uids'] == ['d0', 'd1', 'd2'] and manifest['counts'] == {'dev': 3}
    assert manifest['training_manifest_sha256'] == h.sha256(main / 'MANIFEST.json')
    rows = {arm: h._by_uid([json.loads(x) for x in (main / arm / 'dev.jsonl').read_text().splitlines()]) for arm in h.ARMS}
    for arm in h.ARMS:
        requests, _ = infer.load_eval(out, arm)  # The unchanged eval-data-v1 loader and request validator.
        assert all(r['messages'] == rows[arm][r['uid']]['messages'][:2] and r['split'] == 'dev' for r in requests)
    sizes = h._by_uid(h._rows(out, 'size-estimates.jsonl', manifest['files']['size-estimates.jsonl']))
    for uid, row in rows['placement'].items():
        request = json.loads(row['messages'][1]['content'])
        assert sizes[uid]['estimated_sizes'] == {o['id']: o['size'] for o in request['objects']}
    with pytest.raises(FileExistsError):
        sv.main(['build', '--data', str(main), '--out', str(out)])


@pytest.mark.parametrize('args, env, ok', [
    (['dev', 'joint', '1', '--limit', '16'], {}, True),
    (['dev', 'placement', '4'], {'SLURM_ARRAY_TASK_ID': '3'}, True),
    (['dev', 'placement', '4'], {'SLURM_ARRAY_TASK_ID': '4'}, False),
    (['dev', 'joint', '4'], {}, False),  # Several shards need an array task id.
    (['test', 'joint', '1'], {}, False),  # Test needs a later amendment.
    (['dev', 'both', '1'], {}, False),
    (['dev', 'joint', '1', '--limit'], {}, False)])
def test_sbatch_argument_checks(tmp_path, args, env, ok):
    run = subprocess.run(['bash', str(SCRIPTS / 'isambard_sample_vllm.sbatch'), str(CODE), *args],
                         env={**T.stub_sbatch(tmp_path), **env}, capture_output=True, text=True)
    # Valid arguments pass every check and stop at the missing Slurm allocation; invalid ones exit 2 before it.
    assert (run.returncode, 'Slurm allocation required' in run.stderr) == ((1, True) if ok else (2, False))


def submit_script(tmp_path, references=('sft/infer-joint', 'rpo-dpo1/infer-placement')):
    """submit_sample_vllm.sh with PROJECT_ROOT under tmp_path and the given select --reference dirs."""
    project = tmp_path / 'project'
    for ref in references:
        (project / 'runs/unknownsize-20261013/dev-eval-010' / ref).mkdir(parents=True)
        (project / 'runs/unknownsize-20261013/dev-eval-010' / ref / 'MANIFEST.json').write_text('{}')
    script = tmp_path / 'submit_sample_vllm.sh'
    original = (SCRIPTS / 'submit_sample_vllm.sh').read_text()
    line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    assert original.count(line) == 1
    script.write_text(original.replace(line, f'PROJECT_ROOT={project}\n'))
    return script


def test_submit_chain_with_stub_sbatch(tmp_path):
    for name in ('isambard_sample_vllm.sbatch', 'submit_sample_vllm.sh'):
        subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    sbatch = (SCRIPTS / 'isambard_sample_vllm.sbatch').read_text()
    assert 'VLLM_USE_FLASHINFER_SAMPLER=0' in sbatch and '.cache-jobs/s2/$ARM' in sbatch
    assert sbatch.index("import vllm, shapely, fastfill.comparison.sample_vllm") < sbatch.index('nvidia-smi -L')
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    (tmp_path / 'squeue').write_text('#!/bin/sh\ntest -z "$SQUEUE_FAIL" || exit 1\n'
                                     'test "$*" = "-h -u me -n unknownsize-s2-sample,unknownsize-s2-cpu -o %i" || exit 3\n'
                                     'printf "%s" "$SQUEUE_OUT"\n')
    (tmp_path / 'squeue').chmod(0o755)
    script = submit_script(tmp_path)
    run = subprocess.run(['bash', str(script), str(CODE), '3'], env=env, capture_output=True, text=True, check=True)
    log = [x.replace(f'{SCRIPTS}/', '').replace(f' {CODE} ', ' CODE ') for x in (tmp_path / 'log').read_text().splitlines()]
    kill = '--parsable --kill-on-invalid-dep=yes --dependency=afterok:'
    assert log[0].startswith('--parsable --hold --job-name=unknownsize-s2-cpu ') and 'sample_vllm build ' in log[0]
    assert log[1] == f'{kill}101 --time=02:00:00 isambard_sample_vllm.sbatch CODE dev joint 1 --limit 16'
    assert log[2] == f'{kill}102 --array=0-2 isambard_sample_vllm.sbatch CODE dev joint 3'
    assert log[3].startswith(f'{kill}103 --job-name=unknownsize-s2-cpu ') and 'sample_vllm finalize ' in log[3]
    assert log[4].startswith(f'{kill}104 --job-name=unknownsize-s2-cpu --account=brics.u6uv') and '--cpus-per-task=32' in log[4]
    assert 'best_of_n select ' in log[4] and '--arm joint ' in log[4] and '--workers 32' in log[4]
    assert '/dev-eval-010/sft/infer-joint' in log[4] and '/s2-011/joint-sft/samples ' in log[4]
    assert log[5] == f'{kill}101 --time=02:00:00 isambard_sample_vllm.sbatch CODE dev placement 1 --limit 16'
    assert '/dev-eval-010/rpo-dpo1/infer-placement' in log[8] and len(log) == 9
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101\n'
    assert run.stdout.splitlines()[-1] == 'released 101'
    failed = tmp_path / 'failed'
    failed.mkdir()
    for name in ('squeue',):
        (failed / name).write_bytes((tmp_path / name).read_bytes())
        (failed / name).chmod(0o755)
    bad = subprocess.run(['bash', str(script), str(CODE)],
                         env={**T.stub_sbatch(failed, fail_on=3), 'USER': 'me'}, capture_output=True, text=True)
    assert bad.returncode == 1 and "sbatch failed for 'joint array'" in bad.stderr and 'scancel 101 102' in bad.stderr
    assert not (failed / 'log.scontrol').exists()
    queued = subprocess.run(['bash', str(script), str(CODE)],
                            env={**env, 'SQUEUE_OUT': '7400001\n'}, capture_output=True, text=True)
    assert queued.returncode == 1 and 'scancel 7400001' in queued.stderr
    unreferenced = tmp_path / 'unreferenced'
    unreferenced.mkdir()
    missing = subprocess.run(['bash', str(submit_script(unreferenced, ('sft/infer-joint',))), str(CODE)],
                             env=env, capture_output=True, text=True)  # The select job would fail only at its end.
    assert missing.returncode == 1 and 'rpo-dpo1/infer-placement' in missing.stderr
    assert len((tmp_path / 'log').read_text().splitlines()) == 9  # Nothing new was submitted.
