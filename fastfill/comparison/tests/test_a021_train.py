"""Amendment 021 (a021_train) on CPU: the P-only pool on a tiny real main-data build (rebuild, byte equality, checker,
leak and count refusals), the pinned 27B receipt, the policy identity through dpo_stages / sample_vllm (8B unchanged),
smoke stop rules and N (a crash is no verdict), the vLLM load gate, train / probe with fakes, and the sbatch / submit scripts with stub Slurm commands."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fastfill.comparison import a021_train as a
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import main_data as m
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import sample_vllm as sv
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests import test_main_data as M
from fastfill.comparison.tests import test_sample_vllm as SV
from fastfill.v1_1.tests.test_train_dpo_a021 import Tok

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
GIB = 2 ** 30
GB = 10 ** 9


def collide(data):
    """A train room whose GT collides (J fails) but whose P teacher repairs it (P qualifies): a P-only room."""
    room = M.room('P0')
    room['objects'].append(dict(id='P0:second', category='chair', size=[1, 1, 1], pos=[5.2, 5, 0], yaw=0, parent=None,
                                anchor='floor'))
    return data + [room]


def receipt(path):
    """base_receipt stand-in: 27B for paths naming 27, else the generic 8B (no model_type)."""
    return {'path': str(path), 'files': {'config.json': hashlib.sha256(str(path).encode()).hexdigest()},
            'repository': 'R', 'revision': 'v', **({'model_type': a.MODEL_TYPE} if '27' in Path(path).name else {})}


@pytest.fixture
def world(tmp_path, monkeypatch):
    corpus = M.fixture(tmp_path / 'corpus', collide)
    main = tmp_path / 'main'
    manifest = m.build(corpus, main, max_candidates=8, max_passes=1)
    lineage = tmp_path / 'lineage.json'
    lineage.write_text(json.dumps({'status': 'ERROR', 'summary': {'admission_counts': manifest['dataset_counts']}}))
    monkeypatch.setattr(a, 'LINEAGE_SHA256', h.sha256(lineage))
    monkeypatch.setattr(a, 'P_ONLY', 1)
    monkeypatch.setattr(a, 'base_receipt', receipt)
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: Tok())
    monkeypatch.setattr(d, '_tokenizer', lambda path: Tok())
    held = []
    monkeypatch.setattr(op, '_held_out', lambda roots: (set(held), set(), [{'path': str(r)} for r in roots]))
    args = SimpleNamespace(data=main, corpus=corpus, lineage=lineage, base27=tmp_path / 'qwen27', base8=tmp_path / 'qwen8',
                           dev=[tmp_path / 'dev'], test=[tmp_path / 'test'], out=tmp_path / 'a021/pool', workers=1, sample=3)
    return SimpleNamespace(root=tmp_path, main=main, manifest=manifest, args=args, held=held)


def test_pool_admits_accepted_and_p_only_rooms_in_hash_order_with_p_sft_rows(world):
    assert a.pool(world.args) == {'status': a.POOL_STATUS, 'pool': 7, 'p_only': 1, 'fits': {'27b': 7, '8b': 7}}
    out = world.args.out
    rows = [json.loads(x) for x in (out / 'sft-rows.jsonl').read_text().splitlines()]
    ledger = [json.loads(x) for x in (out / 'ledger.jsonl').read_text().splitlines()]
    assert [r['uid'] for r in rows] == sorted((r['uid'] for r in rows), key=lambda u: hashlib.sha256(f'a021-27b:{u}'.encode()).hexdigest())
    assert {e['uid']: e['origin'] for e in ledger}['P0'] == 'p_only' and len(ledger) == 7
    trained = {r['uid']: r for r in map(json.loads, (world.main / 'placement/train.jsonl').read_text().splitlines())}
    for row in rows:  # P's SFT format: its system/user turns and its target (chosen = rejected).
        assert set(row) == {'uid', 'source', 'group', 'prompt', 'chosen', 'rejected'} and row['chosen'] == row['rejected']
        if row['uid'] in trained:
            assert [row['prompt'], row['chosen'], row['group']] == [
                trained[row['uid']]['messages'][:2], trained[row['uid']]['messages'][2]['content'], trained[row['uid']]['group']]
    p0 = next(r for r in rows if r['uid'] == 'P0')
    assert 'P0' not in trained and json.loads(p0['prompt'][1]['content'])['objects'][0]['size']  # P input: sizes given.
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['admission_counts']['placement_qualified'] - summary['admission_counts']['common_decoder_train'] == 1
    assert summary['rebuild'] == {**summary['rebuild'], 'p_only_ledger_equal': 1, 'accepted_sample_byte_equal': 3}
    assert summary['p_only']['by_source'] == {'Synthetic': 1} and summary['pool']['by_origin'] == {'common': 6, 'p_only': 1}
    assert all(e['tokens']['27b']['fits'] and e['checker']['geometry_success'] for e in ledger)
    assert a.take(a.load_pool(out), '27b', 3) == [r['uid'] for r in rows[:3]]
    with pytest.raises(FileExistsError):
        a.pool(world.args)


def test_rebuild_reproduces_p_sft_rows_byte_for_byte_and_refuses_drift(world, monkeypatch):
    accepted = {r['uid'] for r in map(json.loads, (world.main / 'common-admission.jsonl').read_text().splitlines())
                if r['accepted']}
    rebuilt = a.regenerate(world.args.corpus, world.main, world.manifest, accepted, 1)
    lines = {json.loads(x)['uid']: x for x in (world.main / 'placement/train.jsonl').read_text().splitlines()}
    assert {u: r['line'] for u, r in rebuilt.items()} == lines  # Every accepted room, not only a sample.
    real = a._regen

    def drift(task):
        value = real(task)
        return {**value, 'line': value['line'] + ' '} if value['uid'] == 'T0' else value  # One byte.
    monkeypatch.setattr(a, '_regen', drift)
    monkeypatch.setattr(a, 'rank', lambda uid, salt=a.SALT: '0' if uid == 'T0' else uid)  # T0 is in the sample.
    with pytest.raises(ValueError, match='rebuild differs from main-data-v1 for an accepted room: T0'):
        a.pool(world.args)


def test_pool_refuses_count_lineage_and_leak_mismatches(world, monkeypatch):
    monkeypatch.setattr(a, 'P_ONLY', 2)
    with pytest.raises(ValueError, match='not the audited 2'):
        a.pool(world.args)
    monkeypatch.setattr(a, 'P_ONLY', 1)
    counts = {**world.manifest['dataset_counts'], 'joint_qualified': 5}
    world.args.lineage.write_text(json.dumps({'summary': {'admission_counts': counts}}))
    with pytest.raises(ValueError, match='lineage audit result'):
        a.pool(world.args)
    monkeypatch.setattr(a, 'LINEAGE_SHA256', h.sha256(world.args.lineage))
    with pytest.raises(ValueError, match='admission count joint_qualified: ledger 6, lineage 5'):
        a.pool(world.args)
    world.args.lineage.write_text(json.dumps({'summary': {'admission_counts': world.manifest['dataset_counts']}}))
    monkeypatch.setattr(a, 'LINEAGE_SHA256', h.sha256(world.args.lineage))
    world.held.append('P0')
    with pytest.raises(ValueError, match='share a UID or house group with dev/test'):
        a.pool(world.args)
    assert not world.args.out.exists()


def write_files(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (root / name).write_bytes(payload if isinstance(payload, bytes) else payload.encode())


def test_27b_base_receipt_pins_every_non_weight_file_and_rehashes_shards(tmp_path, monkeypatch):
    base = tmp_path / 'Qwen3.8-27B'
    shards = {'model-1.safetensors': b'w1', 'model-2.safetensors': b'w2'}
    files = {'config.json': json.dumps({'model_type': 'qwen3_5', 'text_config': {'max_position_embeddings': 262144}}),
             'model.safetensors.index.json': json.dumps({'weight_map': {'a': 'model-1.safetensors', 'b': 'model-2.safetensors'}}),
             'tokenizer.json': '{}'}
    files['SHA256SUMS'] = ''.join(f'{hashlib.sha256(v).hexdigest()}  {k}\n' for k, v in shards.items())
    write_files(base, {**files, **shards})
    monkeypatch.setattr(a, 'BASE_27B', {n: h.sha256(base / n) for n in files})
    value = a.base_receipt(base)
    assert value['model_type'] == 'qwen3_5' and value['max_position_embeddings'] == 262144
    assert value['files'] == {**a.BASE_27B, **{k: hashlib.sha256(v).hexdigest() for k, v in shards.items()}}
    (base / 'model-2.safetensors').write_bytes(b'w3')
    with pytest.raises(ValueError, match='artifact hash mismatch: model-2'):
        a.base_receipt(base)
    (base / 'model-2.safetensors').write_bytes(b'w2')
    (base / 'notes.txt').write_text('x')
    with pytest.raises(ValueError, match='non-weight files differ'):
        a.base_receipt(base)
    (base / 'notes.txt').unlink()
    (base / 'tokenizer.json').write_text('{"x": 1}')
    with pytest.raises(ValueError, match='artifact hash mismatch: tokenizer.json'):
        a.base_receipt(base)
    eight = tmp_path / 'Qwen3-8B'
    write_files(eight, {'config.json': json.dumps({'model_type': 'qwen3'})})
    monkeypatch.setattr(h, 'base_receipt', lambda path: {'generic': str(path)})
    assert a.base_receipt(eight) == {'generic': str(eight)}  # The 8B: train_pair_helpers' pin, unchanged.


def test_real_27b_pins_are_the_o2_run_records():
    """The pins are llm-eval-O2's model_files_sha256; the three files pulled to the Mac match them (2026-10-14)."""
    assert a.BASE_27B['config.json'] == '191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab'
    assert a.BASE_27B['chat_template.jinja'] == 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'
    assert a.BASE_27B['generation_config.json'] == 'e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e'
    assert len(a.BASE_27B) == 15 and not any(n.endswith('.safetensors') for n in a.BASE_27B)
    pulls = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013/server-pulls-021')
    for name in ('config.json', 'chat_template.jinja', 'generation_config.json'):
        if (pulls / name).is_file():
            assert h.sha256(pulls / name) == a.BASE_27B[name]


def policy(tmp_path, arm='A'):
    spec = a.ARMS[arm]
    root = tmp_path / 'a021' / spec['dir']
    final = root / 'train/final'
    write_files(final, {'adapter_config.json': '{}', 'adapter_model.safetensors': b'lora', 'tokenizer.json': '{}'})
    base = receipt(tmp_path / ('qwen27' if arm == 'A' else 'qwen8'))
    value = {'arm': arm, 'stage': f'a021-{spec["dir"]}-sft', 'model_class': spec['model_class'],
             'base': {k: base[k] for k in ('path', 'files', 'repository', 'revision')},
             'adapter': {'path': str(final.resolve()), 'files': h.artifact_manifest(final)},
             'training_data': {'manifest_sha256': 'a' * 64}}
    (root / a.POLICY).write_text(json.dumps(value))
    return root, Path(base['path'])


def test_policy_identity_through_dpo_stages_binds_base_and_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'base_receipt', receipt)
    root, base = policy(tmp_path)
    run = SimpleNamespace(base=base, model=root, arm='placement', exact_policy=True)
    identity = d.inference_identity(run)
    assert identity['policy'] == {'mode': 'unmerged_fp32_adapter_on_bf16_training_base', 'training_base': str(base),
                                  'training_base_sha256': identity['base_sha256'],
                                  'adapter': str((root / 'train/final').resolve()),
                                  'adapter_sha256': identity['policy']['adapter_sha256'], 'model_class': a.IMAGE_TEXT}
    assert identity['base']['model_type'] == 'qwen3_5' and identity['stage'] == 'a021-27b-sft'
    assert identity['lineage_sha256'] == h.sha256(root / a.POLICY)
    with pytest.raises(ValueError, match='base differs'):
        d.inference_identity(SimpleNamespace(**{**vars(run), 'base': tmp_path / 'qwen27b'}))
    with pytest.raises(ValueError, match='a placement policy, not --arm joint'):
        d.inference_identity(SimpleNamespace(**{**vars(run), 'arm': 'joint'}))
    (root / 'train/final/adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='adapter bytes differ'):
        d.inference_identity(run)
    assert not a.is_policy(tmp_path / 'dpo-003/placement/dpo1-merged')  # 8B stage models keep the lineage path.


def test_sample_vllm_runs_the_27b_policy_text_only_and_8b_engine_is_unchanged(tmp_path, monkeypatch):
    assert sv.ENGINE == {'enable_lora': True, 'max_lora_rank': 32, 'max_loras': 1, 'max_model_len': 40960,
                         'dtype': 'bfloat16', 'seed': 0, 'tensor_parallel_size': 1}
    world = SV.patched(tmp_path, monkeypatch, SV.make_data(tmp_path))
    base = {'model_type': 'qwen3_5'}
    monkeypatch.setattr(sv.d, 'inference_identity', lambda run: {
        'base': base, 'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': 'a' * 64}, 'stage': 'a021-27b-sft',
        'lineage_sha256': '0' * 64, 'policy': {'training_base': str(tmp_path / 'qwen27'), 'adapter': str(tmp_path / 'a021/27b/train/final')}})
    engines = []
    sv.main(SV.argv(world.root, world.data, 'placement', 'samples-27b'), SV.loader(SV.FakeLLM(world.data), engines))
    assert engines == [{**sv.ENGINE, 'limit_mm_per_prompt': {'image': 0, 'video': 0}, 'model': str(tmp_path / 'qwen27')}]
    base.clear()  # An 8B identity (no model_type): the amendment-011 engine, bit for bit.
    engines.clear()
    sv.main(SV.argv(world.root, world.data, 'placement', 'samples-8b'), SV.loader(SV.FakeLLM(world.data), engines))
    assert engines == [{**sv.ENGINE, 'model': str(tmp_path / 'qwen27')}]


def perf(seconds=2.0, tokens=1000, peak=50 * GIB, steps=30, ranks=range(4)):
    return [{'rank': r, 'steps': [{'step': s + 1, 'seconds': seconds, 'input_tokens': tokens} for s in range(steps)],
             'max_memory_reserved': peak, 'max_memory_allocated': peak - 1, 'device_total_memory': 95 * GIB,
             'device_name': 'GH200'} for r in ranks]


def test_smoke_stop_rules_and_n_are_fixed_in_code():
    assert (a.N_MAX, a.FIT_HOURS, a.STOP_HOURS, a.STOP_MEMORY, a.SMOKE_STEPS, a.WORLD) == (40_000, 10, 12, 90 * GIB, 200, 4)
    assert a.RECIPE['warmup_ratio'] == 0.03 and '--warmup_ratio' in d._flags(a.recipe('A', 1))  # P's 3% warmup.
    assert a.step_tokens([5] * 10) == [5, 5, 5]  # 10 rooms on 4 ranks: 3 steps, the sampler pads the last.
    tokens = list(range(1, 11))
    steps = a.step_tokens(tokens)
    assert len(steps) == 3 and max(steps) == 10 and all(s in tokens for s in steps)
    # 1,000 tokens in 2 s per rank: 500 tokens/s. 2,000-token rooms: 4 s per step of 4 rooms.
    verdict = a.smoke_verdict(perf(), [2000] * 50_000)
    assert verdict['status'] == a.SMOKE_PASS and verdict['reasons'] == []
    assert verdict['n'] == 36_000 and verdict['projected_hours_n'] == 10.0  # 9,000 steps x 4 s.
    assert round(verdict['projected_hours_n_max'], 3) == round(40_000 / 4 * 4 / 3600, 3) <= 12
    assert verdict['save_steps'] == 900 and verdict['steady_steps'] == 20 and verdict['tokens_per_second_per_rank'] == 500
    small = a.smoke_verdict(perf(), [2000] * 1000)
    assert small['status'] == a.SMOKE_PASS and small['n'] == 1000  # The whole pool fits.
    slow = a.smoke_verdict(perf(), [3000] * 50_000)  # 6 s per step: 40,000 rooms = 16.7 h > 12 h.
    assert slow['status'] == a.SMOKE_STOP and slow['reasons'] == ['projected_hours']
    big = a.smoke_verdict(perf(peak=90 * GIB + 1), [2000] * 50_000)  # 90 GiB (amendment 021 addendum A).
    assert big['status'] == a.SMOKE_STOP and big['reasons'] == ['peak_memory']
    assert a.smoke_verdict(perf(peak=90 * GIB), [2000] * 50_000)['status'] == a.SMOKE_PASS
    for broken in (perf(ranks=range(3)), perf(steps=a.WARMUP), []):  # Incomplete records: no verdict at all.
        with pytest.raises(RuntimeError, match='incomplete smoke perf records'):
            a.smoke_verdict(broken, [2000] * 10)


def fake_base(path):
    from safetensors.torch import save_file
    import torch
    write_files(path, {'config.json': '{}', 'tokenizer_config.json': '{}', 'tokenizer.json': '{}'})
    save_file({'w': torch.zeros(1)}, str(path / 'model.safetensors'))
    return path


def fake_torchrun(calls, ranks=4, fail=False, peak=50 * GIB):
    """torchrun train_dpo stand-in: perf records (--perf_out) and a completed final adapter."""
    from fastfill.v1_1 import train_dpo

    def run(command, check=False):
        calls.append(command)
        flags = dict(zip(command[::1], command[1:]))
        if fail:
            return SimpleNamespace(returncode=1)
        if '--perf_out' in command:
            out = Path(flags['--perf_out'])
            out.mkdir(parents=True)
            for record in perf(steps=30, ranks=range(ranks), peak=peak):
                (out / f'perf-rank{record["rank"]}.json').write_text(json.dumps(record))
        count = len(Path(flags['--pairs']).read_text().splitlines())
        final = Path(flags['--out']) / 'final'
        write_files(final, {'adapter_config.json': '{}', 'adapter_model.safetensors': b'lora', 'tokenizer_config.json': '{}',
                            'tokenizer.json': '{}'})
        batches = -(-count // 4)
        train_dpo.write_dpo_completion(final, flags['--out'], count, batches, 4, 1, batches, batches)
        return SimpleNamespace(returncode=0)
    return run


def vllm_load(world, monkeypatch, smoke_dir, b27, script=None):
    """smoke-vllm with sample_vllm's fakes (their tokenizer and fake engine); returns (result, engines, fake)."""
    data = world.root / 'dev-select/data'
    if not data.exists():
        assert SV.make_data(world.root / 'dev-select') == data
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: SV.Tok())
    fake, engines = SV.FakeLLM(data, **({'script': script} if script else {})), []
    try:
        return a.smoke_vllm(SimpleNamespace(data=data, smoke=smoke_dir, base=b27), SV.loader(fake, engines)), engines, fake
    finally:
        monkeypatch.setattr(d, '_model_tokenizer', lambda path: Tok())


def test_smoke_then_train_arm_a_writes_n_before_training_and_a_policy(world, monkeypatch):
    a.pool(world.args)
    monkeypatch.setattr(a, 'SMOKE_STEPS', 1)
    b27, b8 = fake_base(world.root / 'qwen27'), fake_base(world.root / 'qwen8')
    smoke = SimpleNamespace(pool=world.args.out, base=b27, out=world.root / 'a021/smoke')
    calls = []
    verdict = a.smoke(smoke, fake_torchrun(calls))
    assert verdict['status'] == a.SMOKE_PASS and verdict['n'] == 7 and verdict['rows']['rooms'] == 4
    command = calls[0]
    assert command[:8] == [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', '-m',
                           'fastfill.v1_1.train_dpo', '--model']
    for flag in ('--sft_only', '--sft_token_mean', '--answer_logits', '--check_render', '--base_receipt', '--image_text_model'):
        assert flag in command
    assert command[command.index('--lora_regex') + 1] == a.LORA_27B and command[command.index('--save_steps') + 1] == '0'
    assert [command[command.index(f) + 1] for f in ('--lr', '--lora_r', '--lora_alpha', '--epochs', '--bs', '--grad_accum')] == [
        '5e-05', '32', '64', '1', '1', '1']
    smoke_rows = [json.loads(x)['uid'] for x in (smoke.out / 'rows/placement.jsonl').read_text().splitlines()]
    ledger = {e['uid']: e for e in a.load_pool(world.args.out)['ledger']}
    assert [ledger[u]['tokens']['27b']['total'] for u in smoke_rows] == sorted(
        (e['tokens']['27b']['total'] for e in ledger.values()), reverse=True)[:4]  # The longest admitted sequences.
    assert a.smoke(smoke, fake_torchrun(calls, fail=True)) == verdict and len(calls) == 1  # Write-once verdict.
    args = SimpleNamespace(arm='A', pool=world.args.out, base=b27, run_root=world.root / 'a021', smoke=smoke.out)
    with pytest.raises(ValueError, match='vLLM has not served this smoke adapter'):  # Before any training.
        a.train(args, fake_torchrun(calls))
    loaded, engines, fake = vllm_load(world, monkeypatch, smoke.out, b27)
    assert loaded == {'status': 'OK', 'rooms': a.VLLM_SMOKE_ROOMS, 'stopped_on_eos': a.VLLM_SMOKE_ROOMS}
    assert engines == [{**sv.ENGINE, 'limit_mm_per_prompt': {'image': 0, 'video': 0}, 'model': str(b27)}]
    assert fake.calls[0]['lora'] == ('a021-smoke', 1, str(smoke.out / 'train/final'))
    assert all(p['temperature'] == 0.0 for p in fake.calls[0]['params'])
    load = json.loads((smoke.out / a.VLLM_LOAD).read_text())
    assert load['verdict_sha256'] == h.sha256(smoke.out / 'verdict.json') and load['adapter']['files'] == h.artifact_manifest(
        smoke.out / 'train/final')
    with pytest.raises(FileExistsError):
        vllm_load(world, monkeypatch, smoke.out, b27)
    result = a.train(args, fake_torchrun(calls))
    assert result['status'] == 'A021_POLICY_COMPLETE' and result['n'] == 7
    root = world.root / 'a021/27b'
    run_receipt = json.loads((root / 'run_receipt.json').read_text())
    assert run_receipt['n'] == 7 and run_receipt['smoke_verdict_sha256'] == h.sha256(smoke.out / 'verdict.json')
    assert run_receipt['smoke_vllm_load_sha256'] == h.sha256(smoke.out / a.VLLM_LOAD)
    assert calls[-1][calls[-1].index('--warmup_ratio') + 1] == '0.03'  # P's warmup, not train_dpo's default 5%.
    assert run_receipt['recipe']['save_steps'] == verdict['save_steps'] and run_receipt['model_class'] == a.IMAGE_TEXT
    trained = [json.loads(x)['uid'] for x in (root / 'rows/placement.jsonl').read_text().splitlines()]
    assert trained == a.take(a.load_pool(world.args.out), '27b', 7)  # Hash order.
    identity = d.inference_identity(SimpleNamespace(base=b27, model=root, arm='placement', exact_policy=True))
    assert identity['training_data']['count'] == 7 and identity['training_data']['manifest_sha256'] == h.sha256(
        world.main / 'MANIFEST.json')
    assert a.train(args, fake_torchrun(calls))['status'] == 'A021_POLICY_COMPLETE' and len(calls) == 2  # Done: no rerun.
    with pytest.raises(ValueError, match='only the fallback'):
        a.train(SimpleNamespace(arm='B', pool=world.args.out, base=b8, run_root=world.root / 'a021', smoke=None),
                fake_torchrun(calls))


def test_a_crashed_smoke_is_no_verdict_and_reruns_only_the_two_rules_stop(world, monkeypatch):
    a.pool(world.args)
    monkeypatch.setattr(a, 'SMOKE_STEPS', 1)
    b27 = fake_base(world.root / 'qwen27')
    smoke = SimpleNamespace(pool=world.args.out, base=b27, out=world.root / 'a021/smoke')
    failed = world.root / f'a021/smoke.failed-{os.getpid()}'
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    for k, crash in enumerate((fake_torchrun([], fail=True), fake_torchrun([], ranks=3))):  # Exit 1; a rank's perf missing.
        with pytest.raises(RuntimeError, match='no verdict, attempt kept as'):
            a.smoke(smoke, crash)
        assert not smoke.out.exists() and (failed / 'rows/placement.jsonl').is_file()
        failed.rename(world.root / f'a021/inspected-{k}')
    real = a.load_pool
    monkeypatch.setattr(a, 'load_pool', lambda root: {**(v := real(root)), 'summary': {**v['summary'], 'tokenizers': {
        '27b': {**v['summary']['tokenizers']['27b'], 'eos_token': '<|endoftext|>'}}}})
    with pytest.raises(ValueError, match='27b tokenizer EOS .* is not'):
        a.smoke(smoke, fake_torchrun([]))  # Refused before anything is written.
    assert not smoke.out.exists()
    monkeypatch.setattr(a, 'load_pool', real)
    stopped = a.smoke(smoke, fake_torchrun([], peak=90 * GIB + 1))  # A rerun measures; only the rules stop.
    assert stopped['status'] == a.SMOKE_STOP and stopped['reasons'] == ['peak_memory']
    with pytest.raises(ValueError, match='the smoke stopped Arm A'):
        vllm_load(world, monkeypatch, smoke.out, b27)
    assert not (smoke.out / a.VLLM_LOAD).exists()


def test_train_refuses_a_stopped_smoke_and_arm_b_takes_the_whole_pool(world, monkeypatch):
    a.pool(world.args)
    monkeypatch.setattr(a, 'SMOKE_STEPS', 1)
    b27, b8 = fake_base(world.root / 'qwen27'), fake_base(world.root / 'qwen8')
    smoke = SimpleNamespace(pool=world.args.out, base=b27, out=world.root / 'a021/smoke')
    assert a.smoke(smoke, fake_torchrun([], peak=90 * GIB + 1))['reasons'] == ['peak_memory']
    args = SimpleNamespace(arm='A', pool=world.args.out, base=b27, run_root=world.root / 'a021', smoke=smoke.out)
    with pytest.raises(ValueError, match='the smoke stopped Arm A'):
        a.train(args, fake_torchrun([]))
    with pytest.raises(ValueError, match='Arm A trains on qwen3_5'):
        a.train(SimpleNamespace(**{**vars(args), 'base': b8}), fake_torchrun([]))
    calls = []
    result = a.train(SimpleNamespace(arm='B', pool=world.args.out, base=b8, run_root=world.root / 'a021', smoke=None),
                     fake_torchrun(calls))
    assert result['n'] == 7 and '--image_text_model' not in calls[0]
    assert calls[0][calls[0].index('--lora_regex') + 1] == a.LORA_8B and calls[0][calls[0].index('--save_steps') + 1] == '1000'
    with pytest.raises(SystemExit):
        a.parse_args(['train', '--arm', 'A', '--pool', 'p', '--base', 'b', '--run-root', 'r'])  # Arm A needs --smoke.


def test_probe_hf_vllm_report_on_the_same_rooms(tmp_path, monkeypatch):
    data = SV.make_data(tmp_path)
    identity = {'policy': {'training_base': str(tmp_path / 'qwen27'), 'adapter': str(tmp_path / 'a021/27b/train/final'),
                           'model_class': a.IMAGE_TEXT}}
    monkeypatch.setattr(a.d, 'inference_identity', lambda run: identity)
    monkeypatch.setattr(a.d, '_model_tokenizer', lambda path: SV.Tok())
    args = SimpleNamespace(data=data, policy=tmp_path / 'a021/27b', base=tmp_path / 'qwen27', out=tmp_path / 'probe')

    generated = []

    def runtime(policy, fail_after=None):
        assert policy == identity['policy']

        def generate(ids, cap):
            if fail_after is not None and len(generated) == fail_after:
                raise RuntimeError('time limit')
            generated.append(ids)
            text, _ = SV.spaced(json.loads(''.join(map(chr, ids))), None, 0)
            return ids + [ord(c) for c in text] + [T.EOS]
        return SV.Tok(), T.EOS, generate
    with pytest.raises(RuntimeError, match='time limit'):  # A job killed after 3 rooms keeps them.
        a.probe_hf(args, lambda policy: runtime(policy, fail_after=3))
    assert len((args.out / 'hf.partial.jsonl').read_text().splitlines()) == 3 and not (args.out / 'hf.jsonl').exists()
    assert a.probe_hf(args, runtime) == {'status': 'PROBE_HF_DONE', 'rooms': 7}
    assert len(generated) == 7 and not (args.out / 'hf.partial.jsonl').exists()  # The rerun decoded only the other 4.
    engines = []
    shifted = SV.FakeLLM(data, script=lambda request, uid, k: SV.spaced(request, uid, 3 if uid == 'r0' else k))
    assert a.probe_vllm(args, SV.loader(shifted, engines)) == {'status': 'PROBE_VLLM_DONE', 'rooms': 7}
    assert engines == [{**sv.ENGINE, 'limit_mm_per_prompt': {'image': 0, 'video': 0}, 'model': str(tmp_path / 'qwen27')}]
    assert all(p['temperature'] == 0.0 and p['stop_token_ids'] == [T.EOS] for p in shifted.calls[0]['params'])
    assert shifted.calls[0]['lora'] == ('a021', 1, identity['policy']['adapter'])
    report = a.probe_report(args)
    assert report['rooms'] == 7 and report['identical_outputs'] == {'count': 6, 'share': 6 / 7}
    assert report['rates']['hf']['complete'] == report['rates']['vllm']['complete'] == 1.0
    saved = json.loads((args.out / 'report.json').read_text())
    assert saved['checker_success_vllm_minus_hf']['estimate'] == pytest.approx(
        saved['rates']['vllm']['geometry_success'] - saved['rates']['hf']['geometry_success'])
    assert '保真探针' in (args.out / 'report.md').read_text() and (args.out / 'MANIFEST.json').exists()
    for step in (a.probe_hf, a.probe_report):
        with pytest.raises(FileExistsError):
            step(args) if step is a.probe_report else step(args, runtime)


# ---- sbatch and submit ---------------------------------------------------------------------------------------------

COLUMNS = 'JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES'


def stubs(directory):
    (directory / 'squeue').write_text('#!/bin/sh\ncase "$*" in\n'
                                      '  "-h -u me -n unknownsize-021-train -o %i") printf "%s" "$SQUEUE_OUT" ;;\n'
                                      '  "-h -u me -o %i|%j") printf "%s" "$SQUEUE_ALL" ;;\n  *) exit 3 ;;\nesac\n')
    (directory / 'sacct').write_text('#!/bin/sh\ncase "$*" in\n'
                                     f'  "-n -X -P -u me -S 2026-10-14 -o {COLUMNS}") printf "%s" "$SACCT_OUT" ;;\n'
                                     f'  "-n -X -P -j $SACCT_IDS -o {COLUMNS}") printf "%s" "$SACCT_J_OUT" ;;\n'
                                     '  *) exit 3 ;;\nesac\n')
    for name in ('squeue', 'sacct'):
        (directory / name).chmod(0o755)
    return {**T.stub_sbatch(directory), 'USER': 'me'}


def submit_script(tmp_path):
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    for path in ('main-data-v1', 'full-corpus-v2', 'dev-select-v1', 'dev-full-v1', 'eval-data-v1'):
        write_files(run / path, {'MANIFEST.json': '{}'})
    for model in ('Qwen3.8-27B', 'Qwen3-8B'):
        write_files(project / 'models' / model, {'config.json': '{}'})
    (project / 'env/bin').mkdir(parents=True)
    (project / 'env/bin/python').symlink_to(sys.executable)
    script = tmp_path / 'submit_021_train.sh'
    line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    original = (SCRIPTS / 'submit_021_train.sh').read_text()
    assert original.count(line) == 1
    script.write_text(original.replace(line, f'PROJECT_ROOT={project}\n'))
    code = tmp_path / 'main-abc'
    code.mkdir()
    (code / 'fastfill').symlink_to(CODE / 'fastfill')
    docs = code / 'docs/research/20261013-unknown-size-comparison'
    write_files(docs / 'evidence', {'full-audit-data-lineage-7368003-result.json': '{}'})
    write_files(docs, {'amendment-021-diagnostics-and-one-more-arm-20261014.md': '# 021\n'})
    return script, project, run / 'a021', code


def test_submit_021_modes_gpu_cap_held_roots_and_refusals(tmp_path):
    for name in ('isambard_021_train.sbatch', 'submit_021_train.sh'):
        subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    env = stubs(tmp_path)
    script, project, a21, code = submit_script(tmp_path)
    sent = []

    def submit(*argv, **extra):
        before = len(sent)
        done = subprocess.run(['bash', str(script), str(code), *argv], env={**env, **extra}, capture_output=True, text=True)
        log = tmp_path / 'log'
        sent[:] = log.read_text().splitlines() if log.exists() else []
        return done, [x.replace(f'{code}/fastfill/comparison/', '').replace(f' {code} ', ' CODE ') for x in sent[before:]]
    J, gpu4 = 'isambard_021_train.sbatch CODE', '--gpus-per-node=4 --cpus-per-task=72 --mem=0'
    gpu1 = '--gpus-per-node=1 --cpus-per-task=16 --mem=200G'
    done, log = submit('setup', SACCT_OUT='1|unknownsize-021-o2nt|COMPLETED|7200|60|gres/gpu=2|gres/gpu=2\n'
                                          '2|unknownsize-018|COMPLETED|360000|720|gres/gpu=4|gres/gpu=4\n')
    assert done.returncode == 0, done.stderr
    assert 'GPU-h: used 4.00 + ceiling 0 <= 200' in done.stdout and log == [f'--parsable --hold {J} setup']
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101\n'
    done, log = submit('smoke')
    assert log == [] and 'smoke needs RUN/a021/pool and the fla overlay' in done.stderr
    write_files(a21 / 'pool', {'MANIFEST.json': '{}'})
    write_files(project / 'pylib-021', {'.a021-fla-core-0.4.0': 'fla-core==0.4.0\n'})
    done, log = submit('setup')
    assert log == [] and 'nothing to do' in done.stderr
    done, log = submit('smoke')  # smoke-vllm only after a SMOKE_PASS (smoke exits 1 on SMOKE_STOP).
    assert done.returncode == 0 and 'ceiling 7 <=' in done.stdout
    assert log == [f'--parsable --hold {gpu4} --time=01:30:00 {J} smoke',
                   f'--parsable --kill-on-invalid-dep=yes --dependency=afterok:102 {gpu1} --time=01:00:00 {J} smoke-vllm']
    (a21 / 'smoke').mkdir()
    assert 'no verdict: an interrupted attempt; move it aside' in submit('smoke')[0].stderr
    done, log = submit('train')
    assert log == [] and 'no SMOKE_PASS verdict' in done.stderr
    assert 'no SMOKE_PASS verdict' in submit('smoke-vllm')[0].stderr
    (a21 / 'smoke/verdict.json').write_text(json.dumps({'status': a.SMOKE_PASS, 'projected_hours_n': 4.0}))
    assert 'the smoke has a verdict' in submit('smoke')[0].stderr
    assert 'no smoke adapter' in submit('smoke-vllm')[0].stderr
    write_files(a21 / 'smoke/train/final', {'adapter_model.safetensors': b'lora'})
    done, log = submit('smoke-vllm')  # A rerun of the vLLM load alone.
    assert done.returncode == 0 and log == [f'--parsable --hold {gpu1} --time=01:00:00 {J} smoke-vllm']
    done, log = submit('train')
    assert log == [] and 'vLLM has not loaded the smoke adapter' in done.stderr
    (a21 / 'smoke/vllm-load.json').write_text(json.dumps({'status': 'OK'}))
    assert 'vllm-load.json exists' in submit('smoke-vllm')[0].stderr
    done, log = submit('train', '2')  # H = ceil(1.15 x 4 + 1) = 6; 2 x 4 x 6 = 48 GPU-h.
    assert done.returncode == 0, done.stderr
    assert 'GPU-h: used 0.00 + ceiling 48 <= 200' in done.stdout
    first = len(sent) - 1
    assert log == [f'--parsable --hold {gpu4} --time=6:00:00 {J} train A 6',
                   f'--parsable --kill-on-invalid-dep=yes --dependency=afternotok:{100 + first} {gpu4} --time=6:00:00 {J} train A 6']
    done, log = submit('train', '2', SACCT_OUT='5|unknownsize-021-engine|COMPLETED|594000|1800|gres/gpu=1|gres/gpu=1\n')
    assert log == [] and 'GPU cap: used 165.00 + ceiling 48 > 200' in done.stderr  # 165 + 48 > 200.
    done, log = submit('train', '1', '08', SACCT_OUT='5|unknownsize-021-engine|COMPLETED|594000|1800|gres/gpu=1|gres/gpu=1\n')
    assert done.returncode == 0 and 'used 165.00 + ceiling 32 <= 200' in done.stdout  # An explicit HOURS sizes a resume.
    assert log == [f'--parsable --hold {gpu4} --time=8:00:00 {J} train A 8']
    # Pending and running jobs count with their time limit; squeue's ids catch a dependency-pending job sacct -S misses.
    budget = dict(SACCT_OUT='3|unknownsize-021-eval|RUNNING|3600|240|gres/gpu=1|gres/gpu=1\n'
                            '4|unknownsize-021-diag|CANCELLED by 1|1800|600|gres/gpu=2|gres/gpu=2\n',
                  SQUEUE_ALL='3|unknownsize-021-eval\n9|unknownsize-021-eval\n8|unknownsize-018-x\n', SACCT_IDS='3,9',
                  SACCT_J_OUT='3|unknownsize-021-eval|RUNNING|3600|240|gres/gpu=1|gres/gpu=1\n'
                              '9|unknownsize-021-eval|PENDING|0|2475||gres/gpu=4\n')
    done, log = submit('train', '1', '7', **budget)  # 4 (limit, not elapsed) + 1 + 165 + 28 = 198.
    assert done.returncode == 0 and 'GPU-h: used 170.00 + ceiling 28 <= 200' in done.stdout, done.stderr
    done, log = submit('train', '1', '8', **budget)
    assert log == [] and 'GPU cap: used 170.00 + ceiling 32 > 200' in done.stderr
    (a21 / 'smoke/verdict.json').write_text(json.dumps({'status': a.SMOKE_PASS, 'projected_hours_n': 20.0}))
    done, log = submit('train')
    assert log[0] == f'--parsable --hold {gpu4} --time=12:00:00 {J} train A 12'  # Never above 12 h.
    (a21 / 'smoke/verdict.json').write_text(json.dumps({'status': a.SMOKE_STOP, 'reasons': ['peak_memory']}))
    assert 'no SMOKE_PASS verdict' in submit('train')[0].stderr
    assert 'probe needs the Arm A policy' in submit('probe')[0].stderr
    done, log = submit('armB')
    assert done.returncode == 0 and log == [f'--parsable --hold {gpu4} --time=12:00:00 {J} train B 12']
    done, log = submit('armB', '1', '5')
    assert done.returncode == 0 and log == [f'--parsable --hold {gpu4} --time=5:00:00 {J} train B 5']
    write_files(a21 / '27b', {'a021-policy.json': '{}'})
    assert 'Arm B is the fallback only' in submit('armB')[0].stderr
    assert 'Arm A is trained' in submit('train')[0].stderr
    done, log = submit('probe')
    assert done.returncode == 0 and log == [f'--parsable --hold --gpus-per-node=1 --cpus-per-task=16 --mem=200G '
                                            f'--time=03:00:00 {J} probe']
    done, log = submit('probe', SQUEUE_OUT='7500001\n')
    assert log == [] and 'scancel 7500001' in done.stderr
    for bad in (('bogus',), ('train', '4'), ('probe', '1'), ('armB', 'x'), ('train', '1', '13'), ('train', '1', '0'),
                ('train', '1', 'x'), ('smoke-vllm', '1'), ('train', '1', '2', '3')):
        assert submit(*bad)[0].returncode == 2
    os.remove(a21 / '27b/a021-policy.json')
    (a21 / 'smoke/verdict.json').write_text(json.dumps({'status': a.SMOKE_PASS, 'projected_hours_n': 2.0}))
    (a21 / 'smoke/vllm-load.json').write_text(json.dumps({'status': 'OK'}))
    failing = {**env, **T.stub_sbatch(tmp_path, fail_on=len(sent) + 2)}
    done, log = submit('train', '2', **failing)
    assert done.returncode == 1 and "sbatch failed for 'train A segment 2'" in done.stderr
    assert f'scancel {101 + len(sent) - 2}' in done.stderr


@pytest.mark.parametrize('args, ok', [(('setup',), True), (('smoke',), True), (('smoke-vllm',), True),
                                      (('smoke-vllm', 'x'), False), (('train', 'A', '6'), True),
                                      (('train', 'B', '12'), True), (('probe',), True), (('train', 'C', '6'), False),
                                      (('train', 'A', '0'), False), (('train', 'A'), False), (('smoke', 'x'), False),
                                      (('pool',), False)])
def test_sbatch_argument_checks(tmp_path, args, ok):
    done = subprocess.run(['bash', str(SCRIPTS / 'isambard_021_train.sbatch'), str(CODE), *args],
                          env={'PATH': '/usr/bin:/bin'}, capture_output=True, text=True)
    # Valid: past the case checks, stopped at the missing Slurm allocation. Invalid: exit 2 first.
    assert (done.returncode, 'Slurm allocation required' in done.stderr) == ((1, True) if ok else (2, False))
    text = (SCRIPTS / 'isambard_021_train.sbatch').read_text()
    assert '#SBATCH --exclude=nid010216\n' in text and 'export PYTHONPATH="$CODE_ROOT:$OVERLAY"' in text
    assert 'pip install --no-deps --no-cache-dir --target "$STAGE"' in text and '"$PROJECT_ROOT/env"' not in text.replace(
        'PYTHON="$PROJECT_ROOT/env/bin/python"', '')
