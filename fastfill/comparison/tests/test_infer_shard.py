"""Inference v2 plan -> shards (interrupted + resumed) -> merge -> unchanged evaluate_pair; CPU only."""
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import Mock, patch

import pytest

from fastfill.comparison import evaluate_pair
from fastfill.comparison import infer_shard as s
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages

EOS = 999999
OBJECTS = [1, 3, 2, 1, 3, 2, 2]  # 7 rooms, batch 2 -> 4 plan batches; 2 shards -> batches {0,2} and {1,3}.
IDENTITY = {'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': 'a' * 64}}


class Tok:
    """Prompt = user JSON, one token per character, so the fake model can read its input."""
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return messages[1]['content']

    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, truncation=False)
        return {'input_ids': [ord(c) for c in text]}

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


def fake_generate(calls=None, fail_on=None):
    def generate(ids, cap):
        if calls is not None:
            calls.append(cap)
            if fail_on is not None and len(calls) == fail_on:
                raise KeyboardInterrupt()
        width = max(map(len, ids))
        out = []
        for x in ids:
            request = json.loads(''.join(map(chr, x)))
            answer = json.dumps({'placements': [
                {'id': o['id'], **({} if 'size' in o else {'size': [0.5, 0.5, 0.5]}), 'pos': [0.6 + k, 0.6, 0], 'yaw': 0}
                for k, o in enumerate(request['objects'])]})
            out.append([0] * (width - len(x)) + x + [ord(c) for c in answer] + [EOS])
        return out, width
    return generate


def make_data(root, sidecar=b'opaque\n'):
    data = root / 'data'
    (data / 'joint').mkdir(parents=True)
    (data / 'placement').mkdir()
    rooms = []
    for k, count in enumerate(OBJECTS):
        objects = [{'id': f'o{j}', 'category': 'chair', 'size': [1, 1, 1], 'pos': [1 + j, 1, 0], 'yaw': 0}
                   for j in range(count)]
        rooms.append(bind_room({'uid': f'r{k}', 'source': 'IL3D_3dfront', 'group': f'g{k}', 'room_type': 'bedroom',
                                'boundary_type': 'polygon', 'boundary': [[0, 0], [5, 0], [5, 5], [0, 5]],
                                'height': 3, 'objects': objects, 'fixed': []})[0])
    (data / 'rooms.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rooms))
    (data / 'size-estimates.jsonl').write_bytes(sidecar)
    for arm in h.ARMS:
        rows = []
        for room in rooms:
            sizes = {o['id']: o['size'] for o in room['objects']}
            rows.append({'uid': room['uid'], 'source': room['source'], 'group': room['group'],
                         'messages': messages(room, arm, sizes if arm == 'placement' else None)[:2]})
        (data / arm / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    h.write_json(data / 'MANIFEST.json', {'expected_uids': [r['uid'] for r in rooms],
                                          'training_manifest_sha256': 'a' * 64, 'files': h.artifact_manifest(data)})
    return data


def make_plan(root, data, name='plan', tokenizer=None):
    args = s.parse_args(['plan', '--data', str(data), '--base', str(root / 'base'),
                         '--out', str(root / 'v2' / name), '--batch-size', '2'])
    with patch.object(s, '_plan_tokenizer', return_value=(Tok(), tokenizer or {'source': 'test'}, 'b' * 64)):
        s.plan(args)
    return args.out


def shard_args(root, data, plan, arm, index, count=2, resume=False, out=None):
    argv = ['shard', '--data', str(data), '--base', str(root / 'base'), '--adapter', str(root / arm / 'adapter'),
            '--training-run', str(root / arm), '--arm', arm, '--plan', str(plan), '--shard-index', str(index),
            '--shard-count', str(count), '--out', str(out or root / 'v2' / arm / f'shard-{index:02d}')]
    return s.parse_args(argv + (['--resume'] if resume else []))


def run_shard(args, generate=None, tok=None):
    with patch.object(s.i, 'training_identity', return_value=IDENTITY), \
            patch.object(s.i, '_runtime', return_value=(tok or Tok(), generate or fake_generate(), EOS)):
        return s.shard(args)


def merge(root, data, plan, arm, shards, name=None):
    args = s.parse_args(['merge', '--data', str(data), '--plan', str(plan), '--arm', arm,
                         '--out', str(root / (name or f'infer-{arm}-002')), '--shards', *map(str, shards)])
    s.merge(args)
    return args.out


def complete_arm(root, data, plan, arm, count=2):
    shards = [shard_args(root, data, plan, arm, k, count) for k in range(count)]
    for args in shards:
        run_shard(args)
    return [a.out for a in shards]


@pytest.fixture
def setup(tmp_path):
    data = make_data(tmp_path)
    return tmp_path, data, make_plan(tmp_path, data)


def test_plan_shards_resume_merge_and_unchanged_evaluate_pair(setup):
    root, data, plan = setup
    value, plan_sha, _ = s.load_plan(plan)
    assert [b['uids'] for b in value['batches']] == [['r1', 'r4'], ['r2', 'r5'], ['r6', 'r0'], ['r3']]
    outs = {}
    for arm in h.ARMS:
        first = shard_args(root, data, plan, arm, 0)
        calls = []
        with pytest.raises(KeyboardInterrupt):  # Timeout stand-in: second batch of shard 0 dies.
            run_shard(first, fake_generate(calls, fail_on=2))
        status = h.strict_json((first.out / 'status.json').read_text())
        assert (status['status'], status['recorded_count'], status['pending_count']) == ('FAIL', 2, 2)
        assert not (first.out / 'result.json').exists()
        calls = []
        result = run_shard(shard_args(root, data, plan, arm, 0, resume=True), fake_generate(calls))
        assert result['status'] == 'SHARD_COMPLETE' and calls == [value['batches'][2]['max_new_tokens']]
        second = shard_args(root, data, plan, arm, 1)
        run_shard(second)
        outs[arm] = merge(root, data, plan, arm, [first.out, second.out])
        records, receipt = evaluate_pair._inference(outs[arm], data, value['data']['expected_uids'])
        assert [r['uid'] for r in records] == [f'r{k}' for k in range(7)]
        assert receipt['context_plan_sha256'] == plan_sha and receipt['config']['lora'] == 'merge_and_unload_bf16'
        caps = {u: b['max_new_tokens'] for b in value['batches'] for u in b['uids']}
        assert all(r['effective_max_new_tokens'] == caps[r['uid']] for r in records)
        assert all(r['schema']['valid'] and r['stopped_on_eos'] for r in records)
    summary = evaluate_pair.evaluate(data, outs, root / 'geometry-v2', repeats=20)
    assert summary['status'] == 'GEOMETRY_COMPLETE' and summary['expected_rooms_per_arm'] == 7
    assert summary['arms']['joint']['all']['complete_count'] == 7
    assert summary['arms']['placement']['all']['complete_count'] == 7


def test_cap_formula_and_input_only_order():
    decode = {'batch_size': 2, 'context_budget': 2000, 'max_input_tokens': 1999, 'max_new_tokens': 1999,
              'cap_base': s.CAP_BASE, 'cap_per_object': s.CAP_PER_OBJECT}
    items = [{'uid': 'c', 'objects': 2, 'input_tokens': {'joint': 10, 'placement': 30}},
             {'uid': 'a', 'objects': 2, 'input_tokens': {'joint': 30, 'placement': 12}},
             {'uid': 'b', 'objects': 5, 'input_tokens': {'joint': 900, 'placement': 950}},
             {'uid': 'd', 'objects': 1, 'input_tokens': {'joint': 5, 'placement': 2500}}]
    plan = s.batches(items, decode)
    assert [b['uids'] for b in plan] == [['b', 'a'], ['c', 'd']]  # objects desc, then max tokens desc, then UID.
    assert plan[0]['shared_padded_width'] == 950 and plan[0]['max_new_tokens'] == 2000 - 950  # context-bound
    assert plan[1]['shared_padded_width'] == 30  # overlong 2500 is ineligible, not part of W
    assert plan[1]['max_new_tokens'] == 1024 + 128 * 2  # object-bound
    assert s.batches(items, {**decode, 'max_new_tokens': 100})[1]['max_new_tokens'] == 100  # configured-cap-bound
    assert s.batches(items, {**decode, 'context_budget': 950})[0]['shared_padded_width'] == 900  # input >= budget is ineligible


def test_plan_reads_requests_only(tmp_path):
    first = make_data(tmp_path / 'one')
    second = make_data(tmp_path / 'two', sidecar=b'different private sidecar\n')
    rooms = second / 'rooms.jsonl'
    rooms.write_text(rooms.read_text().replace('"chair"', '"target-only"'))
    manifest = h.strict_json((second / 'MANIFEST.json').read_text())
    h.write_json(second / 'MANIFEST.json', {**manifest, 'files': {n: h.sha256(second / n) for n in manifest['files']}})
    plans = [s.load_plan(make_plan(root, data))[0] for root, data in ((tmp_path / 'one', first), (tmp_path / 'two', second))]
    assert plans[0]['batches'] == plans[1]['batches']
    rows = {arm: s.i.load_eval(first, arm)[0] for arm in h.ARMS}
    relabelled = {arm: [{**r, 'source': 'X', 'group': 'Y', 'room_type': 'Z'} for r in rs] for arm, rs in rows.items()}
    assert s.entries(rows, Tok()) == s.entries(relabelled, Tok())


def test_tampered_plan_rejected_even_when_rehashed(setup):
    _, _, plan = setup
    value = h.strict_json((plan / 'plan.json').read_bytes())
    for change in ({'max_new_tokens': 99999}, {'uids': ['r1', 'r1']}):
        forged = {**value, 'batches': [{**value['batches'][0], **change}, *value['batches'][1:]]}
        (plan / 'plan.json').unlink()
        h.write_json(plan / 'plan.json', forged)
        with pytest.raises(ValueError, match='hash mismatch'):
            s.load_plan(plan)
        h.write_json(plan / 'MANIFEST.json', {'files': {'plan.json': h.sha256(plan / 'plan.json')}})
        with pytest.raises(ValueError, match='amendment 002|exactly once'):
            s.load_plan(plan)


def test_merge_rejects_missing_duplicate_and_tampered_shards(setup):
    root, data, plan = setup
    shards = complete_arm(root, data, plan, 'joint')
    for chosen in ([shards[0]], [shards[0], shards[0]]):
        with pytest.raises(ValueError, match='missing or duplicate'):
            merge(root, data, plan, 'joint', chosen)
    receipt = shards[1] / 'receipts' / '00000000.json'
    value = h.strict_json(receipt.read_text())
    receipt.unlink()
    s.i.write_once(receipt, {**value, 'raw': '{"placements": []}'})
    with pytest.raises(ValueError, match='inventory/hash'):
        merge(root, data, plan, 'joint', shards)
    manifest = h.strict_json((shards[1] / 'MANIFEST.json').read_text())
    h.write_json(shards[1] / 'MANIFEST.json', {**manifest, 'files': {
        n: h.sha256(shards[1] / n) for n in manifest['files']}})
    with pytest.raises(ValueError, match='ledger/receipt'):
        merge(root, data, plan, 'joint', shards)
    assert not (root / 'infer-joint-002').exists()


def test_duplicate_uid_across_shards_rejected(setup):
    root, data, plan = setup
    shards = complete_arm(root, data, plan, 'joint')
    real = s._verified_shard
    def duplicated(*args):
        receipt, marker, result, receipts = real(*args)
        return receipt, marker, result, receipts + receipts[:1]
    with patch.object(s, '_verified_shard', side_effect=duplicated), pytest.raises(ValueError, match='duplicate UID'):
        merge(root, data, plan, 'joint', shards)


def test_plan_mismatch_between_arms_rejected(setup):
    root, data, plan = setup
    other = make_plan(root, data, name='plan-other', tokenizer={'source': 'different'})
    joint = merge(root, data, plan, 'joint', complete_arm(root, data, plan, 'joint'))
    placement_shards = complete_arm(root, data, other, 'placement')
    with pytest.raises(ValueError, match='shard plan or arm mismatch'):
        merge(root, data, plan, 'placement', placement_shards)
    placement = merge(root, data, other, 'placement', placement_shards)
    with pytest.raises(ValueError, match='paired context budget plan mismatch'):
        evaluate_pair.evaluate(data, {'joint': joint, 'placement': placement}, root / 'geometry', repeats=20)


def test_orphan_receipt_adopted_and_partial_batch_regenerated(setup):
    root, data, plan = setup
    real_append, real_write = s.i.ledger_append, s.i.write_once
    appended = []
    def dying_append(out, path, record):
        appended.append(path.name)
        if len(appended) == 4:  # Receipt 3 is on disk, its ledger line is not.
            raise KeyboardInterrupt()
        return real_append(out, path, record)
    first = shard_args(root, data, plan, 'joint', 0)
    with patch.object(s.i, 'ledger_append', side_effect=dying_append), pytest.raises(KeyboardInterrupt):
        run_shard(first)
    calls = []
    assert run_shard(shard_args(root, data, plan, 'joint', 0, resume=True), fake_generate(calls))['recorded_count'] == 4
    assert calls == []  # Adopted orphan completed the shard without regeneration.
    written = []
    def dying_write(path, value):
        if path.parent.name == 'receipts':
            written.append(path.name)
            if len(written) == 2:  # Batch 1 (shard 1's first) half recorded, no orphan.
                raise KeyboardInterrupt()
        return real_write(path, value)
    second = shard_args(root, data, plan, 'joint', 1)
    with patch.object(s.i, 'write_once', side_effect=dying_write), pytest.raises(KeyboardInterrupt):
        run_shard(second)
    calls = []
    run_shard(shard_args(root, data, plan, 'joint', 1, resume=True), fake_generate(calls))
    assert len(calls) == 2  # Partial batch 1 regenerated, then batch 3.
    ledger = [h.strict_json(x) for x in (second.out / 'ledger.jsonl').read_text().splitlines()]
    assert [e['receipt'] for e in ledger] == [f'receipts/{k:08d}.json' for k in range(3)]
    merge(root, data, plan, 'joint', [first.out, second.out])


def test_resume_rejects_identity_mismatch_and_tampered_ledger(setup):
    root, data, plan = setup
    first = shard_args(root, data, plan, 'joint', 0)
    with pytest.raises(KeyboardInterrupt):
        run_shard(first, fake_generate([], fail_on=2))
    with pytest.raises(ValueError, match='resume run identity'):
        run_shard(shard_args(root, data, plan, 'joint', 0, count=3, resume=True, out=first.out))
    with pytest.raises(FileNotFoundError):
        run_shard(shard_args(root, data, plan, 'joint', 1, resume=True))
    with pytest.raises(FileExistsError):
        run_shard(first)
    ledger = first.out / 'ledger.jsonl'
    ledger.write_text(ledger.read_text().replace('"generated": true', '"generated": false', 1))
    with pytest.raises(ValueError, match='ledger/receipt'):
        run_shard(shard_args(root, data, plan, 'joint', 0, resume=True))


def test_completed_shard_resume_is_read_only_and_tokenizer_drift_fails_before_generation(setup):
    root, data, plan = setup
    args = shard_args(root, data, plan, 'joint', 0)
    run_shard(args)
    before = h.artifact_manifest(args.out)
    generate = Mock(side_effect=AssertionError('must not generate'))
    assert run_shard(shard_args(root, data, plan, 'joint', 0, resume=True), generate)['status'] == 'SHARD_COMPLETE'
    assert h.artifact_manifest(args.out) == before
    class Drift(Tok):
        def apply_chat_template(self, messages, **kwargs):
            return ' ' + super().apply_chat_template(messages, **kwargs)
    with pytest.raises(ValueError, match='tokenizer differs'):
        run_shard(shard_args(root, data, plan, 'joint', 1), generate, tok=Drift())
    generate.assert_not_called()


def test_runtime_merges_lora_before_generation():
    tok = Mock(pad_token_id=0)
    tok.get_vocab.return_value = {'<|im_end|>': EOS}
    peft_model, merged = Mock(), Mock()
    peft_model.merge_and_unload.return_value = merged
    modules = {
        'torch': types.SimpleNamespace(bfloat16='bf16', inference_mode=contextlib.nullcontext,
            cuda=Mock(device_count=Mock(return_value=1), is_bf16_supported=Mock(return_value=True))),
        'transformers': types.SimpleNamespace(AutoTokenizer=Mock(from_pretrained=Mock(return_value=tok)),
            AutoModelForCausalLM=Mock(), GenerationConfig=lambda **kw: types.SimpleNamespace(**kw)),
        'peft': types.SimpleNamespace(PeftModel=Mock(from_pretrained=Mock(return_value=peft_model)))}
    args = types.SimpleNamespace(adapter=Path('/a'), base=Path('/b'), max_new_tokens=8, context_budget=20)
    with patch.dict(sys.modules, modules), patch.dict('os.environ', {'WORLD_SIZE': '1'}):
        s.i._runtime(args, merge=True)
    peft_model.merge_and_unload.assert_called_once_with(safe_merge=True)
    merged.eval.assert_called_once()
    peft_model.eval.assert_not_called()


def test_submit_chain_dependencies_with_stub_sbatch(tmp_path):
    code = Path(__file__).resolve().parents[3]
    scripts = code / 'fastfill/comparison'
    for name in ('isambard_infer_plan.sbatch', 'isambard_infer_shard.sbatch', 'isambard_infer_merge.sbatch',
                 'isambard_geometry.sbatch', 'submit_infer_v2.sh'):
        subprocess.run(['bash', '-n', str(scripts / name)], check=True)
    stub = tmp_path / 'sbatch'
    stub.write_text('#!/bin/sh\necho "$@" >> "$SBATCH_LOG"\necho "$((100 + $(wc -l < "$SBATCH_LOG")));cluster"\n')
    stub.chmod(0o755)
    env = {'PATH': f'{tmp_path}:/usr/bin:/bin', 'SBATCH_LOG': str(tmp_path / 'log')}
    run = subprocess.run(['bash', str(scripts / 'submit_infer_v2.sh'), str(code), '7365681', '-', '2'],
                         env=env, capture_output=True, text=True, check=True)
    log = (tmp_path / 'log').read_text().splitlines()
    root = '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013'
    kill = '--parsable --kill-on-invalid-dep=yes --dependency=afterok:'
    assert log[0] == f'--parsable {scripts}/isambard_infer_plan.sbatch {code} {root}/infer-v2-002'
    assert log[1] == f'{kill}101:7365681 {scripts}/isambard_infer_shard.sbatch {code} joint 0 2 {root}/infer-v2-002'
    assert log[3] == (f'{kill}102:103 {scripts}/isambard_infer_merge.sbatch {code} joint 2 '
                      f'{root}/infer-v2-002 {root}/infer-joint-002')
    assert log[4].startswith(f'{kill}101 {scripts}/isambard_infer_shard.sbatch {code} placement 0 2 ')
    assert log[7] == f'{kill}104:107 {scripts}/isambard_geometry.sbatch {code} 002 geometry-v2'
    assert run.stdout.splitlines()[-1] == 'geometry 108'
    bad = subprocess.run(['bash', str(scripts / 'submit_infer_v2.sh'), str(code), 'x', '-'], env=env)
    assert bad.returncode != 0 and len((tmp_path / 'log').read_text().splitlines()) == 8
