"""Shared-context integrity and runtime tests, using synthetic inputs only."""
import contextlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from fastfill.comparison.tests import test_infer as fixture

w = fixture.worker
h = fixture.h


def record(uid, length, arm):
    original = fixture.row(uid, arm)
    return {**original, 'messages': [
        {'role': 'system', 'content': 'x' * length}, original['messages'][1]]}


def paired_rows(lengths=((6, 7), (10, 12))):
    return {arm: [record(uid, n, arm) for uid, n in zip(('b', 'a'), ns)]
            for arm, ns in zip(h.ARMS, lengths)}


def test_plan_order_is_arm_independent_and_preserves_frozen_uid_order():
    paired = paired_rows()
    left = w.context_plan(paired, fixture.Tokenizer(), 1, 20, 19, 19)
    right = w.context_plan(dict(reversed(list(paired.items()))), fixture.Tokenizer(), 1, 20, 19, 19)
    assert left == right
    assert w._hash(left) == w._hash(right)
    assert [entry['uids'] for entry in left] == [['b'], ['a']]
    assert [entry['max_new_tokens'] for entry in left] == [10, 8]
    with pytest.raises(ValueError, match='UID mismatch'):
        w.context_plan({**paired, 'placement': paired['placement'][::-1]}, fixture.Tokenizer(), 2, 20, 19, 19)


@pytest.mark.parametrize('long_width', [19, 20, 21])
def test_context_boundary_retains_failures_without_poisoning_valid_rows(long_width):
    paired = paired_rows(((6, 7), (10, long_width)))
    plan = w.context_plan(paired, fixture.Tokenizer(), 2, 20, 19, 19)[0]
    expected_cap = 1 if long_width == 19 else 10
    assert plan['max_new_tokens'] == expected_cap
    for arm in h.ARMS:
        calls = []
        def generate(ids, cap):
            calls.append(cap)
            width = max(map(len, ids))
            return [[0] * width + [65] * (cap - 1) + [999] for _ in ids], width
        rows = w.process_batch(paired[arm], arm, fixture.Tokenizer(), generate,
                               999, 19, 19, 0, 20, expected_cap)
        assert calls == [expected_cap]
        assert [row['uid'] for row in rows] == ['b', 'a']
        for row in rows:
            if arm == 'placement' and row['uid'] == 'a' and long_width >= 20:
                assert row['error'] == 'INPUT_TOO_LONG' and not row['generated']
            else:
                assert row['effective_max_new_tokens'] == expected_cap
                assert row['generated_tokens'] == expected_cap
                assert row['stopped_on_eos'] and row['reached_token_cap']


def test_all_overlong_never_generates_and_configured_cap_is_respected():
    paired = paired_rows(((20, 21), (21, 20)))
    plan = w.context_plan(paired, fixture.Tokenizer(), 2, 20, 19, 3)[0]
    generate = Mock(side_effect=AssertionError('must not generate'))
    rows = w.process_batch(paired['joint'], 'joint', fixture.Tokenizer(), generate,
                           999, 19, 3, 0, 20, plan['max_new_tokens'])
    assert all(row['error'] == 'INPUT_TOO_LONG' for row in rows)
    generate.assert_not_called()
    short = w.context_plan(paired_rows(), fixture.Tokenizer(), 2, 20, 19, 3)
    assert short[0]['max_new_tokens'] == 3


def test_actual_runtime_shared_cap_equals_decode_cap_for_unequal_arm_widths():
    class Batch(dict):
        def to(self, device):
            return self
    tok = fixture.Tokenizer()
    tok.pad_token_id = 999
    tok.get_vocab = lambda: {'<|im_end|>': 999}
    tok.pad = lambda value, **kwargs: Batch(input_ids=types.SimpleNamespace(
        shape=(len(value['input_ids']), max(map(len, value['input_ids'])))), attention_mask=value['attention_mask'])
    model = Mock()
    def model_generate(**kwargs):
        count, width = kwargs['input_ids'].shape
        cap = kwargs['generation_config'].max_new_tokens
        return Mock(cpu=Mock(return_value=Mock(tolist=Mock(
            return_value=[[0] * width + [65] * cap for _ in range(count)]))))
    model.generate.side_effect = model_generate
    modules = {
        'torch': types.SimpleNamespace(bfloat16='bf16', inference_mode=contextlib.nullcontext,
            cuda=Mock(device_count=Mock(return_value=1), is_bf16_supported=Mock(return_value=True))),
        'transformers': types.SimpleNamespace(AutoTokenizer=Mock(from_pretrained=Mock(return_value=tok)),
            AutoModelForCausalLM=Mock(), GenerationConfig=lambda **kw: types.SimpleNamespace(**kw)),
        'peft': types.SimpleNamespace(PeftModel=Mock(from_pretrained=Mock(return_value=model)))}
    args = types.SimpleNamespace(adapter=Path('/synthetic/adapter'), base=Path('/synthetic/base'),
                                 max_new_tokens=19, context_budget=20)
    with patch.dict(sys.modules, modules), patch.dict('os.environ', {'WORLD_SIZE': '1'}):
        _, generate, eos = w._runtime(args)
        for arm, records in paired_rows().items():
            rows = w.process_batch(records, arm, tok, generate, eos, 19, 19, 0, 20, 8)
            assert model.generate.call_args.kwargs['generation_config'].max_new_tokens == 8
            assert all(r['generated_tokens'] == r['effective_max_new_tokens'] == 8 for r in rows)
            assert all(r['reached_token_cap'] and not r['stopped_on_eos'] for r in rows)
        for forged in (0, -1, True, 1.5, 9):
            with pytest.raises(ValueError, match='shared generation cap'):
                generate([[1] * 12], forged)


def run_fixture(root, arm):
    data = root / 'data'
    if not data.exists():
        data.mkdir()
        fixture.eval_fixture(data)
        manifest = json.loads((data / 'MANIFEST.json').read_text())
        for name, rows in paired_rows().items():
            path = data / name / 'requests.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        h.write_json(data / 'MANIFEST.json', {**manifest, 'files': {
            name: h.sha256(data / name) for name in manifest['files']}})
    args = types.SimpleNamespace(data=data, out=root / arm, arm=arm,
        adapter=root / 'adapter', base=root / 'base', training_run=root / 'training',
        batch_size=2, max_input_tokens=19, max_new_tokens=19, context_budget=20)
    def generate(ids, cap):
        receipt = json.loads((args.out / 'run_receipt.json').read_text())
        assert receipt['context_plan_sha256'] == h.sha256(args.out / 'context_plan.json')
        assert w._hash({k: v for k, v in receipt.items() if k != 'run_sha256'}) == receipt['run_sha256']
        width = max(map(len, ids))
        return [[0] * width + [65] * cap for _ in ids], width
    with patch.object(w, 'training_identity', return_value={}), patch.object(
            w, '_runtime', return_value=(fixture.Tokenizer(), generate, 999)):
        w.run(args)
    return args


def test_both_run_receipts_bind_identical_plan_before_generation(tmp_path):
    args = [run_fixture(tmp_path, arm) for arm in h.ARMS]
    receipts = [json.loads((a.out / 'run_receipt.json').read_text()) for a in args]
    assert receipts[0]['context_plan_sha256'] == receipts[1]['context_plan_sha256']
    assert receipts[0]['run_sha256'] != receipts[1]['run_sha256']  # Arm/run identities differ.
    for a, receipt in zip(args, receipts):
        rows = [json.loads(line) for line in (a.out / 'records.jsonl').read_text().splitlines()]
        assert [r['uid'] for r in rows] == ['b', 'a']
        assert all(r['run_sha256'] == receipt['run_sha256'] for r in rows)
        assert all(r['effective_max_new_tokens'] == 8 for r in rows)


@pytest.mark.parametrize('refresh_manifest', [False, True])
def test_geometry_rejects_tampered_plan_even_if_manifest_is_rehashed(tmp_path, refresh_manifest):
    from fastfill.comparison import evaluate_pair
    args = run_fixture(tmp_path, 'joint')
    records, _ = evaluate_pair._inference(args.out, args.data, ['b', 'a'])
    assert len(records) == 2
    plan_path = args.out / 'context_plan.json'
    plan = json.loads(plan_path.read_text())
    h.write_json(plan_path, [{**plan[0], 'max_new_tokens': 1}])
    if refresh_manifest:
        path = args.out / 'MANIFEST.json'
        manifest = json.loads(path.read_text())
        h.write_json(path, {**manifest, 'files': {
            name: h.sha256(args.out / name) for name in manifest['files']}})
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluate_pair._inference(args.out, args.data, ['b', 'a'])


def test_forged_preexisting_plan_directory_is_not_consumed(tmp_path):
    args = types.SimpleNamespace(out=tmp_path / 'out')
    args.out.mkdir()
    h.write_json(args.out / 'context_plan.json', [{'max_new_tokens': 1}])
    with patch.object(w, '_runtime') as runtime, pytest.raises(FileExistsError):
        w.run(args)
    runtime.assert_not_called()


def test_geometry_rejects_forged_plan_and_receipt_under_original_run_identity(tmp_path):
    from fastfill.comparison import evaluate_pair
    args = run_fixture(tmp_path, 'joint')
    plan_path = args.out / 'context_plan.json'
    plan = json.loads(plan_path.read_text())
    h.write_json(plan_path, [{**plan[0], 'max_new_tokens': 1}])
    receipt_path = args.out / 'run_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    forged = {**receipt, 'context_plan_sha256': h.sha256(plan_path)}
    assert w._hash({k: v for k, v in forged.items() if k != 'run_sha256'}) != receipt['run_sha256']
    h.write_json(receipt_path, forged)
    path = args.out / 'MANIFEST.json'
    manifest = json.loads(path.read_text())
    h.write_json(path, {**manifest, 'files': {
        name: h.sha256(args.out / name) for name in manifest['files']}})
    with pytest.raises(ValueError):
        evaluate_pair._inference(args.out, args.data, ['b', 'a'])
