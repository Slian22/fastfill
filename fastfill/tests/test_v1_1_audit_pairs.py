"""Training-only preference audit contracts, with no network/model dependencies."""
import copy
import json
from pathlib import Path

import pytest

from fastfill.v1_1.audit_pairs import audit_pair, sanitize_pairs, size_violations


def example():
    user = {'boundary_type': 'polygon', 'boundary': [[0, 0], [8, 0], [8, 8], [0, 8]],
            'height': 3, 'objects': [{'id': 'bed_1'}, {'id': 'bed_2'}],
            'constraints': [['near', 'bed_1', 'bed_2', .5]]}
    layouts = [{'id': 'bed_1', 'size': [1, 1, .5], 'pos': [2, 2, 0], 'yaw': 0},
               {'id': 'bed_2', 'size': [1, 1, .5], 'pos': [3.5, 2, 0], 'yaw': 0}]
    msgs = [{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': json.dumps(user)},
            {'role': 'assistant', 'content': json.dumps({'placements': layouts})}]
    ref = {'uid': 'train-room', 'source': 'fixture', 'messages': msgs}
    bad = copy.deepcopy(layouts); bad[1]['pos'] = [2, 2, 0]
    pair = {'uid': ref['uid'], 'source': ref['source'], 'prompt': msgs[:2],
            'chosen': msgs[2]['content'], 'rejected': json.dumps({'placements': bad}),
            'reject_kind': 'main_collision'}
    return ref, pair


def test_valid_preference_kept_and_constraint_wrong_positive_dropped():
    ref, pair = example()
    assert audit_pair(pair, ref)['keep']
    bad = json.loads(pair['chosen']); bad['placements'][1]['pos'] = [5, 5, 0]
    result = audit_pair({**pair, 'chosen': json.dumps(bad)}, ref)
    assert not result['keep']
    assert 'chosen:constraint_failure' in result['drop_reasons']


def test_corrected_size_gate_contains_thin_and_large_ground_truth():
    gt = {'rug': {'size': [2, 1, .01]}, 'wall': {'size': [8, .1, 2]}}
    assert size_violations(gt, gt) == []
    assert set(size_violations(gt, gt, legacy=True)) == {'rug', 'wall'}
    bad = {'rug': {'size': [.1, 1, .01]}, 'wall': {'size': [20, .1, 2]}}
    assert set(size_violations(bad, gt)) == {'rug', 'wall'}


def test_size_only_negative_kept_but_unjustified_negative_dropped():
    ref, pair = example()
    # Large Z is below ceiling but outside size range; geometry alone passes.
    bad = json.loads(pair['chosen']); bad['placements'][0]['size'][2] = 1.3
    out = audit_pair({**pair, 'rejected': json.dumps(bad)}, ref)
    assert out['keep'] and out['rejected_violations'] == ['size_distortion']
    neutral = json.loads(pair['chosen']); neutral['placements'][0]['yaw'] = 10
    out = audit_pair({**pair, 'rejected': json.dumps(neutral)}, ref)
    assert not out['keep'] and 'rejected:no_verified_defect' in out['drop_reasons']


def test_foreign_prompt_source_and_duplicate_identity_fail_closed(tmp_path):
    ref, pair = example()
    for patch in ({'uid': 'foreign'}, {'source': 'other'}, {'prompt': pair['prompt'][:1]}):
        with pytest.raises(ValueError): audit_pair({**pair, **patch}, ref)
    train, s1, s2 = write_inputs(tmp_path, [ref], [pair, pair], [pair])
    with pytest.raises(ValueError, match='duplicate'):
        sanitize_pairs(train, s1, s2, tmp_path / 'out', workers=1)
    assert not (tmp_path / 'out').exists()


def write_inputs(tmp_path, refs, first, second):
    paths = [tmp_path / name for name in ('train.jsonl', 's1.jsonl', 's2.jsonl')]
    for path, rows in zip(paths, (refs, first, second)):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return paths


def test_atomic_receipt_idempotency_and_mutated_source_refusal(tmp_path):
    ref, pair = example()
    train, s1, s2 = write_inputs(tmp_path, [ref], [pair], [pair])
    out = tmp_path / 'clean'
    receipt = sanitize_pairs(train, s1, s2, out, workers=1)
    assert receipt['outputs']['stage1']['kept'] == 1
    assert receipt['outputs']['stage1']['dropped'] == 0
    assert (out / 'stage1.cleaned.jsonl').read_text() == s1.read_text()
    assert sanitize_pairs(train, s1, s2, out, workers=1) == receipt
    s1.write_text(s1.read_text() + '\n')
    with pytest.raises(ValueError, match='identity'):
        sanitize_pairs(train, s1, s2, out, workers=1)


def test_output_tamper_and_empty_retained_split_refused(tmp_path):
    ref, pair = example()
    train, s1, s2 = write_inputs(tmp_path, [ref], [pair], [pair])
    out = tmp_path / 'clean'; sanitize_pairs(train, s1, s2, out, workers=1)
    (out / 'stage2.cleaned.jsonl').write_text('tamper')
    with pytest.raises(ValueError, match='hash'):
        sanitize_pairs(train, s1, s2, out, workers=1)
    neutral = {**pair, 'rejected': pair['chosen']}
    s1.write_text(json.dumps(neutral) + '\n')
    with pytest.raises(ValueError, match='No verified'):
        sanitize_pairs(train, s1, s2, tmp_path / 'empty', workers=1)
    assert not (tmp_path / 'empty').exists()


def test_bad_chosen_parse_dropped_and_bad_rejected_parse_is_a_verified_defect():
    ref, pair = example()
    bad = audit_pair({**pair, 'chosen': 'not json'}, ref)
    assert not bad['keep'] and 'chosen:parse_failure' in bad['drop_reasons']
    bad_negative = audit_pair({**pair, 'rejected': 'not json'}, ref)
    assert bad_negative['keep'] and bad_negative['rejected_violations'] == ['parse_failure']


def test_missing_training_membership_and_invalid_worker_count(tmp_path):
    ref, pair = example()
    train, s1, s2 = write_inputs(tmp_path, [], [pair], [pair])
    with pytest.raises(ValueError, match='membership'):
        sanitize_pairs(train, s1, s2, tmp_path / 'absent', workers=1)
    for count in (0, 17):
        with pytest.raises(ValueError, match='workers'):
            sanitize_pairs(train, s1, s2, tmp_path / 'absent', workers=count)


def test_parallel_audit_same_output_as_serial(tmp_path):
    ref, pair = example()
    train, s1, s2 = write_inputs(tmp_path, [ref], [pair], [pair])
    serial = sanitize_pairs(train, s1, s2, tmp_path / 'serial', workers=1)
    parallel = sanitize_pairs(train, s1, s2, tmp_path / 'parallel', workers=2)
    assert serial == parallel


def test_reference_failure_and_floor_support_failure_drop_pairs():
    ref, pair = example()
    bad = json.loads(pair['chosen']); bad['placements'][0]['pos'][2] = .4
    result = audit_pair({**pair, 'chosen': json.dumps(bad)}, ref)
    assert 'chosen:support_fail' in result['drop_reasons']
    wrong_ref = copy.deepcopy(ref); wrong_ref['messages'][2]['content'] = json.dumps(bad)
    result = audit_pair(pair, wrong_ref)
    assert 'reference:support_fail' in result['drop_reasons']


@pytest.mark.parametrize('reason', ['over_length', 'prefix_mismatch'])
def test_dpo_refuses_all_filtered_pairs_before_loading_distributed_model(tmp_path, monkeypatch, reason):
    import sys
    from unittest.mock import Mock
    from safetensors.torch import save_file
    from fastfill.v1_1 import train_dpo

    class TinyTokenizer:
        eos_token, pad_token, pad_token_id = '<end>', '<pad>', 0

        def apply_chat_template(self, messages, **kwargs):
            return 'prompt:'

        def __call__(self, texts, **kwargs):
            ids = [[ord(c) for c in text] for text in texts]
            if reason == 'prefix_mismatch':
                ids = [[999] + row[1:] if len(row) > len('prompt:') else row for row in ids]
            return {'input_ids': ids}

    _, pair = example(); pairs = tmp_path / 'pairs.jsonl'
    pairs.write_text(json.dumps(pair) + '\n')
    model = tmp_path / 'model'
    model.mkdir()
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (model / name).write_text('{}')
    save_file({'weight': train_dpo.torch.zeros(1)}, model / 'model.safetensors')
    monkeypatch.setenv('WORLD_SIZE', '2'); monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(train_dpo.dist, 'init_process_group', lambda **kwargs: None)
    broadcast = Mock(return_value=None)
    monkeypatch.setattr(train_dpo.dist, 'broadcast_object_list', broadcast)
    monkeypatch.setattr(train_dpo.torch.cuda, 'set_device', lambda value: None)
    monkeypatch.setattr(train_dpo.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(train_dpo.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: TinyTokenizer())

    def forbidden(*args, **kwargs):
        pytest.fail('An empty tokenized preference dataset must fail before loading model weights')

    monkeypatch.setattr(train_dpo.AutoModelForCausalLM, 'from_pretrained', forbidden)
    monkeypatch.setattr(sys, 'argv', ['train_dpo', '--model', str(model), '--pairs', str(pairs),
                                    '--out', str(tmp_path / 'run'), '--max_len', '1' if reason == 'over_length' else '8192'])
    with pytest.raises(ValueError, match='No valid DPO pairs'):
        train_dpo.main()
    broadcast.assert_called_once_with([None], src=0)
    assert json.loads((tmp_path / 'run/trainer_identity.json').read_text())['world_size'] == 2
    assert not (tmp_path / 'run/final').exists()


@pytest.mark.parametrize('microbatches', [1, 4, 7, 8])
def test_partial_accumulation_matches_each_actual_group_mean(microbatches):
    import torch
    from fastfill.v1_1.train_dpo import accumulation_window_size

    weight = torch.tensor(2., requires_grad=True)
    values = torch.arange(1, microbatches + 1, dtype=torch.float32)
    gradients = []
    for index, value in enumerate(values):
        divisor = accumulation_window_size(index, microbatches, 4)
        ((weight * value).square() / divisor).backward()
        if (index + 1) % 4 == 0 or index + 1 == microbatches:
            gradients.append(weight.grad.item()); weight.grad = None
    expected = [4 * group.square().mean().item() for group in values.split(4)]
    assert gradients == pytest.approx(expected)
    if microbatches == 7:
        assert accumulation_window_size(6, 7, 4) == 3


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_dpo_loss_fails_before_backward(value):
    import torch
    from fastfill.v1_1.train_dpo import backward_finite_loss

    weight = torch.tensor(2., requires_grad=True)
    loss = weight * torch.tensor(value)
    with pytest.raises(FloatingPointError, match='Non-finite DPO loss'):
        backward_finite_loss(loss)
    assert weight.grad is None


def test_finite_dpo_loss_preserves_normal_backward():
    import torch
    from fastfill.v1_1.train_dpo import backward_finite_loss

    weight = torch.tensor(2., requires_grad=True)
    backward_finite_loss(weight.square())
    assert weight.grad.item() == 4.
