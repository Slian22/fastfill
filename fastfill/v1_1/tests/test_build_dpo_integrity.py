"""Regression tests for preference generation and immutable conversion publication."""
import copy
import json
from pathlib import Path
import random
import sys
import types

import pytest

from fastfill.scene import messages
from fastfill.v1_1 import build_dpo, build_v1_1
from fastfill.v1_1.scene_v1_1 import messages_v1_1


def room():
    return {'boundary_type': 'polygon', 'boundary': [[0, 0], [8, 0], [8, 8], [0, 8]],
            'objects': [{'id': 'rug_1', 'category': 'rug', 'size': [2, 2, .01], 'pos': [2, 2, 0], 'yaw': 0},
                        {'id': 'chair_1', 'category': 'chair', 'size': [.5, .5, .8], 'pos': [5, 5, 0], 'yaw': 0}]}


def row(constraints=None):
    return {'uid': 'fixture', 'source': 'fixture', 'flags': {}, 'messages': messages_v1_1(room(), constraints)}


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return 'prompt'

    def __call__(self, texts, **kwargs):
        return {'input_ids': [[1] for _ in texts]}


def setup_stage1(tmp_path, monkeypatch, answers, record=None, fail=False):
    # Import real torch before monkeypatching sys.modules, so module restoration is harmless.
    import torch
    import transformers
    record = record or row()
    (tmp_path / 'train.jsonl').write_text(json.dumps(record) + '\n')
    model = tmp_path / 'model'; model.mkdir(exist_ok=True)
    (model / 'weights').write_text('frozen-model')
    calls = []

    class LLM:
        def __init__(self, **kwargs):
            calls.append('load')

        def generate(self, prompts, params):
            if fail:
                raise RuntimeError('interrupted inference')
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=a) for a in answers]) for _ in prompts]

    monkeypatch.setitem(sys.modules, 'vllm', types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *args, **kw: Tokenizer())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Weight hashing/format validation has dedicated real-safetensors tests; keep these CPU fixtures minimal.
    monkeypatch.setattr(build_dpo, 'model_identity', lambda path: {'path': str(path), 'bytes': (Path(path) / 'weights').read_text()}, raising=False)
    return record, str(model), calls


def generate(tmp_path, model, **kwargs):
    return build_dpo.build_stage1_vllm(model, tmp_path, tmp_path / 'pairs.jsonl',
                                    kwargs.get('max_pairs', 1), 2, 1024, 1, kwargs.get('seed', 42))


def bad_answer(record):
    bad = json.loads(record['messages'][2]['content'])
    bad['placements'][1]['pos'] = [20, 20, 0]
    return json.dumps(bad)


def test_stage1_thin_ground_truth_is_not_a_negative(tmp_path, monkeypatch):
    record = row()
    _, model, _ = setup_stage1(tmp_path, monkeypatch, [record['messages'][2]['content']])
    generate(tmp_path, model)
    assert (tmp_path / 'pairs.jsonl').read_text() == ''


def test_stage1_verified_bad_candidate_keeps_correct_thin_positive(tmp_path, monkeypatch):
    record = row()
    _, model, _ = setup_stage1(tmp_path, monkeypatch, [record['messages'][2]['content'], bad_answer(record)])
    generate(tmp_path, model)
    pair = json.loads((tmp_path / 'pairs.jsonl').read_text())
    assert pair['chosen'] == record['messages'][2]['content']
    assert pair['rejected'] == bad_answer(record)
    assert pair['reject_kind'] == 'oob'


@pytest.mark.parametrize('stage', ['stage1', 'stage2'])
def test_invalid_reference_constraints_never_become_a_positive(tmp_path, monkeypatch, stage):
    record = row([['near', 'rug_1', 'chair_1', .1]])
    if stage == 'stage2':
        assert build_dpo._make_hard_negative(record['messages'], random.Random(42)) is None
    else:
        _, model, _ = setup_stage1(tmp_path, monkeypatch, [bad_answer(record)], record)
        generate(tmp_path, model)
        assert (tmp_path / 'pairs.jsonl').read_text() == ''


def test_legacy_partial_cannot_be_adopted(tmp_path, monkeypatch):
    _, model, _ = setup_stage1(tmp_path, monkeypatch, [])
    partial = tmp_path / 'pairs.jsonl.partial'
    partial.write_text(json.dumps({'uid': 'foreign', 'chosen': 'X', 'rejected': 'Y'}) + '\n')
    with pytest.raises(ValueError, match='(?i)(unbound|identity|legacy)'):
        generate(tmp_path, model)
    assert not (tmp_path / 'pairs.jsonl').exists()


@pytest.mark.parametrize('change', ['data', 'model', 'settings', 'code'])
def test_bound_resume_rejects_identity_drift(tmp_path, monkeypatch, change):
    record = row()
    _, model, _ = setup_stage1(tmp_path, monkeypatch, [bad_answer(record)])
    generate(tmp_path, model)
    if change == 'data':
        with (tmp_path / 'train.jsonl').open('a') as stream: stream.write('\n')
    elif change == 'model':
        (Path(model) / 'weights').write_text('changed')
    elif change == 'code':
        cache = json.loads((tmp_path / 'pairs.jsonl.partial').read_text())
        cache['identity']['code_sha256']['v1_1/build_dpo.py'] = 'wrong'
        (tmp_path / 'pairs.jsonl.partial').write_text(json.dumps(cache))
    with pytest.raises(ValueError, match='(?i)identity'):
        generate(tmp_path, model, seed=17 if change == 'settings' else 42)


def test_bound_resume_validates_every_cached_pair_and_deduplicates(tmp_path, monkeypatch):
    record = row()
    _, model, calls = setup_stage1(tmp_path, monkeypatch, [bad_answer(record)])
    generate(tmp_path, model)
    generate(tmp_path, model)
    assert calls == ['load']
    cache_path = tmp_path / 'pairs.jsonl.partial'
    cache = json.loads(cache_path.read_text())
    cache['pairs'][0]['chosen'] = cache['pairs'][0]['rejected']
    cache_path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match='(?i)(pair|positive|semantic)'):
        generate(tmp_path, model)


def test_failed_generation_keeps_a_bound_recoverable_cache(tmp_path, monkeypatch):
    record = row()
    _, model, _ = setup_stage1(tmp_path, monkeypatch, [bad_answer(record)], fail=True)
    with pytest.raises(RuntimeError, match='interrupted'):
        generate(tmp_path, model)
    cache = json.loads((tmp_path / 'pairs.jsonl.partial').read_text())
    assert cache['identity'] and cache['pairs'] == []
    assert not (tmp_path / 'pairs.jsonl').exists()


def converter_inputs(tmp_path):
    src = tmp_path / 'src'; src.mkdir()
    record = {'uid': 'fixture', 'source': 'fixture', 'flags': {}, 'messages': messages(room())}
    for split in ('train', 'dev', 'test'):
        (src / f'{split}.jsonl').write_text(json.dumps(record) + '\n')
    for split in ('dev_rooms', 'dev_constrained_rooms', 'test_rooms', 'test_constrained_rooms'):
        (src / f'{split}.jsonl').write_text(json.dumps(room()) + '\n')
    return src


def convert(src, out, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['build_v1_1', '--src', str(src), '--out', str(out), '--workers', '1'])
    build_v1_1.main()


def test_converter_rejects_nonempty_target_without_modifying_it(tmp_path, monkeypatch):
    src = converter_inputs(tmp_path); out = tmp_path / 'out'; out.mkdir()
    stale = out / 'test_rooms.jsonl'; stale.write_text('old')
    with pytest.raises((ValueError, FileExistsError), match='(?i)(existing|empty|replace)'):
        convert(src, out, monkeypatch)
    assert list(out.iterdir()) == [stale] and stale.read_text() == 'old'


def test_converter_requires_all_seven_inputs_before_publication(tmp_path, monkeypatch):
    src = converter_inputs(tmp_path); out = tmp_path / 'out'
    (src / 'test_rooms.jsonl').unlink()
    with pytest.raises((ValueError, FileNotFoundError), match='test_rooms'):
        convert(src, out, monkeypatch)
    assert not out.exists()


def test_converter_failure_preserves_existing_empty_destination(tmp_path, monkeypatch):
    src = converter_inputs(tmp_path); out = tmp_path / 'out'; out.mkdir()
    (src / 'dev.jsonl').write_text('not-json\n')
    with pytest.raises(ValueError):
        convert(src, out, monkeypatch)
    assert out.is_dir() and list(out.iterdir()) == []


def test_converter_publishes_exact_cohort_and_source_hashes(tmp_path, monkeypatch):
    src = converter_inputs(tmp_path); out = tmp_path / 'out'; out.mkdir()
    convert(src, out, monkeypatch)
    manifest = json.loads((out / 'MANIFEST.json').read_text())
    assert len(manifest['files']) == 7
    assert len(manifest['source_sha256']) == 7
    assert json.loads(json.loads((out / 'train.jsonl').read_text())['messages'][1]['content'])['objects'][0] == {'id': 'rug_1'}


@pytest.mark.parametrize('damage', ['malformed', 'duplicate_pair', 'foreign_uid', 'prompt', 'attempts'])
def test_corrupt_bound_cache_fails_closed(tmp_path, monkeypatch, damage):
    record = row()
    _, model, _ = setup_stage1(tmp_path, monkeypatch, [bad_answer(record)])
    generate(tmp_path, model)
    path = tmp_path / 'pairs.jsonl.partial'
    cache = json.loads(path.read_text())
    if damage == 'duplicate_pair': cache['pairs'] *= 2
    if damage == 'foreign_uid': cache['pairs'][0]['uid'] = 'foreign'
    if damage == 'prompt': cache['pairs'][0]['prompt'] = []
    if damage == 'attempts': cache['attempted_uids'] = ['foreign']
    path.write_text('invalid JSON' if damage == 'malformed' else json.dumps(cache))
    before = (tmp_path / 'pairs.jsonl').read_bytes()
    with pytest.raises(ValueError): generate(tmp_path, model)
    assert (tmp_path / 'pairs.jsonl').read_bytes() == before


def test_output_tampering_is_not_silently_replaced(tmp_path, monkeypatch):
    record = row()
    _, model, _ = setup_stage1(tmp_path, monkeypatch, [bad_answer(record)])
    generate(tmp_path, model)
    (tmp_path / 'pairs.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='Existing output differs'):
        generate(tmp_path, model)
    assert (tmp_path / 'pairs.jsonl').read_text() == 'tampered'


def test_generation_identity_drift_refuses_publication(tmp_path, monkeypatch):
    record = row()
    _, model, _ = setup_stage1(tmp_path, monkeypatch, [bad_answer(record)])
    identify = build_dpo._generation_identity
    calls = []
    def drifting(*args):
        value = identify(*args); calls.append(1)
        return value if len(calls) == 1 else {**value, 'changed': True}
    monkeypatch.setattr(build_dpo, '_generation_identity', drifting)
    with pytest.raises(ValueError, match='identity changed'):
        generate(tmp_path, model)
    assert not (tmp_path / 'pairs.jsonl').exists()


def test_generation_skips_unparseable_answers_and_invalid_reference(tmp_path, monkeypatch):
    _, model, _ = setup_stage1(tmp_path, monkeypatch, ['not JSON'])
    generate(tmp_path, model)
    assert (tmp_path / 'pairs.jsonl').read_text() == ''
    broken = row()['messages']; broken[2]['content'] = 'not JSON'
    assert build_dpo._make_hard_negative(broken, random.Random(42)) is None


def test_stage2_emits_only_audited_pairs_with_fixed_and_support_geometry(tmp_path):
    from fastfill.v1_1.audit_pairs import audit_pair
    r = room()
    r['objects'] = [
        {'id': 'desk_1', 'category': 'desk', 'size': [1, 1, .8], 'pos': [2, 2, 0], 'yaw': 0},
        {'id': 'cabinet_1', 'category': 'cabinet', 'size': [1, 1, .8], 'pos': [5, 5, 0], 'yaw': 0},
        {'id': 'lamp_1', 'category': 'lamp', 'size': [.2, .2, .4], 'pos': [2, 2, .8], 'yaw': 0, 'parent': 'desk_1'}]
    r['fixed'] = [{'id': 'door_1', 'category': 'door', 'size': [.2, .9, 2], 'pos': [0, 4, 0], 'yaw': 0}]
    record = {**row(), 'messages': messages_v1_1(r, [['on', 'lamp_1', 'desk_1']])}
    kinds = set()
    for seed in range(30):
        result = build_dpo._make_hard_negative(record['messages'], random.Random(seed))
        assert result is not None
        text, kind = result; kinds.add(kind)
        pair = build_dpo._verified_pair(record, record['messages'][2]['content'], text, kind)
        assert pair is not None and audit_pair(pair, record)['keep']
    assert {'door_block', 'support_z_corrupt', 'collision_shift', 'wall_oob'} <= kinds
    (tmp_path / 'train.jsonl').write_text(json.dumps(record) + '\n')
    build_dpo.build_stage2_hardneg(tmp_path, tmp_path / 'stage2.jsonl', 1, 42)
    assert audit_pair(json.loads((tmp_path / 'stage2.jsonl').read_text()), record)['keep']


def test_converter_audits_cross_category_collisions():
    r = room(); r['objects'][0] = {**r['objects'][0], 'id': 'cabinet_1', 'category': 'cabinet', 'size': [.5, .5, .8], 'pos': [5, 5, 0]}
    record = {'uid': 'collision', 'source': 'fixture', 'messages': messages(r)}
    _, flags = build_v1_1._process_chunk([json.dumps(record)], True)[0]
    assert flags['main_collision'] == 1 and flags['severe_collision'] == 1


def test_converter_rejects_invalid_room_json_atomically(tmp_path, monkeypatch):
    src = converter_inputs(tmp_path); out = tmp_path / 'out'
    (src / 'test_rooms.jsonl').write_text('{"objects":[]}\n')
    with pytest.raises(ValueError, match='Invalid room record'):
        convert(src, out, monkeypatch)
    assert not out.exists()


def test_converter_limit_keeps_all_seven_files_consistent(tmp_path, monkeypatch):
    src = converter_inputs(tmp_path); out = tmp_path / 'out'
    for file in src.iterdir(): file.write_text(file.read_text() * 2)
    monkeypatch.setattr(sys, 'argv', ['build_v1_1', '--src', str(src), '--out', str(out), '--workers', '1', '--limit', '1'])
    build_v1_1.main()
    assert all(len(file.read_text().splitlines()) == 1 for file in out.glob('*.jsonl'))


@pytest.mark.parametrize('max_pairs', [0, -1])
def test_stage2_invalid_limits_cannot_truncate_existing_output(tmp_path, max_pairs):
    (tmp_path / 'train.jsonl').write_text(json.dumps(row()) + '\n')
    target = tmp_path / 'stage2.jsonl'; target.write_text('PREVIOUS VERIFIED DATA\n')
    with pytest.raises(ValueError, match='positive'):
        build_dpo.build_stage2_hardneg(tmp_path, target, max_pairs, 42)
    assert target.read_text() == 'PREVIOUS VERIFIED DATA\n'


def test_stage2_existing_output_is_never_overwritten(tmp_path):
    (tmp_path / 'train.jsonl').write_text(json.dumps(row()) + '\n')
    target = tmp_path / 'stage2.jsonl'; target.write_text('existing')
    with pytest.raises(FileExistsError, match='existing'):
        build_dpo.build_stage2_hardneg(tmp_path, target, 1, 42)
    assert target.read_text() == 'existing'


def test_stage2_generation_lock_rejects_a_second_writer(tmp_path):
    (tmp_path / 'train.jsonl').write_text(json.dumps(row()) + '\n')
    target = tmp_path / 'stage2.jsonl'
    with build_dpo._generation_lock(target):
        with pytest.raises(BlockingIOError):
            build_dpo.build_stage2_hardneg(tmp_path, target, 1, 42)
    assert not target.exists()


def test_stage2_preserves_legacy_partial_and_leaves_no_temp_after_success(tmp_path):
    (tmp_path / 'train.jsonl').write_text(json.dumps(row()) + '\n')
    partial = tmp_path / 'stage2.jsonl.partial'; partial.write_text('legacy artifact')
    target = tmp_path / 'stage2.jsonl'
    build_dpo.build_stage2_hardneg(tmp_path, target, 1, 42)
    assert json.loads(target.read_text())['uid'] == 'fixture'
    assert partial.read_text() == 'legacy artifact'
    assert sorted(p.name for p in tmp_path.glob('.stage2.jsonl.*')) == ['.stage2.jsonl.lock']


def test_stage2_generation_failure_and_data_drift_do_not_publish(tmp_path, monkeypatch):
    source = tmp_path / 'train.jsonl'; source.write_text(json.dumps(row()) + '\n')
    target = tmp_path / 'stage2.jsonl'
    original = build_dpo._make_hard_negative
    def changing(messages, rng):
        result = original(messages, rng)
        source.write_text(source.read_text() + '\n')
        return result
    monkeypatch.setattr(build_dpo, '_make_hard_negative', changing)
    with pytest.raises(ValueError, match='identity changed'):
        build_dpo.build_stage2_hardneg(tmp_path, target, 1, 42)
    assert not target.exists()


def test_stage2_output_appearing_during_generation_is_preserved(tmp_path, monkeypatch):
    (tmp_path / 'train.jsonl').write_text(json.dumps(row()) + '\n')
    target = tmp_path / 'stage2.jsonl'
    original = build_dpo._make_hard_negative
    def another_writer(messages, rng):
        target.write_text('external writer')
        return original(messages, rng)
    monkeypatch.setattr(build_dpo, '_make_hard_negative', another_writer)
    with pytest.raises(FileExistsError, match='existing'):
        build_dpo.build_stage2_hardneg(tmp_path, target, 1, 42)
    assert target.read_text() == 'external writer'


def test_stage2_publication_race_is_create_only_and_cleans_unique_temp(tmp_path, monkeypatch):
    (tmp_path / 'train.jsonl').write_text(json.dumps(row()) + '\n')
    target = tmp_path / 'stage2.jsonl'
    link = build_dpo.os.link
    def racing_link(source, destination):
        Path(destination).write_text('arrived immediately before publish')
        link(source, destination)
    monkeypatch.setattr(build_dpo.os, 'link', racing_link)
    with pytest.raises(FileExistsError):
        build_dpo.build_stage2_hardneg(tmp_path, target, 1, 42)
    assert target.read_text() == 'arrived immediately before publish'
    assert sorted(p.name for p in tmp_path.glob('.stage2.jsonl.*')) == ['.stage2.jsonl.lock']
