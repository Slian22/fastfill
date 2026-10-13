"""Amendments 008/009 DPO stages on CPU: fakes stand in for vLLM, train_dpo, merge_lora and Slurm."""
import copy
import json
import math
from pathlib import Path
import random
import subprocess
import types
from unittest.mock import patch

import pytest

from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair
from fastfill.comparison import geometry as g
from fastfill.comparison import infer_shard as s
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.tests import test_infer_shard as T

EOS_ID = 999999
BASE = {'path': '/models/Qwen3-8B', 'revision': 'rev', 'files': {'config.json': 'd' * 64}, 'receipt_hashes': {}}
ROOT_SHA = s.i._hash(BASE['files'])
TARGETS = {'OptiScene_holodeck': 3, 'IL3D_3dfront': 2}
HEADER = json.dumps({'w': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
SAFETENSORS = len(HEADER).to_bytes(8, 'little') + HEADER + b'\0' * 4


class Tok:
    """Prompt = user JSON; one token per character, the EOS text is one token."""
    eos_token = d.EOS

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert [m['role'] for m in messages] == ['system', 'user']
        return messages[1]['content']

    def _ids(self, text):
        if text.endswith(self.eos_token):
            return [ord(c) for c in text[:-len(self.eos_token)]] + [EOS_ID]
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        assert kwargs.get('add_special_tokens') is False
        return {'input_ids': [self._ids(t) for t in text] if isinstance(text, list) else self._ids(text)}


def obj(oid, category, x, y, z=0, size=(0.6, 0.6, 0.8), parent=None):
    return {'id': oid, 'category': category, 'size': list(size), 'pos': [x, y, z], 'yaw': 0,
            **({'parent': parent} if parent else {})}


def raw_room(uid, source, objects):
    return {'uid': uid, 'source': source, 'group': 'g-' + uid, 'room_type': 'bedroom', 'boundary_type': 'polygon',
            'boundary': [[0, 0], [6, 0], [6, 6], [0, 6]], 'height': 3, 'objects': objects, 'fixed': []}


def spaced(uid, source):
    return raw_room(uid, source, [obj('a', 'cabinet', 1, 1), obj('b', 'sofa', 2.5, 1), obj('c', 'wardrobe', 4, 1)])


ROOMS = [spaced('h0', 'OptiScene_holodeck'), spaced('h1', 'OptiScene_holodeck'),
         raw_room('h2', 'OptiScene_holodeck', [obj('t', 'cabinet', 3, 3, size=(1, 1, 0.7)), obj('s', 'sofa', 1, 1),
                                                obj('l', 'lamp', 3, 3, 0.7, (0.2, 0.2, 0.3), parent='t')]),
         spaced('h3', 'OptiScene_holodeck'),  # Not length-admitted.
         raw_room('h4', 'OptiScene_holodeck', [obj('a', 'cabinet', 1, 1), obj('b', 'sofa', 1.1, 1)]),  # GT collides.
         spaced('f0', 'IL3D_3dfront'),
         raw_room('f1', 'IL3D_3dfront', [obj('a', 'cabinet', 5.9, 1), obj('b', 'sofa', 2, 2)]),  # GT out of room.
         spaced('x0', 'Structured3D'), spaced('x1', 'Structured3D')]
ADMITTED = sorted(r['uid'] for r in ROOMS if r['uid'] != 'h3')


def record(room, arm, split):
    sizes = {o['id']: o['size'] for o in room['objects']}
    return {'uid': room['uid'], 'source': room['source'], 'split': split, 'group': room['group'], 'flags': {},
            'messages': messages(room, arm, sizes if arm == 'placement' else None),
            'target_policy': h.MAIN_TARGET_POLICY, 'target_qualified': split == 'train'}


def build_main(root, rooms=ROOMS):
    data = root / 'main-data-v1'
    bound = {'train': [bind_room(r)[0] for r in rooms], 'dev': [bind_room(spaced('dev0', 'IL3D_3dfront'))[0]]}
    for arm in h.ARMS:
        (data / arm).mkdir(parents=True)
        for split, values in bound.items():
            (data / arm / f'{split}.jsonl').write_bytes(d._jsonl(record(r, arm, split) for r in values))
    ledger = []
    for room in bound['train']:
        entry = {'uid': room['uid'], 'accepted': True, 'eligible': True}
        for arm in h.ARMS:
            entry[arm] = {'qualified': True, 'reasons': [], **h.qualification_hashes(record(room, arm, 'train')['messages'], arm),
                          'counts': {'valid': True, **{k: 0 for k in ('oob', 'support_fail', 'ceiling', 'fixed_blocking', 'main', 'severe')}}}
        ledger.append(entry)
    (data / 'common-admission.jsonl').write_bytes(d._jsonl(ledger))
    for split, values in bound.items():
        (data / f'{split}_rooms.jsonl').write_bytes(d._jsonl(values))
    h.write_json(data / 'MANIFEST.json', {'target_policy': h.MAIN_TARGET_POLICY, 'files': h.artifact_manifest(data),
                                          'counts': {'train': len(rooms), 'dev': 1}})
    return data


def build_sft(root, data):
    pairs, info = h.load_pair(data)
    rows = {a: h._by_uid(pairs['train'][a]) for a in h.ARMS}
    lengths = [{'uid': u, 'lengths': {a: len(Tok().apply_chat_template(rows[a][u]['messages'][:2], tokenize=False,
                                                                       add_generation_prompt=True, enable_thinking=False))
                                      + len(rows[a][u]['messages'][2]['content']) + 1 for a in h.ARMS},
                'answer_tokens': {a: len(rows[a][u]['messages'][2]['content']) + 1 for a in h.ARMS}}
               for u in sorted(rows['joint'])]
    runs = []
    for arm in h.ARMS:
        run = root / f'sft-{arm}-001'
        run.mkdir()
        h.write_json(run / 'lengths.json', {'train': {'accepted_uids': ADMITTED, 'lengths': lengths}, 'dev': {}})
        h.write_json(run / 'run_receipt.json', {'config': {'max_len': d.MAX_LEN},
                                               'data': {'manifest_sha256': info['manifest_sha256']}})
        h.write_json(run / 'status.json', {'status': 'TRAINING_COMPLETE', 'completed_arms': [arm]})
        h.write_json(run / 'SHA256.json', h.artifact_manifest(run))
        runs.append(run)
    return runs


def fake_apply_lora(base, out, adapter):
    out = Path(out)
    out.mkdir()
    (out / 'config.json').write_text(json.dumps({'base': str(base), 'adapter': str(adapter)}))
    for name in ('tokenizer_config.json', 'tokenizer.json'):
        (out / name).write_text('{}')
    (out / 'model.safetensors').write_bytes(SAFETENSORS)


def sft_identity(run):
    return {'training_run_sha256': 'c' * 64, 'base': BASE, 'base_sha256': ROOT_SHA,
            'training_data': {'manifest_sha256': 'a' * 64}, 'adapter_files': {}, 'adapter_sha256': 'e' * 64}


class FakeLLM:
    """Per prompt: capped, unparsable, the positive itself, then a shifted valid layout."""
    def __init__(self, answers, invalid=(), die_after=None):
        self.answers, self.invalid, self.die_after, self.calls = answers, set(invalid), die_after, []

    def generate(self, prompts, params, use_tqdm):
        assert use_tqdm is False and len(prompts) == len(params)
        if self.die_after is not None and len(self.calls) == self.die_after:
            raise KeyboardInterrupt()
        self.calls.append(len(prompts))
        outputs = []
        for prompt, p in zip(prompts, params):
            ids = prompt['prompt_token_ids']
            assert (p['n'], p['temperature'], p['top_p'], p['max_tokens']) == (4, .85, .95, d.MAX_LEN - len(ids))
            uid, answer = self.answers[''.join(map(chr, ids))]
            value = json.loads(answer)
            for o in value['placements']:
                o['pos'][0] = round(o['pos'][0] + .1, 2)
            texts = [(answer[:10], 'length'), ('not json', 'stop'), (answer, 'stop'), (json.dumps(value), 'stop')]
            if uid in self.invalid:
                texts = [('{"placements": []}', 'stop')] * 4
            outputs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t, finish_reason=f, token_ids=[0] * 3)
                                                          for t, f in texts]))
        return outputs


def args(**kw):
    return types.SimpleNamespace(**kw)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """main data -> positives -> merged SFT models -> generations -> stage1/stage2 pairs, all on CPU."""
    monkeypatch.setattr(d, 'TARGETS', TARGETS)
    monkeypatch.setattr(d, '_model_tokenizer', lambda model: Tok())
    monkeypatch.setattr(d.i, 'training_identity', sft_identity)
    monkeypatch.setattr('fastfill.merge_lora.apply_lora', fake_apply_lora)
    data = build_main(tmp_path)
    runs = build_sft(tmp_path, data)
    root = tmp_path / 'dpo-001'
    d.positives(args(data=data, sft_runs=runs, out=root / 'positives', workers=1))
    value, records, _, _ = d.load_positives(root / 'positives')
    answers = {r['messages'][1]['content']: (r['uid'], r['messages'][2]['content']) for a in h.ARMS for r in records[a]}
    fill = value['fill_uids'][0]
    for arm in h.ARMS:
        d.merge_sft(args(arm=arm, base=Path('/models/Qwen3-8B'), sft_run=tmp_path / f'sft-{arm}-001', dpo_root=root))
        llm = FakeLLM(answers, invalid=[fill] if arm == 'placement' else [])
        d.generate(args(dpo_root=root, arm=arm, chunk=2), load=lambda engine: (llm, lambda **kw: kw))
    d.stage1_pairs(args(dpo_root=root, base=Path('/models/Qwen3-8B')))
    d.stage2_pairs(args(dpo_root=root, base=Path('/models/Qwen3-8B')))
    return types.SimpleNamespace(root=root, data=data, runs=runs, value=value, records=records, fill=fill, answers=answers)


def pair_rows(root, stage, arm):
    return [h.strict_json(line) for line in (root / stage / f'{arm}.jsonl').read_text().splitlines()]


def test_selection_is_deterministic_hash_ranked_and_fills_shortfall(monkeypatch):
    monkeypatch.setattr(d, 'TARGETS', {'OptiScene_holodeck': 3, 'IL3D_3dfront': 4})
    rng = random.Random(0)
    entries = [{'uid': f'u{k}', 'source': ('OptiScene_holodeck', 'IL3D_3dfront', 'other')[k % 3],
                'gate': {'joint': k % 7 != 0, 'placement': k % 11 != 0}} for k in range(12)]
    first = d.select(entries)
    for _ in range(5):
        shuffled = copy.deepcopy(entries)
        rng.shuffle(shuffled)
        assert d.select(shuffled) == first
    passing = sorted((e for e in entries if all(e['gate'].values())), key=lambda e: d.rank(e['uid']))
    holo = [e['uid'] for e in passing if e['source'] == 'OptiScene_holodeck'][:3]
    front = [e['uid'] for e in passing if e['source'] == 'IL3D_3dfront']
    assert len(front) == 3 and first['shortfall'] == {'OptiScene_holodeck': 0, 'IL3D_3dfront': 1}
    rest = [e['uid'] for e in passing if e['uid'] not in holo + front]
    assert first['fill_uids'] == rest[:1] and first['uids'] == sorted(holo + front + rest[:1])
    assert d.rank('u1') == __import__('hashlib').sha256(b'dpo-positives-v1:u1').hexdigest()
    assert first['pool']['gate_fail'] == {'joint': 2, 'placement': 2}  # u0, u7 / u0, u11
    monkeypatch.setattr(d, 'TARGETS', {'OptiScene_holodeck': 30, 'IL3D_3dfront': 30})
    with pytest.raises(ValueError, match='too few'):
        d.select(entries)


def test_positives_are_length_admitted_gate_passing_and_target_free_inputs(pipeline):
    value, records = pipeline.value, pipeline.records
    assert value['by_source']['OptiScene_holodeck'] == 3 and value['shortfall'] == {'OptiScene_holodeck': 0,
                                                                                      'IL3D_3dfront': 1}
    assert {'h3', 'h4', 'f1'}.isdisjoint(value['uids']) and {'h0', 'h1', 'h2', 'f0'} <= set(value['uids'])
    assert value['fill_uids'] == [min(['x0', 'x1'], key=d.rank)] and len(value['uids']) == 5
    assert value['pool'] == {'length_admitted': 8, 'gate_pass_both_arms': 6, 'gate_fail': {'joint': 2, 'placement': 2},
                             'gate_pass_by_source': {'IL3D_3dfront': 1, 'OptiScene_holodeck': 3, 'Structured3D': 2}}
    assert 'not unanimous human approval' in value['label']
    for arm in h.ARMS:
        assert [r['uid'] for r in records[arm]] == value['uids']
        for row in records[arm]:  # Model inputs carry no target: J has id/category only, P adds its prior size.
            s.i.validate_request({**{k: row[k] for k in ('uid', 'source', 'group')}, 'messages': row['messages'][:2]}, arm)
            assert value['prompt_tokens'][row['uid']][arm] == len(row['messages'][1]['content'])
    with pytest.raises(FileExistsError):  # Written once.
        d.positives(args(data=pipeline.data, sft_runs=pipeline.runs, out=pipeline.root / 'positives', workers=1))


def test_positives_do_not_depend_on_row_order(tmp_path, pipeline):
    reordered = tmp_path / 'other'
    reordered.mkdir()
    data = build_main(reordered, list(reversed(ROOMS)))
    runs = build_sft(reordered, data)
    d.positives(args(data=data, sft_runs=runs, out=reordered / 'positives', workers=2))  # Spawned gate workers.
    again = d.load_positives(reordered / 'positives')[0]
    assert again['uids'] == pipeline.value['uids'] and again['uids_sha256'] == pipeline.value['uids_sha256']


def test_positives_refuse_disagreeing_sft_length_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'TARGETS', TARGETS)
    data = build_main(tmp_path)
    runs = build_sft(tmp_path, data)
    lengths = h.strict_json((runs[1] / 'lengths.json').read_text())
    lengths['train']['accepted_uids'] = ADMITTED[1:]
    h.write_json(runs[1] / 'lengths.json', lengths)
    h.write_json(runs[1] / 'SHA256.json', h.artifact_manifest(runs[1]))
    with pytest.raises(ValueError, match='disagree'):
        d.positives(args(data=data, sft_runs=runs, out=tmp_path / 'positives', workers=1))
    assert not (tmp_path / 'positives').exists()


def test_generation_resumes_from_receipts_and_binds_identity(tmp_path, monkeypatch, pipeline):
    root = pipeline.root
    result = h.strict_json((root / 'joint/stage1-generations/result.json').read_text())
    assert result['status'] == 'GENERATION_COMPLETE' and result['samples'] == 20
    assert result['finish_reasons'] == {'length': 5, 'stop': 15}
    receipt = h.strict_json((root / 'joint/stage1-generations/run_receipt.json').read_text())
    assert receipt['config']['sampling']['temperature'] == .85 and receipt['config']['engine']['max_model_len'] == 8192
    # Interrupted second chunk, then a resume that regenerates only what has no receipt.
    rmtree = __import__('shutil').rmtree
    rmtree(root / 'joint/stage1-generations')
    dying = FakeLLM(pipeline.answers, die_after=1)
    with pytest.raises(KeyboardInterrupt):
        d.generate(args(dpo_root=root, arm='joint', chunk=2), load=lambda engine: (dying, lambda **kw: kw))
    assert len(list((root / 'joint/stage1-generations/receipts').iterdir())) == 2
    llm = FakeLLM(pipeline.answers)
    d.generate(args(dpo_root=root, arm='joint', chunk=2), load=lambda engine: (llm, lambda **kw: kw))
    assert llm.calls == [2, 1]
    assert h.strict_json((root / 'joint/stage1-generations/result.json').read_text()) == result
    monkeypatch.setattr(d, 'SAMPLING', {**d.SAMPLING, 'temperature': 1.0})
    with pytest.raises(ValueError, match='resume identity'):
        d.generate(args(dpo_root=root, arm='joint', chunk=2), load=None)


def test_generation_refuses_prompt_rendering_drift(pipeline, monkeypatch):
    class Drift(Tok):
        def apply_chat_template(self, messages, **kwargs):
            return ' ' + super().apply_chat_template(messages, **kwargs)
    __import__('shutil').rmtree(pipeline.root / 'joint/stage1-generations')
    monkeypatch.setattr(d, '_model_tokenizer', lambda model: Drift())
    with pytest.raises(ValueError, match='prompt rendering differs'):
        d.generate(args(dpo_root=pipeline.root, arm='joint', chunk=2), load=None)


def test_stage1_pairs_verified_first_valid_differing_sample_same_uids_both_arms(pipeline):
    root, value = pipeline.root, pipeline.value
    kept = sorted(set(value['uids']) - {pipeline.fill})  # P had no valid negative for the fill room.
    summary = h.strict_json((root / 'stage1-pairs/summary.json').read_text())
    assert summary['kept_both_arms'] == 4 and summary['verified_per_arm'] == {'joint': 5, 'placement': 4}
    assert summary['dropped_by_other_arm_only'] == {'joint': 1, 'placement': 0}
    ledger = {r['uid']: r for r in map(h.strict_json, (root / 'stage1-pairs/ledger.jsonl').read_text().splitlines())}
    assert ledger[pipeline.fill]['placement']['rejected_samples'] == ['rejected_schema_invalid'] * 4
    assert not ledger[pipeline.fill]['kept']
    for arm in h.ARMS:
        rows = pair_rows(root, 'stage1-pairs', arm)
        assert [r['uid'] for r in rows] == kept
        for row in rows:
            assert row['sample'] == 3 and row['reject_kind'] == 'sft_generation'
            assert row['chosen'] == pipeline.answers[row['prompt'][1]['content']][1]
            assert ledger[row['uid']][arm]['rejected_samples'] == ['not_stopped_on_eos', 'rejected_schema_invalid',
                                                                   'rejected_equals_positive']
    with pytest.raises(FileExistsError):
        d.stage1_pairs(args(dpo_root=root, base=Path('/models/Qwen3-8B')))


def test_verify_pair_rejects_identity_changes_and_overlength(pipeline, monkeypatch):
    value, records = pipeline.value, pipeline.records
    _, _, rooms, _ = d.load_positives(pipeline.root / 'positives')
    good = pair_rows(pipeline.root, 'stage1-pairs', 'joint')[0]
    row = next(r for r in records['joint'] if r['uid'] == good['uid'])
    tokens = value['prompt_tokens'][row['uid']]['joint']
    assert d.verify_pair(good, row, rooms[row['uid']], 'joint', Tok(), tokens, 'stage1') is None
    for change in ({'chosen': good['rejected']}, {'prompt': [good['prompt'][1]]}, {'uid': 'other'}):
        with pytest.raises(ValueError, match='identity'):
            d.verify_pair({**good, **change}, row, rooms[row['uid']], 'joint', Tok(), tokens, 'stage1')
    with pytest.raises(ValueError, match='prompt tokens'):
        d.verify_pair(good, row, rooms[row['uid']], 'joint', Tok(), tokens + 1, 'stage1')
    monkeypatch.setattr(d, 'MAX_LEN', tokens + len(good['chosen']) + 1)  # Chosen fits exactly; rejected is longer.
    assert d.verify_pair(good, row, rooms[row['uid']], 'joint', Tok(), tokens, 'stage1') == 'overlength'


def test_stage2_identical_schedule_positions_only_targeted_violations(pipeline):
    root, value = pipeline.root, pipeline.value
    _, records, rooms, _ = d.load_positives(root / 'positives')
    rows = {a: {r['uid']: r for r in pair_rows(root, 'stage2-pairs', a)} for a in h.ARMS}
    assert list(rows['joint']) == list(rows['placement']) == value['uids']
    for uid in value['uids']:
        joint, placement = rows['joint'][uid], rows['placement'][uid]
        assert joint['perturbation'] == placement['perturbation'] and joint['reject_kind'] == placement['reject_kind']
        floor = d._floor(next(r for r in records['joint'] if r['uid'] == uid))
        assert joint['perturbation']['op'] in ('collision_shift', 'wall_oob')
        assert {k: v for k, v in joint['perturbation'].items() if k != 'moved'} in d.schedule(uid, floor)
        for arm, pair in (('joint', joint), ('placement', placement)):
            before, after = (h.strict_json(pair[k])['placements'] for k in ('chosen', 'rejected'))
            moved = {o['id'] for o, p in zip(before, after) if o['pos'] != p['pos']}
            assert moved == set(pair['perturbation']['moved'])
            assert [{k: v for k, v in o.items() if k != 'pos'} for o in before] == \
                   [{k: v for k, v in o.items() if k != 'pos'} for o in after]  # Sizes, yaw and support untouched.
            score = g.score(rooms[uid], pair['prompt'], pair['rejected'], arm)
            assert not score['geometry_success']
            assert (score['oob_objects'] if pair['reject_kind'] == 'wall_oob'
                    else score['main_collisions'] + score['severe_collisions']) > 0
    # The cabinet carries its lamp along; the lamp keeps its height on the cabinet.
    cabinet = next(o['id'] for o in h.strict_json(rows['joint']['h2']['chosen'])['placements'] if o['size'][0] == 1)
    lamp = next(o['id'] for o in h.strict_json(rows['joint']['h2']['chosen'])['placements'] if o.get('on'))
    step = {'op': 'wall_oob', 'move': cabinet, 'side': 'x_max'}
    row = next(r for r in records['joint'] if r['uid'] == 'h2')
    rejected, moved = d.perturb(row['messages'][2]['content'], h.strict_json(row['messages'][1]['content']), step)
    assert moved == sorted([cabinet, lamp])
    after = {o['id']: o for o in h.strict_json(rejected)['placements']}
    before = {o['id']: o for o in h.strict_json(row['messages'][2]['content'])['placements']}
    assert after[cabinet]['pos'][0] == 6.35 and after[lamp]['pos'][0] - before[lamp]['pos'][0] == pytest.approx(3.35)
    assert after[lamp]['pos'][2] == before[lamp]['pos'][2]


def test_schedule_is_seeded_by_uid_only():
    floor = ['obj_0001', 'obj_0002', 'obj_0003']
    assert d.schedule('u', floor) == d.schedule('u', list(floor)) != d.schedule('v', floor)
    assert d.schedule('u', ['obj_0001']) == [{'op': 'wall_oob', 'move': 'obj_0001',
                                              'side': d.schedule('u', ['obj_0001'])[0]['side']}]
    assert d.schedule('u', []) == []


def fake_train(calls):
    from fastfill.v1_1 import train_dpo

    def run(command, check, world=None):
        # torchrun DDP as v1.1: per device 1 x WORLD ranks x grad_accum; DistributedSampler pads the last batch.
        assert check is True and command[1:7] == ['-m', 'torch.distributed.run', '--standalone',
                                                  '--nproc_per_node=4', '-m', 'fastfill.v1_1.train_dpo']
        flags = dict(zip(command[7::2], command[8::2]))
        calls.append(flags)
        out, final = Path(flags['--out']), Path(flags['--out']) / 'final'
        final.mkdir(parents=True)
        for name in ('adapter_config.json', 'tokenizer_config.json', 'tokenizer.json'):
            (final / name).write_text('{}')
        (final / 'adapter_model.safetensors').write_bytes(SAFETENSORS)
        world = world or int(command[4].split('=')[1])
        count = len(Path(flags['--pairs']).read_text().splitlines())
        batches = math.ceil(count / (int(flags['--bs']) * world))
        steps = int(flags['--epochs']) * math.ceil(batches / int(flags['--grad_accum']))
        train_dpo.write_dpo_completion(final, out, count, batches, world, int(flags['--epochs']), steps, steps)
    return run


def test_train_both_stages_recipe_guard_merge_and_lineage(pipeline, monkeypatch):
    root, calls = pipeline.root, []
    monkeypatch.setattr(d.subprocess, 'run', fake_train(calls))
    for stage in ('dpo1', 'dpo2'):
        for arm in h.ARMS:
            receipt = d.train(args(dpo_root=root, arm=arm, stage=stage))
            assert receipt['stage'] == stage and receipt['base_sha256'] == ROOT_SHA
    # Amendment 009: official README values; effective batch 4 = 1 per device x 4 GPUs x accumulation 1.
    recipe = {'--beta': '0.1', '--lr': '5e-06', '--epochs': '10', '--bs': '1', '--grad_accum': '1',
              '--max_len': '8192', '--lora_r': '32', '--lora_alpha': '16', '--save_steps': '100', '--seed': '42'}
    assert [{k: v for k, v in c.items() if k in recipe} for c in calls] == [recipe] * 4
    assert d.WORLD_SIZE * d.RECIPE['bs'] * d.RECIPE['grad_accum'] == d.TRAIN_FIXED['effective_batch'] == 4
    assert [(Path(c['--model']).name, Path(c['--pairs']).parent.name) for c in calls] == [
        ('sft-merged', 'stage1-pairs'), ('sft-merged', 'stage1-pairs'),
        ('dpo1-merged', 'stage2-pairs'), ('dpo1-merged', 'stage2-pairs')]
    guards = [h.strict_json((root / a / s_ / 'training_guard.json').read_text()) for a in h.ARMS for s_ in ('dpo1', 'dpo2')]
    assert all(g_['settings'] == {**d.RECIPE, **d.TRAIN_FIXED} for g_ in guards)
    final = d.verify_merged(root / 'placement/dpo2-merged', 'placement', 'dpo2')
    assert final['parent']['stage'] == 'dpo1' and final['training']['pairs']['count'] == 5
    assert final['training']['completion']['steps'] == 10 * math.ceil(5 / 4)  # 5 pairs on 4 ranks: 2 batches/epoch.
    # The fake merge leaves the one tensor unchanged; the real diagnostic compared parent and merged files.
    assert final['merge_diagnostic'] == {'modules': {'other': {'changed': 0, 'total': 1, 'fraction': 0.0}},
                                         'dtypes': ['torch.float32'], 'tensors_only_in_parent': 0,
                                         'tensors_only_in_merged': 0}
    assert d.train(args(dpo_root=root, arm='joint', stage='dpo2'))['stage'] == 'dpo2' and len(calls) == 4  # Done: verify only.
    (root / 'joint/dpo2-merged/model.safetensors').write_bytes(SAFETENSORS[:-1] + b'\1')
    with pytest.raises(ValueError, match='bytes differ'):
        d.verify_merged(root / 'joint/dpo2-merged', 'joint', 'dpo2')
    receipt = h.strict_json((root / 'placement/sft-merged.json').read_text())
    (root / 'placement/sft-merged.json').unlink()
    s.i.write_once(root / 'placement/sft-merged.json', {**receipt, 'training_data': {'manifest_sha256': 'f' * 64}})
    with pytest.raises(ValueError, match='parent lineage receipt changed'):
        d.verify_merged(root / 'placement/dpo2-merged', 'placement', 'dpo2')


def test_train_refuses_incomplete_completion_and_dropped_pairs(pipeline, monkeypatch):
    root = pipeline.root
    def short(command, check):
        fake_train([])(command, check)
        flags = dict(zip(command[7::2], command[8::2]))
        path = Path(flags['--out']) / 'final/completion.json'
        value = json.loads(path.read_text())
        path.write_text(json.dumps({**value, 'admitted_pairs': value['admitted_pairs'] - 1}))
    monkeypatch.setattr(d.subprocess, 'run', short)
    with pytest.raises(ValueError, match='every pair'):
        d.train(args(dpo_root=root, arm='joint', stage='dpo1'))
    assert not (root / 'joint/dpo1-merged').exists()
    # A run that was not the 4-rank DDP recipe (here one process, all steps done) is refused as well.
    monkeypatch.setattr(d.subprocess, 'run', lambda command, check: fake_train([])(command, check, world=1))
    with pytest.raises(ValueError, match='every pair'):
        d.train(args(dpo_root=root, arm='placement', stage='dpo1'))
    assert not (root / 'placement/dpo1-merged').exists()
    monkeypatch.setattr(d, 'MAX_LEN', 50)  # train_dpo would silently skip every pair.
    with pytest.raises(ValueError, match='drop pairs'):
        d.preflight(root / 'placement/sft-merged', root / 'stage1-pairs/placement.jsonl', 4)
    with pytest.raises(ValueError, match='lineage receipt is not'):
        d.verify_merged(root / 'joint/sft-merged', 'placement', 'sft')


def test_inference_identity_uses_generic_lineage_root(pipeline, monkeypatch):
    run = types.SimpleNamespace(model=pipeline.root / 'joint/sft-merged', arm='joint', base=Path('/models/Qwen3-8B'))
    monkeypatch.setattr(d.h, 'base_receipt', lambda base: BASE)
    identity = d.inference_identity(run)
    receipt = h.strict_json((pipeline.root / 'joint/sft-merged.json').read_text())
    assert identity['base_sha256'] == ROOT_SHA != receipt['model_sha256'] == identity['model_sha256']
    assert identity['stage'] == 'sft' and identity['training_data'] == {'manifest_sha256': 'a' * 64}
    monkeypatch.setattr(d.h, 'base_receipt', lambda base: {**BASE, 'revision': 'other'})
    with pytest.raises(ValueError, match='lineage root'):
        d.inference_identity(run)


def test_exact_policy_identity_binds_training_base_and_unmerged_adapter(pipeline, monkeypatch):
    root = pipeline.root
    monkeypatch.setattr(d.subprocess, 'run', fake_train([]))
    monkeypatch.setattr(d.h, 'base_receipt', lambda base: BASE)
    for stage in ('dpo1', 'dpo2'):
        d.train(args(dpo_root=root, arm='joint', stage=stage))
    run = types.SimpleNamespace(model=root / 'joint/dpo2-merged', arm='joint', base=Path('/models/Qwen3-8B'),
                                exact_policy=True)
    identity = d.inference_identity(run)
    assert identity['stage'] == 'dpo2' and identity['base_sha256'] == ROOT_SHA and 'model_sha256' not in identity
    assert identity['policy'] == {  # The stage adapter on the merged previous stage it was trained on.
        'mode': 'unmerged_fp32_adapter_on_bf16_training_base',
        'training_base': str((root / 'joint/dpo1-merged').resolve()),
        'training_base_sha256': h.strict_json((root / 'joint/dpo1-merged.json').read_text())['model_sha256'],
        'adapter': str((root / 'joint/dpo2/final').resolve()),
        'adapter_sha256': s.i._hash(h.artifact_manifest(root / 'joint/dpo2/final'))}
    assert identity['lineage_sha256'] == h.sha256(root / 'joint/dpo2-merged.json')
    # The merged stage model is not what is evaluated, so its bytes are not part of the exact identity.
    (root / 'joint/dpo2-merged/model.safetensors').write_bytes(SAFETENSORS[:-1] + b'\1')
    assert d.inference_identity(run) == identity
    with pytest.raises(ValueError, match='bytes differ'):
        d.inference_identity(types.SimpleNamespace(**{**vars(run), 'exact_policy': False}))
    (root / 'joint/dpo2/final/adapter_model.safetensors').write_bytes(SAFETENSORS[:-1] + b'\1')
    with pytest.raises(ValueError, match='stage adapter bytes differ'):
        d.inference_identity(run)
    (root / 'joint/dpo1-merged/model.safetensors').write_bytes(SAFETENSORS[:-1] + b'\1')
    with pytest.raises(ValueError, match='bytes differ from its lineage receipt'):  # Training base rehashed.
        d.inference_identity(run)
    with pytest.raises(ValueError, match='DPO stages only'):
        d.inference_identity(types.SimpleNamespace(**{**vars(run), 'model': root / 'joint/sft-merged'}))


def test_merge_diagnostic_counts_changed_bf16_weights_per_module(tmp_path):
    import torch
    from safetensors.torch import save_file
    q, down = 'model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.down_proj.weight'
    parent = {q: torch.zeros(4, 4, dtype=torch.bfloat16), down: torch.ones(2, 8, dtype=torch.bfloat16),
              'model.norm.weight': torch.ones(4, dtype=torch.bfloat16), 'lm_head.weight': torch.ones(2, dtype=torch.bfloat16)}
    merged = {k: v.clone() for k, v in parent.items() if k != 'lm_head.weight'}
    merged[q][0, :2] = 1e-3                      # 2 of 16 q_proj weights change.
    merged[down] += 1e-4                         # Below half a BF16 ULP of 1.0: the merge rounds it away.
    for name, tensors in (('parent', parent), ('merged', merged)):
        (tmp_path / name).mkdir()
        items = sorted(tensors.items())
        save_file(dict(items[:2]), str(tmp_path / name / 'model-00001-of-00002.safetensors'))
        save_file(dict(items[2:]), str(tmp_path / name / 'model-00002-of-00002.safetensors'))
    assert d.merge_diagnostic(tmp_path / 'parent', tmp_path / 'merged') == {
        'modules': {'down_proj': {'changed': 0, 'total': 16, 'fraction': 0.0},
                    'other': {'changed': 0, 'total': 4, 'fraction': 0.0},
                    'q_proj': {'changed': 2, 'total': 16, 'fraction': 0.125}},
        'dtypes': ['torch.bfloat16'], 'tensors_only_in_parent': 1, 'tensors_only_in_merged': 0}


def test_runtime_exact_policy_keeps_fp32_adapter_unmerged_on_training_base():
    import contextlib
    import sys
    from unittest.mock import Mock
    tok = Mock(pad_token_id=0)
    tok.get_vocab.return_value = {'<|im_end|>': T.EOS}
    auto, peft = Mock(), Mock()
    modules = {
        'torch': types.SimpleNamespace(bfloat16='bf16', inference_mode=contextlib.nullcontext,
            cuda=Mock(device_count=Mock(return_value=1), is_bf16_supported=Mock(return_value=True))),
        'transformers': types.SimpleNamespace(AutoTokenizer=Mock(from_pretrained=Mock(return_value=tok)),
            AutoModelForCausalLM=auto, GenerationConfig=lambda **kw: types.SimpleNamespace(**kw)),
        'peft': types.SimpleNamespace(PeftModel=peft)}
    run = types.SimpleNamespace(base=Path('/dpo1-merged'), adapter=Path('/dpo2/final'), exact_policy=True,
                                max_new_tokens=8, context_budget=20)
    with patch.dict(sys.modules, modules), patch.dict('os.environ', {'WORLD_SIZE': '1'}):
        s.i._runtime(run, merge=False)
        with pytest.raises(ValueError, match='unmerged adapter'):
            s.i._runtime(run, merge=True)
    assert auto.from_pretrained.call_args.args == ('/dpo1-merged',)
    assert peft.from_pretrained.call_args.args[1:] == ('/dpo2/final',)
    assert peft.from_pretrained.call_args.kwargs['autocast_adapter_dtype'] is True  # FP32 LoRA, as train_dpo.
    peft.from_pretrained.return_value.merge_and_unload.assert_not_called()
    peft.from_pretrained.return_value.eval.assert_called_once()


def test_exact_policy_shards_load_unmerged_policy_and_results_carry_the_stage(tmp_path):
    data = T.make_data(tmp_path)
    plan = T.make_plan(tmp_path, data)
    outs = {}
    for arm in h.ARMS:
        policy = {'mode': 'unmerged_fp32_adapter_on_bf16_training_base', 'training_base': str(tmp_path / arm / 'dpo1-merged'),
                  'training_base_sha256': arm[0] * 64, 'adapter': str(tmp_path / arm / 'dpo2/final'), 'adapter_sha256': 'e' * 64}
        identity = {'base': {}, 'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': 'a' * 64},
                    'stage': 'dpo2', 'policy': policy, 'lineage_sha256': '0' * 64}
        shards = []
        for k in range(2):
            a = merged_shard_args(tmp_path, data, plan, arm, k, out=tmp_path / '004' / arm / f'shard-{k:02d}',
                                  extra=('--exact-policy',))
            with patch('fastfill.comparison.dpo_stages.inference_identity', return_value=identity) as verified, \
                    patch.object(s.i, '_runtime', return_value=(T.Tok(), T.fake_generate(), T.EOS)) as runtime:
                assert s.shard(a)['status'] == 'SHARD_COMPLETE'
            assert verified.call_args.args[0].exact_policy is True and verified.call_args.args[0].model == a.model
            loaded = runtime.call_args
            assert loaded.kwargs == {'merge': False} and not hasattr(loaded.args[0], 'model')
            assert (loaded.args[0].base, loaded.args[0].adapter, loaded.args[0].exact_policy) == (
                Path(policy['training_base']), Path(policy['adapter']), True)
            shards.append(a.out)
        outs[arm] = T.merge(tmp_path, data, plan, arm, shards, name=f'infer-{arm}-004')
    receipt = h.strict_json((outs['joint'] / 'run_receipt.json').read_text())
    assert receipt['config']['lora'] == 'unmerged_fp32_stage_adapter_on_bf16_training_base'
    assert receipt['policy']['training_base_sha256'] == 'j' * 64 and receipt['stage'] == 'dpo2'
    summary = evaluate_pair.evaluate(data, outs, tmp_path / 'geometry-v4', repeats=20)
    assert summary['stage'] == 'dpo2' and h.strict_json((tmp_path / 'geometry-v4/summary.json').read_text())['stage'] == 'dpo2'
    assert '模型阶段（stage）：dpo2' in (tmp_path / 'geometry-v4/report.md').read_text()
    alone = s.parse_args(['shard', '--data', str(data), '--base', str(tmp_path), '--adapter', str(tmp_path / 'a'),
                          '--training-run', str(tmp_path), '--exact-policy', '--arm', 'joint', '--plan', str(plan),
                          '--shard-index', '0', '--shard-count', '2', '--out', str(tmp_path / 'x')])
    with pytest.raises(ValueError, match='--exact-policy'):  # Only with --model (a DPO stage's lineage receipt).
        s.shard(alone)


def merged_shard_args(root, data, plan, arm, index, count=2, out=None, extra=()):
    return s.parse_args(['shard', '--data', str(data), '--base', str(root / 'base'), '--model', str(root / arm / 'merged'),
                         '--arm', arm, '--plan', str(plan), '--shard-index', str(index), '--shard-count', str(count),
                         '--out', str(out or root / 'v3' / arm / f'shard-{index:02d}'), *extra])


def run_merged_stage(root, data, plan, arm, stage, name):
    identity = {'base': {}, 'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': 'a' * 64},
                'stage': stage, 'model_sha256': arm[0] * 64, 'lineage_sha256': '0' * 64}
    shards = []
    for k in range(2):
        a = merged_shard_args(root, data, plan, arm, k, out=root / name / arm / f'shard-{k:02d}')
        with patch('fastfill.comparison.dpo_stages.inference_identity', return_value=identity) as verified, \
                patch.object(s.i, '_runtime', return_value=(T.Tok(), T.fake_generate(), T.EOS)) as runtime:
            assert s.shard(a)['status'] == 'SHARD_COMPLETE'
        assert verified.call_args.args[0].model == a.model and runtime.call_args.args[0].model == a.model
        shards.append(a.out)
    return T.merge(root, data, plan, arm, shards, name=f'infer-{arm}-{name}')


def test_merged_stage_inference_shares_plan_and_evaluates_on_lineage_root(tmp_path):
    data = T.make_data(tmp_path)
    plan = T.make_plan(tmp_path, data)
    outs = {arm: run_merged_stage(tmp_path, data, plan, arm, 'dpo1', '003') for arm in h.ARMS}
    receipts = {arm: h.strict_json((outs[arm] / 'run_receipt.json').read_text()) for arm in h.ARMS}
    assert receipts['joint']['model_sha256'] != receipts['placement']['model_sha256']  # Arm-specific merged models.
    assert receipts['joint']['config']['model'].endswith('joint/merged') and 'adapter' not in receipts['joint']['config']
    assert receipts['joint']['config']['lora'] == 'premerged_bf16_stage_model'
    summary = evaluate_pair.evaluate(data, outs, tmp_path / 'geometry-v3', repeats=20)
    assert summary['status'] == 'GEOMETRY_COMPLETE' and summary['arms']['joint']['all']['complete_count'] == 7
    other = run_merged_stage(tmp_path, data, plan, 'placement', 'dpo2', '004')
    with pytest.raises(ValueError, match='model stage mismatch'):
        evaluate_pair.evaluate(data, {'joint': outs['joint'], 'placement': other}, tmp_path / 'geometry-x', repeats=20)


def test_shard_requires_exactly_one_model_source(tmp_path):
    data = T.make_data(tmp_path)
    plan = T.make_plan(tmp_path, data)
    both = merged_shard_args(tmp_path, data, plan, 'joint', 0, extra=('--adapter', str(tmp_path / 'a'),
                                                                     '--training-run', str(tmp_path)))
    with pytest.raises(ValueError, match='--model'):
        s.shard(both)
    neither = s.parse_args(['shard', '--data', str(data), '--base', str(tmp_path), '--arm', 'joint', '--plan', str(plan),
                            '--shard-index', '0', '--shard-count', '2', '--out', str(tmp_path / 'n')])
    with pytest.raises(ValueError, match='--model'):
        s.shard(neither)


def test_runtime_loads_merged_model_without_adapter():
    import contextlib
    import sys
    from unittest.mock import Mock
    tok = Mock(pad_token_id=0)
    tok.get_vocab.return_value = {'<|im_end|>': T.EOS}
    auto, peft = Mock(), Mock()
    modules = {
        'torch': types.SimpleNamespace(bfloat16='bf16', inference_mode=contextlib.nullcontext,
            cuda=Mock(device_count=Mock(return_value=1), is_bf16_supported=Mock(return_value=True))),
        'transformers': types.SimpleNamespace(AutoTokenizer=Mock(from_pretrained=Mock(return_value=tok)),
            AutoModelForCausalLM=auto, GenerationConfig=lambda **kw: types.SimpleNamespace(**kw)),
        'peft': types.SimpleNamespace(PeftModel=peft)}
    run = types.SimpleNamespace(model=Path('/m'), base=Path('/b'), max_new_tokens=8, context_budget=20)
    with patch.dict(sys.modules, modules), patch.dict('os.environ', {'WORLD_SIZE': '1'}):
        s.i._runtime(run, merge=True)
    assert auto.from_pretrained.call_args.args == ('/m',)
    assert modules['transformers'].AutoTokenizer.from_pretrained.call_args.args == ('/m',)
    peft.from_pretrained.assert_not_called()
    auto.from_pretrained.return_value.eval.assert_called_once()


def test_copy_plan_is_byte_identical_and_new_only(tmp_path):
    data = T.make_data(tmp_path)
    plan = T.make_plan(tmp_path, data)
    outs = [tmp_path / 'infer-v2-003/plan', tmp_path / 'infer-v2-004/plan']
    result = d.copy_plan(args(src=plan, out=outs))
    for out in outs:
        assert (out / 'plan.json').read_bytes() == (plan / 'plan.json').read_bytes()
        assert s.load_plan(out)[1] == s.load_plan(plan)[1] == result['plan_sha256']
    with pytest.raises(FileExistsError):
        d.copy_plan(args(src=plan, out=[tmp_path / 'infer-v2-005/plan', outs[0]]))
    assert not (tmp_path / 'infer-v2-005').exists()


def test_submit_chain_dependencies_with_stub_sbatch(tmp_path):
    code = Path(__file__).resolve().parents[3]
    scripts = code / 'fastfill/comparison'
    for name in ('isambard_dpo_cpu.sbatch', 'isambard_dpo_generate.sbatch', 'isambard_dpo_train.sbatch',
                 'isambard_dpo_infer_shard.sbatch', 'submit_dpo_stages.sh'):
        subprocess.run(['bash', '-n', str(scripts / name)], check=True)
    env = T.stub_sbatch(tmp_path)
    run = subprocess.run(['bash', str(scripts / 'submit_dpo_stages.sh'), str(code), '-', '7365682', '1'],
                         env=env, capture_output=True, text=True, check=True)
    log = [line.replace(f'{scripts}/', '').replace(f' {code} ', ' CODE ') for line in (tmp_path / 'log').read_text().splitlines()]
    root = '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013'
    kill = '--parsable --kill-on-invalid-dep=yes --dependency=afterok:'
    assert log == [
        '--parsable --hold isambard_dpo_cpu.sbatch CODE copy-plans',               # 101
        f'{kill}101:7365682 isambard_dpo_cpu.sbatch CODE positives',               # 102
        f'{kill}101 isambard_dpo_cpu.sbatch CODE merge-sft joint',                 # 103 (joint SFT left Slurm)
        f'{kill}102:103 isambard_dpo_generate.sbatch CODE joint',                  # 104
        f'{kill}101:7365682 isambard_dpo_cpu.sbatch CODE merge-sft placement',     # 105
        f'{kill}102:105 isambard_dpo_generate.sbatch CODE placement',              # 106
        f'{kill}104:106 isambard_dpo_cpu.sbatch CODE stage1-pairs',                # 107
        f'{kill}102 isambard_dpo_cpu.sbatch CODE stage2-pairs',                    # 108
        f'{kill}107 isambard_dpo_train.sbatch CODE joint dpo1',                    # 109
        f'{kill}109:108 isambard_dpo_train.sbatch CODE joint dpo2',                # 110
        f'{kill}107 isambard_dpo_train.sbatch CODE placement dpo1',                # 111
        f'{kill}111:108 isambard_dpo_train.sbatch CODE placement dpo2',            # 112
        f'{kill}109 isambard_dpo_infer_shard.sbatch CODE dpo1 joint 0 1 {root}/infer-v2-003',
        f'{kill}113 isambard_infer_merge.sbatch CODE joint 1 {root}/infer-v2-003 {root}/infer-joint-003',
        f'{kill}111 isambard_dpo_infer_shard.sbatch CODE dpo1 placement 0 1 {root}/infer-v2-003',
        f'{kill}115 isambard_infer_merge.sbatch CODE placement 1 {root}/infer-v2-003 {root}/infer-placement-003',
        f'{kill}114:116 isambard_geometry.sbatch CODE 003 geometry-v3',
        f'{kill}110 isambard_dpo_infer_shard.sbatch CODE dpo2 joint 0 1 {root}/infer-v2-004',
        f'{kill}118 isambard_infer_merge.sbatch CODE joint 1 {root}/infer-v2-004 {root}/infer-joint-004',
        f'{kill}112 isambard_dpo_infer_shard.sbatch CODE dpo2 placement 0 1 {root}/infer-v2-004',
        f'{kill}120 isambard_infer_merge.sbatch CODE placement 1 {root}/infer-v2-004 {root}/infer-placement-004',
        f'{kill}119:121 isambard_geometry.sbatch CODE 004 geometry-v4']
    assert run.stdout.splitlines()[-1] == 'released plans 101'
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101\n'
    bad = subprocess.run(['bash', str(scripts / 'submit_dpo_stages.sh'), str(code), 'x', '-'], env=env)
    assert bad.returncode != 0 and len((tmp_path / 'log').read_text().splitlines()) == 22
    # Amendment 009: 4 GPUs of one node for torchrun DDP within 12 h (inner timeout below it), per-arm compile
    # caches for concurrent generate jobs, exact-policy shards tied to their stage's result root.
    train = (scripts / 'isambard_dpo_train.sbatch').read_text()
    assert '#SBATCH --nodes=1\n' in train and '#SBATCH --gpus-per-node=4\n' in train and '#SBATCH --time=12:00:00\n' in train
    assert 'timeout --signal=TERM --kill-after=120s 705m' in train and 'test "$GPUS" -eq 4' in train
    generate = (scripts / 'isambard_dpo_generate.sbatch').read_text()
    assert all(f'$CACHE/dpo-generate/$ARM/{c}' in generate for c in ('vllm', 'torchinductor', 'triton'))
    infer = (scripts / 'isambard_dpo_infer_shard.sbatch').read_text()
    assert '--model "$MODEL" --exact-policy' in infer
    for stage, tag, ok in (('dpo1', '003', True), ('dpo2', '004', True), ('dpo1', '004', False), ('dpo2', '003', False)):
        shard = subprocess.run(['bash', str(scripts / 'isambard_dpo_infer_shard.sbatch'), str(code), stage, 'joint',
                                '0', '4', f'{root}/infer-v2-{tag}'], env=env, capture_output=True, text=True)
        # Cross-wired: exit 2 at the case check. Matched: passes it and stops at the missing Slurm allocation.
        assert (shard.returncode, 'Slurm allocation required' in shard.stderr) == ((1, True) if ok else (2, False))


def test_submit_aborts_on_failed_sbatch_and_lists_submitted_jobs(tmp_path):
    scripts = Path(__file__).resolve().parents[3] / 'fastfill/comparison'
    env = T.stub_sbatch(tmp_path, fail_on=4)
    run = subprocess.run(['bash', str(scripts / 'submit_dpo_stages.sh'), str(scripts.parents[1]), '1', '2'],
                         env=env, capture_output=True, text=True)
    assert run.returncode == 1 and len((tmp_path / 'log').read_text().splitlines()) == 4
    assert "sbatch failed for 'generate joint'" in run.stderr and 'scancel 101 102 103' in run.stderr
    assert not (tmp_path / 'log.scontrol').exists()
