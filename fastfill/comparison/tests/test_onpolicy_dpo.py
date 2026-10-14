"""Amendment 013 on-policy DPO2 on CPU: fakes stand in for vLLM, train_dpo, merge_lora, the models and Slurm."""
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import types
from unittest.mock import patch

import pytest

from fastfill import scene
from fastfill.comparison import contract
from fastfill.comparison import dev_select as ds
from fastfill.comparison import downstream as dn
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import geometry as g
from fastfill.comparison import infer
from fastfill.comparison import infer_shard as s
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import output_repair as rep
from fastfill.comparison import sample_vllm as sv
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.tests import test_dev_select as DS
from fastfill.comparison.tests import test_dpo_stages as D
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests import test_sample_vllm as V
from fastfill.comparison.tests.test_dev_select import dev  # noqa: F401  Fixture: 5 frozen dev rooms.
from fastfill.comparison.tests.test_dpo_stages import pipeline  # noqa: F401  Fixture: dpo-001 on CPU.

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
KILL = ['--kill-on-invalid-dep=yes']


def ns(**kw):
    return types.SimpleNamespace(**kw)


# ---- subset --------------------------------------------------------------------------------------

def held(root, uids, groups=None):
    """A verified eval-layout stand-in: rooms.jsonl (uid, group) and expected_uids."""
    root.mkdir(parents=True)
    (root / 'rooms.jsonl').write_bytes(d._jsonl({'uid': u, 'group': (groups or {}).get(u, 'held-' + u)} for u in uids))
    h.write_json(root / 'MANIFEST.json', {'expected_uids': list(uids), 'files': h.artifact_manifest(root)})
    return root


@pytest.fixture
def main(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'TARGETS', D.TARGETS)
    data = D.build_main(tmp_path)
    runs = D.build_sft(tmp_path, data)
    positives = tmp_path / 'dpo-001/positives'
    d.positives(D.args(data=data, sft_runs=runs, out=positives, workers=1))
    return ns(root=tmp_path, data=data, runs=runs, positives=positives,
              dev=[held(tmp_path / 'dev-select-v1', ['dev0']), held(tmp_path / 'dev-full-v1', ['dev0', 'dev1'])],
              test=[held(tmp_path / 'eval-data-v1', ['t0'])])


def subset_args(w, out, rooms=2, dev_dirs=None, test_dirs=None):
    return op.parse_args(['subset', '--data', str(w.data), '--sft-runs', *map(str, w.runs), '--exclude', str(w.positives),
                          '--dev', *map(str, dev_dirs or w.dev), '--test', *map(str, test_dirs or w.test),
                          '--out', str(out), '--rooms', str(rooms)])


def test_pick_quotas_are_proportional_largest_remainder_and_order_free():
    pool = {f'a{k}': 'A' for k in range(6)} | {f'b{k}': 'B' for k in range(3)} | {'c0': 'C'}
    uids, info = op.pick(pool, 5)
    # 5 x (6, 3, 1) / 10 = 3, 1.5, 0.5: one remainder seat, B and C tie on it and B wins by name.
    assert info == {'cells': {'A': 6, 'B': 3, 'C': 1}, 'quotas': {'A': 3, 'B': 2, 'C': 0}, 'shortfall': 0}
    ranked = lambda cell: sorted((u for u in pool if pool[u] == cell), key=op.rank)
    assert uids == sorted(ranked('A')[:3] + ranked('B')[:2])
    assert op.pick(dict(reversed(list(pool.items()))), 5) == (uids, info)
    assert op.pick(pool, 20) == (sorted(pool), {**info, 'quotas': {'A': 6, 'B': 3, 'C': 1}, 'shortfall': 10})
    assert op.rank('u') == __import__('hashlib').sha256(b'op-dpo2-v1:u').hexdigest()


def test_subset_is_train_only_positive_free_eval_layout_and_sampleable(main, monkeypatch):
    out = main.root / 'dpo-004/train-subset'
    result = op.subset(subset_args(main, out))
    manifest = h.strict_json((out / 'MANIFEST.json').read_text())
    positives = set(d.load_positives(main.positives)[0]['uids'])
    pool = sorted(set(D.ADMITTED) - positives)  # h3 is not length-admitted; 5 positives leave 3 rooms.
    assert len(pool) == 3 and result['rooms'] == len(manifest['expected_uids']) == 2 and result['shortfall'] == 0
    rooms = {r['uid']: r for r in D.ROOMS}
    cells = {u: f"{op.bo.bin_name(len(rooms[u]['objects']))}|{rooms[u]['source']}" for u in pool}
    assert manifest['expected_uids'] == op.pick(cells, 2)[0] and not set(manifest['expected_uids']) & positives
    assert (manifest['scope'], manifest['split'], manifest['counts']) == (op.SUBSET_SCOPE, 'train', {'train': 2})
    assert manifest['training_manifest_sha256'] == manifest['main_data_manifest_sha256'] == h.sha256(main.data / 'MANIFEST.json')
    prompt_tokens = d._admitted(main.runs, h.load_pair(main.data)[1])[1]  # The SFT run's P prompt token counts.
    assert manifest['sft_prompt_tokens'] == {u: prompt_tokens[u]['placement'] for u in manifest['expected_uids']}
    selection = manifest['selection']
    assert selection['pool'] == {'length_admitted_train': 8, 'excluded_positives': 5, 'eligible': 3}
    assert selection['excluded_positives_manifest_sha256'] == h.sha256(main.positives / 'MANIFEST.json')
    assert [x['path'] for x in selection['held_out']['dev']] == [str(p.resolve()) for p in main.dev]
    train = {a: h._by_uid([json.loads(x) for x in (main.data / a / 'train.jsonl').read_text().splitlines()]) for a in h.ARMS}
    for arm in h.ARMS:
        requests, info = infer.load_eval(out, arm)  # The unchanged eval loader and request validator.
        assert info['expected_uids'] == manifest['expected_uids']
        for r in requests:
            assert r['messages'] == train[arm][r['uid']]['messages'][:2] and r['split'] == 'train'
            assert r['group'] == r['component'] == train[arm][r['uid']]['group']
    sizes = h._by_uid(h._rows(out, 'size-estimates.jsonl', manifest['files']['size-estimates.jsonl']))
    for uid, value in sizes.items():
        request = json.loads(train['placement'][uid]['messages'][1]['content'])
        assert value['estimated_sizes'] == {o['id']: o['size'] for o in request['objects']}
    op.subset(subset_args(main, main.root / 'again'))  # Deterministic.
    assert h.strict_json((main.root / 'again/MANIFEST.json').read_text())['expected_uids'] == manifest['expected_uids']
    with pytest.raises(FileExistsError):
        op.subset(subset_args(main, out))
    leak = held(main.root / 'leak-dev', ['dev0', pool[0]])
    with pytest.raises(ValueError, match='share a UID or house group'):
        op.subset(subset_args(main, main.root / 'x', dev_dirs=[leak]))
    group = held(main.root / 'leak-test', ['t9'], {'t9': rooms[pool[1]]['group']})  # Same house, other UID.
    with pytest.raises(ValueError, match='share a UID or house group'):
        op.subset(subset_args(main, main.root / 'y', test_dirs=[group]))
    assert not (main.root / 'x').exists() and not (main.root / 'y').exists()
    # sample_vllm runs and finalizes on the subset unchanged (rpo-dpo1's exact policy, patched).
    monkeypatch.setattr(sv.d, 'inference_identity', lambda run: {
        'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': manifest['training_manifest_sha256']},
        'stage': 'dpo1', 'lineage_sha256': '0' * 64, 'policy': POLICY})
    monkeypatch.setattr(sv.d, '_model_tokenizer', lambda path: V.Tok())
    sv.main(V.argv(main.root, out, 'placement', 'samples'), V.loader(V.FakeLLM(out)))
    assert V.finalize(main.root, out)['status'] == 'SAMPLING_COMPLETE'


# ---- pairs ---------------------------------------------------------------------------------------

POLICY = {'mode': 'unmerged_fp32_adapter_on_bf16_training_base', 'training_base': '/run/dpo-001/placement/sft-merged',
          'adapter': '/run/dpo-003/placement/dpo1/final'}
SIX = (.6, .6, .8)
SPEC = {'a': [('cabinet', SIX), ('cabinet', SIX), ('sofa', SIX)],
        'b': [('cabinet', (.605, .6, .8)), ('sofa', SIX)],
        'c': [('cabinet', SIX)],
        'd': [('cabinet', SIX), ('cabinet', SIX), ('sofa', SIX), ('wardrobe', SIX)]}


class Tok(D.Tok):
    """D.Tok (prompt = user JSON, one token per character, the EOS text one token) plus sample_vllm's vocab/decode."""
    def get_vocab(self):
        return {d.EOS: D.EOS_ID}

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


def make_subset(root, spec, training='a' * 64, sft_shift=0, scope=op.SUBSET_SCOPE):
    """A train-subset in the eval layout: 7 x 7 m rooms, GT sizes as the P prior; SFT prompt counts = Tok's + sft_shift
    (None: no SFT counts, as a dev data dir)."""
    data, rooms = root / 'train-subset', []
    for uid, objects in spec.items():
        rooms.append(bind_room({'uid': uid, 'source': 'S0' if uid < 'c' else 'S1', 'group': 'g-' + uid,
                                'room_type': 'bedroom', 'boundary_type': 'polygon',
                                'boundary': [[0, 0], [7, 0], [7, 7], [0, 7]], 'height': 3, 'fixed': [],
                                'objects': [{'id': f'o{j}', 'category': c, 'size': list(size), 'pos': [1 + 1.5 * j, 1, 0],
                                             'yaw': 0} for j, (c, size) in enumerate(objects)]})[0])
    files = {'rooms.jsonl': rooms, 'size-estimates.jsonl': [], 'joint/requests.jsonl': [], 'placement/requests.jsonl': []}
    for room in rooms:
        sizes = {o['id']: o['size'] for o in room['objects']}
        files['size-estimates.jsonl'].append({'uid': room['uid'], 'estimated_sizes': sizes})
        for arm in h.ARMS:
            files[f'{arm}/requests.jsonl'].append({**{k: room[k] for k in ('uid', 'source', 'group')}, 'split': 'train',
                                                   'messages': messages(room, arm, sizes if arm == 'placement' else None)[:2]})
    for name, rows in files.items():
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_bytes(d._jsonl(rows))
    sft = {} if sft_shift is None else {'sft_prompt_tokens': {
        r['uid']: len(r['messages'][1]['content']) + sft_shift for r in files['placement/requests.jsonl']}}
    h.write_json(data / 'MANIFEST.json', {'scope': scope, 'expected_uids': [r['uid'] for r in rooms],
                                          'training_manifest_sha256': training, **sft, 'files': h.artifact_manifest(data)})
    return data


def sampled(data, model):
    """sample_vllm run + finalize of the fake rpo-dpo1 policy on DATA -> DATA's sibling samples dir."""
    samples = data.with_name('samples')
    sv.main(['run', '--data', str(data), '--base', str(data.parent / 'base'), '--arm', 'placement', '--model', str(model),
             '--exact-policy', '--out', str(samples)], V.loader(V.FakeLLM(data, script)))
    sv.finalize(sv.parse_args(['finalize', '--data', str(data), '--out', str(samples)]))
    return samples


def text(request, positions):
    return contract._json({'placements': [{'id': o['id'], 'pos': list(p), 'yaw': 0}
                                          for o, p in zip(request['objects'], positions)]})


def nth(request, category, k=0):
    return [j for j, o in enumerate(request['objects']) if o['category'] == category][k]


def script(request, uid, k):
    """a: colocated cabinets every k. b: k = 0 cabinet/sofa collision, else spaced. c: k = 0 floats under the ceiling
    (not XY fixable), k = 1 capped, k = 2 not JSON, k >= 3 = k = 0. d: k = 0, 6, 7, 8 spaced; 1, 4 out of the room;
    2, 5 sofa/wardrobe collisions; 3 colocated cabinets."""
    pos = [[1 + 1.5 * j, 1, 0] for j in range(len(request['objects']))]
    finish = 'stop'
    if uid == 'a':
        pos[nth(request, 'cabinet', 1)] = list(pos[nth(request, 'cabinet')])
    elif uid == 'b' and k == 0:
        pos[nth(request, 'sofa')] = [pos[nth(request, 'cabinet')][0] + .3, 1, 0]
    elif uid == 'c' and k == 1:
        finish = 'length'
    elif uid == 'c' and k == 2:
        return 'not json', 'stop'
    elif uid == 'c':
        pos[0] = [1, 1, 2.5]
    elif uid == 'd' and k in (1, 4):
        pos[3 if k == 1 else 0] = [7.2, 1, 0] if k == 1 else [1, 7.2, 0]
    elif uid == 'd' and k in (2, 5):
        pos[nth(request, 'wardrobe')] = [pos[nth(request, 'sofa')][0] + (.3 if k == 2 else .35), 1, 0]
    elif uid == 'd' and k == 3:
        pos[nth(request, 'cabinet', 1)] = list(pos[nth(request, 'cabinet')])
    return text(request, pos), finish


def rounding_repair(task):
    """Room b's S1 'fix' leaves 9.9 mm of overlap (under downstream's 1 cm); at cm precision it is 1.25 cm (A4)."""
    row, room, *_ = task
    result = REAL_REPAIR(task)
    if room['uid'] != 'b':
        return result
    layout = copy.deepcopy(row['raw_layout'])
    cabinet = next(o for o in layout if o['category'] == 'cabinet')  # 0.605 wide: right edge at x + 0.3025.
    next(o for o in layout if o['category'] == 'sofa')['pos'][0] = cabinet['pos'][0] + .5926
    return {**result, 's1_layout': layout, 's1': {**result['s1'], 'status': 'ok'}}


REAL_REPAIR = rep.repair_row


@pytest.fixture
def world(tmp_path, monkeypatch):
    data = make_subset(tmp_path, SPEC)
    model = tmp_path / 'dpo-003/placement/dpo1-merged'
    model.parent.mkdir(parents=True)
    d.receipt_path(model).write_text('{"stage": "dpo1"}\n')
    monkeypatch.setattr(d, 'inference_identity', lambda run: {
        'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': 'a' * 64}, 'stage': 'dpo1',
        'lineage_sha256': h.sha256(d.receipt_path(model.resolve())), 'policy': POLICY})
    monkeypatch.setattr(d, 'exact_policy', lambda path, arm: ({'stage': 'dpo1', 'training': {'candidate': 'rpo'}}, POLICY))
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: Tok())
    monkeypatch.setattr(op.rep, 'repair_row', rounding_repair)
    monkeypatch.setattr(op, 'MIN_PAIRS', 6)
    monkeypatch.setattr(op, 'MIN_ROOMS', 3)
    return ns(root=tmp_path, data=data, model=model, samples=sampled(data, model))


def pair_args(w, out, data=None, samples=None, dry=None):
    return op.parse_args(['pairs', '--data', str(data or w.data), '--samples', str(samples or w.samples),
                          '--base-model', str(w.model), '--out', str(out), *(['--dry-run', str(dry)] if dry else [])])


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def test_pairs_s1_chosen_fallback_tiers_cap_and_ledger(world):
    out = world.root / 'stage2-pairs'
    result = op.build_pairs(pair_args(world, out))  # Includes dpo_stages.preflight: train_dpo admits every pair.
    rows, summary = lines(out / 'placement.jsonl'), json.loads((out / 'summary.json').read_text())
    ledger = {x['uid']: x for x in lines(out / 'ledger.jsonl')}
    assert set(d._verified_dir(out)) == {'placement.jsonl', 'ledger.jsonl', 'summary.json'}
    assert [r['uid'] for r in rows] == ['a', 'b', 'd', 'd', 'd', 'd'] and result['pairs'] == 6
    manifest = e_verified(world.data)
    rooms = h._by_uid(h._rows(world.data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    requests = h._by_uid(h._rows(world.data, 'placement/requests.jsonl', manifest['files']['placement/requests.jsonl']))
    for r in rows:  # train_dpo reads prompt/chosen/rejected; the chosen always passes both gates, V = 0 < V(rejected).
        assert r['prompt'] == requests[r['uid']]['messages'] and r['reject_kind'] == 'onpolicy' and r['chosen'] != r['rejected']
        row = g.score(rooms[r['uid']], r['prompt'], r['chosen'], 'placement')
        assert row['geometry_success'] and dn.accept(rep.predicted_room(rooms[r['uid']], r['prompt'], row['raw_layout']),
                                                     row['raw_layout'])['accepted']
        assert set(r['violations']['chosen'].values()) == {0} and r['downstream']['chosen'] == []
        assert sum(r['violations']['rejected'].values()) + len(r['downstream']['rejected']) > 0
        assert r['tokens'] == {'prompt': len(r['prompt'][1]['content']), 'chosen': len(r['chosen']) + 1,
                               'rejected': len(r['rejected']) + 1}
    a = rows[0]  # Colocated duplicate fixed by its own S1 repair: positions only, cm precision, canonical, no size.
    assert (a['chosen_from'], a['chosen_k'], a['rejected_k'], a['rejected_tier']) == ('s1_same_sample', 0, 0, 'colocated_duplicate')
    chosen, rejected = json.loads(a['chosen'])['placements'], json.loads(a['rejected'])['placements']
    assert contract._json({'placements': chosen}) == a['chosen'] and all('size' not in p for p in chosen)
    assert [{k: v for k, v in p.items() if k != 'pos'} for p in chosen] == [{k: v for k, v in p.items() if k != 'pos'} for p in rejected]
    assert all(x == scene.num(x) for p in chosen for x in p['pos']) and [p['pos'][2] for p in chosen] == [p['pos'][2] for p in rejected]
    assert a['s1']['status'] == 'ok' and a['s1']['moved'] == 1 and a['s1']['moved_median_m'] > .5
    assert ledger['a']['candidates'] == [{'rejected_k': 0, 'outcome': 'pair', 'chosen_from': 's1_same_sample', 'chosen_k': 0}] + [
        {'rejected_k': k, 'outcome': 'duplicate_rejected_text'} for k in range(1, 9)]  # Identical raws collapse.
    b = rows[1]  # cm rounding undid a mm-scale S1 fix: the best passing sample instead (lowest k).
    assert (b['chosen_from'], b['chosen_k'], b['rejected_k'], b['rejected_tier'], b['s1']) == (
        'best_passing_sample', 1, 0, 'main_collision', None)
    assert ledger['b']['candidates'] == [{'rejected_k': 0, 's1_rejected': 'rounding_broke', 'outcome': 'pair',
                                          'chosen_from': 'best_passing_sample', 'chosen_k': 1}]
    c = ledger['c']  # Nothing passes and S1 cannot fix a ceiling/support failure: no pair, never the GT.
    assert [x['class'] for x in c['samples']] == ['reject', 'incomplete', 'incomplete'] + ['reject'] * 6
    assert [x['finish_reason'] for x in c['samples'][:3]] == ['stop', 'length', 'stop']  # Cap hits never count.
    assert c['pairs'] == 0 and c['candidates'][0]['outcome'] == 'no_chosen' and c['candidates'][0]['s1_rejected'].startswith('s1_')
    assert c['samples'][0]['s1']['status'] in ('partial', 'unchanged') and c['samples'][1]['s1'] is None
    assert [(r['rejected_k'], r['rejected_tier']) for r in rows[2:]] == [  # Tier, then k; at most 4 per room.
        (3, 'colocated_duplicate'), (2, 'main_collision'), (5, 'main_collision'), (1, 'other')]
    assert ledger['d']['candidates'][-1]['rejected_k'] == 4 and ledger['d']['candidates'][-1]['outcome'] == 'room_cap'
    assert all(r['chosen_from'] == 's1_same_sample' or r['chosen_k'] == 0 for r in rows[2:])  # Greedy passes in d.
    assert summary['pairs'] == 6 and summary['rooms_with_pair'] == 3 and summary['pairs_per_room'] == {'0': 1, '1': 2, '4': 1}
    assert summary['drop_reasons'] == {'duplicate_rejected_text': 14, 'no_chosen': 1, 'room_cap': 1}
    assert summary['inputs']['reference_policy'] == POLICY and summary['samples']['incomplete'] == 2
    assert summary['inputs']['reference_lineage_sha256'] == h.sha256(d.receipt_path(world.model.resolve()))
    with pytest.raises(FileExistsError):
        op.build_pairs(pair_args(world, out))


def e_verified(root):
    return op.e._verified(Path(root))


def test_pairs_refuse_bindings_drift_and_floors_and_drop_overlength(world, monkeypatch):
    out = world.root / 'pairs'
    other = make_subset(world.root / 'other', SPEC, training='c' * 64)  # Same rooms, another training manifest.
    with pytest.raises(ValueError, match='not a complete sampling run'):
        op.build_pairs(pair_args(world, out, other))
    with monkeypatch.context() as m:
        m.setattr(d, 'exact_policy', lambda path, arm: ({'stage': 'dpo1', 'training': {'candidate': 'rpo'}},
                                                        {**POLICY, 'adapter': '/elsewhere'}))
        with pytest.raises(ValueError, match="rpo-dpo1's exact policy"):
            op.build_pairs(pair_args(world, out))
        m.setattr(d, 'exact_policy', lambda path, arm: ({'stage': 'dpo1', 'training': {'candidate': 'e1'}}, POLICY))
        with pytest.raises(ValueError, match="rpo-dpo1's exact policy"):
            op.build_pairs(pair_args(world, out))

    class Drift(Tok):
        def apply_chat_template(self, msgs, **kwargs):
            return super().apply_chat_template(msgs, **kwargs) + ' '
    with monkeypatch.context() as m:
        m.setattr(d, '_model_tokenizer', lambda path: Drift())
        with pytest.raises(ValueError, match='template drift'):
            op.build_pairs(pair_args(world, out))
    with monkeypatch.context() as m:
        m.setattr(op, 'MIN_PAIRS', 7)
        with pytest.raises(ValueError, match='too few on-policy pairs: 6 pairs in 3 rooms'):
            op.build_pairs(pair_args(world, out))
        m.setattr(op, 'MIN_PAIRS', 6)
        m.setattr(op, 'MIN_ROOMS', 4)
        with pytest.raises(ValueError, match='too few on-policy pairs'):
            op.build_pairs(pair_args(world, out))
    assert not out.exists()
    with monkeypatch.context() as m:  # verify_pair's length rule: nothing over MAX_LEN reaches train_dpo.
        m.setattr(d, 'MAX_LEN', 200)
        m.setattr(op, 'MIN_PAIRS', 0)
        m.setattr(op, 'MIN_ROOMS', 0)
        op.build_pairs(pair_args(world, out))
    summary = json.loads((out / 'summary.json').read_text())
    kept = lines(out / 'placement.jsonl') if (out / 'placement.jsonl').read_text() else []
    assert summary['drop_reasons']['overlength'] > 0 and len(kept) == summary['pairs']
    assert all(r['tokens']['prompt'] + max(r['tokens']['chosen'], r['tokens']['rejected']) <= 200 for r in kept)


def test_pairs_bind_sft_prompt_tokens_and_the_sampled_training_manifest(world):
    shifted = make_subset(world.root / 'shifted', SPEC, sft_shift=1)  # vLLM saw these prompts; SFT trained on others.
    with pytest.raises(ValueError, match='differently from SFT training'):
        op.build_pairs(pair_args(world, world.root / 'p1', shifted, sampled(shifted, world.model)))
    samples = world.root / 'foreign-samples'  # Same data manifest, a model trained on another training manifest.
    shutil.copytree(world.samples, samples)
    receipt = json.loads((samples / 'run_receipt.json').read_text())
    receipt['model']['training_data']['manifest_sha256'] = 'c' * 64
    receipt['run_sha256'] = infer._hash({k: v for k, v in receipt.items() if k != 'run_sha256'})
    h.write_json(samples / 'run_receipt.json', receipt)
    manifest = json.loads((samples / 'MANIFEST.json').read_text())
    manifest['files']['run_receipt.json'] = h.sha256(samples / 'run_receipt.json')
    h.write_json(samples / 'MANIFEST.json', manifest)
    with pytest.raises(ValueError, match='trained on another training manifest'):
        op.build_pairs(pair_args(world, world.root / 'p2', samples=samples))
    assert not (world.root / 'p1').exists() and not (world.root / 'p2').exists()


def test_pairs_dry_run_projects_the_yield_of_other_samples(world, monkeypatch):
    dev = make_subset(world.root / 'dev', SPEC, sft_shift=None, scope='dev')  # As dev-full-v1: no SFT counts.
    samples = sampled(dev, world.model)
    with pytest.raises(ValueError, match='training subset only'):
        op.build_pairs(pair_args(world, world.root / 'real', dev, samples))
    out = world.root / 'op-yield-013'
    result = op.build_pairs(pair_args(world, out, dev, samples, dry=2))
    summary, ledger = json.loads((out / 'summary.json').read_text()), lines(out / 'ledger.jsonl')
    assert set(d._verified_dir(out)) == {'ledger.jsonl', 'summary.json'}  # No pairs file, nothing trains on it.
    chosen = sorted(sorted(SPEC, key=op.rank)[:2])
    per_room = {'a': 1, 'b': 1, 'c': 0, 'd': 4}  # test_pairs_s1_chosen_fallback_tiers_cap_and_ledger's full-run yield.
    pairs, with_pair = sum(per_room[u] for u in chosen), sum(per_room[u] > 0 for u in chosen)
    assert [x['uid'] for x in ledger] == chosen and (summary['rooms'], summary['pairs']) == (2, pairs)
    assert summary['status'] == result['status'] == 'OP_YIELD_DRY_RUN' and summary['projection'] == {
        'rooms': op.ROOMS, 'pairs': pairs * op.ROOMS / 2, 'rooms_with_pair': with_pair * op.ROOMS / 2,
        'meets_floors': True}
    monkeypatch.setattr(op, 'ROOMS', 2)  # Projected to 2 rooms: under the fixture's floors (6 pairs, 3 rooms).
    projection = op.build_pairs(pair_args(world, world.root / 'small', dev, samples, dry=2))['projection']
    assert projection == {'rooms': 2, 'pairs': pairs, 'rooms_with_pair': with_pair, 'meets_floors': False}


def test_s1_chosen_guards_and_canonical_raw(world):
    manifest = e_verified(world.data)
    room = h._by_uid(h._rows(world.data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))['a']
    request = h._by_uid(h._rows(world.data, 'placement/requests.jsonl', manifest['files']['placement/requests.jsonl']))['a']
    prior = {o['id']: o['size'] for o in room['objects']}
    raw = script(json.loads(request['messages'][1]['content']), 'a', 0)[0]
    value = json.loads(raw)
    op.guard(raw, raw, room, request)
    moved = copy.deepcopy(value)
    moved['placements'][0]['pos'][0] += 1
    op.guard(contract._json(moved), raw, room, request)  # Positions may move.
    turned = copy.deepcopy(value)
    turned['placements'][0]['yaw'] = 90
    with pytest.raises(ValueError, match='beyond positions'):
        op.guard(contract._json(turned), raw, room, request)
    reordered = {'placements': [{'pos': p['pos'], 'id': p['id'], 'yaw': p['yaw']} for p in value['placements']]}
    with pytest.raises(ValueError, match='beyond positions'):  # Key order is part of the text.
        op.guard(contract._json(reordered), raw, room, request)
    sized = {'placements': [{**p, 'size': [.6, .6, .8]} for p in value['placements']]}
    with pytest.raises(ValueError):  # The placement contract has no size field.
        op.guard(contract._json(sized), raw, room, request)
    sample = {'k': 0, 'raw': json.dumps(value), 'error': None, 'finish_reason': 'stop'}  # json.dumps adds spaces.
    row, compact, _ = op._sample(room, request, prior, sample)
    repaired = REAL_REPAIR((row, room, request, op.MAX_TRIALS, op.TIME_LIMIT))
    assert repaired['s1']['status'] == 'ok'
    assert op.s1_chosen(room, request, prior, sample, row, repaired, compact) == (None, 'non_canonical')
    assert op.s1_chosen(room, request, prior, sample, row, {**repaired, 's1': {'status': 'partial'}}, compact) == (
        None, 's1_partial')


# ---- train ---------------------------------------------------------------------------------------

def fake_train_epochs(calls, step_shift=0):
    """D.fake_train plus the per-epoch adapters train_dpo --save_every_epoch writes (each one distinct)."""
    inner = D.fake_train(calls)

    def run(command, check):
        inner(command, check)
        flags = dict(zip(command[7::2], command[8::2]))
        out = Path(flags['--out'])
        count = len(Path(flags['--pairs']).read_text().splitlines())
        per_epoch = math.ceil(math.ceil(count / d.WORLD_SIZE) / int(flags['--grad_accum']))
        for epoch in range(1, int(flags['--epochs']) + 1):
            target = out / f'epoch-{epoch:02d}'
            shutil.copytree(out / 'final', target, ignore=shutil.ignore_patterns('completion.json'))
            (target / 'adapter_model.safetensors').write_bytes(D.SAFETENSORS[:-1] + bytes([epoch]))
            (target / 'epoch.json').write_text(json.dumps({'epoch': epoch, 'step': epoch * per_epoch + step_shift}) + '\n')
    return run


def op_pairs(pipeline, extra=1, lineage=None):
    """dpo-004/stage2-pairs from dpo-001's placement stage-2 rows, one room repeated (several pairs per room)."""
    root = pipeline.root.parent / 'dpo-004'
    rows = D.pair_rows(pipeline.root, 'stage2-pairs', 'placement')
    parent = (pipeline.root.parent / 'dpo-003/placement/dpo1-merged').resolve()
    summary = {'status': 'OP_PAIRS_COMPLETE', 'inputs': {'reference_lineage_sha256': lineage or h.sha256(d.receipt_path(parent))}}
    d._publish_dir(root / 'stage2-pairs', {'placement.jsonl': d._jsonl(rows + rows[:extra]),
                                           'summary.json': op._json_bytes(summary)})
    return root, parent, len(rows) + extra


def test_train_op_dpo2_on_rpo_dpo1_with_epoch_merges_and_exact_policies(pipeline, monkeypatch):
    source, calls, commands = pipeline.root, [], []
    monkeypatch.setattr(d.subprocess, 'run', D.fake_train(calls))
    d.train(D.args(dpo_root=source.parent / 'dpo-003', arm='placement', stage='dpo1', candidate='rpo'))
    root, parent, count = op_pairs(pipeline)
    inner = fake_train_epochs(calls)
    monkeypatch.setattr(op.subprocess, 'run', lambda command, check: (commands.append(command), inner(command, check)))
    final = op.train(ns(dpo_root=root))
    flags, command = calls[-1], commands[-1]
    assert len(calls) == 2 and (flags['--epochs'], flags['--rpo_alpha'], command[-1]) == ('2', '1.0', '--save_every_epoch')
    assert {k: v for k, v in flags.items() if k not in ('--epochs', '--rpo_alpha', '--model', '--pairs', '--out')} == {
        '--' + k: str(v) for k, v in d.RECIPE.items() if k != 'epochs'}  # lr 5e-6, beta 0.1, r32/a16, bs 1 x 4 x 1, seed 42.
    assert (Path(flags['--model']), Path(flags['--pairs']), Path(flags['--out'])) == (
        parent, (root / 'stage2-pairs/placement.jsonl').resolve(), root / 'placement/dpo2')
    guard = h.strict_json((root / 'placement/dpo2/training_guard.json').read_text())
    assert guard['settings'] == {**op.RECIPE, **d.TRAIN_FIXED} and str(Path(op.__file__).resolve()) in guard['code']
    epoch = d.verify_merged(root / 'placement/dpo2-epoch01-merged', 'placement', 'dpo2')
    assert epoch['adapter']['path'] == str((root / 'placement/dpo2/epoch-01').resolve()) and epoch['training']['epoch'] == 1
    assert final['adapter']['path'] == str((root / 'placement/dpo2/final').resolve()) and 'epoch' not in final['training']
    assert final['training']['pairs']['count'] == count == 6 and final['training']['completion']['steps'] == 2 * 2
    assert final['training']['candidate'] == 'op' and final['parent'] == {
        'stage': 'dpo1', 'path': str(parent), 'lineage_sha256': h.sha256(d.receipt_path(parent))}
    first = d.verify_merged(parent, 'placement', 'dpo1')  # SFT + DPO1 + DPO2 lineage.
    assert first['training']['candidate'] == 'rpo' and first['parent']['path'] == str((source / 'placement/sft-merged').resolve())
    monkeypatch.setattr(d.h, 'base_receipt', lambda base: D.BASE)
    for name, adapter in (('dpo2-epoch01-merged', 'epoch-01'), ('dpo2-merged', 'final')):
        identity = d.inference_identity(ns(model=root / 'placement' / name, arm='placement', base=Path('/models/Qwen3-8B'),
                                           exact_policy=True))
        assert identity['stage'] == 'dpo2' and identity['policy']['training_base'] == str(parent)
        assert identity['policy']['adapter'] == str((root / 'placement/dpo2' / adapter).resolve())
    assert op.train(ns(dpo_root=root))['model_sha256'] == final['model_sha256'] and len(calls) == 2  # Rerun: verify only.
    epoch_receipt = d.receipt_path(root / 'placement/dpo2-epoch01-merged').read_bytes()
    shutil.rmtree(root / 'placement/dpo2-merged')  # Interrupted after the epoch merge, before the final one.
    d.receipt_path(root / 'placement/dpo2-merged').unlink()
    merges, real_merge = [], d._merge
    monkeypatch.setattr(d, '_merge', lambda *a: (merges.append(Path(a[2]).name), real_merge(*a))[1])
    again = op.train(ns(dpo_root=root))
    assert merges == ['dpo2-merged'] and len(calls) == 2 and again['adapter'] == final['adapter']  # No retraining.
    assert d.receipt_path(root / 'placement/dpo2-epoch01-merged').read_bytes() == epoch_receipt
    with pytest.raises(ValueError, match='trains in dpo-004'):
        op.train(ns(dpo_root=source.parent / 'dpo-003'))
    with pytest.raises(SystemExit):  # Only placement dpo2: no arm or stage to choose.
        op.parse_args(['train', '--dpo-root', str(root), '--arm', 'joint'])
    assert not (source / 'placement/dpo2').exists() and not (source.parent / 'dpo-003/placement/dpo2').exists()


def test_train_refuses_foreign_pairs_and_a_mislabelled_epoch(pipeline, monkeypatch):
    monkeypatch.setattr(d.subprocess, 'run', D.fake_train([]))
    d.train(D.args(dpo_root=pipeline.root.parent / 'dpo-003', arm='placement', stage='dpo1', candidate='rpo'))
    root, _, _ = op_pairs(pipeline, lineage='0' * 64)
    with pytest.raises(ValueError, match='not sampled from this DPO2 base'):
        op.train(ns(dpo_root=root))
    shutil.rmtree(root / 'stage2-pairs')
    op_pairs(pipeline)
    monkeypatch.setattr(op.subprocess, 'run', fake_train_epochs([], step_shift=1))
    with pytest.raises(ValueError, match='not the end of epoch 1'):
        op.train(ns(dpo_root=root))
    assert not (root / 'placement/dpo2-epoch01-merged').exists() and not (root / 'placement/dpo2-merged').exists()


# ---- dev score, select, merged check, test report -------------------------------------------------

def run_inference(root, data, plan, out, source, identity, kind):
    """Two placement shards on PLAN with a patched identity and fake runtime, merged into OUT (as DS.run_candidate)."""
    shards = []
    for k in range(2):
        shard = out.with_name(f'{out.name}-shard-{k:02d}')
        args = s.parse_args(['shard', '--data', str(data), '--base', str(root / 'base'), '--arm', 'placement',
                             '--plan', str(plan), '--shard-index', str(k), '--shard-count', '2', '--out', str(shard), *source])
        with patch('fastfill.comparison.dpo_stages.inference_identity', return_value=identity), \
                patch.object(s.i, '_runtime', return_value=(T.Tok(), DS.fake(kind), T.EOS)):
            assert s.shard(args)['status'] == 'SHARD_COMPLETE'
        shards.append(shard)
    s.merge(s.parse_args(['merge', '--data', str(data), '--plan', str(plan), '--arm', 'placement', '--out', str(out),
                          '--shards', *map(str, shards)]))
    return out


def identity(training, stage, merged=False):
    return {'base': {}, 'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': training}, 'stage': stage,
            'lineage_sha256': '0' * 64, **({'model_sha256': 'f' * 64} if merged else
                                           {'policy': {'mode': 'x', 'training_base': 'tb', 'adapter': 'ad'}})}


def run_op(w, candidate, kind, plan=None, out=None):
    merged = candidate.endswith('-merged')
    source = ['--model', str(op.model_path(w.root, candidate))] + ([] if merged else ['--exact-policy'])
    return run_inference(w.root, w.data, plan or w.plan, out or w.root / 'dev-eval-013' / candidate / 'infer-placement',
                         source, identity(w.training, 'dpo2', merged), kind)


def dev_world(dev, monkeypatch, rpo='collide', ops=('good', 'good')):
    """dev-eval-010 (sft + rpo-dpo1 dev runs and their scores) and the op checkpoints' dev runs on the same plan."""
    root, training = dev.root, h.sha256(dev.data / 'MANIFEST.json')
    plan = DS.plan_for(dev.out, root / 'dev-eval-010/plan')
    for candidate, kind in (('sft', 'collide'), ('rpo-dpo1', rpo)):
        for arm in h.ARMS:
            DS.run_candidate(root, dev.out, plan, arm, candidate, kind, training)
    with monkeypatch.context() as m:
        m.setattr(ds, 'CANDIDATES', ('sft', 'rpo-dpo1'))
        ds.score(ns(data=dev.out, root=root / 'dev-eval-010', out=root / 'dev-eval-010/scores', workers=1, only='selectable'))
    w = ns(root=root, data=dev.out, plan=plan, training=training, eval=root / 'dev-eval-013')
    for candidate, kind in zip(op.CANDIDATES, ops):
        run_op(w, candidate, kind)
    return w


def score_args(w, out=None):
    return ns(data=w.data, root=w.eval, ref_root=w.root / 'dev-eval-010', out=out or w.eval / 'scores', workers=1)


def fake_sbatch(log):
    def run(argv, capture_output, text, env):
        log.append(argv)
        return ns(returncode=0, stdout='' if argv[0] == 'scontrol' else f'{100 + len(log)};cluster\n', stderr='')
    return run


def test_dev_score_select_adopts_earlier_epoch_and_submits_test_chain(dev, monkeypatch):
    w = dev_world(dev, monkeypatch, ops=('good', 'good'))
    result = op.dev_score(score_args(w))
    scores = h.strict_json((w.eval / 'scores/scores.json').read_text())
    c = scores['candidates']
    assert set(c) == {'sft', 'rpo-dpo1', *op.CANDIDATES} and scores['plan_sha256'] == h.strict_json(
        (w.root / 'dev-eval-010/scores/scores.json').read_text())['plan_sha256']
    rooms = len(h.strict_json((w.data / 'MANIFEST.json').read_text())['expected_uids'])
    assert c['rpo-dpo1']['a_prime'] == c['rpo-dpo1']['accepted_rate'] == 0 and c['rpo-dpo1']['complete_rate'] == 1
    for name in op.CANDIDATES:
        assert (c[name]['a_prime'], c[name]['accepted_rate'], c[name]['both_pass_rate'], c[name]['requests']) == (1, 1, 1, rooms)
        assert c[name]['halves'] == {'paper_sources': {'rooms': 3, 'a_prime': 1.0, 'accepted_rate': 1.0},
                                     'other_sources': {'rooms': 2, 'a_prime': 1.0, 'accepted_rate': 1.0}}
        assert c[name]['inference']['lora'] == ds.EXACT and c[name]['inference']['stage'] == 'dpo2'
        assert c[name]['model'] == str(op.model_path(w.root, name).resolve())
        contrast = scores['contrasts_vs_rpo_dpo1'][name]
        assert contrast['accepted']['estimate'] == contrast['success_debiased']['estimate'] == 1 and contrast['complete']['estimate'] == 0
    assert result['op-dpo2-e1'] == {'a_prime': 1.0, 'accepted_rate': 1.0, 'complete_rate': 1.0}
    rows = lines(w.eval / 'scores/placement-op-dpo2-e1-geometry.jsonl')
    assert len(rows) == rooms and all(r['accepted'] and r['success_debiased'] for r in rows)
    with pytest.raises(FileExistsError):
        op.dev_score(score_args(w))
    DS.plan_for(T.make_data(w.root), w.root / 'infer-v2-002/plan')  # The test plan the adopted chain copies.
    log, sel = [], w.eval / 'selection.json'
    held = op.select(ns(scores=w.eval / 'scores', out=sel, code_root=CODE, shards=2, no_submit=True), run=fake_sbatch(log),
                     which=lambda name: name)  # The chain's select: selection.json only.
    assert held == {'status': 'ADOPTED', 'selected': 'op-dpo2-e1'} and log == [] and not (w.root / 'infer-v2-006').exists()
    older = h.strict_json(sel.read_text())  # Written by an older snapshot: the later submit must still reproduce it.
    sel.unlink()
    h.write_json(sel, {**older, 'code_sha256': {'onpolicy_dpo.py': '0' * 64}})
    picked = op.select(ns(scores=w.eval / 'scores', out=sel, code_root=CODE, shards=2), run=fake_sbatch(log),
                       which=lambda name: name)  # submit_013.sh CODE test.
    assert picked['selected'] == 'op-dpo2-e1'  # Both eligible with equal a': the earlier epoch.
    value = h.strict_json(sel.read_text())
    assert value['adopted'] and value['eligible'] == list(op.CANDIDATES) == value['tied'] and value['rule'] == op.RULE
    assert value['test']['inference'] == str(w.root / 'infer-placement-006') and value['test']['geometry'] == str(w.root / 'geometry-v6')
    assert (w.root / 'infer-v2-006/plan/plan.json').read_bytes() == (w.root / 'infer-v2-002/plan/plan.json').read_bytes()
    S, run = str(SCRIPTS / 'isambard_013.sbatch'), str(w.root)
    shard = ['sbatch', '--parsable', '--hold', *op.SHARD, S, str(CODE), 'shard']
    assert log == [
        [*shard, 'test', 'op-dpo2-e1', '0', '2'], [*shard, 'test', 'op-dpo2-e1', '1', '2'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:101:102', str(SCRIPTS / 'isambard_infer_merge.sbatch'),
         str(CODE), 'placement', '2', f'{run}/infer-v2-006', f'{run}/infer-placement-006'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:103', S, str(CODE), 'test-geometry'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:103', '--time=06:00:00', S, str(CODE), 'test-report'],
        [*shard, 'dev', 'op-dpo2-e1-merged', '0', '2'], [*shard, 'dev', 'op-dpo2-e1-merged', '1', '2'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:106:107', S, str(CODE), 'dev-merge', 'op-dpo2-e1-merged', '2'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:108', S, str(CODE), 'check-merged'],
        ['scontrol', 'release', '101,102,106,107']]
    with pytest.raises(FileExistsError, match='already submitted'):
        op.select(ns(scores=w.eval / 'scores', out=sel, code_root=CODE, shards=2), run=fake_sbatch([]), which=lambda n: n)
    # Merged-weight check (amendment 011 section 5): the BF16 merge as is vs the exact policy on the same dev plan.
    run_op(w, 'op-dpo2-e1-merged', 'good')
    check = op.check_merged(ns(data=w.data, root=w.eval, ref_root=w.root / 'dev-eval-010', out=w.eval / 'merged-check', workers=1))
    value = h.strict_json((w.eval / 'merged-check/check.json').read_text())
    assert check['decision'] == value['decision'] == 'merged_convenience_artifact' and value['byte_identical']['fraction'] == 1
    assert value['merged']['inference']['lora'] == ds.PREMERGED and value['selection_sha256'] == h.sha256(sel)
    shutil.rmtree(w.eval / 'op-dpo2-e1-merged')  # A merge that changes the outputs: adapter + training base only.
    run_op(w, 'op-dpo2-e1-merged', 'collide')
    check = op.check_merged(ns(data=w.data, root=w.eval, ref_root=w.root / 'dev-eval-010', out=w.eval / 'mc-2', workers=1))
    value = h.strict_json((w.eval / 'mc-2/check.json').read_text())
    assert check['decision'] == value['decision'] == 'adapter_and_training_base_only' and not value['merged_listed_as_convenience']
    assert value['byte_identical']['rooms'] == 0 and check['delta']['a_prime'] == -1
    # The sbatch shard preflight: only the adopted checkpoint, by a selection.json bound to its dev scores.
    for argv, ok in ((['test', 'op-dpo2-e1'], True), (['test', 'op-dpo2-e2'], False), (['dev', 'op-dpo2-e1-merged'], True)):
        done = shard_preflight(w, argv)
        assert ('PREFLIGHT FAIL' not in done.stderr and 'module' in done.stderr) if ok else (
            f'PREFLIGHT FAIL: {argv[1]} is not the adopted op checkpoint' in done.stderr), done.stderr
    # Test report: P-op vs P-rpo-dpo1 on one shared plan (the dev rooms stand in for eval-data-v1 here).
    plan = DS.plan_for(w.data, w.root / 'test-plan')
    run_op(w, 'op-dpo2-e1', 'good', plan, w.root / 'infer-placement-006')
    run_inference(w.root, w.data, plan, w.root / 'infer-placement-005',
                  ['--model', str(w.root / 'dpo-003/placement/dpo1-merged'), '--exact-policy'],
                  identity(w.training, 'dpo1'), 'collide')
    report = op.test_report(ns(data=w.data, selection=sel, op=w.root / 'infer-placement-006',
                               ref=w.root / 'infer-placement-005', out=w.root / 'test-report-013', workers=1))
    value = h.strict_json((w.root / 'test-report-013/report.json').read_text())
    assert report['op-dpo2-e1'] == 1 and report['rpo-dpo1'] == 0 and value['selected'] == 'op-dpo2-e1'
    assert value['contrast_op_minus_rpo_dpo1']['accepted']['estimate'] == 1 and value['rooms'] == rooms
    with pytest.raises(ValueError, match='inference is not placement/op-dpo2-e1'):  # Swapped runs are refused.
        op.test_report(ns(data=w.data, selection=sel, op=w.root / 'infer-placement-005',
                          ref=w.root / 'infer-placement-006', out=w.root / 'test-report-x', workers=1))


def test_no_adoption_keeps_rpo_dpo1_and_submits_nothing(dev, monkeypatch):
    w = dev_world(dev, monkeypatch, rpo='good', ops=('collide', 'broken'))
    op.dev_score(score_args(w))
    c = h.strict_json((w.eval / 'scores/scores.json').read_text())['candidates']
    broken = c['op-dpo2-e2']  # Downstream acceptance over every room: incomplete rooms count as failures.
    assert broken['complete_rate'] == broken['accepted_rate'] == broken['a_prime'] == 0 and broken['requests'] == 5
    assert c['op-dpo2-e1']['complete_rate'] == 1 and c['op-dpo2-e1']['accepted_rate'] == 0
    rows = lines(w.eval / 'scores/placement-op-dpo2-e2-geometry.jsonl')
    assert len(rows) == 5 and not any(r['accepted'] or r['complete'] for r in rows)
    sel = w.eval / 'selection.json'
    args = ns(scores=w.eval / 'scores', out=sel, code_root=CODE, shards=2)
    def never(*a, **k):
        raise AssertionError('nothing may be submitted')
    result = op.select(args, run=never, which=lambda name: name)
    value = h.strict_json(sel.read_text())
    assert result['status'] == 'NOT_ADOPTED' and (value['selected'], value['adopted'], value['test']) == ('rpo-dpo1', False, None)
    assert op.select(args, run=never, which=lambda name: name)['status'] == 'NOT_ADOPTED'  # Idempotent, write-once.
    assert not (w.root / 'infer-v2-006').exists()
    with pytest.raises(ValueError, match='not adopted'):
        op.check_merged(ns(data=w.data, root=w.eval, ref_root=w.root / 'dev-eval-010', out=w.eval / 'mc', workers=1))
    sel.unlink()
    sel.write_text(json.dumps({**value, 'selected': 'op-dpo2-e1'}) + '\n')
    with pytest.raises(ValueError, match='exists with another selection'):
        op.select(args, run=never, which=lambda name: name)
    with pytest.raises(ValueError, match='not bound to its dev scores'):
        op.load_selection(sel)
    done = shard_preflight(w, ['dev', 'op-dpo2-e1-merged'])  # The hand-edited selection.json authorizes nothing.
    assert done.returncode == 1 and 'PREFLIGHT FAIL: op-dpo2-e1 is not the adopted op checkpoint' in done.stderr
    assert 'not bound to its dev scores' in done.stderr


def test_dev_score_refuses_plan_mismatch_and_rescoring_drift(dev, monkeypatch):
    w = dev_world(dev, monkeypatch)
    recheck = ns(data=w.data, ref_root=w.root / 'dev-eval-010', workers=1)
    assert op.recheck(recheck) == {'status': 'OP_RECHECK_COMPLETE', 'sft': {'a_prime': 0.0, 'complete_rate': 1.0},
                                   'rpo-dpo1': {'a_prime': 0.0, 'complete_rate': 1.0}}
    real = op._scored

    def other_plan(root, candidate, *rest):
        records, receipt, rows, info = real(root, candidate, *rest)
        return records, {**receipt, 'context_plan_sha256': 'f' * 64} if candidate == 'op-dpo2-e2' else receipt, rows, info
    with monkeypatch.context() as m:
        m.setattr(op, '_scored', other_plan)
        with pytest.raises(ValueError, match='shared dev plan'):
            op.dev_score(score_args(w))
    with monkeypatch.context() as m:  # dev-eval-010's scores of record must reproduce.
        finished = ds._finished
        m.setattr(ds, '_finished', lambda *a: (lambda v, f: ({**v, 'a_prime': v['a_prime'] - .5}, f))(*finished(*a)))
        with pytest.raises(ValueError, match='rescoring differs'):
            op.dev_score(score_args(w))
        with pytest.raises(ValueError, match='sft rescoring differs'):  # The chain's recheck fails before any GPU job.
            op.recheck(recheck)
    with pytest.raises(ValueError, match='inference is not placement/op-dpo2-e2'):
        op._bind(h.strict_json((w.eval / 'op-dpo2-e1/infer-placement/run_receipt.json').read_text()), w.root, 'op-dpo2-e2')
    with pytest.raises(ValueError, match='inference is not placement/op-dpo2-e1-merged'):  # Exact policy != merged bytes.
        op._bind(h.strict_json((w.eval / 'op-dpo2-e1/infer-placement/run_receipt.json').read_text()), w.root,
                 'op-dpo2-e1-merged')
    assert not (w.eval / 'scores').exists()


def m(complete, a, accepted, paper=None, other=None):
    return {'complete_rate': complete, 'a_prime': a, 'accepted_rate': accepted,
            'halves': {'paper_sources': {'a_prime': a if paper is None else paper},
                       'other_sources': {'a_prime': a if other is None else other}}}


def test_rule_inclusive_bounds_per_half_guard_and_epoch_tie_break():
    ref = m(.90, .50, .40)
    edge = {'rpo-dpo1': ref, 'op-dpo2-e1': m(.895, .51, .41), 'op-dpo2-e2': m(.90, .50, .60)}
    assert op.choose(edge) == {'selected': 'op-dpo2-e1', 'adopted': True, 'eligible': ['op-dpo2-e1'],
                               'tied': ['op-dpo2-e1'], 'best_a_prime': .51}  # Every bound inclusive; e2 lacks +0.01 a'.
    assert not op.choose({**edge, 'op-dpo2-e1': m(.894, .6, .6)})['adopted']  # Complete below R - 0.005.
    assert not op.choose({**edge, 'op-dpo2-e1': m(.9, .6, .409)})['adopted']  # Acceptance below R + 0.01.
    assert not op.choose({**edge, 'op-dpo2-e1': m(.9, .6, .6, paper=.49)})['adopted']  # One dev half got worse.
    assert op.choose({**edge, 'op-dpo2-e1': m(.9, .6, .6, other=.5)})['adopted']  # Equal in a half is allowed.
    near = {'rpo-dpo1': ref, 'op-dpo2-e1': m(.9, .600, .6), 'op-dpo2-e2': m(.9, .605, .6)}
    assert op.choose(near)['selected'] == 'op-dpo2-e1' and op.choose(near)['tied'] == list(op.CANDIDATES)  # Within 0.005.
    far = {**near, 'op-dpo2-e2': m(.9, .606, .6)}
    assert op.choose(far)['selected'] == 'op-dpo2-e2'
    assert op.choose({**near, 'op-dpo2-e1': m(.9, .7, .6)})['selected'] == 'op-dpo2-e1'
    with pytest.raises(ValueError, match='must be scored'):
        op.choose({'rpo-dpo1': ref, 'op-dpo2-e1': ref})


# ---- Slurm scripts -------------------------------------------------------------------------------

def python_stub(path):
    """PROJECT_ROOT/env/bin/python: this interpreter, or (PYLOG set) a recorder of its argv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\nif [ -n "$PYLOG" ]; then echo "$@" > "$PYLOG"; exit 0; fi\nexec {sys.executable} "$@"\n')
    path.chmod(0o755)


def shard_preflight(w, argv):
    """isambard_013.sbatch shard SET CAND 0 2 in an allocation, PROJECT_ROOT/runs/unknownsize-20261013 = w.root: it
    stops at the adopted() check, or after it at the cluster's module command."""
    project = w.root / 'project'
    if not project.exists():
        python_stub(project / 'env/bin/python')
        (project / 'runs').mkdir()
        (project / 'runs/unknownsize-20261013').symlink_to(w.root)
        for name in op.MODELS.values():
            (w.root / 'dpo-004/placement').mkdir(parents=True, exist_ok=True)
            d.receipt_path(w.root / 'dpo-004/placement' / name).write_text('{}')
        original = (SCRIPTS / 'isambard_013.sbatch').read_text()
        line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
        assert original.count(line) == 1
        (project / 'isambard_013.sbatch').write_text(original.replace(line, f'PROJECT_ROOT={project}\n'))
    return subprocess.run(['bash', str(project / 'isambard_013.sbatch'), str(CODE), 'shard', *argv, '0', '2'],
                          env={'PATH': '/usr/bin:/bin', 'SLURM_JOB_ID': '1'}, capture_output=True, text=True)


INPUTS = ('main-data-v1/MANIFEST.json', 'sft-joint-001/SHA256.json', 'sft-placement-001/SHA256.json',
          'dpo-001/positives/MANIFEST.json', 'dev-select-v1/MANIFEST.json', 'dev-full-v1/MANIFEST.json',
          'eval-data-v1/MANIFEST.json', 'dpo-003/placement/dpo1-merged.json', 'dev-eval-010/plan/MANIFEST.json',
          'dev-eval-010/scores/MANIFEST.json', 'dev-eval-010/selection.json',
          'dev-eval-010/sft/infer-placement/MANIFEST.json', 'dev-eval-010/rpo-dpo1/infer-placement/MANIFEST.json',
          'context-capacity-001/SHA256.json', 'infer-v2-002/plan/MANIFEST.json', 'infer-joint-002/MANIFEST.json',
          'infer-placement-005/MANIFEST.json')
DOC = 'docs/research/20261013-unknown-size-comparison/amendment-013-onpolicy-dpo2-20261014.md'


def code_root(path, doc=True):
    """A code snapshot: this fastfill plus (doc) the amendment-013 protocol doc."""
    path.mkdir()
    (path / 'fastfill').symlink_to(CODE / 'fastfill')
    if doc:
        (path / DOC).parent.mkdir(parents=True)
        (path / DOC).write_text('# Amendment 013\n')
    return path


def submit_script(tmp_path, meets_floors=True):
    """submit_013.sh with PROJECT_ROOT under tmp_path, every input present and the yield dry run's summary."""
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    for path in INPUTS:
        (run / path).parent.mkdir(parents=True, exist_ok=True)
        (run / path).write_text('{}')
    (project / 'models/Qwen3-8B').mkdir(parents=True)
    python_stub(project / 'env/bin/python')
    (run / 'op-yield-013').mkdir()
    (run / 'op-yield-013/summary.json').write_text(json.dumps(
        {'status': 'OP_YIELD_DRY_RUN', 'projection': {'meets_floors': meets_floors}}))
    script = tmp_path / 'submit_013.sh'
    original = (SCRIPTS / 'submit_013.sh').read_text()
    line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    assert original.count(line) == 1
    script.write_text(original.replace(line, f'PROJECT_ROOT={project}\n'))
    return script, run


def squeue(directory):
    (directory / 'squeue').write_text(
        '#!/bin/sh\ntest "$*" = "-h -u me -n unknownsize-s2-sample,unknownsize-s2-cpu,unknownsize-013 -o %i" || exit 3\n'
        'printf "%s" "$SQUEUE_OUT"\n')
    (directory / 'squeue').chmod(0o755)


def test_submit_013_chain_dependencies_held_root_and_refusals(tmp_path):
    for name in ('isambard_013.sbatch', 'submit_013.sh'):
        subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    squeue(tmp_path)
    script, run = submit_script(tmp_path)
    code = code_root(tmp_path / 'main-abc')
    sent = []

    def submit(*argv, **extra):
        """submit_013.sh CODE ARGV -> (process, the sbatch lines it added, normalized)."""
        before = len(sent)
        done = subprocess.run(['bash', str(script), str(code), *argv], env={**env, **extra}, capture_output=True, text=True)
        log = tmp_path / 'log'
        sent[:] = log.read_text().splitlines() if log.exists() else []
        return done, [x.replace(f'{code}/fastfill/comparison/', '').replace(f' {code} ', ' CODE ') for x in sent[before:]]
    done, log = submit('3', '2')
    assert done.returncode == 0, done.stderr
    kill, sample, shard = '--parsable --kill-on-invalid-dep=yes --dependency=afterok:', \
        '--gpus-per-node=1 --cpus-per-task=16 --mem=100G', ' '.join(op.SHARD)  # The selector's shard resources too.
    assert log == [
        '--parsable --hold isambard_013.sbatch CODE recheck',                                              # 101
        f'{kill}101 isambard_013.sbatch CODE subset',                                                       # 102
        f'{kill}102 {sample} --time=02:00:00 isambard_013.sbatch CODE sample 1 --limit 16',                 # 103
        f'{kill}103 {sample} --time=04:00:00 --array=0-2 isambard_013.sbatch CODE sample 3',                # 104
        f'{kill}104 isambard_013.sbatch CODE finalize',                                                     # 105
        f'{kill}105 --cpus-per-task=72 --mem=192G --time=08:00:00 isambard_013.sbatch CODE pairs',          # 106
        f'{kill}106 --gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=12:00:00 isambard_013.sbatch CODE train',
        f'{kill}107 {shard} isambard_013.sbatch CODE shard dev op-dpo2-e1 0 2',                             # 108
        f'{kill}107 {shard} isambard_013.sbatch CODE shard dev op-dpo2-e1 1 2',                             # 109
        f'{kill}108:109 isambard_013.sbatch CODE dev-merge op-dpo2-e1 2',                                   # 110
        f'{kill}107 {shard} isambard_013.sbatch CODE shard dev op-dpo2-e2 0 2',                             # 111
        f'{kill}107 {shard} isambard_013.sbatch CODE shard dev op-dpo2-e2 1 2',                             # 112
        f'{kill}111:112 isambard_013.sbatch CODE dev-merge op-dpo2-e2 2',                                   # 113
        f'{kill}110:113 isambard_013.sbatch CODE dev-score',                                                # 114
        f'{kill}114 isambard_013.sbatch CODE select']
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101\n'
    assert done.stdout.splitlines()[-1] == 'released recheck 101'
    failed = tmp_path / 'failed'
    failed.mkdir()
    squeue(failed)
    bad = subprocess.run(['bash', str(script), str(code)], env={**T.stub_sbatch(failed, fail_on=4), 'USER': 'me'},
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "sbatch failed for 'array'" in bad.stderr and 'scancel 101 102 103' in bad.stderr
    assert not (failed / 'log.scontrol').exists()  # The held root never ran.
    # A restart: FROM train after a failed train job reruns train (it resumes) and everything after it.
    (run / 'dpo-004/stage2-pairs').mkdir(parents=True)
    (run / 'dev-eval-013/op-dpo2-e1/placement/shard-01').mkdir(parents=True)  # A timed-out dev shard resumes.
    done, log = submit('3', '2', 'train')
    assert done.returncode == 0, done.stderr
    assert log == [
        '--parsable --hold isambard_013.sbatch CODE recheck',                                              # 116
        f'{kill}116 --gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=12:00:00 isambard_013.sbatch CODE train',
        f'{kill}117 {shard} isambard_013.sbatch CODE shard dev op-dpo2-e1 0 2',
        f'{kill}117 {shard} isambard_013.sbatch CODE shard dev op-dpo2-e1 1 2 --resume',
        f'{kill}118:119 isambard_013.sbatch CODE dev-merge op-dpo2-e1 2',
        f'{kill}117 {shard} isambard_013.sbatch CODE shard dev op-dpo2-e2 0 2',
        f'{kill}117 {shard} isambard_013.sbatch CODE shard dev op-dpo2-e2 1 2',
        f'{kill}121:122 isambard_013.sbatch CODE dev-merge op-dpo2-e2 2',
        f'{kill}120:123 isambard_013.sbatch CODE dev-score',
        f'{kill}124 isambard_013.sbatch CODE select']
    refusals = [
        (('3', '2', 'pairs'), {}, f"FROM pairs needs {run / 'dpo-004/samples/MANIFEST.json'}"),
        (('3', '2', 'dev'), {}, f"FROM dev needs {run / 'dpo-004/placement/dpo2-merged.json'}"),
        (('3', '2', 'bogus'), {}, None),
        ((), {}, 'refusing to reuse existing ' + str(run / 'dpo-004/stage2-pairs')),
        (('3', '2', 'train'), {'SQUEUE_OUT': '7400001\n'}, 'scancel 7400001')]
    for argv, extra, message in refusals:
        done, log = submit(*argv, **extra)
        assert log == [] and (done.returncode, message is None or message in done.stderr) == (2 if message is None else 1, True)
    shutil.rmtree(run / 'dpo-004')
    (run / 'op-yield-013/summary.json').write_text(json.dumps(
        {'status': 'OP_YIELD_DRY_RUN', 'projection': {'meets_floors': False}}))
    done, log = submit()
    assert log == [] and done.returncode == 1 and 'does not project the pair floors' in done.stderr
    (run / 'dev-eval-010/selection.json').unlink()
    done, log = submit('3', '2', 'train')
    assert log == [] and done.returncode == 1 and 'dev-eval-010/selection.json' in done.stderr
    (run / 'dev-eval-010/selection.json').write_text('{}')
    done = subprocess.run(['bash', str(script), str(code_root(tmp_path / 'main-nodoc', doc=False))], env=env,
                          capture_output=True, text=True)
    assert done.returncode == 1 and 'no amendment-013 protocol doc' in done.stderr
    # test: select once more on the login node, without --no-submit (it submits the adopted test chain itself).
    done, log = submit('test')
    assert log == [] and done.returncode == 1 and 'no ' + str(run / 'dev-eval-013/selection.json') in done.stderr
    (run / 'dev-eval-013').mkdir(exist_ok=True)
    (run / 'dev-eval-013/selection.json').write_text('{}')
    done, log = submit('test', PYLOG=str(tmp_path / 'python.log'))
    eval_root = run / 'dev-eval-013'
    assert done.returncode == 0 and log == [] and (tmp_path / 'python.log').read_text().split() == [
        '-B', '-m', 'fastfill.comparison.onpolicy_dpo', 'select', '--scores', str(eval_root / 'scores'),
        '--out', str(eval_root / 'selection.json'), '--code-root', str(code), '--shards', '6']
    assert subprocess.run(['bash', str(script), str(code), 'test', '3'], env=env).returncode == 2


@pytest.mark.parametrize('argv, ok', [
    (['yield'], True), (['recheck'], True), (['recheck', 'x'], False), (['subset'], True), (['sample', '1', '--limit', '16'], True), (['sample', '2'], False), (['finalize'], True),
    (['pairs'], True), (['train'], True), (['train', 'x'], False), (['shard', 'dev', 'op-dpo2-e1', '0', '2'], True),
    (['shard', 'test', 'op-dpo2-e2', '5', '6', '--resume'], True), (['shard', 'test', 'op-dpo2-e1-merged', '0', '2'], False),
    (['shard', 'dev', 'rpo-dpo1', '0', '2'], False), (['shard', 'dev', 'op-dpo2-e1', '2', '2'], False),
    (['dev-merge', 'op-dpo2-e2-merged', '2'], True), (['dev-merge', 'sft', '2'], False), (['dev-score'], True),
    (['select'], True), (['test-geometry'], True), (['test-report'], True), (['check-merged'], True), (['bogus'], False)])
def test_sbatch_modes_validate_arguments_before_the_allocation(tmp_path, argv, ok):
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    done = subprocess.run(['bash', str(SCRIPTS / 'isambard_013.sbatch'), str(CODE), *argv], env=env, capture_output=True,
                          text=True)
    # Valid: passes the argument checks and stops at the missing Slurm allocation. Invalid: exit 2 before it.
    assert (done.returncode, 'Slurm allocation required' in done.stderr) == ((1, True) if ok else (2, False))
