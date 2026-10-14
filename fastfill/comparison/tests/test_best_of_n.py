"""S2 best-of-N selection on CPU: samples come from sample_vllm with a scripted fake vLLM engine."""
import json
import random
from types import SimpleNamespace

import pytest

from fastfill.comparison import best_of_n as B
from fastfill.comparison import geometry as g
from fastfill.comparison import infer
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import sample_vllm as sv
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_sample_vllm as S

COUNTS = (2, 3, 12, 2)  # Bins 1-10 and 11-30.
ADDED = ('sample_index', 'selection', 'violations', 'gate_fail', *B.EXPLOITS, *B.CAP_FLAGS)


def script(rooms, arm):
    """k0 all stacked (collides), k1 shrunk to half (J: gated), k2 valid but cut by the cap, k3 one object out of
    the room, k4 valid, k5+ unparsable."""
    def answer(request, uid, k):
        gt = {o['id']: o for o in rooms[uid]['objects']}
        def layout(pos=lambda j, o: o['pos'], scale=1.0):
            return json.dumps({'placements': [
                {'id': o['id'], **({'size': [v * scale for v in gt[o['id']]['size']]} if arm == 'joint' else {}),
                 'pos': pos(j, gt[o['id']]), 'yaw': 0} for j, o in enumerate(request['objects'])]})
        return {0: (layout(lambda j, o: [1, 1, 0]), 'stop'), 1: (layout(scale=0.5), 'stop'), 2: (layout(), 'length'),
                3: (layout(lambda j, o: [7.5, 1, 0] if j == 0 else o['pos']), 'stop'),
                4: (layout(), 'stop')}.get(k, ('nope', 'stop'))
    return answer


@pytest.fixture
def world(tmp_path, monkeypatch):
    data = S.make_data(tmp_path, COUNTS)
    world = S.patched(tmp_path, monkeypatch, data)
    rooms = {r['uid']: r for r in map(json.loads, (data / 'rooms.jsonl').read_text().splitlines())}
    for arm in h.ARMS:
        sv.main(S.argv(tmp_path, data, arm, f'samples-{arm}'), S.loader(S.FakeLLM(data, script(rooms, arm))))
        S.finalize(tmp_path, data, f'samples-{arm}')
    return SimpleNamespace(**vars(world), rooms=rooms)


def evaluate(world, arm, out='sel', **kw):
    return B.evaluate(world.data, world.root / f'samples-{arm}', arm, world.root / f'{out}-{arm}', **kw)


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def cand(k, complete=True, v=0, gate=(), bias=0.0, **counts):
    return {'k': k, 'complete': complete, 'violations': v if complete else None, 'gate_fail': list(gate) if complete else None,
            **{key: counts.get(key, 0) if complete else None for key in B.GUARDED}, 'bias_prior': bias if complete else None}


def test_select_rule():
    select = B.select
    assert select([cand(0, complete=False), cand(1, v=5)], 1) == (1, 'fewest_violations')  # Complete wins.
    fewer = [cand(0, v=3), cand(1, v=2), cand(2, v=1)]
    assert [select(fewer, m) for m in (0, 1, 2)] == [(0, 'greedy'), (1, 'fewest_violations'), (2, 'fewest_violations')]
    assert select([cand(0, v=1), cand(1, v=1)], 8) == (0, 'greedy')  # Greedy wins ties.
    assert select([cand(0, v=3), cand(2, v=1), cand(1, v=1)], 8)[0] == 1  # Then the lower k.
    gated = [cand(0, v=4), cand(1, v=0, gate=['obj_0001'])]
    assert select(gated, 8) == (0, 'greedy') and select(gated, 8, guarded=False) == (1, 'fewest_violations')
    # Review item 1: a gated greedy stays in every pool, so a room greedy passes never fails at m >= 1.
    shrunk_greedy = [cand(0, v=0, gate=['obj_0001']), cand(1, v=3)]
    assert [select(shrunk_greedy, m) for m in (0, 1, 8)] == [(0, 'greedy')] * 3
    nothing = [cand(0, complete=False), cand(1, v=0, gate=['obj_0001']), cand(2, complete=False)]
    assert select(nothing, 8) == (0, 'fallback_greedy')
    with pytest.raises(ValueError, match='start with greedy'):
        select([cand(1), cand(0)], 8)


def test_guards_close_the_size_exploits_relative_to_greedy():
    """Review item 2: a band-passing shrink, a duplicate tucked inside its twin, a flattened box or a same-category
    stack never wins over greedy; the bias guard needs a complete greedy, the counts fall back to 0 without one."""
    select, greedy = B.select, cand(0, v=2, contained_pairs=1, flat_objects=1, stacked_duplicates=1, bias=-0.10)
    for worse in ({'contained_pairs': 2}, {'flat_objects': 2}, {'stacked_duplicates': 2}, {'bias': -0.131}):
        assert select([greedy, cand(1, **worse)], 8) == (0, 'greedy'), worse
    equal = cand(1, contained_pairs=1, flat_objects=1, stacked_duplicates=1, bias=-0.13)
    assert select([greedy, equal], 8) == (1, 'fewest_violations')  # No worse than greedy: admissible.
    broken = cand(0, complete=False)
    assert select([broken, cand(1, contained_pairs=1)], 8) == (0, 'fallback_greedy')
    assert select([broken, cand(1, bias=-0.5)], 8) == (1, 'fewest_violations')
    # The two-wardrobe case: 1.5x and 0.67x of one prior on one spot pass the band and are 'contained', not main.
    prior = {'a': [1.0, 0.6, 2.0], 'b': [1.0, 0.6, 2.0]}
    pair = [{'id': 'a', 'category': 'wardrobe', 'pos': [2, 2, 0], 'size': [1.5, 0.9, 2.0], 'yaw': 0.0, 'parent': None},
            {'id': 'b', 'category': 'wardrobe', 'pos': [2, 2, 0], 'size': [0.67, 0.402, 1.34], 'yaw': 0.0, 'parent': None}]
    assert B.size_gate(pair, prior) == []
    assert B.exploits({'complete': True, 'raw_layout': pair}, {'objects': pair}, prior)['contained_pairs'] == 1


def test_selected_violations_never_rise_with_m():
    rng = random.Random(0)
    for _ in range(2000):
        cands = [cand(k, complete=rng.random() < .7, v=rng.randrange(4), gate=['x'] if rng.random() < .3 else (),
                      bias=rng.uniform(-.1, 0), **{key: rng.randrange(2) for key in B.GUARDED}) for k in range(9)]
        picks = [B.select(cands, m) for m in (0, 1, 2, 4, 8)]
        eligible = [reason != 'fallback_greedy' for _, reason in picks]
        assert eligible == sorted(eligible)  # Once a room has an eligible pick, it keeps one.
        values = [cands[k]['violations'] for (k, _), ok in zip(picks, eligible) if ok]
        assert values == sorted(values, reverse=True)
        if cands[0]['complete']:  # Greedy is in every pool: never a fallback, never above greedy.
            assert all(eligible) and values[0] == cands[0]['violations']


def test_size_gate_inclusive_bounds_include_height_and_violations_sum():
    prior = {'a': [1.0, 2.0, 0.5]}
    def gate(size, band=B.BAND):
        return B.size_gate([{'id': 'a', 'size': size}], prior, band)
    assert gate([0.67, 3.0, 0.5 * 0.67]) == [] and gate([1.5, 2 * 0.67, 0.75]) == []  # Both bounds inclusive.
    assert gate([0.669, 2, 0.5]) == ['a'] and gate([1, 3.01, 0.5]) == ['a']
    assert gate([1, 2, 0.5 * 1.51]) == ['a'] and gate([1, 2, 0.5 * 0.66]) == ['a']  # Height alone fails.
    assert gate([1, 2, 0.5], (1.01, 2)) == ['a']
    row = {'complete': True, **dict(zip(B.VIOLATIONS, (1, 2, 0, 3, 3, 1)))}
    assert B.violations(row) == 10 and B.violations({**row, 'complete': False}) is None


def test_exploits_count_contained_flat_and_reparented_objects():
    def box(oid, category, pos, size, parent=None):
        return {'id': oid, 'category': category, 'pos': pos, 'size': size, 'yaw': 0.0, 'parent': parent}
    layout = [box('cab', 'cabinet', [2, 2, 0], [2, 2, 1]), box('bas', 'basket', [2, 2, 0], [0.5, 0.5, 0.5]),
              box('sofa', 'sofa', [5, 5, 0], [2, 1, 0.03]), box('mat', 'tray', [6, 6, 0], [0.3, 0.3, 0.03]),
              box('lamp', 'lamp', [5, 5, 0.03], [0.2, 0.2, 0.4], parent='sofa')]
    layout.append(box('mat2', 'Tray', [6, 6, 0.03], [0.3, 0.3, 0.03], parent='mat'))  # Same category after norm_cat.
    prior = {'cab': [2, 2, 1], 'bas': [0.5, 0.5, 0.5], 'sofa': [2, 1, 0.8], 'mat': [0.3, 0.3, 0.04], 'lamp': [0.2, 0.2, 0.4],
             'mat2': [0.3, 0.3, 0.03]}
    room = {'objects': [{**o, 'parent': None} for o in layout]}
    assert B.exploits({'complete': True, 'raw_layout': layout}, room, prior) == {
        'contained_pairs': 1, 'flat_objects': 1, 'stacked_duplicates': 1,
        'parent_changed': 2}  # The 3 cm tray had a 4 cm prior: not flattened; the lamp on the sofa is no duplicate.
    assert B.exploits({'complete': False}, room, prior) == dict.fromkeys(B.EXPLOITS)


def test_selection_end_to_end_rows_are_geometry_scores_of_the_chosen_text(world):
    summaries = {arm: evaluate(world, arm, workers=1) for arm in h.ARMS}
    parallel = evaluate(world, 'placement', 'spawned', workers=2)  # Spawned workers: picklable tasks and results.
    assert parallel['by_m'] == summaries['placement']['by_m']
    joint, out = summaries['joint'], world.root / 'sel-joint'
    rates = {m: joint['by_m'][m]['all']['success_rate'] for m in joint['by_m']}
    assert rates == {'0': 0.0, '1': 0.0, '2': 0.0, '4': 1.0, '8': 1.0}  # Shrunk (k1) and capped (k2) never win.
    assert {m: v['all']['success_rate'] for m, v in summaries['placement']['by_m'].items()} == {
        '0': 0.0, '1': 1.0, '2': 1.0, '4': 1.0, '8': 1.0}  # P ignores the gate: k1 is valid at its prior sizes.
    assert joint['by_m']['8']['all']['changed_from_greedy'] == 4 and joint['by_m']['8']['all']['fallback_rooms'] == 0
    # Unguarded, the shrunk k1 (0 violations) would win every J room at m >= 1; P never meets a guard here.
    assert [joint['by_m'][m]['all']['guard_changed_rooms'] for m in ('0', '1', '8')] == [0, 4, 4]
    assert summaries['placement']['by_m']['8']['all']['guard_changed_rooms'] == 0
    assert {'contract.py', 'scene.py', 'validate.py', 'geometry.py'} <= set(joint['code_sha256'])
    assert list(joint['by_m']['8']['by_bin']) == ['1-10', '11-30'] and set(joint['by_m']['8']['by_source']) == {'S0', 'S1'}
    assert joint['by_m']['8']['all']['delta_bias_fp_vs_greedy'] == {'prior': 0.0, 'gt': 0.0, 'rooms': 4}
    assert joint['band'] == [0.67, 1.5] and joint['fidelity'] is None and not joint['quality_claim']
    assert set(h.strict_json((out / 'MANIFEST.json').read_text())['files']) == {
        'joint-greedy-geometry.jsonl', 'joint-bo8-geometry.jsonl', 'joint-samples.jsonl', 'summary.json'}
    compact = rows(out / 'joint-samples.jsonl')
    assert len(compact) == 4 * 9 and not any('raw_layout' in c for c in compact)
    capped = [c for c in compact if c['k'] == 2]  # A stop on 'length' is incomplete even though the text parses.
    assert all(not c['complete'] and c['error'] == r2.CAP and c['finish_reason'] == 'length' for c in capped)
    assert all(c['gate_fail'] for c in compact if c['k'] == 1)
    assert all(c['violations'] == 1 for c in compact if c['k'] == 3)
    samples = {(s['uid'], s['k']): s for s in rows(world.root / 'samples-joint/samples.jsonl')}
    requests = {r['uid']: r for r in infer.load_eval(world.data, 'joint')[0]}
    for name, k in (('joint-bo8-geometry.jsonl', 4), ('joint-greedy-geometry.jsonl', 0)):
        chosen = rows(out / name)
        assert [r['sample_index'] for r in chosen] == [k] * 4
        for row in chosen:
            score = g.score(world.rooms[row['uid']], requests[row['uid']]['messages'], samples[(row['uid'], k)]['raw'], 'joint')
            assert {key: v for key, v in row.items() if key not in ADDED} == json.loads(json.dumps(score))
        assert g.summarize(chosen)['all']['requests'] == 4
    assert [r['selection'] for r in rows(out / 'joint-greedy-geometry.jsonl')] == ['greedy'] * 4
    with pytest.raises(FileExistsError):
        evaluate(world, 'joint')
    with pytest.raises(ValueError, match='ks must'):
        evaluate(world, 'joint', 'other', ks=(2, 1))
    with pytest.raises(ValueError, match='not a complete sampling run'):
        B.evaluate(world.data, world.root / 'samples-placement', 'joint', world.root / 'wrong-arm')


def write_reference(root, data, arm, raws, model, capped=(), drift=()):
    """An HF inference dir in the evaluate_pair._inference layout, bound to the same data; capped rooms reached the
    token cap (with EOS), drift rooms have one prompt token more than the vLLM prompt."""
    ref = root / f'hf-{arm}'
    ref.mkdir()
    identity = {'config': {'arm': arm, 'data': str(data), 'lora': 'merge_and_unload_bf16'},
                'data': {'manifest_sha256': h.sha256(data / 'MANIFEST.json')}, **model}
    run_hash = infer._hash(identity)
    h.write_json(ref / 'run_receipt.json', {**identity, 'run_sha256': run_hash})
    requests = infer.load_eval(data, arm)[0]
    (ref / 'records.jsonl').write_text(''.join(json.dumps(
        {**{k: r[k] for k in ('uid', 'source', 'group')}, 'arm': arm, 'raw': raws[r['uid']], 'error': None,
         'input_tokens': len(infer.prompt_ids(S.Tok(), r)) + (r['uid'] in drift),
         'reached_token_cap': r['uid'] in capped, 'stopped_on_eos': True, 'run_sha256': run_hash}) + '\n' for r in requests))
    h.write_json(ref / 'result.json', {'status': 'INFERENCE_COMPLETE', 'pending_count': 0})
    h.write_json(ref / 'MANIFEST.json', {'files': h.artifact_manifest(ref), 'expected_uids': [r['uid'] for r in requests]})
    return ref


def test_fidelity_reports_exact_match_and_discordant_rooms(world, monkeypatch):
    samples = {(s['uid'], s['k']): s['raw'] for s in rows(world.root / 'samples-joint/samples.jsonl')}
    raws = {uid: samples[(uid, 4) if uid == 'r0' else (uid, 0)] for uid in world.rooms}  # HF passes r0, vLLM greedy fails.
    model = {'base_sha256': 'b' * 64, 'adapter_sha256': 'e' * 64}
    ref = write_reference(world.root, world.data, 'joint', raws, model, capped={'r0'})
    fidelity = evaluate(world, 'joint', reference=ref)['fidelity']
    assert (fidelity['rooms'], fidelity['exact_match'], fidelity['b_hf_only_pass'], fidelity['c_vllm_only_pass']) == (4, 3, 1, 0)
    assert (fidelity['b_either_capped'], fidelity['c_either_capped'], fidelity['prompt_token_counts_equal']) == (1, 0, 4)
    assert (fidelity['hf_success_rate'], fidelity['vllm_greedy_success_rate'], fidelity['both_pass']) == (0.25, 0.0, 0)
    def unscored(*args):
        raise AssertionError('reference checks must run before any scoring')
    monkeypatch.setattr(B.d, '_map', unscored)  # Review item 7: a bad reference fails before the long scoring step.
    for name, kw, match in (('other', {'model': {**model, 'adapter_sha256': 'f' * 64}}, 'not the sampled model'),
                            ('drift', {'model': model, 'drift': {'r2'}}, '1 reference prompts differ in token count')):
        (world.root / name).mkdir()
        bad = write_reference(world.root / name, world.data, 'joint', raws, **kw)
        with pytest.raises(ValueError, match=match):
            evaluate(world, 'joint', name, reference=bad)


def test_calibrate_reports_per_bin_gate_pass_and_flags_strict_bands(world):
    evaluate(world, 'joint')
    result = B.calibrate(world.data, world.root / 'sel-joint/joint-greedy-geometry.jsonl', [(0.67, 1.5), (1.01, 2)])
    assert result['complete_rooms'] == result['rooms'] == 4
    assert result['bands']['0.67,1.5']['1-10'] == {'complete_rooms': 3, 'gate_pass_rate': 1.0, 'too_strict': False}
    assert result['bands']['0.67,1.5']['51+'] == {'complete_rooms': 0, 'gate_pass_rate': None, 'too_strict': False}
    assert result['bands']['1.01,2']['11-30'] == {'complete_rooms': 1, 'gate_pass_rate': 0.0, 'too_strict': True}
    with pytest.raises(ValueError, match='joint rows'):
        B.calibrate(world.data, world.root / 'samples-joint/samples.jsonl', [B.BAND])
