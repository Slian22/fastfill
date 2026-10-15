"""Amendment 025 critic -> constrained repair on synthetic rooms (no network, no server inputs)."""
import json
import math
import random
import struct

import pytest

from fastfill.comparison import critic_025 as c
from fastfill.comparison import train_pair_helpers as h
from fastfill.validate import holds

RECT = [[0, 0], [4, 0], [4, 3], [0, 3]]
DOOR = {'id': 'fixed_0001', 'category': 'door', 'size': [0.1, 0.9, 2.0], 'pos': [0.5, 0.0, 0.0], 'yaw': math.pi / 2}


def obj(i, category, size, pos, yaw=0.0, parent=None):
    return {'id': i, 'category': category, 'size': list(size), 'pos': list(pos), 'yaw': yaw, 'parent': parent}


def room(uid='u1', boundary=RECT, fixed=(DOOR,)):
    return {'uid': uid, 'source': 's', 'group': 'g-' + uid, 'room_type': 'bedroom', 'boundary_type': 'polygon',
            'boundary': boundary, 'height': 2.8, 'fixed': list(fixed)}


def bedroom():
    """Bed against the top-left walls, desk on the right wall facing in, a lamp on it, a chair facing away."""
    return [obj('bed', 'bed', (2.0, 1.6, 0.5), (1.0, 2.2, 0.0)),
            obj('desk', 'desk', (0.5, 1.0, 0.75), (3.75, 1.5, 0.0), math.pi),
            obj('lamp', 'lamp', (0.2, 0.2, 0.4), (3.75, 1.7, 0.75), 0.0, 'desk'),
            obj('chair', 'chair', (0.5, 0.5, 0.9), (2.8, 1.5, 0.0), math.pi)]


def state(uid, layout, r=None):
    r = r or room(uid)
    return {'uid': uid, 'source': 's', 'group': r['group'], 'objects': len(layout), 'room': r, 'layout': layout,
            'metrics': c.metrics(r, layout), 'round': 0, 'active': True, 'history': []}


def answer(edits, f=5, l=5):
    return json.dumps({'scores': {'functionality': f, 'layout': l}, 'edits': edits})


def by(layout):
    return {o['id']: o for o in layout}


# ---------------------------------------------------------------- selection, settings

def test_selection_is_family_balanced_lowest_salted_hash_and_order_free():
    beds, garages, halls = [f'bed-{k}' for k in range(40)], [f'gar-{k}' for k in range(5)], ['hall-0']
    types = {**{u: 'bedroom' for u in beds}, **{u: 'garage' for u in garages}, 'hall-0': 'Hallway', 'none-0': None}
    uids = beds + garages + halls + ['none-0']
    first, counts = c.select(uids, types)
    shuffled = uids[:]
    random.Random(3).shuffle(shuffled)
    assert c.select(shuffled, types) == (first, counts)
    assert first == sorted(beds, key=c.rank)[:31] + sorted(garages, key=c.rank) + halls + ['none-0']
    assert counts == {**{f: 0 for f in c.FAMILIES}, 'bedroom': 31, 'garage': 5, 'circulation': 1, 'none': 1}
    assert list(counts) == list(c.FAMILIES) and c.PER_FAMILY == 31
    assert c.rank('x') == __import__('hashlib').sha256(b'a025:x').hexdigest()


def test_rows_refuses_an_input_run_of_another_model(tmp_path, monkeypatch):
    monkeypatch.setattr(c.A, '_slice', lambda data: ([], {}, {}))
    for config in ({'arm': 'placement', 'model': '/runs/a021/9b', 'lora': c.A.VLLM_LORA},
                   {'arm': 'placement', 'model': '/runs/a021/27b', 'lora': 'hf'},
                   {'arm': 'joint', 'model': '/runs/a021/27b', 'lora': c.A.VLLM_LORA}):
        monkeypatch.setattr(c.e, '_inference', lambda arm, data, expected, config=config: ([], {'config': config}))
        with pytest.raises(ValueError, match='not the placement vLLM run'):
            c.rows(tmp_path / 'data', tmp_path / 'arm', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_s1_settings_are_the_delivered_systems():
    from fastfill.system import pipeline
    assert c.S1 == pipeline.S1 == {'max_trials': 2000, 'time_limit': 60.0}


# ---------------------------------------------------------------- parsing

def test_parse_rejects_forbidden_ops_unknown_ids_and_extra_fields_without_repair():
    ids, targets = {'bed', 'desk'}, {'bed', 'desk', 'fixed_0001'}
    edits = [{'id': 'bed', 'op': 'against_wall', 'reason': 'r'},
             {'id': 'bed', 'op': 'remove', 'reason': 'r'},
             {'id': 'sofa', 'op': 'rotate', 'yaw_deg': 90, 'reason': 'r'},
             {'id': 'bed', 'op': 'move', 'x': 1, 'y': 2, 'size': [1, 1, 1], 'reason': 'r'},
             {'id': 'bed', 'op': 'move', 'x': 1, 'reason': 'r'},
             {'id': 'bed', 'op': 'face', 'target': 'ghost', 'reason': 'r'},
             {'id': 'bed', 'op': 'face', 'target': 'bed', 'reason': 'r'}]
    out = c.parse_edits(answer(edits), ids, targets)
    assert out['error'] is None and out['edits'] == edits[:1]
    assert [r['code'] for r in out['rejected']] == ['forbidden_op', 'unknown_id', 'unexpected_field',
                                                     'missing_field', 'over_limit', 'over_limit']
    out = c.parse_edits(answer(edits[5:]), ids, targets)
    assert [r['code'] for r in out['rejected']] == ['unknown_target', 'self_target'] and out['edits'] == []
    more = [{'id': 'bed', 'op': 'rotate', 'yaw_deg': True, 'reason': 'r'},
            {'id': 'bed', 'op': 'rotate', 'yaw_deg': 12345.5, 'reason': 'r'},
            {'id': 'bed', 'op': 'against_wall', 'reason': 3}, ['bed'],
            {'id': 'desk', 'op': 'face', 'target': 'fixed_0001', 'reason': 'r'}]
    out = c.parse_edits(answer(more).replace('12345.5', '1e999'), ids, targets)       # JSON's way to an infinity
    assert [r['code'] for r in out['rejected']] == ['invalid_number', 'invalid_number', 'invalid_reason', 'not_object']
    assert out['edits'] == more[-1:]
    far = [{'id': 'bed', 'op': 'move', 'x': 1e300, 'y': 1, 'reason': 'r'},
           {'id': 'bed', 'op': 'move', 'x': 1, 'y': -1e6 - 1, 'reason': 'r'},
           {'id': 'bed', 'op': 'move', 'x': 1e6, 'y': -1e6, 'reason': 'r'}]
    out = c.parse_edits(answer(far), ids, targets)                  # shapely raises near 1e300 inside S1
    assert [r['code'] for r in out['rejected']] == ['invalid_number'] * 2 and out['edits'] == far[2:]
    for text, code in (('```json\n{}\n```', 'invalid_json'), ('{"scores": {"functionality": 5, "layout": 5}}',
                                                                 'invalid_keys'),
                       (answer([], f=11), 'invalid_scores'), (answer([], f=5.0), 'invalid_scores'),
                       ('{"scores": {"functionality": 5, "layout": 5}, "edits": {}}', 'invalid_edits'),
                       ('{"scores": {"functionality": 5, "layout": 5, "realism": 2}, "edits": []}', 'invalid_scores')):
        assert c.parse_edits(text, ids, targets)['error'] == code


# ---------------------------------------------------------------- ops

def test_move_and_rotate_carry_the_support_subtree_rigidly():
    layout = bedroom()
    moved, _ = c.apply_edit(room(), layout, {'id': 'desk', 'op': 'move', 'x': 2.0, 'y': 1.0, 'reason': ''})
    m = by(moved)
    assert m['desk']['pos'] == [2.0, 1.0, 0.0] and m['desk']['yaw'] == math.pi
    assert m['lamp']['pos'] == pytest.approx([2.0, 1.2, 0.75]) and m['lamp']['yaw'] == 0.0
    assert m['bed'] == by(layout)['bed'] and by(layout)['desk']['pos'] == [3.75, 1.5, 0.0]   # input untouched
    turned, _ = c.apply_edit(room(), layout, {'id': 'desk', 'op': 'rotate', 'yaw_deg': 270, 'reason': ''})
    t = by(turned)
    assert t['desk']['yaw'] == pytest.approx(3 * math.pi / 2) and t['desk']['pos'] == pytest.approx([3.75, 1.5, 0.0])
    # a +90 degree turn about the desk centre: the lamp's (0, 0.2) offset becomes (-0.2, 0)
    assert t['lamp']['pos'] == pytest.approx([3.55, 1.5, 0.75]) and t['lamp']['yaw'] == pytest.approx(math.pi / 2)


def test_against_wall_puts_the_back_on_the_nearest_wall_facing_in():
    layout = [obj('bed', 'bed', (2.0, 1.6, 0.5), (2.0, 0.9, 0.0), 0.3)]
    out, _ = c.apply_edit(room(), layout, {'id': 'bed', 'op': 'against_wall', 'reason': ''})
    bed = out[0]
    assert bed['pos'] == pytest.approx([2.0, 1.0, 0.0]) and bed['yaw'] == pytest.approx(math.pi / 2)
    assert holds(['against_wall', 'bed'], out, RECT)
    assert min(y for _, y in __import__('fastfill.scene').scene.footprint(bed)) == pytest.approx(0.0)
    assert -math.sin(bed['yaw']) == pytest.approx(-1.0)       # local -X (the back) points at the wall y = 0
    corner = [obj('bed', 'bed', (2.0, 1.6, 0.5), (0.5, 0.3, 0.0))]
    assert c.apply_edit(room(), corner, {'id': 'bed', 'op': 'against_wall', 'reason': ''})[0][0]['pos'] == \
        pytest.approx([0.8, 1.0, 0.0])                        # clamped so the whole width stays on the edge


def test_face_turns_the_front_to_the_target_centre_and_skips_a_coincident_one():
    layout = bedroom()
    out, code = c.apply_edit(room(), layout, {'id': 'chair', 'op': 'face', 'target': 'desk', 'reason': ''})
    assert code is None and by(out)['chair']['yaw'] == pytest.approx(0.0) and holds(['faces', 'chair', 'desk'], out, RECT)
    out, _ = c.apply_edit(room(), layout, {'id': 'chair', 'op': 'face', 'target': 'fixed_0001', 'reason': ''})
    assert by(out)['chair']['yaw'] == pytest.approx(math.atan2(-1.5, -2.3) % c.TAU)
    stacked = [obj('a', 'desk', (1, 1, .7), (1, 1, 0)), obj('b', 'lamp', (.2, .2, .3), (1, 1, .7), 0, 'a')]
    assert c.apply_edit(room(), stacked, {'id': 'b', 'op': 'face', 'target': 'a', 'reason': ''}) == (stacked,
                                                                                                    'degenerate_target')


# ---------------------------------------------------------------- re-check

def test_verdict_is_lexicographic_and_guards_admission():
    ok = lambda key, accepted=True: {'key': key, 'accepted': accepted}
    assert c.verdict(ok([0, 0, 0, 2, 0, 0, 0]), ok([0, 0, 0, 1, 9, 9, 9])) == []      # fewer main beats more oob
    assert c.verdict(ok([0, 0, 0, 1, 0, 0, 0]), ok([0, 0, 0, 2, 0, 0, 0])) == ['checker_worse']
    assert c.verdict(ok([1, 0, 0, 0, 0, 0, 0]), ok([0, 5, 5, 5, 0, 0, 0])) == []      # validity first
    assert c.verdict(ok([0] * 7), ok([0] * 7, False)) == ['admission_worse']
    assert c.verdict(ok([0] * 7, False), ok([0] * 7, False)) == []
    assert c.invariants(bedroom(), bedroom()) == {'count': 0, 'id': 0, 'category': 0, 'size': 0, 'parent': 0}
    changed = bedroom()
    changed[0]['size'] = [1, 1, 1]
    assert c.invariants(bedroom(), changed)['size'] == 1


def test_apply_edits_accepts_a_better_layout_and_keeps_the_previous_one_otherwise():
    layout = bedroom()
    result = c.apply_edits(room(), layout, [{'id': 'chair', 'op': 'face', 'target': 'desk', 'reason': ''}])
    assert result['accepted'] and result['reasons'] == [] and result['yaw_changed'] == 1
    assert by(result['layout'])['chair']['yaw'] == pytest.approx(0.0) and not any(result['invariants'].values())
    assert result['candidate_key'] == result['before_key'] == [0] * 7
    corridor = room(boundary=[[0, 0], [3, 0], [3, 1], [0, 1]], fixed=())
    bench = [obj('bench', 'bench', (0.4, 2.5, 0.5), (1.5, 0.5, 0.0), math.pi / 2)]
    result = c.apply_edits(corridor, bench, [{'id': 'bench', 'op': 'rotate', 'yaw_deg': 0, 'reason': ''}])
    assert not result['accepted'] and result['reasons'] == ['checker_worse', 'admission_worse']
    assert result['layout'] == bench
    assert result['candidate_key'][0] == 1 and result['metrics'] == c.metrics(corridor, bench)
    assert not result['unchanged']
    # a no-op set meets both section-1 conditions: accepted (recorded as unchanged), the layout stays the same
    result = c.apply_edits(room(), layout, [{'id': 'desk', 'op': 'against_wall', 'reason': ''}])
    assert result['accepted'] and result['unchanged'] and result['reasons'] == [] and result['layout'] == layout


# ---------------------------------------------------------------- rounds

def done(text):
    return {'status': 'completed', 'returned_model': 'gpt-6.1-sol', 'raw_final_text': text}


def edit_face(f=5, l=5):
    return done(answer([{'id': 'chair', 'op': 'face', 'target': 'desk', 'reason': 'use the desk'}], f, l))


def test_round_step_stops_on_failure_rejection_scores_and_round_limit():
    row = state('u1', bedroom())
    failed = c._step((row, None, 1))
    assert not failed['active'] and failed['history'][-1]['stop'] == 'call_failed' and failed['layout'] == row['layout']
    for bad in ({**edit_face(), 'status': 'failed', 'error': 'model_mismatch'},       # failed records keep their text
                {**edit_face(), 'returned_model': 'gpt-5'}, {**edit_face(), 'status': None}, done('  ')):
        out = c._step((row, bad, 1))
        assert out['history'][-1]['call'] == 'failed' and out['history'][-1]['stop'] == 'call_failed'
        assert not out['active'] and out['layout'] == row['layout'] and 'edits' not in out['history'][-1]
    invalid = c._step((row, done('not json'), 1))
    assert invalid['history'][-1]['stop'] == 'invalid_answer' and invalid['layout'] == row['layout']
    empty = c._step((row, done(answer([{'id': 'x', 'op': 'move', 'x': 0, 'y': 0, 'reason': ''}])), 1))
    assert empty['history'][-1]['stop'] == 'no_valid_edits' and empty['history'][-1]['rejected'][0]['code'] == 'unknown_id'
    going = c._step((row, edit_face(), 1))
    assert going['active'] and going['history'][-1]['stop'] is None and going['history'][-1]['accepted']
    assert by(going['layout'])['chair']['yaw'] == pytest.approx(0.0) and going['metrics'] == c.metrics(row['room'],
                                                                                                       going['layout'])
    good = c._step((row, edit_face(8, 9), 1))
    assert not good['active'] and good['history'][-1]['stop'] == 'scores_ge_8' and good['layout'] == going['layout']
    last = c._step((row, edit_face(), 2))
    assert not last['active'] and last['history'][-1]['stop'] == 'max_rounds'
    again = c._step((going, edit_face(), 1))                   # the chair already faces the desk: nothing changes
    assert again['active'] and again['history'][-1]['accepted'] and again['history'][-1]['unchanged']
    assert again['history'][-1]['stop'] is None and again['layout'] == going['layout']


def png_size(data):
    assert data[:8] == b'\x89PNG\r\n\x1a\n' and data[12:16] == b'IHDR'
    return struct.unpack('>II', data[16:24])


def test_case_views_are_deterministic_bounded_and_the_prompt_carries_the_tools():
    row = state('u1', bedroom())
    (files, case), (again, _) = c.case_files((row, 1)), c.case_files((row, 1))
    assert files == again and len(files) == 3
    for name in (i['path'] for i in case['images']):
        assert max(png_size(files[name])) <= 1024 and b'Software' not in files[name]
    text = files[case['prompt_file']].decode()
    for needle in ('take precedence over', "a bed's headboard", 'chairs face their desk or table', 'a sofa faces the TV',
                   'clear path from each door', 'nothing blocks a door or a window', '"op": "against_wall"',
                   '"id":"lamp","category":"lamp"', '"parent":"desk"', 'Frozen checker: {"geometry_success":true',
                   'Strict admission', '"failure_reasons":'):
        assert needle in text, needle
    flag, _ = c.flags(row['room'], [*bedroom(), obj('box', 'wardrobe', (1, 1, 2), (1.0, 2.2, 0.0))])
    assert flag['box'] == flag['bed'] == 'main' and flag['desk'] == 'ok'


def test_rounds_and_evaluation_end_to_end(tmp_path):
    rows = [state('u1', bedroom()), state('u2', bedroom()), state('u3', bedroom())]
    per_family = {f: 3 * (f == 'bedroom') for f in c.FAMILIES}
    c._publish_state(tmp_path / 'r0', rows, {'status': 'A025_B_COMPLETE', 'round': 0, 'b_manifest_sha256': None,
                                             'selection': {'per_family': per_family}}, 1)
    mapping = h.strict_json((tmp_path / 'r0/critic-mapping.json').read_text())
    assert mapping == {c.case_id(u, 1): u for u in ('u1', 'u2', 'u3')}
    manifest = h.strict_json((tmp_path / 'r0/critic-manifest.json').read_text())
    assert manifest['expected_model'] == 'gpt-6.1-sol' and len(manifest['cases']) == 3

    def record(state_dir, uid, k, value):                     # identity as critic_batch_025's result files carry it
        case = next(x for x in h.strict_json((state_dir / 'critic-manifest.json').read_text())['cases']
                    if x['id'] == c.case_id(uid, k))
        return json.dumps({'schema_version': 1, 'id': case['id'], 'expected_model': 'gpt-6.1-sol',
                           'content_sha256': c.judge_batch.content_hash(case), **value})
    answers = tmp_path / 'answers'
    (answers / 'nested').mkdir(parents=True)
    (answers / 'nested/a.result.json').write_text(record(tmp_path / 'r0', 'u1', 1, edit_face()))
    (answers / 'b.json').write_text(record(tmp_path / 'r0', 'u2', 1, done(answer([], 9, 9))))
    (answers / 'c.attempt.json').write_text(record(tmp_path / 'r0', 'u3', 1, {'status': 'attempted'}))
    stale = tmp_path / 'answers-stale'                        # same case id, other content: refused, nothing written
    stale.mkdir()
    (stale / 'a.json').write_text(json.dumps({**json.loads(record(tmp_path / 'r0', 'u1', 1, edit_face())),
                                              'content_sha256': '0' * 64}))
    with pytest.raises(ValueError, match="this round's cases"):
        c.round_(tmp_path / 'r0', stale, tmp_path / 'r1-stale')
    assert not (tmp_path / 'r1-stale').exists()
    assert c.round_(tmp_path / 'r0', answers, tmp_path / 'r1')['next_round_rooms'] == 1
    r1 = [json.loads(x) for x in (tmp_path / 'r1/rows.jsonl').read_text().splitlines()]
    assert [r['active'] for r in r1] == [True, False, False]
    assert [r['history'][-1]['stop'] for r in r1] == [None, 'no_valid_edits', 'call_failed']
    assert h.strict_json((tmp_path / 'r1/critic-mapping.json').read_text()) == {c.case_id('u1', 2): 'u1'}
    with pytest.raises(ValueError):                           # answers for cases outside the round are refused
        c.round_(tmp_path / 'r1', answers, tmp_path / 'r2-bad')
    with pytest.raises(ValueError):                           # u1 goes on: round 1 is not the final C
        c.evaluate(tmp_path / 'r0', tmp_path / 'r1', tmp_path / 'eval-r1')
    second = tmp_path / 'answers-2'
    second.mkdir()
    (second / 'a.json').write_text(record(tmp_path / 'r1', 'u1', 2, done(answer([], 6, 6))))
    c.round_(tmp_path / 'r1', second, tmp_path / 'r2')
    judge = tmp_path / 'judge'
    judge.mkdir()
    records = [{'uid': u, 'arm': arm, 'stratum': 's', 'status': 'completed',
                'scores': {'functionality_score': 5 + (arm == 'C' and u == 'u1'), 'layout_score': 5,
                           'aesthetics_score': 5}} for u in ('u1', 'u2', 'u3') for arm in ('B', 'C')]
    (judge / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    sha = lambda name: h.sha256(tmp_path / name / 'MANIFEST.json')
    wrong = c.judge_lineage(sha('r0'), sha('r1'))
    (judge / 'summary.json').write_text(json.dumps({'geometry_manifest_sha256': wrong}))
    with pytest.raises(ValueError, match='judge run was not made'):  # a judge run on another C is refused
        c.evaluate(tmp_path / 'r0', tmp_path / 'r2', tmp_path / 'eval', judge)
    bound = c.render.json_sha256({'B': sha('r0'), 'C': sha('r2')})          # judge_bundle_025.dev's provenance
    (judge / 'summary.json').write_text(json.dumps({'geometry_manifest_sha256': bound}))
    result = c.evaluate(tmp_path / 'r0', tmp_path / 'r2', tmp_path / 'eval', judge)
    value = h.strict_json((tmp_path / 'eval/results.json').read_text())
    assert value['rooms_edited'] == 1 and value['yaw_changed_objects'] == 1 and value['moved_objects'] == 0
    assert value['invariants'] == {'count': 0, 'id': 0, 'category': 0, 'size': 0, 'parent': 0}
    assert value['contrast_c_minus_b']['judge']['estimate'] == pytest.approx(0.5 / 3)
    assert value['contrast_c_minus_b']['geometry_success']['estimate'] == 0 and result['status'] == 'A025_EVALUATED'
    assert value['critic']['1']['edit_sets_accepted'] == 1 and value['critic']['2']['stops'] == {'no_valid_edits': 1}
    assert value['per_family'] == per_family and value['judge']['image_source'] == 'box_views'
    report = (tmp_path / 'eval/report.md').read_text()
    assert '修订 025' in report and '| bedroom | 3 |' in report and 'image_source = box_views' in report
    with pytest.raises(FileExistsError):
        c.evaluate(tmp_path / 'r0', tmp_path / 'r2', tmp_path / 'eval', judge)


def test_adoption_needs_all_three_rules_and_waits_for_the_judge():
    contrast = {'accepted': {'estimate': -0.004}, 'geometry_success': {'estimate': 0.0},
                'judge': {'estimate': 0.3, 'ci95': [0.05, 0.6]}}
    assert c.adoption(contrast)['adopted'] is True
    assert c.adoption({**contrast, 'judge': {'estimate': 0.3, 'ci95': [-0.01, 0.6]}})['adopted'] is False
    assert c.adoption({**contrast, 'judge': {'estimate': None, 'ci95': None}})['adopted'] is False
    assert c.adoption({**contrast, 'accepted': {'estimate': -0.006}})['adopted'] is False
    assert c.adoption({k: v for k, v in contrast.items() if k != 'judge'})['adopted'] is None
