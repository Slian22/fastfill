"""Amendment-025 judge bundles: offline, synthetic rooms, fake views and judge_batch's fake client subprocess."""
import json
from pathlib import Path

import pytest

from fastfill.comparison import dpo_stages as dps, eval_data, judge_batch, judge_bundle_025 as jb
from fastfill.comparison import render_assets as render
from fastfill.comparison.tests.test_judge_batch import CLIENT, fake_client

ROOMS = [{'uid': f'src{i % 2}::room-{i}', 'source': f'src{i % 2}', 'group': f'house-{i // 2}', 'split': 'dev',
          'room_type': 'bedroom', 'boundary_type': 'polygon', 'boundary': [[0, 0], [4, 0], [4, 3], [0, 3]],
          'height': 2.6, 'fixed': [{'id': 'fixed_0001', 'category': 'door', 'pos': [0, 1, 0], 'yaw': 0.0,
                                    'size': [0.1, 0.9, 2.0], 'fixed_kind': 'structure'}],
          'objects': [{'id': 'obj_0001', 'category': 'bed', 'size': [2, 1.6, 0.5], 'parent': None},
                      {'id': 'obj_0002', 'category': 'lamp', 'size': [0.3, 0.3, 0.4], 'parent': 'obj_0001'}]}
         for i in range(3)]


def state(path, rooms, x, summary, active=False):
    """A critic_025 state dir (rows.jsonl, summary.json, MANIFEST.json) with every layout at pos x."""
    rows = [{'uid': r['uid'], 'source': r['source'], 'group': r['group'], 'active': active, 'history': [],
             'room': {k: r[k] for k in ('uid', 'source', 'group', 'room_type', 'boundary_type', 'boundary', 'height',
                                        'fixed')},
             'layout': [{**o, 'pos': [x, 1.5, 0.5 * (o['parent'] is not None)], 'yaw': 0.0} for o in r['objects']]}
            for r in rooms]
    dps._publish_dir(path, {'rows.jsonl': dps._jsonl(rows), 'summary.json': json.dumps(summary).encode()})
    return path, render.file_sha256(path / 'MANIFEST.json')


def dev_dir(tmp_path):
    root = tmp_path / 'dev'
    root.mkdir()
    (root / 'MANIFEST.json').write_text(json.dumps({'main_data_manifest_sha256': 'e' * 64, 'files': {}}))
    return root


def build(tmp_path, views=jb._views, c_rooms=ROOMS):
    data = dev_dir(tmp_path)
    b, b_sha = state(tmp_path / 'round-0', ROOMS, 1.5, {
        'round': 0, 'b_manifest_sha256': None, 'inputs': {'data_manifest_sha256': render.file_sha256(
            data / 'MANIFEST.json')}})
    c, _ = state(tmp_path / 'round-2', c_rooms, 2.5, {'round': 2, 'b_manifest_sha256': b_sha})
    return jb.dev(data, b, c, tmp_path / 'bundle', views=views), tmp_path / 'bundle'


def test_dev_bundle_runs_through_judge_batch_blind_and_paired(tmp_path, monkeypatch):
    seen = []

    def views(task):
        seen.append((task[0]['uid'], task[1][0]['pos'][0], task[2]))
        return jb._views(task)
    result, root = build(tmp_path, views)
    assert result['image_source'] == 'box_views' and result['provenance']['image_source'] == 'box_views'
    assert result['counts'] == {'rooms': 3, 'selected': 6, 'api_cases': 6, 'failed': 0}
    assert result['judge_batch'].startswith('--arms C,B --max-cases 6 --expected-total 6 ')
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    assert manifest['disclosures'] == jb.DISCLOSURES and any('neutral' in d for d in jb.DISCLOSURES)
    shas = {a: render.file_sha256(tmp_path / f'round-{k}/MANIFEST.json') for a, k in (('B', 0), ('C', 2))}
    assert result['provenance']['geometry_manifest_sha256'] == render.json_sha256(shas)  # path-free
    # each arm's own layout; one title per room, the same for both arms
    assert sorted(seen) == sorted((r['uid'], x, r['uid'][-48:]) for r in ROOMS for x in (1.5, 2.5))
    mapping = json.loads((root / 'api-mapping.json').read_text())
    assert list(mapping) == sorted(mapping)
    by_room = {r['uid']: r for r in ROOMS}
    for cid, meta in mapping.items():
        assert meta['stratum'] == by_room[meta['uid']]['source'] and meta['group'] == by_room[meta['uid']]['group']
        assert meta['arm'] not in cid and meta['uid'] not in cid and len(cid) == 64  # opaque ids
    manifest = json.loads((root / 'api-bundle/manifest.json').read_text())
    images = {}
    for case in manifest['cases']:
        assert [Path(i['path']).name for i in case['images']] == ['perspective.png', 'top.png']
        assert all(case['id'] in i['path'] for i in case['images'])
        images[mapping[case['id']]['uid'], mapping[case['id']]['arm']] = [i['sha256'] for i in case['images']]
        prompt = (root / 'api-bundle' / case['prompt_file']).read_text()
        assert prompt.startswith('Evaluate the same room layout') and '"bedroom"' in prompt
        assert '"count": 1' in prompt and '"door"' in prompt
    assert all(images[r['uid'], 'B'][k] != images[r['uid'], 'C'][k] for r in ROOMS for k in (0, 1))
    calls = []
    monkeypatch.setattr(judge_batch.subprocess, 'run', fake_client(calls))
    summary = judge_batch.evaluate(root, tmp_path / 'judge', CLIENT, max_cases=6, expected_total=6, arms=('C', 'B'))
    assert sorted(calls) == sorted(mapping) and summary['counts']['success'] == 6
    assert summary['endpoint']['direction'] == 'C_minus_B' and summary['endpoint']['both_complete'] == 3
    records = [json.loads(line) for line in (tmp_path / 'judge/records.jsonl').read_text().splitlines()]
    assert {(r['uid'], r['arm'], r['group']) for r in records} == {
        (r['uid'], a, r['group']) for r in ROOMS for a in ('B', 'C')}


def test_views_failure_is_recorded_not_replaced(tmp_path, monkeypatch):
    def views(task):
        if task[0]['uid'] == ROOMS[1]['uid'] and task[1][0]['pos'][0] == 2.5:
            return None, 'ValueError: degenerate'
        return jb._views(task)
    result, root = build(tmp_path, views)
    assert result['counts'] == {'rooms': 3, 'selected': 6, 'api_cases': 5, 'failed': 1}
    failures = json.loads((root / 'api-failures.json').read_text())
    assert [(f['uid'], f['arm'], f['failure']['stage']) for f in failures] == [(ROOMS[1]['uid'], 'C', 'box_views')]
    assert not (root / 'api-bundle/images' / failures[0]['id']).exists()
    monkeypatch.setattr(judge_batch.subprocess, 'run', fake_client([]))
    summary = judge_batch.evaluate(root, tmp_path / 'judge', CLIENT, max_cases=6, expected_total=6, arms=('C', 'B'))
    assert summary['counts']['failed'] == 1 and summary['endpoint']['both_complete'] == 2


def test_real_views_failure_is_caught():
    room = {**ROOMS[0], 'boundary': [[0, 0]]}
    images, reason = jb._views((room, [{**o, 'pos': [1, 1, 0], 'yaw': 0.0} for o in ROOMS[0]['objects']], 'x'))
    assert images is None and reason.startswith('ValueError')


def test_dev_refuses_unpaired_states_and_existing_output(tmp_path):
    with pytest.raises(ValueError, match='built on this B'):
        build(tmp_path, c_rooms=ROOMS[:2])
    (tmp_path / 'y').mkdir()
    with pytest.raises(FileExistsError):
        jb.dev(tmp_path / 'dev', tmp_path / 'round-0', tmp_path / 'round-2', tmp_path / 'y')
    b_sha = render.file_sha256(tmp_path / 'round-0/MANIFEST.json')
    c, _ = state(tmp_path / 'round-1', ROOMS, 2.5, {'round': 1, 'b_manifest_sha256': b_sha}, active=True)
    with pytest.raises(ValueError, match='not final'):
        jb.dev(tmp_path / 'dev', tmp_path / 'round-0', c, tmp_path / 'z')
    assert not (tmp_path / 'z').exists()


def s_dir(tmp_path):
    """A visual_024-prepared arm S dir, as far as sc() reads it."""
    s = tmp_path / 'S'
    source = s / 'source'
    source.mkdir(parents=True)
    rows = [{'uid': r['uid'], 'source': r['source'], 'group': r['group'], 'room_type': 'bedroom', 'arm': 'system',
             'stage': 's1', 'complete': True, 'raw_layout': [{**o, 'pos': [1.0, 1.0, 0], 'yaw': 0.0}
                                                             for o in r['objects']]} for r in ROOMS]
    (source / 'system-geometry.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (source / 'summary.json').write_text(json.dumps({'data_manifest_sha256': 'f' * 64}))
    names = ('system-geometry.jsonl', 'summary.json')
    (source / 'MANIFEST.json').write_text(json.dumps({'files': {n: render.file_sha256(source / n) for n in names}}))
    (s / 'preparation.json').write_text(json.dumps({'selected_uids': [ROOMS[0]['uid']]}))
    files = ['preparation.json', *('source/' + n for n in (*names, 'MANIFEST.json'))]
    (s / 'MANIFEST.json').write_text(json.dumps({'files': {n: render.file_sha256(s / n) for n in files}}))
    return s


def test_sc_rows_enter_visual_024_in_s_order(tmp_path):
    critic = [{'uid': ROOMS[0]['uid'], 'layout': [{**o, 'pos': [2.5, 1.0, 0], 'yaw': 1.0}
                                                   for o in reversed(ROOMS[0]['objects'])]}]
    (tmp_path / 'critic.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in critic))
    record = jb.sc(tmp_path / 'critic.jsonl', s_dir(tmp_path), tmp_path / 'SC')
    assert record['data_manifest_sha256'] == 'f' * 64 and record['rooms'] == 1
    eval_data._snapshot(tmp_path / 'SC')  # the sealed-dir check visual_pipeline._inputs runs
    [row] = [json.loads(line) for line in (tmp_path / 'SC/SC-geometry.jsonl').read_text().splitlines()]
    assert row['arm'] == 'SC' and row['group'] == ROOMS[0]['group'] and row['complete'] is True
    assert [o['id'] for o in row['raw_layout']] == ['obj_0001', 'obj_0002']
    assert {o['pos'][0] for o in row['raw_layout']} == {2.5}


@pytest.mark.parametrize('change', ['size', 'parent', 'missing', 'foreign', 'duplicate', 'judge_room'])
def test_sc_refuses_changed_objects_or_rooms(tmp_path, change):
    layout = [{**o, 'pos': [2.0, 1.0, 0], 'yaw': 0} for o in ROOMS[0]['objects']]
    uid = ROOMS[0]['uid']
    if change == 'size':
        layout[0] = {**layout[0], 'size': [1.9, 1.6, 0.5]}
    if change == 'parent':
        layout[1] = {**layout[1], 'parent': None}
    if change == 'missing':
        layout = layout[:1]
    if change == 'foreign':
        uid = 'elsewhere::room'
    if change == 'duplicate':
        layout = layout + [{**layout[1], 'pos': [3.0, 1.0, 0]}]
    if change == 'judge_room':  # a valid row, but not for S's judge room ROOMS[0]
        uid, layout = ROOMS[1]['uid'], [{**o, 'pos': [2.0, 1.0, 0], 'yaw': 0} for o in ROOMS[1]['objects']]
    (tmp_path / 'critic.jsonl').write_text(json.dumps({'uid': uid, 'raw_layout': layout}) + '\n')
    with pytest.raises(ValueError):
        jb.sc(tmp_path / 'critic.jsonl', s_dir(tmp_path), tmp_path / 'SC')
    assert not (tmp_path / 'SC').exists()
