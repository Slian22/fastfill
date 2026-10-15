"""Amendment 024 on a synthetic run 002: visual_pipeline's own fixture, a stub Blender, injected FID features."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fastfill.comparison import fid, judge_batch, mesh_collision, render_assets as render
from fastfill.comparison import visual_024 as v
from fastfill.comparison import visual_pipeline as vp
from fastfill.comparison.tests import test_visual_pipeline as T


def extract(images, **_):
    """One deterministic 2048-d feature per image path (the stub renders identical pixels)."""
    return np.stack([np.frombuffer(hashlib.sha256(r.path.encode()).digest(), np.uint8).astype(float).repeat(64)
                     for r in images])


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


@pytest.fixture
def world(tmp_path, monkeypatch):
    ref, geo, evaluation, cases = T.fixture(tmp_path)
    calls = []
    monkeypatch.setattr(vp.subprocess, 'run', T.fake_blender(calls))
    monkeypatch.setattr(fid, 'verify_weights', lambda path: fid.WEIGHTS_SHA256)
    h = tmp_path / 'H'
    base = h / 'visual-final-selected-002'
    vp.prepare_systems(ref, {'J': geo, 'P': geo}, base / 'visual', '/mock/blender', eval_data=evaluation)
    T.write(base / 'config.json', {'reference_data': str(ref), 'blender': '/mock/blender',
                                   'vendor': str(tmp_path), 'weights': str(tmp_path / 'w')})
    fid.recompute(base / 'visual', h / 'fid-011-visual-002' / 'fid.json', vendor=tmp_path, weights=tmp_path / 'w',
                  extractor=extract)
    mesh_collision.run(base, h / 'mesh-collision-019', workers=1, repeats=20)
    data = render.file_sha256(evaluation / 'MANIFEST.json')
    # Arm A: amendment-021-style placement rows, chair moved; room-1 incomplete (a recorded failure).
    a021 = h / 'a021-test-geometry-arm'
    T.rows(a021 / 'placement-geometry.jsonl', [
        dict(uid=c['uid'], arm='placement', source=c['source'], group='g', complete=c['uid'] != 'room-1',
             raw_layout=[{**c['layouts']['reference'][0], 'pos': [2.5, 2.5, 0]}]) for c in cases])
    T.write(a021 / 'summary.json', {'status': 'GEOMETRY_COMPLETE', 'data_manifest_sha256': data})
    T.seal(a021)
    # Arm S: fastfill.system.pipeline rooms.jsonl and the amendment-022 final test evaluation of it.
    test = h / 'system-022-final' / 'test'
    T.rows(test / 'system' / 'rooms.jsonl', [
        dict(uid=c['uid'], source=c['source'], group='g', room_type='bedroom', stage='placer', complete=True,
             checker_pass=True, candidates=[], placer={},
             layout=[{k: c['layouts']['reference'][0][k] for k in ('id', 'category', 'size', 'pos', 'yaw')}
                     | {'pos': [2.0, 3.0, 0], 'parent': None}]) for c in cases])
    T.write(test / 'eval' / 'results.json', {
        'mode': 'final', 'scope': 'test', 'label': 'system with free O2 escalation',
        'inputs': {'data_MANIFEST': data, 'system': {'rooms_sha256': render.file_sha256(test / 'system/rooms.jsonl')}}})
    return SimpleNamespace(h=h, base=base, out=h / 'visual-024', a=a021 / 'placement-geometry.jsonl',
                           s=test / 'system' / 'rooms.jsonl', calls=calls, cases=cases)


def prepared(world, arms=('A', 'S')):
    for arm in arms:
        v.prepare(arm, world.a if arm == 'A' else world.s, world.base, world.out)


def rendered(world, arms=('A', 'S')):
    prepared(world, arms)
    for arm in arms:
        v.render_arm(arm, world.out)


def test_prepare_reuses_run_002_rooms_and_scenes_and_is_write_once(world, capsys):
    argv = ['prepare', '--arm', 'A', '--rows', str(world.a), '--base', str(world.base), '--out', str(world.out)]
    assert v.main(argv) == 0
    a = world.out / 'A'
    record = json.loads((a / 'preparation.json').read_text())
    assert record['counts'] | {'failures': None} == {'frozen': 4, 'rows_in_cohort': 4, 'attached': 3, 'failed': 1,
                                                    'failures': None}
    assert record['scene_check'] == {'compared_with_run_002': 3, 'identical': True}
    assert record['provenance']['reference_manifest_sha256'] == render.file_sha256(
        world.h.parent / 'reference' / 'MANIFEST.json')
    assert [r['uid'] for r in lines(a / 'A-geometry.jsonl')] == [c['uid'] for c in world.cases]   # 'outside' dropped
    old = {c['uid']: c for c in lines(world.base / 'visual' / 'cases.jsonl')}
    for case in lines(a / 'cases.jsonl'):
        assert case['layouts']['reference'] == old[case['uid']]['layouts']['reference']
        assert ('A' in case['layouts']) == (case['uid'] != 'room-1')
    files = json.loads((a / 'MANIFEST.json').read_text())['files']
    assert set(files) == {'A-geometry.jsonl', 'cases.jsonl', 'attachment-ledger.jsonl', 'preparation.json'}
    assert v._prepared(a) == record
    with pytest.raises(SystemExit):
        v.main(argv)
    assert capsys.readouterr().err.strip().endswith('visual-024/A')
    with pytest.raises(ValueError, match='run-002 label'):
        v.prepare('P', world.a, world.base, world.out)
    # S: the system rows are bound to the evaluation that scored those exact bytes.
    s = v.prepare('S', world.s, world.base, world.out)
    assert s['kind'] == 'system' and s['counts']['attached'] == 4 and (world.out / 'S/source/MANIFEST.json').exists()
    assert lines(world.out / 'S/S-geometry.jsonl')[0]['raw_layout'][0]['pos'] == [2.0, 3.0, 0]
    world.s.write_text(world.s.read_text() + '\n')
    with pytest.raises(ValueError, match='is not the rooms.jsonl'):
        v.prepare('S2', world.s, world.base, world.out)
    assert not (world.out / 'S2').exists() and not list(world.out.glob('.*'))


def test_prepare_refuses_a_scene_or_group_that_differs_from_run_002(world):
    rows = lines(world.a)
    rows[0]['group'] = 'other'
    T.rows(world.a, rows)
    T.seal(world.a.parent)
    with pytest.raises(ValueError, match='house groups differ'):
        v.prepare('A', world.a, world.base, world.out)
    visual = world.base / 'visual'
    cases = lines(visual / 'cases.jsonl')
    cases[0]['height'] = 9
    T.rows(visual / 'cases.jsonl', cases)
    with pytest.raises(ValueError, match='not the file MANIFEST.json records'):
        v.prepare('A', world.a, world.base, world.out)


def test_render_renders_only_the_arm_with_the_run_002_command_and_never_twice(world):
    prepared(world, ('A',))
    before = len(world.calls)
    result = v.render_arm('A', world.out)
    assert result['counts'] == {'frozen': 4, 'rendered': 3, 'failed': 1} and len(world.calls) - before == 2
    old = json.loads((world.base / 'visual/chunks/00/process.json').read_text())['command']
    new = world.calls[-1][0]
    assert [x for x in new if '/' not in x] == [x for x in old if '/' not in x] and new[0] == old[0] == '/mock/blender'
    for chunk in sorted((world.out / 'A/chunks').glob('*/input.json')):
        assert all(set(c['layouts']) == {'A'} for c in json.loads(chunk.read_text())['cases'])
    ledger = {r['uid']: r for r in json.loads((world.out / 'A/ledger.json').read_text())['cases']}
    assert ledger['room-1']['status'] == 'attachment_failed' and ledger['room-1']['system_errors'] == {
        'A': 'schema_incomplete'}
    assert set(ledger['room-0']['images']) == {'A'} and set(ledger['room-0']['images']['A']) == {'perspective', 'top'}
    assert [r['uid'] for r in lines(world.out / 'A/failures.jsonl')] == ['room-1']
    with pytest.raises(FileExistsError):
        v.render_arm('A', world.out)


def test_fid_adds_the_arms_on_the_amendment_011_rule_and_reproduces_fid_011(world):
    rendered(world)
    summary = v.fid_arms(world.out, extract=extract)
    assert summary['reproduces_fid_011'] and summary['not_reproduced'] == []
    pooled = summary['populations']['pooled']
    assert set(pooled) == {'primary', 'agent/A', 'agent/S', 'judged'}
    assert (pooled['agent/A']['n'], pooled['agent/S']['n'], pooled['primary']['n']) == (3, 4, 4)
    assert pooled['agent/A']['systems'] == ['J', 'P', 'A'] and pooled['agent/A']['fid']['A'] > 0
    assert summary['features']['reused'] == 12 and summary['features']['extracted'] == 7   # 4 x ref/J/P; 3 A + 4 S
    result = json.loads((world.out / 'fid-A-S/fid.json').read_text())
    assert result['systems'] == ['J', 'P', 'A', 'S'] and result['judged'] == ['J', 'P']
    assert {(f['uid'], f['arm']) for f in result['failures']} == {('room-1', 'A')}
    with pytest.raises(FileExistsError):
        v.fid_arms(world.out, extract=extract)
    with pytest.raises(ValueError, match='other extraction settings'):
        v.fid_arms(world.out, ['A'], threads=2, extract=extract)


def test_fid_and_bundle_refuse_a_limited_smoke_render(world):
    prepared(world)
    v.render_arm('A', world.out, limit=1)
    v.render_arm('S', world.out)
    assert json.loads((world.out / 'A/ledger.json').read_text())['expected_uids'] == ['room-0']
    with pytest.raises(ValueError, match='smoke render'):
        v.fid_arms(world.out, extract=extract)
    with pytest.raises(ValueError, match='smoke render'):
        v.bundle(world.out)
    with pytest.raises(ValueError, match='positive integer'):
        v.render_arm('S', world.out, limit=0)


def test_mesh_runs_amendment_019_with_the_arms_and_reproduces_gt_and_p(world):
    prepared(world)
    systems = mesh_collision.SYSTEMS
    result = v.mesh_arms(world.out, workers=1, repeats=20)
    assert mesh_collision.SYSTEMS == systems
    assert result['reproduces_019'] and result['analysed_rooms'] == 4 and set(result['mesh_free_500']) == {'A', 'S'}
    summary = json.loads((world.out / 'mesh-A-S/collision/summary.json').read_text())
    assert summary['systems_present'] == ['GT', 'P', 'A', 'S'] and set(summary['paired_vs_P']) == {'GT', 'A', 'S'}
    assert summary['systems']['A']['rooms'] == 3 and summary['systems']['A']['missing_rooms'] == 1
    assert (summary['pitch_m'], summary['sample_spacing_m'], summary['primary_tau_cm3']) == (0.02, 0.005, 500)
    reproduction = json.loads((world.out / 'mesh-A-S/reproduction-019.json').read_text())
    assert reproduction['compared_rows'] == 8 and reproduction['differing'] == []


def test_bundle_is_run_002_format_for_its_judge_rooms_and_passes_judge_batch_preflight(world):
    rendered(world)
    with pytest.raises(ValueError, match='at least two arms'):
        v.bundle(world.out, ['A'])
    result = v.bundle(world.out)
    dest = world.out / 'bundle-A-S'
    mapping = json.loads((dest / 'api-mapping.json').read_text())
    failures = json.loads((dest / 'api-failures.json').read_text())
    api = json.loads((dest / 'api-bundle/manifest.json').read_text())
    assert result['counts'] == {'selected': 8, 'api_cases': 7, 'failed': 1}
    assert {(m['uid'], m['arm']) for m in mapping.values()} == {(f'room-{i}', a) for i in range(4) for a in 'AS'}
    assert [(f['uid'], f['arm'], f['failure']['stage']) for f in failures] == [('room-1', 'A', 'attachment')]
    assert api['expected_model'] == vp.MODEL and all(len(c['images']) == 2 for c in api['cases'])
    old = json.loads((world.base / 'visual/api-bundle/manifest.json').read_text())
    assert {c['prompt_sha256'] for c in api['cases']} == {c['prompt_sha256'] for c in old['cases']}
    provenance = json.loads((dest / 'api-bundle/provenance.json').read_text())
    assert provenance['geometry_manifest_sha256'] == render.json_sha256(
        {a: json.loads((world.out / a / 'preparation.json').read_text())['lineage'][a] for a in 'AS'})
    client = judge_batch._client(judge_batch.Path(judge_batch.__file__).with_name('server_api_judge.py'))
    rows = judge_batch._inputs(dest, None, None, client, 400, 8, ('A', 'S'))[2]
    assert len(rows) == 8
    with pytest.raises(FileExistsError):
        v.bundle(world.out)
