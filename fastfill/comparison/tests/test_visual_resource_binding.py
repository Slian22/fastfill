"""VIS-01: real synthetic freezes and asset inspection, mocked RGB rendering only."""
import json
from pathlib import Path

import pytest

from fastfill.comparison import render_assets as render, visual_cases
from fastfill.comparison import visual_pipeline as pipeline
from fastfill.comparison.tests import test_visual_cases as freeze_tests
from fastfill.comparison.tests import test_visual_pipeline as pipeline_tests


def read(path):
    return json.loads(path.read_text())


def reseal(root):
    metadata = {k: v for k, v in read(root / 'MANIFEST.json').items() if k != 'files'}
    pipeline_tests.seal(root, **metadata)


def cohort(tmp_path, n=3, shared=False, override=False, fixed=False, procedural=False):
    rooms = [freeze_tests.room(f'room-{i}') for i in range(n)]
    if fixed:
        rooms[0]['fixed'] = [dict(id='fixed-chair', category='cabinet', size=[1, 1, 1],
                                  pos=[0, 0, 0], yaw=0)]
    if procedural:
        rooms[0]['fixed'] = [dict(id='door', category='door', structure=True, size=[1, .1, 2],
                                  pos=[0, 0, 0], yaw=0)]
    export, receipt, mapping = freeze_tests.fixture(tmp_path, rooms)
    assets, paths = [], {}
    for i, room in enumerate(rooms):
        for obj in room['objects'] + room['fixed']:
            name = f'{i}-{obj["id"]}'
            directory = tmp_path / 'assets' / name
            directory.mkdir(parents=True)
            mesh, material, texture = [directory / f'mesh.{ext}' for ext in ('obj', 'mtl', 'png')]
            asset_id = name
            if override and i == 0:
                asset_id = 'f350c1ba-8040-4750-9e6e-f45cfbf916a4'
                material = directory / 'raw_model.mtl'
            library = 'model.mtl' if override and i == 0 else material.name
            mesh.write_text(f'mtllib {library}\nv 0 0 0\nv 1 0 0\nv 0 1 1\nusemtl m\nf 1 2 3\n')
            material.write_text(f'newmtl m\nKd 1 1 1\nmap_Kd {texture.name}\n')
            texture.write_bytes(b'frozen-synthetic-texture')
            paths[name] = dict(obj=mesh, mtl=material, texture=texture)
            assets.append(dict(uid=room['uid'], original_instance_id=obj['id'],
                asset_id=asset_id, mesh_path=str(mesh), source=room['source'],
                role='fixed' if obj in room['fixed'] else 'movable', layout_object={'scale': [1, 1, 1]}))
    if shared:
        assets[1] = {**assets[1], **{k: assets[0][k] for k in ('asset_id', 'mesh_path')}}
    pipeline_tests.write(mapping, assets)
    ref, geo, evaluation = [tmp_path / name for name in ('reference', 'geometry', 'eval')]
    visual_cases.freeze_reference(export, receipt, mapping, ref)
    cases = visual_cases._read_rows(ref / 'reference-cases.jsonl')
    assert all(c['reference_eligible'] for c in cases)
    pipeline_tests.rows(evaluation / 'rooms.jsonl', [{'uid': c['uid']} for c in cases])
    pipeline_tests.seal(evaluation, expected_uids=[c['uid'] for c in cases],
                        corpus_manifest_sha256='b' * 64)
    pipeline_tests.write(geo / 'summary.json',
                         {'data_manifest_sha256': render.file_sha256(evaluation / 'MANIFEST.json')})
    for arm in pipeline.ARMS[1:]:
        pipeline_tests.rows(geo / f'{arm}-geometry.jsonl', [dict(uid=c['uid'], arm=arm,
            source=c['source'], group='test-group', complete=True, raw_layout=c['layouts']['reference'])
            for c in cases])
    pipeline_tests.seal(geo)
    return ref, geo, evaluation, paths


def prepare(tmp_path, monkeypatch, inputs, during=None, statuses=None, corrupt=None, timeout=False):
    calls = []
    fake = pipeline_tests.fake_blender(calls, statuses=statuses, corrupt=corrupt, timeout=timeout)

    def run(*args, **kwargs):
        result = fake(*args, **kwargs)
        if during:
            during()
        return result

    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    ref, geo, evaluation, _ = inputs
    out = tmp_path / 'visual'
    result = pipeline.prepare(ref, geo, out, '/mock/blender', workers=1, eval_data=evaluation)
    return out, result, calls


def assert_inventory(out, result, ref, failed):
    uids = read(ref / 'MANIFEST.json')['expected_uids']
    selected = read(ref / 'gpt-selection.json')['selected_uids']
    ledger = read(out / 'ledger.json')['cases']
    assert result['expected_uids'] == uids
    assert result['selected_uids'] == selected
    assert result['counts'] == dict(frozen=len(uids), rendered=len(uids) - len(failed),
        failed=len(failed), gpt_selected=len(selected), api_cases=2 * len(set(selected) - failed))
    assert [c['uid'] for c in visual_cases._read_rows(out / 'cases.jsonl')] == uids
    assert [c['uid'] for c in ledger] == uids
    assert {c['uid'] for c in ledger if c['failure']} == failed
    assert all(c['images'] == {} for c in ledger if c['uid'] in failed)
    fid = read(out / 'fid-manifest.json')
    assert fid['frozen_uids'] == uids
    assert [c['uid'] for c in fid['cases']] == uids
    assert all(c['failure'] and c['images'] == {} for c in fid['cases'] if c['uid'] in failed)
    mapping = read(out / 'api-mapping.json')
    assert mapping == {render.json_sha256([uid, arm]): dict(uid=uid, arm=arm,
        stratum='IL3D_3dfront/bedroom', group='test-group') for uid in selected for arm in pipeline.ARMS[1:]}
    requests = read(out / 'api-bundle/manifest.json')['cases']
    assert {mapping[c['id']]['uid'] for c in requests} == set(selected) - failed
    assert all(set(c) == {'id', 'prompt_file', 'prompt_sha256', 'images'} for c in requests)
    failures = read(out / 'api-failures.json')
    assert {c['uid'] for c in failures} == set(selected) & failed
    assert len(failures) == 2 * len(set(selected) & failed)
    assert {c['uid'] for c in visual_cases._read_rows(out / 'failures.jsonl')} == failed
    provenance = read(out / 'api-bundle/provenance.json')
    assert provenance['reference_manifest_sha256'] == render.file_sha256(ref / 'MANIFEST.json')
    assert provenance['corpus_link'] == 'verified'
    for name, digest in read(out / 'MANIFEST.json')['files'].items():
        assert render.file_sha256(out / name) == digest


def test_current_frozen_schema_accepts_unchanged_resources(tmp_path, monkeypatch):
    inputs = cohort(tmp_path)
    before = {p: p.read_bytes() for p in inputs[0].iterdir()}
    out, result, calls = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, inputs[0], set())
    assert len(calls) == 1
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('phase', ['before_render', 'after_render'])
@pytest.mark.parametrize('kind', ['obj', 'mtl', 'texture'])
@pytest.mark.parametrize('missing', [False, True], ids=['changed', 'missing'])
def test_resource_drift_fails_whole_room_and_keeps_other_rooms(
        tmp_path, monkeypatch, phase, kind, missing):
    inputs = cohort(tmp_path)
    path = inputs[3]['0-original-a'][kind]

    def drift():
        if missing:
            path.unlink()
        else:
            # Still parsable resources; not a renderer syntax rejection.
            replacements = {'obj': (b'v 1 0 0', b'v 2 0 0'),
                            'mtl': (b'Kd 1 1 1', b'Kd 0 1 1'),
                            'texture': (b'frozen', b'change')}
            path.write_bytes(path.read_bytes().replace(*replacements[kind]))

    if phase == 'before_render':
        drift()
    out, result, calls = prepare(tmp_path, monkeypatch, inputs,
                                 during=drift if phase == 'after_render' else None)
    assert_inventory(out, result, inputs[0], {'room-0'})
    failed = read(out / 'ledger.json')['cases'][0]
    assert failed['status'] == 'resource_failed'
    assert failed['failure']['stage'] == 'resource_binding'
    assert failed['failure']['phase'] == phase
    submitted = read(Path(calls[0][0][calls[0][0].index('--manifest') + 1]))['cases']
    assert ('room-0' in {c['uid'] for c in submitted}) == (phase == 'after_render')


@pytest.mark.parametrize('fault', ['absent', 'unbound', 'hash_mismatch'])
def test_resources_manifest_must_be_bound_before_render(tmp_path, monkeypatch, fault):
    inputs = cohort(tmp_path)
    ref = inputs[0]
    if fault == 'hash_mismatch':
        (ref / 'resources.json').write_text('{}')
    else:
        manifest = read(ref / 'MANIFEST.json')
        pipeline_tests.write(ref / 'MANIFEST.json', {**manifest,
            'files': {k: v for k, v in manifest['files'].items() if k != 'resources.json'}})
        if fault == 'absent':
            (ref / 'resources.json').unlink()
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('render launched'))
    with pytest.raises(ValueError):
        pipeline.prepare(ref, inputs[1], tmp_path / 'visual', '/mock')
    assert not (tmp_path / 'visual').exists()


@pytest.mark.parametrize('fault', ['missing', 'wrong_identity', 'missing_material', 'missing_texture', 'override'])
def test_incomplete_frozen_resource_set_is_room_failure(tmp_path, monkeypatch, fault):
    inputs = cohort(tmp_path)
    ref = inputs[0]
    resources = read(ref / 'resources.json')
    key = next(k for k, r in resources.items() if r['asset_id'] == '0-original-a')
    entry = resources[key]
    if fault == 'missing':
        resources = {k: v for k, v in resources.items() if k != key}
    else:
        changes = {'wrong_identity': {'asset_id': 'other'}, 'missing_material': {'materials': []},
            'missing_texture': {'materials': [{**entry['materials'][0], 'texture_sha256': {}}]},
            'override': {'overrides': {'mesh.mtl': 'other.mtl'}}}
        resources = {**resources, key: {**entry, **changes[fault]}}
    pipeline_tests.write(ref / 'resources.json', resources)
    reseal(ref)
    out, result, _ = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, ref, {'room-0'})


@pytest.mark.parametrize('phase', ['before_render', 'after_render'])
def test_shared_resource_checks_are_fresh_and_fail_all_dependent_uids(tmp_path, monkeypatch, phase):
    inputs = cohort(tmp_path, shared=True)
    path = inputs[3]['0-original-a']['texture']
    drift = lambda: path.write_bytes(b'changed')
    if phase == 'before_render':
        drift()
    out, result, _ = prepare(tmp_path, monkeypatch, inputs,
                             during=drift if phase == 'after_render' else None)
    assert_inventory(out, result, inputs[0], {'room-0', 'room-1'})


def test_fixed_furniture_resources_are_bound(tmp_path, monkeypatch):
    inputs = cohort(tmp_path, fixed=True)
    inputs[3]['0-fixed-chair']['texture'].unlink()
    out, result, _ = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, inputs[0], {'room-0'})


def test_override_resource_set_cannot_silently_switch_material(tmp_path, monkeypatch):
    inputs = cohort(tmp_path, override=True)
    material = inputs[3]['0-original-a']['mtl']
    # The previously absent preferred library appears with identical bytes.
    (material.parent / 'model.mtl').write_bytes(material.read_bytes())
    out, result, _ = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, inputs[0], {'room-0'})


def test_all_resources_failed_keeps_full_inventory_without_launch(tmp_path, monkeypatch):
    inputs = cohort(tmp_path)
    for paths in inputs[3].values():
        paths['obj'].unlink()
    out, result, calls = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, inputs[0], {'room-0', 'room-1', 'room-2'})
    assert calls == [] and result['chunks'] == []


def test_render_failures_still_count_with_resource_failures(tmp_path, monkeypatch):
    inputs = cohort(tmp_path)
    inputs[3]['0-original-a']['texture'].unlink()
    out, result, _ = prepare(tmp_path, monkeypatch, inputs, statuses={'room-1': 'failed'})
    assert_inventory(out, result, inputs[0], {'room-0', 'room-1'})
    assert read(out / 'ledger.json')['cases'][1]['failure']['stage'] == 'render'


def test_failed_selected_room_is_not_replaced_by_unselected_success(tmp_path, monkeypatch):
    inputs = cohort(tmp_path, n=30)
    selected = read(inputs[0] / 'gpt-selection.json')['selected_uids']
    uid = selected[0]
    inputs[3][uid.removeprefix('room-') + '-original-a']['texture'].unlink()
    out, result, _ = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, inputs[0], {uid})
    assert result['counts']['gpt_selected'] == 29
    assert result['counts']['rendered'] == 29 and result['counts']['api_cases'] == 56


@pytest.mark.parametrize('as_dict', [False, True])
def test_relative_mapping_paths_bind_to_existing_absolute_freeze(tmp_path, monkeypatch, as_dict):
    inputs = cohort(tmp_path)
    ref = inputs[0]
    cases = visual_cases._read_rows(ref / 'reference-cases.jsonl')
    updated = []
    for case in cases:
        rows = [{**r, 'mesh_path': '../' + str(Path(r['mesh_path']).relative_to(tmp_path))}
                for r in case['asset_mapping']]
        mapping = {r['prompt_id']: r for r in rows} if as_dict else rows
        updated.append({**case, 'asset_mapping': mapping})
    pipeline_tests.rows(ref / 'reference-cases.jsonl', updated)
    reseal(ref)
    out, result, _ = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, ref, set())


def test_unchanged_documented_override_is_accepted(tmp_path, monkeypatch):
    inputs = cohort(tmp_path, override=True)
    out, result, _ = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, inputs[0], set())


def test_procedural_structure_does_not_require_unused_mapping_resource(tmp_path, monkeypatch):
    inputs = cohort(tmp_path, procedural=True)
    inputs[3]['0-door']['obj'].unlink()
    out, result, _ = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, inputs[0], set())


def test_invalid_resources_inventory_rejected_before_output(tmp_path, monkeypatch):
    inputs = cohort(tmp_path)
    pipeline_tests.write(inputs[0] / 'resources.json', [])
    reseal(inputs[0])
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('render launched'))
    with pytest.raises(ValueError, match='resources inventory'):
        pipeline.prepare(inputs[0], inputs[1], tmp_path / 'visual', '/mock')
    assert not (tmp_path / 'visual').exists()


@pytest.mark.parametrize('kind', ['hash', 'camera', 'view', 'layout', 'input'])
def test_image_validation_still_excludes_all_arms_with_bound_resources(tmp_path, monkeypatch, kind):
    def corrupt(receipt):
        changes = {
            'hash': {'image_sha256': {**receipt['image_sha256'], 'joint': {
                **receipt['image_sha256']['joint'], 'top': '0' * 64}}},
            'camera': {'cameras': {}},
            'view': {'images': {**receipt['images'], 'placement': {}}},
            'layout': {'layout_sha256': {}}, 'input': {'input_sha256': '0' * 64}}
        return {**receipt, **changes[kind]}
    inputs = cohort(tmp_path)
    out, result, _ = prepare(tmp_path, monkeypatch, inputs, corrupt=corrupt)
    assert_inventory(out, result, inputs[0], {'room-0', 'room-1', 'room-2'})
    assert all(c['failure']['stage'] == 'image_validation' for c in read(out / 'ledger.json')['cases'])


def test_partial_timeout_keeps_render_and_resource_failure_denominators(tmp_path, monkeypatch):
    inputs = cohort(tmp_path)
    inputs[3]['0-original-a']['texture'].unlink()
    out, result, _ = prepare(tmp_path, monkeypatch, inputs, statuses={'room-1': 'missing'}, timeout=True)
    assert_inventory(out, result, inputs[0], {'room-0', 'room-1'})
    assert result['chunks'][0]['status'] == 'timeout'
    assert [c['status'] for c in read(out / 'ledger.json')['cases']] == ['resource_failed', 'missing', 'rendered']


def test_attachment_failures_preserved_alongside_resource_failures(tmp_path, monkeypatch):
    inputs = cohort(tmp_path)
    path = inputs[1] / 'joint-geometry.jsonl'
    pipeline_tests.rows(path, [r for r in visual_cases._read_rows(path) if r['uid'] != 'room-1'])
    reseal(inputs[1])
    inputs[3]['0-original-a']['texture'].unlink()
    out, result, _ = prepare(tmp_path, monkeypatch, inputs)
    assert_inventory(out, result, inputs[0], {'room-0', 'room-1'})
    assert read(out / 'ledger.json')['cases'][1]['failure']['errors']['joint'] == 'missing_output'


def test_api_bundle_remains_bound_to_geometry_provenance(tmp_path, monkeypatch):
    inputs = cohort(tmp_path)
    first, _, _ = prepare(tmp_path, monkeypatch, inputs)
    summary = inputs[1] / 'summary.json'
    pipeline_tests.write(summary, {**read(summary), 'inference_run_sha256': {'joint': 'f' * 64}})
    reseal(inputs[1])
    second = tmp_path / 'second'
    pipeline.prepare(inputs[0], inputs[1], second, '/mock', workers=1, eval_data=inputs[2])
    assert (first / 'api-bundle/manifest.json').read_bytes() == (second / 'api-bundle/manifest.json').read_bytes()
    old, new = [read(root / 'api-bundle/provenance.json') for root in (first, second)]
    assert old['geometry_manifest_sha256'] != new['geometry_manifest_sha256']
    assert new['geometry_manifest_sha256'] == render.file_sha256(inputs[1] / 'MANIFEST.json')
