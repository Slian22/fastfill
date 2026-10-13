"""Synthetic reference assets only; no Blender, models, or remote calls."""
import copy
import gzip
import json
import math
from pathlib import Path


def test_unknown_room_type_does_not_abort_other_source_inventory():
    from fastfill.comparison import visual_cases
    for value in ('unknown', 'misc', 'other', None, ''):
        assert visual_cases._stratum({'source': 'HSSD200', 'room_type': value}) == 'HSSD200/'

import pytest

from fastfill.comparison import contract, eval_data, render_assets
from fastfill.comparison import visual_cases as v


def room(uid='r', source='IL3D_3dfront', kind='bedroom'):
    return dict(uid=uid, source=source, split='test', room_type=kind,
                boundary_type='polygon', boundary=[[0, 0], [5, 0], [5, 5], [0, 5]],
                height=3, fixed=[], objects=[dict(id='original-a', category='chair',
                size=[1.1234, 2.2345, 3.3456], pos=[1.1234, 2, 0], yaw=.12345)])


def write_json(path, data):
    path.write_text(json.dumps(data))


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def fixture(tmp_path, rooms=None):
    rooms = rooms if rooms is not None else [room()]
    mesh = tmp_path / 'mesh.obj'
    mesh.write_text('mtllib mesh.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 1\nusemtl m\nf 1 2 3\n')
    (tmp_path / 'mesh.mtl').write_text('newmtl m\nKd 1 1 1\n')
    export, receipt, mapping = [tmp_path / n for n in ('export.jsonl.gz', 'receipt.json', 'map.json')]
    with gzip.open(export, 'wt') as stream:
        for r in rooms:
            stream.write(json.dumps(r) + '\n')
    write_json(receipt, dict(rows=len(rooms), gzip_sha256=v.sha256(export),
                            corpus_manifest_sha256='b'*64, consumed_prepared_sha256='c'*64))
    write_json(mapping, [dict(uid=r['uid'], original_instance_id=o['id'], prompt_id='OLD-WRONG',
               asset_id='mesh', mesh_path=str(mesh), source=r['source'], role='movable',
               layout_object={'scale': [-1, 1, 1]}, gt_object={'size': [99, 99, 99]})
               for r in rooms for o in r['objects']])
    return export, receipt, mapping


def freeze(tmp_path, rooms=None):
    args = fixture(tmp_path, rooms)
    out = tmp_path / 'out'
    manifest = v.freeze_reference(*args, out)
    return out, manifest, rows(out / 'reference-cases.jsonl')


def predictions(tmp_path, cases, arm, **changes):
    path = tmp_path / (arm + '.jsonl')
    path.write_text(''.join(json.dumps(dict(uid=c['uid'], source=c['source'], arm=arm,
        complete=True, raw_layout=[{**o, 'size': [4, 5, 6], 'pos': [3, 3, 0]}
        for o in c['layouts']['reference']], **changes))+'\n' for c in cases))
    return path


def test_id_mapping_and_own_prepared_geometry(tmp_path):
    out, manifest, cases = freeze(tmp_path)
    case = cases[0]
    bound, sidecar = contract.bind_room(room())
    assert case['expected_ids'] == [o['id'] for o in bound['objects']]
    assert case['layouts']['reference'] == bound['objects']
    assert case['asset_mapping'][0]['prompt_id'] == sidecar['original_to_bound']['original-a']
    assert case['asset_mapping'][0]['layout_object'] == {'scale': [-1, 1, 1]}
    assert 'gt_object' not in case['asset_mapping'][0]
    assert case['layout_variant'] == 'raw'
    assert manifest['scope'] == 'reference_asset_assisted_visual_comparison'


def test_fixed_request_rounding_flags_and_normalization(tmp_path):
    r = room()
    r['fixed'] = [dict(id='door', category='Door', structure=True, size=[.1234, 1, 2],
                       pos=[.1234, 0, 0], yaw=.1234)]
    _, _, cases = freeze(tmp_path, [r])
    bound, _ = contract.bind_room(r)
    request = json.loads(contract.messages(bound)[1]['content'])
    fixed = cases[0]['fixed'][0]
    assert fixed['size'] == request['fixed'][0]['size']
    assert fixed['pos'] == request['fixed'][0]['pos']
    assert fixed['yaw'] == math.radians(request['fixed'][0]['yaw'])
    assert fixed['category'] == request['fixed'][0]['category']
    assert fixed['structure'] is True
    assert cases[0]['reference_eligible']


@pytest.mark.parametrize('category,flags', [('cabinet', {'structure': True}), ('door', {}),
                                           ('door frame', {'fixed_kind': 'structure'})])
def test_arbitrary_fixed_furniture_never_proxy(tmp_path, category, flags):
    r = room()
    r['fixed'] = [dict(id='fixed', category=category, size=[1, 1, 1], pos=[0, 0, 0], yaw=0, **flags)]
    out, _, cases = freeze(tmp_path, [r])
    assert not cases[0]['reference_eligible']
    assert 'no_asset' in rows(out / 'eligibility.jsonl')[0]['reason']


def test_unknown_height_uniform_three_meters(tmp_path):
    r = {**room(), 'height': None}
    _, _, cases = freeze(tmp_path, [r])
    assert cases[0]['height'] is None
    assert cases[0]['height_provenance']['effective_height'] == 3
    assert cases[0]['height_provenance']['imputed']


def test_malformed_room_and_missing_assets_retained(tmp_path):
    bad = {**room('bad'), 'height': -1}
    args = fixture(tmp_path, [bad, room('missing')])
    write_json(args[2], [])
    v.freeze_reference(*args, tmp_path / 'out')
    ledger = rows(tmp_path / 'out/eligibility.jsonl')
    assert {r['uid'] for r in ledger} == {'bad', 'missing'}
    assert all(not r['eligible'] for r in ledger)


def test_selection_exact_six_strata_first29_no_replacement(tmp_path):
    rooms = [room(f'{i}-{j}', source=s, kind=k) for i, (s, k) in enumerate(eval_data.PAPER_STRATA)
             for j in range(32)] + [room('excluded', kind='livingdiningroom')]
    args = fixture(tmp_path, rooms)
    write_json(args[2], [])
    v.freeze_reference(*args, tmp_path / 'out')
    selection = json.loads((tmp_path / 'out/gpt-selection.json').read_text())
    raw = eval_data.select_gpt(rooms)
    assert selection['raw_candidates'] == raw
    assert len(selection['selected_uids']) == 174
    for key, value in raw['strata'].items():
        assert selection['strata'][key]['selected_uids'] == value['selected_uids'][:29]
    assert all(r['status'] == 'selected_failed' for r in selection['selected'])
    assert selection['formal_success_cap'] == 348


def test_hashes_and_no_replace(tmp_path):
    out, manifest, cases = freeze(tmp_path)
    for name, digest in manifest['files'].items():
        assert v.sha256(out / name) == digest
    assert cases[0]['provenance']['export_sha256'] == v.sha256(tmp_path / 'export.jsonl.gz')
    with pytest.raises(FileExistsError):
        v.freeze_reference(tmp_path / 'export.jsonl.gz', tmp_path / 'receipt.json', tmp_path / 'map.json', out)


@pytest.mark.parametrize('field,value', [('gzip_sha256', '0'*64), ('rows', 999)])
def test_receipt_tamper_rejected(tmp_path, field, value):
    args = fixture(tmp_path)
    receipt = json.loads(args[1].read_text())
    write_json(args[1], {**receipt, field: value})
    with pytest.raises(ValueError):
        v.freeze_reference(*args, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_asset_inspection_cached_across_rooms(tmp_path, monkeypatch):
    original, calls = render_assets.inspect_asset, []
    def inspect(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(render_assets, 'inspect_asset', inspect)
    freeze(tmp_path, [room('a'), room('b')])
    assert len(calls) == 1


def test_attach_raw_both_complete_and_immutable(tmp_path):
    out, _, cases = freeze(tmp_path)
    original = copy.deepcopy(cases)
    j = predictions(tmp_path, cases, 'joint', repaired_layout=[{'wrong': True}])
    p = predictions(tmp_path, cases, 'placement')
    result = v.attach_predictions(out / 'reference-cases.jsonl', j, p)
    assert result['counts']['renderable'] == 1
    case = result['cases'][0]
    assert set(case['layouts']) == {'reference', 'joint', 'placement'}
    assert case['layouts']['joint'][0]['size'] == [4, 5, 6]
    assert case['fixed'] == cases[0]['fixed']
    assert case['cameras'] == cases[0]['cameras']
    assert cases == original


@pytest.mark.parametrize('failure', ['missing', 'incomplete', 'missing_size', 'wrong_ids', 'schema_false'])
def test_attach_failure_no_repair_gt_fill_or_replacement(tmp_path, failure):
    out, _, cases = freeze(tmp_path)
    j = predictions(tmp_path, cases, 'joint')
    p = predictions(tmp_path, cases, 'placement')
    record = rows(p)[0]
    if failure == 'missing':
        p.write_text('')
    else:
        if failure == 'incomplete':
            record['complete'] = False
        if failure == 'schema_false':
            record['schema_complete'] = False
        if failure == 'missing_size':
            del record['raw_layout'][0]['size']
        if failure == 'wrong_ids':
            record['raw_layout'][0]['id'] = 'OLD-WRONG'
        record['repaired_layout'] = cases[0]['layouts']['reference']
        p.write_text(json.dumps(record)+'\n')
    result = v.attach_predictions(out / 'reference-cases.jsonl', j, p)
    assert result['cases'] == []
    assert result['ledger'][0]['status'] == 'failed'
    assert result['ledger'][0]['uid'] == 'r'


def test_duplicate_and_foreign_prediction_uids_rejected(tmp_path):
    out, _, cases = freeze(tmp_path)
    j = predictions(tmp_path, cases, 'joint')
    p = predictions(tmp_path, cases, 'placement')
    text = p.read_text()
    for invalid in (text+text, text.replace('"uid": "r"', '"uid": "foreign"')):
        p.write_text(invalid)
        with pytest.raises(ValueError, match='UID'):
            v.attach_predictions(out / 'reference-cases.jsonl', j, p)


def test_crosswalk_duplicate_and_unknown_original_never_use_old_prompt(tmp_path):
    args = fixture(tmp_path)
    original = json.loads(args[2].read_text())[0]
    for i, mapping in enumerate(([original, original], [{**original, 'original_instance_id': 'wrong'}])):
        write_json(args[2], mapping)
        out = tmp_path / f'out-{i}'
        v.freeze_reference(*args, out)
        assert rows(out / 'eligibility.jsonl')[0]['eligible'] is False


def test_late_input_change_aborts_publication(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    original = v._selection
    def changed(*values):
        result = original(*values)
        args[2].write_text(args[2].read_text()+' ')
        return result
    monkeypatch.setattr(v, '_selection', changed)
    with pytest.raises(ValueError, match='hash changed'):
        v.freeze_reference(*args, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_cli_freeze_and_attach_synthetic_end_to_end(tmp_path, monkeypatch, capsys):
    import sys
    args = fixture(tmp_path)
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['visual_cases', 'freeze', '--export', str(args[0]),
        '--receipt', str(args[1]), '--assetmap', str(args[2]), '--out', str(out)])
    v.main()
    assert json.loads(capsys.readouterr().out)['counts']['eligible'] == 1
    cases = rows(out / 'reference-cases.jsonl')
    j, p = (predictions(tmp_path, cases, arm) for arm in ('joint', 'placement'))
    attached = tmp_path / 'attached'
    monkeypatch.setattr(sys, 'argv', ['visual_cases', 'attach', '--reference-cases',
        str(out / 'reference-cases.jsonl'), '--joint', str(j), '--placement', str(p), '--out', str(attached)])
    v.main()
    assert json.loads(capsys.readouterr().out)['counts']['renderable'] == 1
    assert len(rows(attached / 'cases.jsonl')) == 1
    assert rows(attached / 'failures.jsonl')[0]['status'] == 'renderable'


def test_failed_reference_stays_failed_even_with_valid_predictions(tmp_path):
    r = {**room(), 'fixed': [dict(id='sink', category='sink', size=[1, 1, 1], pos=[0, 0, 0], yaw=0)]}
    out, _, cases = freeze(tmp_path, [r])
    j, p = (predictions(tmp_path, cases, arm) for arm in ('joint', 'placement'))
    result = v.attach_predictions(out / 'reference-cases.jsonl', j, p)
    assert result['cases'] == []
    assert 'reference' in result['ledger'][0]['errors']
