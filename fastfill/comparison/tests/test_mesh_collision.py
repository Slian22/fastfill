"""Amendment 019 mesh collisions on synthetic OBJ meshes; no Blender, no real assets."""
import json

import numpy as np
import pytest

from fastfill.comparison import mesh_collision as m

BOX_FACES = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
TOP = {(4, 5, 6), (4, 6, 7)}  # the z-max face


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]


def write_obj(path, parts):
    """parts: [(world-frame vertices, faces)]; IL3D_3dfront maps raw (x, y, z) to world (z, x, y)."""
    lines, base = [], 0
    for vertices, faces in parts:
        lines += [f'v {y} {z} {x}' for x, y, z in vertices]
        lines += ['f ' + ' '.join(str(base + i + 1) for i in f) for f in faces]
        base += len(vertices)
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def assets(tmp_path):
    cube = write_obj(tmp_path / 'cube.obj', [(box((0, 0, 0), (1, 1, 1)), BOX_FACES)])
    shell = write_obj(tmp_path / 'open.obj', [(box((0, 0, 0), (1, 1, 1)), [f for f in BOX_FACES if f not in TOP])])
    ell = write_obj(tmp_path / 'ell.obj', [(box((0, 0, 0), (1, .4, .5)), BOX_FACES),
                                           (box((0, 0, 0), (.4, 1, .5)), BOX_FACES)])
    return {'cube': cube, 'open': shell, 'ell': ell}


def obj(i, category, size, pos, yaw=0.0, parent=None):
    return {'id': i, 'category': category, 'size': list(size), 'pos': list(pos), 'yaw': yaw, 'parent': parent}


def case(tmp_path, layout, meshes, uid='r1', systems=None):
    paths = assets(tmp_path)
    return {'uid': uid, 'source': 'IL3D_3dfront', 'stratum': 'IL3D_3dfront/bedroom', 'boundary_type': 'polygon',
            'boundary': [[0, 0], [4, 0], [4, 4], [0, 4]], 'height': 3.0, 'fixed': [],
            'expected_ids': [o['id'] for o in layout],
            'asset_mapping': [{'prompt_id': o['id'], 'asset_id': meshes[o['id']], 'mesh_path': paths[meshes[o['id']]],
                               'source': 'IL3D_3dfront', 'layout_object': {'scale': [1, 1, 1]}} for o in layout],
            'layouts': {'reference': layout, **(systems or {})}}


def only(result, system='GT'):
    assert result['error'] is None, result['error']
    return ([r for r in result['rows'] if r['system'] == system][0],
            [p for p in result['pairs'] if p['system'] == system])


def test_two_overlapping_cubes_share_the_known_voxel_volume(tmp_path):
    # x 1.0-1.2 and 1.1-1.3, same y/z 0.2 m: surface voxels 50..60 and 55..65 -> 6 x 11 x 11 shared voxels
    layout = [obj('a', 'cabinet', (.2, .2, .2), (1.1, 1.1, 0)), obj('b', 'cabinet', (.2, .2, .2), (1.2, 1.1, 0))]
    row, pairs = only(m.analyse_case(case(tmp_path, layout, {'a': 'cube', 'b': 'cube'})))
    assert len(pairs) == 1 and pairs[0]['candidate'] and pairs[0]['closed'] == [True, True]
    assert pairs[0]['volume_cm3'] == 6 * 11 * 11 * 8
    assert all(pairs[0][f'collides_{t}'] for t in m.TAUS) and not any(row[f'mesh_free_{t}'] for t in m.TAUS)
    assert pairs[0]['checker'] == 'severe' and pairs[0]['a4'] and pairs[0]['a4_class'] == 'genuine'
    assert row['bbox_max_err_m'] < 1e-9 and row['open_meshes'] == 0


def test_l_shape_box_overlap_without_mesh_contact_is_not_a_collision(tmp_path):
    # L arms x 1.0-2.0 / y 1.0-1.4 and x 1.0-1.4 / y 1.0-2.0; the cube sits in the empty corner, 20 cm from both arms
    layout = [obj('l', 'sofa', (1, 1, .5), (1.5, 1.5, 0)), obj('c', 'table', (.3, .3, .3), (1.75, 1.75, 0))]
    row, pairs = only(m.analyse_case(case(tmp_path, layout, {'l': 'ell', 'c': 'cube'})))
    assert len(pairs) == 1 and pairs[0]['candidate'] and pairs[0]['checker'] == 'contained' and pairs[0]['a4']
    assert pairs[0]['volume_cm3'] == 0 and not any(pairs[0][f'collides_{t}'] for t in m.TAUS)
    assert all(row[f'mesh_free_{t}'] for t in m.TAUS) and row['candidates'] == 1


def test_parent_child_pair_is_excluded(tmp_path):
    layout = [obj('a', 'cabinet', (.2, .2, .2), (1.1, 1.1, 0)),
              obj('b', 'box', (.2, .2, .2), (1.2, 1.1, 0), parent='a')]
    row, pairs = only(m.analyse_case(case(tmp_path, layout, {'a': 'cube', 'b': 'cube'})))
    assert pairs == [] and row['candidates'] == 0 and all(row[f'mesh_free_{t}'] for t in m.TAUS)
    flipped = [{**layout[0], 'parent': 'b'}, {**layout[1], 'parent': None}]
    assert only(m.analyse_case(case(tmp_path, flipped, {'a': 'cube', 'b': 'cube'})))[0]['candidates'] == 0


def test_open_shell_is_flagged_not_closed(tmp_path):
    layout = [obj('a', 'cabinet', (.3, .3, .3), (1, 1, 0)), obj('b', 'cabinet', (.3, .3, .3), (2, 2, 0))]
    row, _ = only(m.analyse_case(case(tmp_path, layout, {'a': 'open', 'b': 'cube'})))
    assert row['open_meshes'] == 1
    m._MESHES.clear()
    plan = m.PLAN(case(tmp_path, layout, {'a': 'open', 'b': 'cube'}))
    placed = [m.place(o, {}) for o in plan['layouts']['reference']]
    assert [p['closed'] for p in placed] == [False, True]
    assert placed[0]['solid'].sum() == placed[0]['shell'].sum() < placed[1]['solid'].sum()


@pytest.mark.parametrize('shift', [0.0, 0.04, 0.1, 0.16, 0.2, 0.24])
def test_tolerances_are_monotone(tmp_path, shift):
    layout = [obj('a', 'cabinet', (.2, .2, .2), (1.1, 1.1, 0)),
              obj('b', 'cabinet', (.2, .2, .2), (1.1 + shift, 1.1, 0))]
    row, pairs = only(m.analyse_case(case(tmp_path, layout, {'a': 'cube', 'b': 'cube'})))
    volume = pairs[0]['volume_cm3'] if pairs else 0.0
    flags = [volume > t for t in m.TAUS]
    assert flags == sorted(flags, reverse=True)
    if pairs:
        assert [pairs[0][f'collides_{t}'] for t in m.TAUS] == flags
    free = [row[f'mesh_free_{t}'] for t in m.TAUS]
    assert free == sorted(free) and free == [not f for f in flags]
    assert volume == pytest.approx(max(0, 11 - round(shift / m.PITCH)) * 121 * 8)
    if shift == 0.2:                      # flush faces share one voxel layer: 968 cm3, all of it surface contact
        assert volume == 968 and pairs[0]['deep_cm3'] == 0 and flags == [True, True, False]


def test_split_transform_matches_render_placement_on_a_rotated_object(tmp_path):
    layout = [obj('l', 'sofa', (1.13, .87, .61), (1.537, 2.011, .003), yaw=0.37)]
    m._MESHES.clear()
    planned = m.PLAN(case(tmp_path, layout, {'l': 'ell'}))['layouts']['reference'][0]
    placed = m.place(planned, {})
    assert placed['bbox_err'] < 1e-9
    tf = planned['transform']
    world = m._affine(planned['resource']['vertices'], tf.linear, np.array(tf.offset))
    point = tf.point(tuple(planned['resource']['vertices'][5]))
    assert np.allclose(world[5], point, atol=1e-12)
    direct = m.voxelise(world / m.PITCH, planned['resource']['faces'], m.SAMPLE_M / m.PITCH)
    assert (direct[0] == placed['lo']).all() and (direct[2] == placed['solid']).all()
    one_by_one = m.voxelise(world / m.PITCH, planned['resource']['faces'], m.SAMPLE_M / m.PITCH, chunk=1)
    assert all((x == y).all() for x, y in zip(direct, one_by_one))


def test_cli_writes_once_and_reports_missing(tmp_path):
    rooms = tmp_path / 'assets'
    rooms.mkdir()
    a = [obj('a', 'cabinet', (.2, .2, .2), (1.1, 1.1, 0)), obj('b', 'cabinet', (.2, .2, .2), (1.2, 1.1, 0))]
    p = [obj('a', 'cabinet', (.2, .2, .2), (1.1, 1.1, 0)), obj('b', 'cabinet', (.2, .2, .2), (2.5, 2.5, 0))]
    cases = [case(rooms, a, {'a': 'cube', 'b': 'cube'}, uid=f'r{i}', systems={'P': p, 'J': a}) for i in range(3)]
    cases[2] = {**cases[2], 'input_error': 'RenderReject: no_asset: fixed_0001', 'layouts': {'reference': a}}
    visual = tmp_path / 'run' / 'visual'
    (visual / 'chunks' / '00').mkdir(parents=True)
    (visual / 'cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases))
    (visual / 'chunks' / '00' / 'input.json').write_text(json.dumps({'cases': cases[:2]}))
    (visual / 'J-geometry.jsonl').write_text(''.join(json.dumps({'uid': c['uid'], 'group': f'g{i}'}) + '\n'
                                                     for i, c in enumerate(cases)))
    (visual / 'attachment-ledger.jsonl').write_text(''.join(json.dumps({'uid': c['uid'], 'errors': {}}) + '\n'
                                                            for c in cases))
    out = tmp_path / 'out'
    summary = m.run(tmp_path / 'run', out, workers=1, repeats=20)
    assert summary['analysed_rooms'] == 2 and list(summary['missing_rooms']) == ['r2']
    assert summary['systems']['GT']['levels']['mesh_free_500']['estimate'] == 0
    assert summary['systems']['P']['levels']['mesh_free_500']['estimate'] == 1
    assert summary['paired_vs_P']['GT']['mesh_free_500']['estimate'] == -1
    assert summary['gt_a4_classes']['genuine'] == {'pairs': 2, 'collides_125': 2, 'collides_500': 2,
                                                   'collides_2000': 2, 'median_volume_cm3': 5808}
    manifest = json.loads((out / 'MANIFEST.json').read_text())
    assert set(manifest['files']) == {'rows.jsonl', 'pairs.jsonl', 'summary.json', 'report.md'}
    assert m.render.file_sha256(out / 'report.md') == manifest['files']['report.md']
    text = (out / 'report.md').read_text()
    assert '修订 019' in text and '三角形采样间距 ≤ 0.5 cm' in text and '仅外壳（填充未增加体素）' in text
    assert len((out / 'rows.jsonl').read_text().splitlines()) == 6
    with pytest.raises(FileExistsError):
        m.run(tmp_path / 'run', out, workers=1, repeats=20)
