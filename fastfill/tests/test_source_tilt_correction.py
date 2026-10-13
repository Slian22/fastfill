"""Source-box regressions; embedded raw fixtures require no mounted dataset.

Fixtures were verified against the exporters and raw source on 2026-10-13.
See docs/research/20261013-unknown-size-comparison/evidence/
source-tilt-correction-20261013.md for paths and transform conventions.
"""
import copy
import itertools
import json
import math
from unittest.mock import patch

import numpy as np
import pytest

from fastfill.adapters import scan2cad, spatialgen


SCAN_BOX = {'furniture_category': 'chair',
 'furniture_instance_id': 'scene0494_00::scan2cad_5',
 'furniture_asset_id': '03001627/e955b63a4bd738857178717457aa5d20',
 'furniture_position': {'x': 0.708803960912392, 'y': 0.8918472396351461, 'z': 0.40660861976074053},
 'position_type': 'oriented_bbox_center',
 'furniture_rotation': {'x': -10.039585163852308, 'y': -0.74180380628862, 'z': 77.91919993980092},
 'furniture_size': {'width': 0.40841571282942496,
                    'length': 0.446364754191654,
                    'height': 0.8936079665390511},
 'source_fields': {'category_synset_id': '03001627',
                   'rotation_matrix_zup_scan_local': [[0.20927341646255887,
                                                       -0.9624080210098739,
                                                       -0.17313590992900918],
                                                      [0.9777717672178058,
                                                       0.20829317016159174,
                                                       0.024019399343087194],
                                                      [0.012946557126997273,
                                                       -0.17431392091359307,
                                                       0.9846050190986673]],
                   'scan2cad_annotation': {'trs': {'translation': [-1.2257661181328232,
                                                                   -0.34619782045789155,
                                                                   -0.9181225466283327],
                                                   'rotation': [0.2040985395151209,
                                                                -0.011566121152010748,
                                                                0.9750109208194173,
                                                                0.0869696228125854],
                                                   'scale': [0.9870788271692251,
                                                             1.193268769126256,
                                                             0.8622489358106685]},
                                           'bbox': [0.20688100159168243,
                                                    0.37443700432777405,
                                                    0.2588375210762024],
                                           'center': [-0.00011900067329406738,
                                                      -0.005470991134643555,
                                                      -0.041844502091407776],
                                           'sym': '__SYM_NONE'}}}

SCAN_TRS = {'translation': [-1.9339642934501171, -0.7889901318485502, 2.9236308932304382],
 'rotation': [-0.7071067811865476, 0.7071067811865475, -0.0, -0.0],
 'scale': [1.0, 1.0, 1.0]}

SCAN_ALIGNMENT = [0.199368,
 0.979925,
 0.0,
 -3.1634,
 -0.979925,
 0.199368,
 0.0,
 0.812662,
 0.0,
 0.0,
 1.0,
 -0.023465,
 0.0,
 0.0,
 0.0,
 1.0]

SPATIAL_BOXES = [{'furniture_category': 'painting, picture',
  'furniture_instance_id': 'spatialgen_scene_00024::obj_0',
  'furniture_position': {'x': -4.31362, 'y': 0.07841, 'z': 0.590044},
  'furniture_rotation': {'x': None, 'y': None, 'z': 90.000003},
  'furniture_size': {'width': 0.8, 'length': 0.04, 'height': 1.180454},
  'source_fields': {'class': 'painting, picture',
                    'transform': [-4.262062e-08,
                                  -0.93292814,
                                  -0.0918232,
                                  -4.31362,
                                  0.97504616,
                                  -4.0779586e-08,
                                  -4.013721e-09,
                                  0.07840963999999984,
                                  1.3340588e-15,
                                  -0.08162061,
                                  1.0495443,
                                  0.5900439000000001,
                                  0.0,
                                  0.0,
                                  0.0,
                                  1.0],
                    'size': [0.8204739990234375, 0.042712600708007815, 1.120449951171875],
                    'index': 0,
                    'transform_scale': [0.97504616, 0.936491772, 1.053553386],
                    'transform_mirrored': False}},
 {'furniture_category': 'painting, picture',
  'furniture_instance_id': 'spatialgen_scene_00030::obj_0',
  'furniture_position': {'x': 5.169046, 'y': 2.048631, 'z': 0.375},
  'furniture_rotation': {'x': None, 'y': None, 'z': -90.000003},
  'furniture_size': {'width': 1.506951, 'length': 0.03, 'height': 0.75},
  'source_fields': {'class': 'painting, picture',
                    'transform': [-4.078972e-08,
                                  0.9848077,
                                  0.17364822,
                                  5.169046000000001,
                                  -0.93315995,
                                  -4.5216524e-08,
                                  4.7117963e-09,
                                  2.0486306,
                                  1.1657018e-08,
                                  -0.17364822,
                                  0.9848077,
                                  0.375,
                                  0.0,
                                  0.0,
                                  0.0,
                                  1.0],
                    'size': [1.6148900146484375, 0.030000200271606447, 0.75],
                    'index': 0,
                    'transform_scale': [0.93315995, 0.999999955, 0.999999955],
                    'transform_mirrored': False}}]


def rotation(yaw, roll=0, pitch=0):
    z, x, y = np.radians([yaw, roll, pitch])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return rz @ ry @ rx


def corners(size, matrix, center):
    signs = np.asarray(list(itertools.product((-0.5, 0.5), repeat=3)))
    return (signs * size) @ np.asarray(matrix).T + center


def quaternion_matrix(q):
    # Independent quaternion-vector formula, not the exporter's entrywise formula.
    w, v = q[0], np.asarray(q[1:])
    return np.column_stack([e + 2 * np.cross(v, w * e + np.cross(v, e)) for e in np.eye(3)])


def scan_native_corners():
    model = SCAN_BOX['source_fields']['scan2cad_annotation']
    trs = model['trs']
    points = corners(2 * np.asarray(model['bbox']), np.eye(3), model['center'])
    world = (points * trs['scale']) @ quaternion_matrix(trs['rotation']).T + trs['translation']
    local = (world - SCAN_TRS['translation']) @ quaternion_matrix(SCAN_TRS['rotation']) / SCAN_TRS['scale']
    alignment = np.asarray(SCAN_ALIGNMENT).reshape(4, 4)
    return local @ alignment[:3, :3].T + alignment[:3, 3]


def scene_for(adapter, furniture):
    room = {'room_id': 'fixture::room_0', 'room_type': 'living room',
            'room_boundary': [[-10, -10], [10, -10], [10, 10], [-10, 10]],
            'scene_shell': None, 'furniture': furniture, 'scan2cad_alignments': furniture}
    return {'scene_id': 'fixture_00', 'rooms': [room],
            'provenance': {'source_coverage': 'fixture', 'scan2cad_scan_to_world_trs': SCAN_TRS}}


def load_fixture(adapter, furniture):
    scene = scene_for(adapter, furniture)
    before = json.dumps(scene, sort_keys=True)
    with patch.object(adapter, 'iter_records', return_value=iter([scene])):
        result = next(adapter.load('/unused'))
    assert json.dumps(scene, sort_keys=True) == before
    return result


def assert_envelope(obj, points, floor=0):
    shifted = points + [10, 10, -floor]  # convert_room boundary-origin normalization
    relative = (shifted - obj['pos']) @ rotation(math.degrees(obj['yaw']))
    size = np.asarray(obj['size'])
    lo = np.array([-size[0] / 2, -size[1] / 2, 0])
    np.testing.assert_allclose(relative.min(0), lo, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(relative.max(0), lo + size, atol=1e-9, rtol=1e-10)
    assert np.all(relative >= lo - 1e-9)
    assert np.all(relative <= lo + size + 1e-9)


def furniture_for(adapter, yaw=37, roll=0, pitch=0, mirrored=False, category='decoration'):
    size, center = np.array([0.7, 1.3, 2.1]), np.array([3.0, 4.0, 2.5])
    matrix = rotation(yaw, roll, pitch)
    tilted = bool(roll or pitch)
    base = {'furniture_instance_id': 'fixture::obj_0', 'furniture_category': category,
            'furniture_position': dict(zip('xyz', center)),
            'furniture_rotation': {'x': roll, 'y': pitch, 'z': yaw},
            'furniture_size': dict(zip(('width', 'length', 'height'), size))}
    if adapter is scan2cad:
        fields = {'rotation_matrix_zup_scan_local': matrix.tolist(),
                  'scan2cad_annotation': {'sym': '__SYM_NONE'}}
        return {**base, 'source_fields': fields}, corners(size, matrix, center)
    scale = np.array([1.2, -0.8 if mirrored else 0.8, 2.3])
    affine = np.column_stack((matrix * scale, center))
    transform = np.vstack((affine, [0, 0, 0, 1])).flatten().tolist()
    fields = {'transform': transform, 'size': (size / abs(scale)).tolist()}
    return {**base, 'source_fields': fields,
            'furniture_rotation': {'x': None if tilted else 0, 'y': None if tilted else 0, 'z': yaw}}, corners(size / abs(scale), matrix * scale, center)


def test_scan2cad_known_raw_0494_box_matches_native_trs_corners():
    points = scan_native_corners()
    f = SCAN_BOX
    exported = corners([f['furniture_size'][k] for k in ('width', 'length', 'height')],
                       f['source_fields']['rotation_matrix_zup_scan_local'],
                       [f['furniture_position'][k] for k in 'xyz'])
    # Canonical Z-up permutes/sign-flips native CAD axes: compare corner sets.
    assert max(min(np.linalg.norm(p - q) for q in exported) for p in points) < 1e-12
    ir = load_fixture(scan2cad, [f])
    obj = ir['objects'][0]
    assert_envelope(obj, points)
    assert obj['tilted'] and ir['meta']['n_tilted'] == 1
    assert obj['yaw'] == pytest.approx(math.radians(f['furniture_rotation']['z'] + 90))
    assert obj['sym'] == '__SYM_NONE'


@pytest.mark.parametrize('f', SPATIAL_BOXES, ids=['raw_00024_obj_0', 'raw_00030_obj_0'])
def test_spatialgen_known_raw_boxes_match_full_affine_corners(f):
    matrix = np.asarray(f['source_fields']['transform']).reshape(4, 4)
    points = corners(f['source_fields']['size'], matrix[:3, :3], matrix[:3, 3])
    ir = load_fixture(spatialgen, [f])
    assert_envelope(ir['objects'][0], points)
    assert ir['objects'][0]['tilted'] and ir['meta']['n_tilted'] == 1
    assert ir['objects'][0]['yaw'] == pytest.approx(math.radians(f['furniture_rotation']['z'] - 90) % (2 * math.pi))
    assert ir['meta']['eval_only'] is True


@pytest.mark.parametrize('adapter', [scan2cad, spatialgen])
@pytest.mark.parametrize('yaw', [-90, 0, 37, 179])
def test_ordinary_yaw_is_invariant(adapter, yaw):
    f, points = furniture_for(adapter, yaw=yaw)
    obj = load_fixture(adapter, [f])['objects'][0]
    assert_envelope(obj, points)
    assert obj['size'] == pytest.approx([1.3, 0.7, 2.1])
    assert obj['pos'] == pytest.approx([13, 14, 1.45])
    offset = 90 if adapter is scan2cad else -90
    assert obj['yaw'] == pytest.approx(math.radians(yaw + offset) % (2 * math.pi))
    assert obj['tilted'] is False


@pytest.mark.parametrize('adapter', [scan2cad, spatialgen])
@pytest.mark.parametrize('roll,pitch', [(9, 0), (11, 0), (23, -17), (90, 0)])
def test_tilt_envelope_keeps_source_yaw_and_tilt_policy(adapter, roll, pitch):
    f, points = furniture_for(adapter, roll=roll, pitch=pitch)
    ir = load_fixture(adapter, [f])
    obj = ir['objects'][0]
    expected_tilt = adapter is spatialgen or roll > scan2cad.MAX_TILT_DEG
    if expected_tilt:
        assert_envelope(obj, points)
    else:
        # Scan2CAD fit noise (<= MAX_TILT_DEG) keeps the CAD-local box, bottom = centre - height / 2.
        assert obj['size'] == pytest.approx([1.3, 0.7, 2.1])
        assert obj['pos'] == pytest.approx([13, 14, 1.45])
    assert obj['tilted'] == expected_tilt
    assert ir['meta']['n_tilted'] == int(expected_tilt)
    offset = 90 if adapter is scan2cad else -90
    assert obj['yaw'] == pytest.approx(math.radians(37 + offset) % (2 * math.pi))


@pytest.mark.parametrize('roll', [0, 27])
def test_spatialgen_signed_scale_is_applied_once_without_front_flip(roll):
    f, points = furniture_for(spatialgen, roll=roll, pitch=0, mirrored=True)
    obj = load_fixture(spatialgen, [f])['objects'][0]
    assert_envelope(obj, points)
    assert obj['yaw'] == pytest.approx(math.radians(37 - 90) % (2 * math.pi))
    assert obj['tilted'] == bool(roll)


def test_scan_floor_votes_with_historical_upright_bottoms_and_tilted_boxes_use_enclosure():
    # Upright fits vote with centre - height / 2 (an enclosure bottom would move the median);
    # the tilted chair at 0.20 m does not vote (it would make the median 0.18).
    furniture, point_sets = [], []
    for index, (bottom, category, roll) in enumerate([(0.10, 'chair', 9), (0.18, 'chair', 6), (0.40, 'chair', 3),
                                                    (0.20, 'chair', 15), (0.7, 'lamp', 15)]):
        f, points = furniture_for(scan2cad, category=category, roll=roll)
        tilted = roll > scan2cad.MAX_TILT_DEG
        dz = bottom - (points[:, 2].min() if tilted else f['furniture_position']['z'] - 2.1 / 2)
        furniture.append({**f, 'furniture_instance_id': str(index),
                          'furniture_position': {**f['furniture_position'], 'z': f['furniture_position']['z'] + dz}})
        point_sets.append(points + [0, 0, dz])
    ir = load_fixture(scan2cad, furniture)
    assert ir['meta']['floor_z'] == pytest.approx(0.14)
    assert ir['meta']['n_floor_snapped'] == 3
    assert [o['pos'][2] for o in ir['objects']] == pytest.approx([0, 0, 0.26, 0, 0.56])
    assert [o['tilted'] for o in ir['objects']] == [False, False, False, True, True]
    for obj in ir['objects'][:3]:
        assert obj['size'] == pytest.approx([1.3, 0.7, 2.1])
    for obj, points in zip(ir['objects'][3:], point_sets[3:]):
        # Snapping intentionally translates the entire envelope; sizes still enclose the source box.
        applied_floor = points[:, 2].min() - obj['pos'][2]
        assert_envelope(obj, points, floor=applied_floor)


@pytest.mark.parametrize('adapter', [scan2cad, spatialgen])
@pytest.mark.parametrize('bad', ['missing_transform', 'bad_shape', 'nan_matrix', 'inf_matrix',
                                 'zero_matrix', 'singular_matrix', 'zero_size', 'negative_size',
                                 'nan_size', 'inf_size', 'bool_size', 'huge_size',
                                 'bad_position', 'missing_yaw'])
def test_invalid_geometry_is_incomplete_without_guessing(adapter, bad):
    f, _ = furniture_for(adapter, roll=12)
    f = copy.deepcopy(f)
    fields = f['source_fields']
    key = 'rotation_matrix_zup_scan_local' if adapter is scan2cad else 'transform'
    if bad == 'missing_transform':
        fields.pop(key)
    elif bad == 'bad_shape':
        fields[key] = [1, 2]
    elif bad in ('nan_matrix', 'inf_matrix'):
        value = float('nan') if bad == 'nan_matrix' else float('inf')
        if adapter is scan2cad:
            fields[key][2][2] = value
        else:
            fields[key][10] = value
    elif bad in ('zero_matrix', 'singular_matrix'):
        matrix = np.zeros((3, 3)) if bad == 'zero_matrix' else np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        if adapter is scan2cad:
            fields[key] = matrix.tolist()
        else:
            fields[key] = np.vstack((np.column_stack((matrix, [3, 4, 2.5])), [0, 0, 0, 1])).flatten().tolist()
    elif bad.endswith('_size'):
        value = {'zero_size': 0, 'negative_size': -1, 'nan_size': float('nan'),
                 'inf_size': float('inf'), 'bool_size': True, 'huge_size': 10**400}[bad]
        if adapter is scan2cad:
            f['furniture_size']['width'] = value
        else:
            fields['size'][0] = value
    elif bad == 'bad_position':
        if adapter is scan2cad:
            f['furniture_position']['z'] = float('nan')
        else:
            fields['transform'][11] = float('nan')
    else:
        f['furniture_rotation']['z'] = None
    ir = load_fixture(adapter, [f])
    assert ir['objects'] == []
    assert ir['meta']['n_incomplete'] == 1


@pytest.mark.parametrize('roll', [9, 12])
def test_scan_degenerate_local_axis_is_not_rescued_by_tilt_and_votes_as_before(roll):
    f, _ = furniture_for(scan2cad, roll=roll, category='chair')
    f = {**f, 'furniture_size': {**f['furniture_size'], 'width': 1e-8},
         'furniture_position': {**f['furniture_position'], 'z': 1.05}}
    ir = load_fixture(scan2cad, [f])
    assert ir['objects'] == []
    assert ir['meta']['n_incomplete'] == 1
    # Historical vote: an upright degenerate fit still votes (1 of the 76 does in the export).
    upright = roll <= scan2cad.MAX_TILT_DEG
    assert ir['meta']['floor_z'] == (0.0 if upright else None)
    assert ir['meta']['n_floor_snapped'] == int(upright)


def test_spatialgen_projective_transform_is_not_treated_as_affine():
    f, _ = furniture_for(spatialgen)
    f = copy.deepcopy(f)
    f['source_fields']['transform'][12] = 0.1
    ir = load_fixture(spatialgen, [f])
    assert ir['objects'] == []
    assert ir['meta']['n_incomplete'] == 1


@pytest.mark.parametrize('scale', [None, [1, 2, 1], [1, float('nan'), 1]])
def test_scan_missing_or_nonunit_scan_scale_is_an_evidence_blocker(scale):
    f, _ = furniture_for(scan2cad)
    scene = scene_for(scan2cad, [f])
    scene['provenance'] = {**scene['provenance'],
                           'scan2cad_scan_to_world_trs': {**SCAN_TRS, 'scale': scale}}
    before = json.dumps(scene, sort_keys=True)
    with patch.object(scan2cad, 'iter_records', return_value=iter([scene])):
        ir = next(scan2cad.load('/unused'))
    assert ir['objects'] == []
    assert ir['meta']['n_incomplete'] == 1
    assert ir['meta']['floor_z'] is None
    assert json.dumps(scene, sort_keys=True) == before


@pytest.mark.parametrize('adapter', [scan2cad, spatialgen])
def test_finite_inputs_with_overflowing_extent_are_incomplete(adapter):
    f, _ = furniture_for(adapter, yaw=0)
    f = copy.deepcopy(f)
    if adapter is scan2cad:
        f['furniture_size']['height'] = 1e308
        f['source_fields']['rotation_matrix_zup_scan_local'][2][2] = 2.0
    else:
        f['source_fields']['size'][2] = 1e308
        f['source_fields']['transform'][10] = 2.0
    ir = load_fixture(adapter, [f])
    assert ir['objects'] == []
    assert ir['meta']['n_incomplete'] == 1


def test_spatialgen_small_tilt_does_not_change_exporter_flag():
    f, points = furniture_for(spatialgen, roll=0.001)
    f = {**f, 'furniture_rotation': {**f['furniture_rotation'], 'x': 0, 'y': 0}}
    obj = load_fixture(spatialgen, [f])['objects'][0]
    assert_envelope(obj, points)
    assert obj['tilted'] is False


@pytest.mark.parametrize('adapter', [scan2cad, spatialgen])
def test_export_file_to_ir_preserves_shell_and_source_contracts(adapter, tmp_path):
    f = SCAN_BOX if adapter is scan2cad else SPATIAL_BOXES[0]
    scene = scene_for(adapter, [f])
    if adapter is scan2cad:
        scene['rooms'][0]['scene_shell'] = {
            'vertices_zup_m': [[-10, -10, 0], [10, -10, 0], [10, 10, 0],
                               [-10, 10, 0], [-10, -10, 3], [10, -10, 3]],
            'planes': [[0, 1, 2, 3], [0, 1, 5, 4]]}
        relative = scan2cad.PATH
        content = json.dumps([scene])
        points = scan_native_corners()
    else:
        relative = 'BillLin66__3D_Room_Collections/SpatialGen_exported/spatialgen.jsonl'
        content = json.dumps(scene) + '\n'
        matrix = np.asarray(f['source_fields']['transform']).reshape(4, 4)
        points = corners(f['source_fields']['size'], matrix[:3, :3], matrix[:3, 3])
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(content)
    ir, = adapter.load(str(tmp_path))
    assert path.read_text() == content
    assert_envelope(ir['objects'][0], points)
    assert ir['source'] == adapter.SOURCE
    assert ir['boundary_type'] == 'polygon'
    assert ir['boundary'] == [[0, 0], [20, 0], [20, 20], [0, 20]]
    assert ir['meta']['n_incomplete'] == 0
    if adapter is scan2cad:
        assert ir['meta']['boundary_source'] == 'scenecad_floor_plane'
        assert ir['group'] == 'scannet:fixture'
    else:
        assert ir['subset'] == 'test' and ir['meta']['eval_only']
        assert ir['group'] == 'spatialgen:fixture_00'


def test_scan_without_cad_alignments_remains_skipped():
    with patch.object(scan2cad, 'iter_records', return_value=iter([scene_for(scan2cad, [])])):
        assert list(scan2cad.load('/unused')) == []
