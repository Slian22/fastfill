"""Amendment 012 adoption rules over manifest-verified synthetic S1 outputs (CPU only)."""
import json
import math

import pytest

from fastfill.comparison import decide_012 as d
from fastfill.comparison import train_pair_helpers as h


def decision_inputs(tmp_path, joint_bias, incomplete=False):
    rooms = [{'uid': f'u{k:02}', 'source': 'S', 'group': f'g{k}', 'boundary_type': 'polygon',
              'boundary': [[0, 0], [10, 0], [10, 10], [0, 10]], 'fixed': [],
              'objects': [{'id': 'obj_0001', 'category': 'chair', 'pos': [5, 5, 0],
                           'size': [1, 1, 1], 'yaw': 0, 'parent': None}]} for k in range(20)]
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'rooms.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rooms))
    h.write_json(data / 'MANIFEST.json', {'files': h.artifact_manifest(data),
                                         'expected_uids': [r['uid'] for r in rooms]})
    for arm in d.ARMS:
        for variant in d.VARIANTS:
            output = tmp_path / 's1' / f'{d.ARMS[arm]}-{variant}'
            output.mkdir(parents=True)
            rows = []
            for k, room in enumerate(rooms):
                done = not (incomplete and arm == 'joint' and variant == 'greedy' and k >= 10)
                bias = joint_bias if not incomplete else (0. if k < 10 else 2 * joint_bias)
                size = math.exp(bias / 2) if arm == 'joint' and variant == 'bo8' else 1
                rows.append({'uid': room['uid'], 'source': room['source'], 'group': room['group'],
                             'complete': done,
                             'raw_layout': [{**room['objects'][0], 'size': [size, size, 1]}] if done else None,
                             's1': {'strict_ok_raw': False, 'strict_ok_s1': variant == 'bo8' or k < 10,
                                    'geometry_success_raw': done, 'geometry_success_s1': done,
                                    'displacement_m': [.1] if done else []}})
            (output / f'{arm}-{variant}-geometry-s1.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            h.write_json(output / 'MANIFEST.json', {'files': h.artifact_manifest(output)})
    return data, tmp_path / 's1'


@pytest.mark.parametrize('bias, passed', [(0., True), (.03, True), (-.03, True), (.031, False), (-.031, False)])
def test_s2_joint_requires_absolute_bias_change_with_inclusive_boundary(tmp_path, bias, passed):
    data, s1 = decision_inputs(tmp_path, bias)
    result = d.decide(data, s1, tmp_path / 'decision')
    joint = result['decisions']['s2_joint']
    assert joint['estimate'] == .5 and joint['ci95'][0] > 0  # The acceptance gain alone passes.
    assert joint['passed'] is passed
    assert joint['bias_check']['passed'] is passed
    assert joint['bias_check']['greedy'] == {'mean': 0., 'complete_rooms': 20}
    assert joint['bias_check']['bo8']['mean'] == pytest.approx(bias)
    assert joint['bias_check']['absolute_difference'] == pytest.approx(abs(bias))
    assert result['decisions']['s2_placement']['passed']  # P does not have J's size-bias adoption gate.
    manifest = h.strict_json((tmp_path / 'decision/MANIFEST.json').read_text())
    h.verify_artifacts(tmp_path / 'decision', manifest['files'])
    with pytest.raises(FileExistsError):
        d.decide(data, s1, tmp_path / 'decision')


def test_bias_uses_each_variants_complete_rooms_without_dropping_failures_from_rates(tmp_path):
    data, s1 = decision_inputs(tmp_path, .02, incomplete=True)
    result = d.decide(data, s1, tmp_path / 'decision')
    joint = result['decisions']['s2_joint']
    assert joint['passed']
    assert joint['bias_check']['greedy']['complete_rooms'] == 10
    assert joint['bias_check']['bo8']['complete_rooms'] == 20
    assert joint['bias_check']['bo8']['mean'] == pytest.approx(.02)  # On common-complete alone this would be zero.
    assert result['rates']['joint-greedy-s1']['complete'] == .5
    assert joint['estimate'] == .5


def test_missing_complete_bias_cannot_authorize_s2(tmp_path):
    data, s1 = decision_inputs(tmp_path, 0.)
    root = s1 / 'joint-sft-greedy'
    path = root / 'joint-greedy-geometry-s1.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [{**r, 'complete': False, 'raw_layout': None,
             's1': {**r['s1'], 'strict_ok_s1': False, 'geometry_success_s1': False}} for r in rows]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    h.write_json(root / 'MANIFEST.json', {'files': {path.name: h.sha256(path)}})
    joint = d.decide(data, s1, tmp_path / 'decision')['decisions']['s2_joint']
    assert joint['estimate'] == 1. and joint['ci95'][0] > 0
    assert not joint['passed'] and not joint['bias_check']['passed']
    assert joint['bias_check']['greedy'] == {'mean': None, 'complete_rooms': 0}
    assert joint['bias_check']['absolute_difference'] is None
