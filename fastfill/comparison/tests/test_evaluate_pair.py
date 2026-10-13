import copy
import json
from pathlib import Path

import pytest

from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import messages
from fastfill.comparison.tests.test_geometry import fixture


def dump(path, value):
    path.write_text(json.dumps(value) + '\n')


def setup(root):
    data = root / 'data'
    data.mkdir()
    room = {**fixture(), 'room_type': 'bedroom', 'source': 'IL3D_3dfront'}
    dump(data / 'rooms.jsonl', room)
    runs = {}
    for arm in h.ARMS:
        (data / arm).mkdir()
        sizes = {o['id']: o['size'] for o in room['objects']}
        msg = messages(room, arm, sizes if arm == 'placement' else None)
        request = {'uid': room['uid'], 'source': room['source'], 'group': room['group'], 'messages': msg[:2]}
        dump(data / arm / 'requests.jsonl', request)
        runs[arm] = root / arm
        runs[arm].mkdir()
        dump(runs[arm] / 'records.jsonl', {**request, 'arm': arm, 'raw': msg[-1]['content'],
                                         'error': None, 'run_sha256': arm})
    dump(data / 'MANIFEST.json', {'expected_uids': [room['uid']], 'training_manifest_sha256': 'a'*64,
                                  'files': h.artifact_manifest(data)})
    for arm, run in runs.items():
        dump(run / 'result.json', {'status': 'INFERENCE_COMPLETE', 'pending_count': 0})
        dump(run / 'run_receipt.json', {'run_sha256': arm, 'data': {'manifest_sha256': h.sha256(data/'MANIFEST.json')},
             'training_data': {'manifest_sha256': 'a'*64}, 'base_sha256': 'base',
             'config': {'arm': arm, 'max_new_tokens': 8192, 'max_input_tokens': 8192,
                        'do_sample': False, 'num_beams': 1, 'dtype': 'bfloat16', 'attention': 'sdpa'}})
        from fastfill.comparison.infer import _hash
        value = json.loads((run / 'run_receipt.json').read_text())
        identity = {k: v for k, v in value.items() if k != 'run_sha256'}
        digest = _hash(identity)
        dump(run / 'run_receipt.json', {**identity, 'run_sha256': digest})
        record = json.loads((run / 'records.jsonl').read_text())
        dump(run / 'records.jsonl', {**record, 'run_sha256': digest})
        dump(run / 'MANIFEST.json', {'expected_uids': [room['uid']], 'files': h.artifact_manifest(run)})
    return data, runs


def test_end_to_end_two_arm_tables(tmp_path):
    data, runs = setup(tmp_path)
    out = tmp_path / 'out'
    result = e.evaluate(data, runs, out, repeats=20)
    assert result['status'] == 'GEOMETRY_COMPLETE'
    assert result['arms']['joint']['all']['size_iou'] == 1
    assert result['arms']['placement']['all']['oor'] == 0
    assert result['paper_strata']['IL3D_3dfront/bedroom']['joint']['requests'] == 1
    assert (out / 'geometry.csv').read_text().count('\n') >= 3
    assert 'OOR' in (out / 'report.md').read_text()
    assert result['stage'] == 'sft' and '模型阶段（stage）：sft' in (out / 'report.md').read_text()  # No stage = SFT run.
    h.verify_artifacts(out, json.loads((out/'MANIFEST.json').read_text())['files'])
    with pytest.raises(FileExistsError):
        e.evaluate(data, runs, out, repeats=20)


@pytest.mark.parametrize('kind', ['hash', 'protocol', 'base', 'pending', 'data'])
def test_refuses_mixed_or_incomplete_evidence(tmp_path, kind):
    data, runs = setup(tmp_path)
    run = runs['placement']
    if kind == 'hash':
        (run/'records.jsonl').write_text('{}\n')
    elif kind == 'pending':
        dump(run/'result.json', {'status': 'RUNNING', 'pending_count': 1})
    else:
        p = run/'run_receipt.json'
        value = json.loads(p.read_text())
        if kind == 'protocol': value['config']['max_new_tokens'] = 4096
        if kind == 'base': value['base_sha256'] = 'different'
        if kind == 'data': value['training_data']['manifest_sha256'] = 'b'*64
        dump(p, value)
    if kind != 'hash':
        manifest = json.loads((run/'MANIFEST.json').read_text())
        dump(run/'MANIFEST.json', {**manifest, 'files': {n:h.sha256(run/n) for n in manifest['files']}})
    with pytest.raises(ValueError):
        e.evaluate(data, runs, tmp_path/'out', repeats=20)


def test_failures_stay_in_denominator(tmp_path):
    data, runs = setup(tmp_path)
    p = runs['placement']/'records.jsonl'
    record = json.loads(p.read_text())
    dump(p, {**record, 'raw': '{}'})
    m = runs['placement']/'MANIFEST.json'
    value = json.loads(m.read_text())
    dump(m, {**value, 'files': {n:h.sha256(runs['placement']/n) for n in value['files']}})
    result = e.evaluate(data, runs, tmp_path/'out', repeats=20)
    assert result['arms']['placement']['all']['complete_rate'] == 0
    assert result['arms']['placement']['all']['oor'] is None
    assert result['arms']['placement']['all']['requests'] == 1
