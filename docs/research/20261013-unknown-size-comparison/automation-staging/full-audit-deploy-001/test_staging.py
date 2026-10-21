"""Offline checks only. Never starts a subprocess, SSH connection, or job."""
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location('audit_' + name, Path(__file__).with_name(name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def controller():
    return module('controller')


@pytest.fixture
def configs(tmp_path):
    base = {'geometry_job_id': 33, 'prerequisite_job_ids': [11, 22],
            'total_timeout_seconds': 432000, 'remote_env_file': '/server/secret',
            'output_root': str(tmp_path / 'old'), 'local_code_root': '/old/code',
            'remote_api_code': '/remote/old', 'api_client_path': '/remote/old/client.py',
            'ssh_control_path': '/tmp/old/socket', 'remote_run_root': '/remote/run'}
    candidate = {**base, 'geometry_job_id': 44, 'output_root': str(tmp_path / 'new'),
                 'local_code_root': '/new/code', 'remote_api_code': '/remote/new',
                 'api_client_path': '/remote/new/client.py', 'ssh_control_path': '/tmp/new/socket'}
    return base, candidate, {**candidate, 'deadline_unix': 2000}


def test_absolute_deadline_is_not_reset(controller, configs):
    base, candidate, spec = configs
    assert controller.effective_config(base, candidate, spec, now=1000)['total_timeout_seconds'] == 990
    assert controller.effective_config(base, candidate, spec, now=1500)['total_timeout_seconds'] == 490
    assert candidate['total_timeout_seconds'] == 432000
    with pytest.raises(TimeoutError):
        controller.effective_config(base, candidate, spec, now=2000)


@pytest.mark.parametrize('job', [11, 22, 33, None, 'invalid'])
def test_geometry_must_be_new_job(controller, configs, job):
    base, candidate, spec = configs
    with pytest.raises(ValueError):
        controller.effective_config(base, {**candidate, 'geometry_job_id': job}, spec, now=1000)


def test_config_limits_and_inputs_cannot_drift(controller, configs):
    base, candidate, spec = configs
    with pytest.raises(ValueError, match='unapproved'):
        controller.effective_config(base, {**candidate, 'remote_env_file': '/other'}, spec, now=1000)
    with pytest.raises(ValueError, match='unapproved'):
        controller.effective_config(base, {**candidate, 'prerequisite_job_ids': [55]}, spec, now=1000)


def test_fresh_output_and_immutable_receipts(controller, configs, tmp_path):
    base, candidate, spec = configs
    Path(candidate['output_root']).mkdir()
    with pytest.raises(FileExistsError):
        controller.effective_config(base, candidate, spec, now=1000)
    receipt = tmp_path / 'receipt.json'
    controller.publish(receipt, {'first': True})
    with pytest.raises(FileExistsError):
        controller.publish(receipt, {'first': False})
    assert json.loads(receipt.read_text()) == {'first': True}


def test_staging_spec_cannot_start_any_subprocess(controller, monkeypatch, tmp_path):
    spec = tmp_path / 'spec.json'
    spec.write_text('{"parent_snapshot_ready": false}')
    monkeypatch.setattr(controller.signal, 'signal', Mock())
    monkeypatch.setattr(controller.signal, 'alarm', Mock())
    monkeypatch.setattr(controller.os, 'umask', Mock())
    popen = Mock(side_effect=AssertionError('external subprocess forbidden'))
    monkeypatch.setattr(controller.subprocess, 'Popen', popen)
    monkeypatch.setattr(controller.sys, 'argv', ['controller.py', '--spec', str(spec), '--state', str(tmp_path / 'state')])
    with pytest.raises(ValueError, match='not authorized'):
        controller.main()
    popen.assert_not_called()
    assert not (tmp_path / 'state').exists()


def test_stop_has_bounded_escalation(controller, monkeypatch):
    proc = Mock(pid=999)
    proc.poll.return_value = None
    proc.wait.side_effect = [controller.subprocess.TimeoutExpired('fixture', 30), 0]
    kill = Mock()
    monkeypatch.setattr(controller.os, 'killpg', kill)
    controller.stop(proc)
    assert [call.args[0] for call in kill.call_args_list] == [999, 999]
    assert [call.kwargs['timeout'] for call in proc.wait.call_args_list] == [30, 5]


def test_only_three_frozen_source_changes(tmp_path):
    freeze = module('verify_freeze')
    names = freeze.EXPECTED_CHANGES | {'server_api_judge.py', 'fastfill/comparison/infer.py'}
    for dirname in ('old', 'new'):
        root = tmp_path / dirname
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('new' if dirname == 'new' and name in freeze.EXPECTED_CHANGES else 'old')
        manifest = {name: freeze.digest(root / name) for name in names}
        (root / 'MANIFEST.json').write_text(json.dumps({'files': manifest}))
    assert freeze.verify(tmp_path / 'old', tmp_path / 'new')['inference_and_driver_unchanged']
    (tmp_path / 'new/server_api_judge.py').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        freeze.verify(tmp_path / 'old', tmp_path / 'new')


def test_real_inference_reader_accepts_original_code_identity(tmp_path):
    # Exercise the real current module against a self-consistent old-code
    # inference receipt; metric code is deliberately not its inference identity.
    from fastfill.comparison.evaluate_pair import _inference
    data, run = tmp_path / 'data', tmp_path / 'infer'
    data.mkdir(); run.mkdir()
    (data / 'MANIFEST.json').write_text('{}')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    identity = {'data': {'manifest_sha256': sha(data / 'MANIFEST.json')},
                'code_sha256': {'infer.py': 'original-inference-code', 'train_pair_helpers.py': 'original-helper'}}
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    (run / 'run_receipt.json').write_text(json.dumps({**identity, 'run_sha256': run_hash}))
    (run / 'result.json').write_text('{"status":"INFERENCE_COMPLETE","pending_count":0}')
    (run / 'records.jsonl').write_text(json.dumps({'uid': 'one', 'run_sha256': run_hash}) + '\n')
    (run / 'MANIFEST.json').write_text(json.dumps({'expected_uids': ['one'],
        'files': {p.name: sha(p) for p in run.iterdir()}}))
    records, receipt = _inference(run, data, ['one'])
    assert len(records) == 1 and receipt['code_sha256'] == identity['code_sha256']
    (run / 'records.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        _inference(run, data, ['one'])
