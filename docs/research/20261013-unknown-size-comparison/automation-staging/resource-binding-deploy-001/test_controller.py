"""Offline handoff invariants; no process, network or scheduler side effects."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('visual_handoff', Path(__file__).with_name('controller.py'))
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class HandoffContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.base = {'geometry_job_id': '7368009', 'prerequisite_job_ids': ['7367626', '7367628'],
                     'local_code_root': str(root / 'old-code'), 'output_root': str(root / 'old-run'),
                     'ssh_control_path': str(root / 'old.sock'), 'total_timeout_seconds': 432000,
                     'remote_api_code': '/remote/old-code', 'api_client_path': '/remote/old-code/client.py',
                     'remote_run_root': '/remote/original-run', 'api_case_cap': 348}
        self.candidate = {**self.base, 'local_code_root': str(root / 'new-code'),
                          'output_root': str(root / 'new-run'), 'ssh_control_path': str(root / 'new.sock')}
        self.specification = {**self.candidate, 'deadline_unix': 10000}

    def test_visual_only_switch_preserves_existing_geometry_job(self):
        got = controller.effective_config(self.base, self.candidate, self.specification, now=1000)
        self.assertEqual(got['geometry_job_id'], '7368009')
        self.assertEqual(got['total_timeout_seconds'], 8990)
        self.assertEqual(got['prerequisite_job_ids'], self.base['prerequisite_job_ids'])
        self.assertEqual(self.candidate['total_timeout_seconds'], 432000)

    def test_any_remote_or_budget_change_is_rejected(self):
        mutations = {'geometry_job_id': '7368010', 'prerequisite_job_ids': ['7'],
                     'remote_api_code': '/remote/new', 'api_client_path': '/remote/new/client.py',
                     'remote_run_root': '/remote/newrun', 'api_case_cap': 349}
        for key, value in mutations.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                controller.effective_config(self.base, {**self.candidate, key: value},
                                            {**self.specification, key: value}, now=1000)

    def test_no_timeout_extension_and_no_output_reuse(self):
        base = {**self.base, 'total_timeout_seconds': 200}
        got = controller.effective_config(base, self.candidate, self.specification, now=1000)
        self.assertEqual(got['total_timeout_seconds'], 200)
        with self.assertRaises(TimeoutError):
            controller.effective_config(self.base, self.candidate, self.specification, now=9999)
        Path(self.candidate['output_root']).mkdir()
        with self.assertRaises(FileExistsError):
            controller.effective_config(self.base, self.candidate, self.specification, now=1000)

    def test_absolute_original_deadline_cannot_be_extended_in_both_fields(self):
        with self.assertRaises(ValueError):
            controller.remaining(1792302801, now=1000)

    def release_fixture(self):
        root = Path(self.temp.name)
        def save(name, value):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(value))
            return str(p), controller.sha(p)
        base_path, base_hash = save('base.json', self.base)
        processes = {'launcher_pid': 10, 'driver_pid': 11, 'master_pid': 12, 'wake_pid': 13,
                     'config_sha256': base_hash}
        receipt, receipt_hash = save('old-processes.json', processes)
        self.old = {'status': 'partial', 'stage': 'jobs', 'api_submission_attempted': False,
                    'config_sha256': base_hash, 'pid': 11}
        old_path, _ = save('old-run/endstate.json', self.old)
        release = {'parent_snapshot_ready': True, 'snapshot_manifest_sha256': 'a'*64}
        for label, value in [('candidate_config', self.candidate), ('test_evidence', {}),
                             ('geometry_submission', {'job_id': '7368009',
                              'dependency': 'afterok:7367626:7367628', 'submission_attempts': 1})]:
            path, sha = save(label + '.json', value)
            release[label + '_path'], release[label + '_sha256'] = path, sha
        save('release.json', release)
        self.release_spec = {**self.specification, 'base_config_path': base_path,
                             'base_config_sha256': base_hash, 'deadline_unix': 1792302800,
                             'snapshot_manifest_sha256': 'a'*64, 'old_endstate': old_path,
                             'old_process_ids': [10, 11, 12, 13],
                             'old_process_receipt_path': receipt,
                             'old_process_receipt_sha256': receipt_hash,
                             'inference_job_ids': ['7367626', '7367628']}
        return root

    def test_release_requires_exact_false_and_current_base_identity(self):
        root = self.release_fixture()
        with patch.object(controller.os, 'kill', side_effect=ProcessLookupError):
            controller.check_release(root, self.release_spec, self.base)
            for field, value in [('api_submission_attempted', None), ('api_submission_attempted', 0),
                                 ('api_submission_attempted', ''), ('config_sha256', 'b'*64), ('pid', 99)]:
                Path(self.release_spec['old_endstate']).write_text(json.dumps({**self.old, field: value}))
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    controller.check_release(root, self.release_spec, self.base)

    def test_release_rejects_unbound_empty_or_wrong_process_ids(self):
        root = self.release_fixture()
        with patch.object(controller.os, 'kill', side_effect=ProcessLookupError):
            for ids in ([], [10], [10, 11, 12, 99], [10, 11, 12, True]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    controller.check_release(root, {**self.release_spec, 'old_process_ids': ids}, self.base)


if __name__ == '__main__':
    unittest.main()
