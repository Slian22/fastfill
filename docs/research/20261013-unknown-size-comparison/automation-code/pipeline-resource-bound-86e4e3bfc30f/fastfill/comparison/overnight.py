"""Bounded local comparison driver. No work occurs on import.

CLI: python -m fastfill.comparison.overnight --config CONFIG.json
Required JSON keys: ssh_host, remote_root, remote_run_root, geometry_job_id,
prerequisite_job_ids, local_code_root, local_python, output_root, reference_data,
blender, vendor, weights, remote_api_code, api_client_path, remote_env_file.
remote_env_file must be /home/u6uv/shanliantian.u6uv/.fastfill_api.env. Remote code/client
paths refer to the SERVER. local_code_root must be a finalized frozen snapshot.
receipt_paths lists exact run-root-relative small training/inference receipts.
Optional remote_python (python3), slurm_account (brics.u6uv),
ssh_control_path (an existing task-owned local SSH control socket),
expected_manifest_sha256 (mapping geometry-v1/eval-data-v1/main-data-v1 to hashes),
final_tables (false), poll_seconds (60), read_retries (3). Time caps in seconds:
total_timeout_seconds=432000, render_timeout_seconds=86400, api_timeout_seconds=36000.
The five-day total bound includes scheduler waiting; it does not extend job runtimes.

Output must not exist. status.json, endstate.json and receipts are atomically
published and fsynced. There is deliberately NO resume or API resubmission.
An interrupted submission is unknown, never retried. A remote Slurm CPU job
has a timeout independent of SSH and leaves job.json/completion.json. On local
termination the API may continue within that deadline; receipts say so.
The parent owns caffeinate/launcher lifetime; this module creates no wake lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time


CAPS = {'total_timeout_seconds': 120 * 3600, 'render_timeout_seconds': 24 * 3600,
        'api_timeout_seconds': 10 * 3600}
TERMINAL_FAILURES = {'BOOT_FAIL', 'CANCELLED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
                     'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'SPECIAL_EXIT', 'TIMEOUT'}
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', '-o',
               'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result = {**result, key: value}
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def relative_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or
            any(ord(c) < 32 for c in name) or PurePosixPath(name).is_absolute() or
            '..' in PurePosixPath(name).parts or name in {'.', ''} or name.startswith('-')):
        raise ValueError('path must be relative and stay inside snapshot')
    return name


def safe_file(root, name):
    path = Path(root)
    if path.is_symlink():
        raise ValueError('snapshot symlink forbidden')
    for part in PurePosixPath(relative_name(name)).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('snapshot symlink forbidden')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_manifest(root, expected=None):
    root = Path(root)
    manifest = safe_file(root, 'MANIFEST.json')
    digest = sha256(manifest)
    if expected is not None and digest != expected:
        raise ValueError('manifest hash mismatch: ' + str(root))
    files = read_json(manifest).get('files')
    if not isinstance(files, dict):
        raise ValueError('manifest requires files mapping')
    for name, entry in files.items():
        wanted = entry.get('sha256') if isinstance(entry, dict) else entry
        if not isinstance(wanted, str) or not re.fullmatch('[0-9a-f]{64}', wanted):
            raise ValueError('invalid manifest hash')
        if sha256(safe_file(root, name)) != wanted:
            raise ValueError('artifact hash mismatch: ' + name)
    if sha256(manifest) != digest:
        raise ValueError('manifest changed during verification')
    return digest


def validate_config(raw):
    required = ('ssh_host remote_root remote_run_root geometry_job_id prerequisite_job_ids '
                'local_code_root local_python output_root reference_data blender vendor weights '
                'remote_api_code api_client_path remote_env_file').split()
    if not isinstance(raw, dict) or any(k not in raw for k in required):
        raise ValueError('missing required config field')
    config = {**{'poll_seconds': 60, 'read_retries': 3, 'receipt_paths': [],
                 'remote_python': 'python3', 'final_tables': False,
                 'slurm_account': 'brics.u6uv', 'slurm_partition': 'workq', 'slurm_qos': 'normal'}, **CAPS, **raw}
    for key in required:
        if key in ('prerequisite_job_ids', 'geometry_job_id'):
            continue
        if not isinstance(config[key], str) or not config[key] or any(ord(c) < 32 for c in config[key]):
            raise ValueError('invalid config field: ' + key)
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', config['ssh_host']) or config['ssh_host'].startswith('-'):
        raise ValueError('invalid ssh_host')
    if 'ssh_control_path' in config:
        path = config['ssh_control_path']
        if (not isinstance(path, str) or not path.startswith('/') or '%' in path
                or '$' in path or '..' in Path(path).parts or len(path.encode()) > 100
                or any(ord(c) < 32 for c in path)):
            raise ValueError('invalid SSH control socket path')
    for key in CAPS:
        val = config[key]
        if type(val) not in (int, float) or not math.isfinite(val) or not 0 < val <= CAPS[key]:
            raise ValueError('invalid bound: ' + key)
    if type(config['poll_seconds']) not in (int, float) or not 60 <= config['poll_seconds'] <= 300:
        raise ValueError('poll_seconds must be in [60, 300]')
    if type(config['read_retries']) is not int or not 0 <= config['read_retries'] <= 10:
        raise ValueError('read_retries must be 0..10')
    if type(config['final_tables']) is not bool:
        raise ValueError('final_tables must be boolean')
    if config['remote_env_file'] != '/home/u6uv/shanliantian.u6uv/.fastfill_api.env':
        raise ValueError('remote_env_file must name the approved server-only env file')
    for key in ('slurm_account', 'slurm_partition', 'slurm_qos'):
        if not isinstance(config[key], str) or not re.fullmatch('[A-Za-z0-9_.-]+', config[key]):
            raise ValueError('invalid Slurm selector: ' + key)
    ids = config['prerequisite_job_ids']
    if not isinstance(ids, list):
        raise ValueError('prerequisite_job_ids must be a list')
    jobs = [str(v) for v in [*ids, config['geometry_job_id']]]
    if any(not re.fullmatch('[1-9][0-9]*', v) for v in jobs) or len(set(jobs)) != len(jobs):
        raise ValueError('invalid or duplicate job IDs')
    if not isinstance(config['receipt_paths'], list):
        raise ValueError('receipt_paths must be a list')
    for path in config['receipt_paths']:
        relative_name(path)
    if len(set(config['receipt_paths'])) != len(config['receipt_paths']):
        raise ValueError('duplicate receipt path')
    root = PurePosixPath(config['remote_root'])
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('remote_root must be absolute')
    run_root = PurePosixPath(config['remote_run_root'])
    if not run_root.is_absolute():
        run_root = root / relative_name(str(run_root))
    if not run_root.is_relative_to(root) or '..' in run_root.parts or run_root == root:
        raise ValueError('remote_run_root must be beneath remote_root')
    config = {**config, 'remote_run_root': str(run_root), 'geometry_job_id': jobs[-1],
              'prerequisite_job_ids': jobs[:-1], 'remote_env_file': config.get(
                  'remote_env_file', str(root / '.fastfill_api.env'))}
    for key in ('remote_api_code', 'api_client_path', 'remote_env_file'):
        if (not isinstance(config[key], str) or any(ord(c) < 32 for c in config[key])
                or not PurePosixPath(config[key]).is_absolute() or '..' in PurePosixPath(config[key]).parts):
            raise ValueError(key + ' must be an absolute remote path')
    if (not isinstance(config['remote_python'], str) or not config['remote_python']
            or config['remote_python'].startswith('-') or any(ord(c) < 32 for c in config['remote_python'])):
        raise ValueError('invalid remote_python')
    for key in ('local_code_root', 'local_python', 'output_root', 'reference_data', 'blender', 'vendor', 'weights'):
        config = {**config, key: str(Path(config[key]).expanduser().absolute())}
    out = Path(config['output_root']).resolve()
    for key in ('local_code_root', 'reference_data', 'vendor'):
        other = Path(config[key]).resolve()
        if out.is_relative_to(other) or other.is_relative_to(out):
            raise ValueError('output must be outside input snapshots')
    return config


def load_config(path):
    return validate_config(read_json(path))


def parse_sacct(text, job_ids):
    records = {}
    for line in text.splitlines():
        parts = line.strip().split('|')
        if len(parts) >= 3 and parts[0] in set(map(str, job_ids)):
            state = parts[1].split()[0].rstrip('+') if parts[1] else 'UNKNOWN'
            record = {'state': state, 'exit_code': parts[2]}
            if parts[0] in records and records[parts[0]] != record:
                raise ValueError('conflicting primary job records')
            records = {**records, parts[0]: record}
    return records


def inventory(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('invalid snapshot directory: ' + str(root))
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('snapshot symlink forbidden')
    return {p.relative_to(root).as_posix(): sha256(safe_file(root, p.relative_to(root).as_posix()))
            for p in sorted(root.rglob('*')) if p.is_file()
            and not any(v in {'.git', '__pycache__', '.pytest_cache'} for v in p.relative_to(root).parts)
            and p.suffix != '.pyc'}


class Driver:
    def __init__(self, config):
        self.config = validate_config(config)
        self.out = Path(self.config['output_root'])
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.deadline = self.started + self.config['total_timeout_seconds']
        self.stage, self.sequence = 'initialize', 0
        self.counts, self.failures, self.frozen = {}, [], None
        self.remote_batch, self.remote_job = None, None
        self.api_attempted = False
        self.child = None
        atomic_json(self.out / 'config.json', self.config)
        self.config_hash = sha256(self.out / 'config.json')
        self.status('running')
        # Even SIGKILL leaves an explicit incomplete result rather than a success.
        atomic_json(self.out / 'endstate.json', self.state('partial', reason='driver_active_or_interrupted'))

    def state(self, status, **extra):
        return {'schema_version': 1, 'status': status, 'stage': self.stage,
                'pid': os.getpid(), 'updated_unix': time.time(),
                'elapsed_seconds': time.monotonic() - self.started,
                'config_sha256': self.config_hash, 'counts': self.counts,
                'failures': self.failures, 'api_submission_attempted': self.api_attempted,
                'remote_batch': self.remote_batch, 'remote_job': self.remote_job, **extra}

    def status(self, status='running'):
        atomic_json(self.out / 'status.json', self.state(status))
        if status == 'running':
            atomic_json(self.out / 'endstate.json', self.state('partial', reason='driver_active_or_interrupted'))

    def remaining(self, limit=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('configured driver deadline exhausted')
        return min(remaining, limit) if limit is not None else remaining

    def stop_child(self):
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.child.wait()

    def command(self, argv, timeout=120, input_text=None):
        timeout = self.remaining(timeout)
        self.sequence += 1
        receipt = self.out / 'receipts' / f'{self.sequence:04d}-{self.stage}.json'
        record = {'argv': list(map(str, argv)), 'timeout_seconds': timeout,
                  'started_unix': time.time(), 'stage': self.stage, 'status': 'attempted'}
        atomic_json(receipt, record)
        try:
            env = {**os.environ, 'PYTHONPATH': self.config['local_code_root'],
                   'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '3', 'MKL_NUM_THREADS': '3'}
            self.child = subprocess.Popen(list(map(str, argv)), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
                cwd=self.config['local_code_root'], env=env)
            stdout, stderr = self.child.communicate(input=input_text, timeout=timeout)
            record = {**record, 'status': 'finished', 'returncode': self.child.returncode,
                      'stdout': stdout, 'stderr': stderr, 'finished_unix': time.time()}
            atomic_json(receipt, record)
            if self.child.returncode:
                raise subprocess.CalledProcessError(self.child.returncode, argv)
            return stdout
        except BaseException as exc:
            self.stop_child()
            atomic_json(receipt, {**record, 'status': 'failed', 'error_type': type(exc).__name__})
            raise
        finally:
            self.child = None

    def ssh_options(self):
        path = self.config.get('ssh_control_path')
        return [*SSH_OPTIONS, *(['-o', 'ControlMaster=no', '-o', 'ControlPath=' + path] if path else [])]

    def ssh(self, command, *, read_only=False, timeout=120, deadline=None):
        for attempt in range(1 + (self.config['read_retries'] if read_only else 0)):
            budget = min(timeout, (deadline or self.deadline) - time.monotonic())
            if budget <= 0:
                raise TimeoutError('SSH stage deadline exhausted')
            try:
                return self.command(['ssh', *self.ssh_options(), self.config['ssh_host'], command],
                                    timeout=self.remaining(budget))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if not read_only or attempt == self.config['read_retries']:
                    raise
                self.pause(min(self.config['poll_seconds'], max(0, (deadline or self.deadline) - time.monotonic())))

    def pause(self, seconds):
        until = min(self.deadline, time.monotonic() + seconds)
        while time.monotonic() < until:
            time.sleep(min(60, self.remaining(), until - time.monotonic()))

    def rsync(self, source, target):
        # macOS openrsync speaks protocol 29 and has no --protect-args. Quote
        # remote shell operands explicitly; local paths remain argv elements.
        def operand(value):
            prefix = self.config['ssh_host'] + ':'
            return prefix + shlex.quote(value[len(prefix):]) if value.startswith(prefix) else value
        self.command(['rsync', '-rt', '--safe-links', '--timeout=120',
                      '-e', shlex.join(['ssh', *self.ssh_options()]), operand(source), operand(target)],
                     timeout=self.remaining(7200))

    def advance(self, stage):
        self.remaining()
        self.stage = stage
        self.status()
        if self.frozen is not None:
            self.verify_freeze()
        downloaded = self.out / 'download-receipt.json'
        if downloaded.exists() and inventory(self.out / 'inputs') != read_json(downloaded)['files']:
            raise ValueError('downloaded input hash mismatch after freeze')
        self.remaining()

    def freeze(self):
        c = self.config
        verify_manifest(c['reference_data'])
        for key in ('local_python', 'blender', 'weights'):
            if not Path(c[key]).is_file():
                raise ValueError('missing local resource: ' + key)
        for name in ('visual_pipeline.py', 'fid.py', 'judge_batch.py'):
            safe_file(c['local_code_root'], 'fastfill/comparison/' + name)
        if c['final_tables']:
            safe_file(c['local_code_root'], 'fastfill/comparison/final_tables.py')
        self.frozen = {'trees': {key: inventory(c[key]) for key in
                       ('local_code_root', 'reference_data', 'vendor')},
                       'files': {key: sha256(c[key]) for key in ('weights', 'blender', 'local_python')}}
        atomic_json(self.out / 'freeze.json', self.frozen)

    def verify_freeze(self):
        if sha256(self.out / 'config.json') != self.config_hash:
            raise ValueError('config receipt hash mismatch')
        if read_json(self.out / 'freeze.json') != self.frozen:
            raise ValueError('freeze receipt changed')
        for key, files in self.frozen['trees'].items():
            if inventory(self.config[key]) != files:
                raise ValueError('frozen tree hash mismatch: ' + key)
        for key, digest in self.frozen['files'].items():
            if sha256(self.config[key]) != digest:
                raise ValueError('frozen file hash mismatch: ' + key)

    def poll_geometry(self):
        jobs = [*self.config['prerequisite_job_ids'], self.config['geometry_job_id']]
        while True:
            self.remaining()
            raw = self.ssh(shlex.join(['sacct', '-n', '-P', '-j', ','.join(jobs),
                                      '--format=JobIDRaw,State,ExitCode']), read_only=True)
            records = parse_sacct(raw, jobs)
            atomic_json(self.out / 'jobs.json', records)
            self.counts = {**self.counts, 'jobs_known': len(jobs), 'jobs_observed': len(records),
                           'jobs_completed': sum(r['state'] == 'COMPLETED' for r in records.values())}
            self.status()
            for job, record in records.items():
                if record['state'] in TERMINAL_FAILURES or (record['state'] == 'COMPLETED' and record['exit_code'] != '0:0'):
                    raise RuntimeError(f'job {job}: {record["state"]} {record["exit_code"]}')
            if len(records) == len(jobs) and all(r['state'] == 'COMPLETED' for r in records.values()):
                return
            queue = self.ssh(shlex.join(['squeue', '-h', '-j', ','.join(jobs), '-o', '%i|%T|%r']), read_only=True)
            atomic_json(self.out / 'queue.json', {'raw': queue})
            for line in queue.splitlines():
                parts = line.split('|')
                if parts[0] in jobs and 'DependencyNeverSatisfied' in line:
                    raise RuntimeError('dependency never satisfied: ' + parts[0])
            self.pause(self.config['poll_seconds'])

    def fetch_inputs(self):
        c, dest = self.config, self.out / 'inputs'
        pinned = c.get('expected_manifest_sha256', {})
        for name in ('geometry-v1', 'eval-data-v1'):
            target = dest / name
            target.mkdir(parents=True)
            self.rsync(f'{c["ssh_host"]}:{c["remote_run_root"]}/{name}/', str(target) + '/')
            digest = verify_manifest(target, pinned.get(name))
            atomic_json(self.out / (name + '-receipt.json'), {'manifest_sha256': digest})
        files = ['main-data-v1/MANIFEST.json', *c['receipt_paths']]
        remote_hashes = self.ssh('cd ' + shlex.quote(c['remote_run_root']) + ' && '
            + shlex.join(['sha256sum', '--', *files]), read_only=True)
        expected = {}
        for line in remote_hashes.splitlines():
            digest, name = line.split(maxsplit=1)
            expected = {**expected, name.lstrip('*'): digest}
        if set(expected) != set(files):
            raise ValueError('missing remote receipt hashes')
        for name in files:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            self.rsync(f'{c["ssh_host"]}:{c["remote_run_root"]}/{name}', str(target))
            if sha256(safe_file(dest, name)) != expected[name]:
                raise ValueError('downloaded receipt hash mismatch: ' + name)
        training_hash = sha256(dest / 'main-data-v1/MANIFEST.json')
        if pinned.get('main-data-v1', training_hash) != training_hash:
            raise ValueError('main-data manifest hash mismatch')
        evaluation = read_json(dest / 'eval-data-v1/MANIFEST.json')
        if evaluation.get('training_manifest_sha256', training_hash) != training_hash:
            raise ValueError('evaluation/training manifest hash mismatch')
        geometry = read_json(dest / 'geometry-v1/summary.json')
        if geometry.get('data_manifest_sha256') != sha256(dest / 'eval-data-v1/MANIFEST.json'):
            raise ValueError('geometry/evaluation manifest hash mismatch')
        atomic_json(self.out / 'download-receipt.json', {'files': inventory(dest)})
        self.counts = {**self.counts, 'downloaded_files': len(inventory(dest)), 'training_receipts': len(c['receipt_paths'])}

    def local_module(self, module, args, timeout):
        return self.command([self.config['local_python'], '-B', '-u', '-m', 'fastfill.comparison.' + module,
                             *map(str, args)], timeout=self.remaining(timeout))

    def render(self):
        c, inputs, out = self.config, self.out / 'inputs', self.out / 'visual'
        self.local_module('visual_pipeline', ['prepare', '--reference-dir', c['reference_data'],
            '--geometry-dir', inputs / 'geometry-v1', '--eval-data', inputs / 'eval-data-v1',
            '--out', out, '--blender-path', c['blender'], '--workers', '2', '--threads', '3',
            '--timeout-seconds', c['render_timeout_seconds']], c['render_timeout_seconds'])
        verify_manifest(out)
        preparation = read_json(out / 'preparation.json')
        if preparation.get('status') != 'prepared' or not isinstance(preparation.get('counts'), dict):
            raise ValueError('missing visual preparation counts')
        self.counts = {**self.counts, 'visual': preparation['counts']}
        if any(type(preparation['counts'].get(k)) is not int or preparation['counts'][k] < 0
               for k in ('frozen', 'rendered', 'failed', 'gpt_selected', 'api_cases')):
            raise ValueError('invalid visual preparation counts')
        if preparation['counts'].get('failed', 0):
            self.failures = [*self.failures, {'stage': 'render', 'reason': 'render_case_failures',
                                           'count': preparation['counts']['failed']}]

    def fid(self):
        c = self.config
        self.local_module('fid', ['--manifest', self.out / 'visual/fid-manifest.json',
            '--out', self.out / 'fid.json', '--vendor', c['vendor'], '--weights', c['weights'],
            '--threads', '3', '--device', 'cpu'], self.remaining())
        result = read_json(self.out / 'fid.json')
        counts = result.get('counts', {})
        if (any(type(counts.get(k)) is not int or counts[k] < 0 for k in
                ('requested', 'reference_eligible', 'shared_success')) or
                counts['requested'] != self.counts['visual']['frozen']):
            raise ValueError('missing or inconsistent FID counts')
        self.counts = {**self.counts, 'fid': counts}
        if result.get('failures'):
            self.failures = [*self.failures, {'stage': 'fid', 'reason': 'fid_case_failures',
                                           'count': len(result['failures'])}]

    def bundle(self):
        visual, dest = self.out / 'visual', self.out / 'api-upload'
        dest.mkdir()
        shutil.copytree(visual / 'api-bundle', dest / 'api-bundle', symlinks=True)
        for name in ('api-mapping.json', 'api-failures.json'):
            shutil.copyfile(visual / name, dest / name)
        manifest = read_json(dest / 'api-bundle/manifest.json')
        cases = manifest['cases']
        mapping = read_json(dest / 'api-mapping.json')
        if len(mapping) > 348 or len(cases) > 348 or len({c['id'] for c in cases}) != len(cases):
            raise ValueError('API duplicate/cap violation')
        for case in cases:
            entries = [{'path': case['prompt_file'], 'sha256': case['prompt_sha256']}, *case['images']]
            for item in entries:
                if sha256(safe_file(dest / 'api-bundle', item['path'])) != item['sha256']:
                    raise ValueError('API artifact hash mismatch')
        atomic_json(dest / 'MANIFEST.json', {'files': inventory(dest)})
        digest = verify_manifest(dest)
        self.remote_batch = self.config['remote_run_root'] + '/gpt-' + digest
        self.counts = {**self.counts, 'api_cases': len(cases), 'api_selected': len(mapping)}
        return dest

    def launch_api(self):
        c, dest = self.config, self.bundle()
        q = shlex.quote
        self.ssh('mkdir -- ' + q(self.remote_batch))  # Exclusive immutable remote namespace.
        self.rsync(str(dest) + '/', c['ssh_host'] + ':' + self.remote_batch + '/bundle/')
        files = inventory(dest)
        checks = '\n'.join(digest + '  ' + name for name, digest in files.items()) + '\n'
        self.ssh('cd ' + q(self.remote_batch + '/bundle') + ' && printf %s ' + q(checks) + ' | sha256sum --check --strict -')
        # Read-only files prevent accidental edits; judge wrapper copies to its own fresh output.
        self.ssh('chmod -R a-w -- ' + q(self.remote_batch + '/bundle'))
        wrapper = c['remote_api_code'] + '/fastfill/comparison/judge_batch.py'
        remote_code = self.ssh(shlex.join(['sha256sum', '--', wrapper, c['api_client_path']]), read_only=True)
        code_hashes = dict((line.split(maxsplit=1)[1], line.split(maxsplit=1)[0])
                           for line in remote_code.splitlines())
        local_wrapper = Path(c['local_code_root']) / 'fastfill/comparison/judge_batch.py'
        if (set(code_hashes) != {wrapper, c['api_client_path']} or
                code_hashes.get(wrapper) != sha256(local_wrapper) or
                any(not re.fullmatch('[0-9a-f]{64}', v) for v in code_hashes.values())):
            raise ValueError('remote API code hash mismatch')
        client_pin = c.get('api_client_sha256')
        if client_pin is None:
            client_relative = PurePosixPath(c['api_client_path']).relative_to(c['remote_api_code'])
            client_pin = sha256(safe_file(c['local_code_root'], client_relative.as_posix()))
        if client_pin != code_hashes[c['api_client_path']]:
            raise ValueError('remote API client hash mismatch')
        atomic_json(self.out / 'api-code-freeze.json', code_hashes)
        allowance = self.remaining(c['api_timeout_seconds'])
        if allowance <= 5:
            raise TimeoutError('insufficient API watchdog time')
        seconds = int(allowance) - 5
        # Queueing may use the remaining driver window; execution is capped at
        # ten hours by both Slurm and timeout. A now+walltime deadline would make
        # almost every queued submission ineligible before it could start.
        self.api_deadline = self.deadline
        judge = [c['remote_python'], '-B', '-u', c['remote_api_code'] + '/fastfill/comparison/judge_batch.py',
                 '--bundle', self.remote_batch + '/bundle', '--output', self.remote_batch + '/results',
                 '--client', c['api_client_path'], '--max-cases', '348', '--timeout', '90',
                 '--expected-total', str(len(read_json(dest / 'api-mapping.json')))]
        script = ('#!/bin/bash\nset -u\ncd ' + q(c['remote_api_code']) + '\n'
            + 'export PYTHONPATH=' + q(c['remote_api_code']) + '\n'
            + 'export FASTFILL_API_ENV=' + q(c['remote_env_file']) + '\n'
            + 'printf %s ' + q(remote_code) + ' | sha256sum --check --strict - && '
            + 'timeout --signal=TERM --kill-after=5s ' + str(seconds) + 's ' + shlex.join(judge) + '\n'
            + 'rc=$?\nprintf \'{"exit_code":%s}\\n\' "$rc" > ' + q(self.remote_batch + '/completion.json.tmp') + '\n'
            + 'mv -- ' + q(self.remote_batch + '/completion.json.tmp') + ' ' + q(self.remote_batch + '/completion.json') + '\nexit "$rc"\n')
        atomic_json(self.out / 'api-launch.json', {'status': 'submission_attempted',
            'remote_batch': self.remote_batch, 'deadline_seconds': seconds, 'argv': judge})
        self.api_attempted = True
        self.status()
        wall = seconds + 5
        walltime = f'{wall // 3600:02d}:{wall % 3600 // 60:02d}:{wall % 60:02d}'
        deadline = (datetime.now(timezone.utc) + timedelta(seconds=self.remaining())).strftime('%Y-%m-%dT%H:%M:%S')
        submission = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=1', '--mem=4G', '--gres=none', '--time=' + walltime,
            '--deadline=' + deadline, '--account=' + c['slurm_account'],
            '--partition=' + c['slurm_partition'], '--qos=' + c['slurm_qos'],
            '--job-name=fastfill-judge', '--output=' + self.remote_batch + '/launch.log',
            self.remote_batch + '/launch.sh']
        command = ('set -eu\nmkdir -- ' + q(self.remote_batch + '/launch.claim') + '\n'
            + 'printf %s ' + q(script) + ' > ' + q(self.remote_batch + '/launch.sh') + '\n'
            + 'TZ=UTC ' + shlex.join(submission) + ' > ' + q(self.remote_batch + '/submission.txt') + '\n'
            + 'job=$(cat ' + q(self.remote_batch + '/submission.txt') + '); job=${job%%;*}\n'
            + 'printf \'{"job_id":"%s"}\\n\' "$job" > ' + q(self.remote_batch + '/job.json.tmp') + '\n'
            + 'mv -- ' + q(self.remote_batch + '/job.json.tmp') + ' ' + q(self.remote_batch + '/job.json')
            + '\nprintf "%s\\n" "$job"')
        job = self.ssh(command).strip()
        if not re.fullmatch('[1-9][0-9]*', job):
            raise ValueError('ambiguous API submission; never resubmit')
        self.remote_job = {'kind': 'slurm', 'job_id': job, 'receipt': self.remote_batch + '/job.json'}
        atomic_json(self.out / 'api-job.json', self.remote_job)
        self.status()

    def poll_api(self):
        q = shlex.quote
        while True:
            if time.monotonic() >= self.api_deadline:
                raise TimeoutError('API deadline exhausted; remote watchdog owns termination')
            job = self.remote_job['job_id']
            raw_job = self.ssh(shlex.join(['sacct', '-n', '-P', '-j', job,
                              '--format=JobIDRaw,State,ExitCode']), read_only=True,
                              deadline=self.api_deadline)
            records = parse_sacct(raw_job, [job])
            atomic_json(self.out / 'api-sacct.json', records)
            raw = self.ssh('if test -f ' + q(self.remote_batch + '/completion.json') + '; then cat '
                          + q(self.remote_batch + '/completion.json') + '; else printf "{}\\n"; fi',
                          read_only=True, deadline=self.api_deadline)
            receipt = json.loads(raw)
            if 'exit_code' in receipt:
                atomic_json(self.out / 'api-completion.json', receipt)
                if type(receipt['exit_code']) is not int or not 0 <= receipt['exit_code'] <= 255:
                    raise ValueError('invalid API completion exit code')
                if records.get(job, {}).get('state') in TERMINAL_FAILURES and receipt['exit_code'] == 0:
                    raise ValueError('API completion disagrees with Slurm failure')
                return receipt['exit_code']
            if records.get(job, {}).get('state') in TERMINAL_FAILURES | {'COMPLETED'}:
                raise RuntimeError('API job terminal without completion receipt: ' + records[job]['state'])
            queue = self.ssh(shlex.join(['squeue', '-h', '-j', job, '-o', '%i|%T|%r']),
                             read_only=True, deadline=self.api_deadline)
            if 'DependencyNeverSatisfied' in queue:
                raise RuntimeError('API dependency never satisfied')
            self.pause(min(self.config['poll_seconds'], max(0, self.api_deadline - time.monotonic())))

    def fetch_results(self):
        dest = self.out / 'remote-results'
        dest.mkdir(exist_ok=True)
        # Includes wrapper output, per-case attempts/results, launch logs and job ID.
        self.rsync(self.config['ssh_host'] + ':' + self.remote_batch + '/', str(dest) + '/')
        atomic_json(self.out / 'results-receipt.json', {'files': inventory(dest)})
        if self.remote_job is None and (dest / 'job.json').exists():
            self.remote_job = {'kind': 'slurm', **read_json(dest / 'job.json')}
        summary = dest / 'results/summary.json'
        if summary.exists():
            data = read_json(summary)
            counts = data.get('counts', {})
            if (any(type(counts.get(k)) is not int or counts[k] < 0 for k in
                    ('selected', 'runnable', 'launched', 'success', 'failed')) or
                    counts['selected'] != self.counts['api_selected'] or
                    not counts['success'] <= counts['launched'] <= counts['runnable'] <= counts['selected'] or
                    counts['success'] + counts['failed'] != counts['selected']):
                raise ValueError('missing or inconsistent judge counts')
            self.counts = {**self.counts, 'judge': counts}
            if data.get('counts', {}).get('failed', 0):
                self.failures = [*self.failures, {'stage': 'api', 'reason': 'judge_case_failures',
                                               'count': data['counts']['failed']}]

    def partial_counts(self):
        """Report observed records without inferring unobserved successes."""
        visual = self.out / 'visual'
        if 'visual' not in self.counts and (visual / 'preparation.json').exists():
            prep = read_json(visual / 'preparation.json')
            ledger = [row for path in visual.glob('chunks/*/render/ledger.json')
                      for row in read_json(path).get('cases', [])]
            self.counts = {**self.counts, 'visual_partial': {
                'frozen': len(prep.get('expected_uids', [])), 'observed': len(ledger),
                'rendered': sum(r.get('status') == 'rendered' for r in ledger),
                'failed': sum(r.get('status') == 'failed' for r in ledger)}}
        results = self.out / 'remote-results/results'
        if 'judge' not in self.counts and results.exists():
            records = []
            if (results / 'records.jsonl').exists():
                for line in (results / 'records.jsonl').read_text().splitlines():
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        break  # Last interrupted line is not a completed receipt.
            self.counts = {**self.counts, 'judge_partial': {
                'selected': self.counts.get('api_selected'), 'recorded': len(records),
                'attempt_receipts': len(list(results.rglob('*.attempt.json'))),
                'completed': sum(r.get('status') == 'completed' for r in records),
                'failed': sum(r.get('status') == 'failed' for r in records)}}

    def aggregate(self):
        judge = self.out / 'remote-results/results/summary.json'
        if not self.config['final_tables']:
            return
        if not judge.exists():
            raise ValueError('final tables require judge summary')
        self.local_module('final_tables', ['--geometry', self.out / 'inputs/geometry-v1',
            '--fid', self.out / 'fid.json', '--judge', judge, '--out', self.out / 'final-results'], self.remaining())

    def execute(self):
        try:
            self.freeze()
            for stage, action in [('jobs', self.poll_geometry), ('download', self.fetch_inputs),
                                  ('render', self.render), ('fid', self.fid), ('api_submit', self.launch_api)]:
                self.advance(stage)
                action()
            self.advance('api_wait')
            code = self.poll_api()
            self.advance('results')
            self.fetch_results()
            if not (self.out / 'remote-results/results/summary.json').exists():
                raise ValueError('missing judge summary after remote completion')
            if code:
                self.failures = [*self.failures, {'stage': 'api', 'reason': 'remote_exit', 'exit_code': code}]
            self.advance('final_tables')
            self.aggregate()
            self.advance('finished')
            state = self.state('partial' if self.failures else 'completed')
        except (Exception, KeyboardInterrupt) as exc:
            self.stop_child()
            failed_stage = self.stage
            self.failures = [*self.failures, {'stage': failed_stage, 'reason': str(exc), 'type': type(exc).__name__}]
            if self.api_attempted and self.remote_batch and self.deadline - time.monotonic() > 0:
                try:
                    self.fetch_results()
                except Exception as fetch_error:
                    self.failures = [*self.failures, {'stage': 'fetch_partial', 'type': type(fetch_error).__name__}]
            state = self.state('partial', reason='interrupted' if isinstance(exc, KeyboardInterrupt) else 'stage_failed',
                               remote_may_be_active=self.api_attempted and not (self.out / 'api-completion.json').exists())
        try:
            self.partial_counts()
        except (OSError, ValueError, TypeError) as exc:
            self.failures = [*self.failures, {'stage': 'partial_counts', 'type': type(exc).__name__}]
        state = {**state, 'counts': self.counts, 'failures': self.failures}
        atomic_json(self.out / 'endstate.json', state)
        atomic_json(self.out / 'status.json', state)
        return state


def run(config):
    return Driver(config).execute()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('signal interruption')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        result = run(load_config(args.config))
    except (OSError, ValueError) as exc:
        parser.exit(2, 'overnight: ' + str(exc) + '\n')
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
