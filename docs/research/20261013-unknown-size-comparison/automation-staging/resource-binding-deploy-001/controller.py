"""Gated visual-only handoff; all remote jobs and API code stay unchanged.

Start only after the parent declares the tested snapshot ready. This opens a
fresh authenticated master and waits for release.json in a NEW state directory.
The parent receipts the old-controller handoff separately. Existing geometry
submission evidence is reused; this handoff cannot change any remote job.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time


CHANGED_CONFIG_KEYS = {'local_code_root', 'output_root',
                       'ssh_control_path', 'total_timeout_seconds'}
ORIGINAL_DEADLINE_UNIX = 1792302800


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    """Atomic, fsynced, exclusive publication; existing receipts never change."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def remaining(deadline, now=None):
    if (not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or
            not math.isfinite(deadline) or deadline > ORIGINAL_DEADLINE_UNIX):
        raise ValueError('deadline must not exceed original authorized cutoff')
    seconds = math.floor(deadline - (time.time() if now is None else now))
    if seconds <= 0:
        raise TimeoutError('original deployment deadline exhausted')
    return seconds


def effective_config(base, candidate, spec, now=None):
    changed = {key for key in set(base) | set(candidate) if base.get(key) != candidate.get(key)}
    if changed - CHANGED_CONFIG_KEYS:
        raise ValueError('unapproved config changes: ' + ','.join(sorted(changed - CHANGED_CONFIG_KEYS)))
    for key in ('local_code_root', 'remote_api_code', 'api_client_path',
                'output_root', 'remote_run_root', 'ssh_control_path'):
        if candidate[key] != spec[key]:
            raise ValueError('deployment/config mismatch: ' + key)
    geometry = str(candidate['geometry_job_id'])
    if (not re.fullmatch('[1-9][0-9]*', geometry) or
            geometry != str(base['geometry_job_id']) or
            geometry in set(map(str, base['prerequisite_job_ids']))):
        raise ValueError('existing geometry ID required; all remote jobs stay fixed')
    if Path(candidate['output_root']).exists():
        raise FileExistsError(candidate['output_root'])
    # Leave a startup margin. The independent controller alarm still enforces
    # the absolute original deadline if the child interpreter starts slowly.
    budget = min(base['total_timeout_seconds'], remaining(spec['deadline_unix'], now) - 10)
    if budget <= 0:
        raise TimeoutError('no driver startup budget before original deadline')
    return {**candidate, 'total_timeout_seconds': budget}


def old_identity(spec, base, old):
    if Path(spec['old_endstate']).resolve() != (Path(base['output_root']) / 'endstate.json').resolve():
        raise ValueError('old endstate must belong to base run')
    if sha(spec['old_process_receipt_path']) != spec['old_process_receipt_sha256']:
        raise ValueError('old process receipt hash mismatch')
    receipt = read(spec['old_process_receipt_path'])
    expected = [receipt[k] for k in ('launcher_pid', 'driver_pid', 'master_pid', 'wake_pid')]
    ids = spec['old_process_ids']
    if (not isinstance(ids, list) or len(ids) != 4 or
            any(type(pid) is not int or pid <= 0 for pid in ids + expected) or
            len(set(ids)) != 4 or set(ids) != set(expected)):
        raise ValueError('old process IDs must exactly match owned-process receipt')
    if (old.get('config_sha256') != spec['base_config_sha256'] or
            receipt.get('config_sha256') != spec['base_config_sha256'] or
            old.get('pid') != receipt['driver_pid']):
        raise ValueError('old endstate/processes must match base config and driver')


def check_release(state, spec, base):
    release = read(state / 'release.json')
    if release.get('parent_snapshot_ready') is not True:
        raise ValueError('parent snapshot-ready receipt required')
    for path_key, digest_key in (('candidate_config_path', 'candidate_config_sha256'),
                                 ('test_evidence_path', 'test_evidence_sha256'),
                                 ('geometry_submission_path', 'geometry_submission_sha256')):
        if sha(release[path_key]) != release[digest_key]:
            raise ValueError('release receipt hash mismatch: ' + path_key)
    if release['snapshot_manifest_sha256'] != spec['snapshot_manifest_sha256']:
        raise ValueError('released snapshot differs from ready master specification')
    old = read(spec['old_endstate'])
    if (old.get('status') != 'partial' or old.get('stage') != 'jobs' or
            old.get('api_submission_attempted') is not False):
        raise ValueError('old run must have stopped in jobs with no API attempt')
    old_identity(spec, base, old)
    for pid in spec['old_process_ids']:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise ValueError('old controller/child still exists: ' + str(pid))
    candidate = read(release['candidate_config_path'])
    submission = read(release['geometry_submission_path'])
    if (str(submission['job_id']) != str(candidate['geometry_job_id']) or
            submission['dependency'] != 'afterok:' + ':'.join(map(str, spec['inference_job_ids'])) or
            submission.get('submission_attempts') != 1):
        raise ValueError('geometry submission identity/dependency mismatch')
    return effective_config(base, candidate, spec)


def stop(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
    except ProcessLookupError:
        proc.poll()


def interrupted(signum, _frame):
    raise SystemExit(124 if signum == signal.SIGALRM else 130)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    children, state, driver = [], None, None
    outcome, code = 'startup_failure', 2
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(signum, interrupted)
    signal.alarm(30)  # Covers configuration I/O too.
    try:
        spec = read(args.spec)
        if spec.get('parent_snapshot_ready') is not True:
            raise ValueError('staging spec is not authorized for launch')
        base = read(spec['base_config_path'])
        if sha(spec['base_config_path']) != spec['base_config_sha256']:
            raise ValueError('base configuration changed')
        if sha(Path(spec['local_code_root']) / 'MANIFEST.json') != spec['snapshot_manifest_sha256']:
            raise ValueError('snapshot manifest differs from parent freeze')
        # Import only the new frozen stdlib driver, not the mutable workspace.
        sys.path.insert(0, spec['local_code_root'])
        from fastfill.comparison.overnight import verify_manifest, validate_config
        verify_manifest(spec['local_code_root'], spec['snapshot_manifest_sha256'])
        if spec['deadline_unix'] > spec['original_deadline_unix']:
            raise ValueError('deadline extension forbidden')
        signal.alarm(remaining(spec['deadline_unix']))
        sock = Path(spec['ssh_control_path'])
        if (not sock.is_absolute() or len(str(sock).encode()) > 100 or
                any(c in str(sock) for c in '%$\n\r') or '..' in sock.parts or sock.exists()):
            raise ValueError('fresh valid control socket required')
        info = sock.parent.lstat()
        if (sock.parent.is_symlink() or not stat.S_ISDIR(info.st_mode) or
                info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError('control parent must be owned private 0700 directory')
        args.state.mkdir(mode=0o700)
        state = args.state
        publish(state / 'spec.json', spec)
        options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
                   '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
                   '-o', 'ForkAfterAuthentication=no']
        with (state / 'master.log').open('x') as log:
            master = subprocess.Popen(['/usr/bin/ssh', *options, '-o', 'ForwardAgent=no',
                '-o', 'ForwardX11=no', '-o', 'ControlPersist=no', '-M', '-N', '-n',
                '-S', str(sock), base['ssh_host']], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(master)
        until = time.monotonic() + min(60, remaining(spec['deadline_unix']))
        while True:
            result = subprocess.run(['/usr/bin/ssh', '-S', str(sock), '-O', 'check', base['ssh_host']],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                break
            if master.poll() is not None or time.monotonic() >= until:
                raise RuntimeError('new master readiness failed; old controller untouched')
            time.sleep(1)
        probe = subprocess.run(['/usr/bin/ssh', '-vv', *options, '-o', 'ControlMaster=no',
            '-o', 'ProxyCommand=/usr/bin/false', '-S', str(sock), base['ssh_host'], 'true'],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True, timeout=30)
        if ('mux_client_request_session: master session id:' not in probe.stderr or
                'Received exit status from master 0' not in probe.stderr):
            raise RuntimeError('actual multiplexed session completion unverified')
        publish(state / 'ready.json', {'launcher_pid': os.getpid(), 'master_pid': master.pid,
            'deadline_unix': spec['deadline_unix'], 'control_path': str(sock),
            'driver_started': False, 'observed_unix': time.time(),
            'multiplex_evidence': [line for line in probe.stderr.splitlines()
                                   if 'mux_client' in line or 'Received exit status from master' in line]})
        outcome = 'waiting_for_parent_release'
        while not (state / 'release.json').exists():
            remaining(spec['deadline_unix'])
            if master.poll() is not None:
                raise RuntimeError('new master lost before release; no driver started')
            time.sleep(1)
        candidate = check_release(state, spec, base)
        verify_manifest(spec['local_code_root'], spec['snapshot_manifest_sha256'])
        config = validate_config(candidate)
        publish(state / 'driver-config.json', config)
        env = {**os.environ, 'PYTHONPATH': config['local_code_root'], 'PYTHONDONTWRITEBYTECODE': '1'}
        with (state / 'driver.log').open('x') as log:
            driver = subprocess.Popen([config['local_python'], '-B', '-u', '-m',
                'fastfill.comparison.overnight', '--config', str(state / 'driver-config.json')],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=config['local_code_root'], env=env)
        children.append(driver)
        wake = subprocess.Popen(['/usr/bin/caffeinate', '-ims', '-w', str(driver.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        children.append(wake)
        publish(state / 'processes.json', {'launcher_pid': os.getpid(), 'driver_pid': driver.pid,
            'master_pid': master.pid, 'wake_pid': wake.pid, 'deadline_unix': spec['deadline_unix'],
            'effective_driver_budget_seconds': config['total_timeout_seconds'],
            'config_sha256': sha(state / 'driver-config.json')})
        outcome = 'driver_active'
        code = driver.wait()
        outcome = 'driver_exited'
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        outcome = 'absolute_deadline' if code == 124 else 'interrupted'
        raise
    except Exception as exc:
        outcome = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        signal.alarm(0)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        errors = []
        for proc in reversed(children):
            try:
                stop(proc)
            except Exception as exc:
                errors.append({'pid': proc.pid, 'error': type(exc).__name__})
        if state is not None:
            publish(state / 'end.json', {'outcome': outcome, 'controller_exit_code': code,
                'driver_exit_code': driver.returncode if driver else None,
                'finished_unix': time.time(), 'cleanup_errors': errors,
                'tracked_children_stopped': all(proc.poll() is not None for proc in children),
                'limitation': 'tracked child groups only; up to35s shutdown grace per child'})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
