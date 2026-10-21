"""Submit only shard 1 and its two CPU successors; release separately after review."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill')
OPS = ROOT / 'ops/a027-resume-20261015-v1'
lock = (OPS / 'submit.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (OPS / 'jobs.json').exists(), 'This recovery was already submitted'
queue = subprocess.check_output(['squeue', '-h', '-u', 'shanliantian.u6uv', '-o', '%i|%j'], text=True)
assert not any('unknownsize-022-test-strict' in line for line in queue.splitlines()), queue
subprocess.run([sys.executable, '-B', str(OPS / 'verify.py'), 'capture'], check=True)
common = ['sbatch', '--parsable', '--account=brics.u6uv', '--partition=workq', '--qos=normal',
          '--nodes=1', '--ntasks=1', '--no-requeue',
          f'--output={ROOT}/runs/isambard/unknownsize-022-recovery-%j.out']
jobs, commands = {}, []
try:
    for stage, resources in (
        ('shard', ['--hold', '--exclude=nid011109', '--gpus-per-node=2', '--cpus-per-task=72',
                   '--mem=200G', '--time=12:00:00']),
        ('finalize', ['--cpus-per-task=72', '--mem=200G', '--time=04:00:00']),
        ('summarize', ['--cpus-per-task=32', '--mem=96G', '--time=06:00:00']),
    ):
        dependency = [] if stage == 'shard' else [
            '--kill-on-invalid-dep=yes', '--dependency=afterok:' + jobs['shard' if stage == 'finalize' else 'finalize']]
        command = [*common, f'--job-name=unknownsize-022-test-strict-recovery-{stage}',
                   *resources, *dependency, str(OPS / 'run.sbatch'), stage]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        assert job.isdecimal(), f'Invalid Slurm job ID: {job!r}'
        jobs = {**jobs, stage: job}
        commands = [*commands, command]
        print(stage, job, flush=True)
except BaseException:
    if jobs:
        subprocess.run(['scancel', *jobs.values()], check=False)
    raise
with (OPS / 'jobs.json').open('x') as stream:
    json.dump({'status': 'SUBMITTED_ROOT_HELD', 'utc': datetime.now(timezone.utc).isoformat(),
               'jobs': jobs, 'commands': commands, 'original_failed_job': '7449644',
               'original_snapshot': str(ROOT / 'snapshots/main-498348274')}, stream, indent=2)
print('Release after review: scontrol release ' + jobs['shard'], flush=True)
