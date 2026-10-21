"""Bounded replacement of five pending jobs, with durable per-action receipts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

helper = Path(sys.argv[1]).resolve()
original = helper.parent.parent
run = original / 'runs/fastfill_v1_1_clean_20261011'
output = run / 'evaluation_full'
frozen = original / 'automation/v11-clean-20261011-3e3eef72c515'
addendum = helper / 'docs/research/20261011-v11-clean-dpo/interpretation-addendum.json'
receipt_path = run / 'audit-launch.json'
old_jobs = ('7252878', '7252879', '7252880', '7252881', '7252889')
if receipt_path.exists():
    raise FileExistsError('Inspect existing audit-launch.json before further submissions')

def now():
    return datetime.now(timezone.utc).isoformat()

def call(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()

def inspect(job):
    return dict(re.findall(r'(\w+)=([^\s]*)', call(['scontrol', 'show', 'job', '-o', job])))

previous = {job: inspect(job) for job in old_jobs}
if any(value.get('JobState') != 'PENDING' or value.get('RunTime') != '00:00:00' for value in previous.values()):
    raise ValueError('Replacement is limited to never-running pending jobs')
manifest = json.loads((helper / 'audit-source-manifest.json').read_text())
state = {'started_at': now(), 'helper_root': str(helper), 'frozen_root': str(frozen),
         'run': str(run), 'output': str(output), 'helper_source_sha256': manifest['source_sha256'],
         'superseded': previous, 'jobs': {}, 'actions': [], 'complete': False,
         'gpu_hour_cap_unchanged': 84, 'cpu_audit_max_hours': 12.5,
         'scheduler_requeue': False, 'training_jobs_unchanged': ['7252874', '7252875']}

def save():
    temporary = receipt_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(receipt_path)

def submit(name, arguments):
    cmd = ['sbatch', '--parsable', '--hold', '--no-requeue', '--job-name=' + name,
           '--output=' + str(run / 'logs' / (name + '-%j.out')), *map(str, arguments)]
    job = call(cmd).split(';')[0]
    if not job.isdigit():
        raise ValueError('Unexpected sbatch job ID')
    state['jobs'][name] = {'job_id': job, 'command': cmd, 'submitted_at': now()}
    save()
    return job

save()
gpu_wrapper = helper / 'fastfill/v1_1/isambard_v1_1_verified_evaluate.sbatch'
cpu_wrapper = helper / 'fastfill/v1_1/isambard_v1_1_audit.sbatch'
continuations = []
for tag in ('dpo1', 'final'):
    args = [gpu_wrapper, helper, frozen, output, run, tag]
    first = submit(f'ff-verified-{tag}-1', ['--dependency=afterany:7252875', *args])
    second = submit(f'ff-verified-{tag}-2', ['--dependency=afterany:' + first, *args])
    continuations.append(second)
sft = submit('ff-cpu-sft-rescore', [cpu_wrapper, helper, output, addendum, 'sft'])
lineage_command = 'cd ' + shlex.quote(str(helper)) + ' && export PYTHONPATH=' + shlex.quote(str(helper))
lineage_command += ' && export TOKENIZERS_PARALLELISM=false && exec ' + shlex.join([
    str(original / 'env/bin/python'), '-m', 'fastfill.v1_1.training_guard', 'verify-lineage',
    '--out', str(run), '--tag', 'dpo1'])
lineage = submit('ff-cpu-dpo1-lineage', ['--account=brics.u6uv', '--partition=workq', '--nodes=1',
    '--ntasks=1', '--gpus=0', '--cpus-per-task=4', '--mem=32G', '--time=00:30:00', '--wrap', lineage_command])
submit('ff-audited-report', ['--dependency=afterany:' + ':'.join([*continuations, sft, lineage]),
                            cpu_wrapper, helper, output, addendum, 'report'])
for name, entry in state['jobs'].items():
    info = inspect(entry['job_id'])
    if info.get('JobState') != 'PENDING' or info.get('Requeue') != '0' or info.get('Priority') != '0':
        raise ValueError('Expected held, non-requeued submitted job')
    tres = dict(part.split('=', 1) for part in info['ReqTRES'].split(','))
    expected_gpu = '1' if name.startswith('ff-verified-') else '0'
    if tres.get('gres/gpu', '0') != expected_gpu:
        raise ValueError('Unexpected GPU request')
    entry['verified_slurm'] = info
save()

# Hold old pending work before the final check to close the scheduler-start race.
for job in old_jobs:
    call(['scontrol', 'hold', job])
    state['actions'].append({'at': now(), 'held_old_pending': job})
    save()
for job in old_jobs:
    info = inspect(job)
    if info.get('JobState') != 'PENDING' or info.get('RunTime') != '00:00:00' or info.get('Priority') != '0':
        raise ValueError('Old job did not remain held and never-running')

# Recheck right before cancellation. The report is canceled first so downstream
# afterany cancellation cannot cause the obsolete synthesis to start.
for job in ('7252889', '7252879', '7252881', '7252878', '7252880'):
    info = inspect(job)
    if info.get('JobState') != 'PENDING' or info.get('RunTime') != '00:00:00' or info.get('Priority') != '0':
        raise ValueError('Old job is no longer safe to replace')
    call(['scancel', job])
    state['actions'].append({'at': now(), 'cancelled_pending': job})
    save()
for entry in state['jobs'].values():
    call(['scontrol', 'release', entry['job_id']])
    state['actions'].append({'at': now(), 'released': entry['job_id']})
    save()
state['complete'] = True
state['completed_at'] = now()
save()
print(json.dumps(state))
