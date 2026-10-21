import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill')
run = root / 'runs/fastfill_v1_1_clean_20261011'
output = run / 'evaluation_full'
launch = json.loads((run / 'audit-launch.json').read_text())
helper, frozen = Path(launch['helper_root']), Path(launch['frozen_root'])
jobs = ['7252874', '7252875', *launch['superseded'], *[v['job_id'] for v in launch['jobs'].values()]]

def call(args):
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return proc.stdout

def identity_check(folder, name):
    manifest = json.loads((folder / name).read_text())
    return {'source_sha256': manifest['source_sha256'], 'files': len(manifest['files']),
            'mismatches': [p for p, h in manifest['files'].items()
                           if hashlib.sha256((folder / p).read_bytes()).hexdigest() != h]}

def logs(path):
    if not path.exists():
        return {'present': False}
    lines = path.read_text(errors='replace').replace('\r', '\n').splitlines()
    parsed = []
    for line in lines:
        if line.startswith('{'):
            try:
                parsed.append(json.loads(line))
            except ValueError:
                pass
    return {'present': True, 'bytes': path.stat().st_size,
            'mtime': datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat(),
            'tail': [line[:1200] for line in lines[-12:]],
            'last_json': parsed[-1] if parsed else None}

result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'original_source': identity_check(frozen, 'source-manifest.json'),
          'helper_source': identity_check(helper, 'audit-source-manifest.json'),
          'squeue': call(['squeue', '-h', '-j', ','.join(jobs), '-o', '%i|%T|%M|%R']),
          'sacct': call(['sacct', '-j', ','.join(jobs), '--parsable2', '--format',
              'JobID,JobName%30,State,ExitCode,Start,End,Elapsed,AllocTRES%100']),
          'verification': {}, 'logs': {}, 'spooled_scripts': {}}
for tag in ('sft', 'dpo1', 'final'):
    p = output / 'verification' / f'{tag}.json'
    result['verification'][tag] = json.loads(p.read_text()) if p.exists() else None
for name in ('ff-cpu-sft-rescore', 'ff-cpu-dpo1-lineage', 'ff-audited-report'):
    result['logs'][name] = logs(run / 'logs' / (name + '-' + launch['jobs'][name]['job_id'] + '.out'))
result['logs']['training'] = logs(run / 'logs/ff-clean-train-1-7252874.out')
for name, script in [('ff-verified-dpo1-1', 'isambard_v1_1_verified_evaluate.sbatch'),
                     ('ff-cpu-sft-rescore', 'isambard_v1_1_audit.sbatch'),
                     ('ff-audited-report', 'isambard_v1_1_audit.sbatch')]:
    entry = launch['jobs'][name]
    contents = call(['scontrol', 'write', 'batch_script', entry['job_id'], '-'])
    result['spooled_scripts'][name] = contents.encode() == (helper / 'fastfill/v1_1' / script).read_bytes()
result['report_exists'] = (output / 'research-summary.json').exists()
print(json.dumps(result, indent=2))
