import json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path('/Users/slian/Desktop/3D/Worldedge/OptiScene')
work=Path(tempfile.mkdtemp(prefix='fastfill-shell-b-repro-'))
project=work/'project'; run=project/'runs/unknownsize-20261013'
bundle=run/'gpt-863099194937c5259a5aa8ac269f7a86f5b40cf752eabb9b789743d655ebcbb3/bundle/api-bundle'
bundle.mkdir(parents=True)
(bundle/'manifest.json').write_text('{}')
base=run/'judge-023-astra-high'
(base/'smoke/results').mkdir(parents=True)
(base/'smoke.completion.json').write_text('{"exit_code": 1}\n')
(base/'smoke/results/failure.json').write_text('{"status": "API_FAILED", "error": "smoke rejected"}\n')
bin_=work/'bin';bin_.mkdir()
(bin_/'squeue').write_text('#!/bin/bash\nexit 0\n')
(bin_/'sbatch').write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$AUDIT_SBATCH_LOG"\necho 12345\n')
for p in bin_.iterdir():p.chmod(0o755)
src=ROOT/'fastfill/comparison/submit_023_judge.sh'
script=work/src.name
script.write_text(src.read_text().replace('PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill',f'PROJECT_ROOT={project}'))
log=work/'sbatch.log'
r=subprocess.run(['bash',str(script),str(ROOT),'full'],text=True,capture_output=True,env={**os.environ,'PATH':str(bin_)+':'+os.environ['PATH'],'AUDIT_SBATCH_LOG':str(log)})
result={'workspace':str(work),'smoke_completion_exit_code':1,'smoke_success_records':0,'full_wrapper_exit':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip(),'submitted_full_batch':log.is_file(),'captured_submit':log.read_text() if log.exists() else None,'source_adaptation':'Only PROJECT_ROOT literal redirected to tmp; sbatch/squeue are local no-op stubs.'}
print(json.dumps({k:v for k,v in result.items() if k!='captured_submit'},indent=2))
assert r.returncode==0 and log.is_file() and '--expected-total 696' in log.read_text()
Path('/tmp/fastfill-audit-20261022/comparison-shell-b-repros.json').write_text(json.dumps(result,indent=2)+'\n')
