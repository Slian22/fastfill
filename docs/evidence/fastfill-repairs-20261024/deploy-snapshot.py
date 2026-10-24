import datetime,hashlib,json,os,stat,subprocess,tarfile
from pathlib import Path
record={'content_sha256': 'daca130611a976b51193b7332c6464ad1b31214e43f2b7f29046dcc103dc17e1', 'archive': '/tmp/fastfill-repairs-20261024/repairs-20261024-daca130611a9.tar.gz', 'archive_sha256': '8b2e42276edf6f69e231f24427c5b13207ae2fb150ae5bcd6e474256096fa914', 'files': 494, 'bytes': 2108947, 'remote_directory': '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/snapshots/repairs-20261024-daca130611a9', 'remote_run': '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/repair-validation-20261024-daca130611a9', 'changes_from_previous_candidate': ['fastfill/v1_1/evaluation_protocol.py', 'fastfill/v1_1/tests/test_evaluation_protocol.py', 'fastfill/v1_1/tests/test_training_guard.py']}
run=Path(record['remote_run']); snapshot=Path(record['remote_directory'])
archive=run/'source.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==record['archive_sha256']
with tarfile.open(archive) as t:
 members=t.getmembers()
 assert all(m.isfile() and m.name.startswith('OptiScene/') and '..' not in Path(m.name).parts for m in members)
 manifest=json.load(t.extractfile('OptiScene/SNAPSHOT_MANIFEST.json'))
 expected={'OptiScene/'+p for p in manifest['files']}|{'OptiScene/SNAPSHOT_MANIFEST.json'}
 assert len(members)==len(expected) and {m.name for m in members}==expected
 assert manifest['content_sha256']==record['content_sha256']
 assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==record['content_sha256']
 snapshot.mkdir(mode=0o700)
 t.extractall(snapshot,filter='data')
code=snapshot/'OptiScene'
for name,digest in manifest['files'].items():
 p=code/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
for root,dirs,files in os.walk(snapshot,topdown=False):
 for name in files:
  p=Path(root)/name; p.chmod(0o555 if p.stat().st_mode&0o111 else 0o444)
 Path(root).chmod(0o555)
verification={'status':'SNAPSHOT_BYTES_VERIFIED','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_root':str(code),'content_sha256':record['content_sha256'],'archive_sha256':record['archive_sha256'],'files_verified':len(manifest['files']),'all_match':True,'read_only':True}
with (run/'snapshot-verification.json').open('x') as f:json.dump(verification,f,indent=2)
with (run/'submission-started.json').open('x') as f:json.dump({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
script=run/'server-validation.sbatch'
r=subprocess.run(['sbatch','--parsable','--output='+str(run/'slurm-%j.out'),'--error='+str(run/'slurm-%j.out'),str(script)],capture_output=True,text=True)
(run/'submission-stdout.txt').write_text(r.stdout);(run/'submission-stderr.txt').write_text(r.stderr)
assert r.returncode==0,(r.returncode,r.stderr)
job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout
submission={'job_id':job,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'cpu_only':True,'api_calls_planned':0,'new_model_training':False,'content_sha256':record['content_sha256']}
with (run/'submission.json').open('x') as f:json.dump(submission,f,indent=2)
print(json.dumps({'snapshot':verification,'submission':submission}))
