from datetime import datetime,timezone
import json,re,subprocess
from pathlib import Path
run=Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/fastfill_v1_1_clean_20261011')
p=run/'audit-launch.json';receipt=json.loads(p.read_text());job=receipt['jobs']['ff-verified-dpo1-1']['job_id']
assert job=='7255085'
def call(argv):return subprocess.run(argv,check=True,text=True,capture_output=True).stdout.strip()
def info():return dict(re.findall(r'(\w+)=([^\s]*)',call(['scontrol','show','job','-o',job])))
def save():
 tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(p)
before=info()
if before.get('JobState')!='PENDING' or before.get('RunTime')!='00:00:00':raise ValueError('Ready-stage optimization only applies before execution')
log=run/'logs/ff-cpu-dpo1-lineage-7255090.out'
proof=json.loads(log.read_text().strip())
if proof['completion']['admitted_pairs']!=7244 or proof['completion']['position']['step']!=906:raise ValueError('Expected full DPO1 proof')
change={'at':datetime.now(timezone.utc).isoformat(),'job_id':job,'reason':'DPO1 complete immutable-stage proof passed; overlap DPO1 full eval with DPO2 training; same recipe/data/budget','before':before,'commands':[],'proof_job':'7255090'}
receipt['post_submission_updates']=[*receipt.get('post_submission_updates',[]),change];save()
for args in (['scontrol','hold',job],):
 call(args);change['commands'].append(args);save()
held=info()
if held.get('JobState')!='PENDING' or held.get('RunTime')!='00:00:00' or held.get('Priority')!='0':raise ValueError('Job did not remain held and never running')
for args in (['scontrol','update','JobId='+job,'Dependency='],['scontrol','release',job]):
 call(args);change['commands'].append(args);save()
change['after']=info();change['completed_at']=datetime.now(timezone.utc).isoformat();save()
print(json.dumps(change))
