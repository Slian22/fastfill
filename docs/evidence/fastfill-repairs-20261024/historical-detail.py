from pathlib import Path
import json,subprocess,hashlib,datetime
root=Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013')
r={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'o2nt':{},'evaluation_dirs':{},'critic':{}}
for cohort in ['dev-select','confirm','cohort']:
 p=root/'a021/o2nt'/cohort
 bad=[]
 for q in (p/'receipts').glob('*.json'):
  v=json.loads(q.read_text())
  answers=v.get('answers',[])
  marked=[i for i,a in enumerate(answers) if a.get('no_reasoning_violation') or a.get('schema',{}).get('error')=='reasoning_in_no_reasoning_system']
  if marked:bad.append({'path':str(q.relative_to(root)),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'receipt':v,'marked_indices':marked})
 r['o2nt'][cohort]={'run_receipt':json.loads((p/'run_receipt.json').read_text()),'violations':bad}
for p in [root/'llm-eval-O2nt-P-v1',root/'a021/o2nt-scores']:
 r['evaluation_dirs'][str(p.relative_to(root))]=[x.name for x in p.iterdir()]
for name in ['round-0','round-1','test-round-0','test-round-1']:
 p=root/'a025'/name;b=p.with_name(name+'.critic')
 s=json.loads((p/'summary.json').read_text())
 r['critic'][name]={k:s.get(k) for k in ['status','round','stats','next_round_cases','code_sha256']}
 results=sorted(b.glob('cases/*/results/*.result.json'))
 r['critic'][name]['raw_result_count']=len(results)
 r['critic'][name]['sample_raw_result_keys']=sorted(json.loads(results[0].read_text())) if results else []
cp=subprocess.run(['sacct','-X','-u','shanliantian.u6uv','-S','2026-10-14','--format=JobID,JobName%80,State,ExitCode','--parsable2'],text=True,capture_output=True,timeout=30)
r['critic']['slurm_records']=[x for x in cp.stdout.splitlines() if '025' in x]
print(json.dumps(r))
