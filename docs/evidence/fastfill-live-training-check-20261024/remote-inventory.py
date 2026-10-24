from pathlib import Path
import json,os,subprocess,datetime
root=Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill')
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':subprocess.check_output(['hostname'],text=True).strip(),'runs':{}}
for arm in ['a021','a028']:
 p=root/'runs/unknownsize-20261013'/arm
 entries=[]
 for d,dirs,files in os.walk(p):
  rel=Path(d).relative_to(p)
  depth=len(rel.parts)
  dirs[:]=sorted(x for x in dirs if depth<3 and x not in {'predictions','raw','receipts','chunks','__pycache__'})
  for f in sorted(files):
   q=Path(d)/f
   if depth<=1 or any(w in f.lower() for w in ['manifest','policy','state','complete','train','report','smoke','step','history','log','config']):
    entries.append({'path':str(q.relative_to(root)),'bytes':q.stat().st_size})
 result['runs'][arm]=entries
commands={
 'queue':['squeue','-u','shanliantian.u6uv','-o','%.18i %.42j %.12T %.12M %.35R'],
 'accounting':['sacct','-X','-j','7461663,7461664,7461665,7461666,7461667,7461668,7461669,7461670,7419267,7406358','--format=JobID,JobName%40,State,ExitCode,Start,End,Elapsed,AllocTRES%70','--parsable2']}
for k,cmd in commands.items():
 cp=subprocess.run(cmd,text=True,capture_output=True,timeout=25);result[k]={'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
print(json.dumps(result,indent=2))
