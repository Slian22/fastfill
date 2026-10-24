from pathlib import Path
import json,hashlib,datetime
root=Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill')
files=[]
for arm,model in [('a021','27b'),('a028','27b-joint')]:
 p=root/'runs/unknownsize-20261013'/arm
 for rel in [f'{model}/a021-policy.json',f'{model}/run_receipt.json',f'{model}/train/run_manifest.json',f'{model}/train/training_guard.json',f'{model}/rows/MANIFEST.json',f'{model}/train/final/adapter_config.json','smoke/verdict.json','smoke/vllm-load.json']:
  files.append(p/rel)
files += [root/'runs/unknownsize-20261013/a028/compare/report.md',root/'runs/unknownsize-20261013/a028/compare/results.json',root/'runs/unknownsize-20261013/a028/rows/summary.json',root/'runs/unknownsize-20261013/a021/final-selection.json']
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':[],'logs':[]}
for p in files:
 b=p.read_bytes();result['files'].append({'path':str(p.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()})
for job in ['7419267','7461666','7461667','7461670']:
 for p in sorted((root/'runs/isambard').glob('*'+job+'*')):
  if not p.is_file():continue
  b=p.read_bytes();t=b.decode(errors='replace'); lines=t.replace('\r','\n').splitlines()
  markers=['step','loss','resume','checkpoint','done','complete','epoch','traceback','error','nan','inf','train','argv','command']
  matched=[l for l in lines if any(s in l.lower() for s in markers)]
  result['logs'].append({'path':str(p.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'line_count':len(lines),'first':lines[:35],'last':lines[-35:],'selected':matched[:12]+matched[-18:]})
print(json.dumps(result))
