from pathlib import Path
import json,hashlib,collections,datetime
root=Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013')
def keys(v):
 if isinstance(v,dict):return {k:(keys(x) if isinstance(x,(dict,list)) else type(x).__name__) for k,x in v.items()}
 if isinstance(v,list):return [keys(v[0])] if v else []
 return type(v).__name__
def marker(v):
 if isinstance(v,dict):return any((k=='no_reasoning_violation' and x is True) or (k=='error' and x=='reasoning_in_no_reasoning_system') or marker(x) for k,x in v.items())
 if isinstance(v,list):return any(marker(x) for x in v)
 return False
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'o2nt':{},'critic':{}}
for cohort in ['dev-select','confirm','cohort']:
 p=root/'a021/o2nt'/cohort
 receipts=sorted((p/'receipts').glob('*.json'));bad=[]
 for q in receipts:
  v=json.loads(q.read_text())
  if marker(v):bad.append({'path':str(q.relative_to(root)),'uid':v.get('uid'),'keys':keys(v)})
 result['o2nt'][cohort]={'receipt_count':len(receipts),'violation_count':len(bad),'violations':bad[:5],'sample_schema':keys(json.loads(receipts[0].read_text())) if receipts else None}
for name in ['round-0','round-1','test-round-0','test-round-1']:
 p=root/'a025'/name; b=p.with_name(name+'.critic')
 answers=sorted((b/'answers').glob('*.json'))
 answerkeys=collections.Counter(); missing=[]
 for q in answers:
  v=json.loads(q.read_text());answerkeys[','.join(sorted(v))]+=1
  if not all(k in v for k in ['content_sha256','expected_model']):missing.append(v.get('id'))
 summary=json.loads((p/'summary.json').read_text()) if (p/'summary.json').exists() else None
 result['critic'][name]={'answer_count':len(answers),'missing_identity_count':len(missing),'answer_keysets':dict(answerkeys),'state_summary':summary,'batch_summary':json.loads((b/'summary.json').read_text()) if (b/'summary.json').exists() else None}
print(json.dumps(result))
