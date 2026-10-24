from pathlib import Path
import hashlib,json,datetime
root=Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill')
p= json.loads((root/'runs/unknownsize-20261013/a021/27b/a021-policy.json').read_text())
j= json.loads((root/'runs/unknownsize-20261013/a028/27b-joint/a021-policy.json').read_text())
checks=[]
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
for name,expected in p['base']['files'].items():
 path=Path(p['base']['path'])/name
 with path.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
 checks.append({'path':str(path),'bytes':path.stat().st_size,'sha256':actual,'expected':expected,'passed':actual==expected})
print(json.dumps({'observed_start_utc':start,'observed_end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'shared_base_identity':p['base']==j['base'],'checks':checks,'passed':p['base']==j['base'] and all(x['passed'] for x in checks)}))
