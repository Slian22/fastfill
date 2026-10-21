import json,os,stat,time,urllib.request,urllib.error
from pathlib import Path
out=Path("/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013/api-egress-001")
env=Path('/home/u6uv/shanliantian.u6uv/.fastfill_api.env')
r={'purpose':'compute_node_connectivity_only','api_scoring_calls':0,'credentials_read':False,'credential_file_present':env.is_file(),'credential_file_readable':os.access(env,os.R_OK),'credential_file_mode':oct(stat.S_IMODE(env.stat().st_mode)) if env.exists() else None,'started_unix':time.time()}
try:
 req=urllib.request.Request('https://api.krill-code.net/codex/v1/models',headers={'User-Agent':'fastfill-evaluation/1.0'})
 try:
  with urllib.request.urlopen(req,timeout=30) as response:r.update(http_status=response.status,network_reachable=True)
 except urllib.error.HTTPError as error:r.update(http_status=error.code,network_reachable=True)
except Exception as error:r.update(network_reachable=False,error_type=type(error).__name__)
r['finished_unix']=time.time()
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
raise SystemExit(0 if r.get('network_reachable') and r['credential_file_readable'] else 1)
