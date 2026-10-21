"""v1.0: read-only MTL/texture path check for the exact six paper strata.
Python 3.13.1 stdlib; no mesh/image decoding or render performed.
"""
import json,shlex,hashlib,itertools
from pathlib import Path
maps=json.load(open('/tmp/agency-harddisk-asset-map-20261013.json'))
selected=[r for r in maps if r['source']=='OptiScene_holodeck' or r['room_type'] in ['bedroom','living room']]
unique={r['asset_id']:r for r in selected};results=[]
for aid,r in sorted(unique.items()):
 p=Path(r['mesh_path']);mtls=[]
 with p.open(errors='replace') as f:
  for line in itertools.islice(f,256):
   if line.startswith('mtllib '):mtls.extend(shlex.split(line.strip())[1:])
 mpaths=[p.parent/x for x in mtls];missingmtl=[str(x) for x in mpaths if not x.is_file()];tex=set();unparsed=[]
 for m in mpaths:
  if not m.is_file():continue
  for line in m.read_text(errors='replace').splitlines():
   parts=shlex.split(line,comments=True)
   if parts and (parts[0].lower().startswith('map_') or parts[0].lower() in ['bump','norm','disp','decal','refl']):
    if len(parts)<2:unparsed.append(line)
    else:tex.add(str(m.parent/parts[-1]))
 results.append({'asset_id':aid,'mesh_path':str(p),'mtls':[str(m) for m in mpaths],'texture_paths':sorted(tex),'missing_mtl':missingmtl,'missing_textures':[x for x in sorted(tex) if not Path(x).is_file()],'mtllib_not_in_first_256_lines':not bool(mtls),'unparsed':unparsed})
report={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'1009 exact-stratum test rooms; path existence only, no render/image decode','asset_count':len(results),'mtllib_header_unresolved':sum(r['mtllib_not_in_first_256_lines'] for r in results),'assets_with_missing_mtl':sum(bool(r['missing_mtl']) for r in results),'assets_with_missing_texture':sum(bool(r['missing_textures']) for r in results),'unique_texture_paths':len({t for r in results for t in r['texture_paths']}),'unparsed_lines':sum(len(r['unparsed']) for r in results),'assets':results}
Path('/tmp/agency-harddisk-material-check-20261013.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='assets'},indent=2));print('ISSUE_EXAMPLES',json.dumps([r for r in results if r['missing_mtl'] or r['missing_textures'] or r['mtllib_not_in_first_256_lines']][:4]))
