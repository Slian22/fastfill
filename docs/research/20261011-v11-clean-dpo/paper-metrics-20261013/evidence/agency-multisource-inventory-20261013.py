"""v1.0 read-only release 18-source inventory; Python 3.13.1 stdlib.
No raw-dataset crawl, model execution or scoring. Output only /tmp.
"""
import json,hashlib,collections,sys
from pathlib import Path
ROOT=Path('/Users/slian/Desktop/3D/Worldedge/OptiScene');release=ROOT/'.release/v3.2/data/v3.2';stats=json.load((release/'stats.json').open());manifest=json.load((release/'MANIFEST.json').open())
counts={}
for source in manifest['args']['sources']:
 st=stats[source];counts[source]={'release_scanned':st['scanned'],'train':st.get('train_S1',0)+st.get('train_S2',0),'dev':st.get('dev',0),'test':st.get('test',0),'test_objects':0,'test_fixed':0,'test_groups':set(),'test_boundary_types':collections.Counter(),'test_flag_rooms':collections.Counter(),'test_tilted_objects':0,'test_unknown_front_objects':0,'test_eval_only_rooms':0,'first_group':None,'first_uid':None}
for line in (release/'test_rooms.jsonl').open():
 r=json.loads(line);d=counts[r['source']];d['test_objects']+=len(r['objects']);d['test_fixed']+=len(r.get('fixed',[]));d['test_groups'].add(r['group']);d['test_boundary_types'][r.get('boundary_type')]+=1;d['test_flag_rooms'].update(k for k,v in r.get('meta',{}).get('flags',{}).items() if v);d['test_tilted_objects']+=sum(bool(o.get('tilted')) for o in r['objects']);d['test_unknown_front_objects']+=sum(o.get('front_known') is False for o in r['objects']);d['test_eval_only_rooms']+=bool(r.get('meta',{}).get('eval_only'))
 if d['first_group'] is None:d['first_group']=r['group'];d['first_uid']=r['uid']
for d in counts.values():d['test_groups']=len(d['test_groups']);d['test_boundary_types']=dict(d['test_boundary_types']);d['test_flag_rooms']=dict(d['test_flag_rooms'])
result={'schema_version':1,'python':sys.version,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'release_manifest_path':str(release/'MANIFEST.json'),'release_manifest_sha256':hashlib.sha256((release/'MANIFEST.json').read_bytes()).hexdigest(),'test_sha256':hashlib.sha256((release/'test_rooms.jsonl').read_bytes()).hexdigest(),'sources':counts,'ir_inventory_names':sorted(p.name for p in Path('/Volumes/harddisk/fastfill_ir').iterdir() if p.suffix=='.jsonl'),'limitations':'Existing release statistics and exact test record metadata only. Mesh coverage was independently checked only for the two source subset; no new raw-source crawl.'}
Path('/tmp/agency-multisource-inventory-20261013.json').write_text(json.dumps(result,indent=2));print(json.dumps({s:{k:v for k,v in d.items() if k not in ['test_flag_rooms','first_uid']} for s,d in counts.items()},indent=2))
