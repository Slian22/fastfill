"""v2.0 Read-only original-asset join audit of synced frozen V1.1 test subset.
Python 3.13.1, stdlib only. Inputs read-only, outputs /tmp/agency-harddisk-*.json.
"""
import sys,json,math,hashlib,collections,zipfile
from pathlib import Path
ROOT=Path('/Users/slian/Desktop/3D/Worldedge/OptiScene')
BASE=Path('/Volumes/harddisk/3D_Room_Collections/imChuling__3D_Room_Collections')
ASSETS=ROOT/'huggingface/3D-SynthPlace_Scenes_Assets/assets_data'
TEST=ROOT/'docs/research/20261011-v11-clean-dpo/paper-metrics-20261013/inputs/server-paper-source-records.json'
def rows(p):
 with p.open() as f:
  for l in f:
   if l.strip():yield json.loads(l)
def sig(o):return(o['category'],tuple(max(round(x,2),.01) for x in o['size']),tuple(round(x,2) for x in o['pos'][:2]),round(math.degrees(o['yaw']))%360)
package=json.load(TEST.open());test=package['data'];uids={x['record']['uid'] for x in test}
ir={r['uid']:r for source in ['OptiScene_holodeck','IL3D_3dfront'] for r in rows(Path('/Volumes/harddisk/fastfill_ir')/(source+'.jsonl')) if r['uid'] in uids}
holo={d['scene_id']:d for d in json.load((BASE/'OptiScene_exported/holodeck.json').open()) if 'optiscene:'+d['scene_id'] in uids}
meta={d['model_id']:d for d in json.load((BASE/'IL3D_exported/IL3D/assets.json').open())}
z=zipfile.ZipFile(BASE/'IL3D_exported/IL3D/layout.zip');layouts={}
missing=[];unmatched=[];maps=[];scene_stats=[];examples={};mesh_cache={}
for wrapper in test:
 room=wrapper['record'];uid=room['uid'];source=room['source'];sr=ir.get(uid)
 scene={'uid':uid,'evaluation_row':wrapper['evaluation_row'],'source':source,'room_type':room['room_type'],'furniture_requested':0,'furniture_mapped':0,'missing_mesh_instances':0,'structures':0}
 if not sr:
  unmatched.append({'uid':uid,'reason':'IR_uid_not_found'});scene_stats.append(scene);continue
 lookup=collections.defaultdict(list)
 for o in sr['objects']:lookup[sig(o)].append(o)
 if source=='OptiScene_holodeck':
  export=holo[uid.removeprefix('optiscene:')]; furniture={f['furniture_instance_id']:f for f in export['rooms'][0]['furniture']}
 else:
  sid=room['meta']['il3d_scene_id']
  if sid not in layouts:layouts[sid]=json.loads(z.read('layout/'+sid+'.json'))
 for role,objects in [('movable',room['objects']),('fixed',room.get('fixed',[]))]:
  for o in objects:
   if o.get('structure') or o.get('fixed_kind')=='structure':scene['structures']+=1;continue
   scene['furniture_requested']+=1;matches=lookup.get(sig(o),[])
   if len(matches)>1:matches=[m for m in matches if round(m['pos'][2],2)==round(o['pos'][2],2)]
   if len(matches)!=1:
    unmatched.append({'uid':uid,'evaluation_row':wrapper['evaluation_row'],'object_id':o['id'],'role':role,'matches':len(matches),'object':o});continue
   original=matches[0];idx=int(original['id'].rsplit('::obj_',1)[1])
   if source=='OptiScene_holodeck':
    f=furniture[original['id']];raw=f['source_fields']['layout_object'];aid=f['furniture_asset_id'];sourceid=f['source_instance_id'];nativepath=None
   else:
    raw=layouts[sid]['objects'][idx];aid=raw['assetId'];sourceid=raw['id'];nativepath=raw.get('object_path')
   if aid not in mesh_cache:
    folder=ASSETS/aid; mesh_cache[aid]=next((str(folder/n) for n in ['raw_model.obj','raw_model3.obj'] if (folder/n).is_file()),None)
   mesh=mesh_cache[aid]
   row={'uid':uid,'evaluation_row':wrapper['evaluation_row'],'source':source,'room_type':room['room_type'],'role':role,'prompt_id':o['id'],'original_instance_id':original['id'],'source_index':idx,'source_instance_id':sourceid,'asset_id':aid,'mesh_path':mesh,'original_usdz_relative_path':nativepath,'gt_object':o,'ir_object':original,'layout_object':raw}
   maps.append(row);scene['furniture_mapped']+=1
   if source not in examples:examples[source]=row
   if not mesh:scene['missing_mesh_instances']+=1;missing.append({k:row[k] for k in ['uid','evaluation_row','prompt_id','asset_id','source','room_type','role','original_usdz_relative_path']})
 scene_stats.append(scene)
z.close()
def summarize(ss):
 selected={s['uid'] for s in ss};mm=[m for m in maps if m['uid'] in selected];missingmm=[m for m in mm if m['mesh_path'] is None]
 return{'rooms':len(ss),'furniture_requested':sum(s['furniture_requested'] for s in ss),'furniture_mapped':sum(s['furniture_mapped'] for s in ss),'structures':sum(s['structures'] for s in ss),'unique_assets':len({m['asset_id'] for m in mm}),'missing_asset_ids':len({m['asset_id'] for m in missingmm}),'missing_mesh_instances':len(missingmm),'rooms_with_missing_mesh':sum(s['missing_mesh_instances']>0 for s in ss),'complete_mesh_rooms':sum(s['furniture_requested']==s['furniture_mapped'] and not s['missing_mesh_instances'] for s in ss),'mapping_incomplete_rooms':sum(s['furniture_requested']!=s['furniture_mapped'] for s in ss)}
cohorts={source:{room:summarize([s for s in scene_stats if s['source']==source and s['room_type']==room]) for room in sorted({s['room_type'] for s in scene_stats if s['source']==source})} for source in ['OptiScene_holodeck','IL3D_3dfront']}
result={'schema_version':2,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version,'input_path':str(TEST),'input_sha256':hashlib.sha256(TEST.read_bytes()).hexdigest(),'source_test_receipt':package['source_test'],'asset_root':str(ASSETS),'summary':summarize(scene_stats),'cohorts':cohorts,'missing':missing,'unmatched':unmatched,'scenes':scene_stats,'examples':examples}
Path('/tmp/agency-harddisk-renderability-20261013.json').write_text(json.dumps(result,indent=2));Path('/tmp/agency-harddisk-asset-map-20261013.json').write_text(json.dumps(maps,indent=2))
print(json.dumps({'summary':result['summary'],'cohorts':cohorts,'unmatched_count':len(unmatched)},indent=2));print('EXAMPLES',json.dumps(examples,ensure_ascii=False)[:6000])
