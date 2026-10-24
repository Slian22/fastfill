from pathlib import Path
import tempfile, json, sys
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from fastfill.v2.roomgenbench import write_benchmark_requests
with tempfile.TemporaryDirectory(prefix='v2-scene-key-') as root:
    root=Path(root)
    reference=root/'reference'; (reference/'scenes').mkdir(parents=True)
    victim=reference/'protected.json'; victim.write_text('{"original":"do not change"}\n')
    source=reference/'scenes'/'scene.json'
    scene={'scene_key':'../reference/protected', 'room_type':'study', 'room':{'dimensions':{'width':5.,'length':4.,'height':3.}},
           'objects':[{'id':'source_desk','type':'desk','description':'wood desk','place_id':'floor'}]}
    source.write_text(json.dumps(scene))
    out=root/'requests'
    summary=write_benchmark_requests(source.parent,out,roomgenbench_root=reference)
    assert out.is_dir()
    overwritten=json.loads(victim.read_text())
    assert 'furniture_list' in overwritten and 'original' not in overwritten
    print(json.dumps({'result':'BUG_REPRODUCED','summary':summary,'output_files':[p.name for p in out.iterdir()],
                      'outside_reference_file_overwritten':True,'victim_after_keys':sorted(overwritten)}))
