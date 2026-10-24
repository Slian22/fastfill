import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path('/Users/slian/Desktop/3D/Worldedge/OptiScene')
sys.path.insert(0,str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1', WORLD_SIZE='1', RANK='0')

def support_handoff(tmp):
    from fastfill.v1_1 import roomgenbench_v1_1 as s
    from fastfill.v1_1.scene_v1_1 import apply_v1_1, gt_placements_v1_1
    from fastfill.v1_1.postprocess import repair_v1_1, check_v1_1
    from fastfill.validate import holds
    from fastfill.v2.direct_layout import export_handoff
    from fastfill.v2.roomgenbench import reference_check
    raw={'scene_key':'declared_support', 'room':{'dimensions':{'width':5.,'length':5.,'height':3.}}, 'objects':[
        {'id':'table_source','type':'table','description':'table','place_id':'floor',
         'position':{'x':1.5,'y':1.5,'z':0.},'rotation':{'x':0.,'y':0.,'z':270.},
         'dimensions':{'length':1.,'width':1.,'height':1.}},
        {'id':'lamp_source','type':'lamp','description':'lamp','place_id':'table_source',
         'position':{'x':1.5,'y':1.5,'z':1.},'rotation':{'x':0.,'y':0.,'z':270.},
         'dimensions':{'length':.2,'width':.2,'height':.4}}
    ]}
    inp=tmp/'scene.json'; inp.write_text(json.dumps(raw))
    room, meta=s.roomgenbench_scene_to_v1_1_room(inp)
    pl=gt_placements_v1_1(room)
    lid=next(o['id'] for o in room['objects'] if o['category']=='lamp')
    pl[lid]={**pl[lid],'on':None,'pos':[3.,3.,0.]}
    repaired=repair_v1_1(room, pl)
    pred=apply_v1_1(room,repaired)
    cond,layout=s.placements_to_v2_handoff(room,meta,repaired)
    dest=tmp/'handoff'; export_handoff(dest,cond,layout)
    ref=reference_check(dest,inp)
    return {'declared_constraints':room['constraints'],'constraint_satisfied':holds(room['constraints'][0],pred['objects'],room['boundary']),
        'repair_geometry_valid':check_v1_1(pred)['valid'],'exported_condition_supports':[o['support_parent'] for o in cond['objects']],
        'exported_constraints':cond['constraints'],'reference_check':ref['checks']}

def changed_pairs_resume(tmp):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM
    from safetensors.torch import load_file
    from fastfill.v1_1 import train_dpo
    torch.set_num_threads(1)
    backend=Tokenizer(WordLevel({'<unk>':0,'<eos>':1,'good':2,'bad':3,'prompt':4},unk_token='<unk>'))
    backend.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',eos_token='<eos>',pad_token='<eos>')
    tok.chat_template='prompt '
    base,out,pairs=tmp/'base',tmp/'train',tmp/'pairs.jsonl'
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=5,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=32,eos_token_id=1))
    model.save_pretrained(base); tok.save_pretrained(base)
    pair={'uid':'u','prompt':[{'role':'user','content':'prompt'}],'chosen':'good','rejected':'bad'}
    pairs.write_text(''.join(json.dumps({**pair,'uid':str(i)})+'\n' for i in range(4)))
    argv=['train_dpo','--model',str(base),'--pairs',str(pairs),'--out',str(out),'--epochs','1','--grad_accum','2',
        '--save_steps','1','--max_len','32','--lora_r','2','--lora_alpha','4']
    captured=io.StringIO()
    with patch.object(sys,'argv',argv),contextlib.redirect_stdout(captured): train_dpo.main()
    before=load_file(out/'final/adapter_model.safetensors')
    pair={**pair,'chosen':'bad','rejected':'good'}
    pairs.write_text(''.join(json.dumps({**pair,'uid':str(i)})+'\n' for i in range(4)))
    with patch.object(sys,'argv',argv),contextlib.redirect_stdout(captured): train_dpo.main()
    after=load_file(out/'final/adapter_model.safetensors')
    (tmp/'resume.log').write_text(captured.getvalue())
    return {'changed_training_pairs_accepted':True,'all_final_tensors_identical':all(torch.equal(before[k],after[k]) for k in before),
        'completion':json.loads((out/'final/completion.json').read_text()),'resume_log':str(tmp/'resume.log')}

if __name__=='__main__':
    tmp=Path(tempfile.mkdtemp(prefix='fastfill-system-v11-repro-'))
    values={'support_handoff':support_handoff(tmp),'changed_pairs_resume':changed_pairs_resume(tmp)}
    path=Path('/tmp/fastfill-audit-20261022/system-v11-repros.json')
    path.write_text(json.dumps(values,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'artifact':str(path),'support_handoff':values['support_handoff'],'resume':{
        k:v for k,v in values['changed_pairs_resume'].items() if k!='completion'}},indent=2,ensure_ascii=False))
