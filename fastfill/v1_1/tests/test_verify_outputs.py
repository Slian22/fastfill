"""CPU re-scoring checks use real geometry, real receipts, and the frozen import path."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from fastfill.v1_1 import automate, evaluate_v1_1 as evaluator, evaluation_protocol as protocol
from fastfill.v1_1 import verify_outputs as verifier


@pytest.fixture
def complete_matrix(tmp_path):
    root = Path(__file__).resolve().parents[3]
    data, run, output = (tmp_path / name for name in ('data', 'run', 'evaluation'))
    data.mkdir(); run.mkdir()
    room = {'uid':'u', 'source':'source', 'group':'g', 'boundary':[[0,0],[4,0],[4,4],[0,4]],
            'boundary_type':'polygon', 'height':3, 'objects':[{'id':'chair_1','category':'chair',
            'size':[.5,.5,1], 'pos':[1,1,0], 'yaw':0, 'anchor':'floor'}]}
    for split in protocol.SPLITS:
        (data / f'{split}.jsonl').write_text('\n'.join(json.dumps({**room,'uid':uid}) for uid in ('u','v'))+'\n')
    for model in automate.MODELS.values():
        path=run/model;path.mkdir()
        for name in ('config.json','tokenizer_config.json','tokenizer.json'):
            (path/name).write_text('{}')
        header=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
        (path/'model.safetensors').write_bytes(len(header).to_bytes(8,'little')+header+b'\0'*4)
    automate.prepare(root,data,run,output)
    frozen=json.loads((output/'protocol.json').read_text())
    for tag, model in automate.MODELS.items():
        model_id=protocol.model_identity(run/model)
        cells={}
        for split in protocol.SPLITS:
            target=output/tag/split;target.mkdir(parents=True)
            identity=automate._identity(frozen,model_id,split)
            text='{"placements":[{"id":"chair_1","size":[.5,.5,1],"pos":[2,2,0],"yaw":0}]}'
            # Real valid JSON, followed by a missing generation to cover failure denominators.
            text=text.replace('[.5,.5,1]','[0.5,0.5,1]')
            rooms=[{**room,'uid':uid} for uid in ('u','v')]
            rows=evaluator._score_chunk(rooms,[text,None],1)
            predicted=[{**row,'output':t,'evaluation_row':i} for i,(row,t) in enumerate(zip(rows,[text,None]))]
            (target/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in predicted)+'\n')
            protocol.write_json_atomic(target/'metrics.json',evaluator.summarize_v1_1(rows))
            protocol.finalize_evaluation(target,identity)
            cells[split]={'status':'complete','identity':identity}
        protocol.write_json_atomic(output/'state'/f'{tag}.json',{'status':'complete','model_identity':model_id,'cells':cells})
    return root, output


def verify(matrix, tags=('sft',), workers=1):
    root, output=matrix
    return verifier.verify_worker(output, tags, root, workers)


def resign_cell(output, tag='sft', split='dev_rooms'):
    target=output/tag/split
    identity=json.loads((target/'completion.json').read_text())['identity']
    protocol.finalize_evaluation(target,identity)


def test_real_full_tag_recomputation_receipt(complete_matrix):
    result=verify(complete_matrix)
    assert result['status']=='complete' and result['verified_cells']==4
    receipt=json.loads((complete_matrix[1]/'verification'/'sft.json').read_text())
    assert receipt['status']=='complete'
    assert receipt['cells']['dev_rooms']['verified_rows']==2
    assert receipt['cells']['dev_rooms']['identity']['code_hashes']
    assert receipt['verifier']['sha256']


def test_all_models_and_parallel_scoring(complete_matrix):
    assert verify(complete_matrix,tuple(automate.MODELS),workers=2)['verified_cells']==12


@pytest.mark.parametrize('kind',['score','metrics','output','evaluation_row','extra_field'])
def test_internally_resigned_but_wrong_artifacts_fail_rescore(complete_matrix,kind):
    output=complete_matrix[1];target=output/'sft'/'dev_rooms'
    if kind=='metrics':
        path=target/'metrics.json';d=json.loads(path.read_text());d['in_dist']['accepted_rate']=1.0
        protocol.write_json_atomic(path,d)
    else:
        path=target/'predictions.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        if kind=='score': rows[0]['zero_collision_accepted']=False
        elif kind=='output': rows[0]['output']='bad output'
        elif kind=='evaluation_row': rows[0]['evaluation_row']=True
        else: rows[0]['invented_score']=1
        path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    resign_cell(output)
    with pytest.raises(ValueError, match='mismatch|differs'):
        verify(complete_matrix)
    assert json.loads((output/'verification'/'sft.json').read_text())['status']=='failed'


def test_missing_completion_replaces_prior_pass_with_failed_receipt(complete_matrix):
    verify(complete_matrix)
    (complete_matrix[1]/'sft'/'test_rooms'/'completion.json').unlink()
    with pytest.raises(ValueError,match='completion'):
        verify(complete_matrix)
    assert json.loads((complete_matrix[1]/'verification'/'sft.json').read_text())['status']=='failed'


def test_wrong_scoring_root_rejected(complete_matrix,tmp_path):
    with pytest.raises(ValueError, match='scoring root|code root'):
        verifier.verify_worker(complete_matrix[1],('sft',),tmp_path,1)


def test_data_identity_drift_rejected(complete_matrix):
    p=complete_matrix[1].parent/'data'/'test_rooms.jsonl'
    p.write_text(p.read_text().replace('"height": 3','"height": 4'))
    with pytest.raises(ValueError,match='dataset|identity|protocol'):
        verify(complete_matrix)


def test_state_identity_drift_rejected(complete_matrix):
    p=complete_matrix[1]/'state'/'sft.json';s=json.loads(p.read_text());s['cells']['dev_rooms']['identity']['settings']['tp']=17
    protocol.write_json_atomic(p,s)
    with pytest.raises(ValueError,match='identity|state'):
        verify(complete_matrix)


def test_model_identity_drift_rejected(complete_matrix):
    p=complete_matrix[1].parent/'run'/'sft_merged'/'model.safetensors'
    p.write_bytes(p.read_bytes()[:-4]+b'\1'*4)
    with pytest.raises(ValueError,match='model|identity'):
        verify(complete_matrix)


@pytest.mark.parametrize('workers', [1, 2])
def test_cli_launches_separate_frozen_import_process(complete_matrix, workers):
    root,output=complete_matrix
    result=subprocess.run([sys.executable,'-m','fastfill.v1_1.verify_outputs','--output',str(output),
        '--tag','sft','--scoring-root',str(root),'--workers',str(workers)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((output/'verification'/'sft.json').read_text())['status']=='complete'


@pytest.mark.parametrize('tags,workers',[((),1),(('bad',),1),(('sft',),0)])
def test_invalid_arguments_rejected(complete_matrix,tags,workers):
    with pytest.raises(ValueError): verify(complete_matrix,tags,workers)


def test_matched_receipt_reuses_scores_but_still_validates_artifacts(complete_matrix, monkeypatch):
    root, output = complete_matrix
    verify(complete_matrix)
    monkeypatch.setattr(verifier, '_rescore', lambda task: pytest.fail('should reuse matched receipt'))
    result = verifier.verify_worker(output, ('sft',), root, 1, reuse_verified=True)
    assert result['cached_cells'] == 4
    path = output/'sft'/'dev_rooms'/'metrics.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='completion'):
        verifier.verify_worker(output, ('sft',), root, 1, reuse_verified=True)


def test_cache_is_invalidated_by_runtime_drift(complete_matrix, monkeypatch):
    root, output = complete_matrix
    verify(complete_matrix)
    original = verifier._runtime()
    monkeypatch.setattr(verifier, '_runtime', lambda: {**original, 'sys_version': 'different version'})
    assert verifier.verify_worker(output, ('sft',), root, 1, reuse_verified=True)['cached_cells'] == 0


def test_inputs_changed_during_scoring_are_rejected(complete_matrix, monkeypatch):
    original = verifier._verify_cell
    def changed(*args):
        result = original(*args)
        path = complete_matrix[1]/'runner.json'
        path.write_text(path.read_text()+'\n')
        return result
    monkeypatch.setattr(verifier, '_verify_cell', changed)
    with pytest.raises(ValueError, match='changed during'):
        verify(complete_matrix)


def test_interpreter_mismatch_rejected(complete_matrix):
    p=complete_matrix[1]/'runner.json';runner=json.loads(p.read_text());runner['python']='/unrelated/python'
    protocol.write_json_atomic(p,runner)
    with pytest.raises(ValueError,match='interpreter'):
        verify(complete_matrix)


@pytest.mark.parametrize('damage', ['bad_json', 'missing_cell', 'false_metrics', 'verifier_digest'])
def test_corrupt_cached_receipts_trigger_full_reverification(complete_matrix, damage):
    root, output = complete_matrix
    verify(complete_matrix)
    path = output/'verification'/'sft.json'
    if damage == 'bad_json':
        path.write_text('{bad')
    else:
        value = json.loads(path.read_text())
        if damage == 'missing_cell': del value['cells']['dev_rooms']
        elif damage == 'false_metrics': value['cells']['dev_rooms']['metrics_equal'] = False
        else: value['verifier']['sha256'] = 'changed'
        path.write_text(json.dumps(value))
    assert verifier.verify_worker(output, ('sft',), root, 1, reuse_verified=True)['cached_cells'] == 0


def test_cli_main_uses_original_python_and_explicit_root(complete_matrix, monkeypatch):
    root, output = complete_matrix
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(verifier.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['verify_outputs.py','--output',str(output),'--all',
        '--scoring-root',str(root),'--workers','2','--reuse-verified'])
    with pytest.raises(SystemExit) as stopped: verifier.main()
    assert stopped.value.code == 0
    assert calls[0][0][0] == str(Path(sys.executable).resolve())
    assert calls[0][0][1] == '-I' and '--reuse-verified' in calls[0][0]
    assert calls[0][1]['cwd'] == root


def test_worker_cli_main_publishes_verified_receipt(complete_matrix, monkeypatch):
    root, output = complete_matrix
    monkeypatch.setattr(sys, 'argv', ['verify_outputs.py','--output',str(output),'--tag','sft',
        '--scoring-root',str(root),'--workers','1','--_worker'])
    verifier.main()
    assert json.loads((output/'verification'/'sft.json').read_text())['status'] == 'complete'
