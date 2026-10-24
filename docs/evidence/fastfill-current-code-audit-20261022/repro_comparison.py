from pathlib import Path
import contextlib, io, json, math, tempfile
from fastfill.comparison.tests import test_llm_open as T
from fastfill.comparison import llm_agent as L, llm_open as O, evaluate_llm as E, decide_012 as D, best_of_n as B, train_pair_helpers as H

def o2nt(reasoning=True):
    with tempfile.TemporaryDirectory(prefix='fastfill-o2nt-review-') as directory:
        root, rooms = T.world.__wrapped__(Path(directory))
        answer = '<think>reasoned</think>{valid}' if reasoning else '{valid}'
        O.main(T.argv(root, 'O2nt', 'run', '--arm', 'placement'), T.loader(T.FakeLLM(rooms, default=answer, arm='placement')))
        result = json.loads((root / 'run/result.json').read_text())
        scored = E.evaluate(root / 'data', root / 'cohort.json', root / 'run', root / 'eval', repeats=2)
        return {'run_valid_final': result['valid_final'], 'rooms': result['rooms'], 'scored_complete': scored['arms']['llm_harness']['all']['complete_count'], 'scored_geometry_success_rate': scored['arms']['llm_harness']['all']['geometry_success_rate']}

def nonfinite():
    with tempfile.TemporaryDirectory(prefix='fastfill-nonfinite-review-') as directory:
        root, rooms = T.world.__wrapped__(Path(directory))
        room = next(iter(rooms.values()))
        object_id = json.loads(room['_placement']['messages'][1]['content'])['objects'][0]['id']
        answer = '{"placements":[{"id":' + json.dumps(object_id) + ',"pos":[1e309,1,0],"yaw":0,"size":[1,1,1]}]}'
        joint = L.parse(answer, room['_request'], 'joint')
        result = {'joint_valid': joint['valid']}
        try:
            L.parse(answer, room['_placement'], 'placement')
        except Exception as exc:
            result['placement_exception'] = type(exc).__name__ + ': ' + str(exc)
        try:
            O.main(T.argv(root, 'O2nt', 'run', '--arm', 'placement'), T.loader(T.FakeLLM(rooms, default=answer, arm='placement')))
        except Exception as exc:
            result['run_exception'] = type(exc).__name__ + ': ' + str(exc)
        result['written_receipts'] = len(list((root / 'run/receipts').glob('*.json')))
        return result

def size_guard():
    def cand(k, bias, violations):
        return {'k': k, 'complete': True, 'violations': violations, 'gate_fail': [], 'bias_prior': bias, **dict.fromkeys(B.GUARDED, 0)}
    candidates = [cand(0, 0, 1), cand(1, 2 * math.log(1.5), 0)]
    with tempfile.TemporaryDirectory(prefix='fastfill-s2-review-') as directory:
        root = Path(directory); data = root / 'data'; data.mkdir()
        base_rooms = []
        for k in range(20):
            base_rooms.append({'uid': f'u{k:02}', 'source': 'X', 'group': f'g{k}', 'boundary_type': 'polygon', 'boundary': [[0,0],[10,0],[10,10],[0,10]], 'objects': [{'id': 'obj_0001', 'category': 'chair', 'pos': [5,5,0], 'size': [1,1,1], 'yaw': 0}], 'fixed': []})
        (data / 'rooms.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in base_rooms))
        H.write_json(data / 'MANIFEST.json', {'files': H.artifact_manifest(data), 'expected_uids': [r['uid'] for r in base_rooms]})
        for arm in D.ARMS:
            for variant in D.VARIANTS:
                out = root / 's1' / f'{D.ARMS[arm]}-{variant}'; out.mkdir(parents=True)
                values=[]
                for k, room in enumerate(base_rooms):
                    size=1.5 if variant == 'bo8' and arm == 'joint' else 1
                    values.append({'uid': room['uid'], 'source': room['source'], 'group': room['group'], 'complete': True, 'raw_layout': [{**room['objects'][0], 'size': [size,size,1]}], 's1': {'strict_ok_raw': False, 'strict_ok_s1': variant == 'bo8' or k < 10, 'geometry_success_raw': True, 'geometry_success_s1': True, 'displacement_m': [.1]}})
                (out / f'{arm}-{variant}-geometry-s1.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
                H.write_json(out / 'MANIFEST.json', {'files': H.artifact_manifest(out)})
        result = D.decide(data, root / 's1', root / 'decision')
        return {'request_only_guard_pick': B.select(candidates, 8), 'actual_bias_delta': 2 * math.log(1.5), 'required_max_abs_delta': D.BIAS_TOL, 's2_joint': result['decisions']['s2_joint']}

if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        result = {'o2nt_with_reasoning': o2nt(), 'o2nt_no_reasoning_control': o2nt(False),
                  's2_bias': size_guard(), 'nonfinite_placement': nonfinite()}
    assert result['o2nt_with_reasoning']['run_valid_final'] == 0
    assert result['o2nt_with_reasoning']['scored_complete'] == 36
    assert result['o2nt_no_reasoning_control']['run_valid_final'] == 36
    assert result['s2_bias']['s2_joint']['passed']
    assert result['nonfinite_placement']['written_receipts'] == 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
