from pathlib import Path
import tempfile, json, sys
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from fastfill.v2 import llm_baseline
from fastfill.v2.evaluate import project_minimal
from fastfill.v2.tests.test_evaluate_fix20261006 import real_rows
from fastfill.v2.tests.test_llm_baseline import _answer_from_targets
row=next(p for p in map(project_minimal,real_rows()) if p is not None)
answer=json.loads(_answer_from_targets(row).removeprefix('```json\n').removesuffix('\n```'))
answer['objects'][0]['x']=-100.0 # Schema-valid initial answer has a repairable out-of-room placement.
first=json.dumps(answer)
assert llm_baseline.to_layout(first,row['condition'])
assert llm_baseline.problems(llm_baseline.to_layout(first,row['condition']),row['condition'])
with tempfile.TemporaryDirectory(prefix='v2-harness-repair-') as root:
    root=Path(root); data=root/'rows.jsonl'; data.write_text(json.dumps(row)+'\n')
    env=root/'dummy.env'; env.write_text('OPENAI_MODEL=unused\n')
    calls=[]
    def ask(*args):
        calls.append(len(args[-1]))
        if len(calls)==1:
            return first
        raise ConnectionError('synthetic repair transport failure')
    summary=llm_baseline.run(data,root/'out',env,mode='harness',workers=1,ask=ask)
    prediction=json.loads((root/'out'/'predictions.jsonl').read_text())
    assert prediction['first_answer_valid'] is True and prediction['layout'] is None
    print(json.dumps({'result':'BUG_REPRODUCED','calls':len(calls),'message_counts':calls,
                      'first_answer_valid':prediction['first_answer_valid'],'layout':prediction['layout'],
                      'error':prediction['error'],'failed':summary['failed'],'attempts':prediction['attempts']}))
