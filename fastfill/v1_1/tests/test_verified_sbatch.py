"""Run the actual shell wrappers against fake Python/evaluation executables."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1]
VERIFIED = SCRIPTS / 'isambard_v1_1_verified_evaluate.sbatch'
AUDIT = SCRIPTS / 'isambard_v1_1_audit.sbatch'


@pytest.fixture
def deployment(tmp_path):
    original = tmp_path / 'original project'
    run = original / 'runs' / 'fastfill_v1_1_clean_fixture'
    output = run / 'evaluation full'
    helper = original / 'automation' / 'new helper $(literal)'
    frozen = original / 'automation' / 'old frozen; literal'
    for path in (output, helper / 'fastfill/v1_1', frozen / 'fastfill/v1_1'):
        path.mkdir(parents=True)
    (helper / 'fastfill/v1_1/verify_outputs.py').write_text('# mocked absolute verifier\n')
    python_code = f'''#!{sys.executable}
import json,os,pathlib,subprocess,sys
args=sys.argv[1:]
event={{'argv':args,'cwd':os.getcwd(),'pythonpath':os.environ.get('PYTHONPATH'),'executable':sys.argv[0]}}
with open(os.environ['MOCK_EVENTS'],'a') as out: out.write(json.dumps(event)+'\\n')
if args and args[0]=='-c':
    raise SystemExit(subprocess.run([sys.executable,*args]).returncode)
raise SystemExit(int(os.environ.get('MOCK_GATE_STATUS','0')))
'''
    train_py, eval_py = original / 'env/bin/python', original / 'vllm/bin/python'
    for executable in (train_py, eval_py):
        executable.parent.mkdir(parents=True)
        executable.write_text(python_code)
        executable.chmod(0o755)
    runner = {'root': str(frozen), 'run': str(run), 'output': str(output), 'python': str(eval_py)}
    (output / 'runner.json').write_text(json.dumps(runner))
    addendum = helper / 'interpretation addendum.json'
    addendum.write_text('{}')
    old_script = frozen / 'fastfill/v1_1/isambard_v1_1_evaluate.sbatch'
    old_script.write_text('''#!/bin/bash
set -euo pipefail
"$MOCK_REAL_PYTHON" -c 'import json,os,sys; open(os.environ["MOCK_EVAL"],"w").write(json.dumps({"argv":sys.argv[1:],"cwd":os.getcwd(),"pythonpath":os.environ.get("PYTHONPATH")}))' "$@"
exit "${MOCK_EVAL_STATUS:-0}"
''')
    events, evaluated = tmp_path / 'events.jsonl', tmp_path / 'evaluated.json'
    env = {**os.environ, 'MOCK_EVENTS': str(events), 'MOCK_EVAL': str(evaluated),
           'MOCK_REAL_PYTHON': sys.executable, 'PYTHONPATH': str(frozen)}
    return {'original': original, 'run': run, 'output': output, 'helper': helper,
            'frozen': frozen, 'addendum': addendum, 'train_py': train_py, 'eval_py': eval_py,
            'events': events, 'evaluated': evaluated, 'env': env}


def invoke(script, args, deployment, **environment):
    return subprocess.run(['bash', str(script), *map(str, args)],
                          env={**deployment['env'], **environment}, cwd=deployment['frozen'],
                          capture_output=True, text=True)


def events(deployment):
    return [json.loads(line) for line in deployment['events'].read_text().splitlines()]


def eval_args(d, tag='dpo1'):
    return [d['helper'], d['frozen'], d['output'], d['run'], tag]


def test_gpu_gate_runs_new_lineage_before_old_frozen_evaluator(deployment):
    d = deployment
    result = invoke(VERIFIED, eval_args(d), d)
    assert result.returncode == 0, result.stderr
    gate = next(row for row in events(d) if row['argv'][:2] == ['-m', 'fastfill.v1_1.training_guard'])
    assert gate['argv'] == ['-m', 'fastfill.v1_1.training_guard', 'verify-lineage', '--out', str(d['run']), '--tag', 'dpo1']
    assert gate['executable'] == str(d['train_py'])
    assert gate['cwd'] == gate['pythonpath'] == str(d['helper'])
    evaluation = json.loads(d['evaluated'].read_text())
    assert evaluation['argv'] == [str(d['frozen']), str(d['output']), 'dpo1']


def test_failed_lineage_never_starts_evaluation(deployment):
    result = invoke(VERIFIED, eval_args(deployment), deployment, MOCK_GATE_STATUS='23')
    assert result.returncode == 23
    assert not deployment['evaluated'].exists()


def test_evaluator_failure_propagates(deployment):
    result = invoke(VERIFIED, eval_args(deployment, 'final'), deployment, MOCK_EVAL_STATUS='19')
    assert result.returncode == 19


def test_mismatched_frozen_runner_stops_before_lineage_and_evaluation(deployment):
    d = deployment
    path = d['output'] / 'runner.json'
    runner = json.loads(path.read_text())
    path.write_text(json.dumps({**runner, 'root': str(d['helper'])}))
    result = invoke(VERIFIED, eval_args(d), d)
    assert result.returncode != 0
    assert not any(row['argv'][:2] == ['-m', 'fastfill.v1_1.training_guard'] for row in events(d))
    assert not d['evaluated'].exists()


@pytest.mark.parametrize('script,arguments', [(VERIFIED, []), (AUDIT, [])])
def test_missing_arguments_exit_two(script, arguments, deployment):
    result = invoke(script, arguments, deployment)
    assert result.returncode == 2
    assert not deployment['events'].exists()


def test_invalid_gpu_tag_fails_before_any_python(deployment):
    result = invoke(VERIFIED, eval_args(deployment, '../../invalid'), deployment)
    assert result.returncode == 2
    assert not deployment['events'].exists()


def test_sft_cpu_audit_uses_absolute_new_verifier_and_old_scoring_root(deployment):
    d = deployment
    result = invoke(AUDIT, [d['helper'], d['output'], d['addendum'], 'sft'], d)
    assert result.returncode == 0, result.stderr
    call = events(d)[-1]
    assert call['executable'] == str(d['eval_py'])
    assert call['cwd'] == call['pythonpath'] == str(d['helper'])
    assert call['argv'] == [str(d['helper'] / 'fastfill/v1_1/verify_outputs.py'),
                            '--output', str(d['output']), '--tag', 'sft',
                            '--scoring-root', str(d['frozen']), '--workers', '32', '--reuse-verified']


def test_cpu_report_uses_new_module_and_propagates_failure(deployment):
    d = deployment
    result = invoke(AUDIT, [d['helper'], d['output'], d['addendum'], 'report'], d, MOCK_GATE_STATUS='2')
    assert result.returncode == 2
    call = events(d)[-1]
    assert call['argv'] == ['-m', 'fastfill.v1_1.audit_report', '--output', str(d['output']),
                            '--interpretation', str(d['addendum']), '--workers', '32']
    assert call['executable'] == str(d['eval_py'])
    assert call['cwd'] == call['pythonpath'] == str(d['helper'])


def test_invalid_cpu_mode_fails_before_any_python(deployment):
    d = deployment
    result = invoke(AUDIT, [d['helper'], d['output'], d['addendum'], 'anything'], d)
    assert result.returncode == 2
    assert not d['events'].exists()


def test_cpu_runner_output_mismatch_fails_closed(deployment):
    d = deployment
    path = d['output'] / 'runner.json'
    runner = json.loads(path.read_text())
    path.write_text(json.dumps({**runner, 'output': str(d['helper'])}))
    result = invoke(AUDIT, [d['helper'], d['output'], d['addendum'], 'sft'], d)
    assert result.returncode != 0
    assert all(row['argv'][0] == '-c' for row in events(d))


def test_slurm_scripts_preserve_resources_disable_requeue_and_parse():
    gpu, cpu = VERIFIED.read_text(), AUDIT.read_text()
    for script, contents in ((VERIFIED, gpu), (AUDIT, cpu)):
        assert '#SBATCH --account=brics.u6uv' in contents
        assert '#SBATCH --partition=workq' in contents
        assert '#SBATCH --time=06:00:00' in contents
        assert '#SBATCH --no-requeue' in contents
        assert '#SBATCH --cpus-per-task=32' in contents
        assert subprocess.run(['bash', '-n', str(script)], capture_output=True).returncode == 0
    assert '#SBATCH --gpus=1' in gpu and '#SBATCH --mem-per-gpu=115000M' in gpu
    assert '#SBATCH --gpus=0' in cpu and '#SBATCH --mem=32G' in cpu
