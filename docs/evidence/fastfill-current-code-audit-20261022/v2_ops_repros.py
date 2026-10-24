import json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path('/Users/slian/Desktop/3D/Worldedge/OptiScene')
sys.path[:0] = [str(ROOT), str(ROOT / 'fastfill/v2/ops')]
from fastfill.v2.evaluate import run_evaluation
from fastfill.v2.ops.stratify import collect

work = Path(tempfile.mkdtemp(prefix='fastfill-ops-repro-'))
fixture = ROOT / 'fastfill/v2/tests/fixtures/validation_rows_fix20261006.jsonl'
rows = [json.loads(line) for line in fixture.read_text().splitlines()]
data = work / 'data.jsonl'
data.write_bytes(fixture.read_bytes())
preds = work / 'predictions.jsonl'
preds.write_text(''.join(json.dumps(r['target']) + '\n' for r in rows))
fit = work / 'fit.jsonl'
fit.write_bytes(data.read_bytes())
run_evaluation(data, work / 'evaluation', predictions=preds, baseline_fit=fit)
first = collect(work / 'evaluation', workers=1, cache=work / 'cache')
changed_fit = json.loads(json.dumps(rows))
for row in changed_fit:
    for obj in row['target']['objects']:
        obj['target_size_local_m'] = [2 * v if v is not None else None for v in obj['target_size_local_m']]
fit.write_text(''.join(json.dumps(r) + '\n' for r in changed_fit))
cached = collect(work / 'evaluation', workers=1, cache=work / 'cache')
fresh = collect(work / 'evaluation', workers=1)

shell = work / 'shell'
(shell / 'env/bin').mkdir(parents=True)
(shell / 'runs').mkdir()
py = shell / 'env/bin/python'
py.write_text('#!/bin/bash\nexit 42\n')
py.chmod(0o755)
proc = subprocess.run(['bash', '-c', 'cd() { builtin cd "$OPS_TEST_WORK"; }; source "$OPS_SRC"'],
    env={**os.environ, 'OPS_TEST_WORK': str(shell), 'OPS_SRC': str(ROOT / 'fastfill/v2/ops/run_structured.sh')},
    text=True, capture_output=True)
result = {'workspace': str(work), 'cache_omits_fit': {
    'before_fit_change_reproduces': first[2]['reproduces_report_exactly'],
    'after_fit_change_cached_reproduces': cached[2]['reproduces_report_exactly'],
    'after_fit_change_fresh_reproduces': fresh[2]['reproduces_report_exactly'],
    'fresh_baseline_mismatches': fresh[2]['request_baseline_mismatches'],
    'cached_baseline_mismatches': cached[2]['request_baseline_mismatches']},
    'structured_shell_failure': {'all_python_commands_exit': 42, 'wrapper_exit': proc.returncode,
    'stdout': proc.stdout.strip(), 'stderr': proc.stderr.strip()}}
print(json.dumps(result, indent=2))
assert first[2]['reproduces_report_exactly'] and cached[2]['reproduces_report_exactly']
assert not fresh[2]['reproduces_report_exactly']
assert proc.returncode == 0 and 'ALL-DONE' in proc.stdout
Path('/tmp/fastfill-audit-20261022/v2-ops-repros.json').write_text(json.dumps(result, indent=2) + '\n')
