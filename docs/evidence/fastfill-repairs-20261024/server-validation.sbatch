#!/bin/bash
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=none
#SBATCH --time=00:20:00
#SBATCH --job-name=fastfill-repair-check
#SBATCH --export=NONE
set -euo pipefail
SNAPSHOT='/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/snapshots/repairs-20261024-daca130611a9/OptiScene'
RUN='/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/repair-validation-20261024-daca130611a9'
CODE="$RUN/test-checkout/OptiScene"
QA_DEPS='/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/repair-qa-20261024-f17/packages'
PYTHON='/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/env/bin/python'
complete() {
  status=$?
  "$PYTHON" -B - "$RUN/completion.json" "$status" "${SLURM_JOB_ID:-unknown}" <<'PYCODE'
import datetime,json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'exit_code':int(sys.argv[2]),'job_id':sys.argv[3],'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+chr(10))
PYCODE
  exit "$status"
}
trap complete EXIT
export PYTHONPATH="$(dirname "$CODE"):$CODE:$QA_DEPS"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$RUN/tmp" TORCHINDUCTOR_CACHE_DIR="$RUN/torchinductor-cache" TRITON_CACHE_DIR="$RUN/triton-cache"
export MPLCONFIGDIR="$RUN/matplotlib-cache"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
"$PYTHON" -B - "$SNAPSHOT" "$CODE" "$RUN" <<'PYCODE'
import hashlib, importlib.metadata, json, shutil, sys
from pathlib import Path
source, target, run = map(Path, sys.argv[1:])
manifest = json.loads((source / 'SNAPSHOT_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
shutil.copytree(source, target)
for path in [target, *target.rglob('*')]:
    path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o200))
for name, expected in manifest['files'].items():
    assert hashlib.sha256((target / name).read_bytes()).hexdigest() == expected, name
environment = {'python': sys.version, 'executable': sys.executable,
               'packages': {name: importlib.metadata.version(name) for name in
                            ('torch', 'transformers', 'peft', 'pytest', 'numpy', 'matplotlib', 'pillow')},
               'test_copy_files_verified': len(manifest['files']),
               'test_copy': str(target), 'immutable_source': str(source)}
with (run / 'test-environment.json').open('x') as stream:
    stream.write(json.dumps(environment, indent=2) + '\n')
PYCODE
cd "$CODE"
"$PYTHON" -B -m pytest --import-mode=importlib -q -o "cache_dir=$RUN/pytest-cache" --junitxml="$RUN/pytest.xml" fastfill/comparison/tests/test_agents_016.py \
  fastfill/comparison/tests/test_critic_batch_handoff.py \
  fastfill/comparison/tests/test_decide_012.py \
  fastfill/comparison/tests/test_judge_agreement_023.py \
  fastfill/comparison/tests/test_llm_open.py \
  fastfill/comparison/tests/test_mesh_collision.py \
  fastfill/comparison/tests/test_submit_023_judge.py \
  fastfill/tests/test_http_deadlines.py \
  fastfill/tests/test_qa_constraint_parity.py \
  fastfill/tests/test_runtime_repairs.py \
  fastfill/tests/test_v1_1_audit_pairs.py \
  fastfill/v1_1/tests/test_dpo_checkpoint_recovery.py \
  fastfill/v1_1/tests/test_roomgenbench_recovery.py \
  fastfill/v1_1/tests/test_train_dpo_a021.py \
  fastfill/v1_1/tests/test_train_dpo_init_adapter.py \
  fastfill/v2/tests/test_autorun.py \
  fastfill/v2/tests/test_repairs_20261024.py \
  fastfill/comparison/tests/test_critic_025.py \
  fastfill/comparison/tests/test_critic_client_025.py \
  fastfill/comparison/tests/test_llm_agent.py \
  fastfill/v1_1/tests/test_training_guard.py \
  fastfill/v1_1/tests/test_evaluation_protocol.py \
  fastfill/v1_1/tests/test_automation_lineage.py \
  fastfill/v2/tests/test_llm_baseline.py \
  fastfill/v2/tests/test_llm_structured.py \
  fastfill/v2/tests/test_roomgenbench.py > "$RUN/pytest.log" 2>&1
PYTHONPATH="$(dirname "$SNAPSHOT"):$SNAPSHOT:$QA_DEPS" "$PYTHON" -B "$SNAPSHOT/docs/evidence/fastfill-repairs-20261024/recover_cached_critic.py" --historical '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013' --output "$RUN/critic-answers" > "$RUN/critic-recovery.json"
"$PYTHON" -B "$RUN/remote-verify.py" > "$RUN/training-preservation-full.json"
"$PYTHON" -B - "$RUN" "$SNAPSHOT" "$CODE" <<'PYCODE'
import datetime, hashlib, json, sys, xml.etree.ElementTree as ET
from pathlib import Path
run, snapshot, code = map(Path, sys.argv[1:])
preservation = json.loads((run / 'training-preservation-full.json').read_text())
recovery = json.loads((run / 'critic-recovery.json').read_text())
manifest = json.loads((code / 'SNAPSHOT_MANIFEST.json').read_text())
assert preservation['passed'], 'training preservation failed'
assert recovery['status'] == 'CACHED_ANSWERS_RECOVERED'
assert recovery['answers'] == 757 and recovery['api_calls'] == recovery['new_geometry_rounds'] == 0
for name, expected in manifest['files'].items():
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected, name
suites = list(ET.parse(run / 'pytest.xml').getroot().iter('testsuite'))
assert suites and all(int(s.attrib['failures']) == int(s.attrib['errors']) == 0 for s in suites)
result = {'status': 'REPAIR_VALIDATION_PASSED', 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'content_sha256': manifest['content_sha256'], 'source_files_verified': len(manifest['files']),
          'test_copy_files_verified_after_execution': len(manifest['files']),
          'testsuites': [s.attrib for s in suites], 'training_preservation_checks': len(preservation['checks']),
          'training_preservation_passed': True, 'critic_answers_recovered': recovery['answers'],
          'api_calls': 0, 'gpu_allocated': False, 'new_model_training': False}
with (run / 'validation.json').open('x') as stream:
    stream.write(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
PYCODE
