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
CODE='/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/snapshots/repairs-20261024-f18da4ce983c/OptiScene'
RUN='/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/repair-validation-20261024-f18da4ce983c'
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
export PYTHONPATH="$(dirname "$CODE"):$CODE"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
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
  fastfill/v1_1/tests/test_automation_lineage.py \
  fastfill/v2/tests/test_llm_baseline.py \
  fastfill/v2/tests/test_llm_structured.py \
  fastfill/v2/tests/test_roomgenbench.py > "$RUN/pytest.log" 2>&1
"$PYTHON" -B docs/evidence/fastfill-repairs-20261024/recover_cached_critic.py --historical '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013' --output "$RUN/critic-answers" > "$RUN/critic-recovery.json"
"$PYTHON" -B "$RUN/remote-verify.py" > "$RUN/training-preservation-full.json"
