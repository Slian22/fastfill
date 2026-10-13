#!/bin/bash
#SBATCH --job-name=unknownsize-dpo-infer-shard
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-dpo-infer-shard-%j.out
# Amendment 008/009 stage inference shard: CODE_ROOT dpo1|dpo2 ARM SHARD_INDEX SHARD_COUNT INFER_ROOT [--resume]
# As isambard_infer_shard.sbatch (amendment 002 plan, decoding and receipts), but the policy is the DPO
# stage's exact trained policy (amendment 009, infer_shard shard --model STAGE-merged --exact-policy): the
# stage adapter dpo-001/ARM/STAGE/final unmerged (FP32, as trained) on its BF16 training base, the merged
# previous stage; both are rehashed against the lineage receipt dpo-001/ARM/STAGE-merged.json.
# The amendment 005 dev-probe gate checks LoRA merging against the unmerged SFT probe; nothing is merged here.
# STAGE is tied to its result root: dpo1 -> infer-v2-003, dpo2 -> infer-v2-004.
# A timed-out shard is resubmitted with the same arguments plus --resume.
set -euo pipefail
test "$#" -eq 6 || { test "$#" -eq 7 && test "$7" = --resume; }
PAIR_CODE_ROOT="$1"
STAGE="$2"
PAIR_ARM="$3"
PAIR_SHARD_INDEX="$4"
PAIR_SHARD_COUNT="$5"
PAIR_INFER_ROOT="$6"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
case "$STAGE" in
  dpo1) test "$PAIR_INFER_ROOT" = "$PAIR_RUN_ROOT/infer-v2-003" || exit 2 ;;
  dpo2) test "$PAIR_INFER_ROOT" = "$PAIR_RUN_ROOT/infer-v2-004" || exit 2 ;;
  *) exit 2 ;;
esac
case "$PAIR_ARM" in joint|placement) ;; *) exit 2 ;; esac
case "$PAIR_SHARD_INDEX" in ''|*[!0-9]*) exit 2 ;; esac
case "$PAIR_SHARD_COUNT" in ''|*[!0-9]*) exit 2 ;; esac
test "$PAIR_SHARD_INDEX" -lt "$PAIR_SHARD_COUNT"
: "${SLURM_JOB_ID:?Slurm allocation required}"
test -f "$PAIR_INFER_ROOT/plan/MANIFEST.json"
MODEL="$PAIR_RUN_ROOT/dpo-001/$PAIR_ARM/$STAGE-merged"
test -f "$MODEL.json"
module load cudatoolkit/24.11_12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH="$CUDA_HOME" PYTHONPATH="$PAIR_CODE_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/tmp PYTHONUNBUFFERED=1
cd "$PAIR_CODE_ROOT"
"$PROJECT_ROOT/env/bin/python" -B - <<'PY'
from pathlib import Path
from fastfill.comparison import train_pair_helpers as h
p = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013/context-capacity-001')
h.verify_artifacts(p, h.strict_json((p / 'SHA256.json').read_text()))
r = h.strict_json((p / 'result.json').read_text())
assert r['status'] == 'CAPACITY_COMPLETE' and r['batch_size'] == 8 and r['cached_tokens'] == 40960
assert r['peak_allocated_bytes'] < .9 * r['gpu_total_bytes']
PY
exec timeout --signal=TERM --kill-after=60s 830m "$PROJECT_ROOT/env/bin/python" -B -u \
  -m fastfill.comparison.infer_shard shard --data "$PAIR_RUN_ROOT/eval-data-v1" \
  --base "$PROJECT_ROOT/models/Qwen3-8B" --arm "$PAIR_ARM" --model "$MODEL" --exact-policy --plan "$PAIR_INFER_ROOT/plan" \
  --shard-index "$PAIR_SHARD_INDEX" --shard-count "$PAIR_SHARD_COUNT" \
  --out "$PAIR_INFER_ROOT/$PAIR_ARM/shard-$(printf %02d "$PAIR_SHARD_INDEX")" ${7:+--resume}
