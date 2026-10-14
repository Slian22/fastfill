#!/bin/bash
#SBATCH --job-name=unknownsize-010-infer-shard
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
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-010-infer-shard-%j.out
# Amendment 010 inference shard: CODE_ROOT dev|test ARM CANDIDATE SHARD_INDEX SHARD_COUNT [--resume]
#   dev   dev-select-v1 on the shared dev plan dev-eval-010/plan -> dev-eval-010/CANDIDATE/ARM/shard-NN
#   test  eval-data-v1 on infer-v2-005/plan (byte copy of infer-v2-002/plan) -> infer-v2-005/ARM/shard-NN,
#         a selected DPO candidate only (a selected SFT reuses infer-ARM-002); submitted by the selector job.
# CANDIDATE sft: the SFT adapter merged at runtime exactly as infer-v2-002 (after that arm's passed amendment
# 005 dev probe gate); e1-dpoK / rpo-dpoK: the dpo-002 / dpo-003 stage's exact trained policy (amendment 009,
# infer_shard --model STAGE-merged --exact-policy); a009-dpoK (dev only, report-only rows of the amendment 010
# addendum): the amendment-009 dpo-001 stage's exact policy. Decoding, receipts and resume as isambard_dpo_infer_shard.
# rpo-dpo1-merged (dev, placement only; amendment 011 section 5 merged-weight check): the delivered BF16 merge
# dpo-003/placement/dpo1-merged loaded as is (no --exact-policy: lora premerged_bf16_stage_model), same dev plan.
# A timed-out shard is resubmitted with the same arguments plus --resume.
set -euo pipefail
test "$#" -eq 6 || { test "$#" -eq 7 && test "$7" = --resume; }
PAIR_CODE_ROOT="$1"
SET="$2"
PAIR_ARM="$3"
CANDIDATE="$4"
PAIR_SHARD_INDEX="$5"
PAIR_SHARD_COUNT="$6"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
case "$PAIR_ARM" in joint|placement) ;; *) exit 2 ;; esac
case "$PAIR_SHARD_INDEX" in ''|*[!0-9]*) exit 2 ;; esac
case "$PAIR_SHARD_COUNT" in ''|*[!0-9]*) exit 2 ;; esac
test "$PAIR_SHARD_INDEX" -lt "$PAIR_SHARD_COUNT"
case "$SET" in
  dev) DATA="$PAIR_RUN_ROOT/dev-select-v1"; INFER_ROOT="$PAIR_RUN_ROOT/dev-eval-010/$CANDIDATE"
       PLAN="$PAIR_RUN_ROOT/dev-eval-010/plan" ;;
  test) DATA="$PAIR_RUN_ROOT/eval-data-v1"; INFER_ROOT="$PAIR_RUN_ROOT/infer-v2-005"; PLAN="$INFER_ROOT/plan" ;;
  *) exit 2 ;;
esac
case "$CANDIDATE" in
  sft) test "$SET" = dev || exit 2
       MODEL_ARGS=(--adapter "$PAIR_RUN_ROOT/sft-$PAIR_ARM-001/$PAIR_ARM/adapter"
                   --training-run "$PAIR_RUN_ROOT/sft-$PAIR_ARM-001") ;;
  e1-dpo1|e1-dpo2) MODEL_ARGS=(--model "$PAIR_RUN_ROOT/dpo-002/$PAIR_ARM/${CANDIDATE#e1-}-merged" --exact-policy) ;;
  rpo-dpo1|rpo-dpo2) MODEL_ARGS=(--model "$PAIR_RUN_ROOT/dpo-003/$PAIR_ARM/${CANDIDATE#rpo-}-merged" --exact-policy) ;;
  rpo-dpo1-merged) test "$SET" = dev -a "$PAIR_ARM" = placement || exit 2
       MODEL_ARGS=(--model "$PAIR_RUN_ROOT/dpo-003/placement/dpo1-merged") ;;
  a009-dpo1|a009-dpo2) test "$SET" = dev || exit 2
       MODEL_ARGS=(--model "$PAIR_RUN_ROOT/dpo-001/$PAIR_ARM/${CANDIDATE#a009-}-merged" --exact-policy) ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
test -f "$PLAN/MANIFEST.json"
if test "$CANDIDATE" = sft; then  # amendment 005 section 6, as isambard_infer_shard.sbatch
  grep -q '"status": "PROBE_PASS"' "$PAIR_RUN_ROOT/infer-v2-002/$PAIR_ARM/probe/probe.json"
else
  test -f "${MODEL_ARGS[1]}.json"
fi
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
  -m fastfill.comparison.infer_shard shard --data "$DATA" \
  --base "$PROJECT_ROOT/models/Qwen3-8B" --arm "$PAIR_ARM" "${MODEL_ARGS[@]}" --plan "$PLAN" \
  --shard-index "$PAIR_SHARD_INDEX" --shard-count "$PAIR_SHARD_COUNT" \
  --out "$INFER_ROOT/$PAIR_ARM/shard-$(printf %02d "$PAIR_SHARD_INDEX")" ${7:+--resume}
