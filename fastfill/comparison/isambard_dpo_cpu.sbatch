#!/bin/bash
#SBATCH --job-name=unknownsize-dpo-cpu
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-dpo-cpu-%j.out
# Amendment 008 CPU steps (no GPU): CODE_ROOT STEP [ARM]
#   copy-plans          infer-v2-002/plan -> infer-v2-003/plan (DPO1) and infer-v2-004/plan (DPO2), byte-identical
#   positives           main-data-v1 + both SFT runs -> dpo-001/positives
#   merge-sft ARM       sft-ARM-001 adapter folded into BF16 Qwen3-8B -> dpo-001/ARM/sft-merged (+ .json lineage)
#   stage1-pairs        both arms' SFT generations -> dpo-001/stage1-pairs
#   stage2-pairs        positional hard negatives -> dpo-001/stage2-pairs
# Every output is new-only; a rerun after success is refused (merge-sft re-verifies and exits 0).
set -euo pipefail
test "$#" -eq 2 -o "$#" -eq 3
PAIR_CODE_ROOT="$1"
STEP="$2"
ARM="${3:-}"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
DPO_ROOT="$PAIR_RUN_ROOT/dpo-001"
BASE="$PROJECT_ROOT/models/Qwen3-8B"
: "${SLURM_JOB_ID:?Slurm allocation required}"
case "$STEP" in
  copy-plans)
    test -z "$ARM"
    test ! -e "$PAIR_RUN_ROOT/infer-v2-003" -a ! -e "$PAIR_RUN_ROOT/infer-v2-004"
    ARGS=(copy-plan --src "$PAIR_RUN_ROOT/infer-v2-002/plan"
          --out "$PAIR_RUN_ROOT/infer-v2-003/plan" "$PAIR_RUN_ROOT/infer-v2-004/plan") ;;
  positives)
    test -z "$ARM"
    ARGS=(positives --data "$PAIR_RUN_ROOT/main-data-v1" --out "$DPO_ROOT/positives" --workers 30
          --sft-runs "$PAIR_RUN_ROOT/sft-joint-001" "$PAIR_RUN_ROOT/sft-placement-001") ;;
  merge-sft)
    case "$ARM" in joint|placement) ;; *) exit 2 ;; esac
    ARGS=(merge-sft --arm "$ARM" --base "$BASE" --sft-run "$PAIR_RUN_ROOT/sft-$ARM-001" --dpo-root "$DPO_ROOT") ;;
  stage1-pairs|stage2-pairs)
    test -z "$ARM"
    ARGS=("$STEP" --dpo-root "$DPO_ROOT" --base "$BASE") ;;
  *) exit 2 ;;
esac
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/tmp
cd "$PAIR_CODE_ROOT"
exec timeout --signal=TERM --kill-after=30s 175m "$PROJECT_ROOT/env/bin/python" -B -u \
  -m fastfill.comparison.dpo_stages "${ARGS[@]}"
