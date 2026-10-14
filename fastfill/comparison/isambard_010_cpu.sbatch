#!/bin/bash
#SBATCH --job-name=unknownsize-010-cpu
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-010-cpu-%j.out
# Amendment 010 CPU steps (no GPU): CODE_ROOT STEP [ARGS]
#   build-dev                  main-data-v1 + both SFT runs' dev length admission -> dev-select-v1 (600 rooms, frozen)
#   dev-plan                   amendment 002 shared length-sorted plan (caps) for dev-select-v1 -> dev-eval-010/plan
#   dev-merge ARM CAND N       dev-eval-010/CAND/ARM/shard-* -> dev-eval-010/CAND/infer-ARM
#   score                      frozen geometry over all candidates (+ report-only a009-*) -> dev-eval-010/scores
#   select                     selection rule -> dev-eval-010/selection.json, then submits the test chain
#   selected-geometry J_DIR P_DIR   evaluate_pair --stage-label selected on eval-data-v1 -> geometry-v5
# Every output is new-only.
set -euo pipefail
test "$#" -ge 2
PAIR_CODE_ROOT="$1"
STEP="$2"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
DEV="$PAIR_RUN_ROOT/dev-select-v1"
EVAL="$PAIR_RUN_ROOT/dev-eval-010"
case "$STEP" in
  build-dev)
    test "$#" -eq 2
    ARGS=(fastfill.comparison.dev_select build --data "$PAIR_RUN_ROOT/main-data-v1" --out "$DEV"
          --sft-runs "$PAIR_RUN_ROOT/sft-joint-001" "$PAIR_RUN_ROOT/sft-placement-001") ;;
  dev-plan)
    test "$#" -eq 2
    ARGS=(fastfill.comparison.infer_shard plan --data "$DEV" --base "$PROJECT_ROOT/models/Qwen3-8B"
          --out "$EVAL/plan" --batch-size 8 --max-input-tokens 40959 --max-new-tokens 40959 --context-budget 40960) ;;
  dev-merge)
    test "$#" -eq 5
    case "$3" in joint|placement) ;; *) exit 2 ;; esac
    case "$4" in sft|e1-dpo1|e1-dpo2|rpo-dpo1|rpo-dpo2|a009-dpo1|a009-dpo2) ;; *) exit 2 ;; esac
    case "$5" in ''|*[!0-9]*) exit 2 ;; esac
    SHARDS=()
    for ((i = 0; i < $5; i++)); do SHARDS+=("$EVAL/$4/$3/shard-$(printf %02d "$i")"); done
    ARGS=(fastfill.comparison.infer_shard merge --data "$DEV" --plan "$EVAL/plan" --arm "$3"
          --out "$EVAL/$4/infer-$3" --shards "${SHARDS[@]}") ;;
  score)
    test "$#" -eq 2
    ARGS=(fastfill.comparison.dev_select score --data "$DEV" --root "$EVAL" --out "$EVAL/scores" --workers 30) ;;
  select)
    test "$#" -eq 2
    command -v sbatch >/dev/null || echo "WARNING: no sbatch in this job; the selector prints the test chain" >&2
    ARGS=(fastfill.comparison.dev_select select --scores "$EVAL/scores" --out "$EVAL/selection.json"
          --code-root "$PAIR_CODE_ROOT" --shards 6) ;;
  selected-geometry)
    test "$#" -eq 4
    case "$3" in "$PAIR_RUN_ROOT"/infer-joint-[0-9][0-9][0-9]) ;; *) exit 2 ;; esac
    case "$4" in "$PAIR_RUN_ROOT"/infer-placement-[0-9][0-9][0-9]) ;; *) exit 2 ;; esac
    ARGS=(fastfill.comparison.evaluate_pair --data "$PAIR_RUN_ROOT/eval-data-v1" --joint "$3" --placement "$4"
          --out "$PAIR_RUN_ROOT/geometry-v5" --workers 24 --repeats 2000 --stage-label selected) ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/tmp
cd "$PAIR_CODE_ROOT"
exec timeout --signal=TERM --kill-after=30s 175m "$PROJECT_ROOT/env/bin/python" -B -u -m "${ARGS[@]}"
