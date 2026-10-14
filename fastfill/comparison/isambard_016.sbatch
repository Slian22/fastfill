#!/bin/bash
#SBATCH --job-name=unknownsize-016
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-016-%j.out
# Amendment 016 section 3 (DPO2 attribution ablation on dev; rft = RUN/dpo-005, pert = RUN/dpo-006): CODE_ROOT MODE [ARGS]
# One script for every step; the header is the CPU default (32 cores, 96G, 3 h). Other steps take their resources on
# the sbatch line (submit_016_ablation.sh), and each mode's inner timeout fits them:
#   train CAND                 --gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=12:00:00   ablation_016 train:
#                              torchrun train_dpo (DDP 4 ranks) from rpo-dpo1's exact policy (--model dpo-001/placement/
#                              sft-merged --init_adapter dpo-003/placement/dpo1/final), op-dpo2 e2's step count;
#                              rft: --sft_only on dpo-004's chosen texts; pert: DPO + RPO 1.0 on dpo-001/stage2-pairs.
#                              -> dpo-00X/placement/dpo2 and dpo2-merged (receipt anchor); resubmitting resumes.
#   shard CAND I N [--resume]  --gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=14:00:00   infer_shard shard as
#                              isambard_013.sbatch: the exact policy of dpo-00X/placement/dpo2-merged on dev-select-v1 and
#                              the shared dev-eval-010/plan -> dev-eval-016/CAND/placement/shard-NN.
#   dev-merge CAND N           CPU                     dev-eval-016/CAND/placement/shard-* -> dev-eval-016/CAND/infer-placement
#   score                      CPU                     ablation_016 score -> dev-eval-016/ablation.json + ablation-report.md
#                              (needs dev-eval-013/scores: amendment 013's dev-score; if it is not there yet the job fails
#                              before writing anything: resubmit sbatch isambard_016.sbatch CODE_ROOT score)
# CAND is rft or pert. Every output is new-only.
set -euo pipefail
test "$#" -ge 2 || exit 2
PAIR_CODE_ROOT="$1"
MODE="$2"
shift 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
DEV="$RUN_ROOT/dev-select-v1"
EVAL="$RUN_ROOT/dev-eval-016"
AB=(fastfill.comparison.ablation_016)
GPUS=0
LIMIT=175m
KIND=cpu
run_of() {  # run_of CAND -> its DPO run directory.
  case "$1" in rft) echo "$RUN_ROOT/dpo-005" ;; pert) echo "$RUN_ROOT/dpo-006" ;; *) exit 2 ;; esac
}
case "$MODE" in
  train)
    test "$#" -eq 1 || exit 2
    DPO=$(run_of "$1")
    KIND=train GPUS=4 LIMIT=705m
    ARGS=("${AB[@]}" train --dpo-root "$DPO" --candidate "$1") ;;
  shard)
    test "$#" -eq 3 || { test "$#" -eq 4 && test "$4" = --resume; } || exit 2
    CAND="$1" INDEX="$2" COUNT="$3"
    DPO=$(run_of "$CAND")
    case "$INDEX" in ''|*[!0-9]*) exit 2 ;; esac
    case "$COUNT" in ''|*[!0-9]*) exit 2 ;; esac
    test "$INDEX" -lt "$COUNT" || exit 2
    MODEL="$DPO/placement/dpo2-merged"
    KIND=shard GPUS=1 LIMIT=830m
    ARGS=(fastfill.comparison.infer_shard shard --data "$DEV" --base "$PROJECT_ROOT/models/Qwen3-8B" --arm placement
          --model "$MODEL" --exact-policy --plan "$RUN_ROOT/dev-eval-010/plan" --shard-index "$INDEX"
          --shard-count "$COUNT" --out "$EVAL/$CAND/placement/shard-$(printf %02d "$INDEX")" ${4:+--resume}) ;;
  dev-merge)
    test "$#" -eq 2 || exit 2
    run_of "$1" > /dev/null
    case "$2" in ''|*[!0-9]*) exit 2 ;; esac
    SHARDS=()
    for ((i = 0; i < $2; i++)); do SHARDS+=("$EVAL/$1/placement/shard-$(printf %02d "$i")"); done
    ARGS=(fastfill.comparison.infer_shard merge --data "$DEV" --plan "$RUN_ROOT/dev-eval-010/plan" --arm placement
          --out "$EVAL/$1/infer-placement" --shards "${SHARDS[@]}") ;;
  score)
    test "$#" -eq 0 || exit 2
    ARGS=("${AB[@]}" score --data "$DEV" --run-root "$RUN_ROOT" --out "$EVAL/ablation.json" --workers 30) ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
if test "$MODE" = shard; then
  test -f "$RUN_ROOT/dev-eval-010/plan/MANIFEST.json"
  test -f "$MODEL.json" || { echo "PREFLIGHT FAIL: no lineage receipt $MODEL.json (train $CAND first)" >&2; exit 1; }
fi
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=/tmp
cd "$PAIR_CODE_ROOT"
case "$KIND" in
  cpu)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ;;
  train|shard)
    module load cudatoolkit/24.11_12.6
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export CUDA_PATH="$CUDA_HOME" WANDB_DISABLED=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ;;
esac
if test "$KIND" = train; then  # One writer per ablation run directory, also against a manual resubmit.
  mkdir -p "$DPO/placement"
  exec 9>"$DPO/placement/.dpo2.lock"
  flock -n 9 || { echo "another job owns $DPO/placement/dpo2" >&2; exit 1; }
fi
if test "$KIND" = shard; then  # The amendment-002 KV-cache capacity evidence, as isambard_013.sbatch.
  "$PYTHON" -B - <<'PY'
from pathlib import Path
from fastfill.comparison import train_pair_helpers as h
p = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013/context-capacity-001')
h.verify_artifacts(p, h.strict_json((p / 'SHA256.json').read_text()))
r = h.strict_json((p / 'result.json').read_text())
assert r['status'] == 'CAPACITY_COMPLETE' and r['batch_size'] == 8 and r['cached_tokens'] == 40960
assert r['peak_allocated_bytes'] < .9 * r['gpu_total_bytes']
PY
fi
if test "$GPUS" -gt 0; then  # At least GPUS (4 = the node, for DDP): whether Slurm hides other GPUs is not relied on.
  nvidia-smi -L
  SEEN=$(nvidia-smi -L | grep -c '^GPU ' || true)
  test "$SEEN" -ge "$GPUS" || { echo "PREFLIGHT FAIL: $MODE needs $GPUS GPUs, sees $SEEN" >&2; exit 1; }
fi
exec timeout --signal=TERM --kill-after=60s "$LIMIT" "$PYTHON" -B -u -m "${ARGS[@]}"
