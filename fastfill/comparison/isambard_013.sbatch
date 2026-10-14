#!/bin/bash
#SBATCH --job-name=unknownsize-013
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-013-%j.out
# Amendment 013 (on-policy DPO2 for arm P on rpo-dpo1, candidate op, RUN/dpo-004): CODE_ROOT MODE [ARGS]
# One script for every step; the header is the CPU default (32 cores, 96G, 3 h). Other steps take their resources
# on the sbatch line (submit_013.sh, onpolicy_dpo select), and each mode's inner timeout fits them:
#   yield                      --cpus-per-task=72 --mem=192G --time=08:00:00   by hand before submit_013.sh, once the
#                              amendment-012 S2 placement samples are finalized: onpolicy_dpo pairs --dry-run 1000 on
#                              s2-011/placement-rpo-dpo1/samples (dev-full-v1; same policy and K) -> op-yield-013
#                              (pair yield projected to 3,000 rooms; submit_013.sh samples only if it meets the floors)
#   recheck                    CPU                     onpolicy_dpo recheck: dev-eval-010's sft / rpo-dpo1 dev scores
#                              reproduce under this code (dev-score's first check, before any GPU work)
#   subset                     CPU                     onpolicy_dpo subset -> dpo-004/train-subset
#   sample COUNT [--limit M]   --gpus-per-node=1 --cpus-per-task=16 --mem=100G --time=04:00:00 [--array=0-(COUNT-1)]
#                              (smoke: COUNT 1, --limit 16, --time=02:00:00). vLLM env, tensor_parallel 1, shard =
#                              SLURM_ARRAY_TASK_ID: sample_vllm run of rpo-dpo1's exact policy (dpo-003/placement/
#                              dpo1-merged --exact-policy) on dpo-004/train-subset -> dpo-004/samples; resumes from receipts.
#   finalize                   CPU                     sample_vllm finalize -> dpo-004/samples (SAMPLING_COMPLETE)
#   pairs                      --cpus-per-task=72 --mem=192G --time=08:00:00   onpolicy_dpo pairs -> dpo-004/stage2-pairs
#   train                      --gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=12:00:00   onpolicy_dpo train:
#                              torchrun train_dpo (DDP 4 ranks) from rpo-dpo1's exact policy, not its BF16 merge:
#                              --model dpo-001/placement/sft-merged (rpo-dpo1's training base) --init_adapter
#                              dpo-003/placement/dpo1/final (policy start and frozen reference) -> dpo-004/placement/dpo2;
#                              merges dpo2-epoch01-merged and dpo2-merged (training base + adapter: receipt anchors and
#                              convenience artifacts); resubmitting resumes (training_guard, ckpt_latest).
#   shard dev|test CAND I N [--resume]   --gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=14:00:00
#                              infer_shard shard as isambard_010_infer_shard.sbatch. CAND op-dpo2-e1 / op-dpo2-e2: the
#                              exact policy of dpo2-epoch01-merged / dpo2-merged (--exact-policy: the DPO2 adapter on
#                              dpo-001/placement/sft-merged, from the receipt's training_base); dev -> dev-eval-013/
#                              CAND/placement/shard-NN on the shared dev-eval-010/plan; test (the adopted CAND only) ->
#                              infer-v2-006/placement/shard-NN on infer-v2-006/plan. op-dpo2-eK-merged (dev only, the
#                              adopted K): the BF16 merge loaded as is, for the merged-weight check.
#   dev-merge CAND N           CPU                     dev-eval-013/CAND/placement/shard-* -> dev-eval-013/CAND/infer-placement
#   dev-score                  CPU                     -> dev-eval-013/scores (sft, rpo-dpo1 from dev-eval-010; op-dpo2-e1/e2)
#   select                     CPU                     -> dev-eval-013/selection.json only (--no-submit); the test chain
#                              of an adopted checkpoint is submitted by hand later (submit_013.sh CODE_ROOT test)
#   test-geometry              CPU                     evaluate_pair --stage-label selected: J-SFT infer-joint-002 vs
#                              P-op infer-placement-006 on eval-data-v1 -> geometry-v6 (amendment-005 delivery rule)
#   test-report                CPU, --time=06:00:00    P-op vs P-rpo-dpo1 (infer-placement-005) -> test-report-013
#   check-merged               CPU                     merged vs unmerged adopted op checkpoint on dev -> dev-eval-013/merged-check
# Test sets and merged runs refuse unless dev-eval-013/selection.json, bound to its dev scores, adopted that candidate.
# Every output is new-only.
set -euo pipefail
test "$#" -ge 2 || exit 2
PAIR_CODE_ROOT="$1"
MODE="$2"
shift 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
DPO="$RUN_ROOT/dpo-004"
DEV="$RUN_ROOT/dev-select-v1"
EVAL="$RUN_ROOT/dev-eval-013"
OP=(fastfill.comparison.onpolicy_dpo)
GPUS=0
LIMIT=175m
KIND=cpu
adopted() {  # adopted CAND: dev-eval-013/selection.json, bound to its dev scores (load_selection), adopted CAND.
  "$PYTHON" -B - "$EVAL/selection.json" "$1" <<'PY' || { echo "PREFLIGHT FAIL: $1 is not the adopted op checkpoint" >&2; exit 1; }
import sys
from fastfill.comparison import onpolicy_dpo as op
value, _ = op._adopted(sys.argv[1])
assert value['selected'] == sys.argv[2], value['selected']
PY
}
case "$MODE" in
  yield)
    test "$#" -eq 0 || exit 2
    LIMIT=470m
    ARGS=("${OP[@]}" pairs --data "$RUN_ROOT/dev-full-v1" --samples "$RUN_ROOT/s2-011/placement-rpo-dpo1/samples"
          --base-model "$RUN_ROOT/dpo-003/placement/dpo1-merged" --out "$RUN_ROOT/op-yield-013" --workers 72
          --dry-run 1000) ;;
  recheck)
    test "$#" -eq 0 || exit 2
    ARGS=("${OP[@]}" recheck --data "$DEV" --ref-root "$RUN_ROOT/dev-eval-010" --workers 30) ;;
  subset)
    test "$#" -eq 0 || exit 2
    ARGS=("${OP[@]}" subset --data "$RUN_ROOT/main-data-v1" --exclude "$RUN_ROOT/dpo-001/positives"
          --sft-runs "$RUN_ROOT/sft-joint-001" "$RUN_ROOT/sft-placement-001" --dev "$DEV" "$RUN_ROOT/dev-full-v1"
          --test "$RUN_ROOT/eval-data-v1" --out "$DPO/train-subset") ;;
  sample)
    case "$#" in
      1) ;;
      3) test "$2" = --limit || exit 2; case "$3" in ''|*[!0-9]*) exit 2 ;; esac ;;
      *) exit 2 ;;
    esac
    COUNT="$1"
    case "$COUNT" in ''|*[!0-9]*) exit 2 ;; esac
    test "$COUNT" -ge 1 || exit 2
    INDEX="${SLURM_ARRAY_TASK_ID:-}"
    if test -z "$INDEX"; then
      test "$COUNT" -eq 1 || exit 2  # Several shards need an array job.
      INDEX=0
    fi
    case "$INDEX" in *[!0-9]*) exit 2 ;; esac
    test "$INDEX" -lt "$COUNT" || exit 2
    LIMIT_ARGS=()
    if test "$#" -eq 3; then LIMIT_ARGS=(--limit "$3"); fi
    KIND=vllm GPUS=1 LIMIT=230m
    ARGS=(fastfill.comparison.sample_vllm run --data "$DPO/train-subset" --base "$PROJECT_ROOT/models/Qwen3-8B"
          --arm placement --model "$RUN_ROOT/dpo-003/placement/dpo1-merged" --exact-policy --out "$DPO/samples"
          --shard-index "$INDEX" --shard-count "$COUNT" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}) ;;
  finalize)
    test "$#" -eq 0 || exit 2
    ARGS=(fastfill.comparison.sample_vllm finalize --data "$DPO/train-subset" --out "$DPO/samples") ;;
  pairs)
    test "$#" -eq 0 || exit 2
    LIMIT=470m
    ARGS=("${OP[@]}" pairs --data "$DPO/train-subset" --samples "$DPO/samples"
          --base-model "$RUN_ROOT/dpo-003/placement/dpo1-merged" --out "$DPO/stage2-pairs" --workers 72) ;;
  train)
    test "$#" -eq 0 || exit 2
    KIND=train GPUS=4 LIMIT=705m
    ARGS=("${OP[@]}" train --dpo-root "$DPO") ;;
  shard)
    test "$#" -eq 4 || { test "$#" -eq 5 && test "$5" = --resume; } || exit 2
    SET="$1" CAND="$2" INDEX="$3" COUNT="$4"
    case "$INDEX" in ''|*[!0-9]*) exit 2 ;; esac
    case "$COUNT" in ''|*[!0-9]*) exit 2 ;; esac
    test "$INDEX" -lt "$COUNT" || exit 2
    case "$CAND" in
      op-dpo2-e1) MODEL_ARGS=(--model "$DPO/placement/dpo2-epoch01-merged" --exact-policy) ;;
      op-dpo2-e2) MODEL_ARGS=(--model "$DPO/placement/dpo2-merged" --exact-policy) ;;
      op-dpo2-e1-merged) test "$SET" = dev || exit 2; MODEL_ARGS=(--model "$DPO/placement/dpo2-epoch01-merged") ;;
      op-dpo2-e2-merged) test "$SET" = dev || exit 2; MODEL_ARGS=(--model "$DPO/placement/dpo2-merged") ;;
      *) exit 2 ;;
    esac
    case "$SET" in
      dev) DATA="$DEV"; INFER_ROOT="$EVAL/$CAND"; PLAN="$RUN_ROOT/dev-eval-010/plan" ;;
      test) DATA="$RUN_ROOT/eval-data-v1"; INFER_ROOT="$RUN_ROOT/infer-v2-006"; PLAN="$INFER_ROOT/plan" ;;
      *) exit 2 ;;
    esac
    KIND=shard GPUS=1 LIMIT=830m
    ARGS=(fastfill.comparison.infer_shard shard --data "$DATA" --base "$PROJECT_ROOT/models/Qwen3-8B" --arm placement
          "${MODEL_ARGS[@]}" --plan "$PLAN" --shard-index "$INDEX" --shard-count "$COUNT"
          --out "$INFER_ROOT/placement/shard-$(printf %02d "$INDEX")" ${5:+--resume}) ;;
  dev-merge)
    test "$#" -eq 2 || exit 2
    case "$1" in op-dpo2-e1|op-dpo2-e2|op-dpo2-e1-merged|op-dpo2-e2-merged) ;; *) exit 2 ;; esac
    case "$2" in ''|*[!0-9]*) exit 2 ;; esac
    SHARDS=()
    for ((i = 0; i < $2; i++)); do SHARDS+=("$EVAL/$1/placement/shard-$(printf %02d "$i")"); done
    ARGS=(fastfill.comparison.infer_shard merge --data "$DEV" --plan "$RUN_ROOT/dev-eval-010/plan" --arm placement
          --out "$EVAL/$1/infer-placement" --shards "${SHARDS[@]}") ;;
  dev-score)
    test "$#" -eq 0 || exit 2
    ARGS=("${OP[@]}" dev-score --data "$DEV" --root "$EVAL" --ref-root "$RUN_ROOT/dev-eval-010" --out "$EVAL/scores"
          --workers 30) ;;
  select)
    test "$#" -eq 0 || exit 2
    ARGS=("${OP[@]}" select --scores "$EVAL/scores" --out "$EVAL/selection.json" --code-root "$PAIR_CODE_ROOT" --shards 6
          --no-submit) ;;
  test-geometry)
    test "$#" -eq 0 || exit 2
    ARGS=(fastfill.comparison.evaluate_pair --data "$RUN_ROOT/eval-data-v1" --joint "$RUN_ROOT/infer-joint-002"
          --placement "$RUN_ROOT/infer-placement-006" --out "$RUN_ROOT/geometry-v6" --workers 24 --repeats 2000
          --stage-label selected) ;;
  test-report)
    test "$#" -eq 0 || exit 2
    LIMIT=350m
    ARGS=("${OP[@]}" test-report --data "$RUN_ROOT/eval-data-v1" --selection "$EVAL/selection.json"
          --op "$RUN_ROOT/infer-placement-006" --ref "$RUN_ROOT/infer-placement-005" --out "$RUN_ROOT/test-report-013"
          --workers 30) ;;
  check-merged)
    test "$#" -eq 0 || exit 2
    ARGS=("${OP[@]}" check-merged --data "$DEV" --root "$EVAL" --ref-root "$RUN_ROOT/dev-eval-010"
          --out "$EVAL/merged-check" --workers 30) ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
if test "$MODE" = shard; then
  test -f "$PLAN/MANIFEST.json"
  test -f "${MODEL_ARGS[1]}.json"
fi
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=/tmp
cd "$PAIR_CODE_ROOT"
if test "$MODE" = shard && { test "$SET" = test || test "${CAND%-merged}" != "$CAND"; }; then adopted "${CAND%-merged}"; fi
case "$KIND" in
  cpu)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ;;
  vllm)  # As isambard_sample_vllm.sbatch; compile caches under PROJECTDIR (scratch file limit).
    PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
    JOB_CACHE="$PROJECT_ROOT/.cache-jobs/op"
    test -f "$DPO/train-subset/MANIFEST.json" || { echo "PREFLIGHT FAIL: $DPO/train-subset missing" >&2; exit 1; }
    PIP_CUDA="$(dirname "$(dirname "$PYTHON")")/lib/python3.11/site-packages/nvidia/cu13"
    if test -x "$PIP_CUDA/bin/nvcc"; then
      CUDA_HOME="$PIP_CUDA"
    else
      module load cudatoolkit || { echo "PREFLIGHT FAIL: module load cudatoolkit" >&2; exit 1; }
      NVCC=$(command -v nvcc) || { echo "PREFLIGHT FAIL: no nvcc after 'module load cudatoolkit'" >&2; exit 1; }
      CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    fi
    "$CUDA_HOME/bin/nvcc" --version | tail -1
    export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
    export HF_HOME=/scratch/u6uv/shanliantian.u6uv/.cache/huggingface XDG_CACHE_HOME="$JOB_CACHE/xdg"
    export VLLM_CACHE_ROOT="$JOB_CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor"
    export TRITON_CACHE_DIR="$JOB_CACHE/triton" VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
    export VLLM_ENGINE_READY_TIMEOUT_S=3600
    mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
    "$PYTHON" -B -c 'import vllm, shapely, fastfill.comparison.sample_vllm' || {
      echo "PREFLIGHT FAIL: $PYTHON cannot import vllm + shapely + fastfill.comparison.sample_vllm" >&2; exit 1; }
    "$PYTHON" -B - "$RUN_ROOT/dev-eval-010/selection.json" <<'PY' || {
import json, sys
assert json.load(open(sys.argv[1]))['selected']['placement'] == 'rpo-dpo1'
PY
      echo "PREFLIGHT FAIL: dev-eval-010/selection.json no longer selects rpo-dpo1 for placement" >&2; exit 1; } ;;
  train|shard)
    module load cudatoolkit/24.11_12.6
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export CUDA_PATH="$CUDA_HOME" WANDB_DISABLED=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ;;
esac
if test "$KIND" = train; then  # One writer per stage directory, also against a manual resubmit.
  mkdir -p "$DPO/placement"
  exec 9>"$DPO/placement/.dpo2.lock"
  flock -n 9 || { echo "another job owns $DPO/placement/dpo2" >&2; exit 1; }
fi
if test "$KIND" = shard; then  # The amendment-002 KV-cache capacity evidence, as isambard_010_infer_shard.sbatch.
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
