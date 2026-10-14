#!/bin/bash
#SBATCH --job-name=unknownsize-018
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --exclude=nid010216
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-018-%j.out
# Amendment 018 sections 2-4 (arm P distilled from O2: SFT-D -> DPO1-D -> DPO2-D, RUN/distill-018; dev selection
# RUN/dev-eval-018; test only if adopted): CODE_ROOT MODE [ARGS]. One script for every step; the header is the CPU
# default (32 cores, 96G, 3 h). Other steps take their resources on the sbatch line (submit_018_chain.sh,
# distill_018_chain select), and each mode's inner timeout fits them. Every job excludes nid010216 (its GPUs failed
# 013's training, commit 7c9d50a9b; a header line, so the select-submitted test chain excludes it too):
#   recheck                CPU                     onpolicy_dpo recheck: dev-eval-010's sft / rpo-dpo1 dev scores
#                          reproduce under this code (dev-score's first check, before any GPU work)
#   tc-data                CPU                     T-C of distill-018/plan as its own data dir -> distill-018/t-c
#   sft-data               --cpus-per-task=72 --mem=192G --time=04:00:00   -> distill-018/sft-d-rows (teacher targets)
#   train CAND             --gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=06:00:00   CAND sft-d | dpo1-d | dpo2-d:
#                          torchrun train_dpo (DDP 4 ranks) on dpo-001/placement/sft-merged with --init_adapter = the
#                          previous stage's adapter (rpo-dpo1, SFT-D, DPO1-D) -> distill-018/placement/CAND and the
#                          CAND-merged receipt anchor; resubmitting resumes (training_guard, ckpt_latest).
#   pairs-dpo1             --cpus-per-task=72 --mem=192G --time=04:00:00   T-A: teacher chosen vs rpo-dpo1's failing
#                          samples (dpo-004/samples) -> distill-018/dpo1-d-pairs
#   sample COUNT           --gpus-per-node=1 --cpus-per-task=16 --mem=100G --time=02:00:00 --array=0-(COUNT-1)
#                          vLLM env, tensor_parallel 1, shard = SLURM_ARRAY_TASK_ID: sample_vllm run of DPO1-D's exact
#                          policy (distill-018/placement/dpo1-d-merged --exact-policy) on T-C (distill-018/t-c) ->
#                          distill-018/dpo2-d-samples; resumes from receipts.
#   finalize               CPU                     sample_vllm finalize -> distill-018/dpo2-d-samples (SAMPLING_COMPLETE)
#   pairs-dpo2             --cpus-per-task=72 --mem=192G --time=08:00:00   onpolicy_dpo PAIR_RULE on T-C, policy DPO1-D
#                          -> distill-018/dpo2-d-pairs
#   shard dev|test CAND I N [--resume]   --gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=04:00:00 (dev) or
#                          --time=12:00:00 (test): infer_shard shard as isambard_013.sbatch. CAND sft-d / dpo1-d / dpo2-d:
#                          the exact policy of distill-018/placement/CAND-merged; dev -> dev-eval-018/CAND/placement/
#                          shard-NN on the shared dev-eval-010/plan; test (the adopted CAND only) -> infer-v2-007/placement/
#                          shard-NN on infer-v2-007/plan. CAND-merged (dev only, the adopted CAND): the BF16 merge as is.
#   dev-merge CAND N       CPU                     dev-eval-018/CAND/placement/shard-* -> dev-eval-018/CAND/infer-placement
#                          dev shards and dev-merge of dpo2-d exit 0 without work when distill-018/dpo2-d-pairs is the
#                          NOT_TRAINED record (pairs-dpo2 below the 013 floors; train dpo2-d is then a no-op too)
#   dev-score              CPU                     -> dev-eval-018/scores (rpo-dpo1 from dev-eval-010; the three candidates)
#   select                 CPU                     -> dev-eval-018/selection.json + selection-report.md only (--no-submit);
#                          the adopted test chain is submitted by hand (submit_018_chain.sh CODE_ROOT test)
#   test-geometry          CPU                     evaluate_pair --stage-label selected: J-SFT infer-joint-002 vs P
#                          infer-placement-007 on eval-data-v1 -> geometry-v7
#   test-report            CPU, --time=06:00:00    adopted candidate vs rpo-dpo1 (infer-placement-005) -> test-report-018
#   s1-test                --cpus-per-task=72 --mem=200G --time=04:00:00   output_repair (amendment-012 S1 settings) on
#                          geometry-v7/placement-geometry.jsonl, every test request -> s1-018-test/placement-distill
#                          (the input of the cohort comparisons with the amendment-016 agents)
#   check-merged           CPU                     merged vs unmerged adopted candidate on dev -> dev-eval-018/merged-check
# Test sets and merged runs refuse unless dev-eval-018/selection.json, bound to its dev scores, adopted that candidate.
# Every output is new-only.
set -euo pipefail
test "$#" -ge 2 || exit 2
PAIR_CODE_ROOT="$1"
MODE="$2"
shift 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
D18="$RUN_ROOT/distill-018"
DEV="$RUN_ROOT/dev-select-v1"
EVAL="$RUN_ROOT/dev-eval-018"
CH=(fastfill.comparison.distill_018_chain)
GPUS=0
LIMIT=175m
KIND=cpu
adopted() {  # adopted CAND: dev-eval-018/selection.json, bound to its dev scores (load_selection), adopted CAND.
  "$PYTHON" -B - "$EVAL/selection.json" "$1" <<'PY' || { echo "PREFLIGHT FAIL: $1 is not the adopted 018 candidate" >&2; exit 1; }
import sys
from fastfill.comparison import distill_018_chain as ch
value, _ = ch._adopted(sys.argv[1])
assert value['selected'] == sys.argv[2], value['selected']
PY
}
candidate() { case "$1" in sft-d|dpo1-d|dpo2-d) ;; *) exit 2 ;; esac; }
case "$MODE" in
  recheck)
    test "$#" -eq 0 || exit 2
    ARGS=(fastfill.comparison.onpolicy_dpo recheck --data "$DEV" --ref-root "$RUN_ROOT/dev-eval-010" --workers 30) ;;
  tc-data)
    test "$#" -eq 0 || exit 2
    ARGS=("${CH[@]}" tc-data --plan "$D18/plan" --out "$D18/t-c") ;;
  sft-data)
    test "$#" -eq 0 || exit 2
    LIMIT=230m
    ARGS=("${CH[@]}" sft-data --data "$RUN_ROOT/main-data-v1" --exclude "$RUN_ROOT/dpo-001/positives"
          --sft-runs "$RUN_ROOT/sft-joint-001" "$RUN_ROOT/sft-placement-001" --teacher "$D18/teacher"
          --base-model "$RUN_ROOT/dpo-003/placement/dpo1-merged"
          --dev "$DEV" "$RUN_ROOT/dev-full-v1" --test "$RUN_ROOT/eval-data-v1" --out "$D18/sft-d-rows" --workers 70) ;;
  train)
    test "$#" -eq 1 || exit 2
    candidate "$1"
    CAND="$1"
    KIND=train GPUS=4 LIMIT=350m
    case "$CAND" in sft-d) STEP=train-sft ;; dpo1-d) STEP=train-dpo1 ;; dpo2-d) STEP=train-dpo2 ;; esac
    ARGS=("${CH[@]}" "$STEP" --run-root "$RUN_ROOT") ;;
  pairs-dpo1)
    test "$#" -eq 0 || exit 2
    LIMIT=230m
    ARGS=("${CH[@]}" pairs-dpo1 --data "$RUN_ROOT/dpo-004/train-subset" --samples "$RUN_ROOT/dpo-004/samples"
          --base-model "$RUN_ROOT/dpo-003/placement/dpo1-merged" --targets "$D18/sft-d-rows" --out "$D18/dpo1-d-pairs"
          --workers 70) ;;
  sample)
    test "$#" -eq 1 || exit 2
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
    KIND=vllm GPUS=1 LIMIT=115m
    ARGS=(fastfill.comparison.sample_vllm run --data "$D18/t-c" --base "$PROJECT_ROOT/models/Qwen3-8B" --arm placement
          --model "$D18/placement/dpo1-d-merged" --exact-policy --out "$D18/dpo2-d-samples" --shard-index "$INDEX"
          --shard-count "$COUNT") ;;
  finalize)
    test "$#" -eq 0 || exit 2
    ARGS=(fastfill.comparison.sample_vllm finalize --data "$D18/t-c" --out "$D18/dpo2-d-samples") ;;
  pairs-dpo2)
    test "$#" -eq 0 || exit 2
    LIMIT=470m
    ARGS=("${CH[@]}" pairs-dpo2 --data "$D18/t-c" --samples "$D18/dpo2-d-samples"
          --base-model "$D18/placement/dpo1-d-merged" --targets "$D18/sft-d-rows" --t-a "$RUN_ROOT/dpo-004/train-subset"
          --out "$D18/dpo2-d-pairs" --workers 70) ;;
  shard)
    test "$#" -eq 4 || { test "$#" -eq 5 && test "$5" = --resume; } || exit 2
    SET="$1" CAND="$2" INDEX="$3" COUNT="$4"
    case "$INDEX" in ''|*[!0-9]*) exit 2 ;; esac
    case "$COUNT" in ''|*[!0-9]*) exit 2 ;; esac
    test "$INDEX" -lt "$COUNT" || exit 2
    candidate "${CAND%-merged}"
    MODEL_ARGS=(--model "$D18/placement/${CAND%-merged}-merged")
    if test "${CAND%-merged}" = "$CAND"; then MODEL_ARGS+=(--exact-policy); else test "$SET" = dev || exit 2; fi
    case "$SET" in
      dev) DATA="$DEV"; INFER_ROOT="$EVAL/$CAND"; PLAN="$RUN_ROOT/dev-eval-010/plan"; LIMIT=230m ;;
      test) DATA="$RUN_ROOT/eval-data-v1"; INFER_ROOT="$RUN_ROOT/infer-v2-007"; PLAN="$INFER_ROOT/plan"; LIMIT=710m ;;
      *) exit 2 ;;
    esac
    KIND=shard GPUS=1
    ARGS=(fastfill.comparison.infer_shard shard --data "$DATA" --base "$PROJECT_ROOT/models/Qwen3-8B" --arm placement
          "${MODEL_ARGS[@]}" --plan "$PLAN" --shard-index "$INDEX" --shard-count "$COUNT"
          --out "$INFER_ROOT/placement/shard-$(printf %02d "$INDEX")" ${5:+--resume}) ;;
  dev-merge)
    test "$#" -eq 2 || exit 2
    candidate "${1%-merged}"
    CAND="$1"
    case "$2" in ''|*[!0-9]*) exit 2 ;; esac
    SHARDS=()
    for ((i = 0; i < $2; i++)); do SHARDS+=("$EVAL/$1/placement/shard-$(printf %02d "$i")"); done
    ARGS=(fastfill.comparison.infer_shard merge --data "$DEV" --plan "$RUN_ROOT/dev-eval-010/plan" --arm placement
          --out "$EVAL/$1/infer-placement" --shards "${SHARDS[@]}") ;;
  dev-score)
    test "$#" -eq 0 || exit 2
    ARGS=("${CH[@]}" dev-score --data "$DEV" --root "$EVAL" --ref-root "$RUN_ROOT/dev-eval-010" --out "$EVAL/scores"
          --workers 30) ;;
  select)
    test "$#" -eq 0 || exit 2
    ARGS=("${CH[@]}" select --scores "$EVAL/scores" --out "$EVAL/selection.json" --code-root "$PAIR_CODE_ROOT" --shards 6
          --no-submit) ;;
  test-geometry)
    test "$#" -eq 0 || exit 2
    ARGS=(fastfill.comparison.evaluate_pair --data "$RUN_ROOT/eval-data-v1" --joint "$RUN_ROOT/infer-joint-002"
          --placement "$RUN_ROOT/infer-placement-007" --out "$RUN_ROOT/geometry-v7" --workers 24 --repeats 2000
          --stage-label selected) ;;
  test-report)
    test "$#" -eq 0 || exit 2
    LIMIT=350m
    ARGS=("${CH[@]}" test-report --data "$RUN_ROOT/eval-data-v1" --selection "$EVAL/selection.json"
          --p18 "$RUN_ROOT/infer-placement-007" --ref "$RUN_ROOT/infer-placement-005" --out "$RUN_ROOT/test-report-018"
          --workers 30) ;;
  s1-test)
    test "$#" -eq 0 || exit 2
    LIMIT=230m
    ARGS=(fastfill.comparison.output_repair --data "$RUN_ROOT/eval-data-v1" --geometry "$RUN_ROOT/geometry-v7/placement-geometry.jsonl"
          --requests-arm placement --out "$RUN_ROOT/s1-018-test/placement-distill" --workers 70 --max-trials 2000
          --time-limit 60) ;;
  check-merged)
    test "$#" -eq 0 || exit 2
    ARGS=("${CH[@]}" check-merged --data "$DEV" --root "$EVAL" --ref-root "$RUN_ROOT/dev-eval-010"
          --out "$EVAL/merged-check" --workers 30) ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
if { test "$MODE" = dev-merge || { test "$MODE" = shard && test "$SET" = dev; }; } && test "${CAND%-merged}" = dpo2-d \
    && (cd "$PAIR_CODE_ROOT" && PYTHONPATH="$PAIR_CODE_ROOT" "$PYTHON" -B -c 'import sys
from fastfill.comparison import distill_018_chain as ch
sys.exit(not ch.not_trained(sys.argv[1]))' "$RUN_ROOT"); then
  echo '{"status": "A018_DPO2_NOT_TRAINED", "skipped": "'"$MODE"' dpo2-d"}'
  exit 0
fi
if test "$MODE" = shard; then
  test -f "$PLAN/MANIFEST.json"
  test -f "${MODEL_ARGS[1]}.json" || { echo "PREFLIGHT FAIL: no lineage receipt ${MODEL_ARGS[1]}.json" >&2; exit 1; }
fi
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=/tmp
cd "$PAIR_CODE_ROOT"
if test "$MODE" = shard && { test "$SET" = test || test "${CAND%-merged}" != "$CAND"; }; then adopted "${CAND%-merged}"; fi
case "$KIND" in
  cpu)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ;;
  vllm)  # As isambard_013.sbatch; compile caches under PROJECTDIR (scratch file limit).
    PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
    JOB_CACHE="$PROJECT_ROOT/.cache-jobs/a018"
    test -f "$D18/t-c/MANIFEST.json" || { echo "PREFLIGHT FAIL: $D18/t-c missing" >&2; exit 1; }
    test -f "$D18/placement/dpo1-d-merged.json" || { echo "PREFLIGHT FAIL: no DPO1-D receipt" >&2; exit 1; }
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
      echo "PREFLIGHT FAIL: $PYTHON cannot import vllm + shapely + fastfill.comparison.sample_vllm" >&2; exit 1; } ;;
  train|shard)
    module load cudatoolkit/24.11_12.6
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export CUDA_PATH="$CUDA_HOME" WANDB_DISABLED=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ;;
esac
if test "$KIND" = train; then  # One writer per stage directory, also against a manual resubmit.
  mkdir -p "$D18/placement"
  exec 9>"$D18/placement/.$CAND.lock"
  flock -n 9 || { echo "another job owns $D18/placement/$CAND" >&2; exit 1; }
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
