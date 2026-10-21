#!/bin/bash
#SBATCH --job-name=unknownsize-021-eval
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --exclude=nid010216,nid011181,nid010569
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-021-eval-%j.out
# Amendment 021 sections 1.4, 3, 4 (RUN/a021; fastfill.comparison.a021_select): CODE_ROOT MODE [ARGS]. One script for
# every step; the header is the CPU default (32 cores, 96G, 3 h). GPU steps take their resources on the sbatch line
# (submit_021_eval.sh) and their inner timeout from HOURS. Paths and model arguments come from a021_select argv, so the
# shell only validates the argument shapes (before the allocation) and sets up the environment:
#   confirm                        CPU   a021_select confirm -> a021/confirm-v1, a021/engine-v1, a021/cohorts
#   plan engine|confirm            CPU   infer_shard plan (pinned Qwen3-8B tokenizer) -> a021/hf/SLICE/plan
#   vllm SLICE CAND BATCH HOURS    --gpus-per-node=1 --cpus-per-task=16 --mem=100G   vLLM env: a021_select greedy
#                                  (CAND r | psft | op13 | d18 | arm; BATCH objects | hash) -> a021/vllm/SLICE/CAND[-hash]
#   hf SLICE CAND I N HOURS [--resume]   --gpus-per-node=1 --cpus-per-task=8 --mem=96G   infer_shard shard (HF exact,
#                                  8B models only) on the SLICE plan -> a021/hf/SLICE/CAND/placement/shard-II
#   hf-merge SLICE CAND N          CPU   infer_shard merge -> a021/hf/SLICE/CAND/infer-placement
#   o2nt SLICE HOURS               --gpus-per-node=2 --cpus-per-task=32 --mem=200G   vLLM env: llm_open run --system O2nt
#                                  --arm placement (SLICE dev-select | confirm | cohort) -> a021/o2nt/SLICE
#   o2nt-score | engine-check | select | test-geometry   CPU (30 workers; test-geometry --time=06:00:00)
# SLICE: dev-select | confirm | engine | test. Every output is new-only; a vLLM or llm_open run resumes from receipts.
# Amendment 028 (fastfill.comparison.a028; submit_028.sh):
#   a028 greedy dev-select|test HOURS   --gpus-per-node=1 --cpus-per-task=16 --mem=100G   vLLM env: J27 greedy (Arm A's
#                                  engine, joint requests) -> a028/vllm/SLICE/j27
#   a028 compare HOURS             CPU (70 workers): J27 geometry rows, S1 and 27B-J vs 27B-P -> a028/compare
set -euo pipefail
test "$#" -ge 2 || exit 2
PAIR_CODE_ROOT="$1"
MODE="$2"
shift 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
A21=(fastfill.comparison.a021_select)
KIND=cpu GPUS=0 LIMIT=175m ITEMS=()
number() { case "$1" in ''|*[!0-9]*) exit 2 ;; esac; }
slice() { case "$1" in dev-select|confirm|engine|test) ;; *) exit 2 ;; esac; }
cand() { case "$1" in r|psft|op13|d18|arm) ;; *) exit 2 ;; esac; }
hours() { number "$1"; test "$1" -ge 1 -a "$1" -le 24 || exit 2; LIMIT="$(($1 * 60 - 5))m"; }
case "$MODE" in
  confirm)
    test "$#" -eq 0 || exit 2
    ARGS=("${A21[@]}" confirm --data "$RUN_ROOT/main-data-v1" --dev "$RUN_ROOT/dev-select-v1" --run-root "$RUN_ROOT"
          --sft-runs "$RUN_ROOT/sft-joint-001" "$RUN_ROOT/sft-placement-001") ;;
  plan)
    test "$#" -eq 1 || exit 2
    case "$1" in engine|confirm) ;; *) exit 2 ;; esac
    ITEMS=(plan "$1") ;;
  vllm)
    test "$#" -eq 4 || exit 2
    slice "$1"; cand "$2"; case "$3" in objects|hash) ;; *) exit 2 ;; esac; hours "$4"
    KIND=vllm GPUS=1 ITEMS=(vllm "$1" "$2" "$3") ;;
  hf)
    test "$#" -eq 5 || { test "$#" -eq 6 && test "$6" = --resume; } || exit 2
    slice "$1"; cand "$2"; number "$3"; number "$4"; test "$3" -lt "$4" || exit 2; hours "$5"
    KIND=shard GPUS=1 ITEMS=(hf "$1" "$2" "$3" "$4") ;;
  hf-merge)
    test "$#" -eq 3 || exit 2
    slice "$1"; cand "$2"; number "$3"
    ITEMS=(hf-merge "$1" "$2" "$3") ;;
  o2nt)
    test "$#" -eq 2 || exit 2
    case "$1" in dev-select|confirm|cohort) ;; *) exit 2 ;; esac; hours "$2"
    KIND=vllm GPUS=2 ITEMS=(o2nt "$1") ;;
  a028)
    test "$#" -ge 2 || exit 2
    case "$1" in
      greedy) test "$#" -eq 3 || exit 2; case "$2" in dev-select|test) ;; *) exit 2 ;; esac; hours "$3"
              KIND=vllm GPUS=1 TAG="a028-greedy-$2" ARGS=(fastfill.comparison.a028 greedy --run-root "$RUN_ROOT" --slice "$2") ;;
      compare) test "$#" -eq 2 || exit 2; hours "$2"
               ARGS=(fastfill.comparison.a028 compare --run-root "$RUN_ROOT" --workers 70) ;;
      *) exit 2 ;;
    esac ;;
  o2nt-score|engine-check|select|test-geometry)
    test "$#" -eq 0 || exit 2
    if test "$MODE" = test-geometry; then LIMIT=350m; fi
    ARGS=("${A21[@]}" "$MODE" --run-root "$RUN_ROOT" --workers 30) ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
if test "$MODE" = hf && test "$2" = psft; then  # Amendment 005 section 6, as isambard_010_infer_shard.sbatch.
  grep -q '"status": "PROBE_PASS"' "$RUN_ROOT/infer-v2-002/placement/probe/probe.json" || {
    echo "PREFLIGHT FAIL: the P-SFT runtime merge has no passed amendment-005 probe" >&2; exit 1; }
fi
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=/tmp
cd "$PAIR_CODE_ROOT"
if test "${#ITEMS[@]}" -gt 0; then  # Paths and model arguments from the one place that defines them.
  TEXT=$("$PYTHON" -B -m "${A21[@]}" argv "${ITEMS[@]}" --run-root "$RUN_ROOT") || {
    echo "PREFLIGHT FAIL: a021_select argv ${ITEMS[*]}" >&2; exit 1; }
  ARGS=()
  while IFS= read -r line; do ARGS+=("$line"); done <<< "$TEXT"
fi
case "$KIND" in
  cpu)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ;;
  vllm)  # As isambard_018.sbatch; compile caches under PROJECTDIR (scratch file limit).
    PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
    JOB_CACHE="$PROJECT_ROOT/.cache-jobs/${TAG:-a021-$(IFS=-; echo "${ITEMS[*]}")}"  # One per concurrent job.
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
    "$PYTHON" -B -c 'import vllm, shapely, fastfill.comparison.a021_select' || {
      echo "PREFLIGHT FAIL: $PYTHON cannot import vllm + shapely + fastfill.comparison.a021_select" >&2; exit 1; } ;;
  shard)
    module load cudatoolkit/24.11_12.6
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export CUDA_PATH="$CUDA_HOME" WANDB_DISABLED=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    "$PYTHON" -B - <<'PY'
from pathlib import Path
from fastfill.comparison import train_pair_helpers as h
p = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013/context-capacity-001')
h.verify_artifacts(p, h.strict_json((p / 'SHA256.json').read_text()))
r = h.strict_json((p / 'result.json').read_text())
assert r['status'] == 'CAPACITY_COMPLETE' and r['batch_size'] == 8 and r['cached_tokens'] == 40960
assert r['peak_allocated_bytes'] < .9 * r['gpu_total_bytes']
PY
    ;;
esac
if test "$MODE" = hf && test "${6:-}" = --resume; then ARGS+=(--resume); fi
if test "$GPUS" -gt 0; then
  nvidia-smi -L
  SEEN=$(nvidia-smi -L | grep -c '^GPU ' || true)
  test "$SEEN" -ge "$GPUS" || { echo "PREFLIGHT FAIL: $MODE needs $GPUS GPUs, sees $SEEN" >&2; exit 1; }
fi
exec timeout --signal=TERM --kill-after=60s "$LIMIT" "$PYTHON" -B -u -m "${ARGS[@]}"
