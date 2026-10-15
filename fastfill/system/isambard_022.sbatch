#!/bin/bash
#SBATCH --job-name=unknownsize-022
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-022-%j.out
# Amendment 022 (fastfill.system.evaluate / package; RUN/system-022): CODE_ROOT STEP MODE [SCOPE HOURS]. The header is
# the CPU default; submit_022.sh sets the job name and the resources of each step on the sbatch line:
#   s1 final                      CPU 72 / 200G: evaluate s1 (output_repair on the delivered W's test rows; R reuses
#                                 RUN/s1-012-test) -> RUN/s1-022/placement-W
#   system MODE SCOPE HOURS       --gpus-per-node=2 (O2, TP 2) --cpus-per-task=72 --mem=200G, vLLM env: evaluate system
#                                 (pipeline.run, placer replayed from its test rows, S1 with 64 workers; cohort: the gate,
#                                 then gpt-6.1-sol with the API env file and the calls left of 1,200, then O2; test: O2)
#                                 -> RUN/system-022/MODE/SCOPE/system
#   summarize MODE SCOPE          CPU 32 / 96G, --time=06:00:00: evaluate summarize (30 workers, 355 min) ->
#                                 RUN/system-022/MODE/SCOPE/eval
#   package MODE                  CPU: package build -> RUN/system-022/MODE/package
# Sharded O2 fallback of the test scope (evaluate plan / shard / finalize; RUN/system-022/MODE/test-sharded):
#   plan MODE N                   CPU 72 / 200G: the placer stage once (S1 with 64 workers), N shards -> plan
#   shard MODE K HOURS            --gpus-per-node=2, vLLM env (cache per shard): O2 on shard K, resumes per chunk
#                                 -> shards/shard-K
#   finalize MODE                 CPU 72 / 200G: pipeline.run on the plan + the merged shards -> system
#   summarize MODE test-sharded   summarize --scope test --sharded -> eval
# MODE: interim (R = rpo-dpo1) | final (amendment-021 §4 delivered placer). SCOPE: cohort | test. Never resumes or
# replaces an output; the API key file is never printed.
set -euo pipefail
test "$#" -ge 3 || exit 2
CODE_ROOT="$1"
STEP="$2"
MODE="$3"
shift 3
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
API_ENV=/home/u6uv/shanliantian.u6uv/.fastfill_api.env
case "$MODE" in interim|final) ;; *) exit 2 ;; esac
scope() { case "$1" in cohort|test) ;; *) exit 2 ;; esac; }
LIMIT=175m GPUS=0 CACHE=""
case "$STEP" in
  s1)
    test "$#" -eq 0 -a "$MODE" = final || exit 2
    LIMIT=235m ARGS=(fastfill.system.evaluate s1 --run-root "$RUN_ROOT" --mode final --workers 70) ;;
  system)
    test "$#" -eq 2 || exit 2
    scope "$1"
    case "$2" in ''|*[!0-9]*) exit 2 ;; esac
    test "$2" -ge 1 -a "$2" -le 24 || exit 2
    LIMIT="$(($2 * 60 - 5))m" GPUS=2 CACHE="022-$MODE-$1"
    ARGS=(fastfill.system.evaluate system --run-root "$RUN_ROOT" --mode "$MODE" --scope "$1" --workers 64)
    if test "$1" = cohort; then ARGS+=(--env-file "$API_ENV"); fi ;;
  summarize)
    test "$#" -eq 1 || exit 2
    LIMIT=355m  # submit_022.sh gives summarize --time=06:00:00
    if test "$1" = test-sharded; then
      ARGS=(fastfill.system.evaluate summarize --run-root "$RUN_ROOT" --mode "$MODE" --scope test --sharded --workers 30)
    else
      scope "$1"
      ARGS=(fastfill.system.evaluate summarize --run-root "$RUN_ROOT" --mode "$MODE" --scope "$1" --workers 30)
    fi ;;
  plan)
    test "$#" -eq 1 || exit 2
    case "$1" in ''|*[!0-9]*) exit 2 ;; esac
    test "$1" -ge 1 -a "$1" -le 16 || exit 2
    LIMIT=235m ARGS=(fastfill.system.evaluate plan --run-root "$RUN_ROOT" --mode "$MODE" --shards "$1" --workers 64) ;;
  shard)
    test "$#" -eq 2 || exit 2
    case "$1" in ''|*[!0-9]*) exit 2 ;; esac
    case "$2" in ''|*[!0-9]*) exit 2 ;; esac
    test "$1" -le 15 -a "$2" -ge 1 -a "$2" -le 24 || exit 2
    LIMIT="$(($2 * 60 - 5))m" GPUS=2 CACHE="022-$MODE-test-sharded-$1"
    ARGS=(fastfill.system.evaluate shard --run-root "$RUN_ROOT" --mode "$MODE" --index "$1") ;;
  finalize)
    test "$#" -eq 0 || exit 2
    LIMIT=235m ARGS=(fastfill.system.evaluate finalize --run-root "$RUN_ROOT" --mode "$MODE" --workers 64) ;;
  package)
    test "$#" -eq 0 || exit 2
    ARGS=(fastfill.system.package build --run-root "$RUN_ROOT" --mode "$MODE") ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=/tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL
cd "$CODE_ROOT"
if test "$GPUS" -gt 0; then  # system / shard: O2 on vLLM, as isambard_021_eval.sbatch; caches under PROJECTDIR.
  PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
  JOB_CACHE="$PROJECT_ROOT/.cache-jobs/$CACHE"
  PIP_CUDA="$(dirname "$(dirname "$PYTHON")")/lib/python3.11/site-packages/nvidia/cu13"
  if test -x "$PIP_CUDA/bin/nvcc"; then
    CUDA_HOME="$PIP_CUDA"
  else
    module load cudatoolkit || { echo "PREFLIGHT FAIL: module load cudatoolkit" >&2; exit 1; }
    NVCC=$(command -v nvcc) || { echo "PREFLIGHT FAIL: no nvcc after 'module load cudatoolkit'" >&2; exit 1; }
    CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
  fi
  export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
  export HF_HOME=/scratch/u6uv/shanliantian.u6uv/.cache/huggingface XDG_CACHE_HOME="$JOB_CACHE/xdg"
  export VLLM_CACHE_ROOT="$JOB_CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor"
  export TRITON_CACHE_DIR="$JOB_CACHE/triton" VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  export VLLM_ENGINE_READY_TIMEOUT_S=3600
  mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
  "$PYTHON" -B -c 'import vllm, shapely, fastfill.system.evaluate' || {
    echo "PREFLIGHT FAIL: $PYTHON cannot import vllm + shapely + fastfill.system.evaluate" >&2; exit 1; }
  test -d "$PROJECT_ROOT/models/Qwen3.8-27B" || { echo "PREFLIGHT FAIL: no O2 weights" >&2; exit 1; }
  if test "$1" = cohort; then
    test -f "$API_ENV" || { echo "PREFLIGHT FAIL: no API env file" >&2; exit 1; }
  fi
  nvidia-smi -L
  SEEN=$(nvidia-smi -L | grep -c '^GPU ' || true)
  test "$SEEN" -ge "$GPUS" || { echo "PREFLIGHT FAIL: system needs $GPUS GPUs, sees $SEEN" >&2; exit 1; }
fi
exec timeout --signal=TERM --kill-after=60s "$LIMIT" "$PYTHON" -B -u -m "${ARGS[@]}"
