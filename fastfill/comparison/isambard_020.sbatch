#!/bin/bash
#SBATCH --job-name=unknownsize-020-sizers
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --exclude=nid010216,nid011181,nid010569
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-020-sizers-%j.out
# Amendment 020 stage 1 (EXPLORATORY): sbatch [resources] isambard_020.sbatch CODE_ROOT MODE [ARGS]
# Job names, resources and time limits come from submit_020_sizers.sh; this header is the CPU default (no --gres line:
# CPU jobs pass --gres=none, O2 shards --gpus-per-node). Every job excludes nodes whose GPUs failed or were held
# (nid010216: amendment 018 rule 8; nid010569, nid011181: research log 19:37 and 20:18 UTC).
#   rooms        CPU: sizers_020 rooms -> RUN/a020-rooms-v1.json (an existing file must equal its rebuild)
#   l-size       CPU, API: sizers_020 run-api -> RUN/sizes-020-L. The API cap preflight runs first (refusal: status
#                CAP_REFUSED, exit 1, no call). Exit 2 (API failure, already retried per call) is retried here at
#                most twice after 10 min; a resubmission resumes from the receipts.
#   o2-size I N  llm_open's O2 GPUs, vLLM env: sizers_020 run-open shard I of N -> RUN/sizes-020-O2 (own compile cache)
#   report       CPU: sizers_020 finalize RUN/sizes-020-O2, then report -> RUN/sizes-020-report-v1 (refuses unless
#                O2-size is complete and while L-size is incomplete; a CAP_REFUSED or never-submitted L-size is reported
#                as not run)
set -euo pipefail
test "$#" -ge 2 || exit 2
CODE_ROOT="$1"
MODE="$2"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
DATA="$RUN/eval-data-v1"
ROOMS="$RUN/a020-rooms-v1.json"
PY="$PROJECT_ROOT/env/bin/python"
VLLM_PY=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
M=(-B -u -m fastfill.comparison.sizers_020)
case "$MODE:$#" in
  rooms:2|l-size:2|report:2) ;;
  o2-size:4) case "$3:$4" in *[!0-9:]*|:*|*:) exit 2 ;; esac ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE_ROOT"
case "$MODE" in
  rooms)
    exec "$PY" "${M[@]}" rooms --data "$DATA" --e2 "$RUN/optiscene-014/e2" --e3 "$RUN/optiscene-015/e3" --out "$ROOMS" ;;
  l-size)
    export FASTFILL_API_ENV=/home/u6uv/shanliantian.u6uv/.fastfill_api.env
    unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL
    END=$((SECONDS + 470 * 60))
    for invocation in 1 2 3; do
      LEFT=$(( (END - SECONDS) / 60 ))
      test "$LEFT" -ge 5 || exit 124
      rc=0
      timeout --signal=TERM --kill-after=30s "${LEFT}m" "$PY" "${M[@]}" run-api --data "$DATA" --rooms "$ROOMS" \
        --run-root "$RUN" --out "$RUN/sizes-020-L" --env-file "$FASTFILL_API_ENV" --concurrency 4 || rc=$?
      test "$rc" -eq 2 || exit "$rc"
      test "$invocation" -eq 3 || sleep 600
    done
    exit 2 ;;
  o2-size)
    SHARD_CACHE="$PROJECT_ROOT/.cache-jobs/sizers-020/O2-s$3"  # Shards run side by side: one compile cache each.
    # vLLM JIT kernels need an nvcc matching torch cu130: prefer the env's pip CUDA 13 toolkit, else the module.
    PIP_CUDA="$(dirname "$(dirname "$VLLM_PY")")/lib/python3.11/site-packages/nvidia/cu13"
    if test -x "$PIP_CUDA/bin/nvcc"; then
      CUDA_HOME="$PIP_CUDA"
    else
      module load cudatoolkit || { echo "PREFLIGHT FAIL: module load cudatoolkit" >&2; exit 1; }
      CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi
    export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$(dirname "$VLLM_PY"):$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
    export HF_HOME=/scratch/u6uv/shanliantian.u6uv/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1 XDG_CACHE_HOME="$SHARD_CACHE/xdg" VLLM_CACHE_ROOT="$SHARD_CACHE/vllm"
    export TORCHINDUCTOR_CACHE_DIR="$SHARD_CACHE/torchinductor" TRITON_CACHE_DIR="$SHARD_CACHE/triton"
    export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
    mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    "$VLLM_PY" -B -c 'import vllm, shapely, fastfill.comparison.sizers_020, fastfill.comparison.llm_open' || {
      echo "PREFLIGHT FAIL: $VLLM_PY cannot import vllm + shapely + sizers_020 + llm_open" >&2; exit 1; }
    nvidia-smi -L
    exec "$VLLM_PY" "${M[@]}" run-open --data "$DATA" --rooms "$ROOMS" --model-root "$PROJECT_ROOT/models" \
      --out "$RUN/sizes-020-O2" --shard-index "$3" --shard-count "$4" ;;
  report)
    "$PY" "${M[@]}" finalize --rooms "$ROOMS" --out "$RUN/sizes-020-O2" || echo "O2-size not finalized (see above)"
    exec "$PY" "${M[@]}" report --data "$DATA" --rooms "$ROOMS" --run-root "$RUN" --l "$RUN/sizes-020-L" \
      --o2 "$RUN/sizes-020-O2" --l-joint "$RUN/llm-eval-L-selected-v1" --out "$RUN/sizes-020-report-v1" ;;
esac
