#!/bin/bash
#SBATCH --job-name=optiscene-014
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/optiscene-014-%j.out
# Amendment 014 (exploratory; "OptiScene code recipe (README, Qwen2.5-7B-Instruct)"): CODE_ROOT MODE [ARGS].
# GPU, CPU and time come from submit_014.sh's sbatch options; this header is the CPU default.
#   train STAGE PREV   1 GPU (sft 6 h, dpo 24 h): optiscene_eval train in the optiscene-014 venv; STAGE sft|dpo1|dpo2,
#                      PREV = the stage's previous job id (train refuses unless it ended COMPLETED or TIMEOUT) or '-'.
#                      Slurm's own time limit ends a segment (state TIMEOUT); the next segment resumes (F3).
#   merge STAGE        CPU: checkpoint rule + their merge_lora (bf16, CPU) -> optiscene-014/STAGE-merged
#   plan-e2            CPU: eval-data-v1 paper strata -> optiscene-014/e2; 20 dev-select-v1 rooms -> e2-dev-smoke
#                      (exposure against the uploaded plan-e1); finished outputs are skipped
#   gen MODEL          1 GPU, vLLM env: sft|dpo1|dpo2 = E1 + E2 (dpo2 also the README reasoning prompt on E1);
#                      8b = the 20-room dev smoke (strict parse >= 0.9 or stop), then E2. Finished outputs are skipped.
#   score              CPU 16: E1/E2/P tables -> optiscene-014/score
set -euo pipefail
test "$#" -ge 2 || exit 2
CODE_ROOT="$1"
MODE="$2"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
OUT="$RUN/optiscene-014"
VENV_PY="$PROJECT_ROOT/envs/optiscene-014/bin/python"
ENV_JSON="$PROJECT_ROOT/envs/optiscene-014.env.json"
DATA="$PROJECT_ROOT/data/optiscene-official"
BASE="$PROJECT_ROOT/models/Qwen2.5-7B-Instruct"
E1="$PROJECT_ROOT/data/optiscene-014-e1"
EIGHT_B="$PROJECT_ROOT/models/OptiScene-Qwen-8B-Final"
VLLM_PY=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
CACHE="$PROJECT_ROOT/.cache-jobs/optiscene-014"
SCRIPT="$CODE_ROOT/fastfill/comparison/optiscene_eval.py"
GPU=0
case "$MODE" in
  train)
    test "$#" -eq 4 || exit 2
    case "$3" in sft|dpo1|dpo2) ;; *) exit 2 ;; esac
    case "$4" in -) PREV=() ;; ''|*[!0-9]*) exit 2 ;; *) PREV=(--prev-job "$4") ;; esac
    GPU=1 ;;
  merge)
    test "$#" -eq 3 || exit 2
    case "$3" in sft|dpo1|dpo2) ;; *) exit 2 ;; esac ;;
  gen)
    test "$#" -eq 3 || exit 2
    case "$3" in sft|dpo1|dpo2|8b) ;; *) exit 2 ;; esac
    GPU=1 ;;
  plan-e2|score) test "$#" -eq 2 || exit 2 ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
export HF_HOME="$PROJECT_ROOT/.cache-jobs/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true HF_DATASETS_CACHE="$CACHE/datasets"
export MODELSCOPE_CACHE="$CACHE/modelscope" XDG_CACHE_HOME="$CACHE/xdg" TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" VLLM_CACHE_ROOT="$CACHE/vllm" TMPDIR=/tmp
mkdir -p "$HF_DATASETS_CACHE" "$MODELSCOPE_CACHE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$VLLM_CACHE_ROOT"
if test "$GPU" = 1; then  # One device only: the Trainer must not wrap the model in DataParallel.
  SEEN=$(nvidia-smi -L | grep -c '^GPU ' || true)
  case "${CUDA_VISIBLE_DEVICES:-}" in ''|*,*) SEEN=bad ;; esac
  test "$SEEN" = 1 || { echo "PREFLIGHT FAIL: need exactly 1 GPU (nvidia-smi $SEEN, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-})" >&2; exit 1; }
fi
cd "$CODE_ROOT"
gen() {  # gen NAME ARGS...: one generate output, skipped once published
  if test -f "$OUT/gen/$1/MANIFEST.json"; then echo "skip $1 (published)"; return; fi
  local name="$1"
  shift
  "$VLLM_PY" -B -u -m fastfill.comparison.optiscene_eval generate --out "$OUT/gen/$name" "$@"
}
case "$MODE" in
  train)
    module load cudatoolkit/24.11_12.6
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export CUDA_PATH="$CUDA_HOME" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
    exec "$VENV_PY" -B -u "$SCRIPT" train --stage "$3" --run "$OUT" --data "$DATA" --base "$BASE" --env "$ENV_JSON" \
      --python "$VENV_PY" ${PREV[@]+"${PREV[@]}"} ;;
  merge)
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16
    exec "$VENV_PY" -B -u "$SCRIPT" merge --stage "$3" --run "$OUT" --base "$BASE" --env "$ENV_JSON" --python "$VENV_PY" ;;
  plan-e2)
    test -f "$OUT/e2/MANIFEST.json" || "$PROJECT_ROOT/env/bin/python" -B -u -m fastfill.comparison.optiscene_eval \
      plan-e2 --eval-data "$RUN/eval-data-v1" --data "$DATA" --e1 "$E1" --out "$OUT/e2"
    test -f "$OUT/e2-dev-smoke/MANIFEST.json" || "$PROJECT_ROOT/env/bin/python" -B -u -m fastfill.comparison.optiscene_eval \
      plan-e2 --eval-data "$RUN/dev-select-v1" --data "$DATA" --e1 "$E1" --out "$OUT/e2-dev-smoke" --limit 20 ;;
  gen)
    # vLLM JIT kernels need an nvcc matching torch cu130: prefer the env's pip CUDA 13 toolkit, else the module.
    PIP_CUDA="$(dirname "$(dirname "$VLLM_PY")")/lib/python3.11/site-packages/nvidia/cu13"
    if test -x "$PIP_CUDA/bin/nvcc"; then
      CUDA_HOME="$PIP_CUDA"
    else
      module load cudatoolkit || { echo "PREFLIGHT FAIL: module load cudatoolkit" >&2; exit 1; }
      CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi
    export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$(dirname "$VLLM_PY"):$CUDA_HOME/bin:$PATH"
    export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
    "$VLLM_PY" -B -c 'import vllm, shapely, fastfill.comparison.optiscene_eval' || {
      echo "PREFLIGHT FAIL: $VLLM_PY cannot import vllm + shapely + optiscene_eval" >&2; exit 1; }
    if test "$3" = 8b; then
      gen 8b-dev-smoke-v1 --plan "$OUT/e2-dev-smoke" --prompt v1 --model "$EIGHT_B" --gate-parse 0.9
      gen 8b-e2-v1 --plan "$OUT/e2" --prompt v1 --model "$EIGHT_B"
    else
      gen "$3-e1-sft" --plan "$E1" --prompt sft --model "$OUT/$3-merged"
      gen "$3-e2-sft" --plan "$OUT/e2" --prompt sft --model "$OUT/$3-merged"
      if test "$3" = dpo2; then gen dpo2-e1-reasoning --plan "$E1" --prompt reasoning --model "$OUT/dpo2-merged"; fi
    fi ;;
  score)
    P_RUN=$("$PROJECT_ROOT/env/bin/python" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["test"]["placement"]["inference"])' \
      "$RUN/dev-eval-010/selection.json")
    exec "$PROJECT_ROOT/env/bin/python" -B -u -m fastfill.comparison.optiscene_eval score --e1 "$E1" --e2 "$OUT/e2" \
      --gen "$OUT/gen" --eval-data "$RUN/eval-data-v1" --p-geometry "$RUN/geometry-v5" --p-run "$P_RUN" \
      --out "$OUT/score" --workers 16 ;;
esac
