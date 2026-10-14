#!/bin/bash
#SBATCH --job-name=unknownsize-llm-open
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-llm-open-%j.out
# Usage: sbatch $(python -m fastfill.comparison.llm_open slurm --system S --mode M) \
#          isambard_llm_open.sbatch CODE_ROOT S smoke|full|pilot [placement|placement-gt]
# placement (amendment 016 section 2): P's request instead of J's (llm_open --arm placement), output
# llm-open-S-P-smoke-001 / llm-open-S-P-001 / llm-open-S-P-pilot-001; placement-gt (addendum A1 supplementary,
# pilot only): P's request with GT sizes (--sizes gt), output llm-open-S-PGT-pilot-001. pilot (addendum A6/B1):
# the cohort subset RUN/a016-pilot-v1.json (agents_016 pilot), smoke-gated like a smoke (it replaces the smoke);
# placement arms only. Without the 4th argument everything below is as before.
# t018 sNN (amendment 018 teacher, submit_018_teacher.sh; O2 only): shard sNN of the distill_018 plan, P's request
# (--arm placement --all-rooms --uids RUN/distill-018/plan/shards/sNN.json, cohort RUN/distill-018/plan/cohort.json),
# output llm-open-O2-T018-sNN-001. submit_018_teacher.sh gives it its own job name (never the singleton below).
# GPUs/CPUs/memory/time per system come from llm_open.SYSTEMS (submit_llm_open.sh passes them).
# smoke: 2 cohort rooms; exits non-zero only when the system cannot load or execute (amendment 006:
# cap-hit reasoning or invalid answers are measured failures, and the full run still follows).
# Preflight (fails fast, before the GPU work): cudatoolkit/nvcc for vLLM JIT kernels, and the vLLM env
# must import shapely + fastfill.comparison.
# Resubmitting the same system and mode resumes from receipts under the same run identity.
set -euo pipefail
test "$#" -eq 3 -o "$#" -eq 4
PAIR_CODE_ROOT="$1"
SYSTEM="$2"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
COHORT="$PAIR_RUN_ROOT/llm-cohort-v1.json"
DATA="$PAIR_RUN_ROOT/eval-data-v1"
case "$SYSTEM" in O1a|O1b|O2|O3|O4|O5|O6|O7) ;; *) exit 2 ;; esac
case "$3" in
  smoke) OUT="llm-open-$SYSTEM-smoke-001"; EXTRA=(--limit 2 --smoke-gate) ;;
  full) OUT="llm-open-$SYSTEM-001"; EXTRA=() ;;
  pilot) OUT="llm-open-$SYSTEM-pilot-001"; EXTRA=(--uids "$PAIR_RUN_ROOT/a016-pilot-v1.json" --smoke-gate) ;;
  t018) test "$SYSTEM" = O2 || exit 2
    case "${4-}" in s[0-9][0-9]) ;; *) exit 2 ;; esac
    DATA="$PAIR_RUN_ROOT/distill-018/plan"; COHORT="$DATA/cohort.json"; OUT="llm-open-$SYSTEM-T018-$4-001"
    CACHE_TAG="-T018-$4"  # Shards run side by side: each compiles into its own cache, never the shared O2 one.
    EXTRA=(--arm placement --all-rooms --uids "$DATA/shards/$4.json") ;;
  *) exit 2 ;;
esac
case "${4-joint}" in
  joint) test "$3" != pilot || exit 2 ;;
  placement) OUT="llm-open-$SYSTEM-P-${OUT#llm-open-$SYSTEM-}"; EXTRA+=(--arm placement) ;;
  placement-gt) test "$3" = pilot || exit 2
    OUT="llm-open-$SYSTEM-PGT-${OUT#llm-open-$SYSTEM-}"; EXTRA+=(--arm placement --sizes gt) ;;
  s[0-9][0-9]) test "$3" = t018 || exit 2 ;;
  *) exit 2 ;;
esac
PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
CACHE=/scratch/u6uv/shanliantian.u6uv/.cache
# Scratch hit its 1.024M-file project limit (2026-10-14); compile caches (many small files) live under
# $PROJECTDIR instead. HF_HOME stays on scratch: offline, read-only model metadata.
SYSTEM_CACHE="/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/.cache-jobs/llm-open/$SYSTEM${CACHE_TAG-}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
# JIT builds (FlashInfer, DeepGEMM) need an nvcc matching torch cu130: DeepGEMM asserts nvcc >= 12.9 and the
# cluster module is 12.6, so prefer the env's pip CUDA 13 toolkit; fall back to the module (v1_1 pattern).
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
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$CACHE/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$SYSTEM_CACHE/xdg" VLLM_CACHE_ROOT="$SYSTEM_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$SYSTEM_CACHE/torchinductor" TRITON_CACHE_DIR="$SYSTEM_CACHE/triton"
# FlashInfer's sampler JIT build fails on this GH200 image; large checkpoints may need a long engine start.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
# DeepSeek-V4 (O5): DeepGEMM JIT fails on this image (nvcc 13.4 vs pip CCCL headers); use non-DeepGEMM FP8 paths.
# Its hyper-connection layer still JIT-builds a DeepGEMM kernel: the image's default host compiler (gcc 7.5)
# lacks the C++20 that DeepGEMM asks for, so nvcc uses the system g++-13; the pip nvcc is 13.4 while the cudart
# headers torch cu130 ships are 13.0, which CCCL's same-version check rejects, so that check is disabled
# (verified on the login node: the hc-prenorm kernel header compiles; the smoke gate checks the outputs).
if test "$SYSTEM" = O5; then
  export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
  test -x /usr/bin/g++-13 || { echo "PREFLIGHT FAIL: /usr/bin/g++-13 missing" >&2; exit 1; }
  export NVCC_PREPEND_FLAGS="-ccbin=/usr/bin/g++-13 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
fi
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
cd "$PAIR_CODE_ROOT"
"$PYTHON" -B -c 'import shapely, fastfill.comparison.llm_open' || {
  echo "PREFLIGHT FAIL: $PYTHON cannot import shapely + fastfill.comparison from $PAIR_CODE_ROOT;" \
    "install shapely into the vLLM env (system $SYSTEM stays 'not run')" >&2; exit 1; }
# The API-agent smoke normally writes the cohort first; it is deterministic and write-once, so a
# concurrent writer is harmless (llm_open rebuilds and compares it anyway).
test "$3" = t018 || test -e "$COHORT" || "$PYTHON" -B -m fastfill.comparison.llm_agent cohort \
  --data "$PAIR_RUN_ROOT/eval-data-v1" --out "$COHORT" || test -e "$COHORT"
nvidia-smi -L
exec "$PYTHON" -B -u -m fastfill.comparison.llm_open run --system "$SYSTEM" \
  --data "$DATA" --cohort "$COHORT" --model-root "$PROJECT_ROOT/models" \
  --out "$PAIR_RUN_ROOT/$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
