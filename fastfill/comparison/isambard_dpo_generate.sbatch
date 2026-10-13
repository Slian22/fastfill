#!/bin/bash
#SBATCH --job-name=unknownsize-dpo-generate
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-dpo-generate-%j.out
# Amendment 008 DPO-stage1 sampling: CODE_ROOT ARM -> dpo-001/ARM/stage1-generations (vLLM, merged SFT model).
# Resubmitting the same arguments resumes from the write-once per-UID receipts under the same run identity.
set -euo pipefail
test "$#" -eq 2
PAIR_CODE_ROOT="$1"
ARM="$2"
case "$ARM" in joint|placement) ;; *) exit 2 ;; esac
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
DPO_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013/dpo-001"
PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
CACHE=/scratch/u6uv/shanliantian.u6uv/.cache
: "${SLURM_JOB_ID:?Slurm allocation required}"
test -f "$DPO_ROOT/$ARM/sft-merged.json"
# As isambard_llm_open.sbatch: vLLM JIT kernels need the cluster nvcc; caches live on scratch, not $HOME.
module load cudatoolkit || { echo "PREFLIGHT FAIL: module load cudatoolkit" >&2; exit 1; }
NVCC=$(command -v nvcc) || { echo "PREFLIGHT FAIL: no nvcc after 'module load cudatoolkit'" >&2; exit 1; }
CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$CACHE/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# Per-arm compile caches: both arms' generate jobs can run at once (as isambard_llm_open.sbatch per system).
export XDG_CACHE_HOME="$CACHE" VLLM_CACHE_ROOT="$CACHE/dpo-generate/$ARM/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/dpo-generate/$ARM/torchinductor" TRITON_CACHE_DIR="$CACHE/dpo-generate/$ARM/triton"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$PAIR_CODE_ROOT"
"$PYTHON" -B -c 'import shapely, vllm, fastfill.comparison.dpo_stages' || {
  echo "PREFLIGHT FAIL: $PYTHON cannot import shapely + vllm + fastfill.comparison.dpo_stages" >&2; exit 1; }
nvidia-smi -L
exec timeout --signal=TERM --kill-after=60s 350m "$PYTHON" -B -u \
  -m fastfill.comparison.dpo_stages generate --arm "$ARM" --dpo-root "$DPO_ROOT"
