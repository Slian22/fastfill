#!/bin/bash
#SBATCH --job-name=unknownsize-jp-pilot
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-pilot-%j.out
set -euo pipefail
test "$#" -eq 3
PAIR_CODE_ROOT="$1"
PAIR_DATA_ROOT="$2"
PAIR_OUTPUT_ROOT="$3"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
: "${SLURM_JOB_ID:?Slurm allocation required}"
test -f "$PAIR_CODE_ROOT/fastfill/comparison/train_pair.py"
test -f "$PAIR_DATA_ROOT/MANIFEST.json"
test ! -e "$PAIR_OUTPUT_ROOT"
module load cudatoolkit/24.11_12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH="$CUDA_HOME"
export PATH="$PROJECT_ROOT/env/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PAIR_CODE_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/tmp PYTHONUNBUFFERED=1
cd "$PAIR_CODE_ROOT"
exec timeout --signal=TERM --kill-after=30s 27m \
  "$PROJECT_ROOT/env/bin/python" -B -u -m fastfill.comparison.train_pair \
  --data "$PAIR_DATA_ROOT" --base "$PROJECT_ROOT/models/Qwen3-8B" \
  --out "$PAIR_OUTPUT_ROOT" --max-steps 8 --max-len 8192 --bs 1 --grad-accum 4
