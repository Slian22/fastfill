#!/bin/bash
#SBATCH --job-name=unknownsize-dpo-train
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-dpo-train-%j.out
# Amendment 008/009 DPO stage: CODE_ROOT ARM dpo1|dpo2 -> dpo-001/ARM/STAGE (train_dpo) -> dpo-001/ARM/STAGE-merged.
# v1.1 train_dpo under torchrun on 4 GPUs of one node (DDP, per device 1 x 4 GPUs x accumulation 1 = effective
# batch 4; GPUs/CPUs as the v1.1 clean DPO job), then merge_lora on CPU, merge diagnostic and lineage receipt.
# Expected ~2-3 h; resubmitting the same arguments resumes: training_guard refuses changed inputs, train_dpo
# continues from ckpt_latest (every 100 optimizer steps and each epoch end), a finished stage is only re-verified.
set -euo pipefail
test "$#" -eq 3
PAIR_CODE_ROOT="$1"
ARM="$2"
STAGE="$3"
case "$ARM" in joint|placement) ;; *) exit 2 ;; esac
case "$STAGE" in dpo1|dpo2) ;; *) exit 2 ;; esac
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
DPO_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013/dpo-001"
: "${SLURM_JOB_ID:?Slurm allocation required}"
module load cudatoolkit/24.11_12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH="$CUDA_HOME" PYTHONPATH="$PAIR_CODE_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TMPDIR=/tmp PYTHONUNBUFFERED=1
cd "$PAIR_CODE_ROOT"
mkdir -p "$DPO_ROOT/$ARM"
# One writer per stage directory, also against a manual resubmit while the first job still runs.
exec 9>"$DPO_ROOT/$ARM/.$STAGE.lock"
flock -n 9 || { echo "another job owns $DPO_ROOT/$ARM/$STAGE" >&2; exit 1; }
nvidia-smi -L
GPUS=$(nvidia-smi -L | grep -c '^GPU ' || true)
test "$GPUS" -eq 4 || { echo "PREFLIGHT FAIL: need 4 GPUs for the DDP recipe, see $GPUS" >&2; exit 1; }
timeout --signal=TERM --kill-after=120s 705m "$PROJECT_ROOT/env/bin/python" -B -u \
  -m fastfill.comparison.dpo_stages train --arm "$ARM" --stage "$STAGE" --dpo-root "$DPO_ROOT"
