#!/bin/bash
#SBATCH --job-name=ff-v11-clean-dpo
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00

# Usage: sbatch --output=LOG this.sbatch CODE_ROOT ORIGINAL_ROOT CLEAN_RUN PAIR_DIR
# Submit at most one afterany continuation; never self-submit without a bound.
set -euo pipefail
CODE_ROOT="${1:?frozen code root required}"
ORIGINAL_ROOT="${2:?original project root required}"
CLEAN_RUN="${3:?separate clean run directory required}"
PAIR_DIR="${4:?audited pair directory required}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT"
module load cudatoolkit
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export PATH="$ORIGINAL_ROOT/env/bin:$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp
PY_TRAIN="$ORIGINAL_ROOT/env/bin/python"
TORCHRUN="$ORIGINAL_ROOT/env/bin/torchrun"
SFT="$ORIGINAL_ROOT/runs/fastfill_v1_1/sft_merged"
"$PY_TRAIN" -m fastfill.v1_1.training_guard isolate --model "$ORIGINAL_ROOT" --out "$CLEAN_RUN"
"$PY_TRAIN" -m fastfill.v1_1.training_guard verify-pairs --out "$PAIR_DIR"
mkdir -p "$CLEAN_RUN"
# Prevent overlapping continuation jobs from writing one checkpoint directory.
exec 9>"$CLEAN_RUN/training.lock"
flock -n 9 || { echo 'Another clean DPO job owns this run'; exit 1; }
echo "Clean DPO | job=${SLURM_JOB_ID:-local} | $(date -Is) | $(hostname)"
nvidia-smi
"$PY_TRAIN" -m fastfill.v1_1.training_guard model-complete --out "$SFT"

train_stage() {
    local base="$1" pairs="$2" adapter="$3" merged="$4" lr="$5"
    "$PY_TRAIN" -m fastfill.v1_1.training_guard bind \
        --model "$base" --pairs "$pairs" --out "$adapter" --lr "$lr" --nproc 4
    if ! "$PY_TRAIN" -m fastfill.v1_1.training_guard adapter-complete --out "$adapter/final"; then
        "$TORCHRUN" --standalone --nproc_per_node=4 -m fastfill.v1_1.train_dpo \
            --model "$base" --pairs "$pairs" --out "$adapter" \
            --beta 0.1 --lr "$lr" --epochs 2 --bs 1 --grad_accum 4 \
            --max_len 8192 --lora_r 32 --lora_alpha 64 --save_steps 150 --seed 42
        "$PY_TRAIN" -m fastfill.v1_1.training_guard adapter-complete --out "$adapter/final"
    fi
    if ! "$PY_TRAIN" -m fastfill.v1_1.training_guard merge-complete \
        --model "$base" --adapter "$adapter/final" --out "$merged"; then
        "$PY_TRAIN" -m fastfill.merge_lora --base_model_path "$base" \
            --lora_path "$adapter/final" --output_path "$merged"
        "$PY_TRAIN" -m fastfill.v1_1.training_guard seal-merge \
            --model "$base" --adapter "$adapter/final" --out "$merged"
    fi
}

train_stage "$SFT" "$PAIR_DIR/stage1.cleaned.jsonl" \
    "$CLEAN_RUN/dpo1_lora" "$CLEAN_RUN/dpo1_merged" 5e-6
train_stage "$CLEAN_RUN/dpo1_merged" "$PAIR_DIR/stage2.cleaned.jsonl" \
    "$CLEAN_RUN/dpo2_lora" "$CLEAN_RUN/final_merged" 2e-6
echo "Clean DPO training and model merges complete | $(date -Is)"
