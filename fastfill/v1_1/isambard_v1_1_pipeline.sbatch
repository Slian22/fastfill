#!/bin/bash
#SBATCH --job-name=fastfill-v1.1
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/fastfill_v1_1/slurm-%x-%j.out

set -euo pipefail

ROOT="/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill"
cd "$ROOT"
module load cudatoolkit
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export PATH="/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export TMPDIR="/tmp"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER:-u6uv}"
export TRITON_CACHE_DIR="/tmp/triton_${USER:-u6uv}"
export VLLM_CACHE_ROOT="/tmp/vllm_${USER:-u6uv}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

PY_TRAIN="/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/env/bin/python"
TORCHRUN="/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/env/bin/torchrun"
PY_VLLM="/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python"

BASE_MODEL="$ROOT/models/OptiScene-Qwen-8B-Final"
V32_DATA="$ROOT/data/v3.2"
V11_DATA="$ROOT/data/v1.1"
OUT_ROOT="$ROOT/runs/fastfill_v1_1"
mkdir -p "$OUT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
SFT_MAX_LEN="${SFT_MAX_LEN:-8192}"
SFT_BS="${SFT_BS:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
DPO_BS="${DPO_BS:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-4}"
TP="${TP:-1}"

echo "============================================================"
echo "FastFill v1.1 Automated Pipeline on Isambard-AI (${NPROC}x GH200, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | Date: $(date -Is)"
echo "============================================================"
nvidia-smi

# Ensure shapely, scipy, and trimesh are available in vllm env for stage1_vllm & stage4
"$PY_VLLM" -c "import shapely, scipy, trimesh" 2>/dev/null || "$PY_VLLM" -m pip install --quiet shapely scipy trimesh

# Fetch warm-start FastFill v1.0 Final model (liantian/OptiScene-Qwen-8B-Final) if not cached locally
if [ ! -f "$BASE_MODEL/model.safetensors-00004-of-00004.safetensors" ] && [ ! -f "$BASE_MODEL/model.safetensors.index.json" ]; then
    echo "[Stage 0] Downloading warm-start model liantian/OptiScene-Qwen-8B-Final -> $BASE_MODEL ..."
    "$PY_TRAIN" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='liantian/OptiScene-Qwen-8B-Final', local_dir='$BASE_MODEL', max_workers=2)
print('Downloaded liantian/OptiScene-Qwen-8B-Final to $BASE_MODEL')
"
fi

# ------------------------------------------------------------------
# STAGE 0: Build FastFill v1.1 Dataset + Full Collision Flagging
# ------------------------------------------------------------------
if [ ! -f "$V32_DATA/train.jsonl" ]; then
    echo "[Stage 0a] Fetching liantian/fastfill-v3 (v3.2/*) to $V32_DATA ..."
    "$PY_TRAIN" -c "
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
p = Path(snapshot_download(repo_id='liantian/fastfill-v3', repo_type='dataset', allow_patterns=['v3.2/*'])) / 'v3.2'
dst = Path('$V32_DATA')
dst.mkdir(parents=True, exist_ok=True)
for f in p.iterdir():
    if f.is_file():
        shutil.copy2(f, dst / f.name)
print('Fetched v3.2 files:', [x.name for x in sorted(dst.iterdir())])
"
fi

if [ ! -f "$V11_DATA/MANIFEST.json" ]; then
    echo "[Stage 0b] Building FastFill v1.1 dataset from $V32_DATA -> $V11_DATA ..."
    "$PY_TRAIN" -m fastfill.v1_1.build_v1_1 \
        --src "$V32_DATA" \
        --out "$V11_DATA" \
        --workers 64
else
    echo "[Stage 0b] $V11_DATA/MANIFEST.json already exists, skipping build."
fi

# Self-checks
"$PY_TRAIN" -m fastfill.v1_1.scene_v1_1
"$PY_TRAIN" -m fastfill.v1_1.postprocess
"$PY_TRAIN" -m fastfill.v1_1.evaluate_v1_1

# ------------------------------------------------------------------
# STAGE 1: DDP LoRA SFT on 100% Collision-Free Ground Truth
# ------------------------------------------------------------------
SFT_DIR="$OUT_ROOT/sft_lora"
SFT_MERGED="$OUT_ROOT/sft_merged"

if [ ! -s "$SFT_DIR/final/adapter_config.json" ] && [ ! -s "$SFT_MERGED/tokenizer_config.json" ]; then
    echo "[Stage 1a] Running ${NPROC}-GPU DDP LoRA SFT on collision-free v1.1 train set..."
    for ckpt in "$SFT_DIR"/checkpoint-*; do
        if [ -d "$ckpt" ] && ( [ ! -s "$ckpt/trainer_state.json" ] || [ ! -s "$ckpt/adapter_model.safetensors" ] ); then
            echo "[Stage 1a] Removing incomplete checkpoint directory: $ckpt"
            rm -rf "$ckpt"
        fi
    done
    RESUME_FLAG=""
    if ls "$SFT_DIR"/checkpoint-* >/dev/null 2>&1; then
        RESUME_FLAG="--resume"
    fi
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m fastfill.train \
        --model "$BASE_MODEL" \
        --data "$V11_DATA" \
        --out "$SFT_DIR" \
        --max_len "$SFT_MAX_LEN" \
        --lora_r 32 \
        --lora_alpha 64 \
        --bs "$SFT_BS" \
        --grad_accum "$SFT_GRAD_ACCUM" \
        --lr 5e-5 \
        --epochs "$SFT_EPOCHS" \
        --save_steps 500 \
        --eval_max_len 8192 \
        --exclude_flags oob_objects overlapping_furniture fixed_collision severe_collision main_collision \
        $RESUME_FLAG
fi

if [ ! -s "$SFT_MERGED/tokenizer_config.json" ] || ( [ ! -s "$SFT_MERGED/model.safetensors" ] && [ ! -s "$SFT_MERGED/model.safetensors.index.json" ] ); then
    echo "[Stage 1b] Merging SFT LoRA -> $SFT_MERGED ..."
    "$PY_TRAIN" -m fastfill.merge_lora \
        --base_model_path "$BASE_MODEL" \
        --lora_path "$SFT_DIR/final" \
        --output_path "$SFT_MERGED"
else
    echo "[Stage 1] $SFT_MERGED already complete, skipping SFT."
fi

# ------------------------------------------------------------------
# STAGE 2: DPO Stage 1 (On-Policy vLLM Collision/OOB Rejection)
# ------------------------------------------------------------------
DPO1_PAIRS="$OUT_ROOT/dpo_stage1_pairs.jsonl"
DPO1_DIR="$OUT_ROOT/dpo1_lora"
DPO1_MERGED="$OUT_ROOT/dpo1_merged"

if [ ! -s "$DPO1_PAIRS" ]; then
    echo "[Stage 2a] Sampling on-policy candidates with vLLM for DPO Stage 1..."
    "$PY_VLLM" -m fastfill.v1_1.build_dpo \
        --mode stage1_vllm \
        --model "$SFT_MERGED" \
        --data "$V11_DATA" \
        --out "$DPO1_PAIRS" \
        --max_pairs 8000 \
        --n_samples 4 \
        --max_len 16384 \
        --tp "$TP"
fi

if [ ! -s "$DPO1_DIR/final/adapter_config.json" ] && [ ! -s "$DPO1_MERGED/tokenizer_config.json" ]; then
    echo "[Stage 2b] Training ${NPROC}-GPU DPO Stage 1..."
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m fastfill.v1_1.train_dpo \
        --model "$SFT_MERGED" \
        --pairs "$DPO1_PAIRS" \
        --out "$DPO1_DIR" \
        --beta 0.1 \
        --lr 5e-6 \
        --epochs 2 \
        --bs "$DPO_BS" \
        --grad_accum "$DPO_GRAD_ACCUM" \
        --max_len 8192 \
        --lora_r 32 \
        --lora_alpha 64 \
        --save_steps 150
fi

if [ ! -s "$DPO1_MERGED/tokenizer_config.json" ] || ( [ ! -s "$DPO1_MERGED/model.safetensors" ] && [ ! -s "$DPO1_MERGED/model.safetensors.index.json" ] ); then
    echo "[Stage 2c] Merging DPO Stage 1 LoRA -> $DPO1_MERGED ..."
    "$PY_TRAIN" -m fastfill.merge_lora \
        --base_model_path "$SFT_MERGED" \
        --lora_path "$DPO1_DIR/final" \
        --output_path "$DPO1_MERGED"
else
    echo "[Stage 2] $DPO1_MERGED already complete, skipping DPO Stage 1."
fi

# ------------------------------------------------------------------
# STAGE 3: DPO Stage 2 (Hard-Negative Geometric Perturbation DPO)
# ------------------------------------------------------------------
DPO2_PAIRS="$OUT_ROOT/dpo_stage2_pairs.jsonl"
DPO2_DIR="$OUT_ROOT/dpo2_lora"
FINAL_MERGED="$OUT_ROOT/final_merged"

if [ ! -s "$DPO2_PAIRS" ]; then
    echo "[Stage 3a] Building geometric hard-negative pairs for DPO Stage 2..."
    "$PY_TRAIN" -m fastfill.v1_1.build_dpo \
        --mode stage2_hardneg \
        --data "$V11_DATA" \
        --out "$DPO2_PAIRS" \
        --max_pairs 12000
fi

if [ ! -s "$DPO2_DIR/final/adapter_config.json" ] && [ ! -s "$FINAL_MERGED/tokenizer_config.json" ]; then
    echo "[Stage 3b] Training ${NPROC}-GPU DPO Stage 2..."
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m fastfill.v1_1.train_dpo \
        --model "$DPO1_MERGED" \
        --pairs "$DPO2_PAIRS" \
        --out "$DPO2_DIR" \
        --beta 0.1 \
        --lr 2e-6 \
        --epochs 2 \
        --bs "$DPO_BS" \
        --grad_accum "$DPO_GRAD_ACCUM" \
        --max_len 8192 \
        --lora_r 32 \
        --lora_alpha 64 \
        --save_steps 150
fi

if [ ! -s "$FINAL_MERGED/tokenizer_config.json" ] || ( [ ! -s "$FINAL_MERGED/model.safetensors" ] && [ ! -s "$FINAL_MERGED/model.safetensors.index.json" ] ); then
    echo "[Stage 3c] Merging DPO Stage 2 LoRA -> $FINAL_MERGED ..."
    "$PY_TRAIN" -m fastfill.merge_lora \
        --base_model_path "$DPO1_MERGED" \
        --lora_path "$DPO2_DIR/final" \
        --output_path "$FINAL_MERGED"
else
    echo "[Stage 3] $FINAL_MERGED already complete, skipping DPO Stage 2."
fi

# ------------------------------------------------------------------
# STAGE 4: Full Evaluation (vLLM) + RoomGenBench 5-Scene Export
# ------------------------------------------------------------------
echo "[Stage 4a] Evaluating SFT, DPO1, and Final models on test_rooms & test_constrained_rooms..."
for TAG_AND_MODEL in "sft:$SFT_MERGED" "dpo1:$DPO1_MERGED" "final:$FINAL_MERGED"; do
    TAG="${TAG_AND_MODEL%%:*}"
    MDL="${TAG_AND_MODEL#*:}"
    if [ ! -s "$OUT_ROOT/eval_${TAG}_test/metrics.json" ]; then
        "$PY_VLLM" -m fastfill.v1_1.evaluate_v1_1 \
            --model "$MDL" \
            --rooms "$V11_DATA/test_rooms.jsonl" \
            --out "$OUT_ROOT/eval_${TAG}_test" \
            --use-vllm \
            --tp "$TP"
    fi
    if [ ! -s "$OUT_ROOT/eval_${TAG}_test_constrained/metrics.json" ]; then
        "$PY_VLLM" -m fastfill.v1_1.evaluate_v1_1 \
            --model "$MDL" \
            --rooms "$V11_DATA/test_constrained_rooms.jsonl" \
            --out "$OUT_ROOT/eval_${TAG}_test_constrained" \
            --use-vllm \
            --tp "$TP"
    fi
done

if [ ! -s "$OUT_ROOT/roomgenbench_v1_1/summary.json" ]; then
    echo "[Stage 4b] Exporting RoomGenBench 5 benchmark scenes with doors/windows + zero-collision repair..."
    "$PY_VLLM" -m fastfill.v1_1.roomgenbench_v1_1 \
        --scenes-dir "$ROOT/RoomGenBench/bench/inputs/scenes" \
        --model "$FINAL_MERGED" \
        --out-dir "$OUT_ROOT/roomgenbench_v1_1" \
        --use-vllm \
        --tp "$TP"
fi

echo "============================================================"
echo "FastFill v1.1 Pipeline Complete! Output: $OUT_ROOT"
echo "============================================================"
