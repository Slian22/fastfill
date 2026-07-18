#!/usr/bin/env bash
# Research v1.5 full-param training. RUN INSIDE tmux (bare nohup loses the
# run to TorchElastic's SIGHUP handling):
#   tmux new -s research_v15_fp
#   bash scripts/train_research_v15.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/essfs10/shanliantian/models/Qwen3-8B}"
test -d "$BASE_MODEL_DIR" \
  || { echo "ERROR: base model dir missing: $BASE_MODEL_DIR (set BASE_MODEL_DIR=...)" >&2; exit 1; }
test -f data/full_research_v15/SNAPSHOT.json \
  || { echo "ERROR: data/full_research_v15 snapshot missing — run rebuild first" >&2; exit 1; }
test ! -e out/research_v15_fp \
  || { echo "ERROR: out/research_v15_fp already exists — will not overwrite" >&2; exit 1; }

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-1,2,3,4,5,6,7}" \
accelerate launch --num_processes "${TRAIN_NPROC:-7}" \
  -m fastfill_train.train_sft --config configs/full_fp.yaml \
  --set "dataset_files=[data/full_research_v15/train.jsonl]" \
  --set output_dir=out/research_v15_fp \
  --set "model_name_or_path=$BASE_MODEL_DIR" \
  2>&1 | tee train_research_v15_fp.log
