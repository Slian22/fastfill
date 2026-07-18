#!/usr/bin/env bash
# Research v1.5 full-param training. RUN INSIDE tmux (bare nohup loses the
# run to TorchElastic's SIGHUP handling):
#   tmux new -s research_v15_fp
#   bash scripts/train_research_v15.sh
#
# Interrupted/crashed run: RESUME=1 bash scripts/train_research_v15.sh
# (resumes from the last checkpoint in out/research_v15_fp; the provenance
# guard re-validates lineage). If it died BEFORE any checkpoint existed,
# remove the dir yourself and start fresh — deletion stays a human decision.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/essfs10/shanliantian/models/Qwen3-8B}"
test -d "$BASE_MODEL_DIR" \
  || { echo "ERROR: base model dir missing: $BASE_MODEL_DIR (set BASE_MODEL_DIR=...)" >&2; exit 1; }
test -f data/full_research_v15/SNAPSHOT.json \
  || { echo "ERROR: data/full_research_v15 snapshot missing — run rebuild first" >&2; exit 1; }

RESUME_ARGS=()
if [ -e out/research_v15_fp ]; then
  if [ "${RESUME:-0}" != "1" ]; then
    echo "ERROR: out/research_v15_fp already exists — will not overwrite." >&2
    echo "  Crashed run with checkpoints: RESUME=1 bash scripts/train_research_v15.sh" >&2
    echo "  Died before first checkpoint: rm -rf out/research_v15_fp, then rerun." >&2
    exit 1
  fi
  ls -d out/research_v15_fp/checkpoint-* >/dev/null 2>&1 \
    || { echo "ERROR: RESUME=1 but no checkpoint-* in out/research_v15_fp —" >&2; \
         echo "  nothing to resume; rm -rf the dir and start fresh." >&2; exit 1; }
  RESUME_ARGS=(--set resume_from_checkpoint=true)
  echo "RESUME MODE: continuing from last checkpoint in out/research_v15_fp"
fi

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-1,2,3,4,5,6,7}" \
accelerate launch --num_processes "${TRAIN_NPROC:-7}" \
  -m fastfill_train.train_sft --config configs/full_fp.yaml \
  --set "dataset_files=[data/full_research_v15/train.jsonl]" \
  --set output_dir=out/research_v15_fp \
  --set "model_name_or_path=$BASE_MODEL_DIR" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a train_research_v15_fp.log
