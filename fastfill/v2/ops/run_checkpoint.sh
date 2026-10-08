#!/bin/bash
# Usage: runs/roomgenbench/run_checkpoint.sh <model-step dir> <tag>; CPU inference (GPUs are training).
# Server copy lives at runs/roomgenbench/run_checkpoint.sh (the autopilot calls it); this is the versioned copy.
set -euo pipefail
cd /home/jovyan/shanliantian/fastfill
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HOME=/home/jovyan/shanliantian/.huggingface
PY=./env/bin/python; CK=$1; OUT=runs/roomgenbench/$2; mkdir -p $OUT; export OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=
[ -d runs/roomgenbench/requests ] || $PY -m fastfill.v2.roomgenbench --requests-from RoomGenBench/bench/inputs/scenes --requests-out runs/roomgenbench/requests
for req in runs/roomgenbench/requests/*.json; do
  key=$(basename $req .json)
  $PY -m fastfill.v2.predict --checkpoint $CK --request $req --output $OUT/$key.prediction.json --export-dir $OUT/$key.handoff --device cpu
  $PY -m fastfill.v2.roomgenbench --handoff $OUT/$key.handoff --output-dir $OUT/$key.layout_boxes --roomgenbench-root RoomGenBench --method layout_boxes --require-placement
done
echo DONE $OUT
