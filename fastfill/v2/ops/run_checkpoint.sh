#!/bin/bash
# Usage: fastfill/v2/ops/run_checkpoint.sh <model-step dir> <tag>; CPU inference (GPUs are training). The autopilot calls it.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HOME=${HF_HOME:-/home/jovyan/shanliantian/.huggingface}
PY=./env/bin/python; CK=$1; OUT=runs/roomgenbench/$2; mkdir -p $OUT; export OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=
[ -d runs/roomgenbench/requests ] || $PY -m fastfill.v2.roomgenbench --requests-from RoomGenBench/bench/inputs/scenes --requests-out runs/roomgenbench/requests
for req in runs/roomgenbench/requests/*.json; do
  key=$(basename $req .json)
  $PY -m fastfill.v2.predict --checkpoint $CK --request $req --output $OUT/$key.prediction.json --export-dir $OUT/$key.handoff --device cpu
  $PY -m fastfill.v2.roomgenbench --handoff $OUT/$key.handoff --output-dir $OUT/$key.layout_boxes --roomgenbench-root RoomGenBench --method layout_boxes --require-placement
done
echo DONE $OUT
