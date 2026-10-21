#!/bin/bash
# LayoutDSL transfer probe, API part (Isambard, CPU, <= 100 gpt-6.1-sol calls). NOT submitted by this workflow.
# Fill REPO / DATA / OUT; uids.json is the deterministic 50-room selection made on the Mac (copy it alongside).
#SBATCH --job-name=probe-layoutdsl --partition=<cpu-partition> --time=03:00:00 --cpus-per-task=4 --mem=16G
set -euo pipefail
REPO=$PROJECTDIR/fastfill/repo            # checkout of OptiScene at tag fastfill-delivery-20261015 or later
DATA=$PROJECTDIR/fastfill/runs/eval-data-v1 # placement/requests.jsonl of the frozen test
OUT=$PROJECTDIR/fastfill/runs/probes-20261021/layoutdsl-transfer
PROBE=$OUT/layoutdsl_probe.py              # this script's sibling, copied with uids.json
export PYTHONPATH=$REPO
python "$PROBE" api --requests "$DATA/placement/requests.jsonl" --uids "$OUT/uids.json" --out "$OUT/api" \
    --env-file ~/.fastfill_api.env --max-calls 100            # hard cap: 50 rooms x 2 arms x 1 round, no repair round
python "$PROBE" report --api "$OUT/api" --uids "$OUT/uids.json" --requests "$DATA/placement/requests.jsonl" --out "$OUT"
