#!/bin/bash
# Two OptiScene-style LLM agent modes on the frozen 300 rows, then the same scorer as every other method.
set -euo pipefail
cd /home/jovyan/shanliantian/fastfill
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
ROWS=runs/llm-prompt-300/rows.jsonl
LLM_ENV=/home/jovyan/shanliantian/.fastfill_api.env
pids=()
for mode in structured structured-harness; do
  ( ./env/bin/python -m fastfill.v2.llm_structured --data "$ROWS" --env "$LLM_ENV" --mode "$mode" --repairs 2 --workers 4 --max-samples 300 \
      --output "runs/llm-$mode-300" > "runs/llm-$mode-300.log" 2>&1
    if ! cmp -s "runs/llm-$mode-300/rows.jsonl" "$ROWS"; then
      echo "$mode ROWS DIFFER" >&2
      exit 1
    fi
    echo "$mode rows identical"
    ./env/bin/python -m fastfill.v2.evaluate --data "runs/llm-$mode-300/rows.jsonl" --predictions "runs/llm-$mode-300/predictions.jsonl" \
      --output "runs/llm-$mode-300-eval" > "runs/llm-$mode-300-eval.log" 2>&1
    echo "$mode scored" ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if (( status != 0 )); then
  exit "$status"
fi
echo ALL-DONE
