#!/usr/bin/env bash
# Serve a merged FastFill planner as an OpenAI-compatible endpoint.
#
# Usage:
#   PORT=8901 bash scripts/serve_vllm.sh out/sft_full_merged [extra vllm args...]
#
# The scenesmith side (growing_world fastfill generator) consumes this
# endpoint via environment variables:
#   export FASTFILL_LLM_BASE_URL=http://<host>:8901/v1
#   export FASTFILL_LLM_API_KEY=dummy           # vLLM accepts any key
#   export FASTFILL_LLM_MODEL=fastfill-planner  # matches --served-model-name
#
# The FastFill codec is LINE-BASED PLAIN TEXT (category|w,d,h|x,y|yaw|flags),
# not JSON — no guided-JSON / structured-output flags are needed; plain chat
# completions are sufficient.
set -euo pipefail

MODEL_DIR="${1:?usage: serve_vllm.sh <merged_model_dir> [extra vllm args...]}"
shift

exec vllm serve "$MODEL_DIR" \
  --port "${PORT:-8901}" \
  --served-model-name fastfill-planner \
  "$@"
