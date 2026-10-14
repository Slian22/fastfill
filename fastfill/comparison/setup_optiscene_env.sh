#!/bin/bash
# Amendment 014 environment, run by the user on the Isambard login node (network needed; no GPU needed):
#   setup_optiscene_env.sh CODE_ROOT
# Before: rsync OptiScene_Original/dataset/*.json to $PROJECT_ROOT/data/optiscene-official/.
# 1. venv $PROJECT_ROOT/envs/optiscene-014 (skipped when it exists): when $PROJECT_ROOT/env is a conda/base env
#    (sys.prefix == sys.base_prefix) a --system-site-packages venv reuses its torch; when it is itself a venv, a plain
#    venv gets the same torch from download.pytorch.org (+ shapely for the write-once publish). Then the pins below;
#    pip installs into the venv only and never uninstalls outside it. pip freeze -> envs/optiscene-014.freeze.txt.
# 2. Qwen/Qwen2.5-7B-Instruct at REV = HfApi().model_info(...).sha -> models/Qwen2.5-7B-Instruct, skipped once
#    $BASE.revision exists (written only after a complete download). REV is saved first in $BASE.revision.pending, so
#    a rerun resumes an interrupted download at the same revision. HF_HOME and the pip cache live under
#    $PROJECT_ROOT/.cache-jobs, never $HOME.
# 3. optiscene_eval env-check (pins, vendored hashes, README DPO argv through the vendored main.py's parser,
#    SFTConfig, data sha256 prefixes, every model file hashed) -> envs/optiscene-014.env.json, written once.
set -euo pipefail
test "$#" -eq 1 || exit 2
CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
V="$PROJECT_ROOT/envs/optiscene-014"
ENV_JSON="$PROJECT_ROOT/envs/optiscene-014.env.json"
DATA="$PROJECT_ROOT/data/optiscene-official"
BASE="$PROJECT_ROOT/models/Qwen2.5-7B-Instruct"
PINS=(trl==0.15.2 transformers==4.49.0 peft==0.14.0 accelerate==1.4.0 datasets==3.3.2 modelscope==1.23.2)
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache-jobs/pip" HF_HOME="$PROJECT_ROOT/.cache-jobs/hf"
export MODELSCOPE_CACHE="$PROJECT_ROOT/.cache-jobs/optiscene-014/modelscope" USE_TF=0 PYTHONDONTWRITEBYTECODE=1
test ! -e "$ENV_JSON" || { echo "refusing: $ENV_JSON exists (env-check is written once)" >&2; exit 1; }
for f in sft_prompts.json dpo_prompts.json dpo2_prompts.json; do
  test -f "$DATA/$f" || { echo "missing $DATA/$f: rsync OptiScene_Original/dataset/*.json first" >&2; exit 1; }
done
if ! test -e "$V"; then
  INFO=$("$PROJECT_ROOT/env/bin/python" -c \
    'import sys, torch; print("base" if sys.prefix == sys.base_prefix else "venv", torch.__version__, torch.version.cuda)')
  read -r KIND TORCH CUDA <<< "$INFO"
  echo "env: $KIND torch $TORCH cuda $CUDA"
  if test "$KIND" = base; then
    "$PROJECT_ROOT/env/bin/python" -m venv --system-site-packages "$V"
  else
    SHAPELY=$("$PROJECT_ROOT/env/bin/python" -c 'import shapely; print(shapely.__version__)')
    "$PROJECT_ROOT/env/bin/python" -m venv "$V"
    "$V/bin/pip" install "torch==${TORCH%%+*}" --index-url "https://download.pytorch.org/whl/cu${CUDA//./}"
    "$V/bin/pip" install "shapely==$SHAPELY"
  fi
  "$V/bin/pip" install "${PINS[@]}"
  "$V/bin/pip" freeze > "$PROJECT_ROOT/envs/optiscene-014.freeze.txt"
fi
"$V/bin/python" -c 'import torch; print("venv torch", torch.__version__, torch.version.cuda)'
if ! test -e "$BASE.revision"; then
  mkdir -p "$(dirname "$BASE")"
  test -s "$BASE.revision.pending" || "$V/bin/python" -c \
    "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen2.5-7B-Instruct').sha)" > "$BASE.revision.pending"
  REV=$(cat "$BASE.revision.pending")
  "$V/bin/huggingface-cli" download Qwen/Qwen2.5-7B-Instruct --revision "$REV" --local-dir "$BASE"
  mv "$BASE.revision.pending" "$BASE.revision"
fi
cd "$CODE_ROOT"
PYTHONPATH="$CODE_ROOT" "$V/bin/python" -B fastfill/comparison/optiscene_eval.py env-check --data "$DATA" \
  --model "$BASE" --revision-file "$BASE.revision" --out "$ENV_JSON"
