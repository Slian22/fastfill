#!/bin/bash
# One-time Isambard-AI setup on a compute node (the login nodes have no GPU): aarch64 env, pinned data, backbone and
# LLM comparison rows, RoomGenBench requests, then the test suite. Rerunning skips what is already in place.
#SBATCH --job-name=ff-setup
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
uname -m; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
git submodule update --init RoomGenBench
[ -x env/bin/python ] || ~/miniforge3/bin/conda create -y -p "$PWD/env" python=3.12 pip
env/bin/pip install 'torch==2.13.0+cu126' --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
# the versions the baseline was trained with (its run manifest)
env/bin/pip install -r fastfill/v2/requirements.txt -r fastfill/v2/requirements-roomgenbench.txt \
  transformers==5.14.1 peft==0.21.0 accelerate==1.15.0 scipy==1.15.2 pytest
env/bin/pip freeze > runs/isambard/env-freeze.txt
env/bin/python -c "import platform, torch; assert torch.cuda.is_available(); print(platform.machine(), torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list(), torch.cuda.is_bf16_supported())"

H=env/bin/hf
# the baseline's backbone: same config.json fingerprint as its run manifest
[ -f models/Qwen3-8B/model.safetensors.index.json ] || $H download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/Qwen3-8B
echo "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30  models/Qwen3-8B/config.json" | sha256sum -c
D=data/main-20261008a  # NEXT_DATA.json of the round-10 rebuild (commit 155a123)
if [ ! -d $D ]; then
  $H download liantian/fastfill-v2 --repo-type dataset --revision a6a292dec3e5026a9560a6d9ade37c666d798bc8 \
    --include 'rebuild-main-20261008a/main/*' --local-dir data/.download-main-20261008a
  mv data/.download-main-20261008a/rebuild-main-20261008a/main $D
fi
(cd $D && sha256sum -c) <<EOF
5d9439f63b32bdfbf8200b52ef1c2681435053a0dc8b88a84234f24f0d8160eb  train.jsonl
93cca6cb1abe80ac6eaa5713143d0b6b44774fa0d874a7c37842f7d514f10a39  validation.jsonl
212fc6bfa2302173757a110a51a5f3ac34ae66111bd5d96860de6997ab82f107  test.jsonl
EOF
# the 300 rows the LLM agents answered: phase F scores the trained model on exactly these
R=runs/llm-prompt-300/rows.jsonl
if [ ! -f $R ]; then
  $H download liantian/fastfill-v2-models llm-comparison/rows-300.jsonl --local-dir runs/.download-llm
  mkdir -p runs/llm-prompt-300 && mv runs/.download-llm/llm-comparison/rows-300.jsonl $R
fi
echo "e02da65e6844630eba506952ee1f059655552a2d14e1af763eb787cc2d0152ea  $R" | sha256sum -c
# made once here, so two autopilots exporting at the same time do not race on it
[ -d runs/roomgenbench/requests ] || env/bin/python -m fastfill.v2.roomgenbench \
  --requests-from RoomGenBench/bench/inputs/scenes --requests-out runs/roomgenbench/requests
env/bin/python -m pytest fastfill/v2/tests -q -x
echo SETUP OK
