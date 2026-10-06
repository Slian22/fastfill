# FastFill v2：Qwen3-8B 服务器启动

日期：2026-10-06。用户已指定正式骨干为 **Qwen/Qwen3-8B**，服务器通过 SSH alias `yxd-dev` 访问，工作目录为 `/home/jovyan/shanliantian/FastFill_v2_20261006_server`。此前 Qwen2.5-0.5B 只是未执行的可选链路模板，不是本次正式模型。

本次依赖审计要求 `anyio>=4.14.2`、`fsspec>=2026.6.0`；在独立环境安装补丁版本，不修改用户的 base 环境。旧冻结包的 requirements 不包含这两个新增下限，以下安装命令明确补齐。

[OptiScene 论文 §4.1 Training Setup](https://arxiv.org/html/2506.07570v1#S4.SS1) 写明使用 Qwen3-8B 进行 SFT/DPO。[官方 README 训练示例](https://github.com/PolySummit/OptiScene#training-pipeline) 使用 Qwen2.5-7B-Instruct；两处并不完全一致，本次按用户选择固定 Qwen3-8B，不将 README 示例模型当作论文同一配置。

本手册使用冻结 review3 包。包内代码、数据和清单保持不变；独立环境、运行配置与训练输出放在包目录之外，基础模型按用户要求重新下载到 `/home/jovyan/shanliantian/models/Qwen3-8B`。**截至本手册更新，服务器准备正在进行；本文不声明 Qwen3-8B pilot、正式训练或模型效果已经通过。** 当前 v2 修改未 commit/push，不能通过 `git pull` 获取这些未提交实现。

## 1. 登录并初始化用户 Conda

从 Mac 登录后，先运行用户提供的 `myconda`：

```bash
ssh yxd-dev
myconda
cd /home/jovyan/shanliantian
```

自动化 SSH 命令使用 `zsh -lic`，在该 shell 内先运行 `myconda`。每个新 SSH 进程都需要重新初始化和激活环境；上一个连接的环境变量不会传到下一个连接。

```bash
ssh yxd-dev 'zsh -lic "
myconda
cd /home/jovyan/shanliantian
python -V
"'
```

随后在服务器 shell 中设置本次工作路径：

```bash
export FASTFILL_WORKDIR=/home/jovyan/shanliantian/FastFill_v2_20261006_server
export FASTFILL_BUNDLE="$FASTFILL_WORKDIR/FastFill_v2_20261006_review3"
export FASTFILL_ENV="$FASTFILL_WORKDIR/env"
export FASTFILL_BACKBONE=/home/jovyan/shanliantian/models/Qwen3-8B
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$FASTFILL_WORKDIR/outputs" /home/jovyan/shanliantian/models
```

用户提供 8×H20Z，每卡 143,771 MiB，driver 570.124.06；主机 RAM 2.8 TiB。用户允许使用物理 GPU 1–7，pilot 先用 GPU 1，保留已有任务。用户报告 `/home/jovyan` 约 1.8 TiB 可用；运行时仍需检查当前空间、显存和占用：

```bash
nvidia-smi
free -h
df -h "$FASTFILL_WORKDIR"
```

## 2. 上传、解包并校验冻结包

Mac 上的新包位于仓库 outputs；原始外盘目录不需要重复上传。以下命令在 Mac 执行：

```bash
scp /Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/FastFill_v2_20261006_review3/upload/FastFill_v2_training_20261006_review3.tar.gz \
    /Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/FastFill_v2_20261006_review3/upload/FastFill_v2_training_20261006_review3.tar.gz.sha256 \
    yxd-dev:/home/jovyan/shanliantian/FastFill_v2_20261006_server/
```

服务器执行以下命令。已解包时先运行校验，避免重复解包覆盖已存在内容：

```bash
cd "$FASTFILL_WORKDIR"
sha256sum -c FastFill_v2_training_20261006_review3.tar.gz.sha256
if [ ! -e "$FASTFILL_BUNDLE" ]; then
  tar -xzf FastFill_v2_training_20261006_review3.tar.gz
fi
cd "$FASTFILL_BUNDLE"
sha256sum -c SHA256SUMS
python3 verify_bundle.py
```

review3 压缩包 SHA256 为 `973ef1743a62bdce94150643d1df2049934edfe8a1bef5c1947e5066b07fb04c`。它包含主数据、完整标签子集、代码、测试及审核记录，不含基础 Qwen 权重或正式训练模型。

## 3. 独立环境、CUDA 检查和 Qwen3-8B 权重

环境使用用户 Conda 的独立 prefix，不修改已有训练环境。如果该 prefix 已建立，直接激活并核对依赖；仅在缺失时创建或安装：

```bash
conda create --prefix "$FASTFILL_ENV" python=3.12 -y
conda activate "$FASTFILL_ENV"
cd "$FASTFILL_BUNDLE/project"
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r fastfill/v2/requirements.txt
python -m pip install transformers==5.14.1 peft==0.21.0 accelerate==1.15.0
python -m pip install 'anyio>=4.14.2' 'fsspec>=2026.6.0'
python -m pip install pytest pytest-cov huggingface_hub
python -m pip freeze > "$FASTFILL_WORKDIR/outputs/environment-server.txt"
```

该 CUDA 12.6 PyTorch build 与用户 570.124.06 驱动相容；不需要更换系统驱动或安装系统 CUDA Toolkit。`nvidia-smi` 的 12.8 表示驱动支持信息，`torch.version.cuda` 可以为 12.6。参见 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/) 和 [CUDA 12.6 驱动要求](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html)。

先验证选中的单卡及 BF16：

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import torch
from accelerate import Accelerator
assert torch.cuda.is_available(), 'CUDA PyTorch 不可用'
assert torch.cuda.is_bf16_supported(), '选中设备不支持 BF16'
assert Accelerator(cpu=False, mixed_precision='bf16').device.type == 'cuda'
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert (x @ x)[0, 0].item() == 16
print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
       'visible_gpu_count': torch.cuda.device_count(),
       'physical_gpu_1': torch.cuda.get_device_name(0)})
PY
```

按用户要求重新下载完整基础模型到指定 models 目录，不引用其他项目的缓存作为本次模型路径。使用官方固定 revision：

```bash
hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir "$FASTFILL_BACKBONE"
```

参见 [Qwen3-8B 官方模型页](https://huggingface.co/Qwen/Qwen3-8B) 和 [HF CLI 文档](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。这是基础权重下载，不是下载我们已训练的模型。无需为本入口安装 MinkowskiEngine、V-DETR 或 vLLM。

## 4. 离线测试和真实 tokenizer 预检

完整离线套件使用 CPU/Gloo，显式隐藏 GPU；它不证明 CUDA/NCCL/BF16 大模型路径通过。Metal 测试在 Linux 上正常跳过。测试及 tiny smoke 的工件放在本次工作目录 outputs：

```bash
cd "$FASTFILL_BUNDLE/project"
CUDA_VISIBLE_DEVICES='' COVERAGE_FILE="$FASTFILL_WORKDIR/outputs/.coverage-server" \
  python -m pytest fastfill/tests fastfill/v2/tests \
  -o "cache_dir=$FASTFILL_WORKDIR/outputs/pytest-cache-server" \
  --cov=fastfill.v2 --cov-config=fastfill/v2/.coveragerc --cov-report=term-missing -q

CUDA_VISIBLE_DEVICES='' python -m fastfill.v2.train \
  --config fastfill/v2/configs/smoke.json \
  --data "$FASTFILL_BUNDLE/data/complete-label-cohort/train.jsonl" \
  --output "$FASTFILL_WORKDIR/outputs/offline-server-smoke" \
  --dry-run --max-samples 2
```

创建新的 Qwen3-8B pilot 配置；不修改冻结包里的默认配置。每次实验使用新的配置与输出路径：

```bash
export FASTFILL_PILOT_CONFIG="$FASTFILL_WORKDIR/outputs/qwen3-8b-pilot.json"
python - <<'PY'
import json, os
from pathlib import Path
cfg = json.loads(Path('fastfill/v2/configs/structured.json').read_text())
cfg = {**cfg,
       'model': {**cfg['model'], 'backbone': os.environ['FASTFILL_BACKBONE'],
                 'backbone_dtype': 'bfloat16', 'local_files_only': True},
       'training': {**cfg['training'], 'cpu': False, 'steps': 20, 'batch_size': 1,
                    'max_length': 4096, 'mixed_precision': 'bf16',
                    'gradient_accumulation_steps': 1,
                    'checkpoint_every': 10, 'validate_every': 5}}
with Path(os.environ['FASTFILL_PILOT_CONFIG']).open('x') as stream:
    stream.write(json.dumps(cfg, indent=2) + '\n')
PY
```

其余采用可执行的开发起点：decoder 128、4 heads、2 layers，yaw 12 bins，LoRA rank 8/alpha 16，learning rate 1e-4，四项几何 loss 权重均为 1，box/collision/boundary 权重为 0。这些不是已调优的正式超参数。

使用真实 Qwen3 tokenizer 和实际启用 objective 检查两个训练场景、全部两个验证场景：

```bash
python - <<'PY'
import json, os
from pathlib import Path
from fastfill.v2.batch import load_tokenizer, tokenize_condition
from fastfill.v2.io import read_samples
from fastfill.v2.losses import LossConfig
from fastfill.v2.model import ModelConfig
from fastfill.v2.train import _preflight, _training_config
cfg = json.loads(Path(os.environ['FASTFILL_PILOT_CONFIG']).read_text())
tok = load_tokenizer(cfg['model']['backbone'], local_files_only=True)
for split, limit in [('train', 2), ('validation', None)]:
    rows = read_samples(Path(os.environ['FASTFILL_BUNDLE']) / 'data/complete-label-cohort' / f'{split}.jsonl',
                        training=split == 'train', max_samples=limit)
    kept, rejected = _preflight(rows, tok, ModelConfig(**cfg['model']),
                                _training_config(cfg['training']), LossConfig(**cfg['loss']))
    print(split, {'kept': len(kept), 'rejected': rejected,
                  'condition_tokens': [len(tokenize_condition(r['condition'], tok)[0]) for r in rows]})
    assert not rejected, 'pilot 集合不完整；显式调整预算后重新预检，不能截断对象'
PY
```

结构化模型读取确定性 condition 文本、提取 hidden states，使用外部连续 heads；它不调用 Qwen 的文本采样或 thinking 生成。文本 SFT 是独立实验路径，不是这次 pilot 的前置。

## 5. 单卡 Qwen3-8B 的 20 更新步 pilot

只有以上检查通过后执行；不加 `--dry-run`。该参数只接受显式 tiny 骨干。

```bash
CUDA_VISIBLE_DEVICES=1 python -m fastfill.v2.train \
  --config "$FASTFILL_PILOT_CONFIG" \
  --data "$FASTFILL_BUNDLE/data/complete-label-cohort/train.jsonl" \
  --validation "$FASTFILL_BUNDLE/data/complete-label-cohort/validation.jsonl" \
  --output "$FASTFILL_WORKDIR/outputs/qwen3-8b-pilot-20steps" \
  --max-samples 2

CUDA_VISIBLE_DEVICES=1 python -m fastfill.v2.evaluate \
  --checkpoint "$FASTFILL_WORKDIR/outputs/qwen3-8b-pilot-20steps/model" \
  --data "$FASTFILL_BUNDLE/data/complete-label-cohort/test.jsonl" \
  --output "$FASTFILL_WORKDIR/outputs/qwen3-8b-pilot-eval" \
  --device cuda --max-length 4096
```

进程内 `cuda:0` 对应物理 GPU 1。训练以 BF16 载入基础骨干、使用 BF16 autocast；连续尺寸指数、pooling 累加等保留代码规定的 float32 数值保护。20 步用于检查真实 8B 权重、CUDA、LoRA、四项梯度、validation、保存和重新载入，不作为布局质量证据。

LoRA 产物不包含完整基础模型；评测和部署仍需同一固定 revision 的 Qwen3-8B，并保留训练配置中的基础路径。当前没有 CLI 自动 resume，不将已保存 `state-step-*` 误称为支持自动恢复的正式训练流程。

全库正式训练还需冻结 batch/context、对象资格、稀有 yaw 采样、地板诊断策略、训练曝光预算和实测显存。训练集可信 yaw 仅 523 个对象，不能把默认均匀 1,000 步配置直接视作充分的正式方案。真实资产、WorldEdge Host、mesh/physics/Solver 和 GLB 展示另行验收。

完整实验与产物说明见 [执行手册](fastfill-v2-runbook.md) 和 [pipeline](fastfill-v2-training-pipeline.md)。
