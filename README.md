# fastfill_train — FastFill 布局模型训练工作区

独立于 scenesmith 的训练仓:在训练服务器上把 FastFill 布局规划器(调用 1 Floor + 调用 2 Surface,
紧凑 codec 文本输出)在 Qwen3-8B(对照底座)与 Qwen3.5-9B(候选,A/B 三项赢两项才切换)上,经 Stage-0 三臂(LoRA r32 / r128 / 全参)决出训练方式后训练;Qwen3.6-27B 仅作探顶与 RSFT 采样器。产出 merge 后的权重目录,由 vLLM 以
OpenAI 兼容端点服务,scenesmith 侧 fastfill 后端零改动接入(改环境变量即可)。

```
数据集(服务器) ──vendor/tools converters──▶ FastFillSample JSONL ──dedup/export──▶ SFT JSONL
        ──train_sft(smoke→full, 模板A/B/C)──▶ LoRA ──merge──▶ merged/
        ──build_dpo_data(stage1 自负例 / stage2 注入负例)──▶ train_dpo ×2 ──merge──▶ final/
        ──scripts/serve_vllm.sh──▶ http://host:8901/v1 ──eval_layout 验收──▶ 权重下载给 scenesmith
```

## 与两个上游的关系

- **scenesmith 仓 = 契约的唯一事实源**。`vendor/` 是从 `../scenesmith` 同步的只读副本
  (schema/codec/validator/repair + tools/fastfill_data 转换器)。改契约先改 scenesmith,
  再跑 `scripts/sync_vendor.sh`。`vendor/VENDOR_VERSION` 记录同步时的 commit。
- **OptiScene 仓 = 训练脚手架参照 + baseline**,不是依赖。TRL SFT/DPO/PEFT/merge 的接线方式
  移植自它;数据格式、prompt、split、负例全部换成 FastFill 自己的。
  **注意官方 README 超参 ≠ 论文配置**(审码结论):README 用 Qwen2.5-7B、SFT epochs=1、
  main.py 默认 lr 5e-5、DPO lr 5e-6/10 epochs/LoRA r32 α16;论文是 Qwen3-8B、SFT lr 5e-6/
  10 epochs/r16 α32 dropout 0.05、DPO lr 5e-7/5 epochs。`configs/` 全部按论文;README 那套
  只当"能跑示例"。另:官方 loader 是随机 80/20 行级 split(有泄漏风险),我们用
  split_key(house-first)哈希切分,不复制它。

## 服务器上手

```bash
# 1) 环境(CUDA 机)
python3 -m venv .venv && source .venv/bin/activate
pip install vllm                          # 先装:让 vllm 钉死它兼容的 torch
pip install -r requirements.txt           # 后装其余;装完前后各记 torch/cuda/vllm 版本+pip check
python3 -c "import torch,vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"
pip check                                 # 安装前后各跑一次,记录进 smoke 报告
export PYTHONPATH=src                     # 包内 bootstrap 会自动接上 vendor/

# 2) 数据:把原始数据集放到 ./data(或 export WORLDEDGE_DATA_DIR=...),然后
python3 vendor/tools/fastfill_data/convert_3d_synthplace.py --out out/conv/synthplace.jsonl
python3 vendor/tools/fastfill_data/convert_m3dlayout.py --split 3dfront --out out/conv/m3dlayout.jsonl
python3 vendor/tools/fastfill_data/convert_il3d.py --out out/conv/il3d.jsonl
python3 vendor/tools/fastfill_data/convert_mansionworld.py --out out/conv/mansionworld.jsonl
python3 vendor/tools/fastfill_data/convert_scenesmith.py --out out/conv/scenesmith.jsonl
python3 vendor/tools/fastfill_data/deduplicate.py --in out/conv/*.jsonl \
    --out out/conv/deduped.jsonl --priority 3d_synthplace,m3dlayout,il3d,mansionworld,scenesmith_scenes
python3 vendor/tools/fastfill_data/export_sft.py --in out/conv/deduped.jsonl --out-dir data/sft
# export 默认剔除 CC BY-NC(铁律 2)与 unverified-yaw 的 Floor 记录(MansionWorld 朝向防毒)

# 3) smoke(T3.1:先证明链路,不求质量;100 步)
python3 -m fastfill_train.train_sft --config configs/sft_smoke.yaml
python3 -m fastfill_train.merge_lora --base Qwen/Qwen3-8B \
    --lora out/sft_smoke/checkpoint-100 --out out/sft_smoke_merged
bash scripts/serve_vllm.sh out/sft_smoke_merged &
until curl -sf http://127.0.0.1:8901/v1/models >/dev/null; do sleep 2; done   # 就绪再评;评完 kill 掉旧服务
python3 -m fastfill_train.eval_layout --records data/sft/floor_sft.jsonl \
    --samples out/conv/deduped.jsonl --endpoint http://127.0.0.1:8901/v1 \
    --model fastfill-planner --limit 50 --out out/eval_smoke.json

# 4) 正式 SFT(论文配置;模板 A/B/C 各跑一档做消融(--set template=plan / plan_nl))
python3 -m fastfill_train.train_sft --config configs/sft_full.yaml
python3 -m fastfill_train.train_sft --config configs/sft_full.yaml \
    --set template=plan --set output_dir=out/sft_full_plan

# 5) DPO(条件项 T3.3,先例提示大概率值得)
#    stage1(near-miss 默认门):rejected 违规码种类 ≤2 才入库,超限跳过计数;
#    --max-reject-codes 0 恢复完整 model-pair 仅作消融。stage2 注入器均为单对象单属性扰动+双向核验。
python3 -m fastfill_train.eval_layout --records data/sft/floor_sft.jsonl \
    --endpoint http://127.0.0.1:8901/v1 --model fastfill-planner \
    --dump-generations out/gens_sft.jsonl --out out/eval_sft.json
python3 -m fastfill_train.build_dpo_data stage1 --contexts data/sft/floor_sft.jsonl \
    --generations out/gens_sft.jsonl --samples out/conv/deduped.jsonl \
    --out data/dpo/stage1_pairs.jsonl
python3 -m fastfill_train.train_dpo --config configs/dpo_stage1.yaml
#    stage2:validator 全谱注入负例(越界/碰撞/堵门/错向/漏必放/悬空/出面/错 parent/超载)
python3 -m fastfill_train.build_dpo_data stage2 --in out/conv/deduped.jsonl \
    --out data/dpo/stage2_pairs.jsonl
python3 -m fastfill_train.train_dpo --config configs/dpo_stage2.yaml
```

## 验收与对照(WP3/WP4 锚点)

- 评测口径与 scenesmith `scripts/fastfill_api_smoke.py` 一致:parse 率、修复前/后通过率、
  violation 直方图、输出长度、延迟。服务器 API teacher 实测参考(2026-07-10,20 runs,
  bathroom+task):parse 75.5%,修复前 50%,修复后 100%,正常路径 2 调/修复路径 3 调,
  均值 12.1s。**T3.2 线:held-out 管线一次通过率 ≥ teacher 的 80–90%**
  (`eval_layout --teacher-report` 直接算比值)。
- 实验矩阵(用户定):纯 API Surface vs 1–2k LoRA Surface vs LoRA+确定性修复,同批
  SupportContext,比:合法支撑率/悬空穿透率/parent 正确率/必放覆盖/validated evidence/
  输出 token/推理时延。OptiScene-style SFT 作 Floor baseline(它的 prompt+格式,同底座),
  证明增益来自 task conditioning+门窗条件+validator+repair,而非仅换数据。

## 权重交付到 scenesmith

```bash
# 训练机:
bash scripts/serve_vllm.sh out/dpo_stage2_merged        # 或任一 merged 目录
# scenesmith 机(或同机):
export FASTFILL_LLM_BASE_URL=http://<train-host>:8901/v1
export FASTFILL_LLM_API_KEY=dummy
export FASTFILL_LLM_MODEL=fastfill-planner
# 之后 scenesmith 的 FastFillGenerator/OpenAIChatBackend、grow --content-mode fastfill、
# scripts/fastfill_api_smoke.py 原样工作。codec 是行式文本,无需 guided JSON。
```

也可以直接把 merged 目录拷回 scenesmith 机本地用 vLLM 起。

## 数据红线(与 scenesmith 侧一致)

- 铁律 1:SceneEval 污染名单走 `export_sft --contamination-list`。
- 铁律 2:CC BY-NC(HSSD 衍生,IL3D 约 27% 样本)默认不进训练。
- 铁律 4:先 dedup(geometry_hash 三源同房去重)再 export;split 永远 house-first。
- MansionWorld 朝向未按资产校准:export 默认剔其 Floor 记录(Surface 保留,局部系不受影响)。
- 3D-SynthPlace 数据许可未落地(代码 MIT ≠ 数据可商用;3D-FRONT 衍生部分受其 ToU 约束):
  先研究用途训练,provenance 里 license_pending 已打标。

## RSFT(SFT 后的拒绝采样自举)

```bash
# 对训练 context 采 K 次(eval_layout --dump-generations 跑 K 轮或采样 n=K)
python3 -m fastfill_train.build_rsft_data --contexts data/sft/floor_sft.jsonl \
    --generations out/gens_k.jsonl --samples out/conv/deduped.jsonl \
    --out data/rsft/round1.jsonl
# 硬门槛:parse + validator 过 + 物体数 >= max(3, 0.6x真值) —— 防"越训越空"坍缩;
# dry_rate > 0.5 触发 collapse_alarm 并 exit 2:该轮禁止入训,别靠人眼盯。
# 每轮报告:mean_object_delta_vs_gt(负值扩大=在缩水)+ 人工抽看 20 例。
```

阶段 0 三臂决策(configs/stage0_*.yaml)。先冻结快照(house-first 切分、先切后抽、泄漏硬失败、SNAPSHOT.json 记 hash/seed/构成;leakage_check=passed 结果才有效):
```bash
python3 -m fastfill_train.make_snapshot --in data/sft/floor_sft.jsonl data/sft/surface_sft.jsonl --out-dir data/stage0 --val-fraction 0.1 --max-train 8000
```

LoRA r32 / r128 全线性层 / 全参各跑,比修复前通过率 + CTR + 严格解析率 +
p50/p95 延迟 + 遗忘;胜者当默认主线,**亚军在全量规模复跑一次确认排序不翻转**。
validator 碰撞校准已完成并同步(vendor @ scenesmith 23e8b57;GT 探针:
SynthPlace 100% / IL3D 93% / M3DLayout 73%),全量建库不再被铁律 3 饿死。

## 历史校准记录与剩余数据问题

- **真值也过不了严检**:真数据探针(40 个 M3DLayout 3dfront 房)显示 ground truth 修复前仅
  10% 通过 validator,违规全部是 `L1_FLOOR_COLLISION`(3D-FRONT 原始布局本身含 bbox 重叠,
  典型是椅子塞在桌下)。这不是转换器 bug,是铁律 3 的入库清洗课题(T2.7):要么给
  椅×桌类容许重叠对做白名单/调容差,要么 validator 过滤+按 repair delta 降权后再入库。
  在解决前,SFT 语料按"validator 过滤后子集 + 全量"两档各备一份做对照。
- codec 面向 token 预算不编码 z_local,floating 类 stage2 负例编码后与正例相同,builder 已
  显式跳过并计数(`identical_completion:floating_surface_object`)——悬空错误谱靠 stage1
  自产负例覆盖。
- `--set` 覆盖走 YAML 1.1 解析:写 `1.0e-4`,别写 `1e-4`(后者会变字符串;有测试钉住)。

## 测试

```bash
PYTHONPATH=src python3 -m pytest tests -q     # 单测不依赖 torch/GPU;以实际 collected/passed 为准
```
