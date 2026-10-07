# FastFill v2 损失权重标定（2026-10-06）

本页记录 `fastfill/v2/configs/qwen3_8b_main_world7.json`（及派生的 `qwen3_8b_pilot_1gpu.json`）中四项几何损失权重的来源。`LossConfig` 是严格 dataclass、`run_training` 只接受已知顶层段，配置文件不能携带 `_calibration` 注释键，所以数值和规则写在这里。

## 规则

1. 在未训练基线下计算每一项的逐对象平均值 `B_k`（定义见下）。
2. 以 position 为参照（`λ_position = 1`），`λ_k = round_1sig(B_position / B_k)`，取一位有效数字。
3. 检查每项基线贡献 `λ_k · B_k` 两两之比不超过 2。

## 基线定义（全部由 `GeometryCriterion` 本身在新 L1 定义下计算）

| 项 | 基线预测 | 损失定义（`losses.py`） |
|---|---|---|
| position | 地面多边形形心、z = `floor_z_m`（房间归一化坐标） | 可学习坐标 L1 之和 / 3；`support_parent == floor` 的 z 固定、贡献 0 |
| size | 按 category 取 20,000 个训练房间的逐轴中位数；类别未见则用全局中位数 | log 尺寸 L1 之和 / 3 |
| yaw_cls | 均匀 logits（12 bin） | 对称感知联合最小的交叉熵 = ln 12 |
| yaw_reg | 零残差 | GT bin 残差 Huber（β = 1） |

yaw 有效性按契约 C4 模拟：`front_known` 且非 tilted 且 yaw 有限；对称阶 MultiScan 为 1、其余为 2。磁盘上的行仍是 strict 策略（抽样 2,000 房间中 yaw 有效 0 个），不模拟则 yaw 两项没有样本。

## 数据与命令

- 验证样本：`/Volumes/harddisk/FastFill_v2_multisource_20261006/eligible-local/validation.jsonl`（8,125 行）中 `random.Random(20261006)` 抽 2,000 房间，24,279 个请求对象；14 个来源，SpatialLM 551 / MansionWorld 276 / IL3D_3dfront 230 / InternScenes_gen 214 / Structured3D 200 / SAGE-10k 121 / 其余 408。
- 训练中位数：同一 RNG 从 `train.jsonl`（124,375 行）抽 20,000 房间，170,875 个完整尺寸标签，5,288 个类别；全局中位数 `[0.381, 0.475, 0.594]` m。验证集 5,028 / 24,279 个对象（20.7%）落到全局中位数回退。
- 匹配关闭（`hungarian=false`）：基线预测在组内完全相同，匹配不影响数值。
- 脚本：会话 scratchpad 的 `calibrate_loss_weights.py`（见附录），运行：
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python calibrate_loss_weights.py`（约 15 s，CPU）。

## 结果

| 项 | 有效对象数 | 基线 `B_k` | `λ_k` | 贡献 `λ_k·B_k` |
|---|---:|---:|---:|---:|
| position | 23,471 | 0.2026 | 1 | 0.203 |
| size | 17,222 | 0.3222 | 0.6 | 0.193 |
| yaw_cls | 23,471 | 2.4849 (= ln 12) | 0.08 | 0.199 |
| yaw_reg | 23,471 | 0.0276 | 7 | 0.193 |

贡献最大/最小 = 1.05，满足因子 2 要求。

说明：
- size 有效数少于 position：抽样中 7,057 个对象（29%）没有尺寸标签（SpatialLM 等来源），与 C2 "按位置匹配、尺寸可缺" 一致。
- yaw_reg 基线很小，因为 81% 的 GT yaw 落在 bin 中心（轴对齐家具，|残差| < 0.01），残差 90/95/99 分位为 0.37 / 0.70 / 0.97。按规则得到 `λ_yaw_reg = 7`；tanh 残差头使单对象 Huber 上界为 0.5，加权后 ≤ 3.5，不会爆炸，但这是本次标定中最偏离 1 的权重。若后续 pilot 显示残差头过拟合零残差，可把它封顶到 2 并记录偏离规则。
- 旧配置 SmoothL1(β = 1) 在归一化坐标上几乎全在二次区（位置误差 < 1），等价于 L2 且随房间尺度变化；本次改为 L1，默认 `position_type = size_type = "l1"`，`smooth_l1_beta` 仅在 `smooth_l1` 下生效。
- 其它配置（`structured*.json`、`qwen3_8b_pilot.json`、`qwen3_8b_bbox_*.json`、`smoke.json`）只补齐新键，权重保持 1.0，不改既有实验定义。

## 附录：标定脚本

```python
"""Untrained-baseline value of each FastFill v2 loss term under the new L1 definitions."""
import json, math, random, statistics, sys, time
from collections import defaultdict

import torch
from shapely.geometry import Polygon

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.losses import GeometryCriterion, LossConfig

ROOT = "/Volumes/harddisk/FastFill_v2_multisource_20261006/eligible-local"
SEED, N_VAL, N_TRAIN = 20261006, 2000, 20000


def sampled_rows(path, total, n, rng):
    chosen = set(rng.sample(range(total), n))
    with open(path) as handle:
        for index, line in enumerate(handle):
            if index in chosen:
                yield json.loads(line)


def emulate_c4_yaw(row):
    source = row["provenance"]["source"]
    front_known = bool(row["provenance"].get("source_meta", {}).get("front_known", False))
    evidence = row["provenance"]["field_evidence"]
    yaw = [o.get("yaw_rad") for o in row["target"]["objects"]]
    valid = [front_known and not fe["tilted"] and isinstance(y, (int, float)) and math.isfinite(y)
             for fe, y in zip(evidence, yaw)]
    return {**row, "validity": {**row["validity"], "yaw": valid,
                                "yaw_symmetry_order": [1 if source == "MultiScan" else 2] * len(valid)}}


def main():
    rng = random.Random(SEED)
    train = list(sampled_rows(f"{ROOT}/train.jsonl", 124375, N_TRAIN, rng))
    per_category, everything = defaultdict(list), []
    for row in train:
        by_id = {o["id"]: o for o in row["condition"]["objects"]}
        for target, size_ok in zip(row["target"]["objects"], row["validity"]["size"]):
            if all(size_ok):
                per_category[by_id[target["id"]]["category"]].append(target["target_size_local_m"])
                everything.append(target["target_size_local_m"])
    median = lambda rows: [statistics.median(r[q] for r in rows) for q in range(3)]
    category_median = {c: median(v) for c, v in per_category.items()}
    global_median = median(everything)

    validation = [emulate_c4_yaw(r) for r in sampled_rows(f"{ROOT}/validation.jsonl", 8125, N_VAL, rng)]
    criterion = GeometryCriterion(LossConfig(hungarian=False))
    sums, counts = defaultdict(float), defaultdict(int)
    for row in validation:
        batch = collate_samples([row], TinyTokenizer(), max_length=10**8, max_objects=10**6)
        room = row["condition"]["room"]
        centroid = Polygon(room["floor_polygon_xy_m"]).centroid
        centre = torch.tensor([centroid.x, centroid.y, room.get("floor_z_m") or 0.])
        n = batch["slot_mask"].shape[1]
        sizes = [category_median.get(obj["category"], global_median) for obj in row["condition"]["objects"]]
        predictions = {
            "position_normalized": ((centre - batch["origin"][0]) / batch["scale"][0]).expand(1, n, 3).clone(),
            "size": torch.tensor(sizes).reshape(1, n, 3),
            "yaw_logits": torch.zeros(1, n, 12), "yaw_residuals": torch.zeros(1, n, 12),
            "slot_mask": batch["slot_mask"]}
        result = criterion(predictions, batch)
        for key in ("position", "size", "yaw_cls", "yaw_reg"):
            sums[key] += float(result["term_sums"][key])
            counts[key] += result["term_counts"][key]
    baseline = {k: sums[k] / counts[k] for k in sums}
    one_sig = lambda x: float(f"{x:.1g}")
    lambdas = {k: one_sig(baseline["position"] / baseline[k]) for k in baseline}
    contributions = {k: lambdas[k] * baseline[k] for k in baseline}
    print(json.dumps({"baseline": baseline, "counts": dict(counts), "lambda": lambdas,
                      "contribution": contributions,
                      "max_over_min_contribution": max(contributions.values()) / min(contributions.values())}, indent=2))


if __name__ == "__main__":
    main()
```
