# 指标口径（2026-10-20）

这是现有输出的写作与核对记录，不是新预注册协议。主文件两张表各自固定人群和系统版本；任何新增实验应在看结果前另锁定协议。

## 主表

- 全部 8,647 请求；raw / S1 实际完整 8,643，最终系统完整 8,647。
- GSR = complete AND 无 OOB、support_fail、ceiling_fail、main/severe collisions、fixed_blocking。所有请求为分母。旧 `check.valid` 不等同 GSR。
- 几何检查：边界容忍 10cm；支撑与层高容忍 5cm；碰撞要求高度交叠 >5cm、旋转足迹交面积 >0.01m²。直接父子、部分扁平物体、tucked/contained 等采用源码冻结的豁免。因此 GSR 不能称绝对零碰撞。
- OOR：忽略 yaw 的宽深矩形，所有可移动物体对交叠面积之和 / 总物体占地面积；不含 fixed，不豁免支撑或高度。不应叫 out-of-room rate；可以大于 1。
- 尺寸 log-MAE：每物体三轴 |ln(pred/ref)| 平均，再按物体、房间平均。当前主表 OOR/尺寸指标条件于各阶段完整输出，失败不填零。
- size-IoU（本简稿省略）：以尺寸为边长、同中心的两个框的体积 IoU；不衡量位置或朝向。
- strict = complete AND endpoint_ok AND attach_ok AND admit_ok。下游适配内部先 snap_inside，所以 raw 行的 strict 也含此固定适配。使用本地 infinite-task-world strict port，不是 RoomGenBench/EmbodiedGen 官方标准。1mm envelope、0.2m 导航半径、0.25m 网格等参数见源码。
- required={}（本简稿省略）不仅撤销类别数量约束，还撤销 required-object reachability 与 floating/collision 的部分约束，不能叫“无约束碰撞率”。

代码出处（相对仓库）：`fastfill/comparison/geometry.py:24,34,84`、`fastfill/validate.py:35,72,124`、`fastfill/system/evaluate.py:327,343,362,372`、`fastfill/comparison/downstream.py:69,237,263,289,309`。

## 视觉表

- A = 27B-P 原始输出；S = amendment-022 旧 checker-trigger 系统。S 的冻结来源在硬盘 `fastfillout/unknownsize-20261013/visual-024/S/source/summary.json`，不能替代 a027。
- FID/KID：相同 1,005 房间、GT reference、2,048维 Inception 特征。1,009 中 4 间缺资产；失败不以黑图补齐。KID 原始尺度，不乘 100 或 1,000；表内 ± 为记录的标准误，不是95%CI。
- Mesh CF@500：2cm voxel；交体积 = 共同体素数 × 8cm³；任一 eligible non-parent movable-object pair >500cm³ 记房间失败，fixed 物体不参与。率按成功分析的 1,005 房间计算。500cm³ 是容忍阈值，不等同严格零相交。网格离散化和资产匹配均有限制。
- 视觉指标是描述性补充；当前新版系统未渲染。不得将不同 population 的 FID 混排后选最小值。
- GPT 裁判：原始 `overall_mean` 由模型单独给出，与三项均值不同。本简稿省略该列。若后续加入，应分别写明 gpt-6-astra high、功能/布局/美学 0–10、173/174 成功房间、提示/视角与失败策略；不能等同 OptiScene 的 GPT-4o。
- 人工 UR 未测，不用自动 checker/strict/mesh 或模型裁判替代。

源码出处：`fastfill/comparison/mesh_collision.py:61,224,296`。冻结协议：amendment-019、024；原始特征/网格/裁判文件哈希见 `evidence.json`。

## 统计与结论边界

- 当前新旧触发差：同一批房间，3,056 个房屋连通组配对 bootstrap，2,000次，seed42，95% percentile CI，未做多重比较校正；不包含训练种子不确定性。
- 原始 8B J0/P0 预注册共同终点：全请求 raw checker + 共同完整房间 size log-MAE，两个终点 Holm 校正。系统 strict 改进是后续描述性实验，不替换这些终点。
- amendment-021 27B 模型按 dev checker 及护栏选择，a027 复用其预测；不能将系统 gain 归因于27B训练本身。
- 测试集已被历史实验曝光；不能称新盲测。不能把早期 V1.1 的 target-derived ordering 诊断与现在输入契约混合。
- 46.0 GPU-h 只对应 a027 O2 分片进程记录（含加载与恢复），不是端到端训练成本或每请求延迟，也不是免费运行。
- 参考论文只是指标依据；当前表没有重新运行 ATISS / DiffuScene / LayoutGPT / 作者 OptiScene 的共同基准。
