# FastFill「渲染 → 评审 → 修复」方案（草案）

**结论**
- 先做 (b)：在推理阶段对模型输出做后处理。
- (a) 训练数据只做确定性回收，再加抽样审计。不对 98,754 间房做 VLM 全量评审。
- 几何问题（碰撞、越界、门窗、支撑）交给确定性 solver。VLM critic 只看 box diorama，只判断语义上是否合理。
- 两种用途都**不改类别、物体集合和 id，尺寸也冻结**。原因有两个：P 臂的尺寸本身就是输入；J 臂如果允许改尺寸，会掩盖尺寸误差。gpt-6.1-sol 的 -17% footprint 偏差就是"把家具缩小来过检"。

## 1. SceneSmith 实际怎么做（已读本地代码 /Users/slian/Desktop/3D/Worldedge/scenesmith）

**结构**
- 三个 agent：planner、designer、critic（gpt-5.2）。
- critic 只读不改，能用的工具有三个：
  - `observe_scene`：Blender EEVEE 渲染，1 张 1024² 顶视加 4 张 512² 30° 斜视，带 id 标签、bbox 线框、朝向箭头，并隐藏靠近相机的墙；
  - 场景 JSON；
  - `check_facing_tool`。
- 物理检查结果以文本注入给 critic，规则是"工具结果优先于视觉判断"。

**critic 的输出**
- 6 项 0-10 分加一段文字评语，不是编辑操作。
- 只要有碰撞，realism 最高 4 分（stateful_critic_agent.yaml:119）。
- 评语原样转给 designer，由 designer 用 add/move/remove/rescale/generate 去改。

**终止规则**（base_furniture_agent.yaml:244-252）
- 最多 3 轮；
- 全部分数 ≥9 时提前停；
- 任一单项降 ≥2 或总分降 ≥1 时回滚。

**几何修复**
- 真正修几何的是 Drake IK 非穿透投影（SNOPT，旋转固定），之后再跑仿真（physical_feasibility.py:516,1548）。

**论文 Table 14**
- NoCritic 在 COL/STB/OOB/NAV/ACC 上持平或更好。
- 人评差异不显著（51.8% / 47.5%）。
- NoCritic 便宜 70%（$4.24 vs $13.98 每场景）。
- critic 的主要作用是让场景多出物体。

**借鉴**：id 标签加朝向箭头（box 没有正面，必须靠箭头）、"工具优先于视觉"、回滚规则、轮数上限。

**不借鉴**
- add/remove/generate/rescale 这些工具；
- 完整度评分，以及"房间太空就加家具"；
- "bbox 重叠 ≠ 碰撞"：我们的 box 就是几何本身，这条要反过来；
- Drake/Blender 代码本身：只支持矩形房间，依赖重，aarch64 上未验证。

另外，SceneSmith 在 FastFill 里只作 test 数据源（multisource_data.py:44）。

## 2. 架构

| 步骤 | 做什么 | 允许改 | 复用 |
|---|---|---|---|
| S0 检查 | `check()` 加严格门（越界 1 mm、门窗 keep-out 1 cm）；通过就原样交付 | 无 | validate.py；interface.py:373 `validation` 的逻辑（需适配 v1 schema） |
| S1 确定性修复 | 贪心搜索位置和 yaw，支撑子树跟着刚性移动；严格门并入接受条件；设位移预算和单间时限 | pos、yaw | pose_targets.repair（文件不改） |
| S2 VLM critic（可选） | 只看 S1 之后通过检查的房间，判断语义 | 只能提受限编辑 | judge_batch 的调用方式 |
| S3 复核 | 每条编辑都重跑 S1、`check` 和严格门 | 不变差才接受 | — |

**S2 的输入**
- 1 张顶视和 1 张斜视 box diorama，带 id、朝向箭头，按 check flag 着色；
- 每个物体的 JSON：id、category、size、pos、yaw、parent；
- checker 摘要。

**S2 的 rubric**：只有 functionality 和 layout 两项（0-10），加一份清单：
- 床头靠墙；
- 椅子朝桌、沙发朝电视；
- 储物家具靠墙且正面可达；
- 门口有通道。

**S2 的输出**
```json
{"scores":{"functionality":7,"layout":6},
 "edits":[{"id":"o3","op":"move|rotate|against_wall|face|reparent","x":1.2,"y":0.4,"yaw_deg":90,"target":"o5","reason":"..."}]}
```
- id 必须已存在；
- 每轮最多 5 条编辑；
- 不允许 add/remove/resize/relabel；
- 朝向类编辑用几何函数验证，不信从图上读出的朝向。

**S3 的接受条件**（三条都要满足，否则丢弃该条编辑）
- checker 计数按 (valid, fixed_blocking, severe, main, oob, support_fail, ceiling) 的字典序不变差；
- 严格门仍然通过；
- critic 分数不降。

按这个构造，修复不可能让 checker 变差。最多 2 轮；没有编辑、本轮没有编辑被接受、或两项都 ≥8 时停止。

## 3. 两种用途

**(a) 训练数据**
- **需要更正一个前提**：实际训练用的 98,754 间已经被 checker 过滤过（main_data.py:147-204），"flag, don't filter"并不描述这个训练集。
- 模型失败主要是模型自己的问题：
  - test 上 GT 能通过的 6,252 间里，J 也只有 0.44；
  - MansionWorld 的 GT 通过率 0.976，J 只有 0.20。
  - 所以清洗训练数据对 checker 指标帮助有限。
- 可以做的三件事：
  - **回收**：用 S1 回收被拒的 25,920 间，作为一个可撤回的独立 tier。P teacher 已在 124,843 间里成功修复 118,058 间，全量 CPU 3h36m。
  - **抽样审计**：VLM 每个源抽 200 间，16 个源约 3,200 次调用。
  - **DPO 对**：用（模型输出，修复后）作 DPO 对。注意 chosen 是 solver 的产物，不一定自然。
- 如果坚持全量 VLM：
  - 调用量：约 10–20 万次（1–2 轮）。
  - 输入：每次约 7–9k token，其中文本按 L 实测约 4.9k，图像 token 未测；合计 0.7–1.8 B token。
  - 输出：0.1–0.3 B token。
  - 耗时：中位延迟 13.5 s、并发 4 时约 4–8 天。
  - 远超现有 348/800 和 4,800 次调用的上限，需要新授权。美元价格未知，不做估算。
- 开源 VLM 走 GH200：
  - O2 是多模态模型，但 llm_open.py 里设置了 `image:0`，需要打开图像并验证它看 box 图的能力；
  - 粗估 8 卡 0.5–1 天，未实测。
- 渲染成本：Cycles 22.6 s/间，全量约 620 CPU 小时，不可行。box 图改用 matplotlib，预计 <1 s/间（未测）。
- 风险：
  - critic 幻觉：VLM 与人的一致率只有 47–59%（SceneCritic）；
  - 布局同质化：都推去靠墙，抹平真实扫描的分布；
  - 训练和测试的 GT 不一致，会带来分布偏移；
  - **绝不改 test/dev GT**。

**(b) 推理后处理**
- 下游 backend 的 1 mm / 1 cm 是硬门，VLM 做不到，所以 S1 必需，S2 可选。

**推荐顺序**
1. (b) 的 S0+S1 做 dev pilot；
2. 在此基础上加 S2 做消融（对标 SceneSmith 的 NoCritic）；
3. 再做 (a) 的回收 tier 和抽样审计；
4. 是否重训另议。

## 4. 评估

- **先锁定 amendment-011，再看结果。** 锁定内容：三个臂（raw / S1 / S1+S2）、指标、阈值，以及 dev 房间 id 清单的哈希。
- **dev pilot**：按源分层抽 400 间，J 和 P 各跑一份。
- **指标**：
  - geometry_success；
  - backend 严格接受率（1 mm / 1 cm）；
  - 尺寸偏差（尺寸已冻结，应为 0，作 sanity 检查）；
  - 每物体位移的中位数和 p95、被移动物体的比例、位移 >0.5 m 的比例、yaw 变化；
  - 修复失败率和单间耗时；
  - GPT judge 的 functionality/layout：盲评臂，且 judge 不能和 critic 是同一模型、同一 prompt。
- **阈值示例**（需要你来定）：S1 被采用的条件是严格接受率提高 ≥10 pp、位移中位数 ≤0.3 m、functionality 降幅 ≤0.2。
- **报告方式**：单列一张表，标为 exploratory，不替换确认性的 raw 结果。dev 过阈值后，test 只跑一次。

## 5. 一天的最小步（只用 CPU，不调用 API/GPU）

1. 新建 `fastfill/comparison/box_render.py`：复用 tools/visualize.py 的 `draw_layout` 出顶视，用 mplot3d 出斜视；按 check 结果着色，带 id 标签和朝向箭头。
2. 新建 `fastfill/comparison/output_repair.py`：`repair_output()` 依次做严格门、调用 pose_targets.repair（尺寸冻结）、按字典序不变差接受、统计位移。不改 pose_targets.py 和 validate.py。
3. 新建 `tests/test_output_repair.py`：断言 id、类别、尺寸不变，且修复后计数不劣于原始。
4. 在 200 间 dev 上跑 raw vs S1，输出 CSV，并抽 20 张渲染图人工看。
5. 起草 amendment-011 和 critic 的 schema/prompt，暂不发任何调用。

## 不确定点

- **pose_targets 的限制**
  - 不改 parent，只试原始 yaw 和四个正交 yaw；
  - 最多 256 个物体，预处理 O(N³)，没有时限；
  - docstring 写明 "Never use this as inference postprocessing"，所以推理阶段使用必须走新 amendment，冻结的比较里不能用。
- **dev 预测不在本地**：本地 results 下只有报告，可能需要你从 Isambard 拷出来。拿不到的话，先用 dev 里 GT 不通过的房间做演练。
- **同类物体重叠的重复物体**：在不删物体的前提下，局部搜索可能解不开，只能标为修复失败。
- **两个门的容差不一致**：`check` 的越界容差是 10 cm，严格门是 1 mm。