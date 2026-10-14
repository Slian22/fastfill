**评审结论**：方向是对的：先做 (b)，几何靠确定性修复，VLM 只作可选的语义层，类别、物体集合、id 和尺寸都冻结。但下面 8 条必须先改，方案才能执行。

## 必须修

**1. S1 过不了 backend 严格门，而且和"不改 pose_targets.py"自相矛盾。**
- pose_targets 的目标是 `check()` 计数之和（pose_targets.py:228-230），用的越界容差是 `oob_tol=0.10`（validate.py:72）。窗户重叠也不算 fixed_blocking（validate.py:108-110）。所以越墙 1–100 mm、挡窗这两类问题，它的搜索根本看不到。"严格门并入接受条件"实际上只能事后拒收，引导不了搜索。
- 它还会把**所有**物体的位置取整到 cm、yaw 取整到整数度，没问题的物体也一样（`_normalize` :163，`_finish` :277-279）。位置最多偏 5 mm；一个 2 m 长的物体，角点会因 yaw 取整偏约 9 mm。原本贴墙合格的物体可能因此掉出 1 mm 门。
- 测试 GT 本身就有约 40% 越墙超过 1 mm（interface.py:71）。
- 改法：
  - S1 拆成两步。先跑 pose_targets（check 级），再加一步独立的严格贴入，推回"地面减去 keep-out"的范围，容差 1 mm。这一步可以复用 `interface.snap_inside`（:255），它已经会保留硬约束、带着支撑子树一起移动、不加深与 fixed box 的重叠。
  - 两步之后都重跑 `check` 和严格门。
  - 位移统计要扣除取整带来的偏移。
  - pose_targets 没有超时参数，单间时限只能在外层进程上做。

**2. dev 上没有 keep-out 几何，"严格接受率"测的不是 backend 的规则。**
- contract.py、comparison/geometry.py、validate.py 里都没有 keep-out，grep 结果为空。`check` 只看门窗自身的 fixed box。backend 的门 keep-out 是门前的净空区。
- 要先定 keep-out 从哪来：按 backend 规则从门窗合成，或者直接用 e2e_backend.py 跑真 backend。否则这个主指标没有效度。

**3. 修复能力的证据不成立。**
- 118,058/124,843（94.6%）的起点是"GT 位姿 + 先验尺寸"，本来就接近可行（MANIFEST）。这推不出它能修好模型输出：模型的失败集中在多物体、同类重叠。
- 3h36m 是 main-data 作业 7365659 整体的墙钟时间，包含先验拟合、资格判定、序列化，而且是多进程跑的。它不是"repair 全量 CPU 时间"（research-log.md:30）。
- 模型输出上的修复率和单间耗时，只能靠 dev pilot 实测。

**4. 回收 tier 的前提有误，而且会破坏配对设计。**
- 25,920 间里，21,527 间是 J 的 GT 本身过不了 check（124,843 − 103,316），6,785 间是 P teacher 已经修复失败。
- P 臂的训练目标本来就是 pose_targets 的产物（main_data.py:1-7, 196）。另外已有 19,135 间是 P 合格、只因 J 不合格被丢掉的。所以"回收"对 P 不是新东西。
- J/P 用的是同一批房间的配对设计。加 tier 就必须两臂同时加，因此只能作为新实验。

**5. 协议完整性。**
- **J−P 对比被混杂。** pose_targets 是 P 臂的训练 teacher，P 的输出本来就接近它的不动点，所以 S1 对 P 的增益会系统性大于 J。S1 之后的 J−P 差异不能当作 J/P 结论。
- **test 不再是盲测。** report-v2-JPL-001:3 已写明"历史测试集曾用于诊断，不是新盲测"；S1 的设计动机（重复物体、大房间越界）也来自 test 上的失败分析。所以 test 上的 S1 结果只能标 exploratory，不能写成"盲测只跑一次"。
- **amendment-011 必须在看任何 dev 修复结果之前锁定。** 内容要包括 critic 的模型、prompt、温度和渲染参数。
- **样本量前后不一。** §5 写 200 间，§4 写 400 间。演练用的房间要从 pilot 名单里排除。
- **GT 演练不能定阈值。** 用 dev 中 GT 不通过的房间演练，起点同样接近可行，结果不能用来定阈值。

**6. S3 的"critic 分数不降"既不可行，又是自己评自己。**
- 每条编辑都要重新渲染、重新调一次 critic，每间最多多出 5×2 = 10 次调用。§3 按每间 1–2 次估的全量成本没有算进去。
- 提议编辑的和打分的是同一个 critic，分数会偏高，这个条件几乎过滤不掉什么。
- 编辑后重跑 S1 时，pose_targets 会重新尝试原 yaw、四个正交 yaw 和别的位置，可能直接把 critic 的编辑撤销；它又没有"钉住"物体的参数。
- 改法：
  - 每条编辑必须附带一个可检验的谓词（例如 facing、against_wall），接受条件改为"跑完 S1 后这个谓词仍然成立"。
  - critic 每轮最多复评一次。

**7. 删掉 `reparent`。**
- 它会改 place（floor ↔ on_object）。interface 把 anchor 当硬约束（interface.py:388-389）；pose_targets 的 docstring（:5）也明确禁止重新分配支撑。
- 如果保留，最多只允许 on_object → on_object。

**8. "修复不可能让 checker 变差"说过头了。**
- 字典序只保证第一个不同的分量变好。比如 fixed_blocking 少 1 个、severe 多 5 个，也会被接受。
- 元组之外的东西没有保护：
  - 请求里的硬约束（faces/near/against_wall）。snap_inside 就出过这种事：推回墙内时打破了 GT 上原本成立的约束。
  - tuck 类碰撞和挡窗的 fixed_collisions。
  - yaw 正交翻转（例如床头转成背墙）。
- 接受条件要加上 `interface.validation` 的硬约束 `holds`，并单独统计 yaw 翻转。

## 应修

9. **评估用 GPT judge 和方案自己引的 SceneCritic 矛盾。** SceneCritic 说 VLM 与人的一致率只有 47–59%（这篇 2026 年预印本我没核实）。
   - raw 臂在渲染图里能直接看到穿插，盲评做不到。
   - judge 只作次要描述，不设采用门槛。
   - 调用量 400 × 2 × 3 = 2,400 次，超过 judge 上限 800（amendment-006:23），需要新授权。

10. **mplot3d 的斜视图深度排序有问题。** Poly3DCollection 按整体排序，偏偏重叠、穿插的盒子最容易画错。
    - 顶视用 matplotlib 2D。
    - 斜视用 Blender Workbench 渲纯盒子，或者自己按面做画家排序。耗时要实测。

11. **很多来源的正面方向（semantic front）没有验证过。** InteriorGS 的 yaw 来自 bbox 轴，SpatialLM 和 SAGE 是经验约定，InternScenes 有 k0 回退（full-audit-data-report-20261013.md:15, 98-114）。
    - (a) 的审计会把标注约定的差异当成布局错误。
    - facing 类检查只在已验证正面方向的来源上做。

12. **抽样审计凑不到每源 200 间，也没有决策规则。**
    - Scan2CAD 只有 11 间 eligible，MultiScan 只有 58 间。
    - 更关键的是没写"审计结果到什么程度就做什么"，要预先写死。

13. **几个数字的来源或解读要改。**
    - **Cycles 22.6 s/间**：在 docs/research 里找不到出处（render_assets.py 用的是 Cycles、8 spp、512²），要么标"未测"，要么给出处。
    - **13.5 s**：这是 gpt-6.1-sol 用纯文本、medium 推理生成布局的中位延迟（JPL-001 report:188），不是看图 critic 的延迟。
    - **"把家具缩小来过检"**：这只是一种解释，没有证实因果。J 自己也有 −8.3% 的 footprint 偏差。

14. **开源多模态候选不止 O2。** O3（gemma-4-31b-it）和 O6 同样被设成了 `image:0`（llm_open.py:66-67, 83），也可以考虑。O2 是 tp=2，8 张卡就是 4 个副本。

## SceneSmith 事实核对（本地代码）

**正确**
- critic 用 gpt-5.2，只读，三个工具。
- 渲染是 1 张 1024² 顶视加 4 张 512² 斜视，相机在四角，仰角 30°（view_generation_mixin.py:13, 107）。
- 有碰撞时 realism 最高 4 分（stateful_critic_agent.yaml:118-120）。
- 最多 3 轮，全部 ≥9 分提前停（base_furniture_agent.yaml:244, 252）。
- 几何修复是 Drake IK，SNOPT，旋转固定，之后跑仿真（physical_feasibility.py:516, 1548）。
- 房间只支持矩形（house.py:544-568）。

**不精确：回滚**
- 循环中途的回滚只是给 planner 的建议（base_furniture_agent.yaml:246-249，原文 "suggested"）。
- 只有结束时会自动退回 N-1（base_stateful_agent.py:479-525），依据是 critic 自己的分数。

**补充：IK 投影**
- influence distance 2 cm，单场景时限 360 s，只是针对小穿插的局部投影（残余平均 3.77 mm）。
- 它不适合我们这种大面积重叠。方案没有搬它，这一点是对的。

**SceneSmith 是否进了训练**
- 直接证据是 main_data.py:30 的 `HELDOUT = {'SceneSmith', 'SpatialGen'}`，以及 MANIFEST 的 by_source 只有 16 个源。multisource_data.py:44 只是 v2 的定义。
- 任务背景里"98,754 间来自含 SceneSmith 的 18 个数据集"这句不对。

**未核对**：论文 Table 14 和成本数字，这次没有联网查。

## 已核实无误的方案数字
- 98,923 → 98,754。
- 25,920；J 合格 103,316，P 合格 118,058。
- 测试 GT 通过率 0.7230，即 6,252 间。
- MansionWorld：GT 0.9764，J 0.2041。
- L 每次调用约 4.9k 输入 token（7,235,552 / 1,469）。
- 调用上限 348 / 800 / 4,800（research-state.yaml:26，amendment-006:23，amendment-002:39）。
- O2 在 llm_open.py 里是 `image:0`。