# 大表 v4：参考文献与计分协议复核

核查日期：2026-10-20。结论：**有一手依据，但 v3 不能称为全部正确。** 此次按世界指标、候选执行证据、统计协议三路独立审核，主审复查关键原文与代码，并补搜邻近评测。没有复现外部系统，也没有新增物理实验。

当前产物：[主表 PDF](../../../output/pdf/WorldEdge-Table1-proposal.pdf)、[LaTeX](../../../paper/WorldEdge-Table1-proposal.tex)、[完整中文设计 v4](table-design-v4.md)。所有结果仍为 NE；参考文献支持设计来源，不代表本项目效果已获验证。

## 1. 已修正的五组问题

| 问题 | v4 修正 |
|---|---|
| SceneCode 有方法行却无引用；7 条旧书目没有正文引用 | 补 SceneCode 原文与官方导出说明；移除本版正文未使用的 HomeRobot、BEHAVIOR、BEHAVIOR-1K、EAI、RoboCasa365、PhyScene3D、Holodeck 书目，旧调研记录保留 |
| “沿用原指标”掩盖具体实现差异 | 第一页注明指标谱系；语义、物理、Reach、SR/GC 均写清继承与改编，新增来源对照表 |
| 漏掉直接评测近邻 | 补 HappyWorld-Bench 的扩展保持评测、WorldArena 的感知/功能评测、WorldWeave 的持久几何扩展；撤掉任何依靠这几个宽泛方向建立独有性的空间 |
| 公平对照仍不够具体 | 种子接纳独立于待测方法表现；机制比较固定任务关键资产；恢复器须通过无增长重建一致性验收 |
| 任务与 GC 可被弱化 | 完整目标集合始终在 GC 分母；约束绑定到具体目标；主任务必须真正依赖早期改动的状态，不能只是再次使用同一实体 |

## 2. 各列到底参考什么

| 主表列 | 一手依据与位置 | 采用范围与限制 |
|---|---|---|
| CNT、ATR、OORel、OARel | [SceneEval v3，§3.2–3.3、Table 4](https://arxiv.org/html/2503.14756v3#S3.S3) | 分母为该类标注约束数，先场景内求率再跨场景宏平均；不是对象数/对象对数。本方案增加明确的 missingness 与汇总规则 |
| COL、STB | [SceneSmith，正式论文 App. M.2.1，PDF p.40](https://raw.githubusercontent.com/mlresearch/v306/main/assets/pfaff26a/pfaff26a.pdf#page=40) | 穿透 >1 mm；重力仿真 5 s、步长 1 ms；稳定条件位移 <1 cm 且旋转 <0.1 rad。须同时披露排除对象、焊接与凸分解，不只摘阈值 |
| Reach | [PhyScene，§4 Metric、Table 3](https://arxiv.org/html/2404.09465v1#S4)；[官方 PDF](https://physcene.github.io/assets/paper/CVPR24_PhyScene.pdf) | 原文为随机起点下对象平均可达；本表改为固定出生点和必需任务目标，缺失目标不可达。不是实际导航成功 |
| E2E SR、GC | [ALFRED，§5.1](https://arxiv.org/pdf/1912.01734v2)；[PARTNR，A.5.3–A.5.5](https://arxiv.org/html/2411.00081v1#A5.SS5) | 借鉴完整任务/部分目标及依赖约束。纳入生成/扩展失败、完整目标分母和具体身份/时序规则是本方案改编，不是直接重报原 benchmark |
| Expand、Expand+Keep、Load、COL_Δ/STB_Δ、资源口径 | 本地协议定义；相关先例见下节 | 整条 g 次扩展的成功和联合保持率、边界对象集合、g=4 及成本范围均为拟议设计。要做 pilot、冻结评测器并验证，不能称为公认标准 |

### 论文与实现之间的差异

SceneEval 的对象缺失不会合法地删掉对应约束；但“删物体必然让得分下降”过强，删掉多余实例也可能改善精确数量约束，v4 已改正。

更实质的问题是：官方 OOR/OAR 实现会将无法映射的关系设为 `Skipped` 且 `satisfied=True`。见固定 commit 的 [OOR 分支](https://github.com/3dlg-hcvc/SceneEval/blob/116881e7945dcf7dbc58f7c64590a4c0b99c5cf5/metrics/obj_obj_relationship.py#L189)、[OAR 分支](https://github.com/3dlg-hcvc/SceneEval/blob/116881e7945dcf7dbc58f7c64590a4c0b99c5cf5/metrics/obj_arch_relationship.py#L257)。本协议要求提前冻结支持的标注词汇；评审失败或关系映射失败作为 unknown，不自动给分。零标注情形预先声明为 N/A 并报告数量，不赋 0 或 100%。

SceneSmith 作者 fork 的[汇总脚本](https://github.com/nepfaff/SceneEval/blob/34afe64a692f47327f162de82efbdde9d550cf85/scripts/analyze_results.py#L111)从文本抽取关系分数，可能遗漏提前返回的失败；碰撞汇总分母则取全部 `obj_ids`，不一定等于排除地毯后的适用对象数。其[稳定判定](https://github.com/nepfaff/SceneEval/blob/34afe64a692f47327f162de82efbdde9d550cf85/metrics/architectural_welded_equilibrium.py#L269)还使用 ≤ 并排除焊接对象，而论文阈值文字为 <。**这些是所检查代码路径的差异，未确认正式发表表格采用哪一路径，不能推断论文成绩错误。**

v4 选择显式适配口径：结构化逐约束汇总；严格论文阈值；地毯不进 COL/STB 的分子或分母；焊接对象不进 STB；公开适用、焊接、未知和 N/A 数量。统一引擎或导出造成的差异要披露。评测实现尚未落地，不能声称已复现 SceneSmith 分数。

ACC 与 Reach 也不能混用：SceneEval ACC 取最佳功能侧的局部空闲比例；OOB 用表面采样点向下落在地板上的比例判断。完整定义见 [SceneEval §3.4](https://arxiv.org/html/2503.14756v3#S3.S4)。两项保留在诊断表。

## 3. 五个候选基线的执行证据

| 工作 | 已核实证据 | 共同执行器之前还缺什么 |
|---|---|---|
| [SceneSmith，App. J、L](https://arxiv.org/html/2602.09153v2#A12) | Drake/SDF/DMD，导出 MJCF/USD；移动 KUKA＋抓取/IK/运动规划，有定量跨房任务 | 在选定统一引擎和执行器中重新验收 |
| [EmbodiedGen V2 v3，§2.5–3.4](https://arxiv.org/html/2607.07459v3) | 多房生成、多模拟器导出；VLA/RL 训练及迁移证据 | 训练提升不等于固定 Agent 的生成器 SR；另核原始[训练论文](https://arxiv.org/abs/2603.18532v4) |
| [SceneCode，§4.4](https://arxiv.org/html/2605.19587v1#S4.SS4)；[官方仓库](https://github.com/wangpuyi/SceneCode) | 可执行程序、SDF/DMD/状态 JSON；关节接触展示 | 未建立定量完整任务 SR 或跨房共同控制器 |
| [EmbodiedSmith，§4.2、8.4–8.6](https://arxiv.org/html/2610.07969v1) | Isaac Sim/CuRobo；固定场景下优化计划和初始站位，随后留出执行 | 不是固定完整策略的原生横向比较；未核实跨房共同接口 |
| [Code2Worlds-2，§V、App. E](https://arxiv.org/html/2610.17651v1)；[官方仓库](https://github.com/AIGeeksGroup/Code2Worlds-2) | MJCF 与任务 overlay；匹配提示下接触球代理控制，含辅助抓取约束 | 不能当完整机器人的同协议 SR；官方 demo 不等于完整实验复现 |

原生 simulator、任务范围和动作辅助均不同。因此候选行保留 **candidate adapter + NE**；先冻结可共同支持的任务子集，不能将“不支持”作为零分惩罚，也不能宣称都已复现。

## 4. 补搜到的相关评测与扩展先例

| 工作 | 对本项目定位的影响 | 证据位置 |
|---|---|---|
| **HappyWorld-Bench** | 已有空间扩展/旧区保持及独立的多步动作视频评测。B.6 的保持主要是视图质量；§5 为完整动作提示驱动的视频预测，未建立本表的联合闭环物理任务试验。不能借此反推本项目首创 | [Table 12、B.6；§5](https://arxiv.org/html/2609.24308v1) |
| **WorldArena** | 世界感知质量与下游功能效用并列已有先例；这种表格组织本身不是贡献 | [§3.1–3.2](https://arxiv.org/html/2602.08971v2) |
| **WorldWeave** | 持久显式几何的增量扩展已有近邻；场景规划 Agent 与任务执行机器人须区分 | [§3.4–3.5、§4](https://arxiv.org/html/2609.34221v1) |
| **WRBench** | 回访后的状态证据和可观测性已有专门评测；不能把画面保持等同物理状态保持 | [§3](https://arxiv.org/html/2606.20545v1) |

HappyWorld-Bench 的关键边界由主审与独立协议审查再次交叉核对。Oneira、Genie/SIMA、AgentOdyssey、Minecraft 等既有反例仍保留于[查新报告](novelty-audit-20261020.md)，没有因为接口不同而删除。

## 5. 现在可以相信到哪一步

**已确认：**核心指标有可访问的一手来源；列名、来源、继承/改编及主要排除项已写清；外部系统的不同执行证据不能混算。

**尚未确认：**共同适配器可运行；评测器实现符合本协议；g=4 可行；样本量充分；新系统在物理任务中优于对照；窄化问题具有首创性。以上不能由引用数量替代。

第一项实际验证仍是设计说明中的单次扩展任务，对照全量重生成＋合格状态恢复，并检查真实轨迹、状态和目标计分链条。此项属于下一轮实验，当前文献审核未执行。


文档验收：PDF 共 6 页，第一页完整主表、第 5 页来源对照；全部页面已渲染检查。19 条书目（17 篇论文、2 个固定版本代码来源）均有正文引用，无未定义引用；修改文档的本地链接已检查。此验收不等于评测代码或物理系统通过测试。
