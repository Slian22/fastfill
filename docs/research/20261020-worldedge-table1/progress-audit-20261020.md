# WorldEdge 大表：指标、现有结果与剩余工作

核查日期：2026-10-20。范围：本地 OptiScene、EmbodiedGen、worldedge 的当前文件与保存的运行产物；重新核对一手论文/官方指标说明。没有查询远端实时队列，没有启动训练或付费实验。当前结果不能解释成远端任务的实时状态。

**当前阶段：大表设计已有草稿，组件评测有结果，完整生成世界上的真实 Agent 同协议评测尚未完成。** 仿真和执行桥接已有实现，并非从零开始；缺的是接通固定控制器、可信轨迹验收及完整对照实验。

**文献边界更新：** 本文件审计本地进度，不证明该方向没有先例。[后续查新与交叉审核](novelty-audit-20261020.md)发现 Oneira、Genie/SIMA、AgentOdyssey、SimWorld 等直接相关工作；“首次可扩展世界＋Agent 任务”不能作为贡献。Retain/Revisit 是拟议具体协议，状态持久化和回访评测本身已有文献。原候选行还应审查 EmbodiedSmith、Code2Worlds-2，并补充预生成及重生成恢复对照。

当前设计已更新为 [v3 两个 panel](table-design-v3.md)，查看 [PDF 第一页](../../../output/pdf/WorldEdge-Table1-proposal.pdf)及[可编辑 LaTeX](../../../paper/WorldEdge-Table1-proposal.tex)。以下指标部分保留审计时的 v2 快照（8 行、14 列），不再代表当前排版。实现与运行证据仍有效；全部新表结果仍为 NE，不能读成零分。

## 1. 审计时的 v2 指标快照（当前设计见 v3）

同一行表示一个完整世界生成系统。左边看世界，右边看同一个固定 Agent 在该世界中的任务表现。当前模板采用以下列：

| 维度 | 指标 | 实际回答的问题 | 当前可填状态 |
|---|---|---|---|
| 世界：交付 | Load ↑ | 请求的世界能否真正导入统一模拟器？ | NE；布局 JSON 完成率不是 Load |
| 世界：语义 | CNT ↑ | 要求的物体类别与数量是否满足？ | NE；需对最终交付世界评测 |
| 世界：语义 | ATR ↑ | 材质、颜色、尺寸等要求是否满足？ | NE |
| 世界：关系 | OORel ↑ | 物体之间的空间关系是否正确？ | NE；不同于 FastFill 的 2D Overlap |
| 世界：关系 | OARel ↑ | 物体与墙、角落、房间等建筑关系是否正确？ | NE |
| 世界：物理 | COL ↓ | 真实碰撞几何是否发生过量穿透？ | NE；旧 mesh CF@500 不等于这项对象级指标 |
| 世界：物理 | STB ↑ | 物体经短时物理仿真后是否稳定？ | NE；几何支撑检查不能替代 |
| 世界：边界 | OOB ↓ | 物体是否越过合法场景边界？ | NE；旧检查器口径不同 |
| 世界：空间使用 | ACC ↑ | 物体功能侧是否有可用空间？ | NE；这是空间代理指标，不是机器人到达率 |
| 世界：持续状态 | Retain ↑ | 修改后经历扩展或卸载重载，身份、位姿、关节与包含关系是否保留？ | NE；已有软件快照/恢复测试，不等于此物理试验 |
| Agent：完整任务 | E2E SR ↑ + 95% CI | 从生成请求起算，最终真正完成任务的比例？ | NE；确认的生成/导入失败保留在分母 |
| Agent：部分进展 | GC ↑ | 在身份、时序等约束下完成多少目标？ | NE；符号求解器完成率不能搬入 |
| Agent：回访任务 | Revisit SR ↑ | 修改、扩展/重载、返回之后能否利用正确的历史状态完成任务？ | NE；独立的 R 任务集 |
| Agent：资源 | Actions given success | 成功任务用了多少动作？ | NE；需同时给成功数，只作资源诊断 |

世界指标主要依据 [SceneEval 官方实现](https://github.com/3dlg-hcvc/SceneEval) 和 [SceneSmith 论文](https://proceedings.mlr.press/v306/pfaff26a.html)。成功率、部分目标完成和行动资源的区分参考 [PARTNR §3.2、§4.2](https://arxiv.org/html/2411.00081)。Load 是本方案的交付记账指标；Retain、Revisit，以及全请求分母的扩展也是本项目拟议协议，不是这些论文已经建立的通用标准。

当前候选行是 SceneSmith、EmbodiedGen V2、SceneCode、WorldEdge，以及 WorldEdge 的 one-shot、去布局优化、去关节资产、去状态保持四个消融。前三个外部系统仍需完成适配可行性审查，不能标记为已复现。

共同任务集 C 包含导航、单房间交互、跨房间任务；持续扩展/重载回访使用独立任务集 R。任务类型用于分层报告。没有实现 R 的外部系统不进入 R 排名，也不与不同任务集合的总平均混排。正式冻结时还需预注册 C 的共同支持范围，并在 R 中区分 reload-only 与实际新增空间：预生成世界的序列化成功不能解释成增量生成能力。

## 2. 当前真实进度

| 工作层 | 已核实的进展 | 结论边界/未完成项 |
|---|---|---|
| 调研与表格 | 本地保存 20 份论文 PDF，已有 3 份专项审阅；最新 LaTeX 包含 10 个世界列和 4 个 Agent 列 | 下载数量不代表所有论文已同等深度审阅；这是有针对性的核查，不是穷尽式系统综述 |
| FastFill 布局 a027 | 8,647/8,647 完整；GSR 89.55%；strict admission 49.42%，旧 a022 为 34.35%（增加 15.07 个百分点） | 已完成的历史组件评测；strict 不是物理世界可用率或 Agent SR |
| 尺寸估计 | 1,877 请求；train-only prior log-MAE 0.267445，L-size 0.327269，O2-size 0.408010 | O2-size 仅 1,876 个完整输出；不是新 Table 1 的任务结果 |
| Mesh/视觉 | 1,005 个分析房间；raw 27B-P / 旧系统 S 的 CF@500 为 57.51% / 77.51%；FID 为 22.883 / 22.786 | 原选取 1,009 个，4 个缺资产；S 是旧 checker-trigger 系统，不能写作 a027 的视觉结果 |
| 仿真与执行桥接 | 已有 MuJoCo 场景加载/步进、Drake 物理检查、RobotSolver 外部策略桥接 | 代码存在不等于交付 mesh/关节世界已通过同协议任务试验；内置 oracle 会 teleport，无机械臂抓取 |
| 持久世界与会话 | 调度、重试、恢复、快照与验收入口已实现 | 集成证据使用 fixture physical verdict，尚不能填 Retain/Revisit |
| C1/C2 Agent 实验 | 有真实 LLM + 符号工具记录；C2 有单臂 held-out 25/34 符号成功记录 | 总报告 complete=false，完整配对比较块为 0，physical_robot_execution_measured=false；不能声称具身学习收益 |
| 新大表完整对照 | 指标、失败记账、固定执行器与统计要求已有设计 | 未找到完整、可比较的物理执行结果；14 列均 NE |

### 数值证据

- 布局：[a027 levels.csv 第 4 行](../20261013-unknown-size-comparison/evidence/a027-resume-20261015/completed/eval/levels.csv)、[a022 levels.csv 第 4 行](../20261013-unknown-size-comparison/results/system-022-test-sharded/levels.csv)。a027 完成收据已于 10 月 16 日核验；不是仍停在提交阶段。
- 尺寸：[sizes-020 levels.csv](../20261013-unknown-size-comparison/results/sizes-020/levels.csv)。
- Mesh/视觉：[审定视觉表](../20261013-unknown-size-comparison/paper-draft-20261020/table-visual-en.tex)、[mesh A 报告](../20261013-unknown-size-comparison/results/visual-024/mesh-report-A.md)、[mesh S 报告](../20261013-unknown-size-comparison/results/visual-024/mesh-report-S.md)。CF@500 是容忍 500 cm³ 体素交叠的房间级指标，仅涉及合格的可移动物体对，不等于严格零碰撞。
- 解释边界：[metric-contract.md](../20261013-unknown-size-comparison/paper-draft-20261020/metric-contract.md)。历史测试集已曝光；现有数值是组件证据，不能拼成一次新的盲测。
- 27B-J/a028：[最后启动记录](../20261013-unknown-size-comparison/evidence/a028-launch-20261016/REPORT.md)与[10 月 20 日产物审计](../20261013-unknown-size-comparison/artifact-audit-20261020.md)未给出可核验的完成分数；本次未验证远端是否仍在运行。

### 实现与实际运行证据

路径相对于 `/Users/slian/Desktop/3D/Worldedge/`。

| 证据 | 能确认的事实 |
|---|---|
| `EmbodiedGen/embodied_gen/world/pipeline/agent.py:1–34`；`fastfill.py:488–489` | 管线调用外部执行器/验收器；当前 bbox 验收明确记录 physical_task_readiness=unknown |
| `EmbodiedGen/embodied_gen/world/robot_eval.py:604,873`；`sim_scene.py:653` | 存在策略桥接与 MuJoCo 实现；oracle 的 teleport/未仿真 containment 必须与真实控制区分 |
| `worldedge/scenesmith/agent_utils/physical_feasibility.py:1423` | 存在 Drake 正向物理仿真实现 |
| `EmbodiedGen/embodied_gen/world/pipeline/session.py:237`；`outputs/rsi-loop-20261018/session-review.md:48` | 软件编排与恢复已实现；现有集成测试用了物理判定替身 |
| `EmbodiedGen/outputs/rsi-full-20261018/live-rpm30/report.json:2249,2324,2358` | symbolic 实验不完整、没有物理机器人测量、配对比较块为 0 |

## 3. “整个世界”还需要明确展示的内容

以下建议针对审计时的 14 列草稿，已由 v3 吸收并调整。若论文强调持续增长的完整世界，仅有单场景质量和 Retain 仍不足。配套信息如下；这些是待冻结设计，不是已经测出的数值。

| 补充维度 | 建议展示 | 放置位置 |
|---|---|---|
| 世界规模、增长与持久状态 | 房间数、物体数、面积/密度、扩展轮次与历史跨度；每轮扩展成功、ID/拓扑/合法状态漂移 | 主表旁的世界统计及 R track 分层结果；Retain 为主列 |
| 全局空间连通 | 要求的房间/门/连接是否齐全；从固定起点到任务目标的可达率、机器人 clearance | 任务合同纳入建筑要求；附表诊断，不能用局部 ACC 或 OARel 代替 |
| 动作中的物理交互 | 关节运动扫掠空间、接触失败、受保护物体扰动；区分抓取和关节操作 | 主任务成功包含保护约束；配套失败诊断，初态 COL/STB 不能替代 |
| 视觉与分布覆盖 | 盲评 realism/faithfulness；不同布局、资产、任务组合的 held-out 分层 | 独立视觉/泛化表；有视觉优势主张时应升为主要实验 |
| 系统成本与失败 | 请求/加载/接纳/执行/成功/未知计数；生成和扩展 p50/p95 时延、总成本及重试成本 | 必需记账表与成本表；不能只报成功样本动作数 |

## 4. 下一个实验与完成条件

优先跑通**一个冻结的双房间物理回合**：开容器 → 取物 → 跨房间搬运 → commit → 实际扩展或卸载重载 → 回访验证。

1. 固定实际 mesh/关节、任务初态、对象身份、控制器和动作预算。
2. 保存完整模拟器轨迹与对象/关节状态，用独立验收器判断目标、时序及保持条件。
3. 验证全流程不依赖 teleport 或 fixture verdict；把失败和未知保留在执行账本。
4. 小规模 pilot 确认可运行且判定一致后，冻结外部基线、任务样本与统计方案，再做完整对照。

这个回合通过只能证明链路成立，不能直接声称全系统完成或论文结果成立。C1/C2 的学习收益仍需固定初始策略、相同经验预算和共同 held-out 世界另测。

## 5. 本次整理与验证

README 的早版方案使用 Ready/Spec/Physics 等摘要分项以及四任务族 Macro；最新 LaTeX 已改为上面的横向 14 列，并分 C/R 两套任务。本次给 README 增加版本提示，保留旧讨论供追溯。

原 LaTeX 标题行误把换行写成显示数学入口，导致编译失败；已修正换行及不存在的审阅索引路径，并给 Load 补上本方案指标标记。PDF 编译成功，共 4 页，逐页检查无裁切/重叠，第一页完整呈现大表。未改实验数字、模型代码或训练配置。

并行审阅分工：一个 subagent 核对物理执行与持久世界链路，一个核对组件结果及 cohort，一个独立检查整世界指标覆盖和 C/R 比较设计。上述结论已综合三份审阅。
