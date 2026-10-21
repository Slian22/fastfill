# WorldEdge Table 1：世界质量与固定 Agent 的任务完成评测

研究日期：2026-10-20。用途：冻结下一轮论文评测设计；不是新增实验结果。

## 决策

Table 1 用同一张表的两个纵向 panel，共享方法行、冻结任务规格与运行预算：

- **A. World generation and verification**：共同报告 Verified-ready yield，以及 Spec / Physics / Access / Interaction / Persistence 五个分项。
- **B. Downstream task success with a fixed agent**：Navigation / Interaction / Cross-room / Revisit-state 四任务族，加 Macro E2E SR。

这张表回答：**同样的任务要求下，哪个完整世界生成管线交付的环境更能支持同一个执行器完成任务？** 它不直接证明 Agent 的能力通过训练提升。C1 课程与 C2 Playbook 的学习收益另做共同 held-out 集实验。

英文可编译模板：[WorldEdge-Table1-proposal.tex](../../../paper/WorldEdge-Table1-proposal.tex)。模板所有结果均为 NE（not evaluated），不从旧指标搬运数字，不把候选基线写成已完成复现。

## 1. 本次核查范围与现状

这是针对表格设计的定向文献核查，不是穷尽式系统综述。并行审阅了当前 pipeline、world-generation primary papers、embodied-task benchmarks，并独立检查实验设计。证据优先级：当前代码和实际运行产物 > 同期交付文档 > 旧规划；文献采用论文正文、正式 proceedings 和官方可执行规范，不采用二手解读作为论据。

当前实际链条：

```text
冻结任务 / reviewed template / C1 选题 + 持久世界
  → TaskPlan（对象身份、初始条件、目标、关节要求）
  → 房间与门连接
  → FastFill 尺寸 / bbox 布局
  → 任务前提、目标未提前满足、门洞等独立检查
  → RoomGenBench 资产与固定布局组装
  → Particulate 指定对象的关节化
  → agent-request
  → 外部具身策略与模拟器适配器
  → 可信轨迹 / 物理 / 任务验收
  → 世界 commit + C1 反馈 + C2 Playbook 门控更新
  → session 下一轮 / 重试 / 扩展
```

应区分三个 Agent：建世界的 LLM、canonical 世界中的符号 LLMSolver、真正加载 mesh/关节并行动的具身执行器。Table 1 B 指第三类。

| 证据 | 当前能支持什么 | 不能支持什么 |
|---|---|---|
| `EmbodiedGen/embodied_gen/world/pipeline/agent.py:1,12–34` | 执行器与验收器接口存在；调用方提供后端 | 仓库已内置并评测真实具身策略 |
| `EmbodiedGen/embodied_gen/world/pipeline/fastfill.py:485–489` | `bbox_and_intended_joint_initialization`；明确 `physical_task_readiness=unknown` | bbox 验收等价物理可执行 |
| `EmbodiedGen/embodied_gen/world/pipeline/session.py:30` 与 `outputs/rsi-loop-20261018/session-review.md:23,48` | scheduler、task bridge、注册和反馈继承已实现；测试用了物理 verdict 替身 | 旧文档所说 scheduler 仍缺失；或测试通过证明机器人成功 |
| `EmbodiedGen/embodied_gen/world/bench_report.py:201–207` | `completed_rate` 是 tier A symbolic validator 指标 | 机器人完整 task SR |
| `EmbodiedGen/embodied_gen/world/bench.py:696–718` | `route_completion` 是房型/对象命中 | 机器人真的走完路线或 SPL |
| `EmbodiedGen/outputs/rsi-full-20261018/live-rpm30/report.json:2249,2324,2358,2398` | 有真实 LLM 和符号工具执行；总体 incomplete、完整 paired block 为 0；scope 明确是 symbolic、robot 未测 | 完整 C1/C2 具身效果表；仅凭单臂 held-out 25/34 宣称优胜 |

上述路径相对 `/Users/slian/Desktop/3D/Worldedge/`。现有 FastFill 的 8,647 房间 Complete/GSR/strict/OOR，以及旧系统 1,005 房间 mesh/FID/KID，属于组件或条件子集证据。新 Table 1 必须同一完整系统、同一任务 cohort 重测，不能拼接这些数字。未找到完整、同协议的生成 mesh/关节世界上的具身任务完成对比；这不等于没有任何 LLM/符号执行记录。

## 2. Panel A 的列和判定规则

所有阈值、机器人形态、物理引擎、碰撞模型、probe、采样规则先在独立开发集确定并冻结。下列是 WorldEdge 的设计建议，不宣称为某个外部 benchmark 的原样指标。

| 英文列名 | 逐请求的通过条件 | 特别防止的漏洞 |
|---|---|---|
| **Verified-ready yield ↑** | 世界物化且所有适用的 Spec、Physics、Access、Interaction、Persistence 检查均获得通过证据 | 必须逐请求求逻辑 AND，不能乘各列均值，也不能把未经检查的条件当通过 |
| **Task-Spec Pass ↑** | 所有必需实例、类别/数量、稳定 ID 绑定、初始关系/状态都正确；目标起初为 false；保留冻结的目标与保护约束 | 不能删难物体、改题、提前开柜/把物体放到目标位置来提高 SR |
| **Physical Validity ↑** | 实际导入并短时仿真；必需物体有有效碰撞体、支撑和尺度；无超阈值严重穿透、漂浮、失稳 | 不能以 2D overlap、GLB 成功导出或 JSON 检查代替；不能删掉掉落物体后只评幸存物 |
| **Target Access ↑** | 指定机器人从合法起点可到达任务所需的所有观察/交互区域；检查跨房间门洞、连通性、障碍和形态 clearance | 最大连通地面占比可能很高，但实际目标仍不可达；房间图 BFS 不是机器人导航成功 |
| **Interaction Verified ↑** | 在同一交付世界的独立 reset/clone 上，以固定参考脚本或控制器完成必需的局部动作 probe，如开到指定关节角、取出、放稳 | 存在 joint / `openable=true` 不等于能操作；scripted probe 成功是正向证据，失败不证明无解 |
| **State Preservation ↑** | 对固定离开/扩展/回访 probe，按事件日志核对旧 ID、资产身份、位姿、关节状态、containment 与拓扑；合法动作变更须保留，其他状态不得被错误改写 | 不是要求回到最初状态；不能只比较文本摘要、同一 JSON 或图像相似度 |

**分母**：Spec、Physics 对全部预注册请求；Access、Interaction、Persistence 对各自生成前定义的适用请求集合，分母对所有方法相同并公开。生成/导入失败保留在相应分母中，计未通过。适用性不能由生成结果决定。Verified-ready 对全部请求求联合认证比例；只对事先不适用的条件设逻辑真，不能把执行失败或未检查解释成不适用。

**认证与运行门控分开**：Verified-ready 是独立认证的正向证据产出率，不是绝对可解率；通过若干原子动作 probe 也不能证明整段长任务可解。不强制把所有 world 诊断指标都变为策略执行门控。若系统确有 mandatory admission gate，另记录其 admission yield；未通过该门且没有执行是系统端到端失败。参考 probe 未成功但策略实际成功，二者可以并存，不能强行令 Agent SR ≤ 独立认证率。

**覆盖率必须可见**：每个方法另附 materialized/requested、admitted/requested、actually-run/requested 的计数。模板表下注释要求这三项；完整评测发布时附同页简短 ledger 或补充表，并在主表明确引用。

### 为何不是 FID/GSR/OOR 六项堆在左侧

SceneEval 将显式要求与碰撞、支撑、可达等隐式可用性分开；FID 无法检查指定杯子/柜门是否存在和可用。SceneSmith 加入真实物理仿真；SceneCode 和 EmbodiedGen V2 强调资产/实例身份和局部编辑，但它们没有提供可直接移植的长期 state-preservation 总分。因此我们应定义自己的持久状态试验，并明确它是新协议。

FID/KID、统一渲染的盲评 realism/faithfulness、穿透深度、OOB、support、SceneEval NAV/ACC、目标级可达率、尺寸误差放附表与定性图。视觉质量仍需评测，只是不作为本篇“任务可用且持久世界”的主表中心。若论文主张包括视觉 SOTA，则必须提升为独立主要实验。

旧 OOR 需写成 **2D Overlap**：OptiScene OOR 指重叠比；SceneEval OOR 指 object–object relationship，同名不同义。

## 3. Panel B：固定执行器，四任务族与一个主终点

| 英文列名 | 任务设计 | 真实成功的依据 |
|---|---|---|
| **Navigation SR ↑** | 无操作、无回访的目标搜索/到达 | 到达符合冻结距离、视角/可见性和 stop 规则的目标位姿 |
| **Interaction SR ↑** | 单房间拾取放置 / 开关门抽屉 / 容器交互 | 指定实例和关节/物理状态正确；放置后保持规定驻留时间；不能依靠 teleport/snap 冒充真实控制 |
| **Cross-room SR ↑** | 无历史回访的跨房间运输与组合目标 | 全部目标、动作依赖与保护条件满足；走到房间不算完成取放 |
| **Revisit-state SR ↑** | 改变状态→离开/扩展→返回→使用变化后的对象/状态完成任务 | 同一身份、历史状态、操作顺序和最终保持同时正确，验证器读取完整状态轨迹 |
| **Macro E2E SR ↑** | 四任务族等权平均 | 衡量固定策略下完整世界管线的任务效用，是本表的主要终点 |

这些任务族是面向本 pipeline 的建议，不是公认统一 benchmark。为避免重复计数，预注册互斥分类优先级：Revisit-state > Cross-room > Local interaction > Navigation。任务长度另按原子动作数/依赖深度分层，不能简单把 cross-room 等同所有 long-horizon。局部交互附表再分 pickup/place 和 articulation。

任务是否成功必须由冻结的独立 ExecutionValidator 从 simulator state/trajectory 判断。全部目标为真还不够：要求初始未完成、同一对象、时序依赖、保护约束和必要的最终保持。PARTNR 提供整段轨迹上的 proposition/dependency/constraint 验证先例；ALFRED 区分完整 task success 与部分 goal-condition success。WorldLines 的 planning-judge 分不是机器人动作执行率。

Panel A 的单位是唯一 world-generation request；Panel B 给每个世界固定 K 个 executor seeds，N_f 是该任务族预注册的 episode slots（世界请求数 × K）。一个世界生成失败对应 K 个失败 slot，但这些不是 K 个独立世界。两张 panel 都要公开唯一世界数和 episode 数，bootstrap 保留世界内嵌套。对方法 m、固定执行器 π、任务族 f：

```
E2E-SR(m, π, f) = 成功且满足冻结任务合同的完整执行数 / 预注册 episode slots N_f
Macro E2E SR = (SR_nav + SR_interaction + SR_crossroom + SR_revisit) / 4
```

生成失败、缺必需资产、错误初始状态、导入失败、系统预算超时、实际执行失败都在系统端到端分母中。不能把它们全部归因为 Agent 能力失败。

另报 `SR | admitted` 作为执行器诊断，并给 admission yield。若 gate 与计数完全匹配，在每一个任务族内 E2E SR = admission yield × conditional SR；**macro 平均的乘积不等于各族乘积的 macro 平均**。不同方法被接纳的世界集合不同，conditional SR 也不是纯策略能力的公平排名。某任务族 A_f=0 时，E2E SR=0，conditional SR 未定义，标 N/A；不能计算 0/0。

基础设施外因故障单列 unknown，按预注册的统一重试规则处理，不伪装成策略失败也不静默删样本。若任务族 f 仍有 U_f 个未决 episode，报告该族 `[S_f/N_f, (S_f+U_f)/N_f]`；Macro E2E 的界为 `[mean_f(S_f/N_f), mean_f((S_f+U_f)/N_f)]`，不能用 pooled bounds 代替。它们是未知结果的识别界，不是抽样置信区间。公开各族 U_f，不能声称完整配对评测已经完成。

## 4. 行怎么选、怎么公平比较

**主表行是完整世界生成配置，不是执行 Agent 型号。** 同一 panel 内固定执行器 checkpoint、prompt、memory、传感器、动作 API、技能实现与预算。第二个不同架构执行器放 robustness 矩阵；不能每个方法挑最有利的执行器。

候选完整系统为 SceneSmith、EmbodiedGen V2、必要时 SceneCode/Holodeck；先审查同一任务与模拟器接口能否成立，再冻结 baseline roster。当前修改过的 EmbodiedGen fork 不能不加说明地冒充原版 baseline；固定官方版本/commit，列出适配部分。

分开两种实验：

1. **完整系统比较**：固定输入任务合同、资源来源和总预算，保留各系统 native 生成策略；明确 native/retrieval/generated 资产差异。这衡量系统效用，不能将差异全部归因 FastFill。
2. **组件消融**：同一 WorldEdge 后端与资产，比较 full / one-shot / without layout refinement / without articulated assets / without state persistence。任务保持不变，不因移除模块而删掉该模块必需的任务；移除 articulation 后“打不开”是该消融在 mandatory track 的真实失败。

两组不要混成一个“相同控制变量”的排行榜。模板用不同 row groups 标识。没有评测或未进入某 track 用 NE/N/A；已加入该 mandatory track 而实现失败用 0/失败记录，不能改成 N/A。缺某任务族的系统不得只平均剩余任务族后发布可比较 Macro SR。

外部系统不支持持续扩展时有两种正当处理：进入明确接受该要求的完整系统 track 并计能力缺失；或仅比较共同静态能力并另开 persistence track。不能对每个系统偷偷换成不同的任务集。

**难度控制在生成前**：冻结对象数/密度范围、合法初始状态、必需房间转换、依赖深度、history lag、可见性和初始目标 false。生成后另报告实际 geodesic distance、clutter、物体尺度与关节需求，检查是否简化任务。不能事后只选择成功且“差不多难”的世界。

## 5. 统计、规模和诊断

按同一个 task card × world seed × executor seed 配对运行方法，按预先确定的 task-template / house-world grouping 聚类或分层 bootstrap。重复动作、同一世界多任务和多次回访不能当独立 N。若 task template 与世界家族交叉，分析模型也须处理这两层依赖，不能只按逐 episode IID 抽样。

预注册主要比较：full WorldEdge 对选定 strongest feasible baseline 的 Macro E2E SR 差值及 95% CI。其他大量列/消融若作确认性胜负判定，控制 multiplicity；否则明确探索性并报告全部结果。主表点估计之外应提供配对 CI，不靠加粗单个最高数宣布显著优势。

不凭空指定“100 个就足够”。先用不进入测试集的小规模 pilot 估计跨方法 discordance、world/template 方差与基础设施失效率，再以预设最小实际重要差异做 cluster-aware power/precision 规划。运行预算应按独立世界/任务模板数确定，而不是将同一世界多 seed 当独立样本凑 N。

附表：Goal/Proposition Completion、Navigation SPL、simulated time/action count、失败分类、world/asset/task composition 的 unseen split、规模/历史跨度分层。SPL 仅用于路径任务，不能把跨房间操作统一改成 SPL。

成本另报：从 task/world-delta 请求到可交付状态的 wall-clock p50/p95、总 GPU/API 成本、失败与重试成本、每个 verified-ready request 的总成本比。固定 warm/cold cache、硬件、并发和预算；不能把旧 O2-only 约46 GPU-hours当完整系统成本。

## 6. C1/C2 的第二个问题

Table 1 评的是生成世界支撑固定策略的能力。C1/C2 若主张提升任务能力，另表固定初始策略、训练/经验预算和最终 held-out 测试世界，比较 no adaptation / random curriculum / C1 / C2 / C1+C2。测试时冻结 Playbook，不把 test feedback 写回后再对同一集算提升。若允许 test-time adaptation，作为另一协议且所有方法同预算。

PAIRED 的启发是困难且可解，而不是“让当前 agent 的成功率越低越好”；反过来，当前 agent 在自产简单题上 SR 更高也不是学习收益。当前 C2 是 Playbook 文本更新，不能称权重训练；symbolic LLMSolver 与物理执行收益必须各自命名。

## 7. 核查的一手来源与采用边界

以下仅概述与决策直接相关的证据。建议指标的阈值、合取方式与四任务族划分是本方案提出的设计，不是直接复制原论文。

| 来源 | 核查位置 | 对本设计的依据及边界 |
|---|---|---|
| [SceneEval](https://arxiv.org/html/2503.14756v3) | §3.3–3.4、Table 4 | 分离显式要求与空间 plausibility；NAV 不是特定目标到达率 |
| [SceneSmith](https://arxiv.org/html/2602.09153v2)；[ICML proceedings](https://proceedings.mlr.press/v306/pfaff26a.html) | Table 2、Appendix M.2.1 | 增加物理碰撞与短时稳定性；3D 物理检查不能用2D重叠代替；密度影响比较 |
| [EmbodiedGen V2](https://arxiv.org/html/2607.07459) | §2.6、§3.3/Table 4、§3.5/Table 5 | 状态保持编辑、人工 world acceptance 与 downstream study分开；不能把自动strict改名为其人工UR |
| [SceneCode](https://arxiv.org/html/2605.19587) | §3.4、Table 1、§4.4、Appendix D.1 | 实例/资产/仿真身份关联；交互演示不等于大规模任务完成率；opening clearance不等于柜门关节成功 |
| [Holodeck](https://arxiv.org/html/2312.09067) | §4.1、§4.3、Tables 1–2 | 资产/布局控制与下游ObjectNav评测分开；人类布局偏好不同于Success/SPL |
| [Infinigen Indoors](https://arxiv.org/html/2406.11824) | §4.1、Appendix H/Table 9 | 视觉与布局 realism；renderer/lighting/assets差异影响感知比较 |
| [InfiniteWorld](https://arxiv.org/html/2412.05789) | §4.4、Tables 2–5 | 导航和操作等任务分开；不是持续增长 persistence通用协议 |
| [OptiScene](https://arxiv.org/html/2506.07570) | Appendix D.2 | 2D bounding-box overlap；其OOR与SceneEval关系指标不可混称 |
| [ALFRED](https://arxiv.org/pdf/1912.01734) | §5.1、Table 3 | binary task success与partial goal-condition success分开 |
| [HomeRobot](https://arxiv.org/html/2306.11565) | Appendix C.1 | FindObj→Pick→FindRec→Place的端到端与阶段成功；明确其仿真Pick snap动作抽象 |
| [PARTNR](https://arxiv.org/html/2411.00081) | §3.2、Appendix A.5.5 | 从全状态序列评谓词、依赖、身份和最终保持；支持跨房间/回访验证 |
| [BEHAVIOR-1K evaluation](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/docs/challenge/evaluation.md)；[tasks](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/docs/omnigibson/tasks.md) | challenge metric与PredicateGoal | 某些官方success score实为部分谓词完成比例；不要直接当binary SR |
| [∞-THOR](https://arxiv.org/html/2505.16928) | §5.2、Table 4 | 区分离线memory QA、在线行动、expert history后末任务和完整episode |
| [WorldLines](https://arxiv.org/html/2606.18847) | §5.4、planning appendix | 持久状态/记忆启发；planning judge probe不是机器人物理执行证明 |
| [Navigation evaluation / SPL](https://arxiv.org/pdf/1807.06757) | §4 | 以成功与路径效率共同评分；不直接适用于统一的所有操作任务 |
| [PAIRED](https://arxiv.org/html/2012.02096) | §4 | curriculum的困难-可解平衡与held-out transfer；本项目未据此声称复现原算法 |

## 8. 现在可以做和不能填的格子

可立即冻结任务合同、指标字典、失败码、baseline adapter要求和统计计划；现有组件结果继续作为 FastFill/mesh 的独立补充表。真实执行适配器接通后，先验证少量真实轨迹的人工/程序判定一致性，再开展同 cohort、同预算的完整实验。

没有执行数据的指标一律 NE。`strict`、symbolic `completed_rate`、route命中、接口单测数量、launch记录，以及不同cohort的最佳数值，都不能填进新agent主表。
