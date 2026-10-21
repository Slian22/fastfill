# FastFill 贡献与新颖性独立审查（2026-10-13）

**判定：研究问题成立；当前尚未证明新的性能贡献。** “未知尺寸”“联合预测尺寸与位姿”“用 LLM 生成三维布局”均已有明确先例，不能作为首创。应分别审查 **A：结构化 FastFill v2 的方法候选贡献** 与 **B：正在运行的文本 J0/P0 受控比较**。B 的结果不能替 A 的双向解码器、数值头或几何损失作证。两条路线都有继续研究的理由，但当前不支持整体优于 OptiScene、优于一般两步法或已达到可部署质量的表述。

本报告直接读取本地原始代码、协议、历史设计及论文，独立形成判断；借鉴固定版本 ARIS 的查新、实验边界、结果到主张方法。**不是正式跨模型家族 ARIS verdict**：可用 Agency 角色继承同一模型，本报告没有调用或伪称异构外部评审。没有重跑训练、指标、服务器查询或质量评分；只新增本文，`current-results.md/json` 由父任务维护。

## 1. 两条研究线及证据状态

| 项目 | A：结构化 FastFill v2 | B：clean unknown-size 文本 J0/P0 |
|---|---|---|
| 输入／任务 | 房间、固定对象、请求清单、可用支撑／关系条件；每请求一个对象 slot | 共同房间／固定对象／类别清单；J 不接收可移动目标尺寸，P 接收 train-only 估计尺寸 |
| 模型与输出 | Qwen 条件编码；外部双向对象 decoder；尺寸、位置、yaw 数值头 | 同一通用 Qwen3-8B 起点的两条自回归 JSON SFT；J 输出 size/pos/yaw，P 输出 pos/yaw，系统尺寸来自先验 |
| 监督 | 几何损失、字段有效性 mask、合法可交换组 matching | assistant-only token CE；共同长度准入；P 的摆放示范适配估计尺寸 |
| 已有证据 | 代码与历史训练／评测记录存在；部分质量问题已有负面记录 | 正式流水线运行中；共同实际 SFT 98,754 行；正式 test 全 18 来源、8,647 请求 |
| 尚缺什么 | 结构化与文本的真正匹配对照、核心机制消融、完整条件留出质量证据；真实资产／物理／Host 闭环 | J0/P0 正式质量结果尚无；强尺寸预测基线、训练随机性及机制归因仍未解决 |
| 当前能说 | 已实现一个 request-bound 结构化布局预测原型，并有历史实验 | 已建立同基座的联合生成与训练集尺寸先验摆放对照；待检验两系统取舍 |

代码分界很明确：[v2/model.py](../../../../fastfill/v2/model.py) 125–160、183–227 行使用 `AutoModel` 特征骨干、`TransformerDecoder` 与几何 heads；[comparison/train_pair.py](../../../../fastfill/comparison/train_pair.py) 166–176、203–246 行加载 `AutoModelForCausalLM` 并做 LoRA SFT；[train_pair_helpers.py](../../../../fastfill/comparison/train_pair_helpers.py) 242–259 行只监督 assistant token。**J0/P0 并未实例化 v2 的 `StructuredFastFillModel` 或 `GeometryCriterion`，没有证据表明这轮评估了 v2 heads。**

状态依据是[当前实测报告](../current-results.md)、[协议](../protocol.md)与[执行锁](../execution-lock.json)。本报告沿用父任务直接 SSH 实测并提供的运行状态，不把它改称“用户转述”，也不假称本审查重新登录服务器验证。质量列仍未测；finite loss、测试通过、覆盖率和 driver 存活只支持相应工程状态。

## 2. 一手相关工作：哪些宽泛主张已经被覆盖

“联合”在下表表示尺寸与位姿属于同一生成任务，不意味着所有工作都采用单次并行前向。类别／数量是模型生成还是外部给定，也必须区分。

| 工作与一手来源 | 已覆盖的内容 | 对本项目的具体约束 | 来源质量／阅读范围 |
|---|---|---|---|
| **SceneFormer**，3DV 2021；[论文](https://arxiv.org/abs/2012.09793)、[作者页面](https://xinpeng-wang.github.io/sceneformer/)、[官方代码](https://github.com/cy94/sceneformer) | Transformer 顺序生成对象；官方训练入口明确区分 `cat/dim/loc/ori` | 类别、尺寸、位置、朝向生成及注意力对象关系不是新概念；其顺序生成不同于 v2 并行 slot | 作者论文摘要、项目说明、代码 README；足以核对任务先例，未复现结果 |
| **ATISS**，NeurIPS 2021；[论文](https://arxiv.org/abs/2110.03675)、[作者方法页](https://research.nvidia.com/labs/toronto-ai/ATISS/)、[官方代码](https://github.com/nv-tlabs/ATISS) | 对象表示含 category/size/location/orientation；按部分场景自回归生成，并支持补全和指定类别对象放置 | 不能称首次联合几何生成、首次以已有对象为条件；固定清单也不自动排除其交互式应用 | 正式论文身份与作者方法说明；没有独立重训 |
| **LayoutGPT**，NeurIPS 2023；[正式论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract.html)、[附录 C](https://proceedings.neurips.cc/paper_files/paper/2023/file/3a7f9e485845dac27423375c934cb4db-Supplemental-Conference.pdf)、[官方代码](https://github.com/UCSB-AI/LayoutGPT) | LLM 通过示例与结构化样式语言生成布局；三维对象明确含类别、位置、尺寸、朝向；三维条件为房型／房间尺寸 | “LLM 预测未知尺寸三维布局”不是首次；我们的 SFT、数据与身份合同有差异，但差异本身不是性能证据 | 正式 proceedings 与其索引到的附录原文；整本 PDF 读取受体积限制，未声称通读 |
| **DiffuScene**，CVPR 2024；[正式论文页](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_DiffuScene_Denoising_Diffusion_Models_for_Generative_Indoor_Scene_Synthesis_CVPR_2024_paper.html)、[论文全文](https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_DiffuScene_Denoising_Diffusion_Models_for_Generative_Indoor_Scene_Synthesis_CVPR_2024_paper.pdf)、[官方代码](https://github.com/tangjiapeng/DiffuScene) | 无序对象集合的尺寸、位置、朝向、语义与几何特征扩散，随后检索资产 | 联合对象属性和全场景交互已有先例；与我们固定请求、确定性数值头／文本 SFT 是不同生成机制 | 正式 CVF 摘要、方法及局限节选；不搬用其成绩作本地基准 |
| **InstructScene**，ICLR 2024；[正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/6d96950e6ce0b92e3d0fb2afa634e0f2-Paper-Conference.pdf)、[作者方法页](https://chenguolin.github.io/projects/InstructScene/)、[官方代码](https://github.com/chenguolin/InstructScene) | 语义图先验与布局 decoder 分工，图 Transformer 生成尺寸、位置、朝向 | 语言／关系条件加独立布局 decoder 不是空白；我们的稳定请求身份与异质字段监督仍需独立证明价值 | 正式论文与作者明确的两阶段定义；不将后续扩展版混为独立重复证据 |
| **DeBaRA**，NeurIPS 2024；[正式论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c55aeaa3a85954615319a7e55c81197a-Abstract-Conference.html)、[作者论文 §3.1–3.3](https://arxiv.org/html/2409.18336v1) | 从 floor plan 与给定语义类别集合生成 position/rotation/dimensions；采用语义感知 Chamfer 处理同类对象置换 | **连“给定清单、未知尺寸、共同生成几何”这一任务收窄也已有很近先例**；不能只与已知尺寸 OptiScene 对比就证明新任务 | 正式会议身份、方法原文；是本次最直接的固定语义集合任务先例 |
| **OptiScene**，NeurIPS 2025；[arXiv v2](https://arxiv.org/abs/2506.07570v2)、[官方代码](https://github.com/PolySummit/OptiScene) | §3.1 先检索资产及 bbox 尺寸，再预测位置／朝向；另有 reasoning SFT 和偏好优化 | 未知尺寸是相对该流程的信息条件改变，不是整个领域首次；P0 是适配基线，不是作者模型 | 本地全文与官方页面交叉核对；README 的 Qwen2.5 配方、Coming Soon 权重不等同论文 Qwen3 配方 |
| **SceneNAT**，2026；[v2 论文](https://arxiv.org/html/2601.07218v2)、[版本／状态](https://arxiv.org/abs/2601.07218) | masked 非自回归 Transformer、少量并行解码轮次、属性／实例 masking、关系辅助监督 | “非自回归／双向对象交互＋mask＋快速场景生成”也不能单独作为 v2 首创；其离散多轮生成不同于当前数值头 | 2026-08-11 v2；官方标注 under review。方法披露是有效先例，效果仍是作者报告 |
| **CasLayout**，2026；[论文](https://arxiv.org/html/2604.27361v1)、[元数据](https://arxiv.org/abs/2604.27361) | 将数量／类别、尺寸／特征、关系 latent、OBB 分阶段处理，并纳入建筑条件 | 分解建模并非天然弱基线；关系建模与建筑约束不是我们独占；P0 中位数不能代表这类学习式分阶段模型 | 2026-04-30 v1；官方 arXiv 标注 SIGGRAPH 2026 Journal Track；本次主要依据作者摘要／架构说明 |
| **SceneSmith**，2026；[v2 论文](https://arxiv.org/html/2602.09153v2)、[官方项目](https://scenesmith.github.io/)、[元数据](https://arxiv.org/abs/2602.09153) | 分层 agent 系统，结合资产生成／检索、目标尺寸、布局与物理处理 | 不应简化为同一种 class/size/pose 网络；其系统范围更广。我们使用 SceneSmith 来源的留出数据，不等于已经评测或击败 SceneSmith 方法 | v2，官方元数据标注 ICML 2026 Spotlight；方法与资产接口节选，未核验其性能复现 |

上述论文足以否定宽泛的 first-ever 表述，**不足以证明我们的全部具体组合已被同一工作覆盖**。本次未发现一个已核验论文完整执行“当前 J0/P0 的共同基座＋相同准入集＋分组 OOF 尺寸先验＋本多源合同”的实验；这只是检索范围内的观察，不是独创性证明。论文与同作者 repo 相互补充，但不是两个独立实验。

OptiScene 的关键差异可直接定位：[本地论文](../../../../tmp/pdfs/optiscene-paper/paper.txt) 262–282 行写出先检索资产 bbox，再输出位置和朝向；表 1/2 与实验设置在 430–458 行附近。因此，已知尺寸论文、legacy 联合输出、新 J/P 必须保持不同证据层。不同数据、输入、模型、render／judge 下的 OOR/FID/GPT 数字不得拼成胜负表。

## 3. A 线：结构化 FastFill v2 贡献矩阵

“已实现”只表示本地代码存在并与说明对应，不等于收益已被实验识别。

| 候选贡献 | 已实现／尚未实现 | 最近先例与剩余差异 | 目前可支持的主张／必要证据 |
|---|---|---|---|
| A1：请求绑定的结构化几何解码 | 已实现 Qwen 条件 memory、对象 span pooling、双向 slot decoder、几何 heads；[model.py](../../../../fastfill/v2/model.py) 125–227 | ATISS、InstructScene、DeBaRA、SceneNAT 已有条件对象建模。差异在具体 Qwen 接口、固定请求身份、单轮结构化参数化组合 | 可以描述实现；不能称首次对象 decoder，不能说优于文本。需同任务、同数据、同基座的数值头 vs 文本 CE 对照，以及去对象 self-attention 对照 |
| A2：异质来源的字段缺失监督 | 已实现 validity masks、几何损失及有效项计数；[objective.py](../../../../fastfill/v2/objective.py) 14–25、[losses.py](../../../../fastfill/v2/losses.py) 124–169 | 工程上避免捏造缺失标签；通用 masked loss 并非新思想。本次没有足够文献证据宣称该多源监督机制首创 | 候选价值是缺字段语料是否增加可用监督并改善泛化。需 complete-only 与 partial-enabled、控制数据量／来源后比较；报告每项有效标签数量 |
| A3：保持身份语义的合法交换匹配 | 已实现对描述、约束与 support 引用的交换认证；[matching.py](../../../../fastfill/v2/matching.py) 30–77 | ATISS 的集合建模、DeBaRA 的同类置换目标已覆盖一般置换问题；我们更窄的差异是“同类但不同角色不得交换” | 可作为明确的设计选择；需 fixed identity vs certified matching 对照，分别看真正可交换组和不可交换角色。不能把 Hungarian 本身当贡献 |
| A4：几何参数化与约束训练 | grid/residual 位置、正尺寸、yaw bin/residual、对称候选已实现；历史主配置 box/collision/boundary 权重均为 **0**；[配置](../../../../fastfill/v2/configs/main7-cell05-main-20261007b-e5.json) 35–51 | 几何参数／损失属于成熟构件；相对独特组合需实证。对象对称候选不等于整房对称训练 | 可以说有这些 heads／可选损失；不能把零权重项写成已产生收益的物理监督。需 raw 解码、spread 后处理、启用项分开归因 |
| A5：关系更好且低延迟 | 历史训练／测量存在；更强关系解码、条件 masked 布局训练与锚点迭代仍在设计中列为 Stage 0/1 计划 | SceneNAT 已有 masked 并行生成；CasLayout 已有关系分阶段建模。不能仅用计划描述抵消历史布局问题 | 目前只能提出质量—速度取舍假设。需完整条件、未参与选模集上的关系与功能质量，加包含预处理／后处理的成本 |
| A6：真实资产与部署闭环 | 离线 contract、bbox 检查、内存 Host 参考实现存在；README 明确真实 mesh／physics／Solver／持久 Host 仍需集成 | SceneSmith 已将真实资产和物理处理纳入系统；我们的 target envelope 不等于已验证资产 | 只能主张离线接口／原型；不能称 simulation-ready、真实机器人可用或完整 WorldEdge 部署成功 |

**两个容易混淆的 mask 必须分开。** 已实现的 validity mask 用于缺失标签：例如尺寸／位置要求该字段三个轴有效，yaw 可另行缺失；它不是 SceneNAT 式的 masked generative modeling，也不是 v2 Stage 1 计划中的条件布局补全训练。[objective.py](../../../../fastfill/v2/objective.py) 14–25；[v2 设计 §10.3](../../../fastfill-v2-design-20261009.md) 1600–1617。

**双向对象 decoder 不自动保证整个模型对顺序不变。** `_slots` 含 slot 索引 embedding，条件仍由 Qwen 编码；合法组 matching 是监督对应机制。若论文声称输入重排／ID 改名稳定，仍需单独实测，而不是从“用了 Transformer/Hungarian”推出。[model.py](../../../../fastfill/v2/model.py) 163–199。

### v2 已有负面证据不能隐去

[根 README](../../../../README.md) 定义 v2 为结构化主架构；[v2 README](../../../../fastfill/v2/README.md) 开头保留未验证与部署限制，839–854 行又有历史训练／评测记录。因此不能笼统说 v2 从未训练，也不能凭“formal runs”称其质量已验证。应引用有日期的具体实验范围。

[2026-10-09 设计 §9.10](../../../fastfill-v2-design-20261009.md) 1540–1568 行记录：step-6357 的三字段验证队列中，关系成立率为 spread **0.433**、argmax **0.253**、LLM harness **0.764**；另有 log-size **0.237**、harness **0.388**，历史延迟中位约 **56 ms** 与 **16.8 s**。这些是该文档的历史测量，本审查未重算；300 行与 checkpoint selection 有重叠，关系公共集为 293 间，代码／解码版本也有限制。它们支持“局部尺寸／延迟优点伴随关系不足”的诊断，不支持留出整体质量优势，更不能移植为 J0/P0 成绩。Stage 0/1 在该设计 1606–1616 行仍标记为计划；本次未取得其完成 receipt。

## 4. B 线：当前 J0/P0 的贡献与可识别范围

| 候选贡献 | 当前实际证据 | 相对先例的剩余价值 | 主张上限 |
|---|---|---|---|
| B1：未知尺寸联合 SFT | [contract.messages](../../../../fastfill/comparison/contract.py) 211–237：J 请求仅含 movable id/category，输出尺寸与位姿 | 与 P 在同一允许信息下检验策略取舍；不是新生成变量或新 LLM 能力定义 | 已实现并正式运行；尚无质量贡献结论 |
| B2：train-only 两步适配基线 | [size_prior.py](../../../../fastfill/comparison/size_prior.py) 128–185、205–236：类别×房型逐轴 log-median，稀疏时回退类别／全局，5 折 group OOF | 更可解释地避免把 GT 尺寸喂给未知尺寸基线；是实验设计质量 | 只能代表这个先验＋摆放器，不能代表 OptiScene 作者模型或全部强两步法 |
| B3：目标无关请求合同与共同训练人口 | [bind_room](../../../../fastfill/comparison/contract.py) 153–181 不按目标几何排序；[common_admission](../../../../fastfill/comparison/train_pair_helpers.py) 268–296 两臂共同准入 | 修复 legacy 混杂，形成可复核对照；不是新架构 | 不变性止于冻结 eligible inventory，不能扩大成原始数据筛选完全独立于几何 |
| B4：跨来源经验规律 | [协议](../protocol.md) 要求全 8,647 test、18 source 标签、source-macro 与 room-micro、来源留出 | 可能回答何时 joint 比简单先验有效／何时不值；科学价值在结果规律而非“用了 18 来源” | 目前是假设；来源数不是 18 个独立同分布实验，也不是已经证明泛化 |
| B5：SFT／DPO 的方法收益 | 当前主端点只是 matched SFT J0/P0；legacy DPO 单列 diagnostic | 后续只有在另锁设计下才能研究偏好优化增量 | 不能从 legacy 链的下降归因 clean J/P、human alignment 或 v2 loss；本轮没有新 DPO 结论 |

现有设计比 legacy 更能回答有限问题，但**不能把它解释成只改变“是否联合预测”一个因素的纯机制实验**：

1. **监督示范不同。** [main_data.py](../../../../fastfill/comparison/main_data.py) 171–203 行：J 检验原 joint target，P 在固定估计尺寸下经过 pose repair 后检验；两边均合格才保留。因此主结果反映“联合原目标训练”与“先验尺寸＋适配示范摆放训练”的系统组合。它不是随意造假，而是保持 P 示范可行的设计；论文必须披露其代价，不能把所有差异归于因子分解。
2. **相同行数不等于相同算力或监督信息。** 当前报告为 98,923 共同合格训练行经 8,192-token 准入后保留 **98,754**，排除 **169**；J/P 监督 token 为 **53,157,852 / 34,454,444**。相同 seed、epoch、LoRA、优化器并不让输出长度和有效监督量相同。训练与长上下文推理也不是同一预算。[当前结果](../current-results.md)
3. **先验有额外合格训练池，且共同训练筛选依赖可行性。** prior pool **124,843** 大于 decoder 共同池；**25,920** 未通过共同资格。先验冻结早于 decoder 准入，[main_data.py](../../../../fastfill/comparison/main_data.py) 108–131、196 行可核对。既不能把它自动叫 test leakage，也不能声称两系统见到的训练统计信息完全相同；配对筛选可能偏向两系统都容易示范的场景，其方向须测量。
4. **基线能力有限。** 逐轴中位数只依赖类别／房型，不利用房间几何、对象共现来选尺寸；J 可利用这些条件。因此赢 P0 不能推出 joint 普遍优于 learned two-stage，也不能排除更好的尺寸估计器已经足够。
5. **总体、泛化和感知结论有限。** test 是历史已暴露划分；SceneSmith/SpatialGen 是相对本次微调与先验的来源留出，不保证通用基座从未见过上游信息。只有一个训练 seed。来源有强烈数量不均；共享资产按各臂尺寸拟合的视觉评估不能单独验证真实资产原生尺度可部署性。[协议](../protocol.md)、[执行锁](../execution-lock.json)

上述限制要求校准结论，不要求停止当前实验；本次也没有发出停止或重训建议。

## 5. 可检验假设与最小消融：按主张决定，不铺满组合

以下均为建议的后续证据，不表示已实施。本报告不改变正在运行的冻结协议，也不授权新增作业。先保留当前 J0/P0 主端点，再在 dev 上确定扩展；历史已暴露 test 上探索出的规律须标为探索性，并在新留出材料复核。

| 假设／要写的主张 | 最小关键对照 | 应看到什么才能支持 | 失败／零收益的解释 |
|---|---|---|---|
| H-B1：房间条件下联合尺寸预测比简单尺寸先验更有用 | 先完成 J0/P0；联合查看完整性、size 合理性、raw 几何、边界／支撑与可用视觉证据 | 不是靠缩小物体、减少成功分母或强修复换来 OOR 改善；收益与代价同时报告 | 若只有 OOR 降而尺寸更差，只支持一种权衡；不能称布局更优 |
| H-B2：收益来自 joint，而非 P0 过弱 | 加一个 train-only、同请求条件的学习式尺寸估计器＋placement；保留简单先验；训练端仍按 group OOF | J 对强两步法仍有可信优势，或揭示明确条件下的差异 | 若强两步法追平，贡献转为尺寸估计能力／上下文的重要性，而非 joint 优越 |
| H-B3：系统差异并非完全由监督构造／预算造成 | 分别做先验 fit pool 限到共同训练人口的敏感性；另在可行小规模集控制 teacher／pose target 差异；报告实际 token 与成本 | 核心规律不依赖单一先验池或 teacher 选择；无需假称所有因素已完全等同 | 若效果反转，主张限定为当前训练包。GT-size oracle 可作输入敏感性，不能保证是数值上界 |
| H-B4：某类场景更需要尺寸适配 | 用结果揭晓前由 train/dev 定义的密度、类别尺寸变异、长尾／来源分层；paired house/alias 统计 | 效应在预定分层有稳定模式，并有机制一致的失败案例 | post-hoc 挑来源不证明泛化；重复训练 seed 才覆盖训练方差，bootstrap 不能代替 |
| H-A1：v2 结构化解码改善质量—成本取舍 | 同冻结输入、数据人口、标签资格与 Qwen 起点的 v2 vs text；再去 slot self-attention；单独计 raw/spread | 保持关系／功能质量时成本更低，或在固定预算下质量更好；包含 Qwen 编码与后处理耗时 | 只更快而关系更差，报告 Pareto 取舍；不能把网络前向毫秒当完整系统延迟 |
| H-A2：部分字段监督提高数据利用率 | complete-only vs partial-enabled，并补控制样本量／来源构成的对照；下游在相同合格人口评价 | 改善不只是加入更多容易来源，缺失字段也未被伪标签污染 | 若无改善，报告标签异质性／噪声限制；不要把“可训练更多行”写成泛化提升 |
| H-A3：身份约束 matching／对称处理有收益 | fixed identity vs certified matching；按合法交换性分组；对称候选开关保持标签依据一致 | 改善多解监督，同时不交换有不同支撑／关系角色的实例 | 若只改参考距离而不改布局质量，则限定为监督稳定性，不能推出功能收益 |
| H-A4：关系改造解决当前 v2 短板 | Stage 0 后处理与 Stage 1 学习改造分开；完整条件的冻结关系集；对照 SceneNAT／DeBaRA 等适配版本时统一输入 | 超越当前 raw/spread 且不是仅靠后处理；多轮方案另报轮数／成本 | 若需多轮关系推理才有效，修改“单次前向足够”的假设；负结果本身有意义 |

最小论文闭环应先选择 **B 的经验研究** 或 **A 的方法研究**。前者不必为了名义完整重训所有文献模型，但若声称广泛优于两步法，强 learned prior 是必要对照；后者不能只拿当前 J/P 代替同任务结构化 vs 文本消融，也不能只与高开销云端 agent 比速度。若声称当前技术领先，DeBaRA 与 SceneNAT 是比“仅引用已知尺寸 OptiScene”更直接的审稿人反例，应安排可比适配或清楚缩小主张。

## 6. 结果到主张：允许负面结论

| 将来实际观察 | 可以写 | 不可以写 |
|---|---|---|
| J0/P0 尚无正式成绩（当前） | 实现、协议、运行状态和待检验假设 | 新 FID/GPT/UR、已胜出、贡献已经实验证实 |
| J0 在多个质量面向优于 P0 | 在本数据、共同合同和训练预算下优于指定中位数先验系统 | joint 胜过所有 two-stage；超过原论文 OptiScene；v2 heads 有效 |
| 两臂差异不确定 | 尚未检出稳定差异；报告区间与可排除的效应范围 | “两者等价”；只有预设等价界与足够精度才能支持等价 |
| P0 更好或更省且质量相当 | 简单训练集先验在该任务下有竞争力；联合尺寸自由度未带来预期收益 | 未知尺寸联合生成普遍无效 |
| J 只靠更小尺寸降低 OOR | 几何指标与尺寸合理性之间的失配／取舍 | 空间理解或物理可行性提升 |
| v2 数值更准、速度快，但关系更差 | 单参考几何回归与场景协调性存在张力，需关系建模 | “整体质量超过 LLM”；用局部最优挑选展示掩盖负面结果 |
| 多源改善只出现在常见来源 | 受限范围的经验收益与长尾失败 | 普遍跨域泛化 |

负结果要成为研究贡献，仍需足够强的对照、可复现人口与对失败原因的分析；**不是“没赢所以自动有论文”**。同时，已有邻近论文也不意味着整个项目必须放弃：被明确覆盖的是宽泛主张，尚可研究的是机制、条件与取舍。

## 7. 诚实的论文定位与 ARIS 用法

**B 线建议定位：**《固定对象清单下的未知尺寸室内布局：联合文本生成与训练集尺寸先验的受控比较》。当前可陈述“建立了匹配 SFT 的比较协议并公开信息边界”，将“发现何时何种方案更好”留到正式结果之后。数据贡献应称现有多来源数据的整理／协议；除非具备可发布的清单、字段语义、资格记录及授权条件，不把来源聚合直接包装为新数据集。

**A 线建议定位：**《部分几何监督下的请求绑定结构化布局预测》。候选机制是身份约束、异质标签利用和条件编码到数值解码的具体组合；真正需要贡献的是这套组合揭示或解决了什么问题。现阶段适合原型／经验结果表述，不以“未知尺寸首创”“端到端物理场景生成”作标题。与 B 合写时，必须有明确实验桥梁，不能跨路线借用最好数字。

ARIS 固定参考来自 [manifest](aris-reference/manifest.json)，commit **`26b95cfa0d8747078e9e43b42e20952709e561b8`**；本次校验了清单所列下载文件的 SHA256 一致。官方仓库是[ARIS](https://github.com/wanshuiyin/auto-claude-code-research-in-sleep)，它是研究流程工具，不是室内布局算法，也不能为贡献提供自动学术认证。

| 借鉴的固定版本方法 | 本报告如何使用 | 明确没有完成／不能宣称 |
|---|---|---|
| [novelty-check](https://github.com/wanshuiyin/auto-claude-code-research-in-sleep/blob/26b95cfa0d8747078e9e43b42e20952709e561b8/skills/novelty-check/SKILL.md) | 拆解主张、找具名先例、区分已有部件与潜在发现；邻近工作不自动否决具体差异 | 没有穷尽每个数据库／每个主张三组检索，也未运行上游 `verify_papers.py`；本文逐项直接核对一手链接，非完整上游执行认证 |
| [experiment-audit](https://github.com/wanshuiyin/auto-claude-code-research-in-sleep/blob/26b95cfa0d8747078e9e43b42e20952709e561b8/skills/experiment-audit/SKILL.md) 与 [integrity](aris-reference/shared-references/experiment-integrity.md) | 区分工程状态、真实测量、proxy、未完成项；核对主张是否越过证据范围 | 本次不是第二轮 metric 实现审计，不输出 ARIS integrity PASS |
| [result-to-claim](https://github.com/wanshuiyin/auto-claude-code-research-in-sleep/blob/26b95cfa0d8747078e9e43b42e20952709e561b8/skills/result-to-claim/SKILL.md) | 预先给出不同可能结果允许支持的结论，拒绝把文件存在等同主张成立 | J/P 结果未完成，不能进行其最终结果裁决；没有伪造已执行上游 evidence checker／jury |
| [research-review](https://github.com/wanshuiyin/auto-claude-code-research-in-sleep/blob/26b95cfa0d8747078e9e43b42e20952709e561b8/skills/research-review/SKILL.md)、[independence](aris-reference/shared-references/reviewer-independence.md)、[scope limits](aris-reference/shared-references/review-scope-limits.md) | 直接读原始路径；提出有对应主张的最小补充实验；保存分歧与限制 | 同模型角色分工不提供跨家族独立性，没有异构评审 trace，也不编造 novelty 数值评分或录用概率 |

因此本次建议是：**继续完成冻结 J0/P0，以受控经验研究审视结果；保留 v2 为独立方法候选，其研究价值需自己的对照证实。** 这不是对最终论文新颖性或完整实验诚信的无条件通过。

## 8. 检索范围、可信度与已读文件

检索日为 2026-10-13；主体为英文室内布局生成文献，重点 2020–2026，并补查 2025/2026 近期相关工作。本次使用网页搜索、arXiv 正文／元数据、NeurIPS/ICLR/CVF 正式论文、作者项目页及官方 GitHub。核心矩阵纳入 **10 项场景工作**，ARIS 单列为方法参考；不是 PRISMA 式穷尽综述，不提供虚构的总检出／排除数量。检索中出现的博客、ResearchGate、聚合解释页均不作结论依据。图像重建／房间壳体估计等任务没有纳入核心比较。

代表性实际检索式包括 `ATISS autoregressive transformers indoor scene synthesis object category size translation rotation official paper`、`LayoutGPT 3D scene synthesis size orientation position official github`、`InstructScene semantic graph prior layout decoder size orientation ICLR 2024`、`SceneFormer dimensions category orientation location official`、`site:arxiv.org 2026 indoor scene synthesis language model parallel decoder layout size`、`site:arxiv.org 2025 2026 indoor scene partial supervision`、`site:arxiv.org CasLayout indoor scene relationship cascade diffusion`、`DeBARA 3D scene arrangement diffusion bounding box attributes NeurIPS 2024`，并以正式论文域名查询具体尺寸／朝向表述。部分大 PDF 无法整本抓取，使用作者方法页、官方代码 README、正式论文索引节选或 arXiv HTML 补证；范围已逐行标注，不假称通读。

可信度分层：**高**——具名先例足以推翻宽泛首创；当前两条代码路径确实不同。**中**——v2 具体组合及 J/P 实证问题可能形成贡献，这是基于现有实现和已检索工作的研究判断。**未定**——新方法质量、最终论文充分性、完整新颖性覆盖；本报告没有新质量数据，也没有独立训练复现。论文自报成绩不等于第三方验证；本地旧设计中的结果是历史记录，不是本审查的新测量。

本地读取范围如下（只列实际读取的文件；长文按相关章节／代码段阅读，不声称全库审查）。审查时 checkout HEAD 为 `8b94c437cbdacc5329b6abd04e26223fd0824e40`，工作区含父任务并发未提交文件，故行号指本次本地读取状态，不等于所有代码已纳入该 commit。代码发现曾尝试 MCP graph，当前 OptiScene 项目未索引，随后按规则回退到局部文本查找。

| 文件 | 实际阅读重点／证据位置 |
|---|---|
| [README.md](../../../../README.md) | 任务定义、Code and experiment versions、v2 数据及实现说明 |
| [fastfill/v2/README.md](../../../../fastfill/v2/README.md) | 1–75、839–854 行及相关状态／模块段落；开头限制与历史运行分开 |
| [docs/fastfill-v2-design-20261009.md](../../../fastfill-v2-design-20261009.md) | §1、模型／训练／交付相关段落，重点 1540–1625、文献定位及计划状态 |
| [docs/fastfill-v2-spec-vs-implementation-20261009.md](../../../fastfill-v2-spec-vs-implementation-20261009.md) | 1–44 行结论；架构、mask、loss、缺失消融等对应条目；作为有日期的历史审查使用 |
| [comparison-design-20261013.md](../../20261011-v11-clean-dpo/comparison-design-20261013.md) | 未知尺寸任务、适配基线、共同起点、OOF、oracle 和解释边界 |
| [protocol.md](../protocol.md) | 主终点、输入合同、人口／split、停止规则与负结果许可 |
| [execution-lock.json](../execution-lock.json) | 起点、共同训练预算、先验、teacher、视觉条件及限制；原推理配置后有修订，不用于报告最新作业参数 |
| [current-results.md](../current-results.md) | legacy 混杂、最新 J/P 尚无分数、准入行数与监督 token、父任务实测来源；没有修改 |
| [tmp/pdfs/optiscene-paper/paper.txt](../../../../tmp/pdfs/optiscene-paper/paper.txt) | §3.1、SFT/DPO、表1/2、实验设置；重点 262–282、430–458 行 |
| [fastfill/comparison/contract.py](../../../../fastfill/comparison/contract.py) | 143–237：绑定身份、请求序列、两臂输入输出 |
| [fastfill/comparison/size_prior.py](../../../../fastfill/comparison/size_prior.py) | 128–236：逐轴统计、回退、分组 OOF |
| [fastfill/comparison/main_data.py](../../../../fastfill/comparison/main_data.py) | 79–203：资格、先验冻结、两臂 target 构造与共同筛选 |
| [fastfill/comparison/train_pair.py](../../../../fastfill/comparison/train_pair.py) | 166–176、203–246：CausalLM 与 SFT，不是 v2 heads |
| [fastfill/comparison/train_pair_helpers.py](../../../../fastfill/comparison/train_pair_helpers.py) | 242–296：assistant CE 与共同长度准入 |
| [fastfill/v2/model.py](../../../../fastfill/v2/model.py) | 125–227：Qwen 特征骨干、双向 decoder、slot 与 heads |
| [fastfill/v2/objective.py](../../../../fastfill/v2/objective.py) | 14–61：字段资格与有效项窗口 |
| [fastfill/v2/losses.py](../../../../fastfill/v2/losses.py) | 117–230：matching、mask、几何／对称损失与可选项 |
| [fastfill/v2/matching.py](../../../../fastfill/v2/matching.py) | 30–77：交换组认证及角色引用 |
| [main7-cell05-main-20261007b-e5.json](../../../../fastfill/v2/configs/main7-cell05-main-20261007b-e5.json) | 1–90：历史 v2 配置，尤其零权重场景正则 |
| [aris-reference/manifest.json](aris-reference/manifest.json) | 官方 commit、源 URL、文件摘要；读取时清单文件摘要核验 |
| [novelty-check/SKILL.md](aris-reference/novelty-check/SKILL.md) | 全文：claims、prior、delta 与 verdict 限制 |
| [experiment-audit/SKILL.md](aris-reference/experiment-audit/SKILL.md) | 前 118 行及审查条目定位：外部审查与诚信边界 |
| [result-to-claim/SKILL.md](aris-reference/result-to-claim/SKILL.md) | 前 125 行及主张定位：结果完成后裁决、存在性检查不等于论证成立 |
| [research-review/SKILL.md](aris-reference/research-review/SKILL.md) | 流程／主张定位及 165–205 行：保存结论、范围、外部评审身份 |
| [shared-references/experiment-integrity.md](aris-reference/shared-references/experiment-integrity.md) | 全文：真实证据、proxy、未完成项与模型独立性 |
| [shared-references/reviewer-independence.md](aris-reference/shared-references/reviewer-independence.md) | 全文：直接读取原始材料、避免引导评审 |
| [shared-references/review-scope-limits.md](aris-reference/shared-references/review-scope-limits.md) | 全文：针对实际问题提出范围内改进 |

本次完成的是贡献／新颖性审查。新 J0/P0 的正式结果仍待产出；本文所有候选收益和所需消融均保留为待验证项。
