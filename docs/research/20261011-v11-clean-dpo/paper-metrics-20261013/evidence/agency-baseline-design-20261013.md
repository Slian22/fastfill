# Agency Research Synthesist：未知尺寸基线设计审稿意见

2026-10-13。供父代理综合与用户选择；仅设计建议，未启动训练、生成、付费评审或修改正式文件。不涉及 v2；当前对象是 V1.1 的直接 placements JSON SFT→DPO1→DPO2。

**推荐收敛方案：主实验只做两个系统臂——共同 fresh base 的联合尺寸/布局预测，以及 train-only SizeEstimator→OptiScene-style 摆放器。全部符合条件的 harddisk 多源纳入同一数据合同；known-size/GT 尺寸只做信息更充分的 oracle 辅表，已有 warm-start V1.1 保留 legacy 诊断。暂不同时重训 paper/README 两套完整原版。**

该 pipeline 比直接给 OptiScene 增加 size 输出更清楚：前者有可识别的“先估尺寸并固定，再摆放”结构；后者若同时换成我们的数据、直接 JSON、程序偏好和训练预算，可能已经与 V1.1 是同一算法。相同算法不能换名称后充当独立 baseline。本轮应主张任务扩展、数据及系统设计，不能预设提出了新的 DPO 算法。

## 依据、范围与命名

本次只复核官方 README，并使用已读用户 PDF §§3/4、附录 B/C/D；没有广泛文献检索、harddisk 新扫描或源码重审。README 是一手发布说明，对公开命令有较高证据权重，但不证明论文实际训练；PDF 是一手论文方法报告，对作者声明有较高权重，但不补足未公开运行细节。两者同一作者来源，不是独立复现。既有本地审计是运行/数据证据，以下设计建议是方法判断，不是已测结果。

- **Paper-spec reimplementation / Qwen3-8B**：PDF §4.1 写 SFT 10 epochs、lr 5e-6、LoRA r16/alpha32/dropout .05；DPO lr 5e-7、5 epochs（未明确两阶段分配）。保留语义 reasoning 监督、人类认可 DPO1 等才接近论文方法。未解决配方、数据及标签来源差异前，不能直接写“paper-faithful reproduction”。依据：本地 `tmp/pdfs/optiscene-paper/paper.txt:353`、`:405`、`:460`。
- **README/code-recipe reimplementation / Qwen2.5-7B-Instruct**：官方 README 当前例子为 SFT 1 epoch、两次 DPO 各 10 epochs，DPO lr 5e-6、r32/alpha16；权重仍列 Coming Soon。可说“按公开配方自行训练”，不能因此称论文 checkpoint 或论文严格复现。若调整划分、输入、数据或超参数，应逐项披露。[官方 README](https://github.com/PolySummit/OptiScene#training-pipeline)，[权重栏目](https://github.com/PolySummit/OptiScene#model-weights)。
- **OptiScene-original-task** 只能表达保留已知尺寸→位置/朝向的任务，不证明模型是原作者权重。自行训练必须再带 `reimplementation` 与实际配方标签。
- `liantian/OptiScene-Qwen-8B-Final` 下载身份已核验，但上游训练/作者身份未知：现有 V1.1 标为 `legacy warm-start diagnostic`，不能拿它与 fresh base 的另一臂作受控方法结论。

## 最小可执行矩阵

两主臂建议从**同一可验证 Qwen3-8B revision** 独立开始；共同固定 tokenizer/chat template、非隐藏推理模式、LoRA、训练阶段、场景呈现预算、开发集选择规则和推理采样预算。这里选择 Qwen3 是主实验的共同控制条件，不使其成为论文复现。成本以实测 tokens/steps/GPU-hours 报告；相同 epochs 不等于相同计算量。

| ID / 优先级 | 系统与任务 | 模型可见的信息及输出 | 训练与偏好 | 能支持的结论 |
|---|---|---|---|---|
| J / P1 必做 | `Joint-V1.1-style / fresh Qwen3-8B` | 房间、对象类别/描述、数量、共同约束；无可移动物体真尺寸；直接输出 size/pos/yaw/on | 新合同下多源重训，程序偏好；保存 SFT/DPO1/DPO2 | 未知尺寸联合预测系统表现；不是旧 checkpoint 的重新命名 |
| P / P1 必做 | `SizeEstimator + OptiScene-style placement / matched-data` | 相同初始输入；估计器输出尺寸，摆放器只输出 pos/yaw，最终尺寸固定为估计值 | 同 base、同多源 scene IDs/划分、共同训练预算；摆放器采用明确标注的数据/任务适配 | 与 J 的联合预测 vs 两步系统比较；不是原论文方法复现，也不分离所有内部机制 |
| O / P2 可选、无新增训练臂 | `P + GT-size oracle input` | 给同一摆放器可移动物体 GT 尺寸 | 不重训，仅替换评测输入 | 额外尺寸信息的敏感性参考；可能有输入分布偏移，**不保证性能上界**；不进主胜负表 |
| L / 既有 | `V1.1 legacy warm-start` | 原冻结合同及已有 checkpoint | 不改变历史数据/评分 | 历史诊断；不归入 fresh-base 因果对照 |

**两个新训练分支足够回答当前主问题。** 保存阶段快照不增加新的训练配方，但阶段差值包含额外训练预算，不能单凭 SFT→DPO 改善证明 DPO 优于等预算继续 SFT。

这里的 P 保留的是 OptiScene 的已知尺寸摆放接口与分阶段训练思想。为与当前直接 JSON V1.1 比较，首轮不另加显式 reasoning 监督；必须因此使用 `style/adapted` 标签。若把 P 叫“原方法”，就必须恢复/核验其实际 reasoning 和偏好来源，并承认比较的是更大系统包。不能一面移除原方法要素，一面声称复现了原论文。

## SizeEstimator 的具体默认方案

先用无需另训大模型的**类别×房型尺寸先验**：仅从 train 统计正尺寸的逐轴 log-space 中位数；小样本回退到类别，再到全局。轴定义、类别归并、回退门槛仅用 train/dev 决定并冻结。无需先加独立神经 SizePredictor；它是之后发现先验为瓶颈时的替换项，不属于必做矩阵。

- train 按房屋/原始场景/跨源重复组做 K-fold。摆放器训练看到的是其他 folds 估计出的 **OOF 尺寸**，避免每条训练场景把自己的尺寸写回先验。最终估计器用全部 train 拟合，dev/test 一律使用这一份 train-only 映射。
- 初始信息保持一致：估计器仅用 J 同样可见的类别、房型/描述等；不能用 test 资产 ID、mesh、真实 bbox、目标尺寸排序或布局姿态。GT 尺寸可在隔离的评测器中评分，不得回流模型或检索器。
- P 的固定尺寸必须进入最终几何评分。不能事后覆盖成 GT，不能以“原模型只摆放”为由不计尺寸错误。J/P 的空间坐标、单位、对象对应关系、yaw 和支撑评价要统一；P 的支撑关系从最终几何用共同规则推导，不能默默供应 GT on。
- 拒绝把“喂估计尺寸”当作无害字段替换：当输入尺寸改变，原 GT 位置可能碰撞/越界。P 的 DPO chosen/rejected 必须在**同一组输入估计尺寸**下重新验证，无法形成有效偏好对时明确记录排除原因。SFT 位置仍可作为示范目标，但需记录在估计尺寸下不再可行的比例，不把它们冒称有效正例。
- 主对照 DPO 采用两臂均能表达的空间偏好，如边界/碰撞/阻挡，保持筛选准则、来源和预算对称；不能给 P 无法改变的 `size_corrupt` 对再指责其学不会。J 的尺寸能力通过尺寸 SFT 与独立尺寸指标评价。若恢复 J 完整五类扰动作为额外系统结果，注明监督包不同，不归因为联合预测结构单独作用。
- 共享 scene IDs 不意味着所有偏好对逐字相同；输入固定尺寸导致可行性差异时，公开两臂准入率及共同可比子集。主结论是受控资源下的**系统设计比较**，不声称已隔离纯粹的联合建模机制。

推荐 P 的理由是接口差异真实、易审计、能定位估尺寸与摆放两种误差；风险是简单先验可能偏弱。若 J 获胜，只能首先声称胜过这个预定义先验 pipeline，不能声称胜过所有尺寸估计器。若 J 未胜，亦不证明联合预测原则失败。

## 多源与评价合同

harddisk 所有来源进入父代理确认的覆盖清单；本意见不把旧审计的来源列表假定为全盘清单。主实验使用所有合格训练来源，整组去重切分，目标无关对象顺序。已有 SceneSmith/SpatialGen 若保持整源留出，应继续只进入 OOD 测试；“来源全纳入研究”不要求把所有来源都喂进 train。每源记录原始、可转换、几何有效、train/dev/test 和资产可渲染数，不能静默漏掉不支持的来源。

**主评价：全多源几何；辅助评价：预先冻结的资产合格子集视觉。** J/P 在相同房间上配对评分；完成率以全请求为分母，OOR 与尺寸质量分别报告，失败不置零以降低 OOR；按 source/room type/对象数分层，报告来源宏平均及总体微平均。真实 mesh 不齐的来源仍保留几何评价。视觉子集、相机、材质/资产选择和失败规则对称；资产/GT 只用于隔离评价，不给 P 偷看真尺寸。视觉结论限于资产合格子集，不能代表所有 harddisk 场景。

先用训练/开发数据做协议检查，再冻结主终点与测试；统计以原始场景/房屋组配对聚类。第一轮各一个训练 seed 可形成条件于该运行的系统结果；不能把成千房间的置信区间当作训练 seed 稳定性证明。需要跨种子稳定性结论时再重复同一对比，优先于扩展方法全集。

## 人类偏好、transfer reference 与可说的结论

论文 §3.4 的 DPO1 是约 2,200 个人类一致认可布局，附录 C 分为 1,100+1,100。程序检查器挑选的好布局，甚至直接用 GT，都不等于该人类一致认可。公开 preference JSON 可称“作者发布偏好对”，缺逐条标注证据时不能补写为已验证的人类一致认可。主实验 J/P 全部标为 **programmatic-preference adaptation**；本轮不新增人工采集作为启动条件。

若需要进一步与外部实现接轨，可自行训练一个 `SizeEstimator + OptiScene-code-reimplementation / transfer-reference`：摆放器按官方公开数据与 Qwen2.5 README 配方训练，尺寸估计器仍 train-only。它与 matched-data P **不是同一解释**：前者测迁移系统，差异还包含训练分布、base 和配方；后者才控制主实验的数据/base。此 reference 暂列后续可选，不应抢在两个主臂之前，也不需要现在同时训练 paper-spec Qwen3 版本。原任务使用 GT 尺寸的模型始终单列信息更充分参考。

可写：在共同 fresh base、共同多源初始信息和明确资源协议下，联合预测系统相对预定义两步 pipeline 的几何/尺寸/覆盖表现，以及各自阶段变化。不能写：已超过原论文表格、已严格复现 OptiScene、人类偏好优于程序偏好、纯新算法贡献、仅靠数据规模造成提升、所有两步系统都劣于 joint、或 legacy 模型从未见过测试场景。主矩阵没有数据消融，**不识别多源数据本身的因果收益**；只有需要该主张时，才增加一个同配方的两来源训练对照，而不是改名制造算法臂。

**执行优先次序：** P0 冻结多源未知尺寸合同、主张与预算 → P1 建 train-only/OOF 先验及两臂输入/偏好可行性审查 → P1 运行 J/P 两分支并保存三个阶段 → P1 全几何主结果、合格资产视觉辅结果 → P2 需要时做 GT-size oracle 输入敏感性 → 仅在对应主张需要时增加外部 transfer、数据或 reasoning 消融。当前没有任何训练执行授权被本意见替代。
