# V1.1 未知尺寸公平比较：最小 J/P 双链

状态：设计建议，未启动训练、模型加载、API 或正式视觉评分。对象仅为 V1.1 直接 JSON SFT→DPO；不替换为 V2。

## Behavior Target and Non-Goals

- 主任务：给定相同房型、边界、物体类别/数量/描述及合法 fixed 输入，联合获得物体尺寸与布局。测试输入不含 movable GT 尺寸/位置，不查测试资产真尺寸。多源全部几何合格场景作主评估；资产合格、共同覆盖的视觉子集作辅评估，报告其选择规则和失败分母。
- 不主张新优化算法。若将 OptiScene 改为未知尺寸、直接 JSON、相同数据/程序偏好/预算，它可能就是 V1.1；不得重命名为第二方法臂。原方法接受 GT 尺寸的结果仅为信息更充分 oracle 参考，不进入主胜负表。

## Fixed Comparator Contract

- J/P 均从同一 fresh `Qwen/Qwen3-8B` 权重 revision、tokenizer/chat-template 启动；锁定同一多源 train/dev/test group 划分、共同训练 UID、长度准入、LoRA/优化配置、更新或 token 预算、解码与几何评估器。当前 GT 面积/位置驱动的 ID 排序须在新数据合同中改为声明输入决定，不能只重排测试集。现有 warm-start 三阶段留作 legacy diagnostic；已验证来自 `liantian/OptiScene-Qwen-8B-Final@e554a8c835cb4b212b0af75a5f9adc969ef11194`，上游训练历史及官方身份未验证。
- **推荐唯一外部适配基线：SizeEstimator+OptiScene pipeline。** J 输出 `size+pos+yaw(+support)`；P 用 train-only 类别/房型尺寸先验，再由保留 known-size 输入、pos/yaw 输出接口的摆放器生成布局。P 的最终尺寸就是估计值。它与 J 的测试初始信息相同，分解结构不同，比给同算法换名更清楚；它是适配 pipeline，不是论文复现。
- P 的 matched 摆放器同 fresh base、共同数据训练。用原作者数据/不同 backbone 得到的摆放器只能另列 transfer reference。README 指向 `Qwen/Qwen2.5-7B-Instruct`（`OptiScene_Original/README.md:43`），故该 code-recipe 复现与 Qwen3 因果主表分开；当前最小矩阵不新增这条链。

## Gate: Preflight

- 最低 **2 次 SFT**：J0（直接 joint JSON）与 P0（known-size 摆放接口）；尺寸先验只做 train-only 统计拟合，无第三次 LLM SFT。优先采用固定类别/房型中位数及预注册回退规则。P0 最小版按原 known-size 合同以训练 GT 尺寸学习位置，测试仅用估计尺寸；训练/测试尺寸噪声差异须明示。若加入估计尺寸训练噪声，必须按 group 做 OOF 估计，且重新检查尺寸变化后目标位置的几何有效性；任何修复/过滤只能发生在 train，并保留计数，不能冒充未变更的 GT 目标。
- 参数起点可共用已审计 V1.1 的 LoRA r32/alpha64/dropout .05、8192 context、SFT lr5e-5、有效 batch32；这只是 fresh-base 待验证起点。GPU/存储总上限由父代理在训练前冻结，未锁定则 gate 为 UNVERIFIED：不能直接承诺一轮完成，也不能据此提交任务。原运行 2-GPU SFT/4-GPU DPO 是容量参考，不是新运行授权。

## Single Change Under Test

| 顺序 | J 链 | P 链 | 可识别比较 / 是否新增 SFT |
|---|---|---|---|
| 必做最小起点 | J0 fresh joint SFT | P0 fresh placement SFT + train-only 尺寸先验 | 联合输出 vs 尺寸估计后摆放；共 2 次 SFT |
| 条件推进 | J1 = J0 + DPO1 | P1 = P0 + 合法 DPO1 | 各链新增偏好阶段的效果；无新 SFT |
| 再条件推进 | J2 = J1 + DPO2 | P2 = P1 + 合法 DPO2 | 各链新增几何偏好阶段的效果；无新 SFT |
| 单列参考 | 现有 warm-start 三阶段 | P 权重测试时接收 GT 尺寸 | legacy / oracle；均不进主胜负表 |

- 不自动追加 reasoning、多数据规模或 Qwen2.5 全因子矩阵。若后续确需可识别消融，新增一个明确因素及相应 SFT：reasoning 有无需要新的合法 train-only 监督；数据规模变化需要额外 SFT，并区分固定更新预算的覆盖效应与增加计算的效应。相同算法不算消融。

## Required Measurements

- DPO1 的 human consensus 与程序几何标签必须诚实区分。仅有程序验证时称 programmatic preferences；GT/模型来源本身不能替代真实人类偏好。每对固定同一 prompt，chosen 必须满足本臂的对象/尺寸/约束合同，reject 有可核验缺陷或真实偏好；过滤后 chosen/rejected 差异必须在 tokenization 后仍存在。欲单独比较人类与程序标签，应在相同候选对/共同合格集合上分支，不同时更换 base、候选池和预算；当前不新增该分支。
- **P 禁用 size-only negative**：尺寸是 prompt 输入且不在其响应空间，同一 prompt 下修改输入尺寸不构成合法 rejected response。共同 DPO2 比较只使用双方响应空间允许的 pos/yaw/support 缺陷，用各自最终尺寸验证；不偷换为 GT 尺寸。J 专属 size-only negatives 若以后加入，必须单列消融，不能说 P 接受了同一种监督。
- 全量报告 raw 与 repaired OOR、完整输出/失败计数、尺寸质量与约束满足；同一场景配对、按 group 给不确定性。视觉只在共同资产合格子集比较且不掩盖检索失败。DPO loss/偏好准确率不能代替 held-out 几何与尺寸质量。只有阶段前后对照时，结论是“增加该训练阶段的效果”，不能把额外计算的收益完全归因于 DPO 算法。

## Promotion or Stop Decision

- 先过 preflight（身份/数据合同/预算）、小步 smoke（mask、非退化偏好、完整保存）、固定 dev signal，再推进 controlled；两链都保留 SFT 检查点。没有 DPO1 的 dev 信号就不默认跑 DPO2；没有合法 P 偏好就停 P 的对应阶段，不伪造 size negatives 补齐。
- 推荐主结论聚焦未知尺寸任务扩展、多源数据和系统分解/阶段贡献。fresh matched J/P 支撑受控系统比较；legacy、oracle、不同 base/data 的 transfer 参考不能与其混合归因。所有选择只依据 train/dev；测试不驱动估计器、阈值或资产尺寸选择。

## Preserved Evidence

- 两条链的 base/tokenizer/data/evaluator digest、共同 UID/group 清单、train-only/OOF 先验统计和回退规则、SFT/DPO 准入及拒绝原因、resolved config、GPU/存储预算、完整 checkpoint inventory/hash/load 记录、固定解码和 held-out 原始输出。真实人类标签才记录为 human，并保存标注来源/一致性说明。
- 已完成的本次诊断产物：`docs/research/20261011-v11-clean-dpo/paper-metrics-20261013/{metric-inputs,oor-summary}.json`；1375 场景 × 3 阶段 = 4125 案例，8246 数值 OOR 对照 + 4 失败缺值核对，0 不一致。SFT 保留 2 条 extra_ids 失败。6 个严格房型分层共 1009 场景，其他 366 场景仍在全来源汇总。9 项本任务测试通过；stdlib trace 可执行行覆盖率 91.7%。这些输出明确仅 diagnostic、无 target-independent 主张。
