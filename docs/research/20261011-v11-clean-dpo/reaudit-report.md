# V1.1 代码、训练方法与自动化独立复审报告

复审日期：2026-10-11。本文区分四类证据：代码逐文件阅读、可复现故障与修复测试、冻结服务器的只读核验、尚待部署或完成的后续工作。服务器状态引用 19:00–19:02 UTC 的审计快照；最终部署与CPU验证完成证据已补记至19:23:03 UTC；训练与DPO比较仍在继续。

## 结论与适用范围

**不能把上一轮“测试通过、任务已提交”理解为代码和科研方法已经完全正确。** 本轮重新分工全读了 V1.1 数据、训练、几何、评测、自动化入口及关键依赖，实际复现了多个此前未覆盖的故障，并发现一个影响科学结论适用范围的 **P1 方法限制：所谓“未知尺寸”的输入仍保留由目标尺寸、位置决定的对象顺序与编号。**

本地已经完成多项 P2 防御性修复与回归测试；服务器正在运行的冻结实验保留原源码、原数据和原评分规则。当前实验可以作为**同一历史输入合同下的 SFT → DPO1 → DPO2 诊断性阶段比较**，不能据此宣称已经验证了任意对象清单、完全未知目标几何的布局生成能力，也不能把程序完成或 loss 下降视作模型质量提升。

截至所引用的服务器快照，SFT 四套全量评测已经完成，共 **33,382 条请求**；干净 DPO1 已训练完成并合并；干净 DPO2 正在训练。八个 DPO 评测单元和最终统计报告尚未完成。没有证据支持“新模型已经更好”的结论。

## 审计范围与证据边界

起始目录清单包含 `fastfill/v1_1/` 的 21 个文件，其中生产代码/脚本 16 个、测试文件 5 个，共 5,825 行。各分工完整阅读了数据转换、偏好构建及审计、协议与 parser、DPO trainer、训练身份与 merge guards、后处理、评测与续跑、自动化、统计报告和三份 Slurm 脚本。关键共享依赖包括 `fastfill/{scene,anchors,validate,build,split,interface,evaluate,train,merge_lora}.py`，以及 V2 handoff/RoomGenBench 适配器和上游装配器。新增 verifier、报告 gate 和修复代码另有相应测试与交叉复审。

Codebase graph 的搜索/索引连接返回 `Transport closed`，因此本轮使用已知路径直接阅读，并记录每份文件的 SHA-256、行数和 full-read 标记。当前版本逐 SHA 去重清单见 `reaudit-coverage.json`：V1.1 **20 个生产文件/脚本（5,598行）及12个测试文件（2,882行）全部全文阅读**；另列13个相关生产依赖也全部全文阅读。各分工的阅读范围存在重叠，**不能将行数相加当作独立覆盖率**。这也不是对整个 OptiScene、全部 V2 代码、所有三方库和全部硬件行为的穷尽证明。

服务器核验覆盖另一种性质的证据：原冻结快照的 **224 个文件逐一与 source manifest 的 SHA-256 匹配**，本地与远端 manifest 字节相同，没有额外未登记的 Python 文件；四个全量评测数据集身份与冻结协议一致；抽查的 Slurm spooled 脚本字节与记录相同。这证明检查时运行来源与记录相符，不代表本轮逐行审计了这 224 个文件，更不证明外部 Python 依赖、驱动和固件完全不可变。

完整原始证据位于 `/tmp/fastfill-v11-reaudit-20261011/`，其中 `inventory.json`、各 `*-coverage.json`、六份分工报告和交叉复审报告分别保留具体范围与版本。新修复的文件哈希与最初 frozen source 的哈希不同是预期现象；两者不得混用。

## P1：输入顺序依赖目标几何，当前结论必须收窄

共享 `scene.canonical` 按实际目标 footprint 面积、类别、尺寸以及同类对象的 `(z,x,y)` 排序并重编号；数据构建将这个顺序用于训练和评测。V1.1 转换器删除显式 `size` 字段后保留了顺序与 ID，评测 prompt 同样保留该顺序。因此，“输入里没有 size”不等于“输入与目标尺寸无关”。现有未采用的 `canonical_v1_1` 也不是完整解决方案，因为它仍通过隐藏的目标 `parent` 排序。

变形测试保持房间边界和请求类别不变，仅改变一把椅子的目标尺寸，V1.1 用户清单就从 `[chair_1, table_1]` 变为 `[table_1, chair_1]`。全量扫描结果如下：

| 数据 | 请求数 | 按 GT 面积降序排列的行数 | 与现有 prior-order helper 顺序不同的行数 |
|---|---:|---:|---:|
| train | 144,150 | 144,150 | 101,728 |
| dev | 8,167 | 8,167 | 5,828 |
| dev constrained | 8,035 | 8,035 | 5,815 |
| test | 8,647 | 8,647 | 6,192 |
| test constrained | 8,533 | 8,533 | 6,177 |

该结果**不是**训练/测试划分重叠的证据，也没有证明模型实际利用了这个线索；它证明了任务条件中确实包含目标派生信息。所有阶段使用相同条件，仍能计算有限范围内的配对差异，但不能将结果外推为真正任意 inventory 输入的泛化能力。

后续若要检验真正未知目标几何的输入合同，需要只依据明确声明的请求信息生成 ID 和顺序，一致重映射关系 ID，重新构建 SFT/偏好数据并采用独立冻结协议。**这项工作尚未完成**；仅重排当前评测输入会改变实验并造成训练/评测不一致，不能当作已修复。

## P2：复现结果、本地修复与现行任务的关系

下表中的“本地已修复”不表示运行中的旧快照已经换成新代码。训练完成证明与自动化 lineage 属于同一条防线的不同环节，不应重复计算成彼此独立的科学证据。

| 环节与缺陷 | 已复现的后果 | 本地处理 | 当前冻结实验影响 |
|---|---|---|---|
| Stage1 尺寸界限与未验证 GT fallback | 合法 1 cm 薄物体被判为错误，甚至生成 chosen/rejected 完全相同的偏好 | 采用既定 sanitizer 尺寸规则，验证 GT fallback，写出前独立审计完整 pair | 当前训练读取事先审计并冻结的 pairs，不执行旧生成器 |
| Stage2 正例未检查提示约束 | 几何有效、但违反 `near` 的 GT 被当作 preferred | 先检查完整参考答案及全部提示约束，再审计生成 pair | 当前 Stage2 冻结 pairs 全量复查无该缺陷 |
| Stage1 无身份的 `.partial` 被续用 | 非训练 UID、错误 prompt 或任意文本可被旧缓存采纳 | 增加文件锁、原子缓存、数据/模型/设置/代码身份与逐 pair 重验 | 防止未来重新生成时污染；没有证据表明现行清洗后输入受此缓存影响 |
| 数据转换复用旧输出目录 | 新 manifest 包含未更新的旧 room 文件，晚期失败还会留下部分新文件 | 要求七份非空输入、拒绝非空输出、临时目录构建、源哈希复查和整目录发布 | 当前冻结四份 room 文件身份已独立核验，无需覆盖 |
| DPO checkpoint 先删旧再发布新 | 注入 rename 失败后丢失唯一恢复点，旧代码可能从零重启 | 保留 `.previous`、发布失败回滚、启动恢复；部分 checkpoint 明确失败 | 真实运行中未观察到 checkpoint 丢失；现行源码不热更新 |
| DPO 恢复位置缺少严格检查 | 篡改 epoch/step 可跳过训练并发布 final | 验证实际 loader、整数 epoch/it/step、optimizer 边界和累计步数 | 新防御检查不代表已发现现行 checkpoint 错误 |
| 结构完整的 adapter 被当作训练完成；eval 缺少 lineage | `afterany` 下，存在模型文件不等于训练成功，缺少训练证据的模型也可能被评测 | 新 trainer 写完成凭据；新 lineage gate 验证 guard、pairs、参数、完成位置和 merge，旧模型采用只读末 checkpoint 证明 | 必须通过独立 gate 后才可作为权威阶段结果，不能给旧产物补造新凭据 |
| 有限 loss 之后出现非有限梯度 | 注入 Inf 梯度仍可到达 optimizer update | 梯度裁剪使用 `error_if_nonfinite=True`，在更新前停止 | 未发现当前训练发生这类数值故障 |
| 续跑只检查 scored chunk 元数据、不验评分内容摘要 | 保留 UID/身份但改一个 accepted 值，旧续跑会发布错误指标 | 新 chunk 带内容 SHA-256，恢复和发布都核验；旧 chunk 不静默接入新协议 | 冻结输出通过外部 CPU 重评分验证，避免改动旧评分代码或重新生成 |
| model symlink 绕过输出隔离 | eval 输出可被写到 symlink 实际指向的模型目录内 | 比较前解析所有模型真实路径 | 真实服务器 eval 目录经检查独立，没有发生此覆盖 |
| 身份重新校验失败后仍计入旧 completed | report 仍将先前完成单元当作本次通过 | 增加 `validation_complete`，失败身份状态不能生成完成/统计结论 | 现行真实 source/data 身份核验均通过；新 gate 继续防止陈旧成功 |
| RoomGenBench 解析失败后保留支持环 | `bad_support_ref` 的部分解析结果进入几何 repair，出现无限遍历 | 整份错误预测替换为标记明确的 category-prior 展示 fallback，先验证声明支持图 | 仅 legacy 展示路径，权威全量评测遇解析错误直接记失败 |
| RoomGenBench 恢复网格被写成 `status=ok` | 无生成 provenance 的合成 cube 能被下游称为生成成功 | 恢复/无来源及旧伪三字段 receipt 标为 fallback，逐场景保留 generation receipt | 仅展示资产状态；GLB 可导出不等于模型或生成器成功 |

交叉复审还发现 Stage2 输出发布的 P2：固定 `.partial` 名称、没有生成锁、不拒绝已有结果，`max_pairs=0` 可把已有文件覆盖成空文件。该问题已修复：独占锁、正数参数、已有输出/软链拒绝、输入身份复核及 `os.link` create-only 原子发布均经过故障测试，竞争写入也不会覆盖。35 个 builder/converter 专项测试、55 个组合测试通过；独立复审通过。该生成入口同样不被当前冻结清洗后训练使用。

两项修复过程中的回归已在独立复审后解决：损坏单个资产 sidecar 不再禁用全部 3D 装配，而是保留原证据并逐资产降级；新的报告 gate 通过绝对路径启动 verifier，避免 ambient working directory 错误优先导入旧模块。主协调者另报告解释附录身份漂移和 malformed synthesis 陈旧成功路径已修复；统一集成验证见文末。

## 数据、训练与评价规则的复核

数据侧重新扫描了实际 `train.jsonl` 的 144,150 个消息三元组。全部目标答案具有完整请求 ID、有限数值和无环合法支持引用，所有随输入提供的训练约束在对应参考答案上成立。1,779,614 个请求对象未显式出现 size/pos/yaw/on/parent 字段，但这一结果必须与前述 GT 排序问题一同解释。本轮没有对每个 SFT 房间重新执行全部碰撞检查，也没有重做所有原始 IR 的内容指纹推导；此前 hash-bound split/alias 审计属于另一份证据。

清洗后的偏好又独立全量复审了一次：Stage1 **7,244/7,244**，Stage2 **12,000/12,000** 通过既定 semantic auditor。Stage1 中 4,402 个 preferred 等于参考答案，Stage2 正例全部为参考答案。训练时 Stage2 另有 32 个超长 pair 被整体排除，实际 admitted 为 11,968；不能混淆审计前与 tokenizer 后分母。

SFT assistant-only CE 与 DPO 的公式、符号、causal shift、mask 和 FP32 completion log-probability 计算没有发现新的数学错误。DPO reference 使用冻结 merged base、禁用 LoRA 且不计算梯度，优化对象为 LoRA 参数。真实两进程 CPU Gloo + tiny Qwen3/LoRA/gradient checkpointing 的三 microbatch 尾部梯度，与合并 batch 对照最大差为 **4.656612873077393e-10**；为了可比较，该探针关闭 dropout，因此不能据此宣称 GPU 训练逐比特一致。

当前 recipe 固定四 rank、每卡 batch=1，尾部累积按实际 microbatch 数归一化；Stage1 7,244 admitted 对应每 rank 1,811 batches、两 epoch 共 906 steps，Stage2 11,968 对应 2,992 batches、共 1,496 steps。RNG 未保存，恢复后的 dropout 不等于不中断运行；直接单进程 CLI 的 shuffle 与通用 batch>1 尾部权重还各有边界限制，不能将现行四卡 recipe 的验证外推到任意配置。进度 loss/reward 是 rank0 最后 microbatch 的训练诊断，不是分布式验证集指标。

主指标仍是固定 repair 后、全部请求分母上的 `zero_collision_accepted`；缺失生成、解析失败、少对象、错误引用均为失败，不用 GT 填成功。repair 可能在找不到可行解时打破原来满足的约束，但 scorer 会重新检查全部约束、边界、fixed blocking、支持、尺寸下限及 main/severe collision，未解的失败不能直接算成功。

必须同时保留这些解释边界：碰撞是 upright bounding-box proxy，包含 tuck/contained、薄物体和直接支持对等排除规则；支持检查主要是高度与中心位置容差，不证明稳定性或完整 footprint 支撑；尺寸 realism 是参考相对下限，size IoU 仅比较尺寸，不包含位置/yaw。原始输出与修复后结果必须分开。部分次要碰撞率和平均尺寸指标以完整解析答案为条件，必须伴随 complete rate 和分母阅读。

wall-anchor repair 使用未序列化的 `anchor` 元数据，因此若扩展到 wall-inclusive 数据需要新协议审查。当前四份全量数据经重新扫描只有 floor/object anchor，wall、ceiling、fixed、缺失均为零，该特殊分支不影响本次 100,146 请求矩阵。

统计程序的配对、分组和失败分母经过对抗测试：按 `(source, group)` 整簇 bootstrap，保留 room-weighted 统计量，共 2,000 次 seed42 抽样；四个 dev 比较各自给 95% percentile CI，不作 familywise 保证。test 仅描述，不驱动下一轮训练；没有匹配的“未清洗控制”实验，不能把阶段改进因果归于清洗。warm-start 历史训练来源未知，当前 split held-out 也不等于从未被更早模型训练见过。

## 服务器与已完成 SFT 输出的独立核验

只读核验确认干净 run 目录隔离、SFT symlink 指向预期模型、训练 guard 与现有 base/pairs/audit/code 一致，实际 CLI 参数与 guard 匹配。干净 DPO1 final adapter 及 merge receipt 校验通过，DPO2 的 final 当时尚不存在，符合进行中状态。原始九作业依赖图无环，采用 `afterany`，原始新增 GPU 上限为 84 GPU 小时；这不是无限自动探索或无限重试承诺。

| SFT 数据集 | 完整请求数 | 生成/评分 chunks | 错误请求数 | zero-collision accepted | 全请求接受率 |
|---|---:|---:|---:|---:|---:|
| dev | 8,167 | 32 / 32 | 25 | 5,193 | 63.59% |
| dev constrained | 8,035 | 32 / 32 | 16 | 3,100 | 38.58% |
| test | 8,647 | 34 / 34 | 24 | 5,443 | 62.95% |
| test constrained | 8,533 | 34 / 34 | 29 | 3,334 | 39.07% |

四个 completion receipt、33,382 条有序 UID、生成到评分的关联、final predictions 与 metrics 发布身份均匹配。服务器原评测 Python 环境从保存的 prediction records 重新汇总，精确复现四份 metrics；训练环境 Python 3.12 与评测 Python 3.11 的少量浮点均值差异至多 1.56e-15，没有整数、计数或率的实质差异。SFT 第二次续跑 16 秒完成，验证既有四单元而未重新生成。

上述检查是对已保存评分记录和指标汇总的一致性验证。更强的新增 `verify_outputs.py` 会在**原冻结 scorer 与原评测解释器**中，逐条读取原始生成文本重新执行评分，再精确比较每个字段和汇总；其实际服务器执行、完成或失败必须有新的 verification receipt。**不能把本地 verifier 测试通过写成服务器 33,382 条重新评分已全部完成。**

快照中的干净 DPO1 已合并，DPO2 最新观察值为 130/1,496 steps；这是 19:00 UTC 附近的状态，不是最终进度。旧 diagnostic pipeline 的 DPO2 完成及其 legacy evaluation，不替代干净三阶段矩阵。当前只证实 **4/12** 个全量评测单元完成。

## 测试、交叉复审与剩余边界

各分工先写失败复现或 RED 合同，再实现局部修复。以下是各自运行时的专项结果，测试范围有重叠，不能直接相加为统一总数；代码若在之后变化，须以文末统一运行记录为准。

| 专项 | 已报告的通过结果 | 覆盖或实测范围 |
|---|---|---|
| 数据构建与转换 | 35 专项测试；组合55通过 | generator 实际调用、Stage2竞争发布、语义审计、缓存篡改/中断、真实多进程转换；build_dpo 行覆盖90.56%，build_v1_1 93.45% |
| 训练、恢复与 merge | 32 新测试；组合67通过 | 真实 tiny-Qwen CPU train/resume、rename 故障注入、完成凭据、Inf梯度阻断；trainer 行覆盖96.25% |
| 训练修复独立交叉复审 | checkpoint + guard 59通过 | 原子回滚、步数边界、旧 checkpoint 兼容及完成凭据消费端 |
| 评测与外部重评分 verifier | 69通过 | chunk篡改、失败分母、真实CPU几何、spawn、解释器隔离、缓存与漂移；verifier行覆盖94.52% |
| 自动化/lineage/统计 | 85通过 | 实际 sanitizer receipt、旧checkpoint兼容、新trainer producer/consumer、陈旧成功、symlink隔离；automate95.38%、report98.57%、guard96.36% |
| RoomGenBench | 16通过；六个几何模块 self-check通过 | 真实handoff/GLB/参考检查、五场景313对象往返、fallback与损坏receipt；展示模块行覆盖86.81%，新增helper100% |
| 新审计报告 gate | 主协调者报告21通过 | 含真实隔离 frozen-module 子进程、附录漂移和 malformed synthesis；行覆盖99.18% |

环境没有 pytest-cov/coverage，所列覆盖率来自标准库 trace 对 executable lines 的统计，不是 branch coverage、全部GPU硬件覆盖或“整库100%”。最终独立 cross-review 还实跑数据34测试、报告gate21测试、shell wrapper12测试、评测46测试和旧评分快照真实双进程兼容 smoke。另有10项部署事务模拟测试；这些结果与统一回归分开记录。

仍需明确保留的工程边界：完整 final 目录发布仍有 remove→rename 空窗，现行 recipe 可由末 checkpoint 重建；首次 checkpoint 写到一半且尚无旧版本时故意 fail closed，需要检查，不能保证任意故障自动恢复；rename 防护不等于所有文件系统写入的断电持久性。source/data hash 验证不提供全环境 hermetic 保证，也不证明模型对新输入合同或真实3D物理有效。

## 部署决策和统一验证记录（主协调者补记）

**原则：不热更新正在运行的冻结旧训练，也不在原 evaluation 目录中混用新旧 scorer/chunk 协议。**新构建器与训练防护属于后续独立来源快照；当前生成输出可通过单独 CPU verifier 与严格 lineage gate 外部核验。若新部署使用修改后的 `build_dpo.py`，应在新目录重新生成 sanitizer receipt，不能手改旧 receipt 的代码哈希。

<!-- ROOT-DEPLOYMENT-EVIDENCE: coordinator owns final updates below. -->

- 独立 helper 快照已安装：`/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/automation/v11-audit-20261011-a560b279eef2`；SHA-256 `a560b279eef244d7e0dc95110d32c542f0f50c5cdbf65fe8359b7deca3568473`，236/236文件验证一致。原快照224/224仍一致。新代码包含未来运行用的修复，但当前只调用外部验证/报告工具；旧trainer、旧scorer、模型和数据没有被热替换。
- 2026-10-11 19:16 UTC 完成作业替换。旧5个任务先hold、确认从未运行，再取消；其Elapsed均0且AllocTRES为空。训练7252874→7252875保留；DPO1评测7255085→7255086，final评测7255087→7255088；均先验lineage，成功后执行原冻结eval。新纯CPU逐文本复算7255089、DPO1完成证明7255090、最终报告7255091已提交，均Requeue=0。
- GPU上限保持84 GPU小时；新CPU审计累计墙钟上限12.5小时，按请求为386 CPU核小时。实际scheduler为CPU作业分配72CPU且无GPU，计费不能按请求核数推断。任务失败会明确报告不完整，不无限重试。
- 最新机器快照时间：2026-10-11T19:16:51.278030+00:00。干净DPO2 **280/1496 steps**；SFT原始4单元完成；CPU复算已真实RUNNING，其余DPO评测与报告等待依赖。随后19:20:23 UTC快照中前三套SFT共24,849条逐文本评分全部相同，第四套仍在计算。DPO1 lineage作业7255090已COMPLETED/0（用时62秒），重算7,244条准入、末步epoch1/it1810/step906，并证明final等于完整checkpoint；同一证明又在最终报告Python环境中以已有CPU分配的overlap step实跑通过，没有申请新GPU或新作业。
- 解释附录`interpretation-addendum.json`绑定原四数据集SHA，固定diagnostic_only；最终report必须先通过三个模型逐文本复算及两阶段lineage，失败不发布推断结论。
- 最终统一命令：`/Users/slian/miniforge3/bin/python -m pytest fastfill/tests fastfill/v1_1/tests -q`，**428 passed,90 subtests passed in18.31s**。`compileall`、shell语法和`git diff --check`通过。封包后逐文件验证本地源码没有漂移。
- Stage2发布补丁已经独立复审；五个seed(0/1/2/42/99)有效约束fixture的新旧输出byte-identical，既定pair策略未变。
- 可复查原始记录：`audit-evidence/audit-launch.json`、`audit-evidence/post-deployment.json`、`audit-evidence/unified-validation.json`。完整部署脚本和10项模拟测试也保存在同目录。

### 实际服务器完成验证 — 2026-10-11 19:23:03 UTC（伦敦20:23）

**SFT全量独立逐文本复算已完成**：CPU任务7255089于19:21:42 UTC以COMPLETED/0结束，用时5分31秒。四套各8,167/8,035/8,647/8,533，共33,382条；每个评分字段及四份metrics均与保存结果完全一致，结束前模型/数据/代码/产物身份复验也通过。正式凭据为`audit-evidence/sft-verification.json`，明确status=complete；这是可复现评分证明，不是模型质量胜出的证据。

DPO1完成来源验证任务7255090以COMPLETED/0结束，用时62秒；最终adapter等于完整末checkpoint，准入7,244对、优化906步，并核对merge lineage。最终报告所用Python环境又在同一已有CPU分配的overlap step中执行同一证明并成功。证据：`audit-evidence/dpo1-lineage.json`和`lineage-eval-runtime.json`。

19:23:03 UTC最新观察：干净DPO2 **340/1496 steps**（约22.7%），训练7252874正常RUNNING。DPO评测8单元和最终研究报告仍等待训练依赖，不能声称整个12单元科研比较已经完成。新门禁任务和最终报告已安排无人值守执行；报告将对DPO模型重复同样验证，并固定携带GT排序解释限制。

最终再次读取服务器原224文件和新236文件快照，全部无hash漂移；scheduler保存的wrapper脚本与部署源码字节一致。真实CPU审计AllocTRES中没有GPU，原训练任务保持原样，5个被替换旧任务均零运行时间且无资源分配。完整快照：`audit-evidence/post-deployment-completion.json`。

未来 request-only ID/顺序、重建训练数据、重新SFT或新增多seed/控制组/外部基线，均是独立协议下的后续研究，本文未将其标为完成。当前研究报告必须携带 GT 派生排序、warm-start provenance 与 box-level 指标限制，任何自动统计结论均限于已声明的诊断性比较。

## 证据索引

审计工作目录：`/tmp/fastfill-v11-reaudit-20261011/`。主要文件为 `data-report.md`、`train-report.md`、`geometry-report.md`、`evaluation-report.md`、`automation-report.md`、`live-report.md`；交叉复审为 `data-cross-review.md`、`geometry-fix-review.md`、`audit-report-review.md`；服务器机器证据为 `live.json`、`live-metrics.json`；数据扫描为 `data-scan.json`、`anchor-stats.json`、`pairs-recheck.json`。文件级阅读范围及哈希由 `inventory.json`、各分工 `*-coverage.json` 和 `root-coverage.json` 保存。

本报告不修改既有预注册事实，也不将修复后的源码版本追溯描述为旧实验当时实际使用的版本。原协议、原 launch/source manifest、原始模型输出与本轮复审证据共同界定可复现的实验来源。

### 已完成DPO1提前并行评测 — 2026-10-11 19:26:43 UTC

独立审查原冻结训练脚本后，确认DPO2正常路径只读DPO1基座；训练续跑会验收并跳过已完成DPO1。基于已经通过的完成/merge证明，首个DPO1评测7255085在hold并确认未运行后解除对整个训练链的等待，再release调度，允许与DPO2训练并行。评测入口仍重新验证lineage，续跑7255086、final和report依赖保持；解码/模型/数据/统计协议不变，不将评测结果反馈给当前训练。84GPUh上限不变，峰值申请可为训练4GPU加评测1GPU。操作回执：`audit-evidence/dpo1-early-start.json`。

19:28:15 UTC实机startup检查：7255085在nid010076 RUNNING，严格lineage再次通过，原frozen runner状态running/attempt1，并以原8192输出token/40960上下文/256chunk全量参数启动DPO1 dev评测。此时DPO2为390/1496steps。见`audit-evidence/dpo1-startup.json`；这时尚未宣称DPO1首chunk或任一完整单元已完成。
