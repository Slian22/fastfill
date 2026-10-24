# FastFill 修复：独立证据与历史影响复核

本复核只读检查修复记录、日志、保存的历史回执和当前文件字节；只写本文件，没有 SSH、API、GPU、训练、提交或推送。生产代码已有独立交叉审阅，本次不冒充再次逐行审查全部源码。

**F01–F17 修复的本地全量、服务器验收及 757 条离线答案恢复均通过独立证据复核，没有未解释的测试失败。** 本地 **3505 passed、6 skipped、305 subtests passed**；全部 **183 个生产 Python 文件语句覆盖率为 84.284628%**。已检查的 O2nt 数值未改变，27B-P/J 的既有训练和部署后的保存检查匹配。不能将这些范围内的通过扩展为全部历史结果已重新评估或全环境无依赖漏洞。

## 1. 独立核对结果

| 检查 | 结果与来源 |
|---|---|
| 审阅代码身份 | 从 [独立安全审阅](independent-security-review.md) 解析 18 条 SHA256，重新读取当前 17 个生产文件及一次性恢复脚本，**18/18 匹配**。 |
| F17 追加审阅身份 | 独立阅读新增生产 diff 与三个确定性协议回归、两项 guard/merge 强化测试；`evaluation_protocol.py` 最终 SHA 为 `3747133048d56591cb71ad95c040141a00b2051907b34e932cd00516bba5aef2`。连同原审阅，共覆盖 18 个生产文件及恢复脚本 19 项代码身份。 |
| 第二候选源码 | 内容身份 `0870dad7e23518463c54362a0e4bffa520f8c5c3eacb141a5b94df5a25df9c3a` 已独立核对 **494/494 文件**和排序 JSON 总哈希，全部匹配。相较第一候选，唯一变化是 `fastfill/tests/test_v1_1_audit_pairs.py`。这是该候选的验证记录，不自动认证后续哈希修复版本。 |
| F17 最终候选源码 | [最新源码清单](source-snapshot-manifest.json) 为 `daca130611a976b51193b7332c6464ad1b31214e43f2b7f29046dcc103dc17e1`；**494/494 文件**及清单总 hash 独立核对通过。相较第二候选恰好只增加三个 F17 生产/测试文件的内容变更，没有其他字节漂移。 |
| 训练证据完整性 | [训练证据 MANIFEST](../fastfill-live-training-check-20261024/MANIFEST.json) 登记的 **43/43 文件**重新核对大小和 SHA256，全部匹配。 |
| 训练核验记录 | 运行身份 146 项、completion 24 项、历史源码及交付绑定 23 项、基座文件 33 项，逐项 `passed` 均为 true。此处是保存回执的独立复核，没有再次连接服务器或重复哈希 55.6 GB 权重。 |
| 修复回归日志 | root 80 passed + 90 subtests；comparison 127 passed，追加整数溢出 6 passed，另有独立 22 + 4 passed；system/v1.1 137 passed；v2 133 passed。各组有重叠，不能直接相加为唯一测试总数。 |
| 初始全量两项失败 | 原全量日志为 3500 passed、2 failed、6 skipped、305 subtests。两项均为旧 `audit_pairs` CPU fixture；[修正记录](audit-pairs-regression.md) 和 [94 passed 日志](audit-pairs-green.log) 已核对。生产代码不变；修正提供可校验的模型资产和既有假分布式初始化所需的广播空桩，保留真实身份校验及空数据拒绝断言。 |
| 最终本地全量 | [完整日志](pytest-final.log)、[JUnit](pytest-final.xml) 和 [汇总](verification.json) 一致：**3505 passed、6 skipped、305 subtests passed，591.59s**；JUnit 共 3816 条 = 3505 + 6 + 305，0 failure、0 error。六项 skip 均有明确外部依赖/数据原因。 |
| 最终服务器验收 | [validation](server-final/validation.json) 为 `REPAIR_VALIDATION_PASSED`，job **7684385 COMPLETED 0:0**；**423 passed、2 skipped、67 subtests passed**。已独立核对 15 份下载文件 SHA/大小、JUnit、提交脚本身份、snapshot 与测试副本各 494 项验证记录。 |

独立复核新 fixture 的范围包括实际 diff：`bind_trainer_identity` 和数据过滤器没有被替换为 mock；禁止载入模型的原断言仍保留；增加 rank-zero 广播 `[None]` 和 world size 2 的断言。该测试仍是 CPU 模拟，不能称为真实多 GPU DDP 验证。

## 2. 全量生产覆盖率口径

最终复核输入为 [原始 coverage-final.json](coverage-final.json)、[最终生产覆盖率报告](production-statement-coverage.json) 和 [源码清单](source-snapshot-manifest.json)。统一规则是清单中的 `fastfill/**/*.py`，仅排除 `/tests/`；`fastfill/comparison/test_apply_012.py` 是生产评估入口，必须保留。独立核对原始 coverage SHA、183 个源码 SHA、完整源码集合、无重复条目和分项求和，全部匹配。

| 范围 | 覆盖语句 / 总语句 | 比例 |
|---|---:|---:|
| **F17 后最终全部 183 个生产 Python 文件** | **32147 / 38141** | **84.284628%** |
| F17 前候选，保留为历史记录 | 32153 / 38149 | 84.282681% |

默认报告未登记 15 个未导入的 `optiscene_official` 与 `v2/ops` 模块。协调者用生成原始数据的 **coverage 7.17.0** 对这些文件直接执行 `analysis2`，共识别 **1198 条语句**，在完整口径中全部计为未覆盖；独立核对它们的 covered 均为 0。最终分母为 `36943 + 1198 = 38141`，不包含测试语句。原 [候选覆盖率](production-statement-coverage-initial.json) 绑定 `server-attempt2` 的旧源码清单并标明被最终报告取代，未覆盖旧证据。

最终语句覆盖率高于 80% 门槛；这不是分支覆盖率、真实 GPU 训练/推理覆盖率或未挂载外部数据的验收。初始运行、F17 前主动中断运行、局部重跑和服务器测试相互重叠，不把它们相加成额外唯一通过项。

## 3. 逐项修复与历史影响界限

| 项目 | 修复证据 | 历史结果能支持的结论 |
|---|---|---|
| F01 支撑合同 | [system/v1.1](system-v11-repair.md)：原请求支撑和 `on` 约束保留，含完整 CPU 假模型展示链。 | 修复后导出行为已有验证；没有在本证据包中重导出并重评全部旧 showcase，不能宣称历史所有支撑统计已更正。 |
| F02 critic 身份交接 | [root](root-repair.md)：batch → round 的成功、失败、陈旧身份集成测试。 | 四批旧 `answers/` 确实缺字段，共 757 条；原 client results 含身份。[离线恢复已完成](server-final/critic-recovery.json)，只补身份，不重算布局、几何或统计。 |
| F03 O2nt 语义失败 | [comparison](comparison-repair.md)：保留明确语义拒绝，同时保留普通 B2 重评分。 | 已扫描 dev-select 600、confirm 3000、cohort 1454 条回执。四个最终回答被标记；所选 8 组 raw/harness 旧新评分仅 error 归因改变，数值无变化。 |
| F04 S2 绝对 bias 门槛 | [comparison](comparison-repair.md)：双向差值、边界、缺失分母等回归。 | 已核对本地冻结 `decision-012/decision.json` 与其 MANIFEST：joint 与 placement 原本均 `passed=false`。增加 AND 条件不会将该决定变为采用；未在本复核中重新计算服务器全部 S1 尺寸统计。 |
| F05 trainer 身份 | [system/v1.1](system-v11-repair.md)：tiny Qwen 恢复、漂移拒绝、受保护旧运行兼容。 | [真实训练审计](../fastfill-live-training-check-20261024/REPORT.md) 证明 27B-P/J 各完成 40000 房、10000 步，数据/模型/源码及交付绑定匹配。没有因 F05 要求这两模型重训的证据。 |
| F06 stratify cache | [v2](v2-repair.md)：report 和实际外部 fit 纳入 cache 身份。 | 修复后依赖变化会失效；未重新生成历史缓存或全量重算旧分层报告。 |
| F07 输出路径 | [v2](v2-repair.md)：非法 scene key、目录范围和排他创建。 | 本地攻击输入回归证明参考文件未被写入；未据合成攻击推断历史服务器曾遭覆盖。 |
| F08 HTTP 读取期限 | [root](root-repair.md)：本地 socket 停滞/滴流、后续请求、串行生成。 | 服务故障模式已复现并修复；没有把本地 socket 证据当作历史服务器故障率测量。 |
| F09 非有限值与超大整数 | [comparison](comparison-repair.md)、[追加整数修复](independent-security-review.md)：坏房间失败而整批继续。 | 没有凭单个缺失 manifest 判定所有旧中止由此造成；未重新运行全部历史推理。 |
| F10 repair-v2 | [v2](v2-repair.md)：保留最后合法布局、显式版本与全新目录。 | 冻结 `llm_baseline.py` 字节保持原样；旧实验不被静默改名或改分。新 harness 的合成回归不证明每个历史回答都遇到该错误。 |
| F11 QA 约束 | [root](root-repair.md)：修复与 runtime 使用相同 constraints/boundary。 | 约束保留回归通过；未原位覆盖历史 QA 统计。 |
| F12 shell 退出码 | [v2](v2-repair.md)：各子进程失败传播，失败链不继续评分。 | 本地子进程空桩覆盖错误路径；不能把旧 `ALL-DONE` 一律解释为真实各子任务成功或失败。 |
| F13 Judge smoke 门槛 | [system/v1.1](system-v11-repair.md)：10 类拒绝及有效 smoke 通过。 | 无真实付费全量提交；现有审计未证明旧 full 曾因该缺陷误发。 |
| F14 缺失 Judge 统计 | [root](root-repair.md)：缺失、常数、空/单样本以 null 和真实分母表示。 | 未重新购买评分，也未补造 overall 或相关性。 |
| F15 网格抽样上限 | [comparison](comparison-repair.md)：上限超过人口时不重复 UID。 | 修复抽样边界；未把所有旧网格结果判为重复或无效。 |
| F16 网格全失败报告 | [comparison](comparison-repair.md)：缺失误差显示为未测量，保留原因。 | 能发布失败诊断不等于碰撞分析成功；未补造网格误差或成功率。 |
| **F17 新增：模型身份缓存** | [服务器真实 probe](identity-stat-probe.json)：5/5 次同 stat 修改漏检；移除跨次缓存后 [5 项 RED](f17-red.log) → [相关 124 项 GREEN](f17-green.log)，最终 diff 已独立审阅。 | [最终服务器 JUnit](server-final/pytest.xml) 中四个原失败及三个新边界测试全部通过。该项独立于原 F01–F16；不据此自动否定已另行流式哈希核对的 27B-P/J。 |

### O2nt 的逐记录检查

独立读取 [historical-o2nt-delta.json](historical-o2nt-delta.json) 的全部 8 项并逐键比较 `old`、`new`：四个 `llm_raw` 完全相同，四个 `llm_harness` 只有 `error` 不同。全部原本 `complete=false`、`geometry_success=false`、`geometry_success_repaired=false`。该结果支持这四个房间的数字不变，不声称重新跑了全部 cohort 或更换模型。

### critic 的实际旧执行

[historical-scan.json](historical-scan.json) 的四批答案数为 **246 + 202 + 174 + 135 = 757**，缺身份数量各自相同。[historical-detail.json](historical-detail.json) 保存的旧 round-1 记录分别有 **217** 与 **157** 个 accepted edit sets，且绑定不同于当前消费者的历史代码 hash。因此不能声称缺少当前所需字段意味着历史 critic 从未执行。新恢复 bundle 的用途是让已验证旧答案可被当前身份合同消费；数值质量与后续几何 round 是独立问题。

## 4. 服务器部署脚本与已有回执

已完整阅读 [deploy-snapshot.py](deploy-snapshot.py)、[server-validation.sbatch](server-validation.sbatch) 及其调用的 [只读训练核验脚本](../fastfill-live-training-check-20261024/remote-verify.py)。本次新增部署脚本没有发现未关闭的覆盖、GPU 或付费调用阻断项：

1. 部署前核对 archive SHA、只允许 `OptiScene/` 内常规文件、拒绝 `..`，核对完整成员集合与 manifest 内容哈希。`snapshot.mkdir` 不允许已存在目标；解包后核对 494 项，再只对这个新快照设置只读权限。没有修改历史训练快照的路径。
2. 提交标记排他创建，然后才执行 `sbatch`。脚本明确 `--gres=none`、`CUDA_VISIBLE_DEVICES=''`、模型离线标志，仅申请 CPU；运行临时文件、Torch/Triton cache、pytest 输出都在独立 RUN。测试中的 tiny CPU 训练用于回归，不代表重训交付模型。
3. 最终脚本先核对 immutable SNAPSHOT 的全部 494 项，把它 `copytree` 到独立 `RUN/test-checkout/OptiScene`；目标已存在时拒绝。仅对此新副本增加可写权限，随后再次核对字节。这样 fixture 的二次 copy 可以修改临时测试文件，而 immutable SNAPSHOT 权限和内容保持不动。26 个模块包含 F17 的 `test_evaluation_protocol.py`。MPL 及依赖安装在独立 QA packages 路径，见 [安装回执](server-qa-dependencies.txt)；本审阅者核对的是该操作边界和回执，未把它称为共享环境的全量字节审计。
4. `set -euo pipefail` 使 pytest 失败时不进入恢复。离线恢复使用 immutable SNAPSHOT 内脚本的绝对路径，并单次重设 `PYTHONPATH` 指向 immutable SNAPSHOT；输出为独立 `RUN/critic-answers`，恢复脚本拒绝现存目标及与历史目录重合，无外部 API 调用。随后训练核验只读旧数据、adapter、源码和回执；报告写到 RUN。
5. 最终 validation 要求训练核验通过、757 条恢复、零 API/新几何 round、immutable SNAPSHOT 与可写测试副本的 494 项源码均保持原 hash、JUnit 无 failure/error，并使用排他创建。已读取 [最终部署/提交回执](deployment-final-stdout.txt)：作业 **7684385**，源码为 `daca130611a9`，实际提交脚本 SHA 与独立审阅的 `1c982f44fe5416fa10fd33f1aca4d1a69e48863380a509f3b23b146e65686e14` 完全匹配。[最终 validation](server-final/validation.json) 记录这些条件全部通过。

已读取 [旧作业 completion](server-attempt1/completion.json)、[旧 pytest XML](server-attempt1/pytest.xml) 与 [旧日志](server-attempt1/pytest.log)：**7682225 exit 2**，3 个 collection errors，首个错误是 Torch cache 的 `/local/user/1483807663` 无写权限，后续为初始化连带错误。没有据此宣称代码回归成功，也没有把未执行的 critic 恢复计为完成。第二候选把 cache 重定向到新 RUN，曾通过 **494 项字节核对**并提交 CPU 作业 **7684339**；该次八个测试失败已另存 [第二轮日志](server-attempt2/pytest.log)，没有进入恢复阶段。两个失败尝试均保留，不以最终通过覆盖它们。

证据归档已完成：[原全量日志](pytest-initial.log)、[原 JUnit](pytest-initial.xml)、[v2 初始 RED](v2-red.log)、[v2 参数 RED](v2-parameters-red.log) 均保存。两份 v2 RED 与原临时文件的 SHA256 相同，分别为 `277d4607a4a3296ae8703a183686be9c1d1303c8231ca6b78cae0ace33df12a8`、`67ee4d3efff8402f28307a43b036e8a1d198a8a692db435ca58a4accd4404822`。

## 5. 最终服务器回执与保存检查

独立重新读取 [下载清单](server-final/download-verification.json) 的 **15/15 文件**并计算 SHA256、检查大小，全部匹配。本审阅者没有再次 SSH；以下远程文件重哈希数量来源于已验证的实际执行回执，不冒充本机持有并重读所有远程数据。

| 回执 | 独立核对结果 |
|---|---|
| [Slurm](server-final/scheduler-final.txt)、[completion](server-final/completion.json) | `7684385 COMPLETED 0:0`，耗时 **1:54**，completion exit 0。调度资源记录没有 GPU；提交脚本也明确 `--gres=none`。 |
| [pytest 日志](server-final/pytest.log)、[JUnit](server-final/pytest.xml) | **423 passed、2 skipped、67 subtests passed，81.34s**；JUnit 的 492 条 = 423 + 2 + 67，failure/error 均 0。子测试不能再次加进主测试通过数。F17 四个原失败及三个确定性新测试全部真实执行并通过。 |
| [源码/测试副本 validation](server-final/validation.json) | 内容身份为 `daca130611a9`；immutable snapshot 和测试副本各 **494 项**在执行后哈希通过。 |
| [critic recovery](server-final/critic-recovery.json)、[输出清单](server-final/critic-answers/MANIFEST.json) | `CACHED_ANSWERS_RECOVERED`，**246 + 202 + 174 + 135 = 757**；清单为 757 个答案 + 1 个 summary，共 758 项。summary 与 recovery 回执字节相同。`api_calls=0`、`new_geometry_rounds=0`。 |
| [历史产物复核](server-final/artifact-verification.json)、[训练保存检查](server-final/training-preservation-full.json) | 758 个新输出重新校验；8 份旧 state/batch manifest 的 SHA 与 recovery 绑定逐一一致，共 **8354 个历史文件**校验通过。训练 **146 项全部通过**，整个 checks 列表与部署前保存的 server-verification 完全一致。 |

服务器两项 skip 分别因为只读归档没有 Git 历史 `cf518a24b`、机器没有 `rebuild-main-20261007b` 数据；没有将它们称为执行通过。最终 [队列记录](server-final/queue-final.txt) 为空。恢复期间没有新模型训练、几何 round 或 API 购买。

训练保存检查重新哈希 adapter、数据、冻结源码和基座元数据，并检查基座权重大小；其 `base_weight_shards_rehashed=false`。不能声称部署后再次完整哈希了 55.6 GB 基座；部署前独立完整基座检查仍由先前 [base-verification.json](../fastfill-live-training-check-20261024/base-verification.json) 支持。

**本地最终完整套件已完成并通过。** 已读取 [中断日志](pytest-interrupted-pre-f17.log) 和 [原因记录](pytest-interrupted-pre-f17-reason.txt)：协调者为先修复 F17，在生产源码变更前主动 SIGINT，中途 2524 passed、6 skipped、281 subtests；该计数不作为失败或最终完成验收，不与最终运行重复相加。最终全量重新执行得到 3505 passed、6 skipped、305 subtests，且 494 项源码 hash 在完成后再次匹配。

六项本地跳过分为两项缺少 TRL、一项未挂载冻结测试集、一项缺少 `FASTFILL_EVAL_DATA`、两项未挂载外部原始数据。它们没有计入通过项；没有剩余未解释 failure/error。服务器的两项环境 skip 与本地六项不合并为一个“全局跳过数”。

## 6. F17：第二轮服务器新增模型身份缓存失效

已独立读取 [第二轮 JUnit](server-attempt2/pytest.xml) 的全部失败栈与 [测试日志](server-attempt2/pytest.log)：**8 failed、389 passed、2 skipped、67 subtests passed**。其中三项缺少 matplotlib，一项 fixture 的 `copytree` 继承了只读快照权限；这些属于隔离测试环境/fixture问题。另四项确实执行了同长度权重修改后，原本应拒绝的 guard、merge 或 lineage 校验没有拒绝，不能归入环境安装问题：

- `test_resume_rejects_changed_training_identity[weights]`；
- `test_merged_output_must_match_exact_base_adapter_and_output`；
- `test_lineage_rejects_independently_corrupted_evidence[wrong_output]`；
- `test_lineage_rejects_independently_corrupted_evidence[wrong_base]`。

只读检查 `evaluation_protocol.py` 并结合 [服务器 probe](identity-stat-probe.json) 确认机制：`_model_file_digest` 的跨调用 LRU 仅以路径及 `(dev, ino, size, mtime_ns, ctime_ns)` 为键，命中后既不重读权重字节，也不再次校验 safetensors。服务器 5 次真实同长度修改中，五项 stat 均完全相同，mtime/ctime 显示整秒精度；实际 SHA 均从 `8fa8d104…` 变为 `f29d2251…`，而 `model_identity` 仍报告旧 SHA。该 probe 绑定旧模块 SHA `750c0582c44e0b07f7646a8a486ec9d3104d496bd3f5c1c94085c106c41699b8`。本审阅者读取并核对回执，未自行连接服务器；确定性回归与最终修复由修复负责人执行。原独立审阅的 17 个生产文件不包含该模块，新改动需追加审阅身份。

最终最小实现已移除 `lru_cache` import、旧 `_model_file_digest` 与 stat signature。`_model_file_identity` 对 safetensors 每次执行原 header 校验，然后直接返回 `file_identity(path)`；身份 JSON schema、排序和 hash 算法保持不变。该共享函数同时服务于训练 guard、直接 trainer、merge、lineage 和 evaluation，修复因此覆盖所有调用者。没有增加 sleep、TTL、额外时间戳启发式或部分权重抽样。

本审阅者独立阅读完整 diff、[RED](f17-red.log) 和 [GREEN](f17-green.log)，未发现新的阻断项。三个新协议测试分别覆盖真实同长度权重改写、tokenizer JSON 改写、同 stat 的非法 safetensors 头；两项既有 guard/merge 拒绝测试也固定完整 stat，从而不再依赖本地高精度时间戳。旧实现 **5 failed in 4.64s**；四个完整 protocol、guard、lineage、checkpoint 模块 **124 passed in 7.93s**。这些日志由修复作者执行，本审阅者没有重复运行。scoped `git diff --check` 独立执行通过；没有发现旧缓存引用残留。

| F17 冻结文件 | 独立核对 SHA256 |
|---|---|
| `fastfill/v1_1/evaluation_protocol.py` | `3747133048d56591cb71ad95c040141a00b2051907b34e932cd00516bba5aef2` |
| `fastfill/v1_1/tests/test_evaluation_protocol.py` | `4d5ae134dae6b3839548ac79ac1a6660e15a2def168bec4e563a316b01925f9e` |
| `fastfill/v1_1/tests/test_training_guard.py` | `638cfe4edcf3ed1f40ee11cdea0b57094be7b9089635c8a2fac04535fb6b75de` |

性能代价是每次身份边界重新读取全部模型文件；对于此项目的 55.6 GB 基座，这是实际 I/O 成本，但原 8 MiB 分块内存上界可保留。可在单次身份构建中去除重复路径（现有 set 已有此行为），本次不引入未经证明的跨边界复用。基于不可变存储的缓存优化需要单独建立合同和测试。该修复不要求修改既有 manifest、重新推理或重训模型。

上述方案解决两次校验之间已经完成的内容变化，不将普通文件哈希描述为可对抗恶意并发原位写入的原子快照。训练历史审计使用独立流式 SHA 核对实际权重和数据，相关已保存证据不因该缓存问题自动失效；同样不能据此断言其他未审计历史运行不受影响。

[最终依赖审计摘要](dependency-audit-final-summary.json) 保留已安装直接依赖的既有告警，范围未覆盖传递依赖。独立补丁审阅没有未关闭的补丁阻断项，不等于依赖环境安全认证。本复核范围内的本地、服务器、恢复及保存验收项均已关闭；提交和推送由协调者单独负责，不是本审阅者执行的动作。
