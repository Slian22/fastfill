# FastFill 修复与服务器验收（2026-10-24）

**状态：F01–F17 修复与本地、服务器验收全部通过。完整本地测试 3505 passed、6 skipped、305 subtests；全部生产代码语句覆盖率 84.28%。服务器 423 passed、2 skipped、67 subtests；757 条历史答案身份恢复完成，146 项训练保全检查通过。** 原审计 F01–F16 之外，服务器验收还发现并修复 F17：同长度权重修改后，基于文件时间戳的缓存可能返回旧模型哈希。此前失败记录全部保留。

基线提交：`873251f0b1f299230c42d510fe5c4f732aa77ce9`。最终候选内容身份：`daca130611a976b51193b7332c6464ad1b31214e43f2b7f29046dcc103dc17e1`，共 494 个源代码、配置、测试及必要参考文件。没有将训练权重、数据或旧实验结果打包进修复源码快照。

## 1. 修复范围

| 范围 | 修复内容 | 详细证据 |
|---|---|---|
| F01、F05、F13 | 导出保留原请求支撑合同；直接 trainer 绑定模型/数据/配方/代码身份并兼容经核验的历史 guard；付费 Judge full 必须通过真实 smoke 回执门槛。 | [system/v1.1 修复](system-v11-repair.md) |
| F02、F08、F11、F14 | critic 答案保留模型与请求身份；HTTP 头/body 共用绝对读取期限；QA 保留原约束；缺失 Judge 指标输出 null 和真实分母。 | [root 修复](root-repair.md) |
| F03、F04、F09、F15、F16 | 保留 O2nt 语义拒绝；补 S2 双向绝对 bias 门槛；隔离非有限值/超大整数的单房失败；网格抽样不重复；全失败诊断仍可输出。 | [comparison 修复](comparison-repair.md)、[整数溢出追加验证](independent-security-review.md) |
| F06、F07、F10、F12 | cache 绑定实际 report/fit；限制输出路径；新增独立 `repair-v2` harness 保留最后合法回答、避免失败后重复购买首答；shell 传播子任务失败。 | [v2 修复](v2-repair.md) |
| **F17，服务器新增发现** | 取消跨 `model_identity` 调用的文件 stat/SHA 缓存，每次重新验证权重头并流式计算文件哈希。身份 JSON 合同与未变输入的哈希不变。 | [F17 诊断与 TDD](server-identity-diagnosis.md)、[服务器 5/5 复现](identity-stat-probe.json) |

F17 的真实服务器探针中，5 次同长度修改均保留相同 dev、inode、size、mtime_ns、ctime_ns，但实际 SHA256 已改变，旧缓存仍返回旧值。确定性本地回归在修复前 **5 failed**，修复后 protocol、guard、lineage、checkpoint 四个完整模块 **124 passed**。修复的代价是每个模型身份校验边界重新读取模型文件；读取使用 8 MiB 分块。没有用睡眠、修改测试文件时间或放宽身份断言掩盖问题。

旧 `fastfill/v2/llm_baseline.py` 保持与历史提交 `510f1e0` 字节一致，SHA256 为 `bc50fa57e9250fe86442ed3981c0e9eef4d6388a7303245df0324e0cd97be459`。新实验使用 `fastfill.v2.llm_baseline_v2` 和全新输出目录；旧实验不被重新标记为新版本。

## 2. 测试与代码复核

最终完整运行：**3505 passed、6 skipped、305 subtests passed，591.59 秒，进程 exit 0**。[完整日志](pytest-final.log)、[JUnit](pytest-final.xml)、[解释器/依赖/精确调用](test-invocation-final.json)、[汇总核验](verification.json)。JUnit 的 3816 条包含 305 个子测试，不能再将其当成 3816 个独立主测试相加。

最终语句覆盖率：**32147 / 38141 = 84.2846%**，覆盖清单中全部 **183 个生产 Python 文件**，包括默认报告漏掉的 **15 个未导入模块**（按零覆盖计入），排除测试与外部参考仓库。达到 80% 门槛；不是分支或 GPU 路径覆盖率。[完整生产覆盖率](production-statement-coverage.json)、[原始覆盖率](coverage-final.json)。

本地 6 项跳过分别是 2 项需要训练环境的 `trl`，以及 4 项需要未挂载/未指定的真实数据。跳过原因逐项保存在核验 JSON，不计为通过。最终源码 494 项再次全部匹配快照；370 个 Python、81 个 shell/sbatch、23 个 JSON 语法检查通过：[语法与源码核验](syntax-verification-final.json)。

保留的验证过程：

1. 首次完整运行：3500 passed、2 failed、6 skipped、305 subtests。两项失败是旧 fake-DDP fixture 缺模型身份资产与广播 mock；补齐后相关 94 项通过，真实身份校验和空数据拒绝断言仍保留。[原日志](pytest-initial.log)、[fixture 修正](audit-pairs-regression.md)。
2. 第二次本地运行在服务器发现 F17 后主动中止，尚未修改源代码：2524 passed、6 skipped、281 subtests。它是未完成的中间运行，不作为最终全量通过证明。[原日志](pytest-interrupted-pre-f17.log)、[中止原因](pytest-interrupted-pre-f17-reason.txt)。
3. F17 后重新运行全部 FastFill 测试并重新收集覆盖率，得到上面的最终完整通过结果。正式覆盖率包含全部生产 Python 文件及未导入模块，排除测试代码，不使用含测试代码的默认高覆盖率作为结论。

生产修复、输入边界和恢复脚本已交叉审阅：[独立安全复核](independent-security-review.md)、[最终证据与 F17 独立复核](final-evidence-review.md)。各组测试有重叠，不将分组通过数直接相加。CPU 假模型/假 API、loopback socket、完整导出与批处理集成链有实际执行；本轮未进行真实多 GPU DDP、付费模型推理或全量场景生成验收。

## 3. 服务器隔离验收

最终快照：`/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/snapshots/repairs-20261024-daca130611a9/OptiScene`。

最终验收目录：`/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/repair-validation-20261024-daca130611a9`。

最终作业 **7684385：COMPLETED，exit 0:0，耗时 1:54**。测试 **423 passed、2 skipped、67 subtests passed**，随后离线恢复与训练保全均通过。源码快照和可写测试副本在执行后各 **494/494 哈希匹配**。[最终验收回执](server-final/validation.json)、[Slurm 记录](server-final/scheduler-final.txt)、[测试日志](server-final/pytest.log)、[源码清单](source-snapshot-manifest.json)、[验收脚本](server-validation.sbatch)。

服务器两项跳过分别需要未包含在源码快照中的 Git 历史 `cf518a24b` 和未安装在该机器的 `rebuild-main-20261007b` 数据集，不作为通过计数。服务器原始证据共 15 个文件下载后再次核对 SHA256 和字节数：[下载核验](server-final/download-verification.json)。最后观测用户队列为空。

验收使用不可变快照的字节一致可写副本，避免测试自身的 `copytree` 夹具继承只读权限。运行前后均验证副本与快照的全部源文件；离线 critic 恢复强制使用不可变快照代码。临时目录、Torch/Matplotlib cache、测试输出均在独立验收目录。绘图依赖安装到独立 QA 目录，共享训练环境没有执行安装或升级。[隔离依赖记录](server-qa-dependencies.txt)。

前两次失败保留：

| 作业 | 真实结果 | 处置 |
|---|---|---|
| 7682225 | 3 个 collection errors；Torch 默认缓存路径无写权限。 | [原回执](server-attempt1/completion.json)、[原日志](server-attempt1/pytest.log)。改为独立可写 cache。 |
| 7684339 | 389 passed、8 failed、2 skipped、67 subtests。3 项缺 matplotlib、1 项复制只读权限、4 项模型身份缓存漏检。 | [原回执](server-attempt2/completion.json)、[原日志](server-attempt2/pytest.log)。前四项修验收环境，后四项按真实 F17 修生产代码。 |

两次失败均在 pytest 阶段停止，没有继续执行历史 critic 恢复。全部验收作业 CPU only，`--gres=none`，CUDA 设备为空；不启动新的实际模型训练或付费 API 调用。

## 4. 对已训练模型与历史结果的实际影响

**27B-P/J 没有因本轮缺陷被判定需要重训。** [独立真实训练审计](../fastfill-live-training-check-20261024/REPORT.md) 已验证两模型分别完成 40000 房、10000 步训练，数据、adapter、历史源码和交付绑定一致，并在此前直接重算 33 个基座文件共 55.6 GB 的哈希。该审计使用直接字节哈希，不依赖 F17 的进程内缓存，不能因新发现缓存风险就宣告真实训练或旧权重损坏。

历史影响核对：

- **O2nt：**扫描 600 + 3000 + 1454 房回执，4 个最终回答带语义标记。8 组 raw/harness 旧新评分中数值全部不变，仅 4 个 error 归因改变；这些回答原本已经解析失败。[逐记录差异](historical-o2nt-delta.json)。
- **critic：**四批旧答案 **246 + 202 + 174 + 135 = 757 条**已在新目录补齐身份；原始 client results 的身份、内容和完成状态先全部核验。恢复没有 API 调用、没有新几何 round，也没有修改旧统计。[恢复回执](server-final/critic-recovery.json)、[新输出 MANIFEST](server-final/critic-answers/MANIFEST.json)。独立重读验证新输出 **758 个文件**，再验证 **8 份旧输入 manifest 及 8354 项历史文件**，全部匹配：[产物与历史输入保全](server-final/artifact-verification.json)。旧 round 有真实执行记录，不能把缺字段解读为历史 critic 从未执行。
- **S2：**已核验的冻结 decision 原本对 joint、placement 都不采用；增加 AND 门槛不会把该决定变为采用。没有重新计算全部历史服务器 S1 尺寸统计。
- **其他历史产物：**修复后的代码行为有回归证据，但没有原位覆盖全部旧 showcase、QA、分层报告或网格结果。逐项历史边界见[独立证据矩阵](final-evidence-review.md)。

最终部署后的训练保全检查 **146/146 通过**，重新读取 adapter、数据、历史源码及回执：[保全记录](server-final/training-preservation-full.json)。大基座分片在该阶段仅检查大小，不能把此前完整 55.6 GB 重哈希冒充部署后重复执行。

## 5. 交付边界

[提交前重新执行的依赖审计](dependency-audit-final-summary.json) 检查 13 项已安装直接依赖，结果仍是 3 个包有既有告警，67 条记录、51 个不同主 ID；传递依赖与独立服务器 QA 绘图工具不在该扫描范围。没有认证整个依赖环境安全，也未在本轮更换训练依赖。补丁复核无未关闭阻断项不等于全环境零漏洞。[原始依赖审计](dependency-audit-final.json)、[精确版本清单](dependency-audit-requirements.txt)。

修复源码与本证据包在同一 Git 提交中发布；[证据文件清单](MANIFEST.json) 登记相对路径、大小和 SHA256，源码身份另由源码清单绑定。原模型、冻结交付、旧代码快照与原始结果保留；本次发布的是独立修复源码与可追溯验证证据。
