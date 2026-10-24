# FastFill 最终修复补丁安全复核（2026-10-24）

**结论：下列代码快照范围内没有未关闭的补丁阻断项。** 交叉审查发现的 F09 超大整数溢出已按 TDD 补齐并验证。此结论不是“全环境无漏洞”：主代理的 [dependency-audit-summary.json](dependency-audit-summary.json) 仍报告依赖告警，依赖及部署环境的最终处置由主代理独立记录。

基线 HEAD：`873251f0b1f299230c42d510fe5c4f732aa77ce9`。逐项阅读全部 16 个已跟踪生产文件的 diff 和相关调用/输出边界，完整阅读新 `llm_baseline_v2.py`、一次性 `recover_cached_critic.py`，并检查对应回归测试。文件快照 hash 见末表。测试、证据与无关子模块/`Ref/` 不是本次生产 diff 的组成部分。

本审阅者此前实现 F03/F04/F09/F15/F16；这些修改另由 `audit_system_v11` 交叉复核，且该审阅者发现了本次关闭的 F09 大整数边界。对其他作者的生产修改，本记录是独立审阅；不将作者自审冒充完全独立复核。除主代理再次明确授权的 F09 小补丁与相应测试外，本轮只读检查源码。

## 编号更正

此前协作消息的技术意见正确，但两组编号互换。正确对应为：

| 编号 | 内容 |
|---|---|
| F02 | `critic_batch_025` 到下一轮的答案身份字段 |
| F08 | HTTP headers/body 的绝对读取期限 |
| F11 | QA 修复保留 prompt 约束 |
| F14 | judge agreement 缺失值、相关性和排名 |

## 已关闭发现：F09 的超大整数仍会中断整批

原先只捕获严格 JSON 序列化的 `ValueError`，未覆盖合法 JSON 整数 `±10**400`：返回 size 与浮点 prior 做 `abs(a-b)`，或父物体 z 与支撑高度相加时，会抛 `OverflowError`。这可以令一个房间的坏回答中断整批模型结果处理。

最小修改位于 `llm_agent._vec`：只允许可表示为有限数值的向量参与尺寸比较和支撑算术；仅捕获 `math.isfinite` 的明确 `OverflowError`。未捕获任意 schema/程序异常，也未把非法坐标替换成合法坐标。超大 pos 交回原合同解析器判失败；返回 size 无论是否可用，仍按 B2 被请求 prior 覆盖，保持尺寸覆盖合同。

原始改动前测试：四个 `±10**400 × size/pos` 案例 **4 failed in 0.42s**；含子物体的 36 房间 runner 案例 **1 failed in 0.61s**。修复后两个完整模块 **44 passed in 20.48s**，且 `git diff --check` 通过。

为保存可重现原始栈，另在独立 Python 进程内仅把 `_vec` 恢复为 HEAD 原函数，其他代码与工作树保持不变，重新得到 **5 failed in 0.57s**：[comparison-f09-large-integer-red.txt](comparison-f09-large-integer-red.txt)。这是修复后的证据回放，日志明确标记；没有将旧代码写回仓库。当前源码下同组测试加原 exponent 溢出对照为 **6 passed in 0.48s**：[comparison-f09-large-integer-green.txt](comparison-f09-large-integer-green.txt)。

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/slian/miniforge3/bin/python -B -m pytest \
  -p no:cacheprovider --import-mode=importlib -q \
  fastfill/comparison/tests/test_agents_016.py \
  fastfill/comparison/tests/test_llm_open.py
```

持久化 GREEN 命令：

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/slian/miniforge3/bin/python -B -m pytest \
  -p no:cacheprovider --import-mode=importlib -q \
  fastfill/comparison/tests/test_agents_016.py::test_size_enforcement_handles_integers_too_large_for_float \
  fastfill/comparison/tests/test_llm_open.py::test_nonfinite_placement_is_one_room_failure_not_run_abort
```

交叉审阅者在最终 `_vec` 上另跑 22 + 4 个定向案例全部通过，记录在 [system-v11-repair.md](system-v11-repair.md) 及其 comparison 测试日志。先前 [comparison-repair.md](comparison-repair.md) 的 127 项结果发生在这次追加修复之前；本节是其追加证据，最终全套回归由主代理执行。

## 已核对的安全边界

| 范围 | 核对结果 |
|---|---|
| F07：`v2/roomgenbench.py` | scene_key 在输出创建前验证：拒绝空值、`.`/`..`、绝对路径和 `/`、`\\`、NUL；解析后父目录必须为指定新输出。全部 payload 准备好后才创建目录，每个文件用 exclusive create。恶意 key 回归检查参考仓库字节不变、无部分输出。 |
| F10：新 harness 与 `v2/autorun.py` | `safe_output(..., create=True)` 拒绝已有目标；新文件 exclusive create。修复回答只有解析和检查都成功后才替换保存的 layout；失败保留前一个有效 layout，不回到外层再买首答。计数参数先验证。新版本、源代码 hash、冻结旧实现 hash、输入、模型/参数、endpoint、mode 和设置进入身份。默认 legacy 只复用身份匹配结果；身份不符或部分输出存在时保留并停止，不删除后重买。新生成须显式选择 repair-v2 与独立新路径。评分变化写新报告目录，保留旧评分。已核对历史 `llm_baseline.py` 与 HEAD 字节相同。 |
| F13：`submit_023_judge.sh` | full 阶段在 sbatch 之前要求 smoke exit_code 严格为整数 0、当前 bundle 前两例与 smoke manifest 一致、两例原始 attempt/result 身份和完成状态有效。`prepare_case` 只准备并校验字节，`reuse_result` 只读旧记录，不调用网络。凭据通过 env 文件读取，没有将 key 加入命令行、错误消息或新增报告。失败 smoke、丢失回执、改模型/请求/结果不能仅凭 results 目录解锁付费 full。 |
| F02：`critic_batch_025.py` | answer 同时携带 expected_model 与内容 hash。hash 基于 case ID、prompt SHA 和 image SHA，不受冻结复制路径变化影响。下一轮仍拒绝陈旧/不匹配身份；失败 client 的答案仍作为失败处理，未新增 retry。真实 batch → round 集成测试覆盖完成、失败、错误 hash；使用 mock transport。 |
| F08：`serve.py` | headers/body 共用 setup 后确定的 monotonic deadline；每次 socket recv 都按剩余预算设 timeout，慢滴字节无法无限续期。body 超时返回 408，不完整 EOF 返回 400；旧 1 MiB 上限保持。模型调用保持单线程串行，生成不计入读取期限，写响应时重设有限 timeout。loopback 测试覆盖 header/body × 停滞/滴流、较长生成及并发请求时模型串行。 |
| F05：`v1_1/train_dpo.py`、`training_guard.py` | 在加载 checkpoint 或重写 run manifest 之前绑定 base 字节、pair 字节、initial adapter、world size、dtype、recipe、audit/外部 guard 和代码；rank 0 身份失败先广播，其他 rank 不继续训练。仅有 checkpoint 进度不足以收养无身份历史目录。兼容旧 guarded run 时核对旧 guard、历史 manifest、原冻结代码，新增独立 sidecar，保留原 guard/manifest。正常 resume、不同数据/模型/recipe、初始 adapter、world size、audit/code 漂移均有回归。 |
| F01/F11：v1.1 handoff 与 QA | handoff 的支撑合同来自原请求，不由预测“自我改写”；requested on 约束继续导出且控制 repaired_valid。QA 把有效 constraints 与原 boundary 传入同一 snap_inside；坏 schema/reference 继续被诊断并跳过修复，未丢掉约束后虚报成功。 |
| F03/F04/F14/F15/F16：评分、选择、统计 | O2nt 仅明确的语义失败阻止重新计成功，普通旧 parser 失败仍可按 B2 重评分；S2 补入锁定的双向绝对 bias guard，并在缺失时不采用；缺失统计/排名输出 None 及实际分母，不填假值；网格抽样上限不会制造重复 UID，全失败报告保留原因。这些变更未引入网络、命令执行或新覆盖写入路径。 |
| F06/F12：cache 与 shell | cache 身份新增实际消费的 report 和外部 fit 文件；没有扩大反序列化来源。structured 包装脚本逐模式遇错停止后续评分，等待每个子进程并返回非零；不会在失败后打印 ALL-DONE。shell 路径参数已引用，未新增 eval 或凭据展开输出。 |

F03 历史影响已独立读取主代理的 [historical-o2nt-delta.json](historical-o2nt-delta.json)：四个真实房间、raw/harness 共 8 组旧新评分，所有数值/complete/success 字段均不变；四个 harness 记录仅 error 改为语义失败归因。不能把代码可构造的 36/36 反例扩大为这些真实旧结果的数值污染。

## 离线 critic 恢复脚本

完整阅读 [recover_cached_critic.py](recover_cached_critic.py)。未发现旧目录写入或网络调用路径：

1. 拒绝已存在/符号链接输出，解析后拒绝与历史输入目录相交；四个目标批次名固定。
2. 创建输出前验证所有旧 batch MANIFEST、state MANIFEST、active case inventory、当前 state/frozen case 内容 hash、原 attempt/result、旧 answers inventory 和文本/模型/错误的一致性。
3. 仅接受实际已完成、身份可核验的 cached answer；不猜测失败请求，不创建新的 geometry round，不请求 API。
4. 输出到全新独立目录，答案文件名为 case ID 的 SHA，采用 exclusive create；发布后重读比较，最终写出 summary 与文件 hash manifest 再验证。

脚本用途是补齐已验证 cached answers 的身份字段。四批 757 条真实结果及服务器执行成功与否以主代理留存的运行回执为准；本审阅者没有 SSH 执行该脚本，也不据此次交接缺陷推断历史 round 几何计算失败。

## 验证界限

- 本次“保留旧产物”结论限于修复的 baseline/rescore 路径和新离线恢复脚本。`autorun` F 阶段已有的 llmrows/test 重评目录替换，以及 v1.1 showcase 已有的场景输出替换未在本轮改变；不能概括为整个历史工具链绝不删除输出。本次服务器恢复使用独立新目录，不经过这些替换路径。
- 不自动重买已存在/身份不符 run，不等于供应商侧 exactly-once。冻结的 legacy HTTP transport 仍保留原有有界超时/服务错误重试；网络结果不确定时提供方是否已计费不能由此补丁证明。未新增无限重试或修复失败后的首答重启。
- 未执行付费 API、SSH、GPU/训练或凭据修改；未修改其他作者的生产文件。未新增依赖、修改 requirements 或执行环境升级。依赖扫描由主代理负责；当前摘要注明仅覆盖 13 项直接要求的已安装版本，未解析传递依赖。本记录不将带告警的环境标为安全洁净。
- 本记录不是公网认证/TLS、服务器 ACL、恶意同用户并发文件替换或整个仓库历史代码的全面认证；范围为下表快照中的补丁及相关信任边界。完整回归和最终提交内容校验由主代理另行留存。

## 审阅代码快照 SHA-256

| 文件 | SHA-256 |
|---|---|
| `fastfill/comparison/critic_batch_025.py` | `06fbc8a1d7304f398658e42147195b5a78f534b8518f1dca2a4664164bd2fa57` |
| `fastfill/comparison/decide_012.py` | `3ccc2ad100defa13db760c8e910abd3ce46864cfd9cdfd8e0022d8f8b88b9155` |
| `fastfill/comparison/evaluate_llm.py` | `e12696b4caebf1a77d5acb4cfb3e094f87538fdc397284252e7cd9c4755dd7f2` |
| `fastfill/comparison/judge_agreement_023.py` | `9da3d21d6b528818e4332b2c0e6cb377ea21b8b97f83dc3f791843b98bd854ca` |
| `fastfill/comparison/llm_agent.py` | `611ce03b0756a96e9f590a88a0a054d66a4a2bf588702bb84a19cb5b0bcfa23f` |
| `fastfill/comparison/mesh_collision.py` | `3b97e511c07eb71a90e298f437b91ef685378bfb44c2c5b819405169531bc7b7` |
| `fastfill/comparison/submit_023_judge.sh` | `e1b677971eff6e79d49449185b3e6ae8377dae1c443e6eef4238d8a0311ede45` |
| `fastfill/serve.py` | `7c116d807edda0a53a721451a480bb9861d0d47ad647d5996ecbe4a96867bfec` |
| `fastfill/tools/qa_report.py` | `82393a23b9a66c43b25a5b788465fcbf60e7362bfb34fea9a8b8685f5827e3bf` |
| `fastfill/v1_1/roomgenbench_v1_1.py` | `986ce5e8deac79727e47db45e80a523028dad5c6fcf1bcc07f63b36e095ed302` |
| `fastfill/v1_1/train_dpo.py` | `979a44c18463e3af364ecf7976a10009ce6e80564b34af29d899a7872cddc723` |
| `fastfill/v1_1/training_guard.py` | `e76663854bcd3570c6e48cc92a0352d26cd1f006f8fd489eaaf8fdad3675beb5` |
| `fastfill/v2/autorun.py` | `a267cc9bc18f2306a3b99156918d47e23b99dda0d2fb3b9b9d9884c4cd6391da` |
| `fastfill/v2/ops/run_structured.sh` | `e44a3a62e69be6f7b37604503cc563161b9f1e2003532ab07dfd6271c8551078` |
| `fastfill/v2/ops/stratify.py` | `260ba720d8bc38dc9027a048f1a9486e186ef7ee6f1b1fd557087184f0da8bb7` |
| `fastfill/v2/roomgenbench.py` | `7582e1a2651a5b8e2b5ff23f11817b22909546cc2103e39a5da62861bd70c87f` |
| `fastfill/v2/llm_baseline_v2.py` | `1278a3ed85a84c083925015304223276785c7c171ec3df5a862f2e2e5d7f0599` |
| `recover_cached_critic.py` | `538cecafa1d5d236ace24cd0ff32a7bcb2aa35e1e8aa0d5a2319b3dcb2546a28` |
