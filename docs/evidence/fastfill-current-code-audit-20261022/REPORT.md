# FastFill 当前代码审查 · 2026-10-22

**优先修复：RoomGenBench 支撑要求被预测结果改写，以及 critic 批处理答案无法进入下一轮。**

本次确认 **16 项问题：2 项 P1、12 项 P2、2 项 P3**。均附当前源码位置及本地复现证据。现有测试合并结果为 **3,394 通过、6 跳过，另有 305 个子测试通过**；全量生产 Python 语句覆盖率 **83.73%**。通过测试不能排除下面已复现的跨模块错误。

审查版本：`825fd13ea18f45835db1a242197359a77ea99f9e`。没有修改生产源码、测试、模型或数据；交付的是审查文档与证据。以下“可触发”均指已在本地验证的当前代码行为，**不等于已经证明某次历史实验、论文数字或交付结果受影响**。

**交付复核：2026-10-24。** HEAD 未变，271 个生产/配置文件与 171 个测试文件共 442 项的 SHA256 全部与审查时一致，范围内没有新增或删除文件。测试及复现执行于 2026-10-22；本次继续交付时核验版本、证据完整性和报告链接，没有重复运行未变化的测试。

## 范围与方法

逐文件审查 `fastfill/` 下 **271 个生产/配置文件、68,202 行**，覆盖 v1、25 个数据适配器、工具、v1_1、v2（含 ops）、system、comparison（含官方代码包装及全部 50 个启动脚本）。其中生产 Python 为 182 个文件，shell/sbatch 为 71 个，JSON/YAML 为 18 个；另阅读相关依赖文件、README、协议与测试契约。

| 范围 | 生产/配置文件 | 行数 |
|---|---:|---:|
| 顶层、adapters、tools | 43 | 7,409 |
| v1_1、system | 28 | 8,386 |
| v2，含 ops | 78 | 16,423 |
| comparison，含脚本和 vendor 配置 | 122 | 35,984 |

完整清单：[逐文件审查 CSV](production-review-coverage.csv)、[逐文件 SHA256 与审阅记录](production-review-coverage.json)。所有 271 项均有完整生产代码阅读记录，交付时再次核对源码哈希未变。**171 个测试文件全部纳入测试收集与执行；测试源码只对相关契约、复现辅助函数和覆盖缺口作重点阅读，不声称逐行读完全部测试。**

遵循项目的图优先导航：审查开始时，MCP 返回的是相邻的另一个 FastFill 仓库索引，不能用于本工作区；当时已查询本地 graphify 图，再依据当前源码核实。未把旧图、旧审计或旧文档中的结论直接沿用。

## 先修复的两个 P1

### F01 · P1 · v1_1 导出把错误支撑关系变成输入要求

位置：[roomgenbench_v1_1.py:320](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/v1_1/roomgenbench_v1_1.py#L320)，另见 `:367`。

原请求明确规定灯在桌上，模型却把灯放在地面。`placements_to_v2_handoff` 从预测的 `on` 生成 condition 的 `support_parent`，同时清空 constraints，导致后续验证针对改写过的条件进行。真实几何修复、导出和 reference_check 的 CPU 复现得到：原 `on` 约束为假，但 FastFill `ok=True`；原始 GT 反而因被改成 floor 要求而 `ok=False`。

修复方向：输入的支撑要求和约束必须从原请求保留并做 ID 映射；随后检验模型预测是否满足要求。不能以预测重写验证条件。范围限于 v1_1 RoomGenBench 展示/导出链，不能推断 amendment-022 system 同样有此缺陷。证据：[system-v11-repros.json](system-v11-repros.json)，细节：[system-v11-review.md](system-v11-review.md)。

### F02 · P1 · critic 批处理与下一轮的答案格式不兼容

位置：[critic_batch_025.py:101](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/comparison/critic_batch_025.py#L101)，消费者 [critic_025.py:683](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/comparison/critic_025.py#L683)。

按 `submit_025.sh` 和批处理文档使用 `OUT/answers` 时，生产者未写 `expected_model`、`content_sha256`，消费者却强制要求两者。用已有假 SSE 传输跑完整批处理，一条成功答案生成后，真实 `round_` 仍报 `answers not made on this round's cases or model`，没有下一轮产物。现有测试在消费者测试里手工补齐身份字段，或仅调用 `_step`，因此没有捕捉这次交接失败。

修复方向：每条答案，包括失败答案，都携带冻结 case 的模型及内容身份；补批处理到 `round_` 的集成测试。保留消费者的身份校验。原始 `cases/*/results/*.result.json` 是另一路保留身份的格式，本问题明确针对文档要求使用的 answers 目录。证据：[root-comparison-probes.json](root-comparison-probes.json)。

## P2：评估与训练身份

| ID / 位置 | 已复现的行为与影响 | 修复方向 |
|---|---|---|
| **F03** · [evaluate_llm.py:109](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/comparison/evaluate_llm.py#L109) | O2nt 返回思考内容后本来应判失败，评分器重新解析布局却恢复为成功。36 个合成房间：runner **0/36 valid**，评分 **36/36 complete、几何成功率 1.0**；正常无思考控制组为 36→36。 | 将 runner 的语义失败持久传入评分器；不能仅重新解析 raw。 |
| **F04** · [decide_012.py:106](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/comparison/decide_012.py#L106) | S2 联合臂未执行预注册的绝对 footprint log-bias 差 ≤0.03。复现差 **0.81093**，仍 `passed=True`。候选生成只防缩小，不能替代绝对差门槛。 | 显式计算并记录协议定义的 bias 差，作为采用条件。 |
| **F05** · [train_dpo.py:438](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/v1_1/train_dpo.py#L438) | 同一路径更换成同长度的偏好数据后，直接 CLI 恢复旧 checkpoint，显示 `step=2/2`，**零步新训练**仍重新写 completion；真实 tiny Qwen 的所有最终 adapter 张量不变。 | 在覆盖 manifest 或读取 checkpoint 前，绑定数据、模型和训练配置身份。 |
| **F06** · [stratify.py:223](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/v2/ops/stratify.py#L223) | 缓存键遗漏外部 baseline fit/report 依赖。改动 fit 后，缓存仍报告 exact=True、0 个不匹配；不使用缓存时 exact=False、3 个不匹配。 | 把 report 和实际消费的所有 fit 依赖纳入缓存身份。 |

F03 的 O2nt 只是参考臂，不进入该选择器的训练模型候选集合；本地没有足够逐房收据证明历史汇总已被污染。F04 的现有冻结 decision 本来就拒绝 S2，因此未发现该份决定被误改成采用。F05 的新版 clean sbatch 外围有 `training_guard`，按该入口运行会拦截所演示的数据漂移；问题是直接 trainer 和未加 guard 的旧入口仍可绕过。

证据：[comparison-probes.json](comparison-probes.json)、[system-v11-repros.json](system-v11-repros.json)、[v2-ops-repros.json](v2-ops-repros.json)。详见 [comparison-review.md](comparison-review.md)、[system-v11-review.md](system-v11-review.md)、[v2-ops-review.md](v2-ops-review.md)。

## P2：输入边界与服务可靠性

| ID / 位置 | 已复现的行为与影响 | 修复方向 |
|---|---|---|
| **F07** · [roomgenbench.py:286](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/v2/roomgenbench.py#L286) | `scene_key='../reference/protected'` 被直接拼成输出路径；临时目录复现成功覆盖输出目录外已有的 reference JSON。 | 使用安全单一文件名、解析后校验仍位于输出目录、排他创建。 |
| **F08** · [serve.py:74](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/serve.py#L74) | 单线程 HTTPServer 无读取期限。一个只发送部分 body 的客户端令第二个完整请求阻塞；补齐第一条后第二条立即恢复。1 MiB 大小限制不能限制等待时间。 | 对连接头和 body 设置期限，保持模型调用串行；不能直接引入无保护的多线程 LLM 调用。 |
| **F09** · [llm_agent.py:185](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/comparison/llm_agent.py#L185) | placement 答案含 `1e309` 与 size 时，尺寸覆盖后的 JSON 序列化抛出未捕获 ValueError，整批推理退出而未形成逐房收据。joint 路径对同类坏答案能返回无效状态。 | 在数值边界统一拒绝非有限数值，让坏答案进入正常单房失败路径。 |
| **F10** · [llm_baseline.py:211](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/v2/llm_baseline.py#L211) | 首答 JSON/对象集合法但几何有缺陷，修复调用发生连接错误时，旧 harness 丢弃首答；复现 `first_answer_valid=True`，最终 `layout=null、failed=1`。 | 保留已有可解析答案并记录修复调用失败；需要单独版本化，避免静默改变已冻结旧实验身份。 |

F08 是旧文档已有、当前仍存在的问题。F10 的新 `llm_structured.py` 路径已经保留首答，不能把旧 harness 缺陷扩展到新路径。证据：[v2-repro-results.json](v2-repro-results.json)、[root-probes.json](root-probes.json)、[comparison-probes.json](comparison-probes.json)。安全复核：[security-review.md](security-review.md)。

## P2：统计与运行状态

| ID / 位置 | 已复现的行为与影响 | 修复方向 |
|---|---|---|
| **F11** · [qa_report.py:201](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/tools/qa_report.py#L201) | QA 修复不传 prompt 约束；同一带 faces 约束的参考布局，QA 报修复后 inside **1/1**，实际约束保护的运行时修复仍在墙外并拒绝。 | 使用与运行时一致的约束和边界参数。该旧问题仍存在。 |
| **F12** · [run_structured.sh:8](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/v2/ops/run_structured.sh#L8) | 所有生成、评分进程均 exit 42，脚本仍输出 `ALL-DONE` 并 **exit 0**。后台 `wait` 未逐 PID 传播失败。 | 每条链失败即停止，收集每个后台 PID 的退出状态。 |
| **F13** · [submit_023_judge.sh:35](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/comparison/submit_023_judge.sh#L35) | smoke 已 exit 1、零成功结果，只因 results 目录存在，`full` 仍会提交 696-case 作业。实际客户端在 API 调用前就创建该目录。复现只用了本地 Slurm 空桩。 | 同时核验成功 completion 与两条预期 case 的模型/配置/有效结果后再允许 full。 |
| **F14** · [judge_agreement_023.py:47](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/comparison/judge_agreement_023.py#L47) | 两个 judge 各有 4 条合法完成结果，但都缺可选 overall 时，`zip(*pairs)` 抛异常，无法生成一致性报告；共同成功样本为零或不足两个也未正确处理。 | 对缺失指标和样本不足输出 NA 与数量，避免对空值强行相关计算/排名，不隐式填补 overall。 |

证据：[root-probes.json](root-probes.json)、[v2-ops-repros.json](v2-ops-repros.json)、[comparison-shell-b-repros.json](comparison-shell-b-repros.json)、[root-comparison-probes.json](root-comparison-probes.json)。未发现真实付费全量作业因 F13 已经误发。

## P3：网格诊断的边界处理

| ID / 位置 | 已复现的行为 | 修复方向 |
|---|---|---|
| **F15** · [mesh_collision.py:477](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/comparison/mesh_collision.py#L477) | 仅 2 个 rendered 房间时使用 limit=100，会重复采样同一 UID，配对统计报重复/覆盖错误，无法发布输出。 | 使用截断后的有效 limit 计算抽样索引。 |
| **F16** · [mesh_collision.py:395](https://github.com/Slian22/fastfill/blob/825fd13ea18f45835db1a242197359a77ea99f9e/fastfill/comparison/mesh_collision.py#L395) | 全部网格失败时，summary 正确保留缺失原因，但 report 把 `None` 按 `.2e` 格式化而崩溃，整份诊断产物不能发布。 | 未测得 bbox 误差时输出“未测量”，保留零成功与缺失报告。 |

证据：[comparison-mesh-probes.json](comparison-mesh-probes.json)。这两项是诊断工具边界缺陷，优先级低于前面的验证和训练身份问题。

## 测试、覆盖率与限制

完整 CPU 套件第一轮：`3376 passed, 18 failed, 6 skipped, 305 subtests passed`，610.09 秒。18 项失败都发生在 spawn 子进程导入 checkout 包 `OptiScene` 时；将 checkout 的父目录加入 `PYTHONPATH` 后，仅重跑这些失败项，结果 **18 passed**，89.07 秒。没有改测试或生产代码来取得通过。最终合计 **3394 passed、0 个未解决测试失败、6 skipped、305 subtests passed**。

默认 Python 环境的旧 transformers 缺少 Qwen3Config，且 pytest 默认导入方式会遇到同名 `test_data` 冲突；正式验证使用现有 miniforge Python 3.12.9、Torch 2.13.0、Transformers 5.14.1、pytest 9.1.1 和 `--import-mode=importlib`。只把 coverage 工具安装在临时 QA 目录，没有改变项目依赖环境。

语法验证：353 个 Python 文件 AST 通过，70 个 bash 脚本/sbatch 与 1 个 zsh 脚本语法通过，17 个 JSON 解析通过。正式覆盖率按 **所有 182 个生产 Python** 重新计算，包含默认报告漏掉的 15 个未导入模块，不包含测试代码，结果 **31,707 / 37,867 = 83.73%**。它是语句覆盖率，未宣称分支覆盖或 GPU 路径覆盖。

| 范围 | 生产语句覆盖率 |
|---|---:|
| 根模块 / adapters / tools | 81.35% / **64.37%** / 77.32% |
| v1_1 / system | 81.31% / 90.33% |
| v2，含 ops | 76.99% |
| comparison，含 vendor 包装 | 89.42% |

整体达到 80% 的语句覆盖要求，适配器、tools 和含 ops 的 v2 尚未达到模块级 80%。跨模块身份、错误状态和失败传播是本次最明显的测试缺口。

六项跳过：两项需要 TRL 训练环境；四项分别需要未挂载的冻结测试集、`FASTFILL_EVAL_DATA` 或外部原始数据。真实 GPU/vLLM 大模型训练推理、Blender 实际渲染、Slurm 调度、付费 API、外盘全数据和已发布结果逐房溯源未在此次执行。局部训练恢复问题使用了真实 CPU tiny Qwen；模型接口复现使用明确的假传输，避免把模型输出概率当成验证依据。

原始记录：[verification.json](verification.json)、[完整测试日志](pytest-supported.log)、[18 项重跑日志](pytest-rerun.log)、[全量生产覆盖率](production-statement-coverage.json)。

## 当前已有的正确处理

当前 Scan2CAD、SpatialGen 的倾斜尺寸投影，以及 IL3D 窗洞完整跨度已有修正；旧文档中这些缺陷不能继续当作当前问题。新版 clean 训练 guard、v1_1 全队列评估的内容身份校验、system 的尺寸冻结及付费调用预算记录均有明确防护。多数比较产物使用哈希与不可覆盖目录保存实验身份。

这些正向证据支持保留现有设计；修复应集中在上述可复现缺口，尤其补齐生产者到消费者的集成测试。现有证据不支持把“测试通过”直接解释为所有数据适配、真实模型质量或历史研究数字已获验证。

## 复现与证据目录

在仓库根目录，用已经安装项目依赖的 Python 运行，例如：

```sh
AUDIT=docs/evidence/fastfill-current-code-audit-20261022
PYTHONPATH=. /Users/slian/miniforge3/bin/python "$AUDIT/root_probes.py"
PYTHONPATH=. /Users/slian/miniforge3/bin/python "$AUDIT/repro_root_comparison.py"
PYTHONPATH=. /Users/slian/miniforge3/bin/python "$AUDIT/repro_comparison.py"
```

其余复现脚本及命令见各分组报告。所有此次执行的复现均使用临时数据、CPU 或本地空桩，未对真实服务/外部账户发起操作。`MANIFEST.json` 记录交付证据文件哈希；源码哈希见 `production-review-coverage.json`。
