# FastFill v2 当前源码只读审查（2026-10-22）

审查基准：`825fd13ea`，工作树 `/Users/slian/Desktop/3D/Worldedge/OptiScene`。本报告来自当前源码全文阅读与当前代码 CPU 复现，没有修改生产源码或测试、没有启动训练、GPU、远程作业或 API 请求。未使用 sibling `Worldedge-fastfill` 的知识图谱推断这里；导航参考本仓库 `graphify-out/graph.json`，结论以源码为准。

## 覆盖

本代理逐文件完整阅读 **59 个非 ops 文件，13,600 行**：39 个顶层生产 Python 模块（含 `__init__.py`），11,571 行；16 个 configs；2 个 requirements 文件；完整 v2 README；`.coveragerc`。逐文件状态见 `v2-files.json`。`fastfill/v2/ops/**` 的 24 个文件、3,758 行由主代理转交 `audit_system_v11` 审查，本报告不冒领其覆盖。

阅读边界涵盖：schema/IO/geometry/batching → condition-only model → legal-group matching/完整字段 masks/geometry losses/regularizers → trainer/resume/DDP diagnostics/text SFT → source migration/independent verification/cohorts/full/minimal derivatives → prediction/spread decoding/evaluation/baselines → resolver/reconciliation/validation/atomic in-memory Host → direct handoff/RoomGenBench/LLM baselines/autorun。测试用于核对合同；全套 pytest 由主代理统一运行，此报告不重复执行或声称全套已通过。

## 当前确认问题

### V2-01 · P2 · RoomGenBench scene_key 能逃逸输出目录并覆盖 reference/source 文件

位置：`fastfill/v2/roomgenbench.py:286`，实际写入 `:292–293`；保护检查只在 `:272–275` 校验顶层 `output_dir`。

`write_benchmark_requests` 将输入 JSON 的 `scene_key` 原样作为 `f'{scene_key}.json'`，最后 `(target / name).write_text(payload)`。既没有单一文件名约束，也没有对每个解析后的输出路径重新检查 reference/input 保护边界，而且 `write_text` 会覆盖已有文件。

最小当前可触发条件：输入场景合法且有一个 floor 对象；`scene_key='../reference/protected'`；新输出为 sibling `requests/`，`reference/protected.json` 已存在。这满足函数对新输出目录的检查，却会把请求 JSON 写进 reference 内并覆盖原文件，`requests/` 为空。绝对路径 key 同样可抛弃 target 前缀。

CPU 复现：

```bash
python /tmp/fastfill-audit-20261022/repro-v2-roomgen-path.py
```

期望：拒绝含路径部分的 key，且 reference/source/已有文件不变。实际：正常返回 1 个场景摘要；`outside_reference_file_overwritten=true`；原保护 JSON 消失，被 `furniture_list/room_size_m/room_type` 替代。复现全在自动清理的临时目录中完成。

建议：在创建输出目录前验证 key 为单一安全 basename；验证每个 resolved 目的路径严格位于 target 内，不能落到 reference/source；使用独占创建，避免别名/已有文件覆盖。此项是合成输入已触发的写边界缺陷，**没有证据表明当前标准五个 benchmark 场景或历史结果已发生该覆盖**。

### V2-02 · P2 · 旧 harness 的修复请求异常会丢掉已经有效的首答布局

位置：`fastfill/v2/llm_baseline.py:207–224`，关键 `:211` 的 `call(messages)` 在保留有效布局的内部 `try` 之外。

首答可被 `to_layout` 正常解析、所有 ID 均正确，但几何有问题从而触发 repair。若修复请求在 chat 自身重试后抛出网络/服务异常，它直接进入外层 except，开始从头重新请求；后续也失败时最终写 `layout:null`，尽管已收到有效布局。内部 `:214–215` 已明确对「修复回复格式错误」保留上一有效布局，但网络失败没有同等保障。当前新 `llm_structured.py:262–267` 对同一情况正确保留。

CPU 复现（注入 ask，无网络）：

```bash
python /tmp/fastfill-audit-20261022/repro-v2-harness-repair.py
```

期望：首答有几何缺陷可以如实计分，修复请求失败时保留其布局并记录修复失败。实际：请求长度 `[1,3,1]`，共 3 次 chat；`first_answer_valid=true`、`attempts=2`、`layout=null`、`failed=1`。这把可评分布局转为 no-layout，额外付一次从头请求成本，并使旧 harness 的有效布局率受 repair 服务可用性影响。

建议：将 repair `call` 纳入保留最后有效布局的失败处理，记录 repair 失败。**版本边界必须保留**：README 明确该旧模块为历史实验 byte-pin；应用独立新实现/版本，旧结果按原实现身份保留，不能静默改变旧实验或缓存标识。当前 `autorun.llm_baselines` 仍调用这一模块，故不只是不可达的遗留函数。**本次只证明当前路径可触发，没有核对历史真实 300 行 API 跑次是否发生该异常**。

## 交叉模块核对

应主代理请求核对 v1_1 handoff 的支持关系：`roomgenbench._handoff` 只验证由 handoff condition/layout 重新计算的 scene 一致，并在 `require_placement` 时拒绝缺失 place_id；没有独立原始 benchmark `place_id` 的比对。`ground_truth_layout` 核对象数/顺序 id/category，`reference_check` 则用传入 handoff 的 condition 验证两份布局。因此如果上游 v1_1 将预测 `on` 改写为 condition 的支持声明，下游不会恢复/检查原始支持合同。证据已直接发给 `audit_system_v11`，该问题归其上游报告，避免重复计数。

## 审查中保留的合同与排除项

- 验证器明确区分 pass/violation/unknown；未知必要支持、mesh/physics/Solver 及真实 Host 不是已经实现/验收的能力。`AtomicMemoryHost` 只是离线原子提交参考。这些是明确的集成限制，不作为新代码缺陷。
- spread 为碰撞感知后处理，使用部分 bounding rectangle/AABB 近似，有些布局仍违反几何。argmax 与 spread 在评测区分；现有代码没有宣称 spread 是完整 solver。
- 训练 loss 是每 global microbatch 均值再按 accumulation window 平均；日志是 count-weighted sums/counts。README 明示二者差异，不能把它误报成未披露的全窗口归一化错误。
- yaw 不参与 matching cost，精确位置/尺寸平局时 yaw loss 可能依赖行序；README round 5 明确列为保留设计，本报告未重复报为新发现。
- historical source axis policies、MansionWorld/OptiScene size masks、Scan2CAD position masks、2 cm floor snap、壁挂/内层支撑扩展等有显式 provenance/qualification，不能把所有 target 字段当作有效完整监督。
- autorun phase F 的 `accepted` 只限产物/schema 级检查（其摘要明确不含 validator/physics）；其缺失布局 gate 仅检查顶层 minimal spread，raw 只看进程退出，full projection 不参与该 gate。由于当前 README/summary 把 full 作为额外报告，本轮没有据此断言其违反已声明的主验收合同；使用者不应把 `accepted` 解释为所有投影/几何/物理成功。
- `--gpus` 小于 4 时 phase F 存在无条件 `g[3]` 索引；帮助文本已经声明 phase F 用前四张卡。此轮将它记为需要提前校验的运维限制，未列作核心缺陷。

## 验证范围和余留不确定性

两个 findings 均由当前 HEAD 下独立 CPU 脚本实际触发，输出汇总于 `v2-repro-results.json`。没有改动 source/test，没有连接 paid API、拉取模型或运行 GPU。没有重跑完整真实外盘数据迁移、核对全部已发布/历史 artifacts、测试分布式大模型训练或真实 WorldEdge Host；这些不能由静态阅读/小型复现证明。根代理的全套测试结果及 ops 代理报告应与本报告一起汇总。
