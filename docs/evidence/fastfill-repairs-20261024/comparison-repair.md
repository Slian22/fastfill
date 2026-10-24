# FastFill comparison 修复复核（2026-10-24）

基线：`873251f0b1f299230c42d510fe5c4f732aa77ce9`。本工作包负责 F03、F04、F09、F15、F16；生产改动仅在 `evaluate_llm.py`、`decide_012.py`、`llm_agent.py`、`mesh_collision.py`。新增或扩展四个测试文件，未修改冻结代码副本、训练权重、历史结果、原协议、`a021_select.py` 或 `best_of_n.py`。本子任务未 SSH、未调用模型/API、未启动 GPU/训练，未 commit/push。

## 复现与最小修复

| 项目 | 已复现的问题 | 修复及边界 |
|---|---|---|
| F03 | O2nt 回答含 reasoning，runner 的 `schema.error=reasoning_in_no_reasoning_system`、`valid_final=0`；原 evaluator 忽略这个语义失败，36/36 又成为完整成功输出。 | `_score_llm` 仅将此明确语义错误传给共享 geometry scorer，保留原有 `record.error` 优先级。普通 parser-invalid 仍重新解析：历史回答中被 B2 允许覆盖的尺寸不会因旧 parser 判定而错误失败。无 thinking 的 36/36 对照不变。`a021_select.o2nt_score` 已调用该函数，无需另改选择逻辑。 |
| F04 | 当 acceptance gain/CI 满足采用条件、但 J 的 footprint bias 向任一方向变化超过 .03，原 `decide_012` 仍采用 S2。候选筛选的单向 prior-bias guard 并不等于协议的双向 GT-bias 采用门槛。 | 对各 variant 自身 complete 房间计算 `report_v2.size_bias` 的房间均值，再检查 `abs(B_bo8-B_greedy) <= .03`（浮点比较容差 1e-12）。缺少任一均值时不采用。报告保存两侧均值、complete 分母、绝对差、容差、判定；记录新引用的 metric 源文件 hash。P 的采用逻辑和候选 request-only guard 不变。 |
| F09 | 数字字面量 `1e309`/`-1e309` 可被 JSON 解成 infinity；placement 返回 size 时，`enforce_sizes` 的严格序列化抛 `ValueError`，整批停止。 | 该序列化失败返回原回答及 `counts=None`，交给原合同解析器判失败。未替换非法坐标为合法值，未改变正常尺寸覆盖或支撑 z 修复。集成测试确认一个失败房间不会中断其他 35 个房间，所有 receipts 都写出。 |
| F15 | 已渲染 2 房间而 `limit=100` 时，原等距抽样索引重复选 r0，统计模块因重复 UID 拒绝运行。 | 抽样公式的分母使用 `min(limit, available)`；请求上限仍保留在报告/manifest。limit 小于或等于总体时抽样不变；超过总体时每个房间恰好一次。 |
| F16 | 全部 mesh 失败或无 render chunk 时，`bbox_max_err_m=None` 被以 `.2e` 格式化，报告发布失败。 | 仅缺失值显示 `—`，保留失败原因、零成功计数、空 rows/pairs 和带 hash 的完整输出。正常数值的格式不变，不将缺失解释成成功。 |

## TDD 与验证

解释器：`/Users/slian/miniforge3/bin/python`。所有 pytest 使用 `--import-mode=importlib`；所有推理与 API 均为已有 CPU fake，mesh 为本地合成 OBJ。

首次在改生产代码之前执行下列新增用例，结果 **14 failed, 5 passed**；失败覆盖全部五项缺陷，正常无思考、旧 parser 重评分、既有 runner error 和正常抽样对照已通过：[comparison-red.txt](comparison-red.txt)。

```sh
/Users/slian/miniforge3/bin/python -m pytest --import-mode=importlib -q \
  fastfill/comparison/tests/test_decide_012.py \
  fastfill/comparison/tests/test_llm_open.py::test_o2nt_semantic_failure_survives_evaluation \
  fastfill/comparison/tests/test_agents_016.py::test_rescoring_keeps_parser_failures_repairable_and_runner_failures_failed \
  fastfill/comparison/tests/test_agents_016.py::test_size_enforcement_leaves_nonfinite_geometry_as_a_parse_failure \
  fastfill/comparison/tests/test_mesh_collision.py::test_limit_samples_each_available_room_at_most_once \
  fastfill/comparison/tests/test_mesh_collision.py::test_no_successful_mesh_analysis_still_publishes_missing_reasons
```

补充运行“全批不中断”和“缺少 complete bias 不采用”两个测试：[comparison-red-extra.txt](comparison-red-extra.txt)。后者最初存在测试夹具 manifest 自哈希错误，不能算作产品缺陷复现。修正夹具后，仅在独立 Python 进程内执行 `git show HEAD:fastfill/comparison/decide_012.py` 的原函数，确认原实现仍错误采用，得到真正 RED：[comparison-red-missing-bias.txt](comparison-red-missing-bias.txt)。未将旧实现写回工作树。首次修复后为 20 passed / 1 fixture failure，原始日志保留在 [comparison-green-initial.txt](comparison-green-initial.txt)。

最终完整验证覆盖四个改动模块及 a021 selector、API agent、test application、best-of-N 的既有测试，结果 **127 passed in 39.73s**：[comparison-green-modules.txt](comparison-green-modules.txt)。

```sh
/Users/slian/miniforge3/bin/python -m pytest --import-mode=importlib -q \
  fastfill/comparison/tests/test_decide_012.py \
  fastfill/comparison/tests/test_llm_open.py \
  fastfill/comparison/tests/test_agents_016.py \
  fastfill/comparison/tests/test_mesh_collision.py \
  fastfill/comparison/tests/test_a021_select.py \
  fastfill/comparison/tests/test_llm_agent.py \
  fastfill/comparison/tests/test_test_apply_012.py \
  fastfill/comparison/tests/test_best_of_n.py
```

新增回归共 21 个参数化案例：O2nt 语义失败/正常 2；非有限值整批 1；旧 parser 与 runner error 2；±infinity 的位置/yaw 4；bias 正负边界/超界/自身 complete 分母/空分母 7；limit 1/2/100 共 3；mesh 全失败/无渲染共 2。`git diff --check` 对本工作包修改通过。未在本子任务声称全仓库覆盖率；主代理另做全套回归和覆盖率。

## 历史服务器结果：只读复核方法与影响边界

先验证对应 MANIFEST、run identity、数据/cohort identity；不要原位重写历史 records、scores 或 manifest。需要重新评分时另建独立输出目录，记录旧 run hash 与新评估代码 hash。

1. **F03：必须逐 variant、逐 UID 对接，不能仅凭出现 reasoning 推断成功率变化。** `receipts/*.json` 中，`llm_raw` 对应首个 answer，`llm_harness` 对应最后一个 answer；读取 `no_reasoning_violation` 和 `schema.error`，再核对 variant `records.jsonl` 的 `schema.error`。和旧 `*-geometry.jsonl` 的 `complete`、`error`、`geometry_success`、`geometry_success_repaired` 连接。语义 marker + 旧 complete=true 才说明完整率应下降；success 指标仅在对应旧 success=true 时下降。已有 `record.error` 或原本 incomplete 的房间不应重复扣减。无 marker 的普通 parser-invalid 回答保留 B2 重评分。O2nt 是 amendment 021 参考线，不在 selector 的 `ORDER=('r','op13','d18','arm')` 候选集；不能直接归因已选 27B-P/J 改变。当前服务器出现数及四个具体房间由主代理独立提取，本文不冒充已完成该逐条 join。
2. **F04：冻结 decision 与原输入一起核对。** 从已验证的 `decision-012/decision.json` 读取采用布尔值；用其绑定 data `rooms.jsonl` 的 GT 与 J greedy/bo8 `*-geometry-s1.jsonl` 的 `complete`、`raw_layout` 重算两侧均值。按各自 complete 集合求均值，不使用仅共同 complete 的子集。S1 不改尺寸，因此 raw_layout 足够。已只读验证本地冻结 `results/decision-012/MANIFEST.json`：J gain=.0029386555650789763、P gain=.020325700991796255，两者均原已不采用，新增约束不会将 false 改成 true；这不替代服务器原始 S1 文件的独立核验。
3. **F09：先查是否发生整批中止。** 观察 Slurm/进程日志中 `Out of range float values are not JSON compliant` 及 `enforce_sizes` 栈、run 完成状态、预期 UID/receipts 的差集。若保留了原始模型文本，检查超出 float 范围的 exponent，特别是 placement 含返回 size 或需要 z 修复的回答。仅凭某个 run 缺少 manifest 不足以归因该 bug；完整已完成 run 也不能反推所有中间尝试都正常。此修复恢复失败计数和继续执行，不应把非法 x/y/yaw 改为有效输出。
4. **F15：读 mesh 命令/manifest 的 limit 与 render chunk 唯一 UID 数。** limit > available 时核查重复 UID 异常与缺失的最终输出；limit <= available 或未设 limit 的已完成结果抽样公式不变。它不改变几何计算或训练。
5. **F16：读每房失败诊断及 rendered 清单。** 原来通常在创建最终 output 前因 None 格式化失败；所以缺少 report 并非成功分析 0 个的证据，需要日志/输入才能确认。新报告明确缺失及原因，不补造碰撞或成功率。

原审计复现材料仍在 `docs/evidence/fastfill-current-code-audit-20261022/comparison-review.md`、`comparison-probes.json`、`comparison-mesh-probes.json`。本次修复只改变后续执行语义；旧报告的版本和原始数值保留。
