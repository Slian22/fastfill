# FastFill comparison 当前代码审查

审查基准：`825fd13ea`，2026-10-22。只读审查当前文件，未修改源码或测试，未调用远程作业、模型 API 或 GPU。

## 覆盖

`fastfill/comparison` 共有 70 个生产 Python 文件（30,668 行）、50 个 shell/sbatch（5,223 行）、5 个 vendored 配置/说明文件（407 行），以及 59 个测试 Python 文件（23,643 行）。

- audit_comparison 完整阅读 59 个生产 Python 文件、25,760 行，以及 5 个配置/说明文件。
- root 完整阅读其余 11 个生产 Python 文件、4,908 行：visual_024、visual_cases、visual_pipeline、judge_batch、judge_agreement_023、judge_bundle_025、server_api_judge、critic_025、critic_batch_025、critic_client_025、render_assets。
- audit_v2、audit_system_v11 各完整阅读排序前/后 25 个脚本；独立结论见 `comparison-shell-a-review.md`、`comparison-shell-b-review.md`。覆盖从两份逐文件记录合并。
- 逐文件覆盖见 `comparison-files.json`：125 个生产/配置文件完整阅读；59 个测试文件静态清点，相关测试辅助函数用于本报告的 CPU 复现，全套测试由 root 执行。
- Python AST 解析全部通过；50 个 shell/sbatch 的 `bash -n` 全部通过。这里只证明语法可解析，不能代替语义审查。

图索引未包含当前 OptiScene（同名 sibling FastFill 项目不适用），经 root 确认后以当前源码与 `rg` 为准。审查跨越数据/房屋分组、尺寸先验、共同准入、SFT/DPO/蒸馏、精确策略与 BF16 merge 的身份、分片推理、LLM 异常与恢复、geometry/repair/downstream、选择/统计报告、官方方法 vendored 包装、delivery。剩余 GPU 库兼容、SLURM 行为、真实推理吞吐、真实远程状态未执行验证。

## 确认问题

### C01 · P2 · O2nt 的禁止思考失败标记在评估时丢失

主要位置：`fastfill/comparison/evaluate_llm.py:109–112`。相关链：`llm_open.py:249–263`、`llm_agent.py:493–504`、`evaluate_pair.py:50–56`、`a021_select.py:553–566`。

触发：O2nt（关闭思考参考臂）实际返回思考文本/思考 token，后接可以正常解析的布局。`llm_open._chunk` 正确把 schema 标为 `valid=False`、`reasoning_in_no_reasoning_system`，且不再修复。`_finalize` 保留该 schema，但 records 的 `error` 仍为 `None`。随后 `_score_llm` 只交给 `_score` 原始布局与 `error`，重新解析已经剥离思考文本的布局，从而把该失败房间重新计入 complete、geometry success。

`a021_select.o2nt_score` 也走同一重解析路径；虽然 `_violations` 统计违规 UID，但不把它们从成功分子移除。与 amendment-021 Addendum B 第 138 行“any reasoning … failure of that room”直接冲突。

CPU 复现：现有 `test_llm_open.world` 合成 36 房，FakeLLM 对 O2nt placement 每房返回 `<think>reasoned</think>{valid}`。运行统计 `valid_final=0/36`，`evaluate_llm` 却产出 `complete=36/36`、`geometry_success_rate=1.0`。同一个复现加入无思考控制组，`valid_final=36/36`，评估亦为 36/36。

影响边界：仅在实际发生 reasoning violation 的输出上触发；正常不思考答案不受影响。直接影响 O2nt raw/harness 的参考评估及从这些行派生的分析。O2nt 不属于 `a021_select.ORDER` 的训练模型候选，不能据此推断候选模型选择或当前交付被污染。

历史核验：本地 JSON/JSONL 未发现 `no_reasoning_violation`、`reasoning_in_no_reasoning_system`、`no_reasoning_violations` 的持久化记录；只有 O2nt 配对汇总、旧审计索引等。旧协议记载一次被中止且未使用的违规运行，以及早期 dev-select 零违规运行；这不证明后来汇总是否受影响。本机 `/Volumes` 仅有 Macintosh HD，存储逐房收据的 harddisk 未挂载。因此报告当前可触发缺陷，不宣称已发表 O2nt 数字被污染。

建议：将 runner 的语义失败作为独立且持久的结果状态传给评分器，尤其不可仅重新解析 `raw`；让 O2nt reference scorer 对违规 UID 应用失败语义，并保留原因计数。补一个“runner 失败、几何却有效”的端到端回归测试。

### C02 · P2 · S2 联合臂的绝对尺寸偏差门槛没有执行

主要位置：`fastfill/comparison/decide_012.py:106–115`。相关位置：`best_of_n.py:101–105`。合同：`docs/research/20261013-unknown-size-comparison/amendment-012-output-side-dev-20261014.md:36`，以及 decide_012 自身 docstring 第 11–12 行。

触发：J 的 B8+S1 接受率增益满足条件，但 B8 与 greedy 的足迹 log-bias 差绝对值超过 0.03。decision 只检查增益及 CI，然后把尺寸条件写为“by construction”；实际上 best-of-N 的逐房 guard 仅禁止 `bias_prior` 比 greedy 低 0.03 以上，没有上界。该 guard 与协议的 `|B(B8)-B(G)| <= .03` 不是同一约束，尤其无法防止变大。

CPU 复现：20 个独立房屋组，greedy 尺寸 1×1，B8 尺寸 1.5×1.5（仍在每轴 0.67–1.5 prior band 内）。更少违规的候选被 `best_of_n.select` 选中，真实 bias 差为 `2*log(1.5)=0.8109302162`。B8+S1 下游接受率 1，G+S1 为 0.5，增益 CI `[0.3,0.7]`；`decide` 仍返回 `s2_joint.passed=True`。

影响边界：可以错误采纳不满足预注册尺寸条件的 J S2。当前仓库冻结 `results/decision-012/decision.json` 的 `s2_joint.passed=False`（增益约 .00294，未达 .03），因此本缺陷没有把这份现有 decision 从拒绝变为采纳。P 的尺寸固定为 prior，不适用 J 的 bias 门槛。

建议：从输入 S1 行和同一 GT 计算协议定义的 B8/G footprint bias，在 decision 中显式应用绝对差条件并记录实测值；不要用候选生成阶段的单侧 guard 代替 adoption 条件。

### C03 · P2 · placement 尺寸覆盖对非有限数值抛出未捕获异常，终止整批推理

主要位置：`fastfill/comparison/llm_agent.py:185`；相关位置 `137–139`、`146–152`、`188–190`、`488–490`，以及 `evaluate_llm.py:100`。

触发：LLM 返回可由 Python JSON 解码为无穷大的有限形式数值字面量（如 `1e309`），并包含任意 `size` 字段或会被重新计算的支撑 z。`strict_json` 拒绝 NaN/Infinity 常量，但 Python 可把 `1e309` 解析为 `inf`；`_vec` 只检查数值类型。enforce_sizes 删除 size 后，以 `allow_nan=False` 重新序列化；这一操作在 try/catch 之外，产生 ValueError。

最小答案：`{"placements":[{"id":"obj_0001","pos":[1e309,1,0],"yaw":0,"size":[1,1,1]}]}`（ID 换为合法请求 ID）。Joint 普通解析返回 `valid=False`；placement 的 `llm_agent.parse` 抛 `ValueError: Out of range float values are not JSON compliant: inf`。FakeLLM 跑 O2nt placement 批次时，同样异常传出 main，尚未发布任何 room receipt。API worker 路径对这类异常会标记 INTERNAL_ERROR 并重抛，而不是把这一房间视为普通坏答案。

影响边界：所有调用 `enforce_sizes` 的 placement 基线/重新评估入口；有限数值正常答案不触发。单个畸形模型答案应作为单房失败/修复候选处理，现在可能中断整批或无法完成评估。未发现历史 occurrence，未对真实模型生成概率作假设。

建议：在输入数值边界统一拒绝非有限值，并保证 enforce_sizes 无法从普通无效输出抛到批处理层；让后续 parser 返回已有的 invalid schema 结果。

### C04 · P3 · mesh_collision 的 limit 超过可渲染房间数时重复采样并失败

位置：`fastfill/comparison/mesh_collision.py:477–479`。

触发：传入正的 `--limit` 大于实际 rendered 房间数。循环次数虽然取了 min，索引分母却仍用原始 limit，因此多次选择开头的同一个房间。结果进入配对统计时被重复 UID 校验拒绝。

CPU 复现：直接使用 `test_mesh_collision.test_cli_writes_once_and_reports_missing` 的 2 个 rendered 房间 fixture；默认 run 成功，随后 `run(..., workers=1, limit=100, repeats=2)` 抛 `ValueError: paired UID coverage mismatch or duplicates`，输出目录未发布。

建议：先把有效 limit clamp 到人口数，再按有效 limit 下采样，或在 limit>=人口数时保留完整顺序。

### C05 · P3 · 全部 mesh 房间失败时，缺失报告本身崩溃

位置：`fastfill/comparison/mesh_collision.py:395`；相关 `356`、`498–501`。

触发：所有网格房间都因缺失资产/无效网格失败，或没有可分析的 render chunk。summarize 正确生成 missing_rooms 并把 bbox_max_err_m 留为 None；report 却无条件用 `.2e` 格式化 None，导致 summary/report/MANIFEST 整体不能发布。这违背 analyse_case 的逐房隔离与缺失报告约定。

CPU 复现：一个 `error='FileNotFoundError: missing mesh'`、rows 为空的结果，summarize 返回 `analysed_rooms=0` 与明确 missing reason；report 抛 `TypeError: unsupported format string passed to NoneType.__format__`。

建议：bbox 未测量时输出“未测量/—”；仍发布零成功、全部缺失的报告和诊断。

## 证据与复现入口

```sh
PYTHONPATH=. /Users/slian/miniforge3/bin/python /tmp/fastfill-audit-20261022/repro_comparison.py
PYTHONPATH=. /Users/slian/miniforge3/bin/python /tmp/fastfill-audit-20261022/repro_mesh_edge_cases.py
```

结果：`comparison-probes.json`、`comparison-mesh-probes.json`。全部仅 CPU 合成数据与已有测试 fixture，运行成功并得到上述结果。全套 pytest 的最终结论请以 root 的统一报告为准。

独立评审发现由 root 汇总，未重复计为本报告 findings：critic_batch answers 缺少 critic round 所需 identity 字段；judge_agreement 对合法缺少 optional overall 分数/无共同完成样本的处理；submit_023 full 的 smoke 目录存在门槛。对应证据分别位于 root comparison probes 与 shell-b 报告。

## 审查中的正向证据与限制

多数关键产物使用写一次清单与 SHA256 绑定；开发选择与测试应用有明确切片；exact-policy 与 BF16 convenience merge 被分开记录；cap failure 在模型选择的主要入口显式应用；报告对历史测试集、单训练种子、prior confounding、debiased 仅诊断等限制有较完整披露。

本报告没有把第三方 vendored 上游保留行为误认成项目新实现，也没有把 GPU/库版本未实跑等同于已确认 bug。对模型权重哈希策略、指标范围文字等未满足当前确定触发/影响证据的疑点，没有列为确认问题。未扫描未挂载 harddisk，故对历史实际污染保持未知。
