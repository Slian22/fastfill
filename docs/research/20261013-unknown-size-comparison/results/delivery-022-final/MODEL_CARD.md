# FastFill 交付系统 / FastFill delivered system — final

**正式版（final）：放置器为修订 021 §4 选定并确认的模型。推荐升级规则 --escalate-on strict（修订 026 附录 B，探索性采纳）；本卡所有实测数字都来自旧的 checker 触发规则（修订 022），strict 规则的证据是修订 026 的离线回放（results/frontier-026-v4）。**

**FINAL: the placer selected and confirmed by amendment 021 §4. Recommended escalation --escalate-on strict (amendment 026 addendum B, exploratory adoption); every measured number in this card used the earlier checker trigger (amendment 022); the strict rule's evidence is the amendment-026 offline replay (results/frontier-026-v4).**

## 中文

**做什么。** 输入一个房间（边界多边形、层高、门窗与固定物体框、房间类型）和需要的家具（类别与数量），输出每件家具的 bbox 尺寸、位置、朝向与支撑父物体（IR 布局），并导出 RoomGenBench 格式（downstream.to_roomgenbench）。

**组成（修订 022 §1）。**
1. 尺寸：训练集先验表 size-priors.json（sha256 073522c18e10ce2443189dfb8ebca90074c16ff1c2a4c292dfd6f788409d2422）。
2. 放置器：021 臂，服务引擎 vllm（评估所用测试行的引擎：vllm）；训练基座 /lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/models/Qwen3.8-27B（公开基座，不随包复制，sha256 a4fca2a27a330e7510b09d7535d92352a6b19917b4b44251cd9a3aa20036db36）+ adapter（placer/adapter，sha256 cc3e286fa715be1892d13c0e4f29721ac32324903d8bb08e81f1951fee5fd1e6）。
3. S1：确定性最小位移修复（max_trials 2000，每间 60 秒）。
4. 升级（只对 S1 后未完成或检查器失败的房间）：gpt-6.1-sol 固定尺寸（一次回答 + 至多一轮修复）→ qwen3.8-27b 固定尺寸 → 最佳候选。API 上限：≤ 600 间 × 2 次 = 1200 次调用；LLM 代理总上限 7700。永不返回空结果、永不丢房间。

**实测结果。** 每个数字来自已校验的 results.json；缺失写 not measured。比率的分母为全部请求。

代理队列（含 API 访问的系统：gpt-6.1-sol 升级，再 O2）：

| 阶段 | 完整率 | 检查器成功 | 严格准入 | required={} 准入 | OOR |
|---|---|---|---|---|---|
| 放置器原始 | 1.0000 | 0.5564 | 0.3494 | 0.7648 | 0.1650 |
| +S1 | 1.0000 | 0.8542 | 0.4532 | 0.8205 | 0.0833 |
| +升级 | 1.0000 | 0.9931 | 0.5199 | 0.8542 | 0.0562 |

全部测试房间（仅 O2 免费升级的系统）：

| 阶段 | 完整率 | 检查器成功 | 严格准入 | required={} 准入 | OOR |
|---|---|---|---|---|---|
| 放置器原始 | 0.9995 | 0.4434 | 0.2161 | 0.5331 | 0.2560 |
| +S1 | 0.9995 | 0.7422 | 0.2913 | 0.6372 | 0.1526 |
| +升级 | 1.0000 | 0.8901 | 0.3435 | 0.6668 | 0.1081 |

全部测试房间，修订 012 §4（R 原始 / +S1，无升级）：

| 阶段 | 完整率 | 检查器成功 | 严格准入 | required={} 准入 | OOR |
|---|---|---|---|---|---|
| — | not measured | not measured | not measured | not measured | not measured |

**升级开销。**
- 代理队列（含 API 访问的系统：gpt-6.1-sol 升级，再 O2）：升级比例 0.1458，API 房间 212，API 调用 225，费用（USD）not measured，最终来源 {"placer": 1247, "gpt": 202, "o2": 5}
- 全部测试房间（仅 O2 免费升级的系统）：升级比例 0.2578，API 房间 0，API 调用 0，费用（USD）not measured，最终来源 {"placer": 7348, "o2": 1297, "fallback": 2}

**公平性。** 含升级的系统使用了 API 或额外的开源大模型，只作为“含 API 访问的系统”报告，从不作为“模型”与单次生成的代理比较。网格级碰撞（修订 019）：未测量（系统输出未渲染）。

**用法。**

```
PYTHONPATH=code python -m fastfill.system.cli run --requests ROOMS.jsonl --prior size-priors.json --prior-sha256 073522c18e10ce2443189dfb8ebca90074c16ff1c2a4c292dfd6f788409d2422 --placer vllm --model /lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/models/Qwen3.8-27B --adapter placer/adapter --escalate l+o2 --escalate-on strict --api-cap 1200 --api-max-rooms 600 --env-file ~/.fastfill_api.env --o2-model /lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/models/Qwen3.8-27B --workers 32 --out NEW_DIR
```

输入格式见 fastfill/system/pipeline.py 文档（eval-data 联合请求或简单房间 JSON）。

**代码。** git commit not measured（dirty: yes）；51 个文件随包复制到 code/，逐个 sha256。

## English

**What it does.** Input: a room (boundary polygon, ceiling height, door / window / fixed boxes, room type) and the furniture it needs (category and count). Output: every object's bbox size, position, yaw and support parent (IR layout), exported to RoomGenBench (downstream.to_roomgenbench).

**Components (amendment 022 §1).**
1. Sizes: the train prior table size-priors.json (sha256 073522c18e10ce2443189dfb8ebca90074c16ff1c2a4c292dfd6f788409d2422).
2. Placer: 021 臂, served on vllm (evaluated test rows from vllm); training base /lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/models/Qwen3.8-27B (public base, not copied, sha256 a4fca2a27a330e7510b09d7535d92352a6b19917b4b44251cd9a3aa20036db36) + adapter (placer/adapter, sha256 cc3e286fa715be1892d13c0e4f29721ac32324903d8bb08e81f1951fee5fd1e6).
3. S1: deterministic minimum-displacement repair (max_trials 2000, 60 s per room).
4. Escalation (only rooms incomplete or failing the checker after S1): gpt-6.1-sol size-fixed (one answer + at most one repair round) -> qwen3.8-27b size-fixed -> best candidate. API cap: <= 600 rooms x 2 = 1200 calls; LLM-agent total cap 7700. Never empty, never drops a room.

**Measured.** Every number comes from a verified results.json; missing ones read not measured. Rates use every request as the denominator.

agent cohort (system with API access: gpt-6.1-sol escalation, then O2):

| stage | complete | checker success | strict admission | required={} admission | OOR |
|---|---|---|---|---|---|
| placer raw | 1.0000 | 0.5564 | 0.3494 | 0.7648 | 0.1650 |
| +S1 | 1.0000 | 0.8542 | 0.4532 | 0.8205 | 0.0833 |
| +escalation | 1.0000 | 0.9931 | 0.5199 | 0.8542 | 0.0562 |

all test rooms (system with free O2 escalation only):

| stage | complete | checker success | strict admission | required={} admission | OOR |
|---|---|---|---|---|---|
| placer raw | 0.9995 | 0.4434 | 0.2161 | 0.5331 | 0.2560 |
| +S1 | 0.9995 | 0.7422 | 0.2913 | 0.6372 | 0.1526 |
| +escalation | 1.0000 | 0.8901 | 0.3435 | 0.6668 | 0.1081 |

all test rooms, amendment 012 §4 (R raw / +S1, no escalation):

| stage | complete | checker success | strict admission | required={} admission | OOR |
|---|---|---|---|---|---|
| — | not measured | not measured | not measured | not measured | not measured |

**Escalation cost.**
- agent cohort (system with API access: gpt-6.1-sol escalation, then O2): escalated share 0.1458, API rooms 212, API calls 225, cost (USD) not measured, final stage {"placer": 1247, "gpt": 202, "o2": 5}
- all test rooms (system with free O2 escalation only): escalated share 0.2578, API rooms 0, API calls 0, cost (USD) not measured, final stage {"placer": 7348, "o2": 1297, "fallback": 2}

**Fairness.** The escalated system uses an API or an extra open model; it is reported as a system with API access and never compared as "the model" against single-shot agents. Mesh collisions (amendment 019): not measured (the system output is not rendered).

**Usage.**

```
PYTHONPATH=code python -m fastfill.system.cli run --requests ROOMS.jsonl --prior size-priors.json --prior-sha256 073522c18e10ce2443189dfb8ebca90074c16ff1c2a4c292dfd6f788409d2422 --placer vllm --model /lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/models/Qwen3.8-27B --adapter placer/adapter --escalate l+o2 --escalate-on strict --api-cap 1200 --api-max-rooms 600 --env-file ~/.fastfill_api.env --o2-model /lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/models/Qwen3.8-27B --workers 32 --out NEW_DIR
```

**Code.** git commit not measured (dirty: yes); 51 files copied to code/, each with sha256. Provenance: provenance/ (eval-data MANIFEST, the inference run receipt, final-selection.json).
