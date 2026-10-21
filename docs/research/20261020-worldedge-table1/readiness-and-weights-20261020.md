# 权重交付与主表实验就绪核查

核查日期：2026-10-20；只读检查，没有启动 GPU/API 生成、训练或物理任务，没有改动分支。**四个大文件的本地交付已完成；真实扩房＋Agent 任务的本版本验收尚未建立。**

## 1. 权重与版本

EmbodiedGen 本地 HEAD、origin/main 及 `git ls-remote` 的远端 main 均为 `b4709dcc2a2e1e9af1ba53f0847c530428c98fe7`，该仓库工作树干净。这不代表相邻 OptiScene 的研究文档已提交；本次协议与表格更新保存在 OptiScene。

下列文件已完整重算 SHA256，与 [COPY_MANIFEST](../../../../EmbodiedGen/thirdparty/worldedge_delivery/COPY_MANIFEST.json)及各自包清单匹配；表中仅缩写 SHA，原清单保留完整值。

| 文件 | 字节数 | SHA256 前 16 位 | 交付状态 |
|---|---:|---|---|
| FastFill LoRA | 870,439,776 | `6e4a0460cbbfc167` | 本地与 HF LFS SHA/大小一致 |
| size-priors.json | 241,067,473 | `073522c18e10ce24` | 本地与同一 HF 库一致 |
| Particulate model.pt | 605,384,630 | `2747c17bd0c8dedc` | 本地与 HF LFS SHA/大小一致 |
| PartField model_objaverse.ckpt | 1,243,631,761 | `463efc8a3afd3913` | 本地、交付清单与原 HF 仓库 LFS SHA/大小一致 |

HF 官方 API 查询确认两个库仍为私有，当前 main 分别等于已指定的固定修订：[FastFill `0e9ebe84755a034efdcadb20e3c8502dde8ab865`](https://huggingface.co/liantian/fastfill-27b-p-lora/tree/0e9ebe84755a034efdcadb20e3c8502dde8ab865)、[Particulate `71a86f7c4a94b1f0f6ab039fd8a5ea32ea33adf2`](https://huggingface.co/liantian/particulate-r7-wise/tree/71a86f7c4a94b1f0f6ab039fd8a5ea32ea33adf2)。未下载远端大权重；认证信息未输出或写入报告。

第三方 PartField 也已通过官方 API 复核：[固定修订 `90b9b1e08b6a12fdcb6ee26b4854a26235e1765f`](https://huggingface.co/mikaelaangel/partfield-ckpt/blob/90b9b1e08b6a12fdcb6ee26b4854a26235e1765f/model_objaverse.ckpt)，完整 SHA256 为 `463efc8a3afd3913142aa025e0125c00f16ef452b8de6a132ebe32bbe7877ee4`，与本地一致。

**依赖范围：**约 55.6 GB 的 Qwen3.8-27B 底座不在本地交付目录，当前该目录的底座权重分片数为 0；RoomGenBench 图像/3D 模型、Linux/CUDA 环境仍属于服务器准备事项。历史服务器曾成功加载底座，不等于此次已重新验证服务器缓存。见[交付说明](../../../../EmbodiedGen/gentask_docs/pipeline_delivery_2026-10-16.md)。

## 2. 已有运行证据的层级

| 证据 | 可以支持 | 不能支持 |
|---|---|---|
| [真实 FastFill 摘要](../../../../EmbodiedGen/outputs/portal-server-20261016/model-evidence-summary.json)，Slurm 7461378 | 2 个房间真实布局推理、接受 2/2，约 316.5 秒 | 当前 HEAD 的完整 mesh/关节/Agent 扩房成功；该次 `prior.status=not_provided`，也未证明使用当前必需尺寸先验 |
| [九房间几何检查](../../../../EmbodiedGen/outputs/portal-server-20261016/server-evidence/real-layout-check/report.json) | 门洞、结构几何和 bbox 连通 | 产物明确 task/asset 未执行、physical_navigation unknown；不能写成导航或操作 SR |
| [Particulate 历史 provenance](../../../../EmbodiedGen/thirdparty/worldedge_delivery/particulate/examples/verified-toilet/PROVENANCE.json) | GPU 作业 7454682、固定 checkpoint 的独立样例 | 同一扩房任务里的关节语义和控制可行性 |
| [最新 session 回归](../../../../EmbodiedGen/tests/test_unit/test_world_pipeline_session_integration.py) | 注册、任务编译、调度、恢复和提交合同 | 生成用 fixture，部分物理 pass 为手填；空动作轨迹不是实际机器人完成任务 |
| [CI 配置](../../../../EmbodiedGen/.github/workflows/ci.yml) | 离线、无模型权重的合同测试流程 | 用户给出的 1536 passed 未在本轮独立核原日志；即使成立，也不证明真实服务器扩房完成 |

## 3. 能读到字段，不等于得到主表成绩

`manifest.status`、`failed_stage`、`stages.*.seconds`、`row.outcome`、`goal_validation`、`state_preservation` 和 `harness_unknown` 等字段已有接口，见 [ops](../../../../EmbodiedGen/embodied_gen/world/ops.py)、[session](../../../../EmbodiedGen/embodied_gen/world/pipeline/session.py)。正式评测还需要按预定请求去重和汇总、记录未终结/缺证据案例，并核清实际测量层级。

canonical 的 [state_io](../../../../EmbodiedGen/embodied_gen/world/state_io.py) 与[提交回执](../../../../EmbodiedGen/embodied_gen/world/pipeline/canonical.py)提供状态重载、版本和摘要，但不能直接证明机器人 qpos/qvel、物体动态状态和控制器内部状态的迁移已完成。保护协议只比较声明的物理/任务状态，不要求模拟器内部索引或接触缓存逐字节一致。

符号 `LLMSolver`、盒体代理的物理检查、独立 ManiSkill PickCube 执行器是不同层级；本轮未定位到将真实生成房间接入下游并完成本协议任务的生产适配器。已有 `DownstreamAgent` / `ExecutionValidator` 接口不等于已有同任务实测结果。见 [agent.py](../../../../EmbodiedGen/embodied_gen/world/pipeline/agent.py)、[executor.py](../../../../EmbodiedGen/worldedge/executor.py)。

## 4. 下一轮应怎样验收

1. 在服务器固定当前代码与模型修订，核查底座、尺寸先验和 RoomGenBench 模型；保留真实加载日志和权重绑定。
2. 用现有 `pipeline_world init --world ...` 和 `advance --execute` 在全新 run 目录真实生成一个新增房间，核对所有交付、加载和失败证据；这一步可能只到 `awaiting_agent`。
3. 接入真实场景加载、物理状态迁移和 Agent 执行适配器，完成[增长协议](comparable-growth-protocol-20261020.md)的 g=1 状态依赖任务及合格恢复基线。
4. 先验收评测器的正/负控制，再开展冻结配置的配对 pilot 和正式实验。

这些是待执行的实验验收，不是本次已完成动作。当前大表保留 NE。
