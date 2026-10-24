# FastFill 实际训练运行核验（2026-10-24）

**结论：服务器上的交付模型 27B-P 和后续 27B-J 都完成了计划中的 40,000 房、10,000 步训练。没有发现本轮代码审阅 F05 所描述的串用旧实验 checkpoint 或零步完成现象；现有证据不支持因 F05 要求这两个模型重训。**

本次直接通过 `u6uv.aip2.isambard` 只读查询。观测时间为 2026-10-24 12:44–12:47 UTC，读取真实 Slurm 记录、完整训练标准输出、完成回执、数据、模型文件和冻结源代码。没有提交作业、重新训练、修改远程文件或更换交付模型。

## 1. 实际运行与完成状态

| 模型 | 训练作业 | 冻结代码 | 实际样本 | 完成步数 | Slurm 状态 / 耗时 |
|---|---|---|---:|---:|---|
| 27B-P，当前交付的摆放模型 | `7419267` | `main-5bc501b3d` | 40,000 | 10,000 / 10,000 | COMPLETED，0:0，5:05:58 |
| 27B-J，后续尺寸与摆放联合模型 | `7461666` | `main-7373495ca9` | 40,000 | 10,000 / 10,000 | COMPLETED，0:0，5:00:16 |

P 于 10 月 15 日完成，J 于 10 月 16 日完成。J 的数据准备、训练冒烟、vLLM 加载、dev/test 推理及比较作业均为 COMPLETED；备用续跑 `7461667` 在首段成功后取消，未启动。观测时该用户的队列为空。原始调度证据见 [server-inventory.json](server-inventory.json)。

两次训练实际都是 `sft_only=true` 的 SFT：4 GPU、有效 batch 4、1 epoch、学习率 5e-5、LoRA rank 32 / alpha 64、seed 42。共享的底层文件叫 `v1_1/train_dpo.py`，完成回执也沿用 DPO 名称；这些名称不代表本次启用了 DPO 损失。真实参数与训练回执明确记录 answer-token NLL、无 DPO 项。

## 2. 核验依据

1. **确实执行了训练。** 两份完整日志均包含 1,001 条预期记录：step 1，以及 step 10、20、…、10,000；记录的 loss 和 chosen NLL 全为有限值。两份日志都没有恢复 checkpoint 的记录，末尾都有保存 final adapter 和 `A021_POLICY_COMPLETE`。这是日志记录频率，不声称逐步梯度都经过本次检查。
2. **计划步数与实际完成回执一致。** 两次 completion 均记录 40,000 admitted pairs、world size 4、expected steps 10,000、最终 position.step 10,000。final 权重均与 `ckpt_latest` 相同，与 `ckpt_latest.previous` 不同，支持实际发生过参数更新。
3. **数据和运行身份一致。** 重新读取两份实际训练 JSONL，分别都是 40,000 个唯一 UID；完整 SHA256、字节数与训练 guard 一致。run receipt、guard、rows manifest、训练参数及 policy 的绑定均匹配。P/J 的实际入训房间并非全部相同，不能把“同样 40,000 房”解释成同一个训练集。
4. **模型和代码一致。** 重新计算两组 final adapter 目录的全部已登记文件哈希；两组各 9 个 policy 登记源码文件与服务器冻结快照、对应历史 Git 提交一致，另核对了 guard 自身源码。基础模型共 33 文件、55,586,117,730 字节（含 18 个权重分片）也重新计算完整 SHA256，全部匹配两次训练共同记录的基座身份。
5. **交付身份一致。** 服务器 27B-P 的 policy、adapter 文件清单和基础模型身份与此前冻结的 `delivery-022-final/delivery.json` 一致。没有发现交付记录指向另一份模型的情况。

关键指纹：

| 文件 | SHA256 |
|---|---|
| P 实际训练数据 `placement.jsonl` | `654061b56bcf72630d1fa1d4871c8b1a2e00e6d4c60f5902d0b98f7165284cc3` |
| J 实际训练数据 `joint.jsonl` | `0270a054e82f52931ddd7770a62a5328e43d475bb23cef517a180b0f09ed1953` |
| P final adapter 权重 | `6e4a0460cbbfc1672d15c091353f618ddc7dbe664f617395b5386fa5f46c015f` |
| J final adapter 权重 | `0414a53c0bc99dfedeb048f05b23b48d54918c6f1dec18f1fbc1cb4b542cbb54` |

自动核验结果分别保存在 [运行身份检查](server-verification.json)（146 项）、[completion 与 checkpoint 检查](completion-verification.json)（24 项）、[历史代码及交付绑定检查](local-code-verification.json)（18 项代码、5 项绑定）、[基础模型完整哈希](base-verification.json)（33 文件）。这些存在交叉核对，不是新增单元测试数量。全部检查通过；独立审阅者还复核了完成回执语义和历史 guard，未发现实质性误判。

## 3. 与代码审阅 F05 的关系

[原审阅报告](../fastfill-current-code-audit-20261022/REPORT.md) 的 F05 指向未经过身份校验的旧 DPO 入口：更换同条数数据后复用旧输出目录，可能沿用旧 checkpoint 并零步完成。

本次实际运行虽然复用了底层 `train_dpo.py`，但由 `comparison/a021_train.py` 包装入口先校验不可变 run receipt，再执行 `training_guard`。核对的是训练时的历史代码版本，保护并非只存在于今天的源码。guard 绑定数据内容、模型、配方与关键代码，并拒绝没有 guard 的既有 checkpoint/产物目录。

结合真实的 step 1→10,000 日志、completion、数据及权重校验，本次没有发现上述失败情形。F05 仍是旧入口的有效代码问题，不能直接套用于这两次已核验运行。

## 4. 后续结果与结论边界

J 的服务器比较文件已完成并同步到 [a028 compare/report.md](server/runs/unknownsize-20261013/a028/compare/report.md)。本轮没有重新生成推理结果或重算这份报告的统计指标，也没有改变冻结交付选择。

这次证据确认训练实际执行、完成状态与产物身份一致。它不单独证明所有标签语义、优化算法、泛化效果或评测实现都正确；原代码审阅中的评测、验收问题仍需按各自影响范围处理。

另保留一条独立历史对照的事实：旧 `optiscene-014-dpo2-s1` 作业 `7406358` 的状态是 TIMEOUT。它不是上表两次 27B 训练，本次未追查该旧对照是否另有后续重跑，不能将其超时归为 27B-P/J 训练失败。

## 5. 原始证据与复核脚本

- [P 完整训练日志](server/runs/isambard/unknownsize-021-train-7419267.out)、[P completion](server/runs/unknownsize-20261013/a021/27b/train/final/completion.json)、[P guard](server/runs/unknownsize-20261013/a021/27b/train/training_guard.json)。
- [J 完整训练日志](server/runs/isambard/unknownsize-028-train-7461666.out)、[J completion](server/runs/unknownsize-20261013/a028/27b-joint/train/final/completion.json)、[J guard](server/runs/unknownsize-20261013/a028/27b-joint/train/training_guard.json)。
- [只读远程核验脚本](remote-verify.py)、[基础模型完整哈希脚本](remote-base-verify.py)。`server-verification.json` 的 `base_weight_shards_rehashed=false` 只描述第一阶段，完整分片哈希由随后独立的 `base-verification.json` 补齐。
- `server/` 下的文本副本均按服务器原始 SHA256 核对；训练数据和模型权重留在服务器，未下载进证据目录。
