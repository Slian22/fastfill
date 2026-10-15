# 022 全测试与 025 盲评修正复核

审查基准 HEAD：`942bc7c2a`（结果提交 `7e2e36384`，修正提交 `579591181`）。主代理与 Model QA Specialist、Reality Checker 两个复用子代理进行本地只读核对；未启动模型推理，未修改运行代码，未查询或干预远端作业，未查看正在运行的 judge 评分。

**结论：全测试主要数字成立，旧规则实测与新规则探索性推荐的区分已明确。正式交付前仍需完成文件包组装核验，并确保 90% 配对覆盖率进入最终 critic 采纳判定。**

## 已核实的全测试结果

QA 核对硬盘逐房输出、请求尺寸与汇总；报告四项 MANIFEST 哈希、原始 rooms.jsonl/result.json 的绑定哈希均通过。未重新运行底层几何评测或 bootstrap。

| 阶段（全部请求 8,647） | 检查器通过 | 独立严格准入 |
|---|---:|---:|
| raw | 3,834（44.339%） | 1,869（21.614%） |
| S1 | 6,418（74.222%） | 2,519（29.131%） |
| system | 7,697（89.014%） | 2,970（34.347%） |

最终按物体数分层的检查器通过：≤20 件 6,732/7,214（93.319%）；21–50 件 771/1,086（70.994%）；>50 件 194/347（55.908%）。[结果表](../results/system-022-test-sharded/levels.csv)、[报告](../results/system-022-test-sharded/report.md)。

- 升级 2,229/8,647（25.778%）；O2 候选通过检查器 1,278/2,229（57.335%）。应写“检查器修复率”，不是所有功能或严格准入均修复。
- 最终检查器增加 1,279 间：O2 贡献 1,278，fallback 另贡献 1。最终来源 placer 7,348 / O2 1,297 / fallback 2。
- 最终空布局 0。逐物体对齐全部请求：108,253 个物体的 ID 顺序一致、尺寸逐值变化 0。这表示保持请求先验尺寸，不表示恢复真实尺寸。
- raw/S1 实际完整数为 8,643，最终才是 8,647；三位小数显示 1.000 会掩盖前两阶段的 4 间未完成。
- 评估表的独立 strict 为 2,970；pipeline 联合 accepted（checker 且 strict）为 2,966。差 4 间是语义区别，不是算术错误。

GT 检查器 6,252/8,647（72.302%），严格准入 2,553/8,647（29.525%），参考表哈希通过。系统在这些规则下得分超过 GT，不代表视觉质量、真实性或网格碰撞优于 GT。该结果属于 **旧 checker 触发、O2 四分片的系统**，不能归为单个 27B 或新 ESC-STRICT 的成绩。

“免费 O2”宜改为“本地 O2 升级，GPT API 调用为零”；本次 O2 有 4,814 次回答调用，GPU 成本并未归零。

## 已落实的修正

- [模型卡](../results/delivery-022-final/MODEL_CARD.md)第 3、5 行明确：新规则是探索性推荐，卡内实测数字来自旧 checker 触发。示例命令已带 `--escalate-on strict`。
- [pipeline.py](../../../../fastfill/system/pipeline.py) 的 strict 排序第二键已改用独立 `_strict`，修复前次独立 strict 与联合 accepted 混用的问题。
- [judge_bundle_025.py](../../../../fastfill/comparison/judge_bundle_025.py) 正式 dev 提示加入每个对象的 `{id, category}`。本地 `a025/judge-dev-v2/MANIFEST.json` 明确该版本，246 房、492 例、构建失败0；B/C 状态绑定为 round-0/round-2。
- Reality Checker 实查正式包 1,234 个清单文件，哈希全部匹配；246 房的两臂提示词逐字一致，映射与布局全部对应。984 张图片哈希与旧包一致，中性视图保留。dev 数据清单及其5个文件也匹配；未核验远端实际执行。
- 主代理执行两个针对新增修正的既有测试：`test_prompt_lists_every_box_id_with_its_category`、`test_strict_trigger_escalates_and_ranks_by_strict_then_required`，**2 passed**。

## 交付前剩余事项

### 文件清单不等于已组装的文件包

[delivery.json](../results/delivery-022-final/delivery.json)第 2 行状态仍为 `MANIFEST_READY`，列出 72 个待复制文件。相关作业脚本 [isambard_022.sbatch](../../../../fastfill/system/isambard_022.sbatch)第 83 行仅调用 `fastfill.system.package build`。`build` 生成清单、模型卡及校验信息，不复制这些文件。

真正组装由 [delivery.py](../../../../fastfill/comparison/delivery.py) 的 `assemble`（281 行）复制并逐文件核验，生成 `ASSEMBLY.json`，状态 `DELIVERY_PACKAGE_ASSEMBLED`。本次本地证据尚未提供该收据，因此只能确认“交付清单及模型卡生成完成”，不能认证一个完整可运行文件包已组装并验证。未查询远端其他目录，不能据此断言远端从未单独执行过组装。

现有交付目录的三项 MANIFEST 哈希通过；51 个清单代码文件的哈希也与本地当前代码全部一致。`git_dirty=true` 不等于无法追溯：逐文件哈希仍在。其余权重/先验/来源文件需由 assemble 统一核验。

### 模型卡和机器可读元数据仍有旧描述

模型卡第 15 行及英文对应段仍写“仅未完成或 checker 失败才升级”；delivery.json 第 65 行仍是旧的候选排序。它们与推荐的 strict 命令不一致。应把推荐方法描述更新为 checker **或** strict 失败触发，并列出独立 strict、required={} 排序；旧实验数字继续保留旧规则身份。

### 90% 覆盖要求已写进协议，自动采纳尚未强制执行

[025 附录 B](../amendment-025-critic-repair-20261015.md)要求至少 222/246 房有 B/C 两臂有效评分，并按臂及房间族报告缺失。当前 [critic_025.py](../../../../fastfill/comparison/critic_025.py) 的 `adoption` 仍只检查评分差 CI、strict 差和 checker 差，没有配对覆盖检查。

正式结果不得只取该函数的 `adopted=True`：必须另核验配对覆盖门槛，最好落实为失败时拒绝采纳的代码条件，并报告缺失。协议门槛和代码是否执行应分别记录。尚未完成的 FID、网格碰撞与 judge 结论仍标未测；本次没有核实预计完成时间。

Reality Checker 本次用真实函数构造反例：246 房中仅221对评分有效，其余缺失，有效对 C−B=+1、几何护栏差为0，仍得到 CI=[1,1]、adopted=true。这是未达222门槛的合成反例，不是声称本轮已出现该缺失率。当前 per_family 只计选样，尚无按族 judge 缺失账。

### 采纳必须绑定正式提示词版本

旧 `judge-dev` 和正式 `judge-dev-v2` 使用相同布局，`geometry_manifest_sha256` 相同。`critic_025.evaluate` 714 行只以几何摘要核对 judge 与 B/C 的对应，不能强制排除已在协议中禁用于采纳的旧无类别诊断评分。最终评估应额外绑定正式 bundle/提示词摘要，并核对模型及运行身份。此项是版本防误用缺口，并未发现实际使用了旧评分。
