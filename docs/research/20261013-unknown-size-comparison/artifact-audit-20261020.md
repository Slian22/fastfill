# FastFill 已有对比与 LaTeX 审核（2026-10-20）

**结论：对比已经做过，且不止当前六页简稿中的几张表。** 本次核对了当前项目、挂载的 `/Volumes/harddisk` 相关实验目录，以及邻近 Worldedge 项目、Downloads、Documents 中的论文候选。没有启动训练、推理或远端任务；原始实验文件及现有论文简稿保持不变。

## 1. 直接使用的文件

| 文件 | 用途与版本 |
|---|---|
| [当前双语 LaTeX](../../../paper/FastFill-evaluation-bilingual.tex) | 2026-10-20 17:12 的单文件评测章节，含全测试阶段对照、固定尺寸外部 LLM 基线及视觉表；用 XeLaTeX 编译，不依赖其他 tex/bib 文件。不是整篇论文。 |
| [当前双语 PDF](../../../paper/FastFill-evaluation-bilingual.pdf) | 上述源文件对应的六页预览。 |
| [已有模型对比补充 LaTeX](../../../paper/FastFill-comparison-supplement.tex) | 本次按已保存 CSV/JSON 整理的中文补充材料；加入旧 8B、全八代理、消融、尺寸估计和历史结果。它是已有证据的整理，不是新增实验。 |
| [已有模型对比补充 PDF](../../../paper/FastFill-comparison-supplement.pdf) | 上述补充材料的可读预览。 |
| [审核材料包](../../../paper/FastFill-audited-materials-20261020.zip) | 两套 LaTeX/PDF、对比报告与结果表、来源和文件校验清单；不包含模型权重、大型渲染资产或逐条预测。 |

**版本区别：** `paper-draft-20261020/main.pdf` 和 `fastfill-evaluation-bilingual.zip` 是四页初稿快照，尚未加入外部模型对比。它们不是当前六页稿。此次交付使用 `paper/` 下的文件。

## 2. 实际完成的比较与证据入口

以下相对路径以本审核文件所在目录为根。不同队列、尺寸条件、模型规模和推理预算不能合并排名。

| 比较 | 已有范围 | 最直接证据 |
|---|---|---|
| 8B 联合预测 J 与先验尺寸放置 P：SFT、DPO1、dev 选模 | 每臂 8,647 test 请求；官方超参 DPO2 全测试未完成，不能用旧协议 DPO2 代替 | [阶段表](results/stage-table-010-001/stage_table.md)、[选定模型报告](results/report-v2-selected-001/report.md) |
| 八种外部代理完整基线 | L、O1a、O1b、O2、O3、O4、O6、O7；各 1,454 房间，自己估计尺寸；raw/harness 均有 | [代理主表 T5](results/report-v2-selected-001/T5.csv)、同目录 T5_contrasts.csv 与 T5_cost.csv |
| 统一 S1 修复 | J/P 全测试 8,647；代理队列 1,454；保留修复前后比较 | [S1 应用报告](results/test-apply-012/report.md)、同目录 levels.csv/contrasts.csv |
| 固定尺寸重新生成 | O2/O6/O4/O7 与 **8B P** 的 1,454 队列；L 另有 300 房固定尺寸试点 | [全量固定尺寸报告](results/compare-016-full-v1/report.md)、[300 房试点](results/compare-016-pilot-v1/report.md) |
| DPO2 归因与尺寸控制 | dev600 的 RFT/on-policy DPO2、种子42/43、合成扰动；尺寸替换与缩放曲线另存 | [消融报告](results/ablation-016/ablation-report.md)、[尺寸曲线](results/scale-curve-016/report.md) |
| 尺寸估计 | 1,877 房间的训练先验、L-size、O2-size | [尺寸评测](results/sizes-020/report.md) |
| 27B 与 8B 选型 | dev600、确认3,000、test8,647；另存 S1 结果 | [选型报告](results/a021-selection/final-report.md)、[测试集 S1](results/a021-test/s1-report.md) |
| 最终系统与升级策略 | 旧 checker 触发和新 strict 触发均有 8,647 实际全测试结果；026 策略分析为保存输出的离线回放 | [旧系统](results/system-022-test-sharded/report.md)、[新 strict 全测试](evidence/a027-resume-20261015/completed/eval/report.md)、[026 回放](results/frontier-026-v4/report.md) |
| 微调贡献与房型分层 | 未微调 O2 对照、15 房型族 | [未微调消融](results/frontier-026-ablation-o2nt/report.md)、[房型族](results/families-015/report.md) |
| 视觉、网格、裁判与 critic | FID/KID、mesh，1,005 成功渲染房间；174 预选裁判房；critic dev246 与 test174，最终有效配对173 | [visual-024](results/visual-024/)、[mesh](results/mesh-collision-019/report.md)、[双裁判一致性](results/judge-agreement-023/)、[critic 测试结果](results/a025-test/) |

更早的独立协议也已有结果：

| 历史材料 | 已核实内容 | 使用限制 |
|---|---|---|
| [clean-DPO 汇总](../20261011-v11-clean-dpo/evaluation_full/research-summary.md) | SFT/DPO1/DPO2 的12格评测完成，共100,146条请求记录 | `diagnostic_only`；输入排序继承 GT 派生信息，不能作为当前任意未知尺寸列表协议的结果 |
| [2026-10-07 训练及300房对比](../../../outputs/fastfill_v2/comparison-20261007b/autorun/SUMMARY.md) | Qwen3-8B训练完成；FastFill raw/spread 对 LLM prompt/harness；另有4,455三字段测试请求 | 联合尺寸/位置/yaw；300房来自选模验证队列 |
| [2026-10-09 布局对比](../../../outputs/fastfill_v2/eval-sets-20261009/arrange-comparison-300/report.md) | 293/300共同成功房间的布局指标、配对区间与房型分层 | 关系保持和碰撞结果也必须一起读，不能只报告对 FastFill 有利的尺寸误差 |

## 3. 硬盘、副本与其他 LaTeX

- `/Volumes/harddisk/fastfillout/unknownsize-20261013/compare-016-full-v1/report.md` 与 `compare-016-pilot-v1/report.md`，分别和项目中的同名报告 SHA256 一致。
- 外盘交付包 `FastFill-Delivery-20261015/evidence/frozen-checker-evaluation/test/levels.csv`、`system-022-final/test-sharded-eval/levels.csv`、`delivery-final/evaluation/test/levels.csv` 三份一致，SHA256 为 `84f37dc7b2b3035018429a6789d18face8997dca21ef260517963a2b5df96a40`；这是旧 checker 触发系统，不能当新 strict 规则的成绩。
- 外盘 `visual-final-selected-002/final-results/` 还有旧 J/P/L/O* 视觉报告。其全系统共同成功 FID 子集仅71房，GPT评分683/696成功；与后来的 visual-024 人群不同。材料包保存这份独立报告，不能把它的分母换成1,005。
- `Downloads/worldedge_table1_nature_latex.zip` 及同名目录确有 LaTeX，但原文件明确写 `UNMEASURED` / `Research template; no experimental results`。这是 WorldEDGE 表格模板，不是 FastFill 实测结果。
- 邻近 `particulate-hinge` 也有 `particulate-fixed-results.tex` 和 `latex-fixed-2026-10-20/main.tex`，属于另一模块。外盘本次所查的 FastFill 相关范围未发现 `.tex`；FastFill 论文源文件在本机项目中。

## 4. 本次校验及审核边界

当前简稿重新在临时目录独立编译：XeLaTeX/latexmk 退出码0，无 LaTeX 警告；重新生成的文本与当前 PDF 完全一致。六页均已渲染检查，无遮挡、缺字或表格溢出。

新增补充材料共七页、十张表，22个来源文件的SHA256全部匹配；通过独立协议/内容审核、XeLaTeX编译及逐页排版检查，无编译警告，也不依赖外部tex/bib/图片。

简稿的11条主表/基线数值行和2条视觉数值行，均从原始 CSV/JSON 重新格式化并逐行对照，中英文两份全部匹配。来源侧记录的20项文件哈希及简稿tex/PDF两项哈希全部一致。另对结果目录 MANIFEST 绑定且本地存在的127个文件核验，未发现哈希不符。这些核验集合有重叠，不应相加当成独立文件数。

外盘相关检索收集3,725个文档/CSV/JSON候选及139份非归档 report.md，并重点核对最终报告和相关副本；**没有逐条重算所有预测，也没有逐份审读3,725个候选**。排除了系统索引、依赖、模型权重、大资产和无关项目；所选范围未遇到权限错误。没有连接远端服务器确认仍在运行的作业。

## 5. 会影响论文写法的区别

1. **状态文件有过期快照。** `current-results.md/json`停在10月13日，结果README停在早期修订；更早README还写“训练未启动”。完成情况应以较新的具体报告、CSV和运行证据为准。
2. **模型和尺寸条件不能混写。** 旧表P为8B，当前A/27B-P为27B。L既做过1,454房自估尺寸，也做过300房固定尺寸试点。事后替换尺寸与给定尺寸重新生成不是同一种实验。
3. **尚无结果的项目要保留为空。** O5 DeepSeek四次引擎/JIT尝试失败，未跑出基线；蒸馏018仅有dev结果、未采纳且未跑test；a028现有本地/外盘证据仍只有启动记录，未找到完成结果，远端状态未查询。现有 OptiScene 配方复现没有完成的可比评测；ATISS、DiffuScene、LayoutGPT在当前稿里是文献参照，不能当本项目重跑基线。
4. **旧模板有文字错误。** `a025-test/report-geometry.md`误写“174 dev”和复制的B描述；实际为test阶段S→SC。交付模型卡的“mesh未测”已被后来的visual-024结果覆盖。原件保留，补充材料按实际证据标注。
5. **结论范围要守住。** 026是探索性离线回放；027才是新strict触发真实全测试。当前27B在固定尺寸外部对比的严格准入上没有胜过这些代理。strict admission不是人工可用率UR，历史测试集不是新盲测，旧系统的视觉成绩不能移给新系统。

材料包中的 `MANIFEST.json` 列出每个成员的原始位置、大小及SHA256；`audit-checks/` 保存本次主稿数值、哈希及编译核验记录。旧报告作为历史证据保留，使用时以本文件说明的协议和时间边界为准。
