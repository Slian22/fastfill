# FastFill 全部历史对比审计 / Complete comparison audit

先打开 [完整双语 PDF](../../../../paper/FastFill-all-comparisons-bilingual.pdf)。对应的 [单文件 LaTeX](../../../../paper/FastFill-all-comparisons-bilingual.tex) 包含全部正文与表格，XeLaTeX 可独立编译，不需要其他 tex/bib 文件。

这是完整对比附录：**68 页、75 张总体汇总表**，另有模型字典、历史指标字典及 96 个汇总来源的完整路径。之前的6页 evaluation 文件是精选简稿，不能当作本次全量审计。

This is the full comparison appendix: 68 pages, 75 overall summary tables, model/metric dictionaries, and 96 exact summary-source paths. The earlier six-page evaluation is a selected short draft. All unfavorable results, incomplete runs, older report versions and scoring caveats are retained.

## 已找回的比较 / Recovered comparisons

| 类别 / Category | 已保存的内容 / Saved evidence |
|---|---|
| 外部LLM / External LLMs | 8种代理配置，首答与多轮；自选尺寸及同先验尺寸；1,454房间及300房间试点；旧v2 GPT prompt/harness同300请求及293共同完成配对 |
| 模型及消融 / Models and ablations | 8B J/P、SFT/DPO/开发集选模、S1/八选一、偏好训练消融、27B与8B、critic、关闭思考控制、旧v2 yaw/解码与朴素基线 |
| 系统策略 / System policies | 8,647房间各阶段、022/027触发规则、026 A/B/C离线回放、成本、分层和尺寸敏感性；复用与旧版本明确分开 |
| 视觉与下游 / Visual and downstream | FID/KID、网格碰撞、两种GPT裁判及一致性、V1.1阶段和尺寸/碰撞分层、RoomGen交接诊断 |
| 资产方法 / Asset methods | RoomGenBench七种方法加SAGE reference，184资产、154描述、1,472逐资产行；与自由布局评价分开 |

## 审计中发现的问题 / Audit findings

1. 旧v2 clean忽略未知mesh/physics检查；GPT prompt的252/293=86.01%与252/300=84.00%是不同分母。旧300场景参与FastFill选模，不能称独立留出测试。Historical clean rates and modern strict admission are different metrics.
2. 旧报告的selected/DPO1 Holm列不能改称原始SFT共同主终点；同名P的历史strict相差1间已分别保留。Later stage comparisons remain descriptive; historical score revisions are preserved.
3. a025-test Markdown的dev/B/“尚无judge”模板过期；已按硬盘test/S→SC精确JSON与后续173配对judge记录说明。The later score artifact takes precedence over stale narrative status.
4. InfinigenV2的60条资产状态在旧评分CSV与当前sidecar间变化；旧评分不冒充更新后的重评。RoomGenBench asset scores are archived snapshots.
5. OptiScene014启动及迁移归档已查，但可访问存档中未找到完成的同基准布局评分；ATISS/DiffuScene/LayoutGPT是文献参考。This does not claim that no newer result exists on a remote server.

## 全量数据与完整路径 / Full data and paths

- `repo-results.{md,json,csv}`：34类当前结果及a027；87份CSV的61,295逻辑行、89份JSON完整保留。JSON另含50张总体表及原文件/行号/JSON pointer。
- `historical-results.{md,json,csv}`：18张历史表、36份原始快照、38,712条数值、18项历史指标字典和资产状态漂移。
- `disk-results.{md,json,csv}`：7张补充表、630条数值；9,160候选路径及归档/镜像检查。候选含副本、计划、单测，不是独立实跑数。
- `ALL-PATHS.txt` / `PATHS.csv`：9,780条去重完整候选及来源路径；`summary-source-manifest.json`含96个主表来源的SHA-256。
- `VALIDATION.json` / `export-validation.json` / `numeric-integrity.json`：编译、全页视觉检查、逐行无损核验、35,465项比例/计数校验结果。

The export preserves raw precision. The PDF rounds standalone decimal values to five significant digits. CSV exports from different audits may reference the same run and must not be summed as independent experiments.

## 编译 / Compile

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error FastFill-all-comparisons-bilingual.tex
```

Overleaf: upload the single `.tex`, select XeLaTeX. All content and tables are embedded; source paths are provenance only.

## 覆盖边界 / Coverage boundary

本次审查可访问仓库、硬盘8个FastFill主目录、补扫目录和相关压缩包成员清单，另检索fastfill_ir/fastfill_ir_v34。没有启动新实验、重跑全部几何或访问远程服务器。14个两侧都存在的关键镜像文件SHA-256一致，未重复计为独立运行。所有目录级清单与排除项保留在三份审计JSON中。

The work audits saved evidence and consistency, rather than recreating every experiment. Local absence is reported as a local-evidence limit, not a global absence claim.
