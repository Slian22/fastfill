# FastFill 实验与评测：中英双语简稿

完整历史比较现见 [`comparison-audit-20261020`](../comparison-audit-20261020/README.md) 及 [`FastFill-all-comparisons-bilingual.tex`](../../../../paper/FastFill-all-comparisons-bilingual.tex)。下列 evaluation 文件仍是精选简稿。

当前版本：[`paper/FastFill-evaluation-bilingual.tex`](../../../../paper/FastFill-evaluation-bilingual.tex) 是包含外部 LLM 实测对照的中英双语单文件；预览见[同名 PDF](../../../../paper/FastFill-evaluation-bilingual.pdf)。本目录的模块化 `.tex`、`main.pdf` 和 ZIP 保留为补充外部对照前的初稿快照；新增数据来源见 `baseline-evidence.json`。

初稿可打开 `main.pdf` 阅读。英文与中文内容对应；这是一段可移入论文的评测章节，不是整篇论文，也没有启动新实验。

## 文件

- `main.tex` / `main.pdf`：双语源文件入口与预览。
- `evaluation-en.tex` / `evaluation-zh.tex`：可单独移入论文的英文、中文章节。
- `table-*.tex` / `numbers.tex`：由已保存 CSV/JSON 提取的表格与数值。
- `references.bib` / `literature-notes.md`：四篇核验过的参考文献及指标出处。
- `evidence.json` / `metric-contract.md`：数值来源、文件哈希及精确指标口径。

## 编译

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Overleaf：上传 `fastfill-evaluation-bilingual.zip`，选择 XeLaTeX，主文件选 `main.tex`。使用 TeX Live 自带 Fandol 中文字体，不依赖本机字体。

## 已确定的报告口径

1. **几何质量：GSR + 尺寸 log-MAE**。OOR 和完整率一起报告；失败不能按 OOR=0 处理。
2. **应用侧：strict admission**。这是本地下游规则，既不是人工 UR，也不代表网格无碰撞。
3. **视觉侧：FID / KID + 网格无超阈值碰撞率**。只在相同渲染子集内比较，明确 N；新 a027 没有对应视觉结果。
4. **成本：优化比例、生成轮次和 GPU 时间**。本地 O2 的零 API 调用不等于零计算成本。

正文主表是同批 8,647 房间的阶段比较。视觉表是旧规则的 1,005 房间子集。旧 V1.1、8B J/P、固定尺寸代理回放不混入这两张表。原预注册比较的终点与本次写作重点分开记录。

本节没有提供可直接声称胜过论文基线的共同 benchmark，也没有人工 UR。GPT 裁判评分保留于原始结果，但简稿不混用裁判单独给出的 overall 与三个子分数的算术平均。
