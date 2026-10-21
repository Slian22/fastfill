"""Render audited summary tables; preserve full exports separately."""
from pathlib import Path
import json
import re

AUDIT = Path(__file__).resolve().parent.parent
ROOT = AUDIT.parents[3]
OUT = ROOT / "paper/FastFill-all-comparisons-bilingual.tex"
LABELS = {
    "method": "Method / 方法", "model": "Model / 模型", "system": "System / 系统", "arm": "Arm / 分支", "stage": "Stage / 阶段", "scope": "Cohort / 队列", "entry": "Policy / 策略",
    "requested": "Requests / 请求", "requests": "Requests / 请求", "rooms": "Rooms / 房间", "layouts": "Complete / 完整", "objects": "Objects / 物体", "scene": "Scene / 场景",
    "complete_rooms": "Complete N / 完整数", "complete_rate": "Complete / 完整率", "geometry_success": "GSR / 几何成功", "geometry_success_rate": "GSR / 几何成功", "checker_rate": "GSR / 几何成功", "strict_rate": "Strict / 严格准入", "noreq_rate": "No-req / 无必选规则",
    "size_log_mae": "Size log-MAE / 尺寸误差", "log_size_error": "Size log error / 尺寸误差", "size_iou": "Size IoU / 尺寸IoU", "oor": "OOR / 重叠比", "oor_fraction": "Old OOR / 旧重叠代理", "bias_fp": "Footprint bias / 占地偏差",
    "bottom_center_error_m": "Position (m) / 位置误差", "yaw_error_rad": "Yaw (rad) / 朝向误差", "bev_iou": "BEV IoU / 平面IoU", "object_collision_rate": "Object collision / 物体碰撞率", "object_hard_violation_rate": "Object violation / 物体违规率",
    "clean_rooms": "Clean N / 已知无失败数", "clean_rate_all": "Clean/all / 全请求无失败", "room_clean_rate": "Clean / 已知检查无失败", "clean_room_rate_incl_failures": "Clean/all / 全请求无失败", "room_collision_rate": "Room collision / 房间碰撞率", "strict_ok_rate": "Old strict / 旧严格检查", "strict_rate_old": "Old strict / 旧严格检查", "interface_pass_rate": "Interface pass / 接口通过率",
    "escalated_share": "Routed / 升级比例", "big_used_share": "Big selected / 采用大模型", "selected": "Selected N / 选定数", "success": "Scored N / 完成评分", "failed": "Failed N / 失败数", "training_loss": "Training loss / 训练损失",
    "boundary_fail_checks": "Boundary fails / 越界项", "obb_overlap_fail_pairs": "OBB pairs / 重叠对数", "all_fail_checks": "All fail checks / 失败检查项",
    "downstream": "Strict / 准入", "functionality_mean": "Func. / 功能", "functionality": "Func. / 功能", "layout_mean": "Layout / 布局", "aesthetics_mean": "Aesth. / 美观", "aesthetics": "Aesth. / 美观",
}


def esc(value, breakable=True):
    text = str(value if value is not None else "--")
    if isinstance(value, float) or re.fullmatch(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", text):
        text = format(float(value), ".5g")
    text = text.replace("−", "-").replace("–", "--").replace("—", "---").replace("′", "'")
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    result = "".join(mapping.get(c, c) for c in text)
    if breakable and not re.fullmatch(r"[+\-\d.eE]+", text):
        result = result.replace(r"\_", r"\_\allowbreak{}")
        result = result.replace("/", r"/\allowbreak{}")
        result = result.replace("-", r"-\allowbreak{}")
        result = result.replace("→", r"\allowbreak{}$\to$\allowbreak{}")
        result = re.sub(r"[a-fA-F0-9]{32,}", lambda m: r"\allowbreak{}".join(m[0][i:i+8] for i in range(0, len(m[0]), 8)), result)
    return result


PREAMBLE = r"""% Standalone bilingual audit. Compile with XeLaTeX (twice).
% All report text and tables are embedded; no external .tex/.bib inputs.
\documentclass[10pt,a4paper,fontset=fandol]{ctexart}
\usepackage[margin=18mm]{geometry}
\usepackage{booktabs,longtable,array,amsmath,enumitem,xurl}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\usepackage{microtype}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{3em}
\linespread{1.0}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\setlist{nosep,leftmargin=*}
\newcommand{\src}[1]{\par{\footnotesize\textbf{Source / 来源:} #1}\par}
\begin{document}
\begin{center}
{\Large\bfseries FastFill: Complete Comparison Audit}\\[4pt]
{\large 全部历史对比结果审计（英文＋中文）}\\[4pt]
20 October 2026
\end{center}
\section*{Scope and reading guide / 审计范围与阅读说明}
This report inventories the locally available FastFill comparison records in the repository and mounted hard disk. It preserves unfavorable results, earlier variants and incomplete runs. It is an audit of saved outputs and their provenance, not a new inference run or an independent recomputation of every geometric metric. Different cohorts, tasks, model versions and denominators are reported separately. The complete CSV/JSON exports retain all strata and sensitivity analyses; the tables below show the overall comparisons and diagnostic summaries.

本报告清点仓库与已挂载硬盘中可访问的 FastFill 历史对比，保留不利结果、旧版本和未完成记录。本次核对的是已保存输出与来源，没有重新生成布局，也没有重新计算全部几何指标。不同队列、任务、模型版本及统计分母分别报告。完整 CSV／JSON 导出保留所有分层及敏感性分析；下表汇总各类整体对照与诊断。

The current experiment archive contains 34 result families plus the completed a027 evidence. All 87 CSV files (61,295 logical data rows) were exported without changes and independently checked against their sources. A further 35,465 rate/count consistency assertions across 18 applicable CSVs passed. Historical repository and hard-disk evidence is recorded separately; duplicate copies and report revisions are not independent replications.

当前实验存档包含 34 类结果及已完成的 a027 证据。87 份 CSV、61,295 行数据已无损导出，并逐行对回原文件；18 份适用 CSV 的 35,465 项比例／计数一致性校验通过。较早的仓库和硬盘结果另行登记；副本与报告修订不能当作独立重复实验。

Displayed decimals are rounded to five significant digits; the exports preserve full recorded precision. Rates remain on the scale stated in each table; decimal rates such as 0.85 mean 85\%.

表中小数显示为五位有效数字；全量导出保留原始精度。比例按各表标注的尺度展示，例如 0.85 表示 85\%。

\paragraph{Metric boundaries / 指标边界。}
For the current unknown-size study, GSR and strict-admission rates use all requests as the denominator. OOR is object-overlap area divided by total footprint area, averaged over complete rooms; it is not an out-of-room rate. Size log-MAE is conditional on complete outputs. Historical v2 ``known-check clean'' excludes unknown mesh/physics checks and must not be relabeled as current strict admission. GPT-judge scores are automated judgments, not human usability scores. FID/KID, asset-generation scores and layout metrics belong to different protocols.

当前未知尺寸实验中，GSR 与严格准入率以全部请求为分母；OOR 是物体交叠面积与总占地面积之比，对完整房间取均值，不是越界率。尺寸 log-MAE 也以完整输出为条件。历史 v2 的“已知检查无失败”不包括未知的网格／物理检查，不能改称当前严格准入。GPT 裁判分数不是人工可用性；FID／KID、资产生成及布局指标采用不同协议。

\paragraph{Evidence status / 证据状态。}
Measured LLM-agent comparisons exist, including early v2 tests and the later fixed-size and self-size cohorts. The OptiScene014 reproduction was launched, but no completed comparable layout score was found in the local records or searched hard-disk archives. Training-resume toy tests and request manifests are not layout results. ATISS, DiffuScene and LayoutGPT are literature references in these records. RoomGenBench contains separate historical asset-generation scores for seven methods plus a reference, evaluated on 184 assets; they do not establish FastFill layout superiority over those methods. These are statements about the available archive, not the live status of a remote server.

确实存在 LLM 代理实测对照，包括早期 v2、后期固定尺寸及自选尺寸队列。OptiScene014 已有复现启动记录，但本地及已检查硬盘归档中未发现完成的可比布局评分；训练恢复单测和请求清单不是布局成绩。ATISS、DiffuScene、LayoutGPT 在这些记录中属于参考文献。RoomGenBench 还保存 184 个资产上七种方法加参考资产的独立评分，不能用于声称 FastFill 布局胜过这些方法。本结论仅针对可访问存档，不代表远程服务器实时状态。
\tableofcontents
\clearpage
"""


def normalize_table(t):
    columns = t["columns"]
    if columns and isinstance(columns[0], dict):
        keys = [c.get("key", c.get("name")) for c in columns]
        labels = [c.get("label", c.get("name", c.get("key"))) for c in columns]
    else:
        keys = columns
        labels = [LABELS.get(c, c.replace("_", " ")) for c in columns]
    data = [[row.get(k, "--") for k in keys] if isinstance(row, dict) else list(row) for row in t["rows"]]
    assert all(len(row) == len(labels) for row in data), t.get("title")
    return labels, data


def render_table(t, index, sources):
    labels, data = normalize_table(t)
    # Long status/provenance text stays lossless in full-width notes, not a
    # narrow numeric-table cell that may become taller than one page.
    cell_notes = {}
    displayed = []
    for row in data:
        output_row = []
        for value in row:
            if isinstance(value, str) and len(value) > 140:
                if value not in cell_notes:
                    cell_notes[value] = f"M{len(cell_notes)+1}"
                output_row.append(cell_notes[value])
            else:
                output_row.append(value)
        displayed.append(output_row)
    title = t.get("title", f"Comparison {index}")
    title_zh = t.get("title_zh", "")
    title = title.replace("Arrangement diagnostics: arrange-comparison-300", "Arrangement: 300-request comparison")
    title_zh = title_zh.replace("布局诊断 arrange-comparison-300", "布局诊断：300请求对照")
    full_title = title + (" / " + title_zh if title_zh else "")
    text = [r"\subsection[" + esc(full_title, False) + "]{" + esc(full_title) + "}"]
    for note in (t.get("note"), t.get("note_zh")):
        if note:
            text.append(esc(note))
    # Wide diagnostic matrices retain every column through separate panels.
    identity = [0, 1] if len(labels) > 1 else [0]
    if t.get("id") in {"same_input_1454", "same_input_300", "judge_common_means"}:
        identity = [0, 1, 2]
    if len(labels) <= 7:
        panels = [list(range(len(labels)))]
    else:
        remainder = [j for j in range(len(labels)) if j not in identity]
        per_panel = 7 - len(identity)
        panels = [identity + remainder[i:i+per_panel] for i in range(0, len(remainder), per_panel)]
    for pi, panel in enumerate(panels, 1):
        n = len(panel)
        first = 0.30 if n > 3 else (0.22 if n == 2 else 0.38)
        widths = [first] + [(0.89 - first) / (n - 1)] * (n - 1) if n > 1 else [0.89]
        if len(panels) > 1 and len(identity) == 3 and n > 3:
            widths = [0.18, 0.15, 0.10] + [0.46 / (n - 3)] * (n - 3)
        elif len(panels) > 1 and n > 2:
            widths = [0.28, 0.13] + [0.48 / (n - 2)] * (n - 2)
        spec = "@{}" + "".join(r">{\raggedright\arraybackslash}p{" + f"{w:.4f}" + r"\linewidth}" for w in widths) + "@{}"
        header = " & ".join(esc(labels[j]) for j in panel) + r" \\"
        text += [r"\begingroup\small", r"\begin{longtable}{" + spec + "}", r"\toprule", header, r"\midrule\endfirsthead", r"\toprule", header, r"\midrule\endhead", r"\bottomrule\endfoot"]
        for row in displayed:
            text.append(" & ".join(esc(row[j]) for j in panel) + r" \\")
        text += [r"\end{longtable}", r"\endgroup"]
    for value, code in cell_notes.items():
        text.append(r"\par{\footnotesize\textbf{" + code + "} " + esc(value) + r"}\par")
    paths = t.get("source_paths", t.get("sources", []))
    if isinstance(paths, str):
        paths = [paths]
    ids = []
    for path in paths:
        path = str(path)
        if path not in sources:
            sources[path] = f"S{len(sources)+1:03d}"
        ids.append(sources[path])
    if ids:
        text.append(r"\src{" + ", ".join(ids) + "}")
    return "\n".join(text)


def main():
    sections = [("repo-results.json", "Current experiments / 当前实验"), ("historical-results.json", "Earlier repository experiments / 较早仓库实验"), ("disk-results.json", "Additional hard-disk comparisons / 硬盘补充对照")]
    parts = [PREAMBLE]
    sources = {}
    table_count = 0
    for filename, title in sections:
        evidence = json.loads((AUDIT / filename).read_text())
        tables = evidence.get("report_tables", [])
        if not tables:
            raise ValueError(f"Missing reviewed summary tables: {filename}")
        parts.append(r"\section{" + esc(title, False) + "}")
        if filename == "repo-results.json":
            zh = {"J": "8B 联合预测尺寸与放置；selected 保留 SFT", "P": "8B 使用先验尺寸放置；selected 为 rpo-dpo1", "A/arm": "训练后的 27B 放置模型，使用先验尺寸", "R/r": "8B rpo-dpo1 参考模型；必须区分推理引擎", "L": "gpt-6.1-sol，medium 推理", "O1a": "Qwen3-8B，关闭思考", "O1b": "Qwen3-8B，开启思考", "O2": "Qwen3.8-27B，开启思考", "O3": "gemma-4-31b，关闭思考", "O4": "gpt-oss-120b，medium 推理", "O5": "DeepSeek-V4-Flash：引擎失败，未运行", "O6": "Qwen3.8-Flash-Next-FP8，思考／medium", "O7": "Muse-Glimmer-30B，默认设置", "O2nt": "未微调 Qwen3.8-27B，关闭思考；不是 A", "S": "旧检查器触发的 27B＋S1＋本地O2系统", "SC": "S 加 critic 的174间测试", "raw suffix": "评分采用首答，不替换为重试答案", "harness suffix": "最多三次回答的解析重试；不是S1", "-P suffix": "提供固定训练先验尺寸，并在评分前恢复", "S1 stage": "生成后的确定性修复；不是解析重试"}
            glossary = {"title": "Model and protocol codes", "title_zh": "模型及协议缩写", "columns": ["Code", "Definition / 定义"], "rows": [[key, value + "；" + zh.get(key, "")] for key, value in evidence["model_registry"].items()], "source_paths": [str(AUDIT / filename)]}
            parts.append(render_table(glossary, 0, sources))
        for table in tables:
            table_count += 1
            parts.append(render_table(table, table_count, sources))
        dictionary = evidence.get("metric_dictionary")
        if dictionary:
            parts.append(r"\subsection{Historical metric dictionary / 历史指标说明}")
            if isinstance(dictionary, dict):
                for key, value in dictionary.items():
                    parts.append(r"\paragraph{" + esc(key, False) + "} " + esc(value))
            elif isinstance(dictionary, list):
                for item in dictionary:
                    parts.append(r"\paragraph{" + esc(item['key'] + " / " + item.get('label_zh', ''), False) + "}")
                    for field in ["definition", "definition_zh", "denominator", "unknown_details"]:
                        if item.get(field):
                            parts.append(esc(item[field]))
    parts += [r"\clearpage", r"\section{Source index / 完整路径索引}", "Each source ID below points to the exact local artifact used in the summary tables. The archive also includes source-paths.txt and the three audit inventories for all discovered result files. Paths are provenance, not compilation dependencies.", "下列编号给出汇总表所用文件的完整本地路径。全量已发现文件另见 source-paths.txt 及三份审计清单；这些路径用于追溯，不是编译依赖。"]
    for path, sid in sources.items():
        parts.append(r"\par\noindent\begin{minipage}{\linewidth}\textbf{" + sid + r"}\quad{\footnotesize\nolinkurl{" + path + r"}}\end{minipage}\par")
    parts.append(r"\end{document}")
    OUT.write_text("\n\n".join(parts) + "\n")
    (AUDIT / "report-source-index.json").write_text(json.dumps({"tables": table_count, "sources": sources}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source": str(OUT), "summary_tables": table_count, "source_paths": len(sources)}))


if __name__ == "__main__":
    main()
