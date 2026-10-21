# 全记录数据审计计划与边界 — 2026-10-13

审计对象是正在使用的 unknown-size J/P 数据和训练输入。目标是用实际全量记录判断现有训练的数据完整性，不研究文献、不计算新的模型质量分数、不修改比较代码或冻结数据、不重训。

## 预先规定的检查

1. 完整流式读取 J/P train、dev、test 请求文件。J movable 精确允许 `id/category`，P 精确允许 `id/category/size`；category 必须为非空字符串。逐条检查系统提示、角色、固定/可移动对象身份、目标完整性及 P 尺寸先验一致性。
2. 对所有 train/dev 成对记录重新进行离线 Qwen tokenizer 编码，核对长度、监督 answer+EOS 长度、完整 UID 共同准入及两臂排除规则。不加载模型，不在源文件上修改标签。
3. 完整读取历史 release、18 个冻结原始 IR、prepared corpus 和 identity。核对 UID/行号/行字节 SHA、原始与发布总体、未发布记录、来源分布；用所有原始 group/alias 边及所有发布 prepared content/layout 边重建 component。不能只查已有组件是否交叉。
4. 从历史 train flags、source-heldout 和重建组件独立恢复 prior-fit pool；重算五折分配、每折补集 UID/component 哈希及六组 axis-log-median 桶。浮点比较预设绝对与相对容差均为 `1e-12`，精确核对整数计数和量化到厘米后的输入尺寸。
5. 区分四种排除：未进入历史 release、发布 train 历史标记不合格、共同几何 qualification 不通过、8192-token 共同长度排除。保持完整 18 来源表，零计数也列出。
6. 单独评估原始/来源几何边界：semantic front、代理 boundary、缺失 wall/height/asset 信息、已知转换缺陷。输入 schema 通过不代表原始几何、网格或物理语义正确。

## 运行与输出边界

- SSH 使用父任务提供的 control socket，主机 `u6uv.aip2.isambard`。初始为 `/tmp/ff-u6uv-koee4y6n/control.sock`；父任务停止旧控制器并通知切换后，结果读取使用 `/tmp/ff-audit-ssh-oo470s58/control.sock`。不创建/关闭父任务的 mux。
- 数据根：`/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill`。
- 所有新增脚本、作业日志、机器结果及报告均使用 `docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-*` 名称。
- CPU 密集扫描只在有时限的 Slurm CPU 作业内运行；不请求 GPU，不修改训练 7365681/7365682、其依赖评估或冻结快照。
- 输入审计作业 7367962：4 请求 CPU、32 GB、45 分钟上限，程序 timeout 43 分钟。只写独立审计输出，其中 SQLite 是工作索引，不是训练数据。
- 来源几何暴露关联作业 7367978：4 请求 CPU、32 GB、30 分钟上限，程序 timeout 28 分钟。逐条扫描冻结 IR / prepared / bound 数据；不打开原始网格。
- 仅全量扫描完成且记录总数、覆盖率及 SHA 对齐后，才允许使用“全量通过”。运行中、超时或未读取的文件均明确标记。
- lineage与P teacher补充作业7368003：4请求CPU、32GB、45分钟上限，零GPU；独立重建组件、五折补集与六套先验，随后检查P train/dev target契约。
- 二进制 mesh/texture/model weights 不在本数据审计中逐个读取或重新认证。IR 的 asset 元数据/路径存在不等于资产完整或可正确渲染。

## ARIS 使用方式

已阅读项目保存的 [research-review](aris-reference/research-review/SKILL.md)、[experiment-audit](aris-reference/experiment-audit/SKILL.md)、[reviewer-independence](aris-reference/shared-references/reviewer-independence.md) 和 [review-scope-limits](aris-reference/shared-references/review-scope-limits.md)。采用直接读取原始证据、作用范围说明和结论分级原则。本数据审计不额外调用API；用户已授权正式评测API。本任务也不是外部跨模型审稿；不会冒称完成 ARIS 指定的外部跨模型验收。完整论文贡献/指标结论由父任务合并，本报告只给出数据证据对论文表述的约束。

## 报告分级

- P0：已证实会使当前实验失效的严重数据问题。
- P1：实质影响当前数据或允许结论的缺陷/限制，必须披露或处理。
- P2：有明确证据的局部问题或复现/验证缺口。
- P3：说明性观察、已正确执行的边界或后续完善点。

未来 loader 补丁不能反向证明冻结训练受污染；必须报告当前实际违规条数。未知准确率保持未知，不能由 finite loss、工程测试或 schema 验证代替。
