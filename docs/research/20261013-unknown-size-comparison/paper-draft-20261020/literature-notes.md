# 评测写法与指标：四篇原论文核对

核对日期：2026-10-20。用途：为中英双语 **Evaluation** 小节选定指标，不是系统综述，也不宣称覆盖截至今日的所有方法。

## 检索范围与证据

针对室内布局生成，检索/读取了 arXiv、NeurIPS Proceedings 和 CVF Open Access；查询词包括论文名加 `evaluation metrics`, `FID`, `OOR`, `KL divergence`。纳入 OptiScene、ATISS、DiffuScene、LayoutGPT 四篇 2021–2025 年的一手研究，读取正式论文/PDF 指标段落与对应表格；排除博客、聚合站二手解释及不属于本次室内场景任务的 2D 图像指标。范围由当前对齐需求确定，不是全面检索后的四篇最佳论文。各研究的指标传统存在继承关系，不能当作四次独立验证。

| 论文 / 证据等级 | 原文指标与位置 | 对本项目有用的写法 | 不可直接照搬之处 |
|---|---|---|---|
| **OptiScene**：一手预印本，arXiv v2；指标定义可信，未据此确认同行评审状态 | §4.1，p.7，Tables 1–3：FID↓、OOR↓、GPT-4o 功能/布局/美学 0–10↑、人工 UR↑；Appendix D.2，p.17：OOR 公式；Table 9，p.21：评审提示 | 自动几何、渲染视觉与人工接受度分开；按数据来源和房型列结果 | UR 是 50 场景、每场景5人、至少3人判定物理及语义有效。自动准入不是 UR；OOR 是面积比，不是碰撞房间比例。§4.1 的简述不如附录公式精确 |
| **ATISS**：NeurIPS 2021，一手同行评审论文 | §4，p.7，Table 1：FID↓、类别 KL↓、真实/生成场景分类准确率；Table 2，p.8：生成耗时 | 分房型报告；明确相机/图像协议；效率单独列 | FID 用 256×256 顶视正交图，FID 和分类准确率重复计算10次取平均；分类准确率目标是 **50%**，不是越高越好 |
| **DiffuScene**：CVPR 2024，一手同行评审论文 | §4，p.5–6，Table 1：FID↓、KID↓、SCA（接近50%）、CKL↓；Table 2，p.7：物体数、对称物体对数、pairwise bbox IoU | 明确 1,000 个生成场景，统一相机、渲染与 256×256 顶视语义图；分布指标与布局统计分表 | PIoU 在该表中以接近 GT 统计为目标，不能简单当作“越低越好”的物理安全率；与 OptiScene OOR 不是同一公式 |
| **LayoutGPT**：NeurIPS 2023，一手同行评审论文 | §5.1，p.8，Table 5，p.9：FID↓、类别 KL↓、out-of-bound rate↓ | 越界率按“至少一个家具超出边界的场景占比”定义；表格明确重跑的基线 | 只评矩形房间，423卧室/53客厅；FID 每场景4视角，和 ATISS 顶视协议不同。不能搬其表格数值来与本项目结果排高低 |

这里的“同行评审”仅描述来源类型，不等于每个实现细节已充分报告。置信度：上述指标名称、主定义、表格位置为高；跨实现逐位等价性未经证明。

## 推荐在本项目写清楚的五点

1. **全量主表突出几何成功率与尺寸误差。** 使用 `Geometry success rate (GSR)` 表示完整输出通过冻结检查器的请求数 / 全部请求数，失败记0，并配合 `Size log-MAE` 检查尺寸估计。严格下游准入作为单独的应用指标，不能改名为论文 UR 或“真实可用率”。原始模型、修复后、升级系统分行，归因到对应阶段。这是回顾性阶段报告，不把新选择指标写成预注册主终点；原匹配实验的 J0/P0 预注册终点保持独立。
2. **OOR 是几何辅助指标。** 按 OptiScene 附录公式报告，并披露本地是否忽略 yaw、高度和支撑关系。失败输出的 OOR 不应记0；明确有效输出分母与失败率。预测尺寸可变化，OOR 降低不能单独证明布局质量提升；尺寸误差须配套报告。
3. **渲染结果另表。** FID↓、网格无碰撞率↑、GPT 功能/布局/美学↑仅对各自实际子集成立；各列注明场景数、成功渲染数及排除原因。固定资产、相机、分辨率、特征网络/版本、参考集、评审模型与提示。同名 FID 不代表不同论文数值可比。
4. **保留任务适用性。** 当前输入若固定物体列表，类别 KL 主要反映列表遵循，而非自主场景生成能力，不宜为了仿论文加入主指标。完整布局可多解，位置误差或 GT bbox IoU可作诊断，不宜取代物理与任务约束评价。已有 KID 结果可列入对应渲染子表；SCA 本次没有结果，不填数值。
5. **外部方法比较须重跑。** 共同测试 ID、输入信息、资产、生成预算、后处理和失败规则一致后才做比较。现有结果可写为阶段/系统比较；不要以不同论文已发表数字证明 SOTA。所有后验选出的指标与系统规则应说明是回顾性分析。

OptiScene 的逐场景公式（二维框集合为 $B_i$）：

$$\mathrm{OOR}(s)=\frac{\sum_{i<j}\mathrm{Area}(B_i\cap B_j)}{\sum_i\mathrm{Area}(B_i)}.$$

它不是 IoU，也不保证不大于1。推荐同时写明聚合：在可评分输出集合 $\mathcal V$ 上计算 $|\mathcal V|^{-1}\sum_{s\in\mathcal V}\mathrm{OOR}(s)$，另报 $|\mathcal V|/N$。这条聚合规范是针对本项目的建议，不能声称原论文完全规定了相同失败/聚合策略。

可直接采用的英文句子：

> We report automatic geometric validity separately from rendered-scene quality. For this retrospective stage comparison, we emphasize geometry success over all requested rooms, with failed generations counted as unsuccessful, and report size log-MAE alongside it. Strict admission is reported separately as an application-specific downstream metric; the original preregistered endpoints are unchanged. Raw model predictions and outputs after repair or escalation are evaluated as separate system configurations. Visual metrics are reported on explicitly identified rendering subsets and are not compared numerically with published results obtained under different protocols.

## 原始来源与引用元数据

全部访问于 **2026-10-20**。PDF 已实际下载并提取正文核对；下列地址可以复查。文献 BibTeX 来自官方 arXiv 导出接口，作者、标题、年份、arXiv编号未从记忆手填；仅把 citation key 改为短名。BibTeX 保留预印本记录类型，会议发表身份另由 NeurIPS/CVF 原始页面确认。

| BibTeX key | 正文 / 正式页面 | 元数据抓取来源 |
|---|---|---|
| `optiscene2025` | [arXiv v2 PDF](https://arxiv.org/pdf/2506.07570v2)；[HTML](https://arxiv.org/html/2506.07570v2) | [arXiv BibTeX](https://arxiv.org/bibtex/2506.07570) |
| `atiss2021` | [NeurIPS论文页](https://proceedings.neurips.cc/paper/2021/hash/64986d86a17424eeac96b08a6d519059-Abstract.html)；[正式PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/64986d86a17424eeac96b08a6d519059-Paper.pdf) | [arXiv BibTeX](https://arxiv.org/bibtex/2110.03675) |
| `diffuscene2024` | [CVF正式PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_DiffuScene_Denoising_Diffusion_Models_for_Generative_Indoor_Scene_Synthesis_CVPR_2024_paper.pdf) | [arXiv BibTeX](https://arxiv.org/bibtex/2303.14207) |
| `layoutgpt2023` | [NeurIPS论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html)；[正式PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/3a7f9e485845dac27423375c934cb4db-Paper-Conference.pdf) | [arXiv BibTeX](https://arxiv.org/bibtex/2305.15393) |

原论文未提供的统一测试协议、失败处理、跨论文可比性由本项目评测合同补充，不能靠引用自动成立。本次没有重新生成场景、没有训练、没有调用付费评审，也没有新增人类 UR 数据。
