# 修订 025：critic → 受限修复（dev 选择，探索性）

B = 27B + S1，C = 27B + S1 + critic 修复（最多 2 轮，最终状态为第 2 轮）。246 间 dev 房间，指标都由布局重新计算；C − B 为配对差，房屋连通组 bootstrap 2000 次，seed 42。

房间选择（Addendum A）：每个修订 015 房间族最多 31 间，族内取 sha256('a025:' + uid) 最小者。

| 房间族 | 房间数 |
|---|---:|
| bedroom | 31 |
| living | 31 |
| dining | 19 |
| kitchen | 31 |
| bathroom | 31 |
| study_office | 31 |
| meeting | 4 |
| balcony | 8 |
| laundry_utility | 3 |
| garage | 2 |
| circulation | 4 |
| storage_closet | 6 |
| public_commercial | 8 |
| other | 9 |
| none | 28 |

| 指标 | B | C | C − B | 95% CI | n |
|---|---:|---:|---:|---:|---:|
| 冻结检查器成功率 | 0.776 | 0.955 | 0.1789 | [0.128, 0.231] | 246 |
| 严格准入率 | 0.390 | 0.695 | 0.3049 | [0.250, 0.362] | 246 |
| required = {} 准入率 | 0.752 | 0.898 | 0.1463 | [0.103, 0.191] | 246 |
| OOR（越小越好） | 0.100 | 0.066 | -0.0338 | [-0.042, -0.026] | 246 |
| GPT judge (功能 + 布局) / 2 | 5.909 | 7.330 | 1.3996 | [1.236, 1.555] | 244 |

被编辑房间：217 / 246（0.882）；被移动物体 1076 个，位移（C 相对 B）中位 0.981 m、p95 6.522 m；朝向改变的物体 274 个。
不变量计数（物体数、id、类别、尺寸、父物体改变，必须全为 0）：{'count': 0, 'id': 0, 'category': 0, 'size': 0, 'parent': 0}。

| 轮 | case | 调用结果 | 有效编辑 | 被拒编辑（代码） | 编辑集 接受 / 总数 | 编辑集被拒原因 | 停止原因 | 进入下一轮 |
|---:|---:|---|---:|---|---:|---|---|---:|
| 1 | 246 | {'answered': 246} | 773 | 无 | 217 / 217（其中无变化 0） | 无 | {'no_valid_edits': 29, 'scores_ge_8': 15} | 202 |
| 2 | 202 | {'answered': 202} | 466 | 无 | 138 / 142（其中无变化 1） | {'admission_worse': 4} | {'edit_set_rejected': 4, 'max_rounds': 109, 'no_valid_edits': 60, 'scores_ge_8': 29} | 0 |

采纳规则（修订 025 §2，三条都要满足）：{'judge_ci_lower_gt_0': True, 'judge_coverage_ge_0.9': True, 'strict_admission_ge_-0.005': True, 'checker_ge_-0.005': True}。结论：**采纳**。

失败或无效的 critic 调用不重试、不替换，房间保持未编辑并计入上表。按构造，C 的检查器和严格准入在每个房间都不劣于 B（编辑集只有在字典序不变差、严格准入不由过变不过时才被接受）。这是 dev 上的探索性结果，不改变任何已报告结果。

披露：dev 房间无法接入 run-002 的 Blender 管线，judge 看的是 critic 的同款盒子视图（judge_bundle_025，image_source = box_views），物体按冻结检查器标记着色；按构造 C 的检查器结果不劣于 B，这种着色可能使 judge 端点偏向 C。
