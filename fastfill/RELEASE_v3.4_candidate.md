# FastFill 数据 v3.4（候选）发布说明

状态：**候选，未提交、未发布**。v3.4 = v3.3 + 源数据保真审计（2026-10-14，`docs/research/20261013-unknown-size-comparison/evidence/source-fidelity-audit-20261014/synthesis.md` 第 4 节）里 B1–B11 的修复。本文所有数字都是在本机实测得到的，原始数字在 `/Volumes/harddisk/fastfillout/data-v34-check/summary.json`，脚本在同目录的 `scripts/`。

## 0. 结论

- 冻结研究用的测试集一共 8,647 间。在 v3.4 下有 8,550 间保留，97 间被拒（96 间因为 B11 `oob_far`，1 间因为 B8 变成 proxy 边界）。被拒的 97 间在冻结 GT 下检查器、严格接受、required={} 三项**全部失败**。
- 只看保留的 8,550 间，把冻结 v1 GT 和 v3.4 GT 逐间配对比较：
  - 检查器：6,252 → 6,308（+66/−10），73.1% → 73.8%
  - 严格下游接受：2,553 → 2,650（+106/−9），29.9% → 31.0%
  - required={} 接受：5,235 → 5,267（+46/−14），61.2% → 61.6%
- 如果按全部 8,647 间计算、把被拒的房间记为失败：检查器 0.723 → 0.730，严格接受 0.295 → 0.306，required={} 0.605 → 0.609。
- 审计的判断不变：GT 的失败绝大多数来自官方数据本身。
  - 严格接受 +97 间，其中 78 间来自 Holodeck 去掉 10 cm 外扩（B4）。Holodeck 的严格接受从 425/503 变成 503/503。
  - 剩下的提升主要来自 B1 按面取整：R5（穿墙 ≤1.2 cm）在 484 间房里消失，只在 7 间房里新出现。
- 训练侧，以 2,000 间分层样本推算到 124,843 间合格池：
  - 保持冻结先验时，P 训练目标（位姿）约 7.8% 改变（221/2,000），主要来自 B7 倾斜盒（扫描数据、SAGE）和 B1。
  - 若重新拟合先验，P 目标约 23.6% 改变（522/2,000）。
  - J 目标约 81% 改变，主要是 B1 的 ±1 cm 尺寸。
  - B11 不影响 P/J 训练池，这些房间原本就因 `oob_objects` 被排除。
- 文本协议没有任何变化。`scene.py` 和 `fastfill/comparison` 的代码都没改。用新代码快照的 contract 从冻结房间重新生成 8,647 条 J 请求，与冻结文件逐字节一致。

## 1. 硬约束：遵守情况

- 冻结数据一律没有重建或修改：
  - eval-data-v1、main-data-v1
  - v3.2/v3.3 的 IR 与发布包：`/Volumes/harddisk/fastfill_ir`、`.release/v3.2`、`unknownsize-source-correction-20261013/ir-candidate-003`
  - 本次只读取了它们。
- v3.4 IR 写在新目录 `/Volumes/harddisk/fastfill_ir_v34/`。逐房构建的 GT 只写到 scratch 目录 `/Volumes/harddisk/fastfillout/data-v34-check/prep/`，没有生成可发布的数据集。
- 使用的代码是 git `06161c75b` 加上工作区里未提交的修复（只涉及 `fastfill/adapters/*`、`anchors.py`、`build.py` 和两个新测试）。
  - 代码先快照到 `data-v34-check/code-new/`，diff 存为 `code-new/working-tree.diff`，所有运行都用这份快照，不受其他工作流改动影响。
  - 对照组用的是 HEAD 的快照 `code-old/`。
- 没有 commit、push、SSH，没有用 GPU 或 API；并行进程不超过 4 个。
- 在代码快照上运行 `pytest fastfill/tests`（包含新的 `test_v34_build.py` 和 `test_source_fidelity_v34.py`）：321 个全部通过。

## 2. 测量方法，以及它能精确复现冻结结果的证据

1. **构建路径**与冻结语料完全相同：`build.prep(IR 行, v3.2 MANIFEST args)` → `contract.bind_room(seed)` → 补上冻结的 group/component/split → `contract.messages(..., 'joint')[:2]`。
   - v3.4 在 args 上追加了 `reject_flagged += ['oob_far']`，这是 `build.main` 的 B11 行为。
2. **复现性检验**（`code-old`，v3.2 IR）：
   - 8,647/8,647 间房的标签和 J 请求与 eval-data-v1 的 `rooms.jsonl`、`joint/requests.jsonl` 完全相等。
   - 冻结 GT 打分为检查器 6,252、严格 2,553，与 scale-curve-016 的 GT 行逐间一致（8,647/8,647，包括 checker、downstream 和 reasons）。
3. **GT 打分**与 scale_curve 在 s = 1 时打 GT 的方式一致：
   - 流程：`report_v2._gt` → `output_repair.predicted_room` → `geometry.geometry`（冻结检查器，`geometry_success`）+ `downstream.accept`。
   - 严格接受 = `accepted`。
   - required={} = `endpoint_ok and attach_ok and admit_no_required_ok`。
4. **归因**用了两种方法：
   - 阶段链：V1（旧代码 + v3.2 IR，等于冻结）→ V2（新代码 + v3.2 IR，只含构建修复）→ V3（+v3.3 IR）→ V4（+v3.4 IR，即 v3.4 GT）。
   - 留一法：在 V4 上逐个关闭 B1/B2/B3/B11（通过 monkeypatch，不改代码）。
5. **训练侧**：
   - 用新旧代码分别 prep 全部 144,150 个训练 uid。
   - 从合格训练池（124,843 间，与 main-data-v1 的 prior_fit_count 一致）中按来源分层抽 2,000 间，按 `main_data._candidate` 原样构造 P/J 训练目标。
   - 先验采用在 v3.2 合格池上重新拟合的 full prior。它对 8,647 间测试房的 `size-estimates.jsonl` 复现了 8,647/8,647，因此与冻结的 full_prior 完全一致。
   - 注意：冻结训练实际使用的是 5 折 OOF 先验，那份文件不在本机。

## 3. 改了什么、为什么、影响多大

下表中，"IR" 指 v3.3 → v3.4 的 IR 逐对象差异；"构建后" 指经过取整（cm / 整度）后模型实际看到的变化。

| 编号 | 层 | 改动（文件） | 为什么 | IR 影响（全部 / 测试房） | 构建后在测试集上的影响与指标翻转 |
|---|---|---|---|---|---|
| B1 | BUILD | 正交盒按两个面取整到半厘米网格，再推出整厘米的中心和尺寸（`build._cm_faces/_cm_candidates/_snap`）；随后 `_round_safely` 在 ±2 cm 内挑一个不因取整而新增穿墙、碰撞或失去支撑的候选；fixed 盒只在没有不变大的候选时才变大 | 旧代码中心和尺寸各自取整，贴墙的面最多移动约 7.5 mm，造成约 2,190 个 ≤1.2 cm 的穿墙 | 不在 IR 层 | 7,385 间房变化：21,184 个物体尺寸变化（x/y 轴上 −1 cm 15,821 次、+1 cm 7,441 次、−2 cm 6 次，平均每轴 −0.04 cm），557 个物体平移 1–2 cm，12,837 个 fixed 盒变小 1 cm（15,151 个轴）。R5 在 484 间消失、在 7 间新出现。19 间带 `rounding_unresolved`。留一法：检查器 +49/−3，严格 +25/−8，required={} +29/−9 |
| B2 | BUILD | 悬空物体如果吸附到 z = 0 会压进下方的盒子，就改为挂到该盒子上（`anchors.support_below`）；build 把它放到父盒顶面（`support_gap`） | 旧逻辑会把衣柜压进沙发、把烤箱压进柜子 | 不在 IR 层 | 33 间房：110 个物体 z 变化，其中 64 个换了父物体；2 个物体和 4 个 fixed 盒被去掉。检查器 +2 |
| B3 | BUILD | SAGE-10k 的门只从结构记录生成，名字里带 door 的道具算作家具（`anchors.structural`） | 这类道具离墙 11–33 cm，旧逻辑把它们当成门开口 | 不在 IR 层 | 15 间 SAGE 房：16 个假门 fixed 盒消失。这些道具悬空、无支撑，会按墙面物件被丢弃，所以既不当门，也不被放置。required={} +6 |
| B4 | ADAPTER | Holodeck 的宽和深各减 0.10 m（`optiscene_holodeck.unpad`，最小保留 1 cm） | 官方 bbox 在网格盒外扩了 10 cm，两份 README 都没说明 | 66,855 物体 / 9,977 房；测试 3,367 物体 / 503 房 | 503 间房的尺寸全部变化。严格接受 +78（425 → 503），检查器不变（本来就是 503/503） |
| B5 | ADAPTER | assets.json 尺寸与 layout bbox 排序后相差 >10 cm 时改用 bbox（`il3d_3dfront.checked_size`，并加 `size_from_bbox` 标记） | IL3D 的 assets.json 里有错误尺寸，例如单位立方体归一化的椅子被记成 2.0 × 2.0 × 1.94 m | 2,134 物体 / 1,738 房（meta 计数 2,141 / 1,745）；测试 81 / 71 | 构建后 53 间房的 58 个物体变化。脚印变小的占 60%，变大的占 40%（新/旧比中位数 0.70，p10 0.15，p90 2.08）。3 间房不再是 oob_far。IL3D 检查器 +4/−3，严格 +2/−1，required={} +7/−4 |
| B6 | ADAPTER | URDF 盒如果 ≥70% 被刚体盒覆盖就丢弃（`hssd200._covered`） | 这些是关节体的残留副本 | 82 盒 / 54 房；测试 8 / 6 | 4 间房去掉 6 个物体，指标不变 |
| B7 | ADAPTER | 倾斜物体改用角点投影的最小面积矩形，朝向在原朝向 ±45° 以内（`internscenes.min_area_yaw/tilted_bbox`，sage10k、scenesmith、hssd、interiorgs、structured3d、il3d、multiscan 共用） | 旧做法是按 yaw 对齐的外包盒，会放大脚印 | 70,761 物体 / 15,716 房；测试 4,989 / 1,015 | 在 cm/度取整后可见的变化只有 1,029 个物体、466 间房（SceneSmith 417，SAGE 292，scannet 141，arkit 99，3rscan 76）。检查器在这些来源上 +5/−4（arkit +1/−1，SAGE +2/−2，SceneSmith +2/−1，S3D 归入 B9） |
| B8 | ADAPTER | MultiScan 判断"落地物体"时，测试范围从 z < 0.05 改为底 ≤ 0.5 m（`multiscan.boundary`） | 起点 6.9 cm 高的沙发漏检，房间被错误地当作 hull | 13 间 hull → proxy，42 间边界变化，12 间里 514 个物体随房间坐标系平移 | 测试 1 间变成 proxy 后被拒（`boundary`） |
| B9 | ADAPTER | Structured3D 丢弃中心在房外的"最近房间"分配物体；门窗只有在官方拓扑把它连到本房墙上时才保留（`structured3d.convert/topology_links`） | 0.5 m 最近房间规则把隔壁的板、门带进本房 | 3,046 盒 / 1,752 房（适配器 meta：最近房间且在房外 13,921 个 / 8,471 房，未连接门窗 583 / 547 房）；测试 108 / 77 | 3 个放置物体、20 个 fixed 门窗被去掉；1 间房不再是 oob_far；检查器 +2 |
| B10 | 数据 | 用修复后的代码重新生成全部 IR | 硬盘上的 `fastfill_ir/MultiScan.jsonl` 和 `Scan2CAD.jsonl` 是旧文件 | 相对旧 MultiScan：858 物体 / 100 房变化；Scan2CAD v3.4 = v3.3（640 物体 / 361 房，属于 v3.3 已发布的修正） | 冻结研究用的是 `.release/v3.2/ir`，那里的 MultiScan 已经是 09-27 的倾斜修正版，因此 B10 不影响冻结测试集 |
| B11 | BUILD | 有物体伸出边界 >0.5 m 的房间标记为 `oob_far`，每次构建都拒绝（`build.prep`/`main`） | 这种物体不在本房（扫描地板不完整、邻房物体、损坏盒），边界不可信 | 不在 IR 层 | 测试 96 间被拒：SpatialLM 39，IL3D_3dfront 36，HSSD 7，SpatialGen 4，gen 3，arkit 2，InteriorGS 2，S3D 2，SceneSmith 1。这 96 间冻结 GT 三项指标全部失败。训练全量中 1,773 间被拒 |

另外有两处 docstring 已更正：mp3d 的说明从 "1696 → 1600" 改为 1601；MultiScan 的说明从 "10,957/10,957" 改为 10,887。

### 3.1 v3.4 IR（`/Volumes/harddisk/fastfill_ir_v34/`，18 个文件全部用修复后的代码重新生成）

| 来源 | 行数 | 与 v3.3 相同 | 与 v3.2 相同 | 原因 |
|---|---:|:---:|:---:|---|
| SpatialLM | 54,376 | 是 | 是 | 不受影响（作为对照重新生成，字节一致） |
| MansionWorld | 27,884 | 是 | 是 | 同上 |
| IL3D_synthetic | 6,954 | 是 | 是 | 同上（不调用 `checked_size`） |
| Scan2CAD | 1,506 | 是 | 否 | v3.3 修正；B10 指的旧文件是 v3.2 版本 |
| SpatialGen | 48 | 是 | 否 | v3.3 修正 |
| OptiScene_holodeck | 9,977 | 否 | 否 | B4 |
| IL3D_3dfront | 21,832 | 否 | 否 | B5（+B7 3 个） |
| HSSD200 | 2,351 | 否 | 否 | B6 + B7 |
| Structured3D | 22,894 | 否 | 否 | B9 |
| MultiScan | 273 | 否 | 否 | B8 + B7 |
| SAGE-10k、SceneSmith、InteriorGS、InternScenes_{3rscan, arkit, gen, mp3d, scannet} | 10,027 / 364 / 6,607 / 1,334 / 5,040 / 14,089 / 1,601 / 1,513 | 否 | 否 | B7 |

各文件的 sha256 见 `summary.json` 的 `ir.files`。

## 4. 测试集：冻结 v1 GT 与 v3.4 GT（逐来源）

下表在 v3.4 保留的房间上配对比较，格式为 冻结 → v3.4（+变好/−变差）。

| 来源 | 测试间数 | v3.4 保留 | 新拒绝 | 检查器 | 严格接受 | required={} |
|---|---:|---:|---:|---|---|---|
| SpatialLM | 2461 | 2422 | 39 | 1478→1493 (+17/−2) | 421→426 (+7/−2) | 788→796 (+12/−4) |
| MansionWorld | 1313 | 1313 | 0 | 1282→1284 (+2/−0) | 736→747 (+12/−1) | 1254→1257 (+4/−1) |
| IL3D_3dfront | 872 | 836 | 36 | 431→439 (+11/−3) | 155→156 (+3/−2) | 492→503 (+15/−4) |
| Structured3D | 747 | 745 | 2 | 464→478 (+15/−1) | 126→123 (+1/−4) | 261→261 (+3/−3) |
| InternScenes_gen | 660 | 657 | 3 | 635→636 (+1/−0) | 79→81 (+2/−0) | 322→323 (+1/−0) |
| OptiScene_holodeck | 503 | 503 | 0 | 503→503 (+0/−0) | 425→503 (+78/−0) | 503→503 (+0/−0) |
| SAGE-10k | 489 | 489 | 0 | 425→431 (+7/−1) | 16→16 (+0/−0) | 380→389 (+10/−1) |
| SceneSmith | 364 | 363 | 1 | 284→287 (+4/−1) | 87→88 (+1/−0) | 360→360 (+0/−0) |
| IL3D_synthetic | 316 | 316 | 0 | 301→301 | 302→302 | 316→316 |
| InteriorGS | 311 | 309 | 2 | 175→177 (+2/−0) | 107→108 (+1/−0) | 179→180 (+1/−0) |
| InternScenes_arkit | 229 | 227 | 2 | 132→136 (+5/−1) | 25→26 (+1/−0) | 149→149 |
| HSSD200 | 138 | 131 | 7 | 73→74 (+1/−0) | 52→52 | 91→91 |
| InternScenes_scannet | 104 | 104 | 0 | 14→15 (+1/−0) | 5→5 | 53→52 (+0/−1) |
| InternScenes_3rscan | 70 | 70 | 0 | 12→12 | 0→0 | 47→47 |
| SpatialGen | 45 | 41 | 4 | 29→28 (+0/−1，v3.3 修正) | 8→8 | 22→22 |
| InternScenes_mp3d | 18 | 18 | 0 | 11→11 | 5→5 | 13→13 |
| MultiScan | 6 | 5 | 1 | 3→3 | 4→4 | 5→5 |
| Scan2CAD | 1 | 1 | 0 | 0→0 | 0→0 | 0→0 |
| **合计** | **8647** | **8550** | **97** | **6252→6308 (+66/−10)** | **2553→2650 (+106/−9)** | **5235→5267 (+46/−14)** |

### 4.1 按阶段归因（配对比较，两个阶段都保留的房间）

| 阶段 | 改动的房间 | 检查器 | 严格 | required={} |
|---|---|---|---|---|
| 构建修复 B1+B2+B3+B11（v3.2 IR 上） | 7,387 间变化，100 间被 B11 拒绝 | +53/−3 | +25/−8 | +35/−9 |
| v3.3 已发布的源修正（Scan2CAD、IL3D 窗宽、SpatialGen） | 16 间 | +0/−1（SpatialGen） | 0 | 0 |
| v3.4 适配器修复 B4–B10 | 1,045 间变化；4 间不再被 B11 拒绝，1 间被 B8 拒绝 | +11/−7 | +80/−1 | +8/−5 |

留一法（在 V4 上关闭单项修复）的结果：

| 关闭的修复 | 检查器 | 严格 | required={} | 备注 |
|---|---|---|---|---|
| B1 | +49/−3 | +25/−8 | +29/−9 | |
| B2 | +2/−0 | 0 | 0 | |
| B3 | 0 | 0 | +6/−0 | |
| B11 | — | — | — | 只影响保留集合：关闭 B11 时这 96 间被保留，打分 0/96、0/96、0/96 |

### 4.2 规则层面：冻结 8,647 间 → v3.4 保留的 8,550 间

| 规则 | 冻结 8,647 间 | v3.4 保留 8,550 间 |
|---|---:|---:|
| R5_OUTSIDE_ENVELOPE | 1,561 | 975 |
| A4_COLLISION | 4,429 | 4,254 |
| R2_OUTSIDE_1MM | 3,195 | 3,064 |
| R2_CHECK | 583 | 467 |
| A2_KEEPOUT_BLOCKED | 2,763 | 2,688 |
| A3_UNREACHABLE_REQUIRED | 2,858 | 2,796 |
| 检查器 main_collisions | 1,832 | 1,740 |
| 检查器 severe_collisions | 924 | 874 |
| 检查器 fixed_blocking | 620 | 575 |
| 检查器 oob_objects | 585 | 470 |

R5 从 18.1% 降到 11.4%。A4 的下降主要来自 Holodeck（B4）。剩下的规则失败绝大多数来自官方数据，见第 6 节。

### 4.3 v3.4 自身的回退（需要知道）

- **B1：变差的房间**。共有严格 8 间、检查器 3 间、required={} 9 间，原因如下：
  - `_round_safely` 只检查房间边界、检查器的碰撞规则（>0.01 m²）和支撑。它不检查下游的门前 keep-out 带（A2/R2_OUTSIDE_1MM，6 间）、A4 的 AABB+TOL 规则（3 间）和 R1（1 间）。
  - 源盒本身已经穿墙超过 1 mm 时，取整不受约束。新取整忠实保留了这个穿墙，旧取整只是碰巧把它藏了起来，所以有 7 间新出现 R5（SpatialLM 5、S3D 2）。例：`spatiallm:scene_005550::room_4` 的冰箱源盒穿墙 5.2 mm，旧显示 0，新显示 5.0 mm。
  - 检查器变差的 3 间（2 间 main_collisions、1 间 fixed_blocking）都是边缘碰撞对。`_round_safely` 用的碰撞判据（`_counted` + 0.01 m²）只是对 `validate.check` 的近似；源盒已经重叠超过 0.01 m² 时，取整不受约束。这 3 间没有逐一核实。
  - 建议：把 keep-out 带和 A4 规则加入安全检查；源盒已穿墙时只允许向内取整。
- **B1：系统性尺寸偏差**。显示尺寸变小的情况多于变大（x/y 轴 −1 cm 15,821 次，+1 cm 7,441 次，平均每轴 −0.04 cm）。fixed 盒几乎只会缩小（`grow=False`：15,151 个轴 −1 cm，只有 2 个轴 +1 cm）。对未知尺寸研究的尺寸目标来说，这是一个很小但方向固定的偏差。
- **B5**：40% 的修正让脚印变大，因为这时采用的是 bbox（3D-FRONT）的尺寸。审计只核对了 35 个资产，3D-FRONT 自身也可能是错的那一方（例如电视柜 3.65 m 对 2.49 m）。IL3D 有 3 间房因此检查器变差。建议对"变大"的那部分再与 3D-FRONT 原始 `size` 交叉核对一次。
- **B11 拒绝整间房，不做修复**。held-out 来源 SpatialGen 失去 4/45 间，IL3D_3dfront 失去 36/872 间，HSSD 失去 7/138 间。来源层面的表格会因此改变分母。

## 5. 训练数据的影响

### 5.1 全量训练集：144,150 间，v3.2 构建 → v3.4 构建

- **被拒 1,812 间**：
  - `oob_far` 1,773 间：IL3D_3dfront 808，SpatialLM 612，HSSD 132，InteriorGS 52，arkit 49，S3D 49，gen 36，MansionWorld 27，mp3d 5，scannet 3
  - `object_count` 27 间：B9 去掉了 S3D 房间里仅有的物体
  - `boundary` 12 间：MultiScan 变成 proxy
- **对 P/J 训练合格池几乎没有影响**。`oob_far` 的房间本来就带 `oob_objects`，已经被 `EXCLUDE_FLAGS` 排除；合格池只少了 10 间（5 间 oob_far、5 间 boundary），124,843 → 124,833。
- **房间内容变化**：133,843 间房有变化，绝大多数是 B1 的 ±1 cm。
  - 430,817 个物体尺寸变化，9,665 个物体平移，10,346 个物体朝向变化（B7）
  - 219,385 个 fixed 盒尺寸变化
  - 217 间带 `rounding_unresolved`，28 间带 `support_lowered`

### 5.2 P 训练目标：2,000 间分层样本（合格池，按来源分层，每源至少 50 间或全部）

构造方法与 `main_data._candidate` 完全相同：bind → 先验尺寸 → `pose_targets.repair` → `messages(placement)` → 准入判定。对比了两种先验：

- **固定先验**：用 v3.2 的 full prior，两边尺寸输入相同，只反映房间本身的变化。
- **重拟先验**：在 v3.4 合格池上重新拟合（124,833 间），同时包含先验漂移。拟合后类别中位数 32,407 个不变（<5 mm），4,933 个变化 <2 cm，1,583 个变化 ≥2 cm。

"总体估计"按分层权重推到 124,843 间合格池。

| 指标 | 样本（2,000） | 总体估计 |
|---|---:|---:|
| P 目标改变（固定先验） | 221（另有 2 间被拒） | ≈9,800（7.8%） |
| 　其中两版都被准入的房间 | 94 / 1,532 | ≈5,200 / 101,000（5.2%） |
| P 请求改变（固定先验；主要是 fixed 盒因 B1 缩小 1 cm） | 1,217 | ≈82,100（65.8%） |
| J 目标改变（B1 的 ±1 cm 尺寸、B4、B7） | 1,602 | ≈101,400（81.2%） |
| 准入变化（固定先验） | +11 / −3 | ≈+700 / −60 |
| P 目标改变（重拟先验） | 522 | ≈29,400（23.6%） |
| 　其中两版都被准入的房间 | 282 / 1,532 | ≈17,900（17.7%） |
| 估计尺寸改变（重拟先验） | 1,603 | ≈103,400（82.8%） |
| P 请求改变（重拟先验） | 1,846 | ≈118,300（94.7%） |

**按来源（固定先验）**：
- P 目标的变化集中在扫描数据和 SAGE，原因是 B7 改变了倾斜盒的 yaw：3rscan 31/59，arkit 33/89，scannet 28/59，mp3d 20/59，SAGE 38/139。
- 其余来源都不超过 8%：SpatialLM 28/380，HSSD 10/63，IL3D_3dfront 10/194，S3D 9/148，IL3D_synthetic 5/112，MansionWorld 2/243，InternScenes_gen 0/172。
- Holodeck 是 0/139：B4 只改 GT 尺寸，P 看不到 GT 尺寸。换成重拟先验后，估计尺寸缩小，Holodeck 有 52/139 的 P 目标改变。

**221 个变化的归因**（看留一法 / 阶段变体的 P 键是否改变）：
- 只有 B1：80
- 只有适配器修复（主要是 B7）：70
- 两者都有：66
- 涉及 B2：5
- 涉及 B3：1
- B11：0（这些房间已经被 `oob_objects` 排除）

**注意**：冻结训练使用的是 5 折 OOF 先验，这里用的是单一 full prior，所以准入的绝对值不同（样本准入率 76.8%，main-data-v1 为 79.2%）；但"改变了多少"这个数是可比的。缓存复用的 672 间房抽查 40 间，与完整重算 0 差异。

## 6. 没有修的问题：它们在官方数据里

以下问题 v3.4 **没有、也不应该**在转换里修，它们决定了 GT 的通过率上限：

- **官方盒子之间的碰撞**：测试 GT 的 3,938 个碰撞对中，3,725 个（94.6%）在官方发布的盒子里就存在。
  - 例：SpatialLM 1,083 对，InternScenes 共 847 对，IL3D_3dfront 852 对。
  - 盒子重叠不等于网格互穿。amendment 019 已经证明 GT 的盒碰撞大多不是网格碰撞。
- **官方穿墙**：超过 5 cm 的约 1,060 个，例如 SpatialLM 567，IL3D_3dfront 176，HSSD 65，InternScenes_arkit 62。
- **门前 keep-out 被挡**（A2/A3）：几乎全部是官方几何。
  - SpatialLM 有 3,006 次侵入。
  - InternScenes 41–63% 的测试门是打开的门扇或者不在地板轮廓上，由它们生成的门带没有意义。
  - 0.10 m 的门盒深度是我们的约定。
  - 检查器的 `fixed_blocking`（冻结时 620 间）没有被任何审计按层分类。
- **SceneSmith（UNIFIED）**：1,792 个资产尺寸是在不考虑 SDF scale 的情况下量的。确认有错的是 scene_077、102、112 里的置物架（6.41 m）、冷藏柜（4.96 m）、卡座、书架。要修需要官方 1.13 TB 的 tar，**本次未修**。
- **Structured3D（UNIFIED）**：
  - 门宽取 max(X 跨度, Y 跨度)，对斜门是错的（1,704 个测试门里有 6 个误差 >5 cm），未修。
  - 最近房间规则已由 B9 处理。
- **InternScenes（UNIFIED/官方）**：
  - 凸包边界比地板网格大 5% 以上的房间，在 421 间 Real2Sim 测试房里有 140 间，未修。
  - 官方还有损坏的盒子（例如 1,910 m 和 803,014 m 高的床），已被适配器的 `MAX_SIZE_M = 20` 丢弃；但审计提到的 8.86 m 床低于这个阈值，仍然保留。
- **MultiScan**：11 个物体盒几乎不包含自身点云，判断不了是官方还是统一化引入的，未修。
- **Scan2CAD**：SceneCAD 布局比扫描小（3 处穿墙，官方），未修。
- **评估端的人为因素**（不是数据问题，v3.4 不改）：
  - A4 把塞在桌下的椅子算作碰撞
  - MansionWorld 门侧 15 cm 余量
  - InteriorGS 把 "Door frame" 家具当作门
  - Holodeck 的吸附 × 外扩问题已由 B4 消除
- **已经训练好的模型**（J/P 用 v3.2 训练）学到了 Holodeck 的 10 cm 外扩（J 预测尺寸与外扩后的 GT 中位差 0.00 m），也学到了 5 mm 穿墙的取整目标。要改变这一点只能在 v3.4 上重新训练。冻结研究的结论按 synthesis 第 6 节处理：Holodeck 的 J/P 对 LLM 比较最多偏 6 pp；R5 以 1 cm 容差做敏感性分析。

## 7. 复现

```
D=/Volumes/harddisk/fastfillout/data-v34-check
# IR v3.4（代码快照 code-new；--only 的源名用空格分开）
cd $D/code-new && python -B -m fastfill.adapters --root /Volumes/harddisk/3D_Room_Collections --out /Volumes/harddisk/fastfill_ir_v34 --only <SOURCES...>
# 逐 uid prep（V1 旧代码 + v3.2 IR = 冻结对照；V4 = v3.4；--off B1|B2|B3|B11 为留一法）
python -B $D/scripts/prep_variant.py --code $D/code-new --ir /Volumes/harddisk/fastfill_ir_v34 --uids $D/test_uids.txt --variant V4_new_v34 --out $D/prep/test_V4_new_v34.jsonl
python -B $D/scripts/score_gt.py --variant V4_new_v34 --out $D/score/V4_new_v34.jsonl      # frozen: --variant frozen
python -B $D/scripts/compare_prep.py test V1_old_v32 V4_new_v34 $D/cmp/test_V1_V4.json
python -B $D/scripts/p_targets.py fit V1_old_v32 check && python -B $D/scripts/p_targets.py sample
python -B $D/scripts/p_targets.py run V4_new_v34 $D/priors/prior_V1_old_v32.json $D/ptargets/V4_prior32.jsonl $D/ptargets/V1_prior32.jsonl
python -B $D/scripts/summarize.py      # -> $D/summary.json
```

正式发布 v3.4 之前还需要：
1. 处理第 4.3 节的 B1 回退和 B5 中"变大"的部分（或者明确接受它们）。
2. 提交代码，再用 `build.py` 在新目录构建 v3.4 数据集。
3. 运行 `train --dry_run`、`tools/qa_report.py`、`tools/ablation_check.py`。
4. 更新 README 与 HF 卡片。
