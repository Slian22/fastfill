# 源修正输入与版本化重建计划 — 2026-10-13

冻结 v3.2 的 build/prep/split 与当前工作区相同；修正重建应采用**冻结代码副本 + 3 个适配器和 1 个独立 helper**。18 来源中重生 3 份完整原始 IR、其余 15 份保持字节一致。stock build 没有历史 split 锁定选项，必须明确核对并报告数据修正引起的去重、准入、组件和 split 差异。

本次只读定位原始源与静态代码，只写本 Markdown/JSON；未改原始数据、源码、冻结包和运行，未启动新作业或重复构建。本数据审计不额外调用API；用户已授权正式评测API。使用已有 ARIS 审计与范围约束，不冒称完成外部跨模型审稿。**55/55个历史病例已完成结构源定位**。逐病例路径、JSON pointer、变换、SHA 和读取范围见同名 JSON；历史已知病例用于回归，不能当作所有原始 mesh 已检查的证明。

新产物根：`/Volumes/harddisk/fastfillout/unknownsize-source-correction-20261013`。父任务 `source-correction-storage-20261013.json` 实测外置盘剩余 **2.822 TB**（用户约2.84 TB）；约90GiB是内部项目盘。运行时约10.9GB可用 RAM，legacy build 将全部 retained rooms 存入内存，不可声称全流程流式或内存必然够用。

## 最终执行顺序与父任务基线更新

已读取外部 root 的 `baseline-ir-regeneration.json` 和 `baseline-historical-diff.json`；未重复扫 IR。父任务基线耗时 **327.421秒**：Scan2CAD 1506行、SpatialGen 48行均与历史 IR **BYTE IDENTICAL**；IL3D 21832行 UID/顺序不变，仅12房间、19个数值字段有差异（pos6项、size13项），最大 `2.220446049250313e-16`。这是已观测的数值漂移，不计为源修正，也不能未经检查假设其永不影响阈值/舍入。

实际冻结修正包为新 root **`corrected-code-001/`**；本次独立核对41个历史代码文件：38个字节不变，仅3个适配器替换，另加 `upright_bounds.py`。build/prep/split/anchors/scene/validate 全部保持历史版本。最终 IL3D SHA 为 `51c600230e1cfb79f929c11e4d98e9d9d7b94fd91f32b8555f3a2024d2e7814a`；其余 SHA 和包 manifest 记录在 JSON，覆盖先前工作树观察版本。

父任务已在 **`corrected-ir-001/`** 重生完整3源；最新报告为进行中，本审计没有把未完成产物当完成证据。接受的下一阶段是：

1. 等待现有 corrected IR 作业完成；对比 **corrected↔baseline** 归因修复，对比 **corrected↔historical** 记录实际版本差异。核对全部源行和对象变化，包括 floor median 连带效应及数值阈值；补齐并验证其他15源原字节。
2. 父任务在远端使用有界 CPU Slurm、明确 RAM/time 配额，以 `corrected-code-001` 与原 manifest 参数构建**新 legacy release 候选**。保留 `constraint_frac=0.3` 等全部旧参数；无需新增 GPU。
3. 构建后完整比较所有 UID 的 membership/split/组件。**任何旧 dev/test 改 split 均阻断下游**，包括 dev↔test 和进入 train；旧 heldout 被删除/新 prep 拒绝也必须单列并解决。新准入、去重或旧 train 分配变化逐项披露，不能只报三个 split count。
4. 通过以上 gate 后才固定新 release manifest，重建 corpus 和完整 main-data prior/OOF/P repair/refit，再生成新 eval-data。现有旧版本 main-data 运行继续保护，不重复启动同一构建，不混合两版产物。

这一路径可执行；当前没有额外源码依赖阻断。尚未证明的是 corrected 全量差异、candidate release 的 split/资格稳定性，以及远端实际资源配额。此计划完成不等于新 release 已通过验收。

## 冻结代码与最小覆盖包

绝对基线：`/Users/slian/Desktop/3D/Worldedge/OptiScene/.release/v3.2/code/fastfill/`。权威 manifest：`.release/v3.2/data/v3.2/MANIFEST.json`（SHA `664eecdcdaea593bc7f8653ea544b46e415f1c0bc8bf7e09065fbe89796caf6e`）。本次完整计算其中 **41 个代码文件 SHA，41/41 匹配**。

`build.py` 包含 `prep`，没有另一个 `prep.py`。构建依赖闭包 `__init__.py/build.py/anchors.py/scene.py/split.py/validate.py` 全部与当前工作区逐字节相同。`build.py` SHA 为 `fbd36a4f841f4606ab0475e097077c4ea906046caa71aecba0295640c831939f`。历史 build_start 与最终代码仅 `tools/visualize.py` 不同，不参与 prep。当前工作区额外改变了 `merge_lora.py`、`tools/visualize.py`，与本次源修正无关，不复制进覆盖包。

父任务已独立复制冻结包到新 root 的 `corrected-code-001/`，仅覆盖：

- `fastfill/adapters/scan2cad.py`
- `fastfill/adapters/spatialgen.py`
- `fastfill/adapters/il3d_3dfront.py`
- 新 `fastfill/adapters/upright_bounds.py`（只依赖 `math`）

保留冻结的 `adapters/__init__.py`、`__main__.py`、`unified.py`、`internscenes.py` 和 `scene.py`。registry 会导入整包后筛选有 SOURCE/load 的模块，因此保留整份冻结 adapter 包；独立 helper 不需要修改 registry。IL3D 的 `upright/tilted_bbox` 依赖 `internscenes.py`，不要因函数名相似顺带替换它。运行依赖包括 numpy、Shapely、ijson；冻结 requirements 不是精确版本锁，需记录实际版本。输出 SHA、容差或 flags 差异若无法由修正解释，先排查依赖版本。当前代码由别的工作者持续修改，本 JSON 记录的是观察时的 SHA，父任务应在验证后再固定最终覆盖包 SHA。

**修正适用范围须全量记录：**当前 Scan2CAD 补丁会给所有有效源盒做完整旋转投影，包含 ≤10° 的小倾斜；它还重算 floor median，可连带改变同房间其他对象的 Z。SpatialGen 补丁对所有有效 affine 盒投影，并用未舍入的原始 translation；IL3D 新逻辑覆盖合法共面 vertical outlines，并增加非法 profile 拒绝。这些不等同于“只改55个历史对象”：应列出小倾斜、舍入、floor 连带、跳过对象、窗口方向/dedup 等实际 delta。全量修正同类源错误是目标，未解释的其他行为变化是阻断项。

## IL3D 普通斜墙窗：代码工作者可立即使用

- consumer：`fastfill/adapters/il3d_3dfront.py::openings`；exporter：`/Volumes/harddisk/3D_Room_Collections/BillLin66__3D_Room_Collections/3DFront_exported/kits/convert_3dfront.py::indoor_window`。
- 原始导出：`/Volumes/harddisk/3D_Room_Collections/BillLin66__3D_Room_Collections/3DFront_exported/3dfront.jsonl`，39个历史已确认saved fixed windows全部匹配源对象。JSON列出每个opening的行号、字节偏移、`/room/windows/<index>`及行SHA。
- 原始房屋：同目录`data/3D-FRONT/<house>.json`；22个病例房屋全部存在且可读。JSON列出源room与Window/WallInner mesh候选指针。Export没有constructid，opening索引不能假定等于raw mesh索引；精确outline归属需复用exporter的有界单房间复现。
- 原始布局：`/Volumes/harddisk/3D_Room_Collections/imChuling__3D_Room_Collections/IL3D_exported/IL3D/layout.zip`的24个`layout/<scene_id>.json`成员均已读。聚合`3dfront.json`及`IL3D/assets.json`存在。USDZ模型目录和`3D-FRONT.zip`不在此处；窗轮廓修正不依赖这些家具网格。
- 历史复现：`.release/audits/2026-09-28/source-check/designer/join_opening.py`、`raw_checks.py`、`opening-target-join.json`、`root-opening-confirmed.json`。脚本硬编码输出历史临时目录，**本次未执行，工作者应提取最小fixture逻辑，不直接重跑覆盖旧证据**。

最小修正：仅对已验证普通平面矩形3D window profile，使用墙向单位向量`u`和`max(profile_xy·u)-min(profile_xy·u)`作为local width；保留sill、height、0.10m代理厚度和frame shift，修正后的footprint也要用于piece dedup。世界XY最大跨度是exporter声明的字段语义，错误在consumer把它当墙向长度；无需重写原始export。轴对齐、循环顶点序、反向顶点序、重复窗片控制应覆盖；不将66个不规则door候选或BayWindow自动纳入已证实修正范围。

示例：`il3d:b57d76c2-86d4-11f0-a478-60cf84ae2082::window_2`，house `71b4daf5-e0e4-47f1-a76f-fabb489d2df2`，`SecondBedroom-198098`，export行25374，byte167096966，width0.26428m，实际profile墙向跨度0.37167528031872177m。全部39个沿墙跨度与历史记录在1e-9内一致。

## SpatialGen 倾斜 bbox

原始根：`/Volumes/harddisk/3D_Room_Collections/BillLin66__3D_Room_Collections/SpatialGen_exported`。下列8份 `data/<scene>/room_layout.json` 均读取并验 SHA；`/bboxes/<index>` 的 class、size、transform 与 `spatialgen.jsonl` 保留的 `source_fields` 逐项一致。

| scene | bbox index | 历史/当前暴露 |
|---|---:|---|
| `scene_00008` | 22 | 历史未保留对象；重生 raw IR 仍需修正 |
| `scene_00009` | 9 | 历史未保留对象；重生 raw IR 仍需修正 |
| `scene_00015` | 10 | 历史未保留对象；重生 raw IR 仍需修正 |
| `scene_00019` | 41 | 历史未保留对象；重生 raw IR 仍需修正 |
| `scene_00024` | 0 | 当前 test target |
| `scene_00027` | 8 | 历史未保留对象；重生 raw IR 仍需修正 |
| `scene_00030` | 0 | 当前 test target |
| `scene_00032` | 2 | 历史未保留对象；重生 raw IR 仍需修正 |

原始 `size` 是 local XYZ 完整米制尺寸，16个 `transform` 数值为 row-major 4×4 affine，translation 索引3/7/11，3×3线性块包含 scale/reflection。用 `±size/2` 八角点经完整变换，投影至已有 yaw 轴取得包围尺寸与 world minZ。不要先用 export scaled size 再重复施加 affine scale。保留 front=-90° 转换与 tilt flags；若在 export frame 改尺寸/center，`convert_room(center_z=True)` 只减一次半高。此为 upright box 包络修正，不证明资产语义正面或实际 mesh 全部准确。

历史复现：`.release/audits/2026-09-28/source-check/generated/spatialgen-geometry.json`、`three-source-final-summary.json` 和 `REPORT.md`；原 exporter/verifier 在源根 `kits/convert_spatialgen.py`、`kits/verify_spatialgen.py::source_corners`。专用历史几何 audit 脚本未在已查归档定位，旧临时目录缺失；角点公式和逐病例数值输出足够构建新的小 fixture。原 verifier 的 yaw-only roundtrip 排除 tilted 对象，单纯通过它不等于验证本修复。

## Scan2CAD 倾斜 bbox

聚合输入（存在，196,265,906 bytes）：`/Volumes/harddisk/3D_Room_Collections/imChuling__3D_Room_Collections/SceneCAD_Scan2CAD_exported/scenecad_scan2cad_combined.json`。聚合器位于同根 `aggregate_scenecad_scan2cad.py`；原始注释在 `source/scan2cad`、`source/scenecad`、`source/scannet`。

8个历史确认对象、6个场景如下；逐项原始指针、CAD/scan TRS、axisAlignment、聚合矩阵与 source annotation 完全匹配结果均已写入 JSON。额外对这8例按原生 CAD 八角点、原始 TRS 与 metadata 独立计算包围尺寸，全部在1e-9内匹配历史数值；这只是有界病例复测，不是全原始数据重审。

| scene | alignment index | 历史 default selected |
|---|---:|---|
| `scene0043_00` | 2 | True |
| `scene0043_00` | 4 | True |
| `scene0401_00` | 28 | True |
| `scene0401_00` | 29 | True |
| `scene0402_00` | 7 | False |
| `scene0483_00` | 12 | True |
| `scene0494_00` | 5 | True |
| `scene0569_01` | 1 | True |

原始标注文件为同根 `source/scan2cad/scan2cad_annotations.zip`，成员 `full_annotations.json`，schema `id_scan/trs/aligned_models`；JSON 为每例提供 `/<scene-index>/aligned_models/<alignment-index>`。ScanNet axisAlignment 位于 `source/scannet/metadata/<scene>.txt`；SceneCAD shell 位于 `source/scenecad/scannet_planes.zip` 的逐场景 member（实际名称和 SHA 在 JSON）。6个源场景的 scan scale 都是 `[1,1,1]`，本次没有把这个局部验证外推为所有1506个标注都满足。源标注按流读取至最后所需场景（1407个标注条目），聚合读取至1193条即停止；未重扫冻结数据，未重算196MB聚合全文件 SHA。

聚合 `furniture_position` 是 CAD box center；`furniture_size` 是已含 CAD scale 的 Z-up 局部尺寸；`source_fields.rotation_matrix_zup_scan_local` 包含 ScanNet axis alignment。在 export yaw 的局部轴计算 `abs(Q(yaw).T @ R) @ size`，随后沿用 +90° semantic-front 转换（XY尺寸相应互换）。底部应由完整角点的 minZ 得到，再使用明确记录的 floor/snap 策略；不应把局部 height/2 当倾斜盒世界半高。保留原始矩阵的数值/反射，不自行单位化消除 alignment 信息；原始 scan TRS 非单位 scale 时应由完整变换处理或显式阻断，不能默默猜测。

历史复现：`.release/audits/2026-09-28/source-check/scanned/scan2cad_final_impact.py`、`scan2cad-final-impact.json`、`scan2cad-source-contract-review.json`。脚本会写同目录，且引用失效的历史临时文件，不直接运行覆盖旧证据；提取八角点公式与已存病例到新版本 fixture。

当前比较中仅1个已准入目标不意味着影响只限1个：7个错误 prior target 位于5个 prior-eligible train rooms，可能通过类别 prior/OOF 影响其他房间。修正后必须重建所有 train-only prior 和 P repair，不局部替换这一条训练样本。

## 仅修正版本的重建顺序

1. 固定独立 frozen code + reviewed adapter overlay，登记 SHA 与依赖；不导入当前工作区其他变更，不从正在变化的工作树直接启动。
2. 三个完整源重新生成到新 root/ir；其余15个从 `.release/v3.2/ir/<source>.jsonl` 原样复制并对照旧 manifest SHA。18是总来源数。对三个源也要求未变语义行保留原行 bytes（可由后处理按 UID 对齐后复用旧行）；真正改变的行输出详细对象/字段 delta，禁止仅按历史病例白名单修补。
3. 发布完整 raw IR 和 delta ledger 后才确定 release：既要修正历史未发布 raw rows，也不能把它们自动加入原比较人群。默认保留历史 released UID 与 split；stock frozen build 只能当新候选，逐 UID 比较后才能声称保留。
4. stock build 依次做 prep、cross-source dedup、组件 split、精确布局去重；同一组件按最小 group 名稳定 hash，但尺寸变更可拆合 content edges。若 membership/split 有变化，保留候选和完整差异，或用独立版本 wrapper 显式继承旧 UID/split 并输出新 prep/manifest。现有 stock CLI 未实现该 wrapper；不修改旧 manifest 的 IR hashes 来伪装原 release 已重建。
5. split gate：旧 dev/test 及其 component 不进 train；重新算全 raw group/alias 与 prepared content/layout closure。父 UID 若被新 prep 拒绝或组件跨 split，输出真实冲突，不静默丢弃。对应旧 count 为 train144150/dev8167/test8647，但 count 相同不证明 UID 分配相同。
6. 单独固定当前 comparison pipeline（它不在 v3.2 冻结包中），生成新 corpus → main-data → eval-data；重新计算资格、全部 train-only OOF priors、P layout repair、共同token准入和测试标签。正在运行的旧 main-data 构建保持原样，不能把途中产物拼进新版本。
7. 再做严格字段/UID/组件/OOF/长度/已知源病例 gate，才进入后续配对训练与正式评测。本任务不提交构建或训练。

准入和指标范围仍继承未修正部分的限制：semantic-front 是各 adapter 的契约，boundary proxy 仍是 proxy，缺 mesh 不能宣称全部源 corrected。IL3D fixed window 虽非生成目标，仍是输入和 fixed-geometry 依据；其修正可能改变相关碰撞/可行性统计，但既有 fixed_blocking 对 opening 有豁免，不能未经复算就宣称它必然改变那个 gate。

## 构建入口与参数边界

已运行的三源再生成不应重复执行。使用父任务实际冻结包 `corrected-code-001/` 与完成后验收的 `corrected-ir-001/`（补齐15个字节相同源）；远端应复制并核验相同包/IR manifest。运行时从冻结包目录导入 Python 模块，避免共享当前工作区优先覆盖。legacy candidate 输出必须是独立新目录，不指向 `.release/v3.2` 或任何现有产物。

后续 full legacy release 参数以旧 `MANIFEST.json.args` 全集为准：`constraint_frac=0.3`、min_objects1、max_vertices0、boundary_types polygon/hull、anchors floor/object、oob_tol0.10、hidden_max0.3、dev/test0.05、field_dropout0.2、desc_dropout0.5、rot90开启；reject_flagged/cap/source_anchors为空，keep_duplicates为false。`--sources` 的18项顺序见 JSON。通用 README 省略 constraint_frac，不能直接视作复现命令。新 manifest 必须绑定新 IR、覆盖包和 split policy，保留父 manifest 引用。

**发布阻断项不是原始输入不足，而是待完成的修复验证与版本闭环：**待验证最终 patch 和完整源差异；stock build split/membership 不变尚未实测；全量 legacy build 的 RAM 预算尚未验证；真实资产 mesh 未全查。原始结构输入足以先做新 raw IR，家具网格缺失不阻断这里的 box/window 修正。所有当前训练、评测依赖、冻结 snapshot、历史数据与证据保持保护。
