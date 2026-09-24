# FastFill v1：单次调用的对象条件布局模型

一句话：给定房型、真实房间几何，以及**全部**待摆对象（每件的局部实际尺寸，可选描述与空间约束），模型**一次生成**所有对象的数值位置、yaw 和支撑关系。

$$F_\theta(R, G, O, C_{\text{layout}}) \rightarrow L$$

整体框架沿用 OptiScene：对象由上游给定，模型只负责摆放，单轮 Qwen3-8B + LoRA SFT。
在此基础上修正了 OptiScene 已核实的几处缺陷：
- 房间只给面积 → 改为给地面多边形；
- h/w/d 轴序混乱 → 改为局部尺寸，并约定 +X 为正面；
- 同描述物体按列表位置对应 → 每件物体一个 id；
- 训练与推理用的 prompt 不一致 → 改为同一个 prompt。

范围和协议以用户冻结的 FastFill v1 定稿为准，接口见附录 B，实现在 `interface.py`。

## 1. 流水线与文件

```
HF 导出(22 源) ──download──▶ 本地汇总 JSON ──adapters/*──▶ IR(统一约定) ──build──▶ train/dev/test ──train──▶ LoRA
                                                                                       │                        │
                                                                          *_rooms.jsonl(评测用 IR)      merge_lora ─▶ merged
                                                                                       └──────── evaluate ◀───────┘
WorldEdge 请求(附录 B) ──interface.request_to_room──▶ 模型一次生成 ──interface.placements_from_text──▶ placements(yaw_rad)
```

| 文件 | 作用 |
|---|---|
| `download.py` / `download_full.sh` | 下载导出数据。前者默认只下汇总文件（33 GB），后者用 `hf download` 逐个数据集下全量 |
| `adapters/*.py` | 每个数据源一份，把各自的约定转成统一 IR。每份都附有证据：front test、kits 源码、20 间房逐字段核对 |
| `scene.py` | IR 约定、SYSTEM_PROMPT、序列化、canonical 排序和 id、`parse`（永不抛异常） |
| `anchors.py` | 剔除结构件，推断落地、放在他物上、挂墙、吊顶四类锚点，保证支撑关系闭合 |
| `build.py` | 过滤和拒收规则、取整对齐、跨源去重、防泄漏划分、增强、约束抽取 |
| `split.py` | 用 house/scan、别名、家具指纹、布局指纹做并查集，按连通分量整体划分 |
| `validate.py` | 越界、支撑、分桶碰撞、OOR 检查，以及约束检查 `holds` |
| `train.py` | LoRA SFT：只监督答案，批量分词，按长度分组，支持 `--resume` 和 `--dry_run` |
| `evaluate.py` | 生成并打分，所有请求都计入统计，分布内和 held-out 分开报告，每项都附 GT 对照 |
| `interface.py` | 附录 B 请求与模型文本之间的双向转换（输出 `position_m` / `yaw_rad` / `support_parent`） |
| `serve.py` | OpenAI 兼容服务端点：给 EmbodiedGen 的 `FastFillBackend` 调用，内部走 `interface.py` |
| `../scripts/merge_lora.py` | 合并 LoRA，兼容 transformers 4.x 和 5.x 的产物 |

## 2. 统一标准（所有数据源进训练前都转成这一套）

**坐标与位姿**
- 单位是米，右手系，Z 轴向上。
- 使用房间局部坐标，边界 AABB 的左下角为 (0,0)，地面 z=0。
- 位置取 bbox **底面中心**。
- 尺寸是对象**局部坐标**下的 extent `[sx,sy,sz]`，不是世界坐标下的 AABB。
- 局部 +X 是**语义正面**：只有 front test 验证过的源才会进训练，房间里只要有一个物体的正面不可靠，整间房拒收。
- yaw 是绕 +Z 从房间 +X 转到对象正面的角度。
  - 接口层和 IR 用弧度（`yaw_rad`）。
  - 模型文本里写整数度，因为 `90` 是精确值且只占 2 个 token；`1.570796` 不精确，还要多 3–6 个 token。
  - 取整误差 ≤0.5°，对应 1 m 处约 0.9 cm。

**房间与对象范围**
- 边界：`polygon` 表示墙体，`hull` 表示扫描地面的凸包。只由物体范围拼出的代理边界不进训练。
- 对象：落地物体，加上放在其他物体上的物体（带 `on`）。挂墙和吊顶的物体 v1 不包括。
- 门、窗、帘等结构件不算待摆对象。数据源自己标记的结构件（HSSD、MultiScan 的 architectural）在 IR 里保留、带 `structure` 标记，只是不摆。
- 地面上有楼梯或柱子的房间直接拒收（模型看不到它们，参考布局里会留一个说不清的空洞）。
- 被删掉的落地物体（结构件、倾斜物体、通用类别等）占地合计 ≥0.3 m² 时，整间房拒收，避免参考布局里出现"隐形障碍"。

**缺失字段与取整**
- 源数据里缺失的字段一律省略，不伪造。房型、层高缺失时直接不写这个键，训练时也会随机丢掉这两个字段。
- 物体高过房间层高时，丢弃层高字段。
- 类别只做可追溯的归一化，原始标签保留在 IR 里；房型归一化时会去掉实例编号。
- 所有校验、约束和评测都用**模型实际看到的取整值**：位置和尺寸取到厘米，yaw 取整数度。所以一个完全照抄参考答案的模型，合法率和约束满足率都是 100%。

## 3. 数据（2026-09-24 按审核修复后重建：`/Volumes/harddisk/fastfill_data/v1` 和 `v1.1`，各 337 MB）

train 96,540 / dev 5,896 / test 6,540（另有 dev_constrained 5,853、test_constrained 6,493）。
- v1.0 = 无约束基线 (R,G,O)→L；v1.1 = 同一批房间，其中 28,755 条训练样本带 1–4 条约束 (R,G,O,C)→L。两版的 dev/test 文件逐字节相同，可以直接配对比较。
- 训练集 token：v1.0 共 8,236 万，v1.1 共 8,340 万；单条 p50 约 720，最长 2,753，没有超过 4096 的。
- 每个目录都有 `MANIFEST.json`（每个输出文件和每个代码文件的 sha256、构建参数）和 `QA.json`（泄漏、闭合、约束校验、原始/修复后合法率、各源拒收原因）。
- 修复前的旧数据移到了 `_superseded_20260924/`，不要再用。

**数据集取舍：Excel 的 22 个 → 下载保留 16 个（11 个产生训练房间、2 个只做评测、3 个是辅助数据）→ 删除 6 个**

每个保留的数据集都有实际用途（下面"用在哪"一栏可在 `adapters/*.py` 里对应核实）。数字为 2026-09-24 第三轮重建（train / dev / test 房间数）。

*A. 产生训练房间（11 个下载 → 16 个训练源）*

| Excel 数据集 | 训练源 | train / dev / test | 为什么用 |
|---|---|---|---|
| SpatialLM | SpatialLM | 31,183 / 1,658 / 1,658 | 设计师户型，官方墙线多边形，房型 100% 有值 |
| IL3D | IL3D_3dfront、IL3D_synthetic | 16,109 / 734 / 885；5,992 / 348 / 316 | 3D-FRONT 设计房间的规范版本（类别、尺寸完整）；程序合成房间 |
| OptiScene | OptiScene_holodeck | 8,890 / 549 / 533 | 参考项目的数据，只用 Holodeck 生成部分（其中 3D-FRONT 部分与 IL3D 重复） |
| InternScenes | InternScenes_gen；arkit / scannet / mp3d / 3rscan | 8,000（上限）/ 668 / 601；3,238 / 768 / 669 / 660（train） | Infinigen 生成房间；四个真实扫描子集，OBB 带语义正面 |
| MansionWorld | MansionWorld | 8,000（上限）/ 867 / 1,104 | 程序生成楼宇，多边形边界，房型 100% |
| SAGE-10k | SAGE-10k | 7,648 / 450 / 418 | Agent 生成的单房间，只用落地层 |
| Structured3D | Structured3D | 4,260 / 260 / 291 | 官方房间多边形和层高 |
| InteriorGS | InteriorGS | 932 / 72 / 46 | 官方房间 profile 多边形 |
| HSSD-200 | HSSD200 | 428 / 4 / 55 | 设计场景的官方区域多边形；它的 `asset_catalog.csv` 还是 IL3D_synthetic 计算物体网格边界必需的 |
| SceneCAD & Scan2CAD | Scan2CAD | 152 / 10 / 9 | ScanNet 扫描 + CAD 对齐得到的朝向 |
| MultiScan | MultiScan | 52 / 2 / 8 | 真实扫描，标注了正面向量；多数房间因地面柱子等被拒 |

*B. 只做评测（held-out，从不进训练）*

| Excel 数据集 | test | 为什么用 |
|---|---|---|
| SceneSmith | 320 | 训练里没有的生成器，测泛化 |
| SpatialGen | 23 | 同上 |

*C. 辅助数据：不产生房间，但删掉会让别的数据变错*

| Excel 数据集 | 被谁用 | 删掉的后果 |
|---|---|---|
| 3D-FRONT | IL3D_3dfront：把 IL3D 按房型合并的房间拆回真实源房间，并取真实地面边界和层高 | 约 1,300 个 IL3D 房间会混着 2–10 个房间的物体，边界也不对 |
| 3RScan | InternScenes_3rscan：reference / rescan 对应表 | 同一空间的多次重扫描可能分到 train 和 test 两边（泄漏） |
| ARKitScenes | InternScenes_arkit：video → visit 对应表 | 同一次 visit 的多段扫描可能分到 train 和 test 两边（泄漏） |

这三个本身不直接进训练：3D-FRONT 原始导出类别和尺寸大面积缺失，已由 IL3D 版本覆盖；3RScan、ARKitScenes 原始导出没有房间边界，房间来自 InternScenes 版本。

*D. 删除（6 个）*

| Excel 数据集 | 删除原因 |
|---|---|
| HM3DSem | 没有 yaw，尺寸是世界坐标 AABB；region 不是房间 |
| MetaScenes | 没有 yaw（世界坐标 AABB）；场景是 ScanNet，已由 InternScenes_scannet 覆盖 |
| ScanNet | 没有 yaw；1,513 个扫描已由 InternScenes_scannet（带正面）覆盖 |
| ScanNet++ | OBB 轴与语义正面无关（front test 各方向均匀，只有 7–9%）；221 个 apartment 未拆成房间；边界是矩形代理 |
| ASE | 约 60% 的实例没有框 |
| M3DLayout | 3D-FRONT、Matterport 部分是重复；Infinigen 部分的边界就是家具范围（3000/3000），不是真实房间 |

- 删除依据来自各数据集的 HF README 和本地实测（front test、边界检查）。ASE、ScanNet++、M3DLayout_infinigen 转好的 IR 仍保留在 `/Volumes/harddisk/fastfill_ir/`，以后可以直接加进 build；`fastfill.adapters` 遇到已删除的源会跳过并提示。
- 保留的 16 个数据集已逐文件对照 HuggingFace 核对完整（含 MultiScan 的网格和贴图，共 83 GB）。重新下载用 `fastfill/download_full.sh <目录> <名字...>`（需要 zsh；名字写错或 30 次重试都失败会报错退出）。

**防泄漏**
- 以下四种关系任意一种相连的房间，整体划入同一个 split：同一 house 或 scan、同一次 visit、相同家具指纹、相同布局指纹（旋转 90° 视为同一布局，换资产也算）。
- 在写出的文件上复核过：train 与 dev/test 之间 group、布局指纹、家具指纹的重合数都是 0；held-out 源在 train 中为 0 行。

## 4. 在服务器上训练

**环境**：直接用 node03 上已验证的环境（见 `../fastfill/requirements.txt`：torch 2.10、vLLM 0.19.0、transformers 4.57.6），再补装本目录的依赖。
- transformers 4.57.6、5.14.1、5.17.0 都做过冒烟测试，流程是 train → merge → evaluate。
- **训练、合并、推理必须在同一个环境里完成。**

```bash
# 代码：https://github.com/Slian22/fastfill （私有仓库；下面所有命令都在仓库根目录运行）
git clone https://github.com/Slian22/fastfill.git && cd fastfill
pip install -r fastfill/requirements.txt          # 主要补 shapely / ijson；已有的 transformers 4.57.6 不会被升级
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B

# 数据：直接拷已构建好的 v1 / v1.1（各 337 MB），或者按第 6 节在服务器上重建
rsync -a /Volumes/harddisk/fastfill_data/v1/ <server>:fastfill/data/fastfill_v1/
rsync -a /Volumes/harddisk/fastfill_data/v1.1/ <server>:fastfill/data/fastfill_v1.1/     # 约束版，训练命令里 data 和 out 换成 v1.1
python -m fastfill.train --model models/Qwen3-8B --data data/fastfill_v1 --out /tmp/x --dry_run   # 核对 token 统计
```

**训练**：有效 batch = bs × grad_accum × GPU 数，保持在 32 左右。换 GPU 数量时要同步调整 grad_accum。

```bash
# 8 卡
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node 8 -m fastfill.train \
    --model models/Qwen3-8B --data data/fastfill_v1 --out outputs/ff-v1 \
    --bs 4 --grad_accum 1 --lr 1e-4 --epochs 2 --save_steps 500
# 只用 1–7 号卡：前面加 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7，并改 --nproc_per_node 7（有效 batch 28 也可以）
# 单卡：CUDA_VISIBLE_DEVICES=0 python -m fastfill.train ... --grad_accum 8 --save_steps 1000
#       （不指定卡时，单进程会把所有卡用成 DataParallel，又慢又占 0 号卡显存）
# 中断后续训：同一条命令末尾加 --resume。在第 2 个 epoch 中途断开时，该 epoch 剩余部分的数据顺序会变
# （HF 的已知行为，最多约 1/4 样本在这个 epoch 多训或少训一次），可以接受
```

**资源估算**（解析估算，未在 GPU 上实测，误差约 ±30%）
- 每卡显存 45–55 GB（bs=4）；bs 不要超过 4。
- 2 个 epoch 的用时：8×H100 约 1–1.5 h；8×A100 约 2.5–3 h；1×H100 约 6–9 h。
- 每个进程都会先在 CPU 上加载一份 16 GB 的权重，8 卡时主机内存峰值约 170 GB。

**合并与评测**

```bash
python scripts/merge_lora.py --base_model_path models/Qwen3-8B --lora_path outputs/ff-v1/final --output_path outputs/ff-v1-merged
python -m fastfill.evaluate --gt-only --rooms data/fastfill_v1/test_rooms.jsonl --out eval/gt            # GT 参考线
python -m fastfill.evaluate --model outputs/ff-v1-merged --rooms data/fastfill_v1/test_rooms.jsonl --out eval/ff-v1
python -m fastfill.evaluate --model outputs/ff-v1-merged --rooms data/fastfill_v1/test_constrained_rooms.jsonl --out eval/ff-v1-cons
```

**看评测结果时注意**
- 优先装 vLLM；走 HF generate 时设 `CUDA_VISIBLE_DEVICES=0`，否则 8B 模型会被切到多张卡上串行跑。
- `metrics.json` 的 `in_dist` 和 `held_out` 分开看，碰撞指标一律和同一批房间的 `*_gt` 对比，不要和 1.0 比。
- 合法（valid）= 不出界、支撑正确、不顶破天花板（物体顶 ≤ 层高 + 5 cm；层高未知时不查）。
- 越界率、碰撞率、OOR 只在完整解析的答案上算；一个完整答案都没有时显示 `null`（未知），不是 0。
- **模型指标只看原始输出**（`valid_rate` 等）。`*_repaired` 是接口把物体推回墙内之后的结果，属于确定性后处理，不算模型能力；`inside_1mm_rate` 是按后端 1 mm 标准的贴墙合格率，原始和修复后分开报。
- **v1.0 训练时没有见过约束输入**，所以 test_constrained 上的约束满足率是"无约束监督基线（零样本）"，不能当作约束能力的结论。

**起服务接 WorldEdge（EmbodiedGen `FastFillBackend`）**

```bash
python -m fastfill.serve --model outputs/ff-v1-merged --port 8001          # vLLM，贪心解码，一次处理一个房间
# EmbodiedGen 那边：WORLDEDGE_FASTFILL_URL=http://<node03>:8001/v1  WORLDEDGE_FASTFILL_MODEL=fastfill
```

- 后端把附录 B 请求的 JSON 原样作为 user 消息发过来；`serve.py` 用 `interface.py` 转成模型文本，模型输出后再转回 `placements`（`position_m` / `yaw_rad` / `support_parent`），放在回复的 content 里。
- 以下情况返回 HTTP 422、不返回布局，由调用方（Harness）决定怎么办：请求不合法；模型输出解析不了（包括漏摆或多摆物体）；修复后仍验收不过（出界、悬空、顶破天花板、超出墙面 1 mm）。FastFill 要么把全部物体合法摆好，要么明确失败，**绝不自己丢物体**。放不下的清单也走这条路。
- 训练数据里物体常常穿墙几厘米（test 参考答案 39.8% 的房间 > 1 mm，OptiScene_holodeck 系统性 4.5 cm），而后端只容忍 1 mm。所以落地物体穿墙或穿进 keepout 不超过 10 cm 时，`interface.py` 会把它推回墙内，放在它上面的东西一起移动。这是**确定性修复，不算模型成功**：回复里同时给 `raw_placements`（模型原始提案）、`placements`（修复后）和 `validation.raw` / `validation.repaired`（各自的验收结果，含每条约束是否满足），日志和实验三者分开记。
- 约束字段（这就是后端文档里说的"FastFill v1 文档"）。**每个谓词只有一份实现：`validate.holds`**。数据里抽约束、模型 prompt 的定义文字、评测、服务端验收都调用它，所以训练、条件、验收三处语义一致；别的系统要判断 FastFill 约束是否满足，读回复里的 `validation.*.constraints`，不要自己重新实现一遍。

| type | 字段 | 定义（`validate.holds`） |
|---|---|---|
| `faces` | `subject`, `target` | subject 的正面方向与"subject 中心 → target 中心"的夹角 ≤ 30° |
| `between` | `subject`, `anchors: [a, b]` | a、b 中心连线与 subject 的旋转 footprint 相交 |
| `supported_by` | `subject`, `parent`, `surface: "top"` | subject 的支撑父物体就是 parent（放在它顶面） |
| `against_wall` | `subject` | subject 旋转 footprint 的某一整条边，每一点离边界都 ≤ 0.1 m |
| `near` | `subject`, `target`, `max_gap_m` | 两个**旋转后 footprint 之间的最短距离** ≤ max_gap_m（间隙，不是中心距离）；训练见过 0.1 / 0.2 / 0.3 / 0.5 |
| `keepout` | `polygon_xy` | 禁放区（门口通道、预留门位）：外扩 0.1 m 后从地面多边形里挖掉，模型当墙看待 |

- 别的系统里叫 near 但定义不同的关系（例如"中心距 ≤ 1.5 m"），**不能**直接改个字段名当作 FastFill 的 `near` 发过来：两者是不同的谓词，需要调用方自己决定怎么对应。

- 物体上带 `"anchor": "wall"` 或 `"ceiling"` 的（镜子、挂画、吊灯），v1 不摆，放进 `unsupported` 返回；引用它们的约束也进 `unsupported`，其余物体照常摆。
- `hard` 字段不读，所有约束都按必须满足发给模型。
- keepout 把房间切成两块或在中间留出孤岛时，请求直接报错（`keepout_not_representable`）。

## 5. 模型输入输出（真实样本，HSSD 走廊，7 件，其中 4 件放在桌上）

```json
user:      {"room_type":"hallway","boundary_type":"polygon","boundary":[[2.35,0],[4.35,0.15],[4.82,0.15],[4.82,4.16],[0,4.16],[0,2.74],[2.35,2.74]],"height":2.8,
            "objects":[{"id":"doormat_1","size":[0.6,1.85,0.01]},{"id":"table_1","size":[0.36,2.94,0.76]},{"id":"seat_1","size":[0.56,0.5,1.02]},
                       {"id":"plant_1","size":[0.33,0.48,0.41]},{"id":"lamp_1","size":[0.27,0.27,0.35]}, ...]}
assistant: {"placements":[{"id":"doormat_1","pos":[3.89,3.86,0],"yaw":270},{"id":"table_1","pos":[1.47,3.98,0],"yaw":270},
            {"id":"plant_1","on":"table_1","pos":[0.96,3.98,0.76],"yaw":270}, ...]}
```

- 对象按占地面积从大到小排列，id 是 `类别_序号`，答案顺序与输入一致。
- 可选的 `desc` 取资产描述的第一句，已去掉摆放从句，训练时整间房以 50% 的概率丢弃。
- 可选的 `constraints` 有五种：`["on",a,b]`、`["faces",a,b]`、`["near",a,b,d]`、`["against_wall",a]`、`["between",a,b,c]`。

## 6. 重建数据 / 下一步

```bash
python -m fastfill.download --out $DATA                     # 汇总文件 33 GB（全量：fastfill/download_full.sh $DATA）
python -m fastfill.adapters --root $DATA --out $IR          # 21 个 adapter → IR（约 20 万个房间）
python -m fastfill.build --ir $IR --out data/fastfill_v1 \
  --sources SpatialLM IL3D_3dfront InteriorGS HSSD200 Structured3D InternScenes_arkit InternScenes_mp3d InternScenes_scannet \
            InternScenes_3rscan Scan2CAD MultiScan OptiScene_holodeck IL3D_synthetic SAGE-10k MansionWorld InternScenes_gen SpatialGen SceneSmith \
  --cap MansionWorld=8000 InternScenes_gen=8000 --source_anchors SAGE-10k=floor
# v1.1（约束监督）：同一条命令加 --constraint_frac 0.3，输出到另一个目录；test 文件不变，可以与 v1.0 配对比较
```

**已知限制**
- **支撑关系**：绝大多数是从 bbox 顶面接触推断出来的，证据偏弱。
- **边界**：IL3D_3dfront 用的是地板凸包；可以用原始 3D-FRONT 房屋 JSON 升级为精确多边形。
- **房型标签**：3D-FRONT 自身有房型标错的噪声。
- **朝向**：扫描源约 15% 的床正面转了 90°（适配器层面的问题，尚未修）；MansionWorld 和 Holodeck 的马桶宽大于深。
- **碰撞检查**：只在 bbox 级别，结果须对照 GT 参考线。
- **尚未支持**：门窗输入；对称物体 yaw 等价类的训练。
- **许可**：大部分数据源仅限非商业使用。
