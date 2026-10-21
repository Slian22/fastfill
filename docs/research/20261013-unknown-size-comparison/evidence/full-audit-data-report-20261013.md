# 当前 unknown-size J/P 全记录数据审计 — 2026-10-13

**处置结论：保留当前配对训练为冻结旧语料上的诊断实验；不因 loader 后续补丁或 8192-token 上限立即取消。已确认当前数据继承已知适配器错误，不能称为“全部来源几何已修正”，也不能作为纠正数据后的最终准确率结论。若论文主张纠正后的源几何/物理准确性，应修复源转换、建立新版本并重建 corpus、组件/先验/共同准入和 J/P 数据后配对重跑。不得静默删除已知问题测试行或事后改变正式分母。**

本报告负责 DATA 与训练输入；不负责文献、贡献新颖性、指标实现、模型效果评估或自动重训。本数据审计不额外调用API；用户已授权正式评测API。训练与依赖评估、冻结快照、其他代理工作均未改动。采用 ARIS 的直接证据与审稿范围原则，并非外部跨模型 ARIS 验收。

输入全扫作业 **7367962** 完成（493.43 秒，峰值 RSS 1,944,576 KiB）；来源暴露关联作业 **7367978** 完成（67.37 秒，峰值 RSS 1,096,832 KiB）。两者是独立 CPU Slurm 作业，分别限时45/30分钟、32GB、无GPU。当前训练是否完成/准确率多少不由本扫描决定；父任务最新状态仍为 RUNNING，尚无新的模型 test metrics。

## P0–P3 与立即处置

- **P0：未发现当前真实请求中的直接尺寸/姿态泄漏、类别类型违规、J/P UID 不一致或长度准入分歧。** 这是全记录数据结论，不是准确率或原始网格认证。组件与OOF独立重算已通过；历史未拷贝辅助文件与已完成的payload补充范围见下节。
- **P1：已知几何错误仍在当前监督/上下文中。** 当前获准训练有 Scan2CAD 1 个错误尺寸目标、1 房间；IL3D 有20个过短窗框固定输入、10房间。测试有 SpatialGen 2 个错误目标、2房间，以及 IL3D 2 个错误窗框、1房间。这11个训练房间/3个测试房间是直接暴露下界，绝非模型影响范围上界。Scan2CAD **7个错误尺寸目标、5个房间进入124,843房间的先验池**，可能影响其他房间的 P 尺寸预测；尚未重算纠正后的中位数或测量影响。
- **P1：跨来源指标必须保留几何代理与语义真值的区别。** 盒体碰撞、边界包含、盒顶支持和参考 yaw 一致性可以按冻结定义报告；不能直接改称 mesh 无碰撞、真实墙面接触、物理稳定性或逐资产 semantic-front 准确率。全部 prepared/当前 bound 对象未发现 v2 的 mask 字段；另一份 v2 数据的修正与 partial masks 不能移植为本数据的认证。
- **P2：上游标记与边界/高度处理限制总体。** release train144,150中19,307因历史 eligibility 不进入先验池；另25,920个 eligible 房间未通过共同 decoder qualification。历史 flags 本身来自参考几何，不构成 prompt 标签泄漏，但把“全部原始来源泛化”的主张限制为本发布与准入总体。27,706个原始 IR 房间未发布，必须保留在原始总体与 alias 闭包治理中。
- **P2：semantic-front 是未验证的来源语义，不能把 k0 暴露数当错误数。** 当前训练1113个 movable targets、30个 fixed、806房间；dev569/8对象、240房间；test473/9对象、209房间。k0 几何 fallback 不等于已证实盒体转换错误，不据此全量旋转或删除。
- **P3：长度尾部有共同、可复现的排除。** train169/98,923 = 0.170840%，dev16/8,167；两臂均整条丢弃，未截断答案。训练8k、推理40,960的差异本身不是必须重训的错误；长场景准确率仍需完整正式评测。两臂监督 token 数不同，不能以 training loss 大小判断哪臂更准确。

## 已完成的全量输入检查

六份 J/P 文件共 **231,474条消息记录（115,737对）**。每臂 train98,923、dev8,167、test8,647；严格检查了每臂1,319,984个 movable 输入对象。J对象精确字段`id/category`；P精确字段`id/category/size`；category必须非空字符串。所有记录检查角色顺序、固定上下文字段、opaque ID、冻结 inventory 投影、成对 UID/顺序/元数据、目标字段与完整性、父子身份及数值有效性，违规为0。

全量 train/dev 使用本地离线 Qwen tokenizer 重新编码，与两个完整 length ledgers 对齐；train共同接纳98,754，dev8,151。每条 train 的系统/用户请求、assistant target 和 P 显示尺寸 SHA 都与共同 qualification ledger 对齐。P尺寸逐条重算自已保存的适用 OOF/full prior，未出现 GT fallback 或不一致；prior数值本身的独立重拟合见下节。test全部8,647条请求均未达到40,960输入token，但这不证明生成后的上下文预算或准确率。

训练获准记录的总token：J111,845,719 / P112,329,829；监督answer+EOS token：J53,157,852 / P34,454,444。assistant-only mask和padding -100的冻结实现与先前 live QA保持一致；本次没有重新截取每个运行中的GPU batch或重载模型权重。

证据：[完整输入结果](full-audit-data-inputs-7367962-result.json)、[作业日志](full-audit-data-inputs-7367962.out)、[扫描脚本](full-audit-data-inputs.py)、[先前 live QA](agency-live-training-qa-20261013.md)。模型是共享Qwen3-8B，锁定revision `b968826d9c46dd6066d109eabc6255188de91218`；权重/LoRA fresh-load gates属于既有运行审计，本次不声称重验了权重二进制。

## 组件、OOF 与文件身份

**核心重算完成，所有已执行核心数据一致性断言通过，后续实际payload清单补充也已通过。** 作业7368003重建全部188,670原始IR的group/alias边及160,964发布prepared的content/layout边，得到**56,352组件**，跨split违规0；逐项核对160,964条identity的15个身份/哈希/flags字段。27,706未发布原始记录的alias边也参与闭包；未发布记录的content/layout不属于发布语料fingerprint域。

独立恢复prior eligible124,843房间、49,163组件、1,580,249个movable尺寸观测，排除344,030个fixed尺寸。完整五折赋值、各折UID/组件补集和哈希全部一致。**六套先验共776,848个bucket、2,330,544个轴中位数值逐项相等，最大绝对/相对差均0**（预设容差1e-12）；仅类别/room_type规范化函数复用冻结代码，拟合与组件算法独立实现。

| prior | fit房间 | 排除本折房间 | fit对象 | fit组件 |
|---|---:|---:|---:|---:|
| full | 124,843 | 0 | 1,580,249 | 49,163 |
| 0 | 100,139 | 24,704 | 1,274,032 | 39,330 |
| 1 | 100,241 | 24,602 | 1,248,226 | 39,330 |
| 2 | 100,384 | 24,459 | 1,267,636 | 39,330 |
| 3 | 98,061 | 26,782 | 1,250,911 | 39,331 |
| 4 | 100,547 | 24,296 | 1,280,191 | 39,331 |

full prior fitted-UID SHA：`65e4abfeba4eb8219d9bda5aac8625eef78bdfe3a4e67a23b28f07bd0bdae5e2`；其余五折拟合/排除UID哈希及组件哈希见[核心重算结果](full-audit-data-lineage-7368003-result.json)。历史train完整144,150条ledger复核通过：J qualified103,316，P qualified118,058，共同qualified98,923；prior在共同准入之前冻结，不在准入后反复拟合。

**原始作业状态为ERROR，不能改称整作业PASS。** 核心计算与10,513,216次数据检查已完成且零数据失败；最后枚举历史release manifest辅助文件时，`dev_constrained_rooms.jsonl`未出现在当前刻意分阶段拷贝的输入子集中而触发FileNotFoundError。该文件不属于当前J/P消费者所用train/dev/test。补充CPU作业**7368012**（2 CPU、8GB、10分钟上限）只完成payload哈希/字段清单与缺失历史辅助文件枚举，不重复组件或先验拟合。补充作业**7368012已PASS**（20.76秒，零违规），完整解析/核对当前实际payload并再次核对核心扫描文件，共74份文件记录。缺失且未认证的历史辅助文件为`dev_constrained_rooms.jsonl`、`dev_rooms.jsonl`、`stats.json`、`test_constrained_rooms.jsonl`、`test_rooms.jsonl`；它们位于`inputs/release/`，不是当前`main-data-v1/dev_rooms.jsonl`等同名有效文件。详见[补充结果](full-audit-data-payloads-7368012-result.json)。原ERROR结果及traceback原样保留，未伪称所有历史release文件都在服务器。

## 全部18来源与各阶段总体

| 来源 | 原始 IR | 未发布 | release train | dev | test | prior eligible | decoder qualified | SFT admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HSSD200 | 2,351 | 251 | 1,876 | 86 | 138 | 1,356 | 956 | 956 |
| IL3D_3dfront | 21,832 | 1,967 | 18,054 | 939 | 872 | 14,520 | 9,237 | 9,237 |
| IL3D_synthetic | 6,954 | 12 | 6,277 | 349 | 316 | 6,256 | 6,090 | 6,090 |
| InteriorGS | 6,607 | 1,151 | 4,877 | 268 | 311 | 3,315 | 2,458 | 2,456 |
| InternScenes_3rscan | 1,334 | 3 | 1,200 | 61 | 70 | 963 | 297 | 297 |
| InternScenes_arkit | 5,040 | 3 | 4,528 | 280 | 229 | 3,952 | 2,098 | 2,098 |
| InternScenes_gen | 14,089 | 49 | 12,552 | 828 | 660 | 12,319 | 11,870 | 11,868 |
| InternScenes_mp3d | 1,601 | 236 | 1,258 | 89 | 18 | 976 | 562 | 562 |
| InternScenes_scannet | 1,513 | 20 | 1,309 | 80 | 104 | 929 | 268 | 268 |
| MansionWorld | 27,884 | 6,022 | 19,396 | 1,153 | 1,313 | 19,359 | 18,343 | 18,245 |
| MultiScan | 273 | 151 | 114 | 2 | 6 | 58 | 15 | 15 |
| OptiScene_holodeck | 9,977 | 31 | 8,946 | 497 | 503 | 8,946 | 8,946 | 8,946 |
| SAGE-10k | 10,027 | 58 | 8,977 | 503 | 489 | 8,947 | 5,736 | 5,670 |
| Scan2CAD | 1,506 | 1,487 | 18 | 0 | 1 | 11 | 5 | 5 |
| SceneSmith | 364 | 0 | 0 | 0 | 364 | 0 | 0 | 0 |
| SpatialGen | 48 | 3 | 0 | 0 | 45 | 0 | 0 | 0 |
| SpatialLM | 54,376 | 7,608 | 42,068 | 2,239 | 2,461 | 33,081 | 24,060 | 24,060 |
| Structured3D | 22,894 | 8,654 | 12,700 | 793 | 747 | 9,855 | 7,982 | 7,981 |

原始IR **188,670**；未发布 **27,706**；发布/prepared **160,964 = 144,150 train + 8,167 dev + 8,647 test**。prior pool124,843；qualified decoder train98,923；共同长度准入98,754。SceneSmith与SpatialGen为source-heldout test，训练与prior均为0。数据名`full-corpus-v2`是当前comparison产物版本，不等于独立的`fastfill.v2`数据修正链。

## 已知病例关联：证据强度及具体身份

这是**历史独立测量病例与当前原始ID/实际bound映射的全记录关联**，并对可用历史IR尺寸值作一致性核对；不是本次独立重新测量原始mesh。Scan2CAD8/8和SpatialGen8/8历史IR尺寸、IL3D39/39历史IR窗宽与当前IR在1e-9容差内一致。当前prepared/bound数值与对象角色随证据保留。

| 历史发现 | 当前prior eligible | 当前SFT admitted | 当前test | 解读 |
|---|---:|---:|---:|---|
| Scan2CAD错误upright尺寸 | 7 targets / 5房间 | 1 target / 1房间 | 0 | 先验传播影响未测，不能仅算1行 |
| IL3D过短斜墙窗框 | 25 fixed / 15房间 | 20 fixed / 10房间 | 2 fixed / 1房间 | 输入错误，不是window placement标签 |
| SpatialGen丢tilt保留local extents | 0 | 0 | 2 targets / 2房间 | test-only |
| SceneSmith bottom-origin错误 | 0直接对象 | 0 | 0直接对象 | raw46对象/31房间存在；间接筛选影响未排除 |
| InteriorGS已知薄盒shortcut | 0直接对象 | 0 | 0直接对象 | raw3对象/1房间；未保留直接监督 |

当前Scan2CAD目标：`Scan2CAD::scene0494_00` / `scene0494_00::scan2cad_5` → `obj_0003`，当前size `[0.45,0.41,0.89]`。SpatialGen目标：`spatialgen_scene_00024::room_0` / `obj_0` → `obj_0008`；`spatialgen_scene_00030::room_0` / `obj_0` → `obj_0006`。原始完整ID、尺寸、fixed/movable角色及每一阶段命中见[病例及暴露摘要](full-audit-data-geometry-summary.json)与[原始完整扫描](full-audit-data-geometry-7367978-result.json)。

有限历史病例列表不是全部未知缺陷的穷举。另有30条IL3D源定位信息，其中6个额外对象解析到5个未发布IR房间；其余24条尚未解析到本冻结IR。k0原始索引20,747项中20,609对象匹配当前IR；138项所属Matterport房间UID不在当前IR，不能冒称已逐资产确认。未发布记录也已完整扫描，未用“未发布”推定原始几何正确。

**窗框与qualification的精确关系：** 冻结`validate.py`把`window/windows`从`fixed_blocking`排除；共同准入和P修复评分使用`fixed_blocking`，不是全部`fixed_collisions`。因此仅改变这批window宽度，不能据当前门槛代码认定会直接改变阻挡qualification；错误输入仍可影响模型条件与`fixed_collisions`诊断，源重建/其他间接影响未量化。证据：[冻结代码SHA及读取范围](full-audit-data-qualification-code.json)。

**58,796条P teacher差异的纠正说明：** geometry作业中的`P_target_differs_from_J_minus_size`是审计器错误断言，不是数据错误。训练P按固定估计尺寸执行独立pose repair，本就允许与J位置/yaw不同；主输入扫描已验证其assistant SHA与qualification记录一致。原始日志与结果保持不改，[按split核对P target的完整结果](full-audit-data-p-target-7368003-result.json)已确认：train98,923中58,796条不同、40,127条相同；dev8,167条全部相同；所有记录ID/support一致，文件SHA一致，违规0。这是预期训练teacher行为，不能称为58,796条坏数据。

## 跨来源指标、墙体、资产与mask边界

实际获准训练98,754房间：polygon95,497、hull3,257，缺少有效正高度30,567；test8,647房间：polygon8,219、hull428，缺高度1,954。`polygon`只说明IR声明，不能认证真实建筑边界；源中的proxy边界657个虽未进入发布集，仍不能把其余polygon等同真实墙体。

| 来源 | test范围及数据约束 | 允许的保守表述 |
|---|---|---|
| HSSD200 | 138；6缺高度；asset front/pivot未逐个验证 | 来源盒体/标注边界的一致性 |
| IL3D_3dfront | 872；871 floor polygon、1 export hull；31缺高度；已知窗宽错误 | 分来源box/边界指标，显式披露开口错误 |
| IL3D_synthetic | 316全部缺高度；asset-origin与bounds证据有限 | 标注平面布局，不宣称真实ceiling/support |
| InteriorGS | 311；1缺高度；yaw来自bbox角点/几何轴 | 几何轴一致性，不是家具semantic front |
| InternScenes_3rscan | 70 hull、全部缺高度、非true room | hull containment、box指标 |
| InternScenes_arkit | 229 hull、全部缺高度、非true room | 同上；k0单列暴露 |
| InternScenes_gen | 660 polygon、全部缺高度；floor_z为median door bottom | 标注边界与假定floor的代理指标 |
| InternScenes_mp3d | 18 hull、全部缺高度、非true room | hull/box指标，不能代表真实墙面 |
| InternScenes_scannet | 104 hull、全部缺高度、非true room | 同上 |
| MansionWorld | 1,313；5缺高度；metadata资产、bbox-top支持 | box支持代理，不是物理承托/稳定性 |
| MultiScan | 6 hull；2不可靠/缺高度；floor slab + wall segment凸包 | 扫描盒体几何，非可重用物体资产成功率 |
| OptiScene_holodeck | 503全部缺高度；bbox宽深约11cm padding | padding后标注布局，非紧致mesh尺寸误差 |
| SAGE-10k | 489；place_id anchor；front经验约定/tilt处理 | 当前统一盒体标签一致性 |
| Scan2CAD | 1；非true room，缺高度；floor-plane polygon | 极小来源样本的box指标，不外推语义或完整room |
| SceneSmith | 364；4缺高度；46已知raw origin错误未直接保留 | 当前test盒体指标，保留间接选择风险 |
| SpatialGen | 45；2已知错误目标；full transform未保留 | 原版全分母与事先定义的敏感性分析分开 |
| SpatialLM | 2,461；2缺高度；front经验约定、边界修复 | 修复后标注几何；非逐资产语义认证 |
| Structured3D | 747；2缺高度；半尺寸/原始basis转换、人工opening厚度 | 统一标注box/墙多边形代理 |

raw4,326,606对象、prepared1,990,430 movable +477,711 fixed、admitted1,085,477 movable +259,695 fixed及test108,253 movable +24,955 fixed均逐条扫描字段。所检查的asset引用字段集合（`furniture_asset_id/asset_id/model_id/mesh_path/mesh/model_path/glb_path/asset_path`）命中为0；这说明当前IR没有这些可直接解析的资产引用，**不等于原始磁盘没有资产**。未打开mesh、纹理、模型权重，也未核验每个真实资产是否缺失、pivot/scale/front正确或可渲染。

全部raw房间`meta.front_known=True`；13,488个raw对象明确`front_known=False`；prepared/current对象未保留该逐对象标记。全部扫描对象无名称含`mask`的字段。不能从房间级True推断逐资产语义真值，更不能把另一个v2产物的partial masks视为本轮assistant loss的有效标签掩码。

## 可复现证据与未验证项

[源/代码读取清单](full-audit-data-source-code-inventory.json)区分全文阅读、指定行阅读、仅SHA比较和已有snapshot receipt；[上下文文档清单](full-audit-data-context-inventory.json)记录known-issues、v2 audit和ARIS资料；[有限历史病例清单](full-audit-data-known-cases.json)包含12份原始证据文件SHA。[统一文件检查清单](full-audit-data-files-checked.json)合并87个已读/已哈希路径及各阶段覆盖范围；[最终机器摘要](full-audit-data-final-summary.json)区分完成状态、已知源缺陷、原审计错误和补充检查。完整扫描结果的`files`字段逐文件记录绝对路径、SHA、大小、记录数、字段计数和complete_scan。扫描脚本及Slurm脚本有单独launch hash。SHA一致只证明身份，不证明几何正确。

未验证：原始二进制资产/mesh重新测量、全部未知源转换缺陷、纠正尺寸后的prior变化/资格变化/训练结果、因历史筛选导致的因果偏差、模型输出质量与正式API判分、最终adapter fresh-load执行结果。本次不通过读取训练loss或672工程测试推断这些结果。论文可以说明“对这个明确冻结、有限来源契约的未知尺寸J/P比较”；要升级为纠正后的数据质量与真实物理准确性主张，需版本化修正及新配对实验。
