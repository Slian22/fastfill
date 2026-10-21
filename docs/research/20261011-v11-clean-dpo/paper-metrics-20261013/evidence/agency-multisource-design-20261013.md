# 多源公平比较最小可执行设计 — 2026-10-13

## 决策与证据边界

Primary contract 已由用户选择：**未知尺寸，联合预测尺寸和布局**。给定房间边界、已知固定context、请求对象类别/可允许描述及显式请求约束，模型输出每个实例的 local size、bottom-center pos、semantic-front yaw；不能通过输入排序、ID或隐藏sidecar把目标size/pose传入。

最小主实验仅共同 fresh base 的 SFT / SFT+DPO1 / SFT+DPO1+DPO2 三行。Known-size 原版 OptiScene 是信息更多的参考，独立表呈现，不混主胜负。已完成的 warm-start V1.1 链保留 legacy 诊断，不能通过事后改 prompt 或重新渲染被重命名为 fresh-base 实验。本报告不启动新训练/API/大批量渲染。

证据仅使用已有 source adapters、`.release/v3.2/data/v3.2/{MANIFEST,stats,QA}.json`、冻结 test 的metadata与先前两个来源asset审计；不重新扫描硬盘原始数据。冻结 test SHA 与当前 V1.1 完全相同：`0b28d10c015bbd9f39151113085c84a05a8117ea3facfa5553e095d10deff303`。清单脚本/结果：`/tmp/agency-multisource-inventory-20261013.{py,json}`（Python3.13.1、stdlib、源码SHA已记录）。

## 最终交付：已验 / 未验 / 后续 gate

- **已验**：冻结18源清单为8,647 test房间、108,253待摆对象；两源真实test的9,072家具已逐对象联结，六房型1,009房间/6,063家具/1,968资产OBJ均存在。材质路径检查发现一种可解释的MTL名称不一致，16间卧室需固定resolver。资产存在性结论已独立归档。
- **已验代码能力，不等于本次逐字段重审**：v2已实现position/size/yaw validity masks及完整box组合mask（`fastfill/v2/objective.py:14–26`）。沿用该机制给每项指标绑定资格与分母；保留全部18源的总体账本，不宣称所有对象具备完整、可信的size/pos/semantic-front yaw。v2历史strict-policy计数、旧source-audit的audit-only和旧SpatialLM whitelist均不能替代当前V1.1逐字段资格（`fastfill/v2/README.md:365–382,391–403`）；早期raw MultiScan builder的限制只适用于该builder（`fastfill/v2/data.py:49–58`）。本轮未重跑当前V1.1的全字段能力审计，不给出新的eligible计数。
- **未验**：其余16源当前test逐资产/材质完整率；全资产实际加载与共同renderer成功率；新schema已实现且目标不干扰；共同fresh-base新链的训练/评测结果。表内C/U仅描述本次视觉证据强度，绝非当前训练或几何eligibility判决。已有逐test mesh路径存在也不等于已渲染成功。
- **Gate 1（输入与身份）**：实现unknown-size schema、不可含目标的ID/顺序，train/dev上通过改变GT size/pose不改变输入的检查；冻结split/group与允许信息边界。
- **Gate 2（指标分母）**：冻结本轮逐字段能力mask及其来源版本。分开报告请求总数、字段合格数、模型失败数；源标签缺证记N/A/unknown，模型在合格请求上漏物体/无效输出记失败。全joint指标仅在共同完整且可信的字段集合上计算，并并列其覆盖率；不能把未标注当模型失败，也不能只展示合格数而隐去总数。
- **Gate 3（视觉）**：非test smoke固定轴、反射winding、bottom pivot、材质resolver、相机和GT/预测共同box-fit；实际load/render通过后才锁定视觉清单。其他源暂报visual=N/A及具体缺证，不从全几何总体删去。
- **Gate 4（最小运行）**：设计review后执行共同fresh base的A/B/C三行；known-size及legacy独立表。父代理保留执行控制。本轮不新增数据家族审计、训练、API或渲染任务。

## 最小实验矩阵

| 行 | 初始化/训练 | 输入 | 用途 |
|---|---|---|---|
| A | 经SHA核验的共同fresh Qwen3-8B base → 新schema SFT | unknown-size共同schema | 主baseline |
| B | A的同一SFT checkpoint → 固定DPO1 | 完全同schema/请求 | A→B阶段效应 |
| C | B → 固定DPO2 | 完全同schema/请求 | B→C和A→C阶段效应 |
| R1 | 原版/复现OptiScene（如具备可核验权重） | known-size、原版conditioning | 信息更充分参考；不参加A/B/C胜负 |
| R2 | 现有V1.1 warm-start SFT/DPO链 | 历史schema，含GT-derived ordering | legacy诊断；保持原结果和缺陷披露 |

首轮只用一个预注册训练seed；A/B/C共用数据split、tokenizer revision、adapter rank、SFT起点、解码、context/token预算、失败规则、repair版本和所有评测请求。记录实际steps/tokens/算力。该设计估计“添加整段DPO训练”的阶段效应（包括其额外数据/计算），不分离DPO loss本身与额外训练预算；共同fresh base是必要控制，并不自动证明算法级因果。若日后声称DPO优于同预算继续SFT，另行预注册一个继续SFT对照，本轮不扩展矩阵。单seed区间仅反映测试群组抽样，不代表训练随机性。

## 目标无关 ID 与顺序：可直接实现

为新协议建立 immutable inventory sidecar：`(source, persistent_scene_key, original_instance_id) -> opaque request id`。只允许来自请求身份/provenance的键，禁止 size/pos/yaw/GT parent、渲染结果、旧category_n或目标面积排序参与。源缺稳定instance id时，在第一次仅处理inventory身份的导入中分配随机nonce并冻结crosswalk；以后复用，不按几何重新生成。

```python
import hashlib, hmac

def bind_inventory(source, scene_key, inventory, protocol_seed):
    # inventory: immutable origin_instance_id + allowed category/description only
    def key(item):
        msg = '\0'.join((source, scene_key, item['origin_instance_id'])).encode()
        return hmac.new(protocol_seed, msg, hashlib.sha256).digest()
    ordered = sorted(inventory, key=key)   # no target geometry read
    bindings = [(f'obj_{i:04d}', item) for i, item in enumerate(ordered, 1)]
    requests = [dict(id=rid, category=item['category'],
                     description=item.get('description', ''))
                for rid, item in bindings]
    sidecar = {rid: item['origin_instance_id'] for rid, item in bindings}
    return requests, sidecar   # model receives requests only
```

`protocol_seed` 是独立预注册字节串，存入manifest，不从模型分数、目标几何或当前目录内容产生。排序只在inventory上运行；标签随后按同一crosswalk重标并按输入顺序输出。GT支持parent、训练/评测约束中引用的ID也在标签层/显式请求层一致重映射；不能用GT parent决定inventory顺序。原始assetId、实例UID、split/group、GT尺寸和房间答案只存评测sidecar，不发给模型。现有资产恢复用GT几何做join的做法仅限归档crosswalk，不允许放入新prompt生成器。

必须通过目标不干扰检查：固定允许输入及实例身份，任意改变GT size/pos/yaw/parent、重新排列标签数组，生成的输入JSON必须逐字相同；不得由GT语义坐标生成排序特征。重复类别对象仍各有稳定实例ID；输出ID恰好覆盖所有请求一次。描述采用冻结的纯对象语义描述或显式用户要求，清理非请求的目标位置、尺寸和“左于某物”等答案信息；当前从GT渲染生成的空间推理描述不得悄悄作为unknown-size无答案输入。inventory、场景身份、描述/类别本身的可用信息边界必须一致。

## 两层评测总体：不丢源、不混FID

1. **共同几何总体**：release 18 source 全部8,647 test房间、108,253个待摆对象；固定context另计。每源保留原始请求分母；按冻结能力mask给出字段合格分母，在合格项上模型失败/缺对象算失败，连续size指标对缺对象有明确惩罚/coverage列。源标签缺证单列unknown/N/A。test大小不因mesh缺失而改变。
2. **真实资产视觉辅助层**：先于各方法输出评分，按源数据可联结、mesh/材质可加载、轴/pivot可核验的共同清单定义eligible UIDs；GT/A/B/C用同一subset、资产、相机和渲染器。报告每源 `N_geometry/N_asset_join/N_mesh/N_material/N_rendered`、失败原因及方法实际生成失败。不得因某stage失败把房间从该stage FID样本里悄悄移除。
3. box-only图和原始扫描图片可以另作几何/观察诊断，但不混入“真实资产render FID”。完整scan mesh包含原摆家具，不能把新布局叠上去当可重排scene；需可分离的实例几何和合理空房context才可进入该层。未核验mesh的源保持几何排名与visual=N/A，明确待查，不叫“差质量”或静默排除。
4. FID按来源/预定义房型计算，GT与预测各一房间同相机/同样本数；top/perspective应固定一种主视角或分开报告，不能临时把多视图当独立房间扩N。不同摄影/扫描/材质域不可池化一个数。小n子集只报告不确定性或N/A，不拿低n FID制造排名。
5. Unknown-size主输出在真实参考资产上按输出size拟合，GT按GT size走相同拟合；以固定参考资产控制外观。该分数评估联合尺寸/布局的渲染效果，不评估模型资产选择能力，也不是论文原生retrieval/Isaac Sim全流程复现。

几何主表同时给出完整可信字段共同集合上的joint validity及其占全请求覆盖率、字段合格requested-instance平均size IoU/误差、对象覆盖、碰撞/越界/支撑/约束失败；每项绑定自身能力mask，OOR独立列。只降低OOR但靠缩小物体换得的结果不能称联合任务获胜。pos/yaw误差为reference agreement诊断，不能把合理替代布局一概判错；yaw按可信semantic front/symmetry明确分母。多指标trade-off直接报告，不在test后临时合成一个“胜率”。接受阈值/size合理范围和任何joint gate只在train/dev上确定并冻结；本任务不改评分实现。

## 18 source矩阵（所有行保留共同几何层）

V=当前逐test家具联结已实测；C=已有catalog/asset路线，当前完整test mesh/material覆盖未实测；U=当前证据不足以作为可重排真实资产视觉场景。C/U都不自动进入FID。表中0 unknown-front flag不代表所有源语义前向已无缺陷；现有InternScenes近垂直front问题仍需单列资格。

| Source | train/dev/test | test objects/groups | group/alias | 坐标/边界/tilt限制 | 真实mesh视觉证据 |
|---|---:|---:|---|---|---|
| SpatialLM | 42068/2239/2461 | 12360/619 | Manycore house + cross-export aliases | polygon; box center→bottom; native front −Y; repaired wall rings flagged (`fastfill/adapters/spatiallm.py:1`) | U: layout/export有家具语义；本次未证明可移动原始mesh联结/完整率 |
| IL3D_3dfront | 18054/939/872 | 5710/284 | 3dfront:house; IL3D merged-room拆分 | 871 polygon +1 hull; raw尺度×scale; native front +Z;保留镜像 (`fastfill/adapters/il3d_3dfront.py:1`) | V:872房间870 mesh完整；论文bed/living506全完整；16bed需MTL resolver |
| InteriorGS | 4877/268/311 | 3878/54 | Manycore house + SpatialLM别名 | polygon; raw corners推front；custom cabinetry无front作fixed；唯一polygon membership recovery (`fastfill/adapters/interiorgs.py:1`) | U:有原始实例/box provenance；完整mesh/原点尚未本次验证 |
| HSSD200 | 1876/86/138 | 1767/14 | hssd:scene | polygon; Y-up→(x,−z,y);两种asset up/front；articulated重建/twin去重 (`fastfill/adapters/hssd200.py:1`) | C:asset_catalog.csv/GLB/URDF映射有证据；当前138test逐asset完整率未验 |
| Structured3D | 12700/793/747 | 3479/172 | structured3d:house | polygon; raw_basis/raw_coeffs毫米半长；front raw−Y；未知/倾斜框分层 (`fastfill/adapters/structured3d.py:1`) | U:box/结构/原始图像可用不等于可移动家具mesh；不自动算视觉合格 |
| InternScenes_arkit | 4528/280/229 | 3381/229 | arkit:video + arkit_visit alias | hull不是精确墙；ZXY OBB；tilt envelope；283 tilted movable (`fastfill/adapters/internscenes_arkit.py:1`) | U:Real2Sim box/scan不能直接变可移动完整家具；需显式asset join |
| InternScenes_mp3d | 1258/89/18 | 145/2 | mp3d:house | hull；重复region去重；ZXY；16 tilted；仅2 test groups (`fastfill/adapters/internscenes_mp3d.py:1`) | U:scan/box可诊断；完整可移动mesh未验 |
| InternScenes_scannet | 1309/80/104 | 2260/49 | scannet:sceneXXXX; rescans共组 | hull；ZXY；451 tilted；与Scan2CAD跨源连通 (`fastfill/adapters/internscenes_scannet.py:1`) | U:scan/OBB不是独立可移动mesh；需重建证据 |
| InternScenes_3rscan | 1200/61/70 | 1077/24 | 3rscan:reference; rescan aliases | hull；ZXY；226 tilted；reference/rescan不能跨split (`fastfill/adapters/internscenes_3rscan.py:1`) | U:同上；原始scan可用不等于干净重排scene |
| Scan2CAD | 18/0/1 | 6/1 | scannet:sceneXXXX; links InternScenes | polygon或proxy；约30%对象有CAD；ShapeNet −Z→front；只1 test (`fastfill/adapters/scan2cad.py:1`) | C:CAD id/9DoF可联结；本次无ShapeNet资源覆盖证明，不可报稳定排名 |
| MultiScan | 114/2/6 | 37/4 | multiscan:space; rescans共组 | 6 hull test；scan floor noise；front/up人工；垂直front不可用yaw (`fastfill/adapters/multiscan.py:1`) | U:整scan及分割不证明完整独立mesh；相机/遮挡域单列 |
| OptiScene_holodeck | 8946/497/503 | 3362/503 | holodeck:source scene | polygon；LH source→RH IR；front−Z；bbox有padding；floor-only (`fastfill/adapters/optiscene_holodeck.py:1`) | V:503 test、3362家具、723资产mesh全部存在；完整原始join已产出 |
| IL3D_synthetic | 6277/349/316 | 1515/316 | il3d:synthetic scene | rectangle；HSSD mesh origin非统一底中心；asset/slot类别有分歧 (`fastfill/adapters/il3d_synthetic.py:1`) | C:HSSD catalog + extra bounds与assetId路线；test逐mesh/纹理未验 |
| SAGE-10k | 8977/503/489 | 24183/489 | sage:scene; multiroom同组 | polygon；front +Y；bottom pivot；2005 tilted；support/pivot修正 (`fastfill/adapters/sage10k.py:1`) | C:原TRELLIS per-scene资产路线已有；当前489test尚未逐asset覆盖验证 |
| MansionWorld | 19396/1153/1313 | 24392/16 | building.split(#)[0], floor-plan variants同组 | polygon；THOR→(x,−z)；wall/surface offsets修正；原点/翻轴特殊 (`fastfill/adapters/mansionworld.py:1`) | C:Objathor/THOR引用及raw surface存在；当前1313test mesh join未验 |
| InternScenes_gen | 12552/828/660 | 12516/293 | internscenes:generated scene | polygon；ZXY；12 tilted；no near-vertical yaw certificate (`fastfill/adapters/internscenes.py:1`) | C:生成原scene/对象可作为候选；当前660test可分离mesh覆盖未验 |
| SpatialGen | 0/0/45 | 226/45 | spatialgen:scene; current-split heldout | polygon；box center→bottom；front−Y；2 tilted (`fastfill/adapters/spatialgen.py:1`) | C:对象/asset来源可追；45test mesh覆盖未验 |
| SceneSmith | 0/0/364 | 7959/210 | scenesmith:parent house/room set; heldout | rectangle；front+Y；mount-specific pivot；1268 tilted；windows元数据不完整 (`fastfill/adapters/scenesmith.py:1`) | C:原DMD/glTF资产路线；364test覆盖未验，不拿5示例代替 |

依据：各adapter文件开头约定及load的group字段；`fastfill/README.md:48–70,97–175`；`fastfill/RELEASE_v3.2.md:9–19,38–54`。原README的InteriorGS train=4867属于旧版，当前manifest/stats为4877；上表以冻结release为准。

## Split、来源泛化与有限样本

已有split由 group、group_aliases、家具/布局fingerprint连通分量隔离（`fastfill/split.py:1–8`）；房间不是独立实验单位。保留相同house/visit/rescan/floor-plan variant在同一split，SAGE multiroom/SceneSmith houses也按parent一起分。扫描别名和跨Manycore导出不能按文件名当独立数据。

- 当前source-heldout只有 SceneSmith364 + SpatialGen45=409房间；当前训练中排除不等于warm-start从未见过。fresh base并审计训练manifest之后才能对新链作更强声明。现有warm-start来源不明以及历史test200曝光均保留披露（`docs/research/20261011-v11-clean-dpo/protocol.md:36–42`）。
- 统计使用同一UID/请求成对比较，按真实连通group聚类，不用对象数扩大独立样本数。提供source-macro和room-micro两种汇总，不能让SpatialLM/MansionWorld数量淹没小源。MansionWorld1313房间仅16 test groups；MP3D18房间仅2 groups；Scan2CAD1 test group不能给可靠泛化区间。
- train/dev/test集合及去重仅基于冻结身份/原始数据，绝不按新test输出分数选择；目前已读过的历史test仍是公开历史诊断。新schema构建检查用train/dev；若需严格未触碰终评，须预先登记未用于设计的group/source候选，不能把已看结果的8,647房间宣称全新盲测。本轮不重新随机切分来掩盖曝光。
- 整个source-heldout实验也应检查跨源同一实体组与相同资产库复用；场景未见和资产未见分开描述。辅助三源（原3D-FRONT、ARKit visit表、3RScan rescan表）是provenance，不多算三批独立训练房间。

## 当前范围之外的历史登记

旧source audit、21份IR文件库存及未进入本冻结18-source清单的源，只保留为带日期的历史provenance。本轮不重扫、不重新裁定这些源的当前eligibility，不以旧audit-only/旧排除原因证明现在不可用，也不将辅助原始导出重复计算为新独立测试群体。新增源须另行形成当前版本的task/字段mask/split证据；不阻塞本轮18源最小矩阵。

## 最小落地顺序与停止条件

1. 冻结 unknown-size schema和允许信息，实现独立inventory/crosswalk；只重建数据副本。跑目标不干扰、实例覆盖、support/constraint重标、split hash/alias检查。发现输入依赖目标即停止，不靠后处理掩盖。
2. 18源全部保留几何清单；按既有字段资格标注真实polygon/hull、front-known、tilt、固定结构/未知障碍等。扫描hull只测hull包含性，不能叫真实墙；unknown front不测faces；tilted upright envelope的分数不叫6DoF mesh物理有效。
3. 在非test smoke验证两种raw轴/反射/镜像/底pivot、材质与联合box-fit，冻结render配置。先用已验证两源的1,009六列作为visual辅助首批，不替代其余源几何；其他C/U仅在独立覆盖审计通过后扩同一清单，不根据方法分数扩/删。
4. 共同fresh base跑同一小规模train/dev端到端核查（父代理review后才执行），再冻结首轮A/B/C三个训练设置和单seed。现有warmstart预测仍只报legacy。
5. 完整18源几何推理一次，所有stage保留原始输出/错误及hash；visual合格subset同管线渲染。GPT评分保留图像、prompt、model版本和每次结果；human UR与API评分独立，不将自动几何通过率改名human usability。
6. 输出每source与geometry/visual资格分层的成对差值、group数、coverage和未评估项。known-size参考单列。任何schema/数据/metric变更另编号，不复用旧test分数作超参选择。

## 已交付可复用资产结论

`/tmp/agency-harddisk-renderability-20261013.md` 给出硬盘source路径、两种实际JSON schema、9,072逐对象join、raw→IR轴转换、材质resolver及最小共同renderer。crosswalk=`/tmp/agency-harddisk-asset-map-20261013.json`。六列1009房间6063家具/1968资产mesh全存在；1种MTL文件名错误有已验证替代，影响16bedrooms。更大1375两源test只有2个living-dining房间缺同一mesh，不能用94/124全库缺失数否定当前subset，也不能把2个失败从全几何分母删除。

本报告未安装依赖、未修改训练/评分/源数据、未运行API/训练/渲染、未扫描无关原始目录。原始mesh路径/材质存在不等于渲染已经成功；其他16源的全test视觉覆盖是明确缺证，而非被证明没有资产。
