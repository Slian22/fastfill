# 真实测试集资产联结与渲染性审计 — 2026-10-13

结论：当前 V1.1 同步的两个来源共 1,375 test 房间、9,072 个家具实例，全部与原始实例唯一联结；1,373 房间家具 OBJ 完整。论文六列限定的 1,009 房间、6,063 家具实例、1,968 种资产，OBJ 文件覆盖 100%。这证明可构建真实资产渲染输入，不等于已完成 Blender 加载、纹理解码或视觉验收。

## 输入与产物

- 精确输入：`/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261011-v11-clean-dpo/paper-metrics-20261013/inputs/server-paper-source-records.json`。`source_test.sha256` 为 `0b28d10c015bbd9f39151113085c84a05a8117ea3facfa5553e095d10deff303`，原 test 共 8,647 行。输入含 data[{evaluation_row,record}] 与三阶段预测；本审计只读取 GT/provenance，未按预测分数选择或调整。
- 同 SHA 的冻结源文件也在 `.release/v3.2/data/v3.2/test_rooms.jsonl`；不是早期硬盘 v3.1 的不同 SHA 文件。
- 可重跑脚本：`/tmp/agency-harddisk-renderability-20261013.py`，Python 3.13.1/stdlib，源码 SHA 存在统计 JSON。
- 汇总：`/tmp/agency-harddisk-renderability-20261013.json`。
- 全部逐对象 crosswalk：`/tmp/agency-harddisk-asset-map-20261013.json`。
- 材质核查脚本/结果：`/tmp/agency-harddisk-material-check-20261013.{py,json}`。材质替代 resolver 的单资产核查结果另存于 JSON 的 validated_readonly_resolver_override。

## 真实原始路径与联结

硬盘根 D=`/Volumes/harddisk/3D_Room_Collections/imChuling__3D_Room_Collections`。

1. Holodeck：D/`OptiScene_exported/holodeck.json`，9,977 场景。`scene_id` 对应 test UID 去掉 `optiscene:`。`rooms[].furniture[]` 保留 `furniture_instance_id`、`source_index`、`source_instance_id`、`furniture_asset_id`、`source_fields.layout_object.assetId`。IR 为 `/Volumes/harddisk/fastfill_ir/OptiScene_holodeck.jsonl`。
2. 3D-Front：使用当前训练的 IL3D 版本，D/`IL3D_exported/IL3D/layout.zip` 内 `layout/<meta.il3d_scene_id>.json`。IR 对象 ID `<scene_uuid>::obj_k` 精确索引原始 `objects[k]`，保留 assetId、id、object_path、position、rotation、scale。资产元数据为同目录 `assets.json`。IR 为 `/Volumes/harddisk/fastfill_ir/IL3D_3dfront.jsonl`。
3. 独立 3D-FRONT 的房屋/墙体：`/Volumes/harddisk/3D_Room_Collections/BillLin66__3D_Room_Collections/3DFront_exported/data/3D-FRONT/<house>.json`。此 export 的 `data/3D-FUTURE-model` 只有 model_info.json；IL3D export README:15–16 同样说明未下载 21GB 模型包。
4. 本机已存在可按相同 assetId 联结的真实 OBJ/MTL/texture：`/Users/slian/Desktop/3D/Worldedge/OptiScene/huggingface/3D-SynthPlace_Scenes_Assets/assets_data/<assetId>/raw_model.obj`（3D-FUTURE 常见）或 `raw_model3.obj`（Holodeck 常见）。按存在文件解析，不能仅凭来源猜文件名。

当前重编号可恢复：以 UID 找 IR，用 category、厘米取整后的 local size/XY、整数度 yaw 联结已保存 GT 对象，再用 Z 解可能重复项。本次 9,072/9,072 唯一，无猜测/最近邻补配。该 GT 几何 join 只用于事后恢复身份，不能成为未来 prompt ID 规则。更稳健的新构建直接保留 source_instance_id sidecar，在过滤、量化、canonical 前携带并输出 crosswalk。依据：`fastfill/adapters/unified.py:67–86`、`fastfill/scene.py:176–193`、`fastfill/build.py:228–249`。

## 六列覆盖率

| 来源/房型 | 房间 | 家具实例 | 缺 mesh |
|---|---:|---:|---:|
| IL3D 3D-Front bedroom | 419 | 1,844 | 0 |
| IL3D 3D-Front living room | 87 | 857 | 0 |
| Holodeck bedroom | 115 | 672 | 0 |
| Holodeck living room | 131 | 1,378 | 0 |
| Holodeck kitchen | 129 | 787 | 0 |
| Holodeck bathroom | 128 | 525 | 0 |

另外 366 房间仍保留在多源几何总体；其中两间 living dining room（evaluation_row 409、505）缺同一 dining table 资产 `d7768e49-b2ff-4821-9e21-8e94b2575c68`。不得用全库 94/124 的缺失数量代替本次精确子集覆盖率。家具以外的 1,657 个固定结构项是门窗等，可由已知 GT/context 程序化构建，不能声称有原始家具 mesh。

## 材质与几何实际检查

六列 1,968 OBJ 的前 256 行均找到 mtllib。可读 MTL 的 2,690 个纹理路径均存在。一个资产 `f350c1ba-8040-4750-9e6e-f45cfbf916a4` 引用 model.mtl，实际有 raw_model.mtl：所有 OBJ usemtl 名称被该文件定义，texture.png 存在。影响六列 16 个 bedroom（另有 1 个 elderly room 在六列外）。允许仅在临时导入副本/内存 resolver 中明确映射文件名，所有方法和 GT 一致记录；不修改源文件或悄悄灰模替代。

样例床 `3da90446-a1e7-4f5c-b4cf-372fea72e9bd`：10,596 vertices/20,657 faces，raw XYZ extents [2.381776,1.261218,2.965929] m，raw min Y=0.002893。样例浴缸 `ab3f548d0e114285a87fbeca08635711`：5,892 vertices/1,964 faces，raw XYZ extents [0.717239,0.700161,1.712212]，raw min Y=-0.001496。浴缸 GT semantic box=[1.83,0.83,0.71]，说明 Holodeck 框含 padding。全资产尚未逐一解码/验证拓扑/渲染。

## 轴、pivot 与最小共同管线

IR 为 RH Z-up、local +X=semantic front、size=[sx,sy,sz]、pos=box bottom center、GT yaw=radians；预测 placements yaw=degrees。`fastfill/scene.py:3–12`、`fastfill/v1_1/scene_v1_1.py:31–35`。

- Holodeck native OBJ v=(x,y,z)：Bv=(-z,x,y)，det(B)=-1。这是 Rz(+90) @ A，A v=(x,z,y)。IR yaw=-raw_rotation.y-90°（`optiscene_holodeck.py:3–8,18–36`）。所以 Rz(IR_yaw)B=Rz(-raw_rotation.y)A；只做 A 再用 IR yaw 会错90°。烘焙反射时翻转面绕序，正确处理法线/切线，不再加 raw rotation。
- 3D-FRONT native OBJ：Bv=(z,x,y)，det(B)=+1。来自 native->Z-up (x,-z,y) 再 Rz(+90)，使 mesh +Z 前向成为 IR +X（`il3d_3dfront.py:3–18,78–84,312–321`）。原始 scale 的负号必须保留；bbox-fit 最后会替代正尺度，勿双重缩放。
- 先对每个资产全部 parts 作 B（及 native scale 符号）变换，联合 AABB，pivot=[center_x,center_y,min_z]；v0=Bv-pivot。若采用未知尺寸主任务的 bbox-fit 可视化，D=diag(target_size / extent(v0))，world=Rz(yaw) D v0 + pos。GT 使用 GT size，预测使用各自输出 size，固定对象使用公共 context；不能把 GT size 注入预测。定义冻结后所有来源/方法一致。
- 这是“固定参考资产的联合尺寸/布局可视化”，资产 ID 是评测端 oracle reference，不是模型预测的检索能力。box-fit 会改变 native geometry 的 padding/比例，必须披露，不能称为原版 Isaac Sim/native-scale 复现。
- 同一房间共享原 boundary、fixed context、材质、floor/walls、相机、灯光、分辨率和 samples。GT/SFT/DPO1/final 全部经过相同代码。保持 polygon，不把 L 型房间改为满矩形。失败、缺对象、非有限输出保留 receipt，不从几何总体删除；mesh/resource 不合格单列视觉不可评估原因。
- 可复用 `RoomGenBench/bench/render.py:24–35,61–120` 的 Cycles、同一 top/perspective camera；它的五场景 main/矩形 metadata 不是本任务 loader。应写独立 manifest-driven wrapper，调用通用函数或直接相同配置。若组装 GLB，IR world (x,y,z)->glTF (x,z,-y)，见 `RoomGenBench/bench/assemble.py:39–41`，Blender importer 再还原为 Z-up，勿重复换基。

非 test smoke：`optiscene:HOLODECK-bathroom-0` 和 `il3d:b57c5850-86d4-11f0-a478-60cf84ae2082` 均不在完整两源 test 名单。它们是非test候选，尚未声称已查其 dev/train 归属。只检查 asymmetric front/handedness、正负 scale、底部 pivot、box envelope、纹理和相机，不按 test FID/GPT 分数选轴或大小策略。此只读任务未执行 Blender。

## 两个实际映射 schema

### IL3D_3dfront

```json
{
  "uid": "il3d:b57ce9be-86d4-11f0-a478-60cf84ae2082",
  "evaluation_row": 138,
  "source": "IL3D_3dfront",
  "prompt_id": "king_size_bed_1",
  "original_instance_id": "b57ce9be-86d4-11f0-a478-60cf84ae2082::obj_0",
  "source_index": 0,
  "source_instance_id": "King-size BedBeds1",
  "asset_id": "3da90446-a1e7-4f5c-b4cf-372fea72e9bd",
  "mesh_path": "/Users/slian/Desktop/3D/Worldedge/OptiScene/huggingface/3D-SynthPlace_Scenes_Assets/assets_data/3da90446-a1e7-4f5c-b4cf-372fea72e9bd/raw_model.obj",
  "original_usdz_relative_path": "3D-FRONT/Beds/3da90446-a1e7-4f5c-b4cf-372fea72e9bd.usdz",
  "gt_object": {
    "size": [
      2.97,
      2.38,
      1.26
    ],
    "pos": [
      2.03,
      1.82,
      0
    ],
    "yaw": 3.141592653589793
  },
  "layout_object": {
    "assetId": "3da90446-a1e7-4f5c-b4cf-372fea72e9bd",
    "id": "King-size BedBeds1",
    "object_name": "3D-FRONT_3da90446-a1e7-4f5c-b4cf-372fea72e9bd",
    "rotation": [
      180,
      -90,
      180
    ],
    "object_path": "3D-FRONT/Beds/3da90446-a1e7-4f5c-b4cf-372fea72e9bd.usdz",
    "category": "Beds",
    "kinematic": true,
    "position": [
      2.026900290812255,
      0.0,
      3.2414665217787832
    ],
    "material": null,
    "scale": [
      1,
      1,
      1
    ],
    "roomId": "BedRoom",
    "bbox": [
      2.386529998779297,
      2.9663101196289063,
      1.2644499969482421
    ]
  }
}
```

### OptiScene_holodeck

```json
{
  "uid": "optiscene:HOLODECK-bathroom-101",
  "evaluation_row": 4038,
  "source": "OptiScene_holodeck",
  "prompt_id": "bathtub_1",
  "original_instance_id": "HOLODECK-bathroom-101::obj_0",
  "source_index": 0,
  "source_instance_id": "bathtub-0",
  "asset_id": "ab3f548d0e114285a87fbeca08635711",
  "mesh_path": "/Users/slian/Desktop/3D/Worldedge/OptiScene/huggingface/3D-SynthPlace_Scenes_Assets/assets_data/ab3f548d0e114285a87fbeca08635711/raw_model3.obj",
  "original_usdz_relative_path": null,
  "gt_object": {
    "size": [
      1.83,
      0.83,
      0.71
    ],
    "pos": [
      2.2,
      0.87,
      0
    ],
    "yaw": 1.5707963267948966
  },
  "layout_object": {
    "assetId": "ab3f548d0e114285a87fbeca08635711",
    "position": {
      "x": 0.7,
      "y": 0,
      "z": -1.1317
    },
    "rotation": {
      "x": 0,
      "y": 180,
      "z": 0
    },
    "object_name": "bathtub-0",
    "description": "large modern freestanding bathtub with a sleek curved design",
    "bbox": [
      71,
      83,
      183
    ]
  }
}
```
