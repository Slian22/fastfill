# FastFill v2：review2 全量审计后续处置

2026-10-06。用户提供的报告针对不可变 review2 训练包。本次检查当前代码，独立复现新增反例并修复；旧包、原始数据和旧 checkpoint 保留。当前任务已改为最小 XY 条件直接 bbox 输出，不能混用历史丰富条件的统计和验收。

## 六项判断

| 项目 | 独立检查与处置 | 当前状态 |
|---|---|---|
| C1 尾累积窗口缩小 | 实际 run_training/Accelerate：K=2、3 microbatch，旧梯度为1/0.5；按K/actual_microbatches在clip前修正。包含真实两 rank Gloo、空标签窗口、CPU GradScaler 回归 | 已修复；K=1保持原行为 |
| C2 禁用 loss 的空监督更新 | review3 已引入 loss-aware preflight 和全rank累积窗口有效目标计数；监督loss=0仍更新，完全无目标不更新 | 已有修复，本次回归保留 |
| C3 文本与 ID 混淆 | RED 复现 ID=category/description 等合法条件被拒绝；只改对象 ID/support_parent 与显式关系引用，保留文本/属性/key/表面/组名 | 已修复，合法角色仍禁止交换 |
| C4 资产失败退出0 | 实际 CLI 子进程：空catalog报告失败却exit0。保存诊断后exit2，stdout显式runtime_ok/committed；纯预测与成功闭环exit0 | 已修复；当前无资产路径不受旧反例影响 |
| D1 Scan2CAD估计地板 | 独立重扫12场景/140对象、124个底面为0、其中118个完整有效position。adapter确实以目标底面中位数估计floor并15cm贴齐 | 历史来源资格问题成立；当前SpatialLM-only数据没有Scan2CAD |
| D2不可表达尺寸 | 独立复算13对象：train10/validation1/test2，至少一轴超ref×exp(±10)。新增配置相关preflight拒绝，记录对象/维度/范围/来源，GT不改 | 当前XY数据100,635轴全在范围内；极薄原始资产真值仍未复核 |

range 检查仅作用于实际启用的 size/box 监督和需要学习的完整有效尺寸维度；可信固定尺寸绕过head，缺标签先过滤。范围端点容差仅处理float32舍入，不用epsilon把标签修成有效。拒绝是整场景资格结果，不静默改mask或size。超范围尺寸来源需要原始几何证据，不能仅扩大参数范围宣布纳米家具有效。

D1 未凭空恢复不存在的 raw z，也未重写旧 v1 adapter 或历史语料。若后续重新使用16家族丰富条件主集，Scan2CAD必须另建来源资格版本：显式标明estimated floor/upstream snap，区分XY参考与未经独立测量的z；当前完整向量loss下屏蔽z会排除相应position项。这个历史问题没有因当前子集排除该来源而变成“已经修正全部源标签”。

## 服务器与训练放行范围

新目录 `/home/jovyan/shanliantian/FastFill_v2_direct_bbox_20261006`，独立环境复用此前env；官方固定revision Qwen3-8B在models/Qwen3-8B。全部新train/validation/test 9,601/539/624行通过真实tokenizer预检，最大716/490/526tokens；无对象、context或尺寸范围拒绝。

新XY协议真实pilot已完成：seed42随机32train/16validation/16test，20次更新，LoRA/decoder/四个heads每步有梯度，保存后重新加载和五对象无资产bbox导出成功。测试schema/ID/正尺寸16/16，目标bbox代理无失败检查仅1/16；边界失败14/44对象、地板下界失败34/44、39个重叠对。原严格资产式Validator还要求未知支撑/天花板，0/16；这个口径不等于当前纯bbox接口。没有mesh/physics/Host/Solver验收。

Pilot为修复C1/C3/C4之前的不可变代码快照，K=1、fixed correspondence、无catalog不触发这些反例。最终训练使用新的project-final及其独立hash，不能就地篡改pilot版本或声称旧pilot已经执行新修复路径。完整GPU/Linux回归和最终hash记录随服务器发布保存。

当前新子集可以用于全数据数值监督实验；它不代表全部16家族或截图32–122对象密集场景。新子集最多26对象、平均3.12，且yaw不认证语义前向。正式科研结论仍需固定曝光预算、同条件对照、合法交换组/顺序扰动与密集场景评测；训练运行正确不保证模型生成有效场景。

## 证据与复现

本地证据分别保存于 `outputs/fastfill_v2/accumulation-tail-audit-20261006`、`direct-bbox-20261006/test-evidence` 和 `server-direct-bbox-20261006/server-evidence`。其中 historical-d1-d2-independent.json是独立只读普查，c4-red.log保留原退出码反例；修复后的实际子进程测试位于test_predict.py。数据和服务器pipeline使用方式见 [直接bbox手册](fastfill-v2-direct-bbox.md)、[训练说明](fastfill-v2-training-pipeline.md) 和 [服务器手册](fastfill-v2-server-start.md)。
