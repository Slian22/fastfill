# FastFill v2 review3：启用目标与累积窗口

日期：2026-10-06。本次只处理 review2 独立复审确认的非默认配置 P2，完善新服务器执行步骤。review2 的包身份、640 tests 与数据修订记录保留为历史事实；本版不覆盖旧包。

## 复审判断

用户反例成立：review2 的 `_preflight` 不读取 `LossConfig`；yaw-only 配置仍保留只有 position/size 标签的样本。真实入口中 loss 与所有梯度均为零，AdamW 仍执行 weight decay，记录了一个不存在 yaw 学习信号的更新。

这个反例不阻塞 review2 默认四项全开的完整标签 pilot，但会污染 yaw-only 等消融的更新次数和训练曝光。不能以“loss 等于零”判断是否有监督：预测正确时也可能是零。

## 本版修复契约

1. 预检和 criterion 共享实际启用目标的资格规则，考虑字段 validity、固定坐标、BEV 标签资格及 scene regularizer 的真实项；没有任何启用目标的样本明确拒绝。
2. 累积整个窗口的有效目标数；同步边界用 int64 汇总全部 rank。整个窗口全空时不执行 AdamW，不递增更新步数。
3. 前一个 microbatch 有有效目标、最后一个没有时保留已累积梯度；某 rank 为空、其他 rank 有目标时仍参加全局 backward/更新。
4. 有有效目标但数值 loss 正好为零时仍允许正常更新；没有目标的窗口清零，完全无更新的 epoch 明确退出，避免死循环。
5. 日志区分窗口有效目标与最后 microbatch 的几何诊断，记录跳过的空窗口；更新后的 validation 与 checkpoint 约定保留。

默认四项 loss、预测参数化和 dataset 标签不变。无目标标签的 boundary/collision-only 训练仍可有合法正则项，不能因为没有 GT 标签而一律拒绝；支撑豁免的无碰撞项也不能冒充有效监督。

## 数据及旧文件

训练数据完整沿用 review2：141,341 场景、1,787,052 对象；完整 cohort 仍为 57/2/6。主目录五文件和 cohort 四文件与已校验的 review2 副本逐项 hash 比较，不重新划分或更改几何。

本次外盘 `/Volumes/harddisk` 未挂载。使用本地独立审核解包副本，在重新执行其完整性校验后生成新包到仓库 `outputs/fastfill_v2/FastFill_v2_20261006_review3/`，没有在 `/Volumes` 创建伪造盘目录。新包不意味着原外盘数据消失；重新挂载后可复制新包到外盘新目录。

## 服务器与放行范围

用户提供 8 张 H20Z、driver 570.124.06、2.8 TiB 主机 RAM；允许使用 GPU 1–7。先选择 GPU 1 做单卡真实 Qwen pilot，保留原任务。使用 PyTorch 2.13.0/cu126，服务器实际回归和 CUDA pilot 尚未执行。

v2 代码当前仍在本地，未 commit/push、未上传服务器/HF。包内包含数据与代码；服务器需安装 CUDA PyTorch/依赖并准备基础 Qwen。完整步骤见 [新服务器启动手册](fastfill-v2-server-start.md)。

新增 **19** 项回归通过；完整套件 **659 tests + 114 subtests passed**，v2 覆盖率 **87.68%**。父数据 hash 和新归档核验的确切结果记录在包内 `RUNBOOK.json` 与 `audit/objective-review/`。这次修复仍不证明稀有 yaw 足够、主语料全部物理合法或真实资产/Host/GLB 已验收。正式训练仍需冻结采样、预算和实际模型；先跑有界 pilot。
