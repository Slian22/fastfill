# 硬盘历史比较结果审核 / Local harddisk comparison audit

审查日期：2026-10-20。原始结果只读，未运行推理、训练、远程作业或联网查询。范围是当前可访问本地存档，不能据此断言远程没有结果。

## 覆盖范围

首次扫描23238目录、8081候选；补扫24850目录，新增1079候选；合计9160个路径（包含重复副本，不是独立实验数）。补扫包含artifacts/data/project/code-snapshots/adapter/checkpoints。只跳模型权重、图片/渲染、依赖环境等体积目录。完整路径、字节数和mtime见disk-results.json。mtime是文件归档时间证据，不保证实验实际生成时间。

- `/Volumes/harddisk/fastfillout`：8901个候选。
- `/Volumes/harddisk/FastFill-Delivery-20261015`：42个候选。
- `/Volumes/harddisk/.FastFill-Delivery-20261015.build`：42个候选。
- `/Volumes/harddisk/FastFill_v2_20261006`：24个候选。
- `/Volumes/harddisk/FastFill_v2_20261006_review2`：38个候选。
- `/Volumes/harddisk/FastFill_v2_multisource_20261006`：93个候选。
- `/Volumes/harddisk/FastFill_v2_multisource_20261006_sync-20261006`：15个候选。
- `/Volumes/harddisk/fastfill_data`：2个候选。

已额外核对FastFill训练tar.gz两份及Delivery.zip成员列表，不解压覆盖原结果。fastfill_ir/fastfill_ir_v34为原始数据JSONL；fastfill_data/v3、v3.1主要构建/QA报告。外层enoxian/SWE-bench、gentask_baseline、particulate/R2/R3等经目录/包内容识别为其他任务，不作为本项目性能基线。

## 本次补回的实测比较

旧v2确有gpt-6.1-sol medium prompt/harness实测，而且LLM布局clean显著高于当时FastFill。FastFill在尺寸、角度、BEV IoU有优势，但位置误差不能笼统声称胜出：朴素位置基线更好。旧v2报告2026-10-09归档，不可与后续27B/system022/strict027混为同一系统。下面所有核心指标一起保留，未按胜负筛选。

所有FastFill旧v2主模型为Qwen3-8B骨干+几何头，LoRA rank8/alpha16；main4的yaw_cls损失权重分别0.08和0.5。main7 max_objects128，main4 max_objects256。模型身份和损失配置见JSON model_identities。

### 旧v2实测：同一300请求，原始报告

Same request hash e02da65e...; reference geometry conditional on valid labels and completed layouts, so GPT prompt has 293 layouts vs 300 for other rows. Clean denominator here is all 300 requests. All old strict rates are 0 because unchecked mesh/physics/solver hard checks count unknown as failure. This is not modern admission. FastFill checkpoint-selection cohorts include these 300 scenes; do not call this held-out FastFill test. Original OOR is a different proxy from modern soft OOR.

原始实测同一300请求（SHA e02da65e…）。GPT prompt只完成293间，其余完成300间；几何误差仅对完成且标签有效对象统计，故完成分母不同。clean_rate_all统一除以300。旧strict把mesh/physics/solver未检查的unknown视为失败，因此全为0，不能等同新027严格准入；旧OOR代理也不同于新版soft OOR。这300场景参与FastFill选模，不能称FastFill独立留出测试。

| method | requested | layouts | clean_rooms | clean_rate_all | strict_rate_old | oor_fraction | bottom_center_error_m | log_size_error | yaw_error_rad | bev_iou |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B FastFill yawcls008 spread | 300 | 300 | 145 | 0.483333 | 0.000000 | 0.000000 | 2.687120 | 0.246490 | 0.685584 | 0.048420 |
| Qwen3-8B FastFill yawcls008 argmax | 300 | 300 | 4 | 0.013333 | 0.000000 | 0.000000 | 2.668111 | 0.245156 | 0.697418 | 0.046616 |
| Qwen3-8B FastFill yawcls05 spread | 300 | 300 | 137 | 0.456667 | 0.000000 | 0.000000 | 2.673574 | 0.247634 | 0.667406 | 0.050464 |
| Qwen3-8B FastFill yawcls05 argmax | 300 | 300 | 8 | 0.026667 | 0.000000 | 0.000000 | 2.623685 | 0.246683 | 0.659938 | 0.048281 |
| gpt-6.1-sol medium prompt | 300 | 293 | 252 | 0.840000 | 0.000000 | 0.000000 | 2.674704 | 0.411670 | 0.781736 | 0.036323 |
| gpt-6.1-sol medium harness | 300 | 300 | 284 | 0.946667 | 0.000000 | 0.000000 | 2.623282 | 0.387905 | 0.767426 | 0.032391 |

可核查来源：
- `/Volumes/harddisk/fastfillout/runs/llmrows-main4-cell05-main-20261008a-e5-yawcls008/report.json`
- `/Volumes/harddisk/fastfillout/runs/llmrows-main4-cell05-main-20261008a-e5-yawcls008-argmax/report.json`
- `/Volumes/harddisk/fastfillout/runs/llmrows-main4-cell05-main-20261008a-e5-yawcls05/report.json`
- `/Volumes/harddisk/fastfillout/runs/llmrows-main4-cell05-main-20261008a-e5-yawcls05-argmax/report.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/llm/llm-prompt-300-eval/report.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/llm/llm-harness-300-eval/report.json`

### 旧v2最终配对：共同完成293间

All four methods evaluated on the same 293 completed rooms; geometry label counts are metric-specific. Final FastFill = yawcls05 step5000; all 300 scenes occur in checkpoint-selection cohorts. Bootstrap 10000 seed0; CI and denominators retained in numeric_records/CSV. Latency FastFill 0.0964185208s uses GPU forward+decode (argmax row reuses spread timing); GPT prompt45.8564348771s, harness33.4993478580s include API wait/retry/backoff, so scopes differ.

四方法共同完成的293间配对；各几何指标有效对象数不同，精确分母与95%CI保留在CSV/JSON。FastFill为yawcls05 step5000，这300个源场景全部在选模队列中。bootstrap10000次seed0。FastFill延迟0.0964秒仅单房GPU前向及解码，argmax复用了spread计时；GPT prompt45.8564秒、harness33.4993秒含API限流等待、重试和退避，计时范围不同。

| method | rooms | bottom_center_error_m | log_size_error | yaw_error_rad | bev_iou | object_collision_rate | room_clean_rate | room_collision_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FastFill 最终·spread（防碰撞后处理） | 293 | 2.714193 | 0.247676 | 0.645369 | 0.059119 | 0.250664 | 0.467577 | 0.498294 |
| FastFill 最终·argmax（模型原始输出） | 293 | 2.661330 | 0.246781 | 0.643502 | 0.056794 | 0.707171 | 0.027304 | 0.740614 |
| LLM prompt | 293 | 2.674704 | 0.411670 | 0.781736 | 0.036323 | 0.064077 | 0.860068 | 0.075085 |
| LLM harness | 293 | 2.595398 | 0.382592 | 0.763836 | 0.037758 | 0.017264 | 0.952218 | 0.027304 |

可核查来源：
- `/Volumes/harddisk/fastfillout/runs/analysis/final/llm-compare.json`

### 旧v2最终配对：300间含失败回退

All 300 requests; missing GPT prompt outputs imputed using room-centre position, leave-one-out category-median size, uniform-yaw expectation and BEV IoU zero; clean=0. These are failure-policy-adjusted geometry values, not original observed-only means. Final FastFill = yawcls05 step5000; all 300 scenes occur in checkpoint-selection cohorts. Bootstrap 10000 seed0; CI and denominators retained in numeric_records/CSV. Latency FastFill 0.0964185208s uses GPU forward+decode (argmax row reuses spread timing); GPT prompt45.8564348771s, harness33.4993478580s include API wait/retry/backoff, so scopes differ.

统一300请求，GPT prompt缺失的7间用房间中心位置、评估集留一法类别尺寸中位数、均匀角度误差期望、BEV IoU=0补入，clean=0；这些是含失败回退的几何数值，不是原始仅完成均值。FastFill为yawcls05 step5000；源场景参与选模。bootstrap10000次seed0，分母和CI保留。FastFill与GPT延迟计时范围不同，不直接作纯推理速度比。

| method | rooms | bottom_center_error_m | log_size_error | yaw_error_rad | bev_iou | clean_room_rate_incl_failures | interface_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FastFill 最终·spread（防碰撞后处理） | 300 | 2.673574 | 0.247634 | 0.667406 | 0.050464 | 0.456667 | 1.000000 |
| FastFill 最终·argmax（模型原始输出） | 300 | 2.623685 | 0.246683 | 0.659938 | 0.048281 | 0.026667 | 1.000000 |
| LLM prompt | 300 | 2.635379 | 0.407668 | 0.782277 | 0.029519 | 0.840000 | 0.976667 |
| LLM harness | 300 | 2.623282 | 0.387905 | 0.767426 | 0.032391 | 0.946667 | 1.000000 |

可核查来源：
- `/Volumes/harddisk/fastfillout/runs/analysis/final/llm-compare.json`

### 旧v2 minimal测试：4455请求，含朴素基线

Minimal=三字段投影; full=完整条件投影. Same input file SHA212fc6bf..., different eligible subsets. main7 max128 objects vs main4 max256; this comparison changes data recipe/capacity, not a pure one-factor ablation. Clean/layout and clean/all are both retained. Old strict includes unknown hard checks and is not amendment027 strict admission. Naive baseline fits evaluation-set leave-one-out and is paired to final-spread label objects, not trained only on training data. Position means show the naive baselines outperform learned heads here. Exact per-metric N and CI preserved; 2000 bootstrap seed0.

三字段minimal投影4455请求，不是完整条件8620请求；同一输入文件SHA212fc6bf…但合格子集不同。main7 max128对象、main4 max256，包含数据配方和容量差异，并非纯单因素消融。表同时保留clean/完成与clean/全部；旧strict包含硬检查unknown，不是027严格准入。朴素基线在评估集留一法拟合，并配对final-spread有效标签对象，不是只用训练集拟合。朴素位置误差优于学习模型，也照实保留。每指标分母和95%CI见CSV，bootstrap2000次seed0。

| method | requested | layouts | bottom_center_error_m | log_size_error | yaw_error_rad | bev_iou | object_collision_rate | object_hard_violation_rate | room_clean_rate | clean_room_rate_incl_failures | strict_ok_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B main7 step6357 argmax | 4455 | 4391 | 2.777136 | 0.402559 | 0.677805 | 0.035763 | 0.715946 | 0.813137 | 0.025734 | 0.025365 | 0.000000 |
| Qwen3-8B main7 step6357 spread | 4455 | 4391 | 2.743763 | 0.402701 | 0.677264 | 0.036947 | 0.282007 | 0.284881 | 0.388977 | 0.383389 | 0.000000 |
| Qwen3-8B main4 yawcls008 step5000 argmax | 4455 | 4447 | 2.871360 | 0.344993 | 0.684038 | 0.032598 | 0.705045 | 0.813182 | 0.012818 | 0.012795 | 0.000000 |
| Qwen3-8B main4 yawcls008 step5000 spread | 4455 | 4447 | 2.835351 | 0.345086 | 0.682304 | 0.034653 | 0.266539 | 0.269634 | 0.373285 | 0.372615 | 0.000000 |
| Qwen3-8B main4 yawcls05 step5000 argmax | 4455 | 4447 | 2.842523 | 0.349674 | 0.658072 | 0.034210 | 0.755817 | 0.838126 | 0.025410 | 0.025365 | 0.000000 |
| Qwen3-8B main4 yawcls05 step5000 spread | 4455 | 4447 | 2.828618 | 0.350016 | 0.658054 | 0.035991 | 0.304150 | 0.306478 | 0.407241 | 0.406510 | 0.000000 |
| room_center_position (paired to final spread) | 4455 | — | 2.499688 | — | — | — | — | — | — | — | — |
| category_mean_position (paired to final spread) | 4455 | — | 2.446585 | — | — | — | — | — | — | — | — |
| category_median_size (paired to final spread) | 4455 | — | — | 0.438754 | — | — | — | — | — | — | — |
| uniform_yaw (paired to final spread) | 4455 | — | — | — | 0.783467 | — | — | — | — | — | — |

可核查来源：
- `/Volumes/harddisk/fastfillout/runs/analysis/final/base-test-minimal-argmax.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/base-test-minimal-spread.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/ctrl-test-minimal-argmax.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/ctrl-test-minimal-spread.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/final-test-minimal-argmax.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/final-test-minimal-spread.json`

### 旧v2 full测试：8620请求，含朴素基线

Minimal=三字段投影; full=完整条件投影. Same input file SHA212fc6bf..., different eligible subsets. main7 max128 objects vs main4 max256; this comparison changes data recipe/capacity, not a pure one-factor ablation. Clean/layout and clean/all are both retained. Old strict includes unknown hard checks and is not amendment027 strict admission. Naive baseline fits evaluation-set leave-one-out and is paired to final-spread label objects, not trained only on training data. Position means show the naive baselines outperform learned heads here. Exact per-metric N and CI preserved; 2000 bootstrap seed0.

完整条件full投影8620请求，不与三字段minimal4455合并；两者同源文件但评测条件及子集不同。main7 max128对象、main4 max256，不能只归因于yaw损失权重。clean/完成和clean/全部均保留；旧strict把硬检查unknown纳入失败，不能套用新版027含义。朴素基线是评估集留一法并配对final-spread有效标签对象，不是训练集统计；其位置误差更好也完整列出。每指标分母与95%CI在CSV，bootstrap2000次seed0。

| method | requested | layouts | bottom_center_error_m | log_size_error | yaw_error_rad | bev_iou | object_collision_rate | object_hard_violation_rate | room_clean_rate | clean_room_rate_incl_failures | strict_ok_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B main7 step6357 argmax | 8620 | 8545 | 3.208097 | 0.348139 | 0.629688 | 0.040431 | 0.672528 | 0.790650 | 0.025629 | 0.025406 | 0.005800 |
| Qwen3-8B main7 step6357 spread | 8620 | 8545 | 3.163790 | 0.348241 | 0.632102 | 0.045155 | 0.388966 | 0.475758 | 0.214043 | 0.212181 | 0.026218 |
| Qwen3-8B main4 yawcls008 step5000 argmax | 8620 | 8595 | 3.221880 | 0.318026 | 0.637227 | 0.038745 | 0.678368 | 0.790849 | 0.018615 | 0.018561 | 0.004872 |
| Qwen3-8B main4 yawcls008 step5000 spread | 8620 | 8595 | 3.194021 | 0.318194 | 0.639921 | 0.041831 | 0.389903 | 0.471799 | 0.209308 | 0.208701 | 0.027378 |
| Qwen3-8B main4 yawcls05 step5000 argmax | 8620 | 8595 | 3.204862 | 0.321373 | 0.600224 | 0.040086 | 0.710309 | 0.809654 | 0.028505 | 0.028422 | 0.004872 |
| Qwen3-8B main4 yawcls05 step5000 spread | 8620 | 8595 | 3.171803 | 0.321637 | 0.603290 | 0.043425 | 0.394509 | 0.478681 | 0.218965 | 0.218329 | 0.026798 |
| room_center_position (paired to final spread) | 8620 | — | 2.743407 | — | — | — | — | — | — | — | — |
| category_mean_position (paired to final spread) | 8620 | — | 2.721142 | — | — | — | — | — | — | — | — |
| category_median_size (paired to final spread) | 8620 | — | — | 0.394503 | — | — | — | — | — | — | — |
| uniform_yaw (paired to final spread) | 8620 | — | — | — | 0.759538 | — | — | — | — | — | — |

可核查来源：
- `/Volumes/harddisk/fastfillout/runs/analysis/final/base-test-full-argmax.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/base-test-full-spread.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/ctrl-test-full-argmax.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/ctrl-test-full-spread.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/final-test-full-argmax.json`
- `/Volumes/harddisk/fastfillout/runs/analysis/final/final-test-full-spread.json`

### 单个训练样本的训练前后诊断

Complete real GPU diagnostic, not self-test; MultiScan::scene_00057_01 training sample only. Before is reconstructed seed42 initialization (no saved pre-training state). After is pilot4steps. Not evidence of held-out generalization.

真实GPU单样本诊断，非self-test；只有MultiScan::scene_00057_01训练样本、3对象。before为按seed42重建的初始化，没有保存的训练前状态；after为4步pilot。不能作为留出集泛化或有效布局质量对照。

| method | rooms | objects | training_loss | bottom_center_error_m | log_size_error | yaw_error_rad | bev_iou |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before | 1 | 3 | 2.862921 | 6.160877 | 0.580686 | 1.722677 | 0.000000 |
| after | 1 | 3 | 2.412009 | 3.745468 | 0.513525 | 1.688001 | 0.000000 |

可核查来源：
- `/Volumes/harddisk/FastFill_v2_multisource_20261006/pilot-fixed-sample-comparison.json`

### RoomGenBench输入场景的交接代理检查

Five scenes,313 objects per model. Two FastFill variants using RoomGenBench inputs, not a RoomGenBench layout-method baseline. Target-bbox proxy tolerance.001m. File explicitly says asset_retrieval=not_attempted, mesh=not_checked, support=unknown, physics=not_checked. OBB overlap fail counts are pairs, not failed-room rates.

5场景、每模型313对象。这是两种FastFill在RoomGenBench输入上的交接代理检查，不是RoomGenBench布局方法基准比较。target_bbox_proxy容差0.001米，OBB失败数是重叠对象对数；资产检索未尝试、网格未检查、支撑unknown、物理未检查，不能解释为真实网格碰撞通过率。

| method | scene | objects | boundary_fail_checks | obb_overlap_fail_pairs | all_fail_checks |
| --- | --- | --- | --- | --- | --- |
| yawcls008 | bathroom | 55 | 0 | 23 | 23 |
| yawcls008 | bedroom | 51 | 1 | 29 | 30 |
| yawcls008 | gym | 32 | 0 | 10 | 10 |
| yawcls008 | livingroom | 53 | 0 | 28 | 28 |
| yawcls008 | restaurant | 122 | 0 | 67 | 67 |
| yawcls05 | bathroom | 55 | 0 | 29 | 29 |
| yawcls05 | bedroom | 51 | 0 | 29 | 29 |
| yawcls05 | gym | 32 | 0 | 11 | 11 |
| yawcls05 | livingroom | 53 | 0 | 35 | 35 |
| yawcls05 | restaurant | 122 | 0 | 93 | 93 |

可核查来源：
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls008/bathroom.handoff/diagnostics.json`
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls008/bedroom.handoff/diagnostics.json`
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls008/gym.handoff/diagnostics.json`
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls008/livingroom.handoff/diagnostics.json`
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls008/restaurant.handoff/diagnostics.json`
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls05/bathroom.handoff/diagnostics.json`
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls05/bedroom.handoff/diagnostics.json`
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls05/gym.handoff/diagnostics.json`
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls05/livingroom.handoff/diagnostics.json`
- `/Volumes/harddisk/fastfillout/runs/roomgenbench/best-main4-cell05-main-20261008a-e5-yawcls05/restaurant.handoff/diagnostics.json`

## 后续unknown-size成果已核对的落盘位置

这些表由仓库结果审查合并展示；此处保留硬盘定位和版本限制，避免重复当作新实验：

- `unknownsize-20261013/system-022-final/test-sharded/eval/`：8647请求的raw/S1/旧checker触发系统；系统checker .890135307042905、strict .3434717242974442。`delivery-final/evaluation/test`与Delivery包为副本。
- `unknownsize-20261013/system-022-final/`：1454对照队列；`compare-016-full-v1`、`compare-016-pilot-v1`包含实际外部模型和harness比较；`frontier-026-v4`为离线策略重放加实际基础输出，不能整体称实际选择性执行。
- `unknownsize-20261013/server-pulls-020`：尺寸先验/LLM预测比较；`server-pulls-021`：尺度/模型实验；N1877、N1009或其它子集要分别列。
- `unknownsize-20261013/visual-024`：A/S FID、mesh等最终视觉；S来自system022旧checker系统，不是strict027。1005共同有效渲染，不能与早期68间交集视觉表混用。
- `unknownsize-20261013/a024/judge-A-S-astra-high`、`a025/judge-SC-astra-high`及`server-pulls-judge-023`：gpt-6-astra high评审；早期gpt-6.1-sol judge是另一版本。
- `unknownsize-20261013/a025/test-eval-geometry/results.json`：amendment-025-test,N174，S→SC checker .9655172413793104→.9827586206896551，strict .5977011494252874→.8908045977011494，OOR .0325330093571717→.016184812153003884。geometry文件judge:null属于该文件时点；后续judge173配对已完成。报告模板里dev/B文字不能覆盖JSON的test protocol。
- `FastFill-Delivery-20261015/evidence/strict-full-test-status.json`写NOT_YET_DELIVERED是2026-10-15打包快照；新a027已在项目仓库completed/eval，不能引用此旧包声称如今未完成。

## 外部论文方法、迁移目录与未完成产物

- 不能只查`unknownsize-20261013/optiscene-014/plan-e1`就下结论：本次同时查了`scratchpad-archive-20261015/{a014,optiscene-014-e1,optiscene-014-e1-r2,e1fix,e2new,e2-r2,e3fix,e3work,review014,full-out}`及tar/zip成员。发现E1/E2/E3请求和暴露计划、MANIFEST、coverage、dry-run及训练resume小测；在这些可访问存档未找到OptiScene014完成布局的summary/metrics。E1版本430→425、E2计划1009等是计划数，不是完成样本数。
- `review014/resume/A/eval_results.json`、`B/eval_results.json`是小adapter训练resume校验，不是场景布局benchmark；`full-out`是scale-curve016相关副本。E2 MANIFEST里Qwen2.5-7B-Instruct是代码配方声明，不是完成训练/评测证据。
- ATISS/DiffuScene出现在`FastFill_v2_20261006/audit/papers/paper-code-verification.md`和review2论文复核文档中，属于文献/代码核验；本次可访问存档未找到它们作为预测方法的完成对照成绩。LayoutGPT/SceneDiffuser/LayoutDiffusion/RoomDesigner等未找到新增完成比较。
- OptiScene、Holodeck、SceneSmith常是数据来源或文件名。必须见生成方法身份和产物后才能列为基线，不能把这些数据源名当已跑方法。
- `FastFill_v2_20261006_review2/audit/review2/roomgenbench-comparison.md`是5场景313实例/184资产条目/154描述、45网页房间/45装配日志/7网格方法的输入和资产接口审核；不是7种布局方法同队列质量评测。上表RoomGen交接实测同样只是两种FastFill变体。
- 旧v2结构化6请求smoke和1请求文本失败、multisource4step单训练样本诊断、Delivery尺寸API的2房间连通性smoke均已找到，但不能当泛化质量对照。

## 使用规则与证据限制

CSV采用长表保留精度、单位相关指标名、逐指标分母、95%CI、JSON selector及来源mtime；JSON保留report_tables与完整候选目录。reference误差按有效标签对象聚合，clean/strict按房间，碰撞/越界是代理度量，各版本不可仅凭同名合并。bootstrap CI来自已有汇总，不是本次重新计算。最终旧v2表同时保存测试/验证24个矩阵的全部20项指标，正文仅展示测试核心指标。所有未完成/未找到结论都严格限定于当前本地存档。

## 镜像一致性与补充范围（本次SHA-256核对）

下表只比较现存对应文件。内容相同的硬盘/仓库路径属于镜像证据，不重复计为独立运行。未在仓库保存JSON/contrasts的组不强行补称一致。

| 硬盘来源目录 | 文件 | 对应仓库目录 | 状态 |
| --- | --- | --- | --- |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/compare-016-full-v1 | levels.csv | compare-016-full-v1 | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/compare-016-full-v1 | contrasts.csv | compare-016-full-v1 | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/compare-016-full-v1 | results.json | compare-016-full-v1 | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/compare-016-full-v1 | report.md | compare-016-full-v1 | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/frontier-026-v4 | levels.csv | frontier-026-v4 | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/frontier-026-v4 | contrasts.csv | frontier-026-v4 | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/frontier-026-v4 | results.json | frontier-026-v4 | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/frontier-026-v4 | report.md | frontier-026-v4 | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/test-sharded/eval | levels.csv | system-022-test-sharded | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/test-sharded/eval | contrasts.csv | system-022-test-sharded | 仓库无该文件 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/test-sharded/eval | results.json | system-022-test-sharded | 仓库无该文件 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/test-sharded/eval | report.md | system-022-test-sharded | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/test-sharded-eval | levels.csv | system-022-test-sharded | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/test-sharded-eval | contrasts.csv | system-022-test-sharded | 仓库无该文件 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/test-sharded-eval | results.json | system-022-test-sharded | 仓库无该文件 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/test-sharded-eval | report.md | system-022-test-sharded | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/cohort-eval | levels.csv | system-022-cohort | 字节一致 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/cohort-eval | contrasts.csv | system-022-cohort | 仓库无该文件 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/cohort-eval | results.json | system-022-cohort | 仓库无该文件 |
| /Volumes/harddisk/fastfillout/unknownsize-20261013/system-022-final/cohort-eval | report.md | system-022-cohort | 字节一致 |

补充文件名检索 `/Volumes/harddisk/fastfill_ir`：实际覆盖21个文件，summary/report/results/eval/compare匹配0。仅为IR数据源文件，未当作方法比较；未读取大型场景JSONL。

补充文件名检索 `/Volumes/harddisk/fastfill_ir_v34`：实际覆盖18个文件，summary/report/results/eval/compare匹配0。仅为IR数据源文件，未当作方法比较；未读取大型场景JSONL。

硬盘候选清单未找到a027命名的完成汇总；仓库`evidence/a027-resume-20261015/completed/eval`存在，不混淆本地两种存档状态。SHA-256全文存于JSON mirror_consistency_checks。
