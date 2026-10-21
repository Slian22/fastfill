# 历史比较结果核查（2026-10-20）

范围是本仓库中 `docs/research/20261013-unknown-size-comparison` 以外的历史结果与文档；新任务目录、外接硬盘和远程服务器由其他审计负责。本次只读取既有文件，核对保存的 JSON/CSV/outcomes/sidecar，没有启动推理、训练、打分或远程任务。通过报告文件名、指标关键字与目录清单定位结果，保留全部识别出的数值报告；原始 Hugging Face 资产、`.claude` 重复工作树、graphify 派生产物和代码依赖不计作独立实验。不声称逐一阅读数十万原始资产。

完整机器可读材料：[JSON](historical-results.json) 保存各比较表、报告原值、SHA256、逐资产表与状态差异；[CSV](historical-results.csv) 平铺已保存报告的全部数值及 JSON Pointer。下表显示适宜审阅的报告精度，JSON 的 snapshots 保留原始数值。`source_paths` 为原文件绝对路径。NaN 改用 JSON null 表达，未改动原文件。

## 证据分层与结论

- **真实历史模型评测，保存报告与逐例输出**：v2 FastFill/Qwen3-8B 与 GPT prompt/harness 同300请求；v2验证/测试；V1.1 SFT→DPO1→final全部12评测单元；布局关系诊断。不得把阶段、解码策略、不同条件/队列合成一条排名。
- **资产任务的历史测量归档**：RoomGenBench有8×184逐资产CSV、每个生成方法184份sidecar、7×5组装场景JSON、原始几何摘要及评估协议。这是7种资产方法+SAGE reference，5房间313实例，非8个单样本，也非自由布局生成。运行收据包含模型版本、seed、耗时和原远程输出路径；当前副本无GLB、PNG或特征载荷，因此只能核验记录一致性，不能独立重算视觉分数。没有足够证据将其称为toy/demo；也不能声称本次重新运行这些方法。
- **版本不一致**：InfinigenV2旧CSV为56 ok/34 fallback/94 failed（90/184覆盖）；当前sidecar为56/94/34（150/184状态可用），60资产发生failed→fallback。既有分数和场景组装计数属于旧快照，不能用新状态替换旧coverage并保留旧视觉分数。支撑分母reference/完整方法146、InfinigenV1 91、V2 71；V2的高支撑分受选择覆盖影响。
- **小样本/运行验证/数据版本**：5房间适配结果、单训练房间前后变化、20步运行pilot、structured/text smoke均保留，但不能外推完整模型胜负。v3.1/v3.2/v3.3是数据/几何版本审计，不代表新模型版本得分。
- **外部布局基线缺口**：OptiScene原论文源码/PDF和继承的`liantian/OptiScene-Qwen-8B-Final`训练链存在，但未建立原作者模型/数据/协议等价性。未找到ATISS、DiffuScene、LayoutGPT的独立推理输出、权重运行收据及本地评测结果；`paper-alignment-20261013.md`里的对照是明确引用的论文数值，不是本项目实跑。这里只说明本次本地审计范围，不能据此排除其他硬盘/机器。
- **视觉与裁判状态**：旧V1.1 `fid-smoke.json`仅2×2048有限特征测试，明写不是FID。GPT pilot仅5房间×2视角渲染，没有judge response/score。RoomGenBench Qwen2.5-VL-7B VQA是指定模型的概率评分，不是GPT裁判；其已存资产视觉分数与上述未完成的布局视觉分数必须区分。

## 分母与可比条件

旧v2共同请求文件SHA256为`e02da65e6844630eba506952ee1f059655552a2d14e1af763eb787cc2d0152ea`。GPT prompt有7请求无完整输出：252/293=86.01%是完整输出中的已知检查clean，252/300=84.00%才是所有请求分母；harness为284/300=94.67%。main7旧step6357与后来硬盘yaw分类模型分开列。所有v2原始报告`target_geometry_valid=0`因支撑/mesh/physics/solver unknown未获认证，`failed_requests`字段不能误解为全都推理失败；另读`inference_failed_requests`。本报告派生clean只是schema完整且没有已知fail/violation，忽略unknown，绝非完整strict通过率。

V1.1原研究明确`diagnostic_only`：对象顺序、类别内ID等继承GT尺寸/位姿排序信息，不能与之后任意unknown-size输入设定直接合并。所有12单元及不利结果保留，尤其dev constrained DPO1−SFT零碰撞接受率下降约1.108个百分点（原cluster bootstrap区间−1.949至−0.264个百分点）；普通房间的改善不能掩盖这一点。尺寸IoU/完整率下降也保留在原始snapshots中。没有匹配old-vs-clean训练数据的消融，不能把阶段变化归因于清洗。

RoomGenBench条件是给定文本、bbox和placement后生成资产，再按bbox组装；Holodeck行是固定top1检索、关闭GPT规划等本地适配，不是原Holodeck完整pipeline。资产指标有每资产4视角、154不同描述的检索候选；room VQA仅5房间。原均值、区间、适用n和失败/fallback均原样归档，跨方法可用支撑对不同。

## 完整比较表

### 历史 v2：同300请求的模型与GPT比较

共同rows SHA256=e02da65e6844630eba506952ee1f059655552a2d14e1af763eb787cc2d0152ea。clean是已保存检查中没有已知fail/violation，忽略unknown；所有行原报告target_geometry_valid=0，支撑/mesh/physics未认证。main7=Qwen3-8B结构化模型step6357，GPT请求model=gpt-6.1-sol/medium；prompt与harness不是相同原始响应的配对修复。此旧模型不等于硬盘后来的yawcls008/05。

| Method | Requests | Complete | Known-check clean / complete | Clean / all requests | Position m | Log-size | Yaw rad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main7 spread | 300 | 300 | 125/300 (41.67%) | 125/300 (41.67%) | 2.7474 | 0.2368 | 0.7145 |
| main7 argmax | 300 | 300 | 8/300 (2.67%) | 8/300 (2.67%) | 2.6864 | 0.2377 | 0.6861 |
| GPT prompt | 300 | 293 | 252/293 (86.01%) | 252/300 (84.00%) | 2.6747 | 0.4117 | 0.7817 |
| GPT harness | 300 | 300 | 284/300 (94.67%) | 284/300 (94.67%) | 2.6233 | 0.3879 | 0.7674 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/comparison-20261007b/llmrows-main7-cell05-main-20261007b-e5/report.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/comparison-20261007b/llmrows-main7-cell05-main-20261007b-e5-argmax/report.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/comparison-20261007b/llm-prompt-300-eval/report.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/comparison-20261007b/llm-harness-300-eval/report.json`

### 历史 v2 验证与测试

位置/尺寸/yaw是各自有标签对象的均值，不是成功房间率；all-unknown strict geometry不通过。test spread report另保存full条件projection。

| Run | Requests | No layout | Condition | Position m | Log-size | Yaw rad | Strict target valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Validation spread | 4056 | 3 | minimal | 2.8960 | 0.2568 | 0.7194 | 0.0 |
| Test spread | 4455 | 5 | minimal | 2.6629 | 0.2963 | 0.7003 | 0.0 |
| Test argmax | 4455 | 5 | minimal | 2.6427 | 0.2964 | 0.6727 | 0.0 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/comparison-20261007b/validation-report.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/comparison-20261007b/test-main7-cell05-main-20261007b-e5/report.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/comparison-20261007b/test-main7-cell05-main-20261007b-e5-argmax/report.json`

### 布局诊断 arrange-comparison-300（1）

请求300，共同有输出293；每格=均值[95%房间bootstrap区间],适用n。仅GT/control的两组不能冒称模型对比。GT派生关系/几何诊断，不等价人工功能评分；各方法使用自身预测尺寸。

| Method | rel_hold | rel_near | rel_pose_if_near | partner_gap | wall_back | wall_back_03 |
| --- | --- | --- | --- | --- | --- | --- |
| gt | 0.7136 [0.6530, 0.7758], n=123 | 0.8390 [0.7947, 0.8819], n=123 | 0.8133 [0.7556, 0.8693], n=120 | 0.4582 [0.3413, 0.5965], n=123 | 0.8891 [0.8596, 0.9170], n=286 | 0.9218 [0.8964, 0.9467], n=286 |
| ff-spread | 0.4330 [0.3727, 0.4988], n=123 | 0.6858 [0.6256, 0.7478], n=123 | 0.6060 [0.5369, 0.6754], n=112 | 0.7728 [0.6235, 0.9490], n=123 | 0.6924 [0.6566, 0.7297], n=286 | 0.9319 [0.9125, 0.9500], n=286 |
| ff-argmax | 0.2535 [0.2001, 0.3118], n=123 | 0.6619 [0.5987, 0.7200], n=123 | 0.3745 [0.3061, 0.4458], n=109 | 0.6326 [0.5118, 0.7681], n=123 | 0.3076 [0.2731, 0.3436], n=286 | 0.4112 [0.3738, 0.4516], n=286 |
| llm-prompt | 0.7304 [0.6726, 0.7840], n=123 | 0.8644 [0.8183, 0.9056], n=123 | 0.8375 [0.7805, 0.8881], n=118 | 0.4271 [0.3338, 0.5312], n=123 | 0.6213 [0.5786, 0.6629], n=286 | 0.8688 [0.8406, 0.8926], n=286 |
| llm-harness | 0.7644 [0.7115, 0.8143], n=123 | 0.8825 [0.8435, 0.9172], n=123 | 0.8652 [0.8128, 0.9104], n=121 | 0.4422 [0.3459, 0.5471], n=123 | 0.6147 [0.5715, 0.6558], n=286 | 0.8756 [0.8491, 0.8997], n=286 |
| room-centre | 0.0432 [0.0175, 0.0758], n=123 | 1.0000 [1.0000, 1.0000], n=123 | 0.0432 [0.0175, 0.0758], n=123 | 0.0000 [0.0000, 0.0000], n=123 | 0.0106 [0.0024, 0.0205], n=286 | 0.0274 [0.0149, 0.0410], n=286 |
| random | 0.2451 [0.2012, 0.2948], n=123 | 0.7529 [0.7009, 0.8060], n=123 | 0.3152 [0.2631, 0.3712], n=117 | 0.3900 [0.2930, 0.5006], n=123 | 0.0109 [0.0023, 0.0221], n=286 | 0.0772 [0.0534, 0.1022], n=286 |
| shuffled | 0.2253 [0.1762, 0.2804], n=123 | 0.5779 [0.5168, 0.6409], n=123 | 0.3716 [0.2997, 0.4443], n=106 | 0.9588 [0.7716, 1.1578], n=123 | 0.3492 [0.3088, 0.3896], n=286 | 0.5350 [0.4931, 0.5754], n=286 |
| yaw-flipped | 0.3414 [0.2829, 0.4004], n=123 | 0.8390 [0.7947, 0.8819], n=123 | 0.3954 [0.3294, 0.4602], n=120 | 0.4582 [0.3413, 0.5965], n=123 | 0.0211 [0.0091, 0.0342], n=286 | 0.0304 [0.0156, 0.0464], n=286 |
| mirrored | 0.7136 [0.6530, 0.7758], n=123 | 0.8390 [0.7947, 0.8819], n=123 | 0.8133 [0.7556, 0.8693], n=120 | 0.4582 [0.3413, 0.5965], n=123 | 0.8891 [0.8596, 0.9170], n=286 | 0.9218 [0.8964, 0.9467], n=286 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-comparison-300/summary.json`

### 布局诊断 arrange-comparison-300（2）

请求300，共同有输出293；每格=均值[95%房间bootstrap区间],适用n。仅GT/control的两组不能冒称模型对比。GT派生关系/几何诊断，不等价人工功能评分；各方法使用自身预测尺寸。

| Method | wall_touch | skewed | faces_wall | floating | collide | out_room |
| --- | --- | --- | --- | --- | --- | --- |
| gt | 0.9276 [0.9022, 0.9508], n=286 | 0.0139 [0.0059, 0.0244], n=292 | 0.0347 [0.0193, 0.0519], n=284 | 0.0255 [0.0137, 0.0397], n=292 | 0.0787 [0.0552, 0.1050], n=292 | 0.0245 [0.0150, 0.0357], n=292 |
| ff-spread | 0.7975 [0.7673, 0.8294], n=286 | 0.0638 [0.0491, 0.0803], n=292 | 0.0226 [0.0099, 0.0379], n=287 | 0.0637 [0.0483, 0.0828], n=292 | 0.1029 [0.0769, 0.1310], n=292 | 0.0166 [0.0051, 0.0303], n=292 |
| ff-argmax | 0.7224 [0.6879, 0.7589], n=286 | 0.0000 [0.0000, 0.0000], n=292 | 0.4912 [0.4525, 0.5276], n=274 | 0.0802 [0.0596, 0.1030], n=292 | 0.4056 [0.3633, 0.4448], n=292 | 0.1755 [0.1456, 0.2085], n=292 |
| llm-prompt | 0.6742 [0.6318, 0.7127], n=286 | 0.0032 [0.0007, 0.0063], n=292 | 0.0185 [0.0087, 0.0301], n=283 | 0.0379 [0.0267, 0.0499], n=292 | 0.1014 [0.0757, 0.1266], n=292 | 0.0166 [0.0091, 0.0262], n=292 |
| llm-harness | 0.6720 [0.6308, 0.7117], n=286 | 0.0023 [0.0005, 0.0049], n=292 | 0.0149 [0.0067, 0.0244], n=283 | 0.0439 [0.0294, 0.0594], n=292 | 0.0702 [0.0514, 0.0892], n=292 | 0.0000 [0.0000, 0.0000], n=292 |
| room-centre | 0.0563 [0.0341, 0.0807], n=286 | 0.0139 [0.0059, 0.0244], n=292 | 1.0000 [1.0000, 1.0000], n=18 | 0.3404 [0.2962, 0.3882], n=292 | 0.8664 [0.8253, 0.9041], n=292 | 0.0034 [0.0000, 0.0086], n=292 |
| random | 0.2532 [0.2174, 0.2883], n=286 | 0.5513 [0.5204, 0.5820], n=292 | 0.5609 [0.5040, 0.6193], n=215 | 0.1464 [0.1204, 0.1745], n=292 | 0.5258 [0.4822, 0.5670], n=292 | 0.0128 [0.0040, 0.0237], n=292 |
| shuffled | 0.4445 [0.4048, 0.4844], n=286 | 0.0432 [0.0307, 0.0571], n=292 | 0.0922 [0.0662, 0.1193], n=259 | 0.0988 [0.0796, 0.1194], n=292 | 0.3243 [0.2886, 0.3609], n=292 | 0.2968 [0.2695, 0.3243], n=292 |
| yaw-flipped | 0.9276 [0.9022, 0.9508], n=286 | 0.0139 [0.0059, 0.0244], n=292 | 0.9800 [0.9671, 0.9913], n=284 | 0.0556 [0.0396, 0.0730], n=292 | 0.0787 [0.0552, 0.1050], n=292 | 0.0245 [0.0150, 0.0357], n=292 |
| mirrored | 0.9276 [0.9022, 0.9508], n=286 | 0.0117 [0.0044, 0.0213], n=292 | 0.0347 [0.0193, 0.0519], n=284 | 0.0255 [0.0137, 0.0397], n=292 | 0.0787 [0.0552, 0.1050], n=292 | 0.0245 [0.0150, 0.0357], n=292 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-comparison-300/summary.json`

### 布局诊断 arrange-comparison-300（3）

请求300，共同有输出293；每格=均值[95%房间bootstrap区间],适用n。仅GT/control的两组不能冒称模型对比。GT派生关系/几何诊断，不等价人工功能评分；各方法使用自身预测尺寸。

| Method | walk_frac | walk_conn | reach | door_blocked | door_any_blocked | doors_reachable |
| --- | --- | --- | --- | --- | --- | --- |
| gt | 0.2959 [0.2727, 0.3210], n=292 | 0.9295 [0.9089, 0.9491], n=292 | 0.8693 [0.8432, 0.8944], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| ff-spread | 0.3094 [0.2843, 0.3346], n=292 | 0.9332 [0.9131, 0.9520], n=292 | 0.8641 [0.8403, 0.8887], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| ff-argmax | 0.3477 [0.3234, 0.3726], n=292 | 0.9365 [0.9159, 0.9546], n=292 | 0.8230 [0.7970, 0.8496], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| llm-prompt | 0.3005 [0.2796, 0.3216], n=292 | 0.9294 [0.9099, 0.9475], n=292 | 0.8762 [0.8535, 0.8994], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| llm-harness | 0.2945 [0.2730, 0.3156], n=292 | 0.9295 [0.9100, 0.9474], n=292 | 0.8828 [0.8613, 0.9052], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| room-centre | 0.3208 [0.2912, 0.3527], n=292 | 0.7234 [0.6861, 0.7612], n=292 | 0.4598 [0.4202, 0.4999], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| random | 0.2166 [0.1939, 0.2407], n=292 | 0.7715 [0.7459, 0.7973], n=292 | 0.6914 [0.6618, 0.7256], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| shuffled | 0.3705 [0.3500, 0.3921], n=292 | 0.9563 [0.9414, 0.9700], n=292 | 0.8954 [0.8760, 0.9153], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| yaw-flipped | 0.2959 [0.2727, 0.3210], n=292 | 0.9295 [0.9089, 0.9491], n=292 | 0.8693 [0.8432, 0.8944], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| mirrored | 0.2962 [0.2729, 0.3212], n=292 | 0.9299 [0.9090, 0.9492], n=292 | 0.8703 [0.8446, 0.8958], n=292 | n/a (n=0) | n/a (n=0) | n/a (n=0) |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-comparison-300/summary.json`

### 布局诊断 arrange-llm300-full-gt（1）

请求300，共同有输出300；每格=均值[95%房间bootstrap区间],适用n。仅GT/control的两组不能冒称模型对比。GT派生关系/几何诊断，不等价人工功能评分；各方法使用自身预测尺寸。

| Method | rel_hold | rel_near | rel_pose_if_near | partner_gap | wall_back | wall_back_03 |
| --- | --- | --- | --- | --- | --- | --- |
| gt | 0.7125 [0.6480, 0.7739], n=126 | 0.8349 [0.7884, 0.8804], n=126 | 0.8164 [0.7561, 0.8697], n=122 | 0.4679 [0.3471, 0.6061], n=126 | 0.8889 [0.8609, 0.9167], n=293 | 0.9212 [0.8965, 0.9461], n=293 |
| room-centre | 0.0421 [0.0173, 0.0727], n=126 | 1.0000 [1.0000, 1.0000], n=126 | 0.0421 [0.0173, 0.0727], n=126 | 0.0000 [0.0000, 0.0000], n=126 | 0.0104 [0.0024, 0.0200], n=293 | 0.0268 [0.0148, 0.0402], n=293 |
| random | 0.2248 [0.1795, 0.2707], n=126 | 0.7358 [0.6807, 0.7900], n=126 | 0.2951 [0.2431, 0.3529], n=120 | 0.4084 [0.2899, 0.5472], n=126 | 0.0106 [0.0022, 0.0220], n=293 | 0.0780 [0.0553, 0.1012], n=293 |
| shuffled | 0.2115 [0.1617, 0.2635], n=126 | 0.5747 [0.5177, 0.6350], n=126 | 0.3316 [0.2669, 0.3992], n=108 | 0.9402 [0.7671, 1.1223], n=126 | 0.3304 [0.2919, 0.3675], n=293 | 0.5308 [0.4917, 0.5701], n=293 |
| yaw-flipped | 0.3333 [0.2754, 0.3922], n=126 | 0.8349 [0.7884, 0.8804], n=126 | 0.3889 [0.3261, 0.4546], n=122 | 0.4679 [0.3471, 0.6061], n=126 | 0.0211 [0.0095, 0.0346], n=293 | 0.0302 [0.0158, 0.0456], n=293 |
| mirrored | 0.7125 [0.6480, 0.7739], n=126 | 0.8349 [0.7884, 0.8804], n=126 | 0.8164 [0.7561, 0.8697], n=122 | 0.4679 [0.3471, 0.6061], n=126 | 0.8889 [0.8609, 0.9167], n=293 | 0.9212 [0.8965, 0.9461], n=293 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-llm300-full-gt/summary.json`

### 布局诊断 arrange-llm300-full-gt（2）

请求300，共同有输出300；每格=均值[95%房间bootstrap区间],适用n。仅GT/control的两组不能冒称模型对比。GT派生关系/几何诊断，不等价人工功能评分；各方法使用自身预测尺寸。

| Method | wall_touch | skewed | faces_wall | floating | collide | out_room |
| --- | --- | --- | --- | --- | --- | --- |
| gt | 0.9271 [0.9041, 0.9498], n=293 | 0.0170 [0.0071, 0.0293], n=299 | 0.0346 [0.0204, 0.0504], n=291 | 0.0251 [0.0136, 0.0384], n=299 | 0.0800 [0.0565, 0.1062], n=299 | 0.0239 [0.0143, 0.0350], n=299 |
| room-centre | 0.0549 [0.0337, 0.0797], n=293 | 0.0170 [0.0071, 0.0293], n=299 | 1.0000 [1.0000, 1.0000], n=18 | 0.3346 [0.2891, 0.3773], n=299 | 0.8696 [0.8328, 0.9064], n=299 | 0.0033 [0.0000, 0.0078], n=299 |
| random | 0.2509 [0.2153, 0.2862], n=293 | 0.5554 [0.5246, 0.5861], n=299 | 0.5513 [0.4933, 0.6059], n=223 | 0.1489 [0.1250, 0.1763], n=299 | 0.5313 [0.4908, 0.5715], n=299 | 0.0125 [0.0042, 0.0234], n=299 |
| shuffled | 0.4369 [0.3980, 0.4742], n=293 | 0.0548 [0.0390, 0.0721], n=299 | 0.0976 [0.0736, 0.1244], n=264 | 0.0963 [0.0788, 0.1165], n=299 | 0.3438 [0.3087, 0.3789], n=299 | 0.3308 [0.3030, 0.3564], n=299 |
| yaw-flipped | 0.9271 [0.9041, 0.9498], n=293 | 0.0170 [0.0071, 0.0293], n=299 | 0.9798 [0.9664, 0.9901], n=291 | 0.0560 [0.0404, 0.0737], n=299 | 0.0800 [0.0565, 0.1062], n=299 | 0.0239 [0.0143, 0.0350], n=299 |
| mirrored | 0.9271 [0.9041, 0.9498], n=293 | 0.0147 [0.0055, 0.0266], n=299 | 0.0346 [0.0204, 0.0504], n=291 | 0.0251 [0.0136, 0.0384], n=299 | 0.0800 [0.0565, 0.1062], n=299 | 0.0239 [0.0143, 0.0350], n=299 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-llm300-full-gt/summary.json`

### 布局诊断 arrange-llm300-full-gt（3）

请求300，共同有输出300；每格=均值[95%房间bootstrap区间],适用n。仅GT/control的两组不能冒称模型对比。GT派生关系/几何诊断，不等价人工功能评分；各方法使用自身预测尺寸。

| Method | walk_frac | walk_conn | reach | door_blocked | door_any_blocked | doors_reachable |
| --- | --- | --- | --- | --- | --- | --- |
| gt | 0.2958 [0.2725, 0.3193], n=299 | 0.9273 [0.9056, 0.9467], n=299 | 0.8673 [0.8421, 0.8913], n=299 | 0.1016 [0.0664, 0.1406], n=218 | 0.1239 [0.0826, 0.1697], n=218 | 0.8624 [0.8165, 0.9083], n=218 |
| room-centre | 0.3265 [0.2986, 0.3556], n=299 | 0.7298 [0.6935, 0.7662], n=299 | 0.4547 [0.4176, 0.4939], n=299 | 0.3326 [0.2729, 0.3945], n=218 | 0.3349 [0.2752, 0.3945], n=218 | 0.6422 [0.5734, 0.7064], n=218 |
| random | 0.2170 [0.1945, 0.2391], n=299 | 0.7648 [0.7398, 0.7897], n=299 | 0.6808 [0.6485, 0.7116], n=299 | 0.4872 [0.4237, 0.5523], n=218 | 0.5092 [0.4450, 0.5734], n=218 | 0.4725 [0.4037, 0.5367], n=218 |
| shuffled | 0.3769 [0.3560, 0.3964], n=299 | 0.9552 [0.9402, 0.9691], n=299 | 0.8888 [0.8696, 0.9072], n=299 | 0.1792 [0.1296, 0.2291], n=218 | 0.2018 [0.1468, 0.2569], n=218 | 0.8853 [0.8440, 0.9266], n=218 |
| yaw-flipped | 0.2958 [0.2725, 0.3193], n=299 | 0.9273 [0.9056, 0.9467], n=299 | 0.8673 [0.8421, 0.8913], n=299 | 0.1016 [0.0664, 0.1406], n=218 | 0.1239 [0.0826, 0.1697], n=218 | 0.8624 [0.8165, 0.9083], n=218 |
| mirrored | 0.2961 [0.2728, 0.3194], n=299 | 0.9276 [0.9063, 0.9465], n=299 | 0.8683 [0.8436, 0.8925], n=299 | 0.6505 [0.5833, 0.7103], n=218 | 0.6743 [0.6055, 0.7339], n=218 | 0.5505 [0.4862, 0.6193], n=218 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-llm300-full-gt/summary.json`

### 布局诊断 arrange-test-doors-300-gt（1）

请求300，共同有输出300；每格=均值[95%房间bootstrap区间],适用n。仅GT/control的两组不能冒称模型对比。GT派生关系/几何诊断，不等价人工功能评分；各方法使用自身预测尺寸。

| Method | rel_hold | rel_near | rel_pose_if_near | partner_gap | wall_back | wall_back_03 |
| --- | --- | --- | --- | --- | --- | --- |
| gt | 0.7453 [0.6958, 0.7957], n=163 | 0.8696 [0.8321, 0.9054], n=163 | 0.8419 [0.7957, 0.8840], n=158 | 0.3659 [0.2739, 0.4741], n=163 | 0.8877 [0.8641, 0.9114], n=298 | 0.9219 [0.9021, 0.9412], n=298 |
| room-centre | 0.0397 [0.0218, 0.0600], n=163 | 1.0000 [1.0000, 1.0000], n=163 | 0.0397 [0.0218, 0.0600], n=163 | 0.0000 [0.0000, 0.0000], n=163 | 0.0151 [0.0050, 0.0285], n=298 | 0.0393 [0.0235, 0.0566], n=298 |
| random | 0.2199 [0.1844, 0.2569], n=163 | 0.7551 [0.7067, 0.7999], n=163 | 0.3045 [0.2577, 0.3501], n=157 | 0.4154 [0.3071, 0.5424], n=163 | 0.0179 [0.0070, 0.0315], n=298 | 0.0699 [0.0527, 0.0881], n=298 |
| shuffled | 0.1774 [0.1462, 0.2120], n=163 | 0.5184 [0.4661, 0.5713], n=163 | 0.3402 [0.2859, 0.3947], n=133 | 0.8698 [0.7265, 1.0250], n=163 | 0.2573 [0.2306, 0.2841], n=298 | 0.4494 [0.4167, 0.4851], n=298 |
| yaw-flipped | 0.3994 [0.3461, 0.4580], n=163 | 0.8696 [0.8321, 0.9054], n=163 | 0.4467 [0.3862, 0.5094], n=158 | 0.3659 [0.2739, 0.4741], n=163 | 0.0252 [0.0124, 0.0404], n=298 | 0.0306 [0.0174, 0.0464], n=298 |
| mirrored | 0.7453 [0.6958, 0.7957], n=163 | 0.8696 [0.8321, 0.9054], n=163 | 0.8419 [0.7957, 0.8840], n=158 | 0.3659 [0.2739, 0.4741], n=163 | 0.8877 [0.8641, 0.9114], n=298 | 0.9219 [0.9021, 0.9412], n=298 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-test-doors-300-gt/summary.json`

### 布局诊断 arrange-test-doors-300-gt（2）

请求300，共同有输出300；每格=均值[95%房间bootstrap区间],适用n。仅GT/control的两组不能冒称模型对比。GT派生关系/几何诊断，不等价人工功能评分；各方法使用自身预测尺寸。

| Method | wall_touch | skewed | faces_wall | floating | collide | out_room |
| --- | --- | --- | --- | --- | --- | --- |
| gt | 0.9239 [0.9039, 0.9421], n=298 | 0.0144 [0.0076, 0.0221], n=300 | 0.0351 [0.0217, 0.0512], n=297 | 0.0271 [0.0174, 0.0386], n=300 | 0.0872 [0.0638, 0.1115], n=300 | 0.0089 [0.0041, 0.0146], n=300 |
| room-centre | 0.0502 [0.0314, 0.0711], n=298 | 0.0144 [0.0076, 0.0221], n=300 | 1.0000 [1.0000, 1.0000], n=26 | 0.3015 [0.2626, 0.3397], n=300 | 1.0000 [1.0000, 1.0000], n=300 | 0.0000 [0.0000, 0.0000], n=300 |
| random | 0.2429 [0.2120, 0.2738], n=298 | 0.5478 [0.5241, 0.5723], n=300 | 0.5482 [0.4973, 0.6012], n=243 | 0.1192 [0.0997, 0.1383], n=300 | 0.6358 [0.6064, 0.6645], n=300 | 0.0025 [0.0000, 0.0061], n=300 |
| shuffled | 0.3923 [0.3625, 0.4230], n=298 | 0.0876 [0.0716, 0.1045], n=300 | 0.1383 [0.1120, 0.1668], n=276 | 0.1113 [0.0926, 0.1304], n=300 | 0.4914 [0.4595, 0.5224], n=300 | 0.3396 [0.3186, 0.3603], n=300 |
| yaw-flipped | 0.9239 [0.9039, 0.9421], n=298 | 0.0144 [0.0076, 0.0221], n=300 | 0.9801 [0.9703, 0.9885], n=297 | 0.0717 [0.0557, 0.0910], n=300 | 0.0872 [0.0638, 0.1115], n=300 | 0.0089 [0.0041, 0.0146], n=300 |
| mirrored | 0.9239 [0.9039, 0.9421], n=298 | 0.0150 [0.0080, 0.0232], n=300 | 0.0351 [0.0217, 0.0512], n=297 | 0.0271 [0.0174, 0.0386], n=300 | 0.0872 [0.0638, 0.1115], n=300 | 0.0089 [0.0041, 0.0146], n=300 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-test-doors-300-gt/summary.json`

### 布局诊断 arrange-test-doors-300-gt（3）

请求300，共同有输出300；每格=均值[95%房间bootstrap区间],适用n。仅GT/control的两组不能冒称模型对比。GT派生关系/几何诊断，不等价人工功能评分；各方法使用自身预测尺寸。

| Method | walk_frac | walk_conn | reach | door_blocked | door_any_blocked | doors_reachable |
| --- | --- | --- | --- | --- | --- | --- |
| gt | 0.2142 [0.1976, 0.2312], n=300 | 0.9145 [0.8960, 0.9319], n=300 | 0.7752 [0.7491, 0.8012], n=300 | 0.0676 [0.0436, 0.0938], n=300 | 0.0967 [0.0633, 0.1300], n=300 | 0.8833 [0.8467, 0.9167], n=300 |
| room-centre | 0.3281 [0.3017, 0.3567], n=300 | 0.7496 [0.7173, 0.7813], n=300 | 0.3403 [0.3157, 0.3664], n=300 | 0.1864 [0.1444, 0.2308], n=300 | 0.2000 [0.1567, 0.2467], n=300 | 0.7100 [0.6600, 0.7600], n=300 |
| random | 0.1440 [0.1303, 0.1580], n=300 | 0.7170 [0.6918, 0.7414], n=300 | 0.5626 [0.5355, 0.5890], n=300 | 0.5194 [0.4675, 0.5732], n=300 | 0.5600 [0.5067, 0.6133], n=300 | 0.3667 [0.3100, 0.4233], n=300 |
| shuffled | 0.3295 [0.3121, 0.3478], n=300 | 0.9408 [0.9255, 0.9555], n=300 | 0.8106 [0.7882, 0.8316], n=300 | 0.1814 [0.1394, 0.2234], n=300 | 0.2067 [0.1600, 0.2567], n=300 | 0.8600 [0.8200, 0.8967], n=300 |
| yaw-flipped | 0.2142 [0.1976, 0.2312], n=300 | 0.9145 [0.8960, 0.9319], n=300 | 0.7752 [0.7491, 0.8012], n=300 | 0.0676 [0.0436, 0.0938], n=300 | 0.0967 [0.0633, 0.1300], n=300 | 0.8833 [0.8467, 0.9167], n=300 |
| mirrored | 0.2141 [0.1976, 0.2310], n=300 | 0.9152 [0.8970, 0.9324], n=300 | 0.7745 [0.7484, 0.8005], n=300 | 0.7576 [0.7094, 0.8024], n=300 | 0.7967 [0.7467, 0.8433], n=300 | 0.4067 [0.3533, 0.4600], n=300 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-test-doors-300-gt/summary.json`

### V1.1 三阶段全部12个评测单元

从冻结摘要中互斥in_dist/held_out加权，失败保留在率分母；OOR条件于完整输出。此项目原始次序/ID继承GT尺寸姿态信息，明确diagnostic_only。普通集提升不能掩盖带约束退步；cleaned与old数据的因果对照未做。

| Stage | Split | Requests | Complete | Raw valid % | Repaired valid % | Zero-collision accepted | Raw OOR | Repaired OOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sft | dev_rooms | 8167 | 8142 | 89.35 | 99.41 | 5193/8167 (63.59%) | 0.090520 | 0.081577 |
| sft | dev_constrained_rooms | 8035 | 8019 | 89.43 | 99.49 | 3100/8035 (38.58%) | 0.092148 | 0.082617 |
| sft | test_rooms | 8647 | 8623 | 89.58 | 99.51 | 5443/8647 (62.95%) | 0.092311 | 0.083388 |
| sft | test_constrained_rooms | 8533 | 8504 | 88.64 | 99.47 | 3334/8533 (39.07%) | 0.093441 | 0.084275 |
| dpo1 | dev_rooms | 8167 | 8140 | 89.16 | 99.46 | 5315/8167 (65.08%) | 0.076832 | 0.069631 |
| dpo1 | dev_constrained_rooms | 8035 | 8005 | 89.12 | 99.44 | 3011/8035 (37.47%) | 0.081274 | 0.069620 |
| dpo1 | test_rooms | 8647 | 8629 | 88.87 | 99.66 | 5592/8647 (64.67%) | 0.074409 | 0.069552 |
| dpo1 | test_constrained_rooms | 8533 | 8510 | 88.41 | 99.59 | 3221/8533 (37.75%) | 0.076459 | 0.069572 |
| final | dev_rooms | 8167 | 8132 | 90.34 | 99.38 | 5459/8167 (66.84%) | 0.058714 | 0.058374 |
| final | dev_constrained_rooms | 8035 | 7997 | 90.19 | 99.33 | 3109/8035 (38.69%) | 0.059116 | 0.058112 |
| final | test_rooms | 8647 | 8615 | 89.92 | 99.51 | 5717/8647 (66.12%) | 0.061925 | 0.060773 |
| final | test_constrained_rooms | 8533 | 8497 | 90.10 | 99.50 | 3291/8533 (38.57%) | 0.061941 | 0.061120 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261011-v11-clean-dpo/evaluation_full/research-summary.json`

### V1.1 下游5房间：原版与clean final

每版5房间/313实例，非大样本benchmark。repaired_valid不要求main碰撞为0；reference_check中GT也有unknown/失败。两版均4/5mesh生成标记成功；不可改称自由布局外部方法胜负。

| Version | Room | Objects | Parsed | Raw main | Repaired main | Support fail | Repaired valid | Mesh success | Reference ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roomgenbench_v1_1 | bathroom | 55 | True | 1 | 0 | 1 | False | True | False |
| roomgenbench_v1_1 | bedroom | 51 | True | 2 | 0 | 1 | False | False | False |
| roomgenbench_v1_1 | gym | 32 | True | 0 | 0 | 1 | False | True | False |
| roomgenbench_v1_1 | livingroom | 53 | True | 3 | 2 | 0 | True | True | False |
| roomgenbench_v1_1 | restaurant | 122 | True | 10 | 1 | 0 | True | True | False |
| roomgenbench_clean_final | bathroom | 55 | True | 4 | 0 | 5 | False | True | False |
| roomgenbench_clean_final | bedroom | 51 | True | 5 | 1 | 3 | False | False | False |
| roomgenbench_clean_final | gym | 32 | True | 4 | 0 | 0 | True | True | False |
| roomgenbench_clean_final | livingroom | 53 | True | 6 | 3 | 0 | True | True | False |
| roomgenbench_clean_final | restaurant | 122 | True | 8 | 0 | 0 | True | True | False |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261011-v11-clean-dpo/roomgenbench_v1_1/summary.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261011-v11-clean-dpo/roomgenbench_clean_final/summary.json`

### RoomGenBench 历史资产比较（1）

8行=7方法+SAGE reference；每方法184资产、154描述，5房间共313实例；不是N=8单样本。资产指标均值及原95%区间；room_vqa为5房间；support只用可用支撑对，分母因方法不同。既有逐资产CSV和sidecar证明保存了历史运行记录；当前仓库没有原mesh/render/features，未重算核验。InfinigenV2评分与60份当前sidecar状态漂移，保留旧评分快照。

| Method | coverage | vqa_desc | vqa_cat | siglip | r1 | r5 |
| --- | --- | --- | --- | --- | --- | --- |
| sage_gt | 100.000000 [100.000000, 100.000000] | 0.505078 [0.453270, 0.557835] | 0.552098 [0.498572, 0.606762] | 13.241792 [12.660375, 13.819131] | 44.021739 [36.956522, 51.086957] | 79.891304 [73.913043, 85.326087] |
| infinigen_v1 | 72.282609 [65.760870, 78.804348] | 0.224769 [0.179233, 0.269952] | 0.327758 [0.275586, 0.380993] | 8.831425 [8.041663, 9.601923] | 19.565217 [14.130435, 25.543478] | 52.173913 [45.108696, 59.782609] |
| infinigen_v2 | 48.913043 [41.304348, 55.434783] | 0.175642 [0.131017, 0.221950] | 0.241724 [0.191308, 0.290318] | 7.769677 [6.920586, 8.552597] | 13.043478 [8.152174, 17.404891] | 36.413043 [29.347826, 43.478261] |
| holodeck_retrieval | 100.000000 [100.000000, 100.000000] | 0.320924 [0.272315, 0.369507] | 0.434572 [0.381931, 0.491455] | 10.781739 [10.247224, 11.309776] | 33.152174 [26.630435, 39.673913] | 69.021739 [62.500000, 75.543478] |
| shap_e | 100.000000 [100.000000, 100.000000] | 0.345592 [0.300074, 0.391624] | 0.406000 [0.349751, 0.461350] | 11.952159 [11.354495, 12.550660] | 33.152174 [26.630435, 40.217391] | 71.195652 [64.673913, 77.717391] |
| trellis_text | 100.000000 [100.000000, 100.000000] | 0.496688 [0.446755, 0.543945] | 0.503881 [0.448833, 0.557082] | 12.811791 [12.247030, 13.356858] | 42.934783 [35.869565, 50.000000] | 78.260870 [72.282609, 84.239130] |
| embodiedgen_v2 | 100.000000 [100.000000, 100.000000] | 0.583229 [0.532897, 0.632666] | 0.527686 [0.474916, 0.576635] | 15.484890 [14.936581, 16.038860] | 54.891304 [47.282609, 62.500000] | 91.847826 [87.500000, 95.652174] |
| hunyuan3d_21 | 100.000000 [100.000000, 100.000000] | 0.606988 [0.555831, 0.655661] | 0.510337 [0.456091, 0.559660] | 15.254747 [14.678178, 15.821804] | 52.173913 [44.565217, 59.782609] | 91.304348 [86.956522, 95.108696] |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/results/eval/scores.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/results/eval/per_asset.csv`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/bench/CONTRACT.md`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/README.md`

### RoomGenBench 历史资产比较（2）

8行=7方法+SAGE reference；每方法184资产、154描述，5房间共313实例；不是N=8单样本。资产指标均值及原95%区间；room_vqa为5房间；support只用可用支撑对，分母因方法不同。既有逐资产CSV和sidecar证明保存了历史运行记录；当前仓库没有原mesh/render/features，未重算核验。InfinigenV2评分与60份当前sidecar状态漂移，保留旧评分快照。

| Method | support | room_vqa | dino_gt | fscore_2 | fscore_5 | chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| sage_gt | 100.000000 [100.000000, 100.000000], n=146 | 0.682334 | 1.000000 [1.000000, 1.000000] | 0.999791 [0.999715, 0.999858] | 0.999998 [0.999993, 1.000000] | 0.005241 [0.005083, 0.005403] |
| infinigen_v1 | 74.725275 [64.835165, 82.445055], n=91 | 0.341663 | 0.402411 [0.378364, 0.425465] | 0.563431 [0.519047, 0.608189] | 0.770756 [0.733875, 0.806633] | 0.033644 [0.030001, 0.037485] |
| infinigen_v2 | 98.591549 [95.774648, 100.000000], n=71 | 0.318877 | 0.371168 [0.344901, 0.395739] | 0.534877 [0.492484, 0.578696] | 0.769109 [0.734055, 0.802994] | 0.034035 [0.030534, 0.037767] |
| holodeck_retrieval | 81.506849 [75.342466, 87.671233], n=146 | 0.466180 | 0.463290 [0.437776, 0.489338] | 0.581801 [0.544902, 0.618911] | 0.821827 [0.793525, 0.847620] | 0.028856 [0.026172, 0.031922] |
| shap_e | 82.876712 [76.712329, 88.356164], n=146 | 0.603145 | 0.469723 [0.445700, 0.492090] | 0.609168 [0.572535, 0.644252] | 0.835199 [0.808834, 0.860161] | 0.026491 [0.024067, 0.029183] |
| trellis_text | 83.561644 [77.397260, 89.726027], n=146 | 0.527263 | 0.604060 [0.578416, 0.630077] | 0.679770 [0.646358, 0.712421] | 0.893916 [0.873093, 0.912055] | 0.021316 [0.019456, 0.023410] |
| embodiedgen_v2 | 86.301370 [80.821918, 91.780822], n=146 | 0.582262 | 0.517655 [0.490881, 0.544234] | 0.653482 [0.616382, 0.688231] | 0.865484 [0.837031, 0.890491] | 0.023572 [0.021408, 0.026155] |
| hunyuan3d_21 | 84.246575 [78.082192, 89.726027], n=146 | 0.649080 | 0.470067 [0.442871, 0.495912] | 0.610629 [0.573011, 0.647416] | 0.835554 [0.806738, 0.861508] | 0.026498 [0.024057, 0.029117] |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/results/eval/scores.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/results/eval/per_asset.csv`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/bench/CONTRACT.md`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/README.md`

### 资产状态漂移与归档完整性

原始scores/CSV的coverage包括fallback。InfinigenV2旧评分有90/184可用，当前sidecar150/184可用；不能用新状态修改旧视觉成绩。当前目录GLB/PNG均不在此仓库副本，不能独立重跑。

| Method | Scored CSV ok/fallback/failed | Current sidecar ok/fallback/failed | Different status | Local asset GLBs | Support n |
| --- | --- | --- | --- | --- | --- |
| sage_gt | 184/0/0 | n/a reference | 0 | 0 | 146 |
| infinigen_v1 | 103/30/51 | 103/30/51 | 0 | 0 | 91 |
| infinigen_v2 | 56/34/94 | 56/94/34 | 60 | 0 | 71 |
| holodeck_retrieval | 177/7/0 | 177/7/0 | 0 | 0 | 146 |
| shap_e | 184/0/0 | 184/0/0 | 0 | 0 | 146 |
| trellis_text | 184/0/0 | 184/0/0 | 0 | 0 | 146 |
| embodiedgen_v2 | 184/0/0 | 184/0/0 | 0 | 0 | 146 |
| hunyuan3d_21 | 184/0/0 | 184/0/0 | 0 | 0 | 146 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/results/eval/per_asset.csv`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/RoomGenBench/results/eval/scores.json`

### 历史小样本与数据比较

这类结果已完成，但仅支持声明范围，不与主模型全量成绩混排。完整原数值在JSON snapshots中。

| Comparison | N / scope | Completed evidence | Meaning |
| --- | --- | --- | --- |
| v3.1 → v3.2 | train 144140→144150; dev8167/test8647 unchanged | added10; changed26 common; split transitions0 | 数据版本差异，不是模型效果 |
| v3.2 GT quality | train144150/dev8167/test8647 | valid134329/7613/8062; inside1mm77542/4459/4814 | 数据GT本身包含几何违规 |
| MultiScan fixed-sample before/after | 1 training room;3 objects | position6.160877→3.745468m; log-size .580686→.513525 | 训练样本诊断；初始化是重建，非存档训练前权重 |
| Qwen3-8B direct-bbox pilot | 20 steps; train32/val16/test16 | 16 parsed;1 bbox no-failed-check;0 strict target valid | 运行/梯度/保存恢复；非正式训练效果 |
| Early structured / text smoke | structured6 vs text1 | structured parsed6; text failed1; no fair shared full cohort | 不可用于架构优越结论 |
| FID feature smoke | 2 features ×2048 | all_finite=true; explicitly not dataset FID | 不是FID成绩 |
| GPT pilot directory | 5 rooms ×2 saved views | render-manifest and10 PNG; no judge score artifact | 已渲染，不代表GPT裁判完成 |

原始来源：

- `/Users/slian/Desktop/3D/Worldedge/OptiScene/.release/v3.2/evidence/dataset-scan-summary.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/.release/v3.2/evidence/version-comparison.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/.release/v3.3/evidence/recheck-front/geometric_distribution.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/multisource-20261006/pilot-fixed-sample-comparison.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/server-direct-bbox-20261006/server-evidence/outputs/pilot-summary.json`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261011-v11-clean-dpo/paper-metrics-20261013/fid-smoke.json`

### V1.1 尺寸与碰撞诊断：原始分层全量

逐个保留原始分层值，不跨层重算尺寸均值；不存在的dev held_out不列。尺寸IoU下降、未完整输出以及raw/repaired碰撞均保留。碰撞百分比沿用原冻结指标定义，完整数与请求分母另列。此为diagnostic_only的V1.1输入协议，不是后来任意unknown-size任务。

| Stage / split | Stratum | Requests | Complete | Size IoU | Raw main % | Repaired main % | Raw severe % | Repaired severe % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dpo1/dev_constrained_rooms | in_dist | 8035 | 8005 | 0.596925 | 35.47 | 1.72 | 20.86 | 1.16 |
| dpo1/dev_rooms | in_dist | 8167 | 8140 | 0.595919 | 34.28 | 1.67 | 20.09 | 1.23 |
| dpo1/test_constrained_rooms | in_dist | 8125 | 8102 | 0.599458 | 34.89 | 1.64 | 20.76 | 1.31 |
| dpo1/test_constrained_rooms | held_out | 408 | 408 | 0.422211 | 38.48 | 3.19 | 22.30 | 2.94 |
| dpo1/test_rooms | in_dist | 8238 | 8222 | 0.600214 | 33.31 | 1.67 | 19.69 | 1.39 |
| dpo1/test_rooms | held_out | 409 | 407 | 0.419406 | 37.10 | 3.93 | 22.85 | 3.69 |
| final/dev_constrained_rooms | in_dist | 8035 | 7997 | 0.595085 | 28.94 | 0.90 | 14.74 | 0.63 |
| final/dev_rooms | in_dist | 8167 | 8132 | 0.593129 | 27.95 | 0.82 | 14.61 | 0.59 |
| final/test_constrained_rooms | in_dist | 8125 | 8091 | 0.596179 | 29.16 | 0.83 | 15.19 | 0.59 |
| final/test_constrained_rooms | held_out | 408 | 406 | 0.403616 | 33.00 | 4.93 | 18.97 | 4.93 |
| final/test_rooms | in_dist | 8238 | 8208 | 0.596184 | 27.90 | 0.95 | 14.27 | 0.66 |
| final/test_rooms | held_out | 409 | 407 | 0.407622 | 30.71 | 3.19 | 16.71 | 2.95 |
| sft/dev_constrained_rooms | in_dist | 8035 | 8019 | 0.603760 | 39.64 | 3.17 | 25.79 | 2.64 |
| sft/dev_rooms | in_dist | 8167 | 8142 | 0.602618 | 38.01 | 3.22 | 24.71 | 2.85 |
| sft/test_constrained_rooms | in_dist | 8125 | 8099 | 0.604643 | 38.54 | 3.42 | 24.21 | 2.84 |
| sft/test_constrained_rooms | held_out | 408 | 405 | 0.446196 | 40.74 | 3.46 | 24.69 | 3.46 |
| sft/test_rooms | in_dist | 8238 | 8218 | 0.604559 | 37.94 | 3.21 | 23.91 | 2.79 |
| sft/test_rooms | held_out | 409 | 405 | 0.444726 | 40.25 | 3.46 | 25.43 | 3.21 |

原始来源：`/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261011-v11-clean-dpo/evaluation_full/research-summary.json`

## 未作为新增独立比较计数的材料

早期hungarian、review3 portable/structured/text、audit-four存档副本可能重用同一tiny cohort；逐个保存存在的report，不能相加为总测试N。数值快照36份、报告表18张，这是归档文件/表数量，不是独立实验数量；清单保留376个可追溯元数据文件。论文PDF、proposal、任务注释、源代码下载、渲染清单和数据版本不能单独证明外部方法已运行。

## 历史布局指标字典

源：`/Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/eval-sets-20261009/arrange-comparison-300/report.md`。所有格按适用房间取均值、2000次房间bootstrap区间；n是适用房间数，不是对象/关系数量。下列定义只陈述归档报告明确的信息，未回收到评分实现的细节列为unknown，尤其不可把floating改称物理支撑失败、把collide改称strict通过率、把out_room改称新任务OOR。

| 字段 | 原始标签 | 中文 | 方向 | 已核实定义 | 未核实实现细节 |
|---|---|---|---|---|---|
| rel_hold | mined relations hold | 挖掘关系满足率 | up | 每房间挖掘出的A/B关系同时满足距离、允许方位和需要时的朝向；报告模板为最近伙伴B，距离阈值max(训练GT间距80分位,0.3m)。 | Gap geometry, exact side/facing angle implementation, and pooling details are not recovered from scorer source. |
| rel_near | partner within mined distance | 伙伴在挖掘距离内 | up | 适用关系的伙伴处于挖掘距离阈值内，尚不要求方位与朝向。 | Exact distance geometry is unverified. |
| rel_pose_if_near | pose ok when near (side+facing) | 已接近时方位与朝向满足 | up | 伙伴已在距离阈值内时，方位与朝向满足的条件比例。 | Exact angular thresholds and near-instance aggregation are unverified. |
| partner_gap | median partner gap m | 伙伴间距中位数（米） | down | 原标签为伙伴间距中位数（米），汇总按房间取均值。 | Exact box-gap geometry is unverified; smaller gaps alone do not establish functional quality. |
| wall_back | wall-type back to wall | 应靠墙类别背部靠墙 | up | 历史wall-type对象的背部靠墙分数。 | Wall-type membership, back-distance and orientation thresholds are unknown from the saved report. |
| wall_back_03 | wall-type back within 0.3 m | 应靠墙类别背部距墙0.3米内 | up | wall-type对象背部距墙0.3米内的历史分数。 | Wall-type membership and exact distance-to-back measurement are unverified. |
| wall_touch | wall-type touching wall | 应靠墙类别接触墙 | up | wall-type对象接触墙的历史分数。 | Contact tolerance and wall-type membership are unknown from the saved report. |
| skewed | yaw off wall axes >5 deg | yaw偏离墙轴超过5度 | down | yaw偏离墙轴超过5度的历史比例。 | Wall-axis extraction, symmetry treatment, and within-room object exclusions are unverified. |
| faces_wall | fronted item faces wall | 有正面的物体面向墙 | down | 被指定正面方向的对象朝向墙的历史分数。 | Fronted-object definition, distance and facing thresholds are unverified. |
| floating | floating mid-room | 房间中央游离（原始标签） | down | 仅确认历史标签为floating mid-room；报告不足以确认这代表垂直悬空或支撑高度违规。 | Exact floating criterion is unknown; do not relabel as physical support failure. |
| collide | collides >20% | 碰撞超过20%（原始阈值） | down | 历史碰撞指标使用超过20%的阈值。 | Overlap numerator, normalization denominator, object-pair exclusions, and 2D/3D choice are unverified; do not equate to strict collision-free rate. |
| out_room | out of room >10% | 越出房间超过10%（原始阈值） | down | 历史越界指标使用超过10%的阈值。 | Outside-area/volume geometry and its within-object denominator are unverified; not interchangeable with later OOR. |
| walk_frac | largest walkable / floor | 最大可行走区域占地面比例 | up | 原标签是最大可行走区域除以地面面积。 | Agent radius, raster resolution, height filtering, and exact floor mask are unknown from saved report. |
| walk_conn | walkable connectivity | 可行走区域连通性 | up | 历史可行走区域连通性分数。 | Exact normalization and connectivity rule are unknown; do not assume largest-component/total without scorer evidence. |
| reach | items reachable | 可达对象比例 | up | 历史对象可达性分数。 | Object denominator, reach distance, path start, and navigability model are unverified. |
| door_blocked | doors with a blocked clear zone (share) | 门前净空被挡比例 | down | 门的净空区域被挡的历史比例。 | Clearance dimensions and blockage geometry are unverified. |
| door_any_blocked | room has a blocked door | 房间存在被挡门 | down | 房间任意一扇门被挡的历史指示量。 | Clearance/blockage implementation is unverified. |
| doors_reachable | every door reachable | 每扇门均可达 | up | 房间所有门均可达的历史指示量。 | Navigation seed, door access target, and reachability rule are unverified. |
