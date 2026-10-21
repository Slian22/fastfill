# 仓库比较结果全量审计（2026-10-20）

已逐目录覆盖 results/ 的34个family，并收入 evidence/ 中a027完成结果及agency核验。源数据只读；本审计没有重跑模型、几何计算或bootstrap。

无损导出：87份CSV、61,295行；89份JSON完整保存。另提供50张总体小表、669行用于双语附录。

## 可用于正文的短总览

这些结果包含真实外部LLM比较，并非只有本模型消融；但尚未找到已完成的ATISS、DiffuScene、LayoutGPT或修订014官方OptiScene配方的同基准评分。所有后续系统结论来自已暴露的历史划分，不能改称独立盲测或追溯性预注册。

| Family | 主要范围与结果 | 解释 |
|---|---|---|
| 原始8B J/P | SFT全8647：GSR J 0.388227、P 0.389615；共同完整8610间size logMAE 0.286077/0.281682 | 原始共同主要终点均未确证J优势；size Holm p=0.066，后续高抽样敏感性接近0.05边界 |
| 8B DPO及选择 | 官方超参DPO1：GSR J 0.213253/P 0.092980，完整率0.882734/0.496241；selected J保留SFT，P rpo-DPO1 GSR 0.398057 | 包含退化结果；DPO2全测试未测，不能写成零；selected是后修订结果 |
| 八外部代理自选尺寸 | N1454；raw→harness GSR：L .934663→.934663，O1a 0→0，O1b .042641→.110041，O2 .808116→.927785，O3 .211142→.241403，O4 .687070→.687070，O6 .700138→.700825，O7 .579092→.592160 | 尺寸自由、首答/最多三次尝试分开；O5引擎失败未运行；全部行含完整率及尺寸误差在下方 |
| 固定尺寸公平输入 | 全1454的O2/O4/O6/O7与8B P、300试点加L；每方法均列自选/先验尺寸、首答/多轮、raw/S1 | 不是给所有八代理补造固定尺寸结果；已实跑者全列，未实跑者不填 |
| S1/B8 | 开发8167：J strict .173258→.253092（S1）→.256030（B8+S1）；P .198849→.291539→.311865 | S1采纳；B8相对贪心+S1增益均低于.03门槛而未采纳 |
| 8B偏好归因 | 600开发，RFT两种子strict .270000/.271667；on-policyDPO .328333/.325000，但完整率降至.941667/.946667；pert .235000 | 预设a-prime及strict双终点要求不能用单个有利strict值代替；非27B归因 |
| 最终27B选择 | 600开发+3000确认；27B确认GSR .443667 vs vLLM 8B R .391000；全测试 .443391 vs .399329 | op13开发GSR更高但关系保持护栏失败；模型选择保留不利证据 |
| 最新27B全测试四阶段 | N8647 strict：raw .216144 →S1 .291315 →旧checker-trigger .343472 →新strict-trigger .494160；GSR .443391/.742223/.890135/.895455 | 027实际完成且0API；只新strict触发较旧系统增15.07个百分点，raw/S1复用不可累加 |
| Frontier A/B/C | A严格触发strict .753333 vs BIG .723333；B-harness .724897 vs BIG .700825；C .643333 vs BIG .620000 | 全部为保存候选回放；各策略全列；C和B有房间/房屋组重叠，非独立验证 |
| 无思考控制 | 训练27B−O2nt的strict差：A strict-trigger +.016667，但checker-trigger −.190000；B-harness +.021321但checker-trigger −.157497 | 不隐去不利触发策略；SYSTEM-022行为旧训练系统参考 |
| Critic | 开发246 strict .390244→.695122；测试174的旧S→SC .597701→.890805；测试judge配对173间FL差+.768786 [.601796,.948795] | 实际最多两轮编辑；开发box views、测试Blender，非统一视觉协议；无全8647critic评估 |
| FID/mesh | J/P N1005 FID22.3362/23.9664；A/旧S FID22.8827/22.7857；mesh-free500 A578/1005、S779/1005 | 各代理交集不同需匹配J/P；不得把旧S视觉值赋给新027 |
| 双judge | 两judge共同675case，overall Spearman .904245；两者排序L>O6>J>P；A/S各173成功，overall6.582659/6.828902 | overall并非保证固定权重；原始三个子分及失败数全部保留，同家族bias未排除 |
| 尺寸单独预测/缩放压力 | 1877尺寸集合：prior MAE .267445，L-size .327269，O2-size .408010（完整1876）；L-joint仅1063 | 尺寸集合不同不能直接跨行归因；缩放曲线是固定姿态压力测试，非新生成方法 |

## 指标与分母

- GSR / checker：完整输出且冻结检查器无边界、支撑、顶棚、main/severe碰撞及fixed blocking失败；不是简单check.valid。
- strict / downstream：本地严格准入规则移植。test-apply-012及compare-016的accepted表示checker与strict同时通过；而ablation/021/025中accepted表示strict。列名同名不代表同义。
- required={}去掉A0计数及required相关A3/A4检查，不只是“忽略类别”；保留在审计附表，不能作为严格准入替代。
- OOR为yaw忽略的轴对齐可动物体footprint两两交叠面积之和/各物体面积之和，非越界率，可大于1；对每臂完整房间取均值。
- logMAE及sizeIoU只比较尺寸；旧J/P主对比使用共同完整集，系统层级表通常用各自完整集，因此均值不同不必然矛盾。
- 全请求成功率将失败计入分母；FID、judge、mesh使用各自可用交集，需始终附N/失败数。
- a-prime、scale-curve为固定姿态缩放压力测试；不声称等同重新生成的无偏布局。

## 历史版本、漏项及歧义

- report-v2-JPL-001→002是报告版本；selected的相同SFT J、stage-table复用结果；frontier-v3→v4增加C及修订分析。它们均非独立重复。
- a021 raw输出、system022的raw/S1、a027的raw/S1相同；visual024复用011参考与J/P特征；a025-dev的geometry和完整results也是两个完成时点。
- findings.md头部最后保存时间2026-10-15 03:40 UTC：014仍写DPO1训练中；结果目录没有014完整评分。此为本地保存状态，非现在服务器状态。修订014应标“OptiScene code recipe (README, Qwen2.5-7B-Instruct)”，非论文Qwen3-8B配方或作者权重。
- ATISS/DiffuScene/LayoutGPT未找到共同基准运行证据，只能作为文献背景；已有八种LLM实跑比较不能被误说成“没有对比”。
- 旧selected/DPO1 T1中的p_holm字段原样保存，但修订011限制确认性Holm为原始SFT共同主要终点；不得追溯扩张预注册。
- selected P的strict有一间历史差异：test-apply-012为1728/8647，families015为1729/8647。本审计未重跑，未消除差异；两种版本分开保存。
- a025-test/report-geometry.md误留dev及B=27B+S1标签，且judge待定文字过期。准确测试几何从硬盘amendment-025-test JSON补入，judge由同目录完成结果支持。
- O4自选尺寸OOR raw .038702→harness .121646，O7 .042276→.092827：原始记录确有恶化，完整集/极值变化不能被择优隐藏。
- frontier无思考report的通用local文字仍沿用训练27B；实际O2nt控制及复用SYSTEM-022需按数据分别解释。
- bootstrap文件p=0只表示有限抽样无反向抽样，不是精确数学零。历史多次开发、共享房屋组、单训练种子限制仍存在。

## 完整导出与源文件定位

`repo-results.csv` 每行仅含 family、source绝对路径、row_number（表头为1的逻辑记录序号）、row_json（原始全部列，数值仍为原始字符串）。不同family的同名指标没有横向拼成一个伪统一schema。
`repo-results.json` 的 csv_tables 保存原始列顺序及每行字典，json_documents保存全部原始JSON；report_tables含可直接生成双语附录的title/title_zh、note/note_zh、columns、rows、source_paths。下方小表数值只为阅读作最多6位显示，原始精度在JSON/CSV。

补充来源只有一份已由硬盘代理核实的critic测试几何JSON，它只用于report_tables，不混入仓库87CSV/89JSON计数。

- **a021-selection** — 27B Arm A vs 8B R=rpo-dpo1 vs on-policy13; dev600 + confirm3000; actual serving engines differ and are recorded; selection/confirmation, not all-test. [final-report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a021-selection/final-report.md); 4 files.
- **a021-test** — 27B A and 8B R on vLLM, full test 8647; actual prediction, prior sizes. R checker=0.3993292471 differs from earlier HF selected P=0.3980571296; do not replace one with the other. [s1-report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a021-test/s1-report.md); 4 files.
- **a025-dev** — Critic actual API edits on dev246, before=27B+S1; judge paired244/246; box views; adoption on dev. [report-geometry.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a025-dev/report-geometry.md); 4 files.
- **a025-test** — Critic actual API edits on test174, before=old checker-trigger system; judge paired173/174, Blender views; not full8647. [report-geometry.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a025-test/report-geometry.md); 4 files.
- **ablation-016** — 8B placement dev600, R/rft/pert/on-policyDPO2; seeds42/43 for key arms; matched optimisation budgets; not27B and not test. [ablation-report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/ablation-016/ablation-report.md); 2 files.
- **agents-sft-paired** — 8 actual agents, J-input unknown/self-chosen sizes, raw first/harness up to3 attempts, paired against8B SFT; report.md + manifests only; later selected comparator differs. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/agents-sft-paired/L/report.md); 16 files.
- **audit-sft-20261014** — Engineering/scientific audit report, not a model evaluation arm. [agency-audit-wf_b7ae99d6-8df.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/audit-sft-20261014/agency-audit-wf_b7ae99d6-8df.json); 1 files.
- **compare-016-full-v1** — 1454 cohort; 8B selected P and O2/O4/O6/O7, self sizes vs fixed prior sizes, raw/harness and raw/S1. Re-generation with prior sizes is actual inference; all subgroup/contrast rows preserved. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/compare-016-full-v1/report.md); 5 files.
- **compare-016-pilot-v1** — 300-room fixed-size pilot includes GPT L plus open agents and8B selected P; not the15-family300 scope C. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/compare-016-pilot-v1/report.md); 5 files.
- **decision-012** — Dev8167 deterministic S1 and best-of8; actual sampled candidates/replayed deterministic repair. S1 adopted; B8 not adopted at locked+0.03 downstream threshold. [MANIFEST.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/decision-012/MANIFEST.json); 2 files.
- **delivery-010-001** — Historical8B selected adapter+base delivery metadata; not current27B package. [MODEL_CARD.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/delivery-010-001/MODEL_CARD.md); 4 files.
- **delivery-022-final** — Frozen27B package; measured full-test old checker trigger and cohort GPT/O2; recommended strict trigger at packaging was exploratory replay. a027 later completed separately. [MODEL_CARD.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/delivery-022-final/MODEL_CARD.md); 5 files.
- **dev-selection-010** — 8B J/P SFT,e1/rpo DPO candidates dev600; official-hparam collapsed candidates report-only; selection J=SFT P=rpo-dpo1. [MANIFEST.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/dev-selection-010/scores/MANIFEST.json); 7 files.
- **families-015** — 15 room-family stratification of8B selected J/P and agent comparisons; exploratory, separate fulltest/cohort populations; no new inference. [mapping.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/families-015/mapping.md); 6 files.
- **fid-011-visual-002** — Actual rendered FID/KID; primary reference∩J∩P1005; each agent intersection differs (O2998 etc); source-specific cohorts must not be pooled blindly. [fid-summary.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/fid-011-visual-002/fid-summary.json); 1 files.
- **frontier-026-ablation-o2nt** — Exploratory offline replay with untuned27B no-thinking local arm replacing trained27B; A300 GPT and B1454 O2 raw/harness; shared selectors/S1, not fresh end-to-end. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-ablation-o2nt/report.md); 4 files.
- **frontier-026-v3** — Historical offline replay A300/B1454 only, superseded by v4 including C; retain as historical not independent replication. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-v3/report.md); 6 files.
- **frontier-026-v4** — Exploratory offline replay A300pilot GPT, B1454 O2 raw/harness, C300(15families×20)GPT; LOCAL/ESC-CHECK/ESC-STRICT/ESC-ANY/BIG and savedSYSTEM022; prior sizes; policies selected after earlier evidence, C overlaps35UID/138rooms in89prior groups. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-v4/report.md); 6 files.
- **gpt-judge-023-astra-high** — gpt-6-astra high judge; actual calls on174selected each of8B J/P,L,O6; counts differ; optional overall differs from equal mean3 endpoint. [NOT_IN_GIT.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/gpt-judge-023-astra-high/NOT_IN_GIT.md); 5 files.
- **gpt-judge-visual-002** — Earlier gpt-6.1-sol judge on same planned174 per arm; actual partial calls; do not average with later judge as replication. [aggregate.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/gpt-judge-visual-002/aggregate.csv); 3 files.
- **judge-agreement-023** — Two-judge agreement over common successful cases675/696; descriptive correlation/Kendall; not new layout generation. [agreement.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/judge-agreement-023/agreement.json); 2 files.
- **mesh-collision-019** — Actual voxel mesh analysis;1009selected/1005analysed rooms; all rendered systemsGT,J,P,L,O1a,O1b,O2,O3,O4,O6,O7 with per-system missing layouts; movable pairs only; tau125/500/2000cm3. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/mesh-collision-019/report.md); 4 files.
- **report-v2-JPL-001** — Historical initial8B SFT report with L only; superseded version retained; not independent experiment. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/report-v2-JPL-001/report.md); 16 files.
- **report-v2-JPL-002** — Historical8B SFT matched J0/P0 full8647, original confirmatory endpoints checker + common-complete sizeMAE; L-only report; latest011 refresh on disk is separate. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/report-v2-JPL-002/report.md); 16 files.
- **report-v2-dpo1-001** — Historical8B official-hparam DPO1 full8647 collapse; secondary/descriptive; historical p_holm fields do not make it confirmatory. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/report-v2-dpo1-001/report.md); 13 files.
- **report-v2-selected-001** — 8B J=SFT,P=rpo-dpo1 full8647 +8agents1454; unknown-sizeJ-input agents self-select sizes, P gets prior; raw/harness distinguish; historical p_holm and a-prime naming superseded by011 interpretation. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/report-v2-selected-001/report.md); 16 files.
- **s1-012** — Four dev8167 rows: J/P greedy andbest-of8, each raw/S1; same sizes/yaw/categories/parents, deterministic repair; vLLM greedy differs HF. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/s1-012/joint-sft-bo8/report.md); 8 files.
- **scale-curve-016** — Offline fixed-position geometry rescore under size scale/substitution; GT/J/P/allagents; not regenerated layouts nor calibrated performance. Full curves/strata/rules kept. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/scale-curve-016/report.md); 9 files.
- **sizes-020** — Sizer actual inference: prior/GPT-size/O2-size on1877rooms; L-joint reference only1063, so not same denominator; prior vs size-only primary reading. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/sizes-020/report.md); 5 files.
- **stage-table-010-001** — 8B SFT/DPO1/selected fulltest and dev stage snapshots; DPO2test stopped before viewing, not measured; training/engine identity preserved. [stage_table.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/stage-table-010-001/stage_table.md); 10 files.
- **system-022-cohort** — Current27B raw/S1/system on1454; old checker-trigger system includes GPT/O2; accompanying all8 self-size agent raw/harness and4 fixed-size agents +GPT300pilot; note agent setting suffix. [NOT_IN_GIT.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/system-022-cohort/NOT_IN_GIT.md); 4 files.
- **system-022-test-sharded** — 27B full8647 old checker-trigger escalation to localO2 only, actual end-to-end;0API. [NOT_IN_GIT.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/system-022-test-sharded/NOT_IN_GIT.md); 4 files.
- **test-apply-012** — 8B selected J/P on8647 raw/S1; all8self-size agents1454 raw/harness raw/S1; all-request denominator. downstream_rate=strict port; accepted_rate=checker AND strict, not synonymous. [report.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/test-apply-012/report.md); 5 files.
- **visual-024** — Actual renders of27Braw(A) andoldO2checker-trigger system(S);1005shared successful/1009selected; FID/KID/mesh +GPTjudge173/174; not a027new trigger. [NOT_IN_GIT.md](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/visual-024/NOT_IN_GIT.md); 15 files.

## 全体总体表（原始分层见无损导出）

### 8b_dev_selection — 8B 开发集全部候选

600间开发集；资格及选择为历史决定，已看过DPO1测试结果后修订。

来源：[stage-table-010-001/stage_dev.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/stage-table-010-001/stage_dev.csv)

| arm | candidate | requests | complete_rate | geometry_success | a_prime | size_log_mae | common_with_sft | eligible | selected | missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J | sft | 600 | 1 | 0.38 | 0.356667 | 0.249675 | 600 | True | True | — |
| J | e1-dpo1 | 600 | 1 | 0.398333 | 0.291667 | 0.259139 | 600 | True | False | — |
| J | e1-dpo2 | 600 | 0.841667 | 0.396667 | 0.376667 | 0.251564 | 505 | False | False | — |
| J | rpo-dpo1 | 600 | 1 | 0.39 | 0.16 | 0.266439 | 600 | False | False | — |
| J | rpo-dpo2 | 600 | 0.985 | 0.435 | 0.283333 | 0.252355 | 591 | False | False | — |
| J | a009-dpo1 | 600 | 0.881667 | 0.181667 | 0.116667 | 0.572862 | 529 | — | — | eligible: 仅报告：修订 010 补充，修订 009 阶段只在 dev 上评分，不可选; selected: 仅报告：修订 010 补充，修订 009 阶段只在 dev 上评分，不可选 |
| J | a009-dpo2 | 600 | 0 | 0 | 0 | — | 0 | — | — | size_log_mae: 未测：无（与 SFT 共同）完整输出; sft_size_log_mae: 未测：无（与 SFT 共同）完整输出; footprint_bias: 未测：无（与 SFT 共同）完整输出; eligible: 仅报告：修订 010 补充，修订 009 阶段只在 dev 上评分，不可选; selected: 仅报告：修订 010 补充，修订 009 阶段只在 dev 上评分，不可选 |
| P | sft | 600 | 0.998333 | 0.408333 | 0.381667 | 0.233965 | 599 | True | False | — |
| P | e1-dpo1 | 600 | 0.998333 | 0.421667 | 0.391667 | 0.233918 | 598 | True | False | — |
| P | e1-dpo2 | 600 | 0.428333 | 0.186667 | 0.176667 | 0.266774 | 256 | False | False | — |
| P | rpo-dpo1 | 600 | 0.998333 | 0.47 | 0.435 | 0.233918 | 598 | True | True | — |
| P | rpo-dpo2 | 600 | 0.968333 | 0.431667 | 0.4 | 0.232489 | 580 | False | False | — |
| P | a009-dpo1 | 600 | 0.476667 | 0.0816667 | 0.0733333 | 0.223418 | 286 | — | — | eligible: 仅报告：修订 010 补充，修订 009 阶段只在 dev 上评分，不可选; selected: 仅报告：修订 010 补充，修订 009 阶段只在 dev 上评分，不可选 |
| P | a009-dpo2 | 600 | 0.05 | 0.0416667 | 0.0383333 | 0.174307 | 30 | — | — | eligible: 仅报告：修订 010 补充，修订 009 阶段只在 dev 上评分，不可选; selected: 仅报告：修订 010 补充，修订 009 阶段只在 dev 上评分，不可选 |

### 8b_stages — 8B SFT、DPO1、最终选择及未测DPO2

历史测试8647间；尺寸误差按各阶段J/P共同完整集计算。DPO2空白表示未测，非零分。a-prime是固定位置缩放压力测试。

来源：[stage-table-010-001/stage_stages.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/stage-table-010-001/stage_stages.csv)

| stage | arm | requests | complete_rate | geometry_success | debiased_success | size_log_mae | size_log_mae_n | size_iou | oor_axis | bias_fp | missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sft | J | 8647 | 0.997109 | 0.388227 | 0.343934 | 0.286077 | 8610 | 0.559261 | 0.146967 | -0.0830815 | oor_rotated: 未测：report_v2 只按论文六层输出旋转 OOR（见六层合并列）; automatic_acceptance: 未测：report_v2 未输出自动接受率字段 |
| sft | P | 8647 | 0.998497 | 0.389615 | 0.394703 | 0.281682 | 8610 | 0.558463 | 0.268209 | 0.0149587 | oor_rotated: 未测：report_v2 只按论文六层输出旋转 OOR（见六层合并列）; automatic_acceptance: 未测：report_v2 未输出自动接受率字段 |
| dpo1 | J | 8647 | 0.882734 | 0.213253 | 0.117844 | 0.590364 | 3851 | 0.297782 | 0.15466 | -0.356489 | oor_rotated: 未测：report_v2 只按论文六层输出旋转 OOR（见六层合并列）; automatic_acceptance: 未测：report_v2 未输出自动接受率字段 |
| dpo1 | P | 8647 | 0.496241 | 0.0929802 | 0.0911299 | 0.275178 | 3851 | 0.565301 | 0.14882 | -0.0114888 | oor_rotated: 未测：report_v2 只按论文六层输出旋转 OOR（见六层合并列）; automatic_acceptance: 未测：report_v2 未输出自动接受率字段 |
| dpo2 | J | — | — | — | — | — | — | — | — | — | requests: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; complete_rate: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; geometry_success: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; debiased_success: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; size_iou: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; size_log_mae: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; oor_axis: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; oor_rotated: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; oor_axis_strata: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; oor_rotated_strata: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; automatic_acceptance: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; bias_fp: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; bias_vol: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; cap_rule_flipped_complete: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析 |
| dpo2 | P | — | — | — | — | — | — | — | — | — | requests: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; complete_rate: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; geometry_success: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; debiased_success: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; size_iou: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; size_log_mae: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; oor_axis: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; oor_rotated: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; oor_axis_strata: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; oor_rotated_strata: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; automatic_acceptance: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; bias_fp: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; bias_vol: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析; cap_rule_flipped_complete: 已停止（修订 010 补充：测试推理在查看前停止，未测）：其部分输出（最大的房间，有偏子集）未分析 |
| selected（sft） | J | 8647 | 0.997109 | 0.388227 | 0.343934 | 0.286062 | 8602 | 0.559261 | 0.146967 | -0.0828839 | oor_rotated: 未测：report_v2 只按论文六层输出旋转 OOR（见六层合并列）; automatic_acceptance: 未测：report_v2 未输出自动接受率字段 |
| selected（rpo-dpo1） | P | 8647 | 0.997687 | 0.398057 | 0.404302 | 0.281422 | 8602 | 0.558572 | 0.247899 | 0.0145318 | oor_rotated: 未测：report_v2 只按论文六层输出旋转 OOR（见六层合并列）; automatic_acceptance: 未测：report_v2 未输出自动接受率字段 |

### sft_confirmatory — 原始SFT共同主要终点

仅原始SFT J0/P0的几何成功及尺寸误差构成Holm校正族；去偏终点为次要。后续selected/DPO旧CSV中的Holm列不构成新的验证性分析。

来源：[report-v2-JPL-002/T1.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/report-v2-JPL-002/T1.csv)

| endpoint | J | P | delta | delta_ci_low | delta_ci_high | n | p | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) 全请求原始几何成功率 | 0.388227 | 0.389615 | -0.00138776 | -0.0121546 | 0.00913963 | 8647 | 0.793 | 0.793 |
| (b) 尺寸 log MAE（共同完整） | 0.286077 | 0.281682 | 0.00439494 | 0.000490411 | 0.00842102 | 8610 | 0.033 | 0.066 |
| (a′) 尺寸去偏几何成功率 | 0.343934 | 0.394703 | -0.0507691 | -0.060487 | -0.0412324 | 8647 | — | — |

### 8b_stage_differences — 8B全部阶段差值

原样保留阶段比较及角色；selected是DPO1测试结果引发修订后的描述性结果。

来源：[stage-table-010-001/stage_differences.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/stage-table-010-001/stage_differences.csv)

| stage | endpoint | delta | delta_ci_low | delta_ci_high | n | p | role | missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sft | (a) 全请求原始几何成功率 | -0.00138776 | -0.0121546 | 0.00913963 | 8647 | 0.793 | 主要（确认性，protocol.md） | — |
| sft | (b) 尺寸 log MAE（共同完整） | 0.00439494 | 0.000490411 | 0.00842102 | 8610 | 0.033 | 主要（确认性，protocol.md） | — |
| sft | (a′) 尺寸去偏几何成功率 | -0.0507691 | -0.060487 | -0.0412324 | 8647 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| sft | complete（房间微平均） | -0.00138776 | -0.00270244 | 0 | 8647 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| sft | geometry_success_repaired（房间微平均） | -0.00289118 | -0.0134506 | 0.0078107 | 8647 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| sft | oor（房间微平均） | -0.114062 | -0.130081 | -0.0989116 | 8610 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| sft | oor_repaired（房间微平均） | -0.114107 | -0.13011 | -0.0990347 | 8610 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| sft | size_iou（房间微平均） | 0.000731429 | -0.00211087 | 0.00368316 | 8610 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| dpo1 | (a) 全请求原始几何成功率 | 0.120273 | 0.109621 | 0.131014 | 8647 | 0 | 描述性 | — |
| dpo1 | (b) 尺寸 log MAE（共同完整） | 0.315186 | 0.30511 | 0.324894 | 3851 | 0 | 描述性 | — |
| dpo1 | (a′) 尺寸去偏几何成功率 | 0.0267145 | 0.0191205 | 0.0346557 | 8647 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| dpo1 | complete（房间微平均） | 0.386492 | 0.373939 | 0.398555 | 8647 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| dpo1 | geometry_success_repaired（房间微平均） | 0.119348 | 0.108533 | 0.129983 | 8647 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| dpo1 | oor（房间微平均） | -0.0401762 | -0.0525692 | -0.0283383 | 3851 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| dpo1 | oor_repaired（房间微平均） | -0.0402608 | -0.0526288 | -0.02839 | 3851 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| dpo1 | size_iou（房间微平均） | -0.263372 | -0.26929 | -0.257152 | 3851 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| selected | (a) 全请求原始几何成功率 | -0.00983 | -0.0212792 | 0.00164274 | 8647 | 0.095 | 次要（交付比较，修订 010） | — |
| selected | (b) 尺寸 log MAE（共同完整） | 0.0046404 | 0.000701741 | 0.00869171 | 8602 | 0.021 | 次要（交付比较，修订 010） | — |
| selected | (a′) 尺寸去偏几何成功率 | -0.0603678 | -0.0711895 | -0.0493161 | 8647 | — | 次要（交付比较，修订 010） | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| selected | complete（房间微平均） | -0.000578235 | -0.00205042 | 0.000852969 | 8647 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| selected | geometry_success_repaired（房间微平均） | -0.0102926 | -0.0219057 | 0.00123727 | 8647 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| selected | oor（房间微平均） | -0.092866 | -0.110975 | -0.0751015 | 8602 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| selected | oor_repaired（房间微平均） | -0.0930866 | -0.111247 | -0.0754147 | 8602 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |
| selected | size_iou（房间微平均） | 0.000679329 | -0.0021884 | 0.0036543 | 8602 | — | 描述性 | p: 不适用：只对 (a)、(b) 计算 p 与 Holm; p_holm: 不适用：只对 (a)、(b) 计算 p 与 Holm |

### agents_self_size — 八个外部代理：自选尺寸、首答及多轮

1454间队列。J为训练后8B联合模型，P为8B先验尺寸放置模型；八个外部代理完整列出。raw评分首答，harness最多三次尝试；此表尚未加S1。O5未运行。

来源：[report-v2-selected-001/T5.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/report-v2-selected-001/T5.csv)

| system | requests | complete_rate | geometry_success_rate | success_debiased_rate | size_log_mae | size_iou | oor | bias_fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J | 1454 | 0.998624 | 0.430536 | 0.393398 | 0.242999 | 0.596836 | 0.110675 | -0.0476473 |
| P | 1454 | 1 | 0.519257 | 0.498624 | 0.238788 | 0.591877 | 0.173479 | -0.0223179 |
| L-raw | 1454 | 1 | 0.934663 | 0.765475 | 0.300281 | 0.50435 | 0.0309544 | -0.175138 |
| L-harness | 1454 | 1 | 0.934663 | 0.765475 | 0.300281 | 0.50435 | 0.0309544 | -0.175138 |
| O6-raw | 1454 | 0.996561 | 0.700138 | 0.279917 | 0.421555 | 0.39892 | 0.0232091 | -0.346635 |
| O6-harness | 1454 | 0.998624 | 0.700825 | 0.280605 | 0.421761 | 0.39886 | 0.0241847 | -0.345875 |
| O4-raw | 1454 | 0.997937 | 0.68707 | 0.303301 | 0.446051 | 0.388056 | 0.038702 | -0.296726 |
| O4-harness | 1454 | 1 | 0.68707 | 0.303301 | 0.446754 | 0.387779 | 0.121646 | -0.296166 |
| O2-raw | 1454 | 0.826685 | 0.808116 | 0.284732 | 0.329836 | 0.458071 | 0.013457 | -0.321731 |
| O2-harness | 1454 | 0.958047 | 0.927785 | 0.327373 | 0.340516 | 0.453391 | 0.0168534 | -0.310904 |
| O3-raw | 1454 | 0.331499 | 0.211142 | 0.0955983 | 0.451837 | 0.370033 | 0.0350771 | -0.440472 |
| O3-harness | 1454 | 0.394085 | 0.241403 | 0.0983494 | 0.450069 | 0.371863 | 0.0335619 | -0.449954 |
| O7-raw | 1454 | 0.980055 | 0.579092 | 0.213205 | 0.384408 | 0.425704 | 0.0422761 | -0.323204 |
| O7-harness | 1454 | 1 | 0.59216 | 0.219395 | 0.385992 | 0.424679 | 0.092827 | -0.323405 |
| O1a-raw | 1454 | 0.348006 | 0 | 0 | 0.608834 | 0.281181 | 0.43011 | -0.417497 |
| O1a-harness | 1454 | 0.349381 | 0 | 0 | 0.608914 | 0.280902 | 0.428636 | -0.418334 |
| O1b-raw | 1454 | 0.236589 | 0.042641 | 0.022696 | 0.534947 | 0.313181 | 0.0770619 | -0.351 |
| O1b-harness | 1454 | 0.574966 | 0.110041 | 0.0577717 | 0.538794 | 0.310265 | 0.0911787 | -0.311941 |

### agent_costs — 已保存的生成成本和模型身份

J/P GPU小时对应全测试生成，代理成本对应队列多轮运行。token统计及范围不同，不可直接当作统一部署成本比较。

来源：[report-v2-selected-001/T5_cost.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/report-v2-selected-001/T5_cost.csv)

| system | model | reasoning | calls | gpu_hours_full_test_generation | gpu_hours | input_tokens | output_tokens | reasoning_tokens | latency_p50_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J | Qwen3-8B + LoRA | — | — | 9.12221 | — | 730499 | 608259 | — | — |
| P | Qwen3-8B + LoRA | — | — | 11.1008 | — | 956719 | 387337 | — | — |
| L | gpt-6.1-sol | medium | 1469 | — | — | 7235552 | 862939 | 345132 | 13.5384 |
| O6 | qwen3.8-flash-next-fp8 | thinking on, medium (amendment 007; the model offers xhigh/medium/low) | 1462 | — | 1.97144 | 833771 | 5718037 | 5122821 | 0.722176 |
| O4 | gpt-oss-120b | medium | 1457 | — | 0.547931 | 820681 | 4319765 | 3824076 | 0.381225 |
| O2 | qwen3.8-27b | thinking on (model default) | 1819 | — | 4.51707 | 1212102 | 20475669 | 19903948 | 2.16562 |
| O3 | gemma-4-31b-it | thinking off (model default) | 3337 | — | 0.961086 | 3332899 | 1515628 | 0 | 0.106543 |
| O7 | muse-glimmer-30b | model default (no "Reasoning strength:" system line) | 1487 | — | 1.62021 | 776397 | 8681904 | 8166897 | 1.11826 |
| O1a | qwen3-8b-nothink | thinking off | 3348 | — | 0.122664 | 2800203 | 1361774 | 0 | 0.0297012 |
| O1b | qwen3-8b-think | thinking on | 3331 | — | 1.09267 | 3044739 | 12054951 | 10649402 | 1.21211 |

### agent_failures — 全部完成及失败计数

历史selected报告；保留截断、JSON、schema及缺失失败，成功率不只按有效答案统计。

来源：[report-v2-selected-001/T6.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/report-v2-selected-001/T6.csv)

| system | requests | complete | token_cap_without_eos | invalid_json | invalid_schema_or_geometry | generation_or_api_failure | missing_output | incomplete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J | 8647 | 8622 | 2 | 1 | 22 | 0 | 0 | 25 |
| P | 8647 | 8627 | 0 | 4 | 16 | 0 | 0 | 20 |
| L-raw | 1454 | 1454 | — | — | — | 0 | 0 | 0 |
| L-harness | 1454 | 1454 | — | — | — | 0 | 0 | 0 |
| O6-raw | 1454 | 1449 | 3 | — | — | 0 | 0 | 5 |
| O6-harness | 1454 | 1452 | 1 | — | — | 0 | 0 | 2 |
| O4-raw | 1454 | 1451 | 0 | — | — | 0 | 0 | 3 |
| O4-harness | 1454 | 1454 | 0 | — | — | 0 | 0 | 0 |
| O2-raw | 1454 | 1202 | 241 | — | — | 0 | 0 | 252 |
| O2-harness | 1454 | 1393 | 58 | — | — | 0 | 0 | 61 |
| O3-raw | 1454 | 482 | 0 | — | — | 0 | 0 | 972 |
| O3-harness | 1454 | 573 | 1 | — | — | 0 | 0 | 881 |
| O7-raw | 1454 | 1425 | 5 | — | — | 0 | 0 | 29 |
| O7-harness | 1454 | 1454 | 0 | — | — | 0 | 0 | 0 |
| O1a-raw | 1454 | 506 | 1 | — | — | 0 | 0 | 948 |
| O1a-harness | 1454 | 508 | 1 | — | — | 0 | 0 | 946 |
| O1b-raw | 1454 | 344 | 16 | — | — | 0 | 0 | 1110 |
| O1b-harness | 1454 | 836 | 9 | — | — | 0 | 0 | 618 |

### s1_bo8_dev — 开发集S1及八选一

8167间开发集，8B模型。S1获采纳；B8相对贪心+S1的严格准入增益，两臂均低于0.03采纳门槛。这里是生成八候选再选择，不是八模型集成。

来源：[decision-012/decision.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/decision-012/decision.json)

| policy | requests | complete | checker | downstream |
| --- | --- | --- | --- | --- |
| joint-greedy-raw | 8167 | 0.998041 | 0.387903 | 0.173258 |
| joint-greedy-s1 | 8167 | 0.998041 | 0.711277 | 0.253092 |
| joint-bo8-raw | 8167 | 0.998041 | 0.435288 | 0.185625 |
| joint-bo8-s1 | 8167 | 0.998041 | 0.728787 | 0.25603 |
| placement-greedy-raw | 8167 | 0.997918 | 0.400269 | 0.198849 |
| placement-greedy-s1 | 8167 | 0.997918 | 0.708461 | 0.291539 |
| placement-bo8-raw | 8167 | 0.99902 | 0.600343 | 0.250398 |
| placement-bo8-s1 | 8167 | 0.99902 | 0.797723 | 0.311865 |

### s1_bo8_dev_contrasts — S1及B8开发集配对差

房屋簇bootstrap配对差；此处downstream为严格准入。

来源：[decision-012/decision.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/decision-012/decision.json)

| comparison | metric | estimate | ci_low | ci_high | n |
| --- | --- | --- | --- | --- | --- |
| joint: G+S1 - G | downstream | 0.0798335 | 0.0743492 | 0.0857509 | 8167 |
| joint: G+S1 - G | checker | 0.323375 | 0.312654 | 0.3341 | 8167 |
| placement: G+S1 - G | downstream | 0.0926901 | 0.0860696 | 0.0995575 | 8167 |
| placement: G+S1 - G | checker | 0.308192 | 0.293451 | 0.323712 | 8167 |
| joint: (B8+S1) - (G+S1) | downstream | 0.00293866 | 0.000490222 | 0.00547858 | 8167 |
| joint: (B8+S1) - (G+S1) | checker | 0.0175095 | 0.0140873 | 0.0210165 | 8167 |
| placement: (B8+S1) - (G+S1) | downstream | 0.0203257 | 0.014754 | 0.0261199 | 8167 |
| placement: (B8+S1) - (G+S1) | checker | 0.0892617 | 0.0805224 | 0.0979439 | 8167 |

### s1_test_test — 8B及代理S1应用：test

分母为全部请求；downstream_rate为严格准入，accepted_rate为检查器及严格准入同时通过。raw及S1复用相同生成输出。

来源：[test-apply-012/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/test-apply-012/levels.csv)

| system | stage | requests | complete_rooms | checker_rate | downstream_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| J | raw | 8647 | 8622 | 0.388227 | 0.17243 | 0.166647 |
| J | s1 | 8647 | 8622 | 0.704984 | 0.25824 | 0.253498 |
| P | raw | 8647 | 8627 | 0.398057 | 0.199838 | 0.19475 |
| P | s1 | 8647 | 8627 | 0.700937 | 0.287845 | 0.28426 |
| GT | raw | 8647 | 8647 | 0.723025 | 0.295247 | 0.290505 |

### s1_test_cohort — 8B及代理S1应用：cohort

分母为全部请求；downstream_rate为严格准入，accepted_rate为检查器及严格准入同时通过。raw及S1复用相同生成输出。

来源：[test-apply-012/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/test-apply-012/levels.csv)

| system | stage | requests | complete_rooms | checker_rate | downstream_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| J | raw | 1454 | 1452 | 0.430536 | 0.233838 | 0.230399 |
| J | s1 | 1454 | 1452 | 0.797111 | 0.372077 | 0.370702 |
| P | raw | 1454 | 1454 | 0.519257 | 0.354195 | 0.350069 |
| P | s1 | 1454 | 1454 | 0.825997 | 0.47249 | 0.469739 |
| GT | raw | 1454 | 1454 | 0.724897 | 0.448418 | 0.446355 |
| L-harness | raw | 1454 | 1454 | 0.934663 | 0.727648 | 0.727648 |
| L-harness | s1 | 1454 | 1454 | 0.959422 | 0.75447 | 0.75447 |
| L-raw | raw | 1454 | 1454 | 0.934663 | 0.727648 | 0.727648 |
| L-raw | s1 | 1454 | 1454 | 0.959422 | 0.75447 | 0.75447 |
| O1a-harness | raw | 1454 | 508 | 0 | 0 | 0 |
| O1a-harness | s1 | 1454 | 508 | 0 | 0 | 0 |
| O1a-raw | raw | 1454 | 506 | 0 | 0 | 0 |
| O1a-raw | s1 | 1454 | 506 | 0 | 0 | 0 |
| O1b-harness | raw | 1454 | 836 | 0.110041 | 0.0687758 | 0.068088 |
| O1b-harness | s1 | 1454 | 836 | 0.271664 | 0.171252 | 0.170564 |
| O1b-raw | raw | 1454 | 344 | 0.042641 | 0.0295736 | 0.0295736 |
| O1b-raw | s1 | 1454 | 344 | 0.110041 | 0.0701513 | 0.0701513 |
| O2-harness | raw | 1454 | 1393 | 0.927785 | 0.722834 | 0.722834 |
| O2-harness | s1 | 1454 | 1393 | 0.947043 | 0.752407 | 0.752407 |
| O2-raw | raw | 1454 | 1202 | 0.808116 | 0.674003 | 0.674003 |
| O2-raw | s1 | 1454 | 1202 | 0.821871 | 0.695323 | 0.695323 |
| O3-harness | raw | 1454 | 573 | 0.241403 | 0.158872 | 0.158184 |
| O3-harness | s1 | 1454 | 573 | 0.350069 | 0.222834 | 0.222834 |
| O3-raw | raw | 1454 | 482 | 0.211142 | 0.138239 | 0.137552 |
| O3-raw | s1 | 1454 | 482 | 0.292985 | 0.191884 | 0.191884 |
| O4-harness | raw | 1454 | 1454 | 0.68707 | 0.479367 | 0.47868 |
| O4-harness | s1 | 1454 | 1454 | 0.862448 | 0.596974 | 0.596974 |
| O4-raw | raw | 1454 | 1451 | 0.68707 | 0.479367 | 0.47868 |
| O4-raw | s1 | 1454 | 1451 | 0.862448 | 0.596974 | 0.596974 |
| O6-harness | raw | 1454 | 1452 | 0.700825 | 0.535763 | 0.534388 |
| O6-harness | s1 | 1454 | 1452 | 0.856259 | 0.632737 | 0.632737 |
| O6-raw | raw | 1454 | 1449 | 0.700138 | 0.535763 | 0.534388 |
| O6-raw | s1 | 1454 | 1449 | 0.855571 | 0.632737 | 0.632737 |
| O7-harness | raw | 1454 | 1454 | 0.59216 | 0.409904 | 0.409904 |
| O7-harness | s1 | 1454 | 1454 | 0.740715 | 0.5 | 0.5 |
| O7-raw | raw | 1454 | 1425 | 0.579092 | 0.400275 | 0.400275 |
| O7-raw | s1 | 1454 | 1425 | 0.724897 | 0.488308 | 0.488308 |

### same_input_1454 — 同输入尺寸条件：1454间

完整保留raw/harness及raw/S1。-P表示生成时给先验尺寸且评分时恢复；无-P表示自选尺寸。P为训练后8B参考；accepted为检查器及严格准入同时通过，downstream为严格准入。跨variant重复的P是同一参考。

来源：[compare-016-full-v1/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/compare-016-full-v1/levels.csv)

| variant | system | stage | requests | complete | checker | downstream | accepted | bias_fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| harness | P | raw | 1454 | 1 | 0.519257 | 0.354195 | 0.350069 | -0.0224016 |
| harness | P | s1 | 1454 | 1 | 0.825997 | 0.47249 | 0.469739 | -0.0224016 |
| harness | O2-harness | raw | 1454 | 0.958047 | 0.927785 | 0.722834 | 0.722834 | -0.310819 |
| harness | O2-harness | s1 | 1454 | 0.958047 | 0.947043 | 0.752407 | 0.752407 | -0.310819 |
| harness | O6-harness | raw | 1454 | 0.998624 | 0.700825 | 0.535763 | 0.534388 | -0.345965 |
| harness | O6-harness | s1 | 1454 | 0.998624 | 0.856259 | 0.632737 | 0.632737 | -0.345965 |
| harness | O4-harness | raw | 1454 | 1 | 0.68707 | 0.479367 | 0.47868 | -0.296034 |
| harness | O4-harness | s1 | 1454 | 1 | 0.862448 | 0.596974 | 0.596974 | -0.296034 |
| harness | O7-harness | raw | 1454 | 1 | 0.59216 | 0.409904 | 0.409904 | -0.323345 |
| harness | O7-harness | s1 | 1454 | 1 | 0.740715 | 0.5 | 0.5 | -0.323345 |
| harness | O2-P-harness | raw | 1454 | 0.935351 | 0.918157 | 0.667125 | 0.667125 | -0.0340513 |
| harness | O2-P-harness | s1 | 1454 | 0.935351 | 0.927785 | 0.700825 | 0.700825 | -0.0340513 |
| harness | O6-P-harness | raw | 1454 | 0.996561 | 0.755846 | 0.526823 | 0.526823 | -0.0229236 |
| harness | O6-P-harness | s1 | 1454 | 0.996561 | 0.883081 | 0.61967 | 0.61967 | -0.0229236 |
| harness | O4-P-harness | raw | 1454 | 1 | 0.65337 | 0.457359 | 0.455296 | -0.0224016 |
| harness | O4-P-harness | s1 | 1454 | 1 | 0.85282 | 0.566713 | 0.566713 | -0.0224016 |
| harness | O7-P-harness | raw | 1454 | 0.996561 | 0.700138 | 0.506878 | 0.505502 | -0.0244054 |
| harness | O7-P-harness | s1 | 1454 | 0.996561 | 0.811554 | 0.581155 | 0.581155 | -0.0244054 |
| raw | P | raw | 1454 | 1 | 0.519257 | 0.354195 | 0.350069 | -0.0224016 |
| raw | P | s1 | 1454 | 1 | 0.825997 | 0.47249 | 0.469739 | -0.0224016 |
| raw | O2-raw | raw | 1454 | 0.826685 | 0.808116 | 0.674003 | 0.674003 | -0.321399 |
| raw | O2-raw | s1 | 1454 | 0.826685 | 0.821871 | 0.695323 | 0.695323 | -0.321399 |
| raw | O6-raw | raw | 1454 | 0.996561 | 0.700138 | 0.535763 | 0.534388 | -0.346724 |
| raw | O6-raw | s1 | 1454 | 0.996561 | 0.855571 | 0.632737 | 0.632737 | -0.346724 |
| raw | O4-raw | raw | 1454 | 0.997937 | 0.68707 | 0.479367 | 0.47868 | -0.296592 |
| raw | O4-raw | s1 | 1454 | 0.997937 | 0.862448 | 0.596974 | 0.596974 | -0.296592 |
| raw | O7-raw | raw | 1454 | 0.980055 | 0.579092 | 0.400275 | 0.400275 | -0.323143 |
| raw | O7-raw | s1 | 1454 | 0.980055 | 0.724897 | 0.488308 | 0.488308 | -0.323143 |
| raw | O2-P-raw | raw | 1454 | 0.796424 | 0.786795 | 0.623796 | 0.623796 | -0.0484738 |
| raw | O2-P-raw | s1 | 1454 | 0.796424 | 0.791609 | 0.65337 | 0.65337 | -0.0484738 |
| raw | O6-P-raw | raw | 1454 | 0.974553 | 0.751719 | 0.525447 | 0.525447 | -0.0255352 |
| raw | O6-P-raw | s1 | 1454 | 0.974553 | 0.870702 | 0.616919 | 0.616919 | -0.0255352 |
| raw | O4-P-raw | raw | 1454 | 0.99381 | 0.65337 | 0.457359 | 0.455296 | -0.0229285 |
| raw | O4-P-raw | s1 | 1454 | 0.99381 | 0.850757 | 0.566025 | 0.566025 | -0.0229285 |
| raw | O7-P-raw | raw | 1454 | 0.965612 | 0.682944 | 0.495186 | 0.49381 | -0.0244184 |
| raw | O7-P-raw | s1 | 1454 | 0.965612 | 0.789546 | 0.5674 | 0.5674 | -0.0244184 |

### same_input_300 — 同输入尺寸条件：300间

完整保留raw/harness及raw/S1。-P表示生成时给先验尺寸且评分时恢复；无-P表示自选尺寸。P为训练后8B参考；accepted为检查器及严格准入同时通过，downstream为严格准入。跨variant重复的P是同一参考。

来源：[compare-016-pilot-v1/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/compare-016-pilot-v1/levels.csv)

| variant | system | stage | requests | complete | checker | downstream | accepted | bias_fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| harness | P | raw | 300 | 1 | 0.513333 | 0.366667 | 0.363333 | -0.0538443 |
| harness | P | s1 | 300 | 1 | 0.826667 | 0.49 | 0.486667 | -0.0538443 |
| harness | O2-harness | raw | 300 | 0.97 | 0.953333 | 0.713333 | 0.713333 | -0.30364 |
| harness | O2-harness | s1 | 300 | 0.97 | 0.966667 | 0.75 | 0.75 | -0.30364 |
| harness | O6-harness | raw | 300 | 1 | 0.706667 | 0.53 | 0.526667 | -0.360458 |
| harness | O6-harness | s1 | 300 | 1 | 0.88 | 0.626667 | 0.626667 | -0.360458 |
| harness | O4-harness | raw | 300 | 1 | 0.696667 | 0.453333 | 0.453333 | -0.297323 |
| harness | O4-harness | s1 | 300 | 1 | 0.836667 | 0.576667 | 0.576667 | -0.297323 |
| harness | O7-harness | raw | 300 | 1 | 0.6 | 0.39 | 0.39 | -0.313892 |
| harness | O7-harness | s1 | 300 | 1 | 0.736667 | 0.463333 | 0.463333 | -0.313892 |
| harness | L-harness | raw | 300 | 1 | 0.933333 | 0.69 | 0.69 | -0.185799 |
| harness | L-harness | s1 | 300 | 1 | 0.96 | 0.726667 | 0.726667 | -0.185799 |
| harness | O2-P-harness | raw | 300 | 0.93 | 0.92 | 0.64 | 0.64 | -0.0748262 |
| harness | O2-P-harness | s1 | 300 | 0.93 | 0.923333 | 0.69 | 0.69 | -0.0748262 |
| harness | O6-P-harness | raw | 300 | 0.993333 | 0.76 | 0.526667 | 0.526667 | -0.0546324 |
| harness | O6-P-harness | s1 | 300 | 0.993333 | 0.88 | 0.613333 | 0.613333 | -0.0546324 |
| harness | O4-P-harness | raw | 300 | 1 | 0.623333 | 0.41 | 0.41 | -0.0538443 |
| harness | O4-P-harness | s1 | 300 | 1 | 0.836667 | 0.55 | 0.55 | -0.0538443 |
| harness | O7-P-harness | raw | 300 | 1 | 0.696667 | 0.466667 | 0.466667 | -0.0538443 |
| harness | O7-P-harness | s1 | 300 | 1 | 0.786667 | 0.53 | 0.53 | -0.0538443 |
| harness | L-P-harness | raw | 300 | 1 | 0.946667 | 0.676667 | 0.676667 | -0.0538443 |
| harness | L-P-harness | s1 | 300 | 1 | 0.976667 | 0.723333 | 0.723333 | -0.0538443 |
| raw | P | raw | 300 | 1 | 0.513333 | 0.366667 | 0.363333 | -0.0538443 |
| raw | P | s1 | 300 | 1 | 0.826667 | 0.49 | 0.486667 | -0.0538443 |
| raw | O2-raw | raw | 300 | 0.83 | 0.82 | 0.653333 | 0.653333 | -0.308849 |
| raw | O2-raw | s1 | 300 | 0.83 | 0.83 | 0.68 | 0.68 | -0.308849 |
| raw | O6-raw | raw | 300 | 0.996667 | 0.706667 | 0.53 | 0.526667 | -0.360627 |
| raw | O6-raw | s1 | 300 | 0.996667 | 0.88 | 0.626667 | 0.626667 | -0.360627 |
| raw | O4-raw | raw | 300 | 1 | 0.696667 | 0.453333 | 0.453333 | -0.297323 |
| raw | O4-raw | s1 | 300 | 1 | 0.836667 | 0.576667 | 0.576667 | -0.297323 |
| raw | O7-raw | raw | 300 | 0.99 | 0.593333 | 0.386667 | 0.386667 | -0.314679 |
| raw | O7-raw | s1 | 300 | 0.99 | 0.73 | 0.46 | 0.46 | -0.314679 |
| raw | L-raw | raw | 300 | 1 | 0.933333 | 0.69 | 0.69 | -0.185799 |
| raw | L-raw | s1 | 300 | 1 | 0.96 | 0.726667 | 0.726667 | -0.185799 |
| raw | O2-P-raw | raw | 300 | 0.8 | 0.793333 | 0.596667 | 0.596667 | -0.0812011 |
| raw | O2-P-raw | s1 | 300 | 0.8 | 0.796667 | 0.636667 | 0.636667 | -0.0812011 |
| raw | O6-P-raw | raw | 300 | 0.976667 | 0.756667 | 0.523333 | 0.523333 | -0.057444 |
| raw | O6-P-raw | s1 | 300 | 0.976667 | 0.876667 | 0.61 | 0.61 | -0.057444 |
| raw | O4-P-raw | raw | 300 | 0.993333 | 0.623333 | 0.41 | 0.41 | -0.055957 |
| raw | O4-P-raw | s1 | 300 | 0.993333 | 0.836667 | 0.55 | 0.55 | -0.055957 |
| raw | O7-P-raw | raw | 300 | 0.97 | 0.68 | 0.46 | 0.46 | -0.0557825 |
| raw | O7-P-raw | s1 | 300 | 0.97 | 0.766667 | 0.523333 | 0.523333 | -0.0557825 |
| raw | L-P-raw | raw | 300 | 1 | 0.946667 | 0.676667 | 0.676667 | -0.0538443 |
| raw | L-P-raw | s1 | 300 | 1 | 0.976667 | 0.723333 | 0.723333 | -0.0538443 |

### 8b_training_ablation — 8B偏好训练归因消融

600间开发集，先验尺寸放置。RFT及on-policy DPO各两种子；mean行仅均值，非独立实验。此消融为8B，不能当作后续27B模型的因果归因。

来源：[ablation-016/ablation.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/ablation-016/ablation.json)

| candidate | rooms | complete_rate | geometry_success_rate | a_prime | accepted_rate | both_pass_rate | size_log_mae_own | footprint_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rpo-dpo1 | 600 | 0.998333 | 0.47 | 0.435 | 0.251667 | 0.25 | 0.234579 | -0.0540297 |
| rft | 600 | 1 | 0.473333 | 0.436667 | 0.27 | 0.261667 | 0.234624 | -0.0541753 |
| rft-s43 | 600 | 0.998333 | 0.486667 | 0.445 | 0.271667 | 0.265 | 0.234404 | -0.0542156 |
| pert | 600 | 0.978333 | 0.406667 | 0.38 | 0.235 | 0.235 | 0.232264 | -0.0565687 |
| op-dpo2-e2 | 600 | 0.941667 | 0.475 | 0.443333 | 0.328333 | 0.328333 | 0.232806 | -0.0601579 |
| op-dpo2-s43 | 600 | 0.946667 | 0.49 | 0.461667 | 0.325 | 0.321667 | 0.231903 | -0.0565208 |
| rft-mean | 600 | 0.999167 | 0.48 | 0.440833 | 0.270833 | 0.263333 | 0.234514 | -0.0541955 |
| op-dpo2-mean | 600 | 0.944167 | 0.4825 | 0.4525 | 0.326667 | 0.325 | 0.232354 | -0.0583393 |

### 8b_ablation_endpoints — 全部归因配对终点

预设解释要求a-prime及严格准入两个区间均高于零；未出现满足该双终点要求的清晰统一归因结论。

来源：[ablation-016/ablation.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/ablation-016/ablation.json)

| comparison | metric | estimate | ci_low | ci_high | n |
| --- | --- | --- | --- | --- | --- |
| rft - rpo-dpo1 | success_debiased | 0.00166667 | -0.0339928 | 0.0357173 | 600 |
| rft - rpo-dpo1 | accepted | 0.0183333 | -0.0113669 | 0.0466362 | 600 |
| rft-s43 - rpo-dpo1 | success_debiased | 0.01 | -0.0265836 | 0.043997 | 600 |
| rft-s43 - rpo-dpo1 | accepted | 0.02 | -0.00805153 | 0.0480157 | 600 |
| pert - rpo-dpo1 | success_debiased | -0.055 | -0.0906097 | -0.0196873 | 600 |
| pert - rpo-dpo1 | accepted | -0.0166667 | -0.046392 | 0.0152834 | 600 |
| op-dpo2-e2 - rpo-dpo1 | success_debiased | 0.00833333 | -0.0320524 | 0.0485676 | 600 |
| op-dpo2-e2 - rpo-dpo1 | accepted | 0.0766667 | 0.0449202 | 0.109039 | 600 |
| op-dpo2-s43 - rpo-dpo1 | success_debiased | 0.0266667 | -0.0147411 | 0.0676603 | 600 |
| op-dpo2-s43 - rpo-dpo1 | accepted | 0.0733333 | 0.0405833 | 0.104212 | 600 |
| op-dpo2-e2 - rft | success_debiased | 0.00666667 | -0.03125 | 0.0466848 | 600 |
| op-dpo2-e2 - rft | accepted | 0.0583333 | 0.0274896 | 0.0910765 | 600 |
| op-dpo2-s43 - rft-s43 | success_debiased | 0.0166667 | -0.0219262 | 0.0547015 | 600 |
| op-dpo2-s43 - rft-s43 | accepted | 0.0533333 | 0.0209298 | 0.0852853 | 600 |
| op-dpo2-e2 - pert | success_debiased | 0.0633333 | 0.0235685 | 0.10068 | 600 |
| op-dpo2-e2 - pert | accepted | 0.0933333 | 0.0608094 | 0.126246 | 600 |
| op-dpo2-s43 - pert | success_debiased | 0.0816667 | 0.0424798 | 0.117757 | 600 |
| op-dpo2-s43 - pert | accepted | 0.09 | 0.0531684 | 0.124166 | 600 |
| pert - rft | success_debiased | -0.0566667 | -0.0912045 | -0.0187642 | 600 |
| pert - rft | accepted | -0.035 | -0.0679942 | -0.00162821 | 600 |
| pert - rft-s43 | success_debiased | -0.065 | -0.101106 | -0.0287905 | 600 |
| pert - rft-s43 | accepted | -0.0366667 | -0.0679146 | -0.00484633 | 600 |

### 27b_dev_selection — 600间开发集的最终放置模型选择

op13检查器成功率最高，但未通过GT关系保留护栏。A/arm是训练后27B学生模型，与未训练O2不同；护栏参考使用相同推理引擎的R。

来源：[a021-selection/dev-selection.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a021-selection/dev-selection.json)

| candidate | size | engine | requests | complete_rate | checker | accepted_rate | oor_mean | holds_rate | checker_21plus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r | 8B | hf | 600 | 0.998333 | 0.47 | 0.251667 | 0.126635 | 0.702228 | 0.0175439 |
| op13 | 8B | hf | 600 | 0.995 | 0.515 | 0.331667 | 0.0504491 | 0.628571 | 0.0526316 |
| arm | 27B | vllm | 600 | 0.996667 | 0.51 | 0.263333 | 0.131678 | 0.740236 | 0.0350877 |
| r-reference | 8B | vllm | 600 | 0.998333 | 0.445 | 0.226667 | 0.138682 | 0.703014 | 0 |

### 27b_confirmation — 3000间模型确认切片

winner为27B A，R为vLLM下8B rpo-DPO1。检查器差+0.0526667，95%CI[0.0368207,0.0684343]；该选择确认不使后续复用的历史测试成为独立测试。

来源：[a021-selection/final-selection.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a021-selection/final-selection.json)

| candidate | engine | requests | complete_rate | checker | accepted_rate | oor_mean | holds_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| winner | vllm | 3000 | 0.999333 | 0.443667 | 0.219333 | 0.244706 | 0.646965 |
| r | vllm | 3000 | 0.998 | 0.391 | 0.202667 | 0.257599 | 0.585076 |

### 27b_vs_8b_test — 全测试27B与相同引擎8B比较

8647间，vLLM；R输出不同于之前HF selected-P输出。A输出被022及027系统复用。

来源：[a021-test/summary-arm.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a021-test/summary-arm.json), [a021-test/summary-r.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a021-test/summary-r.json)

| candidate | requests | complete_rate | checker | accepted_rate | oor_mean | holds_rate |
| --- | --- | --- | --- | --- | --- | --- |
| arm | 8647 | 0.999537 | 0.443391 | 0.216144 | 0.255952 | 0.65111 |
| r | 8647 | 0.998034 | 0.399329 | 0.202151 | 0.252895 | 0.595054 |

### engine_probe — HF与vLLM探针

8B参考的100间探针；尽管整体检查器率相同，仅19份输出完全相同。属于引擎探针，不是独立模型比较。

来源：[a021-test/probe-report.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a021-test/probe-report.json)

| engine | rooms | complete | geometry_success |
| --- | --- | --- | --- |
| hf | 100 | 0.99 | 0.43 |
| vllm | 100 | 0.99 | 0.43 |

### system_cohort_raw — 27B及全部已保存代理条件：raw

队列评估，多数1454间；L-P仅300间试点。代理包含自选/固定尺寸、首答/多轮变体，复用已保存答案。队列system阶段有API访问，不等于无API全测试系统。

来源：[system-022-cohort/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/system-022-cohort/levels.csv)

| system | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | oor | size_log_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| system | 1454 | 1454 | 0.556396 | 0.349381 | 0.764787 | 0.164957 | 0.238788 |
| L-raw | 1454 | 1454 | 0.934663 | 0.727648 | 0.837001 | 0.0309544 | 0.300281 |
| O1a-raw | 1454 | 506 | 0 | 0 | 0 | 0.43011 | 0.608834 |
| O1b-raw | 1454 | 344 | 0.042641 | 0.0295736 | 0.0584594 | 0.0770619 | 0.534947 |
| O2-raw | 1454 | 1202 | 0.808116 | 0.674003 | 0.734525 | 0.013457 | 0.329836 |
| O3-raw | 1454 | 482 | 0.211142 | 0.138239 | 0.21458 | 0.0350771 | 0.451837 |
| O4-raw | 1454 | 1451 | 0.68707 | 0.479367 | 0.61967 | 0.038702 | 0.446051 |
| O6-raw | 1454 | 1449 | 0.700138 | 0.535763 | 0.632737 | 0.0232091 | 0.421555 |
| O7-raw | 1454 | 1425 | 0.579092 | 0.400275 | 0.519257 | 0.0422761 | 0.384408 |
| O2-P-raw | 1454 | 1158 | 0.786795 | 0.623796 | 0.682944 | 0.0108557 | 0.220219 |
| O4-P-raw | 1454 | 1445 | 0.65337 | 0.457359 | 0.63205 | 0.123496 | 0.238914 |
| O6-P-raw | 1454 | 1417 | 0.751719 | 0.525447 | 0.641678 | 0.0344961 | 0.237043 |
| O7-P-raw | 1454 | 1404 | 0.682944 | 0.495186 | 0.598349 | 0.0890785 | 0.239376 |
| L-P-pilot-raw | 300 | 300 | 0.946667 | 0.676667 | 0.786667 | 0.0354917 | 0.231991 |
| L-harness | 1454 | 1454 | 0.934663 | 0.727648 | 0.837001 | 0.0309544 | 0.300281 |
| O1a-harness | 1454 | 508 | 0 | 0 | 0 | 0.428636 | 0.608914 |
| O1b-harness | 1454 | 836 | 0.110041 | 0.0687758 | 0.136864 | 0.0911787 | 0.538794 |
| O2-harness | 1454 | 1393 | 0.927785 | 0.722834 | 0.812242 | 0.0168534 | 0.340516 |
| O3-harness | 1454 | 573 | 0.241403 | 0.158872 | 0.257221 | 0.0335619 | 0.450069 |
| O4-harness | 1454 | 1454 | 0.68707 | 0.479367 | 0.620358 | 0.121646 | 0.446754 |
| O6-harness | 1454 | 1452 | 0.700825 | 0.535763 | 0.633425 | 0.0241847 | 0.421761 |
| O7-harness | 1454 | 1454 | 0.59216 | 0.409904 | 0.533012 | 0.092827 | 0.385992 |
| O2-P-harness | 1454 | 1360 | 0.918157 | 0.667125 | 0.749656 | 0.013963 | 0.233041 |
| O4-P-harness | 1454 | 1454 | 0.65337 | 0.457359 | 0.634113 | 0.132245 | 0.238788 |
| O6-P-harness | 1454 | 1449 | 0.755846 | 0.526823 | 0.647868 | 0.0367913 | 0.238858 |
| O7-P-harness | 1454 | 1449 | 0.700138 | 0.506878 | 0.611417 | 0.0888763 | 0.238216 |
| L-P-pilot-harness | 300 | 300 | 0.946667 | 0.676667 | 0.786667 | 0.0354917 | 0.231991 |

### system_cohort_s1 — 27B及全部已保存代理条件：s1

队列评估，多数1454间；L-P仅300间试点。代理包含自选/固定尺寸、首答/多轮变体，复用已保存答案。队列system阶段有API访问，不等于无API全测试系统。

来源：[system-022-cohort/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/system-022-cohort/levels.csv)

| system | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | oor | size_log_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| system | 1454 | 1454 | 0.854195 | 0.453232 | 0.820495 | 0.0832782 | 0.238788 |
| L-raw | 1454 | 1454 | 0.959422 | 0.75447 | 0.887895 | 0.0310088 | 0.300281 |
| O1a-raw | 1454 | 506 | 0 | 0 | 0 | 0.300669 | 0.608834 |
| O1b-raw | 1454 | 344 | 0.110041 | 0.0701513 | 0.112105 | 0.0512712 | 0.534947 |
| O2-raw | 1454 | 1202 | 0.821871 | 0.695323 | 0.766162 | 0.0134887 | 0.329836 |
| O3-raw | 1454 | 482 | 0.292985 | 0.191884 | 0.267538 | 0.0269023 | 0.451837 |
| O4-raw | 1454 | 1451 | 0.862448 | 0.596974 | 0.764099 | 0.0236004 | 0.446051 |
| O6-raw | 1454 | 1449 | 0.855571 | 0.632737 | 0.782669 | 0.0219929 | 0.421555 |
| O7-raw | 1454 | 1425 | 0.724897 | 0.488308 | 0.638239 | 0.0389452 | 0.384408 |
| O2-P-raw | 1454 | 1158 | 0.791609 | 0.65337 | 0.722146 | 0.0108326 | 0.220219 |
| O4-P-raw | 1454 | 1445 | 0.850757 | 0.566025 | 0.738652 | 0.0488149 | 0.238914 |
| O6-P-raw | 1454 | 1417 | 0.870702 | 0.616919 | 0.76066 | 0.0316468 | 0.237043 |
| O7-P-raw | 1454 | 1404 | 0.789546 | 0.5674 | 0.682256 | 0.0468836 | 0.239376 |
| L-P-pilot-raw | 300 | 300 | 0.976667 | 0.723333 | 0.85 | 0.0354998 | 0.231991 |
| L-harness | 1454 | 1454 | 0.959422 | 0.75447 | 0.887895 | 0.0310088 | 0.300281 |
| O1a-harness | 1454 | 508 | 0 | 0 | 0 | 0.299704 | 0.608914 |
| O1b-harness | 1454 | 836 | 0.271664 | 0.171252 | 0.262724 | 0.0523798 | 0.538794 |
| O2-harness | 1454 | 1393 | 0.947043 | 0.752407 | 0.864512 | 0.0169937 | 0.340516 |
| O3-harness | 1454 | 573 | 0.350069 | 0.222834 | 0.324622 | 0.0254625 | 0.450069 |
| O4-harness | 1454 | 1454 | 0.862448 | 0.596974 | 0.764787 | 0.0259712 | 0.446754 |
| O6-harness | 1454 | 1452 | 0.856259 | 0.632737 | 0.783356 | 0.0223469 | 0.421761 |
| O7-harness | 1454 | 1454 | 0.740715 | 0.5 | 0.654058 | 0.0438406 | 0.385992 |
| O2-P-harness | 1454 | 1360 | 0.927785 | 0.700825 | 0.801238 | 0.0140283 | 0.233041 |
| O4-P-harness | 1454 | 1454 | 0.85282 | 0.566713 | 0.741403 | 0.0488172 | 0.238788 |
| O6-P-harness | 1454 | 1449 | 0.883081 | 0.61967 | 0.770977 | 0.0336525 | 0.238858 |
| O7-P-harness | 1454 | 1449 | 0.811554 | 0.581155 | 0.697387 | 0.0476747 | 0.238216 |
| L-P-pilot-harness | 300 | 300 | 0.976667 | 0.723333 | 0.85 | 0.0354998 | 0.231991 |

### system_cohort_system — 27B及全部已保存代理条件：system

队列评估，多数1454间；L-P仅300间试点。代理包含自选/固定尺寸、首答/多轮变体，复用已保存答案。队列system阶段有API访问，不等于无API全测试系统。

来源：[system-022-cohort/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/system-022-cohort/levels.csv)

| system | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | oor | size_log_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| system | 1454 | 1454 | 0.993122 | 0.519945 | 0.854195 | 0.0561982 | 0.238788 |

### fulltest_27b_systems — 全测试27B系统四阶段

历史测试8647间。022/027完全复用raw及S1，只计一次；两次全测试升级均用本地O2、无API。027于2026-10-16完成归档；旧HTML/findings的待运行描述已过期。本导出没有027的FID/judge/mesh结果。

来源：[system-022-test-sharded/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/system-022-test-sharded/levels.csv), [eval/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/evidence/a027-resume-20261015/completed/eval/levels.csv)

| policy | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | oor | size_log_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 27B raw | 8647 | 8643 | 0.443391 | 0.216144 | 0.533133 | 0.255952 | 0.282419 |
| 27B s1 | 8647 | 8643 | 0.742223 | 0.291315 | 0.637215 | 0.152637 | 0.282419 |
| 27B old checker-trigger O2 | 8647 | 8647 | 0.890135 | 0.343472 | 0.666821 | 0.108074 | 0.282447 |
| 27B new strict-trigger O2 | 8647 | 8647 | 0.895455 | 0.49416 | 0.718399 | 0.0996334 | 0.282447 |

### frontier_A — 离线升级策略回放 A

已保存候选的离线回放，非新端到端API部署。保留全部策略。A为300间GPT试点，B为1454间O2队列，C为300间多房间族GPT队列；A在B内。C与B重叠35个UID，共享89房屋组涵盖138间C房间（包含35间），非独立验证。

来源：[frontier-026-v4/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-v4/levels.csv)

| entry | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | escalated_share | big_used_share | oor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOCAL | 300 | 300 | 0.846667 | 0.446667 | 0.796667 | 0 | 0 | 0.0804652 |
| ESC-CHECK | 300 | 300 | 0.986667 | 0.503333 | 0.843333 | 0.153333 | 0.146667 | 0.0506977 |
| ESC-STRICT | 300 | 300 | 0.986667 | 0.753333 | 0.88 | 0.556667 | 0.406667 | 0.0420117 |
| ESC-ANY | 300 | 300 | 0.986667 | 0.753333 | 0.88 | 0.556667 | 0.406667 | 0.0420117 |
| BIG | 300 | 300 | 0.976667 | 0.723333 | 0.85 | 1 | 1 | 0.0354998 |
| SYSTEM-022 | 300 | 300 | 0.99 | 0.513333 | 0.833333 | 0.153333 | 0.146667 | 0.0511242 |

### frontier_B-raw — 离线升级策略回放 B-raw

已保存候选的离线回放，非新端到端API部署。保留全部策略。A为300间GPT试点，B为1454间O2队列，C为300间多房间族GPT队列；A在B内。C与B重叠35个UID，共享89房屋组涵盖138间C房间（包含35间），非独立验证。

来源：[frontier-026-v4/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-v4/levels.csv)

| entry | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | escalated_share | big_used_share | oor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOCAL | 1454 | 1454 | 0.854195 | 0.453232 | 0.820495 | 0 | 0 | 0.0832782 |
| ESC-CHECK | 1454 | 1454 | 0.913343 | 0.493122 | 0.839752 | 0.145805 | 0.0591472 | 0.0717434 |
| ESC-STRICT | 1454 | 1454 | 0.913343 | 0.69326 | 0.863824 | 0.547455 | 0.267538 | 0.0623362 |
| ESC-ANY | 1454 | 1454 | 0.913343 | 0.69326 | 0.863824 | 0.547455 | 0.267538 | 0.0623362 |
| BIG | 1454 | 1158 | 0.791609 | 0.65337 | 0.722146 | 1 | 1 | 0.0108326 |

### frontier_B-harness — 离线升级策略回放 B-harness

已保存候选的离线回放，非新端到端API部署。保留全部策略。A为300间GPT试点，B为1454间O2队列，C为300间多房间族GPT队列；A在B内。C与B重叠35个UID，共享89房屋组涵盖138间C房间（包含35间），非独立验证。

来源：[frontier-026-v4/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-v4/levels.csv)

| entry | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | escalated_share | big_used_share | oor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOCAL | 1454 | 1454 | 0.854195 | 0.453232 | 0.820495 | 0 | 0 | 0.0832782 |
| ESC-CHECK | 1454 | 1454 | 0.955983 | 0.50619 | 0.843191 | 0.145805 | 0.102476 | 0.0618537 |
| ESC-STRICT | 1454 | 1454 | 0.955983 | 0.724897 | 0.870014 | 0.547455 | 0.330812 | 0.0511762 |
| ESC-ANY | 1454 | 1454 | 0.955983 | 0.724897 | 0.870014 | 0.547455 | 0.330812 | 0.0511762 |
| BIG | 1454 | 1360 | 0.927785 | 0.700825 | 0.801238 | 1 | 1 | 0.0140283 |

### frontier_C — 离线升级策略回放 C

已保存候选的离线回放，非新端到端API部署。保留全部策略。A为300间GPT试点，B为1454间O2队列，C为300间多房间族GPT队列；A在B内。C与B重叠35个UID，共享89房屋组涵盖138间C房间（包含35间），非独立验证。

来源：[frontier-026-v4/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-v4/levels.csv)

| entry | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | escalated_share | big_used_share | oor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOCAL | 300 | 299 | 0.79 | 0.35 | 0.706667 | 0 | 0 | 0.12985 |
| ESC-CHECK | 300 | 300 | 0.99 | 0.43 | 0.77 | 0.21 | 0.203333 | 0.0877724 |
| ESC-STRICT | 300 | 300 | 0.99 | 0.643333 | 0.82 | 0.65 | 0.43 | 0.0787822 |
| ESC-ANY | 300 | 300 | 0.99 | 0.643333 | 0.82 | 0.65 | 0.43 | 0.0787822 |
| BIG | 300 | 300 | 0.973333 | 0.62 | 0.776667 | 1 | 1 | 0.0789884 |

### frontier_contrasts — 全部回放严格准入对比

全部总体严格准入配对差，包括不利的LOCAL及检查器触发结果；其他终点及分层均在无损导出内。

来源：[frontier-026-v4/contrasts.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-v4/contrasts.csv)

| panel | comparison | rooms | components | estimate | ci_low | ci_high | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | LOCAL - BIG | 300 | 257 | -0.276667 | -0.338928 | -0.216762 | 0 |
| A | ESC-CHECK - BIG | 300 | 257 | -0.22 | -0.276229 | -0.162924 | 0 |
| A | ESC-STRICT - BIG | 300 | 257 | 0.03 | 0.0105625 | 0.0511182 | 0 |
| A | ESC-ANY - BIG | 300 | 257 | 0.03 | 0.0105625 | 0.0511182 | 0 |
| A | SYSTEM-022 - BIG | 300 | 257 | -0.21 | -0.269899 | -0.152021 | 0 |
| A | SYSTEM-022 - ESC-CHECK | 300 | 257 | 0.01 | -0.00332226 | 0.0236507 | 0.21 |
| B-raw | LOCAL - BIG | 1454 | 989 | -0.200138 | -0.226476 | -0.170389 | 0 |
| B-raw | ESC-CHECK - BIG | 1454 | 989 | -0.160248 | -0.185894 | -0.133506 | 0 |
| B-raw | ESC-STRICT - BIG | 1454 | 989 | 0.03989 | 0.0299318 | 0.0498064 | 0 |
| B-raw | ESC-ANY - BIG | 1454 | 989 | 0.03989 | 0.0299318 | 0.0498064 | 0 |
| B-harness | LOCAL - BIG | 1454 | 989 | -0.247593 | -0.275842 | -0.219203 | 0 |
| B-harness | ESC-CHECK - BIG | 1454 | 989 | -0.194635 | -0.219634 | -0.169475 | 0 |
| B-harness | ESC-STRICT - BIG | 1454 | 989 | 0.0240715 | 0.0166548 | 0.0320707 | 0 |
| B-harness | ESC-ANY - BIG | 1454 | 989 | 0.0240715 | 0.0166548 | 0.0320707 | 0 |
| C | LOCAL - BIG | 300 | 246 | -0.27 | -0.316503 | -0.222213 | 0 |
| C | ESC-CHECK - BIG | 300 | 246 | -0.19 | -0.233439 | -0.143836 | 0 |
| C | ESC-STRICT - BIG | 300 | 246 | 0.0233333 | 0.00931605 | 0.0413829 | 0 |
| C | ESC-ANY - BIG | 300 | 246 | 0.0233333 | 0.00931605 | 0.0413829 | 0 |

### nothink_A — 未训练无思考控制 A

O2nt是关闭思考的未训练Qwen3.8-27B，训练后A同为27B；复用相同大模型候选。若含SYSTEM-022行，它是旧训练后系统的实际参考，不是O2nt系统。此处为训练系统/控制比较，非纯规模消融。

来源：[frontier-026-ablation-o2nt/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-ablation-o2nt/levels.csv)

| entry | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | escalated_share | big_used_share | oor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOCAL | 300 | 111 | 0.233333 | 0.16 | 0.206667 | 0 | 0 | 0.183249 |
| ESC-CHECK | 300 | 300 | 0.98 | 0.693333 | 0.843333 | 0.766667 | 0.766667 | 0.0347914 |
| ESC-STRICT | 300 | 300 | 0.98 | 0.736667 | 0.856667 | 0.84 | 0.813333 | 0.0339969 |
| ESC-ANY | 300 | 300 | 0.98 | 0.736667 | 0.856667 | 0.84 | 0.813333 | 0.0339969 |
| BIG | 300 | 300 | 0.976667 | 0.723333 | 0.85 | 1 | 1 | 0.0354998 |
| SYSTEM-022 | 300 | 300 | 0.99 | 0.513333 | 0.833333 | 0.153333 | 0.146667 | 0.0511242 |

### nothink_B-raw — 未训练无思考控制 B-raw

O2nt是关闭思考的未训练Qwen3.8-27B，训练后A同为27B；复用相同大模型候选。若含SYSTEM-022行，它是旧训练后系统的实际参考，不是O2nt系统。此处为训练系统/控制比较，非纯规模消融。

来源：[frontier-026-ablation-o2nt/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-ablation-o2nt/levels.csv)

| entry | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | escalated_share | big_used_share | oor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOCAL | 1454 | 513 | 0.19945 | 0.138239 | 0.187758 | 0 | 0 | 0.0840807 |
| ESC-CHECK | 1454 | 1263 | 0.810867 | 0.62586 | 0.733838 | 0.80055 | 0.615543 | 0.0370814 |
| ESC-STRICT | 1454 | 1263 | 0.810867 | 0.661623 | 0.73934 | 0.861761 | 0.65337 | 0.0354479 |
| ESC-ANY | 1454 | 1263 | 0.810867 | 0.661623 | 0.73934 | 0.861761 | 0.65337 | 0.0354479 |
| BIG | 1454 | 1158 | 0.791609 | 0.65337 | 0.722146 | 1 | 1 | 0.0108326 |

### nothink_B-harness — 未训练无思考控制 B-harness

O2nt是关闭思考的未训练Qwen3.8-27B，训练后A同为27B；复用相同大模型候选。若含SYSTEM-022行，它是旧训练后系统的实际参考，不是O2nt系统。此处为训练系统/控制比较，非纯规模消融。

来源：[frontier-026-ablation-o2nt/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-ablation-o2nt/levels.csv)

| entry | requests | complete_rooms | checker_rate | strict_rate | noreq_rate | escalated_share | big_used_share | oor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOCAL | 1454 | 513 | 0.19945 | 0.138239 | 0.187758 | 0 | 0 | 0.0840807 |
| ESC-CHECK | 1454 | 1390 | 0.931912 | 0.663686 | 0.80055 | 0.80055 | 0.73934 | 0.0261368 |
| ESC-STRICT | 1454 | 1390 | 0.931912 | 0.703576 | 0.80674 | 0.861761 | 0.781981 | 0.0239102 |
| ESC-ANY | 1454 | 1390 | 0.931912 | 0.703576 | 0.80674 | 0.861761 | 0.781981 | 0.0239102 |
| BIG | 1454 | 1360 | 0.927785 | 0.700825 | 0.801238 | 1 | 1 | 0.0140283 |

### nothink_all_contrasts — 训练后27B减无思考控制的全部配对差

严格触发时严格准入差为正，但检查器触发时严格准入差为负，因为训练后本地结果触发更少升级；必须同时报告。

来源：[frontier-026-ablation-o2nt/paired.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-ablation-o2nt/paired.json)

| comparison | metric | delta | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| A/LOCAL | checker | 0.613333 | 0.555556 | 0.669697 |
| A/LOCAL | strict | 0.286667 | 0.225914 | 0.345395 |
| A/LOCAL | noreq | 0.59 | 0.533981 | 0.647059 |
| A/LOCAL | escalated | 0 | 0 | 0 |
| A/ESC-CHECK | checker | 0.00666667 | 0 | 0.0170648 |
| A/ESC-CHECK | strict | -0.19 | -0.249169 | -0.135593 |
| A/ESC-CHECK | noreq | 0 | -0.0280374 | 0.0308219 |
| A/ESC-CHECK | escalated | -0.613333 | -0.669841 | -0.555556 |
| A/ESC-STRICT | checker | 0.00666667 | 0 | 0.0170648 |
| A/ESC-STRICT | strict | 0.0166667 | 0 | 0.0338983 |
| A/ESC-STRICT | noreq | 0.0233333 | 0.00673401 | 0.0424837 |
| A/ESC-STRICT | escalated | -0.283333 | -0.342105 | -0.223684 |
| B-raw/LOCAL | checker | 0.654746 | 0.627822 | 0.681177 |
| B-raw/LOCAL | strict | 0.314993 | 0.285326 | 0.34433 |
| B-raw/LOCAL | noreq | 0.632737 | 0.605511 | 0.661349 |
| B-raw/LOCAL | escalated | 0 | 0 | 0 |
| B-raw/ESC-CHECK | checker | 0.102476 | 0.0870432 | 0.118976 |
| B-raw/ESC-CHECK | strict | -0.132737 | -0.158763 | -0.107216 |
| B-raw/ESC-CHECK | noreq | 0.105915 | 0.0857344 | 0.126877 |
| B-raw/ESC-CHECK | escalated | -0.654746 | -0.681342 | -0.62803 |
| B-raw/ESC-STRICT | checker | 0.102476 | 0.0870432 | 0.118976 |
| B-raw/ESC-STRICT | strict | 0.0316369 | 0.0214533 | 0.0418782 |
| B-raw/ESC-STRICT | noreq | 0.124484 | 0.107014 | 0.142588 |
| B-raw/ESC-STRICT | escalated | -0.314305 | -0.344151 | -0.285037 |
| B-harness/LOCAL | checker | 0.654746 | 0.627822 | 0.681177 |
| B-harness/LOCAL | strict | 0.314993 | 0.285326 | 0.34433 |
| B-harness/LOCAL | noreq | 0.632737 | 0.605511 | 0.661349 |
| B-harness/LOCAL | escalated | 0 | 0 | 0 |
| B-harness/ESC-CHECK | checker | 0.0240715 | 0.0161858 | 0.032526 |
| B-harness/ESC-CHECK | strict | -0.157497 | -0.182936 | -0.133763 |
| B-harness/ESC-CHECK | noreq | 0.042641 | 0.0243902 | 0.0595981 |
| B-harness/ESC-CHECK | escalated | -0.654746 | -0.681342 | -0.62803 |
| B-harness/ESC-STRICT | checker | 0.0240715 | 0.0161858 | 0.032526 |
| B-harness/ESC-STRICT | strict | 0.0213205 | 0.0130246 | 0.0292439 |
| B-harness/ESC-STRICT | noreq | 0.0632737 | 0.0491468 | 0.0767296 |
| B-harness/ESC-STRICT | escalated | -0.314305 | -0.344151 | -0.285037 |

### critic_dev — critic受限编辑：dev

开发B=27B+S1，C=B+critic（246间）；测试B=旧022系统S，C=SC（174间）；仓库Markdown误留dev/B模板。最多两轮，几何JSON早于judge完成。

来源：[a025-dev/results.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a025-dev/results.json)

| arm | requests | geometry_success | accepted | admitted_no_required | oor | judge |
| --- | --- | --- | --- | --- | --- | --- |
| B | 246 | 0.776423 | 0.390244 | 0.752033 | 0.100223 | 5.90854 |
| C | 246 | 0.955285 | 0.695122 | 0.898374 | 0.0664476 | 7.32992 |

### critic_dev_contrasts — critic配对差：dev

C减B。开发judge为(functionality+layout)/2，244间配对，因此配对差不等于各臂独立均值之差；测试judge配对另列。

来源：[a025-dev/results.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a025-dev/results.json)

| metric | estimate | ci_low | ci_high | n |
| --- | --- | --- | --- | --- |
| geometry_success | 0.178862 | 0.127655 | 0.230769 | 246 |
| accepted | 0.304878 | 0.249976 | 0.3625 | 246 |
| admitted_no_required | 0.146341 | 0.102767 | 0.190876 | 246 |
| oor | -0.0337754 | -0.0419707 | -0.0256586 | 246 |
| judge | 1.39959 | 1.23622 | 1.5546 | 244 |

### critic_test — critic受限编辑：test

开发B=27B+S1，C=B+critic（246间）；测试B=旧022系统S，C=SC（174间）；仓库Markdown误留dev/B模板。最多两轮，几何JSON早于judge完成。

来源：[test-eval-geometry/results.json](/Volumes/harddisk/fastfillout/unknownsize-20261013/a025/test-eval-geometry/results.json)

| arm | requests | geometry_success | accepted | admitted_no_required | oor | judge |
| --- | --- | --- | --- | --- | --- | --- |
| B | 174 | 0.965517 | 0.597701 | 0.885057 | 0.032533 | — |
| C | 174 | 0.982759 | 0.890805 | 0.95977 | 0.0161848 | — |

### critic_test_contrasts — critic配对差：test

C减B。开发judge为(functionality+layout)/2，244间配对，因此配对差不等于各臂独立均值之差；测试judge配对另列。

来源：[test-eval-geometry/results.json](/Volumes/harddisk/fastfillout/unknownsize-20261013/a025/test-eval-geometry/results.json)

| metric | estimate | ci_low | ci_high | n |
| --- | --- | --- | --- | --- |
| geometry_success | 0.0172414 | 0 | 0.0454545 | 174 |
| accepted | 0.293103 | 0.222222 | 0.364643 | 174 |
| admitted_no_required | 0.0747126 | 0.0359281 | 0.119057 | 174 |
| oor | -0.0163482 | -0.0274674 | -0.00925665 | 174 |

### critic_test_judge — 测试critic评审分数

各臂174间中173间有分；S/SC配对173间、157组。功能与布局均值差0.7687861272，95%CI[0.6017964072,0.9487951807]；不等于可选overall字段。

来源：[a025-test/sc-vs-s-judge.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a025-test/sc-vs-s-judge.json)

| arm | n | fl | overall |
| --- | --- | --- | --- |
| S | 173 | 6.96243 | 6.8289 |
| A | 173 | 6.65896 | 6.58266 |
| SC | 173 | 7.73121 | 7.53931 |

### fid_all_populations — 全部匹配总体的FID/KID

代理交集数量不同，如O1a257、O1b615、O3409、O2998；J/P在各交集重算。不可跨不同总体直接排名FID。A为27B raw，S为旧检查器触发系统；024复用011特征，不是独立重复。

来源：[visual-024/fid-summary-A-S.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/visual-024/fid-summary-A-S.json), [fid-011-visual-002/fid-summary.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/fid-011-visual-002/fid-summary.json)

| population | system | n | fid | kid | kid_se |
| --- | --- | --- | --- | --- | --- |
| agent/A | J | 1005 | 22.3362 | 0.000410637 | 0.000119627 |
| agent/A | P | 1005 | 23.9664 | 0.00135157 | 0.000218434 |
| agent/A | A | 1005 | 22.8827 | 0.000678292 | 0.000140369 |
| agent/L | J | 1005 | 22.3362 | 0.000410637 | 0.000119627 |
| agent/L | P | 1005 | 23.9664 | 0.00135157 | 0.000218434 |
| agent/L | L | 1005 | 32.0814 | 0.00723334 | 0.000522936 |
| agent/O1a | J | 257 | 47.6925 | 0.000366033 | 0.000253612 |
| agent/O1a | P | 257 | 49.0519 | 0.000927887 | 0.000446689 |
| agent/O1a | O1a | 257 | 67.5334 | 0.0126612 | 0.00178835 |
| agent/O1b | J | 615 | 32.0015 | 0.000579483 | 0.00020693 |
| agent/O1b | P | 615 | 33.1777 | 0.00120273 | 0.000266917 |
| agent/O1b | O1b | 615 | 48.3527 | 0.012186 | 0.00104336 |
| agent/O2 | J | 998 | 22.4235 | 0.000408408 | 0.000120333 |
| agent/O2 | P | 998 | 24.0763 | 0.00137372 | 0.000221524 |
| agent/O2 | O2 | 998 | 33.7124 | 0.00949272 | 0.00066763 |
| agent/O3 | J | 409 | 40.6308 | 0.000244389 | 0.000200148 |
| agent/O3 | P | 409 | 42.9013 | 0.0013362 | 0.000356499 |
| agent/O3 | O3 | 409 | 54.8917 | 0.0111646 | 0.0010377 |
| agent/O4 | J | 1005 | 22.3362 | 0.000410637 | 0.000119627 |
| agent/O4 | P | 1005 | 23.9664 | 0.00135157 | 0.000218434 |
| agent/O4 | O4 | 1005 | 37.6873 | 0.0146504 | 0.000843257 |
| agent/O6 | J | 1005 | 22.3362 | 0.000410637 | 0.000119627 |
| agent/O6 | P | 1005 | 23.9664 | 0.00135157 | 0.000218434 |
| agent/O6 | O6 | 1005 | 34.3838 | 0.0102818 | 0.000686536 |
| agent/O7 | J | 1005 | 22.3362 | 0.000410637 | 0.000119627 |
| agent/O7 | P | 1005 | 23.9664 | 0.00135157 | 0.000218434 |
| agent/O7 | O7 | 1005 | 40.3713 | 0.0170094 | 0.000923177 |
| agent/S | J | 1005 | 22.3362 | 0.000410637 | 0.000119627 |
| agent/S | P | 1005 | 23.9664 | 0.00135157 | 0.000218434 |
| agent/S | S | 1005 | 22.7857 | 0.000699795 | 0.000142023 |
| judged | J | 1005 | 22.3362 | 0.000410637 | 0.000119627 |
| judged | P | 1005 | 23.9664 | 0.00135157 | 0.000218434 |
| judged | L | 1005 | 32.0814 | 0.00723334 | 0.000522936 |
| judged | O6 | 1005 | 34.3838 | 0.0102818 | 0.000686536 |
| primary | J | 1005 | 22.3362 | 0.000410637 | 0.000119627 |
| primary | P | 1005 | 23.9664 | 0.00135157 | 0.000218434 |

### mesh_all — 全部mesh碰撞比较及阈值

1009间选中，4间固定资产失败先排除，剩1005；代理schema失败使各臂分母继续减少，表内逐臂列出。主阈值500cm3，敏感性125/2000；为可动物体对体素碰撞，非严格准入。条件几何可有利于自选尺寸过小代理。

来源：[mesh-collision-019/summary.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/mesh-collision-019/summary.json), [visual-024/mesh-summary-A.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/visual-024/mesh-summary-A.json), [visual-024/mesh-summary-S.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/visual-024/mesh-summary-S.json)

| system | rooms | missing_rooms | checker_collision_free | mesh_free_125 | mesh_free_500 | mesh_free_2000 | mesh_free_deep_500 | a4_free |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GT | 1005 | 0 | 0.8 | 0.769154 | 0.797015 | 0.838806 | 0.847761 | 0.58806 |
| J | 1005 | 0 | 0.502488 | 0.383085 | 0.421891 | 0.476617 | 0.494527 | 0.315423 |
| L | 1005 | 0 | 0.99204 | 0.975124 | 0.979104 | 0.98607 | 0.995025 | 0.976119 |
| O1a | 257 | 748 | 0.435798 | 0.33463 | 0.350195 | 0.400778 | 0.40856 | 0.33463 |
| O1b | 615 | 390 | 0.515447 | 0.354472 | 0.393496 | 0.445528 | 0.466667 | 0.36748 |
| O2 | 998 | 7 | 0.996994 | 0.971944 | 0.975952 | 0.983968 | 0.997996 | 0.987976 |
| O3 | 409 | 596 | 0.755501 | 0.564792 | 0.589242 | 0.645477 | 0.647922 | 0.542787 |
| O4 | 1005 | 0 | 0.855721 | 0.79801 | 0.834826 | 0.880597 | 0.888557 | 0.81592 |
| O6 | 1005 | 0 | 0.912438 | 0.863682 | 0.895522 | 0.929353 | 0.937313 | 0.852736 |
| O7 | 1005 | 0 | 0.885572 | 0.829851 | 0.861692 | 0.905473 | 0.906468 | 0.79204 |
| P | 1005 | 0 | 0.621891 | 0.535323 | 0.548259 | 0.58806 | 0.60398 | 0.479602 |
| A | 1005 | 0 | 0.657711 | 0.552239 | 0.575124 | 0.632836 | 0.642786 | 0.464677 |
| S | 1005 | 0 | 0.991045 | 0.754229 | 0.775124 | 0.821891 | 0.832836 | 0.655721 |

### judge_all — 全部已保存视觉judge各臂均值

各臂选174间，列成功数。overall是模型可选返回值及回退规则，不能声称固定加权。标准正式终点为功能/布局/美观均值；critic改用功能/布局均值。

来源：[gpt-judge-visual-002/aggregate.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/gpt-judge-visual-002/aggregate.csv), [gpt-judge-023-astra-high/aggregate.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/gpt-judge-023-astra-high/aggregate.csv), [visual-024/judge-A-S-aggregate.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/visual-024/judge-A-S-aggregate.csv), [a025-test/judge-SC-aggregate.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/a025-test/judge-SC-aggregate.csv)

| judge | arm | selected | success | failed | functionality_mean | layout_mean | aesthetics_mean | overall_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.1-sol | J | 174 | 173 | 1 | 6.24277 | 6.54335 | 6.13873 | 6.3052 |
| gpt-6.1-sol | P | 174 | 170 | 4 | 6.04118 | 6.44706 | 6.00588 | 6.16118 |
| gpt-6.1-sol | L | 174 | 173 | 1 | 7.79769 | 8.43353 | 7.53757 | 7.91387 |
| gpt-6.1-sol | O6 | 174 | 167 | 7 | 6.79042 | 7.14371 | 6.56287 | 6.82994 |
| gpt-6-astra-high | J | 174 | 171 | 3 | 6.21637 | 6.45614 | 6.16374 | 6.27544 |
| gpt-6-astra-high | P | 174 | 172 | 2 | 5.92442 | 6.43023 | 6.11628 | 6.14942 |
| gpt-6-astra-high | L | 174 | 171 | 3 | 7.90058 | 8.49123 | 7.65497 | 8.00936 |
| gpt-6-astra-high | O6 | 174 | 170 | 4 | 6.74118 | 7.15294 | 6.63529 | 6.84 |
| gpt-6-astra-high | A | 174 | 173 | 1 | 6.42775 | 6.89017 | 6.43931 | 6.58266 |
| gpt-6-astra-high | S | 174 | 173 | 1 | 6.61272 | 7.31214 | 6.57225 | 6.8289 |
| gpt-6-astra-high | SC | 174 | 173 | 1 | 7.4104 | 8.05202 | 7.17341 | 7.53931 |
| gpt-6-astra-high | S | 0 | 0 | 0 | — | — | — | — |

### judge_common_means — 两个judge各自及共同成功集均值

696个图像case，675个共同完成；judge一致性不等于人类偏好有效性。区分各自及共同成功总体。

来源：[judge-agreement-023/means.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/judge-agreement-023/means.csv)

| judge | cases | arm | n | functionality | layout | aesthetics | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.1-sol | own | J | 173 | 6.24277 | 6.54335 | 6.13873 | 6.3052 |
| gpt-6.1-sol | own | L | 173 | 7.79769 | 8.43353 | 7.53757 | 7.91387 |
| gpt-6.1-sol | own | O6 | 167 | 6.79042 | 7.14371 | 6.56287 | 6.82994 |
| gpt-6.1-sol | own | P | 170 | 6.04118 | 6.44706 | 6.00588 | 6.16118 |
| gpt-6.1-sol | common | J | 171 | 6.24561 | 6.53801 | 6.14035 | 6.30468 |
| gpt-6.1-sol | common | L | 171 | 7.80117 | 8.44444 | 7.54971 | 7.92222 |
| gpt-6.1-sol | common | O6 | 164 | 6.78659 | 7.14024 | 6.56098 | 6.82683 |
| gpt-6.1-sol | common | P | 169 | 6.02959 | 6.43787 | 6 | 6.15207 |
| gpt-6-astra-high | own | J | 171 | 6.21637 | 6.45614 | 6.16374 | 6.27544 |
| gpt-6-astra-high | own | L | 171 | 7.90058 | 8.49123 | 7.65497 | 8.00936 |
| gpt-6-astra-high | own | O6 | 170 | 6.74118 | 7.15294 | 6.63529 | 6.84 |
| gpt-6-astra-high | own | P | 172 | 5.92442 | 6.43023 | 6.11628 | 6.14942 |
| gpt-6-astra-high | common | J | 171 | 6.21637 | 6.45614 | 6.16374 | 6.27544 |
| gpt-6-astra-high | common | L | 171 | 7.90058 | 8.49123 | 7.65497 | 8.00936 |
| gpt-6-astra-high | common | O6 | 164 | 6.76829 | 7.16463 | 6.64634 | 6.85671 |
| gpt-6-astra-high | common | P | 169 | 5.91716 | 6.4142 | 6.11834 | 6.1426 |

### judge_agreement — 共同case的judge一致性

共同完成675个case；两judge均排序L>O6>J>P，仍不能排除同模型家族偏差。

来源：[judge-agreement-023/agreement.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/judge-agreement-023/agreement.json)

| metric | n | spearman | pearson |
| --- | --- | --- | --- |
| functionality | 675 | 0.862871 | 0.879749 |
| layout | 675 | 0.890508 | 0.903164 |
| aesthetics | 675 | 0.856586 | 0.874488 |
| overall | 675 | 0.904245 | 0.914027 |

### size_only — 尺寸单独预测与训练先验

1877间并集；L-joint只覆盖1063间，O2-size完整1876间。保留各指标n。L-size/O2-size是只预测尺寸的调用，不是新布局；未完成放置部分仍属未测。

来源：[sizes-020/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/sizes-020/levels.csv)

| name | rooms | complete | size_log_mae | size_n | size_iou | bias_fp |
| --- | --- | --- | --- | --- | --- | --- |
| GT | 1877 | 1 | 0 | 1877 | 1 | 0 |
| prior | 1877 | 1 | 0.267445 | 1877 | 0.577011 | -0.00425932 |
| L-joint | 1063 | 1 | 0.262038 | 1063 | 0.532055 | -0.182366 |
| L-size | 1877 | 1 | 0.327269 | 1877 | 0.486756 | -0.198186 |
| O2-size | 1877 | 0.999467 | 0.40801 | 1876 | 0.420157 | -0.280255 |

### families_rescore_all — 房间族审计的总体重评分

属于重复评分，不是新生成。历史P严格通过数与test-apply-012差1间（1729与1728/8647），保留各自来源；逐族结果保留在原始导出。

来源：[families-015/levels.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/families-015/levels.csv)

| population | system | rooms | components | complete_rooms | geometry_success | downstream | size_log_mae | oor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | J | 8647 | 3056 | 8622 | 0.388227 | 0.17243 | 0.286434 | 0.146967 |
| test | P | 8647 | 3056 | 8627 | 0.398057 | 0.199954 | 0.281907 | 0.247899 |
| test | GT | 8647 | 3056 | 8647 | 0.723025 | 0.295247 | 0 | 0.0787176 |
| cohort | J | 1454 | 989 | 1452 | 0.430536 | 0.233838 | 0.242999 | 0.110675 |
| cohort | P | 1454 | 989 | 1454 | 0.519257 | 0.354195 | 0.238788 | 0.173479 |
| cohort | GT | 1454 | 989 | 1454 | 0.724897 | 0.448418 | 0 | 0.048117 |
| cohort | L-raw | 1454 | 989 | 1454 | 0.934663 | 0.727648 | 0.300281 | 0.0309544 |
| cohort | L-harness | 1454 | 989 | 1454 | 0.934663 | 0.727648 | 0.300281 | 0.0309544 |
| cohort | O1a-raw | 1454 | 989 | 506 | 0 | 0 | 0.608834 | 0.43011 |
| cohort | O1a-harness | 1454 | 989 | 508 | 0 | 0 | 0.608914 | 0.428636 |
| cohort | O1b-raw | 1454 | 989 | 344 | 0.042641 | 0.0295736 | 0.534947 | 0.0770619 |
| cohort | O1b-harness | 1454 | 989 | 836 | 0.110041 | 0.0687758 | 0.538794 | 0.0911787 |
| cohort | O2-raw | 1454 | 989 | 1202 | 0.808116 | 0.674003 | 0.329836 | 0.013457 |
| cohort | O2-harness | 1454 | 989 | 1393 | 0.927785 | 0.722834 | 0.340516 | 0.0168534 |
| cohort | O3-raw | 1454 | 989 | 482 | 0.211142 | 0.138239 | 0.451837 | 0.0350771 |
| cohort | O3-harness | 1454 | 989 | 573 | 0.241403 | 0.158872 | 0.450069 | 0.0335619 |
| cohort | O4-raw | 1454 | 989 | 1451 | 0.68707 | 0.479367 | 0.446051 | 0.038702 |
| cohort | O4-harness | 1454 | 989 | 1454 | 0.68707 | 0.479367 | 0.446754 | 0.121646 |
| cohort | O6-raw | 1454 | 989 | 1449 | 0.700138 | 0.535763 | 0.421555 | 0.0232091 |
| cohort | O6-harness | 1454 | 989 | 1452 | 0.700825 | 0.535763 | 0.421761 | 0.0241847 |
| cohort | O7-raw | 1454 | 989 | 1425 | 0.579092 | 0.400275 | 0.384408 | 0.0422761 |
| cohort | O7-harness | 1454 | 989 | 1454 | 0.59216 | 0.409904 | 0.385992 | 0.092827 |

### scale_checker_xy — 离线尺寸敏感性：checker/xy

固定已生成姿态后缩放/替换尺寸；这是压力测试，非重新生成的同尺寸方法。s_match可缺失/不匹配；列含义按来源定义。

来源：[scale-curve-016/summary.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/scale-curve-016/summary.csv)

| scope | system | rooms | rate_s1 | slope | slope_ci_low | slope_ci_high | s_match | unmatched | pass_prior | own_minus_prior | pass_gt | own_minus_gt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all-test | GT | 8647 | 0.723025 | -0.685093 | -0.739655 | -0.628532 | 1 | — | 0.432404 | 0.290621 | 0.723025 | 0 |
| all-test | J | 8647 | 0.388227 | -0.709379 | -0.754782 | -0.667127 | — | below_gt_everywhere | 0.316526 | 0.0717012 | 0.238811 | 0.149416 |
| all-test | P | 8647 | 0.398057 | -0.618712 | -0.655418 | -0.579996 | — | below_gt_everywhere | 0.398057 | 0 | 0.26275 | 0.135307 |
| cohort | GT | 1454 | 0.724897 | -0.664374 | -0.79067 | -0.538451 | 1 | — | 0.431912 | 0.292985 | 0.724897 | 0 |
| cohort | J | 1454 | 0.430536 | -0.862448 | -0.9662 | -0.756394 | — | below_gt_everywhere | 0.371389 | 0.0591472 | 0.238652 | 0.191884 |
| cohort | P | 1454 | 0.519257 | -0.58597 | -0.693092 | -0.484362 | — | below_gt_everywhere | 0.519257 | 0 | 0.275791 | 0.243466 |
| cohort | L-raw | 1454 | 0.934663 | -0.104539 | -0.139938 | -0.0701742 | — | above_gt_everywhere | 0.555021 | 0.379642 | 0.359697 | 0.574966 |
| cohort | L-harness | 1454 | 0.934663 | -0.104539 | -0.139938 | -0.0701742 | — | above_gt_everywhere | 0.555021 | 0.379642 | 0.359697 | 0.574966 |
| cohort | O1a-raw | 1454 | 0 | 0 | 0 | 0 | — | below_gt_everywhere | 0 | 0 | 0 | 0 |
| cohort | O1a-harness | 1454 | 0 | 0 | 0 | 0 | — | below_gt_everywhere | 0 | 0 | 0 | 0 |
| cohort | O1b-raw | 1454 | 0.042641 | -0.0522696 | -0.0829931 | -0.0251661 | — | below_gt_everywhere | 0.0206327 | 0.0220083 | 0.0165062 | 0.0261348 |
| cohort | O1b-harness | 1454 | 0.110041 | -0.140303 | -0.191013 | -0.09007 | — | below_gt_everywhere | 0.0570839 | 0.0529574 | 0.0474553 | 0.062586 |
| cohort | O2-raw | 1454 | 0.808116 | -0.0220083 | -0.0398941 | -0.00664675 | 1.05639 | — | 0.355571 | 0.452545 | 0.20564 | 0.602476 |
| cohort | O2-harness | 1454 | 0.927785 | -0.0385144 | -0.0613672 | -0.0173209 | 1.0771 | — | 0.389959 | 0.537827 | 0.231087 | 0.696699 |
| cohort | O3-raw | 1454 | 0.211142 | -0.18707 | -0.243841 | -0.132945 | — | below_gt_everywhere | 0.0790922 | 0.13205 | 0.0674003 | 0.143741 |
| cohort | O3-harness | 1454 | 0.241403 | -0.246217 | -0.309058 | -0.189727 | — | below_gt_everywhere | 0.0859697 | 0.155433 | 0.0729023 | 0.168501 |
| cohort | O4-raw | 1454 | 0.68707 | -0.451169 | -0.526932 | -0.376792 | 0.908621 | — | 0.224209 | 0.462861 | 0.176066 | 0.511004 |
| cohort | O4-harness | 1454 | 0.68707 | -0.451169 | -0.526932 | -0.376792 | 0.908621 | — | 0.224209 | 0.462861 | 0.176066 | 0.511004 |
| cohort | O6-raw | 1454 | 0.700138 | -0.273728 | -0.333088 | -0.220377 | 0.9275 | — | 0.178817 | 0.52132 | 0.13205 | 0.568088 |
| cohort | O6-harness | 1454 | 0.700825 | -0.273728 | -0.333088 | -0.220377 | 0.93 | — | 0.178817 | 0.522008 | 0.13205 | 0.568776 |
| cohort | O7-raw | 1454 | 0.579092 | -0.345254 | -0.408222 | -0.280651 | — | below_gt_everywhere | 0.255846 | 0.323246 | 0.175378 | 0.403714 |
| cohort | O7-harness | 1454 | 0.59216 | -0.353508 | -0.418346 | -0.287074 | — | below_gt_everywhere | 0.263411 | 0.328748 | 0.177442 | 0.414718 |

### scale_checker_xyz — 离线尺寸敏感性：checker/xyz

固定已生成姿态后缩放/替换尺寸；这是压力测试，非重新生成的同尺寸方法。s_match可缺失/不匹配；列含义按来源定义。

来源：[scale-curve-016/summary.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/scale-curve-016/summary.csv)

| scope | system | rooms | rate_s1 | slope | slope_ci_low | slope_ci_high | s_match | unmatched | pass_prior | own_minus_prior | pass_gt | own_minus_gt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all-test | GT | 8647 | 0.723025 | -0.698971 | -0.752518 | -0.645073 | 1 | — | 0.425465 | 0.29756 | 0.723025 | 0 |
| all-test | J | 8647 | 0.388227 | -0.726495 | -0.772927 | -0.683006 | — | below_gt_everywhere | 0.313172 | 0.0750549 | 0.240546 | 0.147681 |
| all-test | P | 8647 | 0.398057 | -0.636984 | -0.675649 | -0.595825 | — | below_gt_everywhere | 0.398057 | 0 | 0.268417 | 0.12964 |
| cohort | GT | 1454 | 0.724897 | -0.66575 | -0.79139 | -0.540924 | 1 | — | 0.4326 | 0.292297 | 0.724897 | 0 |
| cohort | J | 1454 | 0.430536 | -0.866575 | -0.971321 | -0.759512 | — | below_gt_everywhere | 0.36795 | 0.062586 | 0.238652 | 0.191884 |
| cohort | P | 1454 | 0.519257 | -0.592847 | -0.698279 | -0.492683 | — | below_gt_everywhere | 0.519257 | 0 | 0.275791 | 0.243466 |
| cohort | L-raw | 1454 | 0.934663 | -0.104539 | -0.139938 | -0.0701742 | — | above_gt_everywhere | 0.549519 | 0.385144 | 0.356946 | 0.577717 |
| cohort | L-harness | 1454 | 0.934663 | -0.104539 | -0.139938 | -0.0701742 | — | above_gt_everywhere | 0.549519 | 0.385144 | 0.356946 | 0.577717 |
| cohort | O1a-raw | 1454 | 0 | 0 | 0 | 0 | — | below_gt_everywhere | 0 | 0 | 0 | 0 |
| cohort | O1a-harness | 1454 | 0 | 0 | 0 | 0 | — | below_gt_everywhere | 0 | 0 | 0 | 0 |
| cohort | O1b-raw | 1454 | 0.042641 | -0.0522696 | -0.0829931 | -0.0251661 | — | below_gt_everywhere | 0.0206327 | 0.0220083 | 0.0165062 | 0.0261348 |
| cohort | O1b-harness | 1454 | 0.110041 | -0.140303 | -0.191013 | -0.09007 | — | below_gt_everywhere | 0.0577717 | 0.0522696 | 0.0467675 | 0.0632737 |
| cohort | O2-raw | 1454 | 0.808116 | -0.0220083 | -0.0398941 | -0.00664675 | 1.05639 | — | 0.350069 | 0.458047 | 0.202889 | 0.605227 |
| cohort | O2-harness | 1454 | 0.927785 | -0.0385144 | -0.0613672 | -0.0173209 | 1.0771 | — | 0.381706 | 0.54608 | 0.226272 | 0.701513 |
| cohort | O3-raw | 1454 | 0.211142 | -0.18707 | -0.243841 | -0.132945 | — | below_gt_everywhere | 0.0777166 | 0.133425 | 0.0667125 | 0.144429 |
| cohort | O3-harness | 1454 | 0.241403 | -0.246217 | -0.309058 | -0.189727 | — | below_gt_everywhere | 0.0845942 | 0.156809 | 0.0722146 | 0.169188 |
| cohort | O4-raw | 1454 | 0.68707 | -0.455296 | -0.531752 | -0.379267 | 0.908621 | — | 0.22077 | 0.4663 | 0.174003 | 0.513067 |
| cohort | O4-harness | 1454 | 0.68707 | -0.455296 | -0.531752 | -0.379267 | 0.908621 | — | 0.22077 | 0.4663 | 0.174003 | 0.513067 |
| cohort | O6-raw | 1454 | 0.700138 | -0.273728 | -0.333088 | -0.220377 | 0.9275 | — | 0.178129 | 0.522008 | 0.13205 | 0.568088 |
| cohort | O6-harness | 1454 | 0.700825 | -0.273728 | -0.333088 | -0.220377 | 0.93 | — | 0.178129 | 0.522696 | 0.13205 | 0.568776 |
| cohort | O7-raw | 1454 | 0.579092 | -0.345254 | -0.408222 | -0.280651 | — | below_gt_everywhere | 0.253783 | 0.325309 | 0.173315 | 0.405777 |
| cohort | O7-harness | 1454 | 0.59216 | -0.353508 | -0.418346 | -0.287074 | — | below_gt_everywhere | 0.26066 | 0.331499 | 0.175378 | 0.416781 |

### scale_downstream_xy — 离线尺寸敏感性：downstream/xy

固定已生成姿态后缩放/替换尺寸；这是压力测试，非重新生成的同尺寸方法。s_match可缺失/不匹配；列含义按来源定义。

来源：[scale-curve-016/summary.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/scale-curve-016/summary.csv)

| scope | system | rooms | rate_s1 | slope | slope_ci_low | slope_ci_high | s_match | unmatched | pass_prior | own_minus_prior | pass_gt | own_minus_gt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all-test | GT | 8647 | 0.295247 | -0.87036 | -0.909522 | -0.828522 | 1 | — | 0.208396 | 0.0868509 | 0.295247 | 0 |
| all-test | J | 8647 | 0.17243 | -0.661964 | -0.708162 | -0.618976 | 0.817339 | — | 0.152192 | 0.0202382 | 0.110559 | 0.0618712 |
| all-test | P | 8647 | 0.199838 | -0.606222 | -0.639716 | -0.571395 | 0.847174 | — | 0.199838 | 0 | 0.120736 | 0.0791026 |
| cohort | GT | 1454 | 0.448418 | -0.935351 | -1.04584 | -0.82637 | 1 | — | 0.269601 | 0.178817 | 0.448418 | 0 |
| cohort | J | 1454 | 0.233838 | -1.06327 | -1.17403 | -0.955248 | 0.798837 | — | 0.227648 | 0.00618982 | 0.137552 | 0.0962861 |
| cohort | P | 1454 | 0.354195 | -0.858322 | -0.956778 | -0.759041 | 0.897692 | — | 0.354195 | 0 | 0.175378 | 0.178817 |
| cohort | L-raw | 1454 | 0.727648 | -0.372765 | -0.441802 | -0.300467 | 1.09889 | — | 0.339065 | 0.388583 | 0.20564 | 0.522008 |
| cohort | L-harness | 1454 | 0.727648 | -0.372765 | -0.441802 | -0.300467 | 1.09889 | — | 0.339065 | 0.388583 | 0.20564 | 0.522008 |
| cohort | O1a-raw | 1454 | 0 | 0 | 0 | 0 | — | below_gt_everywhere | 0 | 0 | 0 | 0 |
| cohort | O1a-harness | 1454 | 0 | 0 | 0 | 0 | — | below_gt_everywhere | 0 | 0 | 0 | 0 |
| cohort | O1b-raw | 1454 | 0.0295736 | -0.0371389 | -0.0593124 | -0.0184436 | — | below_gt_everywhere | 0.0110041 | 0.0185695 | 0.00825309 | 0.0213205 |
| cohort | O1b-harness | 1454 | 0.0687758 | -0.125172 | -0.165301 | -0.0877119 | — | below_gt_everywhere | 0.0371389 | 0.0316369 | 0.0281981 | 0.0405777 |
| cohort | O2-raw | 1454 | 0.674003 | -0.283356 | -0.340234 | -0.219796 | 1.07362 | — | 0.194635 | 0.479367 | 0.121733 | 0.55227 |
| cohort | O2-harness | 1454 | 0.722834 | -0.310867 | -0.372615 | -0.242876 | 1.08073 | — | 0.203576 | 0.519257 | 0.130674 | 0.59216 |
| cohort | O3-raw | 1454 | 0.138239 | -0.261348 | -0.335423 | -0.194872 | — | below_gt_everywhere | 0.0495186 | 0.0887208 | 0.0371389 | 0.1011 |
| cohort | O3-harness | 1454 | 0.158872 | -0.314993 | -0.39039 | -0.245293 | — | below_gt_everywhere | 0.0543329 | 0.104539 | 0.03989 | 0.118982 |
| cohort | O4-raw | 1454 | 0.479367 | -0.393398 | -0.466576 | -0.32413 | 1.01364 | — | 0.151307 | 0.328061 | 0.112105 | 0.367263 |
| cohort | O4-harness | 1454 | 0.479367 | -0.393398 | -0.466576 | -0.32413 | 1.01364 | — | 0.151307 | 0.328061 | 0.112105 | 0.367263 |
| cohort | O6-raw | 1454 | 0.535763 | -0.361761 | -0.42847 | -0.294794 | 1.03588 | — | 0.10729 | 0.428473 | 0.0866575 | 0.449106 |
| cohort | O6-harness | 1454 | 0.535763 | -0.361761 | -0.42847 | -0.294794 | 1.03588 | — | 0.10729 | 0.428473 | 0.0866575 | 0.449106 |
| cohort | O7-raw | 1454 | 0.400275 | -0.41403 | -0.487068 | -0.342856 | 0.881429 | — | 0.152682 | 0.247593 | 0.103164 | 0.297111 |
| cohort | O7-harness | 1454 | 0.409904 | -0.429161 | -0.50497 | -0.359092 | 0.906061 | — | 0.156121 | 0.253783 | 0.104539 | 0.305365 |

### scale_downstream_xyz — 离线尺寸敏感性：downstream/xyz

固定已生成姿态后缩放/替换尺寸；这是压力测试，非重新生成的同尺寸方法。s_match可缺失/不匹配；列含义按来源定义。

来源：[scale-curve-016/summary.csv](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/scale-curve-016/summary.csv)

| scope | system | rooms | rate_s1 | slope | slope_ci_low | slope_ci_high | s_match | unmatched | pass_prior | own_minus_prior | pass_gt | own_minus_gt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all-test | GT | 8647 | 0.295247 | -0.884931 | -0.92256 | -0.843432 | 1 | — | 0.20724 | 0.0880074 | 0.295247 | 0 |
| all-test | J | 8647 | 0.17243 | -0.670522 | -0.718194 | -0.627033 | 0.81996 | — | 0.151266 | 0.0211634 | 0.110096 | 0.0623338 |
| all-test | P | 8647 | 0.199838 | -0.613392 | -0.647959 | -0.578916 | 0.848523 | — | 0.199838 | 0 | 0.121314 | 0.0785243 |
| cohort | GT | 1454 | 0.448418 | -0.935351 | -1.04584 | -0.82637 | 1 | — | 0.269601 | 0.178817 | 0.448418 | 0 |
| cohort | J | 1454 | 0.233838 | -1.06602 | -1.17539 | -0.957154 | 0.8 | — | 0.226272 | 0.00756534 | 0.137552 | 0.0962861 |
| cohort | P | 1454 | 0.354195 | -0.861073 | -0.96022 | -0.761214 | 0.897692 | — | 0.354195 | 0 | 0.176066 | 0.178129 |
| cohort | L-raw | 1454 | 0.727648 | -0.378267 | -0.446628 | -0.304843 | 1.09871 | — | 0.339065 | 0.388583 | 0.206327 | 0.52132 |
| cohort | L-harness | 1454 | 0.727648 | -0.378267 | -0.446628 | -0.304843 | 1.09871 | — | 0.339065 | 0.388583 | 0.206327 | 0.52132 |
| cohort | O1a-raw | 1454 | 0 | 0 | 0 | 0 | — | below_gt_everywhere | 0 | 0 | 0 | 0 |
| cohort | O1a-harness | 1454 | 0 | 0 | 0 | 0 | — | below_gt_everywhere | 0 | 0 | 0 | 0 |
| cohort | O1b-raw | 1454 | 0.0295736 | -0.0371389 | -0.0593124 | -0.0184436 | — | below_gt_everywhere | 0.0110041 | 0.0185695 | 0.00825309 | 0.0213205 |
| cohort | O1b-harness | 1454 | 0.0687758 | -0.127923 | -0.168319 | -0.0901807 | — | below_gt_everywhere | 0.0371389 | 0.0316369 | 0.0275103 | 0.0412655 |
| cohort | O2-raw | 1454 | 0.674003 | -0.283356 | -0.340234 | -0.219796 | 1.07362 | — | 0.193948 | 0.480055 | 0.120358 | 0.553645 |
| cohort | O2-harness | 1454 | 0.722834 | -0.317744 | -0.379858 | -0.252929 | 1.08073 | — | 0.202889 | 0.519945 | 0.129298 | 0.593535 |
| cohort | O3-raw | 1454 | 0.138239 | -0.261348 | -0.335423 | -0.194872 | — | below_gt_everywhere | 0.0495186 | 0.0887208 | 0.0371389 | 0.1011 |
| cohort | O3-harness | 1454 | 0.158872 | -0.314993 | -0.39039 | -0.245293 | — | below_gt_everywhere | 0.0543329 | 0.104539 | 0.03989 | 0.118982 |
| cohort | O4-raw | 1454 | 0.479367 | -0.393398 | -0.466576 | -0.32413 | 1.01364 | — | 0.151307 | 0.328061 | 0.112105 | 0.367263 |
| cohort | O4-harness | 1454 | 0.479367 | -0.393398 | -0.466576 | -0.32413 | 1.01364 | — | 0.151307 | 0.328061 | 0.112105 | 0.367263 |
| cohort | O6-raw | 1454 | 0.535763 | -0.361761 | -0.42847 | -0.294794 | 1.03588 | — | 0.10729 | 0.428473 | 0.0873453 | 0.448418 |
| cohort | O6-harness | 1454 | 0.535763 | -0.361761 | -0.42847 | -0.294794 | 1.03588 | — | 0.10729 | 0.428473 | 0.0873453 | 0.448418 |
| cohort | O7-raw | 1454 | 0.400275 | -0.41403 | -0.487068 | -0.342856 | 0.881429 | — | 0.152682 | 0.247593 | 0.102476 | 0.297799 |
| cohort | O7-harness | 1454 | 0.409904 | -0.429161 | -0.50497 | -0.359092 | 0.906061 | — | 0.156121 | 0.253783 | 0.103851 | 0.306052 |

### frontier_resource_costs — 全部回放已记录及分摊资源成本

成本按全部请求房间分摊。A的升级token由总用量及另一次运行的46间估算；BIG总量及调用数实测。B墙钟/GPU秒由批运行按输出token分摊，排除加载/进程开销，不是仅运行升级房间的实测延迟。C有逐房receipt。原始嵌套成本对象完整保留。

来源：[frontier-026-v4/results.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/frontier-026-v4/results.json)

| panel | entry | rooms | escalated | big_used | calls_per_room | gpu_s_per_room | input_tokens_per_room | local_gpu_s_per_room | output_tokens_per_room | reasoning_tokens_per_room | wall_equal_s_per_room | wall_s_per_room |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | LOCAL | 300 | 0 | 0 | 0 | — | 0 | 0.409391 | 0 | 0 | — | — |
| A | ESC-CHECK | 300 | 46 | 44 | 0.153333 | — | 808.703 | 0.409391 | 193.943 | 96.91 | — | — |
| A | ESC-STRICT | 300 | 167 | 122 | 0.556667 | — | 2735.98 | 0.409391 | 362.876 | 198.774 | — | — |
| A | ESC-ANY | 300 | 167 | 122 | 0.556667 | — | 2735.98 | 0.409391 | 362.876 | 198.774 | — | — |
| A | BIG | 300 | 300 | 300 | 1 | — | 4854.38 | 0 | 548.563 | 310.74 | — | — |
| A | SYSTEM-022 | 300 | 46 | 44 | 0.173333 | — | 808.703 | 0.409391 | 193.943 | 96.91 | — | — |
| B-raw | LOCAL | 1454 | 0 | 0 | 0 | 0 | 0 | 0.409391 | 0 | 0 | 0 | 0 |
| B-raw | ESC-CHECK | 1454 | 212 | 86 | 0.145805 | 1.83565 | 175.51 | 0.409391 | 2451.19 | 2421.95 | 0.763982 | 0.917824 |
| B-raw | ESC-STRICT | 1454 | 796 | 389 | 0.547455 | 4.62252 | 472.065 | 0.409391 | 6972.69 | 6875.72 | 2.07117 | 2.31126 |
| B-raw | ESC-ANY | 1454 | 796 | 389 | 0.547455 | 4.62252 | 472.065 | 0.409391 | 6972.69 | 6875.72 | 2.07117 | 2.31126 |
| B-raw | BIG | 1454 | 1454 | 1454 | 1 | 6.12021 | 727.963 | 0 | 9649.32 | 9486.07 | 3.06011 | 3.06011 |
| B-harness | LOCAL | 1454 | 0 | 0 | 0 | 0 | 0 | 0.409391 | 0 | 0 | 0 | 0 |
| B-harness | ESC-CHECK | 1454 | 212 | 149 | 0.289546 | 4.85407 | 392.157 | 0.409391 | 5250.07 | 5193.92 | 2.15286 | 2.42703 |
| B-harness | ESC-STRICT | 1454 | 796 | 481 | 0.830812 | 10.5997 | 842.723 | 0.409391 | 11992.8 | 11834 | 4.93217 | 5.29984 |
| B-harness | ESC-ANY | 1454 | 796 | 481 | 0.830812 | 10.5997 | 842.723 | 0.409391 | 11992.8 | 11834 | 4.93217 | 5.29984 |
| B-harness | BIG | 1454 | 1454 | 1454 | 1.30605 | 12.4466 | 1114.5 | 0 | 14852.4 | 14624.7 | 6.22332 | 6.22332 |
| C | LOCAL | 300 | 0 | 0 | 0 | — | 0 | 0.409391 | 0 | 0 | — | — |
| C | ESC-CHECK | 300 | 63 | 61 | 0.21 | — | 1193.95 | 0.409391 | 360.87 | 151.99 | — | — |
| C | ESC-STRICT | 300 | 195 | 129 | 0.65 | — | 3399.5 | 0.409391 | 655.677 | 296.853 | — | — |
| C | ESC-ANY | 300 | 195 | 129 | 0.65 | — | 3399.5 | 0.409391 | 655.677 | 296.853 | — | — |
| C | BIG | 300 | 300 | 300 | 1 | — | 5051.16 | 0 | 746.997 | 347.453 | — | — |

### mesh_checker_diagnostics — 500cm3阈值下mesh与盒检查器诊断

各比率的物体对分母不同并明确列出：确认率按被盒检查器标记的对，漏报率按mesh碰撞对。这是对级诊断，非房间成功率。

来源：[mesh-collision-019/summary.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/mesh-collision-019/summary.json), [visual-024/mesh-summary-A.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/visual-024/mesh-summary-A.json), [visual-024/mesh-summary-S.json](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/visual-024/mesh-summary-S.json)

| system | mesh_colliding_pairs | checker_main_severe_confirmed | checker_main_severe_confirmed_counts | a4_confirmed | a4_confirmed_counts | checker_missed | checker_missed_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GT | 409 | 0.622691 | 236/379 | 0.402229 | 397/987 | 0.422983 | 173/409 |
| J | 1208 | 0.83556 | 813/973 | 0.70136 | 1186/1691 | 0.326987 | 395/1208 |
| L | 28 | 0.666667 | 6/9 | 0.266667 | 8/30 | 0.785714 | 22/28 |
| O1a | 466 | 0.836478 | 266/318 | 0.604527 | 454/751 | 0.429185 | 200/466 |
| O1b | 959 | 0.832797 | 518/622 | 0.822507 | 899/1093 | 0.459854 | 441/959 |
| O2 | 34 | 1 | 4/4 | 0.307692 | 4/13 | 0.882353 | 30/34 |
| O3 | 283 | 0.796992 | 106/133 | 0.741667 | 267/360 | 0.625442 | 177/283 |
| O4 | 232 | 0.72 | 144/200 | 0.639731 | 190/297 | 0.37931 | 88/232 |
| O6 | 146 | 0.745763 | 88/118 | 0.501992 | 126/251 | 0.39726 | 58/146 |
| O7 | 217 | 0.632432 | 117/185 | 0.493298 | 184/373 | 0.460829 | 100/217 |
| P | 910 | 0.865014 | 628/726 | 0.727796 | 885/1216 | 0.30989 | 282/910 |
| A | 768 | 0.813433 | 436/536 | 0.622924 | 750/1204 | 0.432292 | 332/768 |
| S | 313 | 0.888889 | 8/9 | 0.460548 | 286/621 | 0.974441 | 305/313 |

