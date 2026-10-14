# Execution log

- 2026-10-13: user selected unknown-size joint prediction, then authorized autonomous server operation and full autoresearch until OptiScene-style tables. Existing user jobs preserved. Agency workers assigned independent contract, grouped size prior, server preflight and real-asset rendering tasks. Pilot and scoring budgets locked in protocol before new runs. No chat-loop tool available; implement durable Slurm/receipt continuity instead.

- Pilot-v2 built965 train/58dev requests with identical J/P membership and no test inputs. Common generic Qwen3-8B five complete weight payloads freshly hashed on server; all match pinned download receipts. Data code review found a pre-read-hash race; reproduced by a failing test and fixed with hashing the actual consumed stream (8 data tests pass). Pilot-v3 rebuilding under final hashes.
- Dev/train-only supervision audit: replacing dimensions while retaining reference positions makes660/965 pilot training rooms violate at least one declared geometry check (reference289/965, primarily main-collision category). Engineering pilot remains explicitly non-quality. Before main training, P requires fixed-dimension pose/support repair and joint admission with immutable OOF fitting pool; full release component closure is also required. Dedicated Agency implementation started.

## 2026-10-13 00:36 UTC — execution launched

- CPU preflight completed (258.8s before EOS-cache optimization): common train963/965, dev58/58; paired lengths and token counts archived server-side.
- Pilot GPU job7365478 submitted,1GPU30min,8updates perarm and clean-load probes. First submission attempt had a nested snapshot-path error and created no job; corrected actual path recorded in the job receipt.
- Full-release prepared identity closure CPU job7365479 submitted,16CPU64GB2h cap,12workers, noGPU.
- Image API capability succeeded with existing remote secret and exact returned gpt-6.1-sol; first request403, explicit retry with the already proven application User-Agent succeeded. One synthetic case only, not model-quality measurement.
- Independent pose review two P2 findings and renderer review one P1/two P2 findings delegated for tested fixes before main/bulk execution.

## GPU pilot finding

Job7365478 ended FAILED exit1 after67s. Joint completed8updates (19.21s training,loss0.7819) and saved its adapter, but the strict zero-CUDA-allocation check before cleanreload failed. This is not a completed pilot. Trainer internals can retain allocator tensor references despite cleanup; fix uses a fresh process for checkpoint reload after the training process exits, rather than relaxing provenance or pretending adapter existence is completion. Placement was correctly not launched. Originalrun preserved.

Full-corpus job7365479 passed reconstruction/closure checks but failed final publication because Lustre rejects renameat2(RENAME_NOREPLACE), errno22. No output corpus published; temp files cleaned. Added exclusive mkdir + exclusive hardlinks with MANIFEST-last fallback, 74corpus/eval tests passed. Retry uses newimmutable snapshot/output. Pilot retry7365524 uses isolated process handoff; total requested pilotGPU cap1h.

## Verified pilot and corpus completion

Pilot7365524: botharms8updates saved and reloaded in separate processes, ENGINEERING_COMPLETE after276.66s. J/P trainruntime18.875/19.704s, peakallocated30.62/30.65GB, schema-valid probes0/2 vs2/2. J failures are Markdown code fences (976/236generatedtokens, notcap); retained as strict-format failures, noqualityclaim andno parser relaxation.

Corpus7365528 rebuilt all160964releaseUIDs,188670rawrows including27706unreleased aliases,56352components; zero duplicates, unmatchedUIDs, rejects, source mismatches or cross-split components. Lustre manifest-last publication successful.

## 2026-10-13 05:20 UTC — main data complete, full matched SFT active

Main data job7365659 completed0:0 after3:36:08:98,923 common qualified train,8,167dev,8,647test; prior fit pool124,843train,25,920 eligible candidate rejections retained. Both training jobs7365681/7365682 active,24,689 optimizer steps scheduled. Exact token-length admission in each run lengths.json. Devbatch16 capacity passed; amendment records samegreedy decode+logits_to_keep1 before any new test output. Frozen evalsnapshot eval-6f75587fcaae, SHA8fdc07ad78ad8913eff69dd6bfe129a4b68e5c2786a706d517352c257cc33010. Inferencejobs7367601/7367602 depend respective SFT, geometry7367603 depends both.63 targeted tests passed. Agency quota interruption recovered at user request; no duplicate training/API calls.

## 2026-10-13 — user long-output concern resolved before test inference

Independent Agency review found8k generation insufficient:devlongestJtarget14,395tokens. Actualbasecontext40,960,RoPEunscaled. Capacity7367612passed8×40,960cachedtokens:73,998,618,112allocatedbytes,100,411,637,760reservedbytes,device102,005,473,280. Syntheticchunkeddevtokenstress,notqualityscore. Longcontextamendment retains98754matchedSFTrows (169lengthomissions disclosed,no targettruncation), allowsfullremaining40k contextusingcommonpairedinputwidth,batch8,22hperinference. All8647testretained. Oldinference7367601/2andgeometry7367603cancelledonlyafterPENDINGverification; replacements7367626/7367628/7367629submitted. Frozen eval-context-f1218f816e4b SHA d242c430abeadbfbde00f92501414e49ded0442fdf19f150071bce681ffe69ab. New41focusedtests includehash-boundsharedplansandforgedreceipt rejection. Currentrequestedbudget93.75GPUh,cap96.


## 2026-10-13 自动接续流程已实际启动

用户再次授权发现训练实质错误时修正重跑。独立Agency模型QA核验服务器冻结代码、共同98,754条UID、assistant-only监督、优化器与实时有限损失/梯度，未发现需立即停止或重训的中高严重缺陷；完整训练和最终重载/质量结果尚待完成。记录见 `evidence/agency-live-training-qa-20261013.md`。

完整比较代码672测试通过（18.04秒），正确按文件路径采集生产语句覆盖率90.8%。真实两资产、单个非test合成房间的 visual_pipeline→六张图→便携评分包→离线judge预检已实际通过，未产生模型质量指标。服务器CPU连通probe7367681完成（2秒，HTTPS401无认证），凭据留在服务器；之前两项认证视觉API烟测另存。

最终代码 `pipeline-9c4d97dbd736` 本地/服务器33文件逐项哈希一致。SFT7365681/7365682继续运行；推理7367626/7367628及几何7367629保留依赖。不再使用取消的旧推理作业。

实际后台控制器PID41825（PPID1，独立会话），driver42372，SSHmaster41826，caffeinate42373；配置 `automation-config.json`，输出 `automation-run-002`。已核验6个作业状态、实际SSH multiplex通道和独立父进程退出后的存活。根据排队预估将总等待窗口从72h改为120h，render24h/API10h及348正式请求上限不变；排队预估并非保证，5min CPUprobe实际上立即执行。控制器另设120h SIGALRM，结束时回收其子进程。

部署记录：run001仅作启动/轮询验证，已主动终止、保留partial回执，未发API或改动服务器作业。launchd shell因Desktop访问权限失败，Python后台尝试在读取配置前等待，均已停用；没有更改系统隐私权限。最终采用既有CLI权限上下文的分离进程。`ssh-multiplex-launchd-verification.json`文件名沿用早期方案，最终执行实际为分离CLI进程，并非launchd。进程脱离父进程已有实测；ChatGPT宿主退出、注销、重启或断网情况下的持续性不作保证。Mac保持可用是本地渲染的条件。

自动后续：等待全部既有作业完成→校验并下载→参考/J/P同相机真实资产渲染→共同成功子集FID→服务器Slurm CPU GPT评分→ `automation-run-002/final-results/report.md`、`paper-style.csv`、`results.json`。缺失和失败保留；真人UR未测；新正式分数当前仍为空。不能把legacy分数或工程烟测放入新J0/P0正式结果。

补充部署验证：分离CLI进程中的真实资产渲染再次通过（22.64秒，1房间、6图、2个离线API候选、0正式评分），证据 `evidence/detached-render-deployment-smoke.json`。控制器已自动完成第二轮120秒状态轮询；独立DevOps复核部署脚本与实际进程一致。120h触发清理，三个直接子进程的极端等待合计最多额外105秒；不把直接子进程退出等同于所有另起进程组后代已验证退出。


## 2026-10-13 07:49 UTC — 全量审计收尾与实时状态纠正

实际输入/来源暴露/身份组件/OOF先验/当前消费文件检查已完成，最终状态为COMPLETE_WITH_KNOWN_SOURCE_DEFECTS。56352组件及2330544先验轴值复算一致；已知源错误涉及11个实际训练房间、3个测试房间，另有Scan2CAD先验池暴露，传播影响未测。保留冻结J/P为诊断；纠正版本的性能主张必须先修源转换、重建完整数据/先验/teacher/准入并配对重训。此纠正重训尚未启动。7368003的FAILED保留：核心检查已完成，尾部历史辅助库存缺5文件；7368012补验当前消费文件通过。

三项局部修复经681测试、90.874%语句覆盖及独立复查。修正几何/表格代码pipeline-full-audit-a8726c726f72已部署；旧待运行几何7367629于07:37UTC取消，新7368009等待既有推理7367626/7367628。run002停止留存，新run003控制器40191/driver96318/SSH40239/caffeinate96319已核验存活，无API提交，原120h截止时间未重置。训练冻结代码与数据未热替换。

本轮直接SSH于07:49:46UTC核验两臂RUNNING03:18:56：J4881/24689，P4550/24689；日志显示剩余约12:18/13:24，仅为瞬时估计。当前日志尾部无Traceback/OOM；完整训练、fresh reload、正式几何/FID/GPT均待完成。记录见evidence/resume-live-status-20261013.json。纠正旧状态中的dev-loss表述：8151仅为长度准入，训练器不计算dev loss，只保留2间训练后重载生成探针。

主文档contribution-and-evaluation-audit.md更新贡献判断、完整指标合同、已知数据错误、具体修正次序、实时服务器与ARIS执行边界。此轮未新增训练、付费评分或改变计算/API额度；无限对话token不是无限模型上下文或无限GPU预算。

## 2026-10-13 16:27–18:01 UTC — 推理可行性复核、修订002/003、用户新授权

- 16:27 实时核验：J SFT 18098/24689、P 16665/24689，均健康；本地控制器 run004 于 15:11:04 UTC 因 SSH 主连接断开且 clifton 证书过期（4次重试均 publickey 拒绝）进入 stage_failed，服务器作业不受影响。
- 审查 workflow wf_e5054089-226（8 agent，对抗复核）：按冻结方案（UID 顺序 batch 8、HF greedy、~68 ms/步、无续跑）J 推理预计 24–29 h，超过 1315 分钟上限；P 16–19.5 h 临界；任一臂超时会使几何 7368009 被取消、三张表全空。表格设计缺配对差值表、GT 参考行、P 尺寸=先验标注、原始 OOR 标注、失败分类；3D-FRONT 客厅排除 166 间客餐厅。模型无关参考（CPU 复算两次一致）：GT OOR 全体 0.0787，六层 .0302/.0672/.0035/.0341/.0109/.0081；GT geometry_success 0.723（IL3D .494，holodeck 1.0）；P 先验尺寸 IoU .5582 / log MAE .2824。
- 17:43 `scontrol hold 7367626 7367628`（从未启动）。用户授权：自主 autoresearch、按需申请 GPU、Agency 审计、最终交付一个模型 + 同输入对比表（gpt-6.1-sol 与服务器 vLLM 开源模型）。
- 锁定修订：26a841a3c（已部署对比代码入库）、823cab3e9（修订002：共享长度排序计划、输入派生生成上限、LoRA 合并、分片+续跑+合并、LLM agent 臂、共同主指标、交付规则、GPU 上限200）、780bf7804（修订003：Qwen3-8B 两种模式、Qwen3-VL-32B、Qwen3-VL-235B FP8、条件性 gpt-oss-120b；GPU 上限260）。均在任何测试输出产生之前。
- 已推送 GitHub main（be521c581，干净检出 722 测试通过）。源修正（Scan2CAD/IL3D 开口/SpatialGen 倾斜）代码与 v3.3 数据在 wf_2c387d07-73e 中对原始数据独立核验，通过后才构建并上传 HF。
- 服务器：vLLM 冒烟 7382244 因 flashinfer 采样 JIT 编译失败，改 `VLLM_USE_FLASHINFER_SAMPLER=0` 重提 7382555；gpt-oss-120b 下载 7382259 因 `--exclude` 写法错误失败，重提 7382556。

## 2026-10-13 18:01–18:32 UTC — 部署 API agent、开源模型下载、源修正核验结论

- 源修正核验（wf_2c387d07-73e，6 agent）：SpatialGen 修正正确；Scan2CAD 数学正确但范围过大（13,509 个 tilt≤10° 直立物体尺寸被包围盒放大，高度中位 +3.7 cm，地面穿透比例上升），应只改 tilt>10° 的 640 个；IL3D candidate-001 由过期 adapter 生成（328 个窗 yaw 翻转 π），且拱窗回归、湾窗策略待定。HF 发布 NO-GO；候选 002 修正流程 wf_d06f14d0-a6b 运行中。v1_1 包 + merge_lora 原子保存单独提交推送（051726074）。
- Agency 审计（wf_39f38da6-e7d）：训练正确无需重训；数据无泄漏、无划分污染；指标计算正确。交付规则需尺寸偏差护栏 → 修订 005（8d650094e）。修订 004（926fbcdb9）按 2026-10 主流开源模型重选：Qwen3-8B×2、Qwen3.6-27B、gemma-4-31b-it、gpt-oss-120b、DeepSeek-V4-Flash-0731；Kimi K3 / GLM-5.2 超出单节点不运行。
- 推理 v2（fc41cc4a3）与 LLM agent（ad3b9ecdf）合并入 main，746 测试通过；快照 main-8d650094e 部署服务器（305 文件哈希一致）。
- API agent：冒烟 7382896 通过（2/2 首答合法，~20 s/次），全量 7382898（1,454 间，gpt-6.1-sol medium）运行中；cohort uids_sha256 4afccef7…。
- 模型：gpt-oss-120b 移入 $PROJECTDIR/fastfill/models 并 sha256 校验（61G）；7382840 已下载 Qwen3.6-27B、gemma-4-31b-it，DeepSeek-V4-Flash-0731 下载中；vLLM 0.31 支持全部五种架构。

## 2026-10-13 18:45–19:45 UTC — 推理 v2 链、开源模型链提交；修订 006

- 推理 v2 门禁（wbqi5ejp1）：提交脚本出错即停、dev 探针（合并 LoRA 后 2 个 dev 探针 schema 不得由合法变非法）；追加 plan 先 hold 后 release、分片自查 PROBE_PASS（934192e6d）。取消从未启动的 7367626/7367628/7368009；提交 plan 7383109 → probe J 7383110 / P 7383116 → 分片 J 7383111–14、P 7383117–20 → merge 7383115/7383121 → geometry-v2 7383122（快照 main-934192e6d）。
- 源修正 candidate-002：Scan2CAD 与 SpatialGen 正确；IL3D 宽度正确但 1 cm 垂直间隔规则改动窗户合并 → NO-GO；决定恢复历史合并、只修宽度，candidate-003 流程 wmblf45vp 运行中。
- 开源模型运行器、N 系统视觉、report_v2 合并；修订 006 锁定（O5 开思考；顺序 O5,O4,O2,O3,O1a,O1b；GPU 上限 400；评审 696 次、24 h）。集成修补与两处审查跟进（76323f9f1）：思考系统无推理则冒烟失败；无答案房间不再关闭逐行截断规则。
- 开源模型链（快照 main-76323f9f1，singleton 串行，排在全部推理 GPU 作业之后）：O5 7384533/7384535，O4 7384537/7384538，O2 7384553/7384554，O3 7384555/7384556，O1a 7384557/7384558，O1b 7384561/7384562，status 7384563。vLLM 环境 shapely 2.2.0 可用。

## 2026-10-13 19:45–20:30 UTC — 修订 007：最新主流开源模型

- 用户指出 Qwen3.6 已过时；HF API 逐家核查（wf_94a1b68c-adf）后锁定修订 007（c9d9fd51c）：O1a/O1b Qwen3-8B，O2 Qwen3.8-27B，O3 gemma-4-31B-it，O4 gpt-oss-120b，O5 DeepSeek-V4-Flash-0731（思考，effort high；vLLM deepseek_v4 tokenizer 会把 medium 降为 low），O6 Qwen3.8-Flash-Next-FP8（180B/≈6B 激活，TP4+EP，effort medium，模板确认读取 reasoning_effort），O7 Muse-Glimmer-30B（模板默认 high）。单节点放不下或无权限者列为未运行并写明原因。用户说明不必追求超大模型；多节点可用但本组不需要。
- 代码（0ed0fac8b, 48bbeb99a, dd390623f）：O7 依服务器 chat_template.jinja 改用 ATEM 通道切分（to=self 推理 / to=user 答案 / <|eot|> 结束），之前假定的 </think> 会把推理当答案。submit_llm_open.sh 在已有同名作业时拒绝提交。
- 旧开源模型链（7384533–7384563）取消、从未启动；新链（快照 main-dd390623f）：O6 7385472/3，O5 7385474/5，O4 7385476/7，O2 7385478/9，O3 7385480/1，O7 7385482/3，O1a 7385484/5，O1b 7385486/7，status 7385488。所有模型权重已在 $PROJECTDIR/fastfill/models 并有 SHA256SUMS。

## 2026-10-13 20:30–21:00 UTC — 源修正 candidate-003 通过、v3.3 构建、夜间接力链

- candidate-003（444b2a3f1）：IL3D 只改 63 个斜墙窗宽度（全部变宽），窗户集合与顺序与 v3.2 完全一致；Scan2CAD 640 个倾斜物体；SpatialGen 977 个；其余 15 个来源逐字节等于 v3.2。独立复核 correct，门禁 GO。v3.3 发布工具在 .release/v3.3（git-ignored）；输入上传并逐一校验；构建作业 7385668（先控制重建 v3.2 须逐字节一致，再建 v3.3 + 身份审计 + QA + 扫描；exit 0 才上传 HF 新文件夹）。README 写明未修限制（2D 轮廓开口偏窄、挑高房间窗户合并、Scan2CAD 10–15° 下沉）。
- API agent 完成：1,454/1,454 首答 schema 合法，0 API 失败。
- 已知缺陷敏感性集合冻结（d095394a7）：candidate-003 改动的 47 个测试 UID（IR 层，超集）。
- 夜间接力链（快照 main-d095394a7）：evaluate_llm L 7385871（afterok J/P 合并）；O6..O1b 评估 7385872–7385879（afterok 各自全量 + 合并）；J/P/L 报告与交付判定 7385880（afterok geometry-v2 + L 评估）。clifton 证书 2026-10-14 04:16:58 UTC 过期；之后视觉（本地渲染、FID、GPT 评审）与 HF 上传需用户重新 clifton auth。

## 2026-10-13 21:00–22:30 UTC — OptiScene 多阶段 DPO（修订 008/009）上线

- 用户澄清：训练方法应为论文的 SFT → DPO1 → DPO2（即 V1.1 的实现），交付最终模型 + 评测。修订 008（4a2918ea1）：两臂各自 DPO1（2,200 个 GT-pass 正样本：holodeck 1,100 + IL3D 1,100；负样本为自身 SFT 采样）与 DPO2（位置扰动造成越界/重叠）。审查发现论文正文 lr 5e-7 的更新会被 BF16 合并抹去 → 修订 009（e02c9352b）改用官方 README 超参（lr 5e-6、10 epoch、LoRA r32/α16、有效 batch 4，4 卡 DDP），阶段评测用未合并 LoRA 的精确策略。
- 代码：dpo_stages.py（2b7e6dc62 合并）、amendment 009 实现（7279bb718）、审查跟进（3cdfcc90e：跨臂 LoRA 模式校验、DPO 分片 14 h、合并诊断不阻塞）。
- J SFT 完成（16:22:09），P SFT 完成（17:44:51）；J dev 探针 PROBE_PASS（记录 2/2 合法，合并后 10/10 合法）。v3.3 控制重建与已发布 v3.2 逐字节一致，v3.3 构建进入 QA。
- 提交 DPO 全链 43 个作业（快照 main-3cdfcc90e，见 research-state dpo_chain）与 DPO 阶段报告/对照组评估链（dpo_reports）。用户 22:27 UTC 续期 clifton 证书，至 2026-10-14 10:27 UTC。

## 2026-10-13 22:30–23:00 UTC — FastFill v3.3 发布

- v3.3 构建 7386117 release_ready：v3.2 控制重建逐字节一致（aarch64，numpy 2.4.6 / shapely 2.2.0 / GEOS 3.14.1），9 项条件全部通过；相对 v3.2 改动 43 行 SFT（train 32 / test 11），无新增/删除，无划分迁移，dev/test 人群一致，默认训练集 124,843 → 124,842。
- 发布作业 7389340 → HF liantian/fastfill-v3 commit 6d951e3441eff6854d42cbe21653bd8f9ab5a500（父 d264159e），新增 v3.3/、code_v3.3/、ir_v3.3/ 并追加 README v3.3 章节；88 个文件远端校验通过（20 个按 LFS sha256）。首次发布作业 7389337 误带 --verify（仅核验模式），因无 PUBLISHED.json 失败、未改动远端，已取消。
- 注意：当前 J/P 模型训练于 v3.2；v3.3 只影响 43 行，测试集 11 行变化（主要 SpatialGen），已纳入冻结的 47 UID 敏感性集合。

## 2026-10-14 00:35 UTC — DPO 进展

- DPO 偏好对：stage1 2,200/2,200 两臂均通过验证（IL3D_3dfront 1,100 + OptiScene_holodeck 1,100，负样本均为自身 SFT 采样）；stage2 已构建。
- P DPO1 训练 7389124 完成（1:52:12）；合并诊断：投影层 BF16 权重改变比例 q 12.7% / k 10.9% / v 16.5% / o 18.1% / gate 19.3% / up 18.6% / down 15.3%，other 0%——修订 009 的超参使更新在合并中大部分保留。P DPO2 训练 7389125 与 P DPO1 测试分片 7389134–39 已开始。
- SFT 推理：J 完成并合并（infer-joint-002）；P 剩分片 0（1,280/2,167）。

## 2026-10-14 00:52–01:05 UTC — SFT 阶段主结果（协议主要终点）

- SFT 推理两臂完成（P 分片 0 用时 2:21:42），geometry-v2 7383122、L 评估 7385871、报告 7385880 完成；报告拷至 results/report-v2-JPL-001。
- 主要终点：(a) J 0.3882 vs P 0.3896，Δ −0.0014 [−0.0122, +0.0091]；(b) log MAE J 0.2861 vs P 0.2817，Δ +0.0044 [+0.0005, +0.0084]，Holm p 0.066。J 足迹偏差 −0.083，(a′) Δ −0.051。按修订 005 规则 SFT 阶段会交付 P（正式交付在 DPO2 阶段决定）。
- gpt-6.1-sol（队列 1,454）：几何成功 0.935 vs J 0.431 / P 0.510；尺寸 log MAE 0.300 vs 0.243 / 0.239。
- 综合见 findings.md。O6（Qwen3.8-Flash-Next）冒烟通过，全量运行中；J DPO1 完成，DPO2 训练与 DPO1 测试分片运行中。

## 2026-10-14 01:10–02:10 UTC — Agency SFT 审计、报表修正、O5 修复

- Agency 审计（wf_b7ae99d6-8df，3 审计 + 3 Reality Check）：评分路径无伪影；API agent 强烈缩小尺寸（B_L −0.175），去偏后 L 0.765 仍高于 J 0.371 / P 0.515，但仅略高于 GT；J/P 与 GT 的差距约 70% 来自可移动物体重叠，集中于多物体合成房间与大房间越界；J 的优势主要来自 SceneSmith；GT-pass 层 (b) 反号。findings 重写（adc3a35be），审计记录入库。
- report_v2 修正（7e433e779、281257f25）：房屋组件数与 CI 抑制/标记、所有系统的足迹偏差与 (a′)（EXPLORATORY，按系统自身完整掩码）、交付规则仅 DPO2 有约束力、警告渲染、阶段感知标题。旧 DPO 报告作业 7389197/7389198 取消；以同一快照 main-281257f25 重新生成 SFT 报告 7394712（report-v2-JPL-002）、DPO1 7394713、DPO2+L 7394714。
- O5（DeepSeek-V4-Flash）冒烟 7385474 失败：DeepGEMM JIT 断言 nvcc ≥ 12.9，集群模块为 12.6；改用 vLLM 环境内 pip CUDA 13.4 nvcc（79b049733），重提 O5 冒烟 7394628 / 全量 7394629（singleton 排队）与评估 7394641 / 7394642。O6 全量与评估完成；O4 全量运行中。

## 2026-10-14 02:10–02:30 UTC — scratch 文件数配额、开源模型重提

- O2/O3/O7/O1a/O1b 冒烟失败：scratch 项目文件数配额 1,023,915/1,024,000（容量仅 2.1T/5T）。普查：colmap-build 51 万、particulate-hinge 31 万等为用户其他项目；本研究的缓存约 5.7 万（uv-vlm、llm-open、dpo-generate），已清理（作业 7394871）。开源模型编译缓存改放 $PROJECTDIR/fastfill/.cache-jobs（6ffa6a9f7）。
- O5 重提 7394628 因旧失败目录的运行身份不一致被拒（infer.py 已变）；旧目录改名保留。用快照 main-6ffa6a9f7 重新提交 O5、O2、O3、O7、O1a、O1b（singleton 串行）及其 SFT / DPO2 配对评估。O4 全量与 SFT 评估完成。

## 2026-10-14 02:35 UTC — DPO2 训练完成

- DPO2 训练 J 7389123（1:51:24）、P 7389125（1:57:08）完成；训练后期 dpo_loss≈0、reward_acc 1.0、margin 13–15（硬负样本易分，需在评测中检查是否过度优化）。合并诊断（投影层 BF16 改变比例）：J dpo1 9–19%、dpo2 5–11%；P dpo1 11–19%、dpo2 5–10%；非 LoRA 层 0%。
- DPO2 测试分片 12 个已启动（infer-v2-004）；DPO1 分片完成约 35–47%。O2 全量运行中；O5 第三次（Marlin，关闭 DeepGEMM）排队中。

## 2026-10-14 04:10–04:40 UTC — DPO1（官方超参）测试结果：崩坏；修订 010

- DPO1 几何 7389141、报告 7394713（results/report-v2-dpo1-001）：J 几何成功 0.388→0.213、log MAE 0.286→0.590、足迹偏差 −0.08→−0.36；P 几何成功 0.390→0.093，完整输出仅 4,291/8,647（JSON 无效 2,532）。训练日志 reward margin 震荡并饱和（第 8 epoch 起 loss≈0）——DPO 过度优化/似然位移崩坏；仅保存最终步，无法回溯早停。
- 修订 010（3437401e1，查看 DPO1 测试后制定，已披露；DPO2 测试未查看）：两组正则化候选 e1（官方超参但每阶段 1 epoch）与 rpo（lr 5e-6、3 epoch、RPO α=1），以 600 间 dev 房间（从未用于训练/测试决策）选模：资格 = dev 完整率 ≥ SFT−0.01 且 log MAE ≤ SFT+0.01，取 dev (a′) 最高者，SFT 为兜底；交付规则改为在两臂各自选中的模型间应用。实现 workflow w0sm1g31g 进行中。
- 04:45 UTC：按修订 010 补充说明停止 009-DPO2 测试推理（12 分片及其 merge/geometry/report、各代理 DPO2 配对评估），未查看其输出；009 DPO1/DPO2 改在 600 间 dev 上评分、仅作报告。O2 全量与评估完成；O3 冒烟通过、全量运行中。
- 05:00 UTC：修订 010 实现合并（242d5b3a1：train_dpo --rpo_alpha、e1/rpo 候选、dev-select-v1、dev 精确策略评测、选择器自动提交测试链；审查修复 MAE 资格在同一批房间上比较、a009 阶段 dev 仅报告）。提交 dev 选模链（build-dev 7397532 … score 7397586、select 7397587）。
- 05:55 UTC：修订 010 补充 2（654c8930c，先锁定）：a009-DPO2 的 dev 推理生成到上限（J 分片 40 分钟 16/304，有效 0），选模不再等仅报告行。实现 score --only selectable/report-only，阶段表绑定单独的 a009 评分（缺失则显示未测）。最终报告工具（4e3f06453，报告 selected 阶段、阶段表、交付包、submit_final_010.sh）经 Agency 审查：SAME_NUMBERS 白名单核实无数值变化；阻断项即补充 2 已修。部署快照 main-791d0f03b；取消 score 7397586/select 7397587，重提 score 7397855 → select 7397856（自动提交测试链），score-report-only 7397857。
- 06:45 UTC：dev 打分（7397855）与选模（7397856）完成。J 选 SFT：J 的 DPO 候选都加剧缩尺（dev B：e1-dpo1 −0.127、rpo-dpo1 −0.211、rpo-dpo2 −0.139，SFT −0.087），a′ 低于 SFT；e1-dpo2 a′ 0.377 > SFT 0.357，但完整率 0.842 不合格。P 选 rpo-dpo1：dev a′ 0.435 vs SFT 0.382，原始成功率 0.470 vs 0.408，完整率与 log MAE 不变（P 的尺寸来自先验，DPO 只改位置）。e1-dpo2 P 完整率 0.428。选模自动提交 P-rpo-dpo1 测试推理（6 分片 7398937–42、合并 7398943、geometry-v5 7398944）。部署 main-3837da539，提交最终链：9 个代理与选定模型重新配对评测（7399405–14，O1b/O5 等各自生成作业），final 7399415（report-v2-selected-001、stage-table-010-001、delivery-010-001）。
- 07:15 UTC：a009-DPO2 dev 推理完成（J 分片 2h13m），score-report-only 7397857 完成。仅报告行（dev）：J a009-dpo1 完整率 0.882、成功率 0.182、log MAE 0.573、B −0.288；J a009-dpo2 完整率 0（600 间中 209 间到上限）；P a009-dpo1 完整率 0.477；P a009-dpo2 完整率 0.050。与修订 009 DPO1 测试结果一致（崩溃）。P-rpo-dpo1 测试 6 分片运行 30 分钟，每片约 70/1440。
- 07:50 UTC：O5（DeepSeek-V4-Flash）冒烟 7395110 第三次失败：hyper-connection 层的 DeepGEMM JIT 用系统 gcc 7.5（不支持 C++20），且 pip nvcc 13.4 与 torch cu130 的 cudart 13.0 头文件触发 CCCL 同版本检查。修复（74710f00c，仅 O5）：NVCC_PREPEND_FLAGS="-ccbin=/usr/bin/g++-13 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"，登录节点编译该内核头文件通过。失败目录改名保留；重提 O5 冒烟 7399959 → 全量 7399960，评测 SFT 配对 7399962、选定配对 7399963；final 7399415 依赖改为等待新的 O5 选定配对评测（取代已取消的 7399414）。O1b 全量与 SFT 配对评测完成。
- 08:05 UTC：(1) O5 冒烟 7399959 第四次失败：DeepGEMM 已过，但 TileLang 的 cutlass JIT 在 nvcc 13.4 下 cicc 段错误。O5 记为未运行（4 次引擎构建失败，原因逐条记录），不替换。(2) 07:53–07:55 六个测试分片所在节点全部 NODE_FAIL（not responding，集群事故，与 O5 无关），合并/几何/评测/final 被依赖连带取消。以原 argv + --resume 重提（dev-eval-010/submission-2.json）：分片 7400117–27、合并 7400129、geometry-v5 7400131；最终链重提（main-74710f00c，GEOMETRY_JOB=7400131，8 个代理）评测 7400136–43、final 7400144。视觉模板更新为新作业号。
- 09:05 UTC：代理 SFT 配对评测报告（llm-eval-*-v1/report.md）复制到 results/agents-sft-paired/。全队列（1454）几何成功率 raw/harness：L 0.935/0.935；O2 Qwen3.8-27B 0.808/0.928（241 间思考到上限）；O6 Qwen3.8-Flash-Next 0.700/0.701；O4 gpt-oss-120b 0.687/0.687；O7 Muse-Glimmer-30B 0.579/0.592；O3 gemma-4-31b 0.211/0.241（完整率 0.33/0.39）；O1b Qwen3-8B think 0.043/0.110；O1a Qwen3-8B no-think 0.000/0.000；J 0.431、P 0.510。核查 O1a 为 0：506 个完整输出中 505 个支撑失败（锚点/父对象语义不符契约，例如挂在墙上却落地），属模型行为而非评测缺陷；同一冻结几何对所有系统一致。代理尺寸 log MAE 0.30–0.61（J 0.243、P 0.239），去偏比较见 report_v2 T5。
- 10:00 UTC：用户续证（有效至 21:48 UTC），批准付费 GPT 评分的视觉运行。服务器无 Blender，且 FID 参考图由本机 Blender 渲染，故渲染/FID 仍在本机、GPT 评分为服务器作业。回答用户“为何开源数据过不了几何门”：测试集各来源 GT 通过率（参考行）ScanNet 0.135、3RScan 0.171、3D-FRONT 0.494、HSSD 0.529、InteriorGS 0.563、ARKit 0.576、SpatialLM 0.601、Structured3D 0.621、SceneSmith 0.780、SAGE 0.869、IL3D_synthetic 0.953、InternScenes_gen 0.962、MansionWorld 0.976、holodeck 1.000；真实扫描标注框噪声与 3D-FRONT 穿模为主，另有包围盒级检查放大（椅子入桌下等）；v3 数据“只打标记不过滤”，故训练含不合格标注。但 J/P 超额失败约 73–75% 来自 GT 几乎全合格的多物体合成来源。
- 11:00 UTC 更正：10:00 记录的“v3 数据只打标记不过滤，故训练含不合格标注”对本研究的 J/P 训练不成立。main_data.py（_qualification/_candidate，147–196 行）只收录 J 的 GT 与 P 的姿态教师目标都通过冻结检查器（oob、support、ceiling、fixed_blocking、main、severe 均为 0 且 valid）的训练房间，另按 EXCLUDE_FLAGS 排除。J/P 在全合格目标上训练仍在测试中失败，失败来自生成（同类叠放、多物体），不是学到坏标注。测试 GT 的 28% 不合格只影响 GT 参考线。另：3D 对照页（12 间哈希抽样，真实测试输出，逐件标记与存档计数一致）发布为 artifact JPu5uXFSqHjM9byWLFaYH1，数据在 /Volumes/harddisk/fastfillout/unknownsize-20261013/viz-3d-20261014。
- 10:50 UTC：P-rpo-dpo1 测试推理完成（分片 00 续跑 2h42m），geometry-v5 7400131 与 8 个代理选定配对评测 7400136–43 全部完成，final 7400144 运行中。视觉运行（用户批准付费 GPT 评分）已启动：visual_final 选定模板 → 配置 /Volumes/harddisk/fastfillout/unknownsize-20261013/visual-final-selected-001.config.json（8 个代理全部纳入，评审集 J、P、L、O6），本机冻结代码 code-snapshots/main-74710f00c（judge_batch.py 与服务器快照同哈希 133929fb…），overnight 驱动在本机后台运行（caffeinate），输出 visual-final-selected-001/。用户授权输出端完整研究计划（修订 011 起），工具构建工作流 w8g5ps999 进行中。
- 10:55 UTC：final 7400144 完成（FINAL_010_COMPLETE，8 个代理全部纳入，无排除）。交付规则（修订 005 §1，修订 010 绑定）：交付 **P = rpo-dpo1**（条件 (i) 否：Δ(b) J−P +0.0046 [+0.0007, +0.0087]；条件 (ii) 否：Δ(a′) −0.0604 [−0.0712, −0.0493]）。测试全集：(a) J-sft 0.3882、P-rpo-dpo1 0.3981（Δ −0.0098 [−0.0213, +0.0016]）；(a′) 0.3439 / 0.4043；B_J −0.083、B_P +0.015。队列（1454）代理：raw (a) 与自身 B 去偏 (a′)（EXPLORATORY）——L 0.935→0.766（B −0.175）；O2 harness 0.928→0.327（B −0.311）；O6 0.701→0.281（B −0.346）；O4 0.687→0.303；O7 0.592→0.219；O3 0.241→0.098；O1a/O1b ≈0；P 0.519→0.499；J 0.431→0.393。结果拷贝到 results/ 与移动硬盘。视觉运行 001 因 visual_pipeline 冻结选择校验顺序 bug 失败（列表顺序比较，参考文件按房型名排序）；修复 2d4e40873（集合比较 + 测试），以新快照重跑 visual-final-selected-002（渲染中）。
- 11:40 UTC：最终交付审计 wf_c30fbffe-068（4 个视角 + 汇总，audit-final-20261014/）：无改变交付的错误；阻断项为 FID 全系统交集会清空人群（O1a 在 holodeck 无完整房间）。修订 011（24aef2a87 + 澄清 a61d58b23，锁定于任何 FID/评审输出之前）：FID 主人群 = 参考∩J∩P，代理 = 参考∩J∩P∩代理；GPT 评审按臂与配对差；代理 a′ 加 CI、GT 对照、等尺寸对照、其他来源范围、预算/模式/O5 披露；Holm 只在 SFT 阶段；P-selected − P-SFT 配对对比；合并权重 dev 检查（|Δa′|≤0.01 且 |Δcomplete|≤0.005 才可作便利产物，否则只交付 adapter+基座）；三份阶段报告用同一快照重新生成。更正 06:45 记录：J 的 DPO 候选中 3/4 进一步缩小，e1-dpo2 不缩小（B −0.071）但完整率 0.842 不合格；rpo-dpo1 对 P-SFT 的测试增益 +0.0084 来自 MansionWorld（k=15），dev 增益未复现。findings.md 全面改写。修复与重生成工作流 wz4n06b0h 进行中。
- 12:45 UTC：修订 011 代码（wf_77e61102-fff：report_v2/stage_table/delivery/fid/visual_final/evaluate_llm/dev_select/final_011/submit_011，各自独立复核）与输出端工具（wf_c93484e6-922：downstream.py 端口 EmbodiedGen 路径 A/B 接受规则、output_repair.py S1、sample_vllm.py/best_of_n.py S2）合并为 4e3be1d75；final_tables 改读 FID populations；全套 1020 通过。下游规格要点：窗户无禁区；端点路径对所有物体 1 mm；admit A4 用 AABB、桌下椅子算碰撞；A3 可达性；后端发送的约束字段 id 与我方 interface 读取的 subject 不一致（导致带关系的房间 422，属 FastFill 侧可修）。部署 main-4e3be1d75，提交修订 011 链：合并权重 dev 分片 7403860/7403862 → 合并 7403864 → check-merged 7403866 → final_011 7403868。
- 13:05 UTC：修订 012 锁定（51e697f58，全部 dev 8,167 间：G、G+S1、B_m、B8+S1；主指标下游接受率；决策规则与测试一次性应用；GPU 上限 150）。提交 S2 采样链：build 7403879，J smoke/array/finalize/select 7403880–83，P 7403884–87。
- 13:55 UTC：修订 011 §5 合并权重 dev 检查（7403860/62 → 7403864 → 7403866）：BF16 合并模型在 600 个 dev 房间上 complete 0.998（Δ0）、原始成功率 0.433（未合并 0.470，Δ −0.037）、a′ 0.397（未合并 0.435，Δ −0.038，超出 0.01 阈值）、逐字节相同输出仅 20/600（3.3%）。决定：只交付 adapter + 训练基座（被评测策略），合并权重不列出。含义：BF16 合并抹掉了 rpo-dpo1 的大部分增益（回到 SFT 的 dev a′ 0.382 附近）；后续 DPO2 不能以 BF16 合并的 DPO1 为基座（需 FP32 合并或叠加 adapter），部署需按 adapter 方式加载。S2 冒烟 7403880/84 完成，J/P 各 4 片采样 7403881_[0-3]/7403885_[0-3] 运行中；final_011 7403868 运行中。
- 14:35 UTC：修订 013 工具（on-policy DPO2）与修订 014 工具（官方复现 + 交叉评测）合并 cf518a24b（1104 通过）。由于合并权重检查失败，修订 013 的 DPO2 改为“从 rpo-dpo1 adapter 继续训练、参考 = SFT 基座 + 原 adapter”，修复工作流 wf_56cb0fe6-fca 进行中；修订 013 在其完成后锁定。修订 014 锁定（文档 commit 后部署 main-f788c3c63）：E1 计划本机构建（硬盘 optiscene-014/plan-e1；E1-unseen 按论文分层上限选，3D-FRONT 卧室缺 207 间如实记录），上传 data/optiscene-014-e1 与官方三份数据（sha 前缀核对一致）；登录节点启动 setup_optiscene_env.sh（venv + Qwen2.5-7B-Instruct 下载 + env-check，日志 runs/isambard/setup-optiscene-014.log）。
- 14:55 UTC：修订 011 final_011 7403868 完成（FINAL_011_COMPLETE；SFT 与 selected 报告各含 8 个代理，O5 列入未运行），输出拉到硬盘 server-pulls-20261014/。修订 012 S2 采样全部完成（vLLM 约 20–30 分钟/片），best-of-N（dev 8,167 间，检查器成功率，vLLM 贪心为基线）：P-rpo-dpo1 m0 0.400 → m1 0.465 → m2 0.502 → m4 0.548 → m8 0.600（平均违规 18.7 → 2.1）；J-sft m0 0.388 → m8 0.435（4,297 间房间的候选被尺寸等守卫拦下）。保真：600 间 dev 上 vLLM 贪心对 HF 精确策略，P 0.432 vs 0.470（逐字相同 9%），J 0.402 vs 0.380（19.8%）——vLLM 的 BF16 LoRA 与 BF16 合并一样丢失 DPO1 的大部分增益。S1 作业：P 贪心/bo8（7405930/32）运行中；J 两个作业 shapely "The first input geometry is empty" 失败（7405926/28），诊断作业 7405983。OptiScene 环境：登录节点后台进程被会话清理杀掉（空日志）；计算节点可联网（HF/PyPI 200），改为 CPU 作业 7405987。
- 15:20 UTC：S1 崩溃定位：dev structured3d_00488_room_4256 为 2.5×1 m 房间，2.38 m 宽门的门禁带覆盖整块地面，严格阶段与 downstream.accept 对空地面调用 snap_inside 报错。修复 785ae6a74（空地面跳过严格阶段并标记 strict_skipped=empty_floor；accept 在 R1 不可表示时不做 snap，房间照常判拒收），加回归测试；全套 1105 通过。部署 main-785ae6a74，重提 4 个 S1 作业 7406340–43。OptiScene 环境作业 7405987 完成（trl 0.15.2、transformers 4.49.0、peft 0.14.0、torch 2.13.0+cu126；Qwen2.5-7B-Instruct 下载；env-check ok）。提交修订 014 链：plan-e2 7406350、sft 7406351 … score 7406363。
- 15:50 UTC：DPO2 改为从 rpo-dpo1 adapter 继续训练（ad93a2c51，wf_56cb0fe6-fca 复核无阻断项；train_dpo --init_adapter，默认路径逐位不变；全套 1380 通过）。修订 013 锁定（规则：dev 600 间对 rpo-dpo1：complete ≥ −0.005、a′ ≥ +0.01、下游接受率 ≥ +0.01、两半 a′ 不降；否则 P 仍为 rpo-dpo1）。部署 main-506b1e35c，提交产出率预检 yield 7406580（在修订 012 的 dev 采样上投影 3,000 房的样本对数，达标才开始采样）。
- 16:20 UTC：修订 012 S1 全部完成（dev 8,167 间；下游 strict_ok 原始→S1：J 贪心 0.173→0.253、J B8 0.186→0.256、P 贪心 0.199→0.292、P B8 0.250→0.312；检查器 P B8+S1 0.798；移动物体位移中位 0.11–0.23 m），决策作业 decide_012 7407059 排队中（含 GT 参考行）。修订 013 产出率预检通过（1,000 间 dev → 1,253 对/412 房；投影 3,000 房 3,759 对/1,236 房），提交链 recheck 7407088 … select 7407102。用户要求房型更多：测试集 346 种房型写法（OptiScene 仅四类）；房型大类映射 + 分房型表 + OptiScene 在未见房型上的 E3 工具工作流 wf_8f23c0b3-0f8 进行中，完成后锁定修订 015。
- 16:30 UTC：修订 012 §4 dev 决策（decide_012 7407059，decision-012/）：**S1 采纳**（G+S1−G 下游接受率 J +0.080 [0.074, 0.086]、P +0.093 [0.086, 0.100]；移动物体位移中位 J 0.21 m、P 0.225 m；检查器成功率 J +0.323、P +0.308）；**S2 不采纳**（(B8+S1)−(G+S1) 下游接受率 P +0.020 [0.015, 0.026] < 0.03；J +0.003）。GT 参考（同 8,167 间 dev 的标注布局）：下游接受率 0.293、检查器 0.722。P 贪心+S1 下游 0.292 ≈ GT，P B8+S1 0.312。下一步按修订 012 在测试集上一次性应用 S1：J/P 全部 8,647 间与所有代理的 1,454 间队列输出（同设置），GT 参考行。
- 15:38 UTC：用户贴来两份外部评审，主线定为“尺寸混淆可行性评测 → 控制尺寸的下游评测 → 保尺寸修复偏好学习”。修订 016 锁定（affcc1fea）：缩放-通过率曲线、开源模型固定尺寸重生成、DPO2 归因消融（rpo-dpo1 / RFT / OptiScene 式扰动 DPO2 / 本文 DPO2，步数对齐，dev 单次生成）、OptiScene+同一修复、分层报告、检查器效度。工具工作流 wf_fed13063-1d2 进行中。
- 15:50 UTC：修订 015 锁定（6a48ccd32，346 种房型写法 → 15 个大类，映射表 evidence/room-families-mapping-test-20261014.md；论文四类 5,800 间、其余 2,847 间）。部署 main-6a48ccd32：分房型报告 7408056；OptiScene E3（9 个非论文大类共 868 间）plan-e3 7408058、gen-sft/dpo1/dpo2 7408059–61（依赖修订 014 合并）、score-e3 7408062；8B 的 E3 生成待修订 014 的 8B dev 冒烟发布后补提。
- 16:17 UTC：修订 012 测试集一次性应用（工具 f45907d76，wf_1c0c53ca-727 复核时在本机镜像上完整跑了一遍：J/P 全部 8,647 间 + 8 个代理 × harness/raw 在 1,454 间队列上，S1 设置一致；本机正以 2,000 次重抽样重算汇总，输出 /Volumes/harddisk/fastfillout/unknownsize-20261013/test-apply-012/）。点估计（下游接受率，原始→S1）：全测试 J 0.172→0.258、P 0.200→0.288、GT 0.295；队列 J 0.234→0.372、P 0.354→0.472、GT 0.448、L 0.728→0.754（B −0.175）、O2 harness 0.723→0.752（B −0.311）、O6 0.536→0.633（B −0.346）、O4 0.479→0.597（B −0.296）、O7 0.410→0.500（B −0.323）、O3 0.159→0.223、O1b 0.069→0.171、O1a 0。代理在自选尺寸下的下游接受率高于 P，但都显著缩尺寸——正是修订 016（固定尺寸重生成与尺寸替换）要检验的混淆。
- 16:20 UTC：修订 012 测试集应用汇总完成（2,000 次自举，seed 42），结果入库 results/test-apply-012/（report.md、levels.csv、contrasts.csv、results.json、MANIFEST.json）。全测试 P+S1 − J+S1 = +0.030 [+0.020, +0.039]；队列上 P+S1 落后 L/O2/O6/O4，与 O7 接近，领先 O1a/O1b/O3；所有领先 P 的代理也都高于 GT（0.448），且足迹偏小 16–29 %——尺寸混淆，由修订 016 检验。服务器：我们的 GPU 作业全部 Pending（Resources，Priority=1，fair-share 0.18），013 采样 7407090、014 SFT 7406351、014 8B 生成 7406362 未开始；families 7408056 与 E3 plan 7408058 已完成。
- 16:25 UTC：修订 015 房型族表（作业 7408056）拉回并入库 results/families-015/（15 族；论文四族 5,800 间、非论文族 2,133 间、无房型 673 间）。原始输出检查器成功 P：论文族 0.434 / 非论文族 0.338（GT 0.697 / 0.856）；下游接受 P 0.211 / 0.172（GT 0.285 / 0.346）。最弱：public_commercial（P 检查器 0.090，GT 远高）、meeting（P 0.160）、living（P 0.184 vs GT 0.475）。非论文族上与 GT 的差距更大——测试必须覆盖更多房型（附录 A.7 的措辞规则同样适用）。
- 16:58 UTC：修订 016 工具提交 ab2548d0d（工作流 wf_fed13063-1d2：构建→对抗审查→修复；1481 测试通过）。修复阶段已实现附录 A3/A4（尺寸替换、逐条规则表，scale_curve）与 B1/B2/A1/A2（试点 300 间、强制尺寸、L 与 O2-GT 试点、Kendall τ 比较，agents_016）。附录 C 锁定 8a832aa95（试点分配、尺寸变化下的支撑、xyz 为主替换、B2 不对称、API 上限出处更正为修订 002 的 4,800 次、来源标签、种子），锁定前未看过任何 016 系统级数字（披露见附录 C）。
- 提交（快照 main-8a832aa95）：§2 尺寸固定链 7409061–7409106——试点 O2 7409061、O6 7409065、O4 7409069、O7 7409073、L（API，≤1,200 次）7409077，各接 eval/S1；全量 O2 7409081、O6 7409085、O4 7409089、O7 7409093（等全部试点结束）；O2 GT 尺寸试点 7409103。GPU 上限 2010 GPU 分钟。
- 本机运行 §1/§6（scale_curve，快照 /Volumes/harddisk/fastfillout/unknownsize-20261013/code-snapshots/main-8a832aa95，6 进程，输出 scale-curve-016）。第一次因本机没有 infer-joint-002/infer-placement-005 的推理记录（截断规则输入）失败；改为不传这两个参数重跑——report-v2-selected-001 T6 中 J/P 被截断规则翻转的完整房间均为 0，结果等价。
- 工作流 wz2dlyjv4（wf_3fa8a7f6-960）：附录 A5/C8 种子 43（RFT、op-dpo2）、B 与物体数分层、P 臂标签防护。
