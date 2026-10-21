# V1.1 clean DPO research log

Append-only decisions. The preexisting model/test diagnostics below precede this protocol; they are not confirmatory new-run outcomes. Exact launch and protocol-lock timestamps come from the coordinator's Git/deployment receipts.

| Entry | Date | Type | Evidence and decision |
|---|---|---|---|
| 1 | 2026-10-11 | bootstrap | User requested careful inspection of the already prepared test sets, training correctness and unattended automation, then explicitly invoked autoresearch. Scope set to the current V1.1 SFT/DPO chain on Isambard; preserve prior runs. |
| 2 | 2026-10-11 | data audit | Fresh SHA checks match all7 release JSONLs and18 IRs;160964 UID joins complete. UID/group/alias split intersections are zero. Full room counts are8167/8035/8647/8533. Old200 prefixes cover only HSSD/IL3D and omit all409/408 source-heldout rooms. Evidence: local `data-audit.json`. |
| 3 | 2026-10-11 | protocol reconciliation | V2 frozen300+300 still exist but234/239 room inventories differ from same-UID V1.1 views. Treat current full V1.1 evaluation and V2 Stage0 as different tasks. Warm-start provenance cannot independently certify prior heldout exclusion. |
| 4 | 2026-10-11 | exploratory diagnosis | Initial actual-pair audit found144 constraint-invalid Stage1 positives. A further size/negative audit was required; the initial144 count was not claimed as the total invalid-pair count. Old training continues to be preserved as diagnostic evidence. |
| 5 | 2026-10-11 | exploratory diagnosis | Completed train-only sanitizer retained7244/8000 Stage1 pairs;756 unique rows dropped, with overlapping reasons144 constraint failure/475 corrected size failure/215 rejected without verified defect. All12000 Stage2 pairs retained before tokenizer filtering. Exact hashes and policy recorded in `sanitized/audit.json`. |
| 6 | 2026-10-11 | literature | Verified primary arXiv records for DPO2305.18290, LayoutGPT2305.15393 and InstructScene2402.04717. Saved bounded summaries in `literature/survey.md`; no novelty claim or reproduction of those systems. |
| 7 | 2026-10-11 | preregistration decision | Fix SFT/cleanDPO1/cleanDPO2 × four complete cohorts. H1/H2 use per-dev-cohort paired acceptance differences,2000 source-group cluster bootstrap draws/seed42. CI labels only; coverage/size deltas descriptive, no new0.01/0.02 guard thresholds. Test descriptive only. No causal cleaning claim without an unclean matched control. |
| 8 | 2026-10-11 | continuity decision | `/loop` and agent-bound cron tools are unavailable. Use actual bounded Slurm training/evaluation dependencies plus automated synthesis. Cap new work at two6h4GPU training allocations and two6h1GPU allocations per each of three evaluation chains. Preserve partial state if cap is reached; no claim of an indefinitely running chat agent. |
| 9 | 2026-10-11 | protocol prepared | Five research files drafted in existing docs/research structure. Coordinator must freeze tested code, commit protocol separately before clean runs, submit and verify jobs, and fill receipt/job references. No new clean-run outcome was available to this document author. |
| 10 | 2026-10-11 16:59 UTC | deployment | Coordinator locked protocol at `24342801681e62adc001dd647468e796ee25c5af`, deployed source `3e3eef72c51546cc0a22134e4409853570b988c4e5932823182d42007f6197c4`, verified all 224 source-file hashes, pair hashes and preserved SFT link, and completed preparation. Regression validation: 256 tests and 90 subtests passed. Exact commands and submission timestamps are archived in [launch.json](launch.json). |
| 11 | 2026-10-11 16:59:30 UTC | startup observation | Submitted training chain 7252874 → 7252875; SFT evaluation 7252876 → 7252877; DPO1 evaluation 7252878 → 7252879; final evaluation 7252880 → 7252881; CPU/zero-GPU report 7252889 depends afterany on the three last evaluation jobs. Dependencies are afterany with checkpoint/identity checks that fail closed. At this observation SFT evaluation was RUNNING on nid010057; clean training was PENDING Resources and other jobs PENDING Dependency. No clean training start or improved model quality is inferred. |
| 12 | 2026-10-11 | deployed and startup verification | State now records the real launch receipt, immutable code root, isolated run root and scheduled `evaluation_full/research-summary.md` path. The 84 GPU-hour cap remains in force; real startup/smoke verification is still pending. No protocol or source change was made in this documentation update. |
# GPU startup verification — 2026-10-11 17:04:56 UTC

Live GPU evaluation produced and scored its first 256 requests. Independently verified ordered UIDs, manifest identity and the generation-to-score digest from persisted artifacts. SFT ordinary-dev progress reached 1,024/8,167 rows; no complete full-cohort result yet. Clean training remained PENDING Resources at the 17:04:21 UTC scheduler check. Saved `startup-smoke.json` and `live-startup.json` alongside the research record.

Disabled implicit Slurm requeue for all nine new jobs using recorded `scontrol update JobId=... Requeue=0` commands. Only the explicitly submitted continuation may retry each GPU chain. No old job was changed. The final CPU report remains scheduled after all three evaluation chains and returns incomplete if any authoritative cell is missing.

# Full source re-audit and external validation gates — 2026-10-11 19:16 UTC

Fresh source/data inspection found target-derived ordering across train/dev/test and their constrained variants. The interpretation is narrowed to diagnostic-only under that contract in a separate SHA-bound addendum; the original preregistered scoring/decoding/model recipe and hypotheses are not retrospectively changed. No DPO evaluation outcome drove this decision.

Six parallel audit domains and independent cross-reviews confirmed local fixes for corrupted/unbound caches, unsafe checkpoint/publication recovery, unproven training completion, nonfinite gradients, symlink output isolation, stale successful reports and misleading legacy fallback/assets. Stage2 overwrite discovered by cross-review was also fixed with create-only atomic publication. Integrated regression:428 passed and90subtests; additional deployment transaction simulation10passed. Source read coverage is recorded by exact current hashes.

Deployed immutable helper source a560b279eef244d7e0dc95110d32c542f0f50c5cdbf65fe8359b7deca3568473(236files). Old source3e3eef72c515(224files) still matches. New jobs were first held and checked; all5 superseded jobs were then held/confirmed never-running/cancelled, and only then were replacements released. Old training7252874→7252875 was not modified. DPO1 eval7255085→7255086 and final7255087→7255088 retain one continuation and the84GPUh cap. CPU SFT full-text rescore7255089 and DPO1 lineage7255090 started on compute nodes; report7255091 waits for evaluations and audits. All new jobs have requeue disabled. CPU work is bounded at12.5 cumulative job wall-hours; actual allocation and billing differ from requested core counts. See audit-evidence/audit-launch.json and post-deployment.json.

The future corrected request-only ordering contract requires independently rebuilt data and training; it was not implemented by silently changing this running experiment. Full DPO results and the final inference report remain pending.

# External validation completed — 2026-10-11 19:23 UTC

CPU job7255089 completed0 in5m31s: independently rescored all33382 SFT raw generated outputs across four full cohorts; every scoring field and all metrics matched, final identities rechecked, verification/sft.json statuscomplete at19:21:42UTC. CPU DPO1 lineage7255090 completed0 in62s, proving7244admitted/906steps and final-checkpoint+merge lineage. Repeated successfully under the exact final-report interpreter using an overlap step inside the same existing CPU allocation. No newGPU or additionalbatchjob was created for that environment check.

Final read-only snapshot19:23:03UTC shows DPO2 step340/1496 with no source drift; original224/new236 hashes all match. EightDPOcells+report remain pending. Audit and deployment are complete; current research remains diagnostic-only under GT-derived ordering, with no DPO superiority conclusion.

### 已完成DPO1提前并行评测 — 2026-10-11 19:26:43 UTC

独立审查原冻结训练脚本后，确认DPO2正常路径只读DPO1基座；训练续跑会验收并跳过已完成DPO1。基于已经通过的完成/merge证明，首个DPO1评测7255085在hold并确认未运行后解除对整个训练链的等待，再release调度，允许与DPO2训练并行。评测入口仍重新验证lineage，续跑7255086、final和report依赖保持；解码/模型/数据/统计协议不变，不将评测结果反馈给当前训练。84GPUh上限不变，峰值申请可为训练4GPU加评测1GPU。操作回执：`audit-evidence/dpo1-early-start.json`。

# Completion, paper alignment and Agency acceptance — 2026-10-13

Read-only server verification at 2026-10-12T23:39:03.827031+00:00 found training/evaluation/report jobs completed exit 0. Verified 12/12 cells, 100,146 request records, 36 evaluation artifact hashes and local/remote report identity. Preserved frozen predictions and scoring; archived server-status-20261013.json and paper-oor-strata-20261013.json.

Compared the supplied OptiScene v2 PDF tables and protocol with actual training, checkpoint provenance and published reference code. Five installed Agency roles performed technical and bounded evidence reviews. Their substantive qualifications are integrated in paper-alignment-20261013.md, with three cross-review notes archived under audit-evidence/. Completed diagnostic analysis is distinct from fair paper benchmarking and downstream quality acceptance. Historical pending observations above are superseded; no new training was launched.
