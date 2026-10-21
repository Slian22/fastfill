# TestingRealityChecker — delivery acceptance cross-review (2026-10-13)

**Verdict: training completed; frozen diagnostic evaluation completed; paper benchmarking and overall research acceptance PARTIAL / NEEDS WORK.** “训练好了吗” is yes for execution, not an unconditional quality claim. “指标能否对标论文” is not yet on comparable conditions. “我们弄好了吗” is only true for the declared diagnostic run.

Read-only bounded review of archived evidence, not a new server probe, rescoring, reference-repository scan, statistical review, visual inspection, or checkpoint reload. Only this /tmp review was written. No deployment/security/web-app gates apply to this assignment.

Evidence paths below are relative to `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261011-v11-clean-dpo/`:

- **S:** `audit-evidence/server-status-20261013.json` (2026-10-12 23:39 UTC / Oct 13 London).
- **R:** `evaluation_full/research-summary.json` and `.md`.
- **T:** `audit-evidence/optiscene-training-alignment-20261013.json`.
- **P:** `paper-alignment-20261013.md`.
- **G:** `roomgenbench_clean_final/summary.json`, per-room `generation_receipt.json`, `assembly_receipt.json`, `reference_check.json` and actual local GLBs.

| Acceptance item | Status / gate | Evidence and exact limit |
|---|---|---|
| Training execution | **COMPLETED / PASS** | T.training: SFT 3,221 steps, epoch 1.0; DPO1 906 steps / 7,244 admitted pairs; DPO2 1,496 / 11,968. R.audit.checks lineage/dpo1 and lineage/final record complete-checkpoint matching, not just job submission. S lists training jobs 7252874/7252875 and evaluation/report jobs COMPLETED exit 0. This supports completed training, not paper-quality superiority. |
| Artifact/eval integrity | **COMPLETED for archived diagnostic run / PASS** | All 12 R.evaluations entries are complete; independently summed expected requests = 100,146. Three output-verification receipts have status complete; lineage checkpoint hashes and merged weight identities are archived. Fresh local SHA256 of R equals S's receipt: `3994d56d6d90bf4b173281c3e8f6e395f2a843c30664fef9be1e9a9da3a59656`. S records rechecks of 36 evaluation artifacts, UID/index counts and local/remote report equality. Current merged weights were size-checked, **not freshly rehashed/reloaded**; absence of that new check is not corruption evidence. |
| Heldout protocol | **PARTIAL / paper gate NOT MET** | Current group/source-heldout diagnostics exist, including 409 ordinary / 408 constrained heldout rooms. P:85 and T.warm_start delimit exclusion to current training; original 676 IDs, semantic identity matching across datasets and upstream checkpoint training exposure remain unverified. Unknown upstream exposure is an evidence gap, not proven test leakage. |
| Input contract | **NOT DONE for target-independent benchmark / FAIL** | T.contract and R.interpretation confirm target-derived dimension/pose ordering and IDs. This is an established contract limitation, not merely missing documentation. Movable size is predicted and repair contributes to output; paper task uses known asset sizes. Existing results remain diagnostic_only. New independently ordered input contract/cohorts are required before a fair benchmark. |
| Human preference claim | **NOT DONE / NOT ESTABLISHED** | T.pairs establishes programmatic geometry/constraint selection and GT positives; no evidence of a new human preference collection or human UR study. P:26,47 correctly disclose this. “DPO” does not certify human aesthetic alignment. This does not prove the model lacks any inherited human preference. |
| Original metrics coverage | **PARTIAL / full paper gate NOT MET** | Saved raw/repaired OOR is formula-aligned; P:40–54 explicitly separates its definition from acceptance and size-only IoU. FID, GPT-4o visual quality, human UR, editing and navigation results are absent for these evaluations. Missing measurements are untested, not failed scores. Editing/navigation are additional gates only if those paper claims are sought. |
| RoomGenBench downstream | **PARTIAL / end-to-end acceptance NOT MET** | G records model generation without fallback in 5/5; actual local GLBs exist with the reported sizes. Mesh fitting succeeds in 4/5; bedroom has 2 fit_mismatch objects. Repaired geometry valid in 3/5: bathroom has 5 support failures, bedroom 3 plus severe collision. Livingroom is valid=true despite 3 main / 1 severe collision, proving that this flag is not zero-collision acceptance. All five reference_check.fastfill.ok are false; GT checks are also false, with unknown constraints, so do not attribute every violation to the model. These are five examples, not a RoomGenBench benchmark completion certificate. |
| Paper-comparison claim | **NOT DONE / comparison gate NOT MET** | P:7–9 correctly rejects strict reproduction or wins against paper tables; P:83–107 explains changed cohorts, inputs, warm-start provenance and postprocessing. No justified “matches/exceeds paper” conclusion follows from completed jobs or OOR formula agreement. |

## Completion evidence behind the flags

R.audit.checks `lineage/dpo1.completion`: epoch index 1, iteration 1810, step 906; checkpoint SHA256 `b4745bfac31d8cf7c49ea1b370096608c051a11a094e1ec1c2a565410c456709`.

`lineage/final.completion`: epoch index 1, iteration 2991, step 1496; checkpoint SHA256 `c23305efe204e9859cf36c603ae98e6b9dcd01ba1ca4f8157c03484ea39f4c2b`. Archived final merged weight SHA256: `e14ad78b6a2ca8b8f90f10081ebdf3d5c78655ea5e81c89efead6e2f89c22c51`.

Both use `legacy_final_matches_complete_checkpoint`. SFT parent in this older lineage receipt says `legacy_sft_model_identity_only`; the newer T.training.sft separately supplies terminal trainer-state/epoch evidence. Do not claim the older receipt alone certified SFT training completion. Likewise T.source_hashes says later resume/completion safeguards were not executed in this frozen run.

## Handoff discrepancies the parent should address

1. **Proven stale operational status:** `research-state.yaml:58` still says DPO2 RUNNING 390/1496; `:196` says 4 completed cells; `:238` says report scheduled_not_yet_available; `:265` says DPO2/eight cells/final report pending. `:222` advises continuing Slurm work. `findings.md:9,31,55` also retain pending/no-results wording. S and R conclusively supersede these. Keep historical snapshots dated, but add a current-status pointer or correction so a new operator does not rerun completed work. Do not retrospectively edit frozen receipts.
2. **Main report is appropriately cautious, but completion wording is too broad:** P:32 “当前所有训练、评测和独立复核已经结束” should explicitly mean **本轮预定义12单元诊断评测**. P:7 “当前实验已经完成” is defensible in context; neither should imply all original-paper evaluation is finished.
3. **Material downstream omission:** P mentions the five RoomGenBench examples only in the FID row. It does not expose the bedroom fitting failure, bathroom/bedroom geometry failures, or reference-check failures. Add a compact downstream acceptance row so “all done” is not inferred from exported GLBs. Report exact checker scope rather than claiming visually verified mesh failure.
4. **No acceptance threshold/winner was frozen:** R explicitly treats coverage/size warnings descriptively, reports one constrained-dev stage regression and disallows an automatic winner. Thus “complete” must remain an execution/integrity state; it is not a validated quality threshold.

## Smallest dependency order

1. Parent clarifies the current status and superseded files, with the eight gates above. This needs no new training.
2. Freeze the intended comparison claim, test IDs and overlap policy, target-independent input contract, known-vs-predicted size handling, raw-vs-repaired scoring and failure denominators. Unknown warm-start history must stay disclosed; use a verifiably new cohort if necessary.
3. Validate only the assets/rendering and metric implementations needed for that agreed cohort; resolve the recorded fit/support/checker discrepancies before downstream success claims. Freeze OOR/FID/GPT/human-UR protocols and matched baselines.
4. Generate on those common conditions and evaluate the missing metrics. Reuse existing checkpoints for a disclosed method comparison where appropriate; strict training reproduction separately requires verified starting weights, reasoning supervision and the paper training protocol. New training is not the first dependency.
5. Certify only the claims supported by those results; add editing/navigation if they remain in scope.

Suggested parent answer: **“训练跑完了，12/12诊断评测和结果完整性核验也完成了；但现有结果还不能直接对标论文，RoomGenBench下游验收也只有部分通过。因此本轮训练任务完成，论文对标和整体质量验收尚未完成。”**
