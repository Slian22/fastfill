# Agency Research Synthesist — bounded paper-alignment cross-review

Date: 2026-10-13. Workspace: `/Users/slian/Desktop/3D/Worldedge/OptiScene`.

**Verdict:** the report's central conclusion is supported. The recorded FastFill V1.1 run is **completed** within its frozen diagnostic protocol; paper alignment/report readiness is **partial** pending the narrow qualifications below; a fair paper benchmark is **not done**; faithful reproduction is **not done** (documented task/training differences, not proof that training failed). Ready as an internal gap assessment after minor wording revisions, not as evidence of matching or beating OptiScene.

Scope: reviewed the complete `docs/research/20261011-v11-clean-dpo/paper-alignment-20261013.md` against the supplied paper's extracted text `tmp/pdfs/optiscene-paper/paper.txt`, specifically §§3/4, appendices B/C/D, Tables 1/3/6/9 and relevant prompt tables. Spot-checked saved OOR aggregates and research-summary status. No internet/literature search, new PDF rendering, source inventory, training re-audit, jobs, paid tools, or repository modifications. Line references below refer to the reviewed report snapshot. Parent owns final report.

## Evidence weight and independence

| Source | Evidence weight | Limits |
|---|---|---|
| Supplied arXiv v2 paper, 2025-09-19, through its supplied extraction | Primary author report; high for what the paper states, limited for independently verified execution | Peer-review status not established here; incomplete benchmark implementation details; no independent replication |
| `audit-evidence/paper-oor-strata-20261013.json` | Primary local result aggregation; high for transcription checking | Derived from the same saved predictions, not another experiment |
| `evaluation_full/research-summary.json` | Primary local audit/status artifact; supports completed 12/12 and `diagnostic_only` | Does not establish external benchmark validity |
| `audit-evidence/optiscene-training-alignment-20261013.json` | Prior local technical audit with evidence pointers; accepted within assigned scope | Training/code provenance was deliberately not independently rescanned |
| Alignment report | Secondary synthesis under review | Repetition of these sources adds no independent experimental evidence |

## Corrections recommended

1. **Qualify paper DPO epoch allocation (report line 30; PDF §4.1, p7; extraction lines 460–466).** The paper says its two-stage DPO uses lr=5e-7 and trains for five epochs; it does not explicitly say five epochs **per stage**. This is an unsupported precision, not proof the interpretation is wrong. Replace the paper cell with: `lr=5e-7，论文写5 epochs、cosine；未明确5 epochs如何分配到两个DPO阶段。` Add stage allocation to reproducibility clarifications.

2. **Separate the explicit UR definition from Table 3's less explicit protocol (line 47; PDF §§4.1/4.2, pp7–8; extraction lines 469–475 and 540–547).** §4.1 explicitly specifies 50 scenes, five annotators and a ≥3 vote threshold for physical/semantic validity. Table 3's accompanying text instead says 50 layouts per room type and validity based on collisions/interpenetrations/major structural violations; it does not explicitly restate the five-person procedure. Their relationship is plausible, not completely specified. Suggested replacement: `UR：§4.1规定50场景、5位评审、至少3票认可物理与语义有效；表3另称每房型50次生成的有效率，未单独交代是否完全沿用同一评审流程。两者均不能由当前自动验收率替代。`

3. **Broaden the denominator caveat's possible explanations (line 100; Table 3).** The 2% increment observation is correct for exact binary outcomes from n=50. However, repeated averaging or a different denominator are not the only possible explanations: coarse rounding or a reporting error are also possible. Replace the restrictive explanation with: `这些数值不能直接还原为每类50次、未经取整的单次二元成功率；可能涉及粗粒度取整、重复平均、不同分母或报告误差，论文未交代，需原始计数与聚合规则澄清。` Table 6's 33% UR also lacks a recoverable denominator; do not infer misconduct or demonstrated test leakage.

4. **Make the comparison arm explicit (lines 104–107).** The fair-benchmark versus faithful-reproduction distinction is correct, but a new holdout requires new comparator outputs too. Add: `在同一冻结合同上重新运行可验证的OptiScene/选定基线并保存逐场景输出；新留出集不能沿用论文表1/3/6数字作为对照。若原作者权重不可验证，应标注为方法重实现或其他基线，不能称原论文模型。` Current checkpoint reuse can test a documented information-asymmetric system comparison; it does not establish matched-input method performance unless the asset-size/input contract is actually controlled.

5. **Narrow completion wording (lines 7/32).** Replace the opening with: `本轮冻结的训练与12个诊断评测单元已完成；论文共同基准及完整指标评测尚未完成。` This prevents “all evaluation complete” from being read as completion of the missing paper benchmark. Evidence: research-summary JSON `complete=true`, `completed_evaluations=12`, `expected_evaluations=12`, **`research_status=diagnostic_only`**.

## Checks that passed

- §§3.1/3.3 and Appendix B support retrieved asset bounding-box dimensions as input, position/orientation generation, and semantic-summary plus structured-output SFT. The report correctly distinguishes placement-only joint-size prediction.
- §3.4 supports approximately 2,200 unanimously approved positives and SFT negatives. Appendix C specifies 1,100 Holodeck-Synth plus 1,100 3D-Front and two noise families; adding those details would improve completeness but is not required to preserve the conclusion.
- Table 1's six final OOR values and §4.1's 423/53/50/50/50/50 test counts are correctly transcribed. Table 6's 0.049/0.048/0.033/0.023 values are correct. Local stage means, completion counts and six final strata match the saved OOR JSON to displayed precision.
- Appendix D.2 supports the pairwise intersection-area/total-area formula. The report correctly rejects collision-rate/IoU equivalence and direct cross-benchmark ranking. Formula examples are mathematically consistent; exact author implementation equivalence remains unproved.
- Appendix D.1 supports Isaac Sim rendering, and Table 9 supports render-and-preference-based functionality/layout/aesthetics judging. Table 9 additionally specifies integer 0–10 component scores; an implementation should preserve that, record the actual user-preference payload, and keep its optional overall score distinct from the three Table 2 metrics.
- Existing qualifiers correctly avoid treating missing upstream provenance as demonstrated contamination, treating source-label similarity as identical test IDs, or inferring furniture shrinkage from reduced dimension IoU.

## Experimental work still required

1. Freeze test identities, target-independent inputs, asset dimensions/task definition, source/room strata and failure/aggregation rules; document exposure checks across training, preferences and inherited checkpoint history. Unknown historical exposure remains unknown unless resolved or mitigated with a demonstrably new holdout.
2. Generate fresh FastFill and comparator outputs under that contract; report direct-model and repaired-system results separately. Existing numbers remain diagnostic. Include completion rates and paired common-case analyses/uncertainty when claiming stage or method improvement; stage-specific complete-output cohorts are not identical.
3. Run the missing common rendering/FID pipeline, fixed Table 9 evaluator, and independent five-person majority-vote UR protocol. Preserve identities, raw scores/votes and uncertainty. These are absent evidence, not proven low scores.
4. Strict reproduction additionally requires a verifiable Qwen starting point, paper data/reasoning supervision and stage-specific recipe clarification/retraining. Different hyperparameters alone do not invalidate a fair method comparison. Editing/navigation experiments are required only for those downstream claims, not for static-layout benchmark readiness.

Confidence: high in the textual corrections and transcription checks; moderate in overall experimental status because execution/provenance conclusions rely on prior audits by design. No new evidence here overturns the report's central conclusion.
