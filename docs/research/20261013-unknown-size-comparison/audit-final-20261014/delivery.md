No BLOCKER found. The delivery decision (P) is correct and holds at every stage. There are 5 MAJOR defects, all in claims and their hedging, and 13 MINOR ones.

**Checks that passed (recomputed in python3 from the JSON files)**
- **Delivery rule** (`report-v2-selected-001/results.json` delivery block):
  - Condition (i) fails. Δ(b) J−P is +0.00464 [+0.00070, +0.00869]; the estimate is above zero, so J is not more size-accurate.
  - Condition (ii) fails. The lower bound of Δ(a′) J−P is −0.0712, below the −0.02 margin.
  - So the rule delivers P. It also gives P at the SFT stage (i no, ii no) and at DPO1 (i no, ii yes). The switch to the amendment 010 comparison does not change who is delivered.
- **Holm:** SFT stage gives (a) 0.793, (b) 0.066; selected stage gives 0.095, 0.042. Both match the files.
- **Dev selection** (`dev-selection-010/selection.json`), redone by hand:
  - J: eligible = {sft, e1-dpo1}. e1-dpo2 fails completeness (0.842), rpo-dpo1 fails the size rule (0.2664 > 0.2597), rpo-dpo2 fails completeness (0.985 < 0.99). SFT wins.
  - P: eligible = {sft, e1-dpo1, rpo-dpo1}. rpo-dpo1 has dev a′ 0.435, 0.043 above the next best, so it is selected.
  - `dev_select.choose` (dev_select.py:249-265) implements amendment 010 correctly.
- **Numbers in findings.md that match the result files:** line 9 (JPL-001/002); line 22 (4,291 complete, 2,532 invalid JSON, J 0.213/0.590/−0.356, P 0.093); line 29; line 30 except the items below; lines 40 and 43 (these match JPL-002, i.e. P-SFT).

**MAJOR**

1. **The dev→test consistency claim for P rpo-dpo1 is not supported** (findings.md:29, "consistent with the dev gain direction but smaller").
   - The exact test gain is +0.0084 (+73 net rooms of 8,647), not "≈ +0.009".
   - Splitting it by source (selected T3 minus JPL-001 T3):
     - MansionWorld contributes +82 rooms, from 1,313 rooms in only 15 house clusters (flagged "few clusters").
     - Without MansionWorld the gain is −0.0012.
     - On the two DPO-training sources, which make up half of dev, it is −0.0007: holodeck +0.062, 3D-FRONT −0.037.
     - Reweighted to the dev 50/50 mix, the test gain is +0.005, against a dev gain of +0.062 raw and +0.053 a′.
   - No paired CI for P-rpo vs P-SFT exists anywhere; `stage_differences` contains J−P only.
   - The comparison is post hoc but is not labelled EXPLORATORY.
   - **Fix:** say "+0.008 raw / +0.010 a′ (EXPLORATORY, no CI); it comes from MansionWorld (k=15) and is absent on holodeck+3D-FRONT; the dev gain did not replicate". Add a house-cluster bootstrap of P-selected minus P-SFT to stage_table, using `report_v2.contrasts`.

2. **The (a′) claim for agents is stronger than the evidence** (findings.md:30, "P exceeds every open model once size bias is removed").
   - The point estimates are true: P 0.499 vs the best open model O2-harness 0.327 (strata: 0.571 vs 0.291).
   - There is no CI on any a′ contrast. `T5_contrasts` has only Δ(a) and Δ(b), while amendment 005 §4 requires agent contrasts as estimates with CIs.
   - The rescale is asymmetric:
     - P is scaled by 1.011 per axis; open models by 1.16–1.25, i.e. footprint ×1.35–1.57, at fixed positions.
     - This also punishes per-object scatter and tight packing (open-model log MAE 0.34–0.45 vs P 0.24), not only mean bias.
     - On raw (a) the ranking is reversed: O2, O6, O4 and O7 score 0.59–0.93, P scores 0.519. The ordering comes from the stress test itself.
   - The agents receive J's input, so P is not same-input with them: it gets train-prior sizes that the agents are not allowed.
   - Only the pooled correction factor was used. The audit showed per-source factors move L from 0.765 to 0.708; this was never checked for the open models.
   - **Fix:** reword to "under the fixed-position mean-size stress test (EXPLORATORY, no CI) P's a′ is above every open model's; this is not what agents would score with unbiased sizes, and P receives prior sizes they do not". Add a′ paired contrasts to T5 in report_v2.py.

3. **"Every J DPO candidate shrinks sizes further" is false** (findings.md:25; the same claim is in the research-log.md 06:45 entry).
   - J e1-dpo2 on dev: footprint bias B −0.071 vs SFT −0.087, so it shrinks less.
   - Its size log MAE on the shared rooms is 0.2516 vs 0.2540 for SFT, so its sizes are more accurate.
   - Raw success 0.397 and a′ 0.377 both exceed SFT. It was rejected only for completeness (0.842).
   - The listed B values leave it out, and it contradicts the mechanism sentence "DPO on J trades size fidelity for geometric validity".
   - **Fix:** "3 of 4 J candidates shrink further; e1-dpo2 does not but is 84 % complete". Qualify or drop the mechanism sentence.

4. **The API agent section is stale and contradicts line 30** (findings.md:40-41).
   - It compares the agent with P-SFT (P 0.510; P−L −0.424 [−0.452, −0.398]). Amendment 010 says all agents are compared with the delivered model, which gives P 0.519 and P−L −0.415 [−0.444, −0.387].
   - The a′ values "L 0.765, J 0.371, P 0.515" mix size-correction factors. J and P use full-test factors (1.0424/0.9925); L uses its own cohort factor (1.0915). Audit wf_b7ae99d6-8df flagged this and it was not fixed.
   - With consistent cohort factors, J is 0.394 and P 0.4945 (SFT), or J 0.393 and P 0.499 (delivered). So J's a′ on the same cohort appears in findings.md as both 0.371 and 0.393.
   - The contrasts "L−J +0.394, L−P +0.250" become +0.371/+0.271 with consistent factors (SFT P).
   - **Fix:** use the selected-report T5 numbers and say which stage each one is from.

5. **The delivered weights were never evaluated, and the gap is unbounded** (findings.md:29; MODEL_CARD.md:22, 240).
   - All test numbers come from the unmerged adapter on its base model. The delivered file is the BF16 merge, which changed only 6.75 % of weights (5.4–9.2 % per module), versus 13.5–14.5 % for the amendment 009 DPO1 merge.
   - Amendment 009's own reasoning is that a BF16 merge erases small LoRA updates.
   - The measured test benefit over P-SFT is only +0.008 and comes from a single source, so there is no evidence the delivered file behaves like the evaluated policy, or better than P-SFT.
   - **Fix:** score the merged file on the 600 dev rooms with the same plan, and report exact-match against the unmerged policy plus dev a′. Until then, say in findings.md that the numbers belong to adapter+base, or ship adapter+base as the primary artifact.

**MINOR**

6. findings.md:10 is stale. "Binding delivery decision is at DPO2 (amendment 008)" was replaced by amendment 010. Line 5 ("Open-model agents running…") and the open questions on lines 55-56 are also out of date.
7. findings.md:30 rounding and an overstatement:
   - L a′ is 0.76547, which rounds to 0.765, not 0.766 (line 41 says 0.765).
   - "Base Qwen3-8B ≈ 0" hides O1b-harness, which scores raw 0.110 → a′ 0.058.
8. findings.md:25 says J a009-dpo2 went "complete 0 (all to the cap)". `scores-report-only` shows 209 of 600 rooms hit the token cap (the research-log 07:15 entry also says 209).
9. findings.md:22 compares different room sets:
   - "Log MAE 0.286→0.590, B −0.08→−0.36": the DPO1 values are on the 3,851 rooms where both J and P completed; the SFT values are on 8,610.
   - "Reward accuracy 1.0" is not in any archived artifact. The research log records it for DPO2 training (02:35 entry), not DPO1.
10. Holm and labelling:
    - report_v2.py:567/708 computes and stores `holm_co_primaries` at every stage.
    - At the selected stage, (b) gets Holm p 0.042 in report.md T1, stage_table.md:79 and the MODEL_CARD. This sits next to the confirmatory 0.066 for effectively the same hypothesis: same J model, same P size prior; only the common-complete room set differs (8,602 vs 8,610).
    - report_v2.py:706 hard-codes `'amendment': '006'` in an amendment-010 report.
    - findings.md:27-29 does not label the selected-stage comparison as secondary rather than confirmatory.
    - **Fix:** emit Holm only for the SFT stage.
11. findings.md:9 is missing a hedge from audit F10. The (b) Holm p is 0.0496–0.0524 with 20,000 draws, so "not confirmed" sits on the 0.05 boundary under bootstrap Monte Carlo error. The audit recommended adding this sentence.
12. findings.md:41 "Lead over GT +0.041 [0.009, 0.073]" is not robust. With per-source correction factors, L's a′ is 0.708, below GT's 0.725 (audit). This is not mentioned.
13. Bootstrap p-values are printed as 0 (results.json a′ p 0.0 and B p 0.0; stage_table.md:70-71 shows 0.0000). With 2,000 draws they should read "< 1/R". Source: report_v2.py:176-178.
14. The (a′) CI treats the correction factor exp(−B/2) as fixed (report_v2.py:138-152 computes it once before the bootstrap). The delivery condition (ii) interval therefore ignores uncertainty in B. The decision is safe (−0.071 vs −0.02), but this should be disclosed or B re-estimated per draw.
15. The MODEL_CARD agent table (delivery.py:299-305) shows only raw (a) and (b), with no size bias B or a′. This breaks the study's own rule at findings.md:47. **Fix:** add the `bias_fp` and `success_debiased_rate` columns from T5.
16. The delivered recipe's defining term is missing from provenance. The RPO NLL term (α 1.0) does not appear in stage_table.md:107 or MODEL_CARD.md:29. It is hard-coded text at delivery.py:197, not read from the training receipt; stage_table.py:362-365 does not render it. **Fix:** render `rpo_alpha` from the receipt and fail if it is absent for rpo candidates.
17. research-state.yaml is stale: `cap_gpu_hours` is 260 (line 26; amendments 009/010 say 600), the phase and observed_utc (04:45) predate the final run, and the pending list is out of date.
18. Every T1 interval, including the confirmatory ones, carries the warning `singleton_source_signature_strata`; findings.md does not mention it.

All files are under `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/`:
- findings.md
- research-log.md
- research-state.yaml
- results/report-v2-selected-001/results.json
- results/report-v2-JPL-001/results.json
- results/report-v2-JPL-002/results.json
- results/report-v2-dpo1-001/results.json
- results/stage-table-010-001/stage_table.md
- results/delivery-010-001/MODEL_CARD.md
- results/dev-selection-010/selection.json
- results/dev-selection-010/scores-report-only/scores.json
- results/audit-sft-20261014/agency-audit-wf_b7ae99d6-8df.json

Code files cited: `/Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/comparison/report_v2.py`, `delivery.py`, `stage_table.py`, `dev_select.py`.