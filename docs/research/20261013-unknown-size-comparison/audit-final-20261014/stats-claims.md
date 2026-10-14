**Completeness audit: unknown-size J/P comparison deliverable (read-only, as of 2026-10-14)**

Study root (paths below are relative to it): `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison`

**Checks that passed (recomputed with python3 / shasum)**
- **Package integrity:** all 16 `report-selected/*` hashes in `results/delivery-010-001/SHA256SUMS` match the local files in `results/report-v2-selected-001`, and `selection.json` matches f094395f….
- **Holm adjustment:** selected stage (a) 0.095, (b) 0.042. SFT stage (a) 0.793, (b) 0.066. Both are correct.
- **Scale factors:** exp(−B/2) is correct for J (1.0423), P (0.9928), L (1.0915) and O6 (1.1888).
- **Dev selection:** re-applying the amendment-010 rule to `stage_dev` rows gives J = sft and P = rpo-dpo1. J rpo-dpo2 fails on completeness (0.985 < 0.99). P rpo-dpo2 fails too (0.9683 < 0.9883).
- **Table consistency:** T6 complete counts match the T5 and stage-table complete rates. The T4 per-stratum complete counts add up to the strata_pooled rates (O1a-h 259/1009, O3-h 410/1009, O1b-h 616/1009).
- **Findings:** the numbers on findings.md lines 25 and 29–30 match `stage_dev` and T5.

## DONE
- **One model:** P = rpo-dpo1 is delivered by the amendment-005 rule, binding under amendment 010. Neither condition holds: Δ(b) +0.0046 [+0.0007, +0.0087]; Δ(a′) −0.0604 [−0.0712, −0.0493].
- **Delivery package:** merged BF16 weights, adapter, training base, size prior, hashes and model card.
- **Primary confirmatory result:** SFT J0 vs P0 is in `stage_table` and `report-v2-JPL-002`.
- **Stage ablation:** SFT, amendment-009 DPO1 (collapsed), DPO2 stopped, the selected models, and the dev table.
- **Same-input agent benchmark:** 1,454-room cohort and 1,009 six-strata rooms. Systems: L (gpt-6.1-sol, medium), O6, O4, O2, O3, O7, O1a, O1b. Per system: complete, (a), (b), IoU, OOR, B, a′, cost and failure taxonomy (T5, T5_contrasts, T5_cost, T6).
- **Paper six-strata OOR** for every system (T4).

## PENDING
- **Visual run** `visual-final-selected-002` (rendering): FID/KID for every system, GPT judge on J/P/L/O6 over 174 rooms. Every FID/KID/GPT cell in T4 is "—"; T4_gpt is `[]`.
- **O5 (DeepSeek-V4-Flash): not run** after 4 engine-build failures (research-log 08:05). It was never substituted.
- **Not measured:** human UR; automatic acceptance and all-room rotated OOR in the stage table (promised in amendment 008).

## DEFECTS (most severe first)

**B1 BLOCKER: the FID cohort will be empty or tiny.**
- **Cause:** in systems mode, FID uses the per-stratum intersection of *all* systems (`fastfill/comparison/fid.py:21` and `_stratum_result` at about line 343; `visual_cases.py:345`). `visual_final.py` puts every completed agent into that intersection (research-log 10:50: "8 个代理全部纳入").
- **Effect:**
  - O1a-harness has 0 complete outputs in all four holodeck strata (report.md T4 lines 171, 189, 207, 225). FID is therefore `insufficient_sample_size` for J and P too in 4 of 6 strata.
  - The 3D-FRONT living stratum has N ≤ 13 (O3-harness 13/87).
  - Even without O1a, O3-harness caps holodeck bedroom/living at 33/36, below the N<50 flag.
- **Fix:** before any FID is viewed, lock amendment 011. Compute J/P FID on reference∩J∩P (paper-style primary), and each agent's FID on reference∩J∩P∩that agent, reporting N per system. Alternatively, limit the fid-manifest `systems` to the judged set {J, P, L, O6} and report the others descriptively.

**M1 MAJOR: O5 and the not-run list are missing from every user-facing output.**
- `delivery.json` `agents.excluded = {}`; `MODEL_CARD.md:232` lists only 8 agents; report T5 has no O5 row; findings.md never mentions O5. Amendment 007 says not-run systems are "reported in the tables with reason".
- Cause: final_chain_2 was submitted with 8 agents, so `dev_select.final` never recorded O5 (`submit_final_010.sh:118-131`).
- **Fix:** rerun delivery/report with `excluded = {"O5": "not run: 4 engine-build failures (DeepGEMM nvcc/gcc; TileLang cicc SIGSEGV under nvcc 13.4), never substituted"}`. Add a "not run" row block to T5 that includes the amendment-007 "too large / gated" list.

**M2 MAJOR: the model card's agent table can mislead.**
- `MODEL_CARD.md:116-155` shows only raw (a) and (b), with system IDs and no model names. A reader sees P 0.519 against L 0.935 and O2 0.928 with no undersizing context. This contradicts findings.md:47 ("always report (a) next to B and (a′) … agents included").
- **Fix:** in the delivery.py agent table, add model/reasoning columns (from `results.json` agents) and the `bias_fp` and `success_debiased_rate` columns from T5.csv.

**M3 MAJOR: the delivered bytes were never evaluated.**
- `MODEL_CARD.md:22` and the first snippet (lines 53-66) point users to the merged model, which was not evaluated. The BF16 merge changed only 0.0675 of weights (`delivery.json` `model.merge_changed_fraction`). Amendment 009 showed that a small rounded-away update can erase a DPO stage's effect.
- **Fix:** run a 600-room dev inference with the merged model and report complete/(a)/a′ next to the unmerged policy. Until then, make adapter + training-base the primary usage in the card and mark the merged model as unverified.

**M4 MAJOR: findings.md is stale and internally inconsistent.**
- **Line 10:** "binding delivery decision is at DPO2 (amendment 008)". This is superseded by amendment 010; the binding comparison is the selected models.
- **Line 5:** says "Open-model agents running … visual pending" and cites `report-v2-JPL-001`. The agent runs are done, and JPL-001 is superseded by JPL-002.
- **Lines 38-43 (API-agent section):** the numbers are for **P-SFT** (P 0.510, P−L −0.424) and use full-test factors (a′ J 0.371, P 0.515, from `results/audit-sft-20261014/agency-audit-wf_b7ae99d6-8df.json:28`). Line 30 gives cohort-own-B values for the same J on the same cohort (0.393) and for the delivered P (0.499). Two different a′ values for one model appear unlabelled.
- **Lines 53-57 (open questions):** Q1 is answered (J keeps SFT; P rpo-dpo1 gains +0.0085 raw on test). Q2 is answered (all open models undersize, B −0.30 to −0.45).
- **Fix:** rewrite against the selected report: one agent section, P = rpo-dpo1, cohort-own-B a′, and open questions reduced to visual and O5.

**M5 MAJOR: there is no a′ contrast between the delivered P and the agents.**
- `T5_contrasts` has only Δ(a) and Δ(b) (`report_v2.py` T5 contrasts). Findings.md:30 says "P exceeds every open model … only the API agent stays ahead" on a′ with no interval. The only L−P a′ CI is for P-SFT, from the audit.
- **Fix:** add paired Δ(a′) with a house-cluster CI to T5_contrasts (cohort and strata_pooled).

**M6 MAJOR: the GPT judge output only compares J and P.**
- The output is J−P (`report_v2.py:371-395`, T4_gpt). Once the visual run lands, the report still won't have a pooled 174-room mean per system or P−L / P−O6 contrasts. T4 has only per-stratum means.
- **Fix:** add a pooled per-arm mean and P−L / P−O6 paired differences (estimation only, stratified bootstrap) from the judge records.

**M7 MAJOR: research-state.yaml is stale.**
The file was modified at 11:52, but:
- **Line 6:** `observed_utc` is still 04:45Z.
- **Line 5:** `phase` is "dev_selection".
- **Line 7:** still says "source_correction_candidate_002_in_verification", but v3.3 is published.
- **Line 15:** `api_agent.evaluation: pending_after_J_P_merges`.
- **Line 21:** `open_models_rerun.done` lists only O6/O4.
- **Line 26:** `cap_gpu_hours: 260`; the cap is 600 since amendment 008.
- **Line 28:** `pending: [measured_tables, final_review]`.
- **Line 33:** automation still shows the run004 status (died 2026-10-13 15:11Z).
- **Missing:** no `delivery` (P rpo-dpo1, delivery-010-001) or `visual` run (visual-final-selected-002) entry.
- **Fix:** update these fields and add `delivered` and `visual_run` entries.

**m1 MINOR: findings.md:25 overstates J's shrinkage.** "every J DPO candidate shrinks sizes further" is false for e1-dpo2 (dev B −0.0712 vs SFT −0.0870; `stage_table.md:43`). Fix: say "e1-dpo1/rpo-dpo1/rpo-dpo2".

**m2 MINOR: findings.md:30 "base Qwen3-8B ≈ 0".** O1b-harness has raw (a) 0.110 and a′ 0.058 (T5). Fix: "O1a 0; O1b 0.110 → 0.058". The same wording appears in research-log 09:05 and 10:55.

**m3 MINOR: wrong amendment label.** `results.json` `amendment: "006"` is hardcoded at `report_v2.py:706`; the selected report belongs to amendment 010. `holm_co_primaries` is also a misnomer at the secondary stage.

**m4 MINOR: the T4 note describes a visual run that doesn't exist.** With no visual input, `report.md:229` says "此 J/P 视觉运行不产出 KID，记为未测" (the else branch at `report_v2.py:791-793`). Fix: when `visual_mode.fid` is None, say that visual metrics are pending/not provided.

**m5 MINOR: the selected P recipe reads as plain DPO.** `rpo_alpha` is dropped from the recipe strings at `stage_table.py:364-365` and `delivery.py:205-207`, so `stage_table.md:107` and `MODEL_CARD.md:29` read as 3-epoch DPO. Fix: append `rpo_alpha` when present.

**m6 MINOR: the prior-level split was never produced.**
- `S2` merges pair and category into "N" (warning `prior_level_pair_category_unresolved_without_frozen_size_priors`). That breaks the amendment-005 §3 pair/category/fallback split.
- The frozen `size-priors.json` (073522c1…) is shipped in the package.
- **Fix:** rebuild the reference `prior_levels` with that file and regenerate S2.

**m7 MINOR: agent reports call every agent "L".** `evaluate_llm.py:34,91` labels every agent "L". `results/agents-sft-paired/O*/report.md` titles read "LLM 代理 L", with rows "L-raw/L-harness" for Qwen/gemma etc. (O2's "L-harness 0.9278" collides with gpt-6.1-sol's 0.9347). Fix: use the system label and model name.

**m8 MINOR: superseded files carry no notice.**
- `results/report-v2-JPL-001/report.md:5-9` still shows "交付决定 … 交付：P" for the SFT stage.
- `report-v2-JPL-002/report.md:7-9` and `report-v2-dpo1-001/report.md:7-9` say "binding decision at DPO2, amendment 008".
- `current-results.md:1-3` (2026-10-13) says "必须…重训两臂".
- **Fix:** add a `results/README` that marks these superseded. Don't edit the hashed outputs.

**m9 MINOR: T5 has no GT row for the cohort.** The cohort GT success (0.725) appears only in findings.md:40, via the audit. Findings.md:48 requires GT on the same population. Fix: add GT rows (cohort and strata_pooled) to T5.

**m10 MINOR: no within-arm test contrast for the delivered model.** P-rpo-dpo1 − P-SFT has no paired CI. Findings.md:29 claims +0.009 without an interval; the stage table has only J−P. Fix: add a paired stage contrast for the delivered arm.

**m11 MINOR: the card's inference snippet is not the evaluated path.**
- `MODEL_CARD.md:51-63` uses a per-request cap `1024+128×objects` with batch 1.
- The evaluation used a per-batch cap `min(40960−W, 1024+128×max_objects_in_batch)`, with W shared across both arms and batch 8 left-padded (`infer_shard.py:6,81`).
- **Fix:** state this, and note that greedy outputs may differ from the HF path as well as from vLLM.

## Minimal tables for the final user-facing summary

1. **Delivery decision (full test, 8,647):** J-sft vs P-rpo-dpo1 — (a), (b), (a′), B, complete, rule outcome, GT 0.7230.
   - Source: `results/report-v2-selected-001/T1.csv`, `T1_bias.csv`, `results/delivery-010-001/delivery.json` (`decision`).
2. **Primary confirmatory (SFT J0 vs P0, Holm):**
   - Source: `results/stage-table-010-001/stage_differences.csv` (stage=sft; (a), (b)), cross-checked with `results/report-v2-JPL-002/T1.csv`.
3. **Stage ablation (Table-6 analogue) plus dev selection:**
   - Source: `stage_stages.csv`, `stage_dev.csv`, `stage_decisions.csv`, `stage_lineage.csv` (all in `stage-table-010-001`; recipe needs the m5 fix).
4. **Same-input agent benchmark (cohort 1,454 and strata_pooled 1,009):** model name and reasoning, complete, (a), B, (a′), (b), IoU, OOR. Harness is primary and raw is the sensitivity; add GT rows and an O5/not-run block.
   - Source: `report-v2-selected-001/T5.csv` + `results.json` `agents`, `T5_contrasts.csv` (add Δ(a′)), `T5_cost.csv`, `T6.csv`.
5. **Paper six strata:** OOR (axis-aligned and rotated) per system with GT; FID/KID/GPT once the B1 fix is in and the visual run lands. Paper values as context only.
   - Source: `report-v2-selected-001/T4.csv`, `T4_gpt` (pending), `T4_paper.csv`.
6. **Optional, where the J-vs-P story lives:** per source with GT (`T3.csv`) and sensitivity (`S1.csv`: GT-pass reverses Δ(b); SceneSmith/low-reuse).