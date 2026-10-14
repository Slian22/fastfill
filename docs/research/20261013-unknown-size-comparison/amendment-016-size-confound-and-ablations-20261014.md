# Amendment 016 — size-confound evidence and attribution ablations (paper-focused, EXPLORATORY)

Locked 2026-10-14 (UTC, before any output below exists). Motivated by two external reviews pasted by the user: the paper's thesis is "size choices confound feasibility metrics → size-controlled downstream evaluation → size-preserving repair-based preference supervision", not "we beat OptiScene". Nothing here changes amendments 005–015 results or decisions.

## 1. Scale–pass curves (intervention on sizes; CPU only)

- **Systems** (existing outputs, positions/yaw/support fixed): GT, J, P on all 8,647 test rooms; J, P, GT and every same-input agent (L, O1a, O1b, O2, O3, O4, O6, O7) on the 1,454-room cohort; the reproduced OptiScene stages on their E2 rooms when available.
- **Intervention:** every object's x/y (and, in a second variant, x/y/z) size is multiplied by s ∈ {0.70, 0.75, …, 1.00, 1.05, 1.10} about its own bottom centre. The room is re-scored with the frozen checker and the downstream acceptance port.
- **Outputs:**
  - curves of checker success and downstream acceptance vs s;
  - the slope over 0.80–1.00 per system;
  - the s at which each system's pass rate matches GT at s = 1.

  A system that passes by shrinking has a steep curve; one that passes by placement has a flat curve. House-cluster bootstrap bands are reported (2,000 draws, seed 42).

## 2. Size-fixed regeneration (ranking under fixed sizes)

- **What runs:** the free open-weight agents O2 (Qwen3.8-27B), O6 (Qwen3.8-Flash-Next), O4 (gpt-oss-120b) and O7 (Muse-Glimmer-30B) re-run on the 1,454-room cohort with **P's input**: sizes given as the train prior, the model places only. The prompt is P's system prompt plus the same JSON-only suffix and harness as before; sizes are taken from the request, never from the answer. The API agent L is not re-run (cost).
- **Comparison:**
  - The ranking of {P, agents} under fixed sizes is compared with the size-free ranking (J input, own sizes). Reported for raw and S1-repaired outputs, with paired CIs.
  - Each agent's size-free vs size-fixed difference is reported.
- **Budget:** ≤ 40 GPU-hours.

## 3. DPO2 attribution ablation (dev, single-shot greedy, no repair)

All runs start from rpo-dpo1's exact policy (training base + rpo-dpo1 adapter, `--init_adapter`) and use the same number of optimiser steps as op-dpo2 epoch 2 (amendment 013):

- **(a) rpo-dpo1:** the start point.
- **(b) RFT:** continued SFT (NLL only) on the chosen texts of the amendment-013 pairs.
- **(c) Perturbation DPO2 (OptiScene-style):** DPO on the existing dpo-001 stage-2 pairs (GT chosen, synthetic out-of-bounds/overlap rejected), RPO α 1.0.
- **(d) op-dpo2:** amendment 013.

All are evaluated on the 600 dev-select rooms (HF exact policy) with the amendment-013 metrics: complete, a′, downstream acceptance, size log MAE. Paired CIs vs (a) and vs each other use the house-cluster bootstrap.

**Interpretation, fixed in advance:**
- if (d) > (b) on a′ and acceptance (CI excluding 0), the preference construction has value beyond its repaired data;
- if (d) ≈ (b), the contribution is the repair-based data;
- if (c) ≈ (d), real failures add nothing over synthetic ones.

Test is touched only by amendment 013's adopted model.

## 4. OptiScene + the same repair (system-level baseline)

The adopted S1 repair (amendment 012) is applied at the same settings to the amendment-014 E2 outputs (and to E3 when run). OptiScene+S1 vs P+S1 downstream acceptance is reported next to the single-shot comparison.

## 5. Stratified reporting

Every table of amendments 012–016 is also reported by:
- object-count bin: ≤ 20, 21–50, > 50;
- real scans (InternScenes real, SpatialLM, MultiScan, Scan2CAD, HSSD, InteriorGS, Structured3D) vs procedurally generated / synthetic sources;
- room family (amendment 015).

These breakdowns are descriptive, with no new decision rules.

## 6. Checker validity (descriptive)

GT pass rates and failure reasons per source, and the sensitivity of GT / J / P / agent pass rates to the out-of-bounds tolerance (1 mm, 1 cm, 10 cm) and to counting tucked chairs as collisions (downstream A4) or not (frozen checker).

## Addendum A (locked before any amendment-016 output exists; from a second external review)

1. **Ranking criterion for §2 (fixed in advance).**
   - **Rankings:** systems are ranked by downstream acceptance on the cohort, raw and after S1. Systems: P, O2, O4, O6, O7, and L on its subsample. The size-free ranking uses each agent's own sizes (J input). The size-fixed ranking uses P's input (sizes given).
   - **Statistic:** Kendall τ between the two rankings, with a house-cluster bootstrap CI.
   - **Pre-specified flips:**
     - P vs the strongest open model (O2);
     - P vs L;
     - OptiScene (E2, prior-size mode) vs P.

     Each is reported as a paired difference with CI in both settings.
   - **Which sizes:** the primary result uses the train-prior sizes (P's input). GT sizes are a supplementary figure, run only for O2 and only on the pilot subsample.
2. **gpt-6.1-sol, size-fixed:** L is re-run with P's input on a 300-room stable-hash subsample of the cohort (`sha256('a016-L:' + uid)`, stratified by object-count bin). Its size-free counterpart is the same 300 rooms from its existing run. This is within the remaining amendment-006 API call cap; one attempt plus the same harness.
3. **Primary size evidence for §1:** the size-substitution effect per system. On the same rooms with positions, yaw and support fixed, it is pass(own sizes) − pass(prior sizes) and pass(own) − pass(GT sizes), with paired CIs, shown next to the footprint bias B. The scale curves only show sensitivity, and the GT curve is drawn on the same figure as the reference. "Steep" is never read as "gaming" on its own.
4. **Rule-level failure table (§6):** for GT and every system (raw and after S1), the share of rooms failing each rule:
   - downstream: R1/R2 1 mm envelope, R5, A2 door keep-out, A3 reachability, A4 AABB collisions incl. tucked chairs, A6/A7 door clearance, required-class counts;
   - checker: OOB, support, ceiling, main/severe collisions, fixed blocking.

   This table defends the GT downstream acceptance of 0.29 and shows how the metric definition drives conclusions.
5. **§3 additions:**
   - every arm also reports B and is stratified by object-count bin;
   - RFT and op-dpo2 (the two key arms) get a second training seed (seed 43);
   - rpo-dpo1 and the perturbation arm are single-seed, stated as such.
6. **Pilot first:** §2 runs on a 300-room stable-hash pilot (`sha256('a016-pilot:' + uid)`, stratified by object-count bin) for every system before the full cohort, to get an early reading under GPU queueing. Pilot numbers are labelled as such and never replace the full results.
7. **Wording fallback (fixed now):** if the stratified tables do not show the gap widening in dense (> 50 objects) or real-scan rooms, the paper says "evaluation must cover dense and real-scan rooms" rather than "the real difficulty is there".

## Addendum B (locked before any amendment-016 output exists; implementation details from the same review)

1. **One pilot subsample:** L's size-fixed subsample IS the pilot subsample (`sha256('a016-pilot:' + uid)`, stratified by object-count bin, 300 cohort rooms); the `a016-L:` hash of addendum A is withdrawn. Kendall τ is computed twice: on the 300 pilot rooms including L, and on all 1,454 cohort rooms without L.
2. **Size-fixed runs enforce sizes.** Before scoring, the harness overwrites any size in an answer with the request size, and recomputes the z of every supported object as its parent's top under the request sizes; floor objects keep z = 0 if the answer put them on the floor. The share of answers (and of objects) whose returned sizes differ from the request is reported per model ("unauthorised size changes"), as evidence in itself. An answer that only differs in sizes is not counted as invalid.
3. **Size substitution recomputes support heights.** In §1 (own vs prior vs GT sizes, and the scale curves), after replacing sizes, the z of every supported object is recomputed from its parent's new height before re-scoring, so a lamp is never left floating or embedded by the substitution itself.
4. **Visual metrics caveats.** The GPT judge is gpt-6.1-sol, which is also a compared system (L); its scores are reported as a secondary metric with this conflict stated (no claim rests on them). FID is also size-sensitive (rendered furniture scale), which is stated next to the size-substitution results; no extra experiment.
