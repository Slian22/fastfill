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
