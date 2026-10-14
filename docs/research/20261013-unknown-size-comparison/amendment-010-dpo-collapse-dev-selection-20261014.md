# Amendment 010 — DPO collapse under the official hyper-parameters; regularised DPO candidates selected on dev only

Locked 2026-10-14 ~04:30 UTC. **Disclosure:** this amendment is made *after* viewing the DPO1 test results of amendment 009 (report `results/report-v2-dpo1-001`). DPO2 test outputs have not been viewed. Nothing below uses test data for selection.

## What happened (measured, reported as is)

With the official README hyper-parameters (lr 5e-6, 10 epochs, LoRA r32/α16, β 0.1, 2,200 pairs, 5,500 steps), DPO1 degraded both arms on the full test set: J geometry success 0.388 → 0.213, size log MAE 0.286 → 0.590, footprint bias −0.08 → −0.36; P geometry success 0.390 → 0.093 with only 4,291/8,647 complete outputs (2,532 invalid JSON). Training logs show the reward margin oscillating and saturating (loss ≈ 0 by epoch 8): a classic DPO over-optimisation / likelihood-displacement collapse. Only the final step was saved, so the run cannot be early-stopped retroactively. The amendment 009 DPO1/DPO2 results stay in the stage-ablation table as measured.

## Regularised candidates (both arms, identical settings, same 2,200 positives and the existing stage-1/stage-2 pairs)

| Candidate | Recipe |
|---|---|
| `sft` | the existing SFT model (always eligible) |
| `e1` | official README hyper-parameters but **1 epoch** per stage (550 steps), DPO1 → DPO2 |
| `rpo` | lr 5e-6, **3 epochs**, plus an NLL term on the chosen response (RPO, α = 1.0) per stage, DPO1 → DPO2 |

Each stage trains on the BF16 merge of the previous stage (as before); each candidate stage is evaluated with its exact policy (unmerged adapter on its training base). Checkpoints are saved every epoch.

## Dev-only selection

- Dev set: 600 rooms from the length-qualified dev pool (never used for training or for any test decision), 300 from OptiScene_holodeck + IL3D_3dfront and 300 from all other sources, stable hash `sha256('dev-select-v1:' + uid)`; frozen with its sha before any candidate is scored.
- Per arm, every candidate stage (sft, e1-dpo1, e1-dpo2, rpo-dpo1, rpo-dpo2) is scored on dev with the frozen geometry code and the same inference plan rules (shared length-sorted plan, caps as amendment 002).
- **Eligibility:** dev complete rate ≥ SFT dev complete rate − 0.01 and dev size log MAE ≤ SFT dev log MAE + 0.01.
- **Selection:** among eligible candidates the highest dev size-debiased geometry success (a′, using the candidate's own dev footprint bias); ties within 0.005 go to the candidate with fewer training stages. SFT wins if no DPO candidate is eligible or better.
- Only the selected model of each arm is run on the test set (if it is not already measured). The amendment 005 delivery rule is then applied to the two selected models (J-selected vs P-selected) on the test set; that decision is binding and replaces "J-DPO2 vs P-DPO2" of amendment 008.

## Reporting

The stage-ablation table reports: SFT, amendment-009 DPO1/DPO2 (collapsed), and the dev scores of all regularised candidates; test scores only for selected models. All agents are compared with the delivered model. Budget: GPU cap 600 GPU-hours unchanged.

## Addendum (2026-10-14 ~04:40 UTC, before any amendment-009 DPO2 test output was viewed)

The amendment-009 DPO2 test inference (infer-v2-004) progressed at ~460/8,647 (J) and ~1,020/8,647 (P) rows in its first 2 h — far slower than the SFT and DPO1 runs, so it would exceed the 14 h shard limit while holding 12 GPUs (≈ 170 GPU-h) that the regularised candidates need. Because its starting point (amendment-009 DPO1) collapsed and the final model is now chosen on dev (above), the amendment-009 DPO2 test run is stopped; its partial outputs are not analysed (the processed rooms are the largest ones, a biased subset). Amendment-009 DPO1 and DPO2 are instead scored on the same 600-room dev set as the candidates and reported in the stage-ablation table as dev-only, report-only rows (not selectable). The agent evaluations that were paired with amendment-009 DPO2 are cancelled with it; agents are re-paired with the selected models.
