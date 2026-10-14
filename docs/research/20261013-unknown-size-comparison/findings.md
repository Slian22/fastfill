# Current findings (updated 2026-10-14 01:05 UTC)

## Status

SFT stage (J0/P0) fully measured on all 8,647 test requests (report `results/report-v2-JPL-001`, geometry-v2, eval-data-v1 sha-pinned). DPO1/DPO2 (OptiScene recipe, amendments 008/009) training and stage inference in progress; open-model agents running one by one; visual metrics (FID/GPT judge) pending.

## Current understanding

1. **Primary endpoint (protocol: matched SFT J0 vs P0) shows no geometric advantage for joint size prediction.** All-request raw geometry success J 0.3882 vs P 0.3896, Δ −0.0014 [−0.0122, +0.0091]. Size log MAE on common-complete rooms J 0.2861 vs P 0.2817, Δ +0.0044 [+0.0005, +0.0084] (Holm p 0.066 — not confirmatory at α 0.05). P's sizes are the train-only prior, so J did not beat a category×room-type median prior on average.
2. **J systematically undersizes furniture** (footprint bias B_J −0.083 [−0.095, −0.073], volume −0.121), while the prior is near-unbiased (+0.015). After removing each arm's own bias (a′), J is clearly worse (Δ −0.051 [−0.061, −0.041]): J's apparent parity on (a) is partly bought by shrinking footprints — exactly the confound amendment 005 guarded against.
3. **Where J helps:** long-tail and low-reuse data. Rooms containing global-fallback objects: J log MAE 0.384 vs P 0.483 (object level on fallback objects 0.559 vs 1.232); SceneSmith (held-out source) Δ(b) −0.368; low-reuse sources Δ(a) +0.069 [+0.038, +0.101]. **Where P helps:** high-reuse synthetic sources and the paper strata (holodeck strata Δ(a) −0.17 to −0.35), where catalogue sizes repeat and the prior is near-exact.
4. **Both fine-tuned models are far below GT validity**: GT passes the frozen geometry gate in 72.3 % of test rooms; J/P only 39 %. On holodeck strata GT is 100 % valid but J 14–55 %, P 38–76 %.
5. **The API agent (gpt-6.1-sol, medium reasoning, same input) is much more geometrically valid but less accurate in size.** On the 1,454-room cohort: geometry success 0.935 vs J 0.431 / P 0.510; raw OOR 0.031 vs 0.111 / 0.173 (GT 0.079); size log MAE 0.300 vs J 0.243 / P 0.239. Its first answers were all schema-valid (raw = harness). It achieves validity by placing conservatively (OOR well below GT), not by matching GT layouts.
6. Known-defect sensitivity (3 declared rooms; 47 candidate-003 rooms) and GT-pass stratum leave the conclusions unchanged.

## Patterns and insights

- Imitation (SFT) reproduces GT's own collision/out-of-bounds habits and adds its own; GT itself fails the gate 28 % of the time, so a pure imitation target caps validity. A reasoning agent optimises constraints explicitly.
- Joint size prediction learns a shrinkage bias under a loss that does not penalise collisions; this inflates (a) but hurts size fidelity.
- Value of J is concentrated where priors are weak (unseen categories, real scans), which is what an "unknown size" deployment would face outside the catalogue.

## Lessons and constraints

- Always report geometry success next to footprint bias; raw (a) alone rewards undersizing.
- Six-strata (paper) results differ strongly from the all-source picture; never generalise from strata alone.
- lr 5e-7 LoRA DPO is erased by BF16 merging; use official README hparams (lr 5e-6) — merge diagnostics show 11–19 % of projection weights change.

## Open questions

- Does DPO2 (OOB/overlap hard negatives) close the validity gap to GT/the agent without increasing shrinkage?
- Do open-weight agents (Qwen3.8, DeepSeek-V4, gpt-oss, Gemma-4, Muse) show the same validity/size trade-off as gpt-6.1-sol?
- Visual realism (FID) and judge scores for J/P vs agents — pending renders.
