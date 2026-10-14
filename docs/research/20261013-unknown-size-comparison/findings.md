# Current findings (updated 2026-10-14 02:00 UTC, after the Agency SFT-results audit wf_b7ae99d6-8df)

## Status

SFT stage (J0/P0) fully measured on all 8,647 test requests (`results/report-v2-JPL-001`). DPO1/DPO2 (OptiScene recipe, amendments 008/009) in progress; open-model agents running one by one; visual metrics pending. Everything marked EXPLORATORY is post hoc.

## Confirmatory (locked primary endpoint: matched SFT J0 vs P0, Holm over (a), (b); single seed; historical test split)

- Neither co-primary shows an advantage for joint size prediction. (a) all-request geometry success J 0.3882 vs P 0.3896, Δ −0.0014 [−0.0122, +0.0091], Holm p 0.793. (b) size log MAE J 0.2861 vs P 0.2817, Δ +0.0044 [+0.0005, +0.0084], unadjusted p 0.033, Holm p 0.066 (Holm-consistent 97.5 % CI [−0.0001, +0.0089]). **J was not more size-accurate than the train-only prior; J being worse is not confirmed** (the gap is ~0.4 % either way).
- The amendment 005 rule evaluated at SFT would select P; it is informational — the binding delivery decision is at DPO2 (amendment 008).

## Estimation (pre-declared secondary / sensitivity)

- J undersizes: footprint bias B_J −0.083 [−0.095, −0.073] (prior +0.015). Δ(a′) −0.051 [−0.061, −0.041] is a sensitivity analysis with positions held fixed, not the performance of an unbiased J.
- Known-defect exclusions change nothing material (3 rooms: Δ(a) −0.0013, Δ(b) +0.0044; 47 rooms: −0.0008, +0.0045). In the pre-declared GT-pass stratum (6,252 rooms) (a) stays null but **(b) reverses sign**, −0.0050 [−0.0088, −0.0010]; only the small magnitude of (b) is robust.
- Paper strata (pooled 1,009): P has higher raw geometry success, Δ(a) −0.132 [−0.168, −0.097]; on holodeck J's sizes are *more* accurate (Δ(b) −0.018) while the prior undersizes holodeck furniture (B_P −0.13). EXPLORATORY within-source rescaling shrinks the holodeck gap to −0.10 [−0.15, −0.05].
- J's clearest advantage is SceneSmith (held out of fine-tuning and prior fitting; 45 % of objects get the prior's constant fallback box, B_P +1.01): Δ(b) −0.368 [−0.404, −0.336], Δ(a) +0.148 [+0.106, +0.193]. It supplies most of the low-reuse gain (+0.065 of +0.069). On real scans J is *less* size-accurate than the prior (EXPLORATORY pooled Δ(b) +0.078 [+0.064, +0.091]) and strongly undersizes (B_J −0.29). One held-out source cannot establish a general out-of-catalogue benefit.
- Per-source CIs for sources with < 10 house components (InternScenes_mp3d, MultiScan, Scan2CAD) are not valid intervals and are not cited.

## Why J/P fail the geometry gate (EXPLORATORY, all 8,647 rooms)

- Gate failure J 0.612, P 0.610 vs GT 0.277. Shapley split of the gap: main (movable–movable) collisions ≈ 70 % (J 0.239, P 0.225 of 0.335), out-of-bounds 0.040 / 0.066, support/ceiling/incomplete 0.032 / 0.020, blocked fixed objects (≈ half doors) 0.024 / 0.022.
- Collisions are gross, not near misses (80–92 % of colliding pairs overlap > 30 % of the smaller footprint) and concentrate in many-object rooms: failure rises from ~0.32 (1–5 objects) to ~0.98–1.0 (51+), while GT stays 0.21–0.35; four synthetic sources (MansionWorld, InternScenes_gen, SAGE-10k, OptiScene_holodeck) hold ~73–75 % of the excess. Out-of-bounds grows with room area (≥ 60 m²: J 0.40, P 0.51 vs GT 0.03). Many colliding pairs are co-located duplicates of the same category on a shared support.
- Rounding and fixed-object projection in the request are lossless (0 differences); the gap is not an evaluation artefact. Why SFT outputs fail is not identified beyond these descriptive drivers.

## API agent (gpt-6.1-sol, medium reasoning, identical J input; estimation only; 1,454-room cohort = 1,009 strata + 445 others, not representative of the test set)

- Raw geometry success L 0.935 vs J 0.431 / P 0.510 (J−L −0.504 [−0.531, −0.478]; P−L −0.424 [−0.452, −0.398]); cohort GT reference 0.725. The agent's first answers were all schema-valid (raw = harness).
- **The agent strongly undersizes**: footprint bias B_L −0.175 [−0.193, −0.159] (volume −0.292) vs J −0.048 / P −0.023 on the same rooms. Removing each system's own bias (amendment 005 rule, EXPLORATORY for agents): L 0.765, J 0.371, P 0.515 — the agent's lead over J/P survives (L−J +0.394 [0.359, 0.427], L−P +0.250 [0.214, 0.285]) but its lead over GT shrinks to +0.041 [0.009, 0.073].
- The agent exceeds GT only where GT itself fails the gate (mostly 3D-FRONT box-level collisions, i.e. label noise): clean-GT non-holodeck sources GT 0.920 / L 0.793; holodeck 1.000 / 0.994; 3D-FRONT 0.522 / 0.976. Never cite GT 0.72 as a ceiling the agent beats.
- OOR: L 0.031 vs cohort GT 0.048 (strata 0.011 vs 0.026); J 0.111, P 0.173. Its higher size error (log MAE 0.300 vs 0.243 / 0.239) is mainly per-object scatter, not only shrinkage.

## Lessons and constraints

- Always report geometry success next to footprint bias and (a′) for every system, agents included; raw (a) rewards undersizing.
- Compare against GT on the same population (cohort vs full test differ: GT OOR 0.048 vs 0.079).
- Do not write mechanism stories without evidence; label post hoc analyses EXPLORATORY.
- lr 5e-7 LoRA DPO is erased by BF16 merging; official README hparams (lr 5e-6) leave 11–19 % of projection weights changed after merge.

## Open questions

- Does DPO2 (out-of-bounds/overlap hard negatives) reduce the collision-driven gap without more undersizing? (DPO2 training loss reaches ~0 quickly — watch for over-optimisation.)
- Do open-weight agents show the same validity-vs-size trade-off as gpt-6.1-sol?
- Visual realism (FID) and GPT judge scores, J/P vs agents.
