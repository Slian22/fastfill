# Current findings (updated 2026-10-14 11:40 UTC, after the final-deliverable audit wf_c30fbffe-068; amendment 011 locked)

## Status

All test inference is measured. Delivered model: **P = rpo-dpo1** (amendment 010 binding rule; `results/report-v2-selected-001`, `results/stage-table-010-001`, `results/delivery-010-001`). These three outputs are being regenerated from one corrected code snapshot under amendment 011 (reporting only; no J/P number changes). Pending: visual metrics (render running, FID/KID populations fixed by amendment 011 §1, GPT judge), the merged-weight dev check (amendment 011 §5). O5 (DeepSeek-V4-Flash) was not run after 4 engine-build failures and was never substituted. Everything marked EXPLORATORY is post hoc.

## Confirmatory (locked primary endpoint: matched SFT J0 vs P0, Holm over (a), (b); single seed; historical test split)

- Neither co-primary shows an advantage for joint size prediction. (a) all-request geometry success J 0.3882 vs P 0.3896, Δ −0.0014 [−0.0122, +0.0091], Holm p 0.793. (b) size log MAE J 0.2861 vs P 0.2817, Δ +0.0044 [+0.0005, +0.0084], unadjusted p 0.033, Holm p 0.066 (Holm-consistent 97.5 % CI [−0.0001, +0.0089]; with 20,000 bootstrap draws Holm p is 0.050–0.052, i.e. on the 0.05 boundary). **J was not more size-accurate than the train-only prior; J being worse is not confirmed** (the gap is ~0.4 % either way). All CIs carry the `singleton_source_signature_strata` warning.
- The amendment 005 rule gives P at the SFT, DPO1 and selected stages; only the selected-stage application is binding (amendment 010).

## Estimation (pre-declared secondary / sensitivity)

- J undersizes: footprint bias B_J −0.083 [−0.095, −0.073] (prior +0.015). Δ(a′) −0.051 [−0.061, −0.041] at SFT is a stress test with positions held fixed, not the performance of an unbiased J. Caveat: rescaling also changes GT's own success (GT at J's factor 0.610, at P's 0.733), so a′ gaps partly measure the size of the rescale.
- Known-defect exclusions change nothing material (3 rooms: Δ(a) −0.0013, Δ(b) +0.0044; 47 rooms: −0.0008, +0.0045). In the pre-declared GT-pass stratum (6,252 rooms) (a) stays null but **(b) reverses sign**, −0.0050 [−0.0088, −0.0010]; only the small magnitude of (b) is robust.
- Paper strata (pooled 1,009): P has higher raw geometry success, Δ(a) −0.132 [−0.168, −0.097]; on holodeck J's sizes are *more* accurate (Δ(b) −0.018) while the prior undersizes holodeck furniture (B_P −0.13). EXPLORATORY within-source rescaling shrinks the holodeck gap to −0.10 [−0.15, −0.05].
- J's clearest advantage is SceneSmith (held out of fine-tuning and prior fitting; 45 % of objects get the prior's constant fallback box, B_P +1.01): Δ(b) −0.368 [−0.404, −0.336], Δ(a) +0.148 [+0.106, +0.193]. It supplies most of the low-reuse gain (+0.065 of +0.069). On real scans J is *less* size-accurate than the prior (EXPLORATORY pooled Δ(b) +0.078 [+0.064, +0.091]) and strongly undersizes (B_J −0.29). One held-out source cannot establish a general out-of-catalogue benefit. P's weak spot is the global fallback: 16.2 % of rooms / 7.6 % of objects, object log MAE 1.229.
- Per-source CIs for sources with < 10 house components (InternScenes_mp3d, MultiScan, Scan2CAD) are not valid intervals and are not cited.

## DPO with the official OptiScene hyper-parameters collapsed (amendment 009, measured; descriptive)

- DPO1 (lr 5e-6, 10 epochs, 2,200 pairs, 5,500 steps) degraded both arms on all 8,647 test requests: J geometry success 0.388 → 0.213; P 0.390 → 0.093 with only 4,291 complete outputs (2,532 invalid JSON). On the 3,851 rooms complete in both DPO1 arms, J size log MAE is 0.590 and B_J −0.36 (SFT values 0.286 / −0.08 are on 8,610 rooms; different room sets).
- Training logs show the reward margin oscillating and saturating with DPO loss ≈ 0 by epoch 8 (amendment 010). Only the final step was saved.
- Amendment 010 (disclosed as made after viewing DPO1 test): 1-epoch (e1) and 3-epoch + NLL-on-chosen (rpo, α 1.0) candidates, selected on 600 dev rooms only; SFT wins unless a DPO candidate keeps completeness and size accuracy and improves dev a′ by > 0.005. The amendment-009 DPO2 test run was stopped before any output was viewed (it generated to the token cap and would have held 12 GPUs past the shard limit).
- Dev selection: **J keeps SFT.** Three of four J DPO candidates shrink sizes further (dev B: SFT −0.087, e1-dpo1 −0.127, rpo-dpo1 −0.211, rpo-dpo2 −0.139), so raw success can rise (rpo-dpo2 0.435 vs 0.380) while a′ falls. e1-dpo2 does not shrink (B −0.071, size log MAE 0.2516 vs 0.2540, raw 0.397, a′ 0.377 > SFT 0.357) but is only 84 % complete and therefore ineligible. **P selects rpo-dpo1** (P's sizes come from the prior; DPO moves only positions): dev a′ 0.435 vs 0.382, raw 0.470 vs 0.408, completeness and log MAE unchanged. DPO2 stages hurt completeness (P e1-dpo2 43 % complete, rpo-dpo2 96.8 %).
- Report-only dev rows confirm the amendment-009 collapse: J a009-dpo1 complete 0.88 / success 0.18 / B −0.29; J a009-dpo2 complete 0 (209 of 600 rooms hit the token cap, the rest were invalid); P a009-dpo1 complete 0.48; P a009-dpo2 0.05.

## Final selected models and delivery (amendment 010, test, binding rule; secondary comparison)

- Delivered **P = rpo-dpo1** (prior sizes + model placement). Test (a) J-sft 0.388 vs P-rpo-dpo1 0.398, Δ −0.010 [−0.021, +0.002], unadjusted p 0.095; size log MAE 0.286 vs 0.281, Δ +0.0046 [+0.0007, +0.0087] (J not more accurate); (a′) 0.344 vs 0.404, Δ −0.060 [−0.071, −0.049]. Both delivery conditions fail, so P.
- **The dev gain of rpo-dpo1 over P-SFT did not replicate on test (EXPLORATORY, no CI yet):** +0.0084 raw (73 of 8,647 rooms), coming from MansionWorld (+82 rooms, 15 house clusters); without MansionWorld −0.0012, on holodeck + 3D-FRONT −0.0007. Practically P-rpo-dpo1 ≈ P-SFT on test.
- **The BF16 merge loses the DPO gain (amendment 011 §5 dev check):** merged vs evaluated policy on the 600 dev rooms: complete 0.998 vs 0.998, raw success 0.433 vs 0.470, a′ 0.397 vs 0.435 (Δ −0.038, threshold 0.01), only 3.3 % byte-identical outputs. Delivered artifact = adapter + training base only. Any later stage must not train on a BF16 merge of this stage (use an FP32 merge or stacked adapters).

## Same-input agents (1,454-room cohort; estimation only; selected-stage report T5)

- The cohort is not representative of the test set (holodeck 34.6 % vs 5.8 % of test; SpatialLM 2.1 % vs 28.5 %); cohort J−P is −0.089 [−0.118, −0.062] vs −0.010 on all test rooms.
- Raw geometry success (a): gpt-6.1-sol (L, medium reasoning) 0.935; Qwen3.8-27B (O2, thinking) 0.808 raw / 0.928 with up to 3 attempts; Qwen3.8-Flash-Next (O6) 0.701; gpt-oss-120b (O4) 0.687; Muse-Glimmer-30B (O7) 0.592; gemma-4-31b (O3, thinking off) 0.241; untuned Qwen3-8B 0.000 (O1a, no thinking) / 0.110 (O1b, thinking, 3 attempts); our P 0.519, J 0.431. P−L −0.415 [−0.444, −0.387]; P−O2 (3 attempts) −0.409 [−0.436, −0.380]; P−O6 −0.181 [−0.211, −0.148].
- **Every agent undersizes:** footprint B gpt-6.1-sol −0.175; open models −0.30 to −0.45; our P −0.022, J −0.048 on the same rooms. Sizes were compared with the annotated size of each object.
- **Fixed-position size stress test (a′, own-bias rescale; EXPLORATORY, no CI yet):** L 0.765, P 0.499, J 0.393, O2 0.327, O4 0.303, O6 0.281, O7 0.219, O3 0.098, O1b 0.058, O1a 0. This mostly measures how much each system is enlarged: GT itself drops from 0.725 to 0.093 when enlarged by the open models' factor (×1.17 per axis). The audit's preliminary equal-size control (each object given its GT size, positions fixed) keeps the order but narrows P−O2 from 0.17 to about 0.06 (L 0.369, P 0.273, J 0.235, O2 0.212); amendment 011 reports both with CIs. P also receives prior sizes the agents do not have, so P vs agents is not strictly same-input.
- Harness/budget caveats: "harness" = up to 3 attempts with a generic retry message (exception class only); L had a 65,536-token budget vs 1,024 + 128·objects (+16,384 with thinking) for open models, and 241 of O2's 252 raw failures are token-cap truncations. O1a/O1b near-zero results are failures of the support (`on`) field (505 of O1a's 506 complete rooms), not evidence about layout skill.
- On the 445 cohort rooms outside the six paper strata the order flips for some pairs: J 0.389 > P 0.339 and O2 0.843 > L 0.820.
- gpt-6.1-sol vs GT: it exceeds GT mainly where GT itself fails the gate (3D-FRONT box-level collisions); with per-source size factors its a′ (0.708) is below GT's 0.725, so it has no robust lead over GT.

## Output-side repair and best-of-N (amendment 012, dev 8,167 rooms; decisions per the locked rule)

- **S1 deterministic repair is adopted** (XY moves of support subtrees only; categories, sizes, yaw, parents unchanged). Downstream acceptance (EmbodiedGen path-A port) G → G+S1: J 0.173 → 0.253 (+0.080 [0.074, 0.086]), P 0.199 → 0.292 (+0.093 [0.086, 0.100]); checker success roughly doubles (J 0.388 → 0.711, P 0.400 → 0.708); median displacement of moved objects 0.21–0.23 m.
- **S2 best-of-8 is not adopted**: it raises raw checker success strongly for P (0.400 → 0.600) but after repair adds only +0.020 [0.015, 0.026] downstream acceptance for P (threshold 0.03) and +0.003 for J (size guards block most J alternatives).
- Reference: the GT layouts of the same dev rooms reach 0.293 downstream acceptance (checker 0.722): the downstream rules (every object reachable from the door, AABB collisions incl. tucked chairs, 1 mm envelope) reject most real layouts; P + S1 matches GT on this metric. vLLM greedy (the S2 engine) is not the HF exact policy (P 0.432 vs 0.470 checker success on the 600 dev-select rooms).

## Why J/P fail the geometry gate (EXPLORATORY, all 8,647 rooms)

- Gate failure J 0.612, P 0.610 vs GT 0.277. Shapley split of the gap: main (movable–movable) collisions ≈ 70 % (J 0.239, P 0.225 of 0.335), out-of-bounds 0.040 / 0.066, support/ceiling/incomplete 0.032 / 0.020, blocked fixed objects (≈ half doors) 0.024 / 0.022.
- Collisions are gross, not near misses (80–92 % of colliding pairs overlap > 30 % of the smaller footprint) and concentrate in many-object rooms: failure rises from ~0.32 (1–5 objects) to ~0.98–1.0 (51+), while GT stays 0.21–0.35; four synthetic sources (MansionWorld, InternScenes_gen, SAGE-10k, OptiScene_holodeck) hold ~73–75 % of the excess. Out-of-bounds grows with room area (≥ 60 m²: J 0.40, P 0.51 vs GT 0.03). Many colliding pairs are co-located duplicates of the same category on a shared support.
- J/P were trained only on gate-qualified targets (main_data.py admits a room only if J's GT and P's teacher target pass the frozen checker), so the failures come from generation, not from copying bad labels.
- Rounding and fixed-object projection in the request are lossless (0 differences); the gap is not an evaluation artefact.

## Lessons and constraints

- Always report geometry success next to footprint bias for every system; raw (a) rewards undersizing; a′ is a stress test whose size depends on the rescale factor — add GT and equal-size controls.
- Compare against GT on the same population (cohort vs full test differ: GT OOR 0.048 vs 0.079).
- Do not write mechanism stories without evidence; label post hoc analyses EXPLORATORY; check that a dev gain replicates per source before claiming it.
- lr 5e-7 LoRA DPO is erased by BF16 merging; official README hparams (lr 5e-6) left 11–19 % of projection weights changed after merge; the delivered rpo-dpo1 merge changed only 6.75 % — evaluate merged weights before delivering them.
- 10-epoch DPO on 2,200 pairs over-optimises and collapses generation; save every epoch and select on dev, never on test. Synthetic-perturbation negatives (DPO2) are too easy and do not teach the model's own failure (co-located duplicates).
- FID must not require every system to have rendered a room (one failing system empties the population).

## Open questions

- Visual realism (FID/KID) and GPT judge scores of J/P vs agents (amendment 011 populations).
- Does the delivered BF16 merge behave like the evaluated policy on dev (amendment 011 §5)?
- Output-side improvements (amendment 012, not yet written): deterministic repair, local re-generation, best-of-N with a size gate, on-policy DPO — measured by downstream acceptance (1 mm / 1 cm) with open models given the same repair.
