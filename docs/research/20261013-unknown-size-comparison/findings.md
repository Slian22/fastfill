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

### Test application (amendment 012 §4, EXPLORATORY; `results/test-apply-012/`, 2,000 house-cluster draws)

- Full test (8,647): downstream acceptance raw → S1: J 0.172 → 0.258, P 0.200 → 0.288, GT 0.295 (checker P 0.398 → 0.701, GT 0.723). P+S1 − J+S1 = +0.030 [+0.020, +0.039]. The dev effect of S1 replicates on test.
- Cohort (1,454), all systems given the same S1, downstream acceptance raw → S1 (footprint bias B):

  | System | raw → S1 | B |
  |---|---|---|
  | GT | 0.448 | 0 |
  | P | 0.354 → 0.472 | −0.022 |
  | J | 0.234 → 0.372 | −0.048 |
  | L (gpt-6.1-sol) | 0.728 → 0.754 | −0.175 |
  | O2 harness | 0.723 → 0.752 | −0.311 |
  | O6 | 0.536 → 0.633 | −0.346 |
  | O4 | 0.479 → 0.597 | −0.296 |
  | O7 | 0.410 → 0.500 | −0.323 |
  | O3 | 0.159 → 0.223 | −0.450 |
  | O1b harness | 0.069 → 0.171 | −0.313 |
  | O1a | 0 | −0.421 |

  - With S1, P beats O3, O1a and O1b. It trails L, O2, O6 and O4. Versus O7 the difference is small: −0.028 [−0.054, −0.001] for harness and −0.016 [−0.042, +0.012] for raw.
  - **Every agent that beats P also beats GT itself (0.448) while shrinking furniture by 16–29 % in footprint (B −0.18 to −0.35).** Under self-chosen sizes, downstream acceptance rewards undersizing. This is the size confound that amendment 016 tests with size substitution and size-fixed regeneration; no ranking claim is made from these rows.

## On-policy DPO2 (amendment 013; dev-select, 600 rooms, HF exact policy)

- op-dpo2-e1 is adopted. Against rpo-dpo1:
  - checker success 0.470 → 0.515;
  - a′ 0.435 → 0.488;
  - downstream (strict admission) acceptance 0.252 → 0.332;
  - complete 0.998 → 0.995.
- The gain holds in both dev halves:
  - a′ in OptiScene_holodeck + IL3D_3dfront rooms: 0.467 → 0.563;
  - a′ in other sources: 0.403 → 0.413.
- Epoch 2 over-trains: complete falls to 0.942.
- **Pair data.** 4,496 pairs from 1,514 train rooms. Each rejected sample is the model's own failure; its chosen is that same sample's S1 repair.
- **Test.** No test number yet: the test run is deferred to amendment 021 §4 (013 addendum A).
- **Attribution (amendment 016 §3 ablation, `results/ablation-016/`; dev, 600 rooms; 2 seeds for the key arms).**
  - All arms start from rpo-dpo1 and take the same number of optimiser steps. The pre-registered comparison uses op-dpo2 epoch 2.
  - **(b) RFT** (continued SFT on the same repaired chosen texts) ≈ rpo-dpo1:
    - a′ +0.002 / +0.010;
    - acceptance +0.018 / +0.020;
    - every CI includes 0.
  - **(d) op-dpo2 vs (b) RFT, same data, preference loss.** Both seeds agree on acceptance:
    - downstream acceptance +0.058 [+0.027, +0.091] (seed 42) and +0.053 [+0.021, +0.085] (seed 43);
    - a′ +0.007 / +0.017, CIs include 0;
    - checker ≈ 0;
    - complete −0.05, the epoch-2 over-training also seen in 013.
  - **(c) OptiScene-style synthetic-perturbation DPO2** (GT chosen vs perturbed GT) is worse than its start:
    - a′ −0.055 [−0.091, −0.020];
    - checker −0.063 [−0.102, −0.026].
  - **(d) vs (c), both seeds:** a′ +0.063 / +0.082 and checker +0.068 / +0.083.
  - **Reading.** None of the pre-fixed interpretation cases applies exactly (each requires both a′ and acceptance CIs above 0), so this is reported as measured:
    - pairing the model's own failure against its own repair adds admission gains beyond the repaired data itself;
    - synthetic negatives hurt.

## Why P trails (research memo + skeptical review, `solution-research-20261014b/`; EXPLORATORY)

**Corrected comparison**

- Fair single-shot checker success on the size-fixed pilot: P 0.513, O2 raw (one attempt) 0.793 (its completion rate is 0.80), L 0.947. O2's 0.920 includes up to 3 harness attempts.
- With S1 applied to both: P + S1 0.827 vs O2-raw + S1 0.797, a difference of +0.030 [−0.020, +0.081], i.e. parity.
- Gaps that remain:
  - vs O2-harness + S1: −0.097;
  - vs L + S1: −0.150.

**Verified about P**

- Training targets are size-consistent: P targets are GT poses repaired under the prior sizes and admitted only if they pass.
- Survival is a constant ≈ 0.887 per object.
- 96 % of co-located same-category colliding pairs are exact copies of position and yaw.
- P copies the preceding record's coordinates. A parent listed just before its child makes a collision more likely.

**Verified about reasoning**

- Reasoning length does not predict success: L uses about 311 reasoning tokens per room, O2 15.6k, O7 7.1k, O6 5.2k, O4 3.3k.

**Missed cause**

- The matched J/P design dropped 19,135 train rooms that had a valid P target, because J's GT target failed.
- Test rooms whose GT fails (27.7 %) are where P is weakest: 0.238 vs 0.460.

## Size-fixed regeneration, PILOT (amendment 016 §2, 300 pilot rooms; L and O2 only so far; EXPLORATORY)

Every system gets P's input (train-prior sizes, enforced); rates over all 300 requests, harness variants.

| System | complete | checker raw → S1 | admission strict raw → S1 | admission required={} raw → S1 |
|---|---|---|---|---|
| P (rpo-dpo1) | 1.000 | 0.513 → 0.827 | 0.367 → 0.490 | 0.720 → 0.803 |
| L size-fixed | 1.000 | 0.947 → 0.977 | 0.677 → 0.723 | 0.787 → 0.850 |
| O2 size-fixed | 0.930 | 0.920 → 0.923 | 0.640 → 0.690 | 0.707 → 0.767 |
| L size-free (own sizes) | 1.000 | 0.933 → 0.960 | 0.690 → 0.727 | 0.797 → 0.863 |
| O2 size-free (own sizes) | 0.970 | 0.953 → 0.967 | 0.713 → 0.750 | 0.817 → 0.873 |

- **The size confound does NOT explain the agents' lead.** Given realistic sizes, gpt-6.1-sol and O2 re-plan and still pass the checker at 0.92–0.95, against P's 0.51. Their size-free scores barely change (L) or fall a little (O2 checker −0.03, admission −0.07 to −0.11).
- **The substitution result was misleading as a ranking.** "Under prior sizes P ≥ L" held only for layouts planned around small furniture. Read as a ranking it was wrong, and it is withdrawn as ranking evidence. It still shows that self-chosen sizes are undersized and would yield miniature furniture in RoomGenBench.
- **Where P is closest.** Only on admission with required = {} after S1: P 0.803 vs O2-fixed 0.767 vs L-fixed 0.850. On the checker, P + S1 (0.827) stays well below both (0.92–0.98).
- **Unauthorised size changes:** L returned no sizes in 300 / 300 answers, and one object's z was recomputed.
- **All five agents, PILOT (`results/compare-016-pilot-v1`, `agents_016 compare`).** Checker success with fixed sizes, raw → S1:

  | System | Checker, raw → S1 |
  |---|---|
  | L | 0.947 → 0.977 |
  | O2 | 0.920 → 0.923 |
  | O6 | 0.760 → 0.880 |
  | O7 | 0.697 → 0.787 |
  | O4 | 0.623 → 0.837 |
  | P | 0.513 → 0.827 |

  - **Ranking stability.** By strict admission, Kendall τ-b between the size-free and size-fixed rankings is +0.73 [+0.47, +0.87] for harness raw and +0.87 [+0.60, +0.97] for raw raw. The ranking barely changes when sizes are fixed.
  - **Pre-specified flips.** P − O2 under fixed sizes, harness, is −0.273 [−0.329, −0.219] on admission raw and −0.097 [−0.133, −0.059] on the checker after S1. P − L under fixed sizes is −0.310 [−0.373, −0.247].
  - **One exception.** On admission with required = {} after S1, P (0.803) ranks second, behind L (0.850) and ahead of O2 (0.767), O6, O4 and O7.

## Mesh-level collisions on rendered rooms (amendment 019; 1,005 paper-strata rooms; EXPLORATORY)

Every object's real reference mesh is placed into its bbox exactly as rendered. The primary threshold is τ = 500 cm³ of voxel interpenetration.

**Collision-free rooms (τ = 500)**

| System | Collision-free rooms | 95 % CI |
|---|---|---|
| GT | 0.797 | [0.761, 0.832] |
| P | 0.548 | — |
| J | 0.422 | — |
| L | 0.979 | — |
| O2 | 0.976 | (998 rooms) |
| O6 | 0.896 | — |
| O7 | 0.862 | — |
| O4 | 0.835 | — |

The agents' rows use their own, smaller sizes (size-free runs), and that confound applies here too.

**What box-level "collisions" mean in GT (the user's point that real furniture does not fill its box)**

- Pairs of GT's A4 box pairs that also collide at mesh level:

  | Pair class | Mesh-collision share | Note |
  |---|---|---|
  | tucked (chair under table, nightstand into the bed's box) | 31 % | median interpenetration 0 cm³ |
  | rotated-box artefacts | 0 % | — |
  | small box overlaps | 9 % | — |
  | genuine (checker main/severe) | 63 % | — |
  | contained | 70 % | — |

- Overall, only 40 % of GT's A4 pairs are real mesh collisions.

**P's collisions are real**
- 86.5 % of P's checker main/severe pairs are confirmed at mesh level, against 62 % for GT.
- The frozen checker misses about a third of mesh collisions, mostly through its tuck exemption.

**Flush contact (diagnostic, not pre-registered)**
- Some counted collisions are only a flush contact layer, e.g. 27 % of GT's colliding pairs.
- Collision-free rates computed without that layer: GT 0.848, P 0.604, L 0.995, O2 0.998.

## Size-only LLM sizers (amendment 020 table A; 1,877 E2 ∪ E3 test rooms; EXPLORATORY; `results/sizes-020/`)

| Size source | Size log MAE vs GT | Footprint bias B |
|---|---|---|
| train prior (P's input) | 0.267 | −0.004 |
| gpt-6.1-sol size-only (L-size; 2,010 calls) | 0.327 | −0.198 |
| O2 size-only | 0.408 | −0.280 |
| L-joint (cohort rooms only) | 0.262 | −0.182 |

- **Paired difference, L-size − prior:** log MAE +0.060 [+0.050, +0.069].
- **LLMs undersize even when asked only for real-world dimensions** (no fitting instruction). The agents' undersizing is therefore partly a size-knowledge bias, not only shrinking to fit.
- **The prior is better in most families.** L-size beats it only in sparse non-paper families: storage_closet −0.091 and public_commercial −0.077 log MAE.
- **Consequence for amendment 022:** the delivered system keeps the train prior. A sizer must beat the prior on a dev measurement before it can replace it; table A, measured on test rooms, does not select it.

## Size confound: size substitution (amendment 016 §1, addendum A3; EXPLORATORY; `results/scale-curve-016/`)

Each system keeps its own positions, yaw and support; only the sizes are replaced (x/y/z, primary per addendum C3; supported children follow their parent). Shown: downstream acceptance on the 1,454-room cohort, raw outputs, with house-cluster CIs (2,000 draws).

| System | own sizes | train-prior sizes | GT sizes | B |
|---|---|---|---|---|
| GT layout | 0.448 | 0.270 [0.246, 0.294] | 0.448 | 0 |
| P (delivered) | 0.354 | 0.354 (= own) | 0.176 [0.154, 0.199] | −0.022 |
| J | 0.234 | 0.226 | 0.138 | −0.048 |
| L gpt-6.1-sol | 0.728 | 0.339 [0.315, 0.363] | 0.206 | −0.175 |
| O2 harness | 0.723 | 0.203 [0.183, 0.224] | 0.129 | −0.311 |
| O4 | 0.479 | 0.151 | 0.112 | −0.296 |
| O6 | 0.536 | 0.107 | 0.087 | −0.346 |
| O7 harness | 0.410 | 0.156 | 0.104 | −0.323 |

- **Agents' self-chosen sizes are undersized** (superseded as ranking evidence by the size-fixed pilot above). With realistic sizes the agents' acceptance falls by 0.39–0.52 (L, O2) and 0.32–0.45 (O4, O6, O7). The GT layout falls by only 0.18 under the prior. Under the same prior sizes, P (0.354) is at least as high as L (0.339) and above every open model (0.11–0.20). The checker shows the same pattern: own → prior is L 0.935 → 0.550, O2 0.928 → 0.382, P 0.519, GT 0.725 → 0.433.
- **Caveat.** Substitution keeps a layout that was planned for the agent's own (smaller) sizes, so it measures how much of the pass rate rests on those sizes, not how well the agent would place furniture of the given size. The fair placement test is the size-fixed regeneration (§2, running).
- **Scale curves are sensitivity only, as the protocol says.** GT has the steepest checker curve (slope −0.69 over s 0.80–1.00). Agents are flat below s = 1 and drop sharply above it: they already sit at the knee. Steepness is therefore not evidence of gaming.
- **The downstream rules are strict even for GT** (rule table, all test). GT fails:
  - A4 AABB collisions (incl. tucked chairs) in 51 % of rooms;
  - R2 1 mm envelope in 37 %;
  - A3 unreachable required objects in 33 %;
  - A2 door keep-out in 32 %.

  This is why GT downstream acceptance is 0.295; the metric definition drives the absolute level.

## Admission rule variants (amendment 017; cohort 1,454 rooms, from the S1 rows and GT recomputed with downstream.accept)

Strict = our port (every category required, every door an entry); required = {} = no per-class reachability/collision (upper bound of the real infinite-task-world gate, whose required set is the task's named classes); one door changes almost nothing.

| System | strict raw → S1 | required = {} raw → S1 | checker raw → S1 | B |
|---|---|---|---|---|
| GT | 0.448 | 0.758 | 0.725 | 0 |
| P | 0.354 → 0.472 | 0.718 → 0.798 | 0.519 → 0.826 | −0.022 |
| J | 0.234 → 0.372 | 0.688 → 0.786 | 0.431 → 0.797 | −0.048 |
| L | 0.728 → 0.754 | 0.837 → 0.888 | 0.935 → 0.959 | −0.175 |
| O2 harness | 0.723 → 0.752 | 0.812 → 0.865 | 0.928 → 0.947 | −0.311 |
| O6 | 0.536 → 0.633 | 0.633 → 0.783 | 0.701 → 0.856 | −0.346 |
| O4 | 0.479 → 0.597 | 0.620 → 0.765 | 0.687 → 0.862 | −0.296 |
| O7 harness | 0.410 → 0.500 | 0.533 → 0.654 | 0.592 → 0.741 | −0.323 |
| O3 harness | 0.159 → 0.223 | 0.257 → 0.325 | 0.241 → 0.350 | −0.450 |

- Under the strict port even GT passes 0.448 here (0.295 on all test); the all-required assumption causes most GT refusals (all test: 0.295 → 0.605 with required = {}).
- With required = {}, P + S1 (0.798) is above GT (0.758) and above O6/O4/O7 + S1; L and O2 stay higher but with 18–31 % smaller footprints (see size substitution).

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
- Does the agents' lead survive fixed sizes (amendment 016 §1–2), and does on-policy DPO2 add beyond its repaired data (amendment 013, 016 §3)?
