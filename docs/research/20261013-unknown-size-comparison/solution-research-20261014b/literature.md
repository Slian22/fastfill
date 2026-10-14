# Why P (rpo-dpo1) fails the checker where O2 passes on the same input

## Bottom line

**Verified:** P is trained on checker-valid targets and has seen plenty of dense rooms, so bad labels and missing dense data are ruled out. It still fails at a roughly constant rate per object (success ≈ 0.887^n). The usual failures are gross collisions, and in dense rooms many of these are exact copies of an earlier object of the same category.

O2 almost never produces a colliding layout once it finishes an answer (238 of 240). Every O2 failure on the pilot is an answer that never finished (57 of 60 raw failures hit the token cap; the other 3 are unparseable). It spends about 10.9k thinking tokens per answer.

**Inferred:** P has no way to keep track of where it has already put things, and nothing checks its layout while it is being generated. O2 does that work inside its long reasoning. The training labels are not the problem.

## Ranked root causes

### 1. No occupancy tracking or checking during decoding (strongest)

**Verified:**
- P decodes greedily with thinking off (`infer.py:174,303`). There is no reasoning step and no check inside the loop.
- **Constant per-object failure rate.** Fitted on all 8,647 test rooms: log P(success) = −0.117 − 0.1196·n, i.e. survival ≈ 0.887 per object. Observed vs fitted: n=1 0.903; n=5 0.515 (0.489); n=10 0.242 (0.269); n=20 0.089 (0.081); n≥30 0.0.
- **Simple rooms already fail.**
  - Rooms with 2 objects collide in 100 of 710 cases.
  - Rooms with 1 object fail in 68 of 701 cases (41 block a fixed item such as a door, 25 are out of bounds).
- **Pilot, why rooms fail** (300 rooms, raw checker):

| | Complete answers | Pass rate among complete | Failure type |
|---|---|---|---|
| P | 300 | 0.513 | 89 % of failing rooms have main collisions, 60 % severe |
| O2 size-fixed, raw | 240 | 0.992 | all 60 failures are incomplete (57 token cap, 3 unparseable) |
| O2 size-fixed, harness | 279 | 0.989 | all 21 failures are token-cap truncations |
| L (gpt-6.1-sol) size-fixed | 300 | 0.947 | — |

- **Pilot by object count** (P / O2-raw / O2-harness / L): 1–5: 0.739 / 0.901 / 0.986 / 0.958; 6–10: 0.40 / 0.80 / 0.926 / 0.958; 11–20: 0.262 / 0.667 / 0.833 / 0.929; 21–50: 0.0 / 0.235 / 0.588 / 0.882.
  - The pilot is 279 / 17 / 4 rooms across ≤20 / 21–50 / >50 objects (`a016-pilot-v1.json`). So the 0.51 vs 0.92 gap comes mostly from rooms with 20 or fewer objects.
- **Overlap:** on the 279 rooms all three systems completed, mean OOR is P 0.112, O2 0.017, L 0.029.
- **Errors do not build up along the output sequence.** Objects involved in collisions are spread evenly over the first, middle and last third of the list (11,147 / 11,243 / 11,822). Collision pairs are as far apart in the list as random pairs would be (e.g. 21–50 objects: mean index gap 12.24 vs 12.4 expected). Out-of-bounds objects rise only mildly (733 → 1,034).
- **Thinking tokens per answer** (1,454-room cohort, size-free receipts):

| System | Reasoning tokens / answer | Answer tokens / answer |
|---|---|---|
| O2 | 10,942 (410 cap hits) | 314 |
| O4 | 2,625 | — |
| O6 | 3,504 | — |
| O7 | 5,492 | — |
| L | 237 | — |
| L, size-fixed pilot | about 311 (93,222 / 300) | — |

  O2 thinks about 35 times longer than its answer. Its token cap is 1024 + 128·n + 16,384 (`llm_open.py:65,146`).

**Inference:**
- For O2, the advantage is extra compute at test time: its reasoning presumably does explicit coordinate bookkeeping. This is not checked; the traces are not available locally.
- L shows that a frontier model can emit valid layouts with almost no visible reasoning. So test-time compute is enough to close the gap but is not required for every model.
- Collisions spread evenly over the sequence suggest P is not "forgetting" earlier placements. It places each object at a likely position without reference to what is already there.

### 2. Same-category instances look identical, and P copies their positions

**Verified:**
- Objects carry opaque IDs, a category and a prior size. Same-category instances get identical tokens.
- They appear in HMAC order (`contract.py:166-171`), and the output must follow that order (`contract.py:36,285`).
- Same-category instances are split up in the list in 67 % of 6–10-object test rooms, 95 % of 11–20 and 100 % of 21+.
- **96.1 % of co-located same-category colliding pairs are exact copies of position and yaw** (62,452 of 64,987).
- Share of all collision pairs that are such copies: 11 % (6–10), 22 % (11–20), 39 % (21–50), 48 % (51+). They appear in 698 of 1,080 rooms with 21–50 objects and 320 of 337 with 51+.
- O2 and L produce zero co-located pairs on the pilot.

**Caveat (verified):** removing only the duplicates would flip just 70 rooms (0.8 %). Duplicates almost always come with other collisions. They are a symptom of cause 1, not a separately fixable bug.

### 3. P does not adapt when the room is tight for the sizes it is given

**Verified:**
- P's success depends on whether the GT positions, with the prior sizes swapped in, would pass:
  - 1–5 objects: 0.821 vs 0.484
  - 6–10 objects: 0.497 vs 0.239
  - 11–20 objects: 0.238 vs 0.099
- On the pilot, L does not care (0.948 vs 0.946) and O2-harness only a little (0.948 vs 0.898). P drops from 0.649 to 0.404.
- By fill ratio (prior footprint / room area), P falls from 0.57–0.60 to 0.33 at fill 0.5–0.8. L stays at 0.95–0.96.
- In dense rooms the prior oversizes: footprint bias +0.16 (21–50) and +0.27 (51+). GT positions with prior sizes pass only 0.201 and 0.106, against 0.77 and 0.79 with GT sizes (`scale-curve-016/summary.csv`).
- P is not copying GT. Median object-to-GT distance as a fraction of the room diagonal is P 0.289, O2 0.31, L 0.276.

### 4. The preference stage never targeted the failure

**Verified:**
- DPO1 positives come only from OptiScene_holodeck (1,100) and IL3D_3dfront (1,100) (`dpo_stages.py:69`). In those train sources, an estimated 93–94 % of rooms have 10 or fewer objects.
- Negatives are the SFT model's own samples, with no requirement that they fail; their checker result is only recorded (`dpo_stages.py:532-543`).
- DPO2 used synthetic perturbations.
- Result: on test, rpo-dpo1 is only +0.008 over SFT (findings.md).

**Inference:** the preference stages could not have taught occupancy tracking.

### 5. Model size (8B, LoRA r32): plausible but not shown

**Verified:**
- SFT was 1 epoch with LoRA r32/α64, lr 5e-5, effective batch 4 (`run_receipt.json`).
- No dev loss was computed. Only early training loss exists locally (about 0.36 at roughly 6 % of the epoch; `agency-live-training-qa-20261013.md:99-110`).
- There are no generations on train rooms locally.
- Best-of-8 vs greedy on dev (checker, raw): 1–10 objects 0.57 → 0.81; 11–30 objects 0.116 → 0.28; 31–50 objects 0 → 0.039; 51+ objects 0 → 0. Valid layouts exist in P's sampling distribution for small rooms, but dense rooms compound the per-object failure rate.
- Untuned Qwen3-8B with thinking (O1b) scores 0.110, but that failure is mostly the `on` support field, so it says nothing about layout skill.

**Not determinable locally:** whether P is underfit or overfit.

### Hypotheses that were checked and do not hold

**"P is trained to reproduce positions that collide under the sizes it is told" — refuted for SFT.**

- **Verified from code:**
  - A P target is GT poses repaired by `pose_targets.repair` under the out-of-fold prior sizes (`main_data.py:179-182`, `226-227`).
  - It is scored with the prior sizes (`main_data.py:142,154`).
  - A room is admitted only if both the repaired P target and GT pass (`main_data.py:183-196`).
- **Audit:** 58,796 of 98,923 P train targets differ from GT, i.e. were repaired (`evidence/full-audit-data-p-target-7368003-result.json`).
- **Proxy on 750 test rooms from training sources, 150 per object-count bin** (same code path):

| Objects | GT poses + prior pass | Repair succeeds | Room admitted | Objects moved | Median move of moved objects |
|---|---|---|---|---|---|
| 1–5 | 0.533 | 0.953 | 0.76 | 9–14 % | not given |
| 6–10 | 0.273 | 0.96 | 0.667 | 9–14 % | not given |
| 11–20 | 0.093 | 0.94 | 0.58 | 9–14 % | 0.165 m |
| 21–50 | 0.013 | 0.90 | 0.747 | 9–14 % | 0.26 m |
| 51+ | 0.0 | 0.66 | 0.52 | 9–14 % | 0.297 m |

  The 0.433 GT-with-prior figure from the substitution analysis is therefore real, but P is never trained on those unrepaired positions.

**Not enough dense training rooms — refuted.**
- Estimated from the train `lengths.json` (answer tokens = −33.6 + 34.82·n, residual SD 55): about 9,975 train rooms (10.1 %) have 21–50 objects and 3,273 (3.3 %) have 51+. Test has 12.6 % and 4.0 %.
- Only 169 train rooms were dropped for length, mostly MansionWorld and SAGE.

**Truncation or prompt length — not a cause for P.**
- P completes 300 of 300 pilot rooms and 99.8 % of test rooms (2.9 % incomplete in 51+).
- Prompts are at most 6,183 tokens and answers at most 5,861, against an 8,192 limit.
- Coordinates are absolute meters with 2 decimals in a room frame (boundary corner within 0.5 m of the origin in 100 % of test rooms). Yaw is in integer degrees.
- O2 receives P's system and user messages plus a JSON-only suffix (`llm_open.py` docstring; `llm_agent.py:78`). O2, not P, is the one that gets truncated.

**Scoring asymmetry — minor.** The agents' supported-object z is recomputed before scoring (`llm_agent.enforce_sizes`), while P is scored as produced. Support failures appear in only 4 % of P's failing pilot rooms (9 % on the full test).

## What this suggests (inference, untested)

- **Amendment 018 risk.** It distils only O2's final answers. That likely reproduces the outputs without the bookkeeping that produced them, so P's per-object failure rate may survive. Keeping O2's reasoning traces, or a compact occupancy scratchpad, attacks cause 1 directly.
- **Check at decode time, per object.** With roughly a 12 % per-object failure rate, resampling an object until it passes the checker should help where whole-room best-of-8 cannot.
- **Fix the identical-instance problem.** Emit parents first and keep same-category instances together, or add an instance index such as "chair 3/6".
- **On-policy RL with the checker as reward** (e.g. GRPO), focused on rooms with more than 10 objects.
- **Cheap check before deciding on a bigger model:** greedy-decode P-SFT on about 500 train rooms, stratified by object count, to tell underfitting from failure to generalise. Then consider Qwen3.8-27B as the student.

## Files

All in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/better/rootcause/`:
- Scripts: `pilot.py`, `causes.py`, `peek.py`, `targets.py`, `repr.py`, `order.py`, `gtprior.py`, `gtdist.py`, `seqpos.py`
- Outputs: `pilot_table.pkl`, `order.pkl`, `targets_150.pkl`, `targets_150.log`