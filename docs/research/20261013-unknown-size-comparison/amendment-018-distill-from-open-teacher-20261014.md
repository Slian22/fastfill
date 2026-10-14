# Amendment 018 — distil arm P from the free open teacher O2 (SFT-D → DPO1-D → DPO2-D), selected on dev

Locked 2026-10-14 (UTC), before any amendment-018 sample, pair, training step or score exists. Chosen by the user after the amendment-016 §2 pilot. In that pilot, with P's sizes, O2 (Qwen3.8-27B, free open weights) passes the frozen checker on 0.920 of 300 rooms; gpt-6.1-sol passes on 0.947; P (rpo-dpo1) passes on 0.513. **Disclosure:** designed after viewing that pilot, the amendment-012 test application and the amendment-016 §1 results. Amendments 013 and 016 keep running unchanged; their outputs are reported as they are.

## 1. Teacher data (no API cost)

**Teacher**
- O2 = Qwen3.8-27B with exactly the amendment-007 engine and sampling settings.
- Input is P's (amendment-016 §2 `--arm placement`): the train-prior sizes, enforced as in addendum B2.
- Same harness. Greedy, one answer per room.

**Rooms**
- T-A: the 3,000 amendment-013 train rooms (`RUN/dpo-004/train-subset`; rpo-dpo1's 9 samples per room already exist).
- T-B: 5,000 further train rooms drawn from the SFT common length-admitted train pool, minus the 2,200 DPO1 positives, minus T-A.
  - **Allocation:** one third to each object-count bin (≤ 20, 21–50, > 50; if a bin runs short, the remainder goes to the next denser bin).
  - **Within a bin:** proportional to source, then ranked by `sha256('a018-distill:' + uid)`.
- A room or house group that occurs in dev or test makes the build fail.

**Filter**
- A teacher answer is a teacher target iff it is complete and passes the frozen checker (`geometry_success`, raw, no S1).
- It is rebuilt in P's training output format and re-scored, and must reproduce that verdict.

**Budget:** ≤ 35 GPU-h.

## 2. Training (arm P; all LoRA r32/α16 on the training base `dpo-001/placement/sft-merged`; effective batch 4; seed 42; every stage starts from the previous stage's FP32 adapter; delivered form = training base + one FP32 adapter)

1. **SFT-D** (`--sft_only`, NLL on the target only)
   - Data: every teacher target from T-A ∪ T-B.
   - Initialisation: from the rpo-dpo1 adapter.
   - Recipe: lr 1e-5, 2 epochs.
2. **DPO1-D** (sigmoid DPO, β 0.1, RPO α 1.0, lr 5e-6, 2 epochs)
   - Initialisation: from SFT-D; the reference is the frozen SFT-D policy.
   - Pairs, on T-A only:
     - chosen = the room's teacher target;
     - rejected = up to 2 of rpo-dpo1's own complete samples for the room that fail the checker or the strict admission, picked in amendment-013 tier order (co-located duplicates, main collisions, other, then k).
   - Rejected samples are off-policy by one stage (they come from rpo-dpo1, not SFT-D); this is stated in every report.
3. **DPO2-D** (amendment-013 recipe and `PAIR_RULE`, with the sampling policy = DPO1-D)
   - On-policy pairs on T-C: 3,000 further train rooms drawn as T-B (hash prefix `a018-dpo2:`, disjoint from T-A and T-B).
   - Each room: k = 0 greedy plus k = 1..8 sampled.
   - chosen = the rejected sample's own S1 repair, else the best passing sample; never the GT or the teacher.
   - The reference is the frozen DPO1-D policy.

## 3. Selection on dev (dev-select-v1, 600 rooms, HF exact policy, same plan as amendments 010/013)

**Primary metric (amendment 017):** frozen checker success over all 600 requests (incomplete = fail).
- Also reported: strict admission and the required = {} variant, raw and after S1; complete; OOR; and the size metrics. P's sizes are the prior, so size MAE is unchanged by construction.

**Candidates:** SFT-D, DPO1-D, DPO2-D (final checkpoint of each), against R = rpo-dpo1.

**Eligibility** (all four required):
- complete ≥ complete(R) − 0.005;
- checker success ≥ R + 0.02, with the house-cluster paired CI lower bound > 0;
- strict admission ≥ R − 0.01;
- checker gain ≥ 0 in both dev halves (as in amendment 013).

**Adopted model:** the eligible candidate with the highest checker success; the later stage wins ties within 0.005. If none is eligible, P stays rpo-dpo1.

**Delivered model:** if adopted, the delivered model is training base + that candidate's adapter. Its lineage is SFT → rpo-dpo1 → SFT-D → DPO1-D → DPO2-D; it is "SFT + two-stage DPO" with a distillation SFT in between.

## 4. Test (only if adopted)

**Runs**
- Exact policy on all 8,647 test rooms, then geometry and every amendment-012/016/017 table.
- The adopted model on the 1,454 cohort, compared with the amendment-016 §2 size-fixed agents (same P input) and with the size-free agents.

**Reporting:** paired contrasts against rpo-dpo1 and O2 (house-cluster CIs, EXPLORATORY).

**Merged copy:** a merged-weight check, as in amendment 011 §5, decides whether a merged copy may be listed.

## 5. Budget

**Cap: 150 GPU-h.**

| Item | GPU-h |
|---|---|
| teacher | ≤ 35 |
| training | ~15 |
| DPO2-D sampling | ~2 |
| dev | ~12 |
| test | ~70 (only if adopted) |
