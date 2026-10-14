# Amendment 013 — on-policy DPO2 for arm P on top of rpo-dpo1 (final model = SFT + DPO1 + DPO2), selected on dev

Locked 2026-10-14 ~15:45 UTC, before any amendment-013 sample, pair, training step or dev score exists. Requested by the user: the delivered model should be SFT + two-stage DPO with metrics aligned to the paper. Tooling: `fastfill/comparison/onpolicy_dpo.py`, `isambard_013.sbatch`, `submit_013.sh`, `fastfill/v1_1/train_dpo.py --init_adapter` (commits cf518a24b and ad93a2c51, independently reviewed).

**Disclosures.** This amendment was designed after viewing:
- the amendment-009/010 test results;
- the amendment-011 merged-weight check (the BF16 merge of rpo-dpo1 loses its dev gain);
- the amendment-012 dev best-of-N curves.

Amendment-012 dev samples are also used once, for a pre-sampling yield check of the pair rule (no training on dev rooms).

## 1. Why a new DPO2

The amendment-009/010 DPO2 stages used synthetic position perturbations of GT layouts as rejected samples. They either collapsed completeness or did not improve dev a′. The analysis of the official OptiScene data (`optiscene-dpo-data-20261014/`) showed those pairs are too easy and are not the model's own failures; the model's real failure is co-located duplicates and gross collisions in many-object rooms. Amendment 011 §5 showed that the BF16 merge of a DPO stage erases its effect, so a next stage must not train on that merge.

## 2. Design (arm P only; J stays SFT)

**Training rooms**
- 3,000 train rooms: the SFT common length-admitted train pool minus the 2,200 DPO1 positives.
- Stratified by (object-count bin, source), ranked by `sha256('op-dpo2-v1:' + uid)`.
- A room or house group that occurs in dev or test makes the build fail.

**Samples**
- rpo-dpo1's exact policy run in vLLM (BF16 LoRA; amendment 012 measured that this engine is not identical to the HF exact policy).
- k = 0 greedy and k = 1..8 sampled at T 0.7, top_p 0.95.

**Pairs** (`PAIR_RULE`)
- Every sample is scored with the frozen checker and the downstream acceptance port.
- **Rejected:** complete samples that fail either check. Priority goes to co-located same-category duplicates, then main collisions, then other failures.
- **Chosen:** the rejected sample's own S1 repair. It must have S1 status ok and moved objects only; it must be rebuilt in the model's output format and re-scored, and still pass. Otherwise chosen is the best passing sample of the room; failing that, there is no pair (never the GT).
- Pairs whose chosen is not strictly better are dropped. At most 4 pairs per room. P outputs no sizes, so sizes cannot change.
- **Floors:** ≥ 1,500 pairs and ≥ 600 rooms, otherwise nothing is trained. The yield gate projects this from 1,000 amendment-012 dev rooms before any sampling.

**Training**
- Starts from rpo-dpo1's evaluated policy, not from its merge:
  - training base = `dpo-001/placement/sft-merged` (BF16 SFT merge);
  - the trainable LoRA is initialised from `dpo-003/placement/dpo1/final` (FP32);
  - the DPO reference is a frozen copy of that adapter.
- Recipe: lr 5e-6, 2 epochs, β 0.1, RPO α 1.0, LoRA r32/α16, effective batch 4; a checkpoint is saved every epoch.
- Each checkpoint is again "training base + one FP32 adapter" (the delivered form).

## 3. Selection on dev (`RULE`; dev-select-v1, 600 rooms, HF exact policy, same plan as amendment 010)

P = rpo-dpo1 unless an on-policy DPO2 checkpoint c ∈ {op-dpo2-e1, op-dpo2-e2} is eligible against R = rpo-dpo1. Bounds are inclusive. c is eligible when:
- complete(c) ≥ complete(R) − 0.005;
- a′(c) ≥ a′(R) + 0.01;
- downstream acceptance(c) ≥ acceptance(R) + 0.01, over all 600 rooms, where incomplete counts as not accepted;
- a′(c) − a′(R) ≥ 0 in both dev halves: 300 OptiScene_holodeck + IL3D_3dfront rooms, and 300 other rooms.

Adopted: the eligible c with the highest a′; within 0.005, the earlier epoch. If none is eligible, P stays rpo-dpo1 and nothing runs on test.

## 4. Test (only if adopted)

The adopted checkpoint runs once on all 8,647 test rooms (exact policy; infer-v2-006 → geometry-v6). The amendment-005 rule (J-SFT vs P-op) is applied as in amendment 010 and becomes the binding delivery decision; the delivered model is then SFT + DPO1 + DPO2 (adapter form). Paired P-op − P-rpo-dpo1 contrasts with house-cluster CIs are reported (EXPLORATORY). A merged-weight check like amendment 011 §5 decides whether a merged copy may be listed.

## 5. Budget

GPU cap 150 GPU-h: sampling ≤ 30, training ~15–20, dev ~4, test ~60–80 only if adopted.

## Addendum A (locked 2026-10-14 ~21:20 UTC, after the §3 dev selection, before any amendment-013 test output exists)

**Dev result** (`RUN/dev-eval-013/selection.json`): op-dpo2-e1 is adopted.

| Candidate | Checker success | a′ | Downstream acceptance | Complete |
|---|---|---|---|---|
| rpo-dpo1 | 0.470 | 0.435 | 0.252 | 0.998 |
| op-dpo2-e1 | 0.515 | 0.488 | 0.332 | 0.995 |
| op-dpo2-e2 | 0.475 | 0.443 | 0.328 | 0.942 (fails the complete guard) |

op-dpo2-e1's a′ gain is positive in both dev halves.

**Test step.** The §4 test run is **deferred to amendment 021 §4**, which was locked later. That rule selects the delivered model across amendments 013, 018 and 021 on one engine, confirms it on a disjoint 3,000-room dev slice, and runs only the confirmed winner once on test. op-dpo2-e1 enters 021 §4 as 013's adopted candidate. If it wins and is confirmed, it runs on test exactly once there; no separate 013 test run is made.
