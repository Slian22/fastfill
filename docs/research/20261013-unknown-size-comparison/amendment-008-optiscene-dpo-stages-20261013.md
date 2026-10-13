# Amendment 008 — OptiScene training recipe: SFT → DPO-stage1 → DPO-stage2 for both arms; final model = DPO2

Locked 2026-10-13 ~21:20 UTC. No J/P test output has been looked at (J test shards started ~20:54 UTC; only the 2-room dev probe was read), no agent answer or score has been looked at. The user clarified that the training method is the OptiScene paper's (and our V1.1 implementation's) multi-stage recipe and that the deliverable is the final model plus the evaluations.

## Recipe (both arms J and P, matched, each starting from its own merged SFT model)

Following OptiScene §3.4 / §4.1 / App. C and reusing `fastfill/v1_1/build_dpo.py` and `train_dpo.py` logic, adapted to the unknown-size J/P input contract (no target fields in inputs):

- **Positives (shared by both arms, both stages):** 2,200 train rooms — 1,100 OptiScene_holodeck (3D-SynthPlace analogue) + 1,100 IL3D_3dfront (3D-Front) — from the common length-admitted train pool, whose GT layout passes the frozen geometry gate, selected by stable hash `sha256('dpo-positives-v1:' + uid)`. Shortfalls are filled from other GT-pass train rooms and recorded. This is a programmatic geometry-verified selection, **not** the paper's unanimous human approval; it is labelled as such.
- **DPO-stage1 (consensus alignment):** negatives are the arm's own SFT generations on the same inputs (sampled, v1.1 settings), kept only when they are schema-valid and differ from the positive; pairs verified as in v1.1.
- **DPO-stage2 (geometric plausibility):** negatives are the positive layout with targeted violations (out-of-boundary and object-overlap perturbations of positions only; sizes untouched), same seeds and perturbation schedule for both arms.
- **Hyper-parameters (paper §4.1):** LoRA r 16, α 32, dropout 0.05 on all projections, lr 5e-7 cosine, 5 epochs, β 0.1, on top of the merged previous stage; merge after each stage. Our SFT stage stays as already trained (r 32 / α 64, lr 5e-5, 1 epoch) and is reported as such.

## Evaluation

- Every stage of both arms (SFT, +DPO1, +DPO2) is evaluated on the same 8,647 test requests with the same shared inference plan (amendment 002) and the frozen geometry code: complete rate, geometry success, size IoU/log MAE, raw/rotated OOR, automatic acceptance — the stage-ablation table corresponding to paper Table 6 (human UR not measured).
- The final models (J-DPO2, P-DPO2) replace J0/P0 in the paper-style Tables 1/2 (OOR, FID, GPT judge) and in the agent benchmark (same cohort, same inputs). SFT-stage results stay reported.
- **Delivery:** the amendment 005 rule is applied to J-DPO2 vs P-DPO2; the delivered artifact is the merged DPO2 model of the chosen arm (plus the frozen size prior if P).
- The paper's 'reasoning' SFT variant is not part of this study (inputs/outputs are the J/P JSON contract).

## Budget

GPU cap raised to 600 GPU-hours.
