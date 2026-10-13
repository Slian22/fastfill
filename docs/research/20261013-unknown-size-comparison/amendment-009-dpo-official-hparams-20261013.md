# Amendment 009 — DPO hyper-parameters from the official OptiScene code; exact-policy evaluation

Locked 2026-10-13 ~21:50 UTC, before any DPO job exists and before any J/P test output or agent score has been looked at.

## Why

Amendment 008 copied the paper text (§4.1: LoRA r16/α32, lr 5e-7, 5 epochs). The code review (workflow wf_97e0cb14-a4d) showed that at lr 5e-7 the LoRA update is mostly below half a BF16 ULP of the base weights, so the per-stage BF16 merge would erase most of it (simulated: |B|≈1e-5–5e-5 leaves almost no change). The official OptiScene repository README (provided by the user) trains DPO with different values, and that is the code that produced their model.

## Changes to amendment 008

- DPO-stage1 and DPO-stage2 hyper-parameters follow the official README: **lr 5e-6, 10 epochs, LoRA r 32, α 16**, dropout 0.05, all projections (q,k,v,o,gate,up,down), **effective batch 4** as in the README command (per-device 1 × accumulation 4); run as per-device 1 × 4 GPUs (DDP) × accumulation 1, which gives the same effective batch and number of optimizer steps four times faster, β 0.1 (TRL default), cosine schedule as in the existing train_dpo, BF16, gradient checkpointing. Max length stays 8192 (our prompts are longer than the README's 3200; same admission as SFT).
- As in the official pipeline, each stage trains on the BF16 merge of the previous stage and is merged afterwards.
- **Evaluation of each DPO stage uses the exact trained policy**: the stage's LoRA adapter applied unmerged on its BF16 training base (the merged previous stage). The merged checkpoint is still produced (it is the next stage's base and the delivered artifact) and the fraction of weights changed by each merge is reported.
- Positives require both the J target (GT) and the P target to pass the frozen geometry gate (P targets are teacher-qualified, so this rarely binds); counts recorded.
- The SFT row of the stage-ablation table comes from infer-v2-002 (runtime GPU merge); this is stated next to the table.
- Results summaries carry the stage label (sft / dpo1 / dpo2).

## Budget

Expected ≈ 2,200 pairs × 10 epochs ≈ 22,000 pair passes per arm-stage ≈ 2–3 h on 4 GPUs (≈ 9–12 GPU-h) plus inference; 4 arm-stages ≈ 45 GPU-h; GPU cap 600 unchanged.
