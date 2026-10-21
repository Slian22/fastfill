# V1.1 fresh training and recovery audit — 2026-10-11

This is a fresh full-file audit of `fastfill/v1_1/train_dpo.py`, `training_guard.py`, `isambard_v1_1_clean_dpo.sbatch`, `fastfill/train.py`, `fastfill/merge_lora.py`, and four related test files. The coordinator reported graph indexing unavailable (`Transport closed`), so the audit read the known files directly. `train-coverage.json` records exact reviewed hashes and full-read line counts. No remote files, allocations, or jobs were changed by this agent.

## Findings and implemented changes

1. **P2 — checkpoint publication could delete the only recoverable state.** Deployed snapshot `3e3eef72c515` deletes `ckpt_latest` before publishing its replacement (original `train_dpo.py` lines 147–149). An injected `os.replace` failure left no latest checkpoint; the old resume condition then silently started from step zero when either state or adapter was missing. This affects unattended recovery, even though no actual loss on the running server was observed. The local repair keeps `ckpt_latest.previous`, rolls back a failed publication, recovers a process killed between renames, and rejects a partial checkpoint or an interrupted first save instead of silently restarting. Existing checkpoint serialization remains compatible. Rank zero performs recovery before the distributed barrier. New helpers start at lines 131, 145, and 197.

2. **P2 — resume position was not validated against the exact run.** A readable state with an out-of-range epoch or step could skip training and publish `final`; the old code used permissive `.get` defaults. `validate_resume_position` (line 164) now checks integer epoch/iteration/step, optimizer boundaries, full and partial accumulation windows, and the expected step derived from the loader and epochs. A real tiny Qwen3 CPU run with a tampered step of 999 now fails before completing a resume. Unreadable optimizer state also fails explicitly.

3. **P2 — adapter structural validity did not prove training completion.** Existing `completed_adapter` accepts an adapter config object plus a structurally valid safetensors file; a scratch fixture with only one LoRA tensor and no completion evidence is accepted. The clean batch script uses this as its training skip condition. No claim is made that the actual running adapter has been substituted. The new trainer writes `final/completion.json` inside the temporary final directory before publication, binding final adapter/tokenizer bytes, actual admitted pairs, loader batches, world size, completed epoch/iteration/step, expected total steps, and the enclosing training guard hash. It refuses publication when actual steps do not equal expected total steps. The automation owner implements the authoritative verification path and a read-only legacy checkpoint proof; this agent does not modify `training_guard.py`.

4. **P2 — finite loss alone did not prevent a non-finite update.** A real tiny Qwen3 training test injected an infinite gradient after a finite forward loss. The deployed code's default `clip_grad_norm_` converted that gradient into NaN and reached optimizer update. The repair sets `error_if_nonfinite=True` before the optimizer update. The regression proves the optimizer is not called and no checkpoint or final adapter is published. Finite normal gradients use the same clipping and objective.

All four findings apply to defensive correctness/recovery in the deployed training code. They are not evidence that its current finite run is numerically corrupt or that the completed SFT is invalid. The frozen running source was not hot-edited; the coordinator must choose and document any continuation migration.

## Mathematical and implementation checks

- SFT masks all prompt labels and supervises the assistant answer plus end token, including numeric JSON. Overlong sequences are discarded whole. The current actual tokenizer's EOS was previously verified as `<|im_end|>`, matching SFT and DPO; no tokenizer mismatch was established here.
- DPO uses the correct sign and standard summed completion log-ratio objective: `-logsigmoid(beta * ((policy_chosen-reference_chosen) - (policy_rejected-reference_rejected)))`. A new scratch test checks its gradient direction and the equal-policy loss `log(2)`.
- `sequence_logps` applies the causal shift once, excludes prompt and padding labels using `-100`, casts logits to FP32 before cross-entropy, and sums over supervised completion tokens. A direct numerical test verifies the exact target log probability and zero gradients at excluded token positions.
- The reference shares the frozen merged base while disabling LoRA, under `no_grad`. LoRA parameters alone are optimized. The current Qwen base uses zero base attention dropout; LoRA dropout 0.05 applies to policy only. Keeping that stochastic recipe is deliberate; this audit does not silently change it to a deterministic policy.
- Current authoritative recipe is four ranks and batch size one. DistributedSampler gives equal per-rank lengths; tail accumulation uses actual microbatch count. A fresh real two-rank CPU Gloo check with tiny Qwen3, LoRA, non-reentrant gradient checkpointing, and a three-microbatch tail matched the equivalent combined-batch gradient to **4.656612873077393e-10 on both ranks**. The probe disables dropout to permit exact gradient comparison and does not prove bitwise GPU training equivalence.
- Merge retains the adapter tokenizer and nearest run-manifest family, with compatibility rewrites for the deployed Transformers versions. Existing real tiny-model BF16 merge/reload validation passed. Merge receipts bind base, adapter, merge source, and output bytes; the added automation lineage path verifies completed-training evidence before treating a merged model as authoritative.

## Explicit remaining limits

- RNG states are still not saved. Four-rank sampler order is deterministic for the recorded seed/epoch, but resumed dropout is not bitwise identical to uninterrupted dropout. This was already disclosed in the preregistration and is not changed by the repair.
- Direct single-process CLI training uses `shuffle=True` rather than an epoch-addressable sampler. On a mid-run resume it can replay an epoch-zero ordering in a later epoch. A bounded DataLoader fixture reproduces this. The submitted four-rank training chain does not use this path; report it as a direct-CLI limitation, not a defect in current distributed ordering.
- Generic `bs > 1` can give a smaller final minibatch the same microbatch weight as a full minibatch. The submitted recipe fixes `bs=1`, so this does not affect its updates.
- DPO progress loss/reward logs are rank zero's last microbatch, not a distributed aggregate or validation metric. They cannot establish model quality.
- A checksum/receipt establishes artifact lineage relative to trusted producers; it cannot prove warm-start data exclusion, semantic completeness of all preference labels, physical scene validity, or scientific improvement. Those claims still require the full independent evaluation.
- Checkpoint publication guards process interruption and rename failure; this audit does not claim power-loss durability of every filesystem write.

## Verification

- TDD RED: the first 23 recovery contracts failed on the old implementation, including the injected loss of the only checkpoint. Two completion contracts then failed before receipt implementation. The non-finite-gradient test separately failed on reaching optimizer update before the one-line clipping repair.
- GREEN: **32 new checkpoint/completion/training tests** passed. Combined with existing preference-audit and merge tests: **67 passed**.
- Independent coverage invocation: **52 tests passed**; standard-library executable-line tracing covered **308/320 lines (96.25%)** of current `train_dpo.py`, and **73/74 lines (98.65%)** in the checkpoint/completion helper functions. This is CPU line coverage, not branch coverage or comprehensive GPU hardware coverage. One initial stdin-driven coverage harness failed only because macOS spawn could not reopen `<stdin>`; rerunning from the saved guarded script resolved it.
- Real two-rank tail-gradient recheck: both ranks returned maximum difference **4.656612873077393e-10**.
- `git diff --check` passed for the agent-owned paths.

Artifacts: `test_train_adversarial.py`, `ddp_tail_recheck.py`, `trace_training_tests.py`, `train-test-coverage.json`, and `train-coverage.json` in this directory. The adversarial scratch file intentionally records the old behaviors and is not part of the post-repair production test suite.

Agent-owned production source SHA256 after repairs: `train_dpo.py` = `34884c22c83c8a8a67ef9f9bb5edec558f6f3939a6f13bfbfef7d4c217b58a41`.
