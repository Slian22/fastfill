# Amendment 027 — end-to-end full test of the strict escalation trigger (local O2)

Locked 2026-10-15 (UTC), before any amendment-027 output exists. Follow-up work after the frozen delivery (git tag `fastfill-delivery-20261015`); it changes nothing that was delivered.

## Purpose

Amendment 026 addendum B adopted the strict trigger: escalate when the checker or strict admission fails. That adoption rests on an offline replay. This run measures the trigger end to end on all 8,647 test rooms:
- real escalation volume, failures and retries, time and GPU cost;
- quality;
- whether these match the replay prediction.

## Design

**System.** The amendment-022 pipeline with `--escalate-on strict`:
- the same 27B placer rows (replayed);
- the same S1;
- escalation to local O2 only (no GPT API, as in the 022 test scope);
- the best candidate ranked checker > strict admission (standalone) > required = {} > complete > fewer violations;
- the same fallback.

**Execution.**
- The O2 escalation is sharded as in 022 addendum B: N = 8 shards of 2 GPUs, the same shard rule and chunk rules, per-chunk persistence and resume.
- Output: `RUN/system-022/final/test-strict-sharded`.

**Reported (descriptive; no decision)**
1. Checker, strict admission, required = {}, OOR and size metrics per stage, overall and by object-count bin, real/synthetic and room family.
2. Escalated rooms, the share where O2's answer is used, O2 calls and rounds, O2 wall time and GPU-hours, fallbacks and empty outputs (must be 0).
3. Paired differences against the existing checker-trigger run (`test-sharded`), same rooms, house-cluster bootstrap with 2,000 draws and seed 42.
4. The replay prediction against the measurement. On the cohort rooms, compare with 026 scope B "O2 harness, ESC-STRICT":
   - predicted escalation 54.7 %;
   - predicted checker / strict admission / required = {}: 0.956 / 0.725 / 0.870.

## Budget

- No API calls.
- GPU ceiling 8 × 2 × 12 h = 192 GPU-h; expected well under 80 GPU-h.
