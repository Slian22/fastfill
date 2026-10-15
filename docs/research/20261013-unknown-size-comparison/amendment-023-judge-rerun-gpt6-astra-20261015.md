# Amendment 023 — GPT-judge rerun with a judge that is not a compared system

Locked 2026-10-15 (UTC), before any gpt-6-astra judge output exists.

**Disclosure.** Designed after viewing the run-002 judge results (judge = gpt-6.1-sol; `results/gpt-judge-visual-002`: overall J 6.31, P 6.16, L 7.91, O6 6.83). The user asked to switch the judge to "GPT6-astra" with reasoning effort High. gpt-6.1-sol is also compared system L, so run 002 carries a self-preference conflict. This rerun removes the identity conflict but not the family one: gpt-6-astra is from the same vendor and generation as L. Both judges stay secondary metrics (amendment 017); neither decides anything.

## 1. What changes, what does not

- **Judge:** `gpt-6-astra`, the exact API id from the server's `/models` listing on 2026-10-15 ~02:30 UTC (the ids containing gpt-6 or astra were `gpt-6-astra`, `gpt-6-sol` and `gpt-6.1-sol`).
- **Reasoning:** `{"effort": "high"}`.
- **Output tokens:** `max_output_tokens` 32,768, because reasoning tokens count against it (run 002 used 4,096 without reasoning).
- **Timeout:** 600 s per case (run 002: 90 s).
- **Unchanged:** the same bundle, cases, renders, prompt, instructions and rubric as run 002. The bundle is `RUN/gpt-863099…/bundle`: 174 rooms × arms J, P, L, O6 = 696 selected, 692 runnable.
- **Code:** client `fastfill/comparison/server_api_judge.py`. It is the run-002 client (sha 8bfd7bc4…) plus the options `--model` and `--reasoning-effort`. With both unset, the payload is byte-identical (tested).
- **Runner:** `judge_batch.py --model gpt-6-astra --reasoning-effort high --workers 4`. Four one-case client subprocesses run at once, and records keep mapping order.
- **Accounting:** no retries, no replacements, no resume. Each failure is reported as missing.

## 2. Smoke, then the full run

1. **Smoke.** The client alone runs on the first 2 bundle cases (manifest with absolute paths, `--limit 2`).
   - Pass: both complete with valid scores, and the returned model equals `gpt-6-astra` exactly.
   - Otherwise stop and report. There is no remapping of model names.
2. **Full run.** One run on all 692 runnable cases.
   - Cap: 800 calls for the run, plus 2 for the smoke.
   - The 2 smoke cases are judged again in the full run (no reuse).

## 3. Reporting

- **Per-arm means for both judges.** Functionality, layout, aesthetics and overall, overall and by stratum, side by side.
- **J − P paired endpoint.** Same rule as run 002: rooms where both arms completed; 2,000 draws, seed 42, resampling rooms within stratum.
- **Inter-judge agreement.** Computed on cases both judges completed:
  - Spearman and Pearson correlation per metric;
  - per-room arm-order agreement: Kendall τ of the 4 arms' overall scores, averaged over rooms with all 4 arms complete;
  - whether the arm ranking by mean overall is the same.
- **Self-preference check (descriptive).** L's margin over the mean of J, P and O6 under each judge.
- **Usage.** Median reasoning tokens and wall time per case.

## 4. Budget

- **Calls:** ≤ 802 paid calls in total, all on the existing server API configuration. The key is never printed or copied.
- **Compute:** one CPU Slurm job (1 task, 5 CPUs, 8 GB, 12 h limit). It is not a login-node process.
