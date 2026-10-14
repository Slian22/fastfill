# Amendment 011 — visual-metric populations, reporting corrections and the merged-weight check

Locked 2026-10-14 ~11:20 UTC, before any FID/KID or GPT-judge output of the selected-model visual run (`visual-final-selected-002`, rendering) exists or has been viewed. Triggered by the independent audit `audit-final-20261014/` (workflow wf_c30fbffe-068). No confirmatory result, dev selection or delivery decision changes; everything below is reporting or a separate, labelled check.

## 1. FID / KID populations (replaces the all-system intersection)

The current code scores FID only on rooms rendered for every system; O1a has no complete room in the four holodeck strata, so even J and P would get no FID there. From now on:

- **Primary (J, P):** per paper stratum and pooled, the frozen reference set ∩ rooms rendered for both J and P (paired J/P population).
- **Each agent:** reference ∩ J ∩ P ∩ that agent, reported with its N next to J and P on the same rooms.
- **All-system intersection:** secondary, only for the pre-specified judged set {J, P, L, O6}.
- N < 50 is flagged; no FID is computed below the existing minimum. Renders already produced are reused; only the FID/KID step is recomputed.

## 2. GPT judge reporting

Per-arm pooled means on the frozen 174-room judged selection and paired differences P−L, P−O6 and J−P with stratified bootstrap intervals (estimation only). The judged open model O6 is pre-specified (amendment 005 §5), not the best measured open model.

## 3. Agent comparison reporting (EXPLORATORY throughout)

- a′ ("own-bias rescale with positions fixed") stays a stress test. Added: paired Δ(a′) with house-cluster bootstrap CIs; GT control rows (GT layouts scored at each system's factor); an equal-size control (every system scored after rescaling to GT footprint per object class); an "other sources (445 rooms)" scope; a statement that the cohort is not representative of the test set.
- Disclose per-system output budgets (L 65,536 tokens; open models 1,024 + 128·objects, +16,384 with thinking, capped by the 40,960 context), that "harness" means up to 3 attempts with a generic retry message (exception class only, not the parser text), per-round cap hits, model modes (thinking on/off, reasoning effort), and that O5 (DeepSeek-V4-Flash) was not run after 4 engine-build failures and was never substituted.
- O1a/O1b near-zero success is reported as a failure of the support (`on`) contract field, not as evidence about layout skill.

## 4. Stage reporting

- Holm is applied only at the SFT stage (the confirmatory comparison); other stages show unadjusted p and are labelled secondary/descriptive.
- Add the paired contrast P-selected (rpo-dpo1) − P-SFT on test with house-cluster CIs, labelled EXPLORATORY; the dev gain (+0.062 raw) is reported next to it.

## 5. Merged-weight check (delivery artifact)

All test numbers come from the evaluated exact policy (rpo-dpo1 adapter, unmerged, on its BF16 training base). The BF16 merge changed 6.75 % of target weights. Check: run the merged model on the same 600 dev rooms with the same plan and decoding; report complete rate, raw success, a′ and the fraction of byte-identical outputs versus the unmerged policy's dev outputs.

- If |Δ a′| ≤ 0.01 and |Δ complete| ≤ 0.005, the merged model may be listed as a convenience artifact; otherwise only adapter + training base are delivered.
- In both cases the model card leads with adapter + training base (the evaluated policy).

## 6. Corrections to wording (findings, research log, model card)

As listed in `audit-final-20261014/summary.md` (M2–M5, M12, M13 and minors 2–6, 15, 16): the rpo-dpo1 test gain (+0.0084) comes from MansionWorld and is not a replication of the dev gain; 3 of 4 J DPO candidates shrink sizes further (e1-dpo2 does not); stale statements are removed; superseded reports get a README notice.
