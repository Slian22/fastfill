# Review of the decision memo "closing the gap between P (rpo-dpo1) and O2 when sizes are fixed"

Reviewer role: skeptical senior reviewer. Labels used below:
- **[V]** I checked it in this session, in code, data or a primary source.
- **[NEW]** I computed it in this session; all are EXPLORATORY and most use test rooms.
- **[I]** Inference.
- **[PK]** Prior knowledge, not re-checked in this session.

## 0. Verdict

The memo's mechanistic story is mostly sound as a description of the symptoms. Its framing of the gap is wrong, though, and its priorities do not follow from the evidence.

1. **The headline gap compares a single greedy shot with a 3-attempt harness.**
   - O2's single-attempt checker success on the pilot is **0.793**, not 0.920. Its completion rate is 0.800 (`server-pulls-016/llm-eval-O2-P-pilot-v1/report.md:13`).
   - With the already-adopted S1 applied to both, **P+S1 vs O2-raw+S1 is +0.030 [−0.020, +0.081]**, i.e. parity on the primary metric (`compare-016-pilot-v1/report.md:61`).
   - The gaps that remain at system level are:
     - vs O2-harness: −0.097 [−0.133, −0.059] (`:53`);
     - vs L: −0.150 [−0.190, −0.110] (`:65`);
     - on strict admission vs O2-raw: −0.147 [−0.197, −0.098] (`:61`).
2. **The memo ignores results that are already local.** The O4, O6 and O7 size-fixed pilots and the full pilot comparison (`compare-016-pilot-v1`, commit 7f3a0f98d) contradict two of its premises (§2).
3. **The plan restarts directions the user already dropped.** Per memory 2026-10-14 ~15:40, GRPO and instance indices were "dropped for time", and the deliverable is "ONE model = SFT + two-stage DPO". The plan also ignores that M1 is already running as 016 §3(b), and its GPU and wall-clock estimates break measured job limits.

## 1. Factual errors and unsupported claims

| # | Memo claim | Finding |
|---|---|---|
| 1 | §5: O2's "raw single-attempt rate is 0.92 over all rooms" | **[V] Wrong.** Raw is 0.7933 (complete 240/300); harness is 0.920 (`report.md:13-14`; `levels.csv`). |
| 2 | "238 of 240 complete raw answers pass" (0.99) | **[V/NEW] True but selection-biased.** O2-raw finishes easy rooms: mean 7.1 objects among complete vs 17.0 among incomplete. P passes 0.600 (not 0.513) on those same 240 rooms. O2-raw completes only 6 of 21 rooms with 21+ objects (`review/pilot_cond.py`). |
| 3 | O2 uses "10.9k reasoning tokens", "35×" | **[V]** That figure is from the size-free cohort. On the size-fixed pilot O2 used 4,669,666 reasoning tokens over 394 calls: **11.9k per call and 15.6k per room including retries** (`report.md:77`). |
| 4 | "Same-category instances get identical input tokens (contract.py:166-171)" | **[V] Imprecise.** Each instance gets a distinct sequential id `obj_{i:04d}` in HMAC order (`contract.py:172`); only category and prior size coincide. An implicit positional index already exists, which weakens the "instance index" rationale of M5. |
| 5 | Cause 1: "P does not take into account what it has already placed" | **[NEW] Contradicted.** P does use context: it copies it (§2.2). |
| 6 | "Collisions spread evenly over thirds" as evidence | **[V] Weak.** `rootcause/seqpos.py` counts both members of each pair, which makes the distribution near-uniform by construction. The diagnostic "later member" tally was computed but not reported. |
| 7 | Cause 3, P "does not re-plan", with "82 of 122 gap rooms" | **[NEW] Not P-specific, and a stratum-size artefact.** From GT+prior pass to fail, O4-raw drops 0.754 → 0.518 (−0.236), as much as P (0.649 → 0.404); O2-raw drops 0.888 → 0.717; only L is flat (0.948 / 0.946). The fail stratum is 166 of 300 rooms. Per-room gap vs O2-raw is 0.313 vs 0.239, only 1.3× (`review/gtsplit_pilot.py`). |
| 8 | "~80 % of gap in ≤20-object rooms" | **[V/NEW]** The arithmetic reproduces (0.129 / 0.127 / 0.130 / 0.070 / 0.020). With O2-raw instead it is about 87 %. But it combines P's full-test rates with harness rates from a non-representative cohort (holodeck 34.6 % vs 5.8 % of test; SpatialLM 2.1 % vs 28.5 %; `findings.md:36`). It mainly restates that ≤20-object rooms are 83 % of test. |
| 9 | M1 "dev raw 0.47 → 0.55–0.62 (pass@8)" | **[V] Mixed bases.** The 0.600 best-of-8 is vLLM on all 8,167 dev rooms over a vLLM greedy baseline of 0.400. R = 0.470 is HF on the 600 dev-select rooms (`findings.md:47-48`). |
| 10 | M3 "adds ~6k targets to the 98k" | **[V] Wrong.** SFT-D trains on teacher targets only, initialised from rpo-dpo1 (`amendment-018:27-30`). Teacher yield is skewed to small rooms, so this is a forgetting risk for dense rooms. |
| 11 | "DPO2 used synthetic perturbations. Test gain of rpo-dpo1 over SFT: +0.008" | **[V]** The +0.008 is DPO1, not DPO2 (`findings.md:31`). |
| 12 | "Amendments 020 (diagnostics), 021 (arms)" | **[V] 020 is taken.** It is the two-stage sizer → placer amendment, locked in 510f4c7cc. Renumber to 021 and 022. |
| 13 | Diffusion/flow heads "cannot take our conditioning" | **[V] Overstated.** DeBaRA exists (NeurIPS 2024, arXiv 2409.18336: https://arxiv.org/abs/2409.18336), a conditional score-based arrangement model for bounded rooms. The user had planned a DeBaRA-style line (memory 2026-10-14). The right label is "out of window", not "cannot". |
| 14 | Literature gains (3D-Layout-R1 0.700 → 0.854 → 1.000, etc.) | **[V partial]** 3D-Layout-R1 exists (arXiv 2603.22279) and is a text-guided *editing* benchmark. The collision numbers are not in the abstract and I did not re-verify them. These are few-object tasks and should not anchor estimates for 20–50-object rooms. |

## 2. New evidence that changes the hypotheses

### 2.1 Reasoning length does not predict success

**[V/NEW]** Size-fixed pilot, raw. The first rate is pass among complete answers; reasoning tokens are per room from each `report.md` Table 3.

| System | Pass among complete | Reasoning tokens / room | +S1 |
|---|---|---|---|
| L | 0.947 | 311 | 0.977 |
| O2 | 0.992 | 15.6k | 0.797 (raw) |
| O7 | 0.701 | 7.1k | 0.767 |
| O6 | 0.775 | 5.2k | 0.877 |
| O4 | 0.628 | 3.3k | 0.837 |
| P | 0.513 | 0 | 0.827 |

**[I]** Capability, not visible thinking length, separates the systems. This weakens the premise of M4 and M5 ("explicit reasoning raises the ceiling"). Small students imitating long CoT is a known weak spot (Li et al. 2025, arXiv 2502.12143 [PK]).

### 2.2 P copies context; it does not ignore it

**[NEW]** Natural experiment: HMAC order randomises list position within a room. Data: P's raw test layouts (`review/order_nat2.py`).

| Objects | Parent listed before child: at parent's exact centre / collides | Parent listed after child: at parent's exact centre / collides |
|---|---|---|
| 6–10 | 0.62 / 0.155 | 0.37 / 0.104 |
| 11–20 | 0.53 / 0.277 | 0.34 / 0.235 |
| 21–50 | 0.58 / 0.453 | 0.41 / 0.374 |
| 51+ | 0.64 / 0.546 | 0.47 / 0.447 |

- Same-category, same-support exact duplicates (within 1 cm) do **not** depend on list distance once room size is controlled. For 21–50 objects: adjacent 0.343, 2–5 apart 0.357, more than 5 apart 0.352.
- **[I]** The occupying record is often the immediately preceding one, and P still re-emits its coordinates. Adding footprint boxes to the context (M5) adds information P already has and copies.
- **[I]** M5's "parents then children" order is associated with *more* child collisions under the current model.

### 2.3 Missed cause: training admission drops a room type that is common in test

- **[V]** `main_data.py:183-196` admits a room only if J's GT and P's target both pass. Admission counts from `evidence/full-audit-data-lineage-7368003-result.json`:
  - eligible: 124,843
  - joint-qualified: 103,316
  - placement-qualified: 118,058
  - accepted: 98,923
- **19,135 train rooms have a checker-valid P target but were dropped only because J's GT target failed.** The matched J/P design required that; it does not bind P any more. Recovering those targets needs a check, but the code path exists.
- **[NEW]** In test, 2,395 rooms (27.7 %) have GT failing even with GT sizes. P passes 0.238 there vs 0.460 elsewhere.
  - Within object-count bin × GT+prior cells the penalty is 0.06–0.19.
  - If those rooms matched their cells' rate, test raw would gain at most about +0.035 (`review/gtsplit.py`).
  - Part of this is intrinsic difficulty: O4-raw drops 0.698 → 0.443 and O2-raw 0.835 → 0.693, while L does not drop (0.943 → 0.955).

### 2.4 Ruled out: support-parent errors

**[NEW]** Share of objects whose `on` differs from GT on the pilot: P 0.17, L 0.205, O2 0.121. Within bins, P's pass rate barely depends on mismatch (6–10 objects: 0.345 vs 0.325) (`review/parent_err.py`).

## 3. Cheaper falsification before any new training arm

| Test | Cost | Replaces or decides |
|---|---|---|
| 013 op-DPO2 dev and 016 §3 (RFT(b) = M1 round 1 on 013's repaired chosen; perturbation DPO; seeds) | already queued | Direct test of cause 2 / M1. **Do not start M1 before (b) reads out.** |
| Mine the 013 ledger (`RUN/dpo-004`, 3,000 train rooms, k=0 vLLM greedy plus 8 samples at T0.7, all already scored, `onpolicy_dpo.py` header): (a) train greedy pass by bin vs dev vLLM greedy (0.570 / 0.116 / 0.0 / 0.0 for 1–10 / 11–30 / 31–50 / 51+, `s2-011` summary); (b) single sample k=1 vs greedy, with copy and duplicate rates; (c) D4 recombination at **support-subtree** level | 0 GPU, server CPU | Replaces D2 (2 GPU-h). Tests greedy mode-collapse. Fixes D4, which mixes children and parents from different samples. |
| Prefix-intervention probe, HF teacher forcing, ~200 dev rooms: move the preceding sibling onto P's next predicted spot and measure whether the next record moves; compare log-probs of colliding vs clean continuations | ≤0.5 GPU-h | Mechanism gate for M5 and M7. |
| D3 (sizes ×0.85 / ×1.15) | — | **Covered** by amendment 020's P runs with llm-L / llm-O2 sizes and the O2 GT-size pilot 7409103. Drop D3. |
| D1 (O2 thinking off) | ~3 GPU-h plus a new llm_open spec | **Remap.** It tells whether O2's edge needs thinking (that is, M4), not whether a fine-tuned 27B would win (M6). Zero-shot no-think scores conflate format failures (O1a scored 0 from `on`-field errors, `findings.md:40`). |

## 4. Gains, costs and deadline

**M5**
- `train_pair` refuses `WORLD_SIZE≠1` (`train_pair.py:339,367`); the P SFT ran on 1 GPU (`isambard_main.sbatch:8`).
- Doubling answer tokens implies about 35 h on 1 GPU [I], beyond the 24 h limit (`isambard_main.sbatch:11`, timeout 1435 min at `:36`).
- "6 h on 4 GPUs" would need `train_dpo --sft_only`, whose loss is a per-sequence mean NLL (`train_dpo.py:179`). That is a different recipe, so a matched control is mandatory.
- The parser and contract assume JSON only, with no reasoning block (`scene.py:27`; `infer.py:174`).

**M6 (27B student)**
- Same single-GPU limit; 61 GPU-h means three or more chained jobs; no FSDP code exists.
- Test at about 240 GPU-h does not fit the window.

**M2 (GRPO)**
- The project env deliberately has no TRL (`train_dpo.py:3-8`).
- The only TRL env is OptiScene's (trl 0.15.2, transformers 4.49; research-log 15:20). Transformers 4.49 predates Qwen3 support (4.51) [PK], so a new aarch64 env is needed. The 1.5–2 day estimate is optimistic.
- The reward is the evaluation checker.

**M4 (trace distillation)**
- Teacher traces are 17k tokens or more, above `MAX_LEN 8192` (`dpo_stages.py:71`).
- 018's teacher has no job yet (it is not in the `research-state.yaml` handoff).
- Reasoning is stored per answer (`llm_open.py:202`).

**M1 cost**
- SFT per round is overstated: the measured rate is about 5.6k rooms per GPU-h, so 6–8k targets take about 1.5 GPU-h.
- Wall-clock queueing is the binding cost.

**Budget and deadline**
- 180–260 new GPU-h on top of 018 (≤150), 013 (≤150), 016, 020 (≤30) and 014/015, all at Priority=1 with fair-share 0.18 (log 16:20).
- The only fixed deadline found is the automation watchdog at 2026-10-18 05:53 UTC (`research-state.yaml:42`); the Isambard cert expires 21:48 UTC today (`:89`).
- **[I]** At most one new training arm fits in the window.

**Dev selection**
- About 12 or more candidates on the same 600 dev-select rooms creates a winner's curse.
- Confirm the adopted model on a disjoint dev slice; dev has 8,167 rooms.
- Use one engine for all candidates: M4 needs vLLM, and R's vLLM score is 0.432, not 0.470.

## 5. Fairness of the paper claim

1. **Report single-shot as single-shot.** P raw 0.513 vs O2-raw 0.793 vs L 0.947. Report system rows separately: P+S1 0.827 vs O2-raw+S1 0.797 (parity) vs O2-harness+S1 0.923 vs L+S1 0.977.
2. **Training on the evaluation checker.** RFT, GRPO, S3 and M7 all optimise the frozen checker; the agents never saw it. Primary claims need independent metrics: 019 mesh collisions, strict and required={} admission, OOR, FID/judge, relation hold.
3. **Post-processing favours P.** O2's failures are truncations under *our* cap of 1024+128n+16384 (`llm_open.py:65,146`; 57 of 60 raw failures). S3 cannot repair an incomplete answer, so "S3 for all" helps P disproportionately. The fair counterpart also gives the agents a larger budget or checker-feedback retries. The harness currently passes only exception class names (`findings.md:40`).
4. **Designs derived from test rooms.** The root-cause numbers (8,647 test rooms, the 750-room proxy) and the S3 prototype (`scenesmith/run60.json`, test rooms such as `SceneSmith::scene_207`) all come from test. Disclose this, and re-derive on dev before locking.
5. **"No size shrinkage" is vacuous against size-fixed agents**, whose sizes are enforced (addendum B2). Token-efficiency claims must count think blocks, retries and solver time; "1/35" is unsupported for M5.
6. **Distillation lineage.** 018 and M4 students are not independent evidence of 8B capability relative to O2.
7. **Thesis drift.** The locked thesis is "size confound → size-controlled evaluation → size-preserving repair-based preference supervision". M1 with repaired chosen and the §2.3 data lever fit it; M5, M6 and GRPO do not.

## 6. Corrected priority order

0. **Today, no GPU:**
   - Fix the memo's errors (§1) and renumber the amendments.
   - Ask the user to confirm scope again: GRPO and instance-index were dropped, and the deliverable is "SFT + two-stage DPO".
1. **Read out 013 and 016 §3, already queued.** They are the falsification test for cause 2 and M1.
2. **Mine the 013 ledger on server CPU:** train vs dev greedy, greedy vs single sample, subtree-level D4.
3. **Run the prefix-intervention probe (≤0.5 GPU-h).** It gates M5 and M7.
4. **Productionise S3 projection on CPU.**
   - Select on dev, not test, with keep-out, envelope, relation and displacement guards.
   - Apply it to every system as a system-level row.
   - It is the largest measured lever: S1 0.679 → S1+projection 0.868 bin-weighted on 60 test rooms per bin. This is a prototype estimate that dev must re-establish.
5. **New amendment: P-only data recovery.**
   - The 19,135 rooms with valid P targets dropped by the J-GT gate, used for continued SFT (about 4 GPU-h train plus 4 dev).
   - It fits the thesis; [I] expected gain ≤ +0.035 test raw.
6. **018 unchanged.** Watch its dense-room guards.
7. **D1, remapped to decide M4 only.** Drop D3.
8. **M5 only if items 2–3 support it,** with a matched-recipe control on a ≤20-object subset.
9. **Deferred past the window:** M2, M6, M4, M9, M10.

## Files

Scripts in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/better/review/`:
- `pilot_cond.py`
- `gtsplit.py`
- `gtsplit_pilot.py`
- `order_nat.py`
- `order_nat2.py`
- `parent_err.py`

Key sources:
- `/Volumes/harddisk/fastfillout/unknownsize-20261013/compare-016-pilot-v1/{report.md,levels.csv,contrasts.csv}`
- `/Volumes/harddisk/fastfillout/unknownsize-20261013/server-pulls-016/llm-eval-{O2,O4,O6,O7,L}-P-pilot-v1/`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/amendment-020-two-stage-size-then-place-20261014.md`
- `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-lineage-7368003-result.json`