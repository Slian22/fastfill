# Amendment 021 — the 27B student arm, cheap diagnostics, and the final cross-amendment selection

Locked 2026-10-14 (UTC), before any amendment-021 output exists.

**Disclosure.** Designed after viewing:
- the amendment-016 §2 pilot (all five agents);
- the research memo and its skeptical review (`solution-research-20261014b/`);
- an Agency LLM-post-training review of an earlier draft. The draft's D1 gate, its Arm B and its common-engine rule were replaced on that review's advice.

The pilot rooms are a subset of the test cohort. They motivated this arm and are not used to select anything here. Amendments 013, 016, 018, 019 and 020 run unchanged. Given the Isambard queue (Priority = 1), this amendment adds exactly one training arm.

## 1. Diagnostics and reference rows (descriptive; run in parallel with §2, never gating it)

1. **013 ledger mining** (CPU, already-scored `RUN/dpo-004`; 3,000 train rooms, k = 0 vLLM greedy plus k = 1..8 at T 0.7):
   - greedy pass by object-count bin;
   - k = 1 vs greedy;
   - exact-copy and co-located-duplicate rates;
   - a recombination ceiling at the level of support subtrees.
2. **P-only admission pool** (CPU). Count and re-verify the train rooms whose P target passes the frozen checker but which were dropped because J's GT failed (19,135 per the lineage audit). Break them down by source, bin and family.
3. **Prefix-intervention probe** (HF teacher forcing, 200 dev-select rooms, ≤ 0.5 GPU-h). Move the preceding same-category sibling onto P's next predicted position, then measure whether P's next record moves and the log-probability of colliding vs clean continuations.
4. **Reference rows O2nt** (Qwen3.8-27B at the amendment-007 revision with `enable_thinking = False`, zero-shot, P input, B2 enforcement, same harness and cap):
   - Rooms: dev-select, the confirm slice (§4) and the 1,454 cohort.
   - Reported: all-request and pass-among-complete; reasoning tokens must be 0 (asserted).
   - Purpose: the "fine-tune vs think" contrast and a check of whether fine-tuning helped. It decides nothing.

## 2. Arm A — 27B student (the arm); Arm B only if Arm A is infeasible

**Model.** Qwen3.8-27B (the O2 weights at the amendment-007 revision, thinking off), LoRA, language-model layers only.

**Data**
- Pool: the P-only admitted train pool, i.e. every train room whose P target passes the frozen checker, including the rooms of §1.2. Dev and test rooms and their house groups are excluded.
- The prompt and target are exactly P's SFT format.
- **Disclosed confound:** N is below P's 98.9k rooms, so "base model" attribution is approximate.

**Recipe**
- Same as P's SFT: LoRA r32/α64, lr 5e-5, 1 epoch, effective batch 4 (4 ranks × batch 1 × `--grad_accum 1`).
- Token-mean NLL: summed over the answer tokens of the window, divided by the supervised-token count, all-reduced across ranks.

**Code path:** `train_dpo --sft_only` (DDP). Required changes, each with a test:
- a token-mean loss flag;
- LoRA target regex limited to the language model, plus an assertion that every trainable parameter has a gradient after step 1;
- answer-only logits (`logits_to_keep`);
- assertions that the training EOS equals the inference stop token (`<|im_end|>`) and that the training prompt render is a token prefix of `infer.prompt_ids` with thinking off;
- a pinned 27B base receipt, accepted by `train_dpo`, `dpo_stages.inference_identity` and `sample_vllm`;
- `ckpt_latest` saved regularly so the run can resume.

**Smoke and N**
- A smoke run on the longest admitted sequences and 200 steps measures peak memory and throughput.
- Stop rules: peak memory > 90 GB, or a projected wall-clock > 12 h for N = 40,000.
- **N** = min(40,000, the number of rooms the measured throughput fits in 10 h), taken in `sha256('a021-27b:' + uid)` order. N is written to the run receipt before training starts.

**27B fidelity probe.** HF exact vs vLLM LoRA (TP 1, `max_lora_rank` 32, image/video inputs 0) on 100 dev-select rooms. The checker-success difference and the share of identical outputs are reported. A merged copy may be listed only after an 011-§5-style check.

**Arm B (fallback only if Arm A's engineering or smoke fails).** 8B re-SFT on the same pool with the same `train_dpo --sft_only` path and recipe. It is reported as a fallback.

## 3. Engine rule (pre-registered)

**Engine check.** On the first 1,000 confirm-slice rooms (hash order), R (rpo-dpo1) and P-SFT are scored with both HF exact and vLLM greedy.

**Common engine.** If R's |HF − vLLM| checker gap is ≤ 0.015 there, every candidate is selected on vLLM greedy (BF16 LoRA).

**Otherwise.** Each candidate is selected on the engine it would be served with: 8B models on HF exact, the 27B on vLLM.

**Test reporting.** R is reported on test with the winner's engine next to the winner. R's existing HF test rows are kept as they are.

## 4. Final selection of the delivered model (across amendments 013, 018 and 021)

**Candidates**
- R = rpo-dpo1;
- 013's adopted model, if any;
- 018's adopted model, if any;
- Arm A (or B).

**Selection slice.** The 600 dev-select rooms. Primary metric: raw frozen-checker success over all requests (incomplete = fail), on the §3 engine.

**Guards** (relative to R, same engine):
- complete ≥ R − 0.005;
- OOR ≤ R + 0.01;
- strict admission ≥ R − 0.01;
- GT-relation hold rate (`validate.holds`) ≥ R − 0.03;
- checker success in rooms with 21+ objects ≥ R − 0.02.

**Tie band.** R's run-to-run spread is measured with two vLLM runs of different batch composition; the tie band is max(0.01, that spread). Within the band the cheaper model wins (8B over 27B).

**Confirmation**
- Slice: 3,000 dev rooms disjoint from dev-select, ranked by `sha256('a021-confirm:' + uid)` and stratified by bin.
- Requirement: winner − R checker gain with house-cluster CI lower bound > 0, on the §3 engine.
- If confirmation fails, R stays delivered.

**Test.** Only the confirmed winner runs once on all 8,647 test rooms, alongside R on the same engine. Every amendment-012/016/017/019/020 table is then filled with it.

## 5. Budget (GPU-h)

| Item | GPU-h |
|---|---|
| Engineering smoke | ≈ 2 |
| Arm A training | ≈ 25–30 (one 4-GPU node, ≤ 10 h wall) |
| Fidelity probe | ≈ 1 |
| O2nt reference rows | ≈ 6 |
| Engine check (R and P-SFT on HF + vLLM, 1,000 rooms) | ≈ 15 |
| Dev and confirm on vLLM, all candidates | ≈ 6 |
| Test on vLLM (winner + R) | ≈ 2–8 |
| Probe and ledger | ≤ 1 |
| **Cap** | **75** |

## Addendum A (locked before any amendment-021 output exists; from the tooling and its review)

1. **GPU cap: 200 GPU-h** for the whole amendment. The user said GPUs are sufficient.
   - §5's cap of 75 contradicted §2's "N fits 10 h on 4 GPUs", which alone is ≤ 40 GPU-h of training.
   - The split-engine path can need HF confirmation and HF test runs of up to about 72 and 84 GPU-h.
   - Running and pending jobs count at their time limits; jobs pending on a dependency are looked up by id.
2. **Memory stop rule:** peak reserved memory > 90 GiB on any rank (GH200: 95 GiB visible).
3. **Training details**
   - Arm A loads `Qwen3_5ForConditionalGeneration` (adapter keys `model.language_model.layers.*`, as vLLM maps them). LoRA targets are restricted to the language model by regex.
   - Warmup ratio is 0.03, as in P's SFT.
   - The bf16-autocast difference from P's HF Trainer is disclosed in the run receipt.
   - fla-core is pinned at 0.5.2 in a job-local overlay; a strict kernel check in the GPU job is binding.
   - Before the full run, a smoke-trained adapter must load and generate in vLLM (`smoke-vllm`).
4. **Confirm slice**
   - Pool: the SFT length-admitted dev pool, minus dev-select's UIDs and their house groups.
   - Allocation: proportional over the object-count bins `scale_curve.BINS`.
   - Within a bin: lowest `sha256('a021-confirm:' + uid)` first.
   - The engine slice is the first 1,000 confirm rooms in hash order.
5. **GT-relation hold rate**
   - Relations: GT's `on` and `against_wall` (`build.extract_constraints`), checked with `validate.holds`.
   - Rate: relations held / all relations; an incomplete room holds none.
6. **Guards with missing values.** Missing on both sides passes; missing on one side fails.
7. **Tie order inside the band.** 8B before 27B, then R, then higher checker success, then candidate order.
8. **018 test run.** If 018 adopts a candidate, its own pre-registered HF test run (018 §4) is the run of record that 021 §4 uses for 018's candidate. 013's test is deferred here (013 addendum A).
