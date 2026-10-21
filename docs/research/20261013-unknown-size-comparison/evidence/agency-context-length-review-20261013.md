# Agency context and length review — 2026-10-13

## Status

- **WARN — preserve the running matched 8k SFT pair; amend inference before test execution.** The common admission and no-truncation contract are reasonable. Length coverage and worst-case inference capacity need a larger, finite allowance; no quality or completed-training claim follows from startup receipts.
- This is a local, independent, read-only review except for this requested report. No code, lock, job, credential, server, or API was changed or contacted. The parent's latest server verification is explicitly attributed below.

## Observed Evidence

- Both locally read `lengths.json` files are identical: train **98,754/98,923 retained**, **169/98,923 = 0.170839946220798% excluded**; dev **8,151/8,167 retained**, **16/8,167 = 0.195910371005265% excluded**. These denominators are the already qualified paired pools, not all raw scenes. Both run receipts bind the same data manifest, base revision, tokenizer, seed and 8,192 total SFT context.
- `train_pair_helpers.py:235–292` renders the prompt, masks every prompt label with `-100`, supervises the complete answer plus `<|im_end|>`, and drops a UID from BOTH arms if either complete sequence exceeds 8,192. **No training target is truncated.** Formal inference instead stops on EOS or a finite generation cap and records each separately (`infer.py:156–204`).
- The existing batch-16 capacity receipt covers only **1,610 prompt + 2,048 new tokens**, with **26,188,830,720 allocated**, **28,290,580,480 reserved**, and **102,005,473,280 device bytes**. It does not validate the old 8k+8k worst case, much less full-context batch 16. Local run receipts record context 40,960; the parent additionally reports server/official verification of no RoPE scaling, 36 layers, 8 KV heads, head dimension 128, BF16.

## Failure Classification

- **Coverage/capacity design risk, not an observed training truncation or optimizer failure.** The 8k common pool excludes a small but nonrandom long-scene tail. An 8k generation ceiling can censor a structurally valid long answer; EOS is learned termination and is not guaranteed.
- **Full-context batch-16 KV memory is infeasible on the recorded device under the supplied BF16 KV configuration:** 96.64 GB KV + about 16.4 GB base already exceeds 102.01 GB before adapter/workspace. Batch 8 reduces that KV upper bound to 48.32 GB; actual fit and runtime still require the parent's bounded capacity measurement.

## Next Minimal Test

- Keep the completed checkpoint identities (once available), all training, tokenizer/template, paired dev UIDs, greedy/no-thinking decoding, evaluator and seed fixed. First isolate a **capacity-only change**: exercise the intended padded width and full context allowance on dev-derived inputs at batch 8, preserving terminal and peak allocated/reserved memory evidence. A short natural-EOS run does not exercise worst-case cache; if forcing length is used for a stress probe, label it synthetic capacity evidence and never use that forced-EOS policy for quality evaluation.
- Then isolate the **output-budget change** on both arms for the same predetermined dev cases: compare 8,192 versus full residual context, with the same batch layout. Include all 16 dev length exclusions and a fixed sample from every length tier. Measure EOS/cap/no-EOS, strict schema, complete object coverage, geometry, generated lengths, padded work and elapsed time. Cap censoring predicts previously capped incomplete answers become EOS-terminated and valid; repetition/poor EOS predicts longer invalid output or another cap hit. Stop promotion if capacity fails or length merely increases without completion evidence.

## Stop Condition

- Do not promote full-context batch 16. Do not start formal inference under a new allowance until the parent has frozen the amendment and passed the relevant dev-only capacity gate. Any prompt/context overflow must remain an explicit accounted request failure, never silent input clipping or deletion.
- Preserve the 96 requested GPU-hour ceiling unless explicitly amended. The proposed two 22h inference allocations yield **93.25 planned hours**, leaving **2.75** for additional authorized work; do not treat 22h as measured sufficient runtime. Do not cancel the running pair or unrelated jobs, or launch replacement training from this review.

## Artifacts to Preserve

- Preserve both length ledgers and run receipts, original protocol/locks, capacity receipt, code hashes, shared admitted UID identities, data manifest, tokenizer/config digests, resolved decode configuration and terminal scheduler evidence. Hashes and quantile definitions are recorded below; no raw private examples are included.
- Before formal inference, preserve final adapter inventories, verified manifests and clean-load evidence; existing `infer.py:115–153` enforces terminal training identity. Record the new common budget/batch plan digest, per-request prompt/answer counts, first-EOS status, effective cap, cap-without-EOS, schema/object coverage, memory and runtime. This report is not a checkpoint release or resume approval.

## Risks and Limitations

- Startup receipts do not establish live progress, convergence, held-out quality, or a completed checkpoint. The user's report that training is running is not independently reverified here. Parent-provided architecture facts are not a fresh server/config read by this reviewer. Graph lookup had no indexed OptiScene project; direct reads were used. A requested sidecar agent could not start because the thread limit was reached; this review was completed directly.
- Reference lengths describe supervised answers, not generated-length distributions. EOS may fail; neither a 24,576 allowance nor full residual context promises unlimited output. The historical test is exposed and must not tune any budget, retry, parser, or quality threshold.
- Equal paired rows/epochs do not mean equal supervised tokens: J has **53,157,852** versus P **34,454,444** (J/P **1.542845**), while total input tokens are **111,845,719** versus **112,329,829**. Report the matched exposure contract and this objective difference; do not call it equal assistant-token compute.

## Immediate pre-test recommendation

**Final decision, accepted by the user: finish the current 8k matched SFT pair; disclose 169/98,923 training exclusions; do not launch duplicate 32k training. Evaluate every one of the 8,647 frozen test requests under a full-context inference policy.** Training admission is not an evaluation exclusion policy. All inference errors remain in the denominator. New paired training would replay the 98,754 retained rows to recover 169 rows and would exceed the existing budget; long-context training remains a separately motivated follow-up if dev evidence warrants it.

Parent has now implemented `--context-budget 40960` with `effective_max_new_tokens=min(max_new_tokens,40960-padded_prompt_width)`. The intended settings are **batch 8, max input 40,959, max new 40,959, context 40,960, 22h per arm**. There is no artificial 16,384 input cutoff. Inputs of 40,960 or more tokens explicitly fail rather than being clipped. An admitted 40,959-token input leaves only one output token; finite context cannot guarantee complete output for such a request. This policy uses remaining context and keeps ordinary EOS stopping.

Parent plans a **shared paired batch-width plan derived from both request inputs only**, bound by a recorded hash. For each batch of at most eight consecutive frozen UIDs, use W equal to the largest admitted prompt width across both arms; both arms get `min(40959,40960-W)`. Preserve UID order, record actual padded widths, common W, effective cap and plan digest. A request impossible in either arm must retain an explicit error entry; exclude impossible requests from width arithmetic so one impossible prompt does not create a zero/negative cap for every valid batch member. Pair accounting must retain both requested UIDs regardless of eligibility. Do not parse GT sidecars for scheduling.

The latest locally inspected implementation still uses each arm's own width. That is safe for that arm's hard context, but does not yet implement the identical numeric paired output allowance. The parent's planned common-width amendment resolves this protocol gap. Do not reduce caps just to keep batch 16. The parent's long-context probe is reportedly submitted and its result is pending; 29 passing tests were reported by the parent, not rerun by this reviewer. The old local batch-amendment file still says batch16/8k+8k at review time, so a new immutable amendment must supersede it before formal inference.

## Fast review of the changed inference code and tests

- **P1: none identified in the inspected per-arm residual-context arithmetic.** `infer.py:148–154` validates the context against the pinned base and leaves room for generation; `:179–189` tokenizes without truncation and explicitly accounts oversized inputs; `:196–207` records the effective cap and uses it for termination accounting; `:242–253` calculates the runtime cap from actual padded width. With positive CLI caps and admitted inputs, the cap is positive. The CPU runtime test asserts a width-6 batch under context 20 passes max_new_tokens=14 to generation; the new receipt test checks effective-cap accounting at 14 tokens. Existing EOS tests distinguish first EOS, cap-without-EOS, and EOS exactly at cap.
- **P2: paired allowance still differs until the common plan is integrated.** Current `process_batch` and `_runtime.generate` independently derive caps from each arm's local width. J/P prompts differ; hence they can receive different numeric output budgets. When applying the shared plan, ensure BOTH calculations consume its common cap; changing only the runtime makes receipt cap flags wrong, and changing only accounting can reject valid runtime output. A common helper or explicit resolved-cap handoff should make the generation setting and receipt agree. Hash-bind the plan to both request file hashes, tokenizer/template identity, UID order, context and batch policy; reject altered or mismatched plans before generation.
- **P2: new boundary/common-plan coverage is incomplete in the inspected tests.** Add focused cases for prompt widths C-1, C, C+1; mixed accepted/overlong/all-overlong batches; configured cap smaller than remaining context; and paired unequal widths with exactly equal runtime/receipt caps and a tampered-plan rejection. Preserve first-EOS-at-final-token and post-EOS padding assertions. The observed new single-row receipt test and mocked unequal-width runtime test verify the central calculation but do not establish shared-plan integrity, real GPU fit, or longest-input prefill. This is a test gap, not a demonstrated context-overflow bug.

These findings cover the inspected snapshot only; the parent is editing concurrently. No tests, code, jobs, or server state were modified by this reviewer. The capacity probe must exercise long-input prefill as well as long generation: a short prompt plus enough generated tokens establishes a cache envelope but not worst-case near-40k prefill workspace. Do not label a remaining-context policy “unlimited.”

## Recomputed length evidence

All quantiles below use the actual ledger structure `split.lengths[*].lengths[arm]` and `answer_tokens[arm]`, including dropped rows. Prompt length is `total - answer`; answer includes supervised EOS. Quantiles are nearest-rank `sorted_values[ceil(q*n)-1]`, not an interpolated percentile. No test rows were read or used.

| Split | Arm | Tokens | p50 | p90 | p95 | p99 | p99.9 | Max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | joint | prompt | 506 | 928 | 1,099 | 1,623 | 2,536 | 4,371 |
| train | joint | answer | 272 | 1,270 | 2,083 | 3,928 | 6,664 | 12,684 |
| train | joint | total | 762 | 2,148 | 3,143 | 5,514 | 9,096 | 16,906 |
| train | placement | prompt | 600 | 1,348 | 1,798 | 2,956 | 4,704 | 8,415 |
| train | placement | answer | 170 | 826 | 1,364 | 2,611 | 4,468 | 8,786 |
| train | placement | total | 765 | 2,156 | 3,158 | 5,546 | 9,163 | 17,004 |
| dev | joint | prompt | 539 | 1,008 | 1,207 | 1,676 | 2,601 | 4,619 |
| dev | joint | answer | 318 | 1,409 | 2,497 | 4,103 | 7,089 | 14,395 |
| dev | joint | total | 858 | 2,330 | 3,679 | 5,704 | 9,431 | 19,014 |
| dev | placement | prompt | 657 | 1,454 | 2,005 | 3,075 | 4,994 | 9,673 |
| dev | placement | answer | 199 | 918 | 1,662 | 2,743 | 4,703 | 9,405 |
| dev | placement | total | 862 | 2,341 | 3,695 | 5,730 | 9,486 | 19,078 |

Among **retained** rows, maximum train answer lengths are J 6,211 / P 4,285, and maximum dev answers are J 6,092 / P 4,126. Maximum retained totals are train J 8,159 / P 8,179 and dev J 8,053 / P 8,118. Thus the running SFT fits complete answers, rather than teaching clipped targets.

In the full qualified pool, 28 train and 3 dev paired cases have an answer above 8,192 in at least one arm. Two train and two dev cases have a prompt above 8,192 in at least one arm (all P). These are distinct from the **169/16 total-sequence exclusions**; do not conflate input admission, training sequence admission, and generation termination. Train overlength flags are both arms 161, P-only 7, J-only 1; all 16 dev exclusions exceed the total limit in both arms. Train exclusions include 98 MansionWorld and 66 SAGE-10k cases, so the small aggregate exclusion fraction does not prove representativeness.

Formal inference records first EOS and excludes post-EOS padding from generated token counts. `reached_token_cap=true` together with `stopped_on_eos=true` can mean EOS landed exactly on the last allowed token. Report **cap without EOS** separately; EOS alone also does not establish valid JSON or full object coverage. The trainer's two dev probes have a simpler cap flag and are not the formal termination analysis.

For a lower-compute alternative only, the following dev-only input-length envelope is reproducible. Let P be the larger of the paired prompt lengths and A the larger reference answer. Cumulative dev maxima are:

| P upper bound | Dev rows at or below bound | Maximum A | `ceil_1024(1.25*A+128)` |
|---:|---:|---:|---:|
| 1,024 | 6,108 | 1,090 | 2,048 |
| 2,048 | 7,783 | 2,834 | 4,096 |
| 4,096 | 8,150 | 6,293 | 8,192 |
| 8,192 | 8,165 | 9,521 | 12,288 |
| 16,384 | 8,167 | 14,395 | 18,432 |

These numbers support length-aware scheduling, not a guarantee that generation fits. They are a tokenized request-length proxy; object-count fields are absent from `lengths.json`. A true object-inventory regression requires joining dev request inventories before freezing it. The 16,384 row here is only a descriptive dev envelope, not the final input admission limit. **Given the user's latest VRAM/completion preference, do not adopt these smaller output limits as primary; use the full residual-context rule above.**

## Memory and time envelope

With the parent-verified architecture, dense BF16 KV uses `2(K,V) * 36 * 8 * 128 * 2 bytes = 147456 bytes` per sequence-token. It scales with padded batch width plus generated steps, not just the mean nonpadding prompt. These are cache estimates, not measured peak process memory; full attention also becomes more expensive as context grows, and finished examples can remain padded until the batch finishes.

| Padded prompt + allowed new tokens | Batch-16 KV, decimal GB | Batch-8 KV, decimal GB |
|---:|---:|---:|
| 16,384 | 38.65 | 19.33 |
| 24,576 | 57.98 | 28.99 |
| 32,768 | 77.31 | 38.65 |
| 40,960 | 96.64 | 48.32 |

Full-context batch 8 plus an approximately 16.4 GB base is approximately **64.72 GB**, leaving approximately **37.29 GB** on the recorded device before adapters, workspaces, allocator reservation, and other runtime allocations. This is encouraging, not a measured admission guarantee. Full-context batch 16 is already approximately **113.04 GB** before those overheads. `logits_to_keep=1` avoids retaining all-position vocabulary logits but does not remove KV growth.

The existing pilot produced 22,937 tokens in 138.966s (165.055 tokens/s) at batch 16 and short context. It does not establish batch-8 long-context throughput or 8,647-request completion within 16h or 22h. Halving batch size is not automatically a 2x runtime penalty. Parent should measure representative short/medium/long dev batches, include no-EOS tails and padded work, and estimate both arms' runtime before locking capacity. Larger caps cost little when EOS occurs early, but pathological loops can consume the entire budget. Keep partial/failure accounting if walltime expires; no test-conditioned retries or shortened budgets.

## Training context alternatives

| Option | Data and scientific effect | Requested budget implication | Judgment |
|---|---|---|---|
| Finish current common 8k SFT | 98,754 identical train UIDs; 169 disclosed omissions; original matched estimand preserved | 93.25h total if inference becomes 22h/arm | **Recommended now**, with long-tail limitations explicit |
| Add fresh paired 32k runs; retain current pair as 8k ablation | All 98,923 qualified train rows and all 8,167 dev rows fit; all other training variables fixed | +48h for two 24h jobs; **141.25h** planned with 22h inference, before further ablation inference costs | Scientifically defensible full-pool primary only after an explicit budget/primary-endpoint amendment and dev capacity/signal gates |
| Later paired long-context supplement | Tests rare long scenes without invalidating current primary; identify continuation versus fresh-training estimand | Separately budget and gate | Useful follow-up; no silent merging into current endpoint |

Full qualified train maxima are only 16,906/17,004 and dev maxima 19,014/19,078. Therefore **20,480 already admits all observed train/dev rows; 32,768 supplies additional headroom, not additional observed rows beyond 20,480**. A 16,384 training cap would still drop 3 train and 2 dev paired rows; 12,288 drops 16 train and 3 dev. Increasing the training admission cap with this dynamically padding collator does not pad every batch-1 example to 32k. The recovered rows add **1,664,447 J tokens (1.48816%)** and **1,676,908 P tokens (1.49284%)** to current input totals; their greater attention cost can exceed their token-share cost. New-run expense mainly comes from replaying the existing dataset, not from every sequence suddenly becoming 32k.

The latest reported ~1h/4–6% training progress does not authorize cancellation. Preserving the two running jobs and adding two new jobs uses four protocol GPUs when all four train; unrelated user jobs remain untouched. Gate downstream inference to keep protocol concurrency at most four. The 141.25h plan exceeds the 96h ceiling by 45.25h; do not submit it under the existing lock. If the required primary estimand is explicitly changed to “all qualified training rows,” the larger-context fresh paired experiment is the cleaner design, but this review cannot erase its budget or completion requirements. A later continuation must start from a complete verified checkpoint with a declared optimizer-state contract, not an incomplete periodic save or a claim of unsupported resume.

## Behavior Target and Non-Goals

- Produce complete, schema-valid paired scene layouts and evaluate geometry on held-out requests; avoid incidental length censoring. No unlimited-output claim, test tuning, optimizer tuning, RL escalation, or retroactive quality promotion. Trusted structured targets make the existing SFT the weakest sufficient method.

## Fixed Comparator Contract

- Keep J0/P0 checkpoint identities, paired request UIDs and order, data manifest, source/group split, tokenizer/chat template, greedy single-beam/no-thinking decoding, parser and geometry evaluator fixed. Bind final adapter hashes after successful clean load. P remains the train-only size-prior placement comparator; any base comparator receives the same input-derived allowance and separately declared compute budget.
- Base is `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`; data manifest SHA256 is `e588e2f7c0def2c6515810816e0fd101b6dfdaa907274ce37cc9cd0d2c7c63e6`. Receipt tokenizer SHA256 is `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`; tokenizer-config SHA256 is `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`.

## Gate: Preflight

- Local length/accounting review complete; amended inference **smoke/capacity pending parent evidence**, then dev signal before formal execution. Existing capacity evidence is limited to its measured shape. Final model quality and release remain unverified.

## Single Change Under Test

- Output allowance: old finite 8,192 to finite padding-aware residual context. Establish batch-8 capacity separately before the paired dev comparison so budget and batching effects are not confused. Keep training fixed. Do not simultaneously change sampling, parser, repair, or model weights.

## Required Measurements

- Dev per-request EOS, no-EOS cap, schema, full inventory coverage and geometry; short/medium/long and excluded-tail strata; generated and padded token counts; allocated/reserved VRAM, actual device capacity, batch walltime and throughput. Preserve output storage usage and headroom. The old training receipt's free-disk measurement is not a current inference storage guarantee; large raw-output receipts and duplicate final records also consume space.

## Promotion or Stop Decision

- Promote only after parent verifies context/memory and a dev-based runtime envelope, locks the shared rule and 93.25h revised plan, and confirms complete clean-loaded checkpoints. Failure stops new inference promotion while preserving ongoing authorized training and evidence. More output tokens alone do not establish improved quality.

## Preserved Evidence

The following hashes were calculated locally during the initial review; the inference snapshot was superseded by the follow-up below. Paths are relative to this research directory unless prefixed `fastfill/`. Both length ledgers have the same byte hash. The reviewed trainer and helper hashes match those in both startup receipts.

| Artifact | SHA256 |
|---|---|
| `protocol.md` | `f2fc11ec56d927f551a438fd14e83ff4ca64e5e4a0e60d4b5e47968e7853706e` |
| `execution-lock.json` | `432158fcf1b4b91a9a98f85a3f3cdb0a5ad74d59b4d79b319d38dbfd1349b8bf` |
| `inference-batch-amendment.json` | `25896193b98029830122ab64301715f493fbe2ec436aad8674b9b122e30f924e` |
| `evidence/server-main-start/capacity-001/result.json` | `59d1cddbd99aa857f80f3f3b64e7fd438e92d6e25ae7aa547756c75e48c07fe5` |
| `evidence/server-main-start/sft-joint-001/lengths.json` | `d3dd6be2fe3af27c4ee6157fa2da04028d55b91f0015d7759b19447a0e4a950f` |
| `evidence/server-main-start/sft-joint-001/run_receipt.json` | `9bd27d9495f1a24ec6675796bf4f02c3778df4a1dc1465d7cf1f8933bd1df216` |
| `evidence/server-main-start/sft-placement-001/lengths.json` | `d3dd6be2fe3af27c4ee6157fa2da04028d55b91f0015d7759b19447a0e4a950f` |
| `evidence/server-main-start/sft-placement-001/run_receipt.json` | `0c048b78bfd8009a25c14a8817010f6187a6321133de0aac00e10f9b30d6ff6e` |
| `fastfill/comparison/train_pair.py` | `daf0dcf0c545502947c44e39d41806bd436d25179c88eb08cd24032d094152b0` |
| `fastfill/comparison/train_pair_helpers.py` | `6a8268dc576f34a03d79e67e876417b01b7a7e8ce76241b5299b41e50176ea76` |
| `fastfill/comparison/infer.py` | `650d3e7607fd836c7840ee78bab6b812cb684848b8befcfddea37d69bd379b3e` |
| `fastfill/comparison/isambard_infer.sbatch` | `2613beed228f154547d7f5124338dcf89bee443adcd4d2669503b3b33f589ca2` |
| `fastfill/comparison/contract.py` | `ef1e2cd9e449449f5441348e16d9a2bb2bd2ff8618132e6b6ed378347e64cefc` |

Only this review artifact was written by this reviewer. Implementation and job/amendment changes described above belong to the parent, not this review.

### Follow-up snapshot for residual-context code review

- `fastfill/comparison/infer.py`: SHA256 `650d3e7607fd836c7840ee78bab6b812cb684848b8befcfddea37d69bd379b3e`.
- `fastfill/comparison/tests/test_infer.py`: SHA256 `d5e4943dde08055dddbc8873f71d4c1e47c583617a66f4105944dff3ddc2ac53`.

The shared paired-width plan was still pending at this snapshot; it is not certified by this review. Retain training, complete the shared-plan tests and dev capacity probe, and freeze the superseding inference amendment before test execution.
