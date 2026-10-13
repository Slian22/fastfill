# Amendment 002 — inference throughput, LLM-agent arm, endpoints and delivery rule

Locked 2026-10-13 ~18:00 UTC, **before any J0/P0 test inference has run** (jobs 7367626/7367628 were held at 17:43 UTC; no `infer-*` directory exists on the server). User mandate 2026-10-13 ~17:40 UTC: autonomous autoresearch, request more GPUs as needed, Agency audit, deliver one model plus measured tables including a same-input LLM-agent comparison.

## 1. Why

Review workflow wf_e5054089-226 (8 agents, adversarially verified) estimated J inference at 24–29 h against the 1,315-minute timeout of the frozen plan (UID-ordered batch 8, HF greedy, ~68 ms/decode step, 45 % useful decode slots, no resume). A timeout in either arm cancels geometry 7368009 and leaves every table empty. P is marginal (16–19.5 h). The 96 GPU-hour lock is lifted by the user; new cap below.

## 2. Inference v2 (both arms identical; replaces 7367626/7367628, which are cancelled unrun)

- Same adapters, base revision, chat template, greedy decoding (`do_sample=False`, `num_beams=1`), batch size 8, context budget 40,960, max input 40,959.
- **Batch order**: one shared plan for both arms, ordered by input-derived keys only: number of requested movable objects (descending), then the larger of the two arms' prompt token counts (descending), then UID. Batches are formed from consecutive entries of this order. No output, target or GT field is used.
- **Generation cap** per batch (shared by both arms): `min(40960 − W, 1024 + 128 × max_objects_in_batch)` where W is the shared padded input width of both arms. Train-set answers average ~51 (J) / ~34 (P) tokens per object, so this is ≥2.5× the J rate; hitting it is recorded as `reached_token_cap` and counted as a failure in all-request denominators.
- **LoRA merged** into the BF16 base before generation (`merge_and_unload`) in both arms.
- **Sharding and resume**: the plan is split into N shards by round-robin over plan batches; each shard is one 1-GPU Slurm job with its own receipts/ledger; a shard can resume from its verified ledger after a timeout. A merge step verifies all shard receipts and identity and writes the canonical `records.jsonl` in `expected_uids` order with `INFERENCE_COMPLETE`, so the frozen geometry evaluator is unchanged.
- All 8,647 test requests per arm, no filtering, failures stay in denominators.

## 3. LLM-agent arm (L)

- Model `gpt-6.1-sol` through the server's existing OpenAI-compatible config (`~/.fastfill_api.env`, never copied or printed), reasoning effort medium.
- **Same input as J**: the exact J `system` + `user` messages (unknown sizes, same fields, same output schema). No size tables, priors, statistics or per-request extra information. A fixed instruction suffix about JSON-only output is allowed and identical for every request.
- **Harness**: parse with the same `parse_prediction` used for J; if invalid, up to 2 repair rounds that return only the parser error text (no GT, no geometry hints). Reported separately: `L-raw` (first answer) and `L-harness` (after repair). Calls, tokens, latency and repair counts recorded.
- **Cohort**: all 1,009 paper six-strata test rooms, plus a stable-hash sample of up to 30 rooms from each other source (frozen list written before any call). J/P are compared with L on exactly this cohort (paired), and J vs P on the full 8,647.
- Smoke: 2 calls first. Cap: 1,600 rooms × (1 + 2 repairs) calls; sequential or bounded concurrency with idempotent receipts.

## 4. Endpoints (fixed now)

- **Co-primary (J vs P, full test set)**: (a) all-request raw geometry success rate (failed/incomplete outputs count as 0), room-micro paired difference with house-cluster bootstrap 95 % CI; (b) size log MAE on common-complete pairs (P's sizes are the train-only prior, labelled as such).
- **Secondary**: complete rate, raw and repaired OOR, size IoU, source-macro versions of all of the above, six-strata OOR, FID, GPT judge (pooled 174-room paired difference is inferential; per-stratum descriptive), failure taxonomy, cost.
- **Reference rows (model-free, measured before outputs)**: GT layout OOR and geometry success; P size prior alone on all 8,647 rooms.
- **Sensitivity (pre-declared)**: exclude the 3 test rooms with known source defects; 3D-FRONT living stratum with and without the 166 `living dining room` rooms.

## 5. Delivery rule for "one model"

Deliver the arm (J or P) whose co-primary (a) paired CI excludes zero in its favour. If (a) does not separate, deliver the arm with lower (b). Both arms' numbers are reported regardless; L is a comparator, not a delivery candidate. Single training seed; claims are "for this training run".

## 6. Budget

New GPU cap for this protocol: 200 GPU-hours (93.75 already requested; inference v2 is expected to need ~20–30 GPU-hours). API: ≤ 4,800 LLM-agent calls plus the existing ≤ 360 judge calls.
