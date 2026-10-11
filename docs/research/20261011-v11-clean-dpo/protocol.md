# V1.1 clean preference training: preregistered stage comparison

Protocol date: 2026-10-11. Status: prepared for coordinator freeze and protocol-only Git commit before clean training or new full-cohort outcomes. The commit and deployment receipts establish the actual lock time; this draft date alone does not. The coordinator must freeze the tested source snapshot, pair files, model identity and decoding configuration before launch. Subsequent outcome-driven changes are exploratory and require a separately versioned protocol.

## Question and scope

Starting from the completed V1.1 SFT checkpoint, do two predefined DPO stages with audited preference pairs improve joint size, position, yaw and constraint-following behavior on the exact V1.1 validation and test rooms? The study evaluates JSON layout predictions and their fixed postprocessor. It does not establish a novel learning method, physical mesh validity, or superiority over published scene-generation systems.

The methods are `sft` (the existing completed V1.1 SFT merge), `dpo1` (a fresh DPO run from that SFT using audited Stage 1 pairs), and `final` (fresh Stage 2 DPO initialized from the clean DPO1 merge). Old DPO checkpoints and their runs remain preserved as diagnostic artifacts. The clean chain must not resume an old DPO optimizer, adapter or checkpoint. This comparison measures training-stage effects. There is no matched unclean-control arm in the full matrix, so any improvement cannot be attributed causally to pair cleaning alone.

V1.1 SFT learns assistant JSON, including numeric size/position/yaw tokens, using masked causal cross-entropy. DPO learns sequence preferences relative to the stage's frozen base policy; pair checks provide geometry/constraint supervision indirectly. These are not differentiable distance, collision or mesh losses. The warm start is the existing OptiScene-Qwen-8B-Final model, not an untouched Qwen base.

## Hypotheses fixed before new runs

| ID | Prediction | Confirmatory comparison |
|---|---|---|
| H1 | Audited on-policy preferences improve useful accepted layouts relative to SFT because positive answers satisfy the prompt and rejected answers contain an independently verified defect. | `dpo1 - sft`, separately on ordinary dev and constrained dev |
| H2 | Geometric hard negatives provide further improvement after Stage 1 rather than only changing likelihoods on training pairs. | `final - dpo1`, separately on ordinary dev and constrained dev |
| H3 | Hash-bound resumable evaluation accounts for every original request and safely continues an interrupted task. | All 12 matrix cells reach their exact input counts with one explicit outcome per input row and matching manifests; tested interruption/restart behavior |

H1 and H2 may be supported on one dev cohort and inconclusive or regressive on the other; report this mixed result explicitly. H3 is an engineering coverage hypothesis, not evidence of model quality. New numerical-loss architectures, fresh SFT ordering changes, VLM rewards, paid LLM baselines and extra training sweeps are outside this bounded experiment.

## Data identities and exclusions

The SFT source release has train 144,150 / dev 8,167 / test 8,647 message rows. Five geometry flags retain 103,319 / 5,893 / 6,251 respectively before tokenizer-length exclusions. The room files below retain their original full denominators, including imperfect references and duplicate prompts. These evaluation counts are not the filtered SFT counts.

| Cohort | Original rows | Purpose | SHA256 |
|---|---:|---|---|
| `dev_rooms.jsonl` | 8,167 | validation inference | `e0c2a6fdb44778e83f91273122fb5f7a4489cdb2c6fcf49ffc118ab15671c97f` |
| `dev_constrained_rooms.jsonl` | 8,035 | validation inference with constraints | `d54e28c06b4b3714b6e76b544230c6edb45dd592e966dec6b916ce068f16e2e2` |
| `test_rooms.jsonl` | 8,647 | descriptive final report only | `0b28d10c015bbd9f39151113085c84a05a8117ea3facfa5553e095d10deff303` |
| `test_constrained_rooms.jsonl` | 8,533 | descriptive final report only with constraints | `0bb4edaab1ceca5ee794ba5ae99ff8ee8aff26b48e46ae723259cc50900c727d` |

Each of the three methods is evaluated on each cohort: 12 tasks, 33,382 requests per method and 100,146 total requests. Duplicate user prompts occur 9 / 2 / 6 / 1 times respectively; retain all original requests. Prompt caching may reuse identical generations only if the receipt and each original outcome remain explicit. Parse failures, context failures and failed complete-object checks remain in the denominator. Never repair a missing output using reference geometry and call it a successful model output.

The prior 200-request results used prefix sampling after prompt deduplication, not representative sampling. They cover only HSSD200 and IL3D_3dfront. Ordinary test200 contains 138/62 rooms from those sources; constrained test200 contains 132/68. These outcomes were already seen and are historical diagnostics. The test is therefore not claimed to be entirely unseen by the research process. No new selection, threshold tuning, hyperparameter adjustment or retraining decision may use the new test outcomes.

Source-heldout reporting is limited to exclusion from the current V1.1 split: SceneSmith 364 + SpatialGen 45 = 409 ordinary test rooms, and 363 + 45 = 408 constrained test rooms. The warm-start directory lacks a training manifest and provenance; exclusion from every earlier warm-start stage is not independently verified. Report in-distribution and source-heldout results separately.

Fresh audit verified all seven release JSONL hashes and all 18 IR hashes. All 160,964 persisted UIDs match their IR records. Train/dev/test have 56,145 / 3,163 / 3,324 groups; UID, group and alias intersections between splits are zero. Previously completed hash-bound QA also found zero furniture/layout fingerprint intersections. This does not prove complete physical-alias or asset-library disjointness. The source manifest and audit evidence are `.release/v3.2/data/v3.2/{MANIFEST,QA,VALIDATION}.json` and `/tmp/fastfill-v11-automation-20261011/data-audit.{json,md}`.

The separate V2 Stage0 frozen 300 validation + 300 test cohorts are not replicated by this experiment. Although their UIDs all map to the same V1.1 split, 234/300 validation and 239/300 test inventories differ. V2 includes 12,998 / 13,156 objects versus 8,182 / 7,998 in the corresponding V1.1 views. Comparing those raw metrics as the same task would be invalid. The five RoomGenBench scenes remain separate downstream demonstrations, not replacements for these cohorts.

## Audited preference data and training controls

An initial diagnostic found 144 Stage 1 preferred answers that violated prompt constraints. The completed sanitizer subsequently verifies the exact train UID, source and prompt, reference geometry, all prompt constraints and size limits for every pair. It requires a positive to pass and the rejected answer to contain at least one independently verified semantic defect. It neither accesses evaluation outcomes nor synthesizes replacement answers.

For each GT dimension `g`, the accepted interval is `[min(g, max(0.05, 0.45*g)), max(g, min(6.0, 2.2*g))]`. Including `g` prevents genuine thin or large reference dimensions from failing an arbitrary absolute bound. This is a frozen training check, not proof of semantic size realism. The full audit retains 7,244 of 8,000 Stage 1 pairs and drops 756 unique rows. Reasons overlap: 144 constraint-invalid positives, 475 positives outside corrected size limits and 215 rejected answers without a verified defect. These counts must not be summed as disjoint reasons. All 12,000 Stage 2 pairs survive the same audit before tokenizer-length filtering.

| Artifact | SHA256 |
|---|---|
| actual V1.1 `train.jsonl` | `6ab4c001e566718ce3e89a51de08912edbc9ba2aa740fe4284741e04bd6650a0` |
| original Stage 1 pairs | `376ee1096b2cad5a1f67d2a29dc5b49b18c9a71e6317c5b1a20e771ee8b969b1` |
| audited Stage 1 pairs | `072604820711e5749d11924ef1b3f38be1e662297a4efe80e446f47380447c1f` |
| original and audited Stage 2 pairs | `bf4c535bb299eb7ac2e20e5be05bc13946c466cab817fb04e74d7d4389ad3165` |

Evidence: `/tmp/fastfill-v11-automation-20261011/sanitized/audit.json`; the coordinator must copy this receipt alongside the remote immutable inputs. The sanitizer's recorded code hashes are part of provenance. Record the trainer's final admitted-pair count and length skips separately; 7,244 / 12,000 are pre-tokenizer counts.

The predefined recipe keeps the existing DPO design: LoRA rank 32 and alpha 64; beta 0.1; two epochs per stage; learning rate 5e-6 for Stage 1 and 2e-6 for Stage 2; per-device batch 1, accumulation 4, four GPUs; context 8,192. Full generated completions over the training budget are rejected as whole pairs, never silently truncated. The trainer uses FP32 token cross-entropy/log-probability accumulation with correct assistant masking and frozen reference policy. Preserve optimizer, scheduler and trainer state for a continuation of the same clean stage. The current trainer does not persist RNG state; a resumed dropout sequence is therefore not bitwise identical to an uninterrupted run. Record any continuation explicitly rather than claiming exact replay. The launch manifest must bind all actual parameters, seed and software dependencies; an unannounced deviation is not confirmatory.

## Measurement and uncertainty

The primary outcome is the all-request mean of the frozen evaluator's `zero_collision_accepted` indicator after its fixed repair procedure. For failed requests the value is zero. This is evaluator acceptance at the bounding-box level; it is not verified physical stability or mesh collision freedom. Use the same versioned evaluator and postprocessor for all methods and save raw generations so scoring can be audited.

For H1 and H2, pair original requests by frozen cohort row identity and compute candidate minus baseline separately on ordinary dev and constrained dev. Resample `(source, group)` clusters with replacement, retaining all original room rows within each drawn cluster; use the same sampled clusters for both methods. The statistic is the room-weighted mean difference, including repeated clusters' room multiplicity. Use 2,000 bootstrap draws, seed 42, and percentile 95% intervals. Report group count, input count and paired count. Fewer than two usable clusters or any incomplete/mismatched cohort makes inference inconclusive. These are per-comparison intervals without a familywise claim across the four dev contrasts.

Per comparison, classify `supported` only if the complete valid cohort has at least two clusters and the CI lower bound is strictly greater than zero; `regression` only if the CI upper bound is strictly less than zero; otherwise `inconclusive`. These labels apply to the primary proxy on that cohort. They do not declare an overall winning model or establish improvement in every quality dimension.

Supporting evidence must include raw and repaired collision rates, parsing/complete-object coverage, size realism, dimension-only `size_iou_3d`, support/boundary failures, and constraint satisfaction by type. Report size IoU both on all requests with failure assigned zero and on the common-complete paired subset with its denominator. The latter is conditional on generation success and can be biased by selective failures. Dimension-only IoU ignores position/yaw and must not be described as placed-box 3D IoU. Coverage and size changes are descriptive warning signals; no new 1-percentage-point or 0.02 thresholds are imposed. Keep raw outcomes separate because repaired acceptance can improve while the model itself does not.

Test outcomes are descriptive stage comparisons after the models and protocol are frozen. Synthesis can run when all 12 cells and receipts are complete; before then it reports missing cells and known diagnostics, not final hypothesis support. Source and object-count strata are explanatory, and no subgroup discovery creates a retroactive confirmatory claim.

## Execution, stopping and unattended continuity

The coordinator owns submissions. Bound the clean training chain to at most two six-hour four-GPU allocations, including continuations, and each of the three model-evaluation chains to at most two six-hour one-GPU allocations. The planned maximum is 48 training GPU-hours plus 36 evaluation GPU-hours, excluding earlier jobs and queue time. This is a cap, not a completion guarantee. After the cap, preserve checkpoints and report unfinished work; do not launch an unbounded resubmission loop.

Before launch, freeze tested code and record its hash, commit this protocol separately from results, bind data/pair/checkpoint identities and install resumable Slurm dependencies with distinct output directories. Model fingerprints and completion checks must reject stale 200-row metrics, mismatched inputs or incomplete generations. Atomic per-request outputs, explicit failure rows and resume receipts support H3. Stop or mark failure on hash mismatches, invalid pair provenance, NaN/Inf training, incompatible checkpoints or exhausted allocations; do not erase old results to clear a failure.

The available runtime has neither Claude `/loop` nor an agent-bound `cron.add`. Unattended execution therefore consists of the actually submitted bounded Slurm graph and automated synthesis/report generation, not a permanently thinking chat agent. A future agent can read `research-state.yaml`, `findings.md`, the canonical claude-mem work state and server receipts to continue. No automatic extra training is authorized by a positive scalar alone: further experiments require a documented dev-based hypothesis and a new frozen protocol within the user's authorized scope. Test outcomes must never trigger that decision. No paid model API calls or publishing are part of this experiment.

## Interpretation and follow-up

A positive dev comparison supports further examination of the corresponding stage under this specific SFT checkpoint, seed, corpus and evaluator. A non-positive result is useful evidence that preference likelihood gains did not yield the expected measurable layout gains. Before a future training step, inspect coverage, size changes, raw versus repaired behavior and source strata; then propose a mechanism supported by dev evidence. The present three-paper survey establishes methodological context, not a novelty search or publication-ready claim. Multi-seed replication, matched cleaning controls, protocol-aligned external baselines and physical/visual evaluation remain future work.
