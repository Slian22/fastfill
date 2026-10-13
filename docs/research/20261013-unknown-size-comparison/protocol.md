# Unknown-size J/P comparison: autonomous execution protocol

Locked before new training on 2026-10-13. User authorizes server operation, local frozen IR and Hugging Face assets, Agency delegation and Krill API judging. Goal: measured OptiScene-style tables with truthful scope, not a promised winning result.

## Hypothesis and primary comparison

Compare joint size/position/yaw generation J with train-only estimated-size + placement P. Both start from the same verifiable generic Qwen3-8B revision. P uses group-out-of-fold estimated dimensions during training, full-train prior at evaluation. Category-only inputs in the first controlled version avoid unverified source descriptions. Neither arm sees movable target dimensions/pose, raw asset IDs, or target-derived object ordering. Ground-truth asset mapping is an evaluation-only sidecar.

Primary delivered comparison is same-stage matched SFT J0/P0, using a frozen training budget and the same admitted scene pool. Retain this endpoint even if subsequent DPO is run. DPO comparisons are secondary unless a separate protocol is locked before test inference; both arms need controllable, geometry-verified preferences and matched budgets. Existing V1.1 SFT/DPO outputs remain a separate legacy diagnostic table.

First engineering pilot: at most64 train and4 dev rooms per source from the frozen v3.2 release; no test rows used for training/calibration. One GPU allocation, at most30 minutes; approximately8 updates per arm and two dev generations each. It tests loading, gradients, checkpoint saving/reloading, schema and provenance, not model quality. Data-builder qualification counts and dropped/overlength rows remain visible. Main training scale and token/compute budget are fixed after this dev-only capacity pilot, before any new test result is viewed.

## Data and endpoint contract

Freeze original source identity inventory before binding opaque instance IDs. At the frozen inventory boundary, changing movable size/position/yaw/support labels or shuffling labels must leave the input byte-identical. Existing preprocessing defines eligible inventory/fixed context; do not claim this test establishes invariance of upstream raw-scene eligibility itself. Test split membership inherited from the audited release is historical/exposed, never renamed a new blind test. Source-heldout SceneSmith/SpatialGen remain excluded from fine-tuning and size-prior fitting. Every connected house/alias component stays in one split and OOF fold; fit/test identities and hashes are archived.

Main geometry: raw OOR, schema/object coverage, dimension-only size IoU/log error, boundary/support/constraint validity where annotations are qualified. Raw and repaired results separate. All-request failure rates accompany conditional scores. No single automatic winner; report size/geometry tradeoffs. Source-macro and room-micro results separate; uncertainty uses paired source/house groups, not independent objects or views.

Paper-style six strata: 3D-FRONT bedroom/living and SynthPlace bedroom/living/kitchen/bathroom, plus broader source tables. Visual eligibility frozen from reference asset evidence before scoring. Same assets, room-only cameras and renderer for GT and each output, fitting each to its own dimensions. FID uses2048 Inception features, one fixed perspective per room; GPT uses the same frozen rubric and view policy for both arms. Generation/render/API failures retained; FID on shared successful cohort explicitly conditional, not population performance. No box proxy, black-image padding, score imputation or post-score sample replacement.

Initial visual API cap: 360 successful scoring requests plus at most20 explicit retry attempts; sequential receipt-bound calls. Model alias gpt-6.1-sol from existing server secret file. Do not disclose credentials. Initial matched GPT sample: at most30 rooms per each of6 strata per2 arms, stable hash selection before scoring; run a two-case image-capability smoke first. Compare within this judge; do not call scores numerically interchangeable with paper GPT-4o. Any extension requires a logged budget/scope amendment before scores. Human UR unavailable without actual five-person majority ratings; mark not measured.

## Continuity and preservation

This harness has no chat /loop or cron agent-turn tool; tool discovery verified this limitation. Durable continuity uses isolated immutable code/data snapshots, bounded Slurm job dependencies, idempotent receipts and automatic table generation. Live agent continues while this session runs. Do not claim a perpetual background agent or silently enable unrelated hosted automations. State records job IDs, caps and incomplete work. Never cancel existing user jobs, overwrite old runs or modify credentials. Scheduler requeue disabled unless a bounded recovery is explicitly recorded.

Initial pilot limit is0.5 GPU-hour. Main training/evaluation job graph will record individual wall-time/GPU caps before submission, with at most4 simultaneous newly allocated GPUs and at most96 requested GPU-hours for this protocol including pilot and explicit retries. Avoid duplicate jobs; persistent output validation gates downstream steps. Budget/queue failures produce partial tables with missing cells, never fabricated completion.

## Success and stopping rules

Produce reproducible tables, CSV/JSON raw records, coverage/failed-case accounting, code/data/model hashes and a readable report. Negative or mixed results are valid. Do not tune on test, inflate the strength of a category-median baseline, call programmatic preferences human alignment, equate training loss with quality, or label adapted code as the author's checkpoint. Stop automatic new job submission at the frozen budget; preserve pending work and explain its state.
