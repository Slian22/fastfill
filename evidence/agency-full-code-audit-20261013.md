# Agency project-owned code audit — 2026-10-13

## Decision and boundaries

The active J/P experiment is separate from FastFill v2 continuous-head training and legacy Qwen DPO. No finding here warrants mutating or cancelling training jobs **7365681 / 7365682** or the frozen **pipeline-9c4d97dbd736**. This auditor changed only audit evidence/docs, never production, frozen snapshots, credentials, datasets or jobs.

The highest current J/P concern is **VIS-01: visual resource hashes are recorded at freeze but not enforced against external resource bytes during consumption**. Parent independently reproduced acceptance of changed synthetic assets through full `prepare`. This is a prepublication integrity requirement, not evidence that real corpus assets changed. Geometry/data findings and the two metric fixes remain with their assigned reviewers; zero serialized-input issues does not establish clean raw geometry.

Legacy v2 still has a comparator/selection protocol limitation (**V2-01**), reduced-condition and postprocessing attribution limits (**V2-02**), and a narrower implemented architecture than the original aspirational method (**V2-03**). The current code does **not** support the blanket claim “selection is collapse-only”: both trainer selection and autorun use reference errors. The default evaluation-set leave-one-out baseline is a comparator fit on evaluation labels, not evidence that the neural model trained on test data.

## Coverage and reproducibility

Primary production review is **complete: 142 files /33,470 lines**, comprising **103 fresh full reads /23,218 lines** and **39 hash-validated prior full reads /10,252 lines**. **0 partial, 0 unreviewed, 0 missing; 0 source-hash mismatches** at 2026-10-13T07:52:56.484094+00:00. The earlier 22-file operations handoff is now fully closed by this auditor. Every fresh file has complete recorded line spans; every reused file has matching prior full-read evidence. Primary production inventory fingerprint: `41001ce24850738d793c052e40981ec138f7d05c1288c83c3c23fbcbd19b77f0`.

Tests remain separately owned. This auditor's **116 primary test** entries remain honestly classified as 0 fresh full, 14 reused, 2 partial and 100 unreviewed. Sartre's [separate test ledger](agency-full-test-source-coverage-20261013.json), observed at 2026-10-13T07:52:56.484094+00:00, lists **121 primary/delegated tests: 5 fresh full, 18 reused, 98 unreviewed**, status **in_progress**; all recorded test hashes matched then. These are separate coverage views, not additive counts. **Sartre's agent ID is unknown to this auditor**; no live agent status is asserted. Parent subsequently assigned the three unassigned v2 batches to Sagan/Confucius/Faraday; their forthcoming ledgers must be reconciled separately. No tests were reread to duplicate that work.

The complete executable-source inventory contains **444 files** across scopes. Remaining non-primary classifications below are explicit exclusions/delegations, not hidden coverage:

| Scope / role | Fresh full | Reused | Partial | Unreviewed |
|---|---:|---:|---:|---:|
| primary / production | 103 | 39 | 0 | 0 |
| primary / test | 0 | 14 | 2 | 100 |
| delegated_metric / production | 0 | 1 | 0 | 5 |
| delegated_metric / test | 0 | 0 | 0 | 5 |
| frozen_or_historical_copy / production | 0 | 0 | 0 | 128 |
| ancillary / production | 0 | 0 | 1 | 38 |
| ancillary / test | 0 | 0 | 0 | 4 |
| vendored_ancillary / production | 0 | 0 | 0 | 4 |

Every executable source file is listed with SHA-256, tracked/untracked status, role, scope and exact review class in [the coverage JSON](agency-full-code-coverage-20261013.json). `fresh_full` means all lines read at the recorded hash; `fresh_partial` means only the listed spans; `reused_hash_validated` means an explicit prior full-read assertion matched the current bytes; `unreviewed` means inventory/hash only. A test passing does not make its source fully reviewed.

Reuse evidence is `docs/evidence/fastfill-v2-audit-20261006/fastfill-current-read-coverage.json` and `docs/research/20261011-v11-clean-dpo/reaudit-coverage.json`, including linked dependencies and additional reviewed tests. **Every claimed reused file has an exact SHA-256 match (100% of the reused set); this is not a 100% repository-read claim.** Fresh reads were also rehashed. Manifest hashes and the source inventory fingerprint are in the ledger.

Discovery first attempted `codebase-memory-mcp.search_graph` for OptiScene; it was unindexed, so source discovery used `rg`. Inventory is sorted unique `git ls-files --cached --others --exclude-standard -z`, filtered to regular source files with the extensions recorded in the JSON. Root regular files contain no executable source outside `fastfill`. The ignored-source scan under `fastfill` returned zero matching source files. Dependency declarations and the 16 v2 JSON configs are inventoried separately (21 files); only the main experiment config was freshly read in full, other configs had model-key extraction only. Secrets/config credential stores were not printed.

Primary exclusions: `fastfill/comparison/{geometry,evaluate_pair,fid,final_tables,judge_batch}.py` and associated tests, plus `fastfill/v2/geometry.py`, are delegated metric scope. At the parent's follow-up request, `fastfill/v2/evaluate.py` was read completely to close its integration/selection coverage; no legacy numeric tables were recomputed. The four vendor roots `RoomGenBench`, `V-DETR`, `MinkowskiEngine`, `OptiScene_Original` are excluded from primary review. Vendor internals, actual mesh inventories and dataset contents were not fully read. Docs-side scripts and frozen copies remain separately classified; notably the snapshot-only `server_api_judge.py` transport has no fresh full-read claim here.

The 1,811-line revised design and 475-line spec comparison were consulted selectively, not reread in full. Explicit displayed spans are in `document_reads`; current J/P protocol/research-state were read in full. Source coverage and statement test coverage are different quantities.

## Era view: what was actually built and trained

| Era | Implementation and objective | Training evidence / claim limit | Findings |
|---|---|---|---|
| E1: legacy text placement | `fastfill/train.py`: Qwen causal-LM LoRA, assistant-token SFT; known object dimensions and text placement output. `serve.py`, `interface.py`, `merge_lora.py` form the earlier serving path. | Historical OptiScene Qwen checkpoint lineage is not reconstructed by source alone. Do not relabel it generic pretraining or the v2 continuous model. | OPS-01, IO-01 |
| E2: Oct 5–9 structured v2 | `v2/model.py`: Qwen `AutoModel` feature extraction/LoRA, pooled object spans plus learned slots, Transformer decoder and numeric position/size/yaw heads. `train.py` uses geometry supervision/matching. | Revised design records actual main v2 runs; main7 config has 2 decoder layers, dimension128, 16×16 position grid, 12 yaw bins, LoRA rank8, 6,357 planned steps. Run-completion and numeric results remain the parent's artifact audit. | V2-01–06, IO-02/03, OPS-02/03 |
| E3: Oct 11 legacy clean DPO | `v1_1/train_dpo.py` and build/audit modules apply preference training to the legacy causal-LM path. Preferences are programmatic geometry perturbation/repair judgments, not human ratings. | Reused full-read evidence is hash validated. Durable v1.1 research records are a separate legacy diagnostic; no equivalence to v2 training or J/P. | Separate provenance boundary |
| E4: Oct 13 matched unknown-size J/P | `comparison/train_pair.py`: two independently trained causal-LM LoRA SFT arms starting from the pinned generic Qwen3-8B revision. J predicts size/pose; P receives estimated sizes from component-held-out train priors and uses size-adjusted pose targets. Evaluation prior uses full train pool. | Protocol's primary endpoint is matched SFT J0/P0, not DPO. Current research-state records the protected jobs as running and final measured tables as pending; this is recorded state, not a live scheduler query by this auditor. | C-01, VIS-01 |

Git history supports these layers: current HEAD `8b94c437cbdacc5329b6abd04e26223fd0824e40` locks the Oct13 protocol, `243428016` preregisters Oct11 DPO, `aea29cf02` / `2e86188f8` separate the revised Oct9 design from the old-spec comparison, and Oct8 commits explicitly repair v2 matching/yaw/support issues. Newness alone was not used as correctness evidence.

`v2/text_sft.py` implements a separate text baseline; implementation and a tiny self-test are not proof a full Qwen baseline was trained. `v2/llm_baseline.py` / `llm_structured.py` call external language models and contain no optimizer. Do not count API prompting, old DPO and new matched SFT as one training contribution.

## Findings registry

All severities below are **Moderate**: concrete conditional defects or material scientific claim limits. No Critical claim is made. “Confirmed” identifies a verified mechanism, not demonstrated damage to the live experiment. Fix directions are proposals only.

| ID | Era / responsibility | Files | Type / status | Consequence |
|---|---|---|---|---|
| VIS-01 | E4 / visual provenance | `comparison/visual_cases.py`, `visual_pipeline.py`, `render_assets.py` | Consumer binding gap / Confirmed independently | Frozen selection can accept changed external mesh/material/texture bytes. |
| C-01 | E4 / serialized input contract | `comparison/train_pair_helpers.py`, `tests/test_train_pair.py` | Contract validation / Fixed locally by parent; frozen code unchanged | Extra matching target fields or invalid categories used to pass consumer pairing checks. Full input receipt finds no occurrence. |
| V2-01 | E2 / baseline fit and selection | `v2/autorun.py`, `evaluate.py`, `ops/run_plan.sh`, `train.py` | Protocol mismatch / Confirmed | Default comparison baselines fit evaluation labels; some selection paths inherit them. Trainer train-fitted comparator is different. |
| V2-02 | E2 / task definition and decoding | `v2/batch.py`, `evaluate.py`, `autorun.py` | Attribution/claim limit / Confirmed | Minimal-condition, spread-decoded results cannot establish full-condition or raw-model constraint ability. |
| V2-03 | E2 / model and runtime architecture | `v2/model.py`, `losses.py`, `runtime.py`, `adapters.py`, main config | Implementation-vs-spec limit / Confirmed | No explicit relation/support prediction loss, no point-cloud encoder, no persistent world/mesh/physics implementation. |
| V2-04 | E2 / validation units | `v2/validation.py` | Metres added to radians / Confirmed by synthetic reproduction | Changing spatial tolerance changes facing acceptance. |
| IO-01 | E1 / model publication | `merge_lora.py` | Recovery gap / Confirmed | Failure after moving the old model leaves the canonical output path absent. |
| IO-02 | E2 / dataset publication | `v2/qualified_data.py`, `multisource_data.py` | Partial publication / Confirmed | Interruption between per-file renames can leave an incomplete final directory that blocks rerun. |
| OPS-01 | E1 / HTTP serving | `serve.py` | Availability/security / Open deployment-specific | Slow incomplete request body can monopolize the single unauthenticated handler. |
| OPS-02 | E2 / resumed supervision | `v2/autorun.py` | State-existence gap / Open restart edge | An adopted running trainer with a missing log can crash supervision. |
| V2-05 | E2 / analysis cache provenance | `v2/ops/stratify.py` | Incomplete cache identity / Confirmed from source | Changed report metadata or fit-file bytes can return an earlier cached analysis. |
| V2-06 | E2 / relocated backbone provenance | `v2/ops/isambard_inputs.py`, `isambard_setup.sbatch` | Unsupported weight-identity claim / Confirmed from source | Config equality does not prove the substituted backbone has identical weight bytes. |
| IO-03 | E2 / shared request publication | `v2/ops/run_checkpoint.sh` | Concurrent rename race / Confirmed from source; runtime impact not reproduced | Concurrent exports refreshing stale requests can fail or nest request directories. |
| OPS-03 | E2 / legacy shell supervision | `v2/ops/rescore_llm.sh`, `run_structured.sh` | Failure status masking / Confirmed from source | Final success exit/marker can follow failed child generation or evaluation. |


### VIS-01 — prepublication requirement for the current visual path

`visual_cases.py:131–135` records OBJ/MTL/texture hashes; `:278–279` publishes `resources.json`. `visual_pipeline.py:121–140` uses the snapshot, whose `eval_data._snapshot` verifies **all manifest file bytes**, so the inventory JSON itself is protected. However, there is no equality check between its recorded resources and live files. `render_assets.py:406–447` plans from live resources; `:306` rereads the OBJ during preparation; `:298–302` links/copies live textures. `plan_receipt` (`:611–628`) records the newly inspected hashes. `_checked_images` (`visual_pipeline.py:204–225`) binds cameras, case/layout hashes and image bytes, but not those asset hashes to the frozen inventory.

Own CPU-only reproduction changed synthetic MTL bytes while preserving the case hash and was accepted by planning: [evidence](agency-visual-resource-repro-20261013.json). Parent independently ran real freeze/inspection/full prepare with changed synthetic mesh vertices and MTL, using mocked Blender RGB images: [full-prepare receipt](../docs/research/20261013-unknown-size-comparison/evidence/full-audit-resource-drift-probe.json). It reported one prepared/rendered case and two API-ready arm cases despite resource drift. No actual Blender output, external dataset mutation or API call is asserted; current source hashes match the receipt. This confirms consumer binding, not real-corpus corruption.

Parent's independent [point-in-time asset-byte receipt](../docs/research/20261013-unknown-size-comparison/evidence/full-audit-visual-resource-bytes.json), dated 2026-10-13T07:42:12.492224+00:00, verified **1,945 resources /6,558 OBJ, MTL and texture files /3,632,841,637 bytes (~3.38 GiB), zero mismatches** against the frozen reference. This is evidence of no observed drift at that inspection time, not a fix for VIS-01 or proof of future render-time binding. This auditor checked the receipt, not those asset bytes. Parent's newly inventoried standalone `verify_visual_resources.py` is ancillary, unreviewed here; its appearance is not a claim that this audit certified its implementation.

**Proposed standalone audit guard (no production patch in this audit):**

1. Pin the intended reference `MANIFEST.json` hash from the existing run identity; verify its `files['resources.json']` hash before parsing. Never create a new baseline from today's live bytes.
2. Build a canonical expected inventory of `(asset_id, resolved mesh path, mesh SHA256, MTL paths/hashes, texture paths/hashes, overrides)`. Reject conflicting entries. Apply any documented portability mapping explicitly; retain frozen path identity in the receipt.
3. Immediately before rendering, stream-hash every resource used by eligible cases and require equality to that inventory. Missing, added/substituted or changed resources fail the guard. Record the reference-manifest hash, resource-inventory hash and guard source hash in a separate audit receipt.
4. After all paired renders, repeat live-byte verification and require every render receipt's resource inventory to equal the frozen entries for its case (all reference/J/P/fixed-context assets). Gate FID/API/table publication on both checks. Existing image hashes/camera/layout checks still apply.
5. On mismatch, retain failed-case accounting and stop that visual publication; do not replace cases, refit models or overwrite frozen data. Correct assets can be restored/re-rendered into a new versioned visual artifact without repeating SFT. Parent owns any controller integration.

A before/after guard detects persistent drift and receipt mismatch; it cannot prove bytes were unchanged during rendering if a file changes and is restored between checks. Strong protection requires content-addressed copies or a read-only filesystem snapshot consumed by the renderer, including textures (ordinary symlinks are insufficient). Previously rendered outputs without such binding cannot be retroactively certified solely because current bytes match. This limitation must accompany any guard receipt.

### C-01 — independent review of the parent's exact movable allowlist

The only helper change against the protected snapshot is seven lines at `train_pair_helpers.py:144–150`: exact `{id, category}` for J and `{id, category, size}` for P, plus nonblank string category. Existing size exclusion/paired identity/answer validation remains. Current producer and inference contracts were traced separately; the two new ManifestTests (`test_train_pair.py:202–225`) exercise four shared-extra-field cases (`pos`, `yaw`, `on`, `description`) and four invalid-category cases (missing, empty, whitespace, integer).

Independent test execution: **42 tests passed, 99 subtests passed**. All eight bad cases were independently accepted by the frozen helper. [Review evidence](agency-parent-allowlist-review-20261013.json) records both helper hashes. Current helper hash `7968de59729f9227725679b9c8b4252b9ef3970a9d8c7f0d47e79a5cedebea83`; frozen helper hash `6a8268dc576f34a03d79e67e876417b01b7a7e8ce76241b5299b41e50176ea76`.

Euler's completed [full-input receipt](../docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-inputs-7367962-result.json) reports **231,474 serialized requests, issues `{}`**, with the scanner source hash checked. This auditor inspected the allowlist/full-population scanner spans and receipt, not the data. Scientific conclusion: preventive defense, no demonstrated contamination of frozen inputs and no basis here to stop/retrain the frozen arms. These tests cover the specified regression classes, not every request schema, GPU execution or all raw source semantics.

### V2-01 — evaluation-fitted baseline remains in the older selection path

`autorun.score:44–69` includes size/yaw/position error ratios and distribution/collapse penalties. `evaluate_candidates:249–313` invokes evaluation at `:290–292` without `--baseline-fit`; `evaluate.py:929–935` therefore calls `fit_baselines(samples)` and labels it `evaluation_set_leave_one_out`. `baseline_metrics:652–690` excludes the current **room**, leaving other evaluation rooms in the category statistics. `ops/run_plan.sh` also has default-fit head/CPU/LLM comparisons; its explicit training-file ablation path is distinct.

By contrast `train.py:649` fits baseline statistics from the first 20,000 training rows for trainer validation; selection at `:712` uses the geometry objective. Do not collapse these paths into one assertion. J/P `size_prior.py` uses explicit training identity/component folds and does not inherit this legacy issue.

The revised design already acknowledges the LOO baseline limitation. Before making a deployable-baseline or independent held-out selection claim, use pinned training-only statistics and version the resulting validation/selection reports. Historical LOO numbers must keep their label. No numeric report under `outputs/fastfill_v2/comparison-20261007b` or `eval-sets-20261009` was recomputed by this auditor.

### V2-02 — minimal-condition evaluation and rule-based decoding limit attribution

`batch.render_minimal_condition:137–156` drops fixed geometry, support and constraints; main augmentation sets `minimal_form_p=0.5`. Autorun selects through the minimal/spread path. `evaluate.serialize_predictions:340–380` calls `spread_grid_xy:121–328`, whose hand-coded rules spread collisions, impose support heights, place wall objects and flip floor-facing directions near walls. Those rules are disclosed in the function docstring, including cases where an intentional wall-facing object can be turned.

This is a structured model plus decoder system, not pure learned relational reasoning. Report raw argmax and spread outputs distinctly; full/minimal inputs are different tasks. The selection score does not test real asset resolution or mesh/physics/solver acceptance. The revised design's final limitations section is substantially candid; do not repeat the old spec as an implemented contribution.

### V2-03 — architectural claims must stop at the implemented boundary

The v2 Qwen feature backbone and slot decoder predict position, size and yaw. `losses.py` has no explicit relation/support head or target loss. Relations can enter the condition and affect geometry predictions, so “relations not trained” must mean **no explicit relation/support supervision objective**, not no conditional exposure. Main7 config lines40,45–46 set box/collision/boundary losses to zero; availability of optional losses is not evidence those main runs used them.

`CatalogResolver` filters category/capabilities and ranks log-size distance; it is not semantic text/visual retrieval. Runtime checks actual catalog dimensions rather than the oracle asset resized to a requested box. `AtomicMemoryHost` provides an in-process transaction abstraction; it is not a persistent WorldEdge integration. Required mesh/physics/solver levels have no implemented certifier and cannot be called successful merely because bbox checks passed. No V-DETR, sparse-convolution/Minkowski or point-cloud feature path appears in this v2 training implementation.

Defensible contribution: audited source adapters, task/label qualification and provenance, explicit structured-head geometry training/decoding, and a separately controlled J/P protocol. Do not claim architectural novelty, semantic relation learning, actual asset-level deployability or benchmark superiority from this source review alone.

### V2-04 — spatial slack changes angular acceptance

`validation.py:335` takes `tolerance_m`; `:395` passes it to `_constraint`; facing at `:284` tests `error <= tolerance_rad + tol`. [Synthetic evidence](agency-angular-tolerance-repro-20261013.json): an unchanged angular error0.05rad against allowed0.01rad fails with0.0001m slack but passes with0.1m. Default effect is small; impact on existing tables was not recomputed. Separate angular epsilon from metre slack and add a mixed-units regression before changing future validation code.

### IO-01 / IO-02 — publication and recovery

`merge_lora.py:68–75` moves existing output to a backup, then promotes the temporary directory without rollback if the second replace fails. Old bytes survive in backup but the canonical model path can disappear; “atomic” overstates replacement semantics. Active J/P adapter inference does not use this merger. Proposed: restore backup on failed promotion and test failure between operations.

`qualified_data.py:190–192` and `multisource_data.py:508–510` create final output then rename files individually. A crash can expose a partial directory which `safe_output` refuses to reuse. `legacy_build.py:238–243` and `cohort.py:145–150` instead rename the staged directory. Proposed: same-filesystem directory publication and explicit recovery. No corrupt completed dataset is inferred; consumer hash checks generally fail closed on missing/mismatched artifacts.

### OPS-01 / OPS-02 — runtime conditions, not live-run allegations

Legacy `serve.py:64–81,107,125` binds all interfaces by default and synchronously reads a declared body on single-thread `HTTPServer`, without an application read deadline/authentication. A body within the1MiB cap can be sent incompletely and monopolize the handler; exception details may reveal local paths. No evidence it is exposed by current batch jobs. Before exposure, use loopback/proxy authentication and bounded reads/generation. v2 `serve.py` is an offline runner and must not inherit this network finding.

`autorun.py:197–203` can adopt an existing live trainer from output state; `:215` stats its log without an existence guard. Missing/rotated log raises and stops supervision while training may remain alive. Proposed: disclose missing log and retain process supervision instead of assuming the file exists. Do not modify any protected run to test this.

### V2-05 / V2-06 — source-update and artifact-identity gaps in legacy analysis

`ops/stratify.py:214–241` reads `report.json` and accepts an optional training-fit file, but its cache key at `:225` includes only evaluation directory, outcomes filename/content hash, rows hash and Python code hashes. A cache hit at `:228` returns before baseline reconstruction. Changing `report.json` metadata (for example projection or subset limit) or bytes at the same `baseline_fit_file` path does not invalidate the key if outcomes/rows/code remain unchanged. Cached summaries and reproduction checks can therefore describe the earlier inputs. **V2-05** is an E2 identity-contract gap, not proof any published number is affected. Proposed: bind the complete report hash and referenced fit-file hash, record the fit hash at evaluation time, version the cache, and add a regression that changes each omitted input independently.

`ops/isambard_inputs.py:14–17` and `:253` describe the relocated backbone as the same weights, while `baseline_copy:145–149` verifies only `config.json`. `ops/isambard_setup.sbatch:24–25` skips pinned download whenever the local weight-index file exists and checks only the configuration hash afterward. Identical architecture configuration can accompany different or incomplete weight shards. **V2-06** is a provenance/claim gap, not observed weight corruption; adapter/checkpoint download pins are separate and do not establish base-weight identity. Proposed: record and enforce all referenced base-weight shard hashes and the immutable model revision before relocation/reuse, or narrow the assertion to configuration equality. This legacy setup is distinct from the new J/P protocol; no claim is made that J/P uses this weak check.

### IO-03 / OPS-03 — legacy operations can fail outside the happy path

`ops/run_checkpoint.sh:10–16` explicitly documents the concurrent refresh race: two exports can both find shared requests stale, move the directory and promote separate replacements without a lock. One can fail or produce a nested directory. Initial setup avoids a race only while requests stay unchanged; later source/scene updates can trigger it. **IO-03** was confirmed by source inspection, not by racing real exports. Proposed: use a lock or content-addressed immutable request directories and record the exact directory/hash per export.

`ops/rescore_llm.sh` uses `set -uo pipefail`; each failed evaluator is handled by an echo and the script finishes with `RESCORE-DONE`, allowing exit zero. `ops/run_structured.sh` likewise continues after generation failure, uses argument-free `wait`, and finishes with `ALL-DONE`; a failed child does not reliably fail its parent. **OPS-03** is a source-confirmed success-status gap; no legacy job was launched to reproduce it. Proposed: capture each child PID/status, fail on any required stage and validate output identity/completeness before declaring success. These are older operational wrappers, not the active J/P controller.

## Responsibility view: intentional differences and interface seams

| Responsibility | Implementations / interface | Audit conclusion / IDs |
|---|---|---|
| Input/output representation | IR adapters → `scene/build` → comparison `contract/main_data/eval_data`; v2 schema/batch/legacy_bridge is a separate representation | IR metres/radians/bottom-centre and full XYZ dimensions; text requests use metre numbers and degree yaw. Full source adapter review; C-01. |
| Size estimation / prediction | v2 continuous exp-size head; v2 label baseline; J joint token generation; P train-only log-median prior | Different intended methods, not interchangeable duplicate helpers; V2-01, V2-03. |
| Placement validity / repair | legacy interface, v1.1 postprocessing, v2 validation/runtime/spread, comparison pose retargeting | Shared geometry concepts but different contracts/purposes. Explicitly distinguish target qualification, generated output, decoded output and actual asset validity; V2-02–04. |
| Resource retrieval / rendering | v2 CatalogResolver versus comparison GT-sidecar asset lookup and resized real-mesh rendering | Intentionally distinct: comparison visuals do not prove deployable retrieval. Frozen external-byte binding missing; VIS-01. |
| Publication / immutable artifacts | comparison snapshots and write-once receipts; older per-file v2 builders; legacy merged models and shared request refresh | Comparison hash gating checked; IO-01/02/03 identify older weaker recovery/concurrency paths. |
| Analysis cache / model relocation | v2 stratify cache, Isambard input pinning and local backbone reuse | Code/rows/config bindings do not cover every consumed artifact; V2-05/06. |
| Supervision | comparison Slurm dependencies/overnight receipts; v1.1 automation; v2 autorun; legacy HTTP | Separate eras/deployments. OPS-01/02/03 do not imply current J/P failure. |
| External project interfaces | `v2/roomgenbench.py`, `v1_1/roomgenbench_v1_1.py`, `v2/direct_layout.py`, canonical source adapters | Project-owned wrappers reviewed fresh or hash-reused. RoomGenBench internals/official checkpoints not certified. V-DETR/Minkowski/OptiScene_Original are historical/vendor boundaries, not demonstrated active v2 encoders. |

Possible unused scaffolding: `comparison/train_pair._release_gpu` has no caller found in current project source (training now isolates processes). `matching.permute_relations` is referenced by tests but no relation-training head; this is capability scaffolding, not proof of trained relations. Legacy modules still imported by v1.1 are not safe-to-delete merely because a newer comparison exists. No cleanup performed.

## Standalone state-existence / ordering pass

This pass was separate from similarity/duplication review. “Checked” is source evidence, not a live end-to-end execution certificate.

| Handler/job | State read and creator | Guarantee / finding |
|---|---|---|
| comparison corpus workers / main data | ordered rows, original identity, prior/fold state produced before task dispatch | Explicit identity/hash checks and ordered future consumption; manifest publication after successful work. Checked, no missing-state issue found. |
| train_pair | pinned data/base/tokenizer, child training outputs and adapter files | Preflight validation, subprocess join/exit checks, saved-training marker and generation/reload probes precede completion receipt. Checked. |
| infer | successful training receipt, adapter hashes, evaluation UIDs and token budgets | Explicit status/hash/UID checks, paired context policy and write-once output. Checked; no assumption that scheduled training equals completion. |
| overnight | Slurm job status, downloaded artifacts, downstream manifests | Completed-success dependency checks, verified fetches, bounded polling/cutoffs, wait on submitted futures; uncertain submission preserved to avoid duplicates. Checked in current source. New parent controller is separately reviewed, not certified here. |
| visual pipeline | reference selection, geometry outputs, external resources, image receipts | Case/manifest/camera/image checks exist. External resource predecessor guarantee is absent: VIS-01. |
| v2 trainer / evaluation | worker context, tokenizer, checkpoint config, labels and head outputs | Worker initializer installs context before jobs; checkpoint/config checks and barriers precede dependent work. Evaluation tests supplied row counts and stored condition/IDs. No GPU runtime run by auditor. |
| v2 runtime commit | world version, idempotency map, resolved assets | `AtomicMemoryHost` lock/version/idempotency checks guard memory transaction. Checked; external persistent world is unimplemented. |
| v2 autorun | prior output/resume, process, log, generated data and export | Model/input hash/wait checks exist; adopted log existence is not guaranteed: OPS-02. |
| legacy/v2 publishers | prior model and staged dataset files | IO-01/02 recovery windows; no claim that mere output-directory existence means a complete result. |
| v2 stratify workers/cache | rows/fit/projection state and prior analysis cache | Parent installs `_STATE` before fork and runs initializer before room jobs: checked. Cache identity omits report/fit bytes: V2-05. |
| v2 Isambard setup/analysis | pinned inputs, two completed arm statuses, cohorts, selected score and backbone | Arm completion/cohort/code checks precede selection; `done-with-failures` retains recorded failures, not a clean-science certificate. Local backbone equality remains V2-06. |
| v2 shared export/shell wrappers | requests published by setup/another export; asynchronous child outputs | Source updates reopen shared rename race IO-03; child failure propagation is incomplete OPS-03. |
| v2 verification | immutable parents, source pins, output rows/journal | `legacy_verify` checks original selected identities, source hashes and counts; `multisource_verify` reconstructs transformations, journals and before/after hashes. Source review only; no dataset scan run. |

v1.1 ordering correctness is inherited only through the cited hash-validated prior full review; no fresh all-handler claim. Unreviewed ancillary controllers/API transport and excluded vendor event paths are not certified.

## Standalone units / representation pass

| Value chain | Checked representation | Result |
|---|---|---|
| Canonical IR → training targets | metres; full local dimensions; bottom-centre Z; radians | Reviewed adapter conversions preserve that contract. Qualification masks distinguish unreliable geometry from valid numbers. |
| Holodeck / Structured3D | Holodeck cm→m and source-axis/front rotation; Structured3D half-extents mm→full metres, centre→bottom | Explicit one-time transforms; no duplicate conversion found in traced consumers. |
| HSSD / MultiScan / source mesh frames | Y-up basis and pivot/scale handling; half/full extents; estimated floor metadata | Code and synthetic adapter checks only. Actual full mesh corpus was not inspected; source metadata is not independent physical truth. |
| SAGE / SceneSmith | Parent/pivot and source-specific settling/lift policies | Intentionally different source contracts, not automatically duplicate-logic bugs. SceneSmith's source ordering remains an adapter assumption; no universal source-order guarantee claimed. |
| J/P contract → raw layout → render | input metre values; output position/size rounding to2 decimals; integer degrees converted to radians for IR/render | Current call sites consistent; P's exp(log-median) returns metres with positive floor and rounding. This does not certify raw geometry quality. |
| v2 normalized position / exp size / yaw bins | position uses room origin/scale; size ratios/log errors dimensionless; decoded yaw radians | Traced model, losses, batch, predict, serialize and runtime; raw versus spread distinguished. |
| v2 facing tolerance | radian error/tolerance plus metre slack | Confirmed mismatch V2-04. |
| Durations | hours×3600, monotonic seconds, reported milliseconds×1000 | No inconsistent conversion found in the reviewed call paths. |

## Source-update compatibility

The ledger compares local Python bytes against `pipeline-9c4d97dbd736`. Differences are the parent's helper fix and the two delegated metric fixes; snapshot-only `server_api_judge.py` has no current-root counterpart. A new parent `pipeline-full-audit-*` copy appeared during inventory refresh and is separately classified, not silently substituted for the protected baseline. No live server source was changed by this auditor.

A local source fix cannot retroactively fix code loaded by a frozen job. Future stages must consume a coherent versioned source snapshot with matching data/model/metric manifests; retain the old snapshot identity on old artifacts. Parent's newly gated controller owns deployment of the three authorized patches. This report does not duplicate its delta review or imply that its authorization covers other proposed fixes.

V2 exports, head-output redecodes, LLM reports and fit files also need versioned bindings: `evaluate.py` validates stored condition/IDs but only records an optional head manifest, and baseline-fit metadata is a path label rather than an independent content hash. Treat unchanged filenames as insufficient evidence of unchanged scientific inputs. The completed ops review identifies omitted cache inputs (V2-05), a weaker-than-claimed backbone check (V2-06), concurrent request refresh (IO-03), and masked shell failure (OPS-03). Autorun intentionally permits consistently old evaluator caches across candidates; do not call that deliberate historical comparison a fresh-source rerun. Source hashes alone cannot establish unchanged model/data bytes.

## Verification and remaining limits

- Auditor: independent helper test module **42 passed /99 subtests**; eight frozen counterexamples reproduced; two CPU synthetic repros for angular tolerance and resource drift; eleven adapter module self-checks exited0 (`internscenes`, `multiscan`, `scan2cad`, `spatiallm`, `structured3d`, `scenesmith`, `sage10k`, `mansionworld`, `interiorgs`, `hssd200`, `il3d_3dfront`). Synthetic assets only, no full data/mesh scans.
- Parent receipt: **681 comparison tests passed,90.874448% statement coverage**, from [final validation](../docs/research/20261013-unknown-size-comparison/evidence/full-audit-final-validation.json). All six listed patch hashes matched local files at final validation. This is attributed evidence, not an auditor rerun of the metric patch.
- Euler receipt: all serialized requests scanned, issues empty. Raw geometry errors/exposure and numerical scientific results remain in the parent's/data/metric reports.
- Targeted credential-pattern scan of primary source found no recognized key/private-key markers; this is not a comprehensive security certification. No credential value printed, remote/API call made, real render run or training launched by this auditor.
- No independent subagent could be spawned because the thread limit persisted. VIS-01 obtained independent parent confirmation; other noncritical findings retain their stated source/synthetic limits.

## Risk-priority view and proposed order

1. **Before current visual publication — VIS-01 (E4, visual provenance):** require pinned-resource comparison and receipt binding, disclose TOCTOU limitation. No retraining needed. C-01 is locally fixed and fully scanned inputs have zero reported exposure; preserve frozen jobs.
2. **Before legacy v2 scientific claims — V2-01/02/03 (E2, selection/task/model):** label evaluation-fitted baselines, separate raw/spread/full/minimal, and limit architectural claims to implemented/trained components. Parent owns numeric report verification. Bind cache/fit/backbone identities before rerunning or relocating old results (V2-05/06).
3. **Future correctness/recovery patches — V2-04, IO-01/02/03, OPS-02/03:** separate units, publish atomically with recovery/locking, guard adopted missing logs and propagate child failures; propose scoped regressions before source changes.
4. **Before deploying legacy HTTP — OPS-01:** bound and authenticate service access. Deployment/exposure is unconfirmed; current batch jobs are separate.
5. **Cosmetic/document cleanup:** remove stale trainer comments referring to collapse anchors as the selection score; clarify unused `_release_gpu` and future-only relation scaffolding. These are not Critical risks.

## Per-file primary production coverage

### fresh_full (103)

- `fastfill/__init__.py` (0 lines)
- `fastfill/adapters/__init__.py` (57 lines)
- `fastfill/adapters/__main__.py` (3 lines)
- `fastfill/adapters/ase.py` (46 lines)
- `fastfill/adapters/hssd200.py` (424 lines)
- `fastfill/adapters/il3d_3dfront.py` (394 lines)
- `fastfill/adapters/il3d_synthetic.py` (150 lines)
- `fastfill/adapters/interiorgs.py` (369 lines)
- `fastfill/adapters/internscenes.py` (183 lines)
- `fastfill/adapters/internscenes_3rscan.py` (12 lines)
- `fastfill/adapters/internscenes_arkit.py` (14 lines)
- `fastfill/adapters/internscenes_mp3d.py` (24 lines)
- `fastfill/adapters/internscenes_scannet.py` (10 lines)
- `fastfill/adapters/m3dlayout_infinigen.py` (54 lines)
- `fastfill/adapters/mansionworld.py` (234 lines)
- `fastfill/adapters/multiscan.py` (120 lines)
- `fastfill/adapters/optiscene_holodeck.py` (45 lines)
- `fastfill/adapters/sage10k.py` (255 lines)
- `fastfill/adapters/scan2cad.py` (96 lines)
- `fastfill/adapters/scannetpp.py` (45 lines)
- `fastfill/adapters/scenesmith.py` (129 lines)
- `fastfill/adapters/spatialgen.py` (19 lines)
- `fastfill/adapters/spatiallm.py` (177 lines)
- `fastfill/adapters/structured3d.py` (130 lines)
- `fastfill/adapters/unified.py` (112 lines)
- `fastfill/comparison/__init__.py` (5 lines)
- `fastfill/comparison/contract.py` (288 lines)
- `fastfill/comparison/corpus.py` (334 lines)
- `fastfill/comparison/data.py` (263 lines)
- `fastfill/comparison/eval_data.py` (280 lines)
- `fastfill/comparison/infer.py` (385 lines)
- `fastfill/comparison/isambard_geometry.sbatch` (24 lines)
- `fastfill/comparison/isambard_infer.sbatch` (45 lines)
- `fastfill/comparison/isambard_main.sbatch` (41 lines)
- `fastfill/comparison/isambard_pilot.sbatch` (39 lines)
- `fastfill/comparison/main_data.py` (401 lines)
- `fastfill/comparison/overnight.py` (723 lines)
- `fastfill/comparison/pose_targets.py` (330 lines)
- `fastfill/comparison/render_assets.py` (746 lines)
- `fastfill/comparison/size_prior.py` (236 lines)
- `fastfill/comparison/train_pair.py` (459 lines)
- `fastfill/comparison/train_pair_helpers.py` (400 lines)
- `fastfill/comparison/visual_cases.py` (351 lines)
- `fastfill/comparison/visual_pipeline.py` (418 lines)
- `fastfill/download.py` (116 lines)
- `fastfill/download_full.sh` (27 lines)
- `fastfill/inspect_ir.py` (87 lines)
- `fastfill/merge_lora.py` (88 lines)
- `fastfill/serve.py` (188 lines)
- `fastfill/tools/__init__.py` (0 lines)
- `fastfill/tools/ablation_check.py` (48 lines)
- `fastfill/tools/qa_report.py` (420 lines)
- `fastfill/tools/visualize.py` (146 lines)
- `fastfill/v2/adapters.py` (165 lines)
- `fastfill/v2/autorun.py` (616 lines)
- `fastfill/v2/batch.py` (487 lines)
- `fastfill/v2/cohort.py` (163 lines)
- `fastfill/v2/data.py` (127 lines)
- `fastfill/v2/evaluate.py` (1079 lines)
- `fastfill/v2/io.py` (163 lines)
- `fastfill/v2/legacy_bridge.py` (285 lines)
- `fastfill/v2/legacy_build.py` (267 lines)
- `fastfill/v2/legacy_verify.py` (612 lines)
- `fastfill/v2/llm_baseline.py` (261 lines)
- `fastfill/v2/llm_structured.py` (320 lines)
- `fastfill/v2/losses.py` (239 lines)
- `fastfill/v2/matching.py` (193 lines)
- `fastfill/v2/model.py` (269 lines)
- `fastfill/v2/multisource_data.py` (536 lines)
- `fastfill/v2/multisource_verify.py` (685 lines)
- `fastfill/v2/ops/compare.py` (189 lines)
- `fastfill/v2/ops/crosscheck.py` (50 lines)
- `fastfill/v2/ops/diag_ablation.py` (372 lines)
- `fastfill/v2/ops/isambard_analysis.sbatch` (19 lines)
- `fastfill/v2/ops/isambard_autorun.sbatch` (17 lines)
- `fastfill/v2/ops/isambard_dryrun.sh` (123 lines)
- `fastfill/v2/ops/isambard_inputs.py` (289 lines)
- `fastfill/v2/ops/isambard_setup.sbatch` (87 lines)
- `fastfill/v2/ops/isambard_submit.sh` (10 lines)
- `fastfill/v2/ops/llm_compare.py` (190 lines)
- `fastfill/v2/ops/make_sample.sh` (153 lines)
- `fastfill/v2/ops/rebuild_and_publish.sh` (47 lines)
- `fastfill/v2/ops/report.py` (639 lines)
- `fastfill/v2/ops/rescore_llm.sh` (11 lines)
- `fastfill/v2/ops/run_checkpoint.sh` (24 lines)
- `fastfill/v2/ops/run_plan.sh` (322 lines)
- `fastfill/v2/ops/run_reference.sh` (9 lines)
- `fastfill/v2/ops/run_structured.sh` (15 lines)
- `fastfill/v2/ops/sample_validation.py` (116 lines)
- `fastfill/v2/ops/selftest_shift.py` (55 lines)
- `fastfill/v2/ops/start_autorun_retrain.sh` (12 lines)
- `fastfill/v2/ops/stratify.py` (415 lines)
- `fastfill/v2/ops/vlm_judge.py` (533 lines)
- `fastfill/v2/predict.py` (113 lines)
- `fastfill/v2/qualified_data.py` (217 lines)
- `fastfill/v2/regularizers.py` (100 lines)
- `fastfill/v2/review_data.py` (321 lines)
- `fastfill/v2/runtime.py` (448 lines)
- `fastfill/v2/schema.py` (324 lines)
- `fastfill/v2/serve.py` (87 lines)
- `fastfill/v2/text_sft.py` (242 lines)
- `fastfill/v2/train.py` (809 lines)
- `fastfill/v2/validation.py` (403 lines)

### reused_hash_validated (39)

- `fastfill/anchors.py` (239 lines)
- `fastfill/build.py` (623 lines)
- `fastfill/evaluate.py` (230 lines)
- `fastfill/interface.py` (641 lines)
- `fastfill/scene.py` (379 lines)
- `fastfill/split.py` (106 lines)
- `fastfill/train.py` (283 lines)
- `fastfill/v1_1/__init__.py` (1 lines)
- `fastfill/v1_1/audit_pairs.py` (245 lines)
- `fastfill/v1_1/audit_report.py` (159 lines)
- `fastfill/v1_1/automate.py` (323 lines)
- `fastfill/v1_1/build_dpo.py` (446 lines)
- `fastfill/v1_1/build_v1_1.py` (215 lines)
- `fastfill/v1_1/evaluate_v1_1.py` (687 lines)
- `fastfill/v1_1/evaluation_protocol.py` (279 lines)
- `fastfill/v1_1/isambard_v1_1_audit.sbatch` (60 lines)
- `fastfill/v1_1/isambard_v1_1_clean_dpo.sbatch` (63 lines)
- `fastfill/v1_1/isambard_v1_1_evaluate.sbatch` (42 lines)
- `fastfill/v1_1/isambard_v1_1_pipeline.sbatch` (273 lines)
- `fastfill/v1_1/isambard_v1_1_verified_evaluate.sbatch` (51 lines)
- `fastfill/v1_1/postprocess.py` (644 lines)
- `fastfill/v1_1/research_report.py` (262 lines)
- `fastfill/v1_1/roomgenbench_v1_1.py` (529 lines)
- `fastfill/v1_1/scene_v1_1.py` (320 lines)
- `fastfill/v1_1/train_dpo.py` (409 lines)
- `fastfill/v1_1/training_guard.py` (327 lines)
- `fastfill/v1_1/verify_outputs.py` (263 lines)
- `fastfill/v2/__init__.py` (3 lines)
- `fastfill/v2/audit.py` (247 lines)
- `fastfill/v2/bbox_visualization.py` (127 lines)
- `fastfill/v2/boxes.py` (123 lines)
- `fastfill/v2/direct_layout.py` (373 lines)
- `fastfill/v2/legacy_evidence.py` (157 lines)
- `fastfill/v2/minimal_data.py` (313 lines)
- `fastfill/v2/objective.py` (61 lines)
- `fastfill/v2/roomgenbench.py` (444 lines)
- `fastfill/v2/size_range.py` (38 lines)
- `fastfill/v2/verify_bundle.py` (76 lines)
- `fastfill/validate.py` (191 lines)

### fresh_partial (0)

None.

### unreviewed (0)

None.

