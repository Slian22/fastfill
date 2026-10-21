# Test-source audit v2-1 — 2026-10-13

Completed **26/26 files, 4,356/4,356 lines** in **30 bounded display chunks**. Every file is `fresh_full`; initial and final SHA-256/line checks matched **26/26**, with **0 gaps, overlaps or mismatches**. Found **4 concrete test-assertion weaknesses** and **6 coverage limitations**. Production defects were not assessed.

## Scope and method

Static full-source review only. Every line, including fixtures/helpers, comments and blank lines, was displayed with line numbers in bounded chunks after SHA-256 verification. No project modules were imported, no tests were collected or executed, no remote calls were made, and no other batch or assigned comparison test source was read. Imported out-of-batch helpers were treated as opaque. Findings were cross-checked only against the 26 permitted sources.

Source inventory: `evidence/agency-full-test-source-coverage-20261013.json`, only `delegation_ready_batches[batch == 'v2-1'].paths`. Its SHA-256 at selection was `22eb3fa5c68b63e62230e0301a5936ce97b839ef23f17b79b00e4c095099d4fc`; selected records were unreviewed and the batch was unassigned. Source discovery was already complete; the user reported the prior graph attempt/root as unindexed. The shared ledger was not edited.

Every line includes fixtures, helpers, comments and blanks. Imported helpers outside this batch were not opened. SHA-256 was checked before each displayed chunk and recomputed after review at **2026-10-13T07:58:17.580142+00:00**, together with UTF-8 `splitlines()` counts. The JSON records actual displayed spans, individual review summaries and per-file final hashes.

This is a deterministic source census, not a test-results sample. Confidence intervals, significance, pass rates, runtime coverage and release confidence are **not estimable** from this task. No tests were collected or executed. Writes are limited to this report and its JSON companion.

## Concrete findings

All four findings concern source-visible test oracles, **not confirmed production failures**. Priority is medium (review prioritization, not an estimated failure probability). Counterexamples below are static reasoning; no mutation tests were run.

### V21-F1: Canonical OBB round-trip oracle bypasses the returned yaw

Evidence: `fastfill/v2/tests/test_data.py:47–57`; `fastfill/v2/tests/test_data.py:32–37`.

The arbitrary-angle test builds source corners with the input axes and transforms them back with those same axes. It checks returned size and bottom center, but never uses or asserts box['yaw_rad']. Only the separate axis-permutation example asserts the returned yaw at -pi/2.

**Escape scenario:** Returning an incorrect yaw for an oblique input such as 0.7 radians, while retaining the correct size and bottom center, would not affect this round-trip test's assertions. This is a static counterexample, not an executed mutation.

**Claim affected:** Correct recovery of source orientation and preservation of the canonical world-space box.

**Follow-up:** Reconstruct world corners from returned bottom_center_m, target_size_local_m and yaw_rad; compare them with independently constructed source corners, and assert the semantic-front direction at oblique and wrap-boundary angles.

**Scope:** Independent exporter corner tests exercise supplied yaws, not canonical_obb's recovered yaw. No claim is made about tests outside v2-1 or a production yaw defect.

### V21-F2: Related-scan split regression never constructs related scans

Evidence: `fastfill/v2/tests/test_data.py:170–172`; `fastfill/v2/tests/test_data.py:146–161`; `fastfill/v2/tests/test_data.py:182–195`.

The named related-scans test compares grouped_split('MultiScan','house',42) with the same call and checks membership in the three split names. The nearby builder fixtures write only one eligible sample, so they cannot establish cross-row household grouping.

**Escape scenario:** A deterministic splitter returning one constant valid split passes the named test. A builder grouping different scans from one house by scan identity is also not challenged by these one-sample build assertions. These are static counterexamples.

**Claim affected:** Evidence that correlated scans cannot leak across training, validation and test partitions.

**Follow-up:** Build multiple distinct scans for each of multiple house IDs, inspect every output split, assert each house occupies only one split, and repeat with permuted source ordering. Check expected split decisions or a deliberate multi-split fixture separately from determinism.

**Scope:** Existing holdout-guard tests cover explicit benchmark markers; they do not make this named grouping test discriminating. Other split suites were not inspected; no actual leakage is established.

### V21-F3: LLM repair test does not verify that repaired geometry is persisted

Evidence: `fastfill/v2/tests/test_llm_baseline.py:48–66`; `fastfill/v2/tests/test_llm_baseline.py:20–35`.

The harness fixture returns a stacked layout and then the target-derived good answer. Assertions check two calls, overlap feedback, repair_rounds==1 and problems_final<=mean_problems_first, but never inspect prediction['layout'] against the good answer. Equality of the problem counts is allowed.

**Escape scenario:** The harness could retain the original stacked geometry after requesting a repair, while recording one repair and an unchanged problem count, without violating this test's assertions. The separate exact-position/size round-trip runs the default prompt path, not this repair path.

**Claim affected:** Evidence that harness repairs improve saved layouts and that later geometry metrics score the repaired result.

**Follow-up:** Assert the serialized layout matches the second answer by object ID for position, size and yaw; recompute its geometry checks and compare the remaining problems with the known-good result.

**Scope:** This establishes a missing oracle in the test, not an observed harness persistence bug.

### V21-F4: Model checkpoint round trips compare only size predictions

Evidence: `fastfill/v2/tests/test_model.py:92–98`; `fastfill/v2/tests/test_model.py:124–147`; `fastfill/v2/tests/test_train_loop_fix20261006.py:284–294`.

Both the tiny checkpoint round trip and the local random Qwen LoRA/full/frozen round trips compare only the 'size' output. The intermediate-export check compares two loaded checkpoints and validates layout schema, rather than comparing every geometry head with the original in-memory model.

**Escape scenario:** A save/load regression affecting only yaw or position heads can leave the size comparison unchanged. Comparing two exports that share the same serialization omission does not independently verify the original head values.

**Claim affected:** Faithful checkpoint preservation of the complete geometry predictor, particularly position and orientation metrics.

**Follow-up:** Capture all original geometry outputs with deliberately nondefault head values and compare position, size, yaw logits/residuals, slot masks and grid-head outputs after reload for applicable backbones/head modes.

**Scope:** Resume tests do compare complete saved state dictionaries and trajectories, which is useful distinct coverage. This finding is limited to the named original-model-to-loaded-model round-trip oracles; no serialization failure was executed or confirmed.

## Coverage limitations

### V21-L1: Static source closure does not establish execution coverage or runtime success

The 26-file/4,356-line census closes only the delegated fresh-source-read gap. No tests, imports, collection, mutation tests, measured coverage or benchmark runs were performed. Production code and imported helpers outside the batch remain unreviewed in this task.

**Claim boundary:** No pass/fail count, 80% execution-coverage claim, defect probability, confidence interval or release-readiness conclusion follows from this report.

**Follow-up:** Use separately authorized execution evidence and other auditors' reports to assess those claims; do not treat a fresh_full label as a passing test.

### V21-L2: Backend and benchmark evidence is conditional on dependencies and fixtures

Evidence: `fastfill/v2/tests/test_model.py:124–147`; `fastfill/v2/tests/test_model_review_fixes.py:101–145`; `fastfill/v2/tests/test_regularizer_support_review.py:79–94`; `fastfill/v2/tests/test_accumulation_tail.py:198–206`; `fastfill/v2/tests/test_roomgenbench_fix20261006.py:47–54`; `fastfill/v2/tests/test_round14_eval_math.py:242–300`.

Qwen/PEFT tests use importorskip and a local random small model; MPS/Gloo tests have capability gates. RoomGenBench checks depend on imported external reference paths; two reference-metric tests skip when BATHROOM is absent. The restaurant test gates on BATHROOM, then reads restaurant.json without its own availability check.

**Claim boundary:** These source assertions do not prove that device branches or external benchmark checks ran, or that results generalize to a formal large-model/GPU configuration. External reference content and pinning were not inspected.

**Follow-up:** Record actual skip reasons, backend versions and exact benchmark fixture revisions during a later authorized run; check each required fixture explicitly.

### V21-L3: Spread decoding examples do not cover exhausted placement choices or concave room containment

Evidence: `fastfill/v2/tests/test_spread_decode.py:7–61`; `fastfill/v2/tests/test_round14_eval_math.py:22–107`; `fastfill/v2/tests/test_leak_guards_fix20261006.py:66–93`.

The explicitly constructed spread fixtures exercise feasible rectangular bounds, near-wall placement, modulo-pi facing and a rotated support top. These cases have no explicit concave-decoder containment oracle or inventory for which every candidate collides/is outside. L-shaped/non-convex cases inspect exported shell geometry, a different path. Imported real_rows helpers in online replay tests were not opened, so their broader fixture domain is not asserted here.

**Claim boundary:** Collision-aware spreading in these fixtures is not a guarantee of collision-free, contained layouts for infeasible or non-convex scenes.

**Follow-up:** Add adversarial infeasible and concave-room decode fixtures with explicit inventory-preservation, collision/containment diagnostics and failure/fallback expectations.

### V21-L4: Language-baseline tests establish plumbing, not comparative research quality or exhaustive prompt isolation

Evidence: `fastfill/v2/tests/test_text_sft.py:10–47`; `fastfill/v2/tests/test_llm_baseline.py:10–35`; `fastfill/v2/tests/test_llm_baseline.py:101–143`.

Text SFT tests cover a shared condition prefix, label masking, differentiability and a dry-run generation that need only be a string. LLM success fixtures supply answers derived from targets to test conversion. The prompt guard checks only six-decimal strings of target x coordinates, not every geometry field or alternate numeric spelling. Retry accounting is explicitly checked.

**Claim boundary:** No text-vs-structured quality comparison, real LLM success rate or exhaustive target-leakage claim is established by these fixtures.

**Follow-up:** Keep smoke/conversion evidence distinct from research evaluation; add structural prompt allowlists and target-perturbation invariance checks, then evaluate real saved predictions under a documented shared cohort and budget when authorized.

### V21-L5: Autorun acceptance and cache fixtures support narrower claims than full request success and artifact identity

Evidence: `fastfill/v2/tests/test_autorun.py:100–172`; `fastfill/v2/tests/test_autorun.py:209–240`; `fastfill/v2/tests/test_autorun.py:243–303`; `fastfill/v2/tests/test_autorun.py:386–407`.

The acceptance fixture explicitly accepts 5 inference failures out of 10 when all 5 are over capacity, while reporting them. Candidate checkpoints are empty directories and reports are mocked with checkpoint paths; the handoff reuse test keeps the checkpoint path fixed and changes only implementation identity. In-place checkpoint-byte changes are not exercised.

**Claim boundary:** Accepted status is not evidence of all-request layout success. These cache tests do not establish content-based checkpoint identity. Existing data/implementation/decode invalidation assertions remain useful.

**Follow-up:** Report all-request and in-capacity denominators separately. Add independent cache cases for a changed checkpoint path and changed weights at an unchanged path, with explicit expected invalidation.

### V21-L6: Several metric consistency checks share the implementation under review

Evidence: `fastfill/v2/tests/test_review_repairs_r2.py:26–45`; `fastfill/v2/tests/test_round5_train.py:75–89`; `fastfill/v2/tests/test_round14_eval_math.py:269–279`.

The final ground-truth anchor is recomputed with evaluate.collapse_metrics/collapse_score; distributed diagnostics are compared to the same single-process routines; the uniform-yaw quadrature calls _box_equivalent_errors. These test aggregation or consistency, not an independent definition of every underlying metric.

**Claim boundary:** Agreement on these paths cannot alone rule out a common mathematical error. This batch also contains useful independent expectations: analytic loss values, exhaustive assignment permutations, explicit pooled/room-mean counts and binary GLB corner decoding.

**Follow-up:** Retain consistency checks and supplement research-facing metric definitions with small hand-computed geometry/denominator examples that do not call the production metric to obtain expected values.

## Per-file read and final hash ledger

All entries have `read_status=fresh_full`; scope is **every line 1–N inclusive**. “match” means final bytes and line count equal the recorded ledger/inventory values. No status was written back to the shared ledger.

| Path | Lines / scope | SHA-256 | Read status | Final validation |
|---|---:|---|---|---|
| `fastfill/v2/tests/test_accumulation_tail.py` | 206 / 1–206 | `c7bbdd9b3181a2b257387ab983aba44e8f1a571e2aa6c59554f520710e0ba07a` | fresh_full | match |
| `fastfill/v2/tests/test_autorun.py` | 407 / 1–407 | `36229eda2e01dda704e767221678d160c87e83694d2415a51d73a73dc33a98f7` | fresh_full | match |
| `fastfill/v2/tests/test_constraint_schema.py` | 26 / 1–26 | `f343f639e321dd0b0ecc2f03ba8231400d6f5b3dbb48cb41c366fbcf05e61230` | fresh_full | match |
| `fastfill/v2/tests/test_data.py` | 232 / 1–232 | `f7a0a4b1d8c05616eeb21820036f3a05c2300e97ac805a1cf9eafcd785087ad4` | fresh_full | match |
| `fastfill/v2/tests/test_direct_layout.py` | 469 / 1–469 | `c5c6a77a017bcb415b2ca97645d8a55050e162d9c1d395b64c93f5e97479bf19` | fresh_full | match |
| `fastfill/v2/tests/test_leak_guards_fix20261006.py` | 155 / 1–155 | `31b42f26fd3e9518f020d217fb67fd10dcd8ae7aab9ff02c60159b026b4ea8e0` | fresh_full | match |
| `fastfill/v2/tests/test_legacy_evidence.py` | 126 / 1–126 | `1e41608654f3d4e925fe38498375a70d2522ab290ab8528d8260f55ebdb29842` | fresh_full | match |
| `fastfill/v2/tests/test_legacy_workers.py` | 52 / 1–52 | `03e7fd8d5e382ad937ad6bde6934d711566a29d87bd8bd571c5aadc24c18d38f` | fresh_full | match |
| `fastfill/v2/tests/test_llm_baseline.py` | 174 / 1–174 | `9ad03c6d24c5591f104bb224c50fe7d619db43bb775ba5f0439f00e504134bee` | fresh_full | match |
| `fastfill/v2/tests/test_losses_matching_fix20261006.py` | 125 / 1–125 | `30e404a4a00f0b5aed6578002a337a66f6294f7c004325245af151d2dc4dda0e` | fresh_full | match |
| `fastfill/v2/tests/test_model.py` | 147 / 1–147 | `13324e88996821541e77babe6f5e62e49b099ec014ce23f3a8ea06906090e758` | fresh_full | match |
| `fastfill/v2/tests/test_model_review_fixes.py` | 145 / 1–145 | `d7ec8804c3366def003e5843584357025009c4959013f7dbaf7c258a96c14303` | fresh_full | match |
| `fastfill/v2/tests/test_predict.py` | 67 / 1–67 | `7fcf8d9d0975b03b73592b314533b3749e6ada12fb37bc5cf01d7ff580f345f5` | fresh_full | match |
| `fastfill/v2/tests/test_qualified_data.py` | 127 / 1–127 | `cc5e8b5e172621ba477c0c479598f3bb415ee37f666eceee3106eb431ee0ae83` | fresh_full | match |
| `fastfill/v2/tests/test_regularizer_support_review.py` | 103 / 1–103 | `139e67e3ded4479f01e85384c02bd25bfa2e8ebb699b242439d32f5e7fd372e5` | fresh_full | match |
| `fastfill/v2/tests/test_review_repairs_r2.py` | 153 / 1–153 | `871a4822b58ab7983bd8966dce1c714ef8fcca260c825a58a672a7759b9ee323` | fresh_full | match |
| `fastfill/v2/tests/test_roomgenbench_fix20261006.py` | 202 / 1–202 | `9d9a44ec42555f1764edd70b31c20468ad8b82c3adc82537f3b9736262f939b7` | fresh_full | match |
| `fastfill/v2/tests/test_round14_eval_math.py` | 300 / 1–300 | `351430afee8ad66f78792b850c8b4bcf562f9bd29bd84dfc4ac5ed4719e3a6b3` | fresh_full | match |
| `fastfill/v2/tests/test_round4_matching.py` | 126 / 1–126 | `f47c04fb0a0cb5404c313846e15544474cca6c51b04b7013baae25ca6c24f0a0` | fresh_full | match |
| `fastfill/v2/tests/test_round5_train.py` | 191 / 1–191 | `399cade685234c75c1da8e6eb55e62ca596d78dd31a5d836021ffdccdc6eb425` | fresh_full | match |
| `fastfill/v2/tests/test_runtime_cli.py` | 72 / 1–72 | `a0603b9dbf10274e1c963e2165f67fd86531eca2c249ee5238364dea2303c5c6` | fresh_full | match |
| `fastfill/v2/tests/test_runtime_evaluation.py` | 132 / 1–132 | `737cccce0f82882caee70f8608364d670d2228ba33dbd07786c395090ea3a2cd` | fresh_full | match |
| `fastfill/v2/tests/test_size_range.py` | 73 / 1–73 | `0164559ee9bbd9e029182bf211df876682d6d5a3f3792e717891afac9e740ae5` | fresh_full | match |
| `fastfill/v2/tests/test_spread_decode.py` | 61 / 1–61 | `ea5f089cde017e8b0e37f4714e5f69be1ca6bf169b91758cc5803c1ed6ac988b` | fresh_full | match |
| `fastfill/v2/tests/test_text_sft.py` | 86 / 1–86 | `ad0c6f575b34b659b033d28cd7b7054f0d8b310001af2819bf800ca068e9a5db` | fresh_full | match |
| `fastfill/v2/tests/test_train_loop_fix20261006.py` | 399 / 1–399 | `f2db1fb105f49a3da8ab6926d3f1b2a29c0949cc9fd244ab39acb223303491eb` | fresh_full | match |

## Per-file assessment

- **test_accumulation_tail.py** — Asserts complete/tail accumulation means, empty-label window accounting, clipping before/after AMP scaling, overflow step counts and two-rank Gloo tail behavior using recorded scalar gradients.
- **test_autorun.py** — Asserts paired-baseline scoring, summary denominators, configuration scaling, resume-chain adoption, mocked acceptance/orchestration and cache invalidation for data, code and decode changes.
- **test_constraint_schema.py** — Asserts malformed between-target references are rejected and valid between/against_wall schema validation preserves the condition; geometry satisfaction is not exercised here.
- **test_data.py** — Asserts source OBB extent/unit conversion, invalid geometry rejection, condition isolation, audit inventory and read-only builder guards; arbitrary-angle yaw and related-scan split oracles are incomplete. Findings: V21-F1, V21-F2.
- **test_direct_layout.py** — Asserts request expansion, ID completeness/order, world-space corners, canonical-to-benchmark axes, independent GLB binary/transform decoding, immutable exports, SVG safety and object budgets.
- **test_leak_guards_fix20261006.py** — Asserts explicit benchmark holdout guards, non-convex outward shell walls, bf16 yaw precision, row-level leave-one-out baselines, height epsilon and Scan2CAD symmetry metadata.
- **test_legacy_evidence.py** — Asserts evidence width correction without pose mutation, front-unknown provenance, immutable indexes, missing/partial/malformed evidence behavior, no-match scope and idempotence.
- **test_legacy_workers.py** — Asserts serial/parallel migration byte equality for four dataset outputs, unchanged input snapshots, invalid worker rejection and absence of partial output on total rejection.
- **test_llm_baseline.py** — Asserts target-derived answer conversion, failure retention, repair feedback, retry/first-answer accounting, parameter provenance, credential redaction and mocked transport handling; repaired geometry itself is not asserted. Findings: V21-F3.
- **test_losses_matching_fix20261006.py** — Asserts float64 yaw-bin round trips, batch-owned exchange groups, position-only matching with incomplete sizes, invalid loss settings, analytic L1/smooth-L1 values and term sums/counts.
- **test_model.py** — Asserts model validation, finite positive sizes under CPU fp16, valid-slot gradients, padding invariance, condition influence, fixed dimensions and small local Qwen training modes; checkpoint comparisons inspect size only. Findings: V21-F4.
- **test_model_review_fixes.py** — Asserts target type rejection, hard-floor masks, learnable preflight, finite graph-connected zeros, Gloo uneven/global-zero counts and gated MPS value/gradient parity.
- **test_predict.py** — Asserts condition-only prediction output schema/protected paths and actual subprocess exit distinctions for no catalog, empty catalog and successful in-memory resolution; no audit subprocess was run.
- **test_qualified_data.py** — Asserts rich condition/target preservation, source-specific mask demotion, unclamped size masking, heldout-source rejection, immutable pinned builds and whole-group exchangeability demotion.
- **test_regularizer_support_review.py** — Asserts hard support collision exemptions, soft-support penalties, graph-connected zeros, extreme-scale collision values and a numerical gradient oracle, with gated MPS cases.
- **test_review_repairs_r2.py** — Asserts final-update validation/export and score anchors, minimal-form eligibility, rotated rectangle tolerance, pinned axis-swap/yaw semantics, unknown-floor handoff and backbone-path provenance.
- **test_roomgenbench_fix20261006.py** — Asserts embedded real-room shell/opening mapping, actual reference assembler integration, transformed world corners, lifted floors, opening classification and inferred/declared support thresholds.
- **test_round14_eval_math.py** — Asserts box-preserving wall yaw, rotated support containment, square-grid validation, paired baseline/clean-room denominators, offline head replay equality and externally gated reference matching/baselines.
- **test_round4_matching.py** — Asserts a grid matching counterexample and exhaustive three-object assignment minima across seeds/configurations, fixed-z exclusion, identical-prediction invariance and frozen regression-head cost equivalence.
- **test_round5_train.py** — Asserts unpadded distributed diagnostic counts, old-state selection reset, resumed dropout/validation trajectory equality, reduction-term guards and gated Metal grid/reduction consistency.
- **test_runtime_cli.py** — Asserts complete offline JSON transactions, in-memory commit counts, output guards before input reads, failed-scene no-commit behavior and malformed asset rejection.
- **test_runtime_evaluation.py** — Asserts malformed prediction denominators, model-vs-system failure distinction, bounded repair output geometry, operational latency decomposition, incomplete-label matching and required-mesh unknown status.
- **test_size_range.py** — Asserts out-of-head-range label detection, reference-scaled/exact-boundary eligibility, fixed or masked exceptions, enabled-box requirements and nonmutating preflight rejection records.
- **test_spread_decode.py** — Asserts feasible rectangular-grid spreading, raised support/floor behavior, outside-footprint cell rejection, wall facing, yaw preservation without logits and small boundary corrections.
- **test_text_sft.py** — Asserts text/structured condition token parity and loss masking, differentiable CE, complete-label preflight and protected outputs; generation is a dry-run string smoke check.
- **test_train_loop_fix20261006.py** — Asserts config validation, learning-rate/group behavior, killed/resumed trajectories and full saved states, nonfinite-window handling, weighted logs, validation filtering/baselines, exports, exchangeability preflight, augmentation and Gloo resume.
