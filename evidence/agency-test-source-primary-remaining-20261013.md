# Residual test-source audit — original 15-file tranche

Completed the original 15-file / 5,997-line assignment: **11 fresh full reads (4,372 lines)** and **4 reused full reads (1,625 lines)**. All 15 source SHA-256 hashes and line counts matched at the original validation timestamp; zero source-read gaps remain for those frozen review versions. This is not a current-working-tree all-match claim.

Found **3 concrete test-oracle gaps** and **5 coverage limitations**. These are static observations about the exact sources below, not confirmed production defects or executed mutation results.

## Expected post-audit change

User update: Poincare is authorized to fix valid fixtures only in `fastfill/comparison/tests/test_visual_pipeline.py` after the sourceVIS resource changes. Its reviewed baseline remains **431 lines**, SHA-256 `f7ef18ffa0974ef81d3090f01e575ad8e202c9f85c8d54be43c6cb7a84a1a577`. Marked **expected_post_audit_change**; the new version has not been read or rehashed here and will be covered by independent patch review. Visual-pipeline observations below apply only to the frozen baseline.

The user reported 41 new tests, 202 targeted passes, and 25 existing visual-pipeline fixture failures caused by missing resources. These are attributed context, not verified results of this audit or source-read credit. Original counts remain 15 files / 5,997 lines; new correction files remain excluded.

## Scope and method

- Selector: ledger entries with read_status=unreviewed, minus every delegation_ready_batches[].paths entry.
- Original inventory: 121 files. New test_source* correction files, other batches (including v2-3), and production/adapter changes are excluded.
- Fresh reads cover every line, including fixture data, through bounded, numbered, non-truncated source output; SHA checked for every chunk. Reused files were not freshly reread.
- Only the two requested report files were written. No tests, project imports/execution, remote calls, production/test edits, shared-ledger edits, or subagents.
- The parent's 13 residual gaps reflect reuse of geometry/final_tables. This report retains the original 15-file assignment and also establishes exact full-read reuse for train_pair and size_prior. It is separate from the parent's stated 681-pytest / 90.87% execution result, which was not validated or used as source-read proof here.
- Exact census of the assigned source inventory, not a statistical quality estimate. Confidence intervals, significance, defect probabilities, pass rates, and execution line/branch coverage are not estimable from this evidence and are not asserted.

Ledger: `evidence/agency-full-test-source-coverage-20261013.json`  
Ledger SHA-256: `22eb3fa5c68b63e62230e0301a5936ce97b839ef23f17b79b00e4c095099d4fc`

## Reuse provenance

- geometry, final_tables, and train_pair: `evidence/agency-full-audit-patch-review-20261013.md`, lines 7 and 15 assert direct/complete reads; line 80 records stable rehashes; lines 88–90 list the exact current test hashes. Evidence SHA-256: `a7af0af942c97d0d8fa362de39986f864d857bf90c399df2fdcafb1ec948b7e3`.
- size_prior: `docs/research/20261013-unknown-size-comparison/evidence/agency-unknownsize-data-review-20261013.md`, line 60 explicitly asserts full test reads and line 86 gives the matching hash. Evidence SHA-256: `c85bd612630b5e8802888e070b272031615102ecaa6701ee411742dc6707501a`.
- Metrics audit JSON was consulted first: its geometry/final_tables full-read SHA values differ from current bytes, despite matching sha256_at_handoff values and changed_since_read=true. Independent patch review supplies complete-read evidence for the current versions. No fresh read was added to inflate counts.
- Context-length review's matching infer hash lacks an explicit full-read assertion, so infer was read fresh. Partial reads, manifests, execution receipts, stale hashes, and inaccessible Sartre evidence were not promoted to full-read credit.
- Exact evidence quotes, source hashes, and rejected reuse details are recorded in the companion JSON.

## Per-file record

Every scope is **1 through EOF**. The SHA-256 shown is the inventory/review hash **validated at the original timestamp only**, not a claim about current files. Fresh spans list actual displayed chunks; reused entries have no fresh spans.

| Path | Lines | Read status | Fresh spans | SHA-256 / historical validation |
|---|---:|---|---|---|
| `fastfill/comparison/tests/test_final_tables.py` | 122 | `reused_full` | none (full prior read) | `ac0bb5752b21cfdfe63638fb931ea5ef325302fe30095ebe72efa73f89c95901` / historical match |
| `fastfill/comparison/tests/test_geometry.py` | 271 | `reused_full` | none (full prior read) | `f976be18879d54b2cd8faec6d9e8dcdbbcebb5c4d2c6b012e6ad6b919722c49f` / historical match |
| `fastfill/comparison/tests/test_infer.py` | 415 | `fresh_full` | 1–140, 141–280, 281–415 | `d5e4943dde08055dddbc8873f71d4c1e47c583617a66f4105944dff3ddc2ac53` / historical match |
| `fastfill/comparison/tests/test_main_data.py` | 413 | `fresh_full` | 1–210, 211–413 | `9210de6f73c1b956bc073414e1cfbf9c8a12c39c7291822e2f74b4ad18e8fb05` / historical match |
| `fastfill/comparison/tests/test_overnight.py` | 601 | `fresh_full` | 1–200, 201–400, 401–601 | `db8d64de3d2d06c184f744f5ac2d3b73a47981cf8e93fa7f644737827ef0257b` / historical match |
| `fastfill/comparison/tests/test_pose_targets.py` | 394 | `fresh_full` | 1–200, 201–394 | `1f2ae66289e7b6c42480fe89f7b25ada9e2027f130f168b4270a85380a52ea4d` / historical match |
| `fastfill/comparison/tests/test_render_assets.py` | 556 | `fresh_full` | 1–190, 191–380, 381–556 | `87e4493b7e6bf30f4826eff434bbdce78c2fa8cf036c81822d0e052e58c2f5c9` / historical match |
| `fastfill/comparison/tests/test_size_prior.py` | 315 | `reused_full` | none (full prior read) | `60072f98f8c59bac4507f143b85363df807e27748e154f1203db681030c3f093` / historical match |
| `fastfill/comparison/tests/test_train_pair.py` | 917 | `reused_full` | none (full prior read) | `29bf539d83577e7e943b91844e513d088026142d73d82dac912231100131e1e5` / historical match |
| `fastfill/comparison/tests/test_visual_pipeline.py` | 431 | `fresh_full` | 1–215, 216–431 | `f7ef18ffa0974ef81d3090f01e575ad8e202c9f85c8d54be43c6cb7a84a1a577` / historical match; expected change |
| `fastfill/tests/test_adapter_repairs.py` | 491 | `fresh_full` | 1–170, 171–340, 341–491 | `2306c3044e1b0960fdd0af9cbc6317463d190817c6964769a1cb58cc07d64791` / historical match |
| `fastfill/tests/test_build_repairs.py` | 211 | `fresh_full` | 1–211 | `39e0447b31310a52b77fd843e571078c3e9a01937a775c71c46cff5aa533b7b0` / historical match |
| `fastfill/tests/test_embedded_contracts.py` | 18 | `fresh_full` | 1–18 | `0f2c6ed37414799f3c64adeaf25bccfdf14563ef6ac6d54596097e7399ca4f0d` / historical match |
| `fastfill/tests/test_qa_repairs.py` | 454 | `fresh_full` | 1–180, 181–340, 341–454 | `e833fe783bd12fca3ad7c4b54959760f70f1cd4a6f6d0cfd4675026be49d040a` / historical match |
| `fastfill/tests/test_runtime_repairs.py` | 388 | `fresh_full` | 1–200, 201–388 | `1a4547235a4a0f8969f06e7f96cc1a338169d5810ba5d5d4c694f95bab506aa0` / historical match |

## Reviewed contracts

- **test_final_tables.py**: Reused independent complete-source review: frozen judge validation, positive-request strata, and failure/score retention. This turn validates provenance and hashes, not a new source analysis.
- **test_geometry.py**: Reused independent complete-source review: cumulative raw-relative 0.1 m repair budget, support-tree rollback, and preserving a valid exactly-10 cm move. This turn validates provenance and hashes, not a new source analysis.
- **test_infer.py**: Fresh full read: request/sidecar hashes, ordered cardinality and failure receipts, interruption accounting, training identity, token budgets, paired input plans, and mocked local model/runtime wiring.
- **test_main_data.py**: Fresh full read: training-only and out-of-fold size priors, paired input equality, failure retention, identity closure, teacher/output corruption, rounded geometry, consumed-byte races, manifests, and bounded worker reproducibility.
- **test_overnight.py**: Fresh full read: configuration validation, exclusive artifacts, staged failure/timeout handling, frozen resources, subprocess ownership/retries, and a mocked render/judge pipeline. Failure endstate contents remain weakly asserted in one parametrized regression.
- **test_pose_targets.py**: Fresh full read: inventory preservation, support chains, cardinal placement, concavity, obstacles, rounding, ceilings, graph rejection, bounded search, and authoritative final checks. Acceptance follows the selected geometric policy.
- **test_render_assets.py**: Fresh full read: axes, mesh fitting, normals, triangulation, texture/resource provenance, camera independence, unknown height, malformed ledger records, and mocked Blender publication/imports.
- **test_size_prior.py**: Reused complete-source data review with exact matching SHA: training-only/out-of-fold priors and held-out/development target perturbation checks. This turn validates provenance and hashes, not a new source analysis.
- **test_train_pair.py**: Reused independent complete-source review: model-visible field allowlists and target policy/support-graph checks. This turn validates provenance and hashes, not a new source analysis.
- **test_visual_pipeline.py**: Fresh full read: paired geometry/render/API/FID records, strict hashes, frozen selection, failure denominators, concurrency, malformed records, and synthetic large-denominator retention. The fake renderer emits identical pixels for every identity.
- **test_adapter_repairs.py**: Fresh full read: retained adapter geometry regressions, independent corner/quaternion calculations, tilted/upright front directions, support/floor obstacles, mocked dataset inputs, and Interior room assignment. Two rejection tests permit vacuous loop assertions.
- **test_build_repairs.py**: Fresh full read: output preservation, staging/serialization failure, manifests, dependency drift, invalid input, serialized constraints, and geometry rejection. No new concrete assertion defect identified.
- **test_embedded_contracts.py**: Fresh full read of the wrapper only: seven production modules are invoked as __main__. Their embedded implementation/assertion bodies were not read in this tranche.
- **test_qa_repairs.py**: Fresh full read: ablation equality, schema/reference/truth counts, provenance and replay corruption, streaming, fixed yaw, mandatory failure policy, and CLI report behavior. A metric-summary assertion permits empty or incomplete metric dictionaries.
- **test_runtime_repairs.py**: Fresh full read: request anchors and keepouts, fixed/window policies, support tolerance boundaries, handler envelopes/length limits, and embedded-selfcheck dispatch. Handler I/O and generation are synthetic; embedded production bodies remain outside this read scope.

## Concrete test-oracle gaps

### PR-F1: Interior rejection regressions can pass when all rooms disappear

Priority: medium. Evidence: `fastfill/tests/test_adapter_repairs.py:456–473`.

The overlapping/nearest-room rejection test and parametrized invalid-boundary rejection test put their result assertions inside `for r in load_interior(scene)`. Neither first asserts the returned room count or identities.

For these inputs, a loader returning [] skips every assertion in those loops. This is a static counterexample to the local assertions, not an executed mutation or a claim that every adapter test would pass.

**Claim at risk:** The tests do not distinguish retaining rooms while refusing ambiguous object assignment from silently dropping the rooms, which can change dataset/geometry evaluation denominators.

**Suggested assertion:** Materialize the result, assert the expected room IDs and count for each input, then assert empty object assignments and unresolved reasons.

Other cases do assert room cardinality. New correction tests outside the 121-file inventory were excluded; no claim is made about their coverage or current adapter correctness.

### PR-F2: Four pipeline failure cases do not assert truthful terminal status

Priority: medium. Evidence: `fastfill/comparison/tests/test_overnight.py:471–484`; `fastfill/comparison/tests/test_overnight.py:511–516`; `fastfill/comparison/tests/test_overnight.py:543–548`.

The parametrized downstream-stop regression covers terminal_failure, dependency_never, timeout, and hash_mismatch. It checks that downstream work stops and that an endstate is truthy, but does not assert its failure/partial status, failure stage/reason, or persisted status content.

A nonempty endstate claiming completion for these failure paths would satisfy the endstate assertion if the existing stop/timing behavior were preserved. This was not executed.

**Claim at risk:** Failure retention and research-run completion claims are not fully guarded by this regression; a stopped failed run could be labeled complete.

**Suggested assertion:** Assert the expected terminal status and failure stage/reason per scenario and agreement with any persisted receipt.

Separate ambiguous-submission and interrupt tests assert partial status; this finding is limited to the four listed scenarios and does not establish a production reporting defect.

### PR-F3: QA legality-summary equality accepts missing metric fields

Priority: medium. Evidence: `fastfill/tests/test_qa_repairs.py:322–334`.

The test computes `expected = summarize([score(room, None)])["in_dist"]` and compares `actual` with `{k: expected[k] for k in actual}`. It never asserts the required keys of `actual`.

If the existing `dev/in_dist` summary becomes {}, the comparison is {} == {}. A nonempty subset of the expected metrics can also pass. The preceding constraint-count assertions do not assert these legality-summary fields.

**Claim at risk:** Raw/repaired reference-legality reporting can silently lose metrics while this summary regression remains satisfied.

**Suggested assertion:** Assert the documented required key set and expected values, including independently specified geometry examples for critical legality metrics.

This is an omission-sensitive assertion defect, not evidence that the present QA implementation emits an empty summary; no mutation or tests were executed.

## Coverage limitations

### PR-L1: Identical synthetic images do not prove correct image-to-identity assignment

Evidence: `fastfill/comparison/tests/test_visual_pipeline.py:32–104`; `fastfill/comparison/tests/test_visual_pipeline.py:117–144`.

The fixture uses the same geometry across reference/candidate arms and the fake renderer writes the same flat RGB image for every UID, arm, and view. Hash/provenance assertions can validate records while being unable to distinguish a pixel-preserving permutation of image identities.

**Claim boundary:** These tests support ledger/provenance behavior, not that distinct rendered evidence is paired with the intended room, arm, or view.

**Follow-up:** Use distinct deterministic image patterns per UID/arm/view and assert the selected bytes/features match the intended identity.

### PR-L2: Mocked external stages do not substantiate real render/model/metric performance

Evidence: `fastfill/comparison/tests/test_infer.py:320–360`; `fastfill/comparison/tests/test_render_assets.py:426–445`; `fastfill/comparison/tests/test_render_assets.py:498–556`; `fastfill/comparison/tests/test_overnight.py:574–601`.

Model libraries and generation are substituted; Blender tests use fake bpy/render output; the pipeline uses synthetic image features and a fake judge. These are useful control-flow and provenance tests.

**Claim boundary:** They provide no direct observation of actual Blender appearance, model memory/context fit, learned image-feature behavior, judge quality, or numeric research performance.

**Follow-up:** Keep these tests as offline contracts and require separately scoped, versioned real-backend evidence for those claims. Such work was neither authorized nor performed here.

### PR-L3: Geometric acceptance is relative to an explicit policy

Evidence: `fastfill/comparison/tests/test_pose_targets.py:80–83`; `fastfill/comparison/tests/test_pose_targets.py:111–127`; `fastfill/tests/test_runtime_repairs.py:162–172`; `fastfill/tests/test_runtime_repairs.py:259–294`.

Tests intentionally allow support overhang, selected tuck/window collision cases, and inclusive support tolerances. Pose acceptance also calls the production geometric checker.

**Claim boundary:** Passing these contracts does not imply mesh-level collision freedom or physical stability. The policy choices are not reported as defects.

**Follow-up:** Name these tolerances/exemptions in geometry claims; any stronger physical claim needs separate independent oracles.

### PR-L4: Captured regressions and synthetic denominators cannot estimate population reliability

Evidence: `fastfill/tests/test_adapter_repairs.py:1–5`; `fastfill/tests/test_adapter_repairs.py:23–209`; `fastfill/comparison/tests/test_visual_pipeline.py:400–431`.

The adapter fixtures retain 21 historical geometry-error cases; the visual selection case uses a synthetic 1009-case population with mocked backend failures. These are deliberately constructed regression fixtures, not representative outcome samples.

**Claim boundary:** No real-corpus defect prevalence, model quality, success probability, or statistical confidence interval follows from these cases or their counts.

**Follow-up:** Use corpus-level sampled measurements with explicit selection and denominator rules for population claims; this report only gives exact deterministic source-read counts.

### PR-L5: Embedded selfcheck wrappers do not constitute a read of embedded source

Evidence: `fastfill/tests/test_embedded_contracts.py:1–18`; `fastfill/tests/test_runtime_repairs.py:365–384`.

One wrapper invokes seven production modules with runpy, and runtime tests invoke three other modules plus an extracted serve main block. The serve extraction checks for an ast.Assert; the runpy wrappers primarily rely on invoked code to assert.

**Claim boundary:** Full reading of these wrapper files closes their inventory entries only. It does not establish full-source review or substantive assertion coverage of production selfcheck bodies, and no selfchecks were run.

**Follow-up:** Track the embedded bodies separately if future source-audit scope includes them; do not count them as extra reviewed files here.

## Historical validation and version boundary

- Validation time (UTC): `2026-10-13T08:10:10.933326+00:00`.
- At the original validation timestamp, 15/15 source hashes and 15/15 line counts matched inventory/review versions. Those historical values are preserved; current all-match status is not asserted.
- Fresh read spans cover all 4,372 fresh lines exactly once. Exact prior full-read records cover the other 1,625 lines.
- Ledger selection and ledger hash match the original assignment. Both accepted evidence files and the metrics/context reuse-decision evidence match their recorded hashes.
- Citation bounds, JSON structure, per-file statuses, totals, and reuse assertions were checked locally without running project code.
- `test_visual_pipeline.py` is marked as an expected post-audit change. Its baseline SHA/read spans remain frozen; independent patch review covers the new fixture version. No source reread or rehash was performed for this update.
- The hashes identify reviewed versions; no conclusion is extended to concurrent adapter changes or newly added correction tests.
- This report does not change the shared ledger or claim closure outside the original 15 paths.
