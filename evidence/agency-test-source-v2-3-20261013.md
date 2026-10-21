Batch v2-3 test-source audit — 2026-10-13

Completed **32/32 files, 4,489/4,489 lines**, all `fresh_full`. Every line was displayed and reviewed; **32/32 final SHA256 and line-count checks match the ledger**, with zero unread files or mismatches.

Results: **2 concrete test-oracle findings; 8 scoped coverage limitations; 0 confirmed production defects.** No tests were imported, collected or executed; no remote calls, production edits, other-batch test reads, or shared-ledger edits. Only this report and its JSON companion were written.

Scope: exactly `delegation_ready_batches[batch=v2-3].paths` in `evidence/agency-full-test-source-coverage-20261013.json`, whose selection-time SHA256 was `22eb3fa5c68b63e62230e0301a5936ce97b839ef23f17b79b00e4c095099d4fc`. The ledger marked this batch unassigned and its files unreviewed. Source discovery was already complete; the user reported the graph lookup/root-index limitation. Agent delegation was blocked by the thread limit, so the parent performed all 35 source-display spans.

Method: inspect raw-source bytes at their ledger hashes, review all imports/fixtures/comments/assertions without following out-of-batch dependencies, then rehash every file. Cross-check findings only within this batch. Source coverage is a deterministic census, not executable coverage. Pass rates, confidence intervals, statistical significance and release readiness cannot be inferred from this static audit.

**Concrete test findings**

**V23-F01 — Reference round-trip assertions can pass with missing objects (medium).** `fastfill/v2/tests/test_round7_reference.py:19–33`.

The nine-object round trip iterates zip(layout['objects'], inverse['objects']) without first asserting output length or the complete ID sequence. If inverse['objects'] is empty, lines 30-33 execute zero times; a correct prefix also leaves the omitted objects unchecked. This is a logical counterexample to this test's oracle, not an executed mutation or a claim that production returns such output.

Impact: The unconditional test does not establish completeness or exercise every advertised near-pi yaw when output is truncated. Recommended regression: Assert the full ordered ID list and count before comparing geometry; demonstrate that an empty output and a missing last object fail the oracle.

**V23-F02 — Checkpoint/validation association test never checks the persisted checkpoint (medium).** `fastfill/v2/tests/test_training_record_review.py:34–39`; `fastfill/v2/tests/test_training_record_review.py:49–63`.

test_logged_validation_matches_saved_step_parameters captures hashes during validation and compares them to the final in-memory models[0]. It neither reloads a saved model nor reads the saved step's validation record. A writer that saves stale or otherwise different parameters while leaving the validated in-memory model unchanged is invisible to these assertions, provided run_training returns normally. No such faulty writer was executed or confirmed.

Impact: The test proves validation uses the final in-memory state for its one-step fixture, but does not bind a reported score to the model artifact used for research results. Recommended regression: Reload the exported model for the reported step, hash its parameters, and compare with the captured validation hash and recorded checkpoint identity. Include a stale-export negative case.

test_model_loss_r2.py:125-140 checks a separate model save/load round trip, which does not establish training-log/checkpoint association.

**Coverage limitations**

These are limits of the reviewed assertions, not demonstrated production faults or claims that all other tests lack coverage.

**V23-L01 — Pivot-only transform oracle cannot distinguish rotation and scale errors.** `fastfill/v2/tests/test_runtime.py:43–53`.

The only raw point checked maps to canonical origin, so the fixture verifies translation of the pivot but is insensitive to the linear part applied after canonical conversion. Correct raw-asset orientation, scale and transform composition are not established by this runtime assertion.

Follow-up: Check three non-collinear displaced raw points or the full expected matrix. For this fixture, raw (.5,1,1.5) maps to (5,6,7); raw (1,1,1.5) should map to (5,7,7) after scale 2 and +pi/2 yaw, whereas omitting yaw gives (6,6,7).

**V23-L02 — Partial oriented overlap lacks an independent numeric oracle.** `fastfill/v2/tests/test_supervision.py:98–128`; `fastfill/v2/tests/test_validation.py:30–49`.

The oblique GIoU case checks only finiteness; gradcheck checks derivative consistency. Numeric cases cover identical, touching/disjoint and translated identical tiny boxes. The rotated validation case contains one object; pair-collision fixtures are coincident or axis-aligned touching boxes. A self-consistent but wrong partial intersection or rotated pair-collision calculation is not ruled out by these reviewed assertions.

Follow-up: Add an independent partial-overlap oracle (e.g. centered 2x1 rectangles at relative pi/2 have BEV IoU 1/3), plus oblique pairs whose AABBs overlap but oriented footprints do not, and near-contact tolerance cases.

**V23-L03 — Failure-inclusive latency test does not pin percentile arithmetic.** `fastfill/v2/tests/test_evaluation_failure_latency.py:21–61`.

The two-request fixture checks a failed request's elapsed time, population counts and only a lower bound fastfill_p95 >= 19. It correctly checks null latency for supplied predictions. Using a maximum, an inflated percentile or another incorrect aggregation could satisfy the lower-bound assertion; this source alone does not validate reported tail-latency values.

Follow-up: Use a deterministic clock with several distinct success/failure durations and assert exact aggregate and subgroup percentiles under a documented interpolation rule, including zero observed durations.

**V23-L04 — Uniform-yaw baseline agreement mostly reuses the scored metric.** `fastfill/v2/tests/test_audit_c2_uniform_yaw.py:12–37`.

All five parameter tuples use symmetry order 4. Expected values come from reference_metrics; one elongated swap-enabled case additionally pins an independent numeric value near .7677. Stratification is compared to the same baseline. This is useful consistency coverage, but independent yaw correctness for orders 1/2 and broader joint size/yaw cases is not established by this file.

Follow-up: Add closed-form no-swap uniform-yaw means for orders 1 and 2, independent wrapped-angle examples at branch boundaries, and an independently computed joint-candidate oracle; retain the existing elongated-box check.

**V23-L05 — Generated-mesh fit assertions cover box dimensions, not semantic front.** `fastfill/v2/tests/test_roomgenbench.py:38–47`; `fastfill/v2/tests/test_roomgenbench.py:101–147`; `fastfill/v2/tests/test_runtime.py:185–190`.

Generated assets are translated cuboids. The generated-mesh test checks native/fitted dimensions, scale length and parent metadata, while world-space corner assertions exercise the placeholder-box path. The runtime front case supplies local +X at zero yaw. The receipt explicitly says original_asset_geometry_acceptance='not_checked'. Those checks cannot establish the semantic-front orientation of a generated asymmetric asset or actual mesh/physics acceptance. A 180-degree front reversal of a cuboid preserves its dimensions.

Follow-up: Use an asymmetric mesh with a marked front and offset pivot, reload assembled vertices/landmarks, and assert transformed front at nonzero yaw; keep assembly success separate from asset geometry acceptance.

**V23-L06 — Real-corpus and benchmark assertions depend on optional external fixtures.** `fastfill/v2/tests/test_batch_handoff_r2.py:69–85`; `fastfill/v2/tests/test_batch_handoff_r2.py:229–247`; `fastfill/v2/tests/test_round10_wall_selection.py:242–269`; `fastfill/v2/tests/test_round7_reference.py:36–66`.

The 50-real-row projection check, downstream receipt check, five-room support tally, and bathroom reference checks contain explicit skip conditions for external data/reference availability. Their presence in source cannot establish that real-data/benchmark integration ran. This audit did not inspect those external fixtures or determine actual skip outcomes; test_roomgenbench.py also has reference-dependent tests without these skip guards.

Follow-up: For claim-bearing validation, preserve an execution receipt with fixture hashes and explicit required-test/skip outcomes; retain a small independently authored local geometry fixture so basic reference checks are always available.

**V23-L07 — Toy mode-selection and configuration checks do not establish empirical model superiority.** `fastfill/v2/tests/test_model_loss_r2.py:166–197`; `fastfill/v2/tests/test_model_loss_r2.py:245–261`; `fastfill/v2/tests/test_isambard_inputs.py:16–52`.

The mode-selection example trains a tiny model for 100 steps with seed 0 on one repeated condition and two x positions. Formal epoch checks use the literal cohort count 124375, and arm-selection checks use authored scores/manifests. These are mechanism/configuration regressions, not estimates of generalization, seed stability, production-backbone performance or a measured treatment advantage.

Follow-up: Keep the toy regression, but support comparative research claims with independently held-out scene/house groups, repeated seeds, actual cohort/exposure accounting and paired uncertainty estimates from measured outcomes.

**V23-L08 — VLM judge tests validate rendering and anchors, not judgment validity.** `fastfill/v2/tests/test_vlm_judge.py:1–137`.

Assertions cover plan/camera projection, local-front arrows, deterministic anchors, occlusion and label placement. This file contains no judge-response parsing, model-output scoring, paired-order reversal or calibration assertion; two plan-rendering tests can skip without matplotlib. Correct visual inputs do not establish that a VLM-derived preference or score measures layout quality or is unbiased by presentation order.

Follow-up: Add offline canned judge-response and pair-identity/order tests. Any scientific claim about judge validity additionally needs measured agreement/calibration evidence, which is outside this static audit.

**Per-file read and final-hash inventory**

The SHA256 below is both the read hash and the independently revalidated final hash. Every row has `read_status=fresh_full`, full inclusive source scope, and final hash/line validation `passed`. Validation time: `2026-10-13T07:59:03.064699+00:00`. JSON also records each displayed span, prior status, review summary and finding links.

| Path | SHA256 (read = final) | Lines | Read status | Full read scope | Final validation |
| --- | --- | ---: | --- | --- | --- |
| `fastfill/v2/tests/test_active_objective_training.py` | `a91887bc39b6fc41892836337011d550ef7e1f152565198c673fa1c646a4e2e4` | 199 | fresh_full | 1–199 | passed |
| `fastfill/v2/tests/test_audit_c2_uniform_yaw.py` | `caf9b340c43cc43c234db31dae26ebcb260885ecd7b4534088c7f797496d10ec` | 37 | fresh_full | 1–37 | passed |
| `fastfill/v2/tests/test_batch.py` | `4bfd87258e77adeaee8b955a7c6be63633287949552a0a2f8cf835e6c419f9c8` | 102 | fresh_full | 1–102 | passed |
| `fastfill/v2/tests/test_batch_handoff_r2.py` | `c1815879aee64d8ac612efc1568c250ec3728da7dea85fe741e454fb537db4f3` | 247 | fresh_full | 1–247 | passed |
| `fastfill/v2/tests/test_bounded_io.py` | `8e5e3cf560af60214a7b6f631037d9ed9915dabc17231babfb2957b0ce5bb1a7` | 20 | fresh_full | 1–20 | passed |
| `fastfill/v2/tests/test_complete_cohort.py` | `759e8b4371f86e16b808c7dfe9d9f68172a8e2245455fb4ab01e8ae49ce84b70` | 190 | fresh_full | 1–190 | passed |
| `fastfill/v2/tests/test_data_policy_r2.py` | `1741715ed2a49f1135ecf0511719d64dda5d72cdd8b9531da1f2d160054995e8` | 222 | fresh_full | 1–222 | passed |
| `fastfill/v2/tests/test_evaluation_failure_latency.py` | `05087e2ea6a6eb662c0c281c60904033f9c17ddda79e8c8c983b33250ceca64c` | 74 | fresh_full | 1–74 | passed |
| `fastfill/v2/tests/test_fixed_capability_review.py` | `ec5d1d6b9205c07413580dd306045e7873c15ec4f6bf555a9a19130b98efa455` | 72 | fresh_full | 1–72 | passed |
| `fastfill/v2/tests/test_followups_r2.py` | `ea2349f67aebd4820a320e8505e0bc124ae98558fde4475a80a78ac71c600173` | 25 | fresh_full | 1–25 | passed |
| `fastfill/v2/tests/test_geometry_core.py` | `02a0045211b9981308802b9a2c33e84b39cb81cc2c1b4b6a7fa74d3dd39f8028` | 57 | fresh_full | 1–57 | passed |
| `fastfill/v2/tests/test_isambard_inputs.py` | `df7dbc686157973d6e91ecc8f4f741332d471dba70ace968f90c87030ecd8f5f` | 100 | fresh_full | 1–100 | passed |
| `fastfill/v2/tests/test_legacy_bridge_geometry.py` | `449a5f14ee92ff20f963307c27a12737c99af1dcb7705cff94ad4ad7cde090b3` | 202 | fresh_full | 1–202 | passed |
| `fastfill/v2/tests/test_legacy_verify.py` | `4aae1dde53cbd6347011a7e38ab4c37613be114e4e7631815858d60b0ca8ccb4` | 266 | fresh_full | 1–266 | passed |
| `fastfill/v2/tests/test_loss_config_review.py` | `099603ffd92e2a031b9b7ccd3f33433c84ddee801d38f391749feaf0cbb7911f` | 51 | fresh_full | 1–51 | passed |
| `fastfill/v2/tests/test_matching.py` | `49a0076971a04aadb2f1bc54424a1598943cf01911cff5ccbe0c22088371b241` | 124 | fresh_full | 1–124 | passed |
| `fastfill/v2/tests/test_model_loss_r2.py` | `9bb2ea4addd2f618d55007f706d9bee1fa983efb766cf2226e7c35a03bcdfd33` | 270 | fresh_full | 1–270 | passed |
| `fastfill/v2/tests/test_multisource_verify.py` | `bc2211ad12facc2ef948d8399100338f5c70e226b7cf2af2b5af14a63ae04ff0` | 350 | fresh_full | 1–350 | passed |
| `fastfill/v2/tests/test_roomgenbench.py` | `425cdb3e844c139a8e1cef1d2afd5a05f481801b090f2e88112a40b66d44b185` | 273 | fresh_full | 1–273 | passed |
| `fastfill/v2/tests/test_round10_wall_selection.py` | `73145f4864e7e4e1b9d2929a3d13647a4126901f29f481213d037781b654a4f4` | 297 | fresh_full | 1–297 | passed |
| `fastfill/v2/tests/test_round3_resume.py` | `17b0da3ddd2bd869296b2d0b04ce6cc23fb022e6c3a150a277c203eee1476acb` | 122 | fresh_full | 1–122 | passed |
| `fastfill/v2/tests/test_round5_spread.py` | `8de4bad64859834f2a580a839dd95737619a72d596d6a00ae1b5354df78cb726` | 57 | fresh_full | 1–57 | passed |
| `fastfill/v2/tests/test_round6_leftovers.py` | `01eaa0419aeddf39bbff659c8911a80499647402e769b15cdf5d7d161a646a0b` | 74 | fresh_full | 1–74 | passed |
| `fastfill/v2/tests/test_round7_reference.py` | `60ba06a16c7034173bfb35268171d0d8f88a148d04f6fdcb35e5d2c87969e65f` | 74 | fresh_full | 1–74 | passed |
| `fastfill/v2/tests/test_runtime.py` | `02347d2482a06fc2a7216dd62af5abcfc07f7214843dc207f321c946150fc657` | 216 | fresh_full | 1–216 | passed |
| `fastfill/v2/tests/test_schema_review.py` | `3e39974e02667fdbd44edf507c88f8a66d9c2fbf8fb355a7217115dc88f28e2d` | 83 | fresh_full | 1–83 | passed |
| `fastfill/v2/tests/test_slot_pooling_review.py` | `d42d8882aee7e3e38647bf5b8945f229ad05d5774d33b10500fdf9e6ee9269d9` | 40 | fresh_full | 1–40 | passed |
| `fastfill/v2/tests/test_supervision.py` | `fb9ea54d5235780c1887d633b97c9f2a55df7e9c276efca6f3eb3f23c91f8081` | 168 | fresh_full | 1–168 | passed |
| `fastfill/v2/tests/test_training_record_review.py` | `8a851b7e8e8b89c9e8879a1c509cc83a092e7e28b2b26790d027ad9a2d37f2c3` | 185 | fresh_full | 1–185 | passed |
| `fastfill/v2/tests/test_validation.py` | `b3475409ef63b20722603bd5db6ef69aee9b3e456a9cff2e1ee54d5159011659` | 91 | fresh_full | 1–91 | passed |
| `fastfill/v2/tests/test_verify_bundle.py` | `4fe758935f7b5a38bb35247aa209fd0763b286bdef2c1fdc229467a1c30eb698` | 64 | fresh_full | 1–64 | passed |
| `fastfill/v2/tests/test_vlm_judge.py` | `7cab3052819aac9e6819535e7d8920be17a922f41c5b46b665eea768cd9e0d1c` | 137 | fresh_full | 1–137 | passed |

No source changes were made to address the findings. Imported helpers, production implementations, external fixture availability and actual test outcomes remain outside this audit.
