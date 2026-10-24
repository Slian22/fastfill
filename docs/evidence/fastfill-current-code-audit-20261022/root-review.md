# Root / adapters / tools / visual-and-judge review

Snapshot: `825fd13ea18f45835db1a242197359a77ea99f9e`.
Coverage: 43 top-level/adapter/tool production files plus 11 comparison visual/judge/critic modules. Production bodies read in full; selected embedded checks and test contracts examined. Exact source hashes and line counts are in `root-files.json`. External source datasets and real model/render/API execution are outside this review.

## ROOT-01 — P1: critic batch answers cannot enter the documented next round

`fastfill/comparison/critic_batch_025.py:101–103` publishes `answers/*.json` with only `id/status/error/returned_model/raw_final_text`. `critic_025.round_` at `:683–688` requires both `content_sha256` and `expected_model`. The batch docstring and `submit_025.sh:4` explicitly tell the caller to use the answers directory. Every nonempty set of those producer outputs fails the consumer before any edit can be applied, including successfully completed calls.

Reproduction `repro_root_comparison.py` uses the repository's fake SSE transport, runs the real batch client and immutable artifact writers for one successfully completed case, then passes the actual answers directory to the real `round_`. It raises `answers not made on this round's cases or model` and publishes no next round. No network/API request is made. Existing tests construct complete identity fields for `round_`, or test batch results only through `_step`, bypassing this join.

Correction: preserve the expected model and content hash from the frozen case in every answer, including failures; add a batch→round integration test. Do not remove the consumer's stale-input check. Raw `cases/*/results/*.result.json` are a separate route retaining those fields; this finding concerns the documented answers-directory route. No assertion about whether historical runs used that route.

## ROOT-02 — P2: QA reference repair omits the constraints used by runtime/evaluation

`fastfill/tools/qa_report.py:201` calls `snap_inside` without the saved prompt constraints or walls, whereas `fastfill/evaluate.py:102` supplies both. A constrained training reference can be reported repairable only because the QA calculation allows breaking a currently satisfied hard constraint.

`root_probes.py` constructs a chair 3 cm beyond the south wall and flush to the west boundary, facing a small box within the allowed angle. Unconstrained repair reports inside `1/1`; constraint-preserving runtime repair correctly leaves it outside and rejected (`inside_1mm_repaired=false`, `accepted=false`). The raw reference satisfies its faces constraint. This is a known issue still present at the current revision; it concerns QA statistics, not proof that runtime itself discards the constraint.

Correction: pass the saved constraints and boundary into the same repair and make QA's metric definition explicit. Regression should compare QA and evaluator behavior on the identical constrained reference.

## ROOT-03 — P2: a stalled HTTP request body blocks the entire v1 server

`fastfill/serve.py:74` performs a blocking `rfile.read(length)` without a connection/body deadline; `:125` runs a single-threaded `HTTPServer`. The 1 MiB bound does not bound waiting time. A client sending valid headers and one byte of an unfinished body prevents every later request from being handled.

`root_probes.py` opens a temporary loopback server with a fake model. A second complete request times out while the first body's final three bytes are withheld, and receives its response immediately after those bytes arrive. No public endpoint is touched. This is another previously documented issue still present. Default host is `0.0.0.0`; production exposure and upstream proxy timeouts were not inspected.

Correction: bound header/body read time and release stalled connections. Preserve serialized access to the model; changing to unbounded threaded model calls would introduce a different resource/concurrency problem.

## ROOT-04 — P2: judge agreement crashes on valid results with no overall score

`fastfill/comparison/judge_agreement_023.py:47–50` unconditionally unpacks non-missing metric pairs and calls Pearson correlation. The shared client accepts functionality/layout/aesthetics without the optional overall score, and judge_batch preserves that missingness. Four completed, valid cases across arms P/L with no overall score make `compare()` raise `not enough values to unpack (expected 2, got 0)`; no agreement output is produced. An empty common-completion set triggers the same class of error; one pair is also insufficient for Pearson. Later `:62–65` assumes every arm has a numeric overall mean.

Reproduction: `repro_root_comparison.py`, actual results in `root-comparison-probes.json`. Correction: report NA/counts for insufficient pairs and absent overall means, preserving the stated no-imputation policy; rank only defined arm means or state that the ranking is unavailable. The formal visual prompt asks for overall, but its absence is still accepted by the current client contract. This is not proof that a past agreement report encountered it.

## Positive checks and rejected hypotheses

The current Scan2CAD and SpatialGen adapters use full rotated extents for tilted boxes; the IL3D opening path uses the full projected span. Stale reports of those earlier defects must not be reused as current findings. Geometry consumers consistently use bottom-center positions and radians internally, with degrees confined to serialized prompts. The visual pipeline binds references/resources, preserves all UIDs in ledgers, and records arm-specific failures; a suspected missing-case mesh KeyError was ruled out because `_attach` deliberately writes the full reference cohort.

Additional reviewed modules: all source adapters, canonicalization/group splitting, build/filter/constraint logic, top-level training/inference/merge/download tools, visual_024, visual_cases, visual_pipeline, judge_batch, judge_agreement_023, judge_bundle_025, server_api_judge, critic_025, critic_batch_025, critic_client_025, render_assets. No production source or tests were edited.
