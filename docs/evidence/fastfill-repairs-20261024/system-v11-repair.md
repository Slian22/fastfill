# F01 / F05 / F13 repair evidence — 2026-10-24

Scope: `fastfill/v1_1/roomgenbench_v1_1.py`, `train_dpo.py`, `training_guard.py`, `fastfill/comparison/submit_023_judge.sh`, and their regression tests. The working tree is shared with other repair owners. This work used only local CPU execution and fake Slurm; no SSH, GPU, paid API, commit, or push was performed.

## F01 — preserve requested support during showcase export

The handoff formerly derived `condition.objects[].support_parent` from the prediction's `on`, and discarded declared `on` constraints. A lamp requested on a table could be predicted on the floor, exported as a floor-supported request, and reported as repaired-valid.

The handoff now derives support from the converted source objects' declared parent/wall metadata and translates the original `on` constraints into v2 object IDs. `repaired_valid` requires both the existing v1 geometry checks and the original constraints to hold. The prediction geometry remains unchanged by the exporter. Correct support still reports repaired-valid; missing support does not.

V2 deliberately reports an unverified tabletop as `support_surface_unknown`, including for ground truth. The repair preserves that uncertainty; it does not fabricate verified support surfaces or change v2 validation semantics.

Regression coverage includes source table/lamp conversion, repaired placements, v2 export/reference checking, and full showcase generation with a local fake model. Existing bundled RoomGenBench round-trip tests still pass.

## F05 — bind direct trainer startup and resume to its inputs

The trainer previously trusted a compatible optimizer position and overwrote `run_manifest.json` before verifying identity. A real tiny Qwen3 run could resume at step 2/2 after swapping every chosen/rejected response, changing model weights, or changing beta.

The trainer now checks `trainer_identity.json` before loading tokenizer/checkpoint or writing the run manifest. It binds model/tokenizer assets, pair bytes, initial adapter assets, effective training arguments, world size, dtype, audit receipt, optional external guard, and trainer code. Omitted warmup and explicit `0.05` normalize identically; the output/performance-log destinations are not training parameters. A failed rank-zero check is broadcast to DDP peers before any rank proceeds.

Compatibility behavior:

- Existing `training_guard.json` schema and byte contents are unchanged. Its original data/model hashes, executable recipe flags, world size, initial-adapter digest, audit receipt, and its own frozen code paths are verified.
- An older guarded run also requires a matching original `run_manifest.json` before adopting the new sidecar. Historical manifest bytes are retained. Rich wrapper descriptions are not confused with CLI flags.
- Existing unguarded checkpoints/manifests cannot establish provenance from their step counts; automatic adoption is rejected with an `unbound` diagnostic.
- The checkpoint/completion serialization and legacy lineage verifier are unchanged. Fresh unguarded direct-CLI runs gain the new sidecar without requiring pair UIDs that the trainer previously did not require.

Tests use real tiny Qwen3 training and actual optimizer/checkpoint publication, plus guard/adapter artifact tests. They cover same-input resume; changed pairs/model/beta; missing provenance; guarded legacy adoption; changed original code/guard settings; initial adapter/world-size/audit/code drift; and explicit/default warmup compatibility. Guard and manifest bytes are asserted unchanged on rejected resumes and guarded adoption.

Independent review found that a legacy manifest with JSON root `[]` would raise `AttributeError` outside the rank-zero validation catch and leave peers waiting at the broadcast. The added `manifest_type` regression first failed ([1 failed, 46 deselected](system-v11-review-red.txt)); the validator now checks the root type and raises the same identity `ValueError` before publishing a sidecar. This is included in the final GREEN run.

Two pre-existing CPU test fixtures mocked tokenizer loading without persisting any tokenizer assets. The new identity check correctly rejected those incomplete bases ([4 failed, 5 passed](system-v11-legacy-fixtures-red.txt)). Each fixture now calls its existing fake tokenizer's `save_pretrained` into the tiny base directory. No production validation was weakened and no numerical/adapter-byte assertion was changed. The final test run includes the multimodal Qwen3.5 recipe, init-adapter/reference behavior, and the pre-amendment-013 bit-identical training/adapter comparison.

The two downloaded real 27B receipts (`a021/27b` and `a028/27b-joint`) were additionally checked against the new CLI defaults and legacy guard validator. Both recipes/schemas are compatible and their local receipt bytes remained unchanged. This local check substituted the already recorded dataset/code hashes, so it is a metadata compatibility check, not a new remote hash verification. Actual server artifact verification belongs to `../fastfill-live-training-check-20261024/`; these repairs do not claim either historical training suffered F05.

## F13 — require a successful, verified Judge smoke before full submission

The old gate checked only that `smoke/results` existed, although the client creates it before issuing an API request. Ten failure variants all reached fake `sbatch` before the fix.

The full gate now requires integer exit code zero, the exact first two bundle cases, unchanged prompt/image hashes, and two completed attempt/result pairs matching the current endpoint, model, reasoning effort, and request identity. It reuses the client's strict stored-result validation, including parsed score consistency. This preflight performs no API call. Missing, partial, failed, stale, or malformed records stop before submission.

The local fixture provides fake `squeue`/`sbatch`, an `example.invalid` endpoint, and stored synthetic API results. Ten rejection cases cover failed/missing completion, empty/partial results, failed result, invalid scores, wrong returned model/request identity, changed case ordering, and changed image bytes. A valid two-case smoke still submits the unchanged 696-case/four-worker full command. No real Slurm or paid job was submitted.

## RED evidence

Before production edits:

```text
/Users/slian/miniforge3/bin/python -m pytest --import-mode=importlib -q \
  fastfill/v1_1/tests/test_roomgenbench_recovery.py \
  fastfill/v1_1/tests/test_dpo_checkpoint_recovery.py \
  fastfill/comparison/tests/test_submit_023_judge.py \
  -k 'changed_inputs or unbound_legacy or guarded_legacy or requested_support'
11 failed, 59 deselected in 9.58s

/Users/slian/miniforge3/bin/python -m pytest --import-mode=importlib -q \
  fastfill/comparison/tests/test_submit_023_judge.py
10 failed, 1 passed in 3.71s
```

Original logs are retained in this evidence directory: [F01/F05 RED](system-v11-red.txt), [F13 RED](system-v11-judge-red.txt). The first command selects F01/F05 regressions; the Judge regressions are exercised by the second command. The positive support test and additional resume boundary cases were added during the repair.

## GREEN evidence

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/Users/slian/miniforge3/bin/python -m pytest --import-mode=importlib -q \
  fastfill/v1_1/tests/test_dpo_checkpoint_recovery.py \
  fastfill/v1_1/tests/test_training_guard.py \
  fastfill/v1_1/tests/test_automation_lineage.py \
  fastfill/v1_1/tests/test_roomgenbench_recovery.py \
  fastfill/v1_1/tests/test_train_dpo_a021.py \
  fastfill/v1_1/tests/test_train_dpo_init_adapter.py \
  fastfill/comparison/tests/test_submit_023_judge.py
137 passed in 16.54s

bash -n fastfill/comparison/submit_023_judge.sh
# exit 0

git diff --check -- <owned production and test files>
# exit 0
```

Full output: [final GREEN](system-v11-final-green.txt); the preceding 128-test run is also retained [here](system-v11-targeted-green.txt). Broader regression and coverage are coordinated by the root reviewer. DDP error propagation is implemented but was not executed on multiple GPUs in this local validation.

## Read-only cross-review of the comparison repair owner

Reviewed `evaluate_llm.py`, `decide_012.py`, `llm_agent.py`, and `mesh_collision.py` changes against their callers, tests, and amendment 012 §4. F03 preserves the explicit no-reasoning semantic failure through `geometry.score` and `a021_select.finish`, while ordinary B2 parser rescoring remains possible. F04 uses the declared absolute GT-bias difference, each variant's own complete-room mean, inclusive .03 boundary, and refuses a missing mean. F15 caps the sampling denominator to available rooms; F16 retains missing analysis as missing.

A remaining F09 boundary was reported to its owner and the root reviewer: with normal request sizes `[.5,.5,.5]`, a supported layout whose floor parent's `pos[2]` is JSON integer `10**400` raised `OverflowError` in support-z arithmetic before reaching the new serializer catch. A returned size axis with that integer similarly failed the size-difference arithmetic. This was reproduced directly with the existing `_supported_room` fixture; no source file owned by the comparison agent was edited.

The comparison owner resolved it by requiring finite vectors before size/support arithmetic and handling `math.isfinite` integer-conversion overflow. Independent review confirms that malformed positions stay parser failures, while a returned huge size is still replaced by the request size under B2. No additional blocking issue was found in this repair set.

Independent post-fix tests: [22 passed](system-v11-comparison-cross-review-final.txt), including both full-batch overflow cases, followed by [four passed](system-v11-comparison-overflow-cross-review.txt) for positive/negative huge integers in sizes and parent z. The earlier cross-review baseline was [21 passed](system-v11-comparison-cross-review.txt). Commands:

```text
/Users/slian/miniforge3/bin/python -m pytest --import-mode=importlib -q \
  fastfill/comparison/tests/test_decide_012.py \
  fastfill/comparison/tests/test_llm_open.py::test_o2nt_semantic_failure_survives_evaluation \
  fastfill/comparison/tests/test_llm_open.py::test_nonfinite_placement_is_one_room_failure_not_run_abort \
  fastfill/comparison/tests/test_agents_016.py::test_rescoring_keeps_parser_failures_repairable_and_runner_failures_failed \
  fastfill/comparison/tests/test_agents_016.py::test_size_enforcement_leaves_nonfinite_geometry_as_a_parse_failure \
  fastfill/comparison/tests/test_mesh_collision.py::test_limit_samples_each_available_room_at_most_once \
  fastfill/comparison/tests/test_mesh_collision.py::test_no_successful_mesh_analysis_still_publishes_missing_reasons
# 22 passed in 6.88s

/Users/slian/miniforge3/bin/python -m pytest --import-mode=importlib -q \
  fastfill/comparison/tests/test_agents_016.py::test_size_enforcement_handles_integers_too_large_for_float
# 4 passed
```
