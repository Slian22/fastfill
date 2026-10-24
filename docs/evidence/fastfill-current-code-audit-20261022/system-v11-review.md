# FastFill system + v1_1 current-source review

- Snapshot: `825fd13ea18f45835db1a242197359a77ea99f9e` (2026-10-22).
- Full production coverage: **28 files, 8,386 lines**, comprising every Python, `.sh`, and `.sbatch` file under `fastfill/system/` and `fastfill/v1_1/` excluding test directories. Per-file source hashes and review notes: `system-v11-files.json`.
- Read-only: no project source/test changes, no remote calls, Slurm submissions, GPU work or paid API calls. Review uses current sources, the existing graph for navigation, selected current tests, and CPU reproductions. Parent agent owns full-suite execution.

## Confirmed findings

### SV11-01 — [P1] Preserve declared support constraints when exporting the v1_1 showcase

**Location:** `fastfill/v1_1/roomgenbench_v1_1.py:320–333`, and `:367`.

`roomgenbench_scene_to_v1_1_room` turns an original `place_id` referring to another object into both `room.objects[].parent` and an `on` constraint. However, `placements_to_v2_handoff` constructs the **input condition**'s `support_parent` from the model's predicted `on`, and publishes `constraints: []`. Therefore a parseable prediction that ignores the requested support relationship rewrites the condition to fit its own output. The repair does not reinstate missing declared support, and the showcase's `repaired_valid` does not include prompt constraint satisfaction.

**CPU reproduction:** two objects, table and lamp; the source declares the lamp on the table, but the prediction places it elsewhere on the floor with `on=None`. Real `repair_v1_1`, `export_handoff`, and `reference_check` run without mocking geometry. Results:

- original declared constraint `['on', 'lamp_1', 'table_1']` is **false**;
- repaired geometry `valid=True`;
- exported condition supports become `['floor', 'floor']`, with no constraints;
- `reference_check.checks.fastfill.ok=True`;
- the original reference is judged against this modified condition and becomes `ground_truth.ok=False` (floor support violation).

This is a correctness and evaluation-contract failure, not a claim that every existing benchmark export is affected. It triggers whenever a validly parsed model changes/omits a declared furniture support (and the analogous prediction-first branch can replace a declared wall support). It affects the v1_1 RoomGenBench showcase/export path; do not extrapolate it to the amendment-022 generic system path.

**Suggested correction:** derive condition support and constraints from immutable request declarations, translate their IDs through the sidecar, then evaluate the prediction against that condition. If inferred support is intentionally allowed for an undeclared object, restrict it to that case and label it separately. Add an integration test with a validly parsed but wrong-support model answer; existing tests cover malformed support cycles and reference round trips, not this case.

### SV11-02 — [P2] Bind trainer recovery to the data/model/recipe before replacing the manifest

**Location:** `fastfill/v1_1/train_dpo.py:438–445`, `:496–504`.

The trainer overwrites `run_manifest.json` with current arguments before recovering `ckpt_latest`; checkpoint admission only checks serialization and whether `{epoch,it,step}` matches the current loader length. It never validates model/pair bytes or training settings against the previous run. Thus different pairs with the same admitted row count silently reuse the old checkpoint. This can mix experiments on an interrupted run or claim a completed run with no updates on the new data.

**CPU reproduction:** a real tiny Qwen3 model (16 hidden units, one layer), four preference pairs, one epoch, two optimizer steps. Finish training; swap `chosen` and `rejected` in the same `pairs.jsonl`; rerun the identical command/output directory. Current code reports `Resumed ... step=2/2`, takes zero new optimizer steps, republishes `final/completion.json`, and all final adapter tensors are exactly unchanged. The completion has `training_guard: null` and does not identify the new/old pair bytes.

**Scope qualification:** `isambard_v1_1_clean_dpo.sbatch` invokes `training_guard bind` before the trainer and correctly rejects this drift when that wrapper is consistently used. This does **not** mean the guarded clean run is already corrupted. The direct documented trainer entry point remains vulnerable, as does the legacy `isambard_v1_1_pipeline.sbatch` path, which calls `train_dpo` without that guard (`:168–180`, and its stage-2 equivalent).

**Suggested correction:** enforce a generic immutable training identity inside the trainer before writing any manifest or loading checkpoint state (pair hash, base/initial adapter identity, effective world size and loss/optimizer/sampler settings), or require and validate an externally supplied identity. Preserve intentional legacy recovery through an explicit, audited compatibility path. Test same-length changed pairs and changed model/settings, not just corrupt step counts.

## Verification artifacts

- `/tmp/fastfill-audit-20261022/system_v11_repros.py` — standalone reproduction source; only uses temporary files and CPU.
- `/tmp/fastfill-audit-20261022/system-v11-repros.json` — actual outputs from both reproductions.
- Tiny-Qwen environment: `/Users/slian/miniforge3/bin/python` (local Torch/Transformers/PEFT); no downloads.
- All seven shell/sbatch entry files passed `bash -n`.
- Full-read selected tests: system pipeline/sizing, v1_1 checkpoint recovery, v1_1 RoomGenBench recovery. All 18 scope test files were indexed for coverage/gaps; selected training-guard and lineage tests were also read. The full pytest run is consolidated by the parent agent; no pass count is invented here.

## Positive controls and limitations

The current system preserves fixed sizes across placer/escalation and uses a hard retry-counted API ledger. Missing-size API mode validates the whole input batch and budget before spending, rejects incomplete size responses and out-of-range rounding, and forbids reuse of old placed/O2 results with new sizes. Strict escalation preserves additional ranking flags through shard plans. Existing tests cover default-output stability and sharded/unsharded equivalence.

The new v1_1 full-cohort evaluator retains duplicate rows in order, binds chunks to identities and content digests, fails closed on missing generation and corrupted completion, and independently re-scores persisted outputs before audited research synthesis. Guarded clean training and merged-model lineage are materially stronger than the legacy wrapper.

No conclusion is made here about live remote jobs, actual 27B GPU execution, artifact contents only available on the external disk/server, or paper-quality generalization. Declared-support loss and direct-trainer drift are the two current issues verified in this scope. Other hypothetical concerns were not promoted to findings without a concrete trigger and reproduction.
