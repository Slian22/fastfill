# Independent V1.1 evaluator and completion re-audit — 2026-10-11

Scope: full read of 11 original files, 4,058 lines (evaluator, protocol, parser, repair, both evaluator tests, shared geometry/constraint checker, original evaluator, request repair, anchors and serialization). Initial file hashes and exact full-read coverage are in `evaluation-coverage.json`. Graph MCP was unavailable (Transport closed); explicit known files were read directly. No remote code or scheduler changes were made by this reviewer.

## Result

No confirmed P0/P1 error in the current authoritative all-row scoring path. One reproducible P2 recovery-integrity defect was found and fixed locally for future runs: persisted scored chunk values were trusted without a digest over their payload. The existing frozen `3e3eef72c515` evaluator has that defect. There is no evidence that its actual scores were altered. Its running source must remain unchanged; independent CPU re-scoring provides a stronger check of its completed outputs without re-generating GPU predictions.

### P2: scored payload alteration could survive recovery

Original `evaluate_v1_1.py:447–453` checked chunk identity metadata, list length and ordered UIDs but not the actual score payload. The generated chunk was bound into its scored chunk, yet the score payload itself had no integrity digest. Minimal adversarial reproduction:

1. Generate and score a chunk, interrupt before the next chunk.
2. Change one saved `accepted: true` to false, retaining the UID, row count, identity and generation digest.
3. Resume. The original evaluator accepted the altered score, published metrics and wrote a valid completion marker. A five-row fixture changed accepted rate from 1.0 to 0.8.

`test_evaluation_adversarial.py` records the original four-test reproduction/real-scoring checks. Its first test intentionally demonstrated the defect and is not a desired-behavior regression test.

Fix implemented in `evaluate_v1_1.py`: generated and scored chunks now carry `content_sha256`; loading and publishing verify it, and publishing rechecks generation/score linkage. Missing hashes are explicitly rejected, so old chunks cannot silently enter a newly patched run. Four new regression cases failed before implementation and passed afterwards, including generated payload corruption before any scored chunk exists.

### Separate CPU verification for the already frozen experiment

New `verify_outputs.py` is an external audit CLI. Call the NEW script by absolute path, with the OLD evaluation directory and OLD frozen scoring root:

```text
python NEW_ROOT/fastfill/v1_1/verify_outputs.py \
  --output EVAL_ROOT --tag sft --scoring-root OLD_FROZEN_ROOT \
  --workers 32 --reuse-verified
```

`--all` verifies all three tags. The launcher reads `runner.json[python]` and runs an isolated `-I` child with the explicit frozen root prepended to `sys.path`. The worker rejects a different scorer path, protocol path, or Python interpreter. This matters: Python 3.12 changed float summation; mixing its summary implementation with the original Python 3.11 interpreter can create ~1e-15 differences. No tolerance was introduced; identical original scoring/runtime is required.

Before scoring, each tag requires all four frozen cohort completions. It validates protocol and runner hashes, state identities, complete model assets, full dataset hashes, scorer code hashes, and completion/predictions/metrics hashes. It then independently parses every persisted model output, recomputes every scoring field with the original scorer, verifies ordered UID/evaluation_row and exact canonical-JSON equality, and recomputes the complete metrics summary. Inputs and artifacts are identified again before receipt publication.

Separate `verification/{tag}.json` receipts include identity-bound cells, original artifacts, verifier SHA, Python/sys.version/interpreter path and package versions. Their status is running, complete or failed. An exception cannot leave a newly issued pass receipt. The optional cache accepts only a complete receipt with four exact cells and matching input/artifact/verifier/runtime identities. It does not prove generation authenticity, independent scientific replication, or model quality; it verifies that the persisted outputs produce the reported scores.

## Confirmed properties and interpretation

- Authoritative resume mode requires all rows, no limit, a model and no GT-only mode. Raw duplicate UIDs/prompts remain separate rows.
- Missing generation, malformed JSON, missing/extra/duplicate IDs, nonfinite numbers and cyclic/unknown supports fail the primary all-request acceptance denominator. None is explicitly classified `missing_generation` on this path, so it cannot select the GT-only scoring branch.
- Generation is saved before scoring; a scoring interruption can reuse saved text. One model session is reused. Authoritative mode does not silently change backend.
- Paired generation/score metadata, ordered cohort, full dataset/model/code/settings identities and final publication receipts prevent legacy 200-row metrics from satisfying full-cohort completion. Final source/data/model drift blocks publication.
- Raw and repaired geometric fields remain separate. The primary `zero_collision_accepted` includes fixed repair, size lower-bound checking, constraints and zero main/severe collisions. It is not an assertion of no tuck/contained box overlap, mesh-level physics or unrepaired model skill.
- Primary acceptance, validity and constraint outcomes retain failed requests. Collision/object rates and mean size IoU are conditional on complete parsed answers. Existing report text discloses this; these secondary values must always be read together with `complete_rate`.
- Perturbing GT positions/yaw while leaving the condition and evaluated prediction unchanged did not change the predicted judgment in the actual-scoring regression. Predicted sizes/positions/yaw/parents come from generated text; repair uses predicted geometry.
- Repair/check have a wall-anchor branch based on the reference room's `anchor`, which is not exposed in the model prompt. A fresh full scan of all four current cohorts found zero wall anchors and only floor/object anchors; thus this branch does not affect the current 100,146-request matrix. It requires a new protocol review before applying these metrics to wall-inclusive data.
- Group cardinality using group alone equals source+group cardinality on each of the four current datasets. Statistical cluster identity remains a separate report concern.

## Validation

- Existing evaluator/protocol suite: 38 passed before edits.
- New chunk regression RED: 4 failures, then GREEN with all 42 evaluator/protocol cases passing.
- Final focused evaluator/protocol/independent-verifier suite: **69 passed**, including CPU geometry, successful and missing outputs, completion re-signing after wrong scores/metrics/output, interpreter isolation, multiprocessing spawn, cache validation and mid-run identity drift.
- Standard-library executable-line tracing: new verifier **207/219 = 94.52%**; evaluator chunk/resume/publication region 431–604 **120/147 = 81.63%**; protocol **205/217 = 94.47%**. Subprocess execution was excluded from these conservative counts; CLI and worker paths are also exercised in process. See `evaluation-test-coverage.json`.
- `py_compile` and scoped `git diff --check` passed.

Remaining limits: statistical performance has not been independently established here; model training provenance and GT-derived object ordering are owned by separate audits. The original identity manifest pins repository code and assets but not full installed-package/environment bytes. The new CPU receipt records the active versions and uses the original interpreter; this is not a hermetic environment guarantee. Structure/hash validation cannot itself prove that every model tensor is semantically the intended one; model loading and training lineage validation remain necessary.
