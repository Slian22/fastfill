# V1.1 automation fresh re-audit — 2026-10-11

Scope: full reads of automate.py, research_report.py, evaluation/legacy/clean-training sbatch scripts, five actual remote deployment/launch/status scripts, the submitted launch graph, protocol, the evaluation protocol dependency and training_guard.py, and all directly associated automation/report/lineage/guard tests. Exact full-read file hashes and line counts are in automation-coverage.json. This is a new source audit and adversarial reproduction, not adoption of an earlier review conclusion. Evaluation internals, geometry and trainer mathematics are separately owned by sibling reviewers.

## Confirmed defects and fixes

1. **P2: model-symlink output isolation bypass.** automate.prepare resolved output but compared it to unresolved model children. Reproduced a clean-run sft_merged symlink whose actual old model accepted an evaluation subdirectory, writing protocol/runner/report into the model. Current actual server output is safely separate, so there is no evidence this occurred in the real run. Fixed by resolving every model candidate before ancestor/equality checks; regression test first failed and now passes.
2. **P2: clean evaluator accepted a model without training lineage.** The prior runner used model_identity only; a structurally complete dpo1_merged lacking any adapter, training guard or merge receipt could run all four evaluations. This mattered because afterany dependencies intentionally allow downstream jobs after upstream failures. Fixed for new clean runners with a frozen clean-lineage mode, full preflight and end-of-run lineage checks, and a read-only training_guard verify-lineage CLI. Generic legacy evaluation remains explicit; SFT's earlier provenance limitation is preserved rather than invented.
3. **P2: stale completion after rejected identity revalidation.** Completed cells were still counted as complete after a later run-model failed preflight on identity/lineage drift. Reproduced both report and research synthesis incorrectly claiming old completed cells. Added validation_complete state, set false while validating and only true after successful checks; both reports reject cells from a failed identity revalidation. Existing legacy state files remain readable.

Fixes landed only in local current source; the root coordinator owns frozen deployment, replacement jobs and current runtime evidence. This reviewer made no remote mutations.

## Read-only lineage gate

API: `python -m fastfill.v1_1.training_guard verify-lineage --out CLEAN_RUN --tag sft|dpo1|final`.

For DPO the gate verifies the exact original frozen source files recorded in the training guard, the expected fixed recipe, base model identity, pair bytes and audit receipt, the pair auditor code in that same original frozen package, actual manifest paths/arguments, complete final training evidence, and the exact base/adapter/merged-output receipt. Final-stage verification recursively verifies DPO1. It does not compare historical source paths to the new runtime root and does not write or adopt any old artifact.

New trainers supply final/completion.json bound to guard hash, admitted count, world size, loader batches, expected and observed steps, and adapter/tokenizer hashes. The trainer and verifier schema were cross-reviewed with the training agent, including a direct producer/consumer test.

Existing trainers without completion.json use a strict read-only compatibility check: final adapter/config bytes must match ckpt_latest, CPU tokenization independently recounts admitted pairs using identical EOS/prefix/length rules, and safe CPU torch.load must show the exact final epoch/iteration/step. Missing evidence fails. No completion receipt is synthesized. The legacy recount additionally refuses a current prompt renderer whose source differs from the frozen training source. The world-size/batch/accumulation formulas were independently checked by the training reviewer: 7,244 admitted pairs imply 1,811 batches per rank and 906 total steps; 11,968 imply 2,992 batches and 1,496 steps.

## Independently verified behavior

- Actual original launch.json contains nine unique jobs and an acyclic dependency graph. All dependencies use afterany. Two six-hour four-GPU training allocations plus six six-hour one-GPU evaluations imply 84 GPU hours maximum. Requeue is disabled in the recorded applied update. The CPU synthesis job waits for all three final evaluation attempts. A failed/missing model cannot deadlock these dependencies; the new lineage gate fails the evaluation explicitly instead.
- The paired bootstrap resamples whole (source, group) clusters with room-weighted totals, uses 2,000 seeded draws, and pairs exact ordered uid/source/group identities. A two-cluster unequal-size adversarial example produces the expected 0.75 difference and [0,1] interval; failed generation remains false and all-request size IoU uses zero.
- The report requires all 12 validated cells for final completion, only uses dev for the four statistical comparisons, retains failures, labels fewer than two clusters inconclusive, and documents multiplicity and the absent matched cleaning-control arm.
- Runner commands retain all raw rows with no --limit and pass fixed 8192 completion / 40960 context budgets, chunk size 256 and strict vLLM mode. It records subprocess failure and requires completion receipts, not the mere existence of metrics.json.
- Shell syntax checks passed for evaluation, clean-training and legacy pipelines.

## Remaining limitations / compatibility surface

- The old isambard_v1_1_pipeline.sbatch remains a legacy diagnostic path: its Stage 4 still skips on nonempty metrics.json, evaluates only test cohorts, and lacks the new full-row receipt protocol. It must not be presented as the authoritative full-cohort research entry point. No old running experiment or old output was modified in this audit.
- The original launcher only disabled automatic requeue a few minutes after submission; there is no evidence of an actual requeue, and current receipts show all jobs disabled. Replacement submissions should include --no-requeue immediately, as communicated to root.
- Source identity does not freeze external Python package versions or firmware; the protocol's software manifest remains important. Historical SFT pretraining provenance is still unavailable.
- These tests establish automation correctness and failure behavior, not that training improves model quality. Actual stage results and complete full-cohort outcomes remain necessary.

## Validation

TDD: 18 new lineage/isolation tests first failed, then passed after implementation. Additional stale-success report tests also first failed and then passed. Latest targeted suite: 85 passed, including real sanitizer receipts, copied frozen source roots, new trainer completion producer/consumer, legacy torch checkpoint evidence, drift/corruption variants, and runner integration.

pytest-cov and coverage are not installed in the project Python; no dependencies were installed. Coverage was independently measured with Python stdlib trace over executable source lines (synthetic line 0 excluded):
- automate.py: 248/260 executable lines, 95.38%.
- research_report.py: 207/210 executable lines, 98.57%.
- training_guard.py: 265/275 executable lines, 96.36%.

Detailed line coverage: automation-line-coverage.json. Original launch validation: automation-launch-audit.json. Initial independent scratch reproduction: test_automation_adversarial.py (asserts historical bad behavior and is intentionally not the fixed-source regression suite).

## Deployment wrapper follow-up

Added isambard_v1_1_verified_evaluate.sbatch and isambard_v1_1_audit.sbatch after root assigned ownership. GPU wrapper keeps 1 GPU / 32 CPU / 115000M GPU memory / 6h, validates runner root/run/output, runs new readonly lineage checks, and only then executes the old frozen evaluation script. CPU wrapper requests 0 GPU / 32 CPU / 32G / 6h; SFT independently reproduces scoring using the absolute new verifier with the old scoring root and --reuse-verified, while report mode invokes the new audit_report module. Both explicitly specify brics.u6uv, workq, and --no-requeue. New helper cwd and PYTHONPATH prevent accidental import of old modules. Bootstrap JSON parsing never evaluates path values as shell code.

12 behavioral shell tests first failed then passed, including lineage failure preventing old evaluation, old evaluator return-code propagation, path mismatch blocking execution, unsafe-looking but literal path characters, audit report failure propagation, command arguments, resources, and bash syntax checks. No remote submission was performed by this agent. Added wrapper file hashes are in automation-coverage.json.

## Immutable installation and scheduler replacement review

Fresh readonly review found a pending-to-running race between old-job inspection and scancel. Root fixed this by holding all five old jobs, verifying all remain PENDING with RunTime=00:00:00 and Priority=0, and rechecking immediately before cancellation. New jobs are independently checked for Priority=0, no-requeue, pending state and correct GPU requests before any old work is held/cancelled. The obsolete report is cancelled first, then old continuations, then first evaluations; all seven new jobs are released only after all five cancellations.

10 local fake-command/temporary-archive tests passed: successful order, wrong GPU resources, unexpected unheld job, interrupted submission, old status transition, duplicate-receipt protection, valid install/no overwrite, archive hash mismatch, member hash mismatch, and path traversal rejection. No SSH or real scheduler command was executed by this reviewer. Install publication is a final rename after extraction and hash verification. Original 84 GPUh cap is unchanged. CPU audit wall-time allocation totals 12.5h, equivalent to 386 requested CPU-core-hours at 32/32/4 cores. Exact reviewed hashes and outcomes are in deployment-transaction-review.json.
