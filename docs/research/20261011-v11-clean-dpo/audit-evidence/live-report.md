# V1.1 deployed automation live re-audit

Read-only evidence captured 2026-10-11T19:00:19.366953+00:00–2026-10-11T19:00:45.020629+00:00; exact metric recomputation 2026-10-11T19:02:17.141507+00:00.

## Verified deployment and provenance

- All 224 frozen source files match source-manifest SHA-256; local and remote manifest bytes agree. There are no extra unmanifested Python files. Current evaluation code hashes equal frozen protocol hashes.
- The Slurm-spooled scripts for jobs 7252874, 7252875, 7252878 and 7252880 exactly match the frozen batch script bytes. Submitted code/run paths and afterany dependencies match launch receipt.
- Clean run remains isolated. SFT symlink resolves to the expected original SFT merged model. All four full input dataset identities match frozen protocol.
- DPO1 and DPO2 training guards match current base-model bytes, pair-file bytes, audited pair-receipt bytes and training/merge code. Actual run_manifest CLI settings equal guard settings.
- Clean DPO1 final adapter passes structural checks. Its merge lineage receipt validates against exact SFT base, final adapter, merge code and merged output. DPO2 was still training; its final output was absent, as expected.

## Current jobs

| Job | Name | State | Start UTC | Elapsed |
|---|---|---|---|---|
| 7250194 | fastfill-v1.1 | RUNNING | 2026-10-11T13:55:05 | 05:05:14 |
| 7250195 | fastfill-v1.1 | PENDING | Unknown | 00:00:00 |
| 7252874 | ff-clean-train-1 | RUNNING | 2026-10-11T17:13:13 | 01:47:06 |
| 7252875 | ff-clean-train-2 | PENDING | Unknown | 00:00:00 |
| 7252876 | ff-full-sft-1 | COMPLETED | 2026-10-11T16:59:09 | 01:48:25 |
| 7252877 | ff-full-sft-2 | COMPLETED | 2026-10-11T18:47:43 | 00:00:16 |
| 7252878 | ff-full-dpo1-1 | PENDING | Unknown | 00:00:00 |
| 7252879 | ff-full-dpo1-2 | PENDING | Unknown | 00:00:00 |
| 7252880 | ff-full-final-1 | PENDING | Unknown | 00:00:00 |
| 7252881 | ff-full-final-2 | PENDING | Unknown | 00:00:00 |
| 7252889 | ff-research-report | PENDING | Unknown | 00:00:00 |

Clean DPO1 finished and merged before DPO2 started at approximately 18:46 UTC. DPO2 latest observed step was 130/1496, loss 0.5683. Original diagnostic pipeline completed DPO2 1496/1496 and merge; it is proceeding through its legacy evaluation. No fatal error lines were found in the inspected clean training/SFT/original job logs.

The first clean training allocation ends 23:13 UTC. At the observed approximately 6–7 s/step DPO2 pace, projected completion is around 21:20 UTC; this is an estimate, not a completion claim. One additional bounded continuation remains scheduled.

## Full SFT evaluation independently verified

| Split | Rows | Generated/scored chunks | Parse failures | Error rows | Zero-collision accepted |
|---|---:|---:|---:|---:|---:|
| dev_rooms | 8167 | 32/32 | 3 | 25 | 5193 |
| dev_constrained_rooms | 8035 | 32/32 | 6 | 16 | 3100 |
| test_rooms | 8647 | 34/34 | 5 | 24 | 5443 |
| test_constrained_rooms | 8533 | 34/34 | 7 | 29 | 3334 |

- All four completion receipts validate against current model bytes and frozen data/settings/code identities.
- All 33,382 prediction rows match raw input UID order. Every generated/scored chunk identity, generation-content hash, row count and final prediction publication matches; no chunk issue was found.
- Recomputing every stored metric from predictions in the original vLLM Python environment reproduces all four metrics files exactly.
- Recomputing in the training environment (Python 3.12.15) yields identical integer/count/rate metrics and floating mean differences no larger than 1.56e-15, consistent with Python floating summation changes; no substantive metric discrepancy.
- SFT continuation attempt 2 completed in 16 seconds and validated all four previously completed cells without regenerating them.

Eight DPO evaluation cells remain pending and the final statistical research report has not yet run. This audit does not claim 12/12 completion or improved model quality.

## Scope and limits

This probe only read server state and files; it made no server file edits, submissions, cancellations or model changes. It does not validate all mathematical/scientific choices in the evaluation scorer; separate code reviewers own those findings. The supplied guide’s 60-second polling constraint was observed for repeated scheduler probes. The model inference and training software environments are distinct; environment immutability is not guaranteed by the source-only manifest.

Machine evidence: `live.json`, `live-metrics.json`; reproducible read-only probes: `remote_live_probe.py`, `remote_metrics_probe.py`.
