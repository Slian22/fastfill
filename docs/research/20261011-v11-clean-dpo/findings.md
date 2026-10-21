# V1.1 clean DPO research findings

## Current verified status — 2026-10-13

SFT, clean DPO1 and DPO2 are complete. All 12 frozen diagnostic cells and 100,146 evaluation request records passed archived integrity/scoring checks; these are not independent scenes. Live server verification is recorded in [server status](audit-evidence/server-status-20261013.json). The earlier pending observations below are historical and superseded.

Ordinary test raw OOR is 0.092311 → 0.074409 → 0.061925; repaired zero-collision acceptance is 62.95% → 64.67% → 66.12%. Constrained test acceptance is 39.07% → 37.75% → 38.57%. Results remain diagnostic_only under target-derived ordering; no overall winner or paper-comparison claim follows. See [paper alignment and Agency review](paper-alignment-20261013.md) and [full statistics](evaluation_full/research-summary.md).

Paper benchmarking remains incomplete: common test/input identities, FID, GPT visual ratings and human UR have not been established. The five RoomGenBench examples have 5/5 generation, 4/5 mesh fitting and 3/5 geometry-valid outcomes, not full downstream acceptance.

## Fresh full audit — 2026-10-11 19:16 UTC

The latest full audit supersedes any broader interpretation of earlier passing tests: **all V1.1 prompts inherit target-derived object order/IDs**. This limits the current experiment to a diagnostic comparison under the historical contract; arbitrary size-unknown deployment is not validated. See [the full re-audit](reaudit-report.md), [current-hash read coverage](reaudit-coverage.json) and the frozen [interpretation addendum](interpretation-addendum.json).

All20 current V1.1 production files/scripts and12 tests were read fully;13 linked production dependencies were also reviewed. Confirmed recovery, cache, output isolation, lineage, gradient and legacy display defects were fixed locally and independently reviewed. Final integrated validation:428 tests and90 subtests passed. These tests do not remove the methodological limitation.

SFT completed all4 full cohorts(33382 requests); cleanDPO1 completed and merged; DPO2 continues. The original224-file trainer/scorer snapshot remains intact. A236-file helper snapshot adds real CPU rescoring and completed-training lineage verification;5 never-started evaluation/report jobs were replaced, with the84GPUh cap unchanged. NewDPO1 jobs7255085→7255086; final7255087→7255088; CPU SFT rescore7255089; CPU DPO1 lineage7255090; verified report7255091. Exact receipts live in [audit-evidence/audit-launch.json](audit-evidence/audit-launch.json). Independent SFT full-text verification completed successfully at19:21:42UTC for all33382 rows; DPO1 completion/merge lineage also passed in training and report interpreters. At19:23:03UTC DPO2 was340/1496steps; eight DPO evaluation cells and final synthesis remain pending. See [SFT verification receipt](audit-evidence/sft-verification.json); no DPO quality conclusion is yet available.

## Research question

Do two predefined DPO stages with audited preferences improve useful layouts after the existing V1.1 SFT checkpoint, when evaluated on the complete frozen V1.1 rooms and constraints?

## Current understanding

The dataset split is present and substantially larger than the previously reported 200-room checks. Those checks were file prefixes after prompt deduplication, covering HSSD200 and IL3D_3dfront only. They missed the other 16 test sources and all 409 ordinary / 408 constrained SceneSmith–SpatialGen source-heldout rooms. Training completion and a successful five-scene export therefore did not establish full benchmark quality.

The SFT representation is coherent as a conditional language-model baseline: input objects omit explicit target sizes, while output JSON contains sizes, positions and yaws. Fresh auditing found the input order and IDs still depend on target geometry, so the condition is not target-independent. It does not make numeric distance or physical validity part of the token loss. Preference training can improve useful geometry only insofar as its labels reflect the intended task. Local evidence shows that geometry-only positive selection omitted relational requirements and that both positive size checks and negative labels require explicit auditing.

The initial Stage 1 audit found 144 preferred outputs violating their prompt constraints. The completed sanitizer retains 7,244/8,000 pairs and drops 756 unique pairs. Overlapping reasons are 144 constraint-invalid positives, 475 positives failing the corrected reference-relative size gate and 215 rejected answers with no independently verified semantic defect. All 12,000 Stage 2 pairs pass this audit before token-length filtering. The corrected size bounds explicitly include the GT size, avoiding rejection of genuinely thin/large reference objects. These are label-integrity findings, not measured clean-model improvements.

The completed SFT checkpoint is reusable. The new chain starts fresh DPO1 from SFT and fresh DPO2 from clean DPO1 with frozen, audited pairs and corrected numerical training. Old DPO weights and diagnostics are preserved. Since the full matrix contains no matched unclean control, the planned comparisons cannot isolate the causal effect of cleaning.

## Verified evidence and present results

- All seven frozen release JSONL hashes and all 18 IR hashes were freshly checked against their manifest. Every one of 160,964 persisted UIDs joins to IR. Train/dev/test have 56,145 / 3,163 / 3,324 groups, with zero cross-split UID, group and alias intersections. Prior hash-bound QA supplies furniture/layout fingerprint checks; no asset-disjointness claim follows.
- Full evaluation denominators are 8,167 ordinary dev, 8,035 constrained dev, 8,647 ordinary test and 8,533 constrained test. Original rows and explicit failures are retained. The matrix is three methods × four cohorts, 100,146 requests.
- The separate V2 frozen 300+300 protocol has materially different object inventories: 234/300 validation and 239/300 test rooms differ in count from the matching V1.1 room. It is not reproduced here.
- Warm-start training provenance is incomplete. “Source held out” means excluded from the current V1.1 training split, not proven unseen by all earlier training stages. Historical test200 exposure is disclosed.
- No new full-cohort clean DPO quality result is available at the 2026-10-11 16:59:30 UTC deployment observation. H1, H2 and H3 remain pending. Do not substitute old prefix metrics, pair counts or training loss for these outcomes.

## Interpretation rules

Per-cohort dev paired differences in all-request `zero_collision_accepted` are primary. Use 2,000 paired `(source, group)` cluster bootstrap draws, seed42, room-weighted differences and percentile95% intervals. A positive interval supports only that primary proxy on that cohort; a negative interval indicates regression; overlapping zero or insufficient evidence is inconclusive. Coverage and size differences remain visible descriptive guards, without arbitrary new thresholds. Raw and repaired results are separate because postprocessing can account for apparent gains.

The protocol does not automatically choose a winner or launch another training job. Any further experiment needs a dev-grounded explanation, a new frozen protocol and available authorized resources. New test outcomes are reporting evidence only. A single seed, bounding-box checker acceptance and conditional dimension-only IoU do not establish general scene quality or a publication contribution.

## Lessons and constraints

1. Existing metrics files do not prove completion: match data, checkpoint, code, decoding and expected counts before reusing outputs.
2. A label saying “rejected” is not itself evidence of a defect. Validate both sides of each preference against the exact training prompt/reference.
3. Collision counts alone can reward implausible sizes. Keep size and constraint checks separate and make their reference dependence explicit.
4. Preserve all original requests and failures. Duplicate prompt removal can change denominators; parsed-only quality averages can hide failed rooms.
5. Use source-house cluster uncertainty rather than treating every room as independent.
6. Frozen cohorts from different model protocols are not interchangeable merely because their room UIDs match.
7. Unattended Slurm jobs can persist after chat; no perpetual agent loop is available or claimed. The bounded resource cap may end before all tasks complete, in which case report the missing work honestly.

## Open questions

Does clean DPO improve ordinary and constrained dev together? Do size, raw geometry and completion change in the same direction as repaired acceptance? Is any change concentrated in a few sources or larger rooms? Does Stage 2 add value after Stage 1? Can the full matrix finish inside the bounded allocations with consistent recovery? These questions remain empirical, and negative or mixed results are valid outcomes.

## Trajectory

No confirmatory clean-stage quality measurements yet. Verified label cleanup, the protocol commit and isolated deployment are complete. The next milestones are actual startup verification, completed training and the complete SFT/clean DPO evaluation matrix. Machine-readable outcomes belong alongside the immutable server run; `research-state.yaml` records the scheduled synthesis path without claiming that the report is already available.

## Deployment observation — 2026-10-11 16:59:30 UTC

The coordinator committed the protocol as `24342801681e62adc001dd647468e796ee25c5af` before submitting the clean-run graph. The deployed source fingerprint is `3e3eef72c51546cc0a22134e4409853570b988c4e5932823182d42007f6197c4`; all 224 source-file hashes, pair hashes and the preserved SFT link were verified, and preparation succeeded. The validation suite passed 256 tests and 90 subtests. These checks establish deployment integrity and tested behavior, not trained-model quality. The exact submission receipt is archived as [launch.json](launch.json).

| Chain | Submitted jobs | State at the recorded observation |
|---|---|---|
| Clean DPO training and continuation | 7252874 → 7252875 | First job PENDING Resources; training has not been observed starting |
| Full SFT evaluation and continuation | 7252876 → 7252877 | First job RUNNING on nid010057; continuation waits on dependency |
| Full clean DPO1 evaluation and continuation | 7252878 → 7252879 | PENDING Dependency |
| Full clean final evaluation and continuation | 7252880 → 7252881 | PENDING Dependency |
| Automated synthesis, CPU and zero GPUs | 7252889 | PENDING Dependency after all three final evaluation jobs |

All dependency links use `afterany`, allowing continuations and failure reporting; dependent work validates the required model/checkpoint and provenance and fails closed if they are absent or inconsistent. No successful clean training, smoke result or hypothesis support is implied by submission or RUNNING status. The existing 84 GPU-hour cap remains unchanged.

Code root: `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/automation/v11-clean-20261011-3e3eef72c515`. Run root: `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/fastfill_v1_1_clean_20261011`. Scheduled report: `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/fastfill_v1_1_clean_20261011/evaluation_full/research-summary.md`. Real startup/smoke results will be appended after coordinator verification.

## GPU startup verified — 2026-10-11 17:04:56 UTC

The authoritative SFT evaluation passed its actual GPU startup check. The first 256 generated and scored outcomes were persisted, with ordered UIDs, the frozen manifest digest and the generation-to-score digest independently verified. By this check, ordinary dev had progressed to 1,024/8,167 scored rows. This is execution evidence, not a full-cohort quality result. See [startup-smoke.json](startup-smoke.json).

At the latest queue check (17:04:21 UTC), clean training was still waiting for resources and the remaining jobs were waiting for dependencies. No new-training smoke result is claimed. All nine submitted jobs now have implicit Slurm requeue disabled; recovery is limited to the explicitly submitted continuation in each GPU chain, preserving the 84 GPU-hour allocation cap. The updated scheduler commands are archived in [launch.json](launch.json).
