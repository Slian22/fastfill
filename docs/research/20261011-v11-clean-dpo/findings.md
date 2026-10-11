# V1.1 clean DPO research findings

## Research question

Do two predefined DPO stages with audited preferences improve useful layouts after the existing V1.1 SFT checkpoint, when evaluated on the complete frozen V1.1 rooms and constraints?

## Current understanding

The dataset split is present and substantially larger than the previously reported 200-room checks. Those checks were file prefixes after prompt deduplication, covering HSSD200 and IL3D_3dfront only. They missed the other 16 test sources and all 409 ordinary / 408 constrained SceneSmith–SpatialGen source-heldout rooms. Training completion and a successful five-scene export therefore did not establish full benchmark quality.

The SFT representation is coherent as a language-model baseline: input objects omit target sizes, while output JSON contains sizes, positions and yaws. It does not make numeric distance or physical validity part of the token loss. Preference training can improve useful geometry only insofar as its labels reflect the intended task. Local evidence shows that geometry-only positive selection omitted relational requirements and that both positive size checks and negative labels require explicit auditing.

The initial Stage 1 audit found 144 preferred outputs violating their prompt constraints. The completed sanitizer retains 7,244/8,000 pairs and drops 756 unique pairs. Overlapping reasons are 144 constraint-invalid positives, 475 positives failing the corrected reference-relative size gate and 215 rejected answers with no independently verified semantic defect. All 12,000 Stage 2 pairs pass this audit before token-length filtering. The corrected size bounds explicitly include the GT size, avoiding rejection of genuinely thin/large reference objects. These are label-integrity findings, not measured clean-model improvements.

The completed SFT checkpoint is reusable. The new chain starts fresh DPO1 from SFT and fresh DPO2 from clean DPO1 with frozen, audited pairs and corrected numerical training. Old DPO weights and diagnostics are preserved. Since the full matrix contains no matched unclean control, the planned comparisons cannot isolate the causal effect of cleaning.

## Verified evidence and present results

- All seven frozen release JSONL hashes and all 18 IR hashes were freshly checked against their manifest. Every one of 160,964 persisted UIDs joins to IR. Train/dev/test have 56,145 / 3,163 / 3,324 groups, with zero cross-split UID, group and alias intersections. Prior hash-bound QA supplies furniture/layout fingerprint checks; no asset-disjointness claim follows.
- Full evaluation denominators are 8,167 ordinary dev, 8,035 constrained dev, 8,647 ordinary test and 8,533 constrained test. Original rows and explicit failures are retained. The matrix is three methods × four cohorts, 100,146 requests.
- The separate V2 frozen 300+300 protocol has materially different object inventories: 234/300 validation and 239/300 test rooms differ in count from the matching V1.1 room. It is not reproduced here.
- Warm-start training provenance is incomplete. “Source held out” means excluded from the current V1.1 training split, not proven unseen by all earlier training stages. Historical test200 exposure is disclosed.
- No new full-cohort clean DPO quality result is available at this protocol preparation stage. H1, H2 and H3 remain pending. Do not substitute old prefix metrics, pair counts or training loss for these outcomes.

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

No confirmatory clean-stage quality measurements yet. The first milestone is verified label cleanup and a source/data-bound protocol; the next is the complete SFT baseline and clean DPO matrix. Machine-readable outcomes and deployment receipts belong alongside the immutable server run, with synthesis linked from `research-state.yaml` once available.
