# Independent unknown-size geometry review — 2026-10-13

Read-only review of `fastfill/comparison/geometry.py`, `fastfill/comparison/tests/test_geometry.py`, and `docs/research/20261013-unknown-size-comparison/protocol.md`, with narrow inspection of their contract, validation, and repair dependencies. No code changes, nested agents, datasets, rendering, API calls, inference, training, or server operations. Findings concern implementation behavior, not measured model quality.

**P1 — Freeze component/source strata before conditioning on successful pairs.**

Location: `fastfill/comparison/geometry.py:158–169` (especially the filter at 164 and signature construction at 168); protocol line 17.

`paired_intervals` constructs each component's source signature only from rows where both arms have a non-null metric. Thus a generation failure can move a genuinely cross-source component into a different bootstrap stratum. This conditions the resampling design on model success, and can manufacture zero-width confidence intervals despite variation between independent components. The remaining rows of a component are kept together; the defect is the outcome-dependent stratum assignment.

Adversarial probe 2 used just four paired rows. Both original components span sources S and T:

| UID | Source | Component | J size IoU | P size IoU |
|---|---|---|---:|---:|
| 1S | S | g1 | 0 | 0 |
| 1T | T | g1 | null (failed) | 0 |
| 2S | S | g2 | 1 | 0 |
| 2T | T | g2 | 1 | 0 |

With 2,000 replicates and seed 42 the implementation returns `paired_rooms=3`, `components=2`, `singleton_source_signature_strata=2`, estimate `2/3`, and `ci95=[2/3,2/3]`. Its post-filter signatures are `(S)` and `(S,T)`, making both components unsampleable alternatives. With the original `(S,T)` signatures, the four equiprobable ordered two-component draws yield means `[0, 2/3, 2/3, 1]`, hence percentile endpoints `[0,1]`. This is a demonstration of the resampling defect, not a claim that two clusters suffice for reliable population inference.

Action: construct global component membership and source signatures from the full paired UID inventory before the per-metric null filter. Resample those intact components and then calculate conditional numerators/denominators. Explicitly handle draws with no eligible observations. Return an unavailable/partial-uncertainty status when the frozen design has singleton strata, rather than presenting their unestimated variance as an ordinary complete 95% CI; the existing count is useful but does not fix the interval. Add the four-row regression above, verifying both cross-source integrity and nondegenerate draws under differential failure.

**P2 — Contract-valid support on fixed objects is always rejected by geometry validation.**

Location: `fastfill/comparison/geometry.py:35–43` and `96–101`; dependency `fastfill/validate.py:112–120`. Contract: `fastfill/comparison/contract.py:44` and `268–287`.

The contract explicitly accepts a fixed ID in `on`. The scoring wrapper forwards it to `check`, whose parent lookup contains only movable objects. Consequently any such support is reported missing even when its geometry is exact. This lowers both raw and repaired all-request geometry-success rates for outputs that use a permitted support relation; affected prevalence in real inference is unmeasured.

Adversarial probe 1: a fixed counter of size `[2,2,1]` at `[2,2,0]`, with a movable lamp of size `[0.2,0.2,0.2]` at `[2,2,1]`, explicitly supported by that counter, inside a 5×5 room. `validate_output` accepts the output. Scoring returns `complete=True`, `size_iou=1`, `parent_accuracy=1`, `oor=0`, but `support_fail=1`, `geometry_valid=False`, and both geometry-success flags false.

Action: resolve support parents against the combined movable/fixed identity index while retaining movable-only denominators and overlap calculations. Ensure boundary repair also respects fixed-parent support footprints (`snap_inside` currently looks up parents only in the movable placement map). Add exact fixed-support and invalid-height/footprint cases; do not drop the permitted support relation or substitute GT movable support.

**P2 — Source-macro estimates have no matching paired uncertainty calculation.**

Location: `fastfill/comparison/geometry.py:126–136` and `146–182`; protocol line 17.

`summarize` returns source-macro point estimates, but the only interval routine explicitly estimates a room-micro paired mean. It collapses sampled components to one sum/count and discards the source-level quantities needed for macro uncertainty. Source/stratum tables alone will not supply a source-macro interval. In probe 2, the conditional source-macro J−P difference is `0.75`, whereas the returned room-micro paired difference is `2/3`; the latter interval cannot be attached to the former.

Action: add a separately named paired source-macro estimator and bootstrap interval, or explicitly mark macro uncertainty unmeasured until implemented. Within each replicate, use the same global component multiplicities across all sources, recompute source means, then average sources equally. Freeze and label the source set and conditional cohort: per-arm conditional means and shared-success paired means are different estimands, and sources with no eligible observations must not silently change the comparison. Specify handling of replicates missing an eligible source. Cover unequal source sizes, a cross-source component, and differential failures in a small regression test. This finding does not request running the pending 8,647-room aggregation.

**Test evidence and limitations**

- The six existing synthetic unit tests pass: `PYTHONDONTWRITEBYTECODE=1 python -B -m unittest fastfill.comparison.tests.test_geometry -v` (6 tests, 0.006 s in the captured run). Output: `/tmp/agency-unknownsize-geometry-tests-20261013.txt`. Two small adversarial probe scripts were run; their decisive results are above. No repository test was added or modified. The stated 95.2% statement coverage was supplied by the requester and was not remeasured; it does not establish correct resampling semantics or branch coverage.
- OOR is the sum of pairwise axis-aligned XY overlap areas divided by summed movable footprint area (`validate.py:124–130`), separate from out-of-boundary validity. The existing overlap test checks `0.5` for two coincident unit boxes with zero OOB objects. This legacy overlap quantity ignores yaw and vertical separation and can exceed 1 for many coincident objects. That is the inspected definition, not an out-of-room percentage. No claim of equivalence to a separately reproduced paper implementation is made.
- P sizes are taken from the frozen request at `geometry.py:93–95`; both raw layout and repair use those sizes. Probe 1 additionally held that P request fixed at `[0.1,0.1,0.1]` while changing only GT size labels from `[0.2,0.2,0.2]` to `[0.8,0.8,0.8]`: raw and repaired layouts remained identical, while size IoU changed from `0.125` to `0.001953125` and log MAE from `0.693147` to `2.079442`. This checks evaluation-side size isolation, not train-only prior fitting, OOF construction, upstream eligibility, or actual inference prompts.
- Dimension-only IoU and absolute log error penalize shrinking: halving all three dimensions produces IoU `0.125` and mean absolute log error `log(2)`. They do not identify the direction of size bias; a claim specifically about systematic shrinkage requires a signed size/volume diagnostic. Low overlap alone cannot establish improved layout quality. No quality scores for the new models were measured.
- Failed requests remain in all-request rate denominators and have null conditional scalar scores. Paired scalar intervals retain only shared non-null pairs. These conditional estimates do not identify performance on failed requests, particularly when failure rates differ by arm/source; accompanying failure rates and cohort counts remain essential.
- The six tests do not exercise multi-component cross-source resampling, macro intervals, fixed supports, or uneven cluster-size distributions. The repair test checks size preservation and one movement, but does not verify the total 10 cm displacement bound through a support chain. Repair and raw scoring share one exception path, so resilience of raw scores to repair failures is also untested. No large-data or rendering/API integration was attempted. The planned full-test and source/stratum CLI aggregation remains outside this review, not a missing-result defect.

Reviewed-file SHA-256 values (captured at report creation):

- `fastfill/comparison/geometry.py`: `02e4ba547ec9e319f29e6219aebadea9f3ffe7d64fa0420bd435595efd272b42`
- `fastfill/comparison/tests/test_geometry.py`: `3137ff4bd675202a1a3ca850e39efce1759e1433afaa37b434d7dd3a57e723e7`
- `docs/research/20261013-unknown-size-comparison/protocol.md`: `f2fc11ec56d927f551a438fd14e83ff4ca64e5e4a0e60d4b5e47968e7853706e`
