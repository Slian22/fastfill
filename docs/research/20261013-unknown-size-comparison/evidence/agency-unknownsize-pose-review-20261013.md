Read-only independent review — 2026-10-13

Reviewed `fastfill/comparison/pose_targets.py` and `fastfill/comparison/tests/test_pose_targets.py` in `/Users/slian/Desktop/3D/Worldedge/OptiScene`. Summary: no P1 found; two P2 findings below. Frozen dimensions, final rounded-pose qualification, and supported-subtree motion behaved correctly in the inspected code and executed synthetic checks. This is engineering validation, not real-run quality evidence or approval for main training.

1. 🟡 **P2 — Malformed-input rejections violate the strict-JSON return contract.**

   Location: `fastfill/comparison/pose_targets.py:295–297` (public return contract at line 275); coverage gap: `fastfill/comparison/tests/test_pose_targets.py:297–311`.

   `repair()` returns the detached malformed room unchanged after validation rejects it. For an object with `yaw=float('inf')`, it returns `status='rejected'` and `invalid_geometry_or_category:a`, but `json.dumps(result, allow_nan=False)` raises `ValueError: Out of range float values are not JSON compliant: inf`. NaN geometry and nonfinite invalid search limits have the same persistence problem. The invalid-IR test explicitly supplies infinite yaw but never serializes that rejection; the strict-JSON assertion only runs through `assert_inventory` on other paths.

   Why it matters: a consumer persisting full results in the planned all-failed-cases ledger can abort on precisely the failure it needs to retain. Validated, finite prepared inputs avoid this issue, so this is a rejection-path defect rather than a demonstrated failure of the current pilot.

   Suggestion: preserve the detached raw room if required, but expose a documented, strict-JSON-safe failure receipt containing identity, reason and safely encoded invalid values; alternatively make the public rejection representation itself JSON-safe. Add strict serialization coverage for malformed geometry and limits. Do not silently omit these failures from the ledger or alter frozen dimensions to make them serializable.

2. 🟡 **P2 — Deep support forests require cubic preprocessing outside the search budget.**

   Location: `fastfill/comparison/pose_targets.py:96–107`; called at line 292 before search-limit validation/search.

   Every object's ancestor walk uses linear `cur in path` membership, and `members` then tests every possible ancestor against every stored path using another linear membership check. For a chain of N objects, both parts can require O(N^3) work; stored paths/members also require O(N^2) space. The graph work runs even with `max_candidates=0` or `max_passes=0`. There is no inventory/depth cap in this module.

   Reproduced with isolated synthetic `_forest()` calls, without checker/search overhead: 100 nodes 0.0136 s; 200 nodes 0.0578 s; 400 nodes 0.2578 s; 800 nodes 2.0887 s. These are local microbenchmarks, not estimates of the production workload. Existing budget tests use very small graphs and count checker calls, so they do not exercise this cost.

   Suggestion: validate limits before forest construction; use set-based active-path detection plus memoized traversal/child adjacency, and avoid repeated linear ancestor membership tests. If predictable per-room completion is required, enforce an explicit supported inventory/depth or work budget and retain budget-exhausted cases as failures. Candidate limits alone provide no wall-clock guarantee.

The core invariants otherwise look sound within the declared checker policy:

- **Dimensions and identity:** `_freeze` requires exactly the movable IDs and deep-copies the supplied numerical dimensions. Pose operations never resize or swap axes. `_finish` checks all non-pose object fields against the frozen copy. Fixed geometry and room metadata remain detached and unchanged. Synthetic tests cover GT-size independence and preservation on accepted and rejected searches.
- **Physical counts at returned precision:** normalization, candidates and final output round movable positions to cm and yaw to integer degrees; `_finish` reruns `check` on the actual returned room. Acceptance requires `valid` plus zero main, severe and fixed-blocking collisions. This intentionally retains existing center-support, tuck/contained/window and tolerance semantics; it is not a collision-free mesh guarantee. The final count dictionary also includes OOB, support and ceiling failures.
- **Support graph and movement:** missing/fixed parents and cycles reject explicitly; parent-before-child normalization recomputes Z from frozen heights. Candidate moves transform the full descendant subtree and include child footprints in boundary extents. Siblings may subsequently move independently. The extra offset three-level fixture forced a 90-degree root turn and verified descendant XY offsets and dimensions, then serialized and reconstructed the placement label and obtained exactly the recorded final audit.
- **Search-call bound:** the per-root allowance is shared across descendants and passes. For validated input with R roots and candidate cap C, the implementation makes at most R*C + 3 checker calls. The descendant-budget test confirms this accounting for its fixture. Each checker call itself includes quadratic object-pair work, in addition to the preprocessing finding above.

Verification executed read-only, with bytecode writing disabled:

- `PYTHONDONTWRITEBYTECODE=1 python -B -m unittest fastfill.comparison.tests.test_pose_targets -v`: **29 tests passed**, reported test time **0.116 s**.
- Transient in-memory probes: strict-JSON rejection failure reproduced; support-forest timings measured; offset subtree rotation and actual placement-JSON round-trip audit passed with all six final counts zero.
- No training, release-data scans, remote jobs or nested agents were run. No source/test files were edited. The only written artifact is this requested report.

Before the main teacher is integrated, retain the parent's stated boundaries: fit OOF priors once from the immutable train-only pool before common decoder admission; retain every failed case in the ledger; require both GT-J and P zero-qualified checks on the common decoded representations. This module cannot establish split provenance, paired admission or ledger completeness, and the existing tests do not exercise a real teacher/data run. The parent's running eight-step pilot remains preflight with no quality conclusion.

Reviewed source fingerprints (SHA-256):

- `pose_targets.py`: `2691a80aa7313e0c930993413d4d31ecdafb9bf4264bb9de3ccec0dc3ec9ae3b`
- `test_pose_targets.py`: `cf373bbe4bf195eddfa4e2b2f5c5aa01a3e009097b40d2919ab65ad45d349d5a`

The preservation and authoritative-final-audit design is a good foundation. Address the failure-receipt and graph-cost findings, then validate the paired admission/ledger integration before treating this as ready for the main teacher.
