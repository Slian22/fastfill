# Bounded final review — metric input helper

Verdict: No high-severity or blocking correctness issue identified. No fix requested.

Scope: Static, read-only review of only `docs/research/20261011-v11-clean-dpo/paper-metrics-20261013/build_metric_inputs.py` and its sibling `test_build_metric_inputs.py`. No network, dataset scan, or test rerun; no existing files edited.

- Frozen-score parity: `build_metric_inputs.py:63–84` checks dependency file hashes against the frozen manifest and seven evaluator function source hashes against pinned values. `114–126` checks saved parse/error status and both raw/repaired OOR values with explicit finite numeric validation and 1e-12 tolerances. `259–272` verifies source/input stability before publication and binds output provenance. The tests cover source drift, score mismatch preventing publication, and the summary hash. Historical hash provenance was not independently re-established within this scope.
- Raw/repaired distinction: `129–159` retains raw output and layout separately, repairs only schema-complete predictions, and leaves repaired layout/OOR absent on schema failure. Tests check prediction sizes against different GT sizes, preserve extra IDs, and retain empty layouts on bad JSON/missing generation.
- Judging request isolation: `87–93` uses the frozen request serializer and adds an ID-derived category; `138–147` keeps GT geometry outside both judging fields. The test at `75–85` verifies that changing movable target size, position, yaw, parent, and anchor does not change the request, and checks its object field set. Inherited target-derived inventory order/IDs are explicitly disclosed at `240–241`.
- Failure denominator: `192–206` averages OOR only over schema-complete cases, explicitly reports that denominator, and retains failures in request counts with separate schema/parse counts. Numeric comparisons and failure-absence checks are reported separately at `226–229`; failures are not converted to zero OOR.
- Assets/reference handling: `141` supplies an empty, separate `asset_mapping`; `146` labels actual GT objects explicitly, and `243` disclaims supplied visual scores or asset mappings. No fabricated asset/reference is introduced by this helper.

Validation context supplied by parent, not independently rerun: 50 tests passed (41 client + 9 helper); helper coverage 91.7%; 1,375 scenes × 3 stages = 4,125 cases; 8,246 numeric OOR comparisons with zero mismatch; two SFT schema failures preserved. These counts are consistent with two comparisons per schema-complete case.

The explicit failure accounting and separate raw/repaired fields are clear and appropriate for this diagnostic export. Ready for the parent to wrap the design on the reviewed correctness criteria.
