# Independent three-patch review — 2026-10-13

**Conclusion: no patch-introduced correctness/regression findings; no CRITICAL/HIGH findings or deployment blockers identified in the three reviewed changes.** The changes are compatible with the frozen producer and the planned full evaluation protocol. This is a source review conclusion for the hashes below, not confirmation that any remote deployment occurred.

## Scope and baseline

Read all four requested production files and the three requested tests directly. The tests are under `fastfill/comparison/tests/`, not the repository-root `tests/` directory. Compared production source against `docs/research/20261013-unknown-size-comparison/automation-code/pipeline-9c4d97dbd736`. Independently checked all 33 files against that snapshot's `MANIFEST.json`: zero mismatches.

The actual production delta is exactly three changes:

- `geometry.py:56–66`: enforce the raw-relative 10 cm displacement budget after the candidate repair; roll back the entire candidate on excess.
- `train_pair_helpers.py:144–150`: require exact movable-input fields and a nonempty string category.
- `final_tables.py:13,55–60`: reuse the producer's summary validator and require judge strata to match populated paper strata.

`judge_batch.py` is byte-identical to the baseline. The frozen snapshot contains no copies of the requested test files, so their complete contents are supporting coverage, not a reconstructable test delta. Unchanged implementation issues, style, and speculative infrastructure changes are outside this review.

## Findings and line evidence

**Open findings: none.** The following are verified patch behaviors, not requests for further changes.

### 1. Repair budget: correct cumulative rollback

`fastfill/comparison/geometry.py:58–66` keeps original positions separate from the mutable candidate and compares each final candidate position with its original position using Euclidean distance. The new check follows the complete repair pass, so it catches displacement inherited from an ancestor plus a child's own subsequent move. `fastfill/interface.py:283–324` confirms that each support subtree moves together, which is why checking only individual repair steps was insufficient.

Independent baseline comparison reproduced two- and three-level trees: baseline displacements were `[0.05, 0.15]` and `[0.05, 0.15, 0.15]` metres; the patch returned the untouched entire candidate in both cases. Inputs remained unchanged and rollback returned a distinct copy. A valid two-object tree moving exactly 0.10 m was identical between baseline and patch and retained successful geometry. Existing fixed-support protection remains at `geometry.py:67–71`; scoring still calculates raw and repaired metrics separately at `geometry.py:110–120`.

Focused tests: `tests/test_geometry.py:116–164,238–245` (paths relative to `fastfill/comparison/`). No resizing, parent reassignment, raw-score change, or request removal is introduced by this delta.

### 2. Movable-input fields: matches the real producer

`fastfill/comparison/train_pair_helpers.py:144–150` accepts exactly `id/category` for J and `id/category/size` for P. The actual producer emits precisely those fields at `fastfill/comparison/contract.py:231–232`, normalizing missing/blank categories to `object`. Fixed context remains outside this movable-only check. Pilot/main target policies, size validation, and support-graph equality still run at `train_pair_helpers.py:154–164`.

Independent checks passed producer inputs with categories `None`, empty string, whitespace, and `chair` after producer normalization. A shared injected `pos` field was accepted by the baseline and rejected by the patch with `invalid joint movable input fields`. Loading the real local `data/pilot-v3` through both helper versions produced identical pairs and receipts: **965 train / 58 dev**; manifest SHA256 `144b73e480ee66aef5f51d71058fe833260725544cd9f2b78c2b588baa04c8d6`.

Main-data compatibility was also checked against `main_data.py` production serialization/qualification and its integration tests. As secondary evidence only, the existing local `docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-inputs-7367962-result.json` reports no issues for 98,923 train / 8,167 dev / 8,647 test rows per arm; its audit source at `full-audit-data-inputs.py:94–98` uses the same exact field/category predicate. I did not rerun that remote corpus audit.

Focused tests: `tests/test_train_pair.py:202–286`, including leakage rejection, category validation, main-policy admission, and support-graph agreement.

### 3. Final tables: compatible with frozen judge and full-job denominators

`fastfill/comparison/final_tables.py:55` invokes the unchanged producer validator (`judge_batch.py:335–386`). The producer itself already invokes it from `summarize` at `judge_batch.py:402`. Its counts, both-arm structure, required score counts, and optional overall-score handling therefore agree with this consumer.

The new coverage gate at `final_tables.py:56–60` uses positive **request** counts, not successful geometry/render/API counts. Geometry constructs all six strata at `evaluate_pair.py:129–140`. Selection occurs before eligibility and preserves failures (`eval_data.py:select_gpt`; `visual_cases.py:205–219`); `visual_pipeline.py:303–315` creates mapping records for both arms even when preparation fails. The judge summarizes every selected record (`judge_batch.py:483–495`), preserving such strata with zero successes. Therefore the coverage check does not incorrectly discard or block a populated stratum merely because all its evaluations fail.

Read the actual frozen `data/visual-reference-v1/gpt-selection.json`: six populated strata, 29 selected rooms each, 174 rooms / **348 arm records**, no replacement. Its population sizes are 419, 87, 128, 115, 129, and 131 (as keyed in that file). The full-job driver passes the mapping length as `--expected-total` (`overnight.py:534–537`) and feeds `results/summary.json` to final tables (`overnight.py:651–658`). The validator's 348 cap matches this frozen protocol.

Independent offline probes generated summaries using the **frozen baseline judge producer**, then published them through baseline and patched final tables. All accepted cases produced identical paper-style rows:

| Simulated outcome | Selected | Runnable | Launched | Success | Failed | Patched result |
|---|---:|---:|---:|---:|---:|---|
| All successful, no overall scores reported | 348 | 348 | 348 | 348 | 0 | Published |
| All preparation failures | 348 | 0 | 0 | 0 | 348 | Published, scores null |
| Mixed success/failure, overall scores only on some successes | 348 | 348 | 348 | 232 | 116 | Published |
| One truly zero-population stratum | 290 | 290 | 290 | 290 | 0 | Published |

These are fabricated offline test records, not model-quality measurements or API activity. A separately generated internally consistent judge summary missing one **populated** stratum was published by the baseline but rejected by the patch with `judge stratum coverage mismatch`, before an output directory was created.

Focused tests: `tests/test_final_tables.py:79–122`, plus offline judge, visual-pipeline, and driver integration tests. Null scores, differing geometry/FID/GPT denominators, and zero-success accounting remain intact.

## Tests run independently

Environment: `/Users/slian/miniforge3/bin/python`, Python 3.12.9. Commands ran from the repository root with bytecode and pytest cache writes disabled.

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider   fastfill/comparison/tests/test_geometry.py   fastfill/comparison/tests/test_final_tables.py   fastfill/comparison/tests/test_train_pair.py
```

Result: **67 passed, 101 subtests passed in 2.13 s**.

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider   fastfill/comparison/tests/test_contract.py   fastfill/comparison/tests/test_data.py   fastfill/comparison/tests/test_main_data.py   fastfill/comparison/tests/test_eval_data.py   fastfill/comparison/tests/test_evaluate_pair.py   fastfill/comparison/tests/test_judge_batch.py   fastfill/comparison/tests/test_visual_cases.py   fastfill/comparison/tests/test_visual_pipeline.py   fastfill/comparison/tests/test_overnight.py
```

Result: **349 passed in 9.22 s**. Combined independent selection: **416 tests passed**, plus the 101 subtests above. Eleven additional direct baseline/producer/data probes passed; their observed results are documented above and reproduced verbatim below. ML, Blender, subprocess judging, and remote-driver boundaries in integration tests were mocked. No API requests, SSH, remote jobs, real training, or deployment actions were performed.

The user's separate frozen validation result (681 tests / 90.874% coverage) is not counted as my execution. I did not collect statement coverage in this independent review.

## Reviewed source identity and preservation

All seven requested files were rehashed before report creation; none changed during this review. No production or test source was edited. The only repository file created by this review is this report; temporary probe output was written outside the repository. Existing edits and artifacts were preserved.

| File | Reviewed SHA256 |
|---|---|
| `fastfill/comparison/geometry.py` | `0d3f7bee8b8f5b7c81f98af5dd4ae8cd3f1ef9eb867f631fc0b0df1451dae56a` |
| `fastfill/comparison/final_tables.py` | `b9758ce30311973e812035bd6b2c1a8b525ae2cc3682540dbcbde8ccf28ab0f9` |
| `fastfill/comparison/train_pair_helpers.py` | `7968de59729f9227725679b9c8b4252b9ef3970a9d8c7f0d47e79a5cedebea83` |
| `fastfill/comparison/judge_batch.py` | `0d93b727e99118261be14868779a557854afc08a5adf084b2ac7d5545546d7c8` |
| `fastfill/comparison/tests/test_geometry.py` | `f976be18879d54b2cd8faec6d9e8dcdbbcebb5c4d2c6b012e6ad6b919722c49f` |
| `fastfill/comparison/tests/test_final_tables.py` | `ac0bb5752b21cfdfe63638fb931ea5ef325302fe30095ebe72efa73f89c95901` |
| `fastfill/comparison/tests/test_train_pair.py` | `29bf539d83577e7e943b91844e513d088026142d73d82dac912231100131e1e5` |

## Direct probe output

```json
{
  "cumulative_depth_2": {
    "baseline_displacement_m": [
      0.05,
      0.15
    ],
    "patched_displacement_m": [
      0.0,
      0.0
    ]
  },
  "cumulative_depth_3": {
    "baseline_displacement_m": [
      0.05,
      0.15,
      0.15
    ],
    "patched_displacement_m": [
      0.0,
      0.0,
      0.0
    ]
  },
  "exact_10cm": "baseline == patched; both objects translate 0.10m; geometry succeeds",
  "producer_category_fallback": "4 producer inputs accepted by baseline and patched helper; exact allowlists match",
  "shared_pose_leak": {
    "baseline": "accepted",
    "patched": "invalid joint movable input fields"
  },
  "actual_pilot_v3": {
    "counts": {
      "train": 965,
      "dev": 58
    },
    "same_pairs_and_receipt": true,
    "manifest_sha256": "144b73e480ee66aef5f51d71058fe833260725544cd9f2b78c2b588baa04c8d6"
  },
  "full_job_success": {
    "status": "MEASURED_TABLES_PUBLISHED",
    "judge_counts": {
      "selected": 348,
      "runnable": 348,
      "launched": 348,
      "success": 348,
      "failed": 0
    },
    "same_table_rows_as_baseline": true,
    "overall_score_counts": {
      "joint": 0,
      "placement": 0
    }
  },
  "full_job_failed": {
    "status": "MEASURED_TABLES_PUBLISHED",
    "judge_counts": {
      "selected": 348,
      "runnable": 0,
      "launched": 0,
      "success": 0,
      "failed": 348
    },
    "same_table_rows_as_baseline": true,
    "overall_score_counts": {
      "joint": 0,
      "placement": 0
    }
  },
  "full_job_mixed": {
    "status": "MEASURED_TABLES_PUBLISHED",
    "judge_counts": {
      "selected": 348,
      "runnable": 348,
      "launched": 348,
      "success": 232,
      "failed": 116
    },
    "same_table_rows_as_baseline": true,
    "overall_score_counts": {
      "joint": 0,
      "placement": 116
    }
  },
  "full_job_success_zero_population": {
    "status": "MEASURED_TABLES_PUBLISHED",
    "judge_counts": {
      "selected": 290,
      "runnable": 290,
      "launched": 290,
      "success": 290,
      "failed": 0
    },
    "same_table_rows_as_baseline": true,
    "overall_score_counts": {
      "joint": 0,
      "placement": 0
    }
  },
  "missing_populated_stratum": {
    "baseline": "published",
    "patched": "judge stratum coverage mismatch"
  }
}
```

Review completed (UTC): 2026-10-13T07:32:35.894353+00:00
