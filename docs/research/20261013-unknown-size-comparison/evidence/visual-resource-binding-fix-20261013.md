# VIS-01 bounded resource-binding fix — 2026-10-13

Implemented locally for independent parent review. The authorized legacy-fixture follow-up is complete: all 25 original fixture failures are resolved, and the full comparison suite passes all 722 tests. No deployment, restart, run003 change, remote operation, paid API call, paid job, or actual dataset/asset mutation was performed. Existing frozen references need no schema migration or regeneration. This report does not approve a frozen-pipeline upgrade or a production render.

## Changed files and source identity

Only these four files were written across the initial fix and authorized fixture follow-up:

- `fastfill/comparison/visual_pipeline.py` — require the bound resource manifest and enforce room resource checks on both sides of rendering.
- NEW `fastfill/comparison/tests/test_visual_resource_binding.py` — synthetic freeze-to-bundle tests, with real resource inspection and mocked Blender RGB output.
- `fastfill/comparison/tests/test_visual_pipeline.py` — the shared `fixture` function now creates real temporary OBJ/MTL/texture resources and a manifest-bound resource book. Every existing test-function body is unchanged.
- NEW this report.

No helper was necessary. All other existing test files, renderer, freeze implementation, automation-code copies, and run003 were not edited. The fixture follow-up changed only `test_visual_pipeline.py` and this report; production code and the new binding test file retain their original handoff hashes. The comparison directory already existed as untracked content; Git HEAD alone does not identify its source bytes. Checkout HEAD at inspection: `8b94c437cbdacc5329b6abd04e26223fd0824e40`.

| Source | SHA-256 |
| --- | --- |
| `fastfill/comparison/visual_pipeline.py`, before fix | `4cd83a00fa503aecda90041848eb631c1fcd025b002b3ab1aac7edac023c4f61` |
| `fastfill/comparison/visual_pipeline.py`, after fix | `099ed3467bd3312665fd446d21304f9df176a04998d1c40b697b0e1fd974b29d` |
| NEW `fastfill/comparison/tests/test_visual_resource_binding.py` | `af411bc451f5471acdab3c8e0fbd934e82b6decc5a21d6e4004b4db60f466370` |
| `fastfill/comparison/tests/test_visual_pipeline.py`, baseline before fixture follow-up | `f7ef18ffa0974ef81d3090f01e575ad8e202c9f85c8d54be43c6cb7a84a1a577` |
| `fastfill/comparison/tests/test_visual_pipeline.py`, after fixture follow-up | `365dd23012ebf5d73b536d9bb8135d9c7fa0548bac94036d7bcfd810592b8ffc` |
| Unchanged `fastfill/comparison/visual_cases.py` | `dd9b51883357aab474d5bc46c7986940c61a9011f8aa6fdad2667f0b27911a21` |
| Unchanged `fastfill/comparison/render_assets.py` | `eaf4e127aa08c985d9a53ef318cb61b892d60a7a49ff49b9fc9557aee3340583` |

The before-fix pipeline and renderer hashes exactly match [the confirmed drift probe](full-audit-resource-drift-probe.json). That evidence file's SHA-256 is `8a25a7741e1f2630edea4863772271f37de195ce18945c5ff9a6e6b963b76a33`.

[The existing point-in-time byte audit](full-audit-visual-resource-bytes.json) reports 1,945 resource entries, 6,558 unique files, and zero mismatches at `2026-10-13T07:42:12.492224+00:00`, for reference manifest `dda638b9bf6ea292fa9dbcec5b731cb5a7fc885ef4331a3438a741e6b306573a`. Its evidence-file SHA-256 is `cb8e191b54ea61bd77f12abd00546d24ba20ee9e58adabad0d04c08e67cf3243`. This task read that evidence; it did not repeat or extend that real-asset certification.

## Enforcement and compatibility

1. `_inputs` now requires `resources.json` in the reference `MANIFEST.json`. Existing snapshot validation checks the file inventory and hashes. `evaluation._read` additionally hashes the exact bytes consumed when loading the resource book. An absent, unbound, hash-mismatched, or non-object resource book aborts preparation before output creation or renderer launch.
2. `_resource_failures` uses `visual_cases._planner()`, the same inspection and serialization mechanism used by the freeze, and the renderer's actual plan. It identifies consumed movable and fixed-furniture resources by the freeze's digest of `[resolved_mesh_path, asset_id]`. It compares complete records: identity and path, OBJ SHA-256, MTL paths and SHA-256 values, every texture path and SHA-256, and MTL overrides. No current record replaces a frozen record. Procedural structures consume no external resource and do not require an unused mapping entry to be present in the resource book.
3. Preparation validates attached rooms before `_render`. Failed rooms are omitted from submitted chunks. After rendering completes, preparation validates the submitted rooms again using a new inspection cache. Shared assets are inspected once per pass, and mismatches fail every dependent room. Reference-ineligible or prediction-attachment failures retain their existing failure records and never render.
4. `_combine` records affected rooms as `resource_failed`, with `failure.stage = resource_binding`, `phase = before_render` or `after_render`, and an error reason. A resource failure clears all authoritative GT/reference, joint, and placement images, even if the chunk receipt reports success. Unaffected rooms continue.
5. `cases.jsonl`, the final ledger, FID inventory, original frozen/selected counts, and failure denominators retain all original UIDs. Selected failed rooms keep both opaque API mapping entries and both API failure entries, but receive no runnable API cases. Selection slots are never replaced. API request schema, identity mapping, geometry/reference/evaluation provenance, and bundle hashing remain unchanged; the reference-manifest hash transitively binds `resources.json`.

## Initial bounded-fix TDD and local verification

Environment: local Python 3.13.1, pytest 8.3.5. No packages were installed. Synthetic assets, snapshots, rendered PNGs, and coverage data were confined to temporary locations. Blender execution was mocked throughout the new integration tests; resource parsing and hashes were real.

- RED, before editing production: the initial 27-test binding suite produced **24 failed, 3 passed** in 6.02 s. The changed/missing-resource probes were accepted as three successful rooms and six API cases, instead of two successful rooms, one failure, and four API cases. This establishes behavior failure, not an import or missing-helper failure.
- GREEN after the minimal production change: the same suite produced **27 passed** in 4.24 s.
- Expanded binding/regression suite: **41 new tests pass**. Coverage includes changed and missing OBJ/MTL/texture bytes both before and after mock rendering; shared resources; fixed furniture; missing/incomplete frozen entries; required manifest binding; material override changes and unchanged overrides; relative and dictionary mappings; procedural structures; all-failed inventories; no replacement by an unselected successful room; attachment/render/timeout/image failures; and preserved API provenance.
- Combined new suite plus unchanged freeze, renderer, and FID suites: **191 passed** in 15.79 s.
- Eleven compatible existing pipeline boundary/CLI tests: **11 passed** in 0.39 s. Combined production-file statement-and-branch coverage is **84.87%**, exceeding the enforced 80% threshold. There were 202 passing tests across these two successful commands.
- Python compilation and targeted trailing-whitespace checks passed for both changed Python files. `git diff --check` also passed, but does not inspect the already-untracked comparison files; the targeted checks cover those files.

Successful commands, from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 COVERAGE_FILE=/tmp/optiscene-vis01-coverage python3 -m pytest -p no:cacheprovider fastfill/comparison/tests/test_visual_resource_binding.py fastfill/comparison/tests/test_visual_cases.py fastfill/comparison/tests/test_render_assets.py fastfill/comparison/tests/test_fid.py --cov=fastfill.comparison.visual_pipeline --cov-branch --cov-report=term-missing -q --tb=short

PYTHONDONTWRITEBYTECODE=1 COVERAGE_FILE=/tmp/optiscene-vis01-coverage python3 -m pytest -p no:cacheprovider fastfill/comparison/tests/test_visual_pipeline.py::test_resource_bounds fastfill/comparison/tests/test_visual_pipeline.py::test_cli_timeout_override_and_default fastfill/comparison/tests/test_visual_pipeline.py::test_cli_reports_fatal_validation fastfill/comparison/tests/test_visual_pipeline.py::test_output_cannot_modify_input_snapshot 'fastfill/comparison/tests/test_visual_pipeline.py::test_input_hash_mismatch_before_render[reference]' --cov=fastfill.comparison.visual_pipeline --cov-branch --cov-append --cov-report=term --cov-fail-under=80 -q --tb=short
```

At the initial bounded-scope handoff, the unmodified legacy pipeline suite was not green. A separate run of `test_visual_cases.py`, `test_render_assets.py`, `test_fid.py`, and `test_visual_pipeline.py` produced **161 passed, 25 failed** in 4.18 s. All 25 failures stopped at the new required-file check because `test_visual_pipeline.fixture` created snapshots without `resources.json`; that fixture also used empty asset mappings for furniture. These results remain recorded as baseline evidence. The user subsequently extended ownership to correct the legacy fixtures; the completed follow-up below supersedes the earlier unresolved-test status. No compatibility bypass was added.

## Authorized fixture follow-up and final verification

Only the existing shared `fixture` function needed correction. It now creates a temporary, parseable OBJ with a material and a real PNG texture outside the reference snapshot, supplies a furniture asset mapping, and runs the same `visual_cases._planner()` used by freezing. That inspector computes the actual OBJ, MTL, and texture hashes and the full resource records. The fixture writes those records to `resources.json` before sealing `MANIFEST.json`, and takes the cameras from the validated plan. Its asset directory supports nested fixture roots, as required by the existing overnight driver integration test.

No assertion, test case, renderer stub, or production guard was removed or weakened. Before editing, the exact source of all 20 existing `test_*` function bodies was individually hashed; all 20 hashes match after the fixture update. The baseline and final whole-file hashes are listed above for Faraday's read-only baseline audit and the parent's review. No reviewer audit artifacts were modified. No other legacy test file required edits.

- All **36 legacy pipeline test cases pass**, resolving every one of the original 25 failures.
- The final complete `fastfill/comparison/tests` run passed **722 tests in 31.09 s**, with zero failures, skips, or deselections. This includes all 41 binding tests, the 1,009-UID/174-selection accounting case, the concurrent chunk/no-retry case, and the overnight driver integration using this shared fixture.
- Fresh coverage for the unchanged production `visual_pipeline.py`: **296/309 statements (95.79%)**, **144/180 branches (80.00%)**, **89.98% combined statement-and-branch coverage**. The run enforced `--cov-fail-under=80` and passed. Coverage was measured for this production file, not claimed for every comparison module.
- The first full-suite attempt exposed the nested fixture-directory requirement in `test_real_visual_fid_and_judge_accept_driver_upload` (721 passed, one failed). Adding `parents=True` within the same fixture resolved it; the complete rerun above is the final result.
- Targeted Python syntax and whitespace checks passed. Production, freeze, renderer, and new binding-test hashes still match the initial handoff.

Final full-suite command, from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 COVERAGE_FILE=/tmp/optiscene-vis01-full-coverage python3 -m pytest -p no:cacheprovider fastfill/comparison/tests --cov=fastfill.comparison.visual_pipeline --cov-branch --cov-report=term-missing --cov-report=json:/tmp/optiscene-vis01-full-coverage.json --cov-fail-under=80 -q --tb=short
```

Coverage artifacts are local temporary files. Tests use temporary synthetic inputs and mocked rendering/judging backends; no production rendering or paid judging was launched.

## Limits and review handoff

- These are pre/post render-boundary checks over external assets, with invocation-local caches shared within each pass. They detect persistent drift at those boundaries. They do **not** provide immutable staging, lock files, attest every Blender read, or rule out a transient change restored between checks (or mutation during a cached pass). Stronger read-time guarantees require an additional design and write scope.
- Full resource inspection adds CPU work and file I/O for two passes; real-cohort runtime overhead was not benchmarked. No mtime-only or pre-to-post hash cache is used.
- Raw chunk PNGs and receipts may remain inspectable after a post-render resource failure. They are excluded from the authoritative FID images and runnable API bundle. Consumers must use the final manifests/ledger.
- This validates resource identity, bytes, and dependency sets. It does not certify Blender fidelity, geometry correctness, material appearance, or scientific metric validity. No real Blender/production E2E render, production FID evaluation, or paid judge run was performed.
- The parent must independently review this exact source and decide whether to freeze a new pipeline before rendering. The legacy fixture follow-up is complete. Existing run003 and all existing frozen automation-code copies were left untouched.
