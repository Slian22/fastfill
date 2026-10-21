# V1.1 fresh data / prompt / preference audit — 2026-10-11

This is a fresh source read and independent data scan, not a restatement of the prior successful test suite. All five assigned V1.1 modules were read in full, plus the entire `scene.py`, `anchors.py`, `validate.py`, `build.py`, and `split.py` dependency closure. Code discovery MCP was unavailable (parent observed Transport closed for graph search/index), so known-path reads were used. No remote data, job, or deployed source was changed by this agent. Exact file hashes / lengths / full-read coverage are in `data-coverage.json`.

## P1 scientific scope limitation: size-unknown prompts inherit a target-derived ordering

Confirmed mechanism: `fastfill.scene.canonical` ranks movable objects by actual target footprint area, category, dimensions, and then target `(z,x,y)` pose for ties; it renumbers category IDs in that order. `fastfill.build._write_dataset` applies that function to both train and evaluation rooms. `convert_v1_messages_to_v1_1` removes size fields while preserving the resulting IDs and order. `messages_v1_1` also preserves order for evaluation. Thus deleting explicit size fields does not make these prompts independent of the target. The currently unused `canonical_v1_1` helper is not a complete remedy: it itself consults hidden target `parent` before category priors.

A metamorphic test held the room boundary and requested categories fixed and changed only a chair's target size. The resulting V1.1 user object list changed from `[chair_1,table_1]` to `[table_1,chair_1]`. This does **not** prove the model exploited that cue, and is **not** a claim of train/test split overlap. It establishes that the current task is conditional on target-derived object order and that results should not be generalized to arbitrary inventories whose target sizes/poses are unknown.

Fresh full scan (`data-scan.json`):

| Cohort | Rows | Rows in GT-area-descending order | Rows differing from existing prior-order helper |
|---|---:|---:|---:|
| train | 144150 | 144150 | 101728 |
| dev | 8167 | 8167 | 5828 |
| dev constrained | 8035 | 8035 | 5815 |
| test | 8647 | 8647 | 6192 |
| test constrained | 8533 | 8533 | 6177 |

This affects the existing SFT, sanitized DPO, and full evaluation equally. It does not invalidate the narrow paired comparison under that shared historical condition, but it prevents calling that comparison a clean test of unrestricted size-unknown layout generation. No prompt order or frozen data was hot-patched here. A subsequent data/model version should define IDs and order from declared request information only, remap all relation IDs consistently, rebuild SFT and preferences, and preregister its own evaluation. Reordering evaluation alone would change the experiment and leave training mismatch unresolved.

## Confirmed P2 generator defects, now fixed locally

Pre-fix reproduction called the actual Stage 1 generator with a mocked CPU inference backend, rather than copying its logic (`repro_builders.py`, `builder-repros.json`).

1. **Thin-reference false negative and identical preference.** The original Stage 1 interval `[max(.05,.45*g),min(6,2.2*g)]` excludes a valid 1 cm rug even when prediction equals GT. That candidate became `size_distortion`; unverified GT fallback then produced identical chosen/rejected answers. The unchanged sanitizer correctly rejected the pair. Generation now uses the sanitizer's registered bounds and acceptance policy, verifies GT fallback, and independently verifies each complete pair before writing it.
2. **Stage 2 reference constraint failure.** A geometry-valid GT violating a prompt `near` relation was accepted as the positive and a `wall_oob` negative was generated. The unchanged sanitizer rejected it. Stage 2 now validates all prompt relations and the complete reference before generating a pair, and independently audits the result.
3. **Unbound Stage 1 partial adoption.** A `.partial` row `{uid:not_in_train,chosen:X,rejected:Y}` reached the requested count and was published without source, prompt, model, or semantic checks. Stage 1 now locks the destination and uses one atomically written cache document containing data/model/config/code identity and all complete pairs. Every cached row is independently revalidated against the exact training reference. Unknown/legacy cache formats, identity drift, duplicate/foreign rows, malformed answers, and output tampering fail closed. Interrupted inference leaves a bound recoverable cache. A completed cache remains alongside the final JSONL as its generation provenance.

These paths are present in the old deployed snapshot but are not used by the currently scheduled clean run, which consumes separately sanitized frozen pairs. No current clean-pair label failure was discovered. Because `build_dpo.py` contributes to sanitizer code identity, any future deployment using the new code must regenerate the sanitizer receipt in a new output directory; do not edit the old receipt hashes by hand.

## Confirmed P2 converter publication defect, now fixed locally

Pre-fix CLI reproduction (`repro_converter.py`, `converter-repro.json`) supplied three new SFT files to an existing output directory containing an unrelated old `test_rooms.jsonl`. The command exited zero and included that untouched old room file in the new manifest, because room input files were optional and outputs were reused. Conversion also wrote directly into the final directory before later source errors were discovered.

The converter now requires all seven nonempty expected inputs, refuses a nonempty output, validates copied room JSON, builds in a temporary sibling directory, records original input SHA256s, rechecks those hashes before publication, and atomically publishes the complete directory. Existing empty destinations remain empty if conversion fails before publication. These changes do not modify already frozen data files; the four current evaluation files match the separately audited source hashes.

## Fresh actual-data and pair checks

- Read all 144150 actual V1.1 train message triples (SHA256 `6ab4c001e566718ce3e89a51de08912edbc9ba2aa740fe4284741e04bd6650a0`). All assistant layouts parse with exact requested IDs, finite numeric values and acyclic valid support references. All supplied training constraints hold on their reference answers. This scan did not rerun full collision geometry on every SFT room; five recorded quality flags leave 103319 eligible rows before token-length filtering.
- Across 1779614 train requested objects, no explicit input `size`, `pos`, `yaw`, `on`, or `parent` field occurred. That explicit-field result must be read alongside the target-order limitation above.
- Independently reran the unchanged semantic auditor over every currently sanitized preference: Stage 1 **7244/7244 retained**, of which 4402 positives equal the reference; Stage 2 **12000/12000 retained**, all positives equal the reference. No bad pair appeared (`pairs-recheck.json`). This check uses training references only, not evaluation scores.
- Parsed every room of all four full evaluation files. Anchor counts are exclusively `floor` and `object`: ordinary dev 72350/30213, constrained dev 72175/30209, ordinary test 74246/34007, constrained test 74079/34007. There are zero `wall`, `ceiling`, `fixed`, or missing anchors in these cohorts. General anchor inference depends on GT pose/support, but the wall-only repair/oracle concern reported by other reviewers does not execute on these four current cohorts (`anchor-stats.json`).
- Full split/alias/content derivation is in the separately hash-bound prior data audit. This agent reread the entire split algorithm and fresh-scanned all selected train/evaluation message rows; it did not independently repeat the expensive full raw-IR content/layout fingerprint reconstruction. Prior verified absence of UID/group/alias intersections is not proof of asset-library disjointness or warm-start model provenance.

## Independent cross-review follow-up: Stage 2 publication safety

The independent reviewer reproduced one remaining P2: Stage 2's fixed `.partial` writer and `os.replace` could overwrite an existing output; `max_pairs=0` replaced it with an empty file. Seven added regression tests failed before this follow-up fix. Stage 2 now rejects non-positive limits and all existing outputs/symlinks, shares the exclusive generation lock, rechecks training/config/code identity before publishing, and leaves old `.partial` files untouched. The publisher uses a unique sibling temporary file, flush+fsync, and an atomic create-only hard link, so even a target appearing immediately at publication cannot be overwritten. A dedicated race test verifies preservation and temporary-file cleanup. The same publication primitive improves Stage 1's final output boundary.

The reviewer re-read this fix and reported no remaining confirmed P0/P1/P2 blocker (`data-cross-review.md`). A frozen-old versus new Stage 2 comparison on a valid constrained support fixture across seeds 0,1,2,42,99 produced byte-identical JSONL outputs in every case (`stage2-compat.json`). This is a bounded compatibility regression check, not a claim that the full original 12000 pairs were regenerated.

## Validation

The first new regression suite was run before edits: **15 failed**, reproducing the requested defects and missing contracts. After all fixes, 35 dedicated tests pass, covering mocked model generation, real semantic audit, cache tampering/resume/interruption and data/model/config/code drift, actual multiprocessing dataset conversion, atomic failures and all seven output files. The original sanitizer regression tests also passed unchanged. Whole-file executable-line coverage using a scoped standard-library trace is **90.56% for build_dpo.py** and **93.45% for build_v1_1.py**, with 35/35 tests passing during measurement (`builder-coverage.json`). This is executable-line coverage, not branch coverage. Final parent integration tests and independent review remain the parent's responsibility.
