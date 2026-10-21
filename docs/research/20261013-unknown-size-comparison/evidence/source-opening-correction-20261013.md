# IL3D_3dfront source opening correction — 2026-10-13

Implemented and locally verified. Ownership: `fastfill/adapters/il3d_3dfront.py`, new `fastfill/tests/test_source_opening_correction.py`, this report, and the subsequently requested exact baseline copy. No `unified.py` changes, source-data writes, frozen-IR writes, dataset rebuilds, jobs, or API calls. Other agents' changes were left intact. Parent owns full versioned regeneration.

## Baseline retained for isolated regeneration

Exact prepatch adapter: [il3d_3dfront.before.py](source-opening-correction-20261013/il3d_3dfront.before.py).

- Absolute path: `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/evidence/source-opening-correction-20261013/il3d_3dfront.before.py`.
- 26,474 bytes; SHA-256 `b2eb88d7c2d6b97c27d8551d8ca80dd1d4aa64f6b8fa75b1bd93a1b7cacf779a`.
- Provenance: the initial working-tree status showed this adapter clean, with no staged or unstaged adapter changes. The bytes were subsequently recovered from the observed baseline commit `8b94c437cbdacc5329b6abd04e26223fd0824e40` and checked against the independently recorded **prepatch working-file** hash and length in `source-correction-input-plan-20261013.json`. This is not an assumption that an arbitrary current HEAD equals the original working file. The saved file is byte-identical to the prepatch adapter.
- Parent subsequently independently confirmed prepatch audit SHA = HEAD = frozen v3.2 adapter bytes, and started the full three-source baseline regeneration in an external output. That full regeneration is not duplicated here.
- Use this copy for BEFORE, and the reviewed working adapter for AFTER, with identical common dependencies and raw input bytes. For example, load the saved adapter with `runpy.run_path(path, run_name="il3d_before")`; do not execute its `__main__` block or replace the live adapter while other agents work.

Full regeneration must compare **fresh BEFORE vs fresh AFTER** to isolate this correction. Separately compare **older frozen IR vs fresh BEFORE** to identify prior adapter/input/dependency drift. Do not attribute all older-frozen-to-AFTER changes to this patch. Pin/hash the IL3D scene export, assets, 3D-FRONT JSONL, referenced raw house JSONs, and shared adapter dependencies for both runs. Full regeneration and full-source hashes remain the parent's work.

## Source evidence and exact semantics

Mounted and read:

- `/Volumes/harddisk/3D_Room_Collections/BillLin66__3D_Room_Collections/3DFront_exported/3dfront.jsonl` (370,189,505 bytes).
- Its `kits/convert_3dfront.py`, especially `indoor_window` (lines 437–467), `opening_object` (470–496), and opening dispatch (649–668).
- Its `README.md`, especially the opening schema and representation caveats (92–103).
- Raw house `data/3D-FRONT/71b4daf5-e0e4-47f1-a76f-fabb489d2df2.json`.
- Historical `.release/audits/2026-09-28/source-check/designer/{join_opening.py,opening-target-join.json,join_opening.log,root-opening-confirmed.json,REPORT.md}` and `.release/audits/2026-09-28/FINAL_REPORT.md`.

The export changes source Y-up `[x,y,z]` to house-global Z-up `[x,-z,y]`. Both opening constructors set `width = max(max(X)-min(X), max(Y)-min(Y))`. That is a world-axis horizontal bounding-box span. For a vertical window on a slanted wall, it is shorter than the span along the wall. The previous adapter rotated the box along a profile edge but used this projected number as its local width.

An ordinary 3D window profile comes from a source Window boundary loop matched to a WallInner plane, using approximately `1e-5 m` tolerance. Doors, bay windows, and ordinary windows whose indoor match failed instead carry **2D horizontal convex hulls**, including frame depth. These representations do not justify applying the same width correction indiscriminately.

Raw geometry replay: for house `71b4daf5-e0e4-47f1-a76f-fabb489d2df2`, room `SecondBedroom-198098`, invoking the exporter's pure `group_openings`/`indoor_window` functions in memory reproduced all three window outlines. Window index 2 maps to mesh `5230771593790725573/0`, constructid `523022`, and exactly matches the archived and mounted exported opening. Its XY endpoints are `(-0.01947, 4.67276)` and `(0.24187, 4.40848)`; width changes from `0.26428` to `hypot(0.26134, -0.26428) = 0.37167528031872143 m`. Height remains `1.5 m`, sill remains `0.9 m`. Export line 25374, byte offset 167096966, row SHA-256 `d2bf12d19906eff7b7ac79de497562558aaaf3dd777bd9e7d98af2fa0a86941f`.

The two test-room windows in house `eb1f5725-ab22-470d-b345-cca532813d98`, `SecondBedroom-25551`, indices 0/1, change from `0.6625/0.61717` to `0.8865154905019991/0.8720609732123094 m`. They are in export line 51765, byte offset 341699163, row SHA-256 `fbfe8169ba3cc379ea705daf13948d77d7d475964e5b372f924e778d261122bc`.

The 24 relevant mounted JSONL rows were read using recorded byte offsets, each exact row hash was verified, and all 39 selected source opening dicts matched the recorded dicts. This is bounded source verification, not a full corpus rescan.

## Implementation contract

`_opening_frame` validates finite, consistently dimensional profiles before geometric operations. Missing/malformed profiles, collapsed XY geometry, invalid or zero-area 2D polygons, zero-height 3D windows, and zero-area/invalid vertical projected outlines are skipped. Existing positive-width, valid-position, and minimum-height requirements remain in force.

For a 3D window, find the XY point farthest from the first point, then the point farthest from that endpoint. This full-extent pair gives the candidate wall direction. A short subdivided edge is not used: rounding its endpoints can magnify direction error. Neither the longest 3D edge nor a single subdivided XY edge is taken as the width. Every point must lie within `1e-5 m` of this vertical plane. Project all profile points onto the wall axis and take `max(projected coordinate) - min(projected coordinate)`. This handles subdivided outlines and avoids confusing a tall vertical edge with width. Validate the polygon in `(along-wall coordinate, Z)` before accepting it. Exported widths within `1e-6 m` of the measured span are preserved exactly to avoid changing already-correct axis-aligned values through subtraction noise.

For 2D footprints, keep the existing exported width, choose the side nearest that width, and retain `polygon area / width` depth. Nonplanar 3D profiles retain the previous width/direction behavior; no wall span is claimed for them. Window/3D depth stays the existing `0.10 m` fallback. Heights, sill preference (otherwise center Z minus half height), XY frame shift, source room joins, floor reference, neighbor-door selection, opening IDs, provenance (`listed_on`), and front-axis conversion are unchanged. No input dictionaries or coordinate arrays are mutated.

The existing size/overlap ordering and half-covered-footprint threshold use the corrected footprint. Windows additionally require **Z intervals overlapping by more than `1e-6 m`** before their footprints can count toward merging; touching or disjoint vertical intervals do not qualify. Otherwise widening could newly merge distinct windows on separate levels. Door merging is unchanged. This guard can also retain previously merged vertically separate windows, including 2D windows, independently of the width correction. Both wider-footprint effects and the new window Z guard must be separated in full-corpus retention/ordering measurements. The bounded source cases below show no retention change.

## Historical pool and current admitted exposure

The supplied **20 fixed windows / 10 train rooms** is the actual current exposure in the **98,754 admitted main-data J/P paired-SFT rows**. The **25 windows / 15 rooms** count describes the prior default-selected training pool. These are different, consistent membership scopes, verified in [full-audit-data-final-summary.json](full-audit-data-final-summary.json) under `known_source_exposure.IL3D_3dfront`. Current training-exposure claims should use 20/10. The broader historical archive contains:

| Set | Windows | Distinct rooms |
|---|---:|---:|
| Strict raw geometric candidates | 69 | 46 |
| Prior-pool train, default selected | 25 | 15 |
| Saved train, excluded by flags | 12 | 8 |
| Saved test, excluded by `oob_objects: 1` | 2 | 1 |
| All saved hits | 39 | 24 |

There are no dev hits. All 39 saved entries have saved fixed objects and `whole_fixed_list_matches: true`. These are historical fixed-conditioning defects, not evidence that windows were placement targets. The test room is `il3d:bc59ad12-86d4-11f0-a478-60cf84ae2082`, historical test line 888, source indices 0/1 corresponding to canonical `window_2/window_3`. The 39 saved historical cases remain the broader source-correction regression list; 20/10 is its current admitted training-exposure subset, and 25/15 describes the prior selected pool. Parent should retain these labels when comparing versioned membership.

The strict historical predicates found no qualifying rectangular oblique-door candidates. Earlier exploratory counts of 66 doors / 340 windows are screening counts, not confirmed errors, and do not support a longest-edge rewrite for every opening.

## Tests and bounded effects

TDD RED, before adapter edits: `python -B -m pytest -q -p no:cacheprovider fastfill/tests/test_source_opening_correction.py` → **18 failed, 26 passed**. Failures included all five real-coordinate regressions, subdivided profiles, malformed/degenerate profiles, final IR width, and the in-memory load integration.

Initial GREEN: **44 passed**. Independent code review then identified rounded-subdivision direction error and newly merged windows with disjoint heights. Added regressions produced a second RED of **3 failed, 44 passed**; fixing full-extent direction and adding the window-only vertical merge guard produced **47 passed**. A review follow-up caught decimal-height addition noise (`0.9 + 0.8 > 1.7`); its explicit rounded-height regression was RED (**1 failed, 47 passed**), then the source-precision Z guard produced final GREEN of **48 passed**. All material review findings were addressed. `python -B -m fastfill.adapters.il3d_3dfront` → **self-check OK**. `python -B -m pytest -q -p no:cacheprovider fastfill/tests` → **263 passed, 90 subtests passed**, in 9.32 seconds, on the shared working tree at final validation (other agents added tests concurrently; the initial whole-suite run was 250 passed / 90 subtests). `git diff --check` passed. A local stdlib `sys.settrace`/`dis` probe during the 47-test focused iteration covered all 35 executable outer-function lines in `_opening_frame` and all 36 in `openings`; this is scoped function line coverage, not branch coverage or whole-project coverage.

The successful full-suite command used `/Users/slian/miniforge3/bin/python` (Python 3.12.9), pytest 9.1.1, transformers 5.14.1, torch 2.13.0, peft 0.21.0 and accelerate 1.15.0. This is distinct from `/usr/local/bin/python3` (Python 3.13.1, transformers 4.49.0); no environment was changed. Test counts and code fingerprints above describe this correction revision; the parent separately owns subsequent malformed-metadata guards and their validation.

Tests are self-contained and do not require the mounted corpus. Coverage includes five exact source-shaped windows (including both test-room cases), axis-aligned X/Y windows, tall/subdivided/reversed/closed vertical outlines and real-coordinate rounded subdivisions, axis-aligned/slanted doors and 2D windows, footprints deeper than their width, nonplanar fallback, invalid inputs, no mutation, overlap merging and original indices, separate/touching-height windows including independently rounded decimal heights, adapter-to-IR size/yaw conversion, and a load integration with a source room join, neighbor door, upstairs-door rejection, and a nonzero floor reference.

Read-only old/new `openings` comparison on the same 24 mounted source rows:

| Measurement | BEFORE | AFTER |
|---|---:|---:|
| Emitted structures | 128 | 128 |
| Skipped/merged pieces | 7 | 7 |
| Emitted doors | 50 | 50 |
| Emitted windows | 78 | 78 |
| Changed window widths | — | 39 |
| Retained-ID symmetric difference | — | 0 |

All 50 door dicts are exactly unchanged. All 39 changed window dicts differ only in their width-bearing `furniture_size.length`; height, depth, rotation, position, ID, category, and provenance are identical. The other 39 emitted windows are identical. The selected source lists include 16 horizontal 2D window profiles. Width increases range from `0.021141002596555714` to `0.8001406126942188 m`. Independently replaying all 69 strict historical raw candidate profiles through the corrected adapter matches each archived along-wall measurement to `1e-9 m`.

These are opening-level in-memory measurements plus a synthetic load integration. A full fresh adapter load, rebuilding training/evaluation inputs, admission checks, or downstream quality effects have not been run here.

## Source case ledger

Widths below are metres, shown to nine decimals. `train/selected` means the historical join's `default_selected`, not current-version admission. Export row hashes and source-object equality were verified using the locator plan above. Every source index retains its original `::window_<index>` identity before normal canonical renaming.

| Historical set | House / source room | Source indices: exported → corrected |
|---|---|---|
| train/excluded | `71b4daf5-e0e4-47f1-a76f-fabb489d2df2` / `SecondBedroom-198098` | 2: 0.264280000 → 0.371675280 |
| train/selected | `71b4daf5-e0e4-47f1-a76f-fabb489d2df2` / `MasterBedroom-3901` | 1: 0.362300000 → 0.420903870; 3: 0.312060000 → 0.377637356 |
| train/selected | `2a6d5f03-79ed-4fde-83c4-1548ef7082d0` / `KidsRoom-1779` | 0: 1.323220000 → 1.349748692 |
| train/selected | `beb67301-7ec9-436b-acab-89c7580c59f2` / `LivingDiningRoom-8020` | 0: 5.367680000 → 5.617577204 |
| train/selected | `258b12fb-08e1-43df-a3dd-d88793d32621` / `LivingDiningRoom-35273` | 0: 5.172170000 → 5.972310613 |
| train/selected | `258b12fb-08e1-43df-a3dd-d88793d32621` / `Lounge-165793` | 1: 4.795560000 → 5.537436785 |
| train/excluded | `846ebcb5-074c-4119-9776-48a228426e0e` / `MasterBedroom-90821` | 4: 0.767860000 → 1.066420827 |
| train/excluded | `d6979ebd-f7b2-49c4-bd36-59a7990c5621` / `LivingDiningRoom-7683` | 0: 1.083480000 → 1.182630285; 2: 1.095670000 → 1.182634586 |
| train/selected | `c75f9c75-5902-4ba5-ba0a-62510d5dfaf8` / `Bedroom-10763` | 0: 0.600800000 → 0.837697068; 2: 0.647140000 → 0.837699054; 4: 0.597360000 → 0.837697301 |
| train/selected | `cbce76fc-24b3-460d-baea-540ca4366f3c` / `LivingDiningRoom-515` | 3: 1.035880000 → 1.399951820 |
| train/excluded | `0a482eb4-e8fa-4b44-90d0-3623e0a60c71` / `LivingDiningRoom-515` | 3: 1.035880000 → 1.399951820 |
| train/excluded | `8f442b7e-77e3-4a6c-b640-7222082dc352` / `LivingDiningRoom-7683` | 0: 1.083480000 → 1.182630285; 2: 1.095670000 → 1.182634586 |
| train/excluded | `7d1f5f23-b3dd-47b3-a659-9b74107dfbca` / `LivingDiningRoom-7683` | 0: 1.083480000 → 1.182630285; 2: 1.095670000 → 1.182634586 |
| train/selected | `549659d7-3221-4642-991f-2e25264efb21` / `SecondBedroom-10577` | 0: 0.780880000 → 0.999968105 |
| train/excluded | `447ef925-ee99-46ae-860a-22808a46b664` / `SecondBedroom-198098` | 0: 0.261550000 → 0.367835632 |
| train/selected | `58c874ab-3398-4a1d-80e9-de1af9b00752` / `MasterBedroom-34488` | 0: 0.480570000 → 0.639998011; 2: 0.593270000 → 0.640002264 |
| train/selected | `0e04ad6b-5563-481c-b125-ee5b5e40ff5e` / `Bedroom-8905` | 0: 0.606470000 → 0.815919931; 1: 0.667400000 → 0.868762240; 2: 0.589560000 → 0.776491727 |
| train/selected | `ffab746b-2e40-4bb7-8e69-51bb20f09ce1` / `Bedroom-10763` | 0: 0.600800000 → 0.837697068; 2: 0.647140000 → 0.837699054; 4: 0.597360000 → 0.837697301 |
| train/selected | `942b4c07-a1e9-42e2-bcdd-45a418e6f805` / `MasterBedroom-11342` | 1: 1.363780000 → 1.849289569 |
| train/excluded | `9b4dcb0a-4247-4069-b865-bf67f8b96682` / `LivingDiningRoom-7683` | 0: 1.083480000 → 1.182630285; 2: 1.095670000 → 1.182634586 |
| train/selected | `1d4185fd-6736-4441-9bf8-f653312d7a3b` / `MasterBedroom-12892` | 0: 1.738690000 → 1.759831003 |
| train/selected | `a111cbfa-1ae0-4705-9974-210f1f10731f` / `MasterBedroom-8826` | 0: 0.468350000 → 0.648884824; 2: 0.480880000 → 0.648886680 |
| train/selected | `ca12fa31-b995-4419-b689-08564e9237c7` / `MasterBedroom-8826` | 0: 0.468350000 → 0.648884824; 2: 0.480880000 → 0.648886680 |
| test/excluded | `eb1f5725-ab22-470d-b345-cca532813d98` / `SecondBedroom-25551` | 0: 0.662500000 → 0.886515491; 1: 0.617170000 → 0.872060973 |

## Effects the parent should measure

1. Regenerate BEFORE and AFTER from the same source bytes and shared dependencies into separate new versioned outputs. Keep frozen releases immutable. Record output hashes and join by `(source, uid, object id)`; distinguish source-byte/dependency drift from this patch.
2. Count affected raw windows, retained structures, skipped/merged pieces, rooms, room splits, and IDs across the entire adapter. Check doors and unaffected 2D profiles for exact equality. Expected geometry deltas are width-only on supported vertical 3D windows, except wider-footprint merge consequences and the explicitly added window Z-overlap guard. Count newly retained vertically separate windows separately from changed widths.
3. Verify all 39 saved cases explicitly, including the two test windows, and all 69 raw candidates. Reconcile any loss with merging, admission, or version lineage rather than silently reducing the expected set. Audit preserved sill/bottom Z, yaw, XY shifts, floor and neighbor joins.
4. After normal cm rounding and canonicalization, measure fixed-opening widths, fixed collisions, boundary violations, room admission/rejection, split membership and counts, and serialized input hashes. Widening fixed windows can change collision/admission outcomes and fixed sort order/canonical names even when source IDs stay stable.
5. Treat downstream model/placement effects as unmeasured until the parent rebuilds and evaluates. The proven result here is corrected structural-conditioning geometry, not a model-quality improvement.

## Evidence fingerprints

| File | SHA-256 |
|---|---|
| `fastfill/adapters/il3d_3dfront.py` | `c634223dbf1b358d1cdef1f9652d0f2bdceac48c014e7c085f89ba568c684ac8` |
| `fastfill/tests/test_source_opening_correction.py` | `5238aeb19a8cbf5df3c346e4e79cc3096705fc95ce6c7084ddbef29007dff8d5` |
| `docs/research/20261013-unknown-size-comparison/evidence/source-opening-correction-20261013/il3d_3dfront.before.py` | `b2eb88d7c2d6b97c27d8551d8ca80dd1d4aa64f6b8fa75b1bd93a1b7cacf779a` |
| `.release/audits/2026-09-28/source-check/designer/join_opening.py` | `44030d4163f95361d0553ec482e1abda3118d961bf84e0c8574d25cf56c4df8f` |
| `.release/audits/2026-09-28/source-check/designer/opening-target-join.json` | `6e1badf0dc7302e6a223f5a4696f653831c489fd7498095da199e888ad261ee3` |
| `/Volumes/harddisk/3D_Room_Collections/BillLin66__3D_Room_Collections/3DFront_exported/kits/convert_3dfront.py` | `275e1e156e3dc679ef95dc2e6dfbcea4b4c88bd9a57fcc503d3bcf4d8b11b6c9` |

Full-source JSONL and IL3D input hashes were not recomputed in this bounded task. Selected JSONL rows and the saved baseline are byte-verified as described above.
