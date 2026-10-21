# Scan2CAD / SpatialGen source-box correction — 2026-10-13

Implemented only in `fastfill/adapters/scan2cad.py`, `spatialgen.py`, the new narrowly shared `upright_bounds.py`, and the new `fastfill/tests/test_source_tilt_correction.py`. Both adapters now enclose the complete transformed source box at the existing exported yaw and compute the corresponding bottom. Semantic-front offsets and tilt policies remain intact. No edits were made by this task to `unified.py`, existing tests, IL3D, frozen data, or jobs. Integration and full regeneration belong to the parent/root task.

The implementation is supported by exporter code, native raw fields, independent eight-corner calculations, and fresh before/after loads of the three exposed rooms. It is not a mesh measurement, a full-corpus rebuild, or a claim that every historical geometry defect is fixed.

## Exact prepatch baseline and raw identity

The initial `git status --short` showed neither owned adapter modified. Both complete working files were read before editing, and the first subsequent diff identified the original blobs below. After the user's isolation request, those exact blobs were recovered with `git cat-file blob <observed-blob-id>` and retained. This is recovery of the observed clean working baseline, not an assumption that the current HEAD represents a previously modified working file. The helper did not exist before the patch; no fictional old helper was created.

| Observed prepatch source | Retained copy | SHA-256 |
|---|---|---|
| `scan2cad.py`, blob `bfed6b4ff547f249fb674d473d95b5380b96dd47` | [baseline/scan2cad.py](source-tilt-correction-20261013/baseline/scan2cad.py) | `eb82a6a602c4e6689d9871a4baa443b69502b376b6b751c4fd3e869271b7ac04` |
| `spatialgen.py`, blob `c54b6bd9aa0ec33eb690cdca0dc1fdad515a851b` | [baseline/spatialgen.py](source-tilt-correction-20261013/baseline/spatialgen.py) | `b9344dd58586de328ab25a3d9c66f8da7c75b22d3c7c1714674b38de8a139995` |
| `upright_bounds.py` | Absent before patch | Recorded in manifest |

[Baseline manifest](source-tilt-correction-20261013/baseline/manifest.json) includes absolute paths, sizes, blob IDs, and provenance. [Fresh raw-case validation](source-tilt-correction-20261013/raw-case-validation.json) records source/exporter SHA-256 values and before/after numbers. Both preserved baseline modules and corrected modules were imported independently and given the same raw room dictionaries. Input dictionaries were checked for immutability. The numerical “before” values below are freshly generated from these baseline copies, **not copied from frozen IR**; historical frozen-IR drift is a separate comparison for the parent.

Raw export identities for regeneration:

| Path relative to `/Volumes/harddisk/3D_Room_Collections/` | Bytes | SHA-256 |
|---|---:|---|
| `imChuling__3D_Room_Collections/SceneCAD_Scan2CAD_exported/scenecad_scan2cad_combined.json` | 196265906 | `fb67b1e0485a9c5ae971bb61162bb6cdb189f760b7bbab3cba2c3405a5e3c081` |
| `BillLin66__3D_Room_Collections/SpatialGen_exported/spatialgen.jsonl` | 895419 | `64257ce050cac75526fc3048b8fb756194e6be7d054fdec6c52713b3ae046edf` |

Reads were sequential and streamed. No dataset was copied. A metadata-only census of the Scan2CAD export established that all **1,506 scenes with CAD alignments** carry scan TRS scale `[1, 1, 1]`. The full export bytes were hashed in 1 MiB chunks to support identical-input regeneration.

## Verified source conventions

### Scan2CAD

Verified local exporter: `/Volumes/harddisk/3D_Room_Collections/imChuling__3D_Room_Collections/SceneCAD_Scan2CAD_exported/aggregate_scenecad_scan2cad.py`, especially lines 68–86, 125–169, and 175–193.

Native annotations are retained at `.../SceneCAD_Scan2CAD_exported/source/scan2cad/scan2cad_annotations.zip`, member `full_annotations.json`. Streamed the member through `scene0494_00`; native `aligned_models[5]` exactly equals the exported `source_fields.scan2cad_annotation`, and native scene TRS equals exported provenance TRS.

- Quaternion order is `[w, x, y, z]`; model TRS maps ShapeNet CAD coordinates to world coordinates.
- `bbox` contains native CAD half-extents; `center` is a local CAD box center, not the model translation. The exporter scales and rotates that center, adds model translation, applies inverse scan TRS, then ScanNet axis alignment.
- ShapeNet is Y-up. The exporter's basis `B` maps canonical Z-up coordinates to native CAD coordinates as `[x, y, z] -> [x, z, -y]`.
- Exported full size is `[2*bx*sx, 2*bz*sz, 2*by*sy]`; model scale is already present and must not be applied again.
- `rotation_matrix_zup_scan_local = A * R_scan.T * R_model * B`, where `A` is the upper-left ScanNet axis-alignment matrix. Matrix columns map local source axes into the exported scan frame. Rounded alignment scale is retained; normalizing columns would discard real exported geometry.
- Exported Euler convention is `Rz * Ry * Rx`. The correction uses the full matrix and keeps exported yaw; it does not reconstruct rotation from rounded Euler values.
- The exporter divides centers by scan scale but omits that scale from the exported orientation matrix. The all-unit metadata census establishes completeness for this collection. Missing or nonunit scan scale is explicitly treated as an evidence blocker, producing incomplete objects rather than guessed extents.

Independent native oracle: enumerate `center_CAD + diag(bbox) * sign` for all eight sign triples, apply model scale, quaternion rotation, translation, inverse scan TRS, and axis alignment. The known-case native corner set and exported-center/matrix/scaled-size corner set agree within `1e-12 m`.

### SpatialGen

Verified local exporter: `/Volumes/harddisk/3D_Room_Collections/BillLin66__3D_Room_Collections/SpatialGen_exported/kits/convert_spatialgen.py`, especially lines 124–175 and 252–291. `kits/unify_spatialgen.py` was also read: it copies JSONL records into the array view without recomputing fields.

Opened these native layouts and compared `bboxes[0]` exactly to each exported object's raw `class`, `transform`, and `size`:

- `.../SpatialGen_exported/data/scene_00024/room_layout.json`
- `.../SpatialGen_exported/data/scene_00030/room_layout.json`

`source_fields.transform` is a **row-major 4×4 affine matrix**. Its upper-left 3×3 includes rotation, nonuniform scale, and reflection; its translation column is the box center in Z-up meters. `source_fields.size` is the unscaled full local XYZ extent. Exported `furniture_size` instead contains local extent times column norm, rounded to six decimals. Multiplying that exported size by the full affine matrix would double-apply scale.

The adapter therefore uses the raw unscaled size with the full affine matrix, including its unrounded translation. Exported yaw and x/y tilt indicators remain untouched. The exporter encodes tilted objects with x/y `None`; the correction does not turn those into upright objects. Reflections are retained for bounding geometry without flipping the existing semantic-front convention. Exact ordinary yaw geometry is invariant; real exports can also change below a micrometer because the correction retains raw precision instead of rounded position/size fields.

## Corner math and retained policies

For local full extents `d`, center `c`, complete source linear map `L`, and a sign vector `s` in `{−1, +1}³`, the source corners are:

```text
p(s) = c + L * diag(d / 2) * s
Q    = rotation_about_world_Z(exported_yaw)
e    = abs(Q.T * L) * d
bottom_absolute = c.z - e.z / 2
```

`abs` is elementwise. This gives the minimum enclosing box **at the fixed requested yaw**, including contributions of local height to horizontal bounds and horizontal axes to vertical bounds. The box is centered at `c` because the affine image is centrally symmetric. Reflection and shear do not require discarding signed matrix entries before projection.

The helper computes extents in the exported yaw frame. Existing `convert_room` then applies Scan2CAD's `+90°` or SpatialGen's `−90°` semantic-front offset and swaps horizontal size axes. The tests independently enumerate corners and rotate them into the final IR yaw frame, asserting both extrema and containment; they do not merely mirror the helper formula.

For Scan2CAD, corrected absolute bottoms feed the existing floor-standing category vote, existing `tilt <= 10°` rule, strict `abs(bottom) < 0.25 m` voting band, median floor, and strict `abs(relative_bottom) < 0.15 m` snap rule. The snap intentionally translates the resulting envelope: it is not a claim that snapped IR encloses the unshifted raw corners in absolute Z. Source-local axes below 1 mm still make the object incomplete, even when tilt could inflate its envelope. Such unusable boxes no longer vote for the floor.

For both sources, nonfinite inputs, missing transforms/yaw, malformed matrix shape, nonpositive local size, singular linear maps, and overflowed extents produce incomplete geometry. SpatialGen rejects non-affine last rows. There is no Euler-only or identity fallback. These guards use the existing `n_incomplete` path. No blocker was found for the three exposed real cases.

## Fresh same-raw before/after results

Complete raw rooms were loaded (6 Scan2CAD objects, 99 and 20 SpatialGen objects); the target corners were independently checked. Counts, target yaw, target tilt status, and input dictionaries were unchanged. All three exposed targets remain `tilted=True`.

| Source / target | Before IR size (m) | Corrected IR size (m) | Corrected absolute bottom (m) |
|---|---|---|---:|
| Scan2CAD `scene0494_00::scan2cad_5` | `[0.446364754, 0.408415713, 0.893607967]` | `[0.595311359, 0.420781030, 0.962946057]` | `-0.074864409` |
| SpatialGen `spatialgen_scene_00024::obj_0` | `[0.040000000, 0.800000000, 1.180454000]` | `[0.142731094, 0.800000023, 1.179448088]` | `0.000319856` |
| SpatialGen `spatialgen_scene_00030::obj_0` | `[0.030000000, 1.506951000, 0.750000000]` | `[0.159780606, 1.506950696, 0.743815275]` | `0.003092362` |

Maximum target extent discrepancies against native corner enumeration were respectively `7.78e-16`, `2.50e-16`, and `6.94e-16 m`. Scan2CAD's room floor median changes from `0.041142559` to `0.002206080 m`; target relative bottom before snap is `-0.077070489 m`, so both baseline and corrected IR bottom are zero under the unchanged snap policy. SpatialGen baseline IR bottoms were `-0.000183` and `0`; corrected IR bottoms equal the raw minima above.

These results agree with the historical source-check corner findings. The first painting's historical audit used a slightly different pre-correction bottom precision; the fresh preserved-baseline run above is the correct isolation reference. Historical evidence was consulted at `.release/audits/2026-09-28/source-check/scanned/scan2cad-source-contract-review.json`, `scan2cad_final_impact.py`, and `generated/spatialgen-geometry.json`, alongside [the full data audit](full-audit-data-report-20261013.md).

## TDD and validation

All tests are in the new `fastfill/tests/test_source_tilt_correction.py`; no existing tests were edited.

1. **RED, before implementation:** 32 failed / 16 passed. Failures included all three exposed native cases, projected tilted extents, corrected bottom/floor voting, malformed transforms, and degeneracy. Ordinary yaw controls already passed.
2. **GREEN, initial implementation:** 48 passed.
3. Added scan-scale evidence-blocker regressions before implementing that guard: **3 failed / 51 passed**; then the guard made them pass.
4. Final source-specific suite: **57 passed**. It covers the raw cases; ordinary yaw; tolerated and flagged tilt; mixed roll/pitch; 90° tilt; signed/nonuniform SpatialGen scale; unchanged front/yaw; floor voting/bands/snapping; nonfinite/degenerate/missing/overflow geometry; missing/nonunit scan scale; immutability; and actual JSON/JSONL file-to-IR loads, shell boundary metadata, and no-CAD skip behavior.
5. Combined existing and new regression command: **110 passed** in 0.65 s:

```sh
python -m pytest fastfill/tests/test_source_tilt_correction.py \
  fastfill/tests/test_adapter_repairs.py \
  fastfill/tests/test_embedded_contracts.py -q --tb=short
```

`git diff --check` passed for the owned code. [Coverage receipt](source-tilt-correction-20261013/test-coverage.json) uses Python stdlib `trace` and executable-line enumeration: Scan2CAD **73/79 (92.41%)**, SpatialGen **23/23 (100%)**, helper **26/26 (100%)**. This is executable-line coverage, not branch coverage or a whole-project coverage claim; `coverage.py`/`pytest-cov` were unavailable. The unexecuted Scan2CAD lines are the pre-existing invalid-shell return and `__main__` self-check.

Self-review checked scale application, basis direction, raw center offsets, final front-frame swaps, matrix handedness, finite guards, floor ordering, and immutability. A delegated reviewer was attempted but the agent tool reported its thread limit; no independent-agent review is claimed.

## Parent handoff and limits

Use the preserved baseline modules and corrected adapters against the **same raw bytes**, checking the source hashes first, to isolate adapter changes. Compare that baseline regeneration to frozen IR separately to measure historical drift. The new helper is needed only for the corrected version. Keep the same shared `unified.py` implementation and downstream settings across paired adapter runs.

No complete before/after corpus was regenerated, no prior/qualification/denominator was recomputed, and no training or remote/paid job was launched. Geometry changes can legitimately change downstream flags, selection, floor medians, or priors; those effects belong to integration. This patch preserves source semantic-front offsets without claiming independently verified per-asset semantic fronts, especially for mirrored SpatialGen instances.
