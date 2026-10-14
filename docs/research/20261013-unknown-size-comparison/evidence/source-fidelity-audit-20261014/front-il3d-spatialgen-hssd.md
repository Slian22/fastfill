Every problem we measured in the GT for SpatialLM, Structured3D and InteriorGS is already present in the official released data, apart from two small groups we introduced in `build.py` and a handful from the collaborators' unification. The adapters are clean.

**How I checked.** I traced every test room, not only samples: 2,461 SpatialLM, 747 Structured3D and 311 InteriorGS rooms. Each eval object was mapped back through the release-v3.2 IR (`.release/v3.2/ir/*.jsonl`) to its official record. The frozen eval was built from that IR, and the adapters, `build.py` and `scene.py` at HEAD are byte-identical to `.release/v3.2/code`. I then recomputed footprints, z ranges, boundaries and doors from the official files:

| Source | Official files used | Collaborators' file the adapter reads |
|---|---|---|
| SpatialLM | layout txt from `SpatialLM_exported/raw/layout_source.tar.gz` (the first `split.csv` sample per room, which the exporter used) | `SpatialLM_exported/spatiallm.json` |
| Structured3D | `annotation_3d` + `bbox_3d` from `Structured3D_exported/structure_topology.tar.gz` | `Structured3D_exported/structured3d.jsonl` |
| InteriorGS | `InteriorGS_exported/interiorgs/<scene>/{labels,structure}.json` | `InteriorGS_exported/interiorgs.json` |

All paths are under `/Volumes/harddisk/3D_Room_Collections/imChuling__3D_Room_Collections/`.

**What "preserved" means.** Eval geometry matches the official data to within cm rounding: object footprint Hausdorff distance p99 ≤ 0.9 cm (max 2.0 / 1.9 / 1.0 cm), boundary median 0.4 cm, and frame offsets agree to 1e-6 m. Wherever object geometry is preserved, a problem in the GT is also in the official data.

**Problem labels.** They come from `validate.check` (frozen checker; I re-ran it on 180 rooms and got 0 mismatches) and from the A2 lists in `gt-anatomy/doors/gt2.jsonl`.

## SpatialLM (2,461 test rooms): OFFICIAL, plus BUILD rounding

**How the collaborators' file compares to the release.** It is a lossless parse. All four samples of each room are byte-identical (2,461/2,461), and all 7,777 door/window centres lie exactly on their referenced wall (distance 0.000 m).

**Conventions checked against the official data:**
- **Box centre:** 95.1% of 6,374 floor-furniture bottoms (centre z − sz/2) are within 2 cm of 0.
- **Angle sign:** 36 of 39 oblique objects next to oblique walls are parallel with +angle_z (3 with −angle_z).
- **Ring = inner wall face:** facing edges of neighbouring rooms are 0.24 m apart in 2,201 cases, 0.12 m in 1,639 and 0.20 m in 694; only 9 are 0. Wall thickness is 0.0 in the official txt itself.
- **Ring repair:** our buffer(0) repair (97 test rooms) causes no poke.

**Wall pokes (2,930 objects):**
- 1,602 objects at ≤ 1.2 cm are **BUILD** rounding: inside in the official data, outside only after cm/degree rounding at `build.py:228-233`. In 709 rooms these are the only pokes.
- 331 objects at ≤ 1.2 cm, and every deeper one (430 at 1.2–5 cm, 212 at 5–10 cm, 355 over 10 cm), are **OFFICIAL**. The box centre is always inside the room.
- Of the 567 pokes over 5 cm, the median depth is 0.13 m and 42 go deeper than 0.5 m. Only 12 reach another room's floor. 92 sit at a window and 97 at a door. Most common: bed 66, plants 58, wardrobe 48, sofa 46, cupboard 39.
- Examples: `scene_008003::room_0` carpet obj_0008, eval 0.285 / official 0.287 m; `scene_007159::room_1` bookcase obj_0003, 0.121 / 0.121 m.

**Checker collisions (1,108 pairs):**
- 1,083 **OFFICIAL**. Example: `scene_009798::room_0` washing_machine obj_0006 × sideboard obj_0010, eval 0.226 m² / official 0.230 m², height overlap 1.30 m. Most common pairs: coffee table × sofa 125, floor lamp × sofa 56, bed × wardrobe 51.
- 24 **BUILD** rounding (official overlap ≤ 0.01 m²).
- 1 **BUILD** z-snap: `scene_010932::room_3` wardrobe obj_0003 stands at 0.45–2.60 m over combination_sofa obj_0002; build puts it at z = 0, giving a severe 0.897 m² pair.

**Door blocks (1,444 rooms, 3,006 intrusions):** all **OFFICIAL**; door geometry is unchanged. 1,340 of 2,163 core intrusions are at doors ≥ 1.5 m wide. 21 intrusions are at doors that extend past their wall segment (e.g. `scene_008768::room_4` fixed_0001, 2.33 m).

**Stratified sample** (seed 20261014): genuine `scene_009798::room_0`, `scene_003587::room_2` (sofa × bookcase 0.515 / 0.510 m²), `scene_003011::room_2`, `scene_008768::room_4`; pokes `scene_008003::room_0`, `scene_007159::room_1`, `scene_004975::room_0`, `scene_008582::room_2` (chair 0.112 / 0.115 m); door blocks `scene_007966::room_1`, `scene_007185::room_3`, `scene_012037::room_2`, `scene_007940::room_4`. All OFFICIAL except the rounding pokes. Clean rooms `scene_002693::room_1`, `scene_009588::room_1` and `scene_007159::room_0` are reproduced to ≤ 8 mm.

## Structured3D (747 test rooms): OFFICIAL, with small UNIFIED and BUILD shares

**Boundary.** Re-tracing the official floor plane reproduces the eval boundary within 1.3 cm. The collaborators use the same method as the official floorplan code. Only the first traced ring is kept, which drops a 0.6 m² second part in `structured3d_01200_room_1902` and `structured3d_02506_room_1902`.

**Wall pokes:**
- 429 objects are **BUILD** rounding; in 224 rooms they are the only pokes.
- The rest are **OFFICIAL**: the official box crosses the floor-polygon edge with its centre inside (86 at ≤ 1.2 cm, 124 at 1.2–5 cm, 40 at 5–10 cm, 56 over 10 cm).
  - Example: `structured3d_00102_room_672454` cabinet, bbox 19, is a 4.90 × 1.70 × 2.68 m box with 3.06 m² outside every room; it pokes 1.22 m.
  - Example: `structured3d_00224_room_577003` chair rotated 45°; the box corner pokes 0.084 m.
- 4 are **UNIFIED**: the exporter's rule that assigns an object to the nearest room within 0.5 m when its centre is outside every room. Three of them are over 10 cm:
  - `structured3d_00275_room_606054`: bbox 20 is a 4.76 × 0.00 × 4.40 m flat panel, labelled "table" from only 20 votes, giving a 1.02 m poke.
  - `structured3d_02676_room_619886`: desk, bbox 105, 0.48 m poke.
  - `structured3d_00752_room_715215`: cabinet, bbox 71, 0.26 m poke.

**Checker collisions:**
- 252 **OFFICIAL**. Examples: `structured3d_02941_room_3365` refrigerator × cabinet 0.273 / 0.269 m², and sofa × table 0.47 / 0.46 m²; `structured3d_02796_room_328` sofa × table 1.108 / 1.100 m².
- 7 **BUILD** rounding.
- 1 **BUILD** z-snap: `structured3d_02509_room_687234` table (bbox 77, official z 0.45–1.33 m) stands on a 0.45 m bed platform (bbox 26); build puts it at z = 0, giving a severe 0.22 m² pair.

**Door blocks:**
- 929 of 931 intrusions are at doors that the official topology links to the room (they share lines with one of the room's wall planes).
- 1 is **UNIFIED**: in `structured3d_01827_room_549`, door fixed_0002 (semantic 15) officially belongs to rooms 474 and 187. Its centre is 0.47 m from this room, so the exporter's 0.5 m rule attached it here, and cabinet obj_0001 sits 99 cm into its strip.
- Overall, 4 of 1,704 test doors are attached this way: `02164_room_1127` semantic 19, `02164_room_5457` semantic 35, `02488_room_629897` semantic 14, plus the one above.
- 1 intrusion was at door fixed_0006 in `structured3d_02251_room_352`, which wasn't checked against the official topology.
- Very wide official doors are common, e.g. 3.26 m in `structured3d_02062_room_267` and `_245`.

**Stratified sample** (rooms not named above): `00859_room_2074`, `00783_room_687`, `02850_room_290` (cabinet 0.065 / 0.057 m), `02656_room_1515`, `02164_room_196`, `02981_room_3740` (bed 0.132 / 0.133 m), `02715_room_1701`. All OFFICIAL except `01827_room_549`'s second door. Clean rooms `03186_room_7062`, `00631_room_278` and `02352_room_2295` are reproduced to ≤ 7 mm.

## InteriorGS (311 test rooms): OFFICIAL, plus BUILD z-snap and rounding

**Matching.** Every eval object has its centre inside exactly one room profile (4,344/4,344), and the profile is the inner wall face (wall centre line = profile ± thickness/2). The adapter's membership recovery added no test objects.

**Wall pokes:**
- 105 objects are **BUILD** rounding; in 60 rooms they are the only pokes.
- The rest are **OFFICIAL** (20 at ≤ 1.2 cm, 39 at 1.2–5 cm, 20 at 5–10 cm, 25 over 10 cm).
  - `0745_841600::room_2`: sofa ins 184 pokes 0.576 m and carpet ins 185 pokes 0.994 m, both through an opening into room 3.
  - `0080_839950::room_0`: "Swimming pool", 0.30 m.
  - `0368_840234::room_3`: stool, 0.24 m.

**Checker collisions:**
- 309 **OFFICIAL**. Examples: `0649_841461::room_4` sofa × chair 0.221 / 0.223 m²; `0338_840412::room_0` bed × mattress. Most common: plant × plant 25, coat hanger × trappings 22.
- 3 **BUILD** rounding.
- 16 **BUILD** z-snap or re-stacking:
  - `0456_840233::room_5` (7 pairs): a 0.10 m "chair" box carries 7 chairs at z = 0.10; build drops them to z = 0.
  - `0104_839982::room_0` (6 pairs): sports equipment over 6 cm shelves.
  - `0297_840578::room_8` (1 pair): towels re-stacked.
  - `0649_841461::room_2` (2 pairs): stones.

**Door blocks:** all **OFFICIAL**. The 115 DOOR holes involved match the official profile width exactly, and 74 of them are ≥ 1.5 m.
- 7 intrusions come from a furniture box labelled "Door frame" (`0900_841496::room_0` ins_114, 4.83 m wide) that the downstream door-word match treats as a door.
- In `0811_841226::room_3`, the blocker is a "Spotlight" (ins 403) that is a 3.15 × 2.26 × 0.88 m floor box, which is label noise in the official data.

**Boundaries.** Build's `clean_boundary` removes zero-width spikes from three profiles (e.g. `0456_840233::room_0`: Hausdorff 0.70 m but only 0.08 m² of area). This is harmless.

**Stratified sample** (rooms not named above): `0781_840320::room_5`, `0957_841885::room_0`, `0315_840533::room_1`, `0991_841976::room_1`, `0787_841244::room_0`, `0787_841244::room_2`, `0832_840249::room_2` (a "person" model blocking a door). All OFFICIAL. Clean rooms `0661_841476::room_0`, `0954_841881::room_6` and `0511_840291::room_4` are reproduced to ≤ 9 mm.

## Where the spreadsheet or README disagrees with the data

**SpatialLM:** no material disagreement. The rotation range, centre convention, scale axes and identical samples all hold. Two points are missing or unclear:
- The sheet never says wall thickness is 0.0 everywhere.
- "墙线环" (wall-line ring) is the room's inner face, not a wall centre line.

**Structured3D:**
- Door and window `width` is max(X span, Y span) of the merged bounding box; this is wrong for oblique openings (6 of 1,704 test doors are off by more than 5 cm).
- Doors and windows are assigned to rooms by a 0.5 m centre-distance rule (this is in the README, not the sheet). 4 test doors are not on the room's walls.
- Objects up to 0.5 m outside are assigned to the "nearest" room.
- The exported size/yaw heuristic gives a wrong footprint (> 5 cm) for 453 of 1,230 non-upright boxes. The sheet flags it as non-semantic, and the adapter ignores it and uses `raw_basis`.
- Categories are not official: they come from image voting plus Ctrl-Room labels.

**InteriorGS:**
- The sheet gives door `width` as the horizontal span. It is wrong for 162 of 11,989 oblique openings (the README states the caveat; the sheet does not).
- The sheet says object size is "from the convex hull of the 8 corners". The code actually uses the corners at minimum z and falls back to the first 4 corners, which is wrong for tilted boxes.
- The sheet says doors are assigned to rooms by profile centre. Profile centres lie inside walls, so most doors are not assigned; the adapter re-matches them (5,714 of 5,716).
- Correct as stated: the room_box index (i+1) matches its profile for 6,607/6,607 rooms, and object assignment by unique centre holds.

## Fixes (ADAPTER/BUILD only; nothing changed)

1. **BUILD rounding** (`build.py` `rounded` at :228-233, plus `clean_boundary`): after rounding, if an object's original footprint was inside the room but the rounded one leaves the cleaned boundary by ≤ 1.5 cm, nudge it 1 cm inward. This removes the pokes ≤ 1.2 cm from rounding (1,602 / 429 / 105 objects). It makes 709 / 224 / 60 rooms poke-free, though R5 can still fail them for other reasons.
2. **BUILD z-snap** (`prep` → `z_of`, which forces every parentless object to z = 0): if an object's source z sits within 5 cm of the top of a box underneath its footprint, make that box its parent. Otherwise keep the object as fixed or drop it rather than dropping it into another box. This covers 18 test pairs (1 / 1 / 16).
3. **Adapter guards for the Structured3D unification issues:**
   - Drop objects with `assignment_method == "nearest"` whose centre lies outside the room (12 test objects, 4 pokes).
   - Accept a door only if `structure_topology` links it to the room's wall planes, or, without topology, only if it is parallel to and within 0.3 m of an edge (4 test doors).

## Train rooms

I ran build `prep` with the v3.2 arguments on hashed samples of non-test rooms; test figures come from the eval labels.

| Source | Sample size | Genuine collision (train / test) | Poke > 5 cm (train / test) | Rooms with ≤ 1.2 cm pokes (train / test) |
|---|---|---|---|---|
| SpatialLM | 2,381 | 27.0% / 28.2% | 17.8% / 19.0% | 50.2% / 52.2% |
| Structured3D | 936 | 20.1% / 22.4% | 12.2% / 12.0% | 47.5% / 44.4% |
| InteriorGS | 515 | 28.0% / 29.9% | 12.4% / 10.6% | 32.0% / 28.9% |

The Structured3D sample also has 15 nearest-assigned pokes with the centre outside the room, and z-snap pairs appear in 0.3% / 0% / 1.0% of rooms. The training targets carry the same OFFICIAL issues and the same BUILD artifacts.

## Limits

No meshes are on disk, so I can't separate loose boxes from real interpenetration. Every "OFFICIAL" verdict means the overlap or poke is in the released boxes. Structured3D labels are the collaborators' votes.

All scripts and outputs are in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/fidelity/slm-s3d-igs-audit/`:
- per-room traces: `slm_trace.jsonl`, `s3d_trace.jsonl`, `igs_trace.jsonl` (made by `*_trace.py`)
- counts: `final_counts.py`
- stratified sample: `sample.py`
- SpatialLM conventions: `slm_semantics.py`
- inspectors: `s3d_inspect.py`, `igs_inspect.py`
- train comparison: `train_rates.py` → `rates_*.jsonl`
- labels: `labels.jsonl`, copied from `fidelity/slm-s3d-igs/`