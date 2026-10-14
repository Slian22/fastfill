# Where the GT problems come from: official data, collaborators' unification, or our conversion

Almost all GT problems we measured are already in the official released data. Our conversion adds a small, specific set, and only two of those change any headline number. The collaborators' unified files contain many errors, but nearly all of them stay out of the evaluated GT, because our adapters fix them or don't read those fields. Only two collaborator errors reach the GT: SceneSmith asset sizes that ignore the mesh scale (a few assets in 3–4 rooms) and the Structured3D rule that attaches objects and doors to the nearest room within 0.5 m (4 objects and 4 doors).

All four audits covered every one of the 8,647 test rooms (3,519 + 1,371 + 2,669 + 1,088), not samples. InternScenes_gen training rooms were sampled 1 in 5.

Layer names used below:
- **OFFICIAL**: in the dataset's own released files.
- **UNIFIED**: introduced by the collaborators' unification.
- **ADAPTER**: introduced by our `fastfill/adapters` (raw to IR).
- **BUILD**: introduced by our `build.py` (IR to train/eval data).
- **EVAL**: an artefact of our downstream scoring rules, not of any data layer.

## 1. Re-verification of the most consequential claims (my own scripts, read-only)

**V1. Holodeck 10 cm padding: confirmed exactly, on all official scenes.** The fix is in our adapter, but the padding itself is official.
- Across all 9,977 official HOLODECK scenes in `OptiScene/scenes.tar.gz` (66,855 objects), the released bbox width and depth equal the Objathor mesh box + 0.10 m (range +0.095 to +0.105, median +0.1005 / +0.1001). Height is exact (within ±0.5 cm).
- Padded boxes poke more than 1 cm through the official floor polygon for 57,798 of 66,855 objects. The unpadded mesh boxes poke more than 2 mm for 0 objects.
- The collaborators' README ("size = bbox/100") and the official README both leave the padding out.
- Our adapter keeps the padding on purpose: `optiscene_holodeck.py:25` records `size_note` "padded ~11 cm"; the measured value is 10 cm.

**V2. BUILD rounding is the cause of the ≤1.2 cm wall pokes: confirmed against the official SpatialLM files.**
- I took 4 rooms that fail R5 only on ≤1.2 cm pokes and extracted their official layout txt from `raw/layout_source.tar.gz` myself (all 4 samples of each room are identical).
- All 5 R5 objects lie inside the official wall ring (depth 0.0) and poke exactly 5 mm in eval.
- Mechanism, shown on `spatiallm:scene_000467::room_2` washing machine obj_0001:
  - Officially, the box face sits 0.004 mm inside the wall.
  - `num()` rounds the centre 3.66597 → 3.67 (+4.0 mm), and `shown_size()` rounds the size 0.66806 → 0.67 (+1.0 mm on the half-size). The face ends up 5.0 mm through the wall.
  - Same pattern for: scene_005031::room_4 obj_0007, scene_002374::room_2 obj_0005/obj_0006, scene_005946::room_3 obj_0001.

**V3. IL3D `assets.json` sizes (ADAPTER `il3d_3dfront.placed_size`, :88-94): confirmed.** The bad sizes are in IL3D's own release; the adapter trusts them.
- In a 1-in-8 sample of 3D-FRONT houses, 5,137 IL3D 3D-FRONT assets also appear in the house JSONs. For 4,369 (85%) the IL3D size matches 3D-FRONT's `size` within 2 cm; for 162 (3.2%) it differs by more than 10 cm.
- Some of these are unit-cube-normalised meshes (10 have a maximum extent of exactly 2.0). Example: Dining Chair 6914bc9b is 1.94 × 2.0 × 2.0 m in `assets.json` against 0.48 × 0.50 × 0.75 m in 3D-FRONT.
- The eval GT room `il3d:b72fe1d0…` contains this chair as a 2.0 × 2.0 × 1.94 m object in a 3.53 × 3.10 m room.
- I did not recount the auditor's 2,141 placements / 1,780 IR rooms (8.2%). For some of the 162 mismatches 3D-FRONT itself may be the wrong side (e.g. a TV stand at 3.65 vs 2.49 m).

**V4. SceneSmith scale-unaware sizes (UNIFIED): real, but much narrower than the audit's "4,277 objects (22%)" suggests.**
- `geometry_metadata` has 8,541 assets measured with the scale-aware method and 1,792 with the older "glTF POSITION accessor" method. The older method covers 52 test rooms. The README claims every size includes the SDF scale, which is false for those 1,792.
- By category, the two methods agree: median height ratio 0.98 over 27 shared categories. The share of furniture more than 2× off its category median is 5.1% (old method) vs 4.6% (scaled). So most old-method entries probably have scale 1 and are correct.
- Over all 364 test rooms, floor furniture poking through walls clusters in old-method entries:

| Size method | Floor furniture | Poke > 2 cm | Poke > 10 cm |
|---|---|---|---|
| Old (scale-unaware) | 510 | 16 | 10 |
| Scale-aware | 2,421 | 3 | 0 |

- All 10 pokes over 10 cm are in scenes 077, 102 and 112: backbar shelving unit 1.04 × 2.40 × 6.41 m (six units), refrigerated display case 4.96 m tall, banquette 3.27 m, bookshelf. "Probable" is upgraded to confirmed for those few assets only; meshes are not on disk.

## 2. Problem type by source and layer (test GT)

Collisions are checker main/severe pairs, written as O / U / A / B (OFFICIAL / UNIFIED / ADAPTER / BUILD). Pokes and door counts are as reported by each audit.

| Source (test rooms) | Collision pairs O / U / A / B | Wall pokes | Door blocks |
|---|---|---|---|
| SpatialLM (2,461) | 1,083 / 0 / 0 / 25 (24 rounding, 1 z-snap) | ≤1.2 cm: B 1,602, O 331; 1.2–5 cm: O 430; >5 cm: O 567 | 3,006 intrusions, all O |
| Structured3D (747) | 252 / 0 / 0 / 8 | ≤1.2 cm: B 429, O 86; 1.2–5 cm: O 124, U 1; >5 cm: O 96, U 3 (nearest-room rule) | 931: O 929, U 1, 1 unchecked |
| InteriorGS (311) | 309 / 0 / 0 / 19 (16 z-snap/re-stack) | ≤1.2 cm: B 105, O 20; >1.2 cm: O 84 | all O; 7 intrusions are EVAL (a "Door frame" furniture box matched as a door) |
| SpatialGen (45) | 12 / 0 / 0 / 0 | ≤1 cm: B 9; >5 cm: O 12 | 24 hits, O |
| IL3D_synthetic (316) | 15 / 0 / 0 / 0 | none | none |
| HSSD200 (138) | 146 / 0 / 17 (articulated URDF boxes, 5 of them duplicates) / 4 | >5 cm: O 65 | 24 hits, O |
| IL3D_3dfront (872) | 852 / 0 / 14 (assets.json size) / 8 | ≤1 cm: B 24, O 13; >5 cm: O 176, A 14 | 554 hits, O geometry; 242 are at doors listed under the neighbouring room and attached by the adapter, 240 of them correctly |
| OptiScene_holodeck (503) | none | 1–5 cm: 2,901 objects, O (padding; fix belongs in A) | no doors; A4: 88 pairs / 78 rooms, EVAL (snap × padding) |
| MansionWorld (1,313) | 35 / 0 / 0 / 3 | >1 cm: O 2 | 56 objects / 42 rooms, all EVAL (15 cm side margins; none in official doorBoxes) |
| SAGE-10k (489) | 72 / 0 / 8 (tilt box) / 4 | >1 cm: O 8, A 7 | rooms: EVAL 90, O 5, B 8 (door-prop furniture turned into doors) |
| SceneSmith (364) | 100 / 24 (confirmed for a few assets, see V4) / 6 / 4 | >1 cm: U 19, O 10, A 3 | no doors |
| InternScenes_scannet (104) | 401 / 0 / 12 / 16 | >5 cm: O 5, A 4 | 112 hits: real O 59, unusable leaf 48, U (hull) 3, EVAL 2 |
| InternScenes_arkit (229) | 191 / 0 / 11 / 8 | >5 cm: O 62, A 6 | 113 hits: real O 60, unusable leaf 53 |
| InternScenes_3rscan (70) | 216 / 0 / 15 / 6 | >5 cm: O 9, A 2 | 77 hits: real O 52, unusable leaf 23, EVAL 2 |
| InternScenes_mp3d (18) | 20 / 0 / 0 / 0 | >5 cm: O 3 | 6 hits, all unusable leaf |
| InternScenes_gen (660) | 19 / 0 / 0 / 1 | >5 cm: O 7 | 238 hits: real O 128, unusable leaf 101, EVAL 9 |
| MultiScan (6) | 2 / 0 / 0 / 0 | A 1 (12 cm, `boundary()` rule) | 1, O |
| Scan2CAD (1) | none | O 3 (SceneCAD layout smaller than the scan) | none |

"Unusable leaf" means the official door box is an open leaf or lies off the floor outline, so the door strip built from it is meaningless.

**Totals:**
- **Collision pairs:** 3,938 in all. OFFICIAL 3,725 (94.6%), UNIFIED 24 (0.6%), ADAPTER 83 (2.1%), BUILD 106 (2.7%).
- **Wall pokes over 5 cm:** about 1,060 OFFICIAL against about 50 non-OFFICIAL (UNIFIED about 22, ADAPTER about 28).
- **Wall pokes of 1.2 cm or less:** about 2,190 objects are BUILD rounding.

## 3. Collaborator-processing issues, with evidence

**3a. Issues that reach the evaluated GT:**
1. **SceneSmith sizes (V4).**
   - 1,792 asset entries in 52 rooms were measured without the SDF `<scale>`, although the README says all 19,360 sizes include it.
   - Confirmed wrong: backbar_shelving_unit (6.41 m tall, scene_112), refrigerated_display_case (4.96 m, scene_102), banquette_bench (scene_112), bookshelf (scene_077).
   - Fix: re-extract those 1,792 entries with the scale-aware path; this needs the official 1.13 TB tar.
   - Also: all 910 "windows with location" sit at a room-centre placeholder with z = 0. Our adapter drops them.
2. **Structured3D 0.5 m nearest-room assignment** (in the README, not in the spreadsheet):
   - Objects: 12 test objects, 4 of them poke through walls. Example: `00275_room_606054` bbox 20 is a 4.76 × 0.00 × 4.40 m panel labelled "table" and pokes 1.02 m.
   - Doors: 4 test doors are not on the room's walls. In `01827_room_549` this causes one A2 intrusion.
   - Door width = max(X span, Y span) is wrong for oblique doors (6 of 1,704 test doors off by more than 5 cm).
3. **InternScenes convex-hull boundary.** The hull exceeds the floor mesh by more than 5% in 140 of 421 Real2Sim test rooms, and misplaces 3 door strips (e.g. scannet scene0051_00).

**3b. Claims that are wrong or missing, but don't reach the GT (our adapters fix them or don't read the field):**
- **MansionWorld:**
  - The (x, −z, y) map mirrors the scene.
  - Wall objects are recentred on the vertex mean (816 of 816; median 1.38 m off).
  - The 0.10 m padding is subtracted from wall objects as well (0.12–0.20 m too short).
  - x/z flips are dropped (7.4%).
  - "corridor" means the open hub of each floor.
- **SAGE:**
  - The front is local +Y, not +X (93.0% vs 3.1%).
  - x/y tilts are not noise (9.2% of objects tilt more than 2°).
  - Door- and window-type furniture objects are undocumented.
- **Holodeck:**
  - The padding is undocumented.
  - The `rotation.z` sign is flipped.
  - "0 objects out of bounds" holds for centres only; 86.5% of boxes cross the wall.
- **IL3D:**
  - "No multi-room layouts" is false (1,300 layouts merge 2–10 rooms).
  - The boundary is a 4-point rectangle (area off by more than 5% in 288 of 605 traced rooms).
  - Size is read as [x, z, y], and the layout scale is ignored (IL3D_synthetic: 7,289 of 47,837 objects).
  - Bottom-centre position is false for HSSD meshes.
- **3D-FRONT export:** size length and height are swapped for about two-thirds of placements.
- **HSSD:**
  - Door/window centres estimated from metadata are a median 0.85 m too high (2,956 of 2,956).
  - The "−Z front" claim is wrong: 1,755 of 1,781 objects face +Z.
- **InteriorGS:**
  - The spreadsheet's door width caveat is missing (162 of 11,989 oblique openings wrong).
  - The size rule actually uses the min-z corners.
  - Assigning doors by profile centre leaves most doors unassigned; the adapter re-matches 5,714 of 5,716.
- **InternScenes:**
  - 41–63% of test doors are open or off-outline door leaves.
  - The Gen floor sits at 0.09–0.15 m.
  - Corrupt official boxes remain (beds 1,910 m and 803,014 m tall; an 8.86 m bed).
  - 96 mp3d regions are duplicated.
  - "Copy Room" should be "Copy/Mail Room".
- **MultiScan:**
  - `obb_axes` are rows, not columns.
  - `floor_height` is the top of the floor slab.
  - "Point-cloud containment validated" fails for 11 objects (8 contain 0% of their own points; 3 are in a test room). OFFICIAL vs UNIFIED is undetermined.
- **SpatialLM:** only undocumented points (wall thickness is 0; the ring is the inner wall face).
- **SpatialGen:** all claims check out.

## 4. Our conversion bugs to fix

**B1. BUILD rounding** (`build.py:230`, `rounded`; positions at :241 in `z_of`; fixed boxes at :232-233; via `scene.py:100-107` `num`/`shown_size`).
- Centre and size are rounded independently, so flush faces move up to about 7.5 mm.
- Fix: for boxes square to the room frame, round the two faces to cm and derive centre and size from them. This rounding is monotone, so it creates no new wall pokes or overlaps. For rotated objects, nudge inward while checking collisions.
- In my counterfactual, a naive ≤2 cm nudge removes R5 in 406–478 rooms but creates checker failures in 13–17 rooms. Plain nudging is therefore unsafe.
- Also re-check the 0.01 m² collision threshold after rounding (about 88 marginal pairs).

**B2. BUILD z-snap** (`build.py:238-243` `z_of`, plus `anchors.infer_anchors` at `anchors.py:90`, which only accepts a support within 5 cm).
- Floating objects are dropped to z = 0 and into the furniture below them: 18 pairs in SpatialLM/Structured3D/InteriorGS, InternScenes examples, and MultiScan sofas lifted 0.21–0.36 m.
- Fix: when an object floats, parent it to the box below its footprint.

**B3. BUILD door props** (`anchors.fixed_geometry`, `anchors.py:124`).
- SAGE furniture named "door" becomes a door opening (8 test rooms).
- Fix: build doors only from structure records.

**B4. ADAPTER Holodeck padding** (`optiscene_holodeck.py:25`).
- Fix: use `b[1]/100 - 0.10` and `b[2]/100 - 0.10`, as `mansionworld.py:53` already does.

**B5. ADAPTER IL3D sizes** (`il3d_3dfront.py:88-94`).
- Fix: when the `assets.json` extents differ from the layout bbox / 3D-FRONT size by more than 10 cm (or have a maximum extent of 2.0), use the bbox or flag the object.

**B6. ADAPTER HSSD articulated duplicates** (`hssd200.py:182-185`, URDF boxes).
- Fix: drop URDF boxes that rigid boxes cover by 70% or more (79 boxes in 51 IR rooms).

**B7. ADAPTER tilted-box enclosure** (`internscenes.py:55` `tilted_bbox`; `sage10k.py:55` `upright_box`).
- The yaw-aligned box around the 8 corners inflates the footprint: 38 + 8 + 6 test pairs, about 22 pokes, about 888 training pairs in the scan sources.
- Fix: use the minimum-area rectangle of the projected corners, or flag the object.

**B8. ADAPTER `multiscan.boundary(tol=0.05)`** (`multiscan.py:57`).
- Only objects with z < 0.05 are tested, so 13 of 122 hull rooms escape "proxy".
- Fix: test every floor-standing object up to 0.5 m.

**B9. ADAPTER guards for Structured3D** (`structured3d.py`).
- Drop nearest-assigned objects whose centre lies outside the room.
- Accept a door only if the topology links it to the room.

**B10. Stale IR files.**
- MultiScan predates the 09-27 tilt change (7 non-test rooms).
- Scan2CAD predates the 10-13 fix (7 of 19 kept rooms carry the old size defect, e.g. scene0043_00).
- Regenerate both.

**B11. BUILD oob handling.** Rooms with objects far outside the boundary are kept with only an `oob_objects` flag (e.g. arkit 43828265, 2.88 m out). Reject them or type them as proxy.

**Docstring errors:** the mp3d adapter says "1696 → 1600" (the IR has 1,601); the MultiScan adapter says "10,957/10,957" (true for 10,887).

The Holodeck adapter, `build.py` and `placed_size` are unchanged since before v3.3, so v3.3 carries B1, B2, B4 and B5 as well.

## 5. Effect on GT pass rates and training targets

Baseline from scale-curve-016 at s = 1: checker 6,252 / 8,647 = 0.723; downstream acceptance 2,553 = 0.295. My re-scoring reproduces 2,600 of 2,600 rows exactly.

- **Holodeck fix (B4), measured:**
  - GT acceptance +78 rooms (A4 disappears in all 78; nothing is lost): 0.295 → 0.304. Holodeck alone: 425 → 503 of 503.
  - Checker unchanged (all 503 already pass).
- **Rounding fix (B1), measured:**
  - R5 disappears in 406 (audit-labelled objects only) to 478 (every ≤1.2 cm poke) rooms out of 1,561. The R5 share drops from 18.1% to about 12.6–13.4%.
  - Acceptance: net 0 (+1 room, −1 to −3). These rooms also fail A2/R2/A4.
- **All other ADAPTER/BUILD/UNIFIED collision and >10 cm poke items, joined to the checker:**
  - 40 rooms would flip to checker pass (22 BUILD, 17 ADAPTER of which 15 are IL3D sizes, 1 UNIFIED).
  - Up to about 27 more in SAGE, MansionWorld and SceneSmith (upper bound from the audits' text, not joined).
  - Checker 0.723 → at most about 0.731.
- **Unclassified checker reason:** `fixed_blocking` (620 rooms, objects overlapping a door box) was not layer-classified by any audit. Its inputs (door poses, object boxes) were shown unchanged from the official data for the main sources. The 0.10 m door-box depth is our convention.

**Training targets** (v3.2: 144,150 training rooms; build stats and audit samples):
- **Holodeck:** 8,946 training rooms (6.2%). Every object is +10 cm in width and length. J's predicted sizes for those objects equal the padded GT sizes (median difference 0.00 m), so the model has learned the padding.
- **Rounding:** 5 mm wall-poke targets in about 50% / 47.5% / 32% of SpatialLM / Structured3D / InteriorGS training rooms (about 28,000 rooms).
- **IL3D sizes:** about 8.2% of IL3D_3dfront IR rooms (about 1,480 of the 18,054 training rooms) contain 1.5–3.3× inflated sizes. This matters directly for size targets in the unknown-size study.
- **Smaller effects:** z-snap in 0.3–1.0% of rooms; tilt inflation in about 5% of scan collision pairs; HSSD duplicates in 51 rooms.
- **No training effect:** SceneSmith and SpatialGen are test-only.
- **OFFICIAL collisions and pokes** appear in training rooms at the same rates as in test (e.g. SpatialLM genuine collisions 27.0% train vs 28.2% test).

## 6. Which study conclusions change

1. **GT checker pass 0.723: unchanged** (at most +0.8 pp). About 97% of the GT checker failures come from the released source data, not our pipeline or the collaborators.
2. **GT acceptance 0.295: about 0.304 after the Holodeck fix**, and other fixes add at most about 0.5 pp. Low GT acceptance is still driven by the downstream rules (A4 strictness, door strips, A3U's every-instance required set), not by data corruption.
3. **Rule-level tables need two artefacts relabelled:**
   - R5 at ≤1 cm is our rounding (about 400–480 rooms).
   - Holodeck's "A4 15.5%" is padding plus snap; with the fix it is 0%.
4. **Model failures, Holodeck: the comparison changes.** Removing 10 cm of width and length from J/P's own predicted boxes, the same way as for GT, gives:

| System | Holodeck acceptance | Holodeck checker pass |
|---|---|---|
| J | 125 → 214 (+89) | 175 → 248 |
| P | 298 → 364 (+66) | — |

   - Holodeck is 503 of the 1,454-room pilot cohort (34.6%). The LLM agents (e.g. O2 492/503, L 494/503) predict their own sizes, which carry no padding. So J/P-vs-LLM comparisons on that cohort are biased against J/P by up to about 6 pp.
   - J's other 289 Holodeck failures are real.
   - This assumes a model trained on unpadded targets would make the same layout.
5. **Model failures, R5:** models trained on 5 mm-poke targets probably repeat it. This is not quantified, because there are no per-object depths for model rows. Re-score R5 at 1 cm tolerance as a sensitivity check.
6. **Source-specific:** IL3D_3dfront GT checker pass +15 rooms (+1.7 pp within the source); SceneSmith collaborator pokes affect 3–4 rooms.

## 7. Uncertainty

- **What "OFFICIAL" means:** it means "present in the released boxes". Meshes are absent for most sources (no 3D-FUTURE, SceneSmith, HSSD scene_instance/semantic_config), so box overlap is not proven mesh interpenetration (e.g. the loose MultiScan table box).
- **SceneSmith:** UNIFIED is confirmed only for the implausible assets; the other old-method entries are unresolved.
- **IL3D:** my count covers a 1-in-8 sample of houses.
- **Effect estimates:** the checker-flip count for SAGE, SceneSmith and MansionWorld is an upper bound. The nudge counterfactual uses a whole-cm grid search within ±2 cm with no collision check.
- **Undetermined:** MultiScan's corrupt boxes could be OFFICIAL or UNIFIED.

Nothing in the repo or data was changed. Scripts and outputs are in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/fidelity/synth-verify/`:
- `holo_verify.py`, `slm_round.py` (official layouts extracted to `slm_raw/`)
- `il3d_verify.py`, `ss_pop.py`
- `cf.py` → `cf.jsonl` (Holodeck and rounding counterfactuals)
- `eff_chk.py` (checker flips)
- `holo_models.py` (J/P on Holodeck)