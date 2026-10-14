**InternScenes (gen, arkit, scannet, 3rscan, mp3d), MultiScan and Scan2CAD: where the measured GT problems come from**

**Bottom line:** almost all measured problems are already in the released data. The collaborators' unified export copies the official geometry exactly for all three dataset families. Our conversion adds a small share:
- **ADAPTER:** the tilted-box enclosure in InternScenes; one MultiScan boundary rule; stale MultiScan and Scan2CAD IR files.
- **BUILD:** z-snapping onto the floor, and cm/degree rounding that pushes borderline cases over a threshold.

Several spreadsheet and README claims are wrong or incomplete; they are listed in section 4.

**Scope.** I analysed every test room of my 7 sources (1,088 rooms), not a sample:

| Source | Test rooms |
|---|---|
| InternScenes_gen | 660 |
| InternScenes_arkit | 229 |
| InternScenes_scannet | 104 |
| InternScenes_3rscan | 70 |
| InternScenes_mp3d | 18 |
| MultiScan | 6 |
| Scan2CAD | 1 |

Problem labels come from `validate.check` and from the door-strip rule (A2) as `downstream.accept` computes it. Official files exist on disk for all three families:
- **InternScenes:** `source/Layout_info/*/layout.json`, `StructureMesh/floor.glb`, and the Gen `boundary_points.json`.
- **MultiScan:** the annotated mesh and point cloud with instance ids.
- **Scan2CAD:** `full_annotations.json` and the ScanNet `axisAlignment`.

**Checks that hold everywhere:**
- Re-running `build.prep` (default arguments) on the IR files reproduces the delivered eval rooms exactly (objects, fixed boxes and boundary) for 564 of 564 test rooms. That covers all non-Gen test rooms plus the 136 Gen test rooms in a 1-in-5 Gen sample. So the "train proxy" in section 5 is faithful.
- Re-running the current InternScenes adapter on the unified records gives the delivered IR for all 1,081 rooms (no code drift).

## 1. What the collaborators' export (UNIFIED layer) changes: InternScenes
- **Objects:** the unified objects equal the official `layout.json` exactly in all 1,081 rooms (bbox kept as-is, degrees = radians converted).
- **Boundary:** the unified boundary is exactly the convex hull of `floor.glb` after the [x,−z,y] axis swap (Hausdorff distance 0). For Gen it is exactly `boundary_points.json`.
- **Floor height:** the Real2Sim floor is at z = 0 (|y| < 5e-16). The Gen floor is the slab top at 0.088–0.150 m. The adapter's median-door-bottom shift matches it (median difference 0, max 1.8 cm).
- **Rotation order (ZXY) is correct.** I tested 508 Real2Sim objects with a_x ≈ ±90° and an off-axis a_y. Under ZXY (R = Rz·Rx·Ry), 300 of them have their bottom within 3 cm of the floor. No other rotation order puts more than 132 there.
- **Convex hull:** this is the only geometric approximation the collaborators introduce. The hull exceeds the floor-mesh area by more than 5% in 140 of 421 Real2Sim test rooms. It never removes a problem, because the floor mesh lies inside the hull. Its only measured effect is 3 misplaced door strips (UNIFIED, below).

## 2. Per source (test rooms)

A "pair" below is a checker main/severe collision pair. A pair is OFFICIAL when the official oriented boxes overlap by more than 0.01 m² of true projected footprint **and** more than 5 cm in height, i.e. the checker rule holds on the official boxes; 3D separating-axis penetration is > 0 for almost all of them. A wall poke is OFFICIAL when the official footprint pokes out at least as deep as the eval box, within 1.5 cm.

| Source | Collision pairs (OFFICIAL / ADAPTER / BUILD) | Pokes > 5 cm (OFFICIAL / ADAPTER) | Door-strip hits (genuine / door box unusable / hull / evaluator) |
|---|---|---|---|
| scannet | 429 (401 / 12 / 16) | 9 (5 / 4) | 112 (59 / 48 / 3 / 2) |
| arkit | 210 (191 / 11 / 8) | 68 (62 / 6) | 113 (60 / 53 / 0 / 0) |
| 3rscan | 237 (216 / 15 / 6) | 11 (9 / 2) | 77 (52 / 23 / 0 / 2) |
| mp3d | 20 (20 / 0 / 0) | 3 (3 / 0) | 6 (0 / 6 / 0 / 0) |
| gen | 20 (19 / 0 / 1) | 7 (7 / 0) | 238 (128 / 101 / 0 / 9) |

Collision rooms with no OFFICIAL pair at all: scannet 1 (scene0087_02, rounding), arkit 6, 3rscan 0, mp3d 0, gen 1 (diningroom/7473, chair×chair 0.0079 → 0.0104 m²).

**Collisions in the Real2Sim scans.** InternScenes says it "resolves collisions", but the released boxes collide heavily:
- Pillows lying at floor level inside couch boxes. Example: mp3d S9hNv5qa7GM/region15, pillow 33 against L-couch 1: official 0.140 m², penetration 0.18 m.
- Ovens inside cabinets. Example: arkit 47895962, oven 4 × cabinet 18: 0.377 m².
- Items on a counter inside a mailbox box. Example: scannet scene0018_00, mailbox 15 × printer 12: 0.157 m² by 0.52 m, penetration 0.32 m.
- In 247 of 847 OFFICIAL pairs one object's official box bottom is more than 5 cm below the floor.

**ADAPTER collisions (all tilted objects).** The tilted box is replaced by a yaw-aligned rectangle around its 8 corners, which is larger than its true footprint. Example: 3rscan 422885e0…, rack 20 (0.52×2.86×2.52 m, tilted 48°) × bed 10. Official boxes are separated (−2.2 cm); the eval overlap is 0.205 m². Across eval objects the inflation is small (median +2%, max 1.7×), but about 20% of Real2Sim scan objects are tilted (median 18–24°).

**BUILD collisions** come from two mechanisms:
- **z-snap.** Example: arkit 47895976, oven 1 floats 0.43 m above a cabinet with no height overlap. `anchors.infer_anchors` only looks for a support within 5 cm of the object's bottom, so the oven is snapped to z = 0 into the cabinet. Another example: scannet scene0270_02, a chair sunk 6.8 cm is raised to 0 and now overlaps a book at 0.85 m.
- **Rounding** to cm and whole degrees pushes 0.0095 m² past the 0.01 m² limit.

**Wall pokes.** Pokes through the hull are also pokes through the official floor mesh, so they are OFFICIAL. Extreme cases:
- arkit 43828265: bed 3 and cabinet 4 lie entirely outside `floor.glb` (2.88 m). The official floor mesh is incomplete. BUILD keeps the room with only an oob flag.
- gen bedroom/5524: bed 10 measures 2.71×8.86 m and pokes 5.31 m out. This is a corrupt official box that the 20 m size guard misses.

ADAPTER pokes are the tilted-enclosure inflation. Example: arkit 45261407, a toy pokes 1.4 cm officially and 11.4 cm in eval.

**Door strips (A2).**
- **Genuine:** a closed door on the official floor outline, with the object in its 1 m strip in official geometry. Examples: gen bathroom/6651, sink in front of door 23; scannet scene0136_02, table 10 in front of door 9.
- **Door box unusable for a strip:** the official door box is an open leaf, or more than 0.3 m off the outline, or more than 15° to it. Over all test doors this applies to Gen 463/952, arkit 90/151, 3rscan 18/44, mp3d 12/24 and scannet 38/135. Example: gen bathroom/10159, door 40 is 0.47 m off the wall at 89.8°, so the rule computes a 0.20 m "width".
- **Hull (UNIFIED):** 3 cases in scannet. Example: scene0051_00, door 2 is 0.12 m from the floor mesh but 1.40 m from the hull.
- **Evaluator artefact:** the object's real footprint never touches the strip; the axis-aligned box in the room frame does. Six of these are Gen rooms whose frame is rotated ±45°, e.g. bedroom/12145.

**Clean rooms (3 per source).** Every object is within 6.7 mm in position and 5 mm in size of the official box, with yaw within 0.5°. Examples: scannet scene0065_00/02, scene0087_00; gen bathroom/10010, /10086, /10145.

**MultiScan (6 test rooms).** Official = mesh-sampled points with instance ids.
- **Collisions, OFFICIAL:**
  - scene_00054_00 table 19 × sofas 23 and 22: 20% and 14% of the sofa points fall inside the table's box, but no table points fall inside the sofa boxes. The table's box is loose.
  - scene_00078_00 table 10 × backpack 23 (severe): 79.5% of the backpack points are inside the table's box. The backpack is under the table.
- **Wall poke, ADAPTER:** sofa 25 in scene_00054_00 pokes 12 cm in eval; its mesh points are 9.4 cm outside the hull and there is no wall nearby.
  - Cause: `multiscan.boundary()` only tests objects with z < 0.05, and the sofa bottom is at 0.069 m.
  - Effect: the room stays "hull" instead of "proxy", which build would have rejected. This affects 13 of the 122 hull rooms overall.
- **Door strip, OFFICIAL:** banner 27 stands 0.33 m from door 16.
- **Not checker problems, but corrupt GT:**
  - In scene_00031_01, test objects obj_0004/0005/0007 (sofa.4, sofa.5, sofa.3) have boxes that contain 0% of their own mesh points; the boxes enclose unlabelled points instead. I could not tell OFFICIAL from UNIFIED here, because the official annotations file is not on disk. 11 objects in 8 scans are affected dataset-wide.
  - In scene_00054_00, sofas and chairs whose mesh starts 0.21–0.36 m above the floor are snapped to z = 0 by BUILD.

**Scan2CAD (1 test room, scene0287_00).** The official CAD boxes match the unified record exactly (0.0000 m). They poke out of the SceneCAD floor polygon:
- bed: 15.1 cm (eval 13.7 cm);
- bookshelf: 13.1 cm (eval 11.1 cm);
- table: 5.6 cm (eval 5.6 cm).

The ScanNet bed instance's own box also pokes 12.7 cm, so this is OFFICIAL: the SceneCAD layout is smaller than the scan geometry. The SceneCAD frame transform is right (SceneCAD floor vs ScanNet floor hull: IoU median 0.79 over 994 scans).

## 3. Training rooms
I ran `prep` on all non-test IR rooms (Gen 1 in 5). The same mechanisms appear at the same rates:

| Source | Non-test rooms | Rooms with collision pair | Pairs OFFICIAL / ADAPTER (tilt) / BUILD (rounding) / BUILD (z) | OFFICIAL share | Pokes > 5 cm OFFICIAL / ADAPTER |
|---|---|---|---|---|---|
| scannet | 1,408 | 1,080 | 6219 / 294 / 104 / 65 | 93.1% | 217 / 68 |
| arkit | 4,808 | 2,298 | 4958 / 247 / 150 / 48 | 91.8% | 1083 / 127 |
| 3rscan | 1,261 | 919 | 4634 / 267 / 62 / 63 | 92.2% | 153 / 53 |
| mp3d | 1,347 | 629 | 2103 / 80 / 31 / 24 | 94.0% | 141 / 63 |
| gen (sample) | 2,672 | 58 | 71 OFFICIAL / 2 BUILD (rounding) | — | 24 / 0 |

- arkit also has 86 non-test rooms with an object more than 10 cm outside the boundary.
- **Stale IR used for the delivered data:**
  - MultiScan IR predates the 09-27 tilt change: 7 kept non-test rooms differ, no test rooms.
  - Scan2CAD IR predates the 10-13 tilt fix: only 19 Scan2CAD rooms survive dedup against InternScenes_scannet, and 7 of them carry the old local-size defect (e.g. scene0043_00 chair 0.53×0.61×0.98 should be 0.72×0.70×1.12). The test room is unaffected.

## 4. Where the spreadsheet or README disagrees with the data
1. **InternScenes doors** are marked "✅ 28,987" with positions. In fact many door boxes are leaves: 41–63% of test doors are open or off the floor outline, so no wall position or width can be read from them.
2. **InternScenes boundaries:** `boundary_points.json` holds wall-line samples at slab-top height, not floor points at z = 0. The Gen floor sits at 0.09–0.15 m, which the spreadsheet does not mention.
3. **InternScenes boxes** are described as validated, but corrupt official boxes remain:
   - two Gen beds 1,910 m and 803,014 m tall;
   - an 8.86 m bed in bedroom/5524.
4. **Matterport3D:** 96 regions are duplicated in the official data, and 1,696 counts them twice. Neither the spreadsheet nor the README says so.
5. **ScanNet room type:** the kit's `scannet_scene_types.json` gives "Copy Room" where ScanNet's metadata says "Copy/Mail Room" (scene0205_00/01/02).
6. **MultiScan `obb_axes`:** the spreadsheet says the columns are the local axes. In fact the rows are: 10,957 of 10,957 rows equal the columns of the quaternion's rotation matrix. Our adapter is correct.
7. **MultiScan floor height:** "地面高度见 floor_height" is misleading. `floor_height` is the top of the floor slab; object bottoms sit a median 9.2 cm below it (p90 24.9 cm, max 2.24 m).
8. **MultiScan validation:** "31 checks pass, OBB = source, point-cloud containment" does not hold. 11 objects' boxes contain less than 90% of their own points, 8 of them 0%.
9. **Our own docstrings:** the mp3d adapter says "1696 → 1600", but the IR has 1,601 rooms. The MultiScan adapter says "up = axis 1 for 10,957/10,957", but it holds for 10,887.

## 5. Fixes for the ADAPTER and BUILD parts
1. **Tilted boxes (InternScenes and MultiScan):** fit the minimum-area rectangle of the projected corners instead of the local-X-aligned box, or drop or flag small tilted soft items.
2. **`multiscan.boundary()`:** test every floor-standing object that build will snap (non-hanging, z ≤ 0.5 m), not only those with z < 0.05.
3. **Stale IR:** regenerate the MultiScan and Scan2CAD IR with the current adapters.
4. **`anchors.infer_anchors`:** when an object is floating but something lies under its centre with a lower top, put it on that support instead of snapping it to z = 0.
5. **Rooms with objects far outside the boundary:** reject them, or type them as proxy, instead of only flagging `oob_objects`.
6. **InternScenes doors:** use only closed leaves lying on the floor outline when building door strips, or rebuild each door opening from its leaf.

Everything is in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/fidelity/internscenes/v2/`:
- `is_rooms.jsonl` — one line per InternScenes test room;
- `cols_classified.jsonl`, `pokes_classified.jsonl`, `a2_classified.jsonl` — every problem with its layer;
- `train_proxy.jsonl` — the non-test rooms;
- scripts `analyze_is.py`, `classify_cols.py`, `classify_a2.py`, `convention_full.py`, `multiscan_check.py`, `s2c_check.py`, `train_proxy.py`.