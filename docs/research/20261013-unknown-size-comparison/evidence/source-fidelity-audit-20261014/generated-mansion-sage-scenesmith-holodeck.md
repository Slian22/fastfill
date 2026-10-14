## Fidelity audit: IL3D_3dfront, IL3D_synthetic, SpatialGen, HSSD200

**Bottom line:** almost every problem in these four sources is already in the official released data. Our conversion adds three small things:
- **Adapter, IL3D sizes:** the adapter trusts IL3D `assets.json` sizes, which are wrong for 165 assets.
- **Adapter, HSSD duplicates:** the adapter keeps some HSSD articulated objects that duplicate furniture already in the scene.
- **Build rounding:** cm rounding creates a few ≤1 cm wall pokes and a few marginal collisions.

The collaborators' unified files have several real errors, but none reaches the evaluated rooms, because our adapters ignore those fields and rebuild them from the original files.

**How it was checked.**
- **Which data was evaluated:** all 1,371 GT rows of these sources (HSSD200 138, IL3D_3dfront 872, IL3D_synthetic 316, SpatialGen 45) match `.release/v3.2/ir/*.jsonl` by `raw_sha256`. That data was made with the v3.2 adapters. The current `il3d_3dfront.py` and `spatialgen.py` were changed afterwards (commits 4208f1f0b and 444b2a3f1), but only for window width and tilted-box bounds; no traced problem depends on either.
- **Problem labels:**
  - Genuine collisions come from `validate.check` (main or severe).
  - Wall pokes are raw depth outside the polygon (from the earlier anatomy run's `r125`).
  - Door blocks (A2) come from the earlier anatomy run's `doors/gt2`.
- **Every problem room was traced, not a sample of 12.**
  - Each object was recomputed from the official files with independent code and compared with the GT room.
  - Clean-room checks: SpatialGen 3, IL3D_synthetic 3, HSSD200 4, IL3D_3dfront 40 rooms. Every object matched the official footprint within 0.03 m² (one IL3D room 0.065 m²), i.e. cm rounding only.
- **Build check:** every problem was also recomputed on the pre-build IR (before rounding).

### Verdicts by source

**SpatialGen** (official `room_layout.json` present; test-only, so no training rooms)
- **Official, all of them:**
  - Collisions: 12 pairs in 10 rooms. Official overlap matches GT within 0.007 m². Example: scene_00006 coffee table #17 × sofa #16, 0.935 vs 0.9375 m² official.
  - Wall pokes over 5 cm: 12 objects in 11 rooms, matching within 0.6 cm. Examples: scene_00008 wardrobe #23 sticks out 1.55 m and its centre is outside the floor; scene_00006 cabinet #14 sticks out 0.94 m; scene_00019 two cabinets 0.61 and 0.76 m, in a floor outline that crosses itself in the official data.
  - Door blocks: 24 hits in 16 rooms. Every door box ("screen door, screen") sits in a wall (≤0.15 m from it), and every blocker matches official within 0.04 m².
- **Official label errors:** three bedrooms have the bed labelled "sofa":
  - scene_00023: 2.41 × 1.34 × 0.92 m.
  - scene_00029: 2.53 × 3.01 × 2.5 m, filling a 3.02 × 3.01 m room.
  - scene_00031: 2.41 × 3.01 × 2.54 m.
- **Size convention confirmed:** size × transform column length is correct.
  - Floor objects end up on the floor in 143 cases this way, against 12 using size alone.
  - Furniture ends up flush with a wall in 127 cases, against 45.
- **Build:** 9 pokes of ≤1 cm in 5 rooms are pure rounding (zero in the IR). Those 5 rooms fail the 1 mm wall rule (R5) only because of our rounding.
- **Adapter and unified layers:** no errors found.

**IL3D_synthetic** (official = IL3D `layout.zip`, meshes = HSSD `.glb`)
- **Official:** all 15 collision pairs in 14 rooms reproduce from the official layout plus mesh bounds within 0.004 m². Most come from IL3D's generator swapping in a different mesh for a slot:
  - "laundry_hamper-0" slot holds a wardrobe mesh;
  - "bath_caddy-0" holds a 1.46 m bathtub;
  - "twin_bed-0" holds a dog-bed mesh (eval category dog_bed) that overlaps an armchair (0.333 m², 41%).
- No wall pokes and no door blocks.
- **Unified errors, corrected by the adapter:**
  - Size ignores the layout scale for 7,289 of 47,837 objects (wrong by more than 10 cm for 6,628; median error 0.44 m).
  - "position = bottom centre" is false for HSSD meshes: off by more than 2 cm horizontally for 1,039 objects and vertically for 3,123.

**HSSD200** (the official hssd-hab `scene_instance` and `semantic_config` files are not on disk; the official object, stage and URDF meshes are)
- **Collisions, 167 pairs in 49 rooms:**
  - Official: 146 reproduce from the placed official meshes. Examples: dishwasher × kitchen lower cabinet in 103997562:3 (89%); clusters of plants in 103997562:10.
  - Adapter: 17 pairs in 7 rooms involve 7 articulated objects whose boxes our adapter builds from the URDF. Five of these boxes are at least 72% covered by ordinary (rigid) boxes, i.e. leftover duplicates the adapter's duplicate test missed. Examples: "YOO OH Kitchen 250, AIR" (0.6 × 2.38 × 2.4 m) covers the 2 Door Base cabinets by 87–94% in 107733912:8; "YOO OH Kitchen" is 98% covered in 108294465:0.
  - Build: 4 pairs are marginal and created by rounding. Example: in 105515211:0 the bed and chaise touch exactly in the IR (gap 0.0) and overlap 1 cm × 1.51 m = 0.0151 m² after rounding.
- **Wall pokes over 5 cm, 65 objects in 35 rooms, all official** (64 match the placed mesh within 2 cm):
  - 44 objects cross a wall in the official stage mesh, i.e. they really go through the wall. Most are plants or foliage; there is also a bathtub (0.146 m) and a bed (0.131 m).
  - 21 objects do not cross any wall: the room outline there does not follow walls. For 10, the outside part lies in a neighbouring open-plan room; the others are outdoor rugs and doormats.
  - Across the 138 rooms, a median 85% of each outline lies on a wall. 12 rooms are under 50% (outdoor, terrace, balcony), and 7 of those 12 have pokes.
- **Door blocks, 24 hits in 10 rooms:** door and blocker boxes match the placed meshes (≤0.017 and ≤0.032 m²). One blocker is an adapter-built articulated fridge (108294465:0).

**IL3D_3dfront** (official 3D-FRONT house JSONs present; the 3D-FUTURE meshes are not, so sizes come from IL3D `assets.json`)
- **Room split is correct:** every object in every problem is an official child of the same 3D-FRONT room as our IR room.
- **Collisions, 874 pairs in 378 rooms:**
  - Official: 852. Official overlap matches within 0.02 m² for 871 of 874. Examples: Bookcase × Bookcase 103 pairs; two King-size beds 1.90 m² in bd71353a.
  - "Corner/Side Table" × bed accounts for 32 pairs. These are nightstands under a different official label, so the checker's bed–nightstand exemption misses them.
  - Adapter: 14 pairs disappear once a wrong IL3D asset size is corrected (see below).
  - Build: 8 pairs are marginal rounding.
- **Wall pokes over 5 cm, 190 objects in 159 rooms:**
  - All reproduce from official placement against the official floor mesh, and 188 cross the official wall meshes.
  - 3 objects are official children of a room they do not stand in, e.g. b7a210e8 KidsRoom-2837 wardrobe, 3.25 m out, centred in LivingRoom-1353.
  - 14 disappear with the corrected asset size (adapter).
- **Pokes ≤1 cm:** 24 of 37 are build rounding (≤0.5 cm in the official data). 10 rooms fail R5 only because of rounding.
- **Door blocks, 554 hits in 269 rooms, official geometry:**
  - Door boxes match the official door meshes (median overlap ratio 0.97).
  - 242 hits are on doors that 3D-FRONT lists under the neighbouring room, which our adapter attaches to this room. 240 of them sit on this room's wall, so the attachment is geometrically right. 126 rooms are blocked only through such doors.
- **Adapter defect (the only one with real impact):**
  - IL3D's `assets.json` sizes contradict both IL3D's own layout bbox and the 3D-FRONT size for 165 assets and 2,141 placements.
  - For all 35 problem objects checked, the 3D-FRONT size equals the layout bbox, never `assets.json`.
  - Example: Dining Chair 6914bc9b is 2.0 × 1.94 × 2.0 m in `assets.json` but 0.48 × 0.50 × 0.75 m in the layout bbox. In b72fe1d0 this produces 3 collisions and a 0.65 m wall poke.
  - Using the bbox size removes 14 collision pairs and 14 pokes, fully clearing collisions in 5 rooms and pokes in 12.
  - The problem is in 59 test rooms and in 1,780 of 21,832 IR rooms (8.2%).

### Where the spreadsheet or README disagrees with the data
- **IL3D (imChuling):**
  - "No multi-room layouts" is false: 1,300 layouts merge 2–10 3D-FRONT rooms. Our adapter splits them into 2,858 rooms; in 1,772 of them the room id differs from IL3D's.
  - "Boundary = polygon from the floor mesh": it is IL3D's 4-point rectangle. It differs from the real floor area by more than 5% in 288 of 605 traced test rooms (more than 25% in 167), and belongs to another room in 50.
  - "Size = bbox [x, z, y]": length and height are swapped for whole-number scales (95% of a 19,195-object sample is wrong by more than 2 cm), and scale is ignored otherwise.
  - "Position = bottom centre": false for HSSD meshes (see IL3D_synthetic above).
- **3D-FRONT export (BillLin66):** size length and height are swapped for about two-thirds of placements, because the official `size` field is [x, z, y]. In a sample, height equals depth for 11,942 of 18,050. Our adapter does not use this field.
- **HSSD (XXXpilar):**
  - All 2,956 door and window boxes estimated from metadata have centres off the mesh: a median 0.85 m too high, and 13 shifted more than 5 cm along the wall (up to 0.93 m). Sizes are correct. Our adapter re-centres them.
  - "−Z front" is the Habitat world convention, not the object front: 1,755 of 1,781 beds, wardrobes and similar objects standing against a wall have their −Z side at the wall, so they face +Z. Our adapter uses +Z.
  - "Region polygons are official": cannot be checked here because the `semantic_config` files are absent.
  - The claim that articulated objects' translation is their centre of mass is contradicted by the adapter's own documented check; I did not re-run it.
  - The counts (door 469/15, window 2,487/14, pure-yaw 43,972, empty category 20) are correct.
- **SpatialGen (BillLin66):** every claim checks out (counts, units, centre position, size convention, room height, scene_00019's self-crossing floor).

### Proposed fixes
1. **IL3D size (`il3d_3dfront.placed_size`):** when the `assets.json` size differs from the layout bbox by more than 10 cm and the bbox agrees with the 3D-FRONT size, use the bbox (or flag the room).
2. **HSSD duplicates (`hssd200`):** drop an articulated box, or show it as fixed context, when at least 70% of it is covered by ordinary boxes that overlap it in height. In the IR this affects 79 boxes in 51 rooms.
3. **Rounding (`build.py:228-233`):** round so that a footprint inside the room stays inside, e.g. round sizes down at walls or nudge positions inward, and recheck the 0.01 m² collision threshold after rounding.

### Training rooms
Training rooms go through the same adapter and build code, so the same official problems are present at similar rates. From the v3.2 build stats:
- IL3D_3dfront: 3,120 of 14,449 kept rooms are flagged for objects out of bounds (21.6%; test 16.3%), and 856 for overlapping furniture. The `assets.json` size problem is in 8.2% of IR rooms.
- HSSD200: 532 of 1,575 rooms are flagged out of bounds; the leftover duplicate boxes are in 51 IR rooms.
- IL3D_synthetic: only 3 rooms flagged out of bounds.
- SpatialGen: test only, no training rooms.

### Limits of this check
- "Official" here means the overlap exists between boxes of the official placements. I did not test whether the actual meshes interpenetrate.
- For IL3D_3dfront, the 3D-FUTURE meshes are not on disk, so the assumption that each mesh's origin is at its bottom centre rests on the adapter's earlier check.

Everything is in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/fidelity/il3d_sg_hssd_b/`. The existing `il3d_sg_hssd/` folder belongs to another run and was not touched.
- **Labels:** `labels.jsonl`
- **Build layer:** `build_<source>.jsonl`
- **IL3D_3dfront:** `il3df_trace.jsonl`, `il3d_bad_assets.json`, `il3df_badfix.json`
- **HSSD200:** `hssd_trace.jsonl`, `hssd_regwall.json`
- **Scripts:** `sg_*.py`, `il3ds_*.py`, `il3df_*.py`, `hssd_*.py`, `clean_check.py`