The problems in these four sources mostly come from the official data or from the downstream rules, not from our conversion. The collaborators' processing caused one real problem, a size error in SceneSmith. Our adapter adds a small, marginal share, and build adds a few through rounding.

Headline by source:
- **OptiScene_holodeck:** every problem comes from the official 10 cm box padding. The fix belongs in our adapter.
- **MansionWorld:** the collisions are in the official data, and the door blocks come only from the rule's 15 cm side margins. Our adapter already fixes the four collaborator bugs it has.
- **SAGE-10k:** about 86% of the collisions are official. The rest come from our tilted-object boxes (adapter) and cm rounding (build).
- **SceneSmith:** some problems are official, but most large wall pokes come from asset sizes the collaborators measured without the asset's scale. This is the main collaborator defect.

## Scope and method
- **Labels:** I re-ran `validate.check` and `downstream.accept` on all 2,669 test rooms of these sources: MansionWorld 1,313, SAGE-10k 489, SceneSmith 364, OptiScene_holodeck 503. The reason sets match `gt-anatomy/r125/rows.jsonl` for 2,669 of 2,669 rooms.
- **Coverage:** I traced every problem room, not a sample, plus 3 clean rooms per source. Each room was followed from the test record through the IR line (sidecar `raw_line`), the collaborators' unified JSON and, where it exists, the official file. Footprints were recomputed from the raw data.
- **Adapter inputs:** root is `/Volumes/harddisk/3D_Room_Collections`.
  - `imChuling__3D_Room_Collections/MansionWorld_exported/json_room.json`, plus `data/MansionWorld/mansionworld/<bldg>/floor_*.json`
  - `liantian__3D_Room_Scene_Collections/SAGE10k/json_single.json` and `json_multi_split.json`, plus `sage10k/scenes/*/<id>.json`
  - `SceneSmith_exported/json_single.json` and `multiroom/multiroom_split.json`
  - `OptiScene_exported/holodeck.json`
- **Adapter matches the frozen IR:** the current adapter code, run on these inputs, gives exactly the frozen IR for all 373 traced rooms. All 2,669 IR lines match the sidecar hashes. So "ADAPTER" below means the current code.
- **Official files on disk:**
  - MansionWorld: official floor JSONs, plus the Objathor annotations with mesh bounding boxes.
  - SAGE: the official layout JSONs and code (`sage10k/kits`).
  - Holodeck: the official scene JSONs (`OptiScene/scenes.tar.gz`), plus the Objathor annotations, which cover 66,855 of 66,855 Holodeck objects.
  - SceneSmith: only the official poses (`house.dmd.yaml`). The official asset meshes (the 1.13 TB tar) are absent, so SceneSmith sizes and room boundaries are judged against the collaborators' `geometry_metadata` only.
- **Position convention:** in this asset system the stored position is the centre of the object's box. On MansionWorld floor objects, footprint centre equals position for 4,045 of 4,045 off-centre assets (p90 0.0 cm). The alternative, that position is the asset origin, misses by a median 4.2 cm.

## OptiScene_holodeck (503 rooms; no doors, no genuine collisions)
- **Wall pokes, OFFICIAL:** 2,901 of 3,362 objects poke 1–5 cm through the wall, in 503 of 503 rooms (max 5.0 cm).
  - The released `bbox` is the Objathor mesh plus Holodeck's 10 cm size buffer in width and depth. This holds for 66,855 of 66,855 objects (+9.5 to +10.5 cm each); height is exact.
  - With the true mesh sizes, 0 objects poke.
  - Example: in `optiscene:HOLODECK-bathroom-101`, `bathtub-0` has bbox [71, 83, 183] against a mesh of 0.726 × 1.727 m. The padded box reaches 4.67 cm past the wall; the mesh is 0.5 cm inside it.
  - Our position error is under 1 cm for every object, so this is not ours.
- **A4 failures, an evaluation artefact:** 78 rooms, 88 pairs.
  - None of the 88 pairs overlap in the GT as stored.
  - The R2 snap step pushes the padded boxes 4–5 cm inward, which closes 0.5–3.5 cm gaps between neighbours.
  - The true meshes are 10–13.5 cm apart. Example: `HOLODECK-bathroom-1159`, `toilet-0` × `storage_cabinet-0` (GT gap 0.5 cm, mesh gap 10.5 cm).
- **Fix (ADAPTER), `optiscene_holodeck.py:25`:** use width `b[1]/100 - 0.10` and length `b[2]/100 - 0.10`, the same correction the MansionWorld adapter makes. This removes all 2,901 pokes and all 88 pairs.
- **Train rooms:** affected too. In the IR, 57,798 of 66,855 objects poke more than 1 cm, in 9,977 of 9,977 rooms. In the v3.1 dev rooms, 497 of 497 do.

## MansionWorld (1,313 rooms)
- **Fidelity is good:** I traced 2,284 objects in 76 rooms.
  - Positions are within 0.75 cm, and GT footprints match the raw Objathor boxes (IoU median 0.984).
  - Room height equals the raw `wall_height` in 76 of 76 rooms.
  - Door centres are within 0.18 cm of the raw `doorSegment`, and widths within 4 mm.
- **Genuine collisions:** 31 rooms, 38 pairs, all small objects on furniture (lamp × notebook, book stacks).
  - 35 pairs are OFFICIAL: the raw boxes overlap by the same amount.
  - Example: `.../residential_family_apartment_6f_100_fp001#1/F1_secondary_bedroom`, `notebook-0|study_desk-0` × `desk_lamp-0|study_desk-0`. GT 239 cm² and 16.0 cm deep; raw 256 cm², 16.2 cm deep, 6.5 cm height overlap.
  - 3 pairs are BUILD and marginal: the raw overlap is 99 cm² or 4.8 cm high, just under the checker's cut, and cm rounding pushes it over.
  - Rooms: 28 only official, 1 mixed, 2 build only.
- **Wall pokes over 1 cm, OFFICIAL:** 2 objects.
  - `countertop_sink-0|double_vanity_cabinet-0` in `residential_duplex_townhouse_2f_200_fp001#2/F2_shared_bathroom`: 7.0 cm (raw 7.4).
  - `paper ream-1|printer_cabinet-0`: 4.5 cm (raw 4.8).
- **Door blocks:** 42 rooms, 56 blocking objects. None of the 56 overlaps the official `doorBoxes`, the generator's own 1 m clearance zone; the unpadded and padded footprints give the same result.
  - All 56 enter only the A2 strip's 15 cm side margins (1.5–10 cm in).
  - Example: `.../residential_family_apartment_6f_100_fp001#0/F1_secondary_bedroom`, the `study_desk` is 7.5 cm into the margin of `door|4|F1_hub_room|F1_secondary_bedroom` and 0 cm² inside the door box.
  - This comes from the evaluation rule, not from any data layer.
- **Train rooms:** same mechanisms. In the v3.1 dev rooms, 22 of 1,153 (1.9%) have genuine collisions, 24 of 25 of them between items on the same furniture. In test it is 31 of 1,313 (2.4%).

## SAGE-10k (489 rooms)
- **Fidelity:** the unified `json_single` equals the official layout exactly, over 9,142 objects (maximum difference 0.0).
  - The boundary equals the official room position and dimensions (168 of 170 identical; 2 differ by 0.026 m² or less).
  - Ceiling 2.7 m equals GT height in 170 of 170 rooms.
  - Door centres are within 0.5 cm of the official ones.
  - Untilted objects match (footprint IoU median 0.966).
- **Genuine collisions:** 63 rooms, 84 pairs.
  - **72 OFFICIAL:** raw 3D boxes overlap by more than 100 cm² with more than 5 cm height overlap. Example: `layout_2bdcd8a2::0`, `room_777d68af_phone_7cd2310f` × `room_777d68af_lamp_2bd52fe2`: raw 386 cm², 21.9 cm deep.
  - **8 ADAPTER:** these come from `sage10k.upright_box` (line 75). It replaces a tilted object with a box aligned to the object's facing direction, which is larger than the true footprint.
    - Example: `layout_a0910029::0`, `wineglass_platform6b` (45° tilt) × `bottle_plat6` (40° tilt): raw overlap 17 cm², GT 168 cm².
    - One case comes from `rest()` raising the box in height (`layout_14c1b39f`, notebook × plant: raw height overlap −0.2 cm, GT 6 cm).
  - **4 BUILD:** cm rounding of untilted objects. Example: `layout_b2a3661d`, `candletable` × `vase3`: raw 82 cm², GT 102 cm².
  - Only 8 rooms (1.6%) would lose all their genuine collisions without our conversion.
- **Wall pokes over 1 cm:** 15 objects, all tilted small items on furniture. 8 are OFFICIAL (the raw tilted box itself crosses the wall), 7 are ADAPTER (raw 0).
  - Over 5 cm: `layout_d9e3d8bd` `key`, raw 6.3 cm, GT 11.3 (official, plus 5 cm from the adapter).
  - `layout_a0910029` `candlepd`, raw 8.9 cm, GT 10.4 (official).
  - `layout_8c225f8d` `plantwardrobe` (55° tilt), raw 0, GT 8.2 (adapter).
  - The two rooms over 10 cm out of bounds are these first two.
- **Door blocks:** 103 rooms.
  - 90 rooms: side margins only, an evaluation artefact.
  - 5 rooms: OFFICIAL. 7 objects stand inside the 1 m clearance zone, 0.50–0.95 m from the door, e.g. `layout_ff61d862` `blocks` with 409 cm² inside.
  - 8 rooms: BUILD. The "door" is a generated door-prop furniture object standing 11–33 cm off the wall, e.g. `layout_45e3962a` `room_5561de90_door_7690d30b`. Build's name-based structure detection turns it into a door opening.
- **Train rooms:** same mix. In the v3.1 dev rooms, 66 of 503 (13.1%) have genuine collisions, against 12.9% in test, and 16 rooms have pokes over 1 cm.

## SceneSmith (364 rooms, eval-only: no train rooms)
- **Fidelity:** export positions equal the official poses exactly. The GT boundary equals the collaborators' floor-plan rectangle exactly. Untilted GT boxes match the official pose plus the collaborators' asset box (IoU median 0.964).
- **Collaborator size defect:** 1,792 asset entries (4,277 objects, 22%, in 52 of 364 rooms) were measured by the older extraction path. That path ignores the asset file's `<scale>` (metadata method "glTF POSITION accessor AABB…").
  - Three such assets are taller than their room. `backbar_shelving_unit` in `scene_112` measures 2.40 × 1.04 × 6.41 m in a 3.2 m room, with six units spaced 0.9 m apart, so it is about 2.7× too large.
  - Against walls, furniture sized this way pokes more than 2 cm in 16 of 31 cases (10 by more than 10 cm). Correctly scaled furniture does so in 0 of 72.
  - The official meshes are not on disk, so this is strongly supported but not fully confirmed.
- **Genuine collisions:** 75 rooms, 134 pairs.
  - **100 OFFICIAL:** e.g. `scene_120` `task_chair` × `lcd_monitor`, six pairs of 0.06–0.075 m² with 10.7 cm height overlap.
  - **24 UNIFIED (probable):** 12 rooms, including 9 backbar × backbar pairs of 0.6–1.55 m².
  - **6 ADAPTER:** tilted small items, e.g. `scene_064` spatula at 63°: raw 63 cm², GT 100 cm².
  - **4 BUILD:** rounding, e.g. `scene_013` floor_lamp × side_table, raw 89 cm² to GT 105 cm². One is a support error: `scene_203::workout_area` spray_bottle × weightlifting_platform. The bottle rests on the platform in raw (1.1 cm height overlap) but is left unparented on the floor in GT (6 cm).
- **Wall pokes over 1 cm:** 32 objects.
  - **19 UNIFIED (probable):** backbar 24–54 cm, refrigerated display case 48 cm ×2, banquette 37 cm, `scene_077` bookshelf 27 cm, and others.
  - **10 OFFICIAL (pose and size look right, not mesh-confirmed):** 7 tilted shoes in `scene_168` (raw 4.8–16.9 cm), floor scale 7.9, dumbbell rack 9.6, supply cabinet 3.2.
  - **3 ADAPTER:** e.g. `scene_048::bathroom` towel ladder, raw 0, GT 1.5 cm.
  - Rooms with a poke over 5 cm: 4 collaborator (`scene_077`, `093`, `102`, `112`) and 3 official (`scene_168`, `204::exam_room_2`, `209::gym`).
- **No doors** in this source.

## Where the spreadsheet or README disagrees with the data
**MansionWorld:**
1. "Right-handed, Z-up, (x, −z, y)": this map keeps the handedness of the left-handed Unity coordinates, so the export is a mirror image of the scene. It is internally consistent (the yaw turns counter-clockwise); only shapes with a handedness, like an L-sofa, are mirrored.
2. "Polygon and objects share one translation": false for wall objects. In 816 of 816 wall objects across 150 non-rectangular rooms they were recentred on the vertex mean (median 1.38 m off, max 3.78 m).
3. Wall-object sizes had the 0.10 m padding subtracted even though wall footprints are unpadded; width + length come out 0.12–0.20 m short.
4. "Rotation {0, 0, yaw}" drops the raw x/z flips that 7.4% of small objects have (1,412 of 19,148 sampled).
5. `roomType` "corridor" is the open hub of every floor (368 of 368 sampled), whatever its function.

Our adapter already corrects items 2–5 (it does not mirror the scene back).

**SAGE-10k:**
1. "rotation.z 0° faces +X": wrong. The front is local +Y: it faces into the room for 93.0% of 22,850 wall-side objects; local +X does so for only 3.1%. Our adapter uses +Y.
2. "x/y rotations are non-semantic noise": 9.2% of objects tilt more than 2°, including objects lying on their side, and this changes their footprint and height.
3. "No windows" is true, but the README does not mention the 168 window-type and 72 door-type furniture objects (sampled); the door-type ones cause the 8 build door issues above.
4. Not in the docs: the rendered walls stand 5 cm inside the boundary (`kits/tex_utils_local.py:580-592`). No GT floor object enters that band (0 of 11,753), so it is harmless.

**SceneSmith:**
1. "910 windows with location": all 910 sit at the room-centre placeholder at z = 0, so they carry no position. Our adapter drops them.
2. "All 19,360 sizes include the mesh scale": 4,277 do not (see above).
3. "Centres inside the boundary" is true, but 32 boxes cross the wall by more than 1 cm.

**OptiScene_holodeck:**
1. "size = bbox / 100" leaves out the +10 cm padding in width and depth; the official README does not mention it either.
2. "rotation.z = source rotation.y, horizontal yaw": the sign is wrong in the exported frame. Read that way, 99.2% of 8,245 wall-side objects at ±90° would face the wall. Our adapter uses −rotation.y.
3. "0 objects out of bounds" holds for centres only; 86.5% of boxes cross the wall by 1–5 cm.
4. "Handedness not declared": the source is Unity, which is left-handed, and the [x, z, y] position mapping is correct.

## Fixes
1. **Holodeck padding (ADAPTER):** shown above. This is the only large fix, and it also cleans the training targets.
2. **Tilted-object boxes, SAGE and SceneSmith (ADAPTER, `sage10k.py:55-90`):** use the minimum-area rectangle of the tilted object's projected footprint instead of the box aligned to its facing direction. Expected effect: removes about 14 marginal pairs and 10 pokes. Cost: tilted objects lose their front direction.
3. **Door props (BUILD, `anchors.fixed_geometry`):** stop treating SAGE furniture named "door" as door openings.
4. **cm rounding (BUILD, `build.py:230`):** leave as is; it only pushes 11 marginal pairs over the checker's threshold.
5. **SceneSmith sizes (UNIFIED):** the collaborators should re-measure the 1,792 asset entries with the scale-aware extraction; that needs the official tar. Until then, our adapter could flag the 52 affected rooms. Only test ground truth is affected.

Everything is in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/fidelity/gen4/`. The scripts are `labels.py`, `holodeck.py`, `mansion.py`, `sage.py`, `scenesmith.py` and `regen.py`. The outputs are `labels.jsonl`, the four `*_trace.jsonl` files and `sheet.txt`.