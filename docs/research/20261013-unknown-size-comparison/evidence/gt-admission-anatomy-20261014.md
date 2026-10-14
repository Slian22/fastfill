# GT anatomy under the admission port (workflow wf_a0a4bf65-652, 2026-10-14; all 8,647 test rooms, verdicts reproduce scale-curve-016 rows exactly)

## A4 collisions

**GT A4_COLLISION anatomy (8,647 test rooms)**

GT collides mostly because A4 tests boxes far more strictly than the frozen checker does. Tucked seats, rotated-box artefacts, small items on the same table and items inside shelves all count. Only about 1 in 5 GT rooms has a collision the frozen checker would also call one.

I re-ran `downstream.accept` on every GT room, scoring the GT answer on the joint request the same way `scale_curve` does. The code hashes match the `scale-curve-016/MANIFEST.json` code hashes, and the result matches `rows.jsonl` exactly for every room's reasons and acceptance. The script also asserts that my copy of A4 gives the same pairs as `accept()`.

### How A4 is defined (`fastfill/comparison/downstream.py`)
- **Room frame:** set by `frame()` at :81-96. A plain rectangle room uses theta = 0. Otherwise the frame follows the longest edge of the minimum rotated rectangle, folded into [-45°, 45°).
- **Boxes:** each object becomes the axis-aligned box of its rotated footprint in that frame (`aabb()`, :99-101). A4 runs on the layout after the R2 `snap_inside` step, which moves objects by at most 10 cm (:247-249).
- **Overlap test:** `_overlaps()` at :104-106. The overlap must exceed `TOL_M = 0.01` (:70) on both axes. There is no minimum area and height is ignored.
- **Same parent only:** :266-268, with the rule summarised in the docstring at :36-37. Two floor objects (parent None on both) are compared; a parent and its child never are. `r4_parents` (:209-225) keeps a parent link only if the two boxes overlap by a positive area. A parent that is a fixed item becomes None.
- **The only exemption is "covering":** `covering()` at :228-230 exempts an object with no parent that is under `FLOOR_COVERING_H_M = 0.1` m (:71). A rug is exempt; a plate or pen standing on a table is not.
- **Every reason blocks acceptance** (:309-314).

### How the frozen checker differs (`fastfill/validate.py`)
`check(tol=0.05, min_area=0.01, severe=0.3)` at :72 compares pairs using the true rotated footprints. It skips a pair when:
- one is the parent of the other (:86);
- their heights overlap by 5 cm or less (:88);
- one is a rug/mat or under 6 cm tall (:89-90);
- their true overlap area is 0.01 m² (100 cm²) or less (:92).

`collision_kind` (:55-69) then sorts what remains:
- **"tuck"** means a seat-type object (`SEAT`, :30) overlapping a table-type object (`TABLE`, :31), or a bed overlapping a nightstand/bedside table (:57-61). In plain terms: a chair pushed under a table, or a nightstand slightly inside the bed's box.
- **"contained"** means a smaller box inside a bigger one (:66-68).
- **"main"** is everything else; "severe" is a main overlap covering more than 30% of the smaller footprint (:96).

`geometry_success` (`comparison/geometry.py`:42-43) fails a room only on main/severe collisions. Tucked and contained pairs never fail it. A4 has no tuck exemption, and `scale_curve.py`:50 records that no tuck-exempt downstream variant was computed.

### Scale
- 4,429 of 8,647 rooms (51.2%) fail A4_COLLISION, with 25,771 flagged pairs (median 2 per failing room, mean 5.8).
- 1,335 rooms (15.4%) fail downstream on A4 alone.

### Overlap size (all 25,771 pairs)
- **Box overlap depth** (the smaller of the x/y overlaps in the room frame): p10 1.8 cm, p25 3.2, median 6.5, p75 14.5, p90 30.1.
  - Share of pairs by depth: <2 cm 12.0%, 2–5 cm 28.3%, 5–10 cm 24.6%, 10–25 cm 21.7%, ≥25 cm 13.4%.
- **True depth** (rotated footprints): median 3.5 cm, p75 12.0, p90 27.0.
  - 20.9% of pairs have true footprints that do not touch at all.
  - 0–2 cm 18.4%, 2–5 cm 18.1%, 5–10 cm 14.4%, 10–25 cm 16.8%, ≥25 cm 11.5%.
- **Overlap area as % of the smaller footprint:**
  - Box overlap: median 38.5% (p25 17.6%, p75 87.1%). It can exceed 100% because rotated objects get inflated boxes.
  - True overlap: median 8.9% (p75 41.8%). 42.4% of pairs are under 5%, and 13.6% are 90% or more (one box inside the other).
- **Worst pair per failing room:** median true depth 15.3 cm, median area 41.7%.
- **Where the pairs sit:** 47.7% are two floor objects; 52.3% are items on the same movable parent, such as things on one table or shelf. 66.4% involve an object rotated more than 0.5° from the room frame.

### What each pair is
Each pair is put in the first class that fits, in this order:

| Class | Share of pairs |
|---|---|
| Tucked (checker rule) | 16.8% (seat × table 11.6%, bed × nightstand 5.2%) |
| Box-only: a rotated object whose true footprints do not intersect | 20.5% |
| Item on a shared parent, height overlap ≤ 5 cm (pen × notebook, plate × glass) | 10.4% |
| Rug/mat or under 6 cm tall | 1.5% |
| True overlap ≤ 100 cm² | 30.3% |
| Contained (book/bottle inside a shelf or cabinet box) | 6.2% |
| Main | 8.4% |
| Severe | 5.9% |

How deep the tucked pairs go:
- **Chairs under tables:** median 17.5 cm deep, 30.8% of the chair's footprint (quartiles 8.8–28.8 cm).
- **Nightstands into the bed box:** median 10.5 cm, 19.5%.

### Failing rooms where every A4 pair is of one kind (4,429 rooms)
- **(a) Tucked:** 19.0%. Seat × table alone: 5.3%.
- **(b) Box artefact** (rotated, true footprints apart): 1.6%. Using true depth ≤ 1 cm instead: 2.3%.
- **(a) or (b) only:** 21.0%.
- **(c) Annotation noise:** true depth < 2 cm in 6.3%; < 5 cm in 18.5%.
- **(d) Genuine** (checker main/severe):
  - every pair genuine in 13.6% of failing rooms;
  - at least one genuine pair in 40.1% of failing rooms (1,778 rooms, 20.6% of all GT rooms);
  - at least one genuine pair 5 cm deep or more: 18.7% of all rooms;
  - at least one severe pair: 10.2% of all rooms.

### If some pairs were exempt (cumulative, all 8,647 rooms)

| Exempted | Pass A4 | Downstream accepted |
|---|---|---|
| Nothing (as now) | 4,218 (48.8%) | 0.295 |
| + tucked | 5,058 (58.5%) | 0.325 |
| + box artefacts (**the number you asked for**) | **5,148 (59.5%)** | **0.329** |
| + items with ≤ 5 cm height overlap | 5,277 (61.0%) | 0.339 |
| + rugs/mats / under 6 cm tall | 5,305 (61.4%) | 0.339 |
| + true overlap ≤ 100 cm² | 6,280 (72.6%) | 0.383 |
| + contained (only checker main/severe left) | 6,869 (79.4%) | 0.396 |

Swapping the area cut for a depth cut, after tucked, box-artefact and height-overlap exemptions: depth < 2 cm gives 5,496 (63.6%, accepted 0.349); depth < 5 cm gives 6,021 (69.6%, accepted 0.373).

Even with A4 as lenient as the checker, downstream acceptance only reaches 0.396. The remaining gap to the checker's 0.723 comes from R2, A2, A3 and R5.

### Most common category pairs (number of failing rooms containing them)
- **All A4 pairs:** bed × nightstand 711, dining chair × dining table 290, chair × table 209, king size bed × nightstand 145, coffee table × sofa 138, book × shelf 133, book × book 126, chair × desk 116, bottle × shelf 111, chair × chair 106.
- **Main/severe pairs only** (counted by pair): chair × chair 141, coffee table × sofa 128, bookcase × bookcase 105, sofa × sofa 74, sofa × table 63, cabinet × cabinet 56, floor standing lamp × sofa 56, bed × wardrobe 50.

### By source
Format: A4 fail rate / still failing after exempting tucked + box artefacts / rooms with at least one main/severe pair.

- **Official split (`scale_curve.SOURCE_KIND`):**
  - real, n 4,085: 55.5% / 39.3% / 29.5%
  - synthetic, n 4,562: 47.4% / 41.5% / 12.6%
- **True scans**, n 428 (InternScenes arkit/scannet/3rscan/mp3d, MultiScan, Scan2CAD): 85.7% / 77.8% / 55.8%.
  - scannet 92.3% / 91.3% / 83.7%
  - 3rscan 100% / 90.0% / 78.6%
  - arkit 82.1% / 70.7% / 38.4%
- **Designer layouts**, n 4,529 (SpatialLM, Structured3D, HSSD200, InteriorGS, IL3D_3dfront): 55.4% / 37.9% / 29.7%. 41% of their pairs are tucked.
- **Generated layouts**, n 3,690: 42.1% / 39.2% / 5.3%. Their pairs are 31% box artefacts, 39% overlap ≤ 100 cm² and 16% short items on a shared parent.
  - SAGE-10k fails 86.9%, but 42% of its pairs are box artefacts.
  - OptiScene_holodeck fails 15.5% with no main/severe pairs (all under 5 cm).
  - IL3D_synthetic fails 4.4%; MansionWorld 22.5%.

### Example rooms (A4 is the only downstream reason in each)
1. **Tucked chair — `spatiallm:scene_001842::room_2`** (SpatialLM home office, 3 objects).
   - A dining chair (obj_0001, 0.68×0.63 m) is pushed under a desk (obj_0003, 1.00×2.57 m).
   - Overlap 63.0×51.0 cm, true depth 51.0 cm, 75% of the chair's footprint.
   - Checker: tuck, so the room passes. A4: fail.
   - Also: `structured3d_02979_room_3881` has two nightstands, each 7 cm into the bed box (50×7 cm, 14% of the nightstand).
2. **Box artefact — `InternScenes_arkit::ARKitScenes:arkitscenes/Training/42898722`** (scan, room not rotated).
   - Cabinet obj_0002 (0.50×0.51 m, yaw 184°) and bed obj_0003 (2.08×1.62 m, yaw 183°).
   - Their room-frame boxes overlap by 53.4×9.5 cm, but the real boxes are 1.0 cm apart.
   - The overlap comes only from 3–4° of yaw noise in the scan. The checker passes the room.
3. **Genuine overlap — `structured3d_02659_room_578738`** (living room, 4 objects).
   - Table obj_0002 (0.60×0.60×0.42 m) and sofa obj_0003 (1.53×2.58×0.92 m), both straight.
   - Overlap 60.0×22.5 cm = 1,350 cm², 37.5% of the table's footprint. Checker: severe, so GT fails the checker too.
   - Similar cases:
     - `il3d:bce5c464-86d4-11f0-a478-60cf84ae2082::LivingRoom-515`: an L-shaped sofa × coffee table, 69.0×35.5 cm (46%). The L-sofa's box covers the inner corner where the table sits.
     - `il3d:bd5cd892-86d4-11f0-a478-60cf84ae2082::OtherRoom-9513`: two shoe cabinets overlapping 34.5 cm (39%), a real annotation error.

Everything is in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/gt-anatomy/a4/`. The per-pair data is `pairs.jsonl`, and the summary outputs are `summary.txt` and `ladder.txt`. The scripts are `a4_anatomy.py`, `summarize.py`, `ladder.py` and `explore.py`.

## R1/R2/R5 envelope

**R1, R2 and R5 on the GT test rooms (8,647 rooms, downstream.accept at s=1)**

I rebuilt each GT room the way scale_curve does (`report_v2._gt` + `output_repair.predicted_room`, then `downstream.accept`). All 8,647 reason sets match the GT rows in `scale-curve-016/rows.jsonl` exactly. GT acceptance is 0.295. Shares: R2_OUTSIDE_1MM 3,195 rooms (36.9%), R5_OUTSIDE_ENVELOPE 1,561 (18.1%), R2_CHECK 583 (6.7%), R1_KEEPOUT_NOT_REPRESENTABLE 177 (2.0%).

**1. How the rules are defined**

- **Door strip** (`downstream.py:138-161`, constants at `:75`):
  - The door is matched to the nearest boundary edge.
  - The strip runs along that wall for the door width plus `KEEPOUT_SIDE_MARGIN_M = 0.15` m on each side.
  - It reaches `KEEPOUT_DEPTH_M = 1.0` m into the room and is clipped to the polygon.
  - Every door is treated as an entry (strict variant P4).
- **keepout_floor** (`downstream.py:192-206`):
  - The floor is the polygon minus the door strips (not padded), minus small pockets. Pockets are floor pieces under `POCKET_M2 = 0.5` m² that get cut off once the strips are padded by `KEEPOUT_PAD = 0.08` m (`interface.py:65-67`).
  - **R1** fails if the padded strips, after a `WALL_SNAP = 0.01` opening, split the floor into more than one large piece or leave an island.
- **snap_inside runs first** (`downstream.py:247-249`; `interface.py:255`, target `floor.buffer(1e-3)` at `:277`, `SNAP_TOL = 0.10` at `:71`):
  - An object is moved at most 10 cm, and only if the moved footprint lies completely inside keepout_floor. That floor already excludes the door strips.
  - The move is also refused if it deepens an overlap with a fixed box, or if a child's centre would leave its parent.
  - If no move works, the object stays where it was.
- **R2_CHECK** (`downstream.py:250-252`): `validate.check` on the snapped layout, which is the frozen checker's own rule (out of bounds at `oob_tol = 0.10`, `validate.py:72,78`; plus support and ceiling). In GT all 583 failures are the 10 cm out-of-bounds part.
- **R2_OUTSIDE_1MM** (`:253-254`): every footprint, including rugs and stacked objects, must be covered by `keepout_floor.buffer(ENTRY_TOL_M)`, with `ENTRY_TOL_M = 1e-3` (`:69`, ported from EmbodiedGen `backend.py:129`).
- **R5_OUTSIDE_ENVELOPE** (`:255-256`): every footprint must be covered by `polygon.buffer(1e-3)`.
- **Consequence:** the floor is inside the polygon, so every R5 failure is also an R2_OUTSIDE failure. The R5 rooms (1,561) are exactly the R2 rooms that have a wall poke.

**2. R2_OUTSIDE_1MM breakdown (3,195 rooms, 6,303 objects)**

| Cause | Rooms |
|---|---|
| Door strip only | 1,634 |
| Through a wall only | 857 |
| Both | 704 |

- **Door-strip objects (4,259):** 3,857 (90.6%) reach more than 10 cm into the strip, 275 reach 5–10 cm, 114 reach 1–5 cm, 13 reach 1 cm or less.
  - 500 of them are floor coverings (rugs). A2 exempts rugs but R2 does not, and in 172 rooms rugs are the only strip offenders.
  - 10 objects fail only because of a pocket.
  - 2,201 of the 2,338 rooms with strip encroachment also fail A2_KEEPOUT_BLOCKED, so the same door blocking is counted twice.
- **Deepest reach per room:** more than 10 cm in 2,812 of 3,195 rooms.

**3. R5 breakdown (1,561 rooms, 2,044 objects)**

How far objects stick out through the wall:

| Depth | Objects | Rooms (by deepest object) |
|---|---|---|
| 1 cm or less | 791 (38.7%) | 555 |
| 1–5 cm | 397 | 305 |
| 5–10 cm | 144 | 115 |
| More than 10 cm | 712 (34.8%, median 21 cm) | 586 |

- **1 cm or less is our own rounding, not dataset noise.** 663 objects stick out exactly 5 mm and 56 by 7 mm (5 mm on both axes, at a corner); further clusters sit at 10 and 15 mm. That is the signature of `build.py:228-233`, which rounds positions and sizes to cm and yaw to whole degrees (`scene.py:100-107`).
- **The 10 cm repair almost never helps.** Only 2 of 2,044 R5 objects were moved by the repair:
  - 1,396 (68%) also overlap a door strip, so no move can clear both the wall and the strip.
  - Running the same repair against the bare polygon would fix 1,058 of them (52%), including 734 of the 791 that stick out 1 cm or less.
- **More than 5 cm (856 objects):**
  - 137 are bigger than the room along one axis.
  - 158 stick out through a doorway (the outside part is within 5 cm of a door box).
  - About 561 go through a plain wall. These are asset-box versus wall-line mismatches in the source.
- **Polygon simplification is not a main cause.** Infinigen (InternScenes_gen), the only source with an explicit `boundary_simplify_m = 0.01`, has 1% R5. Hull (scanned) rooms have 24 of 428 (5.6%).

R5 and strip encroachment by source (share of rooms):

| Source | R5 | R5 ≤1 cm | R5 >10 cm | Strip encroachment |
|---|---|---|---|---|
| SpatialLM | 0.39 | 0.16 | 0.13 | 0.48 |
| SpatialGen | 0.36 | – | 0.22 | – |
| Structured3D | 0.33 | 0.15 | – | 0.46 |
| HSSD200 | 0.30 | – | 0.25 | – |
| IL3D_3dfront | 0.24 | – | 0.16 | 0.26 |
| InteriorGS | 0.17 | – | – | 0.34 |
| ScanNet | – | – | – | 0.43 |
| Infinigen | – | – | – | 0.29 |
| SAGE | – | – | – | 0.22 |
| MansionWorld | 0.00 | – | – | 0.03 |
| Holodeck | 0.00 | – | – | 0.00 |
| IL3D_synthetic | 0.00 | – | – | 0.00 |

The 2,044 R5 objects are mostly wardrobes, cabinets, beds and sofas, and 2,021 of them stand on the floor.

**4. R1 (177 rooms)**

115 of these rooms have 5 or more doors. Median area is 32.9 m², and only 7% have a side of 2.3 m or less. Sources: SpatialLM 107, Structured3D 24.

**5. Would a looser tolerance help?**

Same snap as deployed; tolerance applied to both R2 and R5:

| Tolerance | R5 pass | R2_OUTSIDE pass | R1+R2+R5 pass | Overall acceptance |
|---|---|---|---|---|
| 1 mm | 7,086 (0.819) | 5,452 (0.631) | 0.625 | 2,553 (0.295) |
| 1 cm | 7,650 (0.885) | 5,487 (0.635) | 0.629 | 2,567 (0.297) |
| 5 cm | 7,943 (0.919) | 5,605 (0.648) | 0.642 | 2,584 (0.299) |
| 10 cm | 8,061 (0.932) | 5,840 (0.675) | 0.669 | 2,590 (0.300) |

Only 130 rooms (1.5%) fail on R codes alone; the rest also fail A2, A3 or A4. A looser tolerance therefore cannot lift acceptance much. R2 is driven by the door strips, not by the 1 mm.

**6. Example rooms**

- **`spatiallm:scene_000467::room_2`** (laundry room, 1.30 × 4.00 m):
  - The door is 2.10 m wide on the east wall at y=2.35, so its strip covers x 0.30–1.30, y 1.15–3.55. That is 1.0 m of the room's 1.3 m width.
  - The washing machine `obj_0001` (0.67 × 0.61, footprint y 3.335–4.005) sticks 0.5 cm through the north wall (the rounding signature) and reaches 21.5 cm into the strip (0.085 m²).
  - The repair cannot fix both, so the room fails R2_OUTSIDE, R5 and A2.
- **`structured3d_00275_room_603826`** (L-shaped bedroom):
  - Door 0.85 m wide at (0.47, 1.22), strip x 0–1.045, y 1.37–2.37.
  - The bed `obj_0001` (2.00 × 1.98, footprint y 2.05–4.03) is fully inside the walls, but its foot is 32 cm into the strip (0.171 m²), leaving 68 cm clear in front of the door.
  - Fails R2_OUTSIDE and A2, not R5.
- **`il3d:b716143a-86d4-11f0-a478-60cf84ae2082`** (3D-FRONT bedroom, 2.60 × 3.40 m):
  - The King-size Bed `obj_0001` is 2.83 m wide, wider than the room, with footprint x 0.045–2.875.
  - It sticks 27.5 cm through the east wall (0.646 m² outside).
  - Fails R2_CHECK (more than 10 cm), R2_OUTSIDE and R5. No move of 10 cm or less can fix it.

Everything is in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/gt-anatomy/r125/`. `run.py` produces `rows.jsonl` (per room: reasons, depths per failing object, pass/fail at each tolerance); the breakdowns come from `an1.py`–`an6.py` and the example details from `ex.py`.

## A2/A3 doors and reachability

**Door rules on GT rooms, all 8,647 test rooms (GT layout at s=1, same path as scale-curve-016)**

Method: I re-ran `downstream.accept` on all 8,647 GT rooms the same way `scale_curve` scores GT (`report_v2._gt` → `output_repair.predicted_room`). The per-room reason codes match `scale-curve-016/rows.jsonl` for 8,647 of 8,647 rooms. My re-implemented A2 and A3 object lists also match `accept()`'s own violation lists in every room. GT acceptance is 2,553 / 8,647 = 0.295.

## 1. How the rules are defined (`fastfill/comparison/downstream.py`)

**Constants (L69-77)**
- `TOL_M` = 0.01 (from `admit.py:139`)
- `FLOOR_COVERING_H_M` = 0.1
- `AGENT_RADIUS_M` = 0.2, `NAV_GRID_M` = 0.25 (`nav.py:40,43`)
- `MIN_REACHABLE_CELLS` = 5
- `MIN_DOORWAY_M` = 2 × 0.2 = 0.4 m
- `KEEPOUT_DEPTH_M` = 1.0, `KEEPOUT_SIDE_MARGIN_M` = 0.15 (`growth_planner.py:70-71`)
- `DETACHED_DOOR_M` = 0.3 (this one is an approximation, not a port)
- `DOOR_WORDS` = {door, doors, doorway, doorways}

**Which boxes count as doors and where the strip goes (`door_strips`, L138-161)**
- A door is any fixed box whose normalised category contains a door word (L145).
- The door is snapped to the nearest boundary edge (L148).
- Door width = the box's extent along that wall (L154).
- The strip spans the door width plus 15 cm on each side, clamped to the edge (L155-156). It is 1.0 m deep along the inward normal and clipped to the room polygon (L158).
- The navigation seed is the door's projection onto the wall, moved 0.25 m into the room (L159).
- A door more than 0.3 m from every wall is flagged "detached", but its strip is still built on the nearest edge (L160).

**"Blocking the door" = A2_KEEPOUT_BLOCKED (L288-292)**
- The strip is shrunk by `TOL_M` = 1 cm.
- A room fails if any "solid" object's axis-aligned box (in the room's main frame) overlaps the shrunk strip with any area > 0.
- "Solid" means every object except floor coverings, where a floor covering is an object with no parent and under 10 cm tall (L228-230, L266). Stacked objects count, and height is ignored.
- The strips are also cut out of the floor used by R2 (`keepout_floor`, L192-206, L243, L253-254). So an object inside a strip usually fails R2_OUTSIDE_1MM as well. snap_inside moves at most 10 cm.

**"Cannot reach" = A3 rules**

Navigation grid (L270-279):
- Grid cells are 0.25 m.
- A cell is free only if its centre lies inside the room polygon shrunk by 0.2 m and outside every parentless solid object's box grown by 0.2 m.
- Consequence: a gap between two pieces of furniture is always blocked below 40 cm, and only passes reliably at about 65 cm or more, depending on how the grid lines up.

Search (`_reach`, L171-189):
- The seed must lie inside the room's bounding box.
- The straight line from the seed to its start cell centre must not cross any grown furniture box. If it does, the door fails A3_ENTRY_BLOCKED (L297-299).
- From there, a 4-connected breadth-first search over free cells.

Rule outcomes:
- **A3_TOO_FEW_REACHABLE:** fewer than 5 cells reached (L300-301).
- **A3_DOORWAY_TOO_NARROW:** door width ≤ 0.40 m (L293-294).
- **A3_UNREACHABLE_REQUIRED** (L302-307):
  - The required counts are every requested category with its full count (L263). In practice that means every instance must be reachable, cups and books included.
  - An object counts as reached if some reached cell centre falls inside its floor-root object's box grown by 0.45 m (0.2 + 0.25).
  - The rule is checked separately from each door.
- **Strict combination (L314):** `admit_ok` requires no violation at any door, so every door has to work as an entrance.

**Comparison with the frozen checker:** door blocking there is only `fixed_blocking` (`validate.py:99-110`). An object fails only if its footprint overlaps the door box itself (about 10 cm thick) by more than 0.01 m² and their heights overlap by more than 5 cm. There is no clearance strip and no reachability test. Its out-of-room tolerance is 10 cm, against 1 mm in R2/R5.

**Why GT "collides"**, briefly (L265-268): A4 fails any pair of solid objects with the same parent whose boxes overlap by more than 1 cm on both axes. It ignores height and has no tucked-seat exemption. The checker exempts seat×table and bed×nightstand (`validate.py:55-61`) and needs both > 0.01 m² overlap and > 5 cm height overlap. Rotated objects also get enlarged boxes under A4.

## 2. GT results

**Doors per room:** 0 doors in 2,103 rooms, 1 in 4,252, 2 in 1,120, 3 or more in 1,172. That is 13,083 doors in total; 590 are detached. Door widths at q10/q25/q50/q75/q90: 78.8 / 80 / 90 / 109 / 194 cm. Doorless rooms (24.3%) never run A2 or A3.

**Rooms failing only because of doors** (counterfactual: the same `accept()` with the door boxes removed passes):

- **Count:** 1,451 rooms, 16.8% of all rooms and 22.2% of the 6,544 rooms that have a door.
- **Acceptance:** 0.295 with doors, 0.463 without.
- **Which rules:** 1,013 of these rooms involve strip codes (A2 / R2 / R1); 427 fail only A3-family rules.
- **Most common reason sets:**
  - {A3U} 385
  - {A2, R2_OUT} 383
  - {A2, R2_OUT, R5} 164
  - {A2, ENTRY, R2_OUT} 109
- 293 of them also show R5, which disappears once the door strips no longer block the 10 cm snap.

**Other ways of relaxing the rules:**

| Variant | Acceptance |
|---|---|
| Lenient P4 (`accept()`'s best-single-door variant) | 0.297 |
| Drop A3U only | 0.340 |
| Drop all A3 rules | 0.345 |
| A3U reachable from *any* door | 0.297 (+14 rooms) |

- Dropping A2 alone gives no gain, because the same objects also fail R2_OUTSIDE_1MM.
- R2_OUTSIDE_1MM fails in 3,195 rooms. In 1,634 of them (51.1%), every failing object is inside the room polygon, so it fails only because of the door strips.

**A2 (2,763 rooms = 32.0%; 42.2% of rooms with a door)**
- **Doors per A2 room:** 1 door in 1,430, 2 in 510, 3 or more in 823.
- **Blocked doors per A2 room:** 1 in 2,022 rooms; all doors blocked in 1,655 (59.9%).
- **Intrusions:** 5,601 (object × door). 71.5% overlap the door opening itself; the rest touch only the 15 cm side margins. Stacked objects are 3.1%, objects starting ≥ 1 m above the floor 0.7%, and box-only hits (rotated footprint misses the strip) 1.3%.
- **Categories (rooms):**
  - wardrobe 367 (+58 "Wardrobe")
  - cabinet 309
  - bed 306
  - sofa 294
  - dining table combination 242
  - refrigerator 234
  - shoe cabinet 186
  - chair 184
  - combination sofa 144
  - table 127
- **Depth into the 1 m strip, per intrusion** (q10/25/50/75/90): 16 / 36.5 / 77.5 / 99 / 99 cm. Per room, median of the deepest intrusion: 95.5 cm.
- **How real the blocking is:**

| Floor-standing object in front of the opening, nearest edge within … of the door wall | A2 rooms |
|---|---|
| 80 cm | 1,778 (64.4%) |
| 50 cm | 1,427 (51.6%) |
| 5 cm (object essentially in the doorway) | 772 (27.9%) |

- **Marginal cases:** 970 A2 rooms (35.1%) only touch the side margins or reach less than 20 cm into the strip.

**ENTRY_BLOCKED (964 rooms = 11.1%; 1,152 doors)**
- At every one of these doors, some furniture's grown box sits on the seed path, i.e. a box within about 45 cm of the door wall.
- None of these doors is blocked when the room is empty; 85 are detached doors.
- Blockers: sofa 181, dining table combination 125, combination sofa 114, cabinet 74, bed 73. None starts ≥ 1 m above the floor.

**TOO_FEW_REACHABLE:** 112 rooms (1.3%). **DOORWAY_TOO_NARROW:** 238 rooms (2.8%) from 257 doors ≤ 40 cm wide, 251 of them InternScenes door boxes (thin door-leaf boxes).

**A3_UNREACHABLE_REQUIRED (2,858 rooms = 33.1%)**
- **Single vs multi-door:** 1,684 single-door rooms and 1,174 multi-door. In 436 of the multi-door rooms (37.1%) the failure comes from only some doors. Yet "reachable from at least one door" fixes A3U in just 112 rooms (3.9%). The same wedged objects are unreachable from every door.
- **Volume:** 29,798 unreachable (object × door) instances; 28.3% are items on top of a table, cabinet, desk or shelf. Unreachable objects per room: median 2, q90 19 (median 27% of the room's objects).
- **Categories (rooms):**
  - nightstand 503
  - chair 419
  - cabinet 329
  - lamp 283
  - book 268
  - plant 239
  - bottle 212
  - table 201
  - shelf 198
- **Why unreachable:**
  - With all other furniture removed (the object kept), 29,729 instances (99.8%) become reachable. Only 69 (0.2%) stay unreachable in the empty room.
  - No case is caused by own-object-against-wall (0), and none by objects starting ≥ 1 m above the floor (0).
  - 37.4% of instances have no free cell within 45 cm (wedged in). 62.6% have free cells nearby, but in a pocket cut off from the door.
- **Doors are not boxed in:** a failing door still reaches a median 98.1% of the free floor (q25 82.1%), and 3,917 of 5,752 failing doors reach ≥ 90%. The failures are local gaps under 40-65 cm.
- **Free floor:** furnished free cells are a median 45% of the empty-room count. Grouped boxes such as "dining table combination" make this worse.

## 3. Where the doors come from

Every door is a source annotation. No door is inferred by the downstream rules.

| Source group | How doors are annotated | Doors / room | Doors | Detached |
|---|---|---|---|---|
| SpatialLM | `room.doors` with wall index and width (`adapters/spatiallm.py:18-21`) | 2.28 | 5,612 | 0 |
| Structured3D | structure doors | 2.28 | 1,704 | 5 |
| MansionWorld | structure doors | 1.70 | 2,233 | 0 |
| IL3D_3dfront | doors, 448 of them taken from the neighbouring room's list when they open into this room (`il3d_3dfront.py:44`) | 1.27 | 1,104 | 7 |
| InteriorGS | openings matched to room edges (`interiorgs.py:29-37`) | 1.61 | 502 | 1 |
| SAGE-10k | layout doors (`sage10k.py:9`) | 1.03 | 506 | 3 |
| HSSD200 | architectural instances (`hssd200.py:21`) | 0.20 | 28 | 0 |
| InternScenes gen / arkit / scannet / 3rscan / mp3d | annotated door objects; the four real-scan sets also use convex-hull boundaries | — | 952 / 151 / 135 / 44 / 24 | 395 / 95 / 47 / 20 / 11 |

- The InternScenes strips are therefore approximate.
- The rule matching also picks up IL3D_synthetic "shower_door" boxes as room doors (5 boxes). They are shower enclosures, but cause no failures.
- Doorless sources: OptiScene_holodeck 503, SceneSmith 364, IL3D_synthetic 311/316, HSSD200 115/138. Door rules never fire on them.

**Failing only because of doors, by source** (acceptance with → without doors):

| Source | Door-only failures | Acceptance |
|---|---|---|
| SpatialLM | 579 / 2,461 | 0.171 → 0.406 |
| MansionWorld | 282 / 1,313 | 0.561 → 0.775 |
| Structured3D | 234 / 747 | 0.169 → 0.482 |
| InternScenes_gen | 147 / 660 | 0.120 → 0.342 |
| IL3D_3dfront | 76 / 872 | 0.178 → 0.265 |
| InteriorGS | 52 / 311 | 0.344 → 0.511 |
| SAGE-10k | 48 / 489 | 0.033 → 0.131 |

## 4. Examples

**`structured3d_03013_room_271` (bedroom, 3.05 × 2.60 m; fails A2 + ENTRY_BLOCKED + R2_OUT)**
- 80 cm door on the x=0 wall at y 0.19-0.99.
- `obj_0002` is a cabinet, 0.6 × 2.59 × 2.65 m, with box x 0-0.6, y 0.01-2.59. It covers the whole wall, door included. It reaches 99 cm into the strip and overlaps the opening by 0.48 m², 25 cm from the seed. This is real blocking: a source defect or a closet door.
- `obj_0003` (bed) also counts. It reaches only 13 cm into the strip (87 cm from the wall), overlapping the opening by 0.064 m².

**`spatiallm:scene_007725::room_3` (bathroom, 1.6 × 2.2 m; fails only A2 + R2_OUT, so a door-only failure)**
- 80 cm door at x 0.1-0.9 on the y=0 wall.
- The toilet's box is x 0.90-1.60, y 0.90-1.35. It touches only the side margin: 14 cm sideways (x 0.90-1.04) and 10.5 cm into the strip, 89.5 cm from the door wall. Overlap is 0.013 m², with no overlap of the opening itself.
- The frozen checker passes this room.

**`spatiallm:scene_004721::room_3` (bedroom, 5.61 × 4.23 m; fails only A3U from its single door)**
- The door's search reaches 137 of 141 free cells.
- `obj_0003` (nightstand, box x 0.85-1.50, y 3.67-4.16) is unreachable. Only 2 free cells lie within 45 cm of it, and they are cut off from the door. It sits 7.5 cm from the wall and 11.5 cm from the bed.
- Its only access is a 48 cm gap to `obj_0004`, a 1.14 × 1.63 m "leisure table and chair combination" group box. That gap is wider than 40 cm but misses the 25 cm grid alignment. The bed-to-table-group gap is 19 cm.
- Removing either the bed or the table group fixes it. A person could walk through a 48 cm gap.

Files are in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/gt-anatomy/doors/`:
- `run.py`
- `analyze.py`
- `detail.py`
- `gt2.jsonl` (per-room records)
- `analysis2.txt`
- `ex1.txt`

## Port vs infinite-task-world backend

**Port vs EmbodiedGen backend: is our downstream check stricter than the real one?**

Rule for rule, the port is faithful. Its door and tolerance deviations together move GT acceptance only from **0.295 to 0.326**. The one large difference is the "required" set. EmbodiedGen checks reachability (A3U) and collisions (A4) only for the classes the task names, one instance each. The port treats every requested category as required, at full count. If nothing is required, GT acceptance rises to **0.642**. Even under the most lenient reading of the backend, 34% of GT rooms are still refused, mostly for doors blocked at every door, objects through walls, and blocked non-entry door strips.

I checked this against `/Users/slian/Desktop/3D/Worldedge/EmbodiedGen` (fork main `c71f7ce`; the `world/` files last changed in `d361bb8`, 2026-09-27). Every line the port cites still matches, except `backend.py:1896`, which is now `:1898`. Two layers make up "downstream":
- **FastFill's own endpoint** (`fastfill/serve.py` + `interface.py`): rules R1 and R2. This is our code.
- **EmbodiedGen**: `assert_attachable` (R5) and `admit.validate` (A0–A7).

### Rules that match (verified)
- **1 cm slack (`TOL_M`):** `downstream.py:70,104-106` = `admit.py:139,245-252`. A2 shrinks the strip by `TOL_M` and tests for area > 0 (`downstream.py:289-290`); on axis-aligned rectangles this equals `_overlaps` (`admit.py:511-512`).
- **1 mm to the walls (`ENTRY_TOL_M`):** `downstream.py:69,255-256` = `backend.py:129,283,336-343`. The backend's `_shift_ban` puts the box around all objects inside the envelope ±1 mm and raises `BackendError` otherwise. On rectangles this is the same test, object by object.
- **AABB of the rotated box, not the OBB:** used for A2/A3/A4/R4. `downstream.py:99-101` = `backend.footprint_xy` `:229-244`. Rotated objects are inflated on both sides.
- **Floor covering** (no parent and height < 0.10 m, at any z): `downstream.py:71,228-230` = `backend.py:142,247-254`.
- **R4** (parent kept only on positive-area overlap): `downstream.py:109-111,209-225` = `backend.py:413-416,1798-1803`.
- **A4:**
  - Both sides compare same-parent pairs only, ignore height and exempt coverings.
  - **Neither side has a tucked-chair exemption:** `downstream.py:266-268` = `admit.py:686-702`.
  - A grep of `embodied_gen/world` and `demo` finds no tuck handling. The tuck exemption exists only in our frozen checker (`validate.py:55-61`).
- **A2:** every non-covering object counts, stacked and wall-hung ones included, and height is ignored. `downstream.py:288-291` = `admit.py:507-520`. The backend's docstring (`:491-505`) calls this "knowingly conservative".
- **Strip shape:** door width ± 0.15 m, 1.0 m deep, clamped to the wall. `downstream.py:75,153-158` = `growth_planner.py:70-71,197-228`.
- **Navigation:**
  - Agent radius 0.2 m and grid 0.25 m.
  - The seed is one cell inside the door midpoint, the seed-to-cell segment is tested, and the search is a 4-connected BFS.
  - Thresholds: at least 5 cells, door wider than 0.40 m, touch distance 0.45 m to the floor root.
  - Port `downstream.py:72-74,159,171-189,270-307` = `nav.py:40,43,81-136` and `admit.py:131,135,256-297,543-617`.
- **Snapping:** EmbodiedGen never repairs. `ff_to_ours` reads positions as given (`backend.py:1757-1765`), and `_shift_ban` refuses rather than moves (`:262-271`). The ≤10 cm snap is FastFill's own (`interface.py:71,255-325`), and the port applies it before the checks, as deployment does (`downstream.py:247-249`).

### Where the port is stricter
Acceptance figures are cumulative on all 8,647 GT rooms. Each step adds one change; my re-run reproduces the `rows.jsonl` verdicts exactly.

| # | Port | Backend | Acceptance |
|---|---|---|---|
| S1 | Every door is an entry: A2 + A3 from each door (`downstream.py:288-307,314`) | One entry keepout (`growth_planner.py:543`) and one BFS seed (`admit.py:569-572`). Other doors get only A6, which ignores stacked objects and rugs (`admit.py:828-829`) | 0.295 → **0.297** (+12 rooms: 8 InternScenes_gen narrow second doors, 4 MansionWorld offices). A random entry door gives 0.296 |
| S2 | Every door strip is 1.0 m deep (`downstream.py:75,158`) | Planned future doors 0.6 m (`growth_planner.py:89,344-355`); only adopted doors 1.0 m (`:381-389`). Affects A6 and the R2 keep-outs | **0.303** (+56) |
| S3 | `A3_DOORWAY_TOO_NARROW` uses the GT door box width (`downstream.py:154,293`) | Uses the portal width echoed back (`backend.py:2035-2039`; `DOOR_WIDTH_M = 0.9`, `config.py:102`). Cannot fire in deployment | **0.306** (+22) |
| S4a | Every instance of every category must be reachable (`downstream.py:263,306`) | `EntityReq.count = 1` (`spec.py:126`; built with the default count in `canonical.py:326-338`), so one instance per required class suffices (`admit.py:601-609`) | **0.321** (+133) |
| S8 | R2 (our `serve.py`): every footprint, rugs and stacked objects included, must lie in the floor minus the keep-outs ±1 mm. Floor pockets under 0.5 m² are also forbidden (`interface.py:65-66,124-136,384-385`) | EmbodiedGen has only A2 (1 cm slack, rugs exempt) and A6 | Dropping R2_OUTSIDE_1MM: **0.326** (+47) |
| S4b | Required = every requested category | Required = the task-named classes only (`admit.py:326-340`). A4 compares a pair only if one side is required (`:660,691`); A4_FLOATING (`:662-683`) and A3U likewise | With required = {}: **0.642** (+2,906, mostly A4). With R2_OUT also dropped: **0.656** |

More places where the port is stricter:
- **S5:** the backend caps planned-door strips at 35% of the floor (`growth_planner.py:82,391-399`). It also sizes rooms to 2.5× the required footprint plus the strips (`:77,180-194,498-516`). GT keeps every door at its real size, so 177 rooms fail R1, 115 of them with 5 or more doors.
- **S6:** 590 detached doors still get a strip on the nearest wall (`downstream.py:76,160`). Backend doors are always on a wall.
- **S7:** non-rectangular rooms are checked against the true polygon (`downstream.py:255,276-278`). The backend only ever builds rectangles (`backend.py:1898`).
- **S9:** the ceiling check uses the room's own height instead of 3.0 m. This has no GT effect: all 583 R2_CHECK failures are out-of-bounds.
- **S10:** for a class with no instances the port also emits A3U (not filtered at `downstream.py:309`); the backend skips it (`admit.py:599-600`). It always comes with A0, so acceptance is unchanged.

**What the realistic backend number depends on:** it lies between 0.321 and 0.642, depending on which classes a task names. `admit.py:639-656` says scoping A4 to required classes is a known flaw and that widening it to all pairs is a pending decision. The port's choice matches that wider rule.

**What still fails under the most lenient variant (V6):** A2 at every door in 1,773 rooms, R5 1,517, A6 978, R2_CHECK 583, A3_ENTRY_BLOCKED 353, R1 81, A3_TOO_FEW 80.

### Where the port is looser
- **L1 – doorless rooms skip A2 and A3 entirely** (`downstream.py:288` loops over doors). That is 2,103 rooms (24.3%). Their acceptance is 0.552, against 0.213 for rooms with doors. In the backend every room has an entry with a 1.0 m keep-out and A3 (`growth_planner.py:518-545`).
- **L2:** the cm-rounding refusal in `clean_boundary` is not ported (`interface.py:139-143`).
- **L3:** relations and goals are not checked (`admit.py:351-379`, plus `serve.py`'s hard constraints). GT has none.
- **L4:** A1, A5 and A7 are not computed (world-side). A5's 1 cm envelope test is already covered by the 1 mm `_shift_ban`.
- **L5:** R2_CHECK tests the original polygon, whereas `serve.py` tests the boundary with the padded keep-outs cut out. R2_OUTSIDE_1MM covers this, so there is no net effect.

### What "not accepted" means in practice (`chain.grow`)
- **R1/R2 failures:** `serve.py` answers HTTP 422 (`:52-58`), which becomes a `BackendError` (`backend.py:1957-1962`).
- **R5 failures:** `_shift_ban` raises `BackendError` (`backend.py:285-297`).
- **Any `BackendError`:** the room is refused as BACKEND_REFUSED, with no resize and no repair (`chain.py:291-303`).
- **Admission failures:**
  - If every code is in {A0_REQUIRED_MISSING, A4_COLLISION, A6_RESERVED_BLOCKED, A2_*}, the same intent is re-planned 1.0 m larger on the free axis (`2 × GRID_QUANT_M`), up to 8 m. The backend is then called again (`chain.py:83-91,305-326`; `config.py:73,78`).
  - Any A3 code is never resized, so the room is refused (`chain.py:327-336`).
- **After a refusal:**
  - The world is left byte-identical, and an unadmitted room cannot be committed (`admit.py:980-983`).
  - The door (frontier) is sealed after 3 consecutive refusals, or after 2 refusals with the same task, room type and codes (`chain.py:60-73,364-387`).
  - FastFill responses are cached by sha256 at temperature 0 (`backend.py:1935-1946`). A retry of the same request therefore returns the same layout, gets the same refusal, and the door is sealed on the second identical refusal.

Offline GT evaluation has none of this: rooms keep their real size, so there is no resize second chance.

**Context from the backend's own docstrings:** these rules rarely fire in its own setting, which is planner-sized rooms with short object lists.
- A2 tripped in 3 of about 700 rooms (`admit.py:502-505`).
- A4 is now 1–2 per 20-run sweep (`:649-651`).
- A3U was 91 of 105 refusals over 20 no-LLM grid runs, before the placer checked reachability (`nav.py:22-24`).

### Examples
1. **`spatiallm:scene_002926::room_4`** (bedroom, 2.40 × 3.50 m, 2 objects).
   - The carpet `obj_0001` (0.46 × 1.80 × 0.01 m, no parent) lies 100 × 16 cm (1,600 cm²) inside the strip of the 0.90 m door `fixed_0001`.
   - The port refuses it on R2_OUTSIDE_1MM alone.
   - EmbodiedGen exempts floor coverings (`admit.py:508-509,829`) and would admit it. Only our `serve.py` rule rejects it.
2. **`spatiallm:scene_003178::room_5`** (bathroom, 1.71 × 1.62 m, 1 object).
   - The toilet's footprint is x 0.38–0.74, y 0.00–0.66, flat against the south wall. It reaches 0.5 cm (36 × 0.5 cm, 18 cm²) into the strip of the 0.75 m door, which starts at y 0.655.
   - The snap cannot move it because the wall is behind it, so the port fails it on R2_OUTSIDE_1MM.
   - EmbodiedGen's A2 ignores overlaps of 1 cm or less and would pass it.
3. **`spatiallm:scene_004580::room_3`** (study, 5.50 × 2.58 m, notched polygon, 2 bookcases, 1 door).
   - `obj_0002` (box x 3.50–5.00, y 0.005–0.455) is reachable.
   - `obj_0001` (x 5.00–5.40, y 0–0.34, in a wall notch) is not:
     - The grid row in front of it (y 0.625) lies inside `obj_0002`'s box grown by 0.2 m (y ≤ 0.655).
     - The next row (y 0.875) is 53.5 cm from its front, beyond the 45 cm touch distance.
   - The port refuses on A3_UNREACHABLE_REQUIRED. The backend would pass it: with bookcase required at count 1, one of two is reachable; if the task does not name bookcase, it is not checked at all.

Files are in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/gt-anatomy/port-vs-backend/`:
- `variants3.py` – per room, door and strip depth, which rules fire
- `summarize3.py`
- `detail.py`
- `gt_v3.jsonl` – one record per room
- `summary3.txt`

`variants.py`, `variants2.py` and `test.jsonl` are superseded.
