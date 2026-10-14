DPO2 quantification (dpo2_prompts.json, 8,410 pairs). All numbers below were computed with the scripts listed at the end. The reasoning was done in English for the script, so final user-facing text still needs to be turned into Chinese by the orchestrator.

**Bottom line:** every DPO2 rejected answer is the chosen layout with one of two hand-written perturbations, and each pair gets exactly one of them. One pushes several objects 0.3–0.4 m outward from the room centre (out-of-boundary). The other moves 1–2 objects up to 2 m onto a neighbour and often turns them by a multiple of 90° (overlap). This matches paper §3.4, App. C and Fig. 11, which describe two kinds of noise: out-of-boundary and object-overlap. Neither perturbation is Gaussian.

**Structure (hd = HOLODECK, 3d = 3D-FRONT)**
- **Sources:** hd 4,597 pairs (1,156 rooms); 3d 3,813 pairs (984 rooms, 970 distinct chosen layouts). That is 2,140 rooms, close to the paper's 1,100 + 1,100.
- **Pairs per room:** pair1–4 directories each hold about 2,100 pairs. 2,025 rooms have 4 pairs, 87 have 3, 21 have 2 and 7 have 1. All pairs of a room share an identical chosen layout.
- **Prompts:** uid and instruction are identical in all 8,410 pairs, and there is only one instruction text (4,485 chars). Object names, order and counts in chosen and rejected are identical in all pairs. z never changes.
- **Rejected input differs in 813 pairs, all hd.** The only difference is w and d swapped on 1–6 objects (691 pairs have one swapped object). Of the 965 swapped objects, 846 were also turned ±90° in the rejected output and 118 were not turned. Training never sees this, because utils.py builds the prompt from `combined_prompts[0]` only.
- **No-op pairs:** 12 hd pairs have chosen == rejected exactly, so they carry no gradient.
- **Input/output mismatch:** in 183 hd pairs, chosen and rejected both list objects that don't match the input (e.g. 3 armchairs in the input, 2 in the output).

**Objects changed per pair**
- **hd:** 0:12, 1:2,374, 2:577, 3:518, 4:395, 5:297, 6:185, 7:121, 8:72, 9:28, 10:11, 11:5, 12:2. Mean 2.42.
- **3d:** 1:2,163, 2:676, 3:446, 4:275, 5:157, 6:66, 7:20, 8:6, 9:4. Mean 1.93.
- **Rotation:** an object is never turned without also being moved. Pairs containing any rotation: hd 1,453/4,597 (31.6%), 3d 1,059/3,813 (27.8%).

**The two noise types.** Pairs are split at a move of 0.45 m; only 43 hd and 41 3d pairs mix both kinds.

*Out-of-boundary pairs* (hd 1,983, 3d 1,576)
- **Size:** each moved object goes 0.30–0.40 m, close to uniform. Mean 0.349, std 0.031 (a uniform 0.3–0.4 would give 0.029); 5th–95th percentile 0.304–0.396.
- **Direction:** straight out from the room centre. The angle between the move and the object's position vector has a median of 0.4–0.5°, and 98% are within 5°.
- **How many objects:** median 4 per pair in hd (60% of objects), median 3 in 3d (30%).
- **Which objects:** mostly ones already against the edge. In hd, 93% of moved objects sit within 5 cm of the edge of the chosen layout's extent, versus 65% of objects left in place. In 3d the figures are 43% versus 19%.
- **Rotation:** almost none (hd 85 of 7,992 moved objects; 3d 51 of 4,693).

*Overlap pairs* (hd 2,559, 3d 2,196)
- **How many objects:** usually one. hd 1:2,179, 2:294, 3:82, 4:4; 3d 1:1,848, 2:321, 3:24, 4:2, 6:1.
- **Size:** median move 1.48 m in hd and 1.47 m in 3d (5th–95th percentile 0.6–2.3). Each of x and y is capped at ±2.00 m and is roughly flat across that range; 26 values sit exactly at ±2.00.
- **Direction:** inward and toward a neighbour. Distance to the nearest other object's centre drops from a median of 1.20 to 0.75 m in hd, and from 0.96 to 0.64 m in 3d.
- **Rotation:** about half of the moved objects are also turned (hd 1,516 of 3,086; 3d 1,079 of 2,624). The turn is always ±90° or 180°, in roughly equal thirds (hd +90:524, −90:530, 180:547). Rejected angles are rewritten in the 0 to 2π range.
- **Angle cue in 3d:** chosen 3d angles stay within ±π, but 322 of 3,813 rejected 3d answers contain an angle above π. That is a surface clue the model could learn instead of geometry.

**Gaussian fit.** A single Gaussian does not fit. Forcing one onto all non-zero x/y changes gives std 0.62 (hd) / 0.69 (3d), a robust (MAD-based) sigma of 0.42 / 0.45, and excess kurtosis of 2.29 / 1.39. The shape is a mix of a narrow 0.3–0.4 m outward push and a wide move capped at ±2 m.

**Do the perturbations target overlap or out-of-boundary?**
- **Proxies used:**
  - **Footprint:** a rectangle of w along the object's own x and d along its own y, rotated by its angle. This gives almost no overlap in hd chosen layouts (mean overlap rate 0.0005). The 3d chosen layouts overlap under every convention I tried (mean overlap rate 0.066).
  - **Overlap rate:** the paper's formula, using these rotated rectangles.
  - **Out-of-room:** area outside the chosen layout's extent, taken as a box centred on the origin. A square of side √area turned out to be useless: 99% of chosen hd layouts already stick out of it.
  - **Baseline:** the same move size in a random direction, 5 draws per pair.
- **Overlap pairs:**
  - **Overlap goes up:** hd 95.8% of pairs versus 49.0% for the baseline; 3d 80.2% versus 45.2%.
  - **Goes outside the extent:** only hd 24.7% (baseline 65.4%) and 3d 15.6% (baseline 44.9%).
  - **Overlapping object pairs:** hd mean 0.06 → 1.32.
- **Out-of-boundary pairs:**
  - **Goes outside the extent:** hd 96.0% (baseline 90.2%), 3d 86.4% (baseline 64.4%).
  - **Overlap goes up:** only 16% (baseline about 70%). In 3d it falls in 59.7% of pairs, because pushing objects apart removes overlaps already present in the 3d layouts.
- **Neither proxy rises:** hd 1.5% of pairs, 3d 12.7%. The higher 3d share comes from the 3d bounding-box problems and existing overlaps.

**Chosen vs SFT ground truth**
- **Same uid, exact match:** chosen equals the SFT output for the same uid in 5,854 of 8,410 pairs (hd 3,515, 3d 2,339). The input text is identical to SFT's in all of those.
- **Not in SFT:** 2,548 pairs (hd 1,074 / 269 rooms; 3d 1,474 / 377 rooms) have a uid absent from SFT. Their chosen layout appears nowhere in SFT, either as exact text or after parsing and rounding.
- **Wrong uid:** 8 hd pairs (uids `a_kitchen-8200` and `-8207`) carry a chosen layout that matches the SFT row of uid -8198 / -8205.
- **Against DPO1:**
  - **hd:** 4,300 hd pairs match DPO1 exactly on room, pair number, uid, input and chosen layout.
  - **3d:** 870 of 984 3d rooms' chosen layout also appears as a DPO1 chosen. Uid and directory naming differ between the two files.
  - So DPO2's positives are essentially DPO1's positives with rule-based negatives.

**Three examples** (index into dpo2_prompts.json)
1. **#2175, hd pair4, `HOLODECK-a_bathroom-2035`, 6.0 m², 4 objects; out-of-boundary.** All 4 objects pushed outward, no turn:
   - vanity (0.71, −0.45) → (1.04, −0.66)
   - toilet (0.69, 0.75) → (0.95, 1.04)
   - storage cabinet (−0.64, 0.60) → (−0.92, 0.87)
   - basket (−0.87, 1.20) → (−1.10, 1.52)
   - All moves are 0.39 m. Overlap stays 0; area outside the extent goes 0 → 1.049 m².
2. **#3007, hd pair4, `HOLODECK-a_living_room-10008`, 42 m², 7 objects; overlap.** Only the sofa (w 4.28, d 2.43) changes: (2.33, 0.15, 1.57) → (1.97, 1.22, 3.14), a 1.13 m move plus a 90° turn. Its overlap goes 0 → 0.858 m², the layout's overlap rate 0 → 0.051, and the turned sofa also puts 1.37 m² outside the extent.
3. **#1738, 3d pair3, `3DFRONT-f03e152e-…-living room`, 38.19 m², 9 objects; overlap.** Only the "blue pillow and black frame stool" moves: (0.65, 0.30) → (2.46, −1.01), 2.23 m with no turn, landing on the dining table at (2.70, −0.52). Its overlap goes 0 → 0.758 m², the layout's overlap rate 0 → 0.082.
   - **"Neither" case:** pair #0 (3d) moves a shelf 0.92 m and raises neither proxy.

Scripts and intermediate files are in /private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/optiscene-dpo/:
- dpo2_parse.py
- dpo2_struct.py
- dpo2_inputdiff.py
- dpo2_diff.py
- dpo2_mag.py
- dpo2_regime.py
- dpo2_conv.py
- dpo2_geom.py
- dpo2_geom_report.py
- dpo2_rules.py
- dpo2_sft.py
- dpo2_vs_dpo1.py
- dpo2_misc.py
- dpo2_ex3d.py
- dpo2_parsed.pkl
- dpo2_diffs.pkl
- dpo2_geom.pkl