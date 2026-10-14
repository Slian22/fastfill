DPO stage 1 is not built the way the paper describes. All 3,626 3D-FRONT pairs have a "rejected" that is a byte-identical copy of "chosen", so they carry no training signal. Only the 4,325 Holodeck pairs actually differ. Their rejected layouts look like a differently-encoded layout source, not samples from the SFT model run on the chosen prompt.

The file has 7,951 pairs covering 1,989 rooms (Holodeck 1,082 rooms / 4,325 pairs, 3D-FRONT 907 rooms / 3,626 pairs). Each room appears exactly in pair1–pair4, except 5 rooms that have only 3 pairs. The paper says about 2,200 positives (1,100 + 1,100, §3.4 and Appendix C). Training (`utils.py:25-27`) uses only `combined_prompts[0]`'s instruction + input as the prompt, so each rejected record's own input is thrown away.

**How I computed this.** Every pair parsed with no failures. I loaded each JSON file whole (about 140 MB) rather than stream-parsing it. Chosen and rejected objects were matched by (object name, instance index). Footprints use w along x and d along y at angle 0, rotated by the angle; the chairs around the table in a 3D-FRONT example are consistent with that convention. The overlap rate (OOR) is total pairwise intersection area divided by total footprint area. "Paper formula" is the axis-aligned version, which ignores rotation, as in Appendix D.2; "rotation-aware" uses rotated rectangles. A pair counts as overlapping above 0.01 m². Token counts use the Qwen2.5-7B-Instruct tokenizer.

| Metric | Holodeck | 3D-FRONT |
|---|---|---|
| Instruction identical | 100% | 100% |
| Input text identical | 0% | 100% |
| Room type / area identical | 100% / 99.9% | 100% / 100% |
| Object names, same order | 99.8% (8 pairs differ) | 100% |
| Per-instance box in rejected input | h changed in 28,555 of 28,619 instances (16,187 h only; 12,368 h plus w/d swapped); 20 w/d swap only; 44 identical | all 34,375 identical |
| Output text identical | 0% | 100% |
| Instances per layout, median (mean) | 6 (6.57) | 9 (9.48) |
| Instances moved more than 1 cm | 99.6%; every instance moved in 98.0% of pairs | 0 |
| Displacement per instance, median / p90 / max | 3.61 / 6.72 / 11.45 m (n=28,371) | 0 / 0 / 0 |
| Per-pair median displacement, median | 3.42 m | 0 |
| Rotation changed | 77.0% of instances (π/2: 13,080; π: 8,763; none: 6,528) | 0% |
| OOR, paper formula: chosen → rejected, mean (median) | 0.017 (0) → 0.404 (0.377) | 0.086 both |
| OOR, rotation-aware: chosen → rejected, mean (median) | 0.000 (0) → 0.381 (0.355) | 0.065 both |
| Layouts with at least one overlapping pair, chosen → rejected | 4.8% → 99.5% | 86.7% both |
| Rejected has higher rotation-aware OOR than chosen | 99.6% | 0% |
| Layout extent centre distance from origin, median, chosen → rejected | 0.055 → 3.20 m | 0.448 m both |
| Layout extent ÷ room area, median, chosen → rejected | 1.012 → 0.400 | 0.943 both |
| Rejected footprints outside chosen's extent rectangle | 99.7% of instances, 99.8% of layouts | 0 |
| All objects share one y value (layouts with 3+ instances), chosen / rejected | 0% / 98.2% | 0% / 0% |
| Angle values used | chosen {1.57, 3.14, 0, 4.71}; rejected {4.71, 6.28, 3.14, 1.57}, with 0.0 only 25 times | identical; ±1.57 and ±3.14, 90.6% multiples of π/2 |
| Decimals in output numbers / z values | 2 decimals / all 0, both sides | same |
| Output tokens, median, chosen / rejected | 479 / 479; difference median −2, max +93 | 601 / 601 |
| Prompt tokens, median; prompt + response over 3,200 | 1,433; 0 | 1,516; 0 |

- **Outputs never restate sizes.** Each output object has only the keys object, coordinates and rotate.
- **Out-of-room check.** The prompt states the room centroid is (0,0,0), and Holodeck chosen layouts are centred and span the room. So I used the chosen layout's footprint extent as the room rectangle. The obvious alternative, a square of side √area, is useless: 98.6% of Holodeck and 95.4% of 3D-FRONT chosen layouts already exceed it by more than 5 cm, because rooms aren't square.
- **Data defect.** Eight 3D-FRONT pairs (2 rooms) have room area 0.

**Where the rejected samples come from**
- **3D-FRONT.** Rejected is a verbatim copy of chosen. The only difference is the uid, which gets a `#` marker before `.json`: `#` for pair1, `###` for pair2, `####` for pair3, `#####` for pair4. Since chosen equals rejected, the DPO loss is a constant log 2 with zero gradient. That is 45.6% of the DPO1 pairs.
- **Holodeck, rejected input.** The rejected "input" differs from the chosen prompt, and it also differs between the 4 rejected samples of the same room. That points to records re-serialized from a separate scene file, not the SFT model prompted with the chosen input.
- **Holodeck, the "h" field.** Rejected "h" ranges from −7.10 to 6.70, with 51.2% negative and 3.3% zero; chosen h is between 0.01 and 2.21. It holds a coordinate, not a height.
- **Holodeck, constant y.** Rejected y is one constant per layout, always negative, from a small set: −2.5 (1,127 layouts), −1.5 (855), −3.0 (689), −1.75 (560), −2.0 (501), −3.5 (429).
- **These patterns never appear in the SFT data.** Across the 12,253 SFT targets: 0 layouts with all-equal y, 0 inputs with h ≤ 0, 0 angles of 6.28.
- **Not a rigid transform of the chosen layout.** The pairwise inter-object distances match in 0 of 4,317 pairs. Correlation with chosen distances is 0.20 (median) when reading rejected (x, h) as floor-plan coordinates, and 0.13 for (x, y).
- **Possible swapped axis (hypothesis).** Reading (x, h) as the floor plan drops the rotation-aware OOR from 0.381 to 0.043, which suggests y and "h" are swapped. But the extent is still 1.8× the room area. This is a hypothesis, not verified.
- **"pairN" means candidate slots 1–4.** Chosen is identical across a room's 4 pairs in 1,989/1,989 rooms. Holodeck rejected outputs are all distinct in 1,070 rooms, 3 distinct in 12, and 3D-FRONT has 1 distinct (the copy).
- **"no_bbf1" is unexplained.** Neither the repo nor the paper mentions it, and I could not infer its meaning from the data.

**Room reuse across DPO1, DPO2 and SFT.** I normalised uids by stripping `#` and the `.json-<type>` suffix.

| Set | Unique rooms | 3D-FRONT | Holodeck |
|---|---|---|---|
| DPO1 | 1,989 | 907 | 1,082 |
| DPO2 | 2,134 | 978 | 1,156 |
| SFT | 11,957 (from 12,253 rows; 282 uids duplicated) | 4,005 | 7,952 |

| Overlap | By uid | Of which 3D-FRONT | Of which Holodeck |
|---|---|---|---|
| DPO1 ∩ DPO2 | 1,952 (98% of DPO1) | 870 | 1,082 (all) |
| DPO1 ∩ SFT | 1,517 | 630 | 887 |
| DPO2 ∩ SFT | 1,488 | — | — |
| All three | 1,488 | — | — |

- **Exact text matches.** Of DPO1's 1,977 unique chosen inputs, 1,940 also appear in DPO2 and 1,492 in SFT. Chosen outputs are shared with DPO2 in 1,939 cases and with SFT in 1,492.
- **Positives are SFT targets.** In 75.5% of DPO1 pairs, the chosen output equals the SFT ground truth for the same uid.
- **Caveat on 3D-FRONT uids.** The uid is scene + room type, which is not unique per room, so the text-based counts are the stricter measure.

**Training-side consequences**
- **Holodeck rejected inputs never reach training.** Only `combined_prompts[0]` is used as the prompt.
- **Train/eval split.** `dpo_train.py` uses `train_test_split(0.2, seed=42)`, which I reproduced with `datasets` 5.0.1. That gives 6,360 training pairs (3,437 Holodeck / 2,923 3D-FRONT) and 1,591 eval pairs. All 1,591 eval pairs' rooms also appear in training. Only the 3,437 Holodeck training pairs carry a gradient.
- **Code vs. paper settings.** The README points to `dataset/dpo1_prompts.json`, but the file is `dpo_prompts.json`. The README uses lr 5e-6 for 10 epochs on Qwen2.5-7B; the paper says 5e-7, 5 epochs, Qwen3-8B.

**Three example pairs**
1. **HOLODECK a_kitchen-5505, pair4** (area 20.0, 3 objects).
   - Chosen:
     - cabinet (0.00, 2.22), angle 0
     - table (0.00, −1.97), angle 3.14
     - trolley (−1.73, 1.25), angle 4.71
   - Rejected (all at y = −2.5):
     - cabinet (−1.25, −2.5), angle 3.14; h 2.0 → 4.44
     - table (−4.02, −2.5), angle 1.57; box (1.02, 1.98, 1.16) → (0.5, 1.16, 1.98)
     - trolley (−4.23, −2.5), angle 1.57; h 1.05 → 3.0
   - Displacements 4.88 / 4.05 / 4.51 m; rotations changed by π / π/2 / π.
   - Rotation-aware OOR 0 → 0.104; maximum overhang past √20/2 = 2.24 m goes from 0.32 m to 2.36 m. Tokens 258 / 258.
   - The other three rejected samples for this room use h values (−1.0, 3.5, 4.46), (0.0, 0.0, 3.5) and (−3.44, 3.0, −1.5).
2. **HOLODECK a_bedroom-4194, pair1** (area 20.0, 6 instances).
   - Chosen: bed (0.97, −0.25), angle 1.57; nightstands (1.75, −1.5) and (1.75, 1.0); wardrobe (−1.68, −1.75), angle 4.71; dresser (−1.25, 2.17); chair (0.00, 2.02).
   - Rejected, all at y = −2.5:
     - bed x −1.03, angle 4.71, h −0.5
     - both nightstands at the same point (−0.25, −2.5), h −3.0
     - wardrobe x −3.25, w/d swapped, h 4.36
     - dresser x −3.67, w/d swapped, h 2.5
     - chair x −2.0, h 4.04
   - Displacements 3.01 / 2.24 / 4.03 / 1.74 / 5.26 / 4.94 m.
   - Rotation-aware OOR 0 → 0.298; overlapping pairs 0 → 5; extent ÷ area 1.00 → 0.33. Tokens 442 / 440.
3. **3DFRONT 6c294141-0ec8-47cd-b1cb-102d60dc252c, living room, pair1** (area 16.77, 8 instances).
   - Instruction, input and output are byte-identical.
   - The uids differ only by the marker: `…living room.json-living room` vs `…living room#.json-living room`.
   - Rotation-aware OOR 0.057 on both sides; tokens 513 / 513. The chosen output equals the SFT target for this uid.

Everything is in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/optiscene-dpo/dpo1/`:
- `analyze_dpo1.py`
- `summarize.py`
- `holo_probe.py`
- `probe2.py`
- `probe3.py`
- `inspect_room.py`
- `pairs.jsonl`
- `summary.json`
- `summary.txt`
- `paper.txt`