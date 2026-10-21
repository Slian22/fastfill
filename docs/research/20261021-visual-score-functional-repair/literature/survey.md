# Survey: visual layout-quality score (Q1) and functional companion rules for collision repair (Q2)

Date 2026-10-21. All numbers come from the per-paper notes in this directory; items marked unverified could not be confirmed from the paper text. Our setting: furniture list with fixed sizes, room polygon, no free-text description; perspective + top renders.

## Q1. Evidence on evaluation protocols

| Protocol | Judge / metric | Views | Single/pair | Human agreement | Source note |
|---|---|---|---|---|---|
| LLMCompare (pros/cons then pick) | GPT-4o | rendered images (views unspecified) | pairwise | 77.1% (no pros/cons 70.0%) | procedural-scene-programs |
| VQAScore / DSG on same task | CLIP-FlanT5 / QA graph | - | - | 58.6% / 50.7% | procedural-scene-programs |
| Composite pairwise judge | Gemini 3 Flash | top-down (above) + side (below), A left B right, randomised, 3-vote majority | pairwise | tau-b user-LLM 0.50-0.56 vs user-user 0.53-0.56 | r3l |
| Plausibility 1-4 | Gemini 3.1 Pro | unspecified | single | judge top pick = human top pick on 72% (chance 33%); human kappa 0.26 | polylayout |
| GPT-4o 0-10 x3 | GPT-4o | unspecified | single | none reported; scores 8-9.4 for all | optiscene |
| GPT-4o top+side | GPT-4o | top-down + side | scores | tau users-GPT-4o 0.46-0.61 vs within-user 0.50-0.57 | layoutvlm |
| GPT-4o ranking top-down vs first-person | GPT-4o | 1 top-down / multi first-person | 3-way rank | tau 0.27 / 0.39; CLIP 0.06, BLIP 0.07, VQAScore 0.13 | vipscene |
| Gemini-2.5-Pro vs symbolic | VLM / SceneCritic | top x2, left, right, front | pairwise | VLM 58.82% easy / 47.06% complex; symbolic 94.44% / 83.33% | scenecritic |
| VLM-as-judge for constraints | GPT-4.1 / Gemini 2.5 Pro / CLIPScore | - | per-scene verdict | kappa 0.05 / 0.05 / 0.02 (tool-grounded LEGO-Eval 0.63) | lego-eval |

Key findings
1. Prompt-alignment metrics (CLIP, BLIP, VQAScore, DSG) are at or near chance for layout quality: 0.06-0.13 tau (VIPScene), 50.7-58.6% agreement (PSP), kappa 0.02 (LEGO-Eval). Reason: the caption names objects/scene type, which every candidate satisfies.
2. MLLM judges work when (a) the comparison is pairwise/forced-choice, (b) the model writes pros and cons first (+7.1 points, 70.0 -> 77.1%), (c) several views are given (0.27 -> 0.39 tau for first-person multi-view), (d) order is randomised and votes aggregated.
3. MLLM judges are weak at geometry: view sensitivity, left/right mirror asymmetry, re-scoring variance 12-19 points, rankings reverse by view, overlap errors (SceneCritic); direct constraint judgments over-accept (35.7% FP, SceneTeract). Symbolic checks beat them on overlap (100%) and orientation.
4. Realistic ceiling: judge-human agreement is about equal to human-human (tau 0.5-0.6; 72% top-pick; human kappa 0.26-0.72). Single absolute scores saturate (OptiScene 8-9.4) and have no reported validation.
5. All rubrics we found include a description/instruction-following criterion; we must remove it.

## Q2. Functional relation rules in the literature

- SceneCritic: orientation pass <= 75 deg, soft fail <= 150 deg; ontology mining: back-to-wall 45 deg, faces-centre 60 deg, faces-pair 60 deg within 5 m; proximity pass if co-occurrence >= 0.7 and d_min <= 2.0 m, or 0.2-0.7 and d_min <= 3.5 m; overlap fail > 0.2 m; scale outside p5/p95 by 0.05 m soft, factor 2.0 hard.
- SceneEval: next_to 0-0.5 m, near 0.5-1.5 m, across_from 1.5-4 m, far > 4 m; face_to score 1.0 at 0 deg to 0.0 at 30 deg; threshold 0.5; support contact within 1 cm.
- Holodeck: near 50-150 cm, far >= 150 cm; face_to, in-front-of, side-of, center-aligned; hard = no collision/in room, others soft.
- Roomer (practical families): living functional angle >= 135 deg; dining side clearance >= 0.60 m; bed-side clearance / bed width >= 0.30; walkable passage 0.60 m. Commit rule: accept an edit only if target fixed, no new hard violation, every satisfied relation whose dependencies are disjoint from the edit stays satisfied, and global violation falls by >= 1e-6.
- LayoutVLM: relations that hold in the initial VLM layout are the only ones kept in the semantic loss (implicit protection); distance band [d_min,d_max], align, point-towards, against-wall, on-top.
- Infinigen Indoors: relations StableAgainst/SupportedBy/Touching/CoPlanar with margins (sofa 0 floor, 10 cm wall), focus_score for "sofas face TVs", reflectional symmetry of bedside tables, accessibility_cost(dist=1.0 default), chairs 2-6 per table. Concrete nightstand-bed / chair-table distances: unverified (not in retrievable text).
- SceneTeract: clearance/navigability/reach checks per agent; adult task success 59.0%, hasClearance 67.3%.
- Evaluated in practice: proximity, orientation (facing), overlap, clearance, walkable connectivity. Not evaluated anywhere we found with numeric pass rates: specific nightstand-bed adjacency.

## (1) Recommended visual score protocol

Use a two-tier score; do not use CLIP/VQAScore.
- Tier A (symbolic, deterministic): collision (OBB overlap, report fraction and >0.2 m flag), in-bounds, 0.60 m walkable connectivity, door/window clearance, and the companion checks in (2). These carry geometry; MLLMs do not (SceneCritic).
- Tier B (MLLM pairwise, gestalt only): for two candidate layouts of the SAME room and furniture list (e.g. before vs after repair, or model A vs B), render one composite image: A left, B right, each column top-down above and one or two perspective views below (VIPScene: multi-view first-person 0.39 vs 0.27 top-down). Colour/label objects by category in the top view. Prompt: expert judge; criteria = functional grouping, circulation/clearance, wall use, balance; NO description criterion; "list pros and cons of each, then pick; you must pick a winner"; JSON output. Randomise side each call, 3 calls, majority vote (R3L). Report win rate vs a fixed reference with CI; keep the judge model fixed and different from any generator family (PolyLayout notes same-family bias).
- Optional absolute gate: PolyLayout 1-4 anchored scale (1 physical failure, 2 main functional group broken, 3 coherent, 4 polished) for a pass/fail threshold at >=3, used only on top of Tier A.
- Expected failure modes: position/side bias; view sensitivity and re-scoring variance (12-19 points in SceneCritic); mirror asymmetry; missing small overlaps (so Tier A decides collisions); ties on near-identical layouts; saturated scores if absolute 0-10 is used; judge-human agreement realistically 72-77% / tau 0.5. Validate on 50-100 pairs with our own human labels before trusting it.

## (2) Companion-relation taxonomy with thresholds

Protect during repair (a move may not break a satisfied relation; Roomer rule). Distances are between footprint edges unless stated; derived from the literature, to be tuned.
| Relation | Pairs | Rule | Source of numbers |
|---|---|---|---|
| Adjacent/attached | nightstand-bed, desk-chair, chair-table (dining), bedside lamp | edge gap <= 0.5 m (next_to); chair/seat edge gap <= 0.5 m of table | SceneEval next_to 0-0.5 m |
| Group proximity | sofa-coffee table, sofa-TV, armchair-sofa, dresser-bed | gap <= 1.5 m (near) default, <= 2.0 m for high co-occurrence pairs (>=0.7), <= 3.5 m otherwise | SceneEval 1.5 m; SceneCritic 2.0 / 3.5 m |
| Facing | chair->table/desk, sofa->TV/coffee table, TV->sofa, armchair->coffee table | angle between front vector and direction to partner <= 30 deg (full score) , <= 75 deg pass, > 150 deg hard fail; living angle (sofa-coffee-TV) >= 135 deg | SceneEval 30; SceneCritic 75/150; Roomer 135 |
| Back-to-wall | bed, sofa, wardrobe, dresser, bookshelf | back within 45 deg of wall normal; wall gap <= 0.10 m | SceneCritic 45 deg; Infinigen 10 cm margin |
| Symmetry | two nightstands about bed | reflect about bed axis (qualitative) | Infinigen (no numbers) |
| Clearance | around bed sides; dining table sides; door/walkway | side clearance >= 0.60 m (dining); bed-side clearance >= 0.30 x bed width; passage >= 0.60 m | Roomer |
| Hard physical | all | OBB overlap <= 0.2 m tolerated only as soft; no out-of-bounds | SceneCritic 0.2 m |
Repair rule: for any object, moves are accepted only if all relations satisfied beforehand remain satisfied (those with companions of the moved object are excluded from the "disjoint" exemption), collisions decrease, and no new hard violation appears; prefer moving the non-anchor member (chair, nightstand) and translating the pair together when the anchor must move.
Unverified: any nightstand-bed or chair-table distance beyond SceneEval's generic bands; Infinigen's own values.

## (3) Why CLIP is not suitable here

CLIP/CLIPScore compares an image with text; with no description, the text is at most "a top-down view of a bedroom" (Holodeck), which every candidate with the right objects matches. CLIP is near chance on spatial relations: ARO VG-Relation 0.59 (chance 50%) and COCO-Order 0.46 (chance 20%); What'sUp 26.1-31.0% (chance 25%) and COCO/GQA-spatial 46.9-49.5% (chance 50%); VL-CheckList spatial 49.1. On layout judging it tracks humans at tau 0.06 (top-down, VIPScene) and kappa 0.02 (LEGO-Eval); VQAScore, the best-case alternative, gives tau 0.13 and 58.6% agreement. Published CLIP differences between layout methods are ~1 point on a ~20 scale with per-room swings of 6-12 (HDSL). Use it at most as a sanity check that the render looks like the room type.
