# SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis
- Authors: Hou In Ivan Tam, Hou In Derek Pun, Austin T. Wang, Angel X. Chang, Manolis Savva (SFU). 2025. https://arxiv.org/abs/2503.14756

## Metrics
Fidelity: CNT, ATR, OOR (object-object relation), OAR (object-architecture). Plausibility: COL (mesh-based pairwise collision, % objects in collision; tolerance not specified), SUP (support; contacts within 1 cm, centroid inside contact convex hull), NAV, ACC, OOB.
## Relations
- 13 object-object relations: inside_of, outside_of, face_to, side_of, side_region, long_short_side_of, on_top, middle_of, surround, next_to (0-0.5 m), near (0.5-1.5 m), across_from (1.5-4 m), far (>4 m). Architecture relations add inside_room, middle_of_room, on_wall, against_wall, corner_of_room, hang_from_ceiling with same bands.
- Positive/negative at score threshold 0.5. face_to: ray from front vector; score 1.0 at 0 deg dropping to 0.0 as angle approaches 30 deg; no ray hit on B -> negative.
- Navigability/accessibility metre values not found in portion read (unverified).
## Judge and agreement
GPT-4o (2024-08-06) for object matching, attributes, relation mapping; geometry done by code. Manual eval 500 scenes: agreement CNT 89.8, ATR 83.5, OOR 94.6, OAR 94.1 %; Cohen kappa 0.75/0.56/0.72/0.77. User study (10 participants, 125 scenes): 89.5/89.9/91.7/88.1 %; kappa 0.72/0.57/0.58/0.56. Qwen2.5-VL-7B lower agreement.
## Relevance
Gives distance bands and a continuous facing score (cos-like decay to 30 deg) that we can reuse. Its relations come from a text description, which we lack.
