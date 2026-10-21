# PDF audit: EmbodiedGen V2, SceneCode, OptiScene

Audit date: 2026-10-20. This is a targeted primary-paper audit, not an exhaustive review. Included the latest locally downloaded arXiv PDFs of the three systems named in the prior recommendation. Excluded the unrelated 2019 SceneCode paper, project-page claims not substantiated in the PDFs, and tertiary reviews. Read evaluation sections and metric appendices; visually inspected rendered tables. Page numbers below are PDF pages, one-based (also match printed pages here).

## Load-bearing corrections

1. **World quality, human acceptance, scripted asset tests, and policy task success are different outcomes.** The three PDFs do not provide a common world-generation/agent-execution leaderboard that can simply be copied.
2. **SceneCode OOR means object–object relationship satisfaction (higher better). OptiScene OOR means footprint overlap (lower better).** Use descriptive labels in our table; never reuse an ambiguous bare OOR column.
3. **EmbodiedGen's 83.3% is human world acceptance, not autonomous task SR. Its 98.6% Collision Success is scripted grasp-and-lift success, not non-collision rate.**
4. **SceneCode's manipulation evidence is a qualitative MuJoCo demonstration.** Its paper does not report a quantitative articulated-task SR or trial denominator.
5. **EmbodiedGen V2 Table 5 explicitly summarizes a separate study.** It is not an independent replication of the same RL results. Trace to Choi et al., arXiv:2603.18532.
6. Prior “do not multiply rates” requires qualification: multiplying *marginal* world-check pass rates is invalid without independence; multiplying *conditional* cascade pass rates gives the joint yield by the probability chain rule. EmbodiedGen Table 3 uses the latter.

## EmbodiedGen V2

Source: https://arxiv.org/pdf/2607.07459v3 (18 August 2026, 24 pages). Evidence grade: primary systems preprint. Good relevance for pipeline stage reporting; modest weight for universal world usability because manual-review details are limited. RL results are derivative within this paper, not independent evidence.

**Table 2, p.13, §3.1: asset ablation, 200 held-out assets.** Headers: Human Accept. ↑; Collision Success ↑; Time (min) ↓; Visual Mesh (MB) ↓; Collision Mesh (MB) ↓. Human acceptance is annotator judgment of input consistency, geometry, unseen surfaces and sim usability; annotator number/agreement rule are not specified here. Collision Success averages four scripted Franka Panda top-down grasp-and-lift trials per asset at evenly spaced yaw angles in SAPIEN; success is lift above a bbox-height-dependent threshold. Full pipeline 96.5% and 98.6%, respectively. This does not measure whole-world collision-free rate or task-policy competence.

**Table 3, p.14, §3.2: affordance ablation, 200 assets.** Headers: Segmentation Pass Rate ↑; Semantic Validity Rate ↑; Grasp Coverage Rate ↑; Affordance Pass Rate ↑; Runtime (s) ↓. Each stage is conditional on preceding success. Segmentation and semantics use VLM checks plus manual cross-validation. Grasp coverage requires at least one physically validated grasp per object, but intrinsically ungraspable household appliances are exempted. Grasp validation rejects slip exceeding 5 cm or 30 degrees. The final 50.0% affordance rate multiplies conditional stage rates; it is not 50% robot task SR.

**Table 4, p.15, §3.3: 150 task-conditioned worlds.** Categories: Task-to-Graph, Asset Instantiation, Asset QA, World-Level Outcome. Includes 150 worlds, 5.19 interactive instances/world, 128 categories, per-asset generation time, Semantic Appearance 76.2%, Mesh Geometry 75.9%, Cross-modal Text-to-3D Alignment 91.0%, 1.35 attempts/valid asset, 47.7±5.4 min/world, and final environment acceptance 83.3%. Asset QA rates are per generation attempt, not final world percentages. Final acceptance is manual inspection for task-semantic consistency, relations, scale, stability and robot executability. Rater count, agreement rule and a standardized executed task-success test are not supplied. Figure 11 p.17 includes a task already satisfied at initialization, an ungraspably oversized object, and an unstable object; this supports checking task initialization, but does not establish our proposed Task-Spec metric as a standard.

**Table 5, p.18, §3.4:** columns Validation Axis / Downstream Setting / Key Result. The caption and §3.4 explicitly attribute results to companion study [6], *Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds*, https://arxiv.org/abs/2603.18532 . Metrics summarized include simulation task success, completion time, OOD task success, ID–OOD gap, real-world task success and dynamics failure. It is a train-on-generated-worlds transfer experiment, not fixed-agent evaluation of multiple generators. Thus it supports a separate learning/transfer table, not importing 79.8% into our generator leaderboard. A further cube-stacking result comes from RankQ [7], arXiv:2605.11151.

No quantitative revisit-state or state-preservation benchmark found in these evaluation sections. Stateful Vibe Coding is a capability, not a measured persistence-SR column.

## SceneCode

Source: https://arxiv.org/pdf/2605.19587v1 (19 May 2026, 34 pages). Evidence grade: primary preprint, 30 prompts selected from SceneEval-100, six room types, nine human participants. High confidence about what it reports; moderate weight for metric choices; low weight for downstream interaction effectiveness because demonstrations have no task statistics.

**Table 1, p.7, §4.2; definitions Appendix D.1 p.18:** CNT ↑, ATR ↑, OOR ↑, OAR ↑, SUP ↑, ACC ↑, NAV ↑, COL ↓, OOB ↓, OPC ↓, ΔPF ↑. First ten are scene-level SceneEval scores averaged over 30 prompts with 95% CIs. They are separate requirement/object/space/opening quantities, not a shared binary all-world acceptance rate:

- CNT: count-requirement satisfaction after VLM object matching. ATR: attribute-requirement satisfaction judged from rendered asset views, including scale reference.
- OOR/OAR: geometric satisfaction of object–object / object–architecture relations.
- SUP: fraction of objects with support consistent with a VLM-assigned support type, checked by ray casting. This is a geometric support proxy, not measured dynamic stability.
- ACC: fraction of objects with accessible functional sides; VLM identifies sides, 2D occupancy checks obstruction.
- NAV: largest connected free-floor region area / total free-floor area. It does not test an agent traveling to the required target or executing manipulation.
- COL: fraction of objects participating in at least one pairwise mesh intersection. OOB: fraction of objects with fewer than 99% downward surface rays hitting the room floor.
- OPC: fraction of door/window clearance volumes obstructed. It is not cabinet-joint actuation success.

ΔPF compares the same prompt/rater within each baseline group. Appendix G pp.22–23 and Table 6 p.26: 3 raters/group × 30 prompts = 90 paired trials/comparison, 1–5 ratings for prompt faithfulness and realism, pairwise preference. The SceneSmith prompt-faithfulness interval includes zero (+0.14, CI −0.05 to +0.34). Do not claim statistically established superiority over SceneSmith. These are human judgments, separate from automated physical scores.

**Table 2, p.8:** MAT, PBR, NME, FAC, VTX, UVI are asset representation/mesh/material diagnostics; appropriate for an asset appendix, not full world task utility. §4.4 / Figure 5 p.9 shows three articulated MuJoCo examples; no quantitative robot-task SR table or episode count. Persistent state serialization is described in the method, but no persistence stress benchmark is reported.

Source consistency caution: main Table 1 reports SceneCode COL 11.3%, while Appendix F p.21 discusses COL 0.0934. Other appendix values also differ slightly from main-table values. Treat Table 1 as the reported table, do not blend the two sets or copy any into our results. Cost Tables 7–8 are p.27; costs depend on their implementation and budget.

## OptiScene

Source: https://arxiv.org/pdf/2506.07570v2 (19 September 2025, 24 pages). Evidence grade: primary preprint. Clear metric definitions for overlap and human UR; limited navigation protocol disclosure. Useful for the FastFill layout-component comparison, not a common full-world benchmark.

**Tables 1–3, p.7, §4.1–4.2:** Table 1 has FID ↓ / OOR ↓ across 3D-FRONT bedroom/living room and four 3D-SynthPlace room types. Dataset test split: 423 bedrooms and 53 living rooms from 3D-FRONT, plus 50 samples for each of four Holodeck-Synth room types; 15,990 training layouts. Table 2 has GPT-4o Func / Layout / Aes scores, each 0–10. Table 3 is labelled generation success rate, but text identifies it with human usability. UR protocol: 50 scenes per room type, five annotators, usable if at least three judge physical and semantic validity. Thus human UR cannot be replaced with our deterministic strict-admission rate.

Appendix D.2 p.17 defines overlap as sum of all pairwise 2D bbox intersection areas divided by sum of all bbox areas. It is neither the fraction of intersecting boxes nor an out-of-room rate; main prose is less exact than the appendix formula. It ignores the full 3D contact/support problem. FID and VLM scores are useful supplementary layout/visual outcomes, not direct robot performance.

**Tables 4–5, p.9, §4.3:** editing OOR / ESR from 20 cases; object loco-navigation SR / NE with rows GPT-4o, Qwen-3-8B, Chat-GLM4-flash. These rows compare controllers on OptiScene worlds, not world generators with one fixed controller. SR requires target within ±30 degrees of field of view and 2 m distance; NE is final target distance. Table header prints NE ↑ despite this definition; our table should use NE ↓. The paper does not provide enough task count/seeds/control detail here to reconstruct a matched independent navigation benchmark. Appendix D.4 p.18 and later figures show additional examples, not a new numerical navigation protocol.

Minor reporting caution: Table 3 includes 75%, 25%, 35% despite the stated 50 binary scenes/type; exact aggregation/rounding is not explained. Avoid borrowing these values as directly reproducible ground truth.

## Implications for our Table 1 (our synthesis)

Primary companion follow-up: https://arxiv.org/html/2603.18532v4 (29 September 2026), titled *Scaling Sim-to-Real VLA Reinforcement Learning with Generative 3D Worlds*, marked accepted to CoRL 2026 on its arXiv record. Read §4 and Appendices A/C in primary HTML; did not download/render its 47 MB PDF. Its Table 1 reports SR, time-to-finish and ID–OOD gap; 100 episodes/scene, three runs where standard deviations are shown. Time averages successful episodes only. Table 2 reports partial SR (grasp/lift), full SR, dynamics/semantic failure and time over 12 scenes × 10 trials × 2 policies = 240 real trials. Failure categories overlap. Appendix A defines simulated success using contact with the target support, no table contact and no robot contact. Appendix C separately has manual environment acceptance 85% for 100 scenes; do not confuse it with EmbodiedGen V2's 83.3%/150 worlds. Evidence weight is higher for controlled pick/place transfer than the systems paper's copied summary; scope remains pick/place, not navigation, articulation or persistent worlds.

Use two levels without pretending all papers share one score: (A) literature-aligned world quality, with explicit prompt/object/attribute/relation satisfaction and geometric support/accessibility/navigation/collision/boundary metrics; (B) actually executed tasks, with established SR and diagnostic progress/efficiency measures under one fixed agent. Annotator usability can be an additional independent study. Keep asset technical metrics and train-on-generated-worlds transfer separate. Our own task initialization, articulated-test pass, persistence and verified-ready joint yield are legitimate proposed additions, but require new definitions and validation and must be labelled as ours, not existing benchmark standards.

Minimal main-world choice, if width is tight: use six *separate* existing columns CNT, ATR, object-relation satisfaction, COL, ACC, NAV, with OAR/SUP/OOB/OPC in the full diagnostic table and explicit task-specific justification for the subset. Do not manufacture a weighted “physical validity” aggregate. If support/attachment is a main claim, promote SUP to the main table. Preserve the full ten-column SceneEval table in supplement. A new joint ready-yield or human usability endpoint may be added only with its independent protocol; neither substitutes for the transparent component scores. Articulation/persistence tests belong in a clearly labelled new benchmark or task-family result block, not as literature-standard metrics.

All discussed PDFs remain under `docs/research/20261020-worldedge-table1/papers/`. Rendered table pages inspected under `/tmp/pdf-audit-world-systems/`: OptiScene pp.7,9,17; EmbodiedGen V2 pp.13,14,15,18; SceneCode pp.7,9,18.
