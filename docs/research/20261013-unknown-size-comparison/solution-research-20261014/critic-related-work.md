## Critic and repair approaches for 3D indoor layouts, compared with a VLM-critic loop

### SceneSmith, read from the local code at `/Users/slian/Desktop/3D/Worldedge/scenesmith`
- **Loop.** There are three agents: a planner, a designer and a critic. The critic only observes and never edits. Its tools are `observe_scene` (a top view plus 4 side views with the blocking walls hidden, i.e. a cut-away render), `get_current_scene_state`, `check_facing_tool` (orientation checked from geometry, not from the image), and `check_physics` (Drake collisions plus door, window and opening clearance). The prompt at `scenesmith/prompts/data/furniture_agent/stateful_critic_agent.yaml` caps the realism score at 4/10 if any collision exists. It says: "Tool validation overrides visual assessment… camera angles create optical illusions". The designer makes the fixes through `add/move/remove/rescale_furniture_tool`. `configurations/furniture_agent/base_furniture_agent.yaml` sets `max_critique_rounds: 3` and `early_finish_min_score: 9`, and rolls back when any score drops by 2 or more.
- **Geometric repair is not done by the critic.** `scenesmith/agent_utils/physical_feasibility.py` resolves penetrations with Drake IK non-penetration projection (SNOPT, XY only, rotation fixed, 1 mm threshold). A gravity simulation follows.
- **Reported gains** ([arXiv 2602.09153](https://arxiv.org/html/2602.09153v2)):
  - Collisions 1.2% vs 3–29% for the baselines.
  - The critic ablation (NoCritic) was **not significant** in the user study: 51.8% realism and 47.5% faithfulness win rate.
  - NoCritic produced 24% fewer objects and was about 70% cheaper: $4.24 vs $13.98 per scene. The critic itself costs $2.79 per scene.
- **Categories.** Not preserved by default, because the designer can add and remove objects. For our use we would have to remove those tools.

### Other approaches

| Method | What is checked | How repair works | Categories kept? | Reported gain and cost |
|---|---|---|---|---|
| [LayoutVLM](https://arxiv.org/html/2412.02193v3) (CVPR'25) | VLM writes relations (distance, on_top_of, align_with, point_towards, against_wall) plus a physics term | Projected gradient descent (Adam, 400 steps) on a DIoU collision loss plus relation losses; out-of-bounds fixed by projection every 100 steps | Yes, all assets are placed | In-bounds 94.9 vs 14.1 without optimisation; PSA 58.8 vs 6.7. About 1–5 min and 5–10 GPT-4o calls per scene |
| [Holodeck](https://arxiv.org/html/2312.09067v2) | Hard: no collision, inside room. Soft: LLM relations (edge/middle, near/far, face-to, aligned) | DFS on a grid for a fixed 30 s, keeping the candidate that satisfies the most soft constraints (MILP variant via Gurobi) | **No**: an object that cannot be placed is dropped | Human MRR 0.706 vs 0.364 when the LLM writes absolute coordinates |
| [I-Design](https://arxiv.org/html/2404.02838v1) | LLM "Layout Corrector" removes impossible scene-graph edges | Backtracking placement | **No**: deeper objects are removed when placement fails | Out-of-bounds 0% vs 51–64% for LayoutGPT |
| [LayoutGPT](https://arxiv.org/abs/2305.15393) | Nothing | No repair | n/a | 51–64% out-of-bounds (as measured by I-Design) |
| [SceneCraft](https://proceedings.mlr.press/v235/hu24g.html) | GPT-V on renders against a scene graph | LLM edits the Blender code | Mostly yes | "Better constraint adherence"; I did not verify numbers |
| [Infinigen-Indoors](https://arxiv.org/abs/2406.11824) | Python constraint DSL (physics, accessibility, relations) | Simulated annealing with Metropolis-Hastings; discrete moves (add, re-relate) plus continuous moves; large → medium → small objects | No (it can add objects) | Cost not checked |
| [PhyScene](https://arxiv.org/html/2404.09465v2) | 3D IoU collision, wall-box out-of-room, A* reachability | Gradient guidance during diffusion sampling (position and angle only) | Yes | Modest: living-room collisions 0.200 → 0.191. Collision guidance alone raises out-of-room to 0.354, so the terms trade off |
| ATISS / DiffuScene | n/a | No post-processing described (per the PhyScene paper) | n/a | Collisions 0.16–0.59 |
| [OptiScene](https://arxiv.org/html/2506.07570v2) | Training only: DPO-2 rejected samples are GT with injected overlap or out-of-bounds | No repair at inference | Yes | Overlap ratio 0.049 (SFT) → 0.023 (+DPO1+2) |
| [SceneWeaver](https://arxiv.org/html/2509.20414v2) (NeurIPS'25) | MLLM on an annotated top-down render (boxes, labels, front arrows) plus a layout list; scores realism, function, layout, completeness | Planner picks tools (update rotation/size/layout, add relation, remove object, …); executor runs a physics optimisation every step; up to 10 rounds with rollback | **No**: adds and removes objects | 0 out-of-bounds and 0 collisions. The reflection ablation covers only 3 kitchen scenes (Real. 8.0 → 9.0), so the evidence is weak |
| [SceneCritic](https://arxiv.org/html/2604.13035) (Apr 2026, preprint) | Symbolic ontology checks (semantics, orientation, overlap) | Compares heuristic, LLM and VLM critics in a refinement loop | Not stated | Heuristic critic best on overlap (99.3 / 98.9). Image-only VLM best on semantics (78.7) and orientation (55.8) but worse on overlap (97.5 / 96.6). VLM-as-judge agrees with humans only 47–59% |
| [Scenethesis](https://arxiv.org/abs/2505.02836) (ICLR'26) | Judge for spatial coherence | SDF-based physics optimisation plus targeted repair | Mostly | Numbers not verified |
| [FirePlace](https://arxiv.org/abs/2503.04919) | MLLM extracts geometry | Constraint solving, then MLLM prunes candidates | Single-object insertion | Not directly applicable |
| [LEGO-Net](https://arxiv.org/abs/2301.09629) | Learned regularity | Iterative learned denoising of position and yaw | Yes | 2D only; does not handle interpenetration |
| [Co-Layout](https://arxiv.org/abs/2511.12474) (AAAI'26) | LLM-extracted constraints | Grid integer programming, coarse to fine | Unverified | Abstract gives no numbers |
| LLplace | n/a | Edits through dialogue, no geometric repair | n/a | n/a |

### What the literature agrees on
1. **Geometry is always fixed by a solver or optimiser, not by the LLM or VLM.** This holds for SceneSmith (IK projection), LayoutVLM (PGD), SceneWeaver (physics step after every tool), Holodeck (DFS) and Infinigen (SA). Taking the optimiser away is catastrophic: LayoutVLM's in-bounds score falls from 94.9 to 14.1.
2. **The VLM critic adds little that can be measured.** SceneSmith's NoCritic result is not significant and saves 70% of the cost. SceneWeaver's ablation is n=3. SceneCritic finds the VLM critic only helps semantics and orientation, and slightly hurts overlap.
3. **A VLM is a poor gate.** Agreement with humans is about 50% (SceneCritic). Our pass/fail gate should stay `validate.check()`.
4. **Most systems preserve categories only by accident.** Holodeck, I-Design, SceneWeaver and SceneSmith all drop or add objects. If we adopt one, we must remove the add/remove tools and constrain the edit API to move / rotate / snap-to-wall / change-parent per id. Size changes must be forbidden or tightly bounded. Otherwise the critic can win by shrinking furniture, which is gpt-6.1-sol's existing −17% footprint bias.

### Deterministic vs VLM split
- **Deterministic (solver).** Out-of-bounds against the room polygon; door/window keep-out (the backend's 1 mm / 1 cm rules); pairwise collision with the tuck exemptions; support and parent. The fix is minimum-displacement projection with fixed sizes, categories and yaws:
  - Use SceneSmith's IK or LayoutVLM's DIoU+PGD. Note that `small_repair` / `snap_inside` only allow ≤10 cm, which is too small for our gross failures.
  - Co-located duplicates and many-object rooms need a global placement search (Holodeck DFS or Infinigen SA style), because local projection cannot untangle a stack.
  - Cheap: milliseconds to seconds per room on CPU.
- **VLM critic on the box diorama.** Use it for:
  - semantic plausibility: bed headboard against a wall, chairs facing the table, sofa facing the TV, storage against a wall, front accessible;
  - "looks wrong" arrangements.

  Follow SceneSmith's own rule: the VLM proposes intent, and a geometric tool (like `check_facing_tool`) verifies it. Do not trust orientation read from the image alone. The SceneCritic data suggests image-only input is best for the orientation critic.

### Recommendation
- **(b) Inference outputs: do this first.**
  - Pipeline: model → deterministic solver repair (categories, sizes and yaws frozen; positions free; global re-placement only for clusters the projection cannot resolve) → `check()`.
  - Then an optional single VLM pass on the diorama for facing and against-wall edits, through a whitelisted per-id edit API.
  - Re-run the solver after every VLM edit. Accept an edit only if `check()` still passes and the judge score does not regress (SceneSmith's rollback rule). Cap it at 1–3 rounds.
  - The solver alone targets most of the current 61% failures (collisions and out-of-bounds) and is deterministic. The VLM layer should be judged by an ablation, because SceneSmith found it insignificant.
- **(a) Training data: do not VLM-critic all 98,754 rooms.**
  - Run the same deterministic repair on the GT with a displacement budget, and record repaired, unchanged and failed per source (scans 0.14–0.60 and 3D-FRONT 0.49 will dominate).
  - Train on the repaired GT, or filter, as a separate tier so that "flag, don't filter" stays reversible.
  - Use the VLM critic only to audit a stratified sample (a few hundred rooms per source) to see whether the repair damages semantics.
  - The repaired-vs-raw pairs can also serve as on-policy DPO pairs, a stronger version of OptiScene's DPO-2.
- **Not verified:** SceneCraft, Scenethesis and Co-Layout numbers; I-Design runtime; SceneWeaver cost (appendix not read); Infinigen solver cost; the LayoutGPT URL (cited from memory, not fetched).