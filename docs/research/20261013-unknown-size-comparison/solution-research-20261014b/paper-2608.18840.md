# SceneSmith (arXiv 2602.09153) vs FastFill: what we can transfer, and why P fails

Repo: `/Users/slian/Desktop/3D/Worldedge/scenesmith`. Paper: https://arxiv.org/abs/2602.09153 (local PDF `/Users/slian/Desktop/3D/Worldedge/paper/SceneSmith- Agentic Generation of Simulation-Ready Indoor Scenes.pdf`). There is a prior note on the same subject, `/Volumes/harddisk/fastfillout/unknownsize-20261013/solution-research-20261014/critic-scenesmith.md`. I re-checked its key citations; this report adds new measurements on our data.

Labels: **[V]** = I verified it in code, the paper or our data. **[I]** = my inference or an untested hypothesis.

## 0. Bottom line

1. **[V] SceneSmith never asks a model for collision-free coordinates in one pass.** Three things make its layouts feasible:
   - verification after each action: `check_physics` is required after every move, and the designer may not finish while collisions remain;
   - a global nonlinear projection: minimise ‖p − p0‖² subject to signed distance ≥ 1e-5 m, all objects jointly, rotations fixed (paper H.1 eq. 4);
   - freedom we do not have: the designer may delete objects and shrink or regenerate assets to clear collisions.

   P is a single greedy pass with no verifier. That is the main structural difference.
2. **[V, new] P has a copy-the-parent failure mode that GT does not have.**
   - 53.2 % of P's supported children sit within 1 cm of the parent's XY centre; in GT it is 3.7 %.
   - 16.4 % of P rooms have at least two siblings at the same XY; GT 1.8 %; L and O2 0.0–0.1 %.
   - 1,341 of P's 5,185 raw-failing rooms contain such siblings.
   - It holds in every source: SAGE-10k GT 0.1 % vs P 47.1 %; MansionWorld 4.6 % vs 72.8 %; InteriorGS 14.3 % vs 90.3 %.
   - So the model produced this, not the labels.
3. **[V, new] O2's dense-room failures are truncation, not geometry.** In the cohort, O2-harness rooms with 21+ objects: 44 of 102 incomplete, and 43 of those 44 hit the token cap. Once complete, O2 almost always passes. Amendment 018's teacher will therefore have low and biased yield in the >50 bin it allocates a third of T-B to.
4. **[V, new, EXPLORATORY] A SceneSmith-style joint projection on top of S1 lifts P's frozen-checker success sharply in dense rooms.**
   - Prototype on 60 complete test rooms per bin, with the frozen checker reproducing all 300 raw verdicts.
   - Bin-weighted over complete rows: S1 0.679 → S1+projection 0.868.
   - This is still below L and O2. Semantic damage and infinite-task-world admission were not measured.
5. **[V] Do not copy the VLM critic for feasibility.** SceneSmith's own NoCritic ablation has COL 0.8 % vs 1.2 % for the full system, with an n.s. user study (51.8 % realism win), at $4.24 vs $13.98 per scene.

## 1. How SceneSmith generates and fixes layouts

### 1.1 Pipeline and agents [V]
- **Stages:** floor_plan → furniture → wall_mounted → ceiling_mounted → manipuland (README; `configurations/experiment/base_experiment.yaml:26-33`).
- **Three agents per stage** in `scenesmith/agent_utils/base_stateful_agent.py`:
  - The planner (orchestrator) calls `request_initial_design`, `request_critique`, `request_design_change(instruction)` and `reset_scene_to_checkpoint` (`:663-734`).
  - The critic returns `FurnitureCritiqueWithScores` with six 0–10 scores: realism, functionality, layout, holistic_completeness, prompt_following, reachability (`agent_utils/scoring.py:80-109`).
- **Rollback:** if any score drops by ≥2 or the total by ≥1, the planner rolls back (`base_stateful_agent.py:400-439`). The final scene is auto-reset to checkpoint N-1 if the last critique regressed (`:479-525`).
- **Limits** (`configurations/furniture_agent/base_furniture_agent.yaml`): `max_critique_rounds: 3` (:244), early stop when all scores ≥ 9 (:252), reset thresholds (:248-249).
- **The critic only reads:** `observe_scene`, `get_current_scene_state`, `check_facing_tool` (`furniture_agents/stateful_furniture_agent.py:114-135`). Physics and reachability results are injected into its prompt as text (`base_stateful_agent.py:780-801`; `stateful_furniture_agent.py:387-403`).
- **Decomposition:** each room gets a fresh agent instance (paper App. D). Manipulands run per supporting furniture, with fresh agents "to bound token usage" (`manipuland_agents/stateful_manipuland_agent.py:680-860`, loop at :757-858).
- **Memory:** older turns are summarised and images stripped (`base_furniture_agent.yaml:5-22`; `agent_utils/turn_trimming_session.py`). Repeated identical tool calls are blocked (`agent_utils/loop_detector.py`).

### 1.2 Action space [V]
- **Furniture placement:**
  - `add_furniture_to_scene_tool(asset_id, x, y, yaw)` and `move_furniture_tool`: SE(2) only, z = 0 (`furniture_agents/tools/furniture_tools.py:272-344`).
  - The only placement guard is that the object's centre lies inside the room rectangle (`:118-144`). Collisions are not blocked at placement time.
  - Gaussian placement noise is added afterwards (`:477`, `:658`; 3 cm / 1° "natural", 1 mm / 0.1° "perfect", `base_furniture_agent.yaml:351-361`).
- **Size and object-set freedom:**
  - `generate_assets(..., desired_dimensions)` lets the model choose sizes (`furniture_tools.py:217-270`).
  - `rescale_furniture_tool` is uniform and asset-wide (`:375-403`); `remove_furniture_tool` exists (`:346-359`).
  - The designer prompt lists "Delete excess furniture" and "Regenerate smaller assets" as ways to resolve collisions (`prompts/data/furniture_agent/designer_agent.yaml:489-495`, also `:92-94`).
- **Deterministic geometric primitives:**
  - `snap_to_object_tool(object_id, target_id, orientation=toward|away|none)` (`furniture_agents/tools/scene_tools.py:111-154`, impl. `:578-722`; algorithm in paper D.2): AABB push-out, then yaw from `check_facing`, then step toward the target in 1 cm increments until contact, keeping a 1 cm margin (`base_furniture_agent.yaml:329-348`).
  - `check_facing_tool` returns `optimal_rotation_degrees` from a 2D ray-AABB test (`scene_tools.py:72-109`; paper D.1).
- **Manipulands** are placed in a support-surface-local 2D frame: `place_manipuland_on_surface(asset_id, surface_id, position_x, position_z, rotation)` (`manipuland_agents/tools/manipuland_tools.py:557-601`). Placement is rejected if the centre or the convex-hull footprint leaves the surface (`:1046-1091`).
- **Group tools** place several items in one call: `create_stack`, `fill_container`, `create_arrangement` (container-local x, y), `create_pile` (`:696-850`).
- `resolve_penetrations` is commented out with "Disabled: 0% success rate in experiments" (`:984`).
- **Ablation:** NoSpecializedTools (no snap / facing / stack / fill / arrangement / pile) gives 54.8 % realism win (n.s.) and COL 0.5 % (paper Table 1 and Table 14). With GPT-5.2 the primitives are convenience, not a feasibility source.

### 1.3 Verification [V]
- **Collisions:** Drake `ComputeSignedDistancePairwiseClosestPoints`, penetration > 1 mm, 5 cm floor tolerance (`agent_utils/physics_validation.py:155-327`; config `base_furniture_agent.yaml:300-312`). Feedback text names the pair and depth, e.g. "A collides with B (x cm penetration)" (`physics_validation.py:45-56`).
- **Door clearance:** an AABB zone of door width × 0.8 m into the room × door height (`agent_utils/clearance_zones.py:152-220`; `configurations/floor_plan_agent/base_floor_plan_agent.yaml:127-128`). A violation is any furniture AABB intersecting the zone; thin coverings are exempt (`clearance_zones.py:400-459`).
- **Window and wall-height checks** are aggregated in `agent_utils/physics_tools.py:157-253`.
- **Reachability:** floor eroded by half of a 0.3 m robot square, minus buffered furniture OBBs. The number of connected regions is counted, and blockers are found by leave-one-out (`agent_utils/reachability.py:44-124, 212-246`).
  - It is soft: it only caps the critic's reachability score at ≤6 (`stateful_critic_runner_instruction.yaml:83`).
  - **[V code, I effect]** It builds the floor as (0,0)–(L,W) (`reachability.py:127-134`), while the move tool uses the centred frame ±L/2 (`furniture_tools.py:128-134`). This looks like a frame mismatch; not checked at runtime.
- **Prompt rules:**
  - The designer may not finish with collisions (`designer_agent.yaml:475-485`; `designer_critique_instruction_stateful.yaml:25-26`).
  - The critic caps realism at 4/10 if any collision exists (`stateful_critic_agent.yaml:119`).
  - The critic is told "Visual bounding box overlap does NOT indicate collision" (`stateful_critic_runner_instruction.yaml:72`). For box layouts this would have to be inverted.

### 1.4 Constraint solving / repair [V]
- **Projection:** `solve_non_penetration_ik` (`agent_utils/physical_feasibility.py:516-702`).
  - Cost: quadratic distance to the initial position (`:610`).
  - Constraint: `AddMinimumDistanceLowerBoundConstraint(1e-5, influence 0.02)` (`:638`), solved with SNOPT or IPOPT.
  - Rotations fixed; walls welded (`:724`).
  - For scenes with >100 objects, only colliding objects are free (`:814-831`).
- **Surface version:** `apply_surface_projection` keeps each XY inside the surface minus the object footprint (Pontryagin difference) (`:1030`, `:1160`).
- **Where it runs** (`experiments/indoor_scene_generation.py:563-606`, `:834`; paper H.3):
  - after furniture, everything free;
  - per supporting furniture, with that furniture welded;
  - at the end, with manipulands only.
- **Gravity simulation:** Drake, 15 s; furniture tilted more than 45° is removed (`base_experiment.yaml:74-113`; `base_furniture_agent.yaml:311-312`).
- **[V] Door clearance is not a projection constraint.** `physical_feasibility.py` contains no clearance logic.
- **[V] Evaluation asymmetry:** COL is measured after SceneSmith's own projection and simulation, while baselines only get collision geometry added (paper M.2.1, H.3). No pre-projection COL is reported. **[I]** Projection probably accounts for much of the 1.2 % COL.

### 1.5 Model, cost, results [V]
- **Model:** GPT-5.2 for all agents and VLM calls; designer and critic at high reasoning, planner at low (paper A.2; `base_furniture_agent.yaml:255-268`).
- **Cost per scene** (179 rooms, paper Tables 18-19):
  - mean $13.98 (sd 9.98, max $101.93);
  - 3 h 26 min per scene, measured with 25 scenes running concurrently on 8 L40S GPUs;
  - the designer is 75.9 % of cost, the critic 20.0 %;
  - NoCritic: $4.24 and 1:40.
- **Results:** 71.1 objects per room, COL 1.2 %, STB 95.6 % (Table 14).
- **Base-model swap control (Table 15):** giving HSM and Holodeck GPT-5.2 did not consistently help. HSM's COL got worse, 20.6 → 31.8. **[I]** Verification plus projection, not model size alone, drives feasibility.

## 2. New measurements on our data (scripts in my scratch dir)

### 2.1 Checker success by object count [V]

Raw and after S1 (`bins.py`):

**P rpo-dpo1, full test**

| Objects | Rooms | Raw | After S1 |
|---|---|---|---|
| 1–5 | 3,713 | 0.685 | 0.869 |
| 6–10 | 1,877 | 0.339 | 0.761 |
| 11–20 | 1,624 | 0.139 | 0.616 |
| 21–50 | 1,086 | 0.031 | 0.347 |
| 51+ | 347 | 0.000 | 0.084 |

- In the 51+ bin, 280 of P's S1-failing complete rooms still have main collisions. Greedy S1 stalls in dense rooms.

**Agents on the cohort**

| System | 21–50: complete | 21–50: raw | 51+: complete | 51+: raw |
|---|---|---|---|---|
| L (gpt-6.1-sol) | 1.0 | 0.62 | 1.0 | 0.40 |
| O2 harness | 0.561 | 0.488 | 0.600 | 0.500 |

- L after S1: 0.79 / 0.70; its remaining failures are mostly `support`.
- O2's losses are incompleteness; its residual geometry failures after S1 are 0–3 rooms per bin.

### 2.2 Parent-centre copying [V]
Numbers are in §0 point 2 (`coloc.py`, `coloc_gt.py`, `coloc_src.py`).

### 2.3 Joint projection prototype (EXPLORATORY)
`project.py` is a Jacobi position-based relaxation driven by the frozen `validate.check`:
- SAT minimum-translation for main/severe pairs, split by area so the smaller object moves more;
- push-outs for `fixed_blocking` and out-of-bounds;
- child centres clamped back over the parent;
- support subtrees move as units below the pair's common ancestor, as in S1;
- sizes, yaw, parents, z and ids are never changed.

Results on 60 complete test rooms per bin (stable hash), checker success:

| Objects | S1 | Projection on raw | S1 + projection |
|---|---|---|---|
| 1–5 | 0.883 | 0.917 | 0.933 |
| 6–10 | 0.733 | 0.883 | 0.917 |
| 11–20 | 0.517 | 0.767 | 0.867 |
| 21–50 | 0.317 | 0.517 | 0.667 |
| 51+ | 0.083 | 0.283 | 0.533 |

- **Displacement:** median displacement of moved objects 0.14–0.26 m. Projection on raw moves 35 % (6–10) to 69 % (51+) of objects.
- **Runtime:** p50 / max per room is 5.5 / 39 s in 21–50 and 45 / 225 s in 51+ (unoptimised, CPU).
- **Not measured:** infinite-task-world admission, keep-outs and 1 mm envelope, held relations such as `against_wall`, semantic or visual damage. n = 60 per bin, no CIs.

## 3. Transferable mechanisms

### (c) Inference-time loops for a small local model (no retraining first)
1. **C1. Global minimum-displacement projection after S1.** Source: paper H.1; `physical_feasibility.py:516-702, 761-879`.
   - Make the prototype production-grade: add downstream keep-outs and the R2 1 mm envelope as constraints, keep S1's held relations, invariants and rollback guard, and run before `serve.py` validation.
   - Two implementation routes:
     - Drake IK, using IPOPT, since SNOPT is not confirmed in the pip wheel **[unverified]**;
     - the vectorised 2D solver, which is enough because our objects are boxes.
   - Evaluate on the 600 dev rooms with a locked rule before any test use.
   - Expected gain is in §2.3. **[I]** It will not close the gap to O2 alone.
2. **C2. Verify each object as it is generated.** SceneSmith's pattern is add → `check_physics` → fix, with the designer forbidden to finish with collisions.
   - Small-model version: when an object's `on` field closes, run the incremental box check against the boxes already placed, the boundary and the keep-outs. Resample only that object's pos/yaw tokens, up to k times.
   - **[I]** This targets the 51+ bin, where room-level best-of-N has p ≈ 0. Not tested.
3. **C3. Deterministic sibling layout.** Analogue of `create_arrangement`, `create_stack` and the Pontryagin surface regions (`manipuland_tools.py:696-850`; `physical_feasibility.py:1160`).
   - For siblings that P co-locates on one parent, re-pack their XY inside "parent top ⊖ child footprint".
   - This directly addresses the failure in §0 point 2. **[I]** It is cheaper than C1 for those rooms.
4. **C4. Repair round with rollback.** SceneSmith's "critique → design change → re-check, reset on regression" (`base_stateful_agent.py:400-439, 479-525`).
   - Feed P the checker's violation list in SceneSmith's format (`physics_validation.py:45-56`; `clearance_zones.py:39-45`).
   - Accept an edit only if the checker vector does not worsen, as S1's Pareto rule does.
   - Needs A3 below, because P was never trained to edit.

### (a) Training data and targets
1. **A1. Parent-local child coordinates.** SceneSmith places manipulands in surface-local frames and items in container-local frames (`manipuland_tools.py:557-601, 797-846`).
   - Encode a child's position as an offset in the parent's frame, and decode it back to world coordinates.
   - **[I]** This removes the "copy the parent's pos tokens" shortcut behind the 53 % parent-centre rate. Risk: collapse to offset (0, 0). Check that after a small SFT ablation.
2. **A2. Hierarchical sub-samples for dense rooms.** Following SceneSmith's tree of stages and per-furniture branches (paper Fig. 2; `stateful_manipuland_agent.py:757-858`):
   - split each room into one floor-object sample plus per-parent child samples, which shortens outputs;
   - use the same split for the amendment-018 teacher, since O2's dense-room failures are token-cap truncations (§0 point 3). Alternatively, raise O2's budget in the 21+ bins.
3. **A3. Repair trajectories.** Input: P's failing layout plus the violation text. Target: the S1+projection repair, or an edit list.
   - Built from on-policy samples; amendment 013 already has 9 samples per room.
   - Free to build. It trains the "design change" mode that C4 needs.
4. **A4. Projection-repaired "chosen" in dense rooms.** Amendment 013 uses the S1 repair as chosen, and S1 fixes only 8 % (51+) and 35 % (21–50) of test rooms, so dense rooms rarely yield pairs. Projection-repaired layouts raise that to about 53 % / 67 % (n = 60).
   - **[I]** Risk: repaired layouts may be less "model-like". A median displacement of 0.14–0.19 m is worth reporting next to any result.

### (b) Training objectives and rewards
1. **B1. Feasibility-gated reward.** Analogue of the critic's collision cap (`stateful_critic_agent.yaml:119`): semantic or relation reward counts only when the exact checker passes; otherwise a dense violation penalty. Use the exact checker, not a VLM; NoCritic shows the critic does not buy feasibility.
2. **B2. Distance-to-feasible reward.** Use the projection objective ‖p − p0‖ needed to reach feasibility as a continuous reward or a DPO ranking key.
   - **[I]** It gives signal in 51+ rooms where binary pass is 0 for every sample.
   - At the prototype's speed it is affordable for ≤50 objects. Dense rooms need a vectorised solver before RL.
3. **B3. Explicit sibling co-location penalty** per object. It complements amendment 013's tier order, which already puts co-located duplicates first.

## 4. Do not transfer [V reasons]
- **The VLM critic or GPT loop at 98k-room scale:** about $1.38 M by straight multiplication, and NoCritic matches it on hard metrics.
- **Delete, shrink, regenerate or rescale to resolve collisions** (`designer_agent.yaml:489-495`; `furniture_tools.py:375-403`): this breaks P's fixed object list and prior sizes. It is the same size confound amendment 016 found in the agents.
- **The "bbox overlap ≠ collision" rule:** for boxes, the boxes are the geometry.
- **Soft reachability:** our infinite-task-world A3 check is a hard gate.
- **Placement noise.**

## 5. Why our model is weak, through the SceneSmith lens
- **[V]** P is judged raw, single-shot and greedy. SceneSmith never trusts even GPT-5.2's raw coordinates: every action is checked and everything is projected at the end.
- **[V]** P copies the parent's centre for child positions.
- **[V]** P's S1 cannot untangle dense rooms. A joint solver can untangle many of them.
- **[V]** O2's advantage on the same input is real (pilot 0.92 vs 0.51). Its losses in dense rooms are output-length limits, which hierarchical decomposition addresses.
- **[I]** A bigger or thinking model helps because reasoning acts as internal verification. The cheaper route for an 8B model is external verification: C1–C3 now, A1–A3 next.

Files are in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/better/scenesmith/`:
- `project.py` (prototype), `run60.json` (per-room results), `smoke.json`
- `bins.py`, `coloc.py`, `coloc_gt.py`, `coloc_src.py`
- `scenesmith.txt` (paper text)