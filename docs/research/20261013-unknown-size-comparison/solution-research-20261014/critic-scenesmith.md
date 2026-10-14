# SceneSmith critic/repair loop: what it does and what we can reuse for FastFill

## Bottom line

SceneSmith's critic does not make scenes collision-free or keep objects in bounds. Drake physics checks, IK projection and simulation do that.

In the paper's own ablation, the run without the critic is equal or better on every hard metric. It costs a third as much. The critic mainly adds objects. For our failures (gross collisions, out-of-bounds), the part to copy is the deterministic Drake projection step, not the vision-language critic. I recommend:

- **(a) Training data:** deterministic repair plus flags, no critic loop at 98k-room scale. A critic only for sampled audit or DPO pairs.
- **(b) Inference outputs:** deterministic projection is required. The backend's 1 mm / 1 cm tolerances cannot be met by a vision-language model. Add an optional, restricted critic afterwards for semantic issues such as facing direction.

File paths below are relative to `/Users/slian/Desktop/3D/Worldedge/scenesmith`.

## 1. Agent structure (furniture stage)

- **Three agents: planner, designer, critic.** Built in `scenesmith/furniture_agents/stateful_furniture_agent.py:280-286`. They use `openai-agents` (`pyproject.toml:29`) with model `gpt-5.2`, reasoning effort high for designer and critic, low for the planner (`configurations/furniture_agent/base_furniture_agent.yaml:255-268`). The paper uses GPT-5.2 for every agent (arXiv 2602.09153, App. A.2).
- **Pipeline stages:** floor_plan, then furniture, then wall_mounted, then manipuland (`configurations/experiment/base_experiment.yaml:27-33`). Each stage has its own designer/critic prompts under `scenesmith/prompts/data/*_agent/`.
- **Planner tools:** `select_placement_style`, `request_initial_design`, `request_critique`, `request_design_change(instruction)`, `reset_scene_to_checkpoint(reason)` (`scenesmith/agent_utils/base_stateful_agent.py:560-725`).
- **Critique is passed verbatim.** The planner prompt says: "Pass the critic's feedback VERBATIM to the designer… The critic is the authority" (`prompts/data/furniture_agent/stateful_planner_agent.yaml:125-128`).
- **Memory:** both agents keep SQLite sessions with turn trimming and summarisation. Only the last turn keeps its images (`base_furniture_agent.yaml:5-22`).

## 2. What the critic sees

- **The critic only reads; it never edits.** Its tools are `observe_scene`, `get_current_scene_state` and `check_facing_tool` (`stateful_furniture_agent.py:114-135`). It is forced to call `observe_scene` first (`base_stateful_agent.py:294-297`).
- **Renders:** Blender EEVEE (`bpy==4.5.4`). One 1024² top-down view plus four 512² corner views at 30° elevation (`base_furniture_agent.yaml:24-39`; `scenesmith/agent_utils/blender/view_generation_mixin.py:57-123`). Overlays on every view: object-name labels, blue wireframe bounding boxes, cyan forward-direction arrows, walls hidden toward the camera (`yaml:55-68`), plus coordinate-grid markers (`coordinate_grid_mixin.py:42-61`). This is very close to the box diorama the user wants. The direction arrows matter for boxes, because a box has no visible front.
- **Scene-state text:** JSON per object with `object_id`, `description`, `position_x/y`, `rotation_degrees`, and `dimensions` (w/d/h) (`furniture_agents/tools/response_dataclasses.py:89-139`). The tool docstring promises `world_bounds`, but the dataclass does not include them (`scene_tools.py:62-63`).
- **Physics and reachability are injected as text, not discovered by the critic** (`base_stateful_agent.py:780-801`; `stateful_furniture_agent.py:387-403`). The runner prompt says: "Visual bounding box overlap does NOT indicate collision… Always trust the physics validation data over visual interpretation" (`stateful_critic_runner_instruction.yaml:72-75`). For our box layouts that line has to be inverted: the boxes are the geometry.

## 3. The critic's rubric and output (quoted)

- **Output format:** `FurnitureCritiqueWithScores` = a free-text `critique` plus six `CategoryScore(name, grade:int 0-10, comment)` fields: realism, functionality, layout, holistic_completeness, prompt_following, reachability (`scenesmith/agent_utils/scoring.py:15-108`).
- **Fixes are prose, not edit operations.**
- **Key rules in `prompts/data/furniture_agent/stateful_critic_agent.yaml`:**
  - "**COLLISION POLICY**: The presence of ANY physics-validated collisions caps the overall realism score at maximum 4/10" (lines 119-121).
  - Storage furniture belongs against walls (128-131).
  - Nothing may stand in front of wardrobes or cabinets (142-146).
  - Scale checks against bounding-box dimensions, with a `rescale_furniture` recommendation (175-222).
  - Functionality "3 or below" for chairs facing away from tables, TVs facing walls, and similar (436-441).
  - "While ANY critical issue exists… Focus EXCLUSIVELY on critical issues" (325-340).
  - "Tool validation overrides visual assessment" for facing direction (102-105).
- **Unsuitable for us:** "Underfurnished rooms (MAJOR issue)… Recommend specific furniture categories to add" (154-158), and the completeness and prompt-following scores (454-630). These push the designer to add or remove objects, which breaks the user's "keep categories unchanged" requirement.

## 4. Edit tools (the designer's)

All in `scenesmith/furniture_agents/tools/furniture_tools.py`:
- `generate_assets(object_descriptions, short_names, desired_dimensions, style_context=None)` (lines 217-270)
- `add_furniture_to_scene_tool(asset_id, x, y, yaw=0.0)` (272-308)
- `move_furniture_tool(object_id, x, y, yaw=0.0)`: z=0, roll=pitch=0, so only floor position and rotation can change (310-344)
- `remove_furniture_tool(object_id)` (346-359)
- `rescale_furniture_tool(object_id, scale_factor)`: uniform, and applied to the asset, so every instance changes (375-393)
- `list_available_assets()` (361-373)

In `furniture_agents/tools/scene_tools.py`:
- `snap_to_object_tool(object_id, target_id, orientation="toward"|"away"|"none")` (111-154)
- `check_facing_tool(object_a_id, object_b_id, direction)` (72-109)

In `furniture_agents/tools/vision_tools.py`:
- `check_physics()` and `check_reachability()` (96-124)

Caveats:
- Moves only check that the object's centre is inside the room rectangle (`furniture_tools.py:118-144, 594-601`).
- Moves add placement noise. The "perfect" profile is 1 mm / 0.1° (`:657-661`; `yaml:351-361`).
- The designer must finish with zero collisions (`designer_critique_instruction_stateful.yaml:25-26`).
- The manipuland stage has `resolve_penetrations` turned off, with the comment "Disabled: 0% success rate in experiments" (`manipuland_agents/tools/manipuland_tools.py:984`).

## 5. Loop termination

- At most 3 critique rounds (`base_furniture_agent.yaml:244`).
- Mandatory early stop when all six scores are ≥9 (`:252`; planner prompt `:110-115`), or when issues persist after 2+ fix attempts.
- Reset to the previous checkpoint is suggested if any single score drops ≥2 or the total drops ≥1 (`yaml:248-249`; `base_stateful_agent.py:400-439`).
- At the end, the scene is reset automatically to the previous checkpoint (N-1) if the final critique regressed (`base_stateful_agent.py:479-525`).

## 6. Physics checks and the deterministic repair step

- **Collision check:** Drake signed-distance queries. Reported above 1 mm penetration; 5 cm floor tolerance (`yaml:300-312`; `agent_utils/physics_validation.py:155-290`).
- **Openings and other checks:** door clearance is an AABB-vs-clearance-zone test (`agent_utils/clearance_zones.py:400-459`). Window, open-connection and wall-height checks are aggregated in `agent_utils/physics_tools.py:157-253`.
- **Reachability:** 2D occupancy with shapely, robot footprint 0.3 m (`agent_utils/reachability.py:44-124`; `yaml:324-326`).
- **Post-processing projection:** IK non-penetration solve with SNOPT, rotation fixed, 2 cm influence distance. Then a 15 s forward simulation that removes furniture tilted more than 45° (`base_experiment.yaml:74-113`; `agent_utils/physical_feasibility.py:516-560, 1548-1665`; `experiments/indoor_scene_generation.py:563-606`).

## 7. Paper results: cost, ablations, failure modes

Source: https://arxiv.org/html/2602.09153.

**Table 14** (179 room-level scenes):

| Run | #Obj | COL ↓ | STB ↑ | OOB ↓ | NAV ↑ | ACC ↑ |
|---|---|---|---|---|---|---|
| NoCritic | 54.0 | 0.8 | 96.1 | 0.1 | 99.0 | 85.8 |
| SceneSmith (full) | 71.1 | 1.2 | 95.6 | 0.2 | 97.6 | 83.4 |

- **Human study (Table 1):** NoCritic vs full is 51.8% realism and 47.5% faithfulness, neither significant.
- **Paper's own summary (§4.2):** "similar preference scores while being 70% cheaper… 24% fewer objects."
- **Cost (App. O.5, Tables 18-19):** $13.98 per scene on average (max $101.93) vs $4.24 for NoCritic. The designer is 75.9% of cost, the critic 20.0%. Time is 3:26 vs 1:40, measured with 25 scenes running concurrently on 8 L40S GPUs; the units of "3:26" are unconfirmed.
- **Residual collisions:** 3.77 mm mean penetration depth (Table 17).
- **Failure modes (App. S):** wrong articulated retrieval, VLM orientation errors, a TV baked into the TV stand, missed support surfaces. Failure rates are not quantified.
- **Not reported:** round counts, score trajectories, how critic scores correlate with human judgements.

## 8. Is our "SceneSmith" data source from this work?

Yes, with high confidence; I did not check the HF dataset's upstream provenance.
- It is downloaded from HF `imChuling/3D_Room_Collections/SceneSmith_exported` (OptiScene `fastfill/download.py:28`).
- The adapter references SceneSmith's `room.py to_drake_directive` and the Drake model-directive (DMD) export, and describes "179 Room scenes + 31 Houses… the paper's evaluation scenes" (`fastfill/adapters/scenesmith.py:1-24`). 179 matches Table 14 exactly.
- The `scene_087` naming matches SceneSmith's `scene_XXX` output directories (`base_experiment.yaml:6-11`).

**Correction to the task context:** in FastFill, SceneSmith is evaluation-only, not training data. See `EVALUATION_SOURCES = {"SceneSmith", "SpatialGen"}` (`fastfill/v2/multisource_data.py:44`), `fastfill/v2/README.md:325`, and the 0/0/364 split (`fastfill/README.md:117`). I did not re-check v3.2 manifests.

## 9. What it would take to run its critic/repair on our layouts

**Dependencies:**
- Python 3.11, `bpy==4.5.4`, `drake>=1.47`, `openai-agents<0.7`, shapely, trimesh (`pyproject.toml:6-44`), and `OPENAI_API_KEY` for GPT-5.2.
- The asset backends (SAM3D, HSSD, ArtVIP) are only needed for `generate_assets`, which we would drop.
- Not verified: SNOPT bundled in the Drake pip wheel; `bpy` and Drake wheels for aarch64 on Isambard.

**Inputs to construct:**
- A `RoomScene` with `RoomGeometry`. Rooms are axis-aligned rectangles only: `RoomSpec` / `RoomGeometry` have just `width` and `length` (`agent_utils/house.py:544-568, 617-659`). Our non-rectangular floor polygons are not supported in the bounds check, the reachability floor or the walls.
- Openings with clearance boxes (`ClearanceOpeningData`).
- Per object: `SceneObject` with `transform`, `geometry_path` (GLTF for rendering), `sdf_path` (collision geometry), `bbox_min/max` and `description` (`agent_utils/room.py:446-497`). Boxes make both files trivial to write.

**Patches needed:**
1. Remove `generate_assets`, `add` and `remove` from the designer; `rescale` only if size repair is allowed.
2. Strip the completeness and "add furniture" rubric sections.
3. Invert the "bbox overlap ≠ collision" rule, keeping our tuck exemptions.
4. Add `on_object` and wall items. The furniture agent handles floor items only (`stateful_planner_agent.yaml:45-48`).
5. Possible frame mismatch: reachability assumes the floor spans (0,0)–(L,W) (`reachability.py:127-134`), while the move tool and renderer assume a room centred on the origin (`furniture_tools.py:128-134`; `vision_tools.py:62-72`). Not checked at runtime.

**Cheaper path:** reuse the design, not the code.
- Our box-diorama renderer with SceneSmith-style labels and direction arrows.
- A `judge_batch`-style critic that outputs structured edits (id, x, y, yaw), restricted to existing ids.
- Our `validate.py check()` after each edit, plus reset-on-regression.
- Before all of this, a deterministic projection: Drake IK, or a 2D separating solver with `snap_inside`.

**Scale, by straight multiplication of the paper's per-scene cost:**
- Full loop: about $1.38M for 98,754 rooms.
- Critic only: about $275k.
- Our rooms have fewer objects than SceneSmith's ~71, so real costs would be lower, but VLM review of the whole training set is still impractical. It fits a sample audit or DPO "chosen" generation, where the repaired layout is chosen and the original rejected.