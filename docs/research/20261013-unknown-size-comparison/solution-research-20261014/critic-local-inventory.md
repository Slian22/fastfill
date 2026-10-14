**Repo inventory for a box-diorama critic + repair pipeline (FastFill, local SceneSmith at /Users/slian/Desktop/3D/Worldedge/scenesmith)**

Everything here was read only. Nothing was edited, no SSH, no heavy jobs. Unless an absolute path is given, paths are relative to `/Users/slian/Desktop/3D/Worldedge/OptiScene`.

## Main findings

1. **The training set is already filtered by the checker, so "flag, don't filter" does not describe it.** The 98,754 J/P SFT rows only include rooms whose GT passes `check` as valid with zero main, severe and fixed_blocking collisions (`fastfill/comparison/main_data.py:147-164, 171-204`).
   - Release train is 144,150 rooms. 19,307 are excluded by flags (`EXCLUDE_FLAGS`, `fastfill/comparison/data.py:16`: oob_objects, overlapping_furniture, fixed_collision), leaving 124,843 eligible.
   - Of those, 103,316 pass as J (GT unchanged) and 118,058 pass as P (after the pose-only repair). 98,923 pass both; 98,754 remain after the length cut. 25,920 eligible rooms were dropped.
   - Source: `docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-report-20261013.md:43,70`.
2. **The model fails mostly on its own, not because of bad labels.**
   - On the 6,252 test rooms whose GT passes, J still only reaches 0.44 (`results/report-v2-JPL-002/S1.csv`).
   - Its worst sources have nearly clean GT: MansionWorld GT 0.976 vs J 0.20, SAGE-10k 0.869 vs 0.055, InternScenes_gen 0.962 vs 0.30 (T2/T3.csv).
   - So repairing outputs at inference is the main lever. Cleaning the training data can only add rooms back (the 25,920 dropped) or catch problems the checker cannot see.
3. **A repair solver that keeps categories, sizes and support already exists:** `fastfill/comparison/pose_targets.py`. It is greedy, moves only position and yaw, carries whatever stands on an object along with it, and accepts a room only when the frozen `check` passes. It succeeded on 118,058 of 124,843 eligible train rooms (with estimated sizes), and the whole corpus ran in 3h36m (research-log line 30).
4. **The box diorama exists only as a published HTML page, not as repo code.**
   - Page: `/Volumes/harddisk/fastfillout/unknownsize-20261013/viz-3d-20261014/fastfill-layouts.html`, built on three.js r128 from cdnjs.
   - Its data file `viz-data.json` stores, per system, the objects plus checker flags: `oob, collision, collision_pairs, blocking, blocked_fixed, support, ceiling`.
   - The template and extractor exist only in the session scratchpad (`scratchpad/viz/template.html`).
5. **The open model O2 (Qwen3.8-27B, text only) already reaches 0.928 geometry success** on the 1,454-room cohort, against 0.935 for gpt-6.1-sol (`results/agents-sft-paired/O2/report.md`). A repair agent on Isambard without API cost is therefore realistic.

## 1. Rendering

| Asset | Path:line | Notes |
|---|---|---|
| three.js box diorama (the reference look) | `/Volumes/harddisk/.../viz-3d-20261014/fastfill-layouts.html`; template `scratchpad/viz/template.html` (`buildScene` 297, `fit` 334, `renderThumbs`/`toDataURL` 393-401, `kindOf` 224) | Colours: floor blue, on-object green, collision red, OOB orange, door-blocking/support/ceiling purple, fixed tan, blocked door dark orange. Browser only; playwright is not installed. |
| Blender renderer | `fastfill/comparison/render_assets.py` (cut-away walls `make_room` 484-500, `room_camera` 183 with perspective and `top` views, `render_layout` 570) | Already draws `procedural_structure` boxes as cubes (577-582). Everything else uses real meshes; there is deliberately no box fallback for furniture. Cycles on CPU, about 22.6 s per room for 6 images (research-log 51). Blender is installed locally. |
| Isometric SVG and GLB boxes | `fastfill/v2/bbox_visualization.py` (`glb_bytes` 28, `preview_svg` 94) | Pure Python, colour per category. No walls, doors or labels. Faces are painter-sorted, so overlapping boxes can draw wrong. Uses the v2 schema. |
| 2D top-down plan | `fastfill/tools/visualize.py:26` `draw_layout` | matplotlib (installed): boundary, doors/windows, boxes, ids, front arrows. |

## 2. Checker outputs per object (`fastfill/validate.py`)

- `check()` (line 72) returns:
  - id lists: `oob` (79), `ceiling` (81), `support_fail` (122)
  - collision pairs: `collisions.{main,severe,tuck,contained}` (95-97)
  - `fixed_collisions` (107) and `fixed_blocking` (110, door/column/stair; windows excluded)
  - `oor` and `valid` (131)
- `collision_kind` (55) decides tuck exemptions (seat×table, bed×nightstand) and marks same-category boxes overlapping more than half their union as `main` duplicates.
- `fastfill/comparison/geometry.py:34-47` `geometry()` gives the "geometry_success" verdict: valid, no main, no severe, no fixed_blocking.
- The backend-strict gate is `fastfill/interface.py:373-400` `validation()`: `outside_1mm` (383-385), plus keep-outs and hard constraints. It runs on the v1 request schema.
- Magnitudes are not returned: no penetration area, OOB distance or overlap fraction.

## 3. Existing repair and solver code

| Code | Path:line | Keeps categories and sizes? | Notes |
|---|---|---|---|
| Pose-only greedy teacher | `fastfill/comparison/pose_targets.py:293` `repair(room, sizes, max_candidates=200, max_passes=4)`; candidates 180-221; search 239-274; acceptance 277-290 | Yes, plus ids and support | Accepts only valid + zero main/severe/fixed_blocking. Cardinal yaws plus grid/wall positions; not a proof of feasibility. Rejects fixed parents; at most 256 objects (`MAX_OBJECTS`, line 41); support-forest preprocessing is O(N³) (`evidence/agency-unknownsize-pose-review-20261013.md`). The docstring forbids inference post-processing inside the frozen comparison. 29 tests in `fastfill/comparison/tests/test_pose_targets.py`. |
| Wall snap | `fastfill/interface.py:255` `snap_inside`, `_push` 276 (`SNAP_TOL` 0.10, line 71) | Yes | Fixes walls only. It slightly lowers GT success (0.723 → 0.713, T2.csv `gt_success_snapped`). |
| Small repair | `fastfill/comparison/geometry.py:56` `small_repair` | Yes | `snap_inside` plus a 10 cm rollback. Repaired J success is about the same as raw (0.4264 vs 0.4305). |
| v1.1 MTV/spiral de-collision | `fastfill/v1_1/postprocess.py:257` `repair_v1_1` | No: clamps sizes | Gave 63-66 % zero-collision acceptance on v1.1. The agency review says not to reuse it unchanged (`agency-unknownsize-data-review-20261013.md:69`). |
| Text critic feedback | `fastfill/v2/llm_baseline.py:100` `problems()`; harness loop at 207 | n/a | Turns OOB, overlap and floating objects into sentences for re-asking. Uses an axis-aligned approximation, not `check`. |
| Pose-only repair hook | `fastfill/v2/runtime.py:338` `run_pipeline(repair=...)`; `_repair_is_pose_only` 300 | Enforced | v2 schema. |
| growing_world repair | `/Users/slian/Desktop/3D/Worldedge/fastfill/vendor/scenesmith/growing_world/fastfill/repair.py:550` | No | Different schema, and it can drop surface objects (409-464). |

## 4. API / VLM harness and cost

- **`server_api_judge.py`** (`docs/research/20261013-unknown-size-comparison/api-smoke-multiview/server_api_judge.py`):
  - Stdlib only. Calls the streaming `/responses` endpoint with `input_image` data URLs (279-282).
  - Hard-coded 3-score instructions (68-75); `MAX_IMAGES=16` (67); `max_output_tokens` 4096 (282).
  - Write-once reservations, no retries. A two-view (top + perspective) smoke run exists.
- **`fastfill/comparison/judge_batch.py`:** hard caps `CAP=348` and `SYSTEMS_CAP=800` (51-52); 90 s timeout per case (515, 589); one subprocess per case; no retries.
- **`fastfill/comparison/llm_agent.py`:** reasoning effort medium; `REPAIRS=2` (67), but the repair rounds only return parser errors, never geometry problems (362). CLI defaults: concurrency 4, max-calls 4800, timeout 900 (507-508). There is a fsynced call ledger.
- **Measured cost of L** (gpt-6.1-sol, text only, 1,454 rooms; `results/report-v2-JPL-002/report.md:186`): 1,469 calls, 7.24 M input / 0.86 M output / 0.35 M reasoning tokens, median latency 13.5 s. That is about 4.9k tokens in and 0.6k out per call.
  - No image-token or dollar figure is recorded anywhere.
- **`fastfill/comparison/llm_open.py:57-90`:** O2, O3 and O6 are multimodal models, but images are turned off (`limit_mm_per_prompt image:0`, lines 64/67/83). Open-model geometry success: O2 0.928, O4 0.687, O6 0.701, O3 0.241, O1b 0.110.

## 5. Training data format and scale

- **IR rooms** (`fastfill/scene.py`): boundary, boundary_type, height, fixed, and objects with id, category, size, bottom-centre pos, yaw in radians, parent, anchor.
  - Local copies: `/Volumes/harddisk/fastfill_ir/<source>.jsonl`, and `/Volumes/harddisk/fastfill_data/{v3,v3.1}` (rows are `{uid, source, flags, messages}`, written at `fastfill/build.py:559`).
  - v3.2 is on Hugging Face (`liantian/fastfill-v3`). `main-data-v1`, the 98,754 rows actually trained on, exists only on Isambard, together with the rejection-reason ledger.
- **Flags:** computed at `fastfill/build.py:250-284`.
- **Admission rule:** `fastfill/comparison/main_data.py:79-89` and `147-204`.

Per-source counts (from `docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-geometry-summary.json`; "test GT pass" is from T2.csv):

| Source | Train | Flag-excluded | Eligible | Qualified | Rejected | Test GT pass |
|---|---:|---:|---:|---:|---:|---:|
| SpatialLM | 42,068 | 8,987 | 33,081 | 24,060 | 9,021 | 0.60 |
| MansionWorld | 19,396 | 37 | 19,359 | 18,343 | 1,016 | 0.98 |
| IL3D_3dfront | 18,054 | 3,534 | 14,520 | 9,237 | 5,283 | 0.49 |
| Structured3D | 12,700 | 2,845 | 9,855 | 7,982 | 1,873 | 0.62 |
| InternScenes_gen | 12,552 | 233 | 12,319 | 11,870 | 449 | 0.96 |
| SAGE-10k | 8,977 | 30 | 8,947 | 5,736 | 3,211 | 0.87 |
| OptiScene_holodeck | 8,946 | 0 | 8,946 | 8,946 | 0 | 1.00 |
| IL3D_synthetic | 6,277 | 21 | 6,256 | 6,090 | 166 | 0.95 |
| InteriorGS | 4,877 | 1,562 | 3,315 | 2,458 | 857 | 0.56 |
| InternScenes_arkit | 4,528 | 576 | 3,952 | 2,098 | 1,854 | 0.58 |
| HSSD200 | 1,876 | 520 | 1,356 | 956 | 400 | 0.53 |
| InternScenes_scannet | 1,309 | 380 | 929 | 268 | 661 | 0.14 |
| InternScenes_mp3d | 1,258 | 282 | 976 | 562 | 414 | 0.61 |
| InternScenes_3rscan | 1,200 | 237 | 963 | 297 | 666 | 0.17 |
| MultiScan | 114 | 56 | 58 | 15 | 43 | 0.50 |
| Scan2CAD | 18 | 7 | 11 | 5 | 6 | 0.00 |
| SceneSmith / SpatialGen | 0 (test-only, held out) | – | – | – | – | 0.78 / 0.64 |
| **Total** | **144,150** | **19,307** | **124,843** | **98,923** | **25,920** | **0.72** |

## 6. RoomGenBench contract (`RoomGenBench/bench/CONTRACT.md`)

- **Input per scene** (`bench/inputs/scenes/*.json`): `room` (dimensions, walls, ceiling_height), and per object id, type, description, asset_key, position (bottom-centre), rotation.z in degrees, dimensions {width, length, height}, place_id (floor / wall / supporter).
- **Assets:** a GLB with Y up and front facing +Z. The assembler fits the asset's bounding box to the dimensions anisotropically.
- **Scoring:** `bench/score_geom.py` (F-score/Chamfer and ray-cast support) and `bench/score_vlm.py` (Qwen2.5-VL room/asset VQA). There is no layout-collision gate in RoomGenBench itself; the 1 mm / 1 cm refusals come from EmbodiedGen's `FastFillBackend` (see memory).
- **Adapter:** `fastfill/v2/direct_layout.py:235` `layout_to_roomgenbench` and `fastfill/v2/roomgenbench.py` expect the v2 schema. The comparison J/P output (`{placements:[id,size,pos,yaw°,on]}`) is not wired to them, and J/P inputs carry categories only, with no `description`.

## 7. SceneSmith (local)

- **Workflow:** a planner, a designer and a critic agent (`scenesmith/agent_utils/base_stateful_agent.py:94, 235-330, 663-720, 762-806`), using gpt-5.2 with vision detail "high" (`configurations/furniture_agent/base_furniture_agent.yaml:254-262`).
- **Critic tools:** `observe_scene`, `get_current_scene_state`, `check_facing_tool`; physics results are injected as context (`scenesmith/furniture_agents/stateful_furniture_agent.py:114-135`).
  - Prompt: `scenesmith/prompts/data/furniture_agent/stateful_critic_agent.yaml`. Any physics collision caps the realism score at 4 (line 119); the rest covers door clearance, storage against walls, and access in front of storage.
  - Output: `FurnitureCritiqueWithScores`, six 0-10 scores each with a comment (`scenesmith/agent_utils/scoring.py:80-109`).
- **Renders:** one 1024 px top view plus four 512 px side views, with set-of-mark id labels, blue bbox wireframes, direction arrows and partial walls, made in Blender EEVEE through a render server (`base_furniture_agent.yaml:25-60`).
- **Loop control:** `max_critique_rounds: 3`, early finish when every score is at least 9, and a reset to the checkpoint if any single score drops by 2 or the total drops by 1 (`base_furniture_agent.yaml:244-252`).
- **Designer tools:** in `scenesmith/furniture_agents/tools/furniture_tools.py` — `move_furniture_tool(x,y,yaw)` (311), `add` (273), `remove` (347), `rescale` (376), `generate_assets` (218). In `scenesmith/furniture_agents/tools/scene_tools.py` — `snap_to_object_tool` (112). To keep categories and sizes unchanged, expose only move, snap and facing.
- **Deterministic final step:** `scenesmith/agent_utils/physical_feasibility.py:516` `solve_non_penetration_ik` (Drake IK with SNOPT, rotation and optionally z fixed) and `:1548` `apply_physical_feasibility_postprocessing`. Collisions come from Drake on meshes (`scenesmith/agent_utils/physics_validation.py:155`).
- Drake, the Blender server and meshes make this too heavy to reuse as code. Port the pattern; `pose_targets` already plays the role of the projection step.

## 8. Gaps

1. **No batch box renderer in the repo.**
   - Lazy option: render a matplotlib top view with ids and arrows (`tools/visualize.py`) plus one oblique view, coloured by `check` flags.
   - Alternatives: a Blender cube mode by extending `render_assets.py:577` with EEVEE/Workbench, or headless Chrome on the three.js template (needs playwright). The template and extractor should be committed.
2. **`check` gives ids but no magnitudes,** which the critic prompt and repair ordering need.
3. **Tolerance mismatch.**
   - `check`, and so `pose_targets` acceptance: 10 cm OOB tolerance.
   - Backend: refuses anything 1 mm outside the room or 1 cm into a door/window keep-out.
   - Fix: add the `interface.validation`-style `outside_1mm` and keep-out gate to repair acceptance; that function currently runs on the v1 schema.
4. **`pose_targets` limits:**
   - no fixed parents
   - at most 256 objects
   - O(N³) preprocessing
   - no wall-clock bound: R×C `check` calls, each O(n²); SAGE rooms reach 87 or more objects
   - cardinal yaws only
   - the frozen comparison forbids it as inference post-processing, so a repaired arm needs a new amendment and raw vs repaired reported separately.
5. **No critic output schema.** The judge instructions are fixed to three scores. A per-object action schema (move / rotate / reparent only, never add, remove or resize) is needed.
6. **No image-token or dollar cost recorded,** and API usage is governed by the 348/800 caps. A VLM pass over 98k rooms needs new authorization or an open VLM with images enabled in `llm_open`.
7. **No J/P → RoomGenBench/EmbodiedGen adapter,** and no per-object `description`.
8. **Per-source rejection reasons** (J fail vs P-teacher fail) exist only in the server ledger.

## 9. Recommendation

- **(b) Repair model outputs at inference — primary.** Pipeline: model output → `check` plus the 1 mm / keep-out gate → `pose_targets.repair` with the predicted sizes frozen → only if that is rejected or the room looks semantically wrong, a VLM critic on the box renders with move/rotate-only actions → re-check, and keep the change only if the checker counts do not get worse (SceneSmith's reset rule) → hand off.
  - The deterministic step alone targets the dominant failure mode (gross collisions, duplicates, OOB).
- **(a) Clean the training data — secondary.**
  - Reuse `pose_targets.repair` with GT sizes to bring back the about 21.5k rooms where GT fails as J (SAGE-10k and real-scan sources are heavily rejected).
  - Use (model output → repaired) as chosen/rejected DPO pairs.
  - Run the VLM critic only on a sample (functional/semantic issues the checker cannot see), not on all 98k rooms.