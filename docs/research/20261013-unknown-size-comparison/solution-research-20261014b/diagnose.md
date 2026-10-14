# RoomWright (arXiv 2608.18840v1) and how it applies to FastFill arm P

## 0. Bottom line
- **The paper is not a learning method.** "Beyond Placement and Articulation: Usage-Driven Code Scenes for Embodied Interaction" (RoomWright; Xiao, Ye, Hao, Yang, Xie, Yi; SJTU and Meituan; arXiv 19 Aug 2026) is a training-free agentic pipeline built on closed models: GPT-5.4 is the scene agent and Claude Opus 4.8 generates the object code. It has no SFT, no RL, no reward and no training data. One scene takes about 9 h and about 43M tokens.
- **Only one idea transfers to our setting.** The language model emits symbolic relations, and a deterministic geometric solver turns them into collision-free coordinates. A tool-validated critic loop then rejects any edit that adds a collision. The paper does not describe the solver and does not ablate it, so this idea is untested by the paper itself.
- **The rest does not apply to arm P.** Object-set expansion, articulated asset code, trigger–condition–effect interactions and usage-guided manipuland orientation all fall outside P's contract (a fixed object list with given sizes) and outside RoomGenBench's bbox interface.

## 1. Problem (verified, pp. 1–2)
- Code-based scene generators (SceneCode, Code-as-Room) model visual construction and single-object articulation. They do not model how a room is used.
- RoomWright's goal: from a natural-language room description, generate a scene that is entirely code. It contains simulation-ready articulated assets, a metric layout and an executable interaction script (for example, a switch that turns on a lamp). The scene loads into OmniGibson or Isaac Sim.
- Key quote (paper.txt:77–100): "the language model performs semantic and task reasoning, while deterministic geometry and physics modules compute coordinates, orientations, and executable behaviors … allowing the language model to reason about usage while explicit geometry handles numerical coordinates."

## 2. Method (verified, pp. 3–5, Fig. 2, Algorithm 1)
The pipeline has three stages.

**(1) Scene planning**
- It plans the room shell, then expands the named object set O_P into a scene state S = (O, F, G_x), Eq. 3:
  - F is a support forest (which object rests on which surface);
  - G_x is an interaction graph whose edges are e = (s, t, m, φ). The method m comes from a closed vocabulary {binary_switch, multi_knob, binary_knob, binary_remote_button, contact}. The effect φ acts on the light, screen or state channel.
- Algorithm 1 has three steps:
  - zoned completion;
  - Phase 1, task-centric expansion per floor anchor. It names the human task, then admits one-hop companion objects, keeping only the top-k by necessity ("radial, not chained").
  - Phase 2, exhaustive interaction induction for every object.

**(2) Asset generation**
- Articraft is the backend for articulated-object code.
- Each asset carries part annotations A = {(ℓ_i, p_i)} (Eq. 1) and per-link physics π(ℓ) = (m, c, I, μ, C) (Eq. 2): mass, centre of mass, inertia, friction and convex collision geometry.
- Each interaction edge is compiled into a trigger–condition–effect rule. A small generic interpreter runs these rules (App. D, Listings 1–3).
- Textures come from FLUX.2.

**(3) Layout generation**
- A "symbolic layout program expressed in the user's reference frame" contains support relations, user-viewpoint spatial relations and usage-guided orientations. It is "grounded by a geometric solver into collision-free poses" (paper.txt:244–245).
- A vision-in-the-loop critic then looks at rendered views and invokes scene-editing tools. "Each revision is validated by physics-simulation tools, which resolves geometric conflicts and rejects modifications that introduce collisions" (paper.txt:251–252).
- **The solver's algorithm, its objective and its constraint vocabulary are not specified anywhere in the paper**, including the appendix.

**Usage-guided orientation**
- A VLM receives the asset's part annotations plus rendered views. It predicts a frame (f, u), a hand anchor h and an arm anchor a.
- d_use = h − a (Eq. 4) is snapped to the nearest principal box axis orthogonal to up, giving f_use. f_use only initialises the pose.

**Training:** none. **Data:** none, apart from the 28 evaluation prompts.

## 3. Evaluation and numbers (verified)
**Setup**
- 28 prompts: 16 easy, 7 medium, 5 hard, covering 23 room types (App. A, Tables 3–4). Hard prompts describe 15–20 entities.
- Ten SceneEval metrics (Tam et al. 2026, WACV), each reported as a fraction of objects. Collision-free is COL = 1 − (1/N)Σ1_i (Eq. 5), a per-object rate with a perturbation re-check. It is **not** a per-room success rate.
- No seeds, variance, CIs or human study are reported.

**Table 1** (mean over 28 scenes):

| Method | Collision-free | Navig. | In-bound | Opening-clear | Support | Access. | ObjCount | ObjAttr | OO-Rel | OA-Rel |
|---|---|---|---|---|---|---|---|---|---|---|
| RoomWright | **95.9** | 97.3 | **99.7** | 95.7 | 53.6 | **65.6** | **57.0** | **64.6** | 22.4 | 40.6 |
| SceneSmith | 95.0 | **99.6** | 99.7 | **98.0** | **58.4** | 39.8 | 48.8 | 42.5 | **27.0** | 42.2 |
| SceneCode | 91.2 | 99.0 | 81.2 | 96.4 | 26.9 | 26.1 | 38.1 | 30.7 | 16.1 | **55.6** |
| Code-as-Room | 45.5 | 89.5 | 73.3 | 47.2 | 51.9 | 63.1 | 37.0 | 56.7 | 20.5 | 22.2 |

- RoomWright is best on 5 of 10 metrics. Its collision-free lead over SceneSmith is 0.9 points with n = 28 and no uncertainty reported.

**Table 2** (interactive content):

| Method | IO | AO | CHI | MPC |
|---|---|---|---|---|
| RoomWright | 16.4 | 13.2 | 6.4 | 3.6 |
| SceneCode | 4.4 | 4.4 | – | – |
| SceneSmith | 2.4 | 2.4 | – | – |
| Code-as-Room | – | – | – | – |

- App. D: 194 interaction edges over 23 easy and medium scenes. Effect share: state 49 %, light 39 %, screen 12 % (Figs. 7–9).

**Cost** (App. F, Tables 7–8)
- Tokens: 1,204.5M in total; per scene 43.0M on average (27.2M–72.0M).
- Wall-clock per scene: 9:10:38 on average (33,038 s); range 6:34:49 to 12:21:11.

**Ablations:** qualitative only.
- Fig. 5 removes usage-driven reasoning (a bathroom example).
- Fig. 6 removes usage-guided orientation (a keyboard and mouse example).
- There is no ablation of the geometric solver, the critic or the physics validation.

**Stated limitations** (App. G): objects hanging on hooks are not supported, and cloth physics is not supported.

**Limitations I see:**
- the solver is unspecified;
- no density-stratified results;
- per-object metrics only;
- small n with no statistics;
- closed and paid models only.

**Code and data:** the PDF links no repository; the only URL in it is the FLUX.2 blog (checked with pdftotext and grep). The arXiv abstract page links no code, project page or data, and a web search found no repository.

## 4. Our gap in the same frame (numbers I recomputed from local pilot files)
Script: `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/better/roomwright-paper/pilot_bins.py`. It reads the a016 pilot uids and the S1 rows.

Frozen-checker success on the 300 pilot rooms, binned by movable-object count:

| System | 1–10 objects (n = 237) | 11–30 (n = 51) | 31+ (n = 12) | All |
|---|---|---|---|---|
| P raw | 0.603 | 0.216 | 0.000 | 0.513 |
| P + S1 | 0.895 | 0.647 | 0.250 | 0.827 |
| O2 raw | 0.962 | 0.804 | 0.583 | 0.920 |
| L raw | 0.958 | 0.941 | 0.750 | 0.947 |

- The "All" column reproduces findings.md:78–80.
- **Paired, P + S1 vs O2 raw:** O2 passes and P + S1 fails in 39 rooms; the reverse happens in 11 rooms.
- **P's 52 failures after S1:**
  - severe collisions in 28 rooms, main collisions in 31, out-of-bounds in 20, fixed blocking in 8;
  - S1's own diagnosis (`raw_failures.not_xy_fixable`) marks only ≤ 5 rooms as having support_z or ceiling problems. The rest are XY-fixable classes that S1's local push did not solve.
- **Context:**
  - The agent harness gives only parser-error feedback, never geometry (amendment-002:22).
  - L used about 311 reasoning tokens per room (`llm-eval-L-P-pilot-v1/report.md`, Table 3: 93,222 reasoning tokens over 300 rooms).
  - Best-of-8 raises P's raw checker success on dev from 0.400 to 0.600 (findings.md:47).

## 5. Fit with arm P

| RoomWright component | Fits P? | Reason (verified fact, then inference) |
|---|---|---|
| Usage-driven object expansion (Alg. 1) | No | P's contract is "Place every listed movable object exactly once" (P system prompt, requests.jsonl). Adding objects breaks the task. |
| Articulated asset code and interactions | No | RoomGenBench takes one bbox per object, then generates or stretches the asset itself (amendment-017 §1). |
| Usage-guided manipuland orientation | Marginal | The checker does not score yaw, and RoomGenBench builds its own asset, so the front-facing logic has no hook. At most it is a realism or judge-score effect. |
| LLM relations → geometric solver | **Yes, the core idea** | It removes P's weakest skill: emitting globally consistent absolute coordinates for many objects in one pass. P's failure grows with object count (table above), and co-located same-category duplicates are a main failure mode (findings.md:137). Inference: an explicit group relation such as "3 chairs in a row along wall W" lets the solver spread duplicates by construction. |
| Tool-validated critic loop | Yes, as a frozen-checker loop without a VLM | Our frozen checker is the tool. Inference: this is a system-level change, so it must also be applied to the agents to keep comparisons fair, as S1 already is. |
| Paper's absolute performance | No evidence for dense rooms | Its rooms are sparse (hard prompts have 15–20 entities), its metric is per-object, there is no density breakdown, and n = 28. At 43M tokens per scene, running it on our 8,647 test rooms is infeasible, and it needs paid APIs. |

## 6. What we would need to implement (inference and proposals; nothing was run)
We already have some building blocks (verified):
- a relation vocabulary with a checker: on, faces, near, against_wall, between (`fastfill/scene.py:48–50`, `fastfill/validate.py:135` `holds`);
- GT constraint extraction: `fastfill/build.py:347` `extract_constraints`, currently capped at max_k = 4 constraints per room;
- a constraint-preserving snap: `fastfill/interface.py:255` `snap_inside`;
- a local repair (S1): `fastfill/comparison/output_repair.py`. It allows only XY moves; yaw, parent and size are frozen.
- a greedy grid and cardinal-yaw search: `fastfill/comparison/pose_targets.py:209` `_candidates`. Its docstring (line 3) restricts it to an offline teacher, never inference.

Three options, cheapest first:

**A. "S3" global re-placement (CPU only, no retraining; cheapest test)**
- For rooms that still fail after S1: take out the violating support subtrees and re-place them over free floor (polygon minus fixed boxes and door strips), allowing cardinal yaw changes.
- Score candidates on relations inferred from P's own raw output (against_wall, near, faces) to keep its intent.
- This needs a new amendment, because S1 freezes yaw and pose_targets is restricted to offline use.
- Upper bound on the pilot: about 47 of the 52 residual rooms are XY-fixable (≈ 0.98 ceiling vs 0.827 now). This is a ceiling, not an estimate.
- OOR, FID and judge scores must be reported, because a solver can pass the checker with odd placements.

**B. RoomWright-faithful (needs retraining)**
1. Change P's output target from pos/yaw to a relation program. For each object: `on`; an anchor (a wall id, a corner, or a target object); a relation (near, faces, between, beside, in front of); and explicit group or row tokens for repeated categories.
2. Auto-label the about 98k gate-qualified GT rooms by extending `extract_constraints` to cover every object.
3. Write a solver over these relations. Candidates cited by RoomWright itself are Holodeck (Yang et al. 2024) and LayoutVLM (Sun et al. 2025b, differentiable layout optimisation). From prior knowledge, not checked in this session: Holodeck uses a DFS constraint solver.
4. **Gate before any GPU spend:** a CPU round-trip on dev, GT → program → solver. Measure checker pass, relation fidelity and OOR against GT (GT's own checker pass is about 0.72, findings.md:48). If the round-trip loses much below GT, the representation is too lossy.
5. Then SFT Qwen3-8B (or a larger open model) on the programs, and optionally DPO on solver outcomes.

**C. Checker-feedback loop at inference**
- Feed checker violations back to the model and regenerate, mirroring RoomWright's validated edits.
- For a non-thinking 8B model this probably needs multi-turn training data. The existing best-of-N (+0.20 raw on dev) is the cheaper version of the same idea.

**Fairness note:** any solver or loop is a different system class, LLM + solver. The single-shot "LLM in FastFill's slot" comparison must stay as defined (feedback-llm-comparison-protocol), and the solver must be applied to every system's output if we report system-level rows.

Files are in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/better/roomwright-paper/`:
- paper.txt
- pilot_bins.py

Sources:
- [arXiv abstract 2608.18840](https://arxiv.org/abs/2608.18840)
- [PDF](https://arxiv.org/pdf/2608.18840)
- [takara.ai summary](https://tldr.takara.ai/p/2608.18840)
- [SceneCode summary](https://www.opentrain.ai/papers/scenecode-executable-world-programs-for-editable-indoor-scenes-with-articulated--arxiv-2605.19587/)
- SceneSmith (Pfaff et al. 2026), arXiv 2602.09153, as cited in the paper's reference list