# FastFill literature sweep: how to make P's layouts physically valid, especially in dense rooms (2024–2026)

## 0. Bottom line

Facts below are marked **[V]** when I checked them in our files or a primary source. **[I]** marks my own inference.

1. **[V]** The gap to O2 is not about how often the answer is complete. In the size-fixed pilot, O2's single raw attempt was complete in 240 of 300 rooms, and 238 of those 240 pass the checker (0.99). Its 0.920 "harness" score only retries on parse or exception errors, not on checker failures. P is complete in 300 of 300 rooms but passes only 0.513.
   - Source: `server-pulls-016/s1-016-size-fixed/agent-O2-P-pilot-{raw,harness}/summary.json`.
2. **[V]** The pilot cannot show that O2 or gpt-6.1-sol (L) is better in dense rooms. It has 279 rooms with ≤20 objects, 17 with 21–50 and 4 with >50 (`a016-pilot-v1.json` "counts").
   - Rooms with 31–50 objects (n=8): O2-harness 0.50, L 0.75.
   - Rooms with 51+ objects (n=4): both 0.75.
   - In rooms with 1–10 objects, P already trails badly: on the full test P scores 0.569 raw (`test-apply-012/.../placement-rpo-dpo1/summary.json` by_stratum), against O2-harness 0.962 and L 0.958 on the pilot. These are different room sets, so the comparison is approximate.
3. **[V]** Three structural differences between P and the strong agents appear in our own code:
   - **No reasoning for P.** P emits JSON only (`fastfill/comparison/contract.py:48-50`; v1 protocol explicitly has "no reasoning block", `fastfill/scene.py:27`). O2 runs with thinking on (`comparison/llm_open.py:72-73`) and gets a 16,384-token thinking allowance (`llm_open.py:65`).
   - **Random object order.** Objects are listed, and must be output, in an HMAC-random order (`contract.py:166-171`, enforced at `:252,:286`). Training uses the same order (`comparison/main_data.py:21,174,297`). So P often has to place small items and children before their parents, and large furniture last.
   - **S1 is only a local repair.** It is a greedy pairwise push in XY with no yaw change (`comparison/output_repair.py:3-31`). On test it lifts rooms with 31–50 objects only from 0.013 to 0.258, and rooms with 51+ objects from 0.000 to 0.084.
4. **[I]** The literature ranks the levers as follows:
   - **Most direct:** RL with a verifiable geometric reward on top of SFT, ideally after a reasoning or structured-plan SFT. This combination appears in 3D-Layout-R1, LayoutDSL, SPG-Layout, MetaSpatial and ThinkBLOX.
   - **Cheapest, no retraining:** a global non-penetration solver in place of, or after, S1. Examples are LayoutVLM's projected gradient descent and Steerable Scene Generation's minimum-displacement projection.
   - **Cheap, needs one SFT re-run:** list objects large-first and parent-first. Ordering ablations in SPG-Layout and ThinkBLOX support this.
   - **Bigger changes:** a relational placement language instead of raw coordinates (LayoutDSL, imperative scene programs), and step-by-step placement with feedback (ThinkBLOX, ScenePilot, FuncRoom-Agent).

## 1. RL with verifiable geometric rewards (closest to our setup)

| Method | Core idea | Reported gains (exact) | Cost / code | Relevance to us |
|---|---|---|---|---|
| **3D-Layout-R1** (NVIDIA, Mar 2026) [arxiv.org/abs/2603.22279](https://arxiv.org/abs/2603.22279) | SFT on DeepSeek-R1 reasoning traces, then GRPO. Reward = IoU + λ₁·(1 − colliding pairs / N) + λ₂·format. | Qwen3-8B sorting task, collision-free: SFT on answers 0.700 → reasoning SFT 0.854 → plain GRPO 0.954 → reasoning SFT + GRPO 1.000. Mean IoU 0.552 → 0.924. | SFT on 16×A100, RL on 8 GPUs for 15 epochs. No code link found. **[V]** | Same base family and size as P. It is an editing task with few objects, not dense rooms. Strongest evidence that answer-only SFT leaves collisions that reasoning SFT plus GRPO removes. |
| **LayoutDSL** (Jul 2026) [arxiv.org/abs/2608.07547](https://arxiv.org/abs/2608.07547) | Qwen3-4B with LoRA outputs DSL actions (`place/mount/hang` + anchor + 26 alignment types + distance). A deterministic interpreter turns them into coordinates. GRPO on equal-weight collision, OOB, floor-plan, reachability and design-logic rewards. | Mean score: base 0.357 → SFT 0.685 → SFT+RL 0.878. Collision score 0.967. Ablation DSL vs raw parameters: SFT 0.685 vs 0.647, after RL **0.878 vs 0.749**. Without the anti-hacking term the model drops objects (0.825). | SFT on 9,035 rooms; GRPO on 1,000 rooms × 8 samples; 8× RTX 4090. No code. **[V]** | GRPO here is about the same compute as our DPO stages. Relational output beats raw coordinates. Object-dropping hacks are impossible for us because the contract requires each id exactly once. |
| **SPG-Layout** (Jul 2026) [arxiv.org/abs/2607.02407](https://arxiv.org/abs/2607.02407) | Qwen3-4B full SFT (100 epochs), then GRPO (10 epochs, group size 8). Reward = 0.3·exp(−E/σ) on voxel overlap/OOB energy + 0.3·semantic + 0.4·statistical spatial prior. **Places large → medium → small objects.** | Volume-violation score (VBL, ×10³, "all" split): LayoutVLM 941.8, ReSpace 310.3, SPG 29.3. Ablation avg_LV: full 19.5, **without the ordering 132.7**, without geometry reward 78.8. LoRA reached 78% format success vs 99% for full fine-tuning. | Code "will be released"; compute not reported. **[V]** | Direct evidence for ordering and a smooth overlap-energy reward. The LoRA-vs-full finding is a warning, since P is LoRA. |
| **MetaSpatial** (ICLR 2026) [arxiv.org/abs/2503.18470](https://arxiv.org/abs/2503.18470) | Multi-turn refinement trajectories trained with GRPO. Its "3D-SPO" applies physics penalties at object level, on coordinate tokens only. Reward = format + 0.2·collision ratio + 0.2·constraint violations + GPT-4o render score. | Qwen2.5-VL-7B collision rate 38.2% → 11.5%. SFT only 30.5%; SFT + RL 13.4%. | Code: [github.com/PzySeere/MetaSpatial](https://github.com/PzySeere/MetaSpatial). The paper's own ablation reports 27.4% for the same setting, an internal inconsistency I flag. **[V]** | Shows per-object credit assignment, which matters for 50-object rooms. |
| **ThinkBLOX** (Jul 2026) [arxiv.org/abs/2607.13539](https://arxiv.org/abs/2607.13539) | Qwen2.5-VL-7B places one object per step, conditioned on the current partial layout plus a render. RL uses "Tier-Decoupled GDPO": hard / soft / consistency reward tiers normalised separately. Collision reward = −tanh(5·overlap). | Collision-free (CF) / overall score (PSA): one-shot SFT without reasoning 39.4 / 8.4 → SFT with reasoning 63.5 / 21.9 → full 81.6 / 54.6. **Plain SFT+GRPO: CF 82.6 but in-boundary 36.2 and PSA 8.65.** Placement order (PSA): plan-based 54.6, size-based 47.8, random 41.6. Inference 3–5× slower. | 8× RTX PRO 6000; no code. **[V]** | (a) Reasoning and step-wise placement each help a lot. (b) **Warning:** collision-only GRPO games the reward and wrecks every other metric. Keep a realism or NLL anchor. |
| **ScenePilot** (Aug 2026) [arxiv.org/abs/2608.30307](https://arxiv.org/abs/2608.30307) | Adds object groups one at a time. A learned repair policy (Qwen3-VL-8B, SFT on 17k inverse-perturbation trajectories, then GRPO with reward 0.5·physics + …) issues move/rotate/scale actions locally and globally. | Collision loss (PBL, ×10³): HRAP only, no repair 147.7 → without learned repair 123.4 → full 75.4. A GPT-5.2 repair agent reaches 53.5. | No code. **[V]** | A learned replacement for S1. Their inverse-perturbation data is what our DPO2 already builds. |
| **FuncRoom-Agent** (Aug 2026) [arxiv.org/abs/2608.29519](https://arxiv.org/abs/2608.29519) | Distils Codex/GPT-5.5 construction traces in a DSL into Qwen3.6-35B-A3B. The executor rejects collisions. GRPO uses a per-level VLM process reward (ScenePRM). | SceneEval collision (COL) 10.7 vs SceneSmith 18.1 vs LayoutVLM 21.1. Function-room FOP: SFT 44.2 → scene-level RL 51.9 → ScenePRM RL 65.7. 25 min per room vs SceneSmith 4 h 15 min. | No code; I read only part of the text. **[V, partial]** | Per-stage (process) rewards beat whole-scene rewards. A MoE ~3B-active expert is a plausible larger base for us. |
| **DirectLayout** (NeurIPS 2025) [arxiv.org/abs/2506.05341](https://arxiv.org/abs/2506.05341) | Qwen2.5-32B LoRA trained on GPT-4o reasoning annotations (6.5k scenes), then DPO on 12k pairs scored by VLM/LLM judges, plus iterative in-context fixes. | OOB / collision: full 8.74 / 6.31 vs Holodeck 36.33 / 8.32, LayoutGPT 33.91 / 15.89. Without reward: 26.04 / 16.77. | DPO about 11 h on 8× A800; [directlayout.github.io](https://directlayout.github.io/). **[V]** | Preference-only approach like ours, but with reasoning and dense judges. |

## 2. Reasoning and placement order: cheap evidence that matches P's gaps

- **[V] OptiScene's own ablation** ([arxiv.org/abs/2506.07570](https://arxiv.org/abs/2506.07570), Table 6; Qwen3-8B, the recipe we reproduced):
  - success rate UR: SFT 28% → +reasoning 33% → +DPO1 40% → +DPO2 50%;
  - OOR: 0.049 → 0.023.
  - Our v1 protocol dropped the reasoning block on purpose (`scene.py:27`).
- **[V] Ordering effects:**
  - SPG-Layout: removing large-first placement raises avg_LV from 19.5 to 132.7.
  - ThinkBLOX: random order gives PSA 41.6 vs 54.6 for plan-based order.
  - Holodeck places an anchor object first ([arxiv.org/abs/2312.09067](https://arxiv.org/abs/2312.09067)).
  - Infinigen Indoors anneals in stages, floor plan → large furniture → small objects ([arxiv.org/abs/2406.11824](https://arxiv.org/abs/2406.11824)).
- **[I]** Our HMAC-random order is the opposite of all of these. It also scatters same-category duplicates through the list while giving them identical inputs (same category, same prior size). That plausibly explains the co-located duplicate failures. A test: re-train SFT with parent-before-child and descending-footprint order, keeping ids opaque.

## 3. LLM + solver hybrids (inference-time; no retraining)

| Method | Mechanism | Numbers | Cost / code |
|---|---|---|---|
| **LayoutVLM** (CVPR 2025) [arxiv.org/abs/2412.02193](https://arxiv.org/abs/2412.02193) | The VLM gives initial poses plus relation functions. 400 steps of Adam projected gradient descent: a Distance-IoU collision loss on oriented boxes, with objects projected back inside the room every 100 steps. Only "self-consistent" relations are kept. | Collision-free / in-boundary / PSA: 81.8 / 94.9 / 58.8, vs LayoutGPT 83.8 / 24.2 / 16.6 and Holodeck 77.8 / 8.1 / 5.6. **Without optimisation: in-boundary 14.1, PSA 6.7.** Fine-tuning on raw numeric poses *hurt* GPT-4o (PSA 43.2 → 11.9); fine-tuning on the relation representation helped (48.1). | 1–5 min per scene of ~40 assets on one GPU; project page [ai.stanford.edu/~sunfanyun/layoutvlm](https://ai.stanford.edu/~sunfanyun/layoutvlm/). **[V]** |
| **Steerable Scene Generation** (CoRL 2025) [arxiv.org/abs/2505.04831](https://arxiv.org/abs/2505.04831) | Projection: minimise ‖t − t₀‖² subject to pairwise signed distance ≥ 0, orientations fixed, solved with SNOPT, then Drake gravity. Also DDPO post-training and MCTS inpainting search. | Penetration after projection is guaranteed zero. MCTS raised feasible objects from 21 (best of 3) to 34, the training maximum (training mean 17.1). SNOPT can fail on cluttered restaurants. | [github.com/nepfaff/steerable-scene-generation](https://github.com/nepfaff/steerable-scene-generation). **[V]** |
| **SceneSmith** (ICML 2026) [arxiv.org/abs/2602.09153](https://arxiv.org/abs/2602.09153) | GPT-5.2 designer / critic / orchestrator agents with placement, snapping and check tools. Then non-penetration projection (cites Pfaff 2025b; I could not see the reference title) and Drake settling. No learned layout model. | Appendix Table 13: collisions 1.2% vs I-Design 3.0, Holodeck 12.3, LayoutVLM 26–31; stability 95.6%; 71.1 objects per room. The main-text Table 2 looked inconsistent with this, so I flag it. | **$13.98 and ~3.5 h per scene** (Appendix N.2); [github.com/nepfaff/scenesmith](https://github.com/nepfaff/scenesmith). **[V]** |
| **Holodeck** (CVPR 2024) | LLM spatial constraints solved by DFS on a grid with yaw ∈ {0, 90, 180, 270}°, 30 s limit; a MILP (Gurobi) variant also exists. Collision and in-room are hard constraints: a violating object is simply not placed. | No collision rates reported. About $0.20 and 3 min per room. **[V]** | — |
| **Imperative scene programs + LLM-free error correction** (SIGGRAPH Asia 2025) [arxiv.org/abs/2504.05482](https://arxiv.org/abs/2504.05482), [2510.16147](https://arxiv.org/abs/2510.16147) | Each object is placed relative to objects already placed. Coordinate descent, one parameter at a time, reduces an OOB + overlap + support loss. | Human preference over declarative systems: 82% and 94%. **[V]** | — |
| Others: I-Design backtracking ([2404.02838](https://arxiv.org/abs/2404.02838)), Open-Universe DSL + gradient optimisation ([2403.09675](https://arxiv.org/abs/2403.09675)), FirePlace constraints + LLM pruning ([2503.04919](https://arxiv.org/abs/2503.04919)), SceneCraft scene graph → Blender constraints + GPT-V refinement, up to ~100 assets ([2403.01248](https://arxiv.org/abs/2403.01248)), SceneWeaver physics executor ([2509.20414](https://arxiv.org/abs/2509.20414)) | — | Abstract-level only; not quantified here. |

**[I] What this means for us.** Sizes are fixed and nearly all yaws are axis-aligned, so the problem is close to rectangle packing.
- A global minimum-displacement solve should beat greedy S1 on the 11–50-object bins. Options are OR-tools CP-SAT `NoOverlap2D` with an L1 displacement objective, the Steerable-style SNOPT/SLSQP projection, or LayoutVLM-style DIoU projected gradient descent.
- I have not checked feasibility: prior sizes may not fit some rooms. Check the sum of prior footprints against free floor area first.
- It changes no model weights. It would be reported as "S3", next to raw and S1.

## 4. Representation and tokenization

- **[V] Relational or DSL output beats raw coordinates** in both papers that test it directly: LayoutDSL (0.878 vs 0.749) and LayoutVLM's fine-tuning result (above).
- **[V] Motifs and programs for repeated objects.** HSM's 12 motifs (row, grid, stack, …) are at [github.com/3dlg-hcvc/hsm](https://github.com/3dlg-hcvc/hsm) ([2503.16848](https://arxiv.org/abs/2503.16848)). SceneMotifCoder meta-programs are at [2408.02211](https://arxiv.org/abs/2408.02211). **[I]** These target exactly our "co-located duplicates".
- **[V] Discrete-then-continuous pose heads.** NaLA (ECCV 2026, [2606.29395](https://arxiv.org/abs/2606.29395), code [github.com/adamcwan/NaLA-code](https://github.com/adamcwan/NaLA-code)) predicts discrete poses autoregressively, then regresses a continuous refinement. A search snippet attributed collisions to "single reused token" pose regression; I could not confirm that on the abstract page.
- **[V] Number Token Loss** (ICML 2025, [2411.02083](https://arxiv.org/abs/2411.02083)) adds a regression-like loss on digit tokens at no runtime cost. The Qwen tokenizer splits numbers into single digits ([Qwen report 2309.16609](https://arxiv.org/abs/2309.16609), stated for Qwen-1; later versions only by third parties). **[I]** Low priority: our failures are gross overlaps, not small numeric errors.
- **[V] Scene-graph intermediates**: InstructScene ([2402.04717](https://arxiv.org/abs/2402.04717)), 3D-Layout-R1, SG-Layout ([2608.01106](https://arxiv.org/abs/2608.01106), no numbers on the abstract page).

## 5. Diffusion and flow layout models

- **[V] PhyScene** ([2404.09465](https://arxiv.org/abs/2404.09465)) adds collision, room and reachability guidance terms (based on box IoU) to a diffusion model.
  - Living room, floor-conditioned, scene-level collision rate: ATISS 0.85, DiffuScene 0.69, PhyScene 0.63.
  - Ablation: collision guidance alone lowers object collisions 0.200 → 0.111 but raises out-of-room 0.240 → 0.354.
  - Size guidance is disabled because it shrinks objects, which is the same lesson as our size confound.
  - **[I]** Guided diffusion still leaves most living rooms colliding, so it is not a fix for dense rooms on its own.
- Others: MiDiffusion ([2405.21066](https://arxiv.org/abs/2405.21066)); LEGO-Net, a denoiser trained on perturbed clean rooms that is a natural learned S1 ([2301.09629](https://arxiv.org/abs/2301.09629)); LayoutFlow ([2403.18187](https://arxiv.org/abs/2403.18187), **2D graphic layouts only**, code JulianGuerreiro/LayoutFlow).
- **[I]** None of these accept our full conditioning (polygon, doors, fixed boxes, prior sizes, support ids) without substantial work. The most useful role for one is a refiner that starts from P's draft.

## 6. Test-time verification, search and self-refinement

- **[V] Our own dev data:** best-of-8 with the checker raised P from 0.400 to 0.600 raw checker success (`findings.md:47`). It was not adopted only because the downstream acceptance gain after S1 was +0.020, below the threshold.
- **[V] Other evidence:**
  - MCTS search over a diffusion model (Steerable, above).
  - Critic loops (SceneSmith), multi-turn refinement in RL (MetaSpatial).
  - SCoRe ([2409.12917](https://arxiv.org/abs/2409.12917)): offline SFT on correction traces suffers distribution mismatch or collapse, while multi-turn online RL improves self-correction (+15.6% MATH).
- **[I]** A checker-feedback turn (violations listed, then the model fixes them), trained with multi-turn GRPO, is the learned version of S1 plus best-of-N.

## 7. Distillation from a larger model

- **[V]** DeepSeek-R1 ([2501.12948](https://arxiv.org/abs/2501.12948)): on Qwen2.5-32B, distillation beat large-scale RL (AIME 72.6 vs 47.0). That is one experiment at 32B.
- **[V]** DirectLayout, 3D-Layout-R1 and FuncRoom-Agent all distil teacher *reasoning or construction traces*, not only final answers.
- **[V]** Amendment 018 distils O2 *answers only*, rebuilt in P's output format (`amendment-018-…md:20`).
- **[I]** If O2's reasoning text was kept on the server, also test a reasoning-SFT variant. A short structured plan (anchors, zones, occupied rectangles so far) may be better than 16k-token free thinking for 50-object rooms.

## 8. Suggested order (all [I]; each needs the usual dev-selection amendment)

1. **Global solver after S1 ("S3")**: CPU only, about a day of code, no retraining. Target the 11–50-object bins (S1: 0.562 / 0.258).
2. **Large-first / parent-first object order** in `bind_room` for an SFT re-run.
3. **GRPO from the current P**, with:
   - a dense reward: −normalised overlap area − OOB area, + support validity, + a pass bonus;
   - per-object credit (MetaSpatial 3D-SPO) or per-group credit;
   - a curriculum over object-count bins;
   - a KL or NLL anchor so it cannot game the reward the way ThinkBLOX's plain GRPO did.

   Rough cost: about 80k rollouts (5k rooms × 8 samples × 2 epochs) at about 2k tokens each is ~160M generated tokens, roughly 15–30 GPU-h on vLLM plus training. Ballpark, not measured.
4. **Reasoning or plan SFT distilled from O2**, then GRPO (the 3D-Layout-R1 recipe).
5. **Bigger changes**: a DSL/motif output space (handles duplicates); step-wise placement with feedback (3–5× slower inference); a larger base (O2-class 27B, or a MoE like FuncRoom's 35B-A3B), since O2 passes 0.99 of the answers it completes.

## 9. Caveats and things I could not verify

- **Our own comparison:**
  - The pilot is mostly small rooms; only 12 rooms have >30 objects.
  - O2's 0.920 includes up to 3 attempts, while P is single-shot.
  - My stratum numbers for P are on full test; O2's and L's are on the 300-room pilot.
- **Paper numbers:**
  - SceneSmith's main-text Table 2 looked inconsistent with its text and appendix.
  - MetaSpatial reports 11.5% and 27.4% for the same setting.
  - OptiScene's DPO pair counts are inconsistent within the paper.
  - FuncRoom-Agent and ScenePilot were read only partly; their GRPO hyper-parameters are not visible.
  - Most 2026 papers (LayoutDSL, ThinkBLOX, SPG-Layout, ScenePilot) have no released code.
- **Not found:** "DeBaRA" did not turn up in my searches. "LaviGen" was seen only in a secondary summary; I did not find its arXiv entry.
- No benchmark in the literature reports results on rooms with more than 50 objects under fixed sizes. Our setting is harder than theirs.

Scratch directory (created, empty): `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/better/litsweep/`