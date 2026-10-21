# Primary-PDF audit: agent evaluation metrics

Reviewed 2026-10-20. Scope: four named primary papers, their main metric tables, metric definitions, relevant appendices, denominators, action/perception assumptions, and training-versus-evaluation controls. This is a targeted protocol audit, not an exhaustive literature review. Search used arXiv full text/PDFs plus official CVF, PMLR and the PARTNR repository for venue identity; excluded reviews, blogs and search snippets as evidence. Four selected, four included. Page numbers below are 1-based PDF pages of the downloaded arXiv versions in `docs/research/20261020-worldedge-table1/papers/`, not proceedings page numbers.

## Source extraction

### PARTNR

Primary benchmark; reviewed arXiv:2411.00081v1; associated work ICLR 2025 (official repository citation). High weight for this benchmark's metric definitions; moderate for transferring its skill implementation to our robotics claims. Sources: https://arxiv.org/pdf/2411.00081 and https://github.com/facebookresearch/partnr-planner .

- **Table 2, p10:** `Sim Steps ↓`, `Success Rate ↑`, `Percent Complete ↑`, `Planning Cycles ↓`; controls include controllability, skills and observability. **Table 13, p37** provides task-type SR as a secondary breakdown.
- **p9; §A.5.3–5 pp22–23:** PC counts satisfied propositions respecting dependencies and temporal/entity/terminal constraints; success requires complete satisfaction. Evaluation uses simulation-state trajectories, not the planner's declared completion.
- **p7:** validation 1,000 episodes/13 houses, test 1,000/10; benchmark scenes/tasks fixed while planners, skills and perception vary. **p36:** validation/test tables report means and standard errors.
- **p32:** 20,000 simulation-step cap; decentralized 50 versus centralized 100 replanning calls. This refines p9's broad statement of 50 calls. Learned Pick/Place still snap/unsnap when the gripper reaches the target neighborhood.
- **p8:** partial observability can still use privileged simulation perception; ConceptGraphs is a separate non-privileged condition. Do not equate “partial” with RGB-only.

### ALFRED

Primary benchmark, CVPR 2020; reviewed arXiv:1912.01734v2. High weight for task/partial-success and expert-action-normalized efficiency definitions; limited correspondence to continuous contact control. https://arxiv.org/pdf/1912.01734v2 .

- **Table 3, p7:** `Task`, `Goal-Cond` for validation/test × seen/unseen, with their path-weighted versions in parentheses. Table 4 p8 is isolated subgoal evaluation, not complete rollout success.
- **§5.1 p6:** binary Task Success requires final object positions/states satisfying all goals. Goal-Condition Success is the fraction fulfilled per episode, averaged across episodes. Entity consistency can itself be a goal condition.
- **Eq.6 p7:** weighted score is `s × L*/max(L*, Lhat)`; L* is expert demonstration action count, explicitly not guaranteed optimal, and Lhat is agent action count. It is not generic geometric SPL.
- **p6:** terminate after exceeding 1,000 actions or more than 10 failed actions; failed runs remain evaluated episodes. **Table2 p4:** seen/unseen splits; same benchmark is used across models.
- **pp4–6:** expert has privileged planner metadata, evaluated agent sees egocentric observations; interactions use discrete actions plus predicted masks resolved against simulator masks. Subgoal tests first replay the preceding expert trajectory (p7), so their scores cannot substitute for E2E SR.

### HomeRobot OVMM

Primary benchmark, CoRL 2023; reviewed arXiv:2306.11565v2. High weight for cumulative stage-success and overall-success distinction; small real-world study and idealized simulated grasp reduce weight for general physical-competence claims. https://arxiv.org/pdf/2306.11565v2 ; venue: https://proceedings.mlr.press/v229/yenamandra23a.html .

- **Table3 p8:** `FindObj`, `Pick`, `FindRec` cumulative partial SR; `Overall Success Rate` equals successful Place; `Partial Success Metric` averages four stage rates. Skill configuration and Ground Truth/DETIC perception are separate controls.
- **p5:** each *successive* completed stage earns one point; overall success requires all four. Thus cumulative stage SR uses episode denominators, not only episodes reaching that stage.
- **§C.1 p18:** all stages within 1,250 steps; simulated Pick uses magic snap with visible target and end-effector within 0.8 m. Place must settle on the receptacle for 50 contiguous steps below velocity thresholds and avoid robot-scene collision.
- **§C.2 p18:** real execution uses actual grasp/settling and a 300-step cap. **Table4 p8**, **p6:** 20 trials per baseline in a held-out apartment.
- **pp5–6:** 38/12/10 train/validation/test scenes; held-out target instances. Exact simulation episode count was not found in reviewed paper text; do not infer it from percentages.

### Holodeck

Primary scene-generation study, CVPR 2024; reviewed arXiv:2312.09067v2. High weight for identifying its experimental design, moderate for broad generalization because the artist-created test set contains only ten worlds. https://arxiv.org/pdf/2312.09067v2 ; venue: https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Holodeck_Language_Guided_Generation_of_3D_Embodied_AI_Environments_CVPR_2024_paper.html .

- **p6 Figures9–10:** human asset-selection/layout/overall preference and CLIP score. **Table1 p7:** human-ranked layout Mean Reciprocal Rank across four residential types/average; assets held identical when changing layout method.
- **Table2 p8:** `Success (%)`, `SPL` for Office, Daycare, Music Room, Gym, Arcade and average.
- **§5 p8:** common ProcTHOR-pretrained agent initialization, then fine-tuning on generated scenes; 100 scenes per type and 50M training steps. Test on separate artist-designed NoveltyTHOR: ten worlds, two per type. Checkpoints selected on each model's validation scenes.
- Hence this is **training-data utility / transfer**, not an unchanged fixed agent executing in each generator's own worlds. Its numbers cannot be reused as that experiment.
- The reviewed evaluation passage does not state total test episode count or a full SPL equation/termination protocol. Do not invent these. Navigation-only Success/SPL does not certify articulation or persistent-world state.

## Synthesis and corrections to the proposed Table 1

**Strong protocol precedent:** task success + partial task completion; report efficiency without confusing different units. PARTNR and ALFRED independently instantiate this design, although their worlds/action interfaces share the broader embodied-simulation ecosystem. HomeRobot provides additional support for progress diagnostics in sequential mobile manipulation. These studies do not statistically validate one universal metric for all world generators.

**Correction:** four task-family SR columns are defensible *our task-suite design*, but are not the main-table metric schema used across these papers. Recommend the Agent panel center on `Task SR ↑`, `Goal Completion/PC ↑`, and an explicit efficiency measure. Put navigation/manipulation/cross-room/revisit breakdowns in a secondary panel or table; navigation subset can have `SR + SPL`. A family-macro average is our aggregation choice and should be labeled with weights.

**Efficiency:** use LLM calls/planning cycles and simulation/interaction steps with fixed caps and reported failure handling. Never minimize mean steps alone: early failures can appear efficient. Add a success-weighted expert-action metric only when a valid reference trajectory exists and control units match; label it as such rather than calling every normalized score SPL. If using successful-only cost, show its denominator and SR together.

**World versus agent denominator:** standard benchmark SR counts benchmark episodes that already have executable worlds. Counting world-generation/loading failure as zero creates a useful *system* E2E yield, but it is our extra endpoint, not an existing PARTNR/HomeRobot metric. Keep generation yield, conditional Agent SR, and all-request E2E SR distinct, or define their relation transparently.

**Controls:** fixed agent across generated worlds is our proposed world-usability experiment. Common initial policy + equal training budgets + shared held-out test worlds is the distinct Holodeck-style learning-utility experiment. Do not claim either substitutes for the other.

**Evidence gaps:** none of these four main tables establishes a universal persistent-world retention score; revisit state/identity checks need our explicit protocol. Skill success under snap/oracle actions is valid for its declared simulator setting but does not establish physically realized grasp/contact skill. Need our evaluation to declare control fidelity and observation privilege.

## Visual verification

Rendered and inspected full table pages: PARTNR p10; ALFRED p7; HomeRobot p8; Holodeck pp7–8. Images: `/tmp/partnr-table2.png`, `/tmp/alfred-table3.png`, `/tmp/homerobot-table3.png`, `/tmp/holodeck-table1.png`, `/tmp/holodeck-table2.png`. PDF prose/appendix text read for all page references above.

## Explicit proposed protocol for matched world generators (our design)

Let N predeclared task requests and generation seeds be shared across generator rows. Let G_i indicate successfully materialized, executable world i. Define a task's evaluation specification *before* generation, including K_i goal propositions, identity bindings and ordering/terminal constraints. Let c_ik in {0,1} be the evaluator's valid satisfaction of proposition k over the rollout; these are simulation-ground-truth evaluation predicates, not an LLM judgment. For the policy, freeze checkpoint, prompt/memory, sensors, embodiment, skills, action frequency and budgets across generator rows.

- Per-episode partial completion: `PC_i = (1 / K_i) sum_k c_ik`; full success: `S_i = 1[PC_i = 1 and all task-level constraints hold]`. Any global constraints not already reflected in c_ik must still gate S_i. Macro-average PC per episode; do not silently pool all propositions because that weights longer tasks more.
- Executable-world yield: `Y = sum_i G_i / N`.
- Agent SR conditional on a materialized world: `SR_cond = sum_i G_i S_i / sum_i G_i`; conditional PC similarly. Always publish this denominator and task-stratum counts because generators may admit different subsets.
- All-request system outcome: `SR_E2E = sum_i G_i S_i / N = Y * SR_cond`. `PC_E2E = sum_i G_i PC_i / N`, with non-materialized cases assigned zero. These E2E definitions are proposed system-level endpoints; they are not verbatim metrics from the four papers.
- If a valid expert reference with action count L_i* exists, `PW-SR = (1/N) sum_i G_i S_i L_i* / max(L_i*, L_i)` and an analogous PC-weighted score give ALFRED-style efficiency. Without references, report actual action/simulation steps and LLM calls as *resource usage*, jointly with SR and failure counts. For success-only cost, publish N_success and never interpret lower cost alone as superior performance. Main PARTNR prose does not fully specify successful-only versus all-episode aggregation for its main-table step column; do not fill that ambiguity by assumption.

**Recommended compact columns:** `Task SR ↑`, `Goal Completion ↑`, `LLM Calls ↓` (or control steps, clearly labeled resource usage) on the Agent side. In the full world-generation comparison, additionally keep `E2E Task SR ↑` and executable-world count/yield somewhere explicit. If space is tight, headline E2E Task SR, retain conditional SR in the companion accounting table, and name the denominator in the caption. Place Navigation / Manipulation / Cross-room / Revisit in task-stratified results, not as substitutes for success/progress/efficiency. Include `SPL` only for the navigation subset or a fully defined navigation reference.

Three experiments that must not be pooled: (1) **world-usability:** fixed policy, different generated worlds from matched specs; (2) **policy comparison:** fixed benchmark worlds/tasks, different agent policies (PARTNR/ALFRED/HomeRobot); (3) **training utility:** same policy initialization and training budget, different generated training worlds, common independent held-out evaluation worlds (Holodeck). Each answers a different causal question.


## Final recommendation on efficiency after discussion

Move efficiency to the appendix until reference trajectories and control units are stable. A main Agent panel with **E2E Task SR + constraint-aware Goal Completion + Revisit SR (our challenge)** at a disclosed fixed budget is more defensible than adding another composite. Report successful-only mean steps as a diagnostic with N_success and SR. Failure-capped steps are well-defined as `C = (1/N) sum_i [S_i min(L_i,B) + (1-S_i) B]`, setting failed generation to S_i=0, but this is our adaptation; algebraically it is `B(1-SR) + SR E[L|success]` under within-budget successes. It partly repackages success and is not an independently validated standard. Actual calls/runtime can be reported as resource usage, without claiming lower alone means better agent performance.
