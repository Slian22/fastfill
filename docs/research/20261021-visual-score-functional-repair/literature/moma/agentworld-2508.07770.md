# AgentWorld: An Interactive Simulation Platform for Scene Construction and Mobile Robotic Manipulation
- Authors: Yizheng Zhang, Zhenjun Yu, Jiaxin Lai, Cewu Lu, Lei Han (5). arXiv 2508.07770 v2 13 Aug 2025; CoRL 2025 (PMLR v305). https://arxiv.org/abs/2508.07770
- What: procedural household scenes (living rooms, bedrooms, kitchens; multi-floor) built in Unreal PCG by extending ProcTHOR's rule-based algorithms, "2-3 seconds per scene"; semantic asset placement; simulated in Isaac Sim (PhysX 5.0); dual-mode teleoperation for data.
- Validity checks: only "Enhanced inter-room connectivity checks to avoid impractical designs (e.g., isolated rooms)"; no numeric collision/navigation/reachability/articulation-clearance thresholds.
- Robots: Unitree G1 and H1 humanoids (2F-85 or dexterous hand), Franka Panda on a wheeled base, DOBOT X-Trainer (fixed).
- Tasks: Pick & Place, Open & Close (room doors, fridge/microwave/stove doors), Push & Pull (drawers); multistage living-room/kitchen/bedroom tasks. 50 episodes per task, success rate.
- Results (P&P / Open&Close / Push&Pull / Living / Kitchen / Bedroom): BC 52/62/58/10/4/6%; ACT 66/84/72/16/12/14%; DP 64/78/76/28/20/24%; pi0 64/82/70/30/20/18%. Sim-to-real pi0 place-in-bowl 29.3%. Failures not attributed to layout ("all methods struggle with complex mobile manipulation").
- Relevance: long-horizon room-scale MoMa success collapses to 4-30% even in generated houses with only a connectivity check; no layout-side executability guarantees are reported -> gap our checks can fill.
- Tags: scene-gen-for-robots, teleop-data, humanoid, whole-body-wheeled-bimanual, long-horizon, articulated-objects
