# LAMBDA: A Benchmark for Data-Efficiency in Long-Horizon Indoor Mobile Manipulation Robotics
- Authors: Ahmed Jaafar, Shreyas Sundara Raman, Yichen Wei, Sudarshan Harithas, Sofia Juliani, Anneke Wernerfelt, Benedict Quartey, Ifrah Idrees, Jason Xinyu Liu, Stefanie Tellex. arXiv v4 27 Jan 2025. https://arxiv.org/abs/2412.05313 (no venue listed)
- What: 571 language-conditioned pick-and-place tasks with room-to-room and floor-to-floor navigation, human expert demos.
- Robot/scenes: sim = ManipulaTHOR arm in AI2-THOR/RoboTHOR, 521 trajectories over 5 scenes, 20 rooms; real = Spot with arm, 50 trajectories over 3 environments, 10 rooms (incl. two floors joined by stairs).
- Success: 1 point per subtask (nav to object, grasp, nav to goal, place), averaged; no distance thresholds given.
- Results: scene generalisation RT-1 2.7%, MG-MoMa 2.4% (no scene > 5%); LIMP zero-shot 44.4% +- 49.7 on 45 tasks.
- Scene-caused failures (qualitative): lowest score in Scene 1 where the robot spawns in an enclosed room and must leave it; higher scores where targets are near spawn.
- Relevance: exiting/entering an enclosed room is the hardest navigation case for learned policies -> door/entry passage and path length from the entry to task objects are layout-level difficulty drivers.
- Tags: sim-benchmark, long-horizon, VLA-policy
