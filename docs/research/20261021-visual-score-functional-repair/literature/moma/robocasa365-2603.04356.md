# RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots
- Authors: Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, Yuke Zhu. arXiv 4 Mar 2026. https://arxiv.org/abs/2603.04356 (existence verified)
- What: 365 tasks (65 atomic, 300 composite), "2,500 diverse kitchen environments" (pretraining 50 layouts x 50 styles; 10 target scenes); 2,509 objects / 153 categories + new assets in 57 categories; 456 fixture instances in 12 categories; 612 h human + 1615 h MimicGen data.
- Layouts: not procedural; replicas of "50 real-world homes with active listings on Zillow.com" (Bay Area, Austin, Denver, Boston, Atlanta), built with digital cousins "to match the floor plan as closely as possible". No counter height / aisle / clearance rules in the text.
- Robot: Franka Panda on Omron mobile base; actions 7 arm + 5 (base translation/rotation, torso height, base-gating mode); real DROID Panda.
- Mobility: "220 require mobile manipulation, while 145 can be performed without mobility".
- Success: binary task condition within a task-dependent horizon; 30 trials per task. Multi-task avg success: DP 6.1, pi0 15.0, pi0.5 16.9, GR00T N1.5 20.0; pretraining + 100% target 51.1.
- Failure causes: none reported for navigation / base positioning / reachability / occlusion; long-horizon and forgetting noted.
- Relevance: kitchen layouts copied from real listings rather than generated; implies the field treats real floor plans as the reachability ground truth. 220/365 = 60.3% of tasks (my arithmetic) need mobility.
- Tags: sim-benchmark, real2sim/digital-twin, long-horizon, data-generation, VLA-policy
