# Galaxea Open-World Dataset and G0 Dual-System VLA Model
- Authors: Galaxea Team (named: Tao Jiang, Jianning Cui, Xiao Liu, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Shuiqi Cheng, Hang Zhao, Huazhe Xu, Jiyang Gao). arXiv 30 Aug 2025. https://arxiv.org/abs/2509.00576
- What: real-world teleop dataset + dual-system VLA (G0-VLM planner + G0-VLA executor).
- Robot (as stated in paper; secondary to official spec sheet): Galaxea R1 Lite, 23 DoF (two 6-DoF arms, 3-DoF torso "vertical and pitch motion", 6-DoF omnidirectional base); base speed up to 1.5 m/s; reach 60 cm; payload up to 5 kg; height 1280 mm; chassis width 600 mm; overall width 670 mm; head stereo RGB + 2x D405 wrist cameras.
- Data: 500 hours, 100K trajectories, 150 task categories, 50 scenes, 11 physical sites (residential, catering, retail, office), >1,600 objects, 58 skills; includes whole-body motions.
- Eval: progress score over 10 runs: table bussing (6 pts), microwave (5), bed making (4: move to bed, lift torso & grasp quilt, lean back, flatten), block stacking (6). Task scores only in figures. G0-VLM instruction accuracy 83.3/74.2/78.2/75.6 vs Gemini-2.5-pro 32.0/15.8/54.2/55.0.
- Scene-related: collection rule "key objects [are] kept in view during manipulation"; cross-embodiment pretraining weakens chassis and torso actions. No layout failure analysis.
- Relevance: a 600 mm-wide chassis (670 mm overall) is wider than a 0.4 m agent-diameter grid assumption; torso lift is used to reach low/high surfaces (bed).
- Tags: VLA-policy, teleop-data, whole-body-wheeled-bimanual, real-home-deploy
