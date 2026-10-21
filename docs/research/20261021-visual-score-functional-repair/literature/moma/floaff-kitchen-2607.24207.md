# FloAff-Kitchen: Bridging Navigation and Manipulation via Canonical and Progressive Floor Affordance Learning
- Authors: Ping Zhong, Manling Teng, Tao Wu, Bolei Chen, Jiazhi Xia, Jianxin Wang. arXiv 2607.24207v1 (27 Jul 2026); no venue. https://arxiv.org/html/2607.24207
- What: predict floor affordance (base positions leading to manipulation success); benchmark in RoboCasa, skills PnPCounterToCab, OpenSingleDoor, CloseDrawer, CloseDoubleDoor; 24,782 RGB-D observations (Styles 13,610, Layouts 11,172).
- Labels: 2D occupancy grid at 0.05 m; candidates in a target-centred region (ranges not given), kept only if base, torso and arm collision-free and target within reach radius r_arm (value not given); each pose rolled out with a pretrained Diffusion Policy; success -> 1; Gaussian smoothing. Robot not named.
- Results: MoMa-Kitchen RMSE 0.073 / PCC 0.796 vs NavAff 0.147 / 0.680. Downstream SR Styles 0.85 vs NavAff 0.37, C2F-Exp 0.50, N2M 0.44, oracle 0.88; Layouts 0.79 vs 0.27 / 0.49 / 0.47, oracle 0.86. Single-task (Layouts): PnP 0.56, OpenSD 0.79, CloseD 0.99, CloseDD 0.70.
- Not reported: fraction of feasible floor cells.
- Relevance: with a good base pose the policy ceiling is 0.86-0.88; the layout split is harder than the style split, i.e. layout geometry itself changes manipulation success.
- Tags: base-placement, articulated-objects, sim-benchmark
