# MobileManiBench: Simplifying Model Verification for Mobile Manipulation
- Authors: Wenbo Wang, Fangyun Wei, Qixiu Li, Xi Chen, Yaobo Liang, Chang Xu, Jiaolong Yang, Baining Guo. arXiv 2602.05233 (v2 26 Jun 2026). https://arxiv.org/abs/2602.05233
- What: Isaac Sim benchmark; RL experts (MobileManiRL) generate trajectories to train a VLA (MobileManiVLA). 630 objects in 20 categories (PartNet-Mobility, UniDoorManip, YCB); skills open, close, pull, push, pick; "over 100 tasks"; 5 realistic settings (space, wall, door, outdoor, tabletop) x 20 placements (80 train / 20 test); 150K trajectories per robot.
- Robots: G1 (parallel gripper) and XHand (12-DoF hand), each a 7-DoF right arm on a 2-DoF mobile base.
- Success: object grasp point reaches the goal point within T = 300 steps. Spawn (train): base facing the grasp point with 1.0-1.5 m x/y offsets and -15 to 15 degree yaw; fixed-base ablation 0.5-1.0 m. No reachability/visibility filters described.
- Results (G1, mean): RL 89.6%, VLA 56.7%; by skill VLA open 42.9, close 75.8, pull 34.4, push 85.1, pick 28.8. Unseen object+scene: OpenVLA 4.5, CogACT 6.8, π0 11.2, π0.5 18.8, MobileManiVLA 28.2%.
- Failure causes: handle-based objects (car, door, table) harder than lid-flip objects (toilet, laptop, dishwasher); fixed vs mobile base ablation 82.8% vs 25.4% (row labelling ambiguous in paper).
- Relevance: articulated-object approach from 1.0-1.5 m is the standard spawn; handle localisation dominates failures, not layout.
- Tags: sim-benchmark, data-generation, RL-sim2real, articulated-objects, VLA-policy
