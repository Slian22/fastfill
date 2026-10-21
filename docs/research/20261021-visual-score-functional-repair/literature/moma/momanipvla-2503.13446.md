# MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation
- Authors: Zhenyu Wu, Yuheng Zhou, Xiuwei Xu, Ziwei Wang, Haibin Yan. arXiv 17 Mar 2025. https://arxiv.org/abs/2503.13446 (venue not stated on arXiv page)
- What: makes a fixed-base VLA mobile via bi-level trajectory optimisation: upper level picks base waypoints, lower level the end-effector trajectory. Costs: reachability (IK iterations, large constant if IK budget exceeded), smoothness, collision (nvblox ESDF, safety margin eps0 = 0.1); weights lambda1 = 10.0, lambda2 = 1.0, lambda3 = 0.6; Dual Annealing + SLSQP.
- Robot: Stretch in OVMM sim (60 scenes, >18k objects); real: hexman echo plus base + RM65 arm; 10-DoF (7 arm + 3 base).
- OVMM: overall 49.4% vs closest baseline 38.2% (FindObj 66.1, Pick 62.6, FindRec 53.1). Text claims "+4.2%" which does not match the table (paper inconsistent).
- Ablation (overall SR): full 49.4; w/o reachability 48.2; w/o smoothness 48.7; w/o collision 48.9; GT masks removed 11.3.
- Failure breakdown (Table 3): Orient_to_place 72.09% (navigation fails to align with target), Find_recep 14.62%, Other 12.96%, Nav_to_place 0.33%.
- Real world (10 trials each): Stack Block 30.0%, Open Drawer 10.0% (hinge constraint, collision with moving hinge), Put in Bowl 40.0%.
- Relevance: dominant OVMM failure is the final base orientation at the place receptacle, not reaching it -> layout check should require an approach side/pose facing the receptacle; drawer opening fails on swing collision.
- Tags: VLA-policy, base-placement, open-vocab-nav-pick, articulated-objects
