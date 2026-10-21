# UniManip: General-Purpose Zero-Shot Robotic Manipulation with Agentic Operational Graph
- Authors: Haichao Liu, Yuanjiang Xue, Yuheng Zhou, Haoyuan Deng, Yinan Liang, Lihua Xie, Ziwei Wang (NTU Singapore, Tsinghua). arXiv 13 Feb 2026. https://arxiv.org/abs/2602.13086
- What: agentic zero-shot manipulation (operational graph + VLM), tested fixed-base (Galaxea A1) and mobile (differential-drive base + Realman RM65 arm) in an office.
- Results: 14-task pick-and-place SR 93.75 vs π0 41.88, NORA-1.5 71.25; long-horizon ablation full system 80.00 (failure section says "59.7% overall success": paper inconsistent).
- Failure breakdown (share of all trials, before recovery): grasp failure 16.3%, motion error 9.7%, grasp collision 5.3%, grounding error 4.7%, placement error 4.3% (total 40.3%). Motion errors attributed to base-arm coupling: "If the base navigation stopped slightly outside the optimal workspace, the arm occasionally encountered kinematic singularities or failed to find a valid IK solution."
- Base placement: mapless visual-servo docking v_x = K_p (z_obj - d_goal), w_z = K_theta atan2(x_obj, z_obj), K_p = 0.6, K_theta = 0.2; d_goal value not given.
- Relevance: about a quarter of failures are tied to base stand-off (motion error) in a zero-shot MoMa stack.
- Tags: base-placement, open-vocab-nav-pick
