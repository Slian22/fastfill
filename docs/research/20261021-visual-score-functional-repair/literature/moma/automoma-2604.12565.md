# Scalable Trajectory Generation for Whole-Body Mobile Manipulation (AutoMoMa)
- Authors: Yida Niu, Xinhai Chang, Xin Liu, Ziyuan Jiao, Yixin Zhu. arXiv 14 Apr 2026; CVPR 2026 (per CVPR virtual poster page and BIGAI page). https://arxiv.org/abs/2604.12565
- What: GPU trajectory optimisation over an augmented kinematic chain (base + arm + articulated object) to mass-produce MoMa demos.
- Robots: Summit Franka (10-DoF base-arm); also R1 and TIAGo; real UR5-Ridgeback.
- Scenes: "over 500k physically valid trajectories spanning 330 scenes" = 30 curated iTHOR scenes (static appliances swapped for SAPIEN articulated assets) + 300 procedural Infinigen scenes; articulated objects from PartNet-Mobility (microwave, dishwasher, cabinet, door, chair); ~20 AO-Grasp grasps per object.
- Validity pipeline: ~30 IK solutions per grasp, clustered; up to 10 retries; ESDF collision checks with sphere-approximated links in a box around start/goal; waypoints filtered on deviation from attachment (thresholds not given; Table A1 tolerances 0.01 position / 0.01 rotation). Yield and per-cause rejection rates: not reported.
- Throughput: "5,000 episodes per gpu-hour" (>80x CPU baselines ~60 trajectories/hour).
- Policy results: microwave door opening success = target angle within 300 steps, 50 trials; fixed-base 100% with < 800 trajectories; mobile-base saturates ~70% (3,200 trajectories); DP3 at 100k > 50% on 4 of 5 objects; object 46197 limited by articulation workspace.
- Relevance: data-gen pipelines assume IK+collision-feasible base poses around articulated objects exist; the scenes are taken as given and infeasible ones are silently filtered -> no scene repair; workspace around an articulation is the limiting factor.
- Tags: data-generation, whole-body-wheeled-bimanual, articulated-objects, scene-gen-for-robots
