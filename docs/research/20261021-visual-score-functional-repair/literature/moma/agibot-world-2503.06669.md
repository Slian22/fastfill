# AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems
- Authors: AgiBot-World team. arXiv 9 Mar 2025 (v4 Aug 2025). https://arxiv.org/abs/2503.06669
- What: teleop dataset + GO-1 generalist policy (latent action planner).
- Data: 1,001,552 trajectories, 2976.4 hours, 217 tasks, 87 skills, 106 scenes in five domains (domestic, retail, industrial, restaurant, office); >4,000 m^2 facility, >3,000 objects.
- Robot: AgiBot G1, dual 7-DoF arms, "a mobile chassis, and an adjustable waist"; wheel type, lift range and reach not given. Base moved by VR thumbsticks during collection; evaluation tasks do not include navigation.
- Results: seen-scenario score 0.77 (AgiBot-pretrained) vs 0.47 (OXE); OOD 0.67 vs 0.38; protocol 10 rollouts per task (text elsewhere says 30; inconsistent).
- Relevance: the biggest 2025 teleop corpus is effectively fixed-base; no layout or navigation failure evidence.
- Tags: teleop-data, VLA-policy
