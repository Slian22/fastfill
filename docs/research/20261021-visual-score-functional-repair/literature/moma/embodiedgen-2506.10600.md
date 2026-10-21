# EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence
- Authors: Xinjie Wang, Liu Liu, Yu Cao, Ruiqi Wu, Wenkang Qin, Dehui Wang, Wei Sui, Zhizhong Su (Horizon Robotics, GigaAI). Jun 2025. https://arxiv.org/abs/2506.10600 (HTML v1 read)
- What: toolkit of six modules (Image-to-3D, Text-to-3D, texture, articulated object generation, scene generation, layout generation) producing URDF assets with real-world scale. Sec 3.5 "3D Scene Generation" makes backgrounds (panorama -> mesh + 3DGS, recentred to floor), not furnished rooms; layout generation is described only in the Fig. 2 caption ("autonomously generates the scene layout", "semantically and physically plausible configurations").
- Robot eval: qualitative demos only (arm grasping in MuJoCo/RoboSplatter, Isaac Lab grasp eval, RoboTwin dual-arm, quadruped navigation in Gym). No success rates, no navigation/reachability/clearance/visibility checks with thresholds.
- Only numbers: asset quality checker on cups precision 68.7%, recall 76.7% (Sec 3.2).
- Relevance: v1 gives no layout-executability criteria; see embodiedgen-v2-2607.07459.md for the multi-room/task-driven version. (FastFill already targets EmbodiedGen as a downstream backend.)
- Tags: scene-gen-for-robots, data-generation, articulated-objects, real2sim/digital-twin
