# AC-DiT: Adaptive Coordination Diffusion Transformer for Mobile Manipulation
- Authors: Sixiang Chen, Jiaming Liu, Siyuan Qian, Han Jiang, Lily Li, Renrui Zhang, Zhuoyang Liu, Chenyang Gu, Chengkai Hou, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang. arXiv v3 5 Jul 2025. https://arxiv.org/abs/2507.01961 (venue not stated)
- What: end-to-end diffusion policy coordinating base and arms (mobility-to-body conditioning, adaptive 2D/3D perception weighting).
- Sim: ManiSkill-HAB Set-Table (7 tasks), success = done within 200 steps, 100 episodes x 3; RoboTwin (6 bimanual tasks). Real: Agilex Cobot Magic (4 Piper arms, Tracer base), 17 trials per task.
- MSHab results: pick apple 33.3, place apple 33.3, open fridge 90.7, pick bowl 36.0, place bowl 17.3, open drawer 81.3, close drawer 97.3; mean 55.6 (text says 48.7%, inconsistent).
- Failure causes (App. F): locomotion errors (velocity too high -> unrecoverable, too low -> no motion); manipulation triggered after poor base stop pose outside data distribution; external view may not show the target; depth range limits point cloud use while moving.
- Relevance: articulated open/close is near-solved in MSHab while pick/place from receptacles stays ~17-36% -> receptacle clutter/placement geometry is the bottleneck; base stop pose and target visibility are named causes.
- Tags: VLA-policy, whole-body-wheeled-bimanual, sim-benchmark
