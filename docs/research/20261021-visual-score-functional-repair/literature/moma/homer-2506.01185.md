# HoMeR: Learning In-the-Wild Mobile Manipulation via Hybrid Imitation and Whole-Body Control
- Authors: Priya Sundaresan et al. (Stanford; incl. Dorsa Sadigh, Jeannette Bohg; full list not checked). arXiv 2506.01185, Jun 2025. https://arxiv.org/abs/2506.01185
- What: hybrid policy (absolute EE poses for long moves, relative for fine manipulation) on top of a kinematic whole-body controller (MuJoCo + mink IK) that moves base and arm together.
- Robot: TidyBot++ holonomic base + 7-DoF Kinova arm; base velocity caps (0.5, 0.5, π/2) m/s, rad/s. Dimensions/height range not reported.
- Tasks: real home: Pillow, TV Remote (open cabinet, retrieve, place on stand), Sweep Trash; sim: Cube, Dishwasher, Cabinet Opening. Overall 79.17% with 20 demos per task, +29.17% over the next-best baseline (per-task values only in figures).
- WBC: optional self-collision avoidance (arm/gripper/base/camera mounts) with 2 cm margin, 10 cm detection range; external environment collisions not handled.
- Failure causes: decoupled base+arm variant fails "when base misalignment affects arm reachability" and on large appliances (cabinet, dishwasher); dense-delta DP drifts under randomized robot/object poses. Narrow spaces and occlusion not discussed.
- Relevance: whole-body control relaxes, but does not remove, the need for a free base region in front of articulated appliances.
- Tags: real-home-deploy, base-placement, articulated-objects, teleop-data
