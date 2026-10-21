# Whole-Body Mobile Manipulation using Offline Reinforcement Learning on Sub-optimal Controllers (WHOLE-MoMa)
- Authors: Snehal Jauhri, Vignesh Prasad, Georgia Chalvatzaki. arXiv 14 Apr 2026. https://arxiv.org/abs/2604.12509 (read: https://arxiv.org/html/2604.12509)
- What: offline RL over demonstrations from sub-optimal whole-body controllers for articulated-object MoMa.
- Robot: TIAGo++ (holonomic base, two 7-joint arms + 2 gripper joints; 21-D state). Isaac Sim with GAPartNet articulated objects.
- Tasks: L1 Door "push open a door and navigate through without colliding" (600 steps, 15 s); L2 Drawer close-one/open-another with handles "intentionally spaced far apart"; L3 Cupboard open + place with the other arm (900 steps, 22.5 s); 40 Hz.
- Success: binary task completion (r_success = 1.0); no numeric thresholds given.
- Sim success (50 eps): Door WBC 86%, BC 78%, WHOLE-MoMa 98%; Drawer 68 / 70 / 80%; Cupboard 52 / 48 / 78%. Real (25 trials): Drawer 20/25, Cupboard 17/25.
- Scene: no door, corridor or passage widths reported; no scene-caused failure analysis (failures are timing/force/pose tracking).
- Relevance: confirms "push door and drive through" is now a standard MoMa sub-task in 2026 but gives no clearance numbers to reuse.
- Tags: RL-sim2real, whole-body-wheeled-bimanual, articulated-objects
