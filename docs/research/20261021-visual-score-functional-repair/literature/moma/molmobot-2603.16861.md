# MolmoBot: Large-Scale Simulation Enables Zero-Shot Manipulation
- Authors: Abhay Deshpande, Maya Guru, Rose Hendrix + 23 others (26 total; Ai2). arXiv v2 26 Mar 2026. https://arxiv.org/abs/2603.16861
- What: policies trained only on procedurally generated sim data in MolmoSpaces scenes ("94k+ procedurally generated environments with 11k+ object assets"; 5-100 objects sampled per scene), transferred zero-shot to real Franka FR3 and Rainbow RB-Y1 (holonomic base, 6-DoF torso, two 7-DoF arms).
- Data generation checks: grasp candidates filtered for gripper-scene collision and IK reachability; RB-Y1 motion via cuRobo (IK 64 seeds, obstacle activation distance 0.2 m, up to 5 planning attempts); episodes with > 3 retries discarded. Base pose sampling for episodes not described in the text read.
- Open-door task: "Open a nearby hinged door to at least 67% of its hinge joint range"; sim 2,000 episodes: Door Specialist 77.7%, Multitask 70.2%, MolmoBot-SPOC Articulated 58.8%. Real (3 pull doors x 3 trials, varied base positions): handle grasp 4/9, door opened 2/9; failures: hardware e-stop 5/9, base collision 2/9, joint limit 1/9, wrong gripper orientation 1/9.
- Real tabletop pick-and-place: MolmoBot 79.2% vs pi0.5 31.3% (abstract says 39.2%; paper inconsistent).
- Relevance: door opening by a wheeled bimanual robot fails on base collision and joint limits, i.e. on free space beside/behind the door and base pose; argues for a symbolic door-approach zone check in layouts.
- Tags: VLA-policy, data-generation, whole-body-wheeled-bimanual, articulated-objects, RL-sim2real
