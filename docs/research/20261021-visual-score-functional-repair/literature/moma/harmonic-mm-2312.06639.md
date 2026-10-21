# Harmonic Mobile Manipulation
- Authors: Ruihan Yang, Yejin Kim, Rose Hendrix, Aniruddha Kembhavi, Xiaolong Wang, Kiana Ehsani. arXiv 11 Dec 2023 (v3 5 Dec 2024). https://arxiv.org/abs/2312.06639 (read: https://arxiv.org/html/2312.06639v3)
- What: end-to-end RGB RL policy that coordinates base and arm for door opening (push/pull), fridge opening and table cleaning; adds procedural door/fridge articulation to ProcTHOR.
- Robot: Hello Robot Stretch (3-DOF arm), RealSense D455; no dimensions given in the paper.
- Scenes: 700 two-room houses (doors), 2000 houses with 1-3 rooms each with a table (cleaning), 1400 furnished kitchens (fridge); eval 350 unseen houses, 1200 episodes (300 per task). Start 1-4 m from table, 1-3 m from door/fridge; 500-step cap.
- Success: door progress > 90%; fridge > 70%; table cleaning > 75%; grasp if end-effector within 0.2 (units not stated) of the object.
- Sim success / progress (%): table 36.7 +-22.9 / 76.0 +-11.3; door push 80.0 +-19.9 / 89.7 +-5.0; door pull 51.2 +-12.2 / 69.8 +-2.7; fridge 20.0 +-3.1 / 46.2 +-5.4. Real: table 66.6% (12 trials), door pull 9/15, door push 7/10, fridge not evaluated (seal too strong).
- Scene-caused failures stated: door pull suffers from "reduced space as the door opens, especially in narrow areas"; table cleaning hurt by surrounding obstacles/furniture; fridge hurt by small door and "confined placement in rooms". No door width, swing direction or clearance numbers given (unverified beyond the paper text).
- Relevance: pull doors are markedly harder than push (51.2 vs 80.0 in sim) and the cited cause is free space on the swing side -> a layout check should keep the door swing sector plus a robot-sized approach zone free, especially on the pull side.
- Tags: RL-sim2real, articulated-objects, sim-benchmark
