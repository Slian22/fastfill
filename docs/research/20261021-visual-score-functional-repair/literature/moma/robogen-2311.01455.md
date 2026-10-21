# RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation
- Authors: Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson, David Held, Chuang Gan. Nov 2023, ICML 2024. https://arxiv.org/abs/2311.01455 (HTML v3 read)
- What: GPT-4 proposes a task, then assets (Objaverse retrieval, k=10, Sentence-BERT rank, Gemini-Pro verification), asset sizes, initial joint states and object locations; decomposes into subtasks; picks RL / motion planning / trajectory optimisation. Single-scene household tabletop/appliance setups (ovens, microwaves, drawers), no rooms or navigation.
- Robot: fixed robotic arm (Franka example, suction gripper); legged robots for locomotion. No mobile base.
- Success: fraction of runs that learn the skill; "an average success rate of 0.774" over 69 tasks (50 object manipulation, 7 soft-body, 12 locomotion). Per-category rates in App. B.2 (not read, unverified).
- Scene checks: VLM asset filtering; LLM size plausibility ("the size of the drawer should be larger than the book"); initial-state collisions resolved by pushing along collision normals (App. A.2), no thresholds. No reachability/navigation/articulation-clearance check.
- Scene-caused failures: 13 of 155 generated object-manipulation tasks failed from incorrect scene generation: asset lacks required functionality (printer with no movable tray), wrong joint-value-to-semantic-state mapping, no asset with precise spatial match (staples/stapler).
- Relevance: task-first scene generation precedent, but tabletop/fixed-arm; supplies no MoMa layout thresholds.
- Tags: scene-gen-for-robots, data-generation, articulated-objects
