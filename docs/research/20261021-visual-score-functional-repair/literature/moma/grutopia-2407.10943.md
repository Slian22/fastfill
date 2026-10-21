# GRUtopia: Dream General Robots in a City at Scale
- Authors: Hanqing Wang, Jiahe Chen, Wensi Huang, Qingwei Ben, ... Jiangmiao Pang (OpenRobotLab, Shanghai AI Lab). arXiv 15 Jul 2024. https://arxiv.org/abs/2407.10943
- What: Isaac-Sim platform; GRScenes ~100k scenes from designer websites (89 categories), first release 100 curated scenes (70 home, 30 commercial) with 2,956 interactive + 22,001 non-interactive objects from 96 categories; designers "distributed objects to follow human habits"; GRResidents LLM NPCs; GRBench.
- Robots: Unitree H1 (navigation, RGB-D), Unitree Aliengo + Z1 arm with RGB-D on a pole 0.8 m above the base (manipulation); also G1, GR-1 APIs. No footprint/reach numbers.
- Tasks/success: Object Loco-Navigation: success if target "in the agent's field of view and less than 3 m away"; Social Loco-Navigation: same, up to three NPC dialogue rounds; Loco-Manipulation: object placed satisfying all task conditions (at most two). STOP within 14,400 physics steps; 300 episodes/benchmark (100 val, 200 test).
- Results (test SR): Object Loco-Nav GPT-4o 14.00, LLM agents 15.50-22.00; Social 2.00-12.50; Loco-Manipulation 0 for all.
- Failure causes: H1 locomotion success 100% flat -> 58% furnished house, Aliengo 100% -> 14% "mainly caused by the failure of path or motion planning"; manipulation 100% -> 8% in house; loco-manipulation falls attributed to Aliengo's "larger turning radius and base size" and arm collisions with scene objects.
- Relevance: furnished-home clutter alone cuts legged-base navigation from 100% to 14-58%; nav success uses a visibility + 3 m rule (a cheap symbolic proxy for "see each task object"). No clearance threshold given.
- Tags: sim-benchmark, humanoid, open-vocab-nav-pick, scene-gen-for-robots
