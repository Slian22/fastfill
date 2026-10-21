# Mind and Motion Aligned: A Joint Evaluation IsaacSim Benchmark for Task Planning and Low-Level Policies in Mobile Manipulation (Kitchen-R)
- Authors: Nikita Kachaev, Andrei Spiridonov, Andrey Gorodetsky, Kirill Muravyev, Nikita Oskolkov, Aditya Narendra, Vlad Shakhuro, Dmitry Makarov, Aleksandr I. Panov, Polina Fedotova, Alexey K. Kovalev. arXiv 21 Aug 2025. https://arxiv.org/abs/2508.15663 (preprint; used in AIJ Contest 2024)
- What: Isaac Sim 5.0.0 digital-twin kitchen; joint evaluation of VLM planner + low-level policy; 563 instructions from 58 templates, plans of 4-8 move/pick/place steps.
- Robot: Mobile ALOHA (base linear/angular velocity + EE pose + gripper); dims not given.
- Success: navigation = base geometric centre within 10 cm of target; manipulation = object centre within 5 cm of target; 120 s per task; waypoint reached < 0.35 m, final stop distance 0.09 m.
- Results: planning exact match 0.000 -> 0.612 (in-context) -> 0.632 (constrained generation); no online SR reported.
- Failure causes: not quantified; notes collision-mesh errors silently cause grasp failures.
- Relevance: concrete navigation/placement success tolerances (10 cm base, 5 cm object) used by a 2025 MoMa benchmark.
- Tags: sim-benchmark, real2sim/digital-twin, long-horizon
