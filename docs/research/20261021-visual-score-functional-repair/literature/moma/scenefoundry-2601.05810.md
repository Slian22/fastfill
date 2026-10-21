# SceneFoundry: Generating Interactive Infinite 3D Worlds
- Authors: ChunTeng Chen, YiChen Hsu, YiWen Liu, WeiFang Sun, TsaiChing Ni, ChunYi Lee, Min Sun, YuanFu Yang. arXiv 2601.05810v2 (16 Jan 2026); venue not named. https://arxiv.org/html/2601.05810
- What: LLM floor-plan/graph + diffusion layout with PhyScene-style guidance and an Articulated Object Collision Constraint: each articulated box is "heuristically expanded" along its primary articulation axis (b'_i), penalty phi_articoll = sum_i sum_j IoU3D(b'_i, b_j). Joint ranges / sweep size not given. Walkable post-process: R_walkable = (A_room - sum A_i) / A_room; swap largest objects for smaller retrievals until R_walkable >= tau (tau 0.60-0.95 tested). No agent size, grid or robot.
- Results (100 scenes): articulated collision rate R_acoll 0.191 -> 0.109 with the constraint, R_reach 0.742 -> 0.808. Ablation baseline vs full: Col_obj 0.279 -> 0.249, R_walkable 0.774 -> 0.822, R_reach 0.742 -> 0.830. Object-count control success 0.95-0.97. FID 29.02.
- Stated limits: box expansion is a heuristic, may be over-conservative or insufficient for multi-joint objects; ~300 s per three-room apartment.
- Relevance: even with an explicit constraint ~11% of articulated objects stay blocked; "walkable" here is a free-area ratio, not connectivity, so it does not test base access.
- Tags: scene-gen-for-robots, articulated-objects
