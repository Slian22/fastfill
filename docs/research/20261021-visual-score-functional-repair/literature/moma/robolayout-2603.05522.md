# RoboLayout: Differentiable 3D Scene Generation for Embodied Agents
- Author: Ali Shamsaddinlou. arXiv 2603.05522v1, 18 Feb 2026; no venue. https://arxiv.org/html/2603.05522v1
- What: extends LayoutVLM's differentiable optimisation with a robot clearance term: for movable, non-stacked asset pairs, centre distance D_ij >= R_ij = e_i + e_j + 2r, e_i = 0.5 * min(w_i, h_i) (footprint as disc), r = robot radius; loss alpha * max(0, R_ij - D_ij) on squared distances, alpha = 0.01; term off if r <= 0.
- Measurement: analytic pairwise distances only; no grid, path planning, reach, height or interaction regions. Robot radius values not given.
- Results: qualitative only (4 example scenes, 5-10 objects, GPT-4o); no metrics, no baseline numbers.
- Stated open: replace clearance with real motion planning; add vertical reach; sampling/MIP solvers.
- Relevance: shows the field's current "robot-aware layout" term is a pairwise gap of 2r between footprints - weaker than a connected-grid check (a chain of gaps >= 2r does not imply a path to each object) and with no base-pose or reach test.
- Tags: scene-gen-for-robots
