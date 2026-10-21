# Improving Human-Robot Collaboration via Computational Design
- Authors: Jixuan Zhi, Jyh-Ming Lien (George Mason Univ.). arXiv 2303.11425 (2023; venue not stated). https://arxiv.org/abs/2303.11425 (read via ar5iv)
- What: kitchen counter layout (positions/orientations along walls) jointly optimised with human and robot paths for a recipe (two-burger, 7 sub-tasks); simulated annealing + prioritized time-aware RRT (goal bias 80%, <= 5 extension steps), Pareto set of 5; Unity.
- Costs: distance/rotation to wall; path length, completion time, narrowness (clearance). No reachability term; robot size not specified (human and robot treated as similar agents).
- Results: 10 runs -> 50 Pareto solutions; all with zero narrowness cost; separate layout-then-path optimisation "above 10% higher" cost than joint; L-shaped kitchen narrowness about 3x a regular kitchen; 36%-smaller kitchen costs higher. Exact costs only in figures.
- Relevance: environment-side co-design exists but at counter level, path-only; thousands of planner calls per design - an argument for cheap symbolic proxies.
- Tags: (none of the fixed tags applies; environment co-design)
