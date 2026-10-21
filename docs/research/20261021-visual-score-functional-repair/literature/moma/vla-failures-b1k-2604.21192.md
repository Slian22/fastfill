# How VLAs (Really) Work In Open-World Environments
- Authors: Rasouli et al. (full list not read). arXiv 23 Apr 2026. https://arxiv.org/abs/2604.21192
- What: expert video review (8 robotics experts) of the top BEHAVIOR-1K 2025 challenge policies (RLC winner, Comet runner-up): 500 recordings = 50 tasks x 10 trials.
- Taxonomy (10): task confusion, abrupt termination, semantic confusion, navigation failure ("position itself, localize the target, or find a safe passage"), improper object handling, skill failure (incl. open/close), collision (hits furniture/containers not in focus), execution order confusion, placement failure, grasp failure.
- Frequencies: only in figures; text: "Fail to grasp is the most common source of error"; order confusion more likely than collision, almost matching placement failures; handling and skill issues more than navigation. No reachability or occlusion category.
- Scene notes: scenes "are generally not cluttered"; flat objects (books on tables) must be slid "to the edge of the surface" before grasping.
- Context: best 2025 challenge solution "only 12.4% average success rate" (cited in MPVI, arXiv 2606.00985, which adds A* navigation + frontier exploration to a pi0.5 VLA and reports +113% mean Q-score; occlusion failures fixed by exploration, qualitative only).
- Relevance: in the main 2025 MoMa benchmark, failures are dominated by grasp/placement and task logic, with navigation and collision secondary but present; low-clutter scenes are the norm, so clutter is not yet a tested axis.
- Tags: VLA-policy, sim-benchmark, long-horizon
