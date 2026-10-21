# AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation
- Authors: Konstantin Gubernatorov, Artem Voronov, Roman Voronov, Sergei Pasynkov, Stepan Perminov, Ziang Guo, Dzmitry Tsetserukou. arXiv 2509.21006 (25 Sep 2025); no venue. https://arxiv.org/html/2509.21006
- What: SLAM + semantic map + frontier exploration + fine-tuned SmolVLA on an SO-101 arm; pick-and-place in a multi-room lab.
- Base pose rule: segment the support surface, PCA normal, place the base outside the edge at a "user-defined offset" oriented perpendicular to the surface; Nav2 checks reachability/collision; try nearby edge candidates on failure. No offset, inflation or reach values given.
- Results (50 episodes): overall 46%; module SR: SLAM 100, exploration 75, navigation 90, detection 85, VLA manipulation 80 (text says 85). Exploration failures (25%) mostly from "cluttered, tight spaces and corridors". Without fine-tuning overall 10%.
- Relevance: the common real-system base-pose heuristic is "perpendicular to the receptacle edge at fixed standoff" -> a layout check can test that the strip in front of each receptacle edge is free.
- Tags: VLA-policy, open-vocab-nav-pick, base-placement
