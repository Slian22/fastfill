# Designing Human-Robot Coexistence Space
- Authors: Jixuan Zhi, Lap-Fai Yu, Jyh-Ming Lien. arXiv 2011.07374 (14 Nov 2020; venue not stated). https://arxiv.org/abs/2011.07374 (read via ar5iv)
- What: furniture layout optimisation (simulated annealing, up to 700 iterations, M = 10 object sequences) so that an autonomous wheelchair (non-holonomic car, turning radius "three meters") can tour every object in sequence; configurations sampled around each object facing its reference point (count scales with object perimeter and wheelchair size); RRT with Reeds-Shepp curves, path cost = length + accumulated rotation + clearance (mean distance to nearby objects).
- Numbers: planning is 96.06% of compute; their planner reuse cuts planning time 65-70%; full design 3-5 min vs 10-20 min without; small bedroom (36% smaller area) path finding ~30% slower. No minimum passage width, no success rate.
- Limits: few objects/types, rectangular rooms only, sequences grow exponentially.
- Relevance: layout-level "every object has a reachable approach pose from a vehicle with a large footprint" check, done by planning; no arm reach, height or visibility.
- Tags: base-placement
