# ManipulaTHOR: A Framework for Visual Object Manipulation
- Authors: Kiana Ehsani, Winson Han, Alvaro Herrasti, Eli VanderBilt, Luca Weihs, Eric Kolve, Aniruddha Kembhavi, Roozbeh Mottaghi. arXiv 22 Apr 2021, "CVPR 2021 -- (Oral presentation)". https://arxiv.org/abs/2104.11213
- What: AI2-THOR extension with an arm on the agent body; ArmPointNav task: navigate to object, grasp, carry to target, release; source/target given as relative (x, y, z). Dataset APND: 30 kitchen scenes (20/5/5), 12 pickupable target categories, ~72k object locations.
- Robot: Kinova-inspired arm, 6-DoF hand; reach = hemisphere of radius 0.6335 m centred at the first joint; arm height adjustable along the body (range not given, unverified); abstract sphere grasp. Actions: base forward 20 cm, rotate 45 deg; wrist 5 cm steps in +-x/y/z; arm up/down 7 cm. Base footprint and camera height not given (unverified).
- Success: object reaches goal within 200 steps (distance threshold not given, unverified); SRwD additionally requires no collision with / movement of other objects.
- Reachability filter: only object locations "for which a valid agent pose can reach the object" are kept; only visible objects considered.
- Results (Depth model): Test-SeenObj SR 68.7 vs SRwD 39.4, PuSR 89.9; Test-NovelObj SR 62.1 / SRwD 32.7. Gap SR -> SRwD = disturbing other objects (clutter); occlusion, 3D obstacle avoidance and clutter named as difficulties without breakdown.
- Relevance: explicit precedent for "keep only targets that have a reachable base pose" in episode generation (done by sampling, not reported as a rate); reach 0.6335 m is a small-arm reference.
- Tags: sim-benchmark, base-placement
