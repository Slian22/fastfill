# OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics
- Authors: Peiqi Liu, Yaswanth Orru, Jay Vakil, Chris Paxton, Nur Muhammad Mahi Shafiullah, Lerrel Pinto (NYU; AI at Meta). arXiv 2401.12202 (2024). https://arxiv.org/html/2401.12202
- What: zero-shot pick-and-drop on Hello Robot Stretch in real homes: voxel semantic memory + A* navigation + AnyGrasp.
- Base pose choice: minimise weighted sum (weights 1, 8, 8) of distance to target, a capped term max(0, 40 - distance) that prefers standing a set distance away, and inverse distance to the nearest obstacle within a 30-unit cutoff (units not stated, unverified); A* on a 10 cm 2D grid, occupied/unexplored cells blocked plus a 20 cm radius around occupied cells.
- Results: 58.5% success over 171 trials in 10 homes; about 82% (82.4%) in decluttered homes.
- Failure causes: semantic memory retrieval 9.3%, difficult pose for the manipulation module 8.0%, hardware 7.5% (leading three). Over cleanup phases navigation errors 15% -> 12% -> 4% and manipulation errors 25% -> 16% -> 13%. Objects in the middle of a bed or on high places hard to reach; clutter drives manipulation failure.
- Relevance: real-home evidence that clutter around the target and out-of-band heights/depths (bed centre, high shelves) are first-order failure causes for a Stretch; 20 cm obstacle inflation on a 10 cm grid is close to our 0.2 m / 0.25 m check.
- Tags: real-home-deploy, open-vocab-nav-pick, 3D-memory/scene-graph
