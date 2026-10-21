# M3Bench: Benchmarking Whole-body Motion Generation for Mobile Manipulation in 3D Scenes
- Authors: Zeyu Zhang, Sixu Yan, Muzhi Han + 4 others (7 total; BIGAI, lead Hangxin Liu). arXiv 2410.06678 v3; accepted 9 May 2025 (IEEE RA-L per listings; unverified). https://arxiv.org/abs/2410.06678
- What: 30,000 pick / place tasks in 119 household scenes "curated from PhyScene" (32 object types, 588 objects); expert whole-body trajectories from M3BenchMaker (energy-based grasp/placement candidates + VKC trajectory optimisation, "collision-free and kinematically feasible"); conditional scene sampler uses distance/angle thresholds (values not stated).
- Robot: 7-DoF Kinova Gen3 + parallel gripper on an omnidirectional base (footprint not given).
- Success: no collision along trajectory and goal state held for 2 s, verified in Isaac Sim.
- Results, Base Test (Succ / goal dist / env-collision): ModMP[O] pick 20.13% / 0.01 m / 9.70%, place 2.76% / 0.29 m / 2.65%; MDiffusion pick 18.12% / 0.04 m / 19.09%, place 5.83% / 0.04 m / 39.67%; Mpi-Net pick 0.07%, Mpi-Former 0.00%. Novel Scene pick (ModMP[O]) 25.59%.
- Failure causes: optimisation failures and environment collisions; learned methods violate joint limits and collide (place env-collision up to 42.75%).
- Relevance: in generated-layout houses, whole-body pick succeeds <= ~26% and place <= ~6%; environment collision is the dominant learned-policy failure, so free space around receptacles (not just reachability) matters.
- Tags: sim-benchmark, base-placement, data-generation
