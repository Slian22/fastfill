# Mobi-pi: Mobilizing Your Robot Learning Policy
- Authors: Jingyun Yang, Isabella Huang, Brandon Vu, Max Bajracharya, Rika Antonova, Jeannette Bohg. arXiv v2 26 Sep 2025. https://arxiv.org/abs/2505.23692 (venue not stated on arXiv page)
- What: "policy mobilization": find a base pose p in SE(2) in a new scene that puts a fixed-base policy in distribution. Score K(p) = 0 unless object visible (MiniCPM-v2 check) and pose collision-free (occupancy map); else KNN (k=5) DINO-feature distance between 3DGS-rendered view and demo frames; Bayesian optimisation.
- Benchmark: 5 RoboCasa tasks (close door, turn on faucet, turn on microwave, close drawer, turn on stove); Franka on Omron base; policies trained on 300 MimicGen demos over 5 layouts; 5 held-out layouts; horizon 500 steps.
- Metrics: spatial feasibility phi = ln2/gamma, the base-pose noise (m) that halves success; hardest task Turn on Stove phi = 0.031 m. Visual feasibility S_v = mean image fraction covered by task objects at near-optimal poses; Close Drawer spatially easy but visually hardest.
- Results: sim average 0.422 (full) vs 0.006 without K_id; real full success 70% / 100% / 80% (pick chips / grab paper towels / pour tomatoes), navigation 100%; scene up to 36 m^2.
- Relevance: success is extremely sensitive to base pose (3.1 cm halves success on the stove task) and occlusion causes abrupt score jumps -> layout checks should require a free base region plus a clear line of sight to the target, not a single feasible point.
- Tags: base-placement, VLA-policy, real2sim/digital-twin
