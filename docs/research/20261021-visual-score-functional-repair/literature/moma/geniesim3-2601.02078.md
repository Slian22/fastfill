# Genie Sim 3.0: A High-Fidelity Comprehensive Simulation Platform for Humanoid Robot
- Authors: Chenghao Yin, Da Huang, Di Yang + 16 others (19 total; AgiBot). arXiv 2601.02078 v1, 5 Jan 2026. https://arxiv.org/abs/2601.02078
- What: LLM scene generator (intention interpretation -> asset retrieval -> DSL code -> assembly) producing a scene graph with relations on/in/adjacent/aligned/stacked, compiled to USD for Isaac Sim; LLM writes task instructions and eval configs, VLM judges success from observation sequences.
- Validity: waypoints checked for "kinematic reachability, collision avoidance, and anthropomorphic feasibility" in data collection, no thresholds given; failed executions roll back and try the next candidate.
- Robots: AgiBot G1 (sim-to-real) and G2 (wheeled dual-arm form factor: unverified).
- Numbers: 5,140 sim-ready objects / 353 categories, > 10,000 h synthetic data over 200 tasks, > 100,000 eval scenarios; sim-real R^2 = 0.924, slope ~1.045 over 16 models; per task sim/real: Select Color 0.86/0.85, Recognize Size 0.93/0.94, Grasp Targets 0.80/0.71, Organize Objects 0.52/0.60.
- Failure attribution: to contact/friction approximations, not to layout.
- Relevance: industry-scale LLM scene generation with VLM-judged success but no published geometric executability thresholds; shows the field relies on rollout filtering rather than layout-side guarantees.
- Tags: scene-gen-for-robots, sim-benchmark, data-generation, humanoid
