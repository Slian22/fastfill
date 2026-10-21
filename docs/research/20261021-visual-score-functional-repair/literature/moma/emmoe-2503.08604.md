# EMMOE: A Comprehensive Benchmark for Embodied Mobile Manipulation in Open Environments
- Authors: Dongping Li, Tielong Cai, Tianci Tang, Wenhao Chai, Katherine Rose Driggs-Campbell, Gaoang Wang. arXiv v2 15 May 2025. https://arxiv.org/abs/2503.08604 (no venue listed)
- What: language-instructed long-horizon MoMa (short, long >= 10 subtasks, open-ended, logical, human-style); EMMOE-100 tasks; HomieBot (SFT+DPO 7B planner).
- Robot/scenes: Fetch in Habitat-Lab 2.0 (v0.2.3); Replica Challenge scenes; 100 scenarios, 90 train / 10 test.
- Success: full goal completion (TP = 100%); 3 runs per task, 20-step limit; also SER, SRR, PLWSR.
- Results: best SR 30.30 (HomieBot SFT+DPO); o1 28.67; GPT-4o 13.33; Gemini-1.5-Pro 17.33.
- Low-level execution (Table 4): share of single-action errors / per-skill SR: Go to 38.49 / 45.32; Pick 49.77 / 22.45; Place 7.30 / 40.97; Open 3.32 / 43.13; Close 1.11 / 36.45.
- Failure causes: error code F2 (target missing or unrecognizable) dominates failed trajectories; D1 (agent too far from target) is most common in successful ones. Per-category percentages only in figures (unverified).
- Relevance: navigation ("Go to") and pick account for ~88% of execution errors; "too far from target" is a distinct failure -> a layout should leave a free base pose close enough to each pick target.
- Tags: sim-benchmark, long-horizon, VLA-policy
