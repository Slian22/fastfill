# Evaluating Gemini Robotics Policies in a Veo World Simulator
- Authors: Gemini Robotics Team, Google DeepMind. arXiv 2512.10675 (v1 11 Dec 2025, v2 6 Jan 2026). https://arxiv.org/abs/2512.10675
- What: video-model "simulator" for policy evaluation; scene edits by Gemini 2.5 Flash Image on the overhead frame (novel objects, backgrounds = red/green/blue cloth, small ~3-4 inch and large ~10-12 inch distractor plushies), fine-tuned Veo2 completes the other camera views; edited scenes rebuilt physically for real comparison.
- Robot/tasks: ALOHA 2 bimanual, five tasks, eight policy checkpoints, 1600+ real evaluations.
- Numbers: OOD-axis ranking for one policy MMRV 0.06, Pearson 0.86; object change predicted and observed as the largest drop; predicted absolute success lower than real.
- Validity: no physical validity check of edited scenes (floating, reachability, occlusion not checked); hallucinated object appearing mid-interaction reported; contact-rich small-object interaction "remains a challenge".
- Relevance: world-model evaluation is growing but is tabletop and has no geometric layout guarantees; not a substitute for symbolic room-scale MoMa checks.
- Tags: sim-benchmark, VLA-policy
