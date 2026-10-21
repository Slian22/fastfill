# BUMBLE: Unifying Reasoning and Acting with Vision-Language Models for Building-wide Mobile Manipulation
- Authors: Rutav Shah, Albert Yu, Yifeng Zhu, Yuke Zhu, Roberto Martin-Martin (UT Austin). arXiv 8 Oct 2024. https://arxiv.org/abs/2410.06237 (venue not checked)
- What: VLM-driven skill sequencing (up to 12 skills, ~15 min) across 3 university buildings, multiple floors with elevator use; tasks: retrieve marker, retrieve diet soda, rearrange chairs.
- Robot: not named in the text read (unverified).
- Results: 47.1% average over 70 trials (per-setting 60/40/50/40/50/40/50%); COME 35.7%, Inner Monologue 6.7%; offline skill-parameter accuracy 80.2% vs 72.6% / 61.7%.
- Failure causes: 10/38 failures from sensors (bad depth, lidar) or GSAM segmentation; most others VLM reasoning: wrong object, "misadjusting the base", failing interactive navigation. Wrong-object picks 38.9% with 20-25 distractors vs 10.0% with 5-10.
- Relevance: real deployment where base adjustment and distractor density are named failure causes; wrong-object rate quadruples with distractor count -> object density near task objects matters.
- Tags: real-home-deploy, long-horizon, VLA-policy
