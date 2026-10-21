# CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation
- Authors: Yushan Liu, Peibo Sun, Xintao Chao, Zhenyang Yang, ... (11 authors; corresponding Wenbo Ding). arXiv 29 Jul 2026. https://arxiv.org/abs/2607.26789
- What: world-model verifier that decides when to replan a VLA during long-horizon MoMa; evaluated on RoboCasa365 (365 tasks, 2,500 kitchens), training on Human300 (300 tasks x 100 demos); 50 target tasks: 18 atomic-seen, 16 composite-seen, 16 composite-unseen. Robot not named.
- Results (avg %, A-S / C-S / C-U / mean): CheckVLA 63.7 / 30.9 / 10.2 / 36.1; Qwen-RobotManip 68.6 / 20.1 / 14.9 / 35.9; π0.5 (published) 39.6 / 7.1 / 1.2 / 16.9; periodic replanning at matched calls 27.6.
- Residual failures (140, blinded audit, kappa 0.79): missed/late alarm 31.4%, physically unrecoverable state 23.6%, world-model ambiguity under occlusion or contact 18.6%, semantic/action error 14.3%, repair-induced regression 7.1%, unnecessary intervention 5.0%.
- Relevance: composite (multi-step) kitchen MoMa is ~10-30% even for 2026 VLAs; occlusion/contact ambiguity is a named residual cause, base placement is not.
- Tags: VLA-policy, sim-benchmark, long-horizon
