# EchoVLA: Robotic Vision-Language-Action Model with Synergistic Declarative Memory for Mobile Manipulation
- Authors: Min Lin, Xiwen Liang, Bingqian Lin, Jingzhi Liu, ... Xiaodan Liang (15 authors). arXiv v1 22 Nov 2025, v3 7 Aug 2026. https://arxiv.org/abs/2511.18112
- What: VLA with scene memory (spatial-semantic maps) + episodic memory feeding mobile-arm diffusion policies; introduces MoMani, an automated long-horizon MoMa trajectory benchmark built with MLLM-guided planning and feedback refinement, plus real-robot demos on a holonomic platform.
- Results (abstract, sim): 0.52 success on manipulation/navigation tasks, 0.31 on mobile manipulation; +0.20 and +0.11 over π0.5. Real-world 0.44 on TidyBot++ is reported in v2 per a search snippet (unverified).
- Scene-related: no layout failure analysis found; motivation is that MoMa needs memory of out-of-view objects.
- Relevance: memory-augmented VLAs still score ~0.3 on full MoMa tasks in sim; scene memory is the 2025-26 answer to partial observability, not layout changes.
- Tags: VLA-policy, 3D-memory/scene-graph, sim-benchmark, data-generation, long-horizon
