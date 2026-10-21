# OptiScene: LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization
- Authors: Yixuan Yang, Zhen Luo, Tongsheng Ding, Junru Lu, Mingqi Gao, Jinyu Yang, Victor Sanchez, Feng Zheng. 2025. https://arxiv.org/abs/2506.07570

## GPT-4o protocol (Table 2, App. D.2, Table 9)
- Single-layout scoring, integers 0-10, on Functionality, Layout (placement), Aesthetics, plus overall_score (not reported) and 1-3 sentence comments. Follows I-Design protocol. 0 = "completely inconsistent", 10 = "perfectly aligned". "Be fair and critical."
- Role: "expert in interior design and human-centric spatial planning". Functionality: daily activities, accessible groupings (desk and chair), circulation. Layout: within bounds, not blocking windows/doors, wall alignment, overlaps. Aesthetics: balance, proportion, symmetry/grouping.
- NOT reported: views/renders passed, number of scenes, temperature, any human-agreement numbers (unverified/absent).
- Results (Func/Layout/Aes): 3D-Front Bedroom OptiScene 8.6/9.1/9.3 vs DiffuScene 7.8/8.1/8.4; SynthPlace Living 8.9/9.2/9.1 vs DiffuScene 7.9/7.8/8.3; Kitchen 8.4/8.3/8.8; Bathroom 8.7/9.4/9.0. All near ceiling.
- Other: OOR = pairwise 2D bbox intersection area / total box area. Usability Rate (human, 5 annotators, >=3 agree): ablation 28% -> 33% -> 40% -> 50%.
- Prompt-side functional rules: chairs near and facing desks, place along walls, traffic flow, no overlap. No automatic functional metric.

## Relevance
Our own lineage; scores saturate (8-9.4) with no reported human agreement -> not a trustworthy discriminator. Rubric text is reusable.
