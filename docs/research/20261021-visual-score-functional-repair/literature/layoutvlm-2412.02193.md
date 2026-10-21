# LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models
- Authors: Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, Jiajun Wu. 2024 (v3 Mar 2025). https://arxiv.org/abs/2412.02193
- Relations (Table 1): Distance [d_min,d_max] loss clamp(min(|pi-pj|-dmin, dmax-|pi-pj|),0,1); On-top-of (neg DIoU, z set); Align-with 1-cos(vi,vj); Point-towards 0 if vi.dij>0 else 1-cos(vi,dij); Against-wall (clamped corner-wall distances + orientation term). Fixed numeric eps for self-consistency not given.
- Optimisation: L_semantic (only relations satisfied on initial VLM poses, so they act as "protected") + L_physics (Distance-IoU over all pairs); Adam, decay 0.96, 400 steps, boundary projection every 100 its.
- Metrics: Collision-Free, In-Boundary, Positional/Rotational coherency, PSA (0-100). GPT-4o rates from top-down + side renders.
- Human agreement Kendall tau (Table 4): within users Pos 0.51 / Rot 0.57 / PSA 0.50; users vs GPT-4o 0.49 / 0.61 / 0.46 (5 grad students, 495 ratings per method-metric).
- Relevance: filtering to relations satisfied before optimization is a cheap protection rule; GPT-4o top+side agreement ~0.46-0.61 tau.
