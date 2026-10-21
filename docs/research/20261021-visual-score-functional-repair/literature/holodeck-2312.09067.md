# Holodeck: Language Guided Generation of 3D Embodied AI Environments
- URL: https://arxiv.org/abs/2312.09067 (authors/year not captured in my fetch - unverified; CVPR 2024, Yang et al., from memory only)

## CLIP score (Sec. 4.1)
- Text template: "a top-down view of [scene type]". Model: OpenCLIP ViT-L/14, LAION-2B; score = cosine x 100. View: top-down only. Human-designed iTHOR is the upper bound. Numeric values only in Fig. 10 (not extractable) -> unverified; text says Holodeck "exceed[s] ProcTHOR with great margins and closely approach[es]" iTHOR.
- What it measures: scene-type coherence (does a top-down image look like a bedroom). It has no handle on whether items are well arranged; the prompt contains no layout information.

## Human study
680 graduate students. Holodeck vs ProcTHOR: Asset selection 59.8%, Layout coherence 56.9%, Overall 64.4%. Layout ablation MRR (Table 1): Constraint 0.706 avg (bathroom 0.696, bedroom 0.745, kitchen 0.654, living 0.728), Absolute 0.364, Random 0.369, Edge 0.645.

## Constraint types
- Main text: ten types in five categories: Global (edge, middle), Distance (near, far), Position (in front of, side of, above, on top of), Alignment (center aligned), Rotation (face to). Appendix floor-object list has eight (no above/on top of).
- Thresholds: near = 50 cm < d < 150 cm; far = d >= 150 cm.
- Solver: DFS over (x, y, w, d, rotation in {0,90,180,270}); hard = no collision, inside room; soft = relational constraints, ranked by number satisfied; 30 s budget. MILP variant (Gurobi): all hard except near/far.

## Relevance
Source of the near/far bands and the vocabulary of relations an LLM emits; CLIP evidence is of the "scene type" kind only.
