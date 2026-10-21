# SceneCritic: A Symbolic Evaluator for 3D Indoor Scene Synthesis
- Authors: Kathakoli Sengupta, Kai Ao, Paola Cascante-Bonilla. 2026. https://arxiv.org/abs/2604.13035

## What is scored
Floor-plan layouts against SceneOnto (ontology from 3D-FRONT, ScanNet, Visual Genome; 79 categories). Object- and relation-level.
- Spatial semantics = 0.33 S_scale + 0.33 S_cooccur + 0.33 S_complete.
  - Scale: hard fail if dim < p5/k or > p95*k, k=2.0; soft fail if outside [p5-0.05 m, p95+0.05 m]; scores 1 / 0.5 / 0.
  - Co-occurrence plausible if f=max(P(li|lj),P(lj|li)) > 0.01.
  - Proximity: pass if f>=0.7 and d_min<=2.0 m; pass if 0.2<=f<0.7 and d_min<=3.5 m; else fail. (cooccur_thresh 0.2, func_thresh 0.7, tuned on ProcTHOR-10K.)
- Orientation: checks back-to-wall, faces-centre, faces-pair. Pass if angle error <= 75 deg (soft threshold), soft fail <=150 deg, hard fail otherwise; object score = 1 if any applicable check passes. Ontology mining thresholds: back-to-wall within 45 deg, faces-centre within 60 deg, faces-pair within 60 deg and 5 m radius (3D-FRONT only; nearest paired object's direction used at verification).
- Overlap: AABB proximity overlap and OBB (SAT) true overlap, each failing at overlap > 0.2 m.
- Support tolerance 5 cm (ontology construction).
- Examples: back-to-wall: sofa, bed, dresser, bookshelf; faces-centre: sofa, chair, TV. Pair-specific numbers (bed-nightstand etc.) are not quantified in the paper.

## VLM judge baseline
Gemini-2.5-Pro on Blender renders (Objaverse assets scaled to layout dims). Views: top-view (two variants), left, right, front. Prompt text not given (unverified); described as covering "semantic consistency, orientation correctness, and overlap".

## Agreement with humans (16 annotators, 594 pairwise judgments)
- SceneCritic agreement: 94.44% (easy), 83.33% (complex). VLM: 58.82% (easy), 47.06% (complex). Overlap: SceneCritic 100%.
- Fleiss kappa easy (Sem/Ori/Ovlp) 0.30/0.46/0.39; complex 0.64/0.59/0.72. Inter-annotator agreement 81.60% / 91.32%.

## VLM failure modes (quoted findings)
- View sensitivity: rankings reverse depending on view. Re-scoring the same top-view image gave semantic variance 12.46 (LayoutGPT+Gemini), 19.32 (LayoutGPT+Qwen72B).
- Mirror asymmetry: left vs right views of same scene get different orientation scores.
- Overlap errors: variance highest where errors frequent (LayoutVLM+Gemini 11.2 Left, 28.1 Right, 17.5 Front).
- VLM ranked Holodeck+Qwen72B top (59.46); SceneCritic ranked LayoutVLM+Qwen72B top (80.32). Biggest gap: VLM put LayoutGPT 75 points above Holodeck on easy scenes, humans the opposite.
- Authors: VLM "not reliably extracting geometric relationships from 2D views."

## Relevance
Gives a ready-made rule set with thresholds (orientation 75/150, proximity 2.0/3.5 m, overlap 0.2 m) and the single most important caution: single-view single-score VLM judging is unstable and sometimes below chance. Use symbolic checks for geometry, VLM only for gestalt.
