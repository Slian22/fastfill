# Findings

## Current understanding
- S2-F numbers (interim, agent not finished): cohort 1,009 — checker S1 0.924 / S2 0.986 / S2F 0.987; strict 0.534 / 0.814 / 0.812; companion retention of S1 pairs 1.000 / 0.853 / 0.966; GT companion score 0.757 / 0.650 / 0.735; median displacement of moved objects S2 1.00 m vs S2F 0.72 m. Dev 600: S2F keeps 0.989 of S2's checker gain and 0.978 of its strict gain; retention 0.809 -> 0.965.
- But by eye on three of the previously bad rooms (A#06, A#09, A#19) S2-F looks like S2: the nightstand still leaves the bed head (#06) and the TV bench still sits beside the bed (#19); the facing rule is satisfied on paper while the result is still wrong.
- GPT pairwise judge (interim): reference beats raw only 13/20 (65%, below the locked 70% bar); S1 vs S2 10/10; the first-shown layout is chosen 15/20 -> strong position bias.
- (before this project) S2 raises checker/strict a lot, but by eye it makes 9 of 20 rendered rooms worse: companion furniture leaves its anchor.

## Patterns and insights
- **H1b supported (CLIP measured, probes-20261021/clip-metric/report.md):** on 1,005 cohort rooms (identical cameras/settings), GT beats the 27B raw layout in only 43.7-53.3% of rooms across CLIP ViT-L/14 and SigLIP-so400m, text and top/perspective views; in the perspective view GT is even significantly lower (-0.20 [-0.36,-0.06]); Spearman(GT-raw, raw violations) |rho| <= 0.05; AUC checker-pass vs fail 0.46-0.58; image similarity to the GT render rho -0.09..0.00. Shuffled furniture text moves the score by only 0.25-0.35 vs a 2.5-4.1 between-room spread: the score is room type + asset look + framing, not arrangement. On R20, sign agreement with the by-eye S2-vs-S1 labels 3-9 of 13 (chance ~6.5).
- Literature (literature/survey.md, 15 papers): text-image scores do not track layout quality. Kendall tau with human ratings: CLIPScore 0.06, BLIPScore 0.07, VQAScore 0.13, GPT-4o top-down 0.27, GPT-4o multi-view first-person 0.39 (VIPScene arXiv 2506.20601 Table 2, verified). VQAScore 58.6% agreement with human pairwise preference (Procedural Scene Programs); CLIPScore Cohen kappa 0.02 per scene (LEGO-Eval). CLIP is near chance on spatial relations (ARO VG-Relation 0.59 vs 0.50 chance; What'sUp 26-31% vs 25%).
- Others use CLIP for prompt alignment (text-to-scene), not for arrangement quality; geometry is scored symbolically and quality by a VLM judge or humans. Symbolic checks agree with humans 94%/83% vs 59%/47% for a VLM judge (SceneCritic).
- Literature thresholds for companion relations: next_to 0-0.5 m, near 0.5-1.5 m (SceneEval/Holodeck); facing within 30 deg full credit, 75 deg pass (Roomer); passage/dining clearance 0.60 m.

- **H3 partly supported (probes-20261021/s2f/REPORT.md):** S2-F keeps 98.9%/101.6% (dev/cohort) of S2's checker gain and 97.8%/99.3% of its strict gain, companion pairs kept 0.853 -> 0.966 (cohort; nightstand 0.81 -> 0.94, dining chair 0.82 -> 0.98, sofa group 0.88 -> 0.99), median move 1.00 -> 0.72 m; but retention stays below S1 (end to end 0.943 vs 0.974; GT-based 0.735 vs 0.757) -> the 'retention >= S1' part fails. By eye (second rater, non-blind, 40 rooms): S2-F vs S1 better 14 / same 19 / worse 7; vs S2 better 7 / same 32 / worse 1 (25 of 40 layouts identical to S2).
- Rater noise is large: the two by-eye raters disagree on 5 of 12 shared S2-vs-S1 rooms. With a position-biased judge as well, no single visual signal here is reliable yet; human pairwise labels are needed to calibrate.
- Remaining visual failures need rules S2-F lacks: toilet/sink corner, dressing table + chair, generic chair/table, and a TV-stand rule stricter than a facing cone (distance band 1.5-4 m and not adjacent to the bed side).

## Lessons and constraints
- The API (krill-code) returned "servers are currently busy" for every call from ~05:40 local; text calls worked at ~05:32.
- Python urllib needs a User-Agent header (else HTTP 403).

## Open questions
