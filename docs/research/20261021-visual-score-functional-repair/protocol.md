# Protocol (locked before results): visual score + functional-relation-aware repair

Date: 2026-10-21. Context: `../20261013-unknown-size-comparison/method-improvement-research-20261021.md` §8–10.
Delivered pipeline: upstream fields → prior sizes → 27B LoRA placer → S1 → escalation → RoomGenBench (rendered).
The user judges by the rendered room. By-eye look at 20 hash-picked renders: S2 (protect + A4) better 4 / same 7 /
worse 9 vs S1, because it moves companion furniture away from its anchor.

All runs local CPU/MPS + the user's API (gpt-6.1-sol, reasoning medium, `llm_config.json`, git-excluded).
No training, no repo code changes (scripts under the harddisk `probes-20261021/scripts/`), no EmbodiedGen edits.
API cap for this protocol: 200 calls in total; usage logged.

## Q1 — which automatic visual score to adopt

Candidates: (a) pairwise gpt-6.1-sol judge, A/B order hash-randomised, image = perspective + top of each layout;
(b) absolute 0–10 judge with the OptiScene-style rubric already in `visual_pipeline.py` (functionality / layout /
aesthetics); (c) CLIP score (ViT-L/14) between the perspective render and the text "a 3D rendering of a {room_type}
with {furniture list}"; (d) symbolic companion-relation score (below).

- **H1a (validity, sanity pairs GT vs 27B raw):** pairwise judge prefers GT in ≥ 70% of non-tie rooms.
- **H1b (CLIP insensitivity):** CLIP prefers GT over raw in ≤ 60% of rooms and its per-room GT−raw difference does
  not correlate (|Spearman| < 0.2) with the raw layout's checker violations.
- Decision rule: adopt as the visual endpoint the score that passes the validity sanity check with the highest
  agreement with the by-eye labels of the 20 S1-vs-S2 rooms (labels fixed in the report §10 before any judge ran).
  Geometry (collision / OOB) stays with the frozen checker in every case.

## Q2 — functional-relation-aware S2 (S2-F)

S2-F = S2 (wall protection + A4 target, accept-if-not-worse) plus companion protection: companion pairs are read
from the layout being repaired (the model's own S1 output; never GT), by category rules (nightstand ↔ bed head
side; TV stand / TV ↔ bed or sofa it faces; desk/dining chair ↔ desk/table; coffee table / ottoman / armchair /
side table ↔ sofa; toilet / sink ↔ wall). Companions move with their anchor, and a moved companion only takes
candidates that keep the relation. Same frozen sizes, yaw, parents, ids, every object placed.

- **H2 (S2 as is):** pairwise judge win rate of S2 over S1 on the 20 rendered rooms ≤ 45%.
- **H3 (S2-F):** on dev-select 600 and on the 1,009 visual-cohort rooms, S2-F keeps ≥ 80% of S2's checker gain over
  S1 and ≥ 60% of its strict gain, with companion retention ≥ S1's; on 20 freshly hash-picked rendered rooms its
  pairwise win rate over S1 is ≥ 50% (non-inferior), and over S2 > 50%.
- Test split caveat: the visual cohort is test (only rooms with assets); it is used for looking, not for selecting
  parameters. Parameters are fixed in the S2-F code before its cohort run.

Outputs: harddisk `probes-20261021/{clip-metric,judge,s2f}/`; summaries here in `findings.md`.

## Addendum A (locked 2026-10-21, after judge v1 results, before any v2 call)

Judge v1 failed its sanity check (GT over raw 13/20 = 65% < 70%; first position chosen 75% on S1 vs S2).
**H1c (judge v2):** same model, images and rooms; the prompt asks for pros and cons of each layout before the verdict
(R³L / Procedural Scene Programs style), and every pair is judged in both A/B orders; the room verdict is the winner
when both orders agree, else tie. Sanity set: GT vs raw on the 20 R20 rooms (40 calls; total API cap stays 200).
Pass iff order-consistency ≥ 70% of rooms and GT wins ≥ 70% of the consistent rooms. If it passes, v2 is re-run on
S2F vs S1 (sets A+B) with the remaining budget; if it fails, no VLM judge is adopted as an endpoint and human pairwise
labels become the visual endpoint.

## Addendum B (locked 2026-10-21, after the v2 sanity result, before the v2 S2F runs)

v2 sanity: order-consistent 18/20, GT wins 13/18 = 0.72 → passes addendum A (marginally; n = 18).
Budget raised from 200 to 300 successful calls (192 used) to run v2 on S2F vs S1 over sets A + B (40 rooms × 2 orders
= 80 calls). Reading: S2F non-inferior to S1 iff S2F wins ≥ 50% of decisive (order-consistent, non-tie) rooms.
