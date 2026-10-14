# Amendment 020 — two-stage pipelines: an LLM estimates sizes only, a placer places them (by room family)

Locked 2026-10-14 (UTC), before any amendment-020 call, generation or score exists. Requested by the user: "通过调用 api 算一下这个家具的 bbox，然后再利用 OptiScene，我们又没有做实验，比如房型不同的情况下". EXPLORATORY; it changes nothing in amendments 005–019.

## 1. Rooms
- **E2:** the six paper strata of eval-data-v1 (amendment 014 E2, 1,009 test rooms).
- **E3:** the 868 rooms from the 9 non-paper families (amendment 015).
- Every table is reported for E2, for E3 and per family.

## 2. Stage 1 — sizers (each outputs [w, d, h] in metres for every requested object; nothing else)
- **Sizer input:**
  - J's request fields minus every placement field: room type, boundary polygon, height, doors/windows/fixed boxes, and the object list (id, category).
  - The size definition is J's own wording, so the axis convention is identical to J's output.
  - The instruction asks for each object's real-world dimensions. There is no instruction to fit the room and no instruction to shrink.
- **Sizers:**
  - **L-size:** gpt-6.1-sol, API, the amendment-006 settings. It counts toward amendment 002's 4,800 LLM-agent call cap; the job refuses to start if the calls used plus 1,877 × 3 would exceed that cap.
  - **O2-size:** Qwen3.8-27B with the amendment-007 engine settings.
  - **Harness:** same as the agent runs: one answer plus up to 2 repair rounds on parse errors. A missing object counts as a failure for that room.
- **Reference size sources** (no new runs):
  - prior: the train prior, i.e. P's input;
  - GT: oracle;
  - L-joint: the sizes gpt-6.1-sol returned in its amendment-016 J-input run, cohort rooms only.
- **Size metrics:** size log MAE vs GT, footprint bias B, IoU of the boxes centred at the origin, completeness. All are reported per family.

## 3. Stage 2 — placers given each size source
- **OS-DPO2:** the amendment-014 reproduced OptiScene, final DPO2 stage, with its prompt and decoding. New size modes `llm-L` and `llm-O2` join the existing `prior` and `gt`.
- **P:** rpo-dpo1, exact policy, in P's request format with the substituted sizes. It runs on E2 ∪ E3 for the `llm-L` and `llm-O2` size sources. P with prior sizes reuses its existing test outputs where the rooms coincide; everywhere else it runs once.
- **Scoring:** amendment-017 primary metrics (frozen checker success, OOR, size metrics), plus the admission variants (strict and required = {}), raw and after S1. Each placer's requests carry the sizes of its size source; the checker scores each layout with those sizes.

## 4. Report
- **Table A:** sizer × family → size quality.
- **Table B:** (sizer → placer) × family → checker success, admission and OOR.
- **Paired contrasts** (house-cluster CIs, EXPLORATORY): `llm-L` vs `prior` within each placer; OS-DPO2 vs P within each size source; E2 vs E3 descriptive gaps.

## 5. Budget
- **API:** ≤ 1,877 rooms × 3 calls.
- **GPU:**

  | Item | GPU-h |
  |---|---|
  | O2 sizer | ~6 |
  | OptiScene generation, 2 new modes | ~4 |
  | P inference, 2 modes on ≤ 1,877 rooms | ~10 |
  | Total cap | 30 |

## Addendum A (locked before any amendment-020 output exists)

- **The locked cap rule could never pass.** §2's rule "used + 1,877 × 3 ≤ 4,800" fails before a single call is made, because 1,877 × 3 = 5,631 > 4,800.
- **Corrected budget.**
  - The amendment-002 LLM-agent call cap is raised from 4,800 to **6,500** calls.
  - L-size gets one answer plus at most **one** repair round per room, instead of two, so it needs at most 1,877 × 2 = 3,754 calls. The worst case so far is ≤ 2,675 calls used, which gives ≤ 6,429 in total.
  - The expected real spend is about 1,900–2,100 calls, because size-only answers rarely need a repair.
- **O2-size** (free, GPU) keeps the agents' two repair rounds.
- **Failures.** A room still invalid after its repair is a failure. It is carried as a failure and never dropped.
