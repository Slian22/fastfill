# Amendment 022 — the delivered FastFill system: sizes → placer → S1 → escalation (one deliverable)

Locked 2026-10-14 (UTC), before any amendment-022 output exists. Requested by the user: one model, the best available, that takes room information plus the required furniture and outputs bbox + layout; a small amount of API use for correction is allowed.

**Disclosure.** Designed after viewing:
- the amendment-016 §2 size-fixed pilot;
- an offline simulation on those 300 pilot rooms using existing outputs only. P + S1 passes the checker at 0.827. Sending only the failing rooms to gpt-6.1-sol size-fixed (+ S1) gives 0.993, with 52 / 300 rooms needing the API. Sending them to O2 instead gives 0.957.

Those pilot rooms are test-cohort rooms. Nothing below is selected on them.

## 1. The system (input → output)

**Input.** Room boundary, height, doors, windows and fixed boxes, room type, and the requested furniture (category and count).

**Output.** Every object's bbox size, position, yaw and support parent, in our IR / RoomGenBench export.

1. **Sizes.**
   - Default: the train-prior sizes (P's input).
   - Replaced by an amendment-020 size-only sizer only if that sizer beats the prior on a dev measurement, decided by a later addendum before any test use. The amendment-020 test tables are not used for this choice.
2. **Placer.** The model delivered by amendment 021 §4: R = rpo-dpo1, or 013's op-dpo2-e1, 018's adopted model, or 021's arm, whichever is selected and confirmed there. It is served on 021's engine.
3. **S1.** The deterministic repair of amendment 012, at the same settings.
4. **Escalation.** It applies only to rooms that, after S1, are incomplete or fail the frozen checker.
   - **Step 1:** gpt-6.1-sol, size-fixed (P's input, B2 enforcement), one answer + at most one repair round, then S1. Its result is taken if it passes the checker.
   - **Step 2 (free fallback):** O2 size-fixed on the same input plus S1, if Step 1 does not pass.
   - **Otherwise:** the system returns the best candidate it has, ordered as passing checker > complete > fewer checker violations.
   - The system never returns nothing for a complete request, and never drops a room.

## 2. Evaluation (test, after 021 §4 has confirmed the placer)

**Scopes and systems**
- **Cohort (1,454 rooms):** the full system with the gpt-6.1-sol escalation. The API cap is the expected escalation share plus a margin: ≤ 600 rooms × 2 calls. The amendment-002/020 LLM-agent cap is raised by that 1,200 calls, to 7,700.
- **All 8,647 test rooms:** the system with the free O2 escalation only (GPU).

**Reported**
- every stage's contribution:
  - placer raw;
  - + S1;
  - + escalation, with the share of rooms escalated and the API calls and cost;
- checker success, both admission variants, OOR, size metrics, and mesh collisions (amendment 019) where rendered;
- all of the above per room family and per object-count bin, next to the agents' single-shot and harness rows.

**Fairness label.** Escalation is a system with API access. It is reported as such and never compared as "the model" against single-shot agents.

## 3. Delivery

- **Package:** the placer (training base + adapter, or the 27B adapter), the prior-size table, the S1 code, the escalation code with its call caps, a CLI and RoomGenBench export, plus a model card with every measured number and its provenance.
- **Interim package:** before 021 §4 finishes, the system with R = rpo-dpo1 is packaged and labelled "interim". It changes no decision.

## Addendum A (locked before any amendment-022 output exists; from the tooling review)

1. **Fallback stage**, implied by "never returns nothing".
   - **When:** no candidate is complete after every step.
   - **What:** every requested object is placed on the floor at one interior point of the boundary (shapely `representative_point`), yaw 0, then S1 at the same settings.
   - **Ranking:** it is ranked by the same best-candidate rule and reported as stage `fallback`.
2. **Engines.**
   - The placer runs in a separate process, so its GPU memory is freed before O2 loads (tensor parallel 2).
   - `--escalate o2|l+o2` needs two visible GPUs and refuses before any placer or API work.
3. **Caps.**
   - The cohort run passes `--api-max-rooms 600`; the system stops before any API call if more rooms would escalate.
   - The amendment-005 cap rule applies to the placer's stages, not to the agents' or O2's answers, as in amendments 012 and 016.
4. **Evaluation engine.** It is the 021 §3 serving engine of the selected placer and is printed on the model card.
