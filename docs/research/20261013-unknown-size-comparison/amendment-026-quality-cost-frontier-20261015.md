# Amendment 026 — quality versus big-model use: local-first escalation policies vs calling the big model for every room

Locked 2026-10-15 (UTC), before any amendment-026 number exists. Analysis only: no new model calls, API calls or GPU jobs.

## Disclosure

**Motivation.** The user asked why the delivered system is not simply gpt-6.1-sol for every room, i.e. what our trained model is for. The agreed framing is "local first, big model as fallback", compared with "big model for every room" on both quality and cost.

**What was viewed before locking.** The 022 cohort results, including the matched contrasts against L+S1 on the 300 pilot rooms, and the room-type mix of those 300 rooms:
- 35 raw room-type labels, which map to 12 families plus 23 unlabelled rooms;
- 143 bedrooms;
- 72 % IL3D_3dfront + holodeck;
- a median of 6 objects.

The pilot rooms were also used to design 022. Everything here is EXPLORATORY.

## 1. Scopes and the big model

| | Scope A | Scope B |
|---|---|---|
| Rooms | the 300 pilot rooms | the 1,454 cohort rooms |
| Big model | gpt-6.1-sol, prior sizes enforced | O2 (Qwen3.8-27B, thinking), prior sizes enforced |
| Big-model outputs used | amendment-016 rows `L-P-pilot` (`s1-016-size-fixed/agent-L-P-pilot-*`) | amendment-016 rows `O2-P` (`s1-016-size-fixed/agent-O2-P-*`) |
| Variant | the harness variant | both raw (one attempt) and harness |
| Cost unit | tokens and calls (API) | output tokens and GPU wall time (local GPU) |

**Local arm (both scopes).** The 27B placer + S1: the amendment-022 cohort rows, i.e. the 27B test rows + S1.

## 2. Policies (per room; every policy ends with S1 on the chosen layout)

- **LOCAL** — 27B + S1 only; no big-model call.
- **ESC-CHECK** — escalate if 27B + S1 fails the frozen checker. This is the delivered 022 rule.
- **ESC-STRICT** — escalate if 27B + S1 fails the checker or strict admission.
- **ESC-ANY** — escalate if 27B + S1 fails the checker, strict admission or required = {} admission.
- **BIG** — the big model for every room, + S1.

**When a room is escalated,** the big model's output is taken only if it ranks higher than the local layout by this lexicographic rule, otherwise the local layout is kept:
1. checker pass;
2. strict admission pass;
3. required = {} pass;
4. fewer checker violations.

**Simulation, disclosed.** The escalated answer is the big model's existing amendment-016 answer for that room. It is not a fresh call with 022's repair round, and its cost is that answer's recorded usage. For scope A the 022 cohort system's actual outputs on the 300 rooms are also reported as a check of the ESC-CHECK simulation.

## 3. Reported per scope and policy

- **Escalation:** the share of rooms escalated and the share where the big model's output is used.
- **Cost:**
  - scope A: calls and input, output and reasoning tokens per room;
  - scope B: output tokens and the O2 wall time attributed per room;
  - for comparison, the 27B's own GPU time per room (021 test: 8,647 rooms in 59 min on 1 GPU).
- **Quality:** checker success, strict admission, required = {} admission, OOR, size log MAE and B.
- **Paired contrasts:** each policy − BIG, house-cluster bootstrap, 2,000 draws, seed 42.
- **Strata:**
  - object-count bins: ≤ 20, 21–50, > 50;
  - room families (amendment 015);
  - real vs synthetic sources.
- **Frontier plot:** quality against the share of rooms sent to the big model.

## 4. Reading

No policy is adopted here. The delivered system stays as locked in 022.

The frontier answers one question: how much big-model use is needed to match "big model for every room" on each metric, and on which room families local-first is enough.

## Addendum A (locked 2026-10-15 ~11:20 UTC, before any scope-C output exists): scope C, family-stratified rooms

**Why.** The user pointed out that the 300 pilot rooms of scope A are dominated by bedrooms and living rooms (143 + 50) from 2 sources, and asked for a sample with more room types.

**Rooms.** 300 test rooms, 20 from each of the 15 amendment-015 room families:
- bedroom, living, dining, kitchen, bathroom, study_office, meeting, balcony, laundry_utility, garage, circulation, storage_closet, public_commercial, other, none;
- every family has ≥ 41 test rooms;
- within a family, the 20 lowest `sha256('a026c:' + uid)`;
- written to `RUN/a026-diverse-v1.json` before any call.

**Big model.** gpt-6.1-sol, size-fixed (P input, B2 enforcement), with exactly the amendment-016 pilot settings:
- `llm_agent` placement arm, reasoning effort medium;
- harness of up to 3 attempts, then `evaluate_llm` and S1.

**Local side.** The existing 27B test rows + S1 (`s1-022/placement-arm`), restricted to these uids. No new GPU work.

**Policies and reporting.** The same policies and reporting as scopes A and B. In addition, LOCAL vs BIG for every family.

**Cap.** ≤ 900 paid calls (300 rooms × ≤ 3 attempts), plus a 2-room smoke. No retries beyond the harness, and no replacement.
