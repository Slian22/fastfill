# Amendment 025 — SceneSmith-style critic → constrained repair on top of the delivered placer

Locked 2026-10-15 (UTC), before any amendment-025 output exists.

## Disclosure

**The idea.** The user proposed it on 2026-10-14 10:06 UTC: render the outputs, let a SceneSmith-style critic (reviewer) look at them, then repair the placement. Categories stay unchanged.

**Earlier design.** It was drafted in `solution-research-20261014/critic-design.md`. Only its deterministic half was run: S1 (amendment 012) and the escalation of amendment 022.

**What was viewed before locking**
- The delivered system's cohort results (022) and the 27B test results (021).
- The 27B visual metrics (024 arm A).
- The run-002 and 023 judge results, where J and P score 6.2–6.3 against L's 8.0.

**Status.** Exploratory and dev-selected. Nothing already reported changes.

## 1. What the critic stage does

### Input
The 27B placer's output after S1, for a complete room, exactly as the delivered system produces it.

### Critic
- **Model:** gpt-6.1-sol on the server API, reasoning effort medium, as for system L. The user chose this model.
- **It sees:**
  - a top view and one oblique view of the room as boxes, with every object's id label, a front-facing arrow, the boundary, doors and windows, and objects coloured by checker flag;
  - the object list `{id, category, size, pos, yaw, parent}`;
  - the frozen checker summary;
  - the strict-admission rule failures (downstream.accept reasons).
- Tool results take precedence over what the critic sees in the images.

### Output
- JSON: scores for functionality and layout (0–10), plus at most 5 edits per round.
- **Allowed edits:**
  - `move` (x, y);
  - `rotate` (yaw);
  - `against_wall` (id);
  - `face` (id, target id).
- **Forbidden:** adding, removing, resizing, relabelling or reparenting. An edit on an unknown id is rejected.
- Supported children move rigidly with their parent.

### Re-check after every edit set
1. Apply the edits.
2. Run S1 with frozen sizes, at the amendment-012 settings.
3. Accept the result only if both hold:
   - the frozen-checker counts are not worse, compared lexicographically over (valid, fixed_blocking, severe, main, oob, support_fail, ceiling);
   - strict admission does not go from pass to fail.
4. Otherwise keep the previous layout.

### Rounds
- At most 2 rounds per room.
- Stop early if no edit was accepted, or if both critic scores are ≥ 8.
- Categories, object set, ids, sizes and parents never change. Invariant counts are reported and must be 0.

## 2. Dev selection

### Rooms
The 250 dev-select rooms with the lowest `sha256('a025:' + uid)` among those where the 27B's vLLM dev-select output is complete.

### Arms
- **B:** 27B + S1.
- **C:** 27B + S1 + critic repair.

### Metrics (paired, house-cluster bootstrap, 2,000 draws, seed 42)
- checker success;
- strict admission;
- required = {} admission;
- OOR;
- displacement (median and p95 of moved objects);
- share of rooms edited;
- invariant counts;
- **GPT judge** (amendment-023 judge: gpt-6-astra, reasoning high), blind to arm:
  - one judgement per room and arm, on the same kind of image for both arms;
  - Blender renders through the run-002 pipeline if the dev rooms can be attached to assets, otherwise the same box views as the critic sees (disclosed);
  - endpoint: the mean of functionality and layout.

### Adoption rule
The critic stage is adopted only if all three hold on dev:
1. Judge (functionality + layout) / 2: C − B has a 95 % CI lower bound > 0.
2. Strict admission: C − B ≥ −0.005.
3. Checker: C − B ≥ −0.005.

## 3. Test, only if adopted

- **Arms:** the delivered system's test output (024 arm S), and the same output plus the critic stage.
- **Rooms:** the 174 run-002 judge rooms.
- **Reported:** checker, both admission variants, displacement and the 023 judge, next to the 023 rows of J, P, L and O6.
- **Not reported:** FID/KID and mesh collisions, because the room count is too small. They are reported as not measured for this arm.

## 4. Budget

**API calls**, ≤ 1,550 paid in total:

| Item | Calls |
|---|---|
| Critic on dev, 250 rooms × ≤ 2 rounds | ≤ 500 |
| Judge on dev, 250 rooms × 2 arms | ≤ 500 |
| Critic on test, 174 rooms × ≤ 2 rounds | ≤ 348 |
| Judge on test | ≤ 174 |
| Smokes, 2 cases each | — |

No retries and no replacement; a failed call leaves the room unedited, and this is reported.

**Compute:** CPU jobs only, and no GPU.

**Keys:** the API key is never printed or copied.

## Addendum A (locked 2026-10-15 ~11:20 UTC, before any amendment-025 output exists): family-balanced dev rooms

**Why.** The user asked for evaluation rooms with more room types. A plain hash sample of dev-select would mirror its mix: 210 bedrooms, 132 living rooms, 2 garages.

**Change to §2 "Rooms".**
- **Pool:** the dev-select rooms whose 27B vLLM output is complete.
- **Selection:** for each amendment-015 room family, at most 31 rooms, taking the lowest `sha256('a025:' + uid)` within the family. Families with fewer rooms contribute all of them.
- **Expected count:** about 247 rooms across all 15 families. The exact count and the per-family counts are written with the room list before any critic call.

**Budget.** The cap stays ≤ 500 critic + ≤ 500 judge calls on dev. Everything else in amendment 025 is unchanged.

## Addendum B (locked 2026-10-15 ~13:35 UTC, before any output of judge job 7442970 has been read)

**Trigger.** An independent review (`evidence/agency-026-025-review-20261015.md`) found that the dev judge prompt gives only category counts, while the box views label objects only by id. The judge therefore cannot tell which box is which piece of furniture. Swapping two categories leaves its input unchanged. This is verified: the prompt lists `{"category": "bed", "count": 1}` etc., and the images show `obj_0001`…

**Changes**

1. **Formal judge.** A new bundle in which the prompt of both arms also lists every movable object as `{id, category}`. The list is identical for B and C, because ids and categories never change.
   - Everything else is as before: same rooms, neutral views, judge, rubric and settings.
   - Job 7442970, which had no mapping, is reported only as a "category-blind diagnostic". It is never used for adoption.
2. **Coverage rule**, added to the §2 adoption rule.
   - At least 90 % of the dev rooms (≥ 222 of 246) must have both arms judged successfully. Otherwise the critic stage is not adopted.
   - Missing judgements are reported per arm and per room family, and are never imputed.
3. **Budget.** The amendment cap is raised from 1,550 to 2,050 paid calls, covering this extra 492-call judge run.
   - Spent so far: 450 critic calls and 492 calls for job 7442970.
   - Still allowed afterwards: test critic ≤ 348 and test judge ≤ 174.
