# Amendment 024 — visual and physical metrics for the delivered placer and system

Locked 2026-10-15 (UTC), before any amendment-024 render or score exists.

**Disclosure.** Designed after viewing:
- the amendment-021 test result (27B vs R);
- the amendment-022 cohort result;
- the visual run 002 metrics (FID/KID, mesh collisions, GPT judge) of J, P and the agents;
- an independent audit of the draft report. It found that every visual number in the draft belonged to the old 8B P, and that the delivered placer and system had never been rendered (022 §2: "未测量（系统输出未渲染）").

Descriptive only. Nothing is selected or changed by this amendment.

## 1. Arms and rooms

**Arms**
- **A — the 27B placer, single-shot raw.** Its amendment-021 test rows (`RUN/a021/test-geometry-arm`), comparable to the existing single-shot rows of J, P and the agents.
- **S — the delivered system.** The final output of the amendment-022 test run (`RUN/system-022/final/test`): 27B → S1 → O2 escalation → best candidate.
- Arm S is rendered only after that run has finished. If it does not finish, S is reported as not measured.

**Rooms.** The 1,009 paper-strata test rooms of visual run 002.

**Rendering, unchanged from run 002**
- the same reference set (`data/visual-reference-v1`), assets, cameras, views and resolution;
- the same Blender binary and the same `visual_pipeline` rendering code;
- render failures are reported and never replaced.

## 2. Metrics (as in amendments 011, 019 and 023)

1. **FID / KID** with the amendment-011 population rule: per arm, reference ∩ J ∩ P ∩ arm. Same vendored features and weights as `results/fid-011-visual-002`. Reported next to the run-002 rows on the same population rule.
2. **OOR** on the same rooms, as T4 of report-v2.
3. **Mesh collisions** (amendment 019): primary collision-free rate at τ = 500 cm³, same voxel and sampling settings.
4. **GPT judge** with the amendment-023 judge, prompt and rubric:
   - judge: gpt-6-astra, reasoning effort high, max_output_tokens 32,768, timeout 600 s;
   - cases: the 174 run-002 judge rooms × arms A and S = 348 cases;
   - reported next to the 023 judgements of J, P, L and O6 on the same rooms;
   - **cap:** 400 calls plus a 2-case smoke; no retries, no replacement.

## 3. Reading

Arm S is a system with escalation (free O2 on test). It is labelled as such and never presented as a single-shot model next to the agents.
