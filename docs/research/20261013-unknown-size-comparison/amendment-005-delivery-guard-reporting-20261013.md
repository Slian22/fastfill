# Amendment 005 — size-bias guard for delivery, reference rows, reporting rules

Locked 2026-10-13 ~19:00 UTC, before any J/P/agent test output exists (inference v2 not yet submitted). Source: Agency audit in workflow wf_39f38da6-e7d (Data Engineer, LLM Post-Training Engineer, Model QA Specialist, Statistician, Reality Checker). Audit verdicts: training correct, no retraining needed; data mechanics correct (no target leakage, no split contamination); metrics computed correctly. The changes below concern interpretation and the delivery decision.

## 1. Delivery rule (replaces amendment 002 §5)

Co-primary (a), all-request raw geometry success, rewards undersized furniture: shrinking GT footprints by 5 % per axis raises GT success from 72.3 % to ~78.6 %, enlarging by 5 % drops it to ~58 %. J chooses its own sizes; P cannot. Therefore:

- B_arm = mean over common-complete rooms of the per-room mean over objects of log(pred footprint / GT footprint); also the analogous volume bias. Reported with the house-cluster bootstrap CI (2,000 draws, seed 42), overall, per source and per paper stratum, with the P prior's own bias as a reference.
- (a′) size-debiased success: rescale each arm's predicted x/y sizes about object centres by exp(−B_arm/2), keep positions/yaw, re-score with the frozen geometry code.
- **Deliver J iff** (i) J's size log MAE on common-complete pairs is lower than P's with the paired 95 % CI excluding zero, **and** (ii) (a′)J − (a′)P has lower 95 % CI bound ≥ −2.0 percentage points. Otherwise deliver P.
- Raw (a), B, (a′) and the GT reference are always shown beside the decision. Delivery is an artifact choice for the user's pipeline, not a claim of scientific superiority. Single seed.

## 2. Reference rows and strata (all model-free, pre-computed, frozen)

- GT layout rows (raw and boundary-snapped): geometry success 0.7230 overall (IL3D_3dfront 0.494; holodeck 1.000), OOR 0.0787; per source and per stratum.
- P prior-only sizes on all 8,647 rooms: size IoU 0.5582, log MAE 0.2824; per source/stratum.
- GT-pass sensitivity stratum: the 6,252 test UIDs whose GT passes the frozen geometry gate (list frozen with sha256 in `evidence/`).
- Known-defect sensitivity: the 3 declared rooms, plus (separately labelled) every test room whose prepared input or GT changes in the final source-correction candidate.

## 3. Reporting rules

- `reached_token_cap` without `stopped_on_eos` counts as a failure in all-request denominators even if the JSON parses.
- The 10 cm "repair" is reported as "snap to boundary (≤10 cm); can create collisions" — it is non-monotone on GT (success 0.7133, OOR 0.0792 after snapping). Raw stays primary.
- Rotated-footprint OOR (using the rotated polygons the checker already builds) is reported beside the axis-aligned paper OOR; holodeck GT has zero true rotated overlap.
- P is labelled "category × room-type median with constant fallback (0.40, 0.50, 0.65 m)"; fallback rates per source and results split by prior level (pair / category / global fallback).
- Size metrics split into low-reuse sources (real scans, SceneSmith, SpatialGen) and high-reuse synthetic sources (asset sizes repeat across splits).
- Multiplicity: Holm across the two co-primaries for any confirmatory J−P claim; bootstrap two-sided p = 2·min(P(draw ≤ 0), P(draw ≥ 0)).

## 4. LLM / open-model agents

- Primary agent comparator = harness variant; raw = single-shot sensitivity. Same endpoints (a), (b), plus complete rate and raw OOR (descriptive).
- Cohort = every source contributes up to 30 rooms not already in the six strata (1,454 rooms total; the implemented reading).
- Inferential agent contrasts only on (i) the pooled cohort and (ii) the pooled six strata (n = 1,009); per-stratum / per-source cells descriptive. Agent contrasts are estimation-only (CIs, no "significant" wording). J−P on the same cohort always shown beside J−agent and P−agent.
- API failures stay in denominators as failures.

## 5. Visual metrics

- GPT judge endpoint: per-room mean of functionality, layout and aesthetics; J−P paired mean difference with bootstrap over rooms stratified by stratum (2,000 draws, seed 42), cohort "both complete" with count. Sub-scores descriptive. Judged systems: J, P, API-agent harness, and the largest successfully run open model (O5 DeepSeek-V4-Flash, else O4 gpt-oss-120b).
- FID: computed on the common all-system successful intersection per stratum with N reported; KID with SE as secondary; strata with N < 50 flagged.

## 6. Inference gate

Before the test shards of an arm run, a dev probe job regenerates the arm's 2 recorded dev probe prompts with the merged-LoRA batched path and records exact match / schema validity against the unmerged training-time probe; schema-invalid output where the unmerged output was valid fails the chain closed.
