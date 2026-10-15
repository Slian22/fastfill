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

## Addendum B (locked 2026-10-15 ~10:00 UTC, before any amendment-022 test-scope output exists; no O2 answer of job 7426877 has been read)

1. **Why.** The test-scope system job 7426877 might not finish within its 12-hour limit.
   - It escalates about 2,230 rooms to O2 on 2 GPUs, in chunks of 256. Chunk 1 took 2.5 h and chunk 2's first round took 44 min.
   - The projected total of 10–12+ h is at the limit.
   - It writes its receipts only after the last chunk, so a timeout would lose every answer.
   - The limit cannot be raised by the user, and the SSH certificate expires at 15:15 UTC.
   - Only its timing log (`invocations.jsonl`) was read.
2. **Change.** 7426877 and its summarize job 7426878 are cancelled. The test scope reruns with the O2 escalation sharded (`submit_022.sh CODE_ROOT test-sharded 4`, commit 34f240e09):
   - **Shards:** 4 shards by `int(sha256('a022-o2-shard:' + uid), 16) mod 4`, each a 2-GPU job.
   - **Unchanged:** the engine, sampling, chat template, harness rounds, chunk size 256 and within-shard chunk order rule of `o2_answers`.
   - **Writes:** every chunk is written as it finishes, and a killed shard resumes from its first unfinished chunk.
   - **Assembly:** a CPU finalize step assembles the receipts and runs the rest of the system unchanged; then the same summarize.
   - **Output:** `RUN/system-022/final/test-sharded`.
   - **Identity:** it records the sharding (N = 4, the rule, each chunk's uids).
3. **Disclosure.** This is not bitwise the unsharded run.
   - vLLM decoding is not batch-invariant, so chunk composition can change some O2 answers.
   - S1 is re-run in the plan step. It is deterministic except for rooms that hit its 60 s safety timeout.
   - The report labels the test-scope system as "O2 escalation, sharded ×4".
4. **Package.** It reads `test-sharded/eval` for the test scope only when `test/eval` does not exist (`package._eval_root`). The package job is resubmitted after the sharded summarize; job 7426881 is cancelled.
5. **GPU cost.** The ceiling is 4 × 2 GPUs × 12 h = 96 GPU-h; the expected use is about 25 GPU-h.
