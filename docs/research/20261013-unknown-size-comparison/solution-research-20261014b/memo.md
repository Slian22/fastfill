# Decision memo: closing the gap between P (rpo-dpo1) and O2 when sizes are fixed

Date: 2026-10-14. Labels: **[V]** verified in code, data or a primary source; **[I]** inference or estimate; **[NEW]** computed for this memo.

## 0. Summary

- **[V]** P fails even though every training target passes the checker. Three facts show it:
  - Failure per object is roughly constant: survival is about 0.887 per object.
  - Collisions are spread evenly along the output.
  - In dense rooms, same-category instances are copied exactly: 96.1 % of co-located pairs.
- **[V]** O2 almost never collides once it finishes an answer: 238 of 240 complete raw answers pass. It spends about 10.9k thinking tokens per answer to get there.
- **[NEW]** About two thirds of the pilot gap (82 of 122 rooms) is in rooms where the ground-truth (GT) arrangement does not fit once the prior sizes are swapped in. There, P passes 0.40, O2-harness 0.90 and L 0.95.
- **[NEW]** About 80 % of the gap is in rooms with 20 or fewer objects (approximate, see §1). P fails even when a feasible near-GT target exists.
- **[I]** Training to pass the checker on the model's own samples (iterated self-training, then GRPO) is the cheapest fix. It probably saturates around P's best-of-8 rate, about 0.60 on dev.
  - Getting near O2 (0.92) needs something that raises that ceiling: explicit geometric reasoning (an extra written-out step before the answer, or O2's traces) or a stronger base model.
  - Whatever remains can be handled at system level by a verifier or solver.
- Amendment 018 stays as locked. New work goes into new amendments, numbered 020+ (019 is taken). Run ~7 GPU-h of diagnostics first; their results decide which training arms get GPUs.

## 1. Root causes, ranked

| # | Root cause | Evidence | Status |
|---|---|---|---|
| 1 | **P does not take into account what it has already placed.** Each object goes near its usual position; greedy decoding plus identical inputs for same-category instances gives exact duplicates. | **[V]** Fitted on 8,647 test rooms: log P(success) = −0.117 − 0.1196·n, i.e. 0.887 per object.<br>**[V]** Collisions are spread evenly over first / middle / last third (11,147 / 11,243 / 11,822). Collision pairs are as far apart in the list as random pairs (12.24 vs 12.4).<br>**[V]** 96.1 % of co-located same-category colliding pairs are exact copies of position and yaw. Same-category instances get identical input tokens (`contract.py:166-171`). P decodes greedily with thinking off (`infer.py:174,303`).<br>**[V]** O2 uses 10.9k reasoning tokens per answer; L uses only about 311 and still passes 0.947. | Symptom [V]. Mechanism ("marginal-mode placement") [I]. |
| 2 | **No training signal ever came from P's own geometric failures.** | **[V]** SFT is teacher-forced NLL.<br>**[V]** DPO1 negatives are not required to fail, and pairs come from sparse sources only (`dpo_stages.py:69,532-543`).<br>**[V]** DPO2 used synthetic perturbations. Test gain of rpo-dpo1 over SFT: +0.008.<br>**[V]** Best-of-8 on dev raises P from 0.400 to 0.600 (`findings.md:47`). Valid layouts are in P's sampling distribution but are not its mode. | [V] |
| 3 | **P does not re-plan when the GT-like arrangement does not fit the given sizes.** | **[NEW]** On the pilot, rooms where GT positions + prior sizes fail: n=166; P 67, O2-harness 149, L 157 pass (82 of the 122-room gap). Rooms where they pass: n=134; P 87, O2-harness 127 (40 rooms).<br>**[NEW]** Rooms with exactly 2 objects where P collides: 100 of 710. 71 of those 100 are rooms where GT + prior fails. All 100 have both objects on the floor and 0 are co-located; pairs are bed–wardrobe, shower–toilet and similar, and 63 come from SpatialLM. So small-room failures are not the duplicate bug.<br>**[NEW]** It is not that the rooms are infeasible. On 750 proxy rooms where the training-target repair succeeds, P still passes only 0.685 / 0.368 / 0.142 / 0.052 / 0.0 by object-count bin. Where repair fails, P passes 0. | Correlation [V]. Mechanism [I]. |
| 4 | **Capacity or representation:** 8B, LoRA r32, 1 epoch, absolute coordinates. | **[V]** Recipe in `run_receipt.json`. No dev loss was computed.<br>Literature: SPG-Layout reports LoRA 78 % vs full fine-tuning 99 % format success; LayoutDSL reports DSL 0.878 vs raw parameters 0.749 after RL. | Hypothesis. Diagnostics D1/D2 below test it. |
| 5 | **Inference-engine mismatch** (affects the plan, not the gap itself). | **[V]** On the 600 dev rooms, vLLM BF16 LoRA greedy gives 0.432 vs the HF exact policy's 0.470, with only 9 % of outputs identical. A BF16 merge gives 0.433 (research-log 13:55 and 14:55). | [V] |

**Where the gap sits [NEW, I, approximate].** I weighted (O2-harness pilot rate − P full-test rate) by each bin's share of test rooms (3,713 / 1,877 / 1,624 / 1,086 / 347). This mixes room sets, so treat it as approximate.

| Objects | Gap contribution |
|---|---|
| 1–5 | 0.129 |
| 6–10 | 0.127 |
| 11–20 | 0.130 |
| 21–50 | 0.070 |
| 51+ | 0.020 |

About 80 % of the gap is in rooms with ≤20 objects. Rooms with 51+ objects are a separate problem: O2 also fails there, through truncation (43 of its 44 incomplete answers in 21+ rooms hit the token cap).

**Ruled out [V]:**
- Bad labels: P targets are GT repaired under the prior sizes, admitted only if they pass (`main_data.py:179-196`).
- Too few dense training rooms: about 10.1 % of train rooms have 21–50 objects and 3.3 % have 51+, against 12.6 % and 4.0 % in test.
- Truncation of P: 300 of 300 pilot rooms complete.
- Scoring asymmetry: minor; support failures are in 4 % of P's failing pilot rooms.
- **[NEW]** Sizes being constant per category: P's input sizes do vary. Only 52.9 % of objects carry their category's most common size; "chair" alone has 72 distinct sizes.

**Fixing training targets for the given sizes** is not supported as a main lever. Targets are already size-consistent. The only target-side defect is selection bias: rooms whose repair fails are excluded, which is 4–10 % of rooms in bins ≤50 objects but 34 % at 51+. That matters only for the 51+ bin, which holds 2 % of the gap.

## 2. Candidate methods

GPU costs are anchored on measured runs:
- 8B LoRA SFT, 1 epoch on 98.7k rooms: 1 GPU × 17h45m ≈ 18 GPU-h (`isambard_main.sbatch:8`; research-state `sft_placement`).
- On-policy sampling: 27,000 samples in 0.62 GPU-h (log 17:34).
- Dev eval: about 4 GPU-h per candidate (018 budget). Test: about 70 GPU-h.

### M1. Iterated checker-filtered self-training (rejection-sampling fine-tuning, "RFT")
- **Change.** Sample k=8 per train room with the existing `sample_vllm.py`. Keep samples that pass the checker, plus S1/S3-repaired failures if they pass. Fine-tune with `train_dpo --sft_only`. Repeat 2–3 rounds.
- **Cause addressed.** 2, and partly 1: it pushes P's mode towards its own feasible samples.
- **Expected gain [I].** Dev raw from 0.47 towards pass@8, about 0.55–0.62. CasaGPT reports 3 rejection-sampling rounds taking living-room NIRate from 55.9 to 89.0 (earlier output-plan, re-checked there). Amendment 016 §3(b) RFT arm, already running, gives an early reading.
- **Cost.** About 3 GPU-h sampling plus 6–8 GPU-h SFT per round, so about 25 GPU-h for 2 rounds.
- **Effort.** Low; the tools exist.
- **Risk.** Ceiling is about pass@k. It is off-policy, which is harmless here.
- **Novelty.** Low.

### M2. GRPO with the checker as a verifiable reward
- **Change.** Start from R (rpo-dpo1) or from 018's adopted model, with an FP32 LoRA.
  - Reward: pass bonus + dense penalty (normalised overlap area, OOB area, fixed-blocking, support) + format.
  - KL anchor to the start policy.
  - Curriculum over rooms with 1–50 objects. Group size G=8, about 3k rooms. One variant gets per-object credit, as MetaSpatial's 3D-SPO does.
- **Cause addressed.** 1 and 2. The dense reward still gives signal in rooms where all 8 samples fail.
- **Expected gain [I].** About pass@8–16 of the start, roughly 0.60–0.65 dev raw. RLVR mostly turns pass@k into pass@1 (Yue et al. 2025, arXiv 2504.13837; from prior knowledge, not re-checked).
  - Literature: 3D-Layout-R1 plain GRPO raises collision-free 0.700 → 0.954; LayoutDSL raw-parameter SFT 0.647 → RL 0.749.
- **Cost.** About 20–40 GPU-h per run.
- **Effort.** 1.5–2 days. No GRPO code exists in `fastfill/` [V]. TRL GRPOTrainer with vLLM colocate is the shortest path; it is untested on aarch64.
- **Risk.**
  - Reward hacking: ThinkBLOX's plain GRPO reached CF 82.6 while in-boundary fell to 36.2 and PSA to 8.65.
  - Reward = evaluation checker (Goodhart). Independent guards are needed: mesh collisions (019), admission, OOR, GT-relation hold rate.
  - vLLM-vs-HF policy mismatch (cause 5) needs an importance-ratio correction or HF scoring.
- **Novelty.** Medium-low alone: 3D-Layout-R1, LayoutDSL, SPG-Layout, MetaSpatial and ThinkBLOX all do this in 2026.

### M3. Distil from O2's final answers (amendment 018, running): keep unchanged
- **Cause addressed.** 2 (DPO1-D uses a strong chosen against P's own failing samples), and 1 only weakly.
- **Expected gain [I].** +0.03 to +0.15 on dev. Answer-only distillation adds about 6k feasible targets to 98k that already exist. DPO1-D is probably the useful part.
- **Expected teacher yield [I]:**
  - T-A: about 2.7k targets.
  - T-B ≤20 objects: about 1.5k.
  - T-B 21–50: about 0.8k.
  - T-B >50: about 0.8k, low yield because of O2's truncation.
- **Cost.** ≤150 GPU-h cap.
- **Novelty.** Low.

### M4. Distil O2's reasoning traces, then GRPO (the 3D-Layout-R1 recipe)
- **Change.** SFT on O2's reasoning plus answer, taken from the 018 teacher receipts.
  - **[V, code]** Each answer keeps a `reasoning` field (`llm_open.py:202`), and receipts are written per room (`llm_open.py:321-330`). That they are on the server is not verified.
- **Cause addressed.** 1 (bookkeeping learned from the teacher) and the capability ceiling.
- **Expected gain [I].** Larger than M3. In 3D-Layout-R1, reasoning SFT reached collision-free 0.854 vs 0.700 for answer-only, and 1.000 with GRPO added.
- **Cost.** About 8k × ~17k tokens ≈ 130M tokens, roughly 8–15 GPU-h. Inference grows about 35×: about 5 GPU-h on vLLM for dev+test, but HF-exact eval is impractical.
- **Risk.**
  - The student inherits O2's truncation in dense rooms.
  - 8k traces may be too few for the 8B to learn the skill.
  - It must be evaluated on vLLM (cause 5).
- **Novelty.** Low-medium.

### M5. Written-out geometry step generated from the training targets (no teacher)
- **Change.**
  - Re-run SFT on the 98k targets with a deterministic `<think>` block, P's equivalent of the agents' thinking. It lists each object's footprint box, computed from size and yaw (x0, y0, x1, y1).
  - The block uses a planned order: large floor objects, then parents, then children. Sizes are known in the input.
  - Each object also gets an instance index (chair 3/6). The JSON that follows is in contract order.
- **Cause addressed.** 1 (explicit occupancy in the context), 3 (size arithmetic becomes explicit tokens), the duplicate bug (instance index) and order.
- **Expected gain [I].** Unknown; nothing on our data yet. Supporting evidence:
  - 3D-Layout-R1: reasoning SFT +0.154 collision-free.
  - SPG-Layout ordering ablation: avg_LV 19.5 with large-first vs 132.7 without.
  - ThinkBLOX: plan-based order PSA 54.6 vs random 41.6.
- **Cost.** About 20–25 GPU-h; answers get roughly 2× longer.
- **Effort.** About 1 day: data builder plus a round-trip test.
- **Risk.**
  - Boxes make overlap visible but do not prevent it, so M5 should be combined with M1/M2.
  - Rooms with 51+ objects exceed the 8,192 training length; exclude them or raise the limit.
- **Novelty.** Medium-high in this field, pending a novelty check. It is teacher-free, built from valid targets, and the reasoning it contains can be checked exactly.

### M6. Switch the base model: Qwen3.8-27B student (O2's base), thinking off
- **Change.** Same SFT data (plus 018 targets), LoRA, then M1/M2 on top.
- **Cause addressed.** 4.
- **Expected gain.** Unknown until D1. If O2 without thinking passes about 0.8 on the pilot, this is the strongest lever.
- **Cost.**
  - SFT: about 61 GPU-h for a full epoch (18 × 27/8), or about 19 GPU-h on 30k rooms.
  - Inference: about 3.4×, so test about 240 GPU-h with HF exact, or use vLLM.
- **Risk.** Memory on GH200: use gradient checkpointing or FSDP over 4 GPUs. The BF16-LoRA fidelity issue also applies here.
- **Novelty.** Low, but it is a required control ("base or thinking?"). It also allows a clean contrast: same base, fine-tuned without thinking vs zero-shot with 11k thinking tokens.

### M7. Per-object verifier during decoding (backtracking)
- **Change.**
  - After each object's record closes, check it against the boxes already placed, the boundary and the keep-outs.
  - If it fails, resample only that object, up to k times; otherwise keep the least-violating draw.
  - Implemented with vLLM prefix caching and a stop at each record end.
  - Optional: use its outputs as M1 targets (expert iteration).
- **Cause addressed.** 1, at test time.
- **Expected gain [I, idealised independence model].** If a failed object is fixed with probability q=0.8, per-object survival becomes about 0.977, giving 0.79 / 0.63 / 0.31 at n = 10 / 20 / 50 (room-level best-of-8: 0.81 for 1–10 objects).
  - SMC-style control in another domain: 0.063 → 0.419 (arXiv 2504.13139, cited in the earlier output plan).
  - D4 below gives a ceiling from existing samples.
- **Cost.** About 1.3–2× greedy tokens.
- **Effort.** 1.5–2 days.
- **Risk.** Later objects get squeezed. It is a system-class row, so agents need the same verifier.
- **Novelty.** Medium, if the claim is amortised into the model via M1.

### M8. Solver hybrid: S3 global minimum-displacement projection after S1
- **Change.** Fixed sizes and yaw; min ‖p − p0‖² subject to non-overlap, polygon containment and keep-outs; support subtrees move rigidly.
- **Cause addressed.** System-level coverage of everything above.
- **Expected gain [V, exploratory].**
  - 60 rooms per bin: S1 0.679 → S1+projection 0.868, bin-weighted over complete rooms (`scenesmith/run60.json`).
  - On the pilot, at most about 47 of P+S1's 52 failures are XY-fixable, a ceiling of about 0.98.
- **Cost.** CPU only. Runtime p50 45 s and max 225 s at 51+ objects (unoptimised).
- **Effort.** About 1 day to productionise.
- **Risk.**
  - Semantic damage: median moves 0.14–0.26 m. OOR, FID and judge scores must be reported.
  - It must be applied to all systems.
- **Novelty.** Low: SceneSmith H.1, Steerable Scene Generation, LayoutVLM.

### M9. Relation program plus a deterministic interpreter (RoomWright / LayoutDSL)
- **Change.** The LLM outputs `on`, an anchor (wall, corner or object), a relation, and row/group tokens; a solver produces coordinates.
- **Cause addressed.** 1, 3 and duplicates, by construction.
- **Expected gain.** LayoutDSL: 0.878 vs 0.749 after RL. LayoutVLM: fine-tuning on raw poses hurt (PSA 43.2 → 11.9), while the relation representation helped (48.1). RoomWright does not specify or ablate its solver.
- **Cost / effort.** 4–7 days. Out of scope for this window except a CPU gate: GT → program → solver round-trip on dev, compared with GT's own checker pass of 0.722.
- **Novelty.** Medium.

### M10. Hierarchical split for dense rooms (floor pass, then one pass per parent)
- **Cause addressed.** 51+ objects (P 0.0) and O2's truncation as a teacher.
- **Expected gain.** At most about 0.02–0.07 of the overall gap.
- **Status.** Defer.

**Not worth doing [V reasons]:**
- A VLM critic loop: in SceneSmith's NoCritic ablation, collisions are 0.8 % vs 1.2 % for the full system, at $4.24 vs $13.98 per scene.
- Diffusion or flow heads: no evidence for 50+ objects, and they cannot take our conditioning.
- Number Token Loss: our failures are gross overlaps, not small numeric errors.
- More synthetic-perturbation DPO.

## 3. Plan for the next 2–4 days

Isambard queueing is the real limit, not GPU-h: our jobs run at Priority=1 with fair-share 0.18. Bundle work into few 4-GPU jobs. Per standing rules, I write scripts and the user submits after giving the go-ahead. Each step needs an amendment locked before its outputs exist.

### Day 0: diagnostics that act as decision gates (about 7 GPU-h plus CPU), amendment 020, descriptive

| ID | What | GPU-h | Decision rule (fixed in advance) |
|---|---|---|---|
| D1 | O2 with thinking off (new spec entry, `enable_thinking: False`) on the 300-room pilot with P's input. Optionally also O2 with a 2k thinking cap. | ~3 | If O2 without thinking ≥ 0.80 → run M6. If ≤ 0.60 → reasoning is the lever: M4/M5 first, skip M6. In between → both. |
| D2 | P greedy (HF exact) on 500 train rooms, stratified by object count | ~2 | If train ≈ test success → P underfits; more capacity / M6 / RL. If train ≫ test → it is a generalisation problem. |
| D3 | P on the pilot with input sizes ×0.85 and ×1.15 | ~2 | If the median position shift is < 2 cm → P ignores sizes, which supports M5. |
| D4 | CPU on the server: combine per object from 013's 9 samples per room (3,000 train rooms). For each object, take the first sampled placement that does not collide with those already placed. | 0 | The ceiling for M7/M1. If ≥ 0.8 on ≤20-object rooms → build M7. |
| — | Check that O2's `reasoning` is stored in 018 teacher receipts (`RUN/llm-open-O2-*/receipts/*.json`). | 0 | If it is → M4 costs no extra teacher GPU. |

Engineering on the same day: the M2 reward wrapper and a GRPO smoke test on 20 rooms; the M5 data builder with a round-trip test (the stripped JSON must equal the target).

### Days 1–2: training arms, amendment 021 (single dev-selection rule)

1. **M1 round 1** from R: about 10k train rooms × 8 samples. Start immediately; the tools exist. About 12 GPU-h.
2. **M5** re-SFT from base, with instance index and planned order. About 22 GPU-h (about 6 h on 4 GPUs).
3. **M2 GRPO** from R once the smoke test passes. About 30 GPU-h.
4. **Conditional on D1:** M6 27B SFT on 30k rooms (about 20 GPU-h) for a quick dev reading.
5. **After 018's teacher finishes:** M4 trace SFT. About 15 GPU-h.
6. **Day 2:** M1 round 2 or M2 on top of the best of {M5, 018-adopted, M6}. About 30 GPU-h.

### Days 3–4: dev selection and one test run

- **Dev:** 600 dev-select rooms, frozen checker raw over all requests, the same engine for every candidate (HF exact where feasible). About 4 GPU-h per 8B candidate, so about 30 GPU-h.
- **Test:** only the adopted model, about 70 GPU-h. The 27B student would cost about 3× that.
- **System-level rows** for P and all agents: +S1, +S3, ±M7.

**Budget (new work, on top of 018's ≤150): about 180–260 GPU-h.**

**Success criteria (dev):**
- **To continue an arm:** ≥ R + 0.05 checker raw (R = 0.470), with the house-cluster CI lower bound > 0 and a gain in both dev halves.
- **Guards:**
  - complete ≥ R − 0.005;
  - OOR not worse than R;
  - GT-relation hold rate (`validate.holds`) within −0.03 of R;
  - median object-to-GT distance within +0.03 of the room diagonal;
  - no regression on mesh collisions (019).
- **Stretch goal by day 4:** dev raw ≥ 0.70, and ≥ 0.90 after S1+S3.

**How this fits with 018:**
- **Keep:** 018 exactly as locked. Its DPO1-D (teacher chosen vs P's own failures) is its strongest part; its teacher run also produces M4's traces. Keep 013 op-DPO2 and the 016 ablations, which the paper's attribution claims need. Keep S1.
- **Add:** amendment 020 (diagnostics) and amendment 021 (M1, M2, M5, M4, and M6 if D1 says so, under one selection rule). S3 is a system-level row only.
- **Drop:** new synthetic-perturbation DPO; VLM critics; diffusion heads; the full relation DSL in this window (only its CPU round-trip gate); Number Token Loss.
- **Flag, without changing the lock:** 018's T-B puts a third of its rooms in the >50 bin, where O2's yield is about 0.5. Expect few dense targets.

## 4. What counts as a paper contribution and what is engineering

**Likely contributions:**
1. **Diagnosis (cheap and solid now).** Under fixed sizes, an 8B SFT+DPO model trained only on checker-valid targets shows:
   - a constant per-object failure rate;
   - identical instances collapsing to exact duplicates under greedy decoding;
   - no re-planning when the given sizes do not fit the GT arrangement.

   O2 passes 0.99 of the answers it completes, but spends 35× more tokens. Together with 016, the story becomes: sizes confound rankings when systems choose their own sizes, and once sizes are fixed, what remains is a planning gap rather than a label problem.
2. **A size-preserving, verifier-driven training method (needs results).** M5's teacher-free geometry step, combined with exact-checker RFT/GRPO.
   - The claim to aim for: "closes X % of the gap to a 27B thinking model at about 1/35 of the tokens, with no size shrinkage". Agents shrink footprints by 16–31 %; P cannot by construction.
   - What is new: the reasoning comes from valid targets without any teacher and can be checked exactly; and the setting is fixed-size, dense, 18-source rooms. Plain GRPO for layout is not new (five 2026 papers). Run `aris:novelty-check` before claiming anything.

**Engineering (needed as baselines or ablations, not a contribution on its own):** S1/S3 projection, best-of-N, the 27B student, answer-only distillation (018), object ordering, instance indices, and verifier decoding without amortisation.

**What reviewers will ask for:**
- comparisons at equal tokens per room;
- the same verifier and solver applied to the agents;
- independent metrics, because the reward is the same checker as the evaluation (019 mesh collisions, admission, FID, judge);
- two seeds for the adopted arm;
- dense and real-scan strata.

## 5. Main uncertainties
- Every gain estimate for M1, M2, M4, M5 and M6 is [I]. None has been measured on our data.
- The pilot is 93 % rooms with ≤20 objects, and only 12 rooms have >30 objects.
- Bin-weighted gap shares mix pilot and full-test room sets.
- O2's 0.920 includes up to 3 harness attempts. Its raw single-attempt rate is 0.92 over all rooms but 0.99 among complete answers.
- GRPO on aarch64 with TRL/vLLM is untested. The vLLM-vs-HF mismatch could hide or inflate gains.

## Files
- New scripts: `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/better/lead/small.py` (2-object room breakdown). The other [NEW] numbers came from inline analyses of `scratchpad/better/rootcause/{order.pkl,targets_150.pkl,pilot_table.pkl}`, `/Volumes/harddisk/fastfillout/unknownsize-20261013/scale-curve-016/rows.jsonl` and the eval-data `placement/requests.jsonl`.
- Sources read: `/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/{findings.md,amendment-016-*.md,amendment-017-*.md,amendment-018-*.md,amendment-019-*.md,research-log.md,research-state.yaml}` and `/Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/comparison/{llm_open.py,contract.py,isambard_main.sbatch}`.