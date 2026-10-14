# Merged audit: unknown-size J/P comparison, delivery P (rpo-dpo1)

**Re-checked from the CSVs before merging:**
- The four audits agree on the numbers except one: the completeness audit's "+0.0085" for the rpo-dpo1 test gain is a rounding slip. The correct value is 0.39806 − 0.38961 = +0.0084 (73 of 8,647 rooms).
- The blocker's cause is confirmed. `fid.py` treats a room as a candidate only when every system has an image (`len(good)==len(arms)`). T4 shows O1a-harness with 0 complete rooms in holodeck bedroom and living (and in the other holodeck strata, per the audits).

Fixes that need a new run are marked "(needs your go-ahead)". The visual run `visual-final-selected-002` is still rendering.

## BLOCKER

**B1. FID will be empty or tiny, for J and P as well as the agents.**
- **Where:** `fastfill/comparison/fid.py` (~291 and `_stratum_result` ~343), `visual_pipeline.py:438`, `visual_cases.py:345`, `visual_final.py`.
- **Cause:** FID uses only rooms where all 10 systems have an image. O1a has 0 complete rooms in all 4 holodeck strata, so J and P get no FID there.
- **Upper bound on usable rooms:** IL3D bedroom 68 of 419, IL3D living 3 of 87. Even without O1a, O3 limits holodeck bedroom and living to 33 and 36 rooms, below the N<50 flag.
- **Fix:** lock amendment 011 before anyone looks at an FID.
  - Primary J/P FID uses reference∩J∩P.
  - Each agent's FID uses reference∩J∩P∩that agent, with its N reported.
  - The all-system intersection is secondary, or limited to the judged set {J, P, L, O6}.
  - Implement this through the fid-manifest `systems` option.
- **Files:** new `amendment-011.md`, `visual_final.py` / `visual_pipeline.py`.

## MAJOR

**M1. The delivered merged weights were never evaluated, yet the model card leads with them.**
- Every test number comes from the unmerged adapter on its base model.
- The BF16 merge changed only 6.75 % of target weights (5.4–9.2 % per module), against 77.6 % for the SFT merge. Amendment 009's own argument is that a BF16 merge erases small updates.
- MODEL_CARD.md:240 says the merge "folds" the update in. findings.md:50 ("11–19 % changed") does not describe the delivered model.
- **Fix:** make `training-base/` + `adapter/` the primary snippet and label `model/` "unevaluated BF16 merge (6.75 % changed)". Optionally score `model/` on the 600 dev rooms and report exact-match against the unmerged policy plus dev a′ (needs your go-ahead).
- **Files:** `delivery.py:185-190, 236-253, 329-336` (→ MODEL_CARD.md:22, 56-58, 80, 240), `findings.md:29, 50`.

**M2. The claim that rpo-dpo1 beats P-SFT on test is not supported** (findings.md:29, "≈+0.009, consistent with dev").
- The exact gain is +0.0084 raw. It comes entirely from MansionWorld (+82 rooms, only 15 house clusters).
- Without MansionWorld the gain is −0.0012. On holodeck+3D-FRONT it is −0.0007. Reweighted to the dev mix it is +0.005, against a dev gain of +0.062.
- There is no paired CI, and the comparison is not labelled EXPLORATORY.
- **Fix:** reword as "+0.008 raw (EXPLORATORY, no CI), comes from MansionWorld only, the dev gain did not replicate". Add a house-cluster bootstrap of P-selected − P-SFT using `report_v2.contrasts`.
- **Files:** `findings.md:29`, `stage_table.py`.

**M3. a′ comparisons across systems mainly measure how much each system was rescaled.**
- **The rescaling is unequal:** P is enlarged 1.1 % per axis, while the open models are enlarged 16–25 %. That lowers their scores for scatter and tight packing, not only for mean bias.
- **Control, GT layouts on the cohort at each factor:**

  | factor | 1.0 | 1.011 | 1.024 | 1.092 | 1.168 | 1.189 |
  |---|---|---|---|---|---|---|
  | GT success | 0.725 | 0.697 | 0.660 | 0.246 | 0.093 | 0.079 |

- **Equal-size control:** L 0.369, P 0.273, J 0.235, O2 0.212. The order holds, but the P−O2 gap shrinks from 0.17 to about 0.06.
- **Not same-input:** P gets prior sizes that the agents do not.
- **No CI:** none of the a′ contrasts has one.
- **Fix:**
  - Reword findings.md:30 as "fixed-position stress test, EXPLORATORY".
  - Add paired Δ(a′) with house-cluster CIs to T5_contrasts (cohort and strata_pooled).
  - Add GT rows to T5 (cohort and strata_pooled), both unscaled and at the same factors, or replace T5 a′ with the equal-size control.
- **Files:** `report_v2.py` (T5, T5_contrasts), `findings.md:30`.

**M4. findings.md is stale and contradicts itself.**
- Line 10 still says the binding decision is at DPO2 (amendment 008); amendment 010 replaced that.
- Line 5 says agents are "running" and cites JPL-001.
- Lines 38-43 use P-SFT (P 0.510, P−L −0.424) and mix correction factors. J and P use full-test factors; L uses its own cohort factor. As a result J's a′ on the same cohort appears as both 0.371 and 0.393.
- Lines 53-57 list open questions that are already answered.
- **Fix:** rewrite against the selected-report T5 with stage labels (L 0.7655, J 0.3934, P 0.4986; P−L −0.415 [−0.443, −0.387]), keep one agent section, and reduce the open questions to the visual run and O5.
- **File:** `findings.md`.

**M5. "Every J DPO candidate shrinks sizes further" is false** (findings.md:25; also research-log 06:45).
- e1-dpo2 on dev: B −0.071 vs SFT −0.087, so it shrinks less. Its size log MAE is 0.2516 vs 0.2540, so its sizes are more accurate.
- It was rejected only for completeness (0.842).
- **Fix:** "3 of 4 J candidates shrink further". Qualify the sentence "DPO on J trades size fidelity for validity".
- **File:** `findings.md:25`.

**M6. O5 not being run is disclosed nowhere a reader looks.** Amendment 007:20 requires it.
- `delivery.json` has `agents.excluded={}`, MODEL_CARD.md:232 lists 8 agents, and T5 and findings.md are silent.
- Cause: O5 was left out of the final chain's arguments (`submit_final_010.sh:118-131`).
- **Fix:** pass `excluded={"O5": "not run: 4 engine-build failures, never substituted"}`. Have `report_v2` render the not-run list from `llm-open-status.json`, including the amendment-007 "too large / gated" list.
- **Files:** `report_v2.py`, `delivery.py:318-321`, `findings.md`.

**M7. The model-card agent table can mislead.**
- It shows only raw (a) and (b), with system IDs and no model names. A reader sees P 0.519 beside L 0.935 and O2 0.928 with no undersizing context, which breaks the study's rule in findings.md:47.
- **Fix:** add model/reasoning, `bias_fp`, `success_debiased_rate`, raw OOR, and a caveat that the cohort is not representative.
- **File:** `delivery.py:299-305`.

**M8. The P request snippet fails on a real room.**
- `bind_room` and `messages()` require ground-truth `size`/`pos`/`yaw` for every object. A room with only id and category raises `ValueError: bed.size: expected 3 numbers`.
- **Fix:** document the placeholder rule (size [1,1,1], pos [0,0,0], yaw 0 do not change the prompt), or add a request builder that does not need labels.
- **Files:** `delivery.py` (→ MODEL_CARD.md:93, 97), `contract.py:123-125`.

**M9. No code or library versions are pinned.**
- The snippet depends on `contract.py`, `size_prior.py` and `scene.py`, none of which is shipped. delivery.json has no commit, no code hashes and no torch/transformers/peft versions.
- **Fix:** record commit 26a841a3c, the three file hashes and the inference receipt's `versions`. Ideally also copy the three files into the package.
- **File:** `delivery.py:125-150`.

**M10. The "repair harness" is really a blind retry.**
- The repair message is only `"<ExceptionClass>: invalid contract output/request"`, while amendment 002 §3 and every agents report say the parser error text is returned.
- **Fix:** disclose it as "generic retry, exception class only". Optionally, by amendment, send `str(exc)` and rerun only the harness rounds.
- **Files:** `train_pair_helpers.py:362-364`, `llm_open.py:217-219`, `llm_agent.py:378-379`, report text.

**M11. Output budgets differ by system and are not disclosed.**
- L gets 65,536 tokens. Open models get 1,024 + 128·obj (+16,384 with thinking), capped at 40,960 minus the prompt.
- 241 of O2-raw's 252 failures are cap truncations, so O2-raw 0.808 vs L 0.935 is largely a budget artefact.
- The T6 cap-hit count (O2: 410) covers all rounds but is printed on both the raw and harness rows.
- **Fix:** add the cap formula and per-round cap hits to T5_cost, state that harness means up to 3 attempts, and label the T6 count "all rounds".
- **File:** `report_v2.py` (T5_cost, T6).

**M12. The agent cohort is not representative, and nothing says so.**
- Holodeck is 34.6 % of the cohort vs 5.8 % of the full test set; SpatialLM 2.1 % vs 28.5 %.
- Cohort J−P is −0.0887, against −0.0098 on the full test set.
- On the 445 rooms outside the six strata the order flips: J 0.389 > P 0.339, and O2 0.843 > L 0.820.
- **Fix:** add an "other sources (445)" scope plus a caveat in T5 and in the model card.
- **File:** `report_v2.py:803`, `delivery.py`.

**M13. O1a ≈ 0 is a failure on the `on` (support) field, not evidence that fine-tuning teaches layout.**
- O1a completes 0 of the 749 rooms that have no fixed items, and 505 of its 506 complete rooms fail the support check.
- **Fix:** state this in T5 and findings. Optionally add a lenient-`on` sensitivity for O1a and O1b, labelled EXPLORATORY.
- **Files:** `report_v2.py` (T5 note), `findings.md`.

**M14. The GPT judge output only compares J and P.**
- **Fix:** add pooled 174-room means per arm and P−L / P−O6 paired differences (estimation only, stratified bootstrap).
- **File:** `report_v2.py:371-395`.

**M15. research-state.yaml is stale.**
- `cap_gpu_hours` is 260 (it has been 600 since amendment 008). `phase`/`observed_utc` read dev_selection/04:45Z. Data status, `api_agent`, `open_models_rerun`, `pending` and the automation status are all out of date.
- There is no `delivered` or `visual_run` entry.
- **Fix:** update these fields and add `delivered` and `visual_run` entries.
- **File:** `research-state.yaml:5-33`.

## MINOR

1. **Holm at the secondary stage.** Holm is computed at every stage, so the selected stage shows (b) Holm 0.042 next to the confirmatory 0.066. The amendment is hard-coded as "006", and `holm_co_primaries` is a misnomer there. Fix: emit Holm only for the SFT stage, take the amendment label from an argument, and label the selected stage secondary in findings.md:27-29. Files: `report_v2.py:567, 706, 708`.
2. **Rounding and a hidden model in findings.md:30.** L a′ is 0.765, not 0.766. "Base Qwen3-8B ≈ 0" hides O1b, which scores 0.110 → 0.058. Files: `findings.md:30`; research-log 09:05 and 10:55.
3. **a009-dpo2 "complete 0 (all to the cap)" is wrong.** Only 209 of 600 rooms hit the cap. File: `findings.md:25`.
4. **findings.md:22 mixes room sets and cites an unarchived number.** The log MAE and B values for DPO1 are on 3,851 rooms, those for SFT on 8,610. "Reward accuracy 1.0" belongs to DPO2 training and is in no archived artifact. File: `findings.md:22`.
5. **Missing bootstrap-error hedge.** With 20,000 draws, the (b) Holm p is 0.0496–0.0524, so "not confirmed" sits on the 0.05 boundary. File: `findings.md:9`.
6. **"Lead over GT" is not robust.** With per-source factors, L's a′ (0.708) is below GT's (0.725). File: `findings.md:41`.
7. **Bootstrap p-values print as 0.** They should read "< 1/R". File: `report_v2.py:176-178`.
8. **The a′ CI ignores uncertainty in B.** The correction factor is computed once, outside the bootstrap. The decision is safe (−0.071 vs −0.02), but this should be disclosed or B re-estimated per draw. File: `report_v2.py:138-152`.
9. **`rpo_alpha` (1.0) is missing from the recipe and the hyper-parameter table.** It appears only as hard-coded prose. Fix: render it from the receipt and fail if it is absent for rpo candidates. Files: `delivery.py:197, 205-207`, `stage_table.py:362-365`.
10. **The `singleton_source_signature_strata` warning goes unmentioned.** It is attached to every T1 CI, including the confirmatory ones. File: `findings.md`.
11. **The snippet drops the ID mapping and never mentions size attachment.** Fix: `room, sidecar = bind_room(...)`, map IDs through `sidecar['bound_to_original']`, attach `sizes[id]`, and say that P outputs no sizes. File: `delivery.py` (→ MODEL_CARD.md:93).
12. **The snippets do not reproduce the evaluated path.**
    - The evaluation used a per-batch cap min(40959, 40960−W, 1024+128·max objects), with batch 8 and left padding.
    - It stopped only at `<|im_end|>`; the snippet's generation config also stops at `<|endoftext|>`.
    - It parsed strictly; the snippet uses a bare `json.loads`.
    - `request` is used before it is defined, and the `messages` variable shadows the imported function.
    - Fix: set `eos_token_id` to `<|im_end|>`, call `validate_output`, state the batching caveat, and reorder the snippets with the list renamed to `msgs`.
    - Files: `delivery.py` (→ MODEL_CARD.md:51-68, 97), `infer_shard.py:76-81`.
13. **The package is not self-contained.** It holds 4 metadata files, yet the status is `DELIVERY_PACKAGE_READY`. Fix: add an assembly step (copy each `files_to_copy` entry, then `sha256sum -c SHA256SUMS`) or rename the status "manifest ready". File: `delivery.py`.
14. **The size prior's provenance cannot be checked from the package.** Fix: add `eval-data-v1/MANIFEST.json` and the inference `run_receipt.json` to `files_to_copy`. File: `delivery.py`.
15. **P's fallback weak spot is undisclosed.** 1,403 of 8,647 rooms (16.2 %) and 8,238 of 108,253 objects (7.6 %) use the global fallback size, with log MAE 1.229. File: MODEL_CARD Limitations via `delivery.py`.
16. **a′ is diagnostic-only.** It uses true test sizes, so a user cannot obtain it. File: `delivery.py` (→ MODEL_CARD.md:5, 110).
17. **The agents-sft-paired reports label every system "L".** Their P column is P-SFT (0.5103), not the delivered P, and the stage is not stated. Fix: label rows by system and model from `summary.json` and regenerate; no rescoring needed. Files: `evaluate_llm.py:34, 91-93, 116-117`.
18. **T6 does not split agent invalid outputs into bad JSON vs bad schema.** Fix: pass the agents' raw text into `taxonomy`. File: `report_v2.py:681`.
19. **The T4 note describes a visual run that does not exist.** Fix: when `visual_mode.fid` is None, say visual metrics are pending. File: `report_v2.py:791-793`.
20. **Model modes are under-disclosed.** O3 is the only large model with thinking off. O7's template default effort is high, but T5 shows "model default". gemma is ranked without its mode. Files: `report_v2.py` (T5), `findings.md:30`.
21. **The judged open model (O6) is not the measured best (O2 0.928 vs 0.701).** Fix: word it as "pre-specified". File: report text.
22. **The same rescaling issue affects the J/P delivery rule's a′.** GT scores 0.6104 at J's factor and 0.7331 at P's, a gap of −0.123, larger than the observed −0.060. The decision does not change because condition (i) already fails. Fix: add a GT control row beside T1. File: `report_v2.py`.
23. **The pair/category prior-level split in S2 was never produced.** Fix: regenerate S2 with the frozen `size-priors.json` (073522…). File: `report_v2.py` (S2).
24. **Superseded reports carry no notice.** Affected: JPL-001 and -002, dpo1-001, and `current-results.md`. Fix: add a `results/README` marking them superseded; do not edit the hashed outputs.
25. **Metrics promised by amendment 008 are missing without a "not measured" note.** Automatic acceptance and all-room rotated OOR are absent from the stage table; human UR is also not measured. File: `stage_table.py`.

## 中文执行摘要（仅含审计已核实的数字）

1. 交付模型为 P（rpo-dpo1，RPO α=1.0），按修订 010 规则选出：dev 上 a′ 0.435，比其余合格候选高 0.043。J 保持 SFT，交付规则在 SFT、DPO1、最终三个阶段都给出 P。
2. 交付比较（全测试集 8,647 间）：几何成功率 (a) J 0.388 对 P 0.398，Δ −0.010 [−0.021, +0.002]，Holm p 0.095，不显著。
3. 尺寸 log MAE (b)：J 0.286 对 P 0.281，Δ(J−P) +0.0046 [+0.0007, +0.0087]，J 不更准；去偏成功率 a′ J 0.344 对 P 0.404，Δ −0.060 [−0.071, −0.049]。两条交付条件都不成立，因此交付 P。
4. 预注册的确认性比较（SFT 阶段）：(a) Δ −0.0014 [−0.012, +0.009]，Holm p 0.793；(b) Δ +0.0044 [+0.0005, +0.0084]，Holm p 0.066。两项都未确认，(b) 处在 0.05 边界、受 bootstrap 误差影响。
5. 尺寸偏差：J 系统性偏小（足迹 B −0.083），P 基本无偏（+0.015）。
6. rpo-dpo1 相对 P-SFT 的测试增益只有 +0.0084（净 73 间），几乎全部来自 MansionWorld（15 个房屋簇），去掉后为 −0.0012。dev 上 +0.062 的增益没有复现，应标为探索性。
7. 交付的 BF16 合并权重从未评测：合并只改动了 6.75% 的目标权重。全部测试数字来自 adapter+base，建议把 adapter+base 作为主交付形式。
8. 同输入代理（1,454 间）原始成功率：L（gpt-6.1-sol）0.935、O2 0.928、O6 0.701、O4 0.687、O7 0.592、P 0.519、J 0.431，P−L −0.415 [−0.443, −0.387]。开源模型全部尺寸偏小（B −0.30 到 −0.45）。O2 的部分失败来自 token 上限截断，各系统预算不同。
9. a′ 上 L 0.765、P 0.499、J 0.393、O2 0.327，但这个指标主要反映各系统被放大的幅度：GT 按 1.17 倍放大后只剩 0.093。在等尺寸对照下 P−O2 差距从 0.17 缩到约 0.06，只能作为探索性结论。
10. 未完成两项。一是视觉评测（FID/KID/GPT 评审）仍在渲染，而按当前"全系统交集"计算，6 个分层中有 4 个连 J/P 都拿不到 FID（BLOCKER），需在看到 FID 前锁定修订 011。二是 O5 因 4 次引擎构建失败未运行，而且至今没有在任何表里披露。