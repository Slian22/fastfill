**Audit: fairness and validity of the same-input agent comparison (T5 and agents-sft-paired)**

I re-scored the published rows locally on the same frozen code (geometry.py sha 0d3f7bee… matches geometry-v5). Inputs: `/Volumes/harddisk/fastfillout/unknownsize-20261013/visual-final-selected-002/inputs/` (geometry-v5, llm-eval-*-selected-v1, eval-data-v1) and `server-results/llm-agent-001`.

**Checked and correct**
- Every T5 (a) value recomputes exactly.
- Each system's own cohort footprint bias B recomputes exactly: J −0.0476, P −0.0223, L −0.1751, O2 −0.3109, O6 −0.3459, O4 −0.2962, O7 −0.3234, O3 −0.4500, O1a −0.4183, O1b −0.3119.
- All agents get the same suffix (`llm_open` reuses `llm_agent.SUFFIX`) and the same parser.
- The token-cap rule changes no agent number. No capped answer was scorable: cap&complete = 0 for every system and variant.
- The 180 GPT-judge rooms (`gpt-selection.json`) were fixed before any prediction existed.
- L never needed more than 12,545 output tokens (most of 1,454 answers), so L gained nothing in practice from its larger budget.

**BLOCKER**

1. **The running visual run will produce almost no FID, including for J and P.**
   - Where: `fid.py:291` counts a room only if every system has an image (`len(good)==len(arms)`), and `visual_pipeline.py:438` puts all 10 systems in that check. The run `visual-final-selected-002` (still in the render stage) includes O1a, O1b and O3.
   - Effect: O1a has 0 harness-complete rooms in all four holodeck strata. Upper bounds on the shared room count per stratum (before any render failure):
     - holodeck bathroom / bedroom / kitchen / living: 0 / 0 / 0 / 0
     - IL3D bedroom: 68 of 419
     - IL3D living: 3 of 87
   - Without O1a, O1b and O3 it would be 417, 82, and every holodeck room.
   - So T4 FID is unmeasured for J and P on 4 of 6 strata, and very selective on the other 2.
   - Fix: lock an amendment before any FID is viewed. Make the primary FID per stratum use only reference∩J∩P, or reference plus the judged set J/P/L/O6. Report the all-system intersection as secondary, or leave systems with near-zero completion in a stratum out of it.

**MAJOR**

2. **The "repair harness" is really a blind retry.**
   - Where: `train_pair_helpers.py:362-364` replaces every error with the fixed text `"<ExceptionClass>: invalid contract output/request"`. That text is the repair message at `llm_open.py:217-219` and `llm_agent.py:378-379`.
   - The validator's own messages (e.g. "yaw must be integer degrees", "invalid support ID") are fixed strings with no data in them, but they are thrown away.
   - Amendment 002 §3 and every agents report say the repair returns the parser error text, which implies a diagnostic.
   - Effect: O1a invalid 948 → 946 and O3 972 → 881 after repair; the weak models cannot learn what to fix.
   - Fix: disclose it as "generic retry, exception class only". Optionally, by amendment, send `str(exc)` from `_validate_output` and rerun only the harness rounds.

3. **Output budgets differ by system, and T5 does not say so.**
   - L: `max_output_tokens` 65,536 (`llm_agent.py:507`; confirmed in the L run receipt).
   - Open models with thinking: 1,024 + 128 × objects + 16,384, also capped at 40,960 minus prompt (`llm_open.py:51, 130-132, 198`). Without thinking: 1,024 + 128 × objects.
   - O2-raw: 241 of its 252 failures are cap truncations (median 15 objects). Its 410 capped answers are about 241 + 111 + 58 across the three rounds, so O2's repairs are mostly re-runs with a fresh budget.
   - Effect: O2-raw 0.808 vs L-raw 0.935 is largely a budget artefact.
   - Fix: in T5_cost add the cap formula, the context limit, and cap hits per round. State that harness means up to 3 attempts.

4. **O5 being not run is disclosed nowhere a reader looks.**
   - `amendment-007:20` requires not-run systems to be "reported in the tables with reason". `report.md`, `MODEL_CARD.md` and `findings.md` never mention O5, nor the not-run list (Kimi, GLM and others).
   - `research-state.yaml` `o5_rerun_3` does record "O5 NOT RUN after 4 engine-build attempts".
   - `delivery.json` has `agents.excluded = {}` because O5 was removed from the final chain's argument list, so the disclosure path at `delivery.py:318-321` never fires.
   - O5 was also the originally designated judged open model (amendments 005 §5 and 006).
   - Fix: have `report_v2` read the `llm-open-status.json` that `llm_open.write_status` already writes and render a "未运行" table; or pass O5 into the final step as excluded with its reason.

5. **The own-bias rescaled success (a′) does not compare fairly across systems.**
   - The per-axis enlargement each system gets: J 2.4%, P 1.1%, L 9.2%, O4 16.0%, O2 16.8%, O7 17.6%, O6 18.9%, O3 25.2%.
   - Control: GT layouts on the cohort, scaled by the same factors, with the same `_derive` path:

     | factor | 1.0 | 1.0112 | 1.0241 | 1.0915 | 1.1682 | 1.1888 |
     |---|---|---|---|---|---|---|
     | GT success | 0.7249 | 0.6974 | 0.6596 | 0.2462 | 0.0928 | 0.0791 |

     So (a′) mostly measures how much a system was rescaled.
   - `findings.md:30` says "P exceeds every open model" (P 0.499 vs O2 0.327).
   - An equal-size control (GT x/y sizes at each system's own position and yaw; my approximation of audit wf_b7ae99d6-8df F3) gives L 0.369, P 0.273, J 0.235, O2 0.212, O4 0.210, O6 0.175, O7 0.158. The ordering holds, but the P−O2 gap shrinks from 0.17 to about 0.06.
   - Fix: add a GT-under-the-same-factor row, or replace T5 (a′) with the equal-size control; reword `findings.md:30`.

6. **The cohort is not representative, and T5 does not warn.**
   - Its header (`report_v2.py:803`) just says "same frozen cohort, same input".
   - Composition vs the full test set: holodeck 34.6% vs 5.8%; IL3D_3dfront 36.9% vs 10.1%; SpatialLM 2.1% vs 28.5%; MansionWorld 2.1% vs 15.2%.
   - Cohort J−P is −0.0887, against −0.0098 on the full test set (about 9× larger).
   - On the 445 rooms outside the six strata the ordering flips: J 0.389 beats P 0.339, and O2-harness 0.843 beats L 0.820.
   - Fix: add an "other sources (445)" scope next to cohort and strata_pooled, plus a representativeness caveat in T5 and in the MODEL_CARD.

7. **O1a ≈ 0 is a failure on the `on` (support) field, not proof that fine-tuning teaches layout.**
   - O1a-raw completes 0 of 749 cohort rooms that have no fixed items, and 506 of 705 that do.
   - In its complete outputs, 3,980 of 5,266 objects are placed "on" a fixed item, and 505 of 506 complete rooms fail the support check. The schema in the prompt shows `"on":support_id` on every placement.
   - Amendment 003:9 and 004:13 frame O1a as isolating the effect of fine-tuning, so J−O1a +0.43 really measures learning the output contract.
   - Fix: say this in T5 and findings; optionally add an EXPLORATORY lenient-`on` sensitivity for O1a and O1b.

**MINOR**

8. **The agents-sft-paired reports mislabel every system as L.**
   - Every report is titled "LLM 代理 L" and its rows read L-raw / L-harness; the table also has "API 失败" and "调用（含失败重试）" columns. The causes are `evaluate_llm.py:34`, `91-93` and `116-117`. For example, `O1a/report.md` shows "L-raw 0.0000".
   - The P column is P-SFT (0.5103), not the delivered P (0.5193), and the stage is not stated.
   - "修复后几何成功率" there means the 10 cm boundary snap, not the repair harness.
   - Fix: label rows by system and model from `summary.json` and regenerate; no rescoring is needed.
9. **T6 does not split agent invalid outputs into bad JSON vs bad schema** (`report_v2.py:681` calls `taxonomy` without records). So parser strictness (code fences, float yaw, `"on": null`) cannot be judged for O3 (880 harness invalid) or O1b (609). Fix: pass the agents' raw text into `taxonomy`.
10. **Wrong T4 note when no visual inputs exist.** With `visual_mode.fid` None, `report_v2.py:791-793` prints "此 J/P 视觉运行不产出 KID…" as if a run existed. It should say visual metrics are not measured yet.
11. **`findings.md:41` mixes definitions.** It calls the values "own bias", but J and P use full-test factors (1.0424/0.9925) while L uses its cohort B, and P is P-SFT. Audit F5 already flagged this. Fix: use the selected-stage T5 values (L 0.7655, J 0.3934, P 0.4986).
12. **T6 "代理达上限回答数" counts all rounds** (O2: 410) and is printed identically on the raw and harness rows. Fix: label it "all rounds".
13. **Model modes are under-disclosed.** O3 is the only large model run with thinking off; O7's template default effort is high (research-log line 90), but T5 shows "model default". `findings.md:30` ranks gemma 0.241 without stating its mode.
14. **The judged open model is not the strongest one.** O6 is chosen by the fixed priority in `amendment-007:26` ("strongest … O6, else O5"), but the measured best open model is O2 (0.928 vs O6 0.701). Fix: word it as "pre-specified".
15. **Same rescaling issue in the J/P delivery rule; the decision is unchanged.** GT on the full test set scores 0.6104 under J's factor (1.0423) and 0.7331 under P's (0.9928), a gap of −0.123. Observed Δ(a′) J−P is −0.060. The decision does not change because condition (i) already fails (Δ(b) +0.0046 [+0.0007, +0.0087]). Fix: add the GT control row beside T1.
16. **MODEL_CARD agent table (`delivery.py:299-305`) leaves things out.** It omits raw OOR (an amendment 005 §4 endpoint) and B/(a′), and it says "same frozen cohort and inputs" without a caveat.

Scripts that reproduce these numbers are in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/`:
- `rew.py`: T5 rates and source reweighting
- `bias.py`: own B per system
- `cap.py`: cap hits
- `o1a.py`: support-field analysis
- `gtscale.py`, `gtfull.py`: GT scaled by each system's factor
- `oracle.py`: equal-size control