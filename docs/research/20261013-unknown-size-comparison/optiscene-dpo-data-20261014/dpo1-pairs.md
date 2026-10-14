I compared the paper, the official release and our implementation for both DPO stages. The main finding is that our stage-1 DPO follows the paper's text, but the released `dpo_prompts.json` does not. Released stage 1 is "correct layout vs badly broken layout", and almost half of its pairs carry no gradient. Ours paired each correct layout with one of the model's own plausible samples and trained 10 epochs. That setup is the one most likely to collapse.

**Sources and method.**
- Paper text is from `pdftotext -layout` of the PDF: §3.2, §3.4, §4.1, Table 6, App. C and D.2.
- Official code: README, `main.py`, `utils.py`, `scripts/dpo_train.py`, `scripts/sft_train.py`.
- Our side: `dpo_stages.py`, `build_dpo.py`, `train_dpo.py`, amendments 008–010, `results/stage-table-010-001/stage_table.md`, `research-log.md` and `findings.md`.
- The three datasets were stream-parsed with ijson. Token counts use the local Qwen2.5-7B-Instruct tokenizer and chat template (the README base model); token counts for completions are +2 for the end marker.
- OOR follows the paper's App. D.2 formula; I used each object's w×d footprint, rotated and turned into an axis-aligned box, with sizes from the chosen side's input.
- Out-of-bounds could not be checked directly because the data gives only the room area. I used a proxy: whether the rejected layout reaches more than 5 cm past the chosen layout's own outer extent. A "square room of that area" check was unusable: 98.6% of correct holodeck layouts already fail it, so I dropped it.
- The length numbers for our side come from pilot-v3 training rows (only the holodeck and 3D-Front sources, 128 rows), not from the real DPO pairs.

## Side-by-side

| Item | Paper text | Official released data / code (measured) | Ours (amendments 008–010) |
|---|---|---|---|
| **Positives** | §3.4: about 2,200 layouts with unanimous annotator approval; App. C: 1,100 Holodeck-Synth + 1,100 3D-Front | dpo1 has 1,977 unique positives (1,082 HD + 895 3D-Front). dpo2 has 2,127 (1,156 HD + 971 3D-Front). 1,940 are shared. Chosen equals an SFT training target word for word in 75.6% of dpo1 pairs (82.0% of HD pairs) and 69.7% of dpo2 pairs. | 2,200 (1,100 holodeck + 1,100 IL3D 3D-Front), picked by a geometry check plus a hash ranking, not by people. Chosen is the SFT training answer in 100% of pairs. Same positives in both stages. |
| **DPO1 negatives** | The SFT model generates several candidates; they are "semantically plausible but deviate from human intuition" | **3D-Front: chosen == rejected in all 3,626 pairs** (the rejected uid has `#####` inserted), so zero gradient. **HD: 4,325 pairs**, 4 distinct negatives per positive (1,070 of 1,082). In 98.1% of them all objects sit on one y line. 99.8% of objects are moved; median move 3.61 m. OOR 0.381 vs chosen 0.0005. 100% reach past the chosen layout's extent (median 2.0 m). The rejected entry's own bounding-box input differs from the prompt's in 4,325/4,325 pairs; training ignores it (utils.py uses only [0] for the prompt). No rejected output appears in the SFT data. | One negative per positive: the arm's own SFT sample (T 0.85, top-p 0.95, n 4), the first one that ended on EOS, is valid and differs from the positive. **No requirement that the negative be geometrically worse**; `rejected_geometry_success` is only recorded. 2,200 pairs, kept in both arms. |
| **DPO2 negatives** | Perturb the correct layout: objects past the boundary, objects overlapping (App. C, two noise types) | 8,410 pairs: 3,813 3D-Front + 4,597 HD; only 12 identical. About 4 negatives per positive. One object moved in 56.7% (3D-Front) and 51.8% (HD) of pairs, otherwise several. Median move 0.37–0.38 m (p25 0.33, p75 0.75–1.15). 28–32% of pairs also rotate objects by 90° or 180°. OOR goes up in only 54% (3D-Front) and 62% (HD) of pairs. 9.6% of 3D-Front pairs show neither proxy violation. | One per positive. One floor object (and anything it supports) is moved either onto another object's centre ±8 cm or 0.35 m past the wall. Positions only; no rotation noise. Each pair is checked for the targeted violation. 2,200 pairs. |
| **Pair counts / steps** | Not given | 80/20 random split (`dpo_train.py`): DPO1 trains on 6,360 pairs, DPO2 on 6,728. On one GPU at effective batch 4 that is about 15,900 / 16,820 steps over 10 epochs. In DPO1 only 54.4% of pairs carry gradient. | 2,200 pairs. Amendment 009: 5,500 steps (10 epochs). e1: 550 per stage. rpo: 1,650. |
| **Prompt / reasoning** | An SFT "semantic reasoning" summary comes before the layout (Table 6 has a Reasoning column) | One 4,485-character instruction shared by SFT and both DPO files. It has **no reasoning step**. 0 of 12,253 SFT outputs and 0 DPO outputs contain reasoning. It is not the inference prompt in `prompt.py`, which asks for `<reasoning>`/`<answer>`. | J/P JSON format, no reasoning, thinking off; the same format for SFT, DPO and inference. |
| **What the output contains** | Coordinates + rotation (sizes are an input) | Same: x, y, z + angle | **J outputs sizes too** (size + pos + yaw). P outputs only pos + yaw, which matches the official setup. |
| **Lengths** | — | Prompts 1,263–2,487 tokens; chosen 236–1,857 (p50 about 480–650). Over 3,200 tokens: 0 dpo1 pairs, 40 dpo2 pairs (1.0% of 3D-Front, so truncated). | max_len 8192, nothing truncated. Pilot proxy: J answer p50 275 tokens (p90 594), P p50 171 (p90 370). |
| **lr / epochs** | DPO lr 5e-7, 5 epochs, cosine | README: 5e-6, 10 epochs. No scheduler is passed, so I assume the Hugging Face default (linear, no warmup; not checked against a specific TRL version). | 009: 5e-6, 10 epochs, cosine with 5% warmup. 010: e1 = 1 epoch; rpo = 3 epochs + chosen NLL at α 1.0. |
| **β** | Not given | Not passed, so the TRL default 0.1 (sigmoid loss) | 0.1, sigmoid. Our own trainer; log-probs summed over completion tokens. |
| **LoRA** | DPO "same configuration as SFT": r16 / α32 / dropout 0.05 | r32 / α16, all 7 projections, dropout 0.05 by default | 009/010: r32 / α16 / 0.05, all 7 projections |
| **Batch** | — | per-device 1 × gradient accumulation 4. GPU count unknown; plain `python` with several GPUs would use DataParallel. | 1 × 4 GPUs (DDP) × accumulation 1 = 4 |
| **Base per stage** | Qwen3-8B; SFT then DPO1 then DPO2 | Qwen2.5-7B-Instruct (README). DPO1 trains on the merged SFT model, DPO2 on the merged DPO1 model; reference = adapter turned off. | Qwen3-8B; same chain and same reference. Our SFT: r32/α64, lr 5e-5, 1 epoch. Paper SFT: 10 epochs, lr 5e-6. Official SFT code: r16/α32, lr 5e-5, 1 epoch. |
| **Checkpoint choice** | Not stated | Saves every 100 steps, evaluates on the held-out 20% every 500 steps; the README merges `checkpoint-XXXX`, **picked by hand with no stated rule** | 009 kept **only the final step**. 010 saves every epoch and selects on a 600-room dev set. |
| **Evaluation** | OOR, FID, GPT-4o, UR (5 annotators). Table 6: OOR 0.049 → 0.048 (+reasoning) → 0.033 (+DPO1) → 0.023 (+DPO2); UR 28% → 33% → 40% → 50%. No validity/completeness metric. | No layout evaluation code; only DPO loss/accuracy on the held-out pairs | 8,647 test rooms: complete rate, geometry success, size log MAE, OOR, footprint bias; dev-set selection |
| **Runnable?** | — | The README command does not run as released: `--dataset_file` is never registered with TrlParser (main.py:49-50, and `dpo_train` reads `script_args.dataset_file`), and the README names `dpo1_prompts.json` but the file is `dpo_prompts.json`. This is from reading the code; I did not run it. | — |

## Our results (from the stage table and research log)

- **Amendment 009 DPO1 on test:**
  - J: geometry success 0.388 → 0.213, complete rate 0.997 → 0.883, size log MAE 0.286 → 0.590, footprint bias −0.083 → −0.357.
  - P: 0.390 → 0.093, complete rate 0.496.
- **Training logs:** DPO1 loss ≈ 0 from epoch 8 with reward accuracy 1.0. DPO2 also ended with loss ≈ 0, reward accuracy 1.0 and margin 13–15.
- **Amendment 009 DPO2 on dev:** complete rate J 0.000, P 0.050 (generation runs to the token cap).
- **Selected models:** J keeps SFT. P picks rpo-dpo1: dev 0.470 vs 0.408, test 0.398 vs 0.390, so only +0.009 on test.

## Differences that could explain the collapse or the small gain

Items 1–3 rest on data I measured; the mechanisms are hypotheses I have not tested.

1. **Our stage-1 negatives are much harder than the released ones.** Released HD negatives are collapsed layouts (98.1% on one line, OOR 0.381 vs 0.0005, median move 3.6 m), so they are trivially separable. Ours are the model's own plausible samples with no quality filter. Pushing down sequences the model already rates highly fits likelihood displacement, which matches loss → 0 at epoch 8 alongside invalid JSON and over-long outputs. Hypothesis.
2. **The released stage-1 signal is diluted.** 45.6% of dpo1 pairs (all of 3D-Front) are identical and give zero gradient, so the informative pairs get roughly half the effective learning rate per batch. Ours has no such dilution, and our 3D-Front positives get real negatives.
3. **J's DPO also acts on size tokens.** The official output has no sizes. Our J sizes kept shrinking (dev bias −0.087 → −0.127 / −0.211; test −0.08 → −0.36), which is why J ends up on SFT. P has no size channel, and it is the only arm where DPO helped.
4. **No early stopping in 009.** The official run keeps a checkpoint every 100 steps and picks one by hand; their delivered model may be an early one. We kept only step 5,500. Amendment 010 fixed this, and the e1/rpo results show shorter training is safer.
5. **Hyper-parameters are not what the paper says.** The paper's lr 5e-7 / 5 epochs / r16 α32 is far more conservative. We used the README values, which are 10× the lr and 2× the epochs. (Amendment 009 records that 5e-7 would be erased by the BF16 merge.)
6. **Stage-2 negatives differ.** Released: about 4 different negatives per positive, mixed translation and rotation, moves around 0.37 m, about 10% of 3D-Front pairs with no clear violation. Ours: one negative per positive, translation only, always a verified violation, so it separates easily (margin 13–15) and the pair repeats every epoch. Ours also starts from a DPO1 that had already collapsed (009) or drifted (e1/rpo). DPO2 cut the complete rate in every candidate (e.g. P e1-dpo2 0.428).
7. **The paper's positives and reasoning are not in the release.** "Unanimous human approval" can't be checked; the data has 1,977 / 2,127 unique positives, and 76% / 70% of pairs reuse an SFT target verbatim (ours 100%). There is no reasoning anywhere in SFT or DPO, so Table 6's Reasoning row cannot be reproduced from the released files. Neither our pipeline nor the released data contains the paper's actual DPO1 (human positives vs plausible SFT samples).
8. **The paper's metrics would not catch our failure.** Table 6 measures OOR and UR only, with no complete-rate or validity metric, and the official code has no generation-based evaluation. A model that breaks output format or shrinks sizes would not show up in their ablation.

**Not checkable here (no SSH):**
- What fraction of our stage-1 negatives pass the geometry check. It is in the `rejected_geometry_success` field of `dpo-001/stage1-pairs/{joint,placement}.jsonl` on the server. If it is high, our stage-1 labels are noisy.
- Token lengths of our real DPO pairs; the pilot numbers above are only a proxy.

Files are in `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/f2530c85-f966-4ee4-9e65-58f0d9e9e62c/scratchpad/optiscene-dpo/cmp_agent/`:
- `analyze.py`
- `extra.py`
- `dpo2_proxy.py`
- `ours_len.py`
- `stats.json`
- `paper_layout.txt`