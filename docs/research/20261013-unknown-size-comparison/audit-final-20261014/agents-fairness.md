**Delivery-package audit: P-selected (rpo-dpo1), delivery-010-001**

No BLOCKER. The hash chain is internally consistent and every number I recomputed matches the report. The problems are in what the card tells a downstream user to run, and in what the package leaves out.

**Recomputed and consistent (no action)**
- **Merged model hash:** `model_sha256` bbbb2c… matches the hash of `model.files` in delivery.json and of the `model/*` lines in SHA256SUMS.
- **Adapter hash:** the `adapter/*` lines hash to 306d1a…, which matches `evaluated_policy.adapter_sha256`.
- **Training base hash:** the `training-base/*` lines hash to 272c5a…. That matches `evaluated_policy.training_base_sha256` and the P sft `model_sha256` in stage_lineage.csv.
- **Lineage receipts:** 245dab… (sft-merged) and a6d383… (dpo1-merged) match stage_lineage.csv. The delivered model's `lineage_receipt_sha256` is a6d383….
- **SHA256SUMS:** its 40 lines match `files_to_copy` exactly, in the same order.
- **MANIFEST.json:** the hashes of MODEL_CARD.md, SHA256SUMS and delivery.json are correct.
- **Local copies:** all 16 `report-selected/*` files and `dev-selection/selection.json` match SHA256SUMS against the local copies.
- **Size prior:** `size-priors.json` is listed with hash 073522… (`delivery.json.size_prior.sha256`), under key `full_prior`.
- **Inference run:** `inference_run_sha256` 8a0e46… matches the report's MANIFEST `input_sha256.placement`.
- **Dev selection:** recomputed from selection.json, the eligible set is {sft, e1-dpo1, rpo-dpo1} with best a′ 0.435, so rpo-dpo1 is selected.
- **Delivery rule:** recomputed (Δb CI [+0.0007, +0.0087] is not below zero; a′ CI lower bound −0.071 < −0.02), so P is delivered.
- **Card cells:** T1, T1b, T6, T1_bias, T2-all and S1 all match results.json, including Holm 0.042 / 0.095.
- **Fallback size:** the "(0.40, 0.50, 0.65) fallback" matches the frozen prior. S2.csv has 8,238 objects with exactly that triple.
- **Adapter precision:** PEFT's default `autocast_adapter_dtype=True` gives the same FP32 adapter as the evaluation (infer.py:289-290).
- **Binding seed:** the default `bind_room` seed equals the evaluation SEED (eval_data.py:22).

**Defects**

1. **MAJOR: the card leads with the unevaluated merged weights and over-states what the merge kept.**
   - MODEL_CARD.md:22 names the merged `model/` as the "Delivered weights". The first snippet (:56-58) and the vLLM snippet (:80) both load it.
   - The merge changed only 0.0675 of the targeted weights (5.4–9.2 % per module, stage_table.json). The SFT merge changed 0.776.
   - Amendment 009 §Why itself says updates below half a BF16 ULP are erased by the merge. So the merged bytes may have lost much of the rpo-dpo1 gain, which is the only reason it beat SFT on dev (a′ 0.435 vs 0.382). Its behaviour has never been measured.
   - :240 says the merged model "folds that update into BF16 weights"; most of the update was rounded away. findings.md:50 ("11–19 % changed") also does not describe the delivered model.
   - Fix (delivery.py:185-190, 236-253, 329-336): make `training-base/` + `adapter/` the primary snippet. Label `model/` "unevaluated BF16 merge (6.75 % of target weights changed)". Optionally score `model/` on the 600 dev rooms; that needs your go-ahead before anything runs.

2. **MAJOR: the P request snippet fails on a real room.** MODEL_CARD.md:93,97; contract.py:123-125.
   - `bind_room` and `messages()` require ground-truth `size`, `pos` and `yaw` for every movable object, plus `source` and `uid`.
   - I reproduced it: a user room with only `id` and `category` raises `ValueError: bed.size: expected 3 numbers`.
   - Placeholder values (size [1,1,1], pos [0,0,0], yaw 0) work and do not change the prompt. That is how I tested it.
   - Fix: document that placeholder rule, or add a request builder that does not need labels.

3. **MAJOR: no code or library versions are pinned.**
   - The P snippet imports `fastfill.comparison.contract`, `fastfill.comparison.size_prior` and `fastfill.scene` (which needs shapely). The JSON key order, cm rounding, category and room-type normalisation, and HMAC object order all come from that code.
   - None of it is in the package. delivery.json has no git commit, no `code_sha256` for these files, and no torch/transformers/peft versions (the inference receipt's `versions` are not copied).
   - Current hashes: contract.py ef1e2cd9…, size_prior.py e7a546b8…, scene.py 91712134…. These files have not changed since commit 26a841a3c.
   - Fix (delivery.py:125-150): record the commit, these three file hashes and the inference receipt's versions in delivery.json. Ideally also copy the three files into the package.

4. **MINOR: the snippet drops the ID mapping and does not say how to assemble the output.** MODEL_CARD.md:93.
   - `bind_room(...)[0]` throws away the sidecar, so the opaque `obj_000N`/`fixed_000N` IDs (and the `on` support references) cannot be mapped back to the user's objects.
   - The card also never says that P outputs no sizes: the request sizes must be attached to each placement.
   - Fix: `room, sidecar = bind_room(...)`, map via `sidecar['bound_to_original']`, and attach `sizes[id]`.

5. **MINOR: the stated output cap is not the one the evaluation used.**
   - MODEL_CARD.md:51 and :63 give a per-request cap: min(40959, 40960 − own tokens, 1024 + 128 × own objects).
   - The evaluation applied one shared cap per batch: min(40959, 40960 − W, 1024 + 128 × max objects in the batch), where W is the shared padded width (infer_shard.py:76-81, infer.py:218-221).
   - The snippet's cap can therefore be smaller. The practical effect for P is near zero (T6 shows 0 P outputs hit the cap).
   - Fix: describe it as the batch cap, and say that per-request use is an approximation.

6. **MINOR: the snippets will not reproduce the evaluated outputs exactly.**
   - **Stop token:** the evaluation stopped only at `<|im_end|>` (infer.py:282, 303). The snippet uses `generation_config.json`, which for Qwen3 also stops at `<|endoftext|>`.
   - **Parsing:** the evaluation counted an answer as complete only after a strict parse (`h.parse_prediction` + contract). The snippet uses a bare `json.loads` (:65).
   - **Batching:** the evaluation ran batches of 8 with left padding on a single GPU. The snippet runs batch 1 with `device_map='auto'`. Greedy BF16 outputs can differ, yet :68 says the snippet gives "the evaluated policy itself (the numbers above)".
   - **Ordering:** :59 uses `request` before :97 defines it. If the transformers snippet then reruns, its `messages = …` replaces the imported `messages()` function, so a loop breaks on the second room.
   - Fix: pass `eos_token_id=tok.convert_tokens_to_ids('<|im_end|>')`, call `validate_output(raw, room, 'placement')`, state the batching caveat, and reorder the snippets with the list renamed to `msgs`.

7. **MINOR: the package is not self-contained and has no assembly instructions.**
   - delivery-010-001 holds only 4 metadata files; the weights stay at the `/lus` paths. Even so, the status is `DELIVERY_PACKAGE_READY`.
   - The card never mentions `files_to_copy` or `sha256sum -c SHA256SUMS`.
   - Fix: add a "Package assembly" step (copy each source to its dest, then check), and/or change the status to "manifest ready".

8. **MINOR: the size prior's provenance cannot be checked from the package.**
   - The prior hash 073522… is tied to the evaluation only through `eval-data-v1/MANIFEST.json` (`input_consumed_sha256`), which is not shipped. The report MANIFEST only gives that manifest's own hash (6f9d80…).
   - The inference `run_receipt.json` is not shipped either.
   - Fix: add both files to `files_to_copy`.

9. **MINOR: the card does not disclose P's weak spot.**
   - 1,403 of 8,647 test rooms (16.2 %) contain at least one object sized by the global fallback. That covers 8,238 of 108,253 objects (7.6 %), and their P size log MAE is 1.229 (S2.csv, S1).
   - Fix: add one line to Limitations.

10. **MINOR: a′ is not a number a user can get.** MODEL_CARD.md:5, :110. a′ rescales sizes by a footprint bias measured against the test set's true sizes, which a deployed model cannot know. The card should say it is a diagnostic used only for the delivery rule.

11. **MINOR: the RPO weight is missing from the hyper-parameter table.** MODEL_CARD.md:29 claims to quote the lineage receipt but omits `rpo_alpha`, the setting that defines the selected candidate (`α 1.0` appears only in hard-coded prose at :24). Fix: add `rpo_alpha` to the key list at delivery.py:205-206.

12. **MINOR: the missing agent is not disclosed.**
    - MODEL_CARD.md:232 lists 8 agents and delivery.json has `agents.excluded = {}`.
    - O5 (amendment 007) was never run (research-state: "O5 NOT RUN … never substituted"). It was left out of the final job's agent list, so the "exclusions stated" mechanism never recorded it.
    - Fix: pass O5 as excluded with its reason, or add one line to the card.

**What a downstream user needs to run P**
- **Weights:**
  - To reproduce the evaluated results: `training-base/` (6 files) plus `adapter/` (7 files).
  - Or `model/` (7 files), which is unevaluated.
- **Size prior:** `size-priors.json`, using key `full_prior`.
- **Code, not shipped:** `fastfill/comparison/contract.py`, `fastfill/comparison/size_prior.py` and `fastfill/scene.py`, plus shapely.
- **Libraries, versions not recorded:** torch, transformers (must load Qwen3 and `chat_template.jinja`), peft.

Steps:
1. Copy each `files_to_copy` source to its dest, then run `sha256sum -c SHA256SUMS`.
2. Build a prepared room with:
   - `source`, `uid`, a valid counter-clockwise `boundary`, and `boundary_type`;
   - optional `height` and `room_type`;
   - `fixed` items with id, category, size, pos, yaw and optional parent;
   - `objects` with id and category, plus placeholder size, pos and yaw.
3. Run `room, sidecar = bind_room(prepared)`.
4. Compute sizes with `predict_sizes({room_type, [{id, category}]}, full_prior)`, rounding each value to 2 decimals with a 0.01 floor.
5. Build the prompt with `messages(room, 'placement', estimated_sizes=sizes)[:2]`.
6. Apply the chat template with `enable_thinking=False` and tokenize with `add_special_tokens=False`. Reject prompts over 40959 tokens.
7. Load the training base in BF16 and `PeftModel.from_pretrained(base, 'adapter')`, so the adapter stays FP32. Use sdpa attention.
8. Generate greedily, stopping at `<|im_end|>`, with cap = min(40959, 40960 − prompt length, 1024 + 128 × objects).
9. Treat reaching the cap without an end token as a failure. Then run `validate_output(raw, room, 'placement')`.
10. Map the IDs and `on` references back through the sidecar, and attach `sizes[id]`. Positions are bottom-centre in metres; yaw is integer degrees counter-clockwise from +X.

Missing from the package or card: the code and its pin, library versions, the placeholder rule, the sidecar mapping, the size-attachment step, output validation, and the assembly instructions.

Files reviewed:
- /Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/results/delivery-010-001/{MODEL_CARD.md, delivery.json, SHA256SUMS, MANIFEST.json}
- /Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/comparison/{delivery.py, contract.py, size_prior.py, eval_data.py, infer.py, infer_shard.py, stage_table.py, dev_select.py}