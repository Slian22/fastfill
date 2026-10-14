# Amendment 014 — official OptiScene code recipe reproduced on Isambard and evaluated with our checker (EXPLORATORY)

Locked 2026-10-14 ~14:30 UTC, before any training or generation of this amendment exists. Requested by the user ("按照他们官方流程弄他们的数据集，在我们这测"). Separate from amendments 005–013: it changes no earlier result or decision. Tooling: `fastfill/comparison/optiscene_official/` (vendored upstream commit 409f1317, every fix listed in VENDORED.md / UPSTREAM.json), `optiscene_eval.py`, `setup_optiscene_env.sh`, `submit_014.sh` (commit cf518a24b, independently reviewed).

## 1. What is reproduced

- **Label:** every number of this amendment is labelled "OptiScene code recipe (README, Qwen2.5-7B-Instruct)". It is neither the paper's recipe (Qwen3-8B, lr 5e-7) nor the authors' weights, which are unreleased.
- **Data:** their released files `sft_prompts.json` (12,253), `dpo_prompts.json` (7,951 pairs) and `dpo2_prompts.json` (8,410 pairs), with their 80/20 split (`train_test_split(0.2, seed 42)`). Known data defects are kept as released and documented: 45.6 % of DPO1 pairs have identical chosen/rejected texts; there are odd Holodeck rejected inputs; and there are areas of 0.
- **Recipe (README):**
  - Base: Qwen2.5-7B-Instruct, at a revision pinned at download.
  - SFT: 1 epoch, lr 5e-5, per-device batch 8, LoRA r16/α32, max_seq 2048.
  - DPO1 and DPO2: 10 epochs each, lr 5e-6, LoRA r32/α16, β 0.1 (TRL default), max_length 3200.
  - A merge follows each stage. Training runs on one GPU per job, with no DataParallel.
- **Fixes:** only those needed to run, each documented in VENDORED.md. They are the dataset filename, the unregistered `--dataset_file` flag, gradient checkpointing for SFT memory, and resume across 24 h jobs.
- **Checkpoints:** the last checkpoint of each stage. Theirs was chosen by hand with no stated rule.

## 2. Evaluation

**E1 (their data)**
- **E1-unseen:** 3D-SynthPlace scenes not covered (≥ 50 % on description and position at 0.1 m) by any SFT or DPO training row, and whose rendered input is no training input. Selected by a stable hash, capped at the paper's per-stratum counts. Shortfalls are recorded and never filled; plan-e1 recorded one for 3D-FRONT bedroom (216 of 423).
- **E1-heldout20:** their own 20 % SFT eval rows not covered by the 80 % training rows.
- **Metrics:**
  - the paper's OOR (App. D.2, axis-aligned);
  - parse rate;
  - our frozen checker after the documented frame and unit conversion. The front convention is valid for Holodeck only; 3D-FRONT yaw is not used for orientation analysis.

**E2 (our data)**
- **Rooms:** our test rooms of the six paper strata (eval-data-v1; counts = the frozen GPT-selection populations), converted into their input format.
- **Size modes:**
  - `gt`: GT sizes, as their method assumes retrieved assets;
  - `prior`: our train-prior sizes, for a like-for-like comparison with our delivered P.
- **Scoring:** outputs are converted back to our IR and scored with `geometry.score`, `downstream.accept`, OOR and our OOR. A training-exposure audit marks E2 rooms whose 3D-SynthPlace source scene is covered by their training data; rooms seen in training are reported separately.

**Systems**
- **Stages:** the reproduced SFT, DPO1 and DPO2.
- **Reference checkpoint:** the existing `models/OptiScene-Qwen-8B-Final`. Its provenance is unknown and its training exposure is not audited, so it is reported as "provenance unknown".
- **Comparison with our P:** our delivered P (rpo-dpo1, exact policy) on the same E2 rooms from its existing test outputs (geometry-v5). Paired OptiScene(prior) − P differences use house-cluster bootstrap CIs (EXPLORATORY).

**Decoding:** vLLM greedy, seed 0, the model directory's chat template; OptiScene max_tokens 2048. The README samples instead; this deviation is recorded in each generation MANIFEST.

## 3. Budget and limits

- **GPU cap:** 110 GPU-hours; expected 50–80.
- **Visual metrics** (FID/GPT judge) for E1 are not part of this amendment, since 3D-SynthPlace rendering needs their assets. E2 rooms can be added to the visual run later by a separate amendment.
- **Failure handling:** a stage that does not complete within two 24 h jobs is reported as incomplete; it is never substituted.
