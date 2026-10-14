# Vendored OptiScene code (amendment 014)

Copied from [PolySummit/OptiScene](https://github.com/PolySummit/OptiScene) at commit `409f1317e1a8afe852f4cdc99dfefaed30aeb425` (local read-only clone:
`OptiScene_Original/`). Machine-readable record: `UPSTREAM.json`. Run every task with this directory as the working
directory (`python main.py --task ...`), as its imports (`scripts.*`, `utils`, `prompt`) require. Results are labelled
"OptiScene code recipe (README, Qwen2.5-7B-Instruct)": not the paper's recipe and not the authors' weights.

## Files

| File | Upstream sha256 | Vendored sha256 | Fixes |
|---|---|---|---|
| `main.py` | `3e5a46ca05e1623a47f93cb3e7cb0735f30b0e3e9eb5eab2b6b16ea5f035bea1` | `c808d2967f5097fbec7ddf80da1ae0cfebf7f9534e97d12b5b9b336b41f69378` | F2 |
| `utils.py` | `09b46eb14a264e68e65b42b07320bd77326efadd4e082c41875e5d958254846f` | `09b46eb14a264e68e65b42b07320bd77326efadd4e082c41875e5d958254846f` | none |
| `prompt.py` | `dff6e81dfe8013975f29722780759ba3cff9fcc93e917b862ca8cc7cfd63eb56` | `dff6e81dfe8013975f29722780759ba3cff9fcc93e917b862ca8cc7cfd63eb56` | none |
| `default_prompt.txt` | `39cd6c2331bc74b76bec7e87e62365e5d86b1947b9a5660c22fed0a92444ad55` | `39cd6c2331bc74b76bec7e87e62365e5d86b1947b9a5660c22fed0a92444ad55` | none |
| `environment.yml` | `490ed14240ee7b21bbc46124dac2eceefd59b714874f37722f6b0f96565ce950` | `490ed14240ee7b21bbc46124dac2eceefd59b714874f37722f6b0f96565ce950` | none |
| `README.md` | `4a489c398b28745793d4b729a55703cd24c4c3e3cf109d36b8fd9e6dfe1c32f7` | `4a489c398b28745793d4b729a55703cd24c4c3e3cf109d36b8fd9e6dfe1c32f7` | none |
| `scripts/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | none |
| `scripts/dpo_train.py` | `0cd3668d4079890ad4be61f97a1bbde1254f7298bf263b219ab02ee4f271c73c` | `34ba07e471a66ab95eb62a5d437d9fa6d224b8fd1465a18d471a9d7b66cee360` | F3 |
| `scripts/sft_train.py` | `a3539ea3eb356228eddab5845034c8836e220f5387306b93b437b3e36086a2c1` | `342b854a642e58bdebe62c2048e555871261615c33443359bed8d5e68d261450` | F5 |
| `scripts/merge_lora.py` | `0aa8bc3c60b056964ee1f999446dcb2cac38dd3cd8548b66f1a0d46704549ad6` | `0aa8bc3c60b056964ee1f999446dcb2cac38dd3cd8548b66f1a0d46704549ad6` | none |
| `scripts/inference.py` | `36c259b1d7f8ea6615482116d9adff61d0b211208ae48765b6a8d3d313966c19` | `36c259b1d7f8ea6615482116d9adff61d0b211208ae48765b6a8d3d313966c19` | none |

`dataset/*.json` are not vendored; jobs read them from `$PROJECT/data/optiscene-official/` and check the sha256
prefixes `12b3ccc10e8e` (sft_prompts.json), `0d331989` (dpo_prompts.json), `ed6c1bae` (dpo2_prompts.json).

## Fixes (complete list)

- **F1 (command line only, no file change):** the README passes `dataset/dpo1_prompts.json`; the released file is
  `dpo_prompts.json`. `optiscene_eval.dpo_argv` passes the released name.
- **F2 (`main.py`):** the DPO task fails on the README command twice: `--dataset_file` is never registered with the
  `TrlParser` that `main.py` builds, and TRL 0.15.2 declares `ScriptArguments.dataset_name: str` without a default.
  Fix: one dataclass `OptiSceneScriptArguments(ScriptArguments)` with `dataset_name: str = "local-json"` (unused) and
  `dataset_file: str = None`, used in `TrlParser((OptiSceneScriptArguments, DPOConfig, ModelConfig))`.
- **F3 (`scripts/dpo_train.py`):** `trainer.train(resume_from_checkpoint=get_last_checkpoint(training_args.output_dir))`
  (plus its import) so a DPO stage continues across 24 h jobs. In a fresh output directory this is `None`, so
  behaviour is unchanged. `optiscene_eval train` creates the output directory first (`get_last_checkpoint` lists it).
- **Operational command-line additions (DPO), no effect on optimisation:** `--save_total_limit 2`, `--report_to none`.
- **F4 (conditional, not applied):** if `env-check` cannot import `modelscope`, replace `from modelscope import` with
  `from transformers import` in `scripts/sft_train.py`, `scripts/merge_lora.py` and `scripts/inference.py`, and update
  this file and `UPSTREAM.json`.
- **F5 (`scripts/sft_train.py`, one line):** `gradient_checkpointing=True` in `SFTConfig`. The README SFT (batch 8 x
  2048 tokens, no checkpointing, LoRA dropout on 7 modules) needs roughly 150 GB of activations plus fp32 logits on one
  95 GiB GH200, so it would run out of memory in its first steps. Memory only: TRL 0.15.2 calls
  `enable_input_require_grads` for reentrant checkpointing and `torch.utils.checkpoint` restores the RNG state, so the
  dropout masks and gradients are unchanged (a tiny Qwen2 SFT on CPU gives a byte-identical adapter with and without it).

## Kept as released (recorded in every training receipt)

- `main.py` remaps `--eval_strategy` to `--evaluation_strategy`; this still works on transformers 4.49.
- The DPO policy loads in fp32 (`ModelConfig.torch_dtype` defaults to None; `--bf16` only enables autocast).
- DPO: linear learning-rate schedule, no warmup, beta 0.1 (DPOConfig defaults).
- SFT: `max_seq_length` 2048 (truncates about 3,965 long rows), cosine schedule with warmup 0.1 (main.py defaults);
  `.to("cuda")` on a single GPU; no `save_model`, so `checkpoint-1226` comes from `--save_strategy=epoch`.
- SFT: `pad_token = eos_token` (`<|im_end|>`), so the collator masks every `<|im_end|>` out of the labels; the loss
  covers the whole sequence, prompt included (no completion-only collator).
- DPO: `get_bin_dataset` pairs the chosen row's prompt with both outputs. The rejected row's own input differs from it
  in 4,325 of 7,951 DPO1 pairs (4,317 box sizes only, 8 object groups) and 813 of 8,410 DPO2 pairs (box sizes only), so
  those rejected layouts were made for other sizes; the trained pairs themselves carry no size change.
- Every stage uses the scripts' own `train_test_split(test_size=0.2, seed=42)`.

Not kept: the README inference samples with the model's `generation_config` (temperature 0.7, top-p 0.8, top-k 20,
repetition penalty 1.05); amendment 014 decodes greedily (temperature 0, seed 0, vLLM), recorded in every generate
MANIFEST.
