# Audit-pairs full-regression fixture repair

The first full repair regression reported **3,500 passed, 2 failed, 6 skipped and 305 subtests passed**. Both failures were the two variants of `test_dpo_refuses_all_filtered_pairs_before_loading_distributed_model`, for `over_length` and `prefix_mismatch`.

## Cause and bounded change

The test simulates rank zero of a two-rank run by mocking `init_process_group` and CUDA operations. F05 now validates the input identity and broadcasts any identity error before tokenization. The older fixture still passed `--model unused`, without identity assets, and did not mock `broadcast_object_list`. Its missing process group error therefore masked the earlier missing-model identity error, preventing the test from reaching its intended empty-tokenized-dataset rejection.

Only `fastfill/tests/test_v1_1_audit_pairs.py` changed:

1. Supply local minimal configuration/tokenizer files and a valid one-tensor safetensors file so the real identity validator can execute.
2. Replace the distributed broadcast with a spy alongside the existing fake process-group initialization; assert that it receives `[None]` from rank zero, showing identity validation succeeded.
3. Retain the expected `No valid DPO pairs` error and forbidden model-loader assertion, and verify the generated identity records world size two.

Neither `bind_trainer_identity` nor the tokenization filter is mocked. The synthetic weights are only hashed and structurally checked; the existing forbidden model-loader assertion still prevents loading them. Production code and training semantics are unchanged. This is CPU fixture coverage, not execution of an actual multi-rank NCCL process group.

## RED and GREEN

[RED log](audit-pairs-red.log): **2 failed in 12.41 seconds**, reproduced before changing the fixture.

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/slian/miniforge3/bin/python -B -m pytest -p no:cacheprovider --import-mode=importlib -q \
  fastfill/tests/test_v1_1_audit_pairs.py::test_dpo_refuses_all_filtered_pairs_before_loading_distributed_model
```

[GREEN log](audit-pairs-green.log): **94 passed in 7.93 seconds**, including the complete affected module and the training-identity/checkpoint regression modules.

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/slian/miniforge3/bin/python -B -m pytest -p no:cacheprovider --import-mode=importlib -q \
  fastfill/tests/test_v1_1_audit_pairs.py \
  fastfill/v1_1/tests/test_training_guard.py \
  fastfill/v1_1/tests/test_dpo_checkpoint_recovery.py
```

The scoped `git diff --check` passes. Comparison with the existing 494-file `source-snapshot-manifest.json` identifies only this test file as changed at this repair checkpoint; the root agent owns updating the final snapshot and full-suite evidence.

| File | SHA-256 after fixture repair |
|---|---|
| `fastfill/tests/test_v1_1_audit_pairs.py` | `70a7b1f4bc9d3222c1f0eb44f46e1565c9e10034d342fdff89217322853c4fe9` |
| `fastfill/v1_1/train_dpo.py` (unchanged) | `979a44c18463e3af364ecf7976a10009ce6e80564b34af29d899a7872cddc723` |
| `fastfill/v1_1/training_guard.py` (unchanged) | `e76663854bcd3570c6e48cc92a0352d26cd1f006f8fd489eaaf8fdad3675beb5` |

No SSH, external API, GPU execution, training, dependency change, commit or push was performed by this repair worker.
