# Isambard unknown-size J/P preflight — 2026-10-13

Read-only observations via `u6uv.aip2.isambard`, 00:13–00:16 UTC; login host `login41`, user `shanliantian.u6uv`. No jobs submitted/cancelled/modified, downloads, installations, credential changes, or intentional remote file writes. The only deliverable written is this local report.

**Result:** existing environment and generic upstream Qwen3-8B cache are suitable candidates for the requested **one-GPU, ≤30-minute sequential J then P pilot**. Both arms should start independently from that same generic checkpoint, run about 8 optimizer steps and 2 dev generations each. Parent supplies ≤64 train/source + 4 dev/source with original IDs and frozen IR. GPU execution, parent data, and the new J/P runner remain unvalidated.

## Exact paths and environment

All server paths below were observed, unless explicitly marked pending.

| Item | Validated path / fact |
|---|---|
| Project, ROOT | `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill` |
| Generic model, BASE | `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/models/Qwen3-8B` |
| Training Python | `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/env/bin/python` → `python3.12` |
| torchrun | `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/env/bin/torchrun` (unneeded for one process) |
| Existing log directory | `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard` |
| Environment receipt | `ROOT/runs/isambard/env-freeze.txt` |
| CUDA module | `cudatoolkit/24.11_12.6`; module maps CUDA to `/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6` |
| Existing HF cache settings | `HF_HOME=/scratch/u6uv/shanliantian.u6uv/.cache/huggingface`; `HF_HUB_CACHE=$HF_HOME/hub` |

Live read-only package inspection: aarch64, Python **3.12.15**, torch **2.13.0+cu126**, transformers **5.14.1**, peft **0.21.0**, accelerate **1.15.0**, tokenizers **0.22.2**, huggingface_hub **1.33.0**, safetensors **0.8.0**, numpy **2.4.6**. Torch/Transformers/PEFT/Accelerate imports passed, with torch built for CUDA 12.6. Offline `AutoConfig`, `AutoTokenizer`, `enable_thinking=False` chat-template rendering, and native `Qwen3ForCausalLM` import passed. Model weights were not loaded on the login node.

`datasets`, `trl`, `bitsandbytes`, and `flash_attn` are absent. Use native Transformers + PEFT, BF16 and SDPA; do not rely on those missing packages. Separate vLLM Python exists at `/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python`, but was not import-tested and is unnecessary for four total dev generations.

## Generic base provenance and integrity

- Repository: **`Qwen/Qwen3-8B`**, revision **`b968826d9c46dd6066d109eabc6255188de91218`**. This is upstream Qwen3-8B, not the separately named Qwen3-8B-Base and not liantian fine-tuned weights.
- Provenance: `ROOT/fastfill/v2/ops/isambard_setup.sbatch:24` explicitly pins that repository/revision/local directory; cached model README agrees.
- Tree receipt: `BASE/.cache/huggingface/trees/b968826d9c46dd6066d109eabc6255188de91218.json`.
- Per-file receipts: `BASE/.cache/huggingface/download/<filename>.metadata`; every observed receipt names the same revision.
- All five indexed shards exist, have receipt-matching lengths, and pass bounded safetensors header/end-offset checks (399 tensors total). Shard mtimes precede their 2026-10-08 download receipts.
- `BASE/config.json` SHA-256 freshly verified: `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30`.
- `BASE/model.safetensors.index.json` SHA-256: `f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc`.
- Config, index, tokenizer config, and generation config Git-blob hashes also match their download receipts.

Receipt LFS SHA-256 values below are **recorded expectations, not freshly recomputed payload hashes**:

| Shard (`model-NNNNN-of-00005.safetensors`) | Bytes observed | Receipt SHA-256 |
|---|---:|---|
| 00001 | 3996250744 | `31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f` |
| 00002 | 3993160032 | `5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282` |
| 00003 | 3959604768 | `c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836` |
| 00004 | 3187841392 | `b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a` |
| 00005 | 1244659840 | `20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff` |

The normal scratch HF hub has no `models--Qwen--Qwen3-8B` directory. **Use the absolute BASE path**, not a hub ID. No upstream freshness query or 16.4 GB payload rehash was performed. “Fresh” here means a new arm initialized from this cached generic checkpoint; it does not mean today's latest upstream revision.

## Scheduler and protected work

- Cluster `gracehopper`; account **`brics.u6uv`**; user QoS **`normal`**; partition **`workq`**, up/default.
- Partition QoS `workq_qos`: **24-hour max walltime**, max 256 running / 512 submitted jobs per user, `DenyOnLimit`. The partition's displayed unlimited MaxTime does not remove its QoS cap.
- `workq` has 1,320 nodes, 4 GPUs/node; inspected node features include **NVIDIA_GH200, ARM**. Node resources: 288 CPUs / 460000M RAM. An actual mixed node showed an allocation of **1 GPU / 72 CPUs / 115000M**, confirming fractional-node GPU use.
- Defaults per GPU: 72 CPUs / 115000M. Proposed pilot: 1 node, 1 task, **1 GPU, 8 requested CPUs, 64G host RAM, 00:30:00**, no exclusivity or requeue. CPU/socket rounding and billing may exceed the requested CPU count.
- Maximum requested GPU time: **0.5 GPU-hour for both arms together**, not per arm. Monetary allocation balance/remaining credits were not exposed by inspected association/share fields; blank limits are not proof of unlimited budget.
- Queue is changing: initial snapshot had 2 idle and 292 mixed nodes; inspected previously idle node became mixed. No start-time or free-GPU guarantee, reservation, or scheduler test submission was made.

User jobs observed unchanged through 00:15:38 UTC:

| Job | Name | State / GPU request |
|---|---|---|
| 7354342 | ph-train-r4-repaired-20261011-step12000 | RUNNING, 4 GPUs, nid010551 |
| 7354344 | ph-train-r4-repaired-20261011-step16000 | PENDING Dependency, 4 GPUs |
| 7354347 | ph-train-r4-repaired-20261011-step20000 | PENDING Dependency, 4 GPUs |
| 7354353 | ph-heldout-eval | PENDING Dependency, 1 GPU |

All four use workq / brics.u6uv / normal and 23-hour limits. Do not reuse their outputs, attach to their allocations, alter dependencies, or cancel them.

## Pilot launch envelope — proposal only, not executed

The following allocation/environment is grounded in observed server settings. Parent must first stage and validate its isolated code, frozen data manifest, and sequential driver. **No exact J/P driver path or CLI has been validated yet.** Save this as a new wrapper only during the subsequent deployment task, then invoke `sbatch wrapper.sbatch DRIVER.py [parent-validated arguments...]` with `PILOT_CODE_ROOT` pointing at the isolated staged code.

```bash
#!/bin/bash
#SBATCH --job-name=agency-unknownsize-pilot
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/agency-unknownsize-%j.out
set -euo pipefail
ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${PILOT_CODE_ROOT:?parent must stage and verify isolated code}"
test -d "$PILOT_CODE_ROOT"
test "$#" -ge 1
test -r "$1"                       # absolute reviewed Python driver path
module load cudatoolkit/24.11_12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH="$CUDA_HOME"
export PATH="$ROOT/env/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PILOT_CODE_ROOT"
export BASE_MODEL="$ROOT/models/Qwen3-8B"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/tmp PYTHONUNBUFFERED=1
cd "$PILOT_CODE_ROOT"
# Preserve Slurm's CUDA_VISIBLE_DEVICES; driver asserts exactly one visible GPU.
# One shared timeout covers J training+dev, then independent P training+dev.
exec timeout --signal=TERM --kill-after=30s 27m   "$ROOT/env/bin/python" -B -u "$@"
```

Driver recommendations: explicit `local_files_only=True` / `trust_remote_code=False`; independent fresh LoRA initialization for J and P; BF16/SDPA, batch 1, gradient checkpointing, explicit 8-optimizer-step cap (not epochs or microsteps); bounded sequence length and generation tokens; exactly 2 dev generations per arm. P's size estimator fits train only. Bind both arms to the parent's original-ID frozen-IR manifest. Keep separate new outputs, unload J before loading P, and log arm/global-step, finite loss, elapsed time, peak GPU memory, dev IDs, and termination status. Fail on OOM/nonfinite loss; no resubmit, model publishing, or writes into existing checkpoints. A timeout is incomplete, not a successful pilot.

**Existing-template pitfalls:** `ROOT/fastfill/v1_1/isambard_v1_1_pipeline.sbatch` requests 4 GPUs/6 hours, uses `models/OptiScene-Qwen-8B-Final`, and contains install/download stages. `ROOT/fastfill/v2/ops/isambard_submit.sh` submits multiple jobs; `isambard_autorun.sbatch` requests 4 GPUs/24 hours. None is suitable unchanged. Frozen clean-DPO template exists at `ROOT/automation/v11-clean-20261011-3e3eef72c515/fastfill/v1_1/isambard_v1_1_clean_dpo.sbatch`, but also uses 4 GPUs and old SFT weights.

The observed `ROOT/fastfill/train.py` (SHA-256 `a208244735cbe2ad3297914b888df793d3319405136b763eb3eec7d3d2d2aa74`) supports `--mem_test`, BF16/SDPA and LoRA, but **has no `--max_steps` or dev-generation CLI** and eagerly reads whole train/dev files. It cannot enforce the requested J/P pilot unchanged. Old `data/v1.1/train.jsonl` is 551 MB and is not the parent's bounded contract.

**Remaining uncertainty:** GPU driver/runtime compatibility, actual peak memory and completion time, immutable weight-byte integrity beyond receipts/header checks, final parent driver/data hashes, and future allocation availability. No paper research or project-code changes were performed.

