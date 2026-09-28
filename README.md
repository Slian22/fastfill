# FastFill

FastFill generates an object-conditioned room layout in one model call. Given a floor polygon, optional room type and ceiling height, fixed obstacles, all objects to place and optional spatial relations, it predicts each object's position, yaw and top-surface support parent.

This is the current implementation for [Slian22/fastfill](https://github.com/Slian22/fastfill). The repository uses `main` as its development branch. Its Python package is `fastfill`; commands run from the repository root. The earlier `src/fastfill_train` Floor/Surface implementation remains in Git history and is a different protocol.

## Code and experiment versions

| Item | Version and purpose |
|---|---|
| Current implementation | The verified v3.2 data-pipeline, SFT, evaluation and service implementation, with clarified repository documentation |
| Main constraint-supervision experiment | Paired **v3 / v3.1** datasets; keep the same evaluation inputs and training configuration |
| v3.2 dataset | Separately rebuilt constrained dataset with source geometry corrections and recovered rooms |
| Model text protocol | FastFill v1: `placements`, `pos`, integer-degree training targets for `yaw`, optional `on` |

Dataset version numbers are not model architecture or training-stage numbers. v3.2 changes geometry and rows, so comparing v3 with v3.2 is not a constraint-only ablation. Existing training runs should retain their original data and code snapshots. See [version and repository map](docs/versions.md) and [known limitations](docs/known-issues.md).

## Data

Data is maintained separately at [liantian/fastfill-v3](https://huggingface.co/datasets/liantian/fastfill-v3). Repository access and upstream dataset terms continue to apply.

| Directory | Train | Dev | Test | Purpose |
|---|---:|---:|---:|---|
| `v3/` | 144,140 | 8,167 | 8,647 | No training constraints |
| `v3.1/` | 144,140 | 8,167 | 8,647 | Same rooms/targets; 42,743 training rows carry constraints |
| `v3.2/` | 144,150 | 8,167 | 8,647 | Corrected constrained dataset; 10 added train rows, 25 changed train rows and 1 changed test row |

The dataset repository also preserves `code/` for v3/v3.1 and `code_v3.2/` for v3.2. This Git repository contains implementation and tests; raw scenes, meshes, weights and generated training JSONLs are not checked in.

## Install and verify

Install a PyTorch build suitable for your CPU/CUDA environment first. Use one compatible environment for training, merging and serving.

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -B -m pytest -q -p no:cacheprovider fastfill/tests
```

The offline regression suite covers geometry conversion, build transactions, QA, runtime validation, and a tiny real CPU LoRA merge. It does not replace GPU training, model-quality evaluation, deployment testing or mesh-level validation. vLLM is optional for evaluation and required by the provided service.

The optional `fastfill.tools.visualize` utility also requires `pip install matplotlib`.

## Train and evaluate

```bash
# Choose v3 OR v3.1 for the paired experiment; use a separate run for v3.2.
hf download liantian/fastfill-v3 --repo-type dataset \
  --include 'v3/*' --include 'v3.1/*' --include 'v3.2/*' --local-dir data

python -m fastfill.train --model /path/to/Qwen3-8B \
  --data data/v3.1 --out outputs/ff-v3.1 --dry_run

python -m fastfill.train --model /path/to/Qwen3-8B \
  --data data/v3.1 --out outputs/ff-v3.1

python -m fastfill.merge_lora --base_model_path /path/to/Qwen3-8B \
  --lora_path outputs/ff-v3.1/final --output_path outputs/ff-v3.1-merged

python -m fastfill.evaluate --model outputs/ff-v3.1-merged \
  --rooms data/v3.1/test_constrained_rooms.jsonl --out eval/ff-v3.1-constraints
```

Select memory, batch size and training budget using the actual tokenizer profile and target hardware. Report both original and deterministically repaired outputs, completeness, constraint satisfaction and collision metrics. `valid` and an accepted HTTP response do not guarantee a collision-free or physically stable scene.

## Implementation

| Path | Responsibility |
|---|---|
| `fastfill/adapters/` | Source-to-canonical-IR conversions |
| `fastfill/build.py`, `split.py`, `anchors.py` | Dataset preparation, leakage groups and support handling |
| `fastfill/scene.py` | Shared text protocol, coordinate conventions and parsing |
| `fastfill/train.py`, `merge_lora.py` | LoRA SFT and adapter merging |
| `fastfill/evaluate.py`, `validate.py` | Model and reference scoring |
| `fastfill/interface.py`, `serve.py` | WorldEdge request conversion and HTTP service |
| `fastfill/tools/`, `fastfill/tests/` | Data QA, ablation checks and regressions |

See the [technical documentation](fastfill/README.md) and [v3.2 release notes](fastfill/RELEASE_v3.2.md). The implementation follows the object-conditioned layout-learning approach of [OptiScene](https://github.com/PolySummit/OptiScene). Its original `main.py` / DPO scripts and schema are not entry points for this package; preference training requires data in a consistent FastFill protocol.
