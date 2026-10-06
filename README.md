# FastFill

FastFill v2 predicts one local bbox size, bottom-center position and yaw for each
requested object from **room type + room size + furniture list**. It exports bbox
scene JSON, a RoomGenBench handoff and colored GLB/SVG without retrieving assets.
See the [direct bbox guide](docs/fastfill-v2-direct-bbox.md).

The preserved FastFill v1 path takes a floor polygon, optional room type/height,
fixed obstacles, objects with known geometry and optional relations, and predicts
position, yaw and support parent. The v1 commands below retain that protocol.

This is the current implementation for [Slian22/fastfill](https://github.com/Slian22/fastfill). The repository uses `main` as its development branch. Its Python package is `fastfill`; commands run from the repository root. The earlier `src/fastfill_train` Floor/Surface implementation remains in Git history and is a different protocol.

## Code and experiment versions

| Item | Version and purpose |
|---|---|
| Current v2 experiment | Qwen3-8B + bidirectional object decoder; minimal-condition joint bbox prediction and downstream export |
| Preserved v1 implementation | v3.2 data-pipeline, text SFT, evaluation and service |
| Main constraint-supervision experiment | Paired **v3 / v3.1** datasets; keep the same evaluation inputs and training configuration |
| v3.2 dataset | Separately rebuilt constrained dataset with source geometry corrections and recovered rooms |
| Model text protocol | FastFill v1: `placements`, `pos`, integer-degree training targets for `yaw`, optional `on` |

Dataset version numbers are not model architecture or training-stage numbers. v3.2 changes geometry and rows, so comparing v3 with v3.2 is not a constraint-only ablation. Existing training runs should retain their original data and code snapshots. See [version and repository map](docs/versions.md) and [known limitations](docs/known-issues.md).

## Data

Data is maintained separately at [liantian/fastfill-v3](https://huggingface.co/datasets/liantian/fastfill-v3). Repository access and upstream dataset terms continue to apply.

FastFill v2 uses the separate **private** dataset
[liantian/fastfill-v2](https://huggingface.co/datasets/liantian/fastfill-v2), historical snapshot
`review3-20261006`, fixed commit `96f4946624b46bf8dc99bf94311b5d31290ea09a`.
The historical review3 splits contain 124,589/8,137/8,615 scenes with
field-validity masks; the historical complete-geometry cohort contains 57/2/6 scenes.
The new minimal XY-condition `direct-bbox-20261006` derivative contains
9,601/539/624 scenes and 33,545 objects, currently qualified from SpatialLM.
Its yaw is pi-periodic bbox-axis orientation, not certified semantic front.
The new private snapshot is fixed at `35f5272330d37771eea2d11925c42aeec9d917d4`.
See the [dataset download and usage guide](docs/fastfill-v2-dataset-release.md).
The v1 dataset table below remains historical and is not the v2 training input.

| Directory | Train | Dev | Test | Purpose |
|---|---:|---:|---:|---|
| `v3/` | 144,140 | 8,167 | 8,647 | No training constraints |
| `v3.1/` | 144,140 | 8,167 | 8,647 | Same rooms/targets; 42,743 training rows carry constraints |
| `v3.2/` | 144,150 | 8,167 | 8,647 | Corrected constrained dataset; 10 added train rows, 25 changed train rows and 1 changed test row |

The dataset repository also preserves `code/` for v3/v3.1 and `code_v3.2/` for v3.2. This Git repository contains implementation and tests; raw scenes, meshes, weights and generated training JSONLs are not checked in.

## Install and verify

The commands below are for FastFill v1. Install a PyTorch build suitable for your CPU/CUDA environment first. FastFill v2 has its own [server environment and training guide](docs/fastfill-v2-server-start.md).

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -B -m pytest -q -p no:cacheprovider fastfill/tests
```

The offline regression suite covers geometry conversion, build transactions, QA, runtime validation, and a tiny real CPU LoRA merge. It does not replace GPU training, model-quality evaluation, deployment testing or mesh-level validation. vLLM is optional for **v1 text evaluation** and required by the **v1 text service**. The v2 structured training and evaluation entry points do not use vLLM.

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

## FastFill v2 implementation

The separate experimental [FastFill v2 package](fastfill/v2/README.md) jointly
predicts target local size, bottom-center position and yaw for a specified object
inventory. See the [selected-corpus review](docs/fastfill-v2-review-20261005.md) before
retraining. The [training pipeline walkthrough](docs/fastfill-v2-training-pipeline.md)
and [execution manual](docs/fastfill-v2-runbook.md) describe the reviewed upload bundle,
optimizer steps and deployment artifacts. The [server setup guide](docs/fastfill-v2-server-start.md)
gives the upload, CUDA environment and **Qwen3-8B** BF16 pilot commands. The
selected model is separate from the historical 0.5B development template;
`fastfill/v2/configs/qwen3_8b_pilot.json` records the historical pilot settings;
`qwen3_8b_bbox_pilot.json` is the separate 20-step minimal-input template.
The new data reaches at most 26 objects per scene; RoomGenBench's dense 32–122
object examples require separate training coverage and evaluation.
Existing FastFill v1 entry points and v3-series datasets are preserved.
