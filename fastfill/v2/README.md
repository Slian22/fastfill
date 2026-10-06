# FastFill v2 — experimental joint geometry generator

Current task: **room type + room XY size + furniture list** predicts one target
local full size, bottom-center and yaw per requested ID, then exports bbox scene
JSON and colored GLB/SVG directly to downstream systems. No asset retrieval is
required. The richer condition/asset-loop path remains optional.
Qwen encodes condition tokens only. An external bidirectional object decoder
cross-attends to all condition hidden states and produces continuous geometry.
Text SFT is an independent token-CE comparison. Neither requires MinkowskiEngine.

The architecture implements the 2026-10-05 design; the direct minimal-input
boundary follows the user's 2026-10-06 clarification. It is **not a validated method**.
No production Qwen training, quality improvement, real Host deployment, mesh or
physical simulation success is claimed. Existing `fastfill/train.py`,
`evaluate.py`, `serve.py`, and v3/v3.1/v3.2 dataset/experiment lineage are retained.
Model version v2 is separate from the older dataset's v3-series numbering.

## What is implemented

| Module | Behavior |
|---|---|
| `schema.py`, `geometry.py`, `batch.py` | Strict finite-number/ID protocol, condition-only normalization/token spans, masks, fixed geometry, complete-sample budget rejection |
| `model.py` | Qwen-family condition backbone (LoRA/full/frozen), request-bound slots, bidirectional self-attention + cross-attention, positive exponential size, position, yaw logits/residuals, checkpoints |
| `matching.py`, `losses.py` | Fixed identity or explicitly certified within-group Hungarian, detached assignment, differentiable original tensors, complete-field masks, symmetry-aware paired yaw loss, global counts |
| `boxes.py`, `regularizers.py` | Optional BEV oriented convex-hull GIoU, normalized collision-volume and convex-room boundary losses; no implicit 3D GIoU |
| `audit.py`, `adapters.py`, `data.py`, `legacy_build.py`, `legacy_bridge.py`, `legacy_evidence.py` | Source inventory, immutable migration of the selected v3.2 corpus with inherited splits and evidence-based field masks; explicit raw MultiScan audit/smoke adapter |
| `train.py`, `text_sft.py`, `evaluate.py` | Joint optimizer, assistant-only text CE, checkpoint inference, all-request failures, legal matching reference metrics and separated asset/system metrics |
| `direct_layout.py`, `bbox_visualization.py`, `minimal_data.py` | Strict three-field request, source-audited XY dataset projection, canonical/RoomGenBench JSON, generation registry, no-asset bbox GLB/SVG and proxy diagnostics |
| `predict.py` | `--request` minimal-input inference/export; richer `--condition` and legacy catalog path retained |
| `runtime.py`, `validation.py`, `serve.py` | Catalog resolver contract, target/actual separation, pivot-to-canonical transforms, verified support propagation, bounded reselect/translation repair, atomic in-memory Host reference |

The latest [objective update](../../docs/fastfill-v2-review3-objective.md) adds enabled-loss preflight and a globally counted accumulation-window optimizer guard. See the [server setup guide](../../docs/fastfill-v2-server-start.md) for the supplied H20Z host. The historical [review2 audit](../../docs/fastfill-v2-review-20261006.md) separates repaired defects, data provenance and remaining training limits.

## Preserved rich-condition data and release history

The local source audit is in [docs/fastfill-v2-data-audit.md](../../docs/fastfill-v2-data-audit.md).
The historical broad training source is **our selected, frozen `.release/v3.2` corpus**,
derived from the 5 local roots / 16 dataset families and represented by 18 source
names in the frozen IR. Its saved splits contain 160,964 scenes (144,150 train,
8,167 dev, 8,647 test); the existing training flag policy retains 124,843 of the
old training scenes. These are historical selection counts, not new v2 geometry
eligibility counts. The new bridge writes
`data/build/fastfill_v2/selected-v3.2-20261005/`; its `manifest.json` records the
actual migrated splits, field-valid counts and explicit rejection reasons.

Migration reads the frozen release and audit evidence, without redownloading or
rewriting source data. It verifies source/legacy-code/evidence hashes, inherits
each saved scene's train/dev/test assignment (dev becomes validation), and applies
the old flag filter only to training. The converted dataset is staged and exposed
atomically at a new output path. Request IDs and input order are independent of
target geometry; asset/source identifiers remain provenance. Continuous targets
come from the original IR precision, rather than rounded legacy answer text.

Field eligibility is explicit. Recorded window-width evidence corrects new
condition geometry; unresolved oblique-window geometry is rejected. InternScenes
`k0` vertical-axis/front evidence controls position, size and yaw masks. MansionWorld
footprint proxies and padded Holodeck boxes do not provide true local-size labels.
The strict front policy enables semantic yaw supervision only for MultiScan
objects with documented front evidence; other useful position/size labels remain
available with yaw masked. Legacy convention yaw is an explicit alternate policy,
not verified semantic front. Old inferred `on` relations are omitted, rather than
turning bbox contact into support evidence. Tilted fixed geometry is rejected;
unsupported requested-object geometry is masked. A scene with no reliable geometry
field is rejected. Thus the historical broad structured corpus contains partial labels;
it must not be described as fully supervised size/position/yaw for every object.

Known room/support evidence is preserved, and missing evidence remains unknown.
For example, selected MultiScan conditions retain `boundary_known=false`,
`floor_known=false` and missing height/contact labels. The runtime refuses commit
when a required check is unknown. Dataset migration does not establish asset
coverage, mesh validity, physical stability or successful room/asset validation.

The final selected build passed an independent streaming audit of **141,341
scenes / 1,787,052 targets**, with no protocol/split/accounting errors:
**124,589 train / 8,137 validation / 8,615 test**. The 316 extra v2 rejections
are recorded individually. Valid complete fields are position 1,732,321;
size 1,224,670; yaw/full geometry 581. Only 57/2/6 scenes have every requested
object fully labeled under the strict policy. The training set has only 523 yaw
labels across 58 MultiScan scenes: choose a coverage/sampling protocol before
production training; uniform 1,000-step development training is insufficient
as an evidence claim. Source/field integrity does not certify all meshes/fronts.

The portable upload bundle is `outputs/fastfill_v2/FastFill_v2_20261006_review3/`.
Its [execution manual](../../docs/fastfill-v2-runbook.md) gives Linux/macOS
checksum, environment, smoke, training, evaluation and runtime commands.
The repository rebuild commands below require the original frozen `.release`
inputs; the portable bundle uses the supplied data and the execution manual.
Historical review3 local tests: **659 passed + 114 subtests**, v2 coverage **87.68%**.

## Current minimal-input bbox path

Read the [direct bbox guide](../../docs/fastfill-v2-direct-bbox.md) for the exact
request and downstream boundary. The new XY derivative has 9,601/539/624 scenes,
33,545 objects, all complete geometry labels. It is qualified from SpatialLM,
inherits parent splits, omits source height from condition, and supervises
pi-periodic **bbox axes** (not semantic front). Correspondence is fixed; no
exchangeable groups were reconstructed. Maximum scene size is 26 objects.
The old 16-family snapshots and richer-condition pilot remain separate evidence.

```bash
python -m fastfill.v2.predict \
  --checkpoint "$FASTFILL_MODEL" \
  --request fastfill/v2/configs/direct_request.json \
  --output outputs/bbox-prediction-new.json \
  --export-dir outputs/bbox-handoff-new --device cuda
```

This writes geometry JSON, RoomGenBench SceneSpec/registry, GLB, SVG and proxy
diagnostics without a catalog or Host. Unknown height/support stays unknown;
the original RoomGenBench fixed scene/site/render harness needs separate scene
registration and height/metadata handling. Export success is not model quality.
Commands below describe the preserved broader-condition and optional asset paths.

The [training pipeline walkthrough](../../docs/fastfill-v2-training-pipeline.md) explains data, each optimizer step, deployment artifacts, and the RoomGenBench task boundary. review2 withdraws unsupported source-derived `faces` while retaining UID/splits/targets/masks; it adds below-floor diagnostics without rewriting labels. The older bundle remains a historical snapshot.

## Run locally

Run commands from the OptiScene repository root. Choose a **new output path**
for each build/run. No CLI is allowed to write below
`/Volumes/harddisk/3D_Room_Collections`. Runtime CLI produces a new report file;
training/evaluation produce new directories.

```bash
python -m pip install -r fastfill/v2/requirements.txt
python -m pip install pytest pytest-cov

python -m fastfill.v2.audit \
  --source-root /Volumes/harddisk/3D_Room_Collections \
  --output data/build/fastfill_v2/audit-new.json

python -m fastfill.v2.data \
  --source selected-v3.2 \
  --release-root .release/v3.2 \
  --evidence-root .release/audits/2026-09-28/source-check \
  --front-policy strict --workers 4 \
  --output data/build/fastfill_v2/selected-v3.2-new
```

`selected-v3.2` is the default `data` CLI source. The dated primary build path is
`data/build/fastfill_v2/selected-v3.2-20261005/`; use a fresh suffix for reproduction.
Four workers preserve the same deterministic output order as one worker, with
bounded conversion submissions. No existing output is overwritten.

A reproducible complete-label comparison cohort has also been built at
`data/build/fastfill_v2/selected-complete-v3.2-20261006/` (57/2/6 scenes).
Each retained JSONL row is byte-identical to its parent row and retains its split.
To reproduce into a fresh directory:

```bash
python -m fastfill.v2.cohort \
  --data-root data/build/fastfill_v2/selected-v3.2-20261005 \
  --output data/build/fastfill_v2/selected-complete-new
```

Use this same cohort for both structured and text controlled comparisons.
Text training checks complete labels before loading the backbone.

The local complete-label smoke subset at
`outputs/fastfill_v2/selected-full-geometry-smoke-20261005/` contains **57 train /
2 validation / 6 test** scenes from default-kept selected MultiScan, with one
incomplete-label scene excluded. It exercises every geometry head and allows a
text/structured plumbing comparison on the same labels. It is a smoke subset,
not the primary corpus or a completed model-quality benchmark.

For an offline optimizer/checkpoint smoke:

```bash
python -m fastfill.v2.train \
  --config fastfill/v2/configs/smoke.json \
  --data outputs/fastfill_v2/selected-full-geometry-smoke-20261005/train.jsonl \
  --output outputs/fastfill_v2/selected-all-heads-smoke-new \
  --dry-run --max-samples 2

python -m fastfill.v2.evaluate \
  --checkpoint outputs/fastfill_v2/selected-all-heads-smoke-new/model \
  --data outputs/fastfill_v2/selected-full-geometry-smoke-20261005/test.jsonl \
  --output outputs/fastfill_v2/selected-all-heads-eval-new \
  --max-length 32768 --device cpu

python -m fastfill.v2.text_sft \
  --data outputs/fastfill_v2/selected-full-geometry-smoke-20261005/train.jsonl \
  --output outputs/fastfill_v2/selected-text-smoke-new \
  --backbone tiny --dry-run --batch-size 1 --max-length 32768

python -m fastfill.v2.evaluate --baseline text \
  --checkpoint outputs/fastfill_v2/selected-text-smoke-new \
  --data outputs/fastfill_v2/selected-full-geometry-smoke-20261005/test.jsonl \
  --output outputs/fastfill_v2/selected-text-eval-new \
  --device cpu --max-length 32768 --max-new-tokens 4096 --max-samples 1

python -m pytest fastfill/tests fastfill/v2/tests \
  --cov=fastfill.v2 --cov-config=fastfill/v2/.coveragerc --cov-report=term-missing -q
```

The explicit `tiny` backbone is a causal byte-token GRU for offline plumbing,
never a Qwen replacement in experimental conclusions. Tests also instantiate
local randomly initialized Qwen2 + PEFT models without downloading weights, to
verify LoRA/full/frozen gradient paths and checkpoint round-trips.

The raw-drive adapter remains available explicitly for an optional audit/smoke
build. It creates separate house-grouped splits and **does not replace or inherit
the selected v3.2 corpus**:

```bash
python -m fastfill.v2.data --source multiscan \
  --source-root /Volumes/harddisk/3D_Room_Collections \
  --output data/build/fastfill_v2/multiscan-audit-new
```

## Train Qwen and evaluate

The reviewed data is published privately at
[liantian/fastfill-v2](https://huggingface.co/datasets/liantian/fastfill-v2), fixed
commit `96f4946624b46bf8dc99bf94311b5d31290ea09a`, snapshot `review3-20261006`.
Use the [dataset guide](../../docs/fastfill-v2-dataset-release.md) to download and
verify the main masked corpus and the complete-label pilot cohort. Raw v1 SFT
messages are not structured v2 samples.

`configs/structured.json` chooses `Qwen/Qwen2.5-0.5B-Instruct` as a development
default, with LoRA, 128-dimensional two-layer decoder, 12 yaw bins, all four base
weights 1, and no scene/box loss. The selected production backbone is now
**Qwen3-8B** on the user's H20Z server (`ssh yxd-dev`). Override the frozen
development template with a separate runtime configuration; the server model
path is `/home/jovyan/shanliantian/models/Qwen3-8B`. See the
[server instructions](../../docs/fastfill-v2-server-start.md) for BF16 preflight
and the bounded 20-step pilot; `configs/qwen3_8b_pilot.json` provides the selected
8B pilot settings. Freeze sampling, context and the full training
budget after pilot measurements; the template's default budget is not validated.

```bash
python -m fastfill.v2.train \
  --config fastfill/v2/configs/structured.json \
  --data data/build/fastfill_v2/selected-v3.2-20261005/train.jsonl \
  --validation data/build/fastfill_v2/selected-v3.2-20261005/validation.jsonl \
  --output outputs/fastfill_v2/selected-qwen-structured-new

# Optional multi-GPU launch using the same configuration and fresh run path:
accelerate launch -m fastfill.v2.train \
  --config fastfill/v2/configs/structured.json \
  --data data/build/fastfill_v2/selected-v3.2-20261005/train.jsonl \
  --validation data/build/fastfill_v2/selected-v3.2-20261005/validation.jsonl \
  --output outputs/fastfill_v2/selected-qwen-distributed-new

python -m fastfill.v2.evaluate \
  --checkpoint outputs/fastfill_v2/selected-qwen-structured-new/model \
  --data data/build/fastfill_v2/selected-v3.2-20261005/test.jsonl \
  --output outputs/fastfill_v2/selected-qwen-eval-new --device cuda
```

Tune dtype/batch/context/budget for the selected machine in a new config. The
defaults are development starting points, not measured optimal settings.
An unrestricted structured run currently loads the train and validation JSONL
samples into memory; plan host RAM against the migrated corpus size. Structured
`--max-samples` bounds a streamed training prefix for smoke runs, not a complete
randomized training subset. Text SFT also loads its input JSONL into memory.

A full text SFT comparison uses the materialized 57/2/6 complete-label cohort
with complete size/position/yaw labels for every requested object and inherited
splits. Its builder and artifact are provided above. Select shared context/object
eligibility and training budgets for the formal research protocol, then use that
**same cohort for the controlled structured comparison**. Comparing text trained
on complete labels against structured training on all partial-label scenes would mix the
training objective with data coverage. The full masked structured run is a
separate experiment whose scope and eligible-label counts must be reported.

Structured training records complete-sample context rejection, source hashes,
implementation hashes, package versions, unweighted losses, effective global
counts, head/backbone gradient norms and trainable parameter count. Validation
loss is a mean-of-batches diagnostic; final all-request metrics come from
`evaluate.py`. Accelerate checkpoint states are diagnostic recovery artifacts;
automatic optimizer resume is not exposed by the initial CLI. Deployment
checkpoints contain backbone adapter/full weights where needed and decoder/heads.

## Geometry and exact losses

- Right-handed Z-up meters. Local +X is the canonical bbox axis; the current
  geometric-yaw dataset does not certify semantic front. `(w,d,h)` are full
  X/Y/Z lengths, invariant under yaw. Positions are bbox bottom-center.
- Position normalization uses input room origin and XY extent; missing height
  uses fixed 3m. Trusted floor-support z and requested fixed size dimensions are
  used directly. Unconstrained head tensors are retained separately for diagnosis.
- `size = positive_reference * exp(clamp(u,-10,10))`; reference is fixed `(1,1,1)`
  meters, with no validation/test statistics. Head residuals use `tanh` by default.
- SmoothL1 beta=1 for normalized bottom-center and log size ratio. Each term uses
  only complete valid field vectors and at least one learned coordinate, then
  sum learned coordinate losses /3 / valid instance count. Missing labels are
  filtered before arithmetic. Fixed dimensions do not reduce the `/3` denominator.
- Yaw CE + SmoothL1 on **GT-bin** residual in training; argmax predicted bin in
  inference. Explicit justified symmetry orders choose the same equivalent angle
  by the joint weighted CE/residual minimum. Box symmetry never implies semantic
  front symmetry. Detached group cost uses position L1 + log-size L1; no global
  matching, detection classification, objectness, NMS or repeated GT.

For DDP, all-reduced valid counts and world-size compensation preserve the
per-global-microbatch objective under averaged gradients. Accumulation averages
microbatch-normalized losses; it is not a new valid-count reduction across the
whole accumulation window. Logging averages the corresponding rank contributions.

Optional `box_operator=bev_oriented_giou_convex_hull` is piecewise differentiable
oriented rectangle intersection with convex-hull enclosure of both boxes.
Vertices are calculated in a common local float64 frame to protect small valid
boxes from global-coordinate cancellation. Intersection/hull topology decisions
are detached; position/size/yaw tensors retain gradients within topology regions.
Contacts/coincident edges are nonsmooth; numerical-gradient tests use
nondegenerate boxes. BEV does not supervise height/z. GIoU and true BEV IoU have
separate functions/metrics. This implementation is correctness-oriented and
Python-loop based; benchmark before large-batch GPU use. No 3D GIoU is claimed.

Collision/convex-room boundary regularizers are separate flags, default zero.
Unknown/concave boundary input is refused by the boundary training operator.
Declared support pairs are exempt from volume regularization; runtime contact
still needs actual evidence. These losses do not replace runtime validation.

## WorldEdge contract and offline asset loop

```bash
python -m fastfill.v2.predict \
  --checkpoint outputs/fastfill_v2/selected-qwen-structured-new/model \
  --condition REQUEST.json --output outputs/fastfill_v2/prediction-new.json \
  --device cuda

python -m fastfill.v2.serve \
  --condition REQUEST.json --prediction PREDICTION.json --catalog CATALOG.json \
  --output outputs/fastfill_v2/runtime-new.json \
  --max-asset-retries 2 --max-repair-calls 2 --repair-step-m 0.25 \
  --commit-in-memory

python -m fastfill.v2.evaluate \
  --checkpoint outputs/fastfill_v2/selected-qwen-structured-new/model \
  --data EVALUATION.jsonl --catalog CATALOG.json \
  --output outputs/fastfill_v2/asset-eval-new --device cuda \
  --asset-retries 2 --repair-calls 2 --repair-step-m 0.25 \
  --commit-in-memory
```

Catalog records carry `ref`, category, actual local size, a raw-to-canonical
4x4 transform, optional semantic front, capability names and verified horizontal
support polygons with local heights. `retrieval_tolerance_log` (or legacy alias
`retrieval_tolerance`) is scalar/three-vector of maximum absolute log-size ratio.
Category/capability/fixed/range checks precede size ranking. This deterministic
catalog example does not implement nuanced natural-language or learned asset
retrieval. Nonempty typed `attributes` requests are rejected because the current
Asset contract has no verified attribute evidence; actual validation reports
hard `attributes_unverified`. An external resolver alone cannot bypass this:
support requires an explicit asset evidence contract and a matching validator.
Description text is not verified proof of color/material or other attributes.
Known semantic-front metadata must already point along canonical +X. A +Y-front
asset must first normalize its transform, local size axes and support polygons;
the runtime rejects that unnormalized metadata. Missing front remains unknown.
Catalog JSON accepts an array or `{"assets": [...]}`. Asset provenance is a
list of source references.

Target size/pose, first-pass actual objects and final repaired actual objects are
retained separately. No asset shrink/delete occurs. Reconciliation applies
`T(bottom_center) @ Rz(yaw) @ canonical_transform`; actual support surfaces, not
bbox tops, update dependent child heights. Translation repair moves support
descendants consistently, is bounded, and revalidates actual geometry.

Runtime geometry uses upright OBBs and polygon checks. Unknown required support,
boundary or semantic front blocks commit. Nonempty openings without a verified
clearance converter produce hard `openings_unchecked` diagnostics. Actual
capabilities are rechecked after asset filtering. Conservative OBBs can reject
valid contents of hollow furniture or shelves; this initial validator does not
infer cavity geometry or bypass penetration checks by furniture category.
Mesh/physics/Solver are explicitly
unavailable; requiring these validation levels also blocks commit. Time budgets
are checked between callbacks; external callbacks must enforce their own I/O
timeouts. `AtomicMemoryHost` demonstrates all-or-fail versioned/idempotent commit.
**Real WorldEdge persistent Host integration requires a concrete atomic adapter**;
this repo exposes the protocol and does not mutate a remote or existing world.
The existing sibling WorldEdge resolver's `resolve(FloorObjectSpec)` contract is
for v1 and cannot silently substitute for v2 target-before-asset resolution.

Evaluation always retains failed requests in schema/ID/validity and system-rate
denominators. Valid-label reference means disclose their eligible-object counts.
Malformed text JSON, missing assets and incomplete validation have distinct
diagnostics. Latency is p50/p95; raw model, asset-first-pass and final stages are
logged separately. Bbox acceptance does not establish task/physics success.
Operational end-to-end latency adds model generation and asset/runtime time,
excluding reference scoring and matching. Supplied-prediction evaluation reports
runtime latency and leaves end-to-end latency unknown; evaluation wall time is
recorded separately. Incomplete labels in anonymous groups use diagnosed fixed
reference correspondence; ineligible reference scores never change model-schema
success or skip the asset loop. `--required-levels bbox mesh` in either CLI
requests mesh evidence and blocks offline commit while that evidence is unknown.

## Required comparisons

1. Preserve existing v1 as an explicitly **bbox-conditioned oracle-size** reference
   using its own existing entry point. It has richer size information; label that
   difference rather than reporting it as the same hidden-size task.
2. Train v2 text SFT and its controlled structured counterpart on the same
   minimal XY dataset, condition information and documented sample exposure.
   For optional asset-loop experiments, additionally fix asset/repair budgets.
3. Train structured fixed correspondence with `configs/structured_fixed.json`.
4. Train structured exchangeable-group Hungarian with `configs/structured.json`.
5. If useful, compare `configs/structured_bev_box.json` at equal training budget.

No such quality comparison has been run. Raw/log size, sin/cos yaw, additional
matching cost and scene regularizers remain separately selected future ablations,
not claimed improvements. No V-DETR default weight is claimed optimal.

Official [V-DETR criterion](https://github.com/V-DETR/V-DETR/blob/main/criterion.py)
and [main](https://github.com/V-DETR/V-DETR/blob/main/main.py) were consulted only
for matcher/criterion separation and angle supervision. The implementation here
is specific to requested layout generation. Point-cloud encoders, 3DV-RPE,
one-to-many, large detection query banks and MinkowskiEngine are not dependencies.

## Recorded local validation — 2026-10-05

The selected-corpus structured smoke used two complete-label training scenes and
one optimizer step with the explicit tiny backend. Its checkpoint evaluation
completed the **6 selected test scenes / 37 requested objects**, with schema,
ID and positive-size success for each. Target-geometry validation was **0%**;
unknown boundary/support evidence remains a blocker and usable layouts have not
been demonstrated. Reports are at
`outputs/fastfill_v2/selected-all-heads-smoke-20261005/run_manifest.json` and
`outputs/fastfill_v2/selected-all-heads-eval-20261005/report.json`.

The selected text smoke performed a token-CE optimizer step on the same
complete-label training cohort. Its bounded one-request generation check failed
JSON/schema parsing. Report:
`outputs/fastfill_v2/selected-text-eval-20261005/report.json`. These plumbing runs
are not a controlled quality comparison. A separate single-fixture overfit test
checks decreasing geometry loss and gradients to all heads.

Offline regression tests include immutable release/hash checks, inherited splits,
train-only flag filtering, atomic migration failure and deterministic one/two-worker
output equivalence, together with geometry/model/runtime checks. Final suite and
coverage totals are recorded by the final verification run; historical coverage
artifacts do not describe the entire repository.

The server and Qwen3-8B checkpoint have been chosen and the historical pilot ran.
Full training exposure/budget and quality comparisons remain unvalidated.
No quality gain or actual persistent-world success has been measured.
