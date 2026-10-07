# FastFill v2 — experimental joint geometry generator

Main task: **room geometry + object requests + fixed objects + available support
and relation constraints** predicts one target local full size, bottom-center and
yaw per stable requested ID. Continuous heads provide structural supervision;
Hungarian matching is restricted to legal exchangeable groups. Deployment design
then resolves actual assets, reconciles actual geometry and validates the whole
scene before atomic Host commit. Current code implements the offline reference
contracts; real mesh/physics/Solver checkers and a persistent WorldEdge Host
adapter remain integration work.
The separate three-field `reference_extent` view is a simplified-input ablation
that exports bbox JSON and GLB/SVG without an asset library. It does not replace
the main full-condition task or its deployment acceptance requirements.
Qwen encodes condition tokens only. An external bidirectional object decoder
cross-attends to all condition hidden states and produces continuous geometry.
Text SFT is an independent token-CE comparison. Neither requires MinkowskiEngine.

The architecture follows the 2026-10-05 design. The direct minimal-input boundary
is an independent ablation. It is **not a validated method**.
No production Qwen training, quality improvement, real Host deployment, mesh or
physical simulation success is claimed. Existing `fastfill/train.py`,
`evaluate.py`, `serve.py`, and v3/v3.1/v3.2 dataset/experiment lineage are retained.
Model version v2 is separate from the older dataset's v3-series numbering.

The final mesh consumer is **RoomGenBench**. `predict --export-dir` now saves
canonical `condition.json` / `layout.json` alongside SceneSpec, asset registry,
GLB/SVG and diagnostics. `python -m fastfill.v2.roomgenbench` accepts any such
handoff and an independent output directory, optionally consuming generated
GLB/JSON sidecars. It preserves declared support, constraints and fixed bbox
proxies, counts failed/fallback/missing assets, and records native/fitted sizes.
See the [downstream contract and commands](../../docs/fastfill-v2-roomgenbench-interface-20261006.md).
This mesh-fit adapter does not run learned generation or certify physics/Host.

The formal runs (round 2) are `configs/qwen3_8b_main_4gpu_regression.json`
(4 ranks x B1 x K24, regression position head) and
`configs/qwen3_8b_main_3gpu_grid.json` (3 ranks x B1 x K32, grid_residual head):
both global batch 96, 3,887 updates (3 epochs of 124,375 scenes), warmup 117,
validation and checkpoint every 500. They inherit everything else from
`configs/qwen3_8b_main_world7.json` (the earlier seven-rank candidate). Pass
them explicitly with the qualified dataset; generic defaults remain
development configurations. Launch commands are in the round-2 section below. Latest audit, source pins and publication state are
in the [design audit](../../docs/fastfill-v2-design-audit-20261006.md).

## What is implemented

| Module | Behavior |
|---|---|
| `schema.py`, `geometry.py`, `batch.py` | Strict finite-number/ID protocol, condition-only normalization/token spans, masks, fixed geometry, complete-sample budget rejection |
| `model.py` | Qwen-family condition backbone (LoRA/full/frozen), request-bound slots, bidirectional self-attention + cross-attention, positive exponential size, position (`model.position_head` regression or grid_residual), yaw logits/residuals, checkpoints |
| `matching.py`, `losses.py` | Fixed identity or explicitly certified within-group Hungarian (groups from `validity.exchangeable_group`), detached assignment, differentiable original tensors, complete-field masks, L1 / smooth-L1 selectable position and size terms, symmetry-aware paired yaw loss, box-symmetric size/yaw candidates for `size_axis_swap_allowed`, per-term sums and counts for window logging |
| `boxes.py`, `regularizers.py` | Optional BEV oriented convex-hull GIoU, normalized collision-volume and convex-room boundary losses; no implicit 3D GIoU |
| `audit.py`, `adapters.py`, `data.py`, `legacy_build.py`, `legacy_bridge.py`, `legacy_evidence.py` | Source inventory, immutable migration of the selected v3.2 corpus with inherited splits and evidence-based field masks; explicit raw MultiScan audit/smoke adapter |
| `qualified_data.py` | New full-condition main derivative with pinned parent/source hashes, D1/D2 qualification changes, fixed-identity fallback for groups losing complete geometry, per-member change records and unchanged target numbers |
| `train.py`, `text_sft.py`, `evaluate.py` | Joint optimizer with warmup/cosine schedule, decoder LR, seeded augmentation, resume and per-checkpoint export; assistant-only text CE; checkpoint inference, all-request failures, legal matching reference metrics, mean-predictor baselines, collapse metrics and separated asset/system metrics |
| `direct_layout.py`, `bbox_visualization.py`, `minimal_data.py`, `multisource_data.py` | Independent three-field ablation, explicit reference-extent semantics, source-audited partial-mask projection, canonical/RoomGenBench JSON with polygon-wall shell, door/window openings, declared/inferred/unknown `place` and benchmark-convention shared `asset_key`, no-asset bbox GLB/SVG and proxy diagnostics |
| `predict.py` | Main `--condition` checkpoint inference; separate `--request` ablation and optional offline catalog execution; `--max-length` defaults to the checkpoint's training value |
| `runtime.py`, `validation.py`, `serve.py` | Catalog resolver contract, target/actual separation, pivot-to-canonical transforms, verified support propagation, bounded reselect/translation repair, atomic in-memory Host reference |

The latest [objective update](../../docs/fastfill-v2-review3-objective.md) adds enabled-loss preflight and a globally counted accumulation-window optimizer guard.

## 2026-10-06 audit fixes

Contract changes from the three-group audit (schema, serialization, yaw policy, matching, losses, trainer, evaluation, RoomGenBench handoff):

- **Row schema.** `condition.objects[i]` carries only `id`, `category`, `description` and optional `support_parent`. Exchangeable groups moved to `validity.exchangeable_group` (`list[str|null]`, target order) and `validity.yaw_symmetry_order` (`list[int]`, 1 = semantic front, 2 = axis mod pi) is always present. `io.read_samples` migrates old rows (`schema.migrate_legacy_row`) once; `validate_condition` rejects `exchangeable_group` in objects.
- **Condition text.** `batch.condition_segments` renders `schema_version, room, constraints, objects` in that fixed order (objects last, so causal object tokens see the room), drops `room.boundary_quality` and every `_`-prefixed key; object span tagging and the prompt header are unchanged. The direct request path therefore renders the same room fields as a training rectangle.
- **Yaw policy at build.** Default `--front-policy axis`: yaw valid = upright, finite, `front_known`; `yaw_symmetry_order` 1 for MultiScan, 2 elsewhere; round 2: `validity.size_axis_swap_allowed` (box may equally be (sx, sy, yaw) or (sy, sx, yaw + pi/2)) is true for InternScenes arkit/3rscan/mp3d/scannet, InteriorGS, HSSD200 categories containing `chair` and MultiScan categories containing `bed`, and those objects carry order 4 (old rows migrate to all false). Floor support comes from the source annotation only: an `anchor = floor` (not inferred) object on a known floor is declared `support_parent: floor` and its target z snapped when within 2 cm (`field_evidence.legacy_z_snap_applied_to_target`), otherwise left undeclared with free z; the manifest counts `floor_declarations_written` / `floor_declaration_z_snapped` / `floor_declaration_skipped_floating` and `source_size_axis_swap_allowed_objects`. `strict` and `legacy-convention` remain selectable (`legacy_bridge.FRONT_POLICIES`). Descriptions are the source `desc` or the category (`provenance.descriptions = source_desc_or_category`). Groups need complete **position** only; signature = category/description/support_parent.
- **Qualification.** Any size axis < 3 mm masks the whole size vector (`degenerate_axis_lt_3mm`); a target above `room.height_m + 0.05` keeps labels and the declared height and is flagged in `provenance.height_conflict = {objects, max_excess_m}` (journal field `provenance.height_conflict`, reason `target_exceeds_declared_height_flag_only`; round 2 reverts the round-1 height drop); the five RoomGenBench SAGE layouts move to the test split with `provenance.holdout_reason = roomgenbench_benchmark_room`. `legacy_verify` / `multisource_verify` recompute `source_yaw_valid_objects`, `source_yaw_symmetry_order_counts`, `exchangeable_group_counts`, `yaw_policy`, `descriptions` and accept the holdout.
- **Matching and losses.** `match_batch` reads `batch["exchangeable_group"]` (per request slot). Group cost = `alpha_position * L1(position)` always, plus `alpha_size * L1(log size)` only when every member has a complete size label; position-incomplete groups keep fixed identity. `loss.position_type` / `loss.size_type` in {`l1`, `smooth_l1`} (default `l1`), `loss.smooth_l1_beta` (default 1). Main-config weights 1.0 / 0.6 / 0.08 / 7.0 (round 2 caps yaw_reg at 2.0) come from the [calibration note](../../docs/fastfill-v2-loss-calibration-20261006.md). The criterion exposes `term_sums` / `term_counts` so window logs are count-weighted means over every microbatch.
- **Trainer.** New sections `optimizer` (`warmup_steps` default 3 % of steps, `schedule` cosine to 10 % LR, `decoder_lr`), `augmentation` (`rotate90`, `mirror`, `shuffle_objects`, `drop_constraints_p`, `drop_support_p`, `category_only_description_p`, round 2 `minimal_form_p`; training rows only, seeded per (seed, epoch, row)), `validation.exclude_flags` (default the three training flags), `training.resume`, `training.export_model_every_checkpoint`. Unknown keys raise. `--resume <state-step-n>` continues into a new output directory with identical config/data; each checkpoint writes `state-step-<n>/` and `model-step-<n>/`; `train.log` tees stdout; bf16 windows with nonfinite gradients are skipped and counted; a room-centre / category-median-size / uniform-yaw baseline is logged before training.
- **Evaluation.** `report.json` adds `baselines` (room centre, category mean position, category median size, uniform yaw; leave-one-out or `--baseline-fit <jsonl>`), `collapse` (predicted vs ground-truth overlap, stacking, wall distance, central fraction), `by_source` and `yaw_error_rad_by_symmetry_order`.
- **RoomGenBench export.** `room.walls` from the floor polygon (0.1 m thick, height = room height or 2.7), `doors` / `windows` from fixed objects attached to the nearest wall, `place` / `support_status` (round 2, `direct_layout.infer_support`): a declared support parent or hard `on` gives `floor` / `on_object` / `wall` with `declared`; otherwise `floor` when the bottom is within 2 cm of a known floor height (none when `floor_known` is false, e.g. `reference_extent`), else `on_object` for the highest strictly-lower predicted box whose top is within 3 cm and whose footprint contains the object's centre, both `inferred`; anything else `unknown` (wall is never inferred). `assets.jsonl` and assembly receipts carry the same `place` and `support_status`; `asset_key = slug(type)[:24] + "_" + sha1(description)[:8]` shared by identical type+description (benchmark convention), per-instance dimensions in the scene. Tests need the reference assembler at `RoomGenBench/` or `FASTFILL_ROOMGENBENCH_ROOT`. See the [server setup guide](../../docs/fastfill-v2-server-start.md) for the supplied H20Z host. The historical [review2 audit](../../docs/fastfill-v2-review-20261006.md) separates repaired defects, data provenance and remaining training limits.

Review repairs on top of the above (same day):

- **Augmentation.** `rotate90` / `mirror` snap rigid-motion float noise (`8.8 - 1.1` renders as `7.7`, full-precision source values are kept), and `train._Rows` falls back to the preflighted row when an augmented condition exceeds `max_length` (`run_manifest.augmentation_fallbacks`). After `drop_constraints_p` / `drop_support_p` / `category_only_description_p` fire, exchangeable groups are recomputed with the build rule (`matching.group_labels`): requests made textually identical by a drop are matched by geometry instead of fixed identity; build-time groups whose wider candidate fails certification keep their members under a `kept_<label>` label.
- **Benchmark rooms.** `io.ROOMGENBENCH_HOLDOUT_GROUPS` is the single producer copy (`multisource_verify` mirrors it independently); `io.read_samples(training=True)` (train.py, text_sft.py) and `cohort` refuse rows whose `provenance.group` is a benchmark room or that carry `holdout_reason`.
- **RoomGenBench assembly.** `roomgenbench._shell_parts` builds each wall with the reference `build_shell` but takes the outward normal from the floor-polygon winding (the reference uses the bbox centre, which extrudes some walls of non-convex rooms into the room); receipt `room.wall_normal_source = floor_polygon_winding`.
- **Smaller.** `decode_yaw` computes bin centres in at least float32 under bf16 residuals; evaluation baselines are row-level leave-one-out (every label of the evaluated row is excluded, not only the object's own); the declared-height conflict adds a 1e-6 m epsilon so float32-rounded sizes do not flag a real height; `legacy_bridge` reads Scan2CAD `sym` (`__SYM_ROTATE_UP_4` -> `yaw_symmetry_order` 4, `__SYM_ROTATE_UP_INF` -> yaw unsupervised; `legacy_verify` accepts orders 1/2/4).

## 2026-10-07 round 2

Contract K1–K10 on top of commit 7ea1a0d. Rows built before round 2 still load
(`schema.migrate_legacy_row` adds an all-false `size_axis_swap_allowed`), but
K1–K3 only reach the training data through the rebuild below.

- **K1 box symmetry.** `validity.size_axis_swap_allowed` (bool, target order, always written): the labelled box may equally be (sx, sy, yaw) or (sy, sx, yaw + pi/2). The axis policy sets it for InternScenes arkit/3rscan/mp3d/scannet and InteriorGS (all objects), HSSD200 categories containing `chair` and MultiScan categories containing `bed`; those objects carry `yaw_symmetry_order` 4. Collate exposes `batch["size_axis_swap_allowed"]` (b x n bool). Loss and matching: see "Geometry and exact losses". Evaluation scores these objects' `log_size_error` / `yaw_error_rad` as the box-equivalent minimum over the same four candidates, keeps `*_plain_convention` (written axis order, yaw modulo pi for swap objects and modulo the recorded order otherwise) and counts `box_equivalent_objects`; the category-median-size baseline is box-equivalent too.
- **K2 height conflict.** Flag only; the round-1 height drop is reverted (see "Qualification" above).
- **K3 floor support.** Source `anchor = floor` only, 2 cm snap, farther anchors stay undeclared with a free z (see "Yaw policy at build" above); `legacy_verify` recomputes the manifest counts.
- **K4 MansionWorld sizes.** IR objects carry only `id, category, size, pos, yaw, anchor, parent, tilted, desc, structure`; no field separates annotated bbox sizes from footprint proxies, so the source-level mask stays (every MansionWorld and OptiScene_holodeck size invalid). Using `anchor` as an indirect proxy is an open decision.
- **K5 three-field contract.** `batch.render_minimal_condition` keeps `schema_version`, room {frame, axis-aligned bounding rectangle, floor_z_m, floor_known, boundary_known = true, height_m, room_type}, `constraints: []` and objects {id, category, description}. `direct_layout.request_to_condition` calls it, so a default (`rectangular`) request renders byte-identically to the projection of a training row with the same room type, size and inventory (`boundary_quality` is never rendered). The separate `reference_extent` ablation profile still sets `boundary_known` false after the projection, as its own frozen data does. Augmentation `minimal_form_p` (default 0.5) fires only when `batch.minimal_form_eligible`: a 4-point axis-aligned rectangle (1 cm tolerance, +1e-9 m float margin so rotations agree) whose `boundary_known` is not false, so hull and `reference_extent` rectangles are never rewritten as known; it regroups exchangeable groups and drops floor-fixed z for that sample. `evaluate.project_minimal` is the shared projection (same predicate; other rows count as `skipped_non_rectangular_rooms`) for `evaluate --projection full minimal` and the trainer's periodic validation, so a `reference_extent` ablation run has no minimal projection and no `selection_metric.best`.
- **K6 hand-off place.** `place` / `support_status` in {declared, inferred, unknown}; see "RoomGenBench export" above. Hand-offs exported before round 2 fail the scene check of `roomgenbench._handoff`; re-export them with `predict --export-dir`.
- **K7 grid head.** See "Geometry and exact losses".
- **K8 binding.** `run_manifest_start.json` before the first update (config, data / validation / implementation sha256, resolved backbone path + HF snapshot revision + config.json sha256 (a plain local directory is identified by config.json only, which does not pin the weights), tokenizer sha256, augmentation, world size, admitted `supervised_samples` / `rejected_samples` / `validation_samples` / `validation_minimal_samples`, selection metric); `run_manifest.json` at the end (adds `selection_metric.best`); `checkpoint_manifest.json` in `model/` and every `model-step-<n>/`. `--expect-data-sha256` / `--expect-validation-sha256` abort before any output. `evaluate` / `predict` take `max_length` from the checkpoint (`io.load_checkpoint_config`) unless `--max-length` is explicit, and model settings from its `model_config.json`; evaluation reports record `checkpoint`, `checkpoint_binding`, `max_length_source`, `data_sha256`, `implementation_sha256`, `code_commit` / `code_dirty`.
- **Selection metric.** `validation.minimal.collapse.score` = predicted BEV overlap rate (IoU > 0.3) + central-quarter fraction on the minimal projection of the validation rooms, lower is better. Each validation runs the rectangular subset a second time. The labels' own score on the same projection is recorded once as `selection_metric.ground_truth` in `run_manifest_start.json` and `run_manifest.json`; a predicted score below it means a more spread-out layout than the data, not a more accurate one (the score has no accuracy term; read it with `validation.minimal.unweighted`). The final update is always validated and checkpointed as well (unless the interval is 0), so `steps` need not be a multiple of the interval. Keep `validate_every == checkpoint_every` so the best step has a `model-step-<n>/`.
- **K9 configs.** Every Qwen config uses `max_length` 8192 (including the Qwen2.5-0.5B `structured*` templates); main and `pilot_1gpu` use `yaw_reg` 2.0, `position_cell` 0.04, `position_residual` 0.4, `minimal_form_p` 0.5; formal configs as at the top of this README.
- **K10 validator.** `validate_scene` check status is `pass` / `violation` / `unknown` (was `fail`) with `counts`; `ok` is unchanged (every hard check must pass, unknown is not pass). This changes the JSON of serve.py / runtime reports. `evaluate` reports `target_validation_checks` per check code and `target_validation_not_run`; ceiling checks on `provenance.height_conflict` rows count as `ceiling_on_height_conflict_rows`. `direct_layout.bbox_diagnostics` (hand-off proxy diagnostics) keeps its own pass/fail words.

Rebuild (CPU, about 1–1.5 h; `OUT` must be new and outside `.release` and the
external disk). The parent hash pins still name review3, so the qualified and
verify steps pin the fresh parent explicitly:

```bash
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
R=/Users/slian/Desktop/3D/Worldedge/OptiScene/.release/v3.2
E=/Users/slian/Desktop/3D/Worldedge/OptiScene/.release/audits/2026-09-28/source-check
OUT=outputs/fastfill_v2/rebuild-20261007
python -m fastfill.v2.legacy_build --release-root $R --evidence-root $E --front-policy axis --workers 4 --output $OUT/bridge
python -m fastfill.v2.legacy_verify --data-root $OUT/bridge --output $OUT/reports/legacy-verify.json
python -m fastfill.v2.review_data build --parent-root $OUT/bridge --output $OUT/reviewed
python -m fastfill.v2.review_data verify --parent-root $OUT/bridge --data-root $OUT/reviewed
python -m fastfill.v2.qualified_data --parent-root $OUT/reviewed --output $OUT/main --ir-root $R/ir \
  --frozen-manifest $R/data/v3.2/MANIFEST.json --pin-current-parent
python -m fastfill.v2.multisource_verify --full-condition --parent-root $OUT/reviewed --data-root $OUT/main \
  --ir-root $R/ir --parent-manifest $OUT/reviewed/manifest.json --output $OUT/reports/main-verify.json
# Optional three-field ablation view and its independent verification:
python -m fastfill.v2.multisource_data --parent-root $OUT/reviewed --ir-root $R/ir \
  --frozen-manifest $R/data/v3.2/MANIFEST.json --output $OUT/data-minimal-reference
python -m fastfill.v2.multisource_verify --parent-root $OUT/reviewed --data-root $OUT/data-minimal-reference \
  --ir-root $R/ir --parent-manifest $OUT/reviewed/manifest.json --output $OUT/reports/minimal-verify.json
```

A bounded smoke build adds `--source NAME` (repeatable) and
`--max-scenes-per-source N` to `legacy_build` (not forwarded by `fastfill.v2.data`).

Formal runs, side by side on GPUs 1–7 (GPU0 stays free). `--nproc_per_node`
must match the config (any other world size changes the global batch of 96):

```bash
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
MAIN=/path/to/rebuild-20261007/main
D=$(sha256sum $MAIN/train.jsonl | cut -d' ' -f1); V=$(sha256sum $MAIN/validation.jsonl | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.run --standalone --nproc_per_node=4 --module fastfill.v2.train \
  --config fastfill/v2/configs/qwen3_8b_main_4gpu_regression.json \
  --data $MAIN/train.jsonl --validation $MAIN/validation.jsonl \
  --expect-data-sha256 $D --expect-validation-sha256 $V --output ../run-main-4gpu-regression
CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=3 --module fastfill.v2.train \
  --config fastfill/v2/configs/qwen3_8b_main_3gpu_grid.json \
  --data $MAIN/train.jsonl --validation $MAIN/validation.jsonl \
  --expect-data-sha256 $D --expect-validation-sha256 $V --output ../run-main-3gpu-grid
```

Check `run_manifest_start.json` right after launch: `world_size` 4 / 3 and
`supervised_samples`. 3,887 updates are 3 epochs only for 124,375 admitted
scenes; the trainer's own preflight now uses 8192 tokens, so for another count N
write a new config with `steps = ceil(3 N / 96)` and `warmup_steps = round(0.03 steps)`.
Pick the checkpoint by `run_manifest.json` `selection_metric.best.step`, never by
test; then `python -m fastfill.v2.evaluate --checkpoint <run>/model-step-<best>
--data $MAIN/test.jsonl --projection full minimal --device cuda --output <new dir>`.

Recorded local validation (2026-10-07, CPU, tiny backbone, no Qwen weights):
`pytest fastfill/tests fastfill/v2/tests` (with `FASTFILL_ROOMGENBENCH_ROOT` set): 1119 passed, 114 subtests passed, no skips (after the H2 review repairs, `tests/test_review_repairs_r2.py`). End-to-end smoke in the session scratchpad: a 200-scene bounded
build (HSSD200, MultiScan, InternScenes_arkit, MansionWorld, IL3D_synthetic x 40)
passed `legacy_verify`, review build/verify, qualified build and
`multisource_verify --full-condition` (ok, no errors) plus the minimal view and
its verifier (478 swap objects, 1,134 floor declarations of which 91 snapped,
4 floating anchors skipped, one height-conflict flag); a tiny grid_residual run
with `minimal_form_p` 0.5 trained 6 steps, and `--resume state-step-4` reproduced
the step-6 weights exactly (max |diff| 0); `evaluate --projection full minimal`
on 50 cross-source rows bound `max_length` to `checkpoint_manifest.json`
(50 full / 17 rectangular minimal requests, 135 box-equivalent objects); one
room was exported by `predict --condition` and `--request` and assembled by the
real RoomGenBench assembler. These are code-path checks, not model quality.

Open decisions: K4 above; the literal HSSD200 `chair` rule matches 27 objects
while 2,982 HSSD200 `seat` objects are not marked (and `bed net` / `desk and chairs`
are); the selection score has no accuracy term (see "Selection metric").

## Main full-condition data and preserved release history

The new main derivative is `outputs/fastfill_v2/multisource-20261006/data`, with
portable package `/Volumes/harddisk/FastFill_v2_multisource_20261006` and server
copy `/home/jovyan/shanliantian/FastFill_v2_multisource_20261006`. Final build
counts, hashes and actual tokenizer eligibility belong to its new manifests and
preflight, not to a historical smoke or single-source pilot.

The original selected 16-family system remains: 11 training-room families expand
to 16 training source tags, SceneSmith and SpatialGen remain evaluation-only,
and three families supply auxiliary inputs. The new main preserves full parent
room/fixed/support/relation conditions and all target numbers. Scan2CAD's
estimated floor is marked unknown and its whole position vectors are masked;
13 targets outside the default size head range receive whole-size masks.
If any member loses complete position labels, all members of that
exchangeable group lose its tag, with logged fixed-identity fallback. IDs,
order, support, relations and targets remain unchanged. Main yaw keeps the 581
inherited valid semantic labels, with no SpatialLM geometric-yaw promotion.
The independent simplified-input view is `data-minimal-reference`; its XY
translation and geometric-axis yaw policy are separate experimental protocols.

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
The default axis front policy admits every upright finite source yaw with front
evidence and records its symmetry order (1 = MultiScan semantic front, 2 = axis
mod pi); the strict policy keeps semantic yaw for MultiScan only, masking the rest.
Legacy convention yaw is an explicit alternate policy, not verified semantic front. Old inferred `on` relations are omitted, rather than
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

## Simplified-input bbox ablations and historical SpatialLM pilot

Read the [direct bbox guide](../../docs/fastfill-v2-direct-bbox.md) for the exact
request and downstream boundary. The historical XY derivative has 9,601/539/624 scenes,
33,545 objects, all complete geometry labels. It is qualified from SpatialLM,
inherits parent splits, omits source height from condition, and supervises
pi-periodic **bbox axes** (not semantic front). Correspondence is fixed; no
exchangeable groups were reconstructed. Maximum scene size is 26 objects.
The source whitelist was applied before other geometry checks; these counts do
not prove other sources ineligible. This historical pilot is separate from the
new full-condition multi-source main. The new `data-minimal-reference` ablation
retains partial masks from multiple sources rather than demanding complete yaw
from every source, and explicitly keeps physical boundary/floor unknown.

```bash
python -m fastfill.v2.predict \
  --checkpoint "$FASTFILL_MODEL" \
  --request fastfill/v2/configs/direct_request.json \
  --room-size-semantics reference_extent \
  --output outputs/bbox-prediction-new.json \
  --export-dir outputs/bbox-handoff-new --device cuda
```

This writes geometry JSON, RoomGenBench SceneSpec/registry, GLB, SVG and proxy
diagnostics without a catalog or Host. Unknown height/support stays unknown;
the original RoomGenBench fixed scene/site/render harness needs separate scene
registration and height/metadata handling. Export success is not model quality.
Main inference uses `--condition` and its original qualification fields. Commands
below exercise the full-condition trainer and the offline asset-runtime contract.

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
  --front-policy axis --workers 4 \
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

The reviewed data is published at
[liantian/fastfill-v2](https://huggingface.co/datasets/liantian/fastfill-v2)
(public + gated with manual approval since 2026-10-07 by the user's decision; private at each earlier release), fixed
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
`evaluate.py`. Every checkpoint writes an Accelerate `state-step-<n>/` and a
deployable `model-step-<n>/`; `--resume <state-step-n>` continues into a new
output directory (see the audit-fixes section). Deployment checkpoints contain
backbone adapter/full weights where needed and decoder/heads.

## Geometry and exact losses

- Right-handed Z-up meters. Local +X is the canonical semantic front for the main
  protocol; sources without front evidence have yaw masked. The separate
  geometric-yaw ablation does not certify semantic front. `(w,d,h)` are full
  X/Y/Z lengths, invariant under yaw. Positions are bbox bottom-center.
- Position normalization uses input room origin and XY extent; missing height
  uses fixed 3m. Trusted floor-support z and requested fixed size dimensions are
  used directly. Unconstrained head tensors are retained separately for diagnosis.
- `size = positive_reference * exp(clamp(u,-10,10))`; reference is fixed `(1,1,1)`
  meters, with no validation/test statistics. Head residuals use `tanh` by default.
- L1 by default (`loss.position_type` / `loss.size_type`; `smooth_l1` with
  `loss.smooth_l1_beta` selectable) for normalized bottom-center and log size ratio. Each term uses
  only complete valid field vectors and at least one learned coordinate, then
  sum learned coordinate losses /3 / valid instance count. Missing labels are
  filtered before arithmetic. Fixed dimensions do not reduce the `/3` denominator.
- Yaw CE + SmoothL1 on **GT-bin** residual in training; argmax predicted bin in
  inference. Explicit justified symmetry orders choose the same equivalent angle
  by the joint weighted CE/residual minimum. Box symmetry never implies semantic
  front symmetry. Detached group cost uses position L1, plus log-size L1 only when
  every group member has a complete size label; no global
  matching, detection classification, objectness, NMS or repeated GT.
- `model.position_head = grid_residual` (round 2): logits over
  `position_grid` x `position_grid` cells (default 16) of normalized XY plus a
  tanh XY residual per cell in half-cell units; training uses CE on the GT cell +
  L1 on the GT cell's residual (`loss.position_cell` / `loss.position_residual`
  relative weights, formal 0.04 / 0.4) + the regression z share |dz|/3; inference
  takes the argmax cell + its residual. `predictions["position_normalized"]` is
  always decoded, so matching, regularizers, evaluation and hand-off are
  head-agnostic. `term_sums["position"]` is the decoded-position L1/3 under both
  heads; `position_cell` / `position_residual` / `position_z` are logged as well.
- Box symmetry (round 2): for `validity.size_axis_swap_allowed` objects the loss
  picks the detached joint minimum of size + yaw CE + yaw residual over
  k in {0..3} (yaw + k pi/2, size xy swapped for odd k); a fixed size coordinate
  pins the written order and leaves yaw only the box's pi symmetry (order 2); matching uses min(cost(sx,sy), cost(sy,sx)) per pair.
  `loss.yaw_reg` is capped at 2.0 in the main configs
  ([calibration, round-2 section](../../docs/fastfill-v2-loss-calibration-20261006.md)).

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
unavailable: the current Validator unconditionally emits unknown for each, with
no connected checker or external evidence-ingestion interface. Requiring these
validation levels therefore blocks commit. Time budgets
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
   full-condition complete-label view, condition information and documented
   sample exposure. Current text SFT explicitly rejects incomplete labels;
   do not silently erase the main corpus's partial masks. Simplified-input
   comparisons use separate manifests. Freeze asset/repair budgets for the
   main downstream runtime evaluation.
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
