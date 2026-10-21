# Renderer handoff

Implemented only `fastfill/comparison/render_assets.py` and `tests/test_render_assets.py` in the repository.

44 tests pass. Combined unit tests + actual Blender smoke statement coverage: **91%** (unit-only 66%).
Two RGB 512x512 images completed, CPU 4 threads, 8 samples. The user subsequently authorized one additional dev perspective: 32 samples with CPU OpenImageDenoise; completed in 7.04 seconds including startup and coverage tracing. No API calls or metric scores.
Earlier `rendered/ledger.json` and `views/ledger.json` are failed **pre-render** receipts; keep them.
The successful receipt is `smoke-final/ledger.json`. `summary.json` records paths, checksums, assets, and limitations.

## Additional authorized clean dev smoke

Use this image for the parent's API capability check:
`/tmp/unknownsize-render-smoke/smoke-clean32/0000-66c0df2bfe17/candidate/perspective.png`

Receipt: `/tmp/unknownsize-render-smoke/smoke-clean32/ledger.json`.
Actual wall time: **7.04 s** for one 512x512, 32-sample, denoised perspective, CPU 4 threads, including startup and coverage tracing. Geometry is clearly visible; fine bed texture is softened. The old images and both failed pre-render receipts remain untouched. Total successful images: three, including this explicitly authorized extra dev smoke.

`--denoise` now explicitly enables OpenImageDenoise and is recorded in the ledger. The default remains off to preserve earlier runs. For future frozen manifests, `--resolution 1023 --top --samples 32 --denoise` produces both views with identical settings. Samples 64 is also supported. Neither future preset has been benchmarked or frozen here. Rough estimate for this same two-asset room at 1023: 10-20 s per view; larger scenes may differ.

````sh
/Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --threads 4 --python-exit-code 1 \
  --python /Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/comparison/render_assets.py -- \
  --case /tmp/unknownsize-render-smoke/case.json \
  --output /tmp/unknownsize-render-smoke/clean32-replay \
  --threads 4 --samples 32 --resolution 512 --denoise
````

## Images

- Perspective: `/private/tmp/unknownsize-render-smoke/smoke-final/0000-66c0df2bfe17/candidate/perspective.png`
- GPT top: `/private/tmp/unknownsize-render-smoke/smoke-final/0000-66c0df2bfe17/candidate/top.png`

## Safe command from any working directory

The replay output must be fresh. This command is provided for review; do not replay merely to inspect the existing images.

```sh
/Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --threads 4 --python-exit-code 1 \
  --python /Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/comparison/render_assets.py -- \
  --case /tmp/unknownsize-render-smoke/case.json \
  --output /tmp/unknownsize-render-smoke/replay \
  --threads 4 --samples 8 --resolution 512 --top
```

For the parent-selected future visual manifest, use `--manifest /absolute/path/manifest.json` instead of `--case`, and `--resolution 1023 --top` for both views. Freeze samples and the remaining protocol before evaluation. The smoke's 8 samples are visibly noisy. No formal quality preset has been selected here.

## Case / manifest schema

`case.json` here is a complete portable example. A manifest is `{"cases": [case, ...]}`.
Every case requires:

- `uid`, `source` (`OptiScene_holodeck` or `IL3D_3dfront`), `boundary` as a simple polygon of [x,y] vertices, positive `height`, and `fixed` list.
- `expected_ids`: explicit complete inventory. Every layout must contain exactly these IDs once.
- `layouts`: dictionary of named canonical lists, e.g. `{"rawGT": [...], "sft_armA": [...], "sft_armB": [...]}`. Each placement requires `id`, positive `size=[sx,sy,sz]`, `pos=[x,y,z]` at bbox bottom-center, and `yaw` in radians. All coordinates are metres, RH Z-up.
- `asset_mapping`: crosswalk rows with `prompt_id`, `asset_id`, `mesh_path`, `source`, optional `layout_object.scale` (native signs retained, positive magnitudes superseded by fitting). A dictionary keyed by prompt ID is also accepted.
- Each layout supplies its own dimensions. GT geometry in the crosswalk is never used as a prediction fallback. Fixed furniture also requires a crosswalk; only explicitly marked door/window/wall/column/beam structures can be procedural context.

Relative `mesh_path` values resolve against the input JSON folder. `--asset-root ROOT` relocates archived mesh paths to `ROOT/asset_id/<archived OBJ basename>`. Paths are not guessed from source.

The optional archived adapter requires all of `--metric-inputs FILE --asset-map FILE --uid UID --stage STAGE`, selects exactly one case, and uses raw candidate layout by default (`--repaired` is explicit). This adapter was not used to render old test predictions.

`--dry-run` runs in standard Python without bpy, validates cases/resources, and emits a readiness/rejection ledger. Missing objects, assets/materials/textures, nonfinite geometry, and unsupported sources have receipts. Incomplete cases do not publish partial images. Existing ledger output directories are rejected to prevent overwrites.

## Axis implementation / review targets

Native scale signs precede Holo `(-z,x,y)` / Front `(z,x,y)`. Fit the transformed union AABB at its local bottom-center to the current layout size; apply canonical yaw and world position last. No raw rotation is reapplied. Holo reflections and any negative native scale flip complete face corner order; normals use inverse transpose. OBJ imports use explicit Blender `forward_axis=Y, up_axis=Z` to preserve already-transformed coordinates.

Cameras and light placement depend only on polygon extent and room height. All layouts get the same floor, cutaway walls, fixed context, seed, material import, color management, and settings. Both camera specs are in the ledger.

The MTL filename repair is restricted to `f350c1ba-8040-4750-9e6e-f45cfbf916a4`: missing `model.mtl` resolves to `raw_model.mtl` only after resource validation. Temporary local texture symlinks (copy fallback) avoid Blender's absolute-path import bug. Explicit pixel access validates Blender's lazily decoded textures. No raw meshes, MTLs, or textures are modified.

## Quality limits

8-sample noise is visible. The selected bed has 268 identical surface-face duplicates in the raw OBJ; they remain unchanged and their effect on the render is unproven. Blender ignores `map_Ka` but loads `map_Kd`. No claim of original Isaac/native-scale shading fidelity is made. Holo padding and nonuniform bbox fit can change asset proportions. Procedural openings are solid context objects, not architectural holes. Final visual preset and asymmetric semantic-front approval belong to the queued independent review and the parent's frozen manifest decision.

Reference: Blender 5.2 OBJ import API: https://docs.blender.org/api/5.2/bpy.ops.wm.html#bpy.ops.wm.obj_import
