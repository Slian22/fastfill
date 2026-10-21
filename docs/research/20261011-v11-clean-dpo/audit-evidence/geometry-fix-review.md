# RoomGenBench patch independent review

Compared `fastfill/v1_1/roomgenbench_v1_1.py` against its deployed frozen copy in `/tmp/fastfill-v11-automation-20261011/source.tar.gz`. Reviewed new recovery tests and followed handoff assembly into `fastfill/v2/roomgenbench.py::_asset_parts` and `_objects`, plus geometric repair into `postprocess.py` and `interface.py::snap_inside`.

The patch correctly labels recovered/unprovenanced mesh assets as fallback instead of inventing generator success. The known legacy three-field `status: ok` record is downgraded. Existing genuine generator records remain intact. The downstream assembler carries fallback status into `geometry_kind` and `generated_mesh_success` rather than turning renderability into generation success.

Malformed model placements now discard the whole parsed subset, avoiding retained cyclic support graphs. Fallback sizes/positions/yaws come from category priors, boundary, and declared anchor/support metadata, not reference object geometry. Invalid declared support graphs fail before geometric repair. The generation receipt retains parse errors and `model_generation_succeeded: false`; fallback geometry cannot pass the model's repaired-valid summary field.

One new P2 regression was reproduced during review: a single malformed existing sidecar raised during receipt stamping and caused the caller to disable the entire asset directory. The worker fixed it before this review closed. Malformed JSON and invalid UTF-8 are now preserved and diagnosed, with per-asset downstream rejection; they do not disable other assets. A newly recovered mesh gets a fallback receipt plus the prior receipt error.

The newly added benchmark reference roundtrip check respects the existing V1.1 1 cm minimum dimension contract. The converter/training/evaluation quantization contract was not changed to satisfy a raw-dimension equality assertion.

Validation after the fix: 16 recovery tests and 11 audit-report tests passed together (27 total, 5.68 s). `git diff --check` passed for the reviewed paths. No remaining confirmed P0/P1/P2 finding in this patch. No server modification was made by this reviewer.
