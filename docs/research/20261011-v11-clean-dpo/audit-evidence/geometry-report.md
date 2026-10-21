# V1.1 geometry and RoomGenBench audit

Audit performed 2026-10-11. Scope: full reads of postprocess, scene/protocol, validator/anchors/interface, RoomGenBench v1.1 entry, v2 direct-layout/geometry/assembly adapters, and upstream RoomGenBench assembler. The adjacent evaluator's generation/judge/score contract was also inspected, with its complete audit owned by a separate agent. Exact full-read file hashes and line counts are in `geometry-coverage.json`; partial reads are explicitly distinguished.

## Result

No confirmed P0/P1 was found in the current authoritative full-cohort geometry scoring path. Two confirmed P2 defects were found in the legacy RoomGenBench showcase path and corrected locally with regression tests. They are not executed by the currently frozen 3-model × 4-split evaluation jobs. No deployed frozen source was changed by this agent.

### P2 G1: invalid model support links could hang the showcase fallback

Original `roomgenbench_v1_1.py:422-442` parsed an invalid output, retained its parsed subset and filled only missing IDs. A two-node support cycle therefore survived `bad_support_ref`. `repair_v1_1` calls `interface.snap_inside`, whose `_push` subtree walk at `interface.py:302-305` assumes a validated acyclic graph and would grow forever for an outside cyclic object.

Reproduction: `geometry-repro.py` constructed `a.on=b, b.on=a`, with both boxes outside the room. The parser correctly returned `bad_support_ref`; the previous showcase recovery left this graph unchanged, and the repair subprocess timed out after 4.027 seconds. Evidence: `geometry-cycle-input.json`, `geometry-repro-result.json`.

Fix: `roomgenbench_v1_1.py:74-94,471-472` discards the entire malformed prediction for the display fallback and builds fresh category-prior boxes from request inventory, boundary and declared support metadata. Declared supports are first checked for missing/self/cyclic references. Reference box sizes, positions and yaw are not used. Each handoff now includes `generation_receipt.json` and its summary distinguishes `model`, `reference`, and `category_prior_fallback`; a fallback never sets `repaired_valid` true. Raw invalid output is retained. The authoritative scorer already rejects parse errors rather than taking this showcase fallback, and its behavior was not changed.

### P2 G2: mesh recovery fabricated generator-success receipts

Original `roomgenbench_v1_1.py:49-55,92-98` wrote `{asset_key,method,status:"ok"}` for missing sidecars regardless of the source mesh's generation provenance. This made downstream `_asset_parts` and `generated_mesh_success` able to interpret recovered or arbitrary meshes as successful generated assets.

Reproduction: 184 synthetic box GLBs without any generation receipt were given to the old fast path. It created `status=ok`; the real downstream `_asset_parts` loaded/fitted the cube and returned status `ok`. Evidence is in `geometry-repro-result.json`. Local existing RoomGenBench assets were inspected read-only: 184 sidecars currently have only the same three fields. This audit did not rewrite those real files.

Fix: `roomgenbench_v1_1.py:44-71,102-105,134-158` labels absent/unprovenanced and recovered-web meshes `fallback`, records provenance, and downgrades the exact old fabricated three-field `ok` receipt shape. Rich existing generator receipts, including failures, are retained when an existing mesh is reused. Newly extracted web meshes remain fallback even if a prior sidecar had another status. A malformed existing sidecar remains intact, emits a warning, and is rejected per asset by the downstream adapter; one bad receipt no longer disables every scene's assembly. A malformed old receipt encountered during new extraction records its error class in the new fallback receipt.

## Verified contracts and scientific limits

- `apply_v1_1` replaces sizes/positions/yaws/parents with predictions; current postprocessing obtains those fields from predicted placements, not reference geometry. It uses prompt-side room/fixed/constraint information and category labels.
- Wall repair/check additionally uses `room.objects[].anchor`, a field not serialized in the V1.1 prompt. The data audit independently scanned all four live evaluation splits: only floor/object anchors occur, with wall/ceiling/fixed/null counts zero. Therefore this privileged wall branch does not affect current authoritative results. RoomGenBench wall handling uses its requested `place_id=wall` declaration and must be described as deterministic postprocessing.
- The solver can abandon previously held constraints when no collision-free placement satisfying them is found (`postprocess.py:410,524`). It does not guarantee every room becomes feasible. The authoritative judge rechecks every constraint and boundary/fixed/support condition; unresolved rooms fail acceptance. The module docstring's blanket guarantee is stronger than the implementation and should not be repeated as a research claim.
- `zero_collision_accepted` is a box-level proxy: rugs, items under 6 cm high, direct support pairs, tucked seats/tables and contained boxes have deliberate exclusions. Severe collisions are a subset of main collisions. It does not certify mesh intersections, stability, contact physics or useful room design.
- Support checks require height agreement and child-center containment within tolerance, not full child-footprint support or stability. The final snap can reveal remaining failures; the evaluator recomputes them.
- Evaluation's `size_realistic` is a reference-relative lower-size guard, not a complete category-realism model or symmetric size-range constraint. The reference is used for evaluation, and separate size IoU is reported. Improvements in zero-collision acceptance alone do not prove realistic furnishing quality.
- Full five-scene conversion roundtrip preserves all 313 objects, original order, source position/yaw (modulo 360), doors/windows and declared wall count; dimensions retain the existing 1 cm minimum. `--gt-only` additionally serializes through the existing two-decimal protocol and is not an exact raw-float export. No size-quantization contract was changed by this audit.
- The v2 handoff assembly correctly distinguishes generated/fallback/placeholder meshes and reports support/physics/host commit as unchecked. Mesh fit uses anisotropic scaling and yaw/tip adjustment to the predicted box; successful assembly cannot be presented as native mesh-quality or room-feasibility proof.

## Validation

TDD: initial 10 focused tests failed against the old code (the old hanging main-path E2E was deliberately excluded until repaired). After fixes, the new real-path E2E runs the showcase main through actual handoff export, reference check and GLB creation with only model generation stubbed. Additional tests exercise synthetic mesh fit, web-GLB extraction, reference roundtrip, input immutability, support-graph failures, and malformed receipts. Independent reviewer found the one-bad-sidecar regression; three RED cases were added and repaired.

Final dedicated regression suite: **16 passed**.

The environment has neither pytest-cov nor coverage installed. A dependency-free CPython `sys.settrace`/`dis.findlinestarts` measurement reports **362/417 executable lines (86.81%)** for the full modified showcase module and **100% for both new helpers**. This is line coverage, not branch coverage. Script and output: `trace-geometry-tests.py`, `geometry-line-coverage.json`.

All six existing module self-check suites passed: `fastfill.scene`, `fastfill.validate`, `fastfill.anchors`, `fastfill.interface`, `fastfill.v1_1.scene_v1_1`, `fastfill.v1_1.postprocess`. `git diff --check` also passed.

Modified production scope is only `fastfill/v1_1/roomgenbench_v1_1.py`; dedicated tests are `fastfill/v1_1/tests/test_roomgenbench_recovery.py`. Current frozen authoritative evaluation remains unchanged. Root should retain these fixes in the next independently identified source snapshot and continue to distinguish the old showcase from scientific full-cohort results.
