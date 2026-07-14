# FastFill — task-conditioned one-pass room content generation (WP1)

FastFill replaces the open-ended designer/critic loop with **≤3 fixed model
calls per room**: one floor-level furniture call, one batched
support-surface call, and at most one batched semantic-repair call. Layouts
are checked by a deterministic validator (L0–L3 + support suite) and fixed by
bounded deterministic repair. See `worldedge_fastfill_执行方案_v1.4_定稿.md`
(repo parent dir) for the full plan; this package is WP1 (T1.1–T1.7).

**Boundary**: the WorldEdge growth loop (a collaborator's system) generates
the room *shell* — floor polygon, doors, windows, portals. FastFill only
consumes that as `RoomContext` and produces room *content*. `FloorLayout`
means "floor-standing furniture layout", **not** an architectural floor plan.

## Coordinate contract (frozen — every converter and generator obeys this)

| Item | Convention |
|---|---|
| Axes | **Z-up**, right-handed (Drake / scenesmith room frame) |
| Units | meters; angles in **degrees** |
| Room frame | origin at floor-polygon centroid, `z = 0` on the floor plane |
| Yaw | `yaw_deg`, CCW rotation about +Z, normalized to `[-180, 180)` |
| Facing | at `yaw_deg = 0` an object's *front* faces **+Y**; converters must normalize source conventions to this (verified visually via `tools/fastfill_data/visualize_sample.py`) |
| `dimensions` | `[width, depth, height]` = full extents along the object's **local** x/y/z before yaw |
| Furniture position | room frame, `position_xy` (+ `z`, 0 for floor-standing) at the **footprint center** |
| Small objects | support-surface **local** frame: origin at surface-polygon centroid, axes = parent furniture local axes, `z_local = 0` on the surface plane; `yaw_deg_local` relative to parent yaw |

Y-up sources convert with `(x, y, z)_yup → (x, -z, y)_zup` (right-handed,
`transforms.yup_to_zup_point`); a yaw about +Y maps to the same-sign yaw
about +Z. Per-source quirks (half-extents, cm units, bbox axis order) live in
the converters under `tools/fastfill_data/` and are unit-tested against the
real downloads in `<repo>/../data/`.

## Modules

| File | Role |
|---|---|
| `schema.py` | Frozen pydantic v2 contract: RoomContext → FloorLayout / SupportContext → SurfaceObjectGroup → RoomContentLayout, plus Violation / ValidationReport / ValidatedTaskEvidence / ProvenanceMeta / OutputBudget |
| `transforms.py` | Coordinate conversions (Y-up→Z-up, recentering, local frames) with hand-derived test vectors |
| `provenance.py` | `geometry_hash` / `layout_hash` for 铁律-4 cross-source dedup; split keys are house-first |
| `interfaces.py` | Protocols: LLMBackend, generator, asset resolver, context backend |
| `generator.py` | Line-codec API generation (1 floor call, 1 call per support surface, per-failed-surface semantic repair); prompts are training-parity (no system message, instruction + codec) |
| `validator.py` | Deterministic L0–L3 + SupportValidatorSuite; rebuilds task evidence independently — model self-claims never count |
| `repair.py` | Bounded deterministic repair (floor ≤20 steps, ≤10 per surface, identity on valid layouts) |
| `codec.py` | Compact training codec (cm / degree integers) with lossless-under-quantization roundtrip |
| `asset_resolver.py` | CanonicalAssetResolver (category-table bbox proxies + canonical top surfaces); real-library resolver is an adapter point |
| `context_builder.py` | v1 backend: build RoomContext from growing_world room data (hook T1.1) |

Data converters, the label sanitizer and SFT export live in
`tools/fastfill_data/` (repo root). Pipeline order: convert →
`deduplicate.py --contamination-list` (dedup #1 + hash closure on raw
hashes) → `sanitize.py` (validate/repair labels, restamp hashes) →
`deduplicate.py` (dedup #2) → `export_sft.py`.

## Running tests

Server: `uv run pytest tests/unit/fastfill tests/unit/fastfill_data -q`
Mac (no Drake wheel, so no `uv sync`): `.venv/bin/python -m pytest tests/unit/fastfill -q`

Converter tests read the real datasets from `$WORLDEDGE_DATA_DIR`
(default `<repo>/../data`) and skip cleanly when a dataset is absent.

## Quick commands (data pipeline + smoke)

```bash
# Convert (each writes <out>.jsonl + <out>.stats.json; --limit optional)
uv run python tools/fastfill_data/convert_3d_synthplace.py --out out/synthplace.jsonl
uv run python tools/fastfill_data/convert_m3dlayout.py --split 3dfront --out out/m3dlayout.jsonl
uv run python tools/fastfill_data/convert_il3d.py --out out/il3d.jsonl
uv run python tools/fastfill_data/convert_mansionworld.py --out out/mansionworld.jsonl
uv run python tools/fastfill_data/convert_scenesmith.py --out out/scenesmith.jsonl

# 铁律 4: cross-source dedup by geometry_hash (priority wins collisions)
uv run python tools/fastfill_data/deduplicate.py \
  --in out/*.jsonl --out out/deduped.jsonl \
  --priority m3dlayout,il3d,3d_synthplace,mansionworld,scenesmith_scenes

# SFT export (floor_sft.jsonl + surface_sft.jsonl; NC license excluded by default)
uv run python tools/fastfill_data/export_sft.py --in out/deduped.jsonl --out-dir out/sft

# Visual audit (front arrows = facing-convention check)
uv run python tools/fastfill_data/visualize_sample.py --in out/m3dlayout.jsonl --index 0 --out out/viz.png

# WP1 acceptance: 20 real-API rooms, fixed-call budget, validate + repair
uv run python scripts/fastfill_api_smoke.py --runs 20 --room-type bathroom \
  --task "brush teeth at the sink" --config scenesmith/growing_world/llm_config.json \
  --out out/fastfill_smoke

# Growth loop with fastfill hook (context-only without an LLM config)
uv run python -m scenesmith.growing_world.grow run --task "..." --out out/world \
  --expansions 3 --content-mode fastfill
```

## Per-source facing calibration (measured, 2026-07-10)

The contract front is +Y at yaw 0. Wall-backed-category audit (do objects
face the room?): 3d_synthplace is NOT homogeneous — calibrate by
`upstream_dataset`: Holodeck-Synth 100% raw → no offset; 3D-FRONT (incl. the
48 source-less all-UUID scenes) 2.5% raw / 97% after **+180°** → +180° applied
(full-corpus audit 2026-07-14, 22,393 Holodeck vs 11,895 3D-FRONT wall-backed
objects). Separately, 3D-FRONT bbox axis order is a per-record HWD/DWH mix
(~73/27), so 3D-FRONT floor records are ISOLATED at export (kept for dedup +
contamination closure) until per-record mesh-vertex resolution. m3dlayout 5%
raw → **+180° applied** (95% after); il3d 8% raw → **+180° applied** (92% after);
scenesmith already contract-aligned (mesh canonicalization front=+Y);
mansionworld is inconsistent PER CATEGORY (cabinets 3% vs toilets 100% —
objaverse assets lack a shared canonical front), so no offset is applied and
samples carry `notes="yaw_facing=unverified_per_asset"` — fix via per-asset
`pose_z_rot_angle` canonicalization is WP2 work.
