# VIS-01 independent read-only review — 2026-10-13

**Verdict: NO HIGH findings in the reviewed persistent-drift boundary fix.** No CRITICAL or HIGH defect was identified. The implementation meets the scoped pre/post resource-binding and paired-failure requirements below. This is an exact-source review for the parent's deployment decision, not deployment execution or an assertion of immutable asset reads.

Production implementation and test hashes remained unchanged throughout this review. `resource_binding.py` is **absent** in this workspace; the fix lives in `visual_pipeline._resource_failures`. The author's fix report explicitly says no helper was necessary, and the pipeline does not import that missing file. No nonexistent helper has been certified.

## Exact reviewed source

| File | SHA-256 |
|---|---|
| `fastfill/comparison/visual_pipeline.py` | `099ed3467bd3312665fd446d21304f9df176a04998d1c40b697b0e1fd974b29d` |
| `fastfill/comparison/resource_binding.py` | Absent; not part of executable implementation |
| `fastfill/comparison/tests/test_visual_resource_binding.py` | `af411bc451f5471acdab3c8e0fbd934e82b6decc5a21d6e4004b4db60f466370` |
| `fastfill/comparison/visual_cases.py` | `dd9b51883357aab474d5bc46c7986940c61a9011f8aa6fdad2667f0b27911a21` |
| `fastfill/comparison/render_assets.py` | `eaf4e127aa08c985d9a53ef318cb61b892d60a7a49ff49b9fc9557aee3340583` |
| `fastfill/comparison/eval_data.py` | `ddd0ca5729739f4603db1e0eb5e53f851b1f02539563b663d944c418f087104b` |
| `fastfill/comparison/fid.py` | `b6c62df6f4a603e43e586814561bff811357c27b855f028515b3f58e5031a36d` |

Read-only review: no production or repository test files were edited. The legacy `test_visual_pipeline.py` fixture update was left to the other worker; it changed concurrently and is not part of this review's production attestation. Adversarial source/assets/images were generated only in temporary directories. No SSH, deployment, API request, real asset mutation, package installation, or full source regeneration was performed.

## Trace and findings

- **Freeze binding** — `visual_pipeline.py:123` requires `resources.json` in the validated reference manifest. The existing snapshot validator checks the complete file inventory and hashes. `evaluation._read` hashes the exact consumed resource-book bytes. Missing, unbound, mismatched or malformed books reject before creating output or launching rendering.
- **Exact UID and consumed asset set** — `visual_pipeline.py:173` invokes the actual renderer planner through the freeze's `_planner`, gathering resources from fixed furniture and every GT/J/P layout. The case's hash-bound mapping carries UID/source/prompt identity; the global resource book is keyed by the digest of `[resolved_mesh_path, asset_id]`, intentionally allowing shared assets across UIDs. Full records must match: mesh identity/path/hash, MTL dependency paths/hashes, texture dependency paths/hashes, and material overrides. Foreign keys, wrong identities, missing dependencies and same-byte/different-resolved-path substitution fail. Procedural structures correctly require no unused external asset.
- **Before and after checks** — `visual_pipeline.py:393` checks attached rooms before launching and submits only those that pass. It checks submitted rooms again after chunks finish with a fresh cache. The independent shared-asset probe confirms one inspection per unique resource per pass, repeated in the next pass. A shared mismatch fails every dependent UID.
- **Entire GT/J/P failure** — `visual_pipeline.py:281` lets resource failure override successful render receipts and empties the authoritative image map. This applies to the entire room, not just one model arm. Fixed-resource, image-validation, attachment and render failures retain their existing failure behavior.
- **No authoritative image/FID/API leak from persistent post-render drift** — `visual_pipeline.py:335` omits failed rooms from runnable requests and does not copy their PNGs or create their prompts. Both selected model-arm failure/mapping entries remain. `visual_pipeline.py:374` includes the failed UID in FID with empty images and the declared failure; the downstream FID inspector marks it noncandidate. The independent test checked actual consumer output and filesystem absence of the failed room's API PNG directories/prompt, not just counts.
- **Full failure denominators** — cases, ledger, FID frozen inventory and selection retain original UIDs. Failed selected slots are not replaced with unselected successes. Mixed resource/render/attachment failures, timeout and all-failed tests pass. Filtering successful images does not reduce the requested/frozen cohort.

No production weakening or compatibility bypass is required for these results.

## Old frozen reference compatibility

Independently read and validated the existing `data/visual-reference-v1` snapshot without modifying it. Its reference-manifest SHA-256 is `dda638b9bf6ea292fa9dbcec5b731cb5a7fc885ef4331a3438a741e6b306573a`, matching the pre-fix audit. Its bound `resources.json` hash is `9a6e8b278c30d2b0f6bdcc3963dbdf39a28da87accd4202e6116f5d8e392edea`.

The snapshot has **1,009 frozen cases, 1,005 eligible cases and 1,945 resource records**. Every resource-book key and record schema was checked. All **6,018 consumed mappings** in eligible cases have the exact case UID/source and resolve to an existing frozen resource key; **883 procedural objects** correctly require no external resource. The old snapshot is already compatible; it needs no schema migration. References lacking the resource book were never valid freezes under this producer contract. No new certification of the real OBJ/MTL/texture bytes is claimed: this review checked the frozen snapshot, mapping coverage and schema, not all 3.6 GB of external assets.

## Independent verification

Environment: `/Users/slian/miniforge3/bin/python`, Python 3.12.9, pytest 9.1.1. No environment changes.

1. `python -B -m pytest -q -p no:cacheprovider fastfill/comparison/tests/test_visual_resource_binding.py fastfill/comparison/tests/test_visual_cases.py fastfill/comparison/tests/test_render_assets.py fastfill/comparison/tests/test_fid.py --tb=short` — **191 passed**, 11.57 s. This includes all 41 new binding tests and the freeze/renderer/FID regressions. The other worker owns the legacy fixture repair and full suite.
2. `python -B -m pytest -q -p no:cacheprovider /tmp/optiscene-vis-independent-20261013-cths629b/test_adversarial_binding.py --tb=short` — **10 passed**, 2.66 s. These independently authored temporary probes cover same-byte texture symlink substitution before/after render, correct resource under a wrong key, unsealed UID/asset/path/source mapping tampering, cache freshness/sharing, authoritative FID/API exclusion, and a deliberately characterized transient-drift limitation.

The persistent post-drift probe retained all three frozen/selected UIDs: one declared failure, two successes, all six opaque API identities, two failed API entries, and four runnable API requests. The failed UID had zero FID image entries and no API PNG directory or prompt. All 18 raw mock-render chunk PNGs remained diagnostic; only eight PNGs for the successful J/P cases were copied into the API bundle.

## Limitations and deployment interpretation

**Boundary checks are not an immutable render snapshot.** The independent transient probe changed a synthetic texture during mocked rendering and restored it before postcheck; the pipeline accepted the outputs. This is consistent with the fix's declared scope. It does not prove bad real Blender pixels, because rendering was mocked, but it demonstrates that the gate cannot certify every read. Continuous changes inside a cached pass are likewise outside its guarantee. If a deployment requires detection of transient restored changes or attestation of every Blender read, this review does not establish that stronger property.

**Raw diagnostic artifacts remain.** Chunk PNGs and nested receipt paths are not deleted after failure. Final FID/API consumers must use the final manifests and authoritative `images`/`failure` fields, not glob chunk images or promote nested receipt paths. Persistent boundary drift does not enter the runnable API bundle or FID candidate set in the tested paths.

No real Blender run, FID feature computation, judge/API call, production end-to-end render, I/O performance benchmark, or full external-asset-byte reaudit was performed. No claim is made about geometry/material correctness or metric quality. The supplied manifest is trusted frozen provenance, not cryptographic authentication against an actor able to replace the entire snapshot consistently.

The **NO HIGH** result applies only to the enumerated hashes and this pre/post persistent-drift contract. Parent should combine it with the other worker's full-suite result when deciding whether to freeze/deploy. A later production revision or newly introduced helper requires its own delta review.

Machine-readable evidence, commands, source hashes, environment, probe source and limits: [visual-binding-independent-review-20261013.json](visual-binding-independent-review-20261013.json).
