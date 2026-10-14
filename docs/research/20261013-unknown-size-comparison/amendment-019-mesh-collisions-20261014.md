# Amendment 019 — mesh-level collisions on the rendered rooms (EXPLORATORY, descriptive)

Locked 2026-10-14 (UTC), before any mesh-collision output exists. Prompted by the user: real furniture does not fill its bbox (an L-shaped sofa occupies half of its square box; chairs go under a tabletop), so a box overlap is not a collision. Nothing in amendments 005–018 changes.

## Population and systems
- **Rooms:** the amendment-011 visual run `visual-final-selected-002`, i.e. its asset-eligible rooms (paper strata, 3D-FRONT / Holodeck reference meshes).
- **Systems:** every system rendered there (GT reference, J, P, L, O2, O3, O4, O6, O7).
- **Placement:** each object's reference mesh is placed exactly as `render_assets.py` places it. That is the same geometry contract, the same bbox fit, yaw and position, using that run's asset mapping and case files. No new placement code.
- Rooms or objects without a mesh are reported as missing, never filled in.

## Metric
- **Candidate pairs:** two distinct objects whose world AABBs intersect. Parent–child pairs are excluded, because a supported object touches its support by design; objects on the same support are included.
- **Solids:** each placed mesh is voxelised at a 2 cm pitch, and interior holes are filled (`scipy.ndimage.binary_fill_holes`) to approximate a solid. A mesh that is not closed stays a shell; the counts of such meshes are reported.
- **Interpenetration volume:** shared voxels × 8 cm³.
- **Collision:** a pair mesh-collides at tolerance τ when its interpenetration volume exceeds τ, for τ ∈ {125, 500, 2000} cm³. The primary τ is 500 cm³.
- **Room level:** a room is mesh-collision-free at τ when no pair collides.

## Reporting
- **Per system:**
  - the share of rooms that are mesh-collision-free;
  - the share of the frozen checker's main/severe pairs and of the A4 box pairs that also mesh-collide;
  - the share of mesh-colliding pairs that the checker misses.
- **For GT:** the A4 categories of amendment-017 evidence (tucked, box artefact, small overlap, contained, genuine), each against the mesh verdict.
- **CIs:** house-cluster bootstrap CIs on paired differences vs P.
- **Status:** descriptive; no decision rule.

## Addendum A (locked before the full run; implementation details from the tool review)

**Disclosure.** Tooling agents ran the tool on 20 and 40 rooms to measure timing and check placement. They reported mesh-collision counts only as tooling diagnostics (for example, 3 of GT's 7 colliding pairs at τ = 500 in 20 rooms are mostly a flush contact layer). No system-level comparison was seen.

1. **Triangle sampling** is ≤ 0.5 cm, i.e. pitch/4. Some assets have holes (open boundary edges); at 1 cm sampling the hole-filling leaked through them and under-counted solids by up to 43%. Voxelisation is chunked; the results are identical.
2. **Systems.** Every system rendered in the visual run is reported, including O1a and O1b (labelled).
3. **"Shell only" column.** It means the fill added no voxels. This is a voxel-level property and depends on the system's sizes; it is not mesh watertightness.
4. **Flush contact.** Two flat faces in contact can share one 2 cm voxel layer, about 2 cm × the contact area. In that case a flush contact is counted at the primary τ = 500 cm³. The primary metric stays exactly as locked.
   - Added diagnostic, labelled "not pre-registered": `deep_cm3`, the interpenetration volume after removing the voxels that lie on both objects' surface shells.
   - No claim may rest on the diagnostic alone.
5. **Rooms and outputs left out.**
   - Rooms whose fixed items have no asset are reported as missing.
   - Each system's outputs that could not be placed are counted per system.
   - After the render completes, the run is checked against the render ledger (`chunks/*/render/ledger.json`).
