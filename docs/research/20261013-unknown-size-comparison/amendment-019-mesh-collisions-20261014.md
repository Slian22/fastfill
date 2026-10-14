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
