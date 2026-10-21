# SceneEval (navigability / accessibility metrics only; full note in ../sceneeval-2503.14756.md)
- Authors: Hou In Ivan Tam, Hou In Derek Pun, Austin T. Wang, Angel X. Chang, Manolis Savva. 2025 (WACV 2026 per search). https://arxiv.org/html/2503.14756v3
- NAV: project scene to a 2D occupancy mask; NAV = largest connected free-space area / total free space (connected components). ACC: a VLM marks each object's functional sides; ACC = unoccupied pixels / total functional area (best side if several). Agent radius, mask resolution, height slice and functional-region size not given in the part read (unverified).
- Results (Table 4, NAV / ACC %): ATISS 99.83 / 85.55; DiffuScene 99.44 / 81.86; LayoutGPT 99.99 / 47.26; InstructScene 98.99 / 77.47; LayoutVLM 99.65 / 85.19; Holodeck 99.60 / 89.65.
- Relevance: NAV saturates (~99-100%) and cannot rank methods, while front-access ACC ranges 47-90% -> per-object access-region checks are the informative signal; no agent size, so not robot-calibrated.
- Tags: sim-benchmark
