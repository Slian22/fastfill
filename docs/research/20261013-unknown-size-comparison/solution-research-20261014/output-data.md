## Output representation and decoding for LLM layout generation

Sources were checked by WebFetch/WebSearch on 2026-10-14. Items marked **[unverified]** come from secondary pages or were not read in the full text.

### Methods table

| Method | Year | Link | What it fixes | Reported numbers | Retrain? | Fits FastFill one-call text? |
|---|---|---|---|---|---|---|
| LayoutGPT | 2023 | [2305.15393](https://arxiv.org/abs/2305.15393) | CSS-style in-context layout, values in px normalised to a 256 max | 3D-FRONT out-of-bounds 43.26% bedroom / 73.58% living, vs ATISS 49.88 / 83.02 | No (ICL) | Yes |
| LLplace | 2024 | [2406.03866](https://arxiv.org/abs/2406.03866) | Llama-3 SFT; JSON with 2-decimal coordinates; sizes are given as input | Average FID 126.8, overlap rate (OOR) 0.264, vs LayoutGPT 152.7 / 0.283 | SFT | Yes (same as ours) |
| OptiScene | 2025 | [2506.07570](https://arxiv.org/abs/2506.07570) | JSON plus a reasoning block; SFT then 2-stage DPO | OOR 0.049→0.023 and usability 28%→50% from SFT to DPO-1+2 (Table 6). No out-of-bounds metric. Max objects per room not stated | Yes | Yes (our template) |
| ReSpace | 2025 | [2506.02459](https://arxiv.org/abs/2506.02459) | 2-decimal SSR; random object permutation in training; **one object per call, conditioned on the partial scene**; best-of-N | Full scene VBL (×1e3) 341.8, vs MiDiffusion 414.5 and ATISS 739.8. Its collision term MBL is *worse* (181.6 vs 87.1). Out-of-bounds and VBL fall as N goes 1→8. Trained on rooms with 3–50 objects | SFT + pref | No (N calls) |
| Pix2Seq | 2022 | [2109.10852](https://arxiv.org/abs/2109.10852) | Quantised coordinate tokens (1000–2000 bins). Random order works best. Nucleus sampling (p=0.4) beats argmax. "Noise object" augmentation targets early EOS and duplicates | AR@100 55.4→61.7 with augmentation; AP roughly flat (43.7→44.7) | Yes | Yes (detection analogue) |
| SpatialLM | 2025 | [2506.07491](https://arxiv.org/abs/2506.07491) | Code-script boxes. Coordinates quantised to 1280 bins at 2.5 cm **[unverified, Voxel51 docs](https://docs.voxel51.com/dataset_zoo/datasets_hf/spatial_lm_dataset.html)** | No collision metric | Yes | Yes |
| LayoutNUWA | 2024 | [2309.09506](https://arxiv.org/abs/2309.09506) | 2D layout written as HTML code with masks to complete | FID gains above 50% on some 2D datasets | Yes | Yes |
| Number Token Loss | 2025 | [2411.02083](https://arxiv.org/abs/2411.02083) | Cross-entropy treats digits as unordered classes; NTL adds an Lp/Wasserstein numeric term | Better numerical accuracy on math tasks; no layout results | Loss term only | Yes |
| LayoutVLM | 2025 | [2412.02193](https://arxiv.org/abs/2412.02193) | Model outputs **poses + relations as code**, then a DIoU collision loss plus boundary projection (400 Adam steps). 5–10 calls per 40-asset scene (grouped). Up to 80 assets | Collision-free / in-boundary 81.8 / 94.9, vs LayoutGPT 83.8 / 24.2 and Holodeck ([2312.09067](https://arxiv.org/abs/2312.09067)) 77.8 / 8.1. Fine-tuned LLaVA in-boundary: numeric poses 18.5 vs their representation 70.4 | Optional | Yes if the solver runs post-hoc |
| ATISS | 2021 | [2110.03675](https://arxiv.org/abs/2110.03675) | Set-based autoregression, permutation-invariant | Out-of-bounds 16–17% ([MiDiffusion T1](https://arxiv.org/abs/2405.21066)). Col_obj 0.25–0.59 ([PhyScene T3](https://arxiv.org/abs/2404.09465)) | Yes | No |
| MiDiffusion | 2024/WACV26 | [2405.21066](https://arxiv.org/abs/2405.21066) | Mixed discrete/continuous diffusion. Its furniture-arrangement mode takes labels and sizes as given, which matches arm P | Arrangement mode: out-of-bounds 7.3–8.3% vs DiffuScene-FA 8.0–17.8%; box IoU 0.55–0.98% | Yes | No |
| DeBaRA | 2024 | [2409.18336](https://arxiv.org/abs/2409.18336) | Score-based model. Input is floor plan + categories; output is size, position and rotation, which matches arm J. Llama-3 proposes object sets and the model self-scores them | Living OBA 167.8 m², vs DiffuScene 341.1 and LayoutGPT 2913.6. No collision metric, and the authors admit collisions | Yes | No |
| PhyScene | 2024 | [2404.09465](https://arxiv.org/abs/2404.09465) | Sampling-time guidance on 3D IoU, layout and reachability | Collision guidance alone: Col_obj 0.200→0.111, but out-of-bounds rises 0.240→0.354. All three guidances together: 0.191 / 0.219. 3D-FRONT GT Col_obj is about 0.21 | Guidance needs no retrain (but needs a diffusion model) | No |
| Forest2Seq | 2024 | [2407.05388](https://arxiv.org/abs/2407.05388) | Ordering prior: cluster tree, BFS linearisation | FID/KL gains claimed; numbers not retrieved **[unverified]** | Yes | Yes (as an ordering convention) |
| Steerable Scene Gen | 2025 | [2505.04831](https://arxiv.org/abs/2505.04831) | Diffusion + MCTS + projection/simulation | MCTS reaches 34 feasible objects; training scenes average 17.1 | Partial | No |
| HSM | 2025 | [2503.16848](https://arxiv.org/abs/2503.16848) | Hierarchy of support surfaces; motifs for small and repeated objects | Fidelity and user preference gains; no box-collision numbers retrieved **[unverified]** | Partial | Partial |

### Takeaways for FastFill

- **The 2-decimal text format is not the bottleneck.** The LLM layout systems we compare against (LLplace, OptiScene, ReSpace) all emit 2-decimal JSON. No paper found shows that quantised coordinate tokens (Pix2Seq, SpatialLM) reduce collisions; they change precision and convergence. Number Token Loss is a cheap add-on that may lower position and size error, but nothing suggests it fixes co-located duplicates. Don't spend the next cycle on tokenisation.

- **Duplicates look like a decoding and ordering problem, and the low-cost fixes need no new architecture.**
  - Pix2Seq found that a deterministic order plus argmax gives early stopping and duplicates. Random order, nucleus p=0.4 and noise-object augmentation fix it (AR@100 +6.3).
  - ReSpace trains on random object permutations and gets lower out-of-bounds and VBL from best-of-N.
  - For FastFill this means: sample K=4–8 at temperature 0.6–0.8 and choose with the frozen `validate.check`. This is still one model and one prompt, but K samples cost K× compute.
  - My proposal, not from the literature: add DPO rejected samples that copy a same-category sibling's position (the "co-located duplicate" negative), in place of some of the random position perturbations.

- **The largest measured gain on collisions and boundaries comes from letting the LLM propose and a solver fix the result.** LayoutVLM's in-boundary score is 94.9 vs LayoutGPT's 24.2. Fine-tuning on relations plus poses beats fine-tuning on numeric poses alone (in-boundary 70.4 vs 18.5).
  - FastFill could keep its one-call output and add a deterministic CPU post-pass. It would push apart overlapping boxes with pairwise DIoU, project boxes back inside the floor polygon and out of door keep-outs, and keep each child on its parent (`on`). This also maps directly onto the backend rule that refuses any footprint >1 mm outside or >1 cm into a door/window keep-out.
  - Costs: boxes drift away from GT. PhyScene also shows that pushing objects apart and keeping them in bounds conflict (Col_obj fell 0.200→0.111 while out-of-bounds rose 0.240→0.354), so the solver needs both terms at once. Gross overlaps above 30% may need re-placement rather than a push.

- **For rooms with 51+ objects, change the sequence order without leaving one call.**
  - Emit large floor furniture first, then each parent's supported children directly after it, possibly with coordinates relative to the parent (the HSM, Forest2Seq and LayoutVLM grouping ideas). For repeated items, consider a count/pattern field (for example "6 × chair, row along table") in place of 6 separate boxes. That would directly stop 167 microwaves landing in one spot.
  - Cost: re-SFT, plus a deterministic expansion step that must still return every object instance, because the object set has to be preserved.
  - Calling the model once per object (ReSpace, ATISS) is better supported by evidence, but it breaks the one-call contract. ReSpace's full-scene collision term (MBL) was still worse than MiDiffusion's.

- **Diffusion or flow heads are the principled fix for v2's "averaged multimodal placements", but they are a new model, not a patch.**
  - MiDiffusion's furniture-arrangement mode matches arm P (out-of-bounds about 7–8%, IoU about 1%). DeBaRA matches arm J.
  - All reported results are on 3D-FRONT-size rooms. No paper reports 50+ object rooms or procedural sources such as SAGE.
  - Supervised imitation is also capped by how valid the GT is: 3D-FRONT GT itself has Col_obj ≈ 0.21 (PhyScene), and 28% of our own GT rooms fail the checker. So filtering or repairing the training targets is needed whatever head is used.