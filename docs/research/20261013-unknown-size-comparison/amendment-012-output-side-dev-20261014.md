# Amendment 012 — output-side improvements on dev: S1 deterministic repair and S2 best-of-N

Locked 2026-10-14 ~13:00 UTC, before any S1 or S2 output on real model layouts exists. The user authorised the full output-side programme and asked that comparisons use free open-weight models, that failed rooms are never discarded, and that the deliverable plugs into the downstream room generator. Tooling: `fastfill/comparison/downstream.py`, `output_repair.py`, `sample_vllm.py`, `best_of_n.py` (commit 4e3be1d75, independently reviewed). Nothing here changes any confirmatory or delivery result of amendments 005–011; all outcomes are reported as a separate, exploratory output-side table.

## 1. Data and models

- **Dev:** all 8,167 dev rooms of main-data-v1 (`RUN/dev-full-v1`, built by `sample_vllm build`; it contains the 600 dev-select-v1 rooms). Test is not touched by this amendment.
- **Models:** the dev-selected models of amendment 010 — J = SFT adapter on Qwen3-8B; P = rpo-dpo1 (adapter on its BF16 training base).
- **Engine:** vLLM with BF16 LoRA. It is not bit-identical to the HF runs. Agreement with the HF outputs on the 600 dev-select rooms is reported (`best_of_n --reference`), and every S2 gain is measured against vLLM greedy, never against HF numbers.

## 2. Arms (per model)

- **G:** vLLM greedy (k = 0), raw.
- **G+S1:** G after the S1 repair (XY translations of support subtrees only; ids, categories, sizes, yaw, parents, z and the object set never change; rolled back if any audit component or any downstream flag gets worse).
- **B_m:** best of the greedy plus m samples (m ∈ {1, 2, 4, 8}; T 0.7, top_p 0.95, per-(uid, k) seeds `sha256('s2-bo-v1:' + uid + ':' + k)`), selected by the request-only rule of `best_of_n.RULE`. Fewest checker violations wins; greedy wins ties. An alternative must pass the guards against greedy: per-object size band 0.67–1.5 × the train prior; contained pairs, flat objects and same-category stacks no more than greedy's; footprint bias no more than 0.03 below greedy's.
- **B8+S1:** B8 after S1.
- Rooms where nothing is complete fall back to greedy. No room is dropped; every rate uses all 8,167 requests as the denominator.

## 3. Metrics

- **Primary for this amendment:** downstream acceptance = `downstream.accept(...)` is accepted, i.e. the EmbodiedGen path-A rules as ported (1 mm floor envelope after keep-outs, door strips, A4 AABB overlaps, A3 reachability, R5). Windows have no keep-out (verified in the backend). Rooms with proxy boundaries are included and also reported separately.
- **Secondary:**
  - frozen-checker geometry success;
  - complete rate;
  - footprint bias B and size log MAE vs GT (S1 cannot change them; S2 can for J);
  - per-object displacement (median and p95 over moved objects, share of objects moved, share moved > 0.5 m);
  - S1 status (ok / partial / rolled back / timed out);
  - the exploit counters (contained pairs, flat objects, same-category stacks, parent changes);
  - GPU-seconds and CPU-seconds per room.
- **Breakdowns:** object-count bins (1–10, 11–30, 31–50, 51+), source, rectangle vs not, rooms with vs without doors.
- **Intervals:** paired differences between arms use the existing house-cluster bootstrap (2,000 draws, seed 42).

## 4. Decisions (dev only, recorded before test)

- **S1 enters the delivered pipeline** if G+S1 − G downstream acceptance has a paired 95 % CI above 0 for both J and P. It must also keep the median displacement of moved objects ≤ 0.30 m and leave geometry success not worse (CI upper bound ≥ 0).
- **S2 (B8) enters the pipeline** if (B8+S1) − (G+S1) downstream acceptance ≥ +0.03 with CI above 0. For J, |B(B8) − B(G)| must also be ≤ 0.03. Otherwise the smallest m whose gain over G+S1 is within 0.01 of B8's is adopted if it passes the same tests; if none does, S2 is not adopted.
- **After the dev decision**, the adopted pipeline is applied once to the test outputs:
  - our selected J/P on all 8,647 test rooms;
  - S1 (and nothing else, at the same settings) on every agent's existing 1,454-room cohort outputs, so the open models and the API agent receive the same repair.
  - Best-of-N for agents is not run here; open-model best-of-N is a later amendment with its own GPU budget.

## 5. Budget and limits

- **GPU cap:** 150 GPU-hours for S2 sampling (2 arms × 8,167 rooms × 9 generations).
- **S1:** CPU only, max_trials 2,000 per stage, 60 s time limit per room (timeouts reported, not reproducible).
- **Paid API:** none in this amendment.
