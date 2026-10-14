# Amendment 015 — results per room family, and the official OptiScene recipe on room families it never saw (EXPLORATORY)

Locked 2026-10-14 (UTC), before any per-family result or E3 generation exists. Requested by the user: OptiScene covers bedroom, living room, kitchen and bathroom; our test set has 346 raw room-type strings. Tooling: `fastfill/comparison/room_families.py` and the E3 extension of `optiscene_eval.py` plus `submit_015.sh` (independently reviewed; the E1/E2 outputs of amendment 014 are byte-identical apart from the implementation hash). No earlier result or decision changes.

## 1. Frozen mapping (room_families.family)

All 346 raw strings of the 8,647 test rooms map to one family. The full string-level table is `evidence/room-families-mapping-test-20261014.md`.

| family | paper family | test rooms |
|---|---|---:|
| bedroom | yes | 2,544 |
| living | yes | 1,410 |
| kitchen | yes | 778 |
| bathroom | yes | 1,068 |
| dining | no | 343 |
| study_office | no | 690 |
| meeting | no | 125 |
| balcony | no | 178 |
| laundry_utility | no | 127 |
| garage | no | 68 |
| circulation | no | 118 |
| storage_closet | no | 127 |
| public_commercial | no | 357 |
| other | no | 41 |
| none (no room type) | no | 673 |

Mapping rules:
- Compound names take the first-named room.
- A hub room counts as an office.
- An office pantry counts as a kitchen.
- Hotel and dorm rooms count as bedrooms.

## 2. Per-family tables

`room_families report` gives, per family and for the paper-family and other-family pools:
- requests, complete rate, checker success and downstream acceptance (raw);
- footprint bias B and size log MAE versus GT, and OOR.

It covers J, P, GT and the S1-repaired J/P on all test rooms, and every agent on the 1,454-room cohort.

Paired differences use the house-component bootstrap (2,000 draws, seed 42): J − P on test, and P − agent on the cohort. A scope with n < 30 is flagged; no CI is given with fewer than 10 house components. These tables are descriptive and set no decision rule.

## 3. E3: OptiScene on unseen room families

- **Rooms:** test rooms of the 9 non-paper families with a usable type (dining, study_office, meeting, balcony, laundry_utility, garage, circulation, storage_closet, public_commercial). Each family contributes at most 100 rooms, chosen by sha256('optiscene-015-e3:' + uid); garage takes all 68. That gives 868 rooms.
- **Inputs:** built exactly as in E2 (amendment 014) in both size modes. The 23 rooms covered by OptiScene training are reported separately.
- **Models:** the reproduced SFT/DPO1/DPO2 checkpoints and the existing 8B.
- **Metrics:** per family and pooled — parse rate, checker success, downstream acceptance, OOR, and the paired OptiScene(prior) − P difference with house-cluster CIs.
- **Repair:** OptiScene + the same S1 repair (amendment 016 §4) is added once S1 is applied.
