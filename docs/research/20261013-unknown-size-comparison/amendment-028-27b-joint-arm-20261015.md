# Amendment 028 — 27B joint (J) arm: a fair "27B joint vs 27B prior-size placement" comparison

Locked 2026-10-15 (UTC), before any amendment-028 output exists. This is follow-up research; the frozen delivery (`fastfill-delivery-20261015`) does not change.

## Why

**What has been compared.** At 8B, joint prediction (J: no size input; it outputs size and pose) and prior-size placement (P) were compared under the same protocol:
- geometry success: J 0.3882 vs P 0.3896;
- size log MAE: J 0.2861 vs P 0.2817;
- neither difference was confirmed.

**What is missing.** Amendment 021 trained only the selected P route at 27B, so "27B-J vs 27B-P" is unknown. It matters for objects the size table does not cover. At 8B, J beat the prior on the held-out SceneSmith source, where 45 % of objects hit the prior's fallback box. That is one source only.

## Design

**Arm J27**
- **Model and recipe:** Qwen3.8-27B (the same base and revision as 021 Arm A), LoRA on the language model only, thinking off. The recipe is identical to 021 Arm A: r32/α64, lr 5e-5, 1 epoch, effective batch 4, token-mean NLL on the answer, warmup 0.03, `ckpt_latest` resume.
- **Format:** J's own SFT format, as in the 8B J run: the request carries no movable sizes, and the target is size, position, yaw and parent.
- **Data:** J's training rows (`main-data-v1/joint`, the matched 98,754-room train pool), N = 40,000 rooms taken in `sha256('a021-27b:' + uid)` order. This is the same rule and salt as Arm A, to maximise room overlap with 27B-P. The overlap with Arm A's 40,000 rooms is reported.
- **Smoke:** a smoke on the longest sequences measures memory and time. If the projected training exceeds 12 h, N is reduced by the 021 §2 rule and recorded before training.

**Evaluation**
- vLLM greedy (BF16 LoRA, as for Arm A).
- Scored with joint semantics (the `evaluate_pair` J rules, amendment-005 cap rule).
- Rooms: dev-select (600) and all 8,647 test rooms.

**Pre-registered comparison on test** (27B-J vs 27B-P, the 021 Arm A test rows, same engine)
- **(a)** All-request geometry success (frozen checker).
- **(b)** Size log MAE (P's sizes are the prior).
- Paired house-cluster bootstrap, 2,000 draws, seed 42; Holm over (a) and (b), as in amendment 005.
- **Secondary:**
  - footprint bias B;
  - a′ (amendment-005 rescale);
  - the SceneSmith source;
  - the prior-fallback objects;
  - object-count bins and room families;
  - S1 applied to both.

**Reading.** This is descriptive for delivery: it does not reopen today's delivery. It informs whether a joint 27B is worth adopting later, e.g. for objects the table cannot size.

## Budget

| Item | GPU-h |
|---|---|
| Training | ≈ 25–35 (4 GPUs, ≤ 12 h wall, segments with resume) |
| Smoke | ≤ 3 |
| Inference (dev + test, 1 GPU each) | ≤ 6 |
| **Cap** | **80** |

No API calls.
