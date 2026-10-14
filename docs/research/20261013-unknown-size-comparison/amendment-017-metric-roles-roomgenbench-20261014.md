# Amendment 017 — what "downstream acceptance" is, and which metrics speak for RoomGenBench (reporting only)

Locked 2026-10-14 (UTC), before any final table is built. This is a post hoc reporting correction, prompted by the user ("RoomGen 在 RoomGenBench 底下，我们生成 layout+bbox 接这个") and by a code check. No decision of amendments 005–016 changes: the amendment-012 S1 adoption also holds on the frozen checker (P 0.398 → 0.701), and every 016 analysis is reported under both metric families.

## 1. Facts, from the code

**RoomGenBench** (`OptiScene/RoomGenBench`, `bench/CONTRACT.md`, `assemble.py`, `score_geom.py`, `score_vlm.py`):
- **Input:** one 3D bbox per object, plus a text description and a place type (floor / wall / on_object).
- **Assembly:** each generated asset is scaled anisotropically to fill its bbox, then placed at the bbox's bottom-centre.
- **Scoring:** asset fidelity (VQA, SigLIP-2, R@5, DINO / F-score) and support (rays cast down from an object's base must hit its supporter within 2 cm).
- **No placement gate:** it has no collision, door keep-out, reachability or 1 mm envelope check, and it never refuses a layout.

**"Downstream acceptance" (`downstream.accept`)** is a port of EmbodiedGen's room-admission gate:
- **Where it runs:** `chain.grow` → `backend.realize_room` → `admit.validate`, in the infinite-world growth loop. A refused room is either re-planned larger or refused for that frontier.
- **Our port is stricter than the original in two ways:**
  - **Required objects (P5).** `admit` counts as "required" only the classes the task names (`spec.required_entities`; A3 and A4 are scoped to them, `admit.py:303-381, 620-702`). We made every requested category required.
  - **Doors (P4).** `admit` has one entry door; other doors get A6 floor strips only. We treat every door as an entry.
- **Effect on GT (all 8,647 test rooms):**

  | Variant | GT acceptance |
  |---|---|
  | strict (as reported) | 0.295 |
  | lenient: one entry door | 0.297 |
  | required = {} | 0.605 |
  | R1/R2/R5 envelope only | 0.625 |

  So most GT refusals come from the all-categories-required assumption. J and P raw follow the same pattern: J 0.172 / 0.173 / 0.483, P 0.200 / 0.201 / 0.474.

## 2. Metric roles in every final table

1. **RoomGenBench-relevant — primary for the paper's feasibility and realism claims:**
   - frozen checker success (OOB, support, ceiling, main and severe collisions, fixed blocking; tucked chairs exempt);
   - size realism: size log MAE, footprint bias B, and the amendment-016 size-substitution results on the checker;
   - OOR;
   - visual metrics (FID, GPT judge, the latter secondary per addendum B4).
2. **EmbodiedGen admission (strict port) — secondary.** It is relabelled everywhere from "downstream acceptance" to "EmbodiedGen admission (strict port)". It is reported next to two variants: lenient (one entry door) and required = {}. These two bracket the real gate, whose required set depends on the task.
3. Numbers already reported under the old label stay as they are, with the new label attached. No result is recomputed or replaced.

## Provenance correction (same day, before any final table)

- `admit.validate` and `chain.grow` are not official EmbodiedGen code. They sit in `embodied_gen/world/` of the local repo `Slian22/infinite-task-world`, which is a fork of EmbodiedGen. The official history (Xinjie, from 2025-06-11) has no `world/` module. All 135 commits of `world/` are by Slian, starting 2026-08-16.
- `fastfill/comparison/downstream.py` is our own port of that `admit.py`, with the two stricter assumptions P4 and P5.
- So neither RoomGenBench nor official EmbodiedGen imposes these rules. The label becomes **"infinite-task-world admission (strict port)"**.
