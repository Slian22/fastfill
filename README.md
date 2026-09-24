# FastFill v1

Object-conditioned room layout in one LLM call: given the room type, the real floor polygon, **all** objects to place
(real local sizes, local +X = front) and optional spatial constraints, Qwen3-8B + LoRA outputs every object's
position, yaw and support parent.

$$F_\theta(R, G, O, C_{\text{layout}}) \rightarrow L$$

Full handoff doc (Chinese): [`fastfill/README.md`](fastfill/README.md) — data sources and why each is kept or dropped,
unified conventions, build, training on the server, evaluation, serving for WorldEdge / EmbodiedGen.

| Path | What |
|---|---|
| `fastfill/` | adapters (18 sources -> one IR), build (filters, leakage-safe split, constraints), train, evaluate, interface, serve |
| `scripts/merge_lora.py` | merge the LoRA into the base model (transformers 4.x and 5.x) |
| `tools/qa_report.py` | the QA numbers of a built dataset (leakage, closure, constraint checks, raw vs repaired legality) |
| `tools/ablation_check.py` | v1.1 minus its constraints must equal v1.0 byte for byte |
| `tools/e2e_embodiedgen.py` | contract test through EmbodiedGen's real FastFillBackend (`EMBODIEDGEN=path/to/EmbodiedGen`) |

Built data (v1.0 / v1.1, ~337 MB each) is not in git: copy it to the server (commands in `fastfill/README.md` section 4).
The previous FastFill line (codec text, Floor + Surface calls, `vendor/`) is kept under the tag `legacy-2026-07`.
