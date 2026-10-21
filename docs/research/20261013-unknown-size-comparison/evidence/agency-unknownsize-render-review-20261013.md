# Independent renderer review — 2026-10-13

**Disposition: hold bulk FID/GPT evaluation.** The reviewed transform math is consistent with the archived axis contract. One P1 integration concern and two P2 reliability/provenance concerns remain. The existing synthetic smoke demonstrates rendering capability, not readiness of the archived rawGT/candidate evaluation path. No model-quality opinion or metric score is issued.

Review started 00:31:36 UTC; completed within the requested five minutes. Independent static review; no code changes, test execution, Blender invocation, metric computation, API calls, or nested agents. Only this requested report was created.

Scope deviation: a line-limited `head` command unintentionally read the entire single-line `metric-inputs.json` while checking the adapter's schema. This exceeded the intended code-only/data-read scope. Data inspection then stopped; no dataset analysis or processing followed. Findings below rely on cited source contracts, tests, and existing receipts; no new cohort counts were computed.

## Findings requiring resolution before bulk evaluation

### 1. P1 — Valid archived rooms with unknown height cannot reach rendering

**Observation.** `metric_case` copies `record["height"]` unchanged, but `plan_case` immediately requires a finite numeric height. Unknown ceiling height is a legitimate upstream state, not necessarily a broken prediction or missing asset.

**Evidence.** `fastfill/comparison/render_assets.py:353–355,392–400`; `fastfill/scene.py:8`; `fastfill/adapters/unified.py:88–91`; `fastfill/build.py:285–289`. The build pipeline deliberately changes contradictory heights to `None`. The archive builder preserves that value with `room.get("height")` at `docs/research/20261011-v11-clean-dpo/paper-metrics-20261013/build_metric_inputs.py:144–148`. Renderer tests always give successful fixtures a numeric height (`tests/test_render_assets.py:117–123`); the archive-adapter test inherits that fixture (`:201–211`).

**Impact.** Every otherwise valid case with `height=null` is rejected as `invalid_number` before assets or either layout can render: zero images for that case. The affected cohort count is not established here. Using only successful renders without an explicit policy would exclude these rooms from visual metrics for a context-data reason, independently of model output quality.

**Recommendation.** Freeze an explicit unknown-height policy before the real-case pilot. Either supply a disclosed render-only height derived solely from shared room context/a fixed protocol default, or formally classify these cases as visually unavailable and retain their denominators. Do not infer height from GT or candidate furniture. Record original/effective height and its provenance. Add an adapter regression for an otherwise valid archived record with null height and verify identical room framing across all arms.

### 2. P2 — Some malformed cases escape the promised rejection ledger and stop later cases

**Observation.** Nested input types are not fully validated, and archive adaptation occurs before the receipt machinery. The current exception boundary therefore does not cover all malformed cases.

**Evidence.** `render_assets.py:376–378` calls `.items()` on any truthy `layouts`; `:594–596` does not catch `AttributeError`. A case with valid room fields, `fixed=[]`, `asset_mapping=[]`, `expected_ids=["chair"]`, and `layouts=["candidate"]` reaches this call without accessing any mesh. It raises `AttributeError`, leaving the already-written receipt at `pending` and aborting subsequent cases. Separately, a selected archived record missing `raw_layout` raises `KeyError` in `metric_case` (`:399–400`) when called by `main` (`:628`), before `run_cases` creates a ledger (`:631`). These are static control-flow reproductions, not executed tests. Existing malformed-case tests (`tests/test_render_assets.py:230–234`) cover outer shapes/missing keys, not these paths.

**Impact.** One malformed nested record can prevent all later manifest cases from receiving receipts. A malformed selected archive record can produce no ledger at all. That breaks failure accounting precisely when visual availability must be distinguished from model quality. This does not negate the correctly handled ordinary missing-mesh path.

**Recommendation.** Validate nested containers and convert schema failures into `RenderReject`; bring selected-record adaptation within a receipt-producing boundary. Test a malformed case followed by a valid case, plus a selected archive record missing a required field. Require terminal rejection reasons and continued accounting without partial images.

### 3. P2 — Receipts do not bind scores to exact layouts, materials, or raw/repaired selection

**Observation.** The ledger hashes the renderer and OBJ files, but not the canonical input case, MTL/texture content, or final images. The adapter's `repaired` selection is not recorded.

**Evidence.** `render_assets.py:392–400` maps either raw or repaired geometry into the same `candidate` label; `:558–575,590–593` records settings, UID/stage, mesh hashes, overrides, cameras, image paths, and Blender version. It omits the selected layout mode and any layout/input digest. `inspect_asset` hashes only OBJ bytes (`:268–270`), although material and texture files drive appearance. The existing `smoke-clean32/ledger.json` has the same omissions. Its script digest does match the currently reviewed renderer.

**Impact.** Changing candidate size/pose, switching raw to repaired layouts, or changing texture/MTL bytes can leave the recorded script/mesh hashes and room cameras unchanged. The receipt alone cannot distinguish those materially different scoring inputs. No existing score contamination is alleged: the supplied smoke reports zero metric/API calls.

**Recommendation.** Before collecting formal scores, bind each case to an immutable canonical-input digest, explicit raw/repaired mode, source/crosswalk identity, material/texture digests, and image digests. A separate frozen sidecar is sufficient if its digest is recorded in the ledger. Link FID/GPT outputs to those image identities. Add a regression demonstrating that layout or texture changes change the bound identity.

## Contracts that passed static review

All test references below are under `fastfill/comparison/tests/test_render_assets.py`.

| Contract | Assessment and evidence |
| --- | --- |
| Holo `(-z,x,y)` | Matches the archived basis, determinant −1 (`render_assets.py:64–70,111–125`). Native scale signs precede the basis. No extra raw rotation is applied. |
| Reflection normals/winding | The normal matrix is the inverse transpose of the fitted linear transform; negative determinant reverses whole OBJ face corners, retaining UV/normal indices (`:120–125,296–303`). Tests cover a non-axis-aligned normal under nonuniform fit and yaw (`tests:64–73`). |
| Front `(z,x,y)` and scales | Proper cyclic basis; negative native scale changes handedness, while positive magnitudes are superseded by bbox fitting (`:102–125`; `tests:50–57`). This matches the archived visualization contract. |
| Union bbox and bottom pivot | Bounds include all OBJ vertex records, across parts; signed-permutation bounds feed a local XY-center/min-Z pivot, then requested size, canonical yaw, and position (`:113–125,236–270`). |
| Prediction sizes / no GT fallback | `mapped_object` requires placement size/pos/yaw and uses those values; it does not consult crosswalk `gt_object` geometry (`:308–341`). Tests separately change GT sizes and poison crosswalk GT dimensions (`tests:171–179,201–211`). |
| RawGT/candidate adapter | Uses `gt_canonicalobjects` and `raw_layout` by default; repair is explicit. The archive builder already exports layout yaw in radians, so no degree conversion is missing here. Fixed context comes from canonical `record["fixed"]`, not degree-valued `judging_request["fixed"]` (`:392–400`; archive builder `:4–5,145–148`). |
| Shared room cameras/context | Cameras depend only on room boundary and height (`:170–183,387–389`). The scene is reset and configured identically per layout. Concave floor triangulation is retained; wall selection and lighting are room-dependent (`:423–438,460–496,509–541`). |
| Ordinary asset failures | Missing mesh/mapping, missing MTL/texture, and incomplete inventory raise recorded rejection paths; complete-case images publish only after every layout/view finishes (`:236–267,319–341,381–386,568–597`). Tests cover these paths. The broader receipt gap is finding 2. |
| Actual texture-path repair | Prepared MTLs use local texture aliases rather than problematic absolute paths; validation forces pixel access before checking lazy image decode (`:273–288,499–506`). Tests cover the alias and lazy decode behavior (`tests:141–150,265–286`). The successful existing smoke was produced by the current renderer hash. This supports the fix for that smoke, not universal decoding/material fidelity across all assets. |

## Evidence limits and the pre-bulk gate

- The archived renderability report establishes file linkage/existence, not complete Blender import or texture-decode coverage. Its six-stratum asset coverage must not be promoted to a universal rendering pass.
- The handoff reports 44 passing tests and 91% combined unit/smoke statement coverage (66% unit-only). Those are supplied results, not independently rerun here. Statement coverage does not close the specific adapter and receipt gaps above.
- The existing clean receipt contains one synthetic room, a `candidate` perspective only, two source assets, 512×512, 32 samples, OpenImageDenoise enabled, and Blender 5.2.1 LTS. It does not exercise a real archived rawGT/candidate pair or the metric/API path. The parent's prior image inspection is accepted; no duplicate image inspection was needed.
- **512×512 is the selected reasonable default**, matching the CLI. The handoff's future 1023 examples and the test accepting 1023 are not a requirement and are not a blocking defect. Freeze samples/denoising consistently across views and arms; the CLI still defaults to 8 samples without denoising unless explicitly set.
- After resolving the above, the next gate is the parent's separately authorized, bounded real-case paired pilot exercising actual FID feature/scoring and GPT judging paths with matched image receipts, source-axis/asymmetric-front inspection, and visible failure accounting. Tiny-pilot FID is an integration check, not a statistically persuasive model comparison. No such pilot, API call, or bulk work was performed or authorized by this review.
- Bbox fitting uses oracle reference asset identities and can alter padding/proportions. Solid procedural doors/windows and cutaway walls are the disclosed common visualization protocol, not native Isaac shading or architectural-opening fidelity.

## Reviewed revision

- Renderer SHA-256: `06083e8d8e33295b609971f456bc701383aa290d0ed1db89e1c0a5b10d105309`.
- Tests SHA-256: `5dd655e829ec57ed01316ac9f91c941caa067ae0455b8eded3fd7beabd2d2fae`.
- Primary evidence: `/tmp/unknownsize-render-smoke/HANDOFF.md`, its existing `summary.json`, `case.json`, and `smoke-clean32/ledger.json`; `docs/research/20261011-v11-clean-dpo/paper-metrics-20261013/evidence/agency-harddisk-renderability-20261013.md`.
- Graph lookup reported that OptiScene was not indexed; direct source inspection was used. The referenced `docs/CODEX-NAVIGATION-GUIDE.md` was absent.
