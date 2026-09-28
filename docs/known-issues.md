# Verified limitations and remaining issues

These notes describe the audited v3.2 implementation. A frozen and hash-verified release is reproducible; it does not imply that all defects are resolved.

## Open implementation issues

- `fastfill/serve.py` bounds Content-Length but does not set a request-read timeout. The synchronous HTTP server can wait indefinitely for an incomplete request body. Add a request deadline before relying on this server for clients that may stall; concurrent model execution is a separate design decision.
- In `fastfill/adapters/internscenes.py`, objects whose source semantic front is near vertical may be re-expressed using a horizontal geometric axis without propagating object-level `front_known=False`. Such objects can still receive yaw supervision. Confirmed examples include a TV in `InternScenes_mp3d::Matterport3D:matterport3d/1LXtFkjw3qL/region11` and a monitor in `InternScenes_3rscan::3RScan:3rscan/0f2f2723-b736-2f71-8c94-f692cca76661`. Representative rows remain in v3, v3.1 and v3.2; the corpus-wide incidence has not been measured in this audit.

## Metric and capability boundaries

- `valid` checks room containment with tolerance, support and ceiling limits. It excludes neither every object-object collision nor every physical failure.
- Runtime acceptance also checks the 1 mm floor-containment rule, non-window fixed obstacles and hard relations. Collisions between objects to place are reported but do not reject the layout. Report severe collisions separately.
- Support means the child's bottom is near the parent's box top and its center is near the parent's footprint. This is not mesh contact, center-of-mass or rigid-body stability verification.
- Fixed geometry may represent existing furniture as obstacles, but relation constraints cannot refer to fixed IDs. Adding a cup onto an existing fixed table is not supported by this protocol. `on` relates objects generated in the same request.
- Integer-degree training targets introduce up to half a degree of angular quantization; geometry still uses floating-point calculations.
- Constraints in v3.1/v3.2 are extracted from reference layouts where they already hold. Counterfactual tests with the same room/objects and different feasible constraints are needed to demonstrate condition-responsive control.
- The earlier standalone, single-room side-by-side plotting scripts are not this package's benchmark. They use a different output schema and should not be used as evidence of FastFill v3-series performance.

## Verification scope

The current offline suite passed 138 tests and 90 subtests in the recorded audit. It includes a tiny real CPU adapter merge; generation and HTTP tests also use mocks. Production GPU training, vLLM deployment, arbitrary mesh correctness and full physical simulation were not validated by that suite. Dependency vulnerability auditing requires a reachable vulnerability database and was unavailable in the consolidation environment.
