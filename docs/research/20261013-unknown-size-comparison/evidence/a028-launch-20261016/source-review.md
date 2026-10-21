# Amendment 028 source review

Final verdict: no remaining launch-blocking findings after fixing two issues.

1. Retried training now requires successful smoke-vLLM AND failure of training segment one. Upstream dependency cancellation cannot allocate a retry behind a failed gate.
2. J27 records a reduced N fitting ten hours when full-N training is slow, preserving memory and zero-N stops. Arm A retains its previous behavior.

Five new regression cases were first observed failing, then passed. Independent reviewer Sartre re-reviewed the final diff. Model/recipe, joint row semantics, held-out isolation, deterministic selection and overlap, separate outputs, checkpoint resume, cap handling and bootstrap/Holm statistics were reviewed.

Code: 7373495ca9e35391a5981653997be5485f64e9b6. Source was deployed as an immutable snapshot, with 456 file checks and actual-server CPU import/arm preflight. No change to the existing 27B-P delivery.
