# SceneTeract: Agentic Functional Affordances and VLM Grounding in 3D Scenes
- Authors: Leopold Maillard, Francis Engelmann, Tom Durand, Boxiao Pan, Yang You, Or Litany, Leonidas Guibas, Maks Ovsjanikov. 31 Mar 2026. https://arxiv.org/abs/2603.29798
- Agent profile: clearance width, eye height, shoulder height, reach radius, locomotion (Adult, Child, Wheelchair). Checks: navigability, reachability, interactivity, clearance, visibility. Numeric values not in text read (unverified); reach radius +-21.9 cm changes pass rates < 3%.
- 3D-FRONT functional failures (Adult/Child/Wheelchair): task success 59.0/66.0/42.5%; isNavigableTo 84.7/89.2/73.0; isReachable 91.9/93.7/80.7; isInteractable 75.6/73.4/64.1; hasClearance 67.3/66.3/66.6; isVisible 98.7/97.8/97.7.
- VLM judges: direct judgments over-accept (Gemini-3-Flash-Preview FP rate 35.7%).
- Relevance: ~40% of professional-grade scenes fail an adult task check -> functional clearance is a real defect class; use for access-clearance rules.
