# Robot specs: Galaxea R1 / R1 Pro / R1 Lite (Galaxea Dynamics), and the BEHAVIOR challenge R1 Pro
- Vendor: Galaxea Dynamics (星海图). Sources (official docs): [R1] https://docs.galaxea-dynamics.com/R1/docs/ (Chinese "技术规格" page); [PRO25] https://docs.galaxea-dynamics.com/R1Pro/en/docs/2025/ ; [PRO26] https://docs.galaxea-dynamics.com/R1Pro/en/docs/2026/ ; [LITE25] https://docs.galaxea-dynamics.com/R1Lite/en/docs/2025/ ; [LITE26] https://docs.galaxea-dynamics.com/R1Lite/en/docs/2026/ ; [NAV] https://docs.galaxea-dynamics.com/R1Pro/en/docs/2025/function_guide/navigation/r1pro_navigation_guide ; [B1K] https://behavior.stanford.edu/challenge/archive/2025/call_for_participation.html , https://behavior.stanford.edu/omnigibson/robots.html

| Quantity | R1 | R1 Pro (2025 / 2026) | R1 Lite (2025 / 2026) | Source |
|---|---|---|---|---|
| Height | 1700 mm | "Full-body height: 1700 mm", torso 800 mm (2025) / 835mm (2026), chassis 346 mm | 1281 mm / "Maximum Height 1560 mm" | R1, PRO25/26, LITE25/26 |
| Width | 675 mm | chest 348 mm, shoulder 615mm, chassis 675 mm | 596 mm chassis, 670 mm platform / chassis 601mm, platform 820mm | same |
| Chassis L x W x H | 675L x 636W x 346H mm | 675L x 636W x 346H mm | 596L x 600W x 250H mm (both years) | same |
| Weight | 96 KG (with battery) | (2025 not listed) / 126kg | 55 kg / 96kg | same |
| Vertical operating range | 0 ~ 2000 mm | 0 ~ 2000 mm | 0-1.7m (2025) / (not listed 2026) | same |
| Horizontal reach | 700 mm (860 mm with gripper) | radius 716 mm (861mm with gripper); single-arm working radius 620 mm | arm fully extended 600mm long | same |
| Arm payload | rated 2.5 KG@0.5 m, max 5 KG@0.5 m | rated 3.5 KG@0.5 m, max 5 KG@0.5 m (spec table); arm table 3.5 kg@0.6m / 5 kg@0.6m | rated 3kg@0.6m, peak 5kg@0.6m | same |
| DoF | 24: chassis 6, torso 4, arm+gripper 7 each | 26: arm 7, gripper 1, chassis 6, torso 4 | 23: arm 6, gripper 1, torso 3, chassis 6 | same |
| Base | chassis "Ackermann/translation/rotation" (3 steerable wheels on Lite: "unlimited 360° rotation") | same | same | R1, PRO25, LITE26 |
| Nav | goal points: "ensuring that the R1 center is at least 45 cm away from obstacles"; autonomous max travel speed 0.6 m/s; maps "no larger than 100 square meters" | | | NAV |
| Min passage width / door opening | (not found) | (not found) | (not found) | - |

BEHAVIOR challenge 2025: "we use Galaxea's R1 Pro robot, a wheeled humanoid, as the default embodiment"; OmniGibson models R1/R1 Pro with a "holonomic base (which we model as a 3-DOF (x,y,rz) set of joints)", "4-DOF torso", "x2 7-DOF arm" (R1: "x2 6-DOF arm"), "x2 2-DOF parallel jaw grippers" [B1K]. No dimensions on those pages.

Layout-check notes
- Large rectangular base (675 x 636 mm, about 0.46 m half-diagonal) and the vendor's 45 cm centre-to-obstacle rule for navigation goals: a 0.20 m agent radius is far too small for R1; passages need roughly >= 0.9 m (2 x 0.45 m, derived from the vendor goal-clearance rule, not a vendor passage spec).
- The torso lift/pitch gives a 0-2.0 m vertical working range (R1/R1 Pro), so floor to high shelves is in range; the binding constraint is base access (free floor in front of the receptacle within ~0.7 m arm radius), not height.
