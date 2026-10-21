# Robot specs: PAL Robotics TIAGo / TIAGo++ / TIAGo Pro
- Vendor: PAL Robotics. Sources (official): [DS] TIAGo technical specifications PDF https://pal-robotics.com/datasheet/tiago/ (covers LITE / TIAGo / ++ ); [WEB] https://pal-robotics.com/robot/tiago/ ; [PRO-DS] TIAGo Pro datasheet PDF https://pal-robotics.com/datasheet/tiago-pro/ ; [PRO-WEB] https://pal-robotics.com/robot/tiago-pro/ ; [NAV] https://github.com/pal-robotics/pal_navigation_cfg_public/blob/humble-devel/pal_navigation_cfg_params/params/tiago_nav2.yaml (tiago_dual_nav2.yaml identical values)

| Quantity | TIAGo / TIAGo++ | TIAGo Pro | Source |
|---|---|---|---|
| Height | "110 - 145 cm" | "120-155 cm" | DS, PRO-DS |
| Footprint | "Ø54 cm" | "50 x 72 cm" | DS, PRO-DS |
| Weight | "70 kg" | "80 kg" | DS, PRO-DS |
| Torso lift | "35 cm" | "35 cm" ("Torso Range") | DS, PRO-WEB |
| Arm reach | "87 cm (without end-effector)" | vertical "92 cm", horizontal "236 cm" | DS, PRO-WEB |
| Payload | "3kg (without end-effector)" at full extension | "3 kg (excluding end-effector)" per arm | DS, PRO-WEB |
| Arms | 1 (TIAGo) or 2 (++) x 7 DoF | 2 x 7 DoF | DS, PRO-WEB |
| Base | Differential, omnidirectional optional; "Max. speed 1.5 m/s" | omnidirectional, 4 mecanum wheels | DS, PRO-WEB |
| Workspace | - | "7 m³" total | PRO-DS |
| Nav footprint | robot_radius: 0.275, inflation_radius: 0.55, resolution: 0.05 | (not found) | NAV |
| Reachable height range floor-to-max | (not found as a spec); "1.75 m" per-arm figure for TIAGo++ only in a search summary (unverified); Pro "can reach objects on the floor and on high shelves" (marketing text, no number) | | - |
| Min passage width / door opening | (not found) | (not found) | - |

Layout-check notes
- Round base Ø54 cm, nav radius 0.275 m: free passages need > 0.55 m plus inflation; again above a 0.20 m agent radius.
- Arm reach 87 cm from the shoulder on a torso with 35 cm travel; robot height 1.10-1.45 m. Tables and counters reachable from a frontal base pose; exact min/max end-effector heights are not published (would need FK on the TIAGo URDF; not done).
