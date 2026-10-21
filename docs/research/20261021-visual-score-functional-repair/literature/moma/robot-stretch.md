# Robot specs: Hello Robot Stretch (RE2 / Stretch 2, Stretch 3, Stretch 4)
- Vendor: Hello Robot. Sources (all official): [S3-DS] Stretch 3 datasheet image (Rev 2.14.2024) https://docs-arch.hello-robot.com/0.3/hardware/images/Stretch_3_Datasheet_2.2024.png ; [S3-HG] Stretch 3 hardware guide https://docs-arch.hello-robot.com/0.3/hardware/hardware_guide_stretch_3/ ; [RE2] RE2 spec sheet image https://docs-arch.hello-robot.com/0.3/hardware/images/re2/hardware_specification.png ; [JL] joint-limit figure https://docs-arch.hello-robot.com/0.3/getting_started/images/stretch_arm_lift_joint_limits.png ; [S4] Stretch 4 datasheet https://hello-robot.com/wp-content/uploads/2026/05/HelloRobot-DataSheet-Stretch-4-Rev5_AsLaunched.pdf and https://hello-robot.com/stretch-4/ ; [NAV] stretch_nav2 params https://github.com/hello-robot/stretch_ros2/blob/humble/stretch_nav2/config/nav2_params.yaml ; [URDF] https://github.com/hello-robot/stretch_urdf (SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf)

| Quantity | Value (exact) | Source |
|---|---|---|
| Footprint | RE2: "Footprint (Stowed): 34cm W x 33 cm L"; S3: "Size 33 x 34 x 141 cm"; S4: "43 cm Diameter" (datasheet) vs "45 cm Diameter" (product page) | RE2, S3-DS, S4 |
| Height | RE2 "Height: 141 cm"; S4 "160 cm Height" | RE2, S4 |
| Weight | RE2 "23 Kg"; S3 "24.5 Kg"; S4 "46 kg / 33 kg ballast off" | RE2, S3-DS, S4 |
| Lift travel | RE2 "Range of motion: 110cm", figure "~1.1m" from "zero"; URDF joint_lift upper="1.1"; S4 "120 cm (Lift)" | RE2, JL, URDF, S4 |
| Arm extension | RE2 "Range of motion: 52cm"; S3 "Reach: 0.52m"; S4 "55 cm + 6 cm wrist" | RE2, S3-HG, S4 |
| Payload | RE2 "2.0Kg (no gripper)"; S3 "Recommended Max payload - 2kg"; S4 "2.5 kg" extended / "4 kg" retracted | RE2, S3-HG, S4 |
| Base | RE2 "Differential two wheel", "25mm ground clearance", "Max speed: 0.3 m/s"; S3 drive wheels "4 inch diameter"; S4 3 omniwheels, "Top Speed: 60 cm/s", "up to 20 mm step/threshold clearance" | RE2, S3-HG, S4 |
| Nav footprint | robot_radius: 0.22, inflation_radius: 0.55, costmap resolution: 0.05 (m) | NAV |
| Grasp-centre height, wrist level | 0.11 m (lift 0) to 1.21 m (lift 1.1) above floor | derived: FK on URDF (base_link z=0 at floor; wheel axle z=0.0508 = 2 in radius) |
| Grasp-centre lateral offset | 0.415 m (retracted) to 0.935 m (extended) from base_link origin, along the robot's -y (sideways) | derived: FK on URDF |
| Head camera height | 1.322 m (head at zero pose) | derived: FK on URDF |
| Min passage width / door opening | (not found in vendor docs) | - |

Layout-check notes
- The arm telescopes sideways (perpendicular to the drive direction): to reach a target the base must be parked parallel to the receptacle edge, with free floor along the receptacle, not facing it. The S3 guide says the retracted arm/wrist "fit within the footprint of the base".
- Reachable surface heights roughly 0.1-1.2 m with level wrist (derived); floor pickup needs the wrist pitched down (URDF grasp centre goes below z=0 at lift 0, derived, not a vendor claim). Horizontal reach beyond the base edge is about 0.52 m for Stretch 2/3.
- Nav stack uses a circle of 0.22 m radius; an agent radius of 0.2 m (our downstream grid) is slightly smaller than Stretch's official 0.22 m.
