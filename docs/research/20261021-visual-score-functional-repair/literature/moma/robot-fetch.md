# Robot specs: Fetch Mobile Manipulator (Fetch Robotics, now Zebra)
- Vendor: Fetch Robotics (acquired by Zebra). docs.fetchrobotics.com no longer resolves; official docs source used instead: [DOC] https://github.com/fetchrobotics/docs/blob/master/source/mechanical_spec.rst (+ sensors.rst); [NAV] https://github.com/fetchrobotics/fetch_ros/blob/melodic-devel/fetch_navigation/config/costmap_common.yaml ; [URDF] https://github.com/fetchrobotics/fetch_ros/blob/melodic-devel/fetch_description/robots/fetch.urdf . The Zebra product page https://www.zebra.com/us/en/products/autonomous-mobile-robots/fetch100-research.html has no specs and the Zebra brochure PDF returned 404.

| Quantity | Value (exact) | Source |
|---|---|---|
| Weight | "about 250 lbs (113.3 kgs)" | DOC |
| Torso lift travel | torso_lift_joint prismatic "400mm" (+) / "0mm" (-), "0.1m/s" | DOC |
| Height | "109.6 cm (149.1 cm \| 58.7 in fully extended)" | (secondary) robotsguide.com/robots/fetch |
| Base width / diameter | "50.8 cm" width | (secondary) robotsguide.com |
| Base diameter | "508mm (20in) wide, 559mm (22in) dia." | (unverified) only seen in a search snippet of a Scribd copy of the spec sheet; page not readable |
| Payload | "payload of 6 kilograms" | (secondary) robotsguide.com; Wevolver also lists 6 kg |
| Arm reach | (not found in official docs). URDF zero pose (arm straight forward): gripper_link 1.128 m in front of base_link origin | derived: FK on URDF |
| Shoulder height | 0.726 m (torso 0) to 1.126 m (torso 0.4 m) above base_link | derived: FK on URDF |
| Head camera height | 1.061 m to 1.461 m above base_link | derived: FK on URDF |
| Nav footprint | robot_radius: 0.30, inflation_radius: 0.70 | NAV |
| Ramps | "ready for ADA-compliant ramps, which are at no more than a 1/12 slope" | DOC |
| Base laser | "range of 25m, 220° field of view" | DOC (sensors.rst) |
| Min passage width / door opening | (not found) | - |

Layout-check notes
- Round differential base, official nav radius 0.30 m: a passage must be wider than 0.60 m plus margin; 0.20 m agent radius under-models Fetch.
- 7-DOF arm mounted on the torso facing forward: the base can face the receptacle. Heights relative to base_link (wheel axle at z=0.055 m in URDF, so base_link is close to the floor; exact offset not checked). With the shoulder at 0.73-1.13 m and ~1.1 m arm length (derived), table/counter heights are easy; floor and high shelves need specific poses (no vendor number found).
