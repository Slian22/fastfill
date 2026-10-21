# Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation
- Authors: Zipeng Fu, Tony Z. Zhao, Chelsea Finn (Stanford). arXiv 4 Jan 2024. https://arxiv.org/abs/2401.02117
- What: low-cost whole-body teleop rig + ACT imitation learning co-trained with static ALOHA data; root of the 2024-26 wheeled-bimanual teleop line.
- Robot: AgileX Tracer differential-drive base; "footprint of 90cm x 135cm"; arms reach 65 cm to 200 cm above ground vertically and extend 100 cm beyond the base; payload 1.5 kg (750 g per arm); 100 N pull at 1.5 m height; crosses 10 mm obstacles, 8 degree slopes; 30 mm ground clearance.
- Tasks / success % (co-train vs none, 50 demos unless noted): Wipe Wine 95/50; Cook Shrimp 40/20 (20 demos, 5 trials); Rinse Pan 80/0; Use Cabinet 85/85; Call Elevator 95/0; Push Chairs 80/0; High Five 85/85 (20 demos).
- Scene causes: authors flag the footprint as a problem for narrow paths (exact wording unverified); fixed arm height limits reach to lower cabinets, ovens and dishwashers; elevator door has only 30 cm clearance; open-loop 180 degree turn of ~1 m radius drifts "more than 10cm".
- Relevance: a large 0.90 x 1.35 m footprint and a fixed 0.65-2.0 m reach band; receptacles below 0.65 m are unreachable for this class of robot.
- Tags: teleop-data, whole-body-wheeled-bimanual, articulated-objects
