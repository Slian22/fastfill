# TidyBot++: An Open-Source Holonomic Mobile Manipulator for Robot Learning
- Authors: Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao, Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg. arXiv 11 Dec 2024 (CoRL 2024, PMLR 270). https://arxiv.org/abs/2412.10447 (numbers checked against the PDF text)
- What: open-source holonomic base (powered casters) + phone teleop + diffusion policies.
- Robot: base footprint 50x54 cm (text: "54 x 50 cm"), top of base ~37.5 cm above ground, 34 kg, payload 60 kg (conservative), 1 m/s, 8 h, $5.4k; Kinova Gen3 7-DoF arm (+12 kg) that can "comfortably reach down towards the ground and up towards tabletops and countertops" (no reach numbers).
- Table 1 comparison footprints (cm): Stretch 33x34, Tracer 57x69, Ranger Mini 50x74, Husky 67x99, Fetch 51x56, Tiago 54x54; weights Stretch 24.5 kg, Fetch 113 kg, Tiago 70 kg.
- Tasks / success (Table 2, diffusion policy): open fridge 10/10, wipe countertop 9/10, load dishwasher 7/10, take out trash 10/10, load laundry 7/10, water plant 6/10. Per-task success criteria not defined.
- Holonomic vs differential (wipe countertop): 9/10 vs 4/10; 2.03 m / 27.4 s vs 4.03 m / 65.2 s; "Everyday tasks such as opening doors and cabinets often require sideways motions of the mobile base to improve the workspace of the arm."
- Clearance: small footprint enables "narrow doorways and hallways" (no numbers).
- Relevance: authoritative footprint list for common MoMa bases (33-99 cm side); opening doors/cabinets needs lateral standing room beside the swing, not just in front.
- Tags: real-home-deploy, teleop-data, VLA-policy, articulated-objects
