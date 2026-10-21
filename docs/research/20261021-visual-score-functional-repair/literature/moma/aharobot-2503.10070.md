# AhaRobot: A Low-Cost Open-Source Bimanual Mobile Manipulator for Embodied AI
- Authors: Haiqin Cui, Yifu Yuan, Yan Zheng, Jianye Hao (Tianjin University). arXiv 2503.10070, Mar 2025. https://arxiv.org/abs/2503.10070
- What: ~$1,000 open-source bimanual mobile manipulator (SCARA-like arms on a vertical axis) + RoboPilot teleop (26-faced marker handle).
- Specs (Table II): differential drive (two BLDC front wheels + rear caster); turning sweep radius 500 mm; size 550*500*1550 mm (axes not labelled); Z-axis reach 1250 mm and "reach the floor"; max X-Y reach 750 mm; payload 1.5 kg per arm; 51 kg; gripper 120 mm.
- Results: teleop comparison over 3 tasks: RoboPilot 100% (45.22 s), SpaceMouse 81.5% (58.71 s), leader-follower 100% (65.33 s). Imitation learning, 10 trials per task: values 10/10, 10/10, 6/10, 6/10, 8/10, 7/10, 7/10 (task mapping unclear in text).
- Scene causes: Mobile ALOHA lacks a lift and could not do the fridge-low to upper-table task; light cups knocked over. No shelf-height, passage or door trials.
- Relevance: floor-to-~1.25 m vertical range is now a stated design target for home MoMa; footprint ~0.55 x 0.50 m with 0.5 m turning sweep.
- Tags: teleop-data, whole-body-wheeled-bimanual
