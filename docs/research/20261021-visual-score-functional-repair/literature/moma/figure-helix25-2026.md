# Helix 2.5: Zero-Shot 30-Home Generalization (Figure AI blog)
- Figure AI, 17 Sep 2026. https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization (company post, not peer reviewed)
- What: humanoid running the Helix 2.5 VLA, evaluated in 30 unseen Bay Area homes with no data collected there; robot model not named on the page.
- Tasks and success definition (full task, no partial credit): living-room tidy ("All 13-15 toys scattered in the scene are picked and placed in the basket"), towel folding (all towels folded and in basket), bed making (both pillows and comforter corners at the top, comforter smooth). Timeouts 1 min per toy, 3 min per towel, 1 min per pillow or comforter side.
- Result: 56% overall zero-shot (from-scratch model 9%). Per-task 67/62/40% and 237/420 trials appear only in secondary write-ups (unverified).
- Scene statements: homes are "tight, cluttered spaces"; wheeled robots need "enough open floor"; robot "may need to walk to find an object, then shift its stance to reach it". No failure analysis.
- Relevance: industry framing that free floor area and stance space near targets are the home constraint; no numbers on clearances.
- Tags: humanoid, VLA-policy, real-home-deploy
