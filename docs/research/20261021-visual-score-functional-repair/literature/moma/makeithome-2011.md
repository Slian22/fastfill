# Make it Home: Automatic Optimization of Furniture Arrangement
- Authors: Lap-Fai Yu, Sai-Kit Yeung, Chi-Keung Tang, Demetri Terzopoulos, Tony F. Chan. ACM SIGGRAPH 2011 (TOG, DOI 10.1145/1964921.1964981). Full text: https://ww3.math.ucla.edu/camreport/cam11-49.pdf
- What: simulated annealing (Metropolis-Hastings) over furniture poses with ergonomic cost terms learned from examples.
- Accessibility: each object face gets an "accessible space" (size from examples or "the dimension of an average-sized adult"); cost C_a = sum_i sum_j sum_k max(0, 1 - ||p_i - a_jk|| / (b_i + ad_jk)) penalises any object i penetrating accessible space k of object j (b_i = object half-diagonal, ad_jk = space diagonal).
- Visibility: viewing frustum (series of rectangles) in front of TV/paintings, penetration cost of the same form.
- Pathway between doors: cubic Bezier curve covered by rectangles, furniture inside penalised (same form); control points are also optimised. No numeric walkway width in the text (pathway width "should depend on the habitant's body width", citing Ching & Binggeli 2005).
- Perceptual study: 25 participants, 70 pairs; only 3 of 21 synthesized arrangements differed significantly (p < 0.05) from designer layouts.
- Relevance: earliest layout optimiser with free-space (accessibility), visibility and door-path terms as box-penetration costs - exactly the symbolic form we need; human-sized, not robot-sized, and no reach/height.
- Tags: (none of the fixed tags applies; human-centred layout optimisation)
