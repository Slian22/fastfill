# Procedural Scene Programs for Open-Universe Scene Generation: LLM-Free Error Correction via Program Search
- Authors: Maxim Gumin, Do Heon Han, Seung Jean Yoo, Aditya Ganeshan, R. Kenny Jones, Kailiang Fu, Rio Aguina-Kang, Stewart Morris, Daniel Ritchie. 2025 (SIGGRAPH Asia 2025).
- URL: https://arxiv.org/abs/2510.16147 (read via /html/2510.16147)

## Protocol (Sec. 6.2 "Automated Evaluation")
- Scored: whole layouts vs the scene prompt. Judge sees rendered images only (not code).
- Judge: GPT-4o. Pairwise: two layouts + general task prompt; judge first lists pros and cons of each relative to the scene prompt, then picks the better one ("LLMCompare"). Exact prompt text and camera views NOT given in the text I could read (unverified). Human study rendered colored boxes with names on hover.
- Ablation: removing the pros/cons step lowers human agreement.

## Agreement with human majority vote (Table 3)
- LLMCompare 77.1%; LLMCompare (no +/-) 70.0%; VQAScore 58.6%; DSG 50.7%.
- Table 4 (Ours win rate) automated vs human: vs DeclBase 77.1% vs 82.9%; vs Holodeck 90.0% vs 94.3%; vs FlairGPT 85.7% (automated only). Abstract: humans preferred the method 82% and 94% vs two declarative baselines.
- Table 5 (vs Holodeck by object count): 10-20: 66.7%; 20-30: 80.0%; 30-40: 72.9%; 40+: 95.2%.

## Failure modes
- VQAScore recognizes the scene type even when layout is poor -> near chance. DSG yes/no questions often tie. CLIP not used as metric (only for retrieval).
- Authors: judge is "an empirical proxy whose biases may steer future systems toward its own blind spots."

## Relevance
Strongest direct evidence that (a) a pairwise MLLM judge with pros/cons beats prompt-text-alignment metrics, (b) VQAScore/DSG are ~chance for layout. Note: their prompt contains a scene prompt; ours has only room type + categories, so the judge prompt must be rewritten without a description.
