# Apples on the Table? Evaluating Text-Guided 3D Scene Synthesis via Fine-Grained Constraint Verification (LEGO-Eval)
- Authors: Minseok Kang, Dongwook Choi, Gyeom Hwangbo, Seungwon Lim, Kai Tzu-iunn Ong, Jinyoung Yeo. 2025. https://arxiv.org/abs/2511.03001
- Protocol: VLM agent with 21 tools (AI2-THOR); five steps (constraint identification, tool planning, argument selection/execution, validation, aggregation); verdict valid only if all binary judgements valid. Constraint types: floor layout, material, object selection, object placement.
- Holistic (Table 2): LEGO-Eval (GPT-4.1) F1 0.82, kappa 0.63; VLM-as-judge GPT-4.1 F1 0.40, kappa 0.05; Gemini 2.5 Pro F1 0.38, kappa 0.05; CLIPScore F1 0.49, kappa 0.02. Human agreement for LEGO-Eval 72.6-89.8% across methods.
- VLM-judge failure: "fails to identify and ground the scene components referenced in the description."
- Relevance: plain VLM judging of constraint satisfaction is ~chance (kappa 0.05); CLIPScore kappa 0.02. Tool-grounded geometry is needed.
