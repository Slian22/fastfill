# Video Perception Models for 3D Scene Synthesis (VIPScene)
- Authors: Rui Huang, Guangyao Zhai, Zuria Bauer, Marc Pollefeys, Federico Tombari, Leonidas Guibas, Gao Huang, Francis Engelmann. 2025. https://arxiv.org/abs/2506.20601
- Top-down metrics: CLIPScore, BLIPScore, VQAScore, GPT-4o 3-way ranking of top-down views (prompt: "Compare the room layout rationality of three methods, all generated from the same text description." rank on Semantic Correctness, Layout Correctness, Overall Preference).
- Agreement with human preference (Kendall tau, Table 2): CLIPScore 0.06; BLIPScore 0.07; VQAScore 0.13; GPT-4o single top-down 0.27; GPT-4o on multiple first-person views (FPVScore) 0.39.
- Table 1 top-down: Holodeck 29.17 / 51.27 / 81.43 / 1.98; Architect 29.95 / 49.72 / 78.34 / 1.90; VIPScene 29.98 / 54.36 / 82.13 / 2.12 (CLIP/BLIP/VQA/GPT-4o). "nearly indistinguishable".
- Relevance: the cleanest numbers that top-down CLIP/VQA do not track humans and that adding perspective (first-person) views raises GPT-4o agreement 0.27 -> 0.39.
