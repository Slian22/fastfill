# Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)
- Authors: Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, Deva Ramanan. https://arxiv.org/abs/2404.01291
- Definition: question "Does this figure show '{text}'? Please answer yes or no."; score = P(Yes). CLIP-FlanT5 (FlanT5-11B main), finetuned on 665K VQA mix.
- Numbers (CLIPScore vs VQAScore): Winoground text/image/group 27.8/11.5/7.8 vs 60.0/57.5/46.0; EqBen 35.0/35.0/25.0 vs 59.3/63.6/47.9; TIFA160 pairwise 54.1 vs 71.2; Pick-a-Pic 76.0 vs 84.0. Per-skill spatial breakdown not extractable (unverified).
- 3D: averages scores over rendered views of 3D assets. StanfordT23D: CLIPScore pairwise 61.0, Pearson 48.1, Kendall 32.6; VQAScore 68.6/64.3/48.7.
- Relevance: improves on CLIP for compositional text, but needs a text caption; in layout-judging it scored 58.6% human agreement (Procedural Scene Programs) and Kendall 0.13 top-down (VIPScene).
