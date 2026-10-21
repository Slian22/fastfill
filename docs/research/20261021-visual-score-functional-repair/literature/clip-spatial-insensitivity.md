# CLIP insensitivity to spatial relations (three benchmarks)
1. ARO (Yuksekgonul, Bianchi, Kalluri, Jurafsky, Zou) https://arxiv.org/abs/2210.01936 : CLIP VG-Relation 0.59 (chance 50%), VG-Attribution 0.62 (50%), COCO-Order 0.46 (20%), Flickr30k-Order 0.59 (20%).
2. What'sUp (Kamath, Hessel, Chang) https://arxiv.org/abs/2310.19785 : CLIP ViT-B/32 31.0 / 47.4 / 46.9 (What'sUp / COCO-spatial / GQA-spatial, avg 41.8); ViT-L/14 26.1 / 49.5 / 47.3 (avg 41.0); random 25/50/50; human 100 / 97.3 / 99.0.
3. VL-CheckList (Zhao et al.) https://arxiv.org/abs/2207.00221 : CLIP Object 82.83, Attribute 67.93, Relation 64.19, Relation-Action 74.07, Relation-Spatial 49.1.
Take-away: CLIP is at chance on swapping spatial relations; on a single image it sees object presence, not arrangement.
