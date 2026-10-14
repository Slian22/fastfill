# FastFill unknown-size layout model: P-selected (rpo-dpo1)

**Delivered arm:** P (placement: predicts position and yaw; sizes come from the frozen training prior, category × room-type median, then category median, with constant fallback (0.40, 0.50, 0.65 m) when neither bucket has ≥ 5 training observations).

Chosen by the amendment 005 §1 rule applied to J-selected vs P-selected (amendment 010: each arm's model selected on dev only; replaces J-DPO2 vs P-DPO2 of amendment 008): deliver J iff J−P size log MAE on common-complete rooms has its paired 95 % CI below zero and the size-debiased success (a′) J−P has 95 % CI lower bound ≥ −0.02; otherwise P; any undefined interval delivers P. Delivery is an artifact choice for the user pipeline, not a claim of scientific superiority. Single training seed. Every number below is copied from the selected-stage report_v2 output (results.json sha256 9539d1044b070f8408fe874831f944e6f3b60070f263ab9056fc17e22bc8951a), the dev selection.json or the lineage receipts.

## Dev selection (amendment 010)

Each arm's model was selected on 600 dev rooms only (never used for training or a test decision; uids sha256 `2f04f2246315de81359447ae0b9d4ccb16bfd90a7de2a38923ce3b7e1104c4df`, selection.json sha256 `f094395f3be366930a39dec66c141410f2f6db560b18dd32448de35602114e13`) by the rule: eligible: SFT always; DPO candidate iff dev complete rate >= SFT - 0.01 and its dev size log MAE <= SFT's dev size log MAE + 0.01, both on the rooms complete in both it and SFT. selected: highest dev a' (own footprint bias rescale) among eligible; candidates within 0.005 of the best tie and go to fewer training stages, then higher a', then protocol order sft, e1-dpo1, e1-dpo2, rpo-dpo1, rpo-dpo2. P: eligible sft, e1-dpo1, rpo-dpo1; best dev a′ 0.4350; within 0.005 of it: rpo-dpo1; selected **rpo-dpo1**. The other arm selected sft.

| candidate | dev complete rate | dev size log MAE | SFT size log MAE (same rooms) | common rooms | own footprint bias B | dev a′ | eligible | selected |
|---|---|---|---|---|---|---|---|---|
| sft | 0.9983 | 0.2340 | 0.2340 | 599 | -0.0556 | 0.3817 | yes | no |
| e1-dpo1 | 0.9983 | 0.2339 | 0.2339 | 598 | -0.0540 | 0.3917 | yes | no |
| e1-dpo2 | 0.4283 | 0.2668 | 0.2668 | 256 | -0.0614 | 0.1767 | no | no |
| rpo-dpo1 | 0.9983 | 0.2339 | 0.2339 | 598 | -0.0540 | 0.4350 | yes | yes |
| rpo-dpo2 | 0.9683 | 0.2325 | 0.2325 | 580 | -0.0539 | 0.4000 | no | no |

## Model

- Base: Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218` (lineage root base_sha256 `69d069331c87531e7aa054e70085cef25752b3e6e3614b03e851a2e997a4b85b`).
- Delivered weights: merged BF16 model `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013/dpo-003/placement/dpo1-merged` (fastfill.merge_lora: BF16 PeftModel.merge_and_unload); model_sha256 `bbbb2c18a276d84cdd4700405e71e72bbae7db7473be287fe899baf7ef359a58` (per-file hashes in delivery.json and SHA256SUMS). Also listed: the stage adapter (`adapter/`) and its BF16 training base (`training-base/`), which together are the evaluated exact policy (amendment 009).
- Training data: FastFill data v3.2 (main-data manifest sha256 `e588e2f7c0def2c6515810816e0fd101b6dfdaa907274ce37cc9cd0d2c7c63e6`).
- Recipe (OptiScene, amendments 008/009; amendment 010 candidate rpo-dpo1): SFT → DPO-stage1 (consensus alignment; negatives = own SFT samples). e1 = the official README hyper-parameters for 1 epoch per stage; rpo = 3 epochs plus an NLL term on the chosen response (α 1.0). Each DPO stage trains a LoRA on the BF16 merge of the previous stage and is merged afterwards. Positives are a programmatic geometry-verified selection, not human approval.

| stage | hyper-parameters (lineage receipt) | DPO pairs | optimizer steps | fraction of weights changed by the BF16 merge |
|---|---|---|---|---|
| sft | SFT as trained: LoRA r32 a64, lr 5e-5, 1 epoch | n/a | n/a | 0.7762 |
| dpo1 | lr 5e-06, epochs 3, lora_r 32, lora_alpha 16, lora_dropout 0.05, beta 0.1, effective_batch 4, max_len 8192, scheduler cosine, 5% warmup, target_modules q,k,v,o,gate,up,down_proj | 2200 | 1650 | 0.0675 |

Pair summaries: dpo1 kept 2200 of 2200 positives (both arms).

## Input / output contract

Two chat messages: the fixed system prompt of the arm and a user JSON with the room boundary, boundary_type, optional height, room_type, fixed items and the movable objects (opaque IDs, categories, and sizes from the frozen prior). The answer is one JSON object `{"placements": [...]}` in listed order, as the system prompt states. Example: the shortest frozen test request of this arm (uid `il3d:b7897ce0-86d4-11f0-a478-60cf84ae2082`, historical test split; format illustration only).

```text
You are a room layout model. Place every listed movable object exactly once, in listed order.
Frame: meters, right-handed, Z up; floor z=0. boundary is a counter-clockwise floor polygon; boundary_type polygon denotes walls, hull denotes a scanned floor hull, proxy an approximate boundary. height is the ceiling height; a missing field means unknown.
Objects have opaque instance IDs and explicit categories. Fixed items have explicit categories and known size, pos and yaw; they cannot move, are not output placements, and must remain unobstructed.
size=[sx,sy,sz] denotes positive extents along local X,Y,Z; local +X is the front. pos=[x,y,z] is the bottom center. yaw is integer degrees in [0,360), counter-clockwise from room +X. on is the supporting movable or fixed ID; omit on for floor support. Supports must exist and cannot be self-references or cycles.
Use the supplied object sizes. Output only JSON {"placements":[{"id":id,"pos":[x,y,z],"yaw":deg,"on":support_id},...]}. Do not output size or change supplied dimensions.
```

```json
{"boundary_type":"polygon","boundary":[[0,0],[1.88,0],[1.87,3.66],[0,3.66]],"height":2.8,"objects":[{"id":"obj_0001","category":"bar","size":[0.55,1.25,0.92]}]}
```

## Inference

Evaluated as the exact trained policy (unmerged_fp32_stage_adapter_on_bf16_training_base) with transformers, greedy decoding (do_sample=False, num_beams=1), dtype bfloat16, thinking off (enable_thinking=False), batch 8 with left padding, context budget 40960 tokens, at most 40959 input tokens, and max_new_tokens = min(40959, 40960 − prompt tokens, 1024 + 128 × movable objects). An answer that reaches the cap without EOS counts as a failure.

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = 'model'  # the copied merged directory
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='auto')
messages = request['messages'][:2]  # system + user, exactly as in the example
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
objects = len(json.loads(messages[1]['content'])['objects'])
cap = min(40959, 40960 - ids.shape[1], 1024 + 128 * objects)
out = model.generate(ids, max_new_tokens=cap, do_sample=False, num_beams=1)
placements = json.loads(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))['placements']
```

The evaluated policy itself (the numbers above), from the listed adapter and base:

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained('training-base', torch_dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base, 'adapter')  # unmerged FP32 adapter, as evaluated (amendment 009)
```

vLLM (same prompt token ids, greedy):

```python
from vllm import LLM, SamplingParams
llm = LLM(model='model', dtype='bfloat16', max_model_len=40960)
ids = llm.get_tokenizer()(text, add_special_tokens=False)['input_ids']
answer = llm.generate([{'prompt_token_ids': ids}], SamplingParams(temperature=0, max_tokens=cap))[0].outputs[0].text
```

vLLM was not used for the reported numbers; its outputs can differ token by token.

P needs the frozen size prior shipped as `size-priors.json` (sha256 `073522c18e10ce2443189dfb8ebca90074c16ff1c2a4c292dfd6f788409d2422`); request sizes are built exactly as the evaluation did:

```python
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.size_prior import predict_sizes
prior = json.load(open('size-priors.json'))['full_prior']
room = bind_room(prepared_room)[0]
features = {'room_type': room.get('room_type'),
            'objects': [{'id': o['id'], 'category': o.get('category')} for o in room['objects']]}
sizes = {k: [max(.01, round(v, 2)) for v in vals] for k, vals in predict_sizes(features, prior).items()}
request = {'messages': messages(room, 'placement', estimated_sizes=sizes)[:2]}
```

## Measured results: secondary delivery comparison J-selected (sft) vs P-selected (rpo-dpo1) (one shared inference plan, frozen geometry code, full test set)

This comparison decides which arm is delivered (amendment 010: the two dev-selected models, replacing J-DPO2 vs P-DPO2 of amendment 008); it is secondary, not confirmatory. The protocol's primary (confirmatory) comparison stays the same-stage matched SFT J0 vs P0 (protocol.md), reported in the SFT-stage report and the stage table, not in this card. Holm p below adjusts only within (a), (b) of this comparison.

Intervals: house-component cluster bootstrap, 2000 draws, seed 42; they cover test-room sampling only, not training randomness.

| endpoint | J-selected (sft) | P-selected (rpo-dpo1) | Δ J−P [95% CI] | n | bootstrap p | Holm p (within (a), (b); secondary) |
|---|---|---|---|---|---|---|
| (a) 全请求原始几何成功率 | 0.3882 | 0.3981 | -0.0098 [-0.0213, +0.0016] | 8647 | 0.0950 | 0.0950 |
| (b) 尺寸 log MAE（共同完整） | 0.2861 [0.2806, 0.2913] | 0.2814 [0.2757, 0.2875] | +0.0046 [+0.0007, +0.0087] | 8602 | 0.0210 | 0.0420 |
| (a′) 尺寸去偏几何成功率 | 0.3439 | 0.4043 | -0.0604 [-0.0712, -0.0493] | 8647 | n/a (tested: (a), (b) only) | n/a (tested: (a), (b) only) |
| complete rate (all requests) | 0.9971 | 0.9977 | -0.0006 [-0.0021, +0.0009] | 8647 | n/a | n/a |
| footprint size bias B = mean log(pred/GT) | -0.0829 [-0.0942, -0.0724] | 0.0145 [0.0029, 0.0265] | n/a | n/a | n/a | n/a |

Delivery rule: condition (i) no, condition (ii) no → **P**. Model-free GT reference on the same requests: geometry success 0.7230, axis-aligned OOR 0.0787, rotated OOR 0.0716; P prior alone: size IoU 0.5582, size log MAE 0.2824. GT is a reference, not a ceiling.

Agent benchmark (same frozen cohort and inputs; estimation only, no significance claims):

| scope | system | requests | complete rate | (a) geometry success | (b) size log MAE |
|---|---|---|---|---|---|
| cohort | J | 1454 | 0.9986 | 0.4305 | 0.2430 |
| cohort | P | 1454 | 1.0000 | 0.5193 | 0.2388 |
| cohort | L-raw | 1454 | 1.0000 | 0.9347 | 0.3003 |
| cohort | L-harness | 1454 | 1.0000 | 0.9347 | 0.3003 |
| cohort | O6-raw | 1454 | 0.9966 | 0.7001 | 0.4216 |
| cohort | O6-harness | 1454 | 0.9986 | 0.7008 | 0.4218 |
| cohort | O4-raw | 1454 | 0.9979 | 0.6871 | 0.4461 |
| cohort | O4-harness | 1454 | 1.0000 | 0.6871 | 0.4468 |
| cohort | O2-raw | 1454 | 0.8267 | 0.8081 | 0.3298 |
| cohort | O2-harness | 1454 | 0.9580 | 0.9278 | 0.3405 |
| cohort | O3-raw | 1454 | 0.3315 | 0.2111 | 0.4518 |
| cohort | O3-harness | 1454 | 0.3941 | 0.2414 | 0.4501 |
| cohort | O7-raw | 1454 | 0.9801 | 0.5791 | 0.3844 |
| cohort | O7-harness | 1454 | 1.0000 | 0.5922 | 0.3860 |
| cohort | O1a-raw | 1454 | 0.3480 | 0.0000 | 0.6088 |
| cohort | O1a-harness | 1454 | 0.3494 | 0.0000 | 0.6089 |
| cohort | O1b-raw | 1454 | 0.2366 | 0.0426 | 0.5349 |
| cohort | O1b-harness | 1454 | 0.5750 | 0.1100 | 0.5388 |
| strata_pooled | J | 1009 | 1.0000 | 0.4490 | 0.1875 |
| strata_pooled | P | 1009 | 1.0000 | 0.5986 | 0.1883 |
| strata_pooled | L-raw | 1009 | 1.0000 | 0.9851 | 0.2551 |
| strata_pooled | L-harness | 1009 | 1.0000 | 0.9851 | 0.2551 |
| strata_pooled | O6-raw | 1009 | 0.9990 | 0.7859 | 0.3840 |
| strata_pooled | O6-harness | 1009 | 1.0000 | 0.7859 | 0.3840 |
| strata_pooled | O4-raw | 1009 | 1.0000 | 0.7691 | 0.4108 |
| strata_pooled | O4-harness | 1009 | 1.0000 | 0.7691 | 0.4108 |
| strata_pooled | O2-raw | 1009 | 0.9227 | 0.9009 | 0.3054 |
| strata_pooled | O2-harness | 1009 | 0.9931 | 0.9653 | 0.3044 |
| strata_pooled | O3-raw | 1009 | 0.3290 | 0.2151 | 0.4211 |
| strata_pooled | O3-harness | 1009 | 0.4063 | 0.2517 | 0.4240 |
| strata_pooled | O7-raw | 1009 | 0.9841 | 0.6650 | 0.3438 |
| strata_pooled | O7-harness | 1009 | 1.0000 | 0.6759 | 0.3445 |
| strata_pooled | O1a-raw | 1009 | 0.2557 | 0.0000 | 0.5034 |
| strata_pooled | O1a-harness | 1009 | 0.2567 | 0.0000 | 0.5036 |
| strata_pooled | O1b-raw | 1009 | 0.2597 | 0.0406 | 0.4936 |
| strata_pooled | O1b-harness | 1009 | 0.6105 | 0.1100 | 0.5028 |

| scope | comparison | Δ(a) [95% CI] | Δ(b) [95% CI] |
|---|---|---|---|
| cohort | J−P | -0.0887 [-0.1176, -0.0622] | +0.0045 [-0.0008, +0.0097] |
| cohort | J−L-raw | -0.5041 [-0.5305, -0.4783] | -0.0570 [-0.0631, -0.0507] |
| cohort | P−L-raw | -0.4154 [-0.4435, -0.3867] | -0.0615 [-0.0680, -0.0550] |
| cohort | J−L-harness | -0.5041 [-0.5305, -0.4783] | -0.0570 [-0.0631, -0.0507] |
| cohort | P−L-harness | -0.4154 [-0.4435, -0.3867] | -0.0615 [-0.0680, -0.0550] |
| cohort | J−O6-raw | -0.2696 [-0.3015, -0.2372] | -0.1796 [-0.1875, -0.1716] |
| cohort | P−O6-raw | -0.1809 [-0.2110, -0.1483] | -0.1834 [-0.1918, -0.1755] |
| cohort | J−O6-harness | -0.2703 [-0.3020, -0.2373] | -0.1795 [-0.1874, -0.1715] |
| cohort | P−O6-harness | -0.1816 [-0.2119, -0.1499] | -0.1832 [-0.1914, -0.1753] |
| cohort | J−O4-raw | -0.2565 [-0.2875, -0.2269] | -0.2037 [-0.2121, -0.1958] |
| cohort | P−O4-raw | -0.1678 [-0.1976, -0.1394] | -0.2077 [-0.2169, -0.1990] |
| cohort | J−O4-harness | -0.2565 [-0.2875, -0.2269] | -0.2036 [-0.2121, -0.1957] |
| cohort | P−O4-harness | -0.1678 [-0.1976, -0.1394] | -0.2080 [-0.2170, -0.1992] |
| cohort | J−O2-raw | -0.3776 [-0.4053, -0.3493] | -0.1091 [-0.1177, -0.1010] |
| cohort | P−O2-raw | -0.2889 [-0.3170, -0.2589] | -0.1119 [-0.1202, -0.1041] |
| cohort | J−O2-harness | -0.4972 [-0.5239, -0.4704] | -0.1055 [-0.1133, -0.0979] |
| cohort | P−O2-harness | -0.4085 [-0.4355, -0.3802] | -0.1092 [-0.1171, -0.1016] |
| cohort | J−O3-raw | +0.2194 [+0.1902, +0.2471] | -0.2200 [-0.2380, -0.2018] |
| cohort | P−O3-raw | +0.3081 [+0.2804, +0.3358] | -0.2216 [-0.2396, -0.2030] |
| cohort | J−O3-harness | +0.1891 [+0.1588, +0.2184] | -0.2250 [-0.2403, -0.2095] |
| cohort | P−O3-harness | +0.2779 [+0.2498, +0.3058] | -0.2252 [-0.2402, -0.2093] |
| cohort | J−O7-raw | -0.1486 [-0.1822, -0.1152] | -0.1420 [-0.1498, -0.1349] |
| cohort | P−O7-raw | -0.0598 [-0.0917, -0.0286] | -0.1458 [-0.1539, -0.1379] |
| cohort | J−O7-harness | -0.1616 [-0.1940, -0.1292] | -0.1427 [-0.1504, -0.1357] |
| cohort | P−O7-harness | -0.0729 [-0.1042, -0.0426] | -0.1472 [-0.1551, -0.1395] |
| cohort | J−O1a-raw | +0.4305 [+0.4035, +0.4573] | -0.3176 [-0.3364, -0.2984] |
| cohort | P−O1a-raw | +0.5193 [+0.4916, +0.5491] | -0.3460 [-0.3665, -0.3257] |
| cohort | J−O1a-harness | +0.4305 [+0.4035, +0.4573] | -0.3177 [-0.3364, -0.2984] |
| cohort | P−O1a-harness | +0.5193 [+0.4916, +0.5491] | -0.3462 [-0.3665, -0.3259] |
| cohort | J−O1b-raw | +0.3879 [+0.3601, +0.4152] | -0.3048 [-0.3271, -0.2816] |
| cohort | P−O1b-raw | +0.4766 [+0.4485, +0.5061] | -0.3088 [-0.3329, -0.2850] |
| cohort | J−O1b-harness | +0.3205 [+0.2922, +0.3474] | -0.3081 [-0.3214, -0.2950] |
| cohort | P−O1b-harness | +0.4092 [+0.3812, +0.4375] | -0.3126 [-0.3261, -0.2987] |
| strata_pooled | J−P | -0.1497 [-0.1873, -0.1112] | -0.0008 [-0.0057, +0.0041] |
| strata_pooled | J−L-raw | -0.5362 [-0.5708, -0.5020] | -0.0676 [-0.0753, -0.0606] |
| strata_pooled | P−L-raw | -0.3865 [-0.4210, -0.3520] | -0.0668 [-0.0738, -0.0601] |
| strata_pooled | J−L-harness | -0.5362 [-0.5708, -0.5020] | -0.0676 [-0.0753, -0.0606] |
| strata_pooled | P−L-harness | -0.3865 [-0.4210, -0.3520] | -0.0668 [-0.0738, -0.0601] |
| strata_pooled | J−O6-raw | -0.3370 [-0.3802, -0.2937] | -0.1964 [-0.2048, -0.1878] |
| strata_pooled | P−O6-raw | -0.1873 [-0.2297, -0.1451] | -0.1957 [-0.2039, -0.1872] |
| strata_pooled | J−O6-harness | -0.3370 [-0.3802, -0.2937] | -0.1964 [-0.2048, -0.1878] |
| strata_pooled | P−O6-harness | -0.1873 [-0.2297, -0.1451] | -0.1957 [-0.2039, -0.1872] |
| strata_pooled | J−O4-raw | -0.3201 [-0.3623, -0.2790] | -0.2233 [-0.2328, -0.2137] |
| strata_pooled | P−O4-raw | -0.1705 [-0.2074, -0.1340] | -0.2225 [-0.2323, -0.2126] |
| strata_pooled | J−O4-harness | -0.3201 [-0.3623, -0.2790] | -0.2233 [-0.2328, -0.2137] |
| strata_pooled | P−O4-harness | -0.1705 [-0.2074, -0.1340] | -0.2225 [-0.2323, -0.2126] |
| strata_pooled | J−O2-raw | -0.4519 [-0.4915, -0.4144] | -0.1203 [-0.1307, -0.1109] |
| strata_pooled | P−O2-raw | -0.3023 [-0.3379, -0.2707] | -0.1175 [-0.1270, -0.1086] |
| strata_pooled | J−O2-harness | -0.5164 [-0.5500, -0.4836] | -0.1171 [-0.1269, -0.1083] |
| strata_pooled | P−O2-harness | -0.3667 [-0.4011, -0.3339] | -0.1161 [-0.1251, -0.1076] |
| strata_pooled | J−O3-raw | +0.2339 [+0.1970, +0.2717] | -0.2319 [-0.2538, -0.2095] |
| strata_pooled | P−O3-raw | +0.3835 [+0.3497, +0.4188] | -0.2267 [-0.2475, -0.2041] |
| strata_pooled | J−O3-harness | +0.1972 [+0.1572, +0.2378] | -0.2387 [-0.2570, -0.2205] |
| strata_pooled | P−O3-harness | +0.3469 [+0.3111, +0.3823] | -0.2328 [-0.2498, -0.2152] |
| strata_pooled | J−O7-raw | -0.2161 [-0.2639, -0.1685] | -0.1565 [-0.1647, -0.1486] |
| strata_pooled | P−O7-raw | -0.0664 [-0.1085, -0.0236] | -0.1555 [-0.1640, -0.1474] |
| strata_pooled | J−O7-harness | -0.2270 [-0.2723, -0.1817] | -0.1570 [-0.1653, -0.1490] |
| strata_pooled | P−O7-harness | -0.0773 [-0.1188, -0.0359] | -0.1562 [-0.1650, -0.1482] |
| strata_pooled | J−O1a-raw | +0.4490 [+0.4149, +0.4825] | -0.2820 [-0.3080, -0.2543] |
| strata_pooled | P−O1a-raw | +0.5986 [+0.5628, +0.6322] | -0.2914 [-0.3187, -0.2605] |
| strata_pooled | J−O1a-harness | +0.4490 [+0.4149, +0.4825] | -0.2822 [-0.3080, -0.2547] |
| strata_pooled | P−O1a-harness | +0.5986 [+0.5628, +0.6322] | -0.2916 [-0.3189, -0.2606] |
| strata_pooled | J−O1b-raw | +0.4083 [+0.3735, +0.4400] | -0.3116 [-0.3379, -0.2846] |
| strata_pooled | P−O1b-raw | +0.5580 [+0.5223, +0.5936] | -0.3072 [-0.3336, -0.2796] |
| strata_pooled | J−O1b-harness | +0.3389 [+0.3023, +0.3719] | -0.3153 [-0.3312, -0.3001] |
| strata_pooled | P−O1b-harness | +0.4886 [+0.4531, +0.5236] | -0.3146 [-0.3301, -0.2984] |

GPT judge: not measured in this report.

FID/KID: not measured in this report.

Human user rating (UR): not measured.

Same-input agents compared in the report: L, O1a, O1b, O2, O3, O4, O6, O7.

## Limitations

- Single training seed; intervals reflect test-room sampling only.
- DPO preferences are programmatic (geometry gate, own SFT samples, position perturbations), not human preferences.
- The test split is historical and was used for earlier diagnostics; it is not a fresh blind test.
- Human UR not measured. Geometry success is a bounding-box check with tolerances and category exemptions, not mesh-level physics.
- Evaluated policy ≠ delivered bytes: the numbers come from the exact policy (unmerged_fp32_stage_adapter_on_bf16_training_base); the delivered model folds that update into BF16 weights (fraction of weights changed by its merge: 0.0675). The merged bytes were not re-evaluated; the listed adapter and base reproduce the evaluated policy.
- Model selection (amendment 010) used dev rooms only and was locked after the amendment-009 DPO1 test results had been viewed (disclosed in the amendment); no test output was used to select.
- Known source defects: trained on FastFill data v3.2. The sensitivity rows below are the report's own: the GT-pass stratum, the declared known-defect test rooms excluded and, when the report was given that frozen list, the test rooms changed by the later v3.3 source correction excluded.

| sensitivity scope (report wording) | requests | Δ(a) J−P | Δ(b) J−P | Δ(a′) J−P |
|---|---|---|---|---|
| GT 通过冻结几何门（6252） | 6252 | -0.0200 [-0.0339, -0.0060] | -0.0048 [-0.0087, -0.0007] | -0.0696 [-0.0824, -0.0560] |
| 排除声明的已知缺陷房间（3） | 8644 | -0.0098 [-0.0214, +0.0012] | +0.0046 [+0.0004, +0.0084] | -0.0604 [-0.0716, -0.0494] |
| 排除源修正候选变化房间（47） | 8600 | -0.0098 [-0.0216, +0.0014] | +0.0047 [+0.0009, +0.0085] | -0.0599 [-0.0709, -0.0495] |

Report warnings: prior_level_pair_category_unresolved_without_frozen_size_priors.

