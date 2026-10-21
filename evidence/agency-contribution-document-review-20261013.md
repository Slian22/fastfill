# Contribution document review — 2026-10-13

**Two material corrections; no new CRITICAL/HIGH finding or deployment blocker identified by this document review.** The document appropriately separates the three model families, avoids a cross-protocol OptiScene victory claim, and limits known-source-error conclusions to the evidence actually checked. This conclusion does not close the already documented VIS-01 resource-binding gap or certify forthcoming visual results.

Reviewed `docs/research/20261013-unknown-size-comparison/contribution-and-evaluation-audit.md` in full. Initial version: 24,115 bytes, SHA256 `35bb76ff96fc3712a8286f73a30d9885f003006a95f7f361254015f82e3db6b0`. A concurrent addition of §5.4 was also read and its 18-source counts checked. Final reviewed version: 27,662 bytes, SHA256 `5b20cba1e6b225efc191b6fd487bd969cbf6ac259325b50a76a3bb9ff43a5679`. Line references below use this later version; section/quoted-text anchors remain usable if subsequent additions shift lines.

## 🟡 D1 — “dev-loss用8151/8167” claims an evaluation the frozen trainer does not perform

**Location:** §5.2, [document](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/contribution-and-evaluation-audit.md:125); phrase `dev-loss用8151/8167`.

The numbers are correct, but their role is not. The frozen trainer computes length admission for train and dev, then selects only the first two admitted dev UIDs for post-training reload/generation probes. It does not evaluate loss on the 8,151 admitted dev examples.

**Direct evidence:**

- [train_pair.py](/Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/comparison/train_pair.py:140): length-admits both splits; lines 148–154 select two dev UIDs and return only encoded training data plus those two dev records.
- [train_pair.py](/Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/comparison/train_pair.py:225): `eval_strategy='no'`. Lines 230–232 pass `train_dataset` only, with no `eval_dataset`.
- [train_pair.py](/Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/comparison/train_pair.py:290): the dev worker generates outputs and checks the output contract; its returned evidence is `clean_load`, `dev_count`, and `schema_valid`, not dev loss.
- Both downloaded `evidence/server-main-start/sft-{joint,placement}-001/run_receipt.json` records bind this exact source SHA256: `daf0dcf0c545502947c44e39d41806bd436d25179c88eb08cd24032d094152b0`. Their `/config/dev_generations_per_arm` is 2 and `/config/evidence_scope` is `matched_sft_execution_two_dev_cases_no_quality_promotion`. Both `lengths.json` records give `/dev/input_count=8167`, `/dev/accepted_count=8151`.

**Why material:** “dev-loss” implies a held-out loss measurement across thousands of rooms and overstates the validation performed. Length qualification is not model evaluation. The same wording in linked `current-results.md` should not be treated as independent corroboration.

**Suggested replacement:**

> dev的长度准入为8151/8167（16条超长排除）；当前冻结训练器不计算dev loss，仅从共同准入dev中选固定2间用于训练后重新加载与生成合同探针。正式test仍保留8647全部请求。

This is a documentation correction, not a request to change the frozen training protocol or start another job.

## 🟡 D2 — State the historical v2 table's object denominators and matching convention

**Location:** §6.1, [document](/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/contribution-and-evaluation-audit.md:182) and its three-column error table (original 24 KB version: lines 154–159).

The displayed values match the original reports, but “4455 requests” is not their averaging denominator. §4.1 explicitly defines the new J/P errors as room-weighted means; the historical table needs its different convention stated beside the numbers.

**Direct evidence:** Both original `report.json` files cited by `full-audit-legacy-measurements.json` were read and their SHA256 values matched that evidence file:

| Original report | Size-log-error denominator | Position-error denominator | Yaw-error denominator |
|---|---:|---:|---:|
| `test-main7-cell05-main-20261007b-e5-argmax/report.json` | 42,857 valid objects | 52,090 valid objects | 52,090 valid objects |
| `test-main7-cell05-main-20261007b-e5/report.json` | 42,857 valid objects | 52,090 valid objects | 52,090 valid objects |

Paths are under `outputs/fastfill_v2/comparison-20261007b/`. Exact JSON fields are `/model/reference/log_size_error/valid_objects`, `/model/reference/bottom_center_error_m/valid_objects`, and `/model/reference/yaw_error_rad/valid_objects`. Both reports have `/requests=4455`, `/inference_failed_requests=5`, and `/evaluation_matching="exchangeable_groups"`; the reference metrics condition on parsed predictions with valid labels.

[evaluate.py](/Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/v2/evaluate.py:450) implements object statistics and pools by `valid_objects` at lines 454–458. [evaluate.py](/Users/slian/Desktop/3D/Worldedge/OptiScene/fastfill/v2/evaluate.py:502) documents legal exchangeable matching and the box-equivalent size/yaw convention for axis-swap-eligible objects. The original reports record 371 box-equivalent objects and separate `*_plain_convention` fields, so the published columns are not simply fixed-ID, fixed-axis errors as in the new J/P contract.

**Why material:** The denominator and matching convention determine what these errors measure. Without this footnote, readers can apply §4.1's room-weighted definition to these historical numbers or treat 4,455 as the contributing error denominator. No displayed numeric value needs changing.

**Suggested table footnote:**

> 本表为minimal投影下、成功解析且对应标签有效的对象加权误差，非§4.1新J/P的房间等权指标。4455为请求数，其中5条推理失败；尺寸误差有效对象n=42857，位置与yaw误差n=52090。使用合法exchangeable_groups配对，并对允许轴交换的对象采用box-equivalent尺寸/yaw误差；不与固定ID的新J/P误差直接合并或排名。

## Checks and pending scope

Read-only checks performed: original v2 report and relation-summary SHA verification; exact comparison of every source-exposure summary count with its hash-matched full audit; recomputation of the legacy acceptance percentages; verification of all 18 rows and totals in the newly added source table (98,754 train / 8,647 test); reading both frozen training receipts and dev-length counts; comparison with the local paper text's tables 1–3, §4.1 and appendix D. These were artifact/claim checks, not new model evaluations or test-suite runs.

The known Scan2CAD, IL3D, SpatialGen, InternScenes, SceneSmith and InteriorGS exposure wording requires no additional material correction on the evidence checked. The document already distinguishes confirmed examples, metadata flags, direct exposure, and possible upstream effects. Its bootstrap/FID/GPT qualifications and the separation of exploratory results from confirmatory claims are appropriately bounded.

The live [official OptiScene repository](https://github.com/PolySummit/OptiScene) still supports the stated README/paper-recipe discrepancy and unreleased-weight wording. Primary pages for [DeBaRA](https://arxiv.org/html/2409.18336v1), [SceneNAT](https://arxiv.org/html/2601.07218v2), [LayoutGPT](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract.html), and [ATISS](https://arxiv.org/abs/2110.03675) were consulted for the narrowly stated related-work claims; no new exhaustive novelty review is claimed.

The subsequently added §7.1 was checked against the existing code-audit findings and the supplied [resource-drift probe](../docs/research/20261013-unknown-size-comparison/evidence/full-audit-resource-drift-probe.json). The probe records different frozen/current mesh hashes, `accepted_changed_assets=true`, and `pipeline_status="prepared"`, with one rendered case, zero failures and two prepared API cases. Its declared scope is synthetic asset inspection with mocked Blender RGBs; these counts are not evidence of actual Blender rendering or API execution. Both recorded source hashes match the current `visual_pipeline.py` and `render_assets.py`. Direct reads of `visual_cases.py:131–135,278–279`, `render_assets.py:338–353,406–437`, and `visual_pipeline.py:204–225` are consistent with the stated freeze/consumption gap. The document correctly distinguishes this demonstrated acceptance gap from actual dataset drift, which has not been observed. No additional material correction is needed in §7.1. This was inspection of the supplied probe and source, not an independent rerun of the probe. The follow-up document snapshot was 27,846 bytes, SHA256 `2a867faede6758beac73181748022f66d000a70ca4528a5c853992c07412fce9`.

**Pending by the user's instruction:** the separate code-coverage, lineage/code-ledger and final data sections. Their absence is not a finding and this review does not wait for them or certify their future content. The existing 681-test/90.87% sentence agrees with its cited validation receipt, but no coverage run was performed in this document review.

No document, production source, tests, data, or other collaborator artifacts were edited. Only this requested review report was created. No SSH, judging API, training, rendering, or deployment actions were taken.

Completed UTC: 2026-10-13T07:37:36.075172+00:00
