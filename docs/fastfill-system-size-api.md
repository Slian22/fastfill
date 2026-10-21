# 缺失家具尺寸的 API 兜底

运行时已支持 `--size-fallback api`：尺寸表没有足够样本，或没有提供尺寸表时，先调用现有 gpt-6.1-sol
API 估尺寸，再把确定后的尺寸交给 27B-P 摆放。这里提供的是功能，尚无这个混合策略的新质量评测。

此功能在当前仓库代码中；冻结的 `fastfill-delivery-20261015` 与硬盘 `delivery-final/` 不被改写。
可以复用交付包的 `placer/adapter` 和 `size-priors.json`，但应使用当前仓库的运行代码。

## 如何决定哪些家具调用 API

1. 类别＋房型的统计桶达到表中的 `min_count`，使用该尺寸。
2. 否则，类别桶达到 `min_count`，使用该尺寸。
3. 两者均不足时，API 只估计这些缺失家具；已有尺寸保持不变。

判断依据是桶的样本数，不是尺寸是否碰巧等于全局兜底值。每个受影响房间是一份请求，其中保留房间、
固定物体信息，只列需要估尺寸的可移动家具。没有表时，该房间的全部可移动家具都需要估尺寸。
这不补全缺失的房间边界或固定物体几何，也不新增自然语言家具描述输入。

## 使用方式

从当前仓库根目录运行。`--model` 是实际 Qwen3.8-27B 底座目录，`--adapter` 可以指向现有交付包。
输入文件沿用现有每行一个房间的 JSONL 格式。

有尺寸表，仅补缺失项：

```bash
python -m fastfill.system.cli run \
  --requests ROOMS.jsonl \
  --prior /Volumes/harddisk/fastfillout/unknownsize-20261013/delivery-final/size-priors.json \
  --model /path/to/Qwen3.8-27B \
  --adapter /Volumes/harddisk/fastfillout/unknownsize-20261013/delivery-final/placer/adapter \
  --placer vllm --escalate none \
  --size-fallback api --size-api-cap 40 --size-api-max-rooms 20 \
  --env-file ~/.fastfill_api.env \
  --out OUT_WITH_SIZE_API
```

没有尺寸表：使用同一命令，省略 `--prior`。显式指定一个不存在的文件也允许兜底；损坏文件、无权限文件、
目录或 SHA256 不匹配仍报错。设置了 `--prior-sha256` 时，表必须存在并匹配，不允许悄悄放弃绑定。

先预检：在上述命令后加 `--dry-run`，并选择一个新的 `--out` 路径。若存在缺失尺寸，输出
`DRY_RUN_SIZE_ESTIMATION_REQUIRED` 和 `would_estimate_sizes`，不加载模型、不读取 API 配置、不发请求。
此时布局升级量未知，`would_escalate` 为 `null`，不会输出虚构的布局或质量分数。
如果尺寸已全部被表覆盖，沿用原来的 dry-run 行为：运行本地放置器和 S1，不调用 API/O2。

需要同时启用现有布局升级时，将 `--escalate none` 改为 `--escalate l+o2 --escalate-on strict`，
并按原命令提供 `--api-cap`、`--api-max-rooms` 和 `--o2-model`。O2 仍需其原有 GPU 配置。
尺寸 API 可独立于布局升级使用；不需要为了估尺寸配置 O2。

## 预算与失败行为

| 参数 | 作用 |
|---|---|
| `--size-api-cap` | 尺寸阶段的 HTTP 请求硬上限，包含传输重试与格式修复 |
| `--size-api-max-rooms` | 最多允许多少个房间请求尺寸；超出时在调用前拒绝 |
| `--api-cap` | 原有布局升级的 HTTP 请求上限，独立于尺寸上限 |

需要调用尺寸 API 时，必须显式给出 `--size-api-cap`；至少要覆盖每个受影响房间的一次请求。
每房间最多一次回答＋一次格式修复，每次回答沿用客户端最多两次传输重试，因此最坏可达六次 HTTP 请求。
例如上面的 40 次预算足够 20 间房各请求两次，但不保证还覆盖所有传输重试；硬上限始终有效。
开启两种 API 阶段时，全流程上限是两个 cap 之和，实际总量写入 `api_calls_total`。
token 用量只累计已返回响应的用量；传输失败尝试可能没有可记录的 token。

返回的物体 ID 必须精确覆盖请求；尺寸必须是三个有限、正的数字且各轴小于 1000 米，不能多物体、漏物体、
使用重复键或返回位置代替尺寸。不完整的 API 响应也不采用。之后按原流程取厘米精度，最小 0.01 米。
尺寸顺序是局部 `[sx, sy, sz]`，局部 `+X` 为正面方向；不是随意交换的“宽、深”。
这些是格式、数值校验，不证明尺寸符合真实家具；后续检查器也不能代替真实尺寸验证。

预算耗尽、网络/API 失败或格式修复后仍无有效尺寸时，停止整个运行，不调用放置器，不回退到统一尺寸。
失败证据保留在 `OUT/size-api/`。CLI 仍要求输出目录全新，不会自动重用失败运行的付费响应；再次运行前应
检查已有记录，避免把重跑误认为免费续跑。

## 输出与追踪

- `rooms.jsonl`：增加 `size_sources`，按原始家具 ID 标注 `prior_category_room_type`、`prior_category` 或 `api`。
- `size-estimates.jsonl`：记录最终送入放置器的尺寸、来源，以及对应放置请求的 SHA256。
- `size-api/`：请求身份哈希、脱敏响应回执、逐次调用账本、终止状态与累计费用用量信息；未调用 API 时不创建。
- `result.json`：增加 `sizing`、尺寸预算与 `api_calls_total`；`report.md` 区分尺寸与布局 API 调用。

尺寸确定后，27B、S1、GPT/O2 布局升级和 RoomGenBench 导出均使用同一份尺寸。
该功能目前接在普通 `fastfill.system.cli run` 入口，不能与已有 `placed` 分片缓存或预先构造的 `o2` 回调混用，
以免升级答案仍对应估尺寸之前的请求；正常运行时按最终尺寸新建 GPT/O2 请求。
不带 `--size-fallback api` 时沿用原流程、原全局兜底与原输出格式。
