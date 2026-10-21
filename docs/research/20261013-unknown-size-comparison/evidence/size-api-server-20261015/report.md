# 尺寸 API 服务器部署实测

状态：**REAL_SERVER_SMOKE_VERIFIED**。代码已部署到 Isambard，真实 API 与真实 27B-P 完整运行，结果已拉回硬盘。

| 情况 | 输入家具 | API 补尺寸 | 完成作业 | 完整布局 | 检查器 / 严格准入 |
|---|---:|---:|---|---|---|
| 表缺类别（卫生间） | 2 | 1 | 7450324 | 通过 | 通过 / 通过 |
| 不提供表（厨房） | 2 | 2 | 7450423 | 通过 | 通过 / 通过 |

两间均使用实际 27B-P（vLLM BF16 LoRA），没有用确定性兜底代替模型，也没有布局 GPT/O2 升级。
从 API 回执、最终尺寸表、27B 请求哈希、S1 后布局到 RoomGenBench SceneSpec 逐项核对：
物体 ID 完整且唯一，尺寸和来源一致，表内已知尺寸保持不变。

## 实际家具尺寸

以下为局部 `[sx, sy, sz]`，单位米；API 和先验尺寸都是估计，不是真实测量。

| 房间 | 家具 | 尺寸 | 来源 |
|---|---|---|---|
| bathroom | floor vanity cabinet | 0.57 × 0.98 × 0.98 | prior_category_room_type |
| bathroom | small bathroom shelf | 0.25 × 0.45 × 0.8 | api |
| kitchen | bench | 0.45 × 1.2 × 0.45 | api |
| kitchen | lamp stand | 0.4 × 0.4 × 1.6 | api |

## 调用与故障记录

共 **3 次真实 API HTTP 请求**，无 API 格式重试；预算上限 12 次。
首次无表作业 `7450325` 在获得尺寸后因 GPU 空闲显存不足停止（81.95 GiB < 87.4 GiB），
其 1 次调用计入总量。增加调用 API 前的显存检查，排除该节点，以剩余 5 次额度重跑；
`7450423` 成功且仅使用 1 次。原始失败日志与回执均保留。
成功作业各申请 1 GPU、16 CPU、100 GB RAM；Slurm 实际历时均为 5 分 43 秒。
没有修改模型、训练参数或推理参数来绕过错误。

## 服务器入口与证据

- 主机：`u6uv.aip2.isambard`
- 部署：`/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/deploy-size-api-20261015-v1`
- 结果：`/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013/size-api-runtime-20261015-v1`
- 启动方式：[服务器运行说明](/Volumes/harddisk/fastfillout/unknownsize-20261013/delivery-size-api-20261015/server/README.md)；核验：[verification.json](verification.json)；完整回执和日志在 `results/`。

Agency DevOps 代理独立复核了启动器与核验程序；核验包括失败尝试的调用量，避免漏计费用。
原冻结交付包不变；新功能部署于独立目录，代码文件 SHA256 在服务器与本地均核验。
这里只是 **2 间开发集房间的功能实测**，不据此推断整体质量或尺寸准确度提升。
