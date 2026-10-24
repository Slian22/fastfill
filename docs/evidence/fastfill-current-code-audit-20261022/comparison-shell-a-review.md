# Comparison shell 第一半只读审查（2026-10-22）

基准 `825fd13ea`。范围为 `fastfill/comparison` 下 `.sh` / `.sbatch` 按相对路径排序的前 25 个，从 `isambard_010_cpu.sbatch` 到 `setup_optiscene_env.sh`（含），**25 文件 / 2,161 行**。逐行全文语义阅读已完成；逐文件覆盖和源码 SHA256 在 `comparison-shell-a-files.json`。

本部分**未确认新增缺陷**。25 个文件均 `bash -n` 通过（`comparison-shell-a-syntax.json`）；该结果仅证明 shell 语法可解析，不能证明 Slurm、CUDA、vLLM、网络下载或训练执行成功。本轮没有执行脚本本身、安装依赖、提交 sbatch、调用 API 或访问远程机器。

审查重点包括：CLI 分支/枚举/参数个数，分片 index/count 与输出命名，准确模型/未合并 policy 选项，dev-only/test-selection 门槛，GPU/CPU 分配与超时时限，Slurm 要求，训练目录 flock，resume 开关，后台/退出状态，env 分离，模型离线加载，API 限额/重试路径，以及缓存和完成标记的使用。

两个看似可疑的跨模块路径已与负责 Python 的代理交叉核对并排除，不报猜测：

- `isambard_014.sbatch` 的 `gen()` 遇已有 MANIFEST 会跳过；怀疑 8B dev-smoke 的 parse gate 失败后被重跑绕过。当前 `optiscene_eval.generate` 在 `:899–900` gate 失败直接抛错，直到 `:912` 才发布文件，失败不会留下完成 MANIFEST，因此该复现前提不成立。
- `isambard_021_train.sbatch:158–159` 的 probe 在 `hf.jsonl` / `vllm.jsonl` 存在时跳过；怀疑增量生成中断造成未完成 probe 被跳过。当前 `a021_train.probe_hf` 增量写的是 `hf.partial.jsonl`（`:719,728–732`），全量结束才发布 `hf.jsonl`（`:733`）；vLLM 在完整 decode 返回后发布（`:800`）；probe_report 对 UID 完整顺序有校验（`:827–829`）。故逐 room 中断导致该问题的前提不成立。

剩余后 25 个脚本由 `audit_system_v11` 阅读，Python 实现由 `audit_comparison` 主审。没有把此处 25 个语法检查夸大为全部 comparison 或集群端验证。
