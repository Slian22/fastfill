# 定向安全复核（只读，2026-10-22）

基准 `825fd13ea`。范围仅限 FastFill HTTP 请求 I/O 阻塞和 RoomGenBench 请求文件输出边界。已复核当前源码、`root_probes.py` / `root-probes.json` 及此前 v2 临时目录复现；没有重复运行可信探针、扩展扫描、修改源码/测试或访问远端。

**确认：HTTP 请求体可无限占住整个串行服务。** `fastfill/serve.py:74` 按 Content-Length 阻塞读取，Handler 未设置 socket timeout；`:125` 使用单线程 HTTPServer。现有本机探针先声明 4 字节但只发送 1 字节，第二个完整请求在 0.3 秒读取期限内收不到回复；补齐第一个请求后第二个立即收到 400。探针记录 `handler_socket_timeout=null`，不涉及模型调用。源码证明没有服务端读取期限，1 MiB 大小上限不能约束等待时间。任何能连接监听端口的客户端均能触发服务不可用；默认监听 `0.0.0.0`（`:107`），但本轮未证明公网实际可达或已有利用。

修复建议应优先保留 HTTPServer 和模型串行调用：在连接 setup、解析请求行/headers 之前设置有限 I/O timeout，给请求体读取增加总截止时间，超时/短读后关闭连接并在可行时返回明确错误。仅设置 `HTTPServer.timeout` 或只在 `do_POST` 内设置超时不能完整覆盖已接受连接的 header/body 读取；单次 socket idle timeout 也不能限制持续滴送字节的总时间。当前 Python 标准库的 StreamRequestHandler.setup 会把 Handler.timeout 应用到 accepted socket，BaseHTTPRequestHandler.handle_one_request 会捕获 TimeoutError 并关闭连接，可作为最小修复的入口。

**不能直接将其替换为 ThreadingHTTPServer 后就认为修完。** `:112` 只创建一个 LLM，`:115–121` 的 generate 闭包共享该实例；直接线程化会并发进入同一 `llm.generate`，改变文件 `:18` 明示的 one-request-at-a-time 合同。若以后要并发处理网络 I/O，须有有界接入/队列，并由单一推理 worker 或显式锁串行执行该模型调用。此次建议不要求改变推理并发。

**确认：RoomGenBench scene_key 可以跨越写入保护边界。** `fastfill/v2/roomgenbench.py:286,292–293` 将未验证的 scene_key 当作路径写入；`:272–275` 只校验顶层目录。已保存的 CPU 复现用 `../reference/protected`，在全新输出目录条件下成功覆盖 sibling reference 内已有 JSON，且返回成功。该缺陷影响运行进程可写的相应 `.json` 文件；触发需要导入这种场景 JSON。没有证据证明标准 benchmark 输入或历史数据已被覆盖，也没有由此推断远程代码执行。

修复建议：先验证所有 scene_key 为单一安全 basename、排除绝对路径/分隔符/路径跳转，再检查每个 resolved 输出的直接父目录确实为 target；保留 reference/input 保护；所有名称与 payload 验证完毕后才写入，使用独占创建防止已有文件覆盖。复现和原始详情见 `repro-v2-roomgen-path.py`、`v2-repro-results.json`、`v2-review.md`。

结论：两项均有当前触发证据，可纳入本轮审查。HTTP 建议须保证网络读取有界、单实例 LLM 仍串行；路径问题须在每个实际输出路径上封闭边界。未发现本次证据涉及凭据泄漏，无凭据轮换操作。
