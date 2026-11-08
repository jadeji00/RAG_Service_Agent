# 踩坑记录

> 只记录**真实发生过**的问题：现象、根因、修复、防复发。
> 面试里「你遇到过什么问题」的答案，应该来自这里，而不是临场编。

## P1. 向量库路径跟着启动目录跑，出现多个 `chroma_db`

- **现象**：同一个项目在不同目录启动，页面读不到知识库；磁盘上出现四个 `chroma_db` 目录。
- **根因**：Chroma 的 `persist_directory` 传相对路径时，按**进程工作目录**解析，而不是按项目根目录。
- **修复**：初始化 `VectorStoreService` 时判断是否为绝对路径，不是则用 `get_abs_path` 锚到项目根。
- **防复发**：启动日志打印最终 `persist_directory` 与向量数；路径相关逻辑统一走 `utils/path_tool.py`。
- **状态**：已修复并实测（空库可自动重建，重复同步幂等）。
- **2026-11-06 复核**：根库 `agent` 为 66 条，66 个 ID 与内容均唯一；`rag/chroma_db`（111 条）、`agent/chroma_db`（0 条）、`agent/tools/chroma_db`（0 条）是修复前的历史残留，均被 `.gitignore` 忽略，可删除后由启动自检重建。
- **另一个常见误会**：单个 `chroma_db` 里的 `chroma.sqlite3` + 一个 UUID 目录**不是两份数据**。前者存 collection/ID/元数据，后者是 HNSW 向量索引文件，两者是同一份库的不同部分。

## P2. YAML 单引号让分片分隔符静默失效

- **现象**：`chunk_size=200` 配好了，但段落级、行级切分没有生效，块内混着多段内容。
- **根因**：`separators` 写成单引号 `'\n\n'`，YAML 不转义反斜杠，解析出来是**字面反斜杠 + n**，匹配不到真实换行。
- **修复**：改为双引号 `"\n\n"`、`"\n"`；重建向量库后核对块数与长度。
- **防复发**：配置加载后校验关键分隔符；改切分参数必须重建库并对比块数。
- **状态**：已修复；切分形态已实测，但「修复后召回质量是否提升」尚未做同 query 对比。

## P3. `stream_mode="values"` 把中间消息推给前端

- **现象**：页面会冒出工具调用的原始 JSON、用户消息重复显示；所谓「流式」其实是整段返回后按字符 `sleep` 播放。
- **根因**：`values` 每次返回完整状态，最后一条消息不一定是 `AIMessage`；`app.py` 又用逐字符打印伪装流式。
- **修复**：改为 `stream_mode=["messages","updates"]` 并只收 `metadata["langgraph_node"] == "model"` 的增量文本，工具事件从 `updates` 取；前端直接渲染增量，不再逐字打印。
- **防复发**：`tests/test_chat_runner.py` 与 `tests/test_api.py` 断言事件序列、delta 数量与最终 answer；`app.py` 改用 `st.write_stream`。
- **状态**：**已修复**。改为 `stream_mode=["messages","updates"]`，只收 `metadata["langgraph_node"] == "model"` 的增量文本；`updates` 单独用于工具开始/结束事件。`app.py` 改用 `st.write_stream`，不再逐字 `sleep`。实测 QA 回答按 token 到达（首 token 约 3.7 秒，来自 RAG 工具耗时），报告首 token 约 0.4–0.8 秒。

## P4. 导入即初始化，启动失败绕过页面兜底

- **现象**：`app.py` 明明有 `try/except`，但配置错误、模型 key 缺失时页面仍直接报错退出。
- **根因**：`agent/tools/agent_tools.py` 在模块顶层执行 `rag = RagSummarizeService()`；`model/factory.py` 也在模块顶层构造模型。这些发生在 `import` 阶段，早于 `app.py` 的 `try`。
- **修复**：RAG 改为惰性单例 `get_rag()`；FastAPI 在 `lifespan` 里才导入并构造 `ReactAgent`，导入 `server.app` 不会拉起向量库和模型；启动失败时保留 `startup_error`，`/health` 返回 503 与可读原因，`/chat/stream` 返回 `not_ready` SSE 事件。
- **防复发**：`tests/test_api.py` 覆盖「runner 未就绪时 health=503、stream 返回 error」；启动自检仍会打印向量数。
- **状态**：**已修复**（FastAPI 入口）。Streamlit 入口仍会在页面脚本导入 `ReactAgent` 时构造，失败由页面 `try/except` 显示错误，不再直接抛 traceback。

## P5. 外部 md5 台账与向量库不同步

- **现象**：删掉 `chroma_db` 后重新启动，程序认为「文件没变」而跳过灌库，知识库一直是空的。
- **根因**：去重依据放在外部 `md5.text`，它与向量库是两份状态，任何一方单独删除都会不一致。
- **修复**：把 `file_md5` 写进每个 chunk 的 metadata，以向量库自身为准；增加 `ensure_knowledge_base()` 空库自愈与已删除源清理。
- **防复发**：不再维护第二份状态；启动日志打印同步前后向量数。
- **状态**：已修复并实测（删除源后自动补回；排除本地 PDF 后当前 66 条）。

## P6. 报告字段核对只看字面，被单位和改写误伤

- **现象**：首版确定性核对把「覆盖率下降 1 个百分点」判为没有覆盖 CSV 的 `-1%`，把「主刷剩余寿命约 28 天」判为没有覆盖 `主刷寿命:剩余28天`，报告字段完整率被低估。
- **根因**：用整串 `in` 判断，没有拆「标签 + 数值」；也没有统一 `个百分点/%`、`平方米/㎡`，正负号还被当成标点删掉。
- **修复**：拆标签与数值锚点；4 字以上标签按二字片段容错；数值优先匹配；统一单位；符号只用于存在性，变化方向交给 LLM judge。
- **防复发**：单测覆盖「1 个百分点 / 1 平方米 / 主刷剩余寿命」等改写；字段覆盖只作为下界，不冒充语义等价。
- **状态**：已修复并实测（`rep-001` 从 3/4 恢复到 4/4）。

## P7. 报告题有 5/10 跳过 `fill_context_for_report`

- **现象**：最终基线里 `rep-002 / 003 / 006 / 007 / 008` 没有调用 `fill_context_for_report`；报告提示词没有切换，回答缺「特征」字段，字段完整率只有 50%。更早一次冒烟是 3/10（`rep-002 / 005 / 008`，完整率 70%），说明跳过比例会随运行波动。
- **根因**：报告流程靠系统提示词要求模型「自觉」按顺序调用工具；模型可能把「取到数据」当成任务完成，跳过只负责标记上下文的空壳工具。
- **影响**：不是生成能力问题，而是控制流不可靠；提示词约束不能保证副作用工具一定执行。另有一道 `rep-004` 调用了完整工具链但正文漏写 `耗材`，属于字段覆盖问题，不是本条。

**旧机制（有缺陷）**

1. 报告请求进入同一个 ReAct 循环，模型自己决定调哪些工具；
2. 提示词要求先调空壳工具 `fill_context_for_report`，中间件把 `runtime.context["report"]` 置为 `True`；
3. `dynamic_prompt` 看到标记后才切换报告提示词；
4. 模型再调 `fetch_external_data` 取数，最后写报告。

控制流的关键一步（切提示词）依赖模型是否记得调标记工具。模型跳过第 2 步时，第 4 步仍可能发生，但报告提示词没生效，于是字段缺失。中间件后来用「`fetch_external_data` 成功也切上下文」兜底，字段不再丢，但工具序列仍可能不完整，且报告阶段仍可能被诱导多调一次检索。

**新机制（已实施）**

1. `execute_stream` 先用纯函数 `is_report_query` 判断意图，不调用模型；
2. 命中报告后不进 ReAct，直接走 `report_pipeline`：
   - `resolve_report_request` 解析参数：显式入参 > 问题里的 4 位 ID / 年月 > mock `get_user_id` / `get_current_month`；
   - 直接调用一次 `fetch_external_data`，参数由代码传入；
   - 把原始 CSV 记录注入报告系统提示词；
   - 用 `temperature=0` 的模型**只写一次**报告；
3. 子流程用 `update_state` 把一问一答写回同一个 `thread_id`，会话历史不会断；
4. 记录为空时直接返回「未找到…」，不调用模型；
5. 子流程异常或路由未命中时，才回退 ReAct；这时 `fetch_external_data` 成功仍会兜底切换提示词。

区别不在「提示词写得更严」，而在**控制权从模型移到代码**：模型不再选择报告工具，不存在漏调标记工具；参考实现中的 `fill_context_for_report` 已从工具注册表和主提示词中删除。代价是路由规则需要维护，且报告写作仍是单次长输出（延迟问题见 P8、`docs/testing.md` 5.6）。

- **防复发**：`tests/test_report_router.py` 用 80 题数据集逐题断言路由结果与 `type` 完全一致；`tests/test_report_pipeline.py` 断言只取数一次、记录注入系统提示、`tool_calls` 只有 `fetch_external_data`、空记录不调用模型。
- **状态**：**已彻底修复**。2026-11-07 完整基线 10/10 报告题全部 `route=report_pipeline`、`pipeline_ok=True`、`fetch_args_ok=True`，字段完整率 100%；报告题 pass^3 严格通过率 100%、零波动。完整数据见 `eval/results/answer_baseline.md` 与 `eval/results/report_pass3.md`。

## P8. 报告提示词列了无关工具，模型多跑一轮

- **现象**：新版报告提示词保留 `rag_summarize` 说明后，同一条 `1001 / 2026-10` 的报告请求在 `fetch_external_data` 之后又调用 `rag_summarize`，模型轮次从 2 变成 3，耗时从 14.4 秒涨到 24.1 秒。
- **根因**：提示词里的工具说明是行动暗示。报告阶段只需要取数 + 写报告，列出检索、天气、定位工具会诱导模型「顺手」多查一次。
- **修复**：报告提示词只保留 `fetch_external_data`，明确写「不要调用 rag_summarize、get_weather、get_user_location」；报告事实只以 CSV 记录为准。
- **防复发**：剖析报告链路时同时看模型轮次、每轮输入/输出字数和耗时；正常报告应为「1 轮取数 + 1 轮生成」。
- **状态**：已修复并实测（同一请求 24.1 秒 → 9.5 秒，2 轮）。
- **2026-11-07 更新**：报告请求已改为确定性子流程，只调用一次写作模型，工具轮次由代码固定为「取数 1 次 + 写作 1 次」，不再存在模型顺手多查一次的可能。延迟剩余部分来自最终 400–600 字输出本身。

## P9. 流式回答不增量，token 用量也取不到

- **现象**：接入 SSE 后发现两件事：一是回答不是逐 token 到达，而是最后一次性返回；二是 `UsageMetadataCallbackHandler` 拿到的是空字典，账本没有 token。
- **根因**：DashScope `ChatTongyi` 在**带工具**的流式调用中默认不增量输出，需要显式 `model_kwargs={"incremental_output": True}`；用量也不在 LangChain 统一的 `usage_metadata`，而在最后一帧的 `response_metadata["token_usage"]`。
- **修复**：主模型与报告模型都设 `streaming=True` + `incremental_output=True`；`observability/usage.py` 同时兼容两种字段来源，并用供应商 `request_id` 去重，避免同一帧跨多次回调重复累加。
- **防复发**：`tests/test_observability.py` 覆盖字段归一、提取与去重；SSE 实跑断言 delta 数量 > 1；账本只在 `usage_source=provider` 时算成本。
- **状态**：已修复并实测。QA 单次 3,415–3,482 tokens（2 次模型调用），报告 829–837 tokens；报告首 token 约 0.4–0.8 秒。

## P10. 护栏提示不能直接插 `SystemMessage`

- **现象**：连续拦截工具后想给模型注入一条系统提示，调用报错或提示词不生效。
- **根因**：DashScope `ChatTongyi` 要求 system 消息必须位于消息列表首位，中途追加 `SystemMessage` 不合法。
- **修复**：改用带 `additional_kwargs={"guardrail": True}` 的 `HumanMessage` 注入；`load_history` 跳过这类消息，不会出现在界面历史里。
- **防复发**：`tests/test_middleware.py` 断言注入消息带标记、只注入一次；历史读取有过滤分支。
- **状态**：已修复。

## P11. 提示词要求「输出思考过程」，把工具调用前的旁白推给了用户

- **现象**：SSE 里先出现「我需要调用 rag_summarize 获取资料」「我将为您查询……」等文字，然后才是工具事件和正式回答。
- **根因**：主提示词写着「每次调用工具前必须输出真实的自然语言思考过程」；模型照做，这段文字进入 `AIMessage.content`，被当成 delta 推给前端。
- **修复**：提示词改为「调用工具时直接发起，不要输出思考过程、工具名、参数或『我将调用……』」；最终回答只给结论和建议。
- **防复发**：实跑检查第一段模型输出的 `content` 应为空、只有 `tool_calls`；SSE 首屏不应出现工具名旁白。
- **状态**：已修复并实测（首个模型步骤 content 为空，回答只保留最终文本）。

## P12. `execute_stream` 语义变了，Streamlit 还在按「整段回答」处理

- **现象**：`execute_stream` 改为按 token 产出后，Streamlit 仍把最后一个 chunk 当作完整回答写入历史，且继续逐字 `sleep` 播放。
- **根因**：接口语义从「一次性整段」变成「增量片段」，旧调用方没有同步改；这是接口演进时的兼容性遗漏。
- **修复**：`app.py` 改用 `st.write_stream(res_stream)`，把返回的片段拼接后写入历史，删除逐字 `sleep`。
- **防复发**：`execute_stream` 的 docstring 明确写「只吐文本增量」；新代码统一用 `iter_chat` 的 `result.answer` 作为最终答案，不再取最后一个 delta。
- **状态**：已修复；Streamlit 页面未纳入自动化测试，只做了代码路径检查。
