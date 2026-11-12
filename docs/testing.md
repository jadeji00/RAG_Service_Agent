# 测试与评测说明

> 测试要回答三个问题：**评测集和标签有没有错？指标算得对不对？系统现在到底是什么水平？**
>
> 这份文档说明当前测了什么、怎么测、结果如何、为什么这样设计，以及哪些还没测。

## 1. 为什么要测试

1. **评测集是尺子，尺子错了会带偏优化。** 如果 gold 标错，调切分、换 embedding、加重排都可能朝着错误方向走；所以先把数据校验做成自动化。
2. **RAG 失败要能定位到层。** 是文件没召回、片段没命中，还是答案生成了但没引用？不分层就只会说“感觉不太准”。
3. **改检索必须有回归基线。** M2 做了 QA 切分、混合召回和重排，靠同一套题和同一套指标才能证明每一步是进步还是退步——事实上切分单独上线时 Span@3 是掉的，没有分阶段基线就会得出错误结论（见 5.13）。
4. **面试要报数字，数字要能复现。** 题量、标注方式、指标定义、运行命令、失败案例，缺一项都会被追问穿。

## 2. 测试分层

| 层 | 文件 | 测什么 | 是否调用模型 | 速度 |
|---|---|---|---|---|
| 数据契约 | `eval/run_eval.py --validate-only` | easy 80 题 + hard 25 题 schema、ID、gold 标签、报告数据 | 否 | 毫秒级 |
| 检索纯函数单测 | `tests/test_eval_metrics.py` | 匹配规则、排名、MRR、分位数 | 否 | 毫秒级 |
| 判分纯函数单测 | `tests/test_judge.py` | JSON 解析、单位归一、报告字段核对、拒答信号 | 否 | 毫秒级 |
| 数据集单测 | `tests/test_dataset.py` | 题量分布、ID 唯一、gold 与当前数据一致 | 否 | 毫秒级 |
| 报告路由单测 | `tests/test_report_router.py` | 意图识别、参数抽取、80 题路由与 `type` 一致 | 否 | 毫秒级 |
| 报告子流程单测 | `tests/test_report_pipeline.py` | 只取数一次、记录注入、空记录不调用模型 | 否（mock 模型/工具） | 毫秒级 |
| 工具护栏单测 | `tests/test_guardrails.py` | 参数校验、预算、重复成功拦截、失败重试一次、参数规范化 | 否 | 毫秒级 |
| 越界路由单测 | `tests/test_scope_router.py` | 15 道 easy 负样本、washing-machine 反例、报告优先级 | 否 | 毫秒级 |
| 中间件单测 | `tests/test_middleware.py` | 报告上下文切换、非法参数前置拦截、连续拦截后强制收口只注入一次 | 否 | 毫秒级 |
| 观测底座单测 | `tests/test_observability.py` | 用量归一/提取/去重、成本估算、trace/span 落库查询、按会话删除 | 否 | 毫秒级 |
| 会话索引单测 | `tests/test_session_index.py` | 线程过滤、UUIDv6 时间解析、首问标题与截断、checkpoint 分组排序、按线程删除 | 否 | 毫秒级 |
| 事件流单测 | `tests/test_chat_runner.py` | SSE 事件序列、trace 落库、错误/取消、缺用量不伪造成本 | 否（fake agent） | 毫秒级 |
| API 单测 | `tests/test_api.py` | health、chat SSE、会话列表/历史/删除、trace、usage、503 not_ready；登录态、401/403、会话按归属过滤、`owner_user_id` 透传 | 否（fake runner） | 毫秒级 |
| 身份单测 | `tests/test_auth.py` | 密码哈希加盐与校验、畸形哈希拒绝、账号种子幂等、`demo_users()` 不含密码、环境变量覆盖密码、登录/登出/过期令牌、会话认领与归属过滤、旧会话收养、关闭鉴权仍可用 | 否 | 毫秒级 |
| pass^k 单测 | `tests/test_passk.py` | 严格通过、至少一次、波动题、repeat=1 | 否 | 毫秒级 |
| 切分单测 | `tests/test_chunking.py` | QA 对/错误码块/编号列表不切断、preamble 归属、元数据、切分签名随参数变化、真实语料零丢行 | 否 | 毫秒级 |
| 混合检索单测 | `tests/test_hybrid_retrieval.py` | 分词（ASCII + CJK 单/双字）、BM25 打分与截断、RRF 融合、vector 模式退化、索引重建、stats 字段 | 否（fake 向量库） | 毫秒级 |
| 重排单测 | `tests/test_reranker.py` | 属性/字典两种响应解析、越界 index、非 200 / 异常 / 空结果退回 None、空文档不调 API | 否（stub `_call`） | 毫秒级 |
| RAG 服务单测 | `tests/test_rag_service.py` | 无证据兜底话术、越界优先于兜底、有证据仍调模型、兜底命中拒答信号、`rag.yml` 重排配置装配 | 否（stub chain） | 毫秒级 |
| 阈值标定 | `eval/calibrate_min_score.py` | 全量题跑真实管线，统计各组最高重排分分布并给出可分阈值 | 是（embedding + 重排） | 约 40 秒 |
| 检索端到端 | `eval/run_eval.py` | 真实 embedding + Chroma + BM25 + 重排的 Recall/MRR/延迟 | 是（embedding + 重排） | 约 20 秒 / 套 |
| 回答级端到端 | `eval/run_answer_eval.py` | easy 55+15+10 或 hard 20+5；correctness / faithfulness / 拒答 / 字段完整性 | 是 | easy 约 5 分钟，hard 约 3 分钟 |
| pass^k | `eval/run_answer_eval.py --repeat k` | 同题独立跑 k 次，严格 pass^k / 至少一次 / 波动题 | 是 | 约 k 倍回答级耗时 |

**为什么把单测和端到端分开**：单测不依赖 API Key、网络和向量库，提交前可以随时跑；端到端会调用 embedding，结果受数据和模型影响，单独作为基线命令执行。

## 3. 具体测了什么

### 3.1 数据集契约（`validate_rows`）

| 检查项 | 规则 | 失败示例 |
|---|---|---|
| 题型 | 只能是 `answerable / negative / report` | 拼错 `answerble` |
| ID | 全库唯一 | 两题都写 `qa-001` |
| 问题 | 非空且不重复 | 复制粘贴后忘改 |
| gold_answer | 非空 | 只写标签没写答案 |
| 来源文件 | `gold_sources` 必须存在 | 改名 `选购指南.txt` 后没同步 |
| 章节/题号 | 必须能在某个 gold 来源中找到 | 数据改版后 `Q3` 消失 |
| 关键词 | 必须能在某个 gold 来源中找到 | 写了原文没有的“高温烘烤” |
| 多来源题 | 章节/关键词命中任意一个来源即可 | 把两个来源做笛卡尔积误报 |
| 负样本 | `expected` 必须是 `refuse` | 写成 `answer` |
| 报告题 | `user_id + month` 必须存在于 `records.csv` | 选了不存在的 `1011/2026-12` |
| 报告字段 | `expected_fields` 必须是 CSV 真实列 | 写了 `总结` 但 CSV 没这列 |

这一步是**防止“假失败”**：如果标签本身错了，跑出来的低分不能用来指导优化。

### 3.2 数据与检索单测（10 个）

| 测试 | 验证点 | 为什么重要 |
|---|---|---|
| `test_normalize_removes_whitespace` | 换行/空格不影响匹配 | 中文标题常被切分器换行 |
| `test_source_basename_handles_windows_path` | Windows 绝对路径取文件名 | metadata 里是 `D:\...`，gold 里是文件名 |
| `test_term_match_requires_gold_source` | 关键词命中但来源不对，判不相关 | 防止别的文件“碰巧”出现同词 |
| `test_section_fallback_when_no_terms` | 无关键词时按章节/题号匹配 | Q&A 文件按 Q 号标注 |
| `test_rank_is_one_based` | 排名从 1 开始 | MRR 的分母不能从 0 开始 |
| `test_summarize_ranks` | Recall@k 与 MRR@k 的数学 | 指标算错，后面全错 |
| `test_percentile` | P50/P95 取最近秩 | 延迟统计不引入 numpy |
| `test_dataset_size_and_distribution` | 80 题 = 55/15/10 | 防止误删或漏加题 |
| `test_all_ids_unique` | ID 无重复 | 逐题结果能对上 |
| `test_gold_labels_match_current_data` | 80 题 gold 全部可追溯 | 数据一改就能发现标签过期 |

### 3.3 判分单测（12 个）

| 测试 | 验证点 | 为什么重要 |
|---|---|---|
| `test_extract_json_from_code_block` | 兼容 ```json 代码块 | 模型常把 JSON 包进 Markdown |
| `test_extract_json_from_prose` | 兼容前后解释文字 | 判分器要能提取真正对象 |
| `test_extract_json_invalid` | 非法输出返回 None | 不把错误输出当 0 分 |
| `test_compact_removes_markdown_and_unifies_units` | 去 Markdown、统一 `%`/`㎡` | 报告排版不同不等于事实缺失 |
| `test_complete_answer_covers_all_fields` | 完整报告四字段全过 | 核对逻辑的正例 |
| `test_paraphrased_units_still_match` | 「1 个百分点」「主刷剩余寿命」 | 防止改写被误判缺字段 |
| `test_missing_comparison_is_detected` | 缺对比字段能抓出来 | 核对逻辑的负例 |
| `test_positive_change_without_plus_sign` | CSV `+1%` 对报告「提升 1%」 | 正负号不能导致漏判 |
| `test_baseline_month_without_comparison` | 「基线数据，无对比」 | 基线月没有变化是合理的 |
| `test_refusal_phrases` | 拒答/无法回答/不在范围 | 规则信号用于诊断 |
| `test_normal_answer_is_not_refusal` | 正常回答不误报 | 避免规则把业务回答当拒答 |
| `test_load_records_indexes_user_and_month` | CSV 按 `(用户ID, 月份)` 建索引 | 报告题取数不能串行 |

### 3.4 报告路由与子流程单测（10 个）

| 测试 | 验证点 | 为什么重要 |
|---|---|---|
| `test_parse_user_id_excludes_year` | 4 位 ID 不把 `2026` 年份当用户 | 报告问题里同时有年月和 ID |
| `test_parse_month_formats` | `2026-10` / `2026/9` / `2026年7月` 统一 | 用户输入格式不固定 |
| `test_report_query_with_explicit_params` | 明确写「报告」即命中 | 报告意图的主路径 |
| `test_report_query_with_explicit_word` | 「使用记录 / 使用情况」也命中 | 同义表达不能漏 |
| `test_knowledge_question_is_not_report` | 「耗材多久换一次」不误判 | 防止知识问答被送进报告模板 |
| `test_dataset_routing_split` | 80 题的路由结果与 `type` 完全一致 | 路由规则的回归基线 |
| `test_generate_report_injects_record_and_fetches_once` | 只取数一次、记录进系统提示、工具序列只有 fetch | 控制流由代码固定，不靠模型自觉 |
| `test_empty_record_skips_model_write` | 空记录直接返回、不调用模型 | 避免无数据时编造报告 |
| `test_missing_params_fall_back_to_mock_tools` | 缺 ID/月份时用 mock 工具补齐 | 保留演示路径 |
| `test_explicit_month_is_normalized` | 显式月份也归一成 `YYYY-MM` | 防止 `2026-9` 与 CSV 的 `2026-09` 对不上 |

### 3.5 中间件兜底单测（8 个）

| 测试 | 验证点 | 为什么重要 |
|---|---|---|
| `test_fetch_switches_report_context` | 取数成功即切换报告上下文 | 路由未命中、回退 ReAct 时的兜底 |
| `test_rag_does_not_switch_report_context` | 普通检索不误切报告模式 | 防止问答被报告模板污染 |
| `test_failed_fetch_does_not_switch_report_context` | 取数失败不切换 | 失败时不应假装进入报告场景 |
| `test_invalid_args_are_blocked_before_handler` | 非法参数不进入工具函数 | 护栏必须在副作用之前 |
| `test_should_switch_report_tool_set` | 触发工具集合固定为 `fetch_external_data` | 防止新增工具时误配；`fill` 已删除 |
| `test_no_injection_before_two_blocks` | 只拦截 1 次不注入 | 避免单次参数错误就打断正常重试 |
| `test_injects_once_after_two_blocks` | 连续 2 次拦截注入且只注入一次 | 防止模型无限试错 |
| `test_no_guard_state_is_noop` | 没有 context 时不报错 | 单测/旧调用方可降级运行 |

### 3.5.1 工具护栏单测（7 个）

| 测试 | 验证点 |
|---|---|
| 参数校验 | `fetch_external_data` 的 ID/月份、天气城市、RAG query 长度、无参工具拒绝 payload |
| 预算 | 第 7 次工具调用被拦截 |
| 重复成功 | 相同参数成功执行后再次调用被拦截 |
| 失败重试 | 失败后允许同一调用重试一次，第二次失败后拦截 |
| 参数规范化 | 字典键顺序不同视为同一调用 |

### 3.5.2 越界路由单测

- 15 道 easy 负样本全部路由到 `out_of_scope`，不调用模型；
- 可回答的 easy/hard 题不被误判为越界；
- 「拖布能用洗衣机洗吗」保留为正常问答，「上门修洗衣机」「洗衣机报 E01」判越界；
- 报告优先于越界：同时包含报告词与越界词的请求会先走报告，这是当前的已知边界，未为混合意图单独设计分流。

### 3.5.3 观测与事件流单测

- 用量：`prompt_tokens/completion_tokens` 与 `input_tokens/output_tokens` 都能归一；同一 `request_id` 只累加一次。
- 成本：模型无单价或缺用量时返回 `None`，不估默认值。
- trace：路由/工具 span 可查；未知字段写入被拒绝；不存在的 trace 返回 `None`。
- SSE：事件顺序为 `meta → tool(start) → tool(end) → delta… → done`；异常只给通用 `error` 且 trace 记 `status=error`；客户端关闭生成器时 trace 记 `cancelled`。
- 越界请求的 `usage_source="none"`，与「供应商漏报 `missing`」区分。

### 3.6 pass^k 单测（3 个）

| 测试 | 验证点 | 为什么重要 |
|---|---|---|
| `test_strict_and_at_least_once` | 严格 pass^k、至少一次、平均单次、波动题 | 指标定义不能含糊 |
| `test_repeat_one_returns_none` | `repeat=1` 不输出 pass^k | 单次运行没有稳定性含义 |
| `test_attach_runs_keeps_first_run_fields` | 追加多次结果时保留首次字段 | 不破坏原有逐题输出 |

### 3.7 检索端到端

对 easy 55 道 / hard 20 道 `answerable` 题：

1. 用题目原文调用 `RagSummarizeService.retrieve(question, k=8)`；
2. 取回 Top-8 片段，按 `gold_sources` + `gold_terms`/`gold_sections` 判断相关性；
3. 记录首次命中排名、Top 来源、单题延迟；
4. 汇总 Span Recall@k、Span MRR@k、Source Recall@k 和 P50/P95。

**Span 与 Source 分开报**的原因：Source 高但 Span 低，说明文件找对了、片段没切对；两者都低，才是召回或 embedding 的问题。

### 3.8 回答级端到端

三条链路分开跑，避免用同一种「满意」口径糊在一起：

| 类型 | 实际走的链路 | 怎么判 |
|---|---|---|
| answerable | `RagSummarizeService.retrieve(k=3)` + `answer()` | LLM judge 对照 gold answer 打 correctness / faithfulness |
| negative | 同一条 RAG 链路 | LLM judge 判是否拒答；规则关键词只做诊断 |
| report | `is_report_query` 命中后直接 `generate_report`（解析 → fetch → 一次写作）；未命中才回退 `ReactAgent.invoke` | 确定性核对字段 + LLM judge 判忠实度与建议 |

几个关键取舍：

- **问答/拒答只评 RAG 子链路**：避免 Agent 的工具选择噪声掩盖生成问题；报告题才评控制流。
- **报告题优先走确定性流程**：评测直接调 `agent.generate_report()`，拿到 `route`、`tool_calls`、`fetch_args_ok` 等可审计字段；只有路由未命中才回退 `ReactAgent.invoke()`，并如实记为 `pipeline_ok=False`。
- **报告字段不交给模型判**：CSV 是事实源，标签 + 数值 + 单位归一能稳定抓漏字段；模型只判「数字是否忠实、建议是否可执行」。
- **判分器强制 JSON、失败重试一次**：解析失败记为 `judge_error`，不静默当 0 分。
- **pass^k 在同一进程内重复调用**：每题用独立 `thread_id`/独立生成，不复用上一轮回答；报告子流程本身无会话依赖，直接重复调用。

### 3.9 M2 检索管线单测（42 个）

检索管线是 M2 唯一「代码写错就悄悄掉分」的地方，所以四段各有一层单测，全部不调模型：

| 段 | 文件 / 数量 | 关键验证点 |
|---|---|---|
| 切分 | `tests/test_chunking.py`（8） | `Q1：…A1：…` 不跨 chunk；`【E05】` 错误码块的原因与处理绑在一起；编号列表按组聚合且每组重复标题；小节未超限不硬切；开头 preamble 独立成块；`section`/`unit`/`mode` 元数据齐全；改 `chunk_size` 后 `split_signature` 变化；真实 5 个语料文件切分后**一行都不丢** |
| 召回 | `tests/test_hybrid_retrieval.py`（14） | 分词把 ASCII 词整词保留、CJK 出单字 + 双字；BM25 让稀有词（`E05`）压过高频词；`top_n` 截断；未知词返回空；两路都命中的 chunk 在 RRF 后上浮；`mode=vector` 时融合结果与向量序完全一致（保证 A/B 只有一条代码路径）；候选数不会小于 `k`；语料指纹变化才重建索引、不变时 `refresh()` 是 no-op；stats 暴露延迟与指纹 |
| 重排 | `tests/test_reranker.py`（9）+ 同文件重排段（6） | 响应既支持对象属性也支持字典；返回的 `index` 越界时跳过该项而不是错位；非 200 / 抛异常 / 空结果都退回 `None`；空文档列表根本不发请求；上层拿到 `None` 时**保持 RRF 顺序**并记 `rerank=fallback`；`min_score` 能剔掉弱候选；全部低于阈值时返回空列表 |
| 生成 | `tests/test_rag_service.py`（5） | 上下文为空 → 返回 `NO_EVIDENCE_ANSWER` 且**不调用模型链**；越界判定优先于无证据兜底；有上下文时链路正常被调用且 context 里带原文；兜底话术能被 `refusal_cue` 识别为拒答；`rag.yml` 的 `enabled/model/candidates/min_score` 真的被装配进 pipeline |

设计取舍：**重排器把 `_call` 单独拆出来当测试缝**，这样解析、排序、降级三条分支都能离线测，不需要 mock 整个 `dashscope` 模块；BM25 用自己的 4 条假语料而不是真实向量库，保证「稀有词压过高频词」这类断言不受语料改版影响。

### 3.10 无证据阈值标定（`eval/calibrate_min_score.py`）

`min_score` 不能拍脑袋写：太高会把能答的题拒掉，太低就形同没有兜底。标定脚本做三件事：

1. 把 easy 80 题 + hard 25 题**全部**跑一遍真实管线（向量 + BM25 + RRF + 重排，k=3），记录每题的最高重排分、保留条数、候选池大小与延迟；
2. 按 `answerable / negative / report` 分组统计分布；negative 额外区分**是否已被 `scope_router` 拦下**——已拦下的那 15 条根本不会走到阈值判断，把它们算进来会把阈值抬高、误伤可答题；
3. 在所有观测到的分数里搜索「answerable 保留率 100%」约束下拒答率最高的分界点，两组可分时取间隔中点作为建议值。

输出 `eval/results/min_score_calibration.{json,md}`，含分布表、建议阈值和边界样本表（分数最低的 answerable 与最高的 negative 各若干条），换语料或换重排模型后重跑即可。

### 3.11 身份与会话归属单测（27 个）

鉴权这类代码的特点是「漏一个分支就等于没有」，所以两层都按分支铺满，全部离线：

| 文件 / 数量 | 关键验证点 |
|---|---|
| `tests/test_auth.py`（17） | 同一密码两次哈希结果不同（盐生效）、`verify_password` 对正确/错误密码分别真/假、畸形哈希（段数不对、算法不认识、盐不是 hex）一律当失败而不是抛异常；同盐同迭代次数可复现；账号种子重复跑不会覆盖已有行；`demo_users()` 只回 `username/display_name/user_id`，**不含哈希**；`AUTH_DEMO_PASSWORD` 能覆盖配置密码；空用户名/空密码/不存在账号/错密码都登录失败；绑定的业务用户 ID 会跟着 `User` 一起传出去；登出只撤销那一个令牌；`token_ttl_hours=-1` 时令牌立刻查不到；乱写的令牌查不到；`claim_thread` 先写先占（第二个人认领返回真正的归属者）；`owned_thread_ids` / `owner_filter` 只放行自己的线程；`release_thread` 删除归属行；`adopt_threads` 只收养没有归属的行、不抢别人已认领的；`enabled=False` 时所有方法仍可用 |
| `tests/test_api.py::AuthApiTest`（10） | 业务接口无令牌 401（`/sessions`、`/history`、`/usage`、`/traces`、`DELETE`、`/chat/stream`），`/health`、`/`、`/auth/bootstrap` 公开；裸令牌与错误 scheme 不放行；`bootstrap` 不泄露任何凭据字段；密码错误 401；`/auth/me` 与登出；令牌换个 `TestClient` 仍然有效、过期后失效；别人的线程 403（对话/历史/删除三条路径）且 `owner_user_id` 透传正确；会话列表按归属过滤；不带 `thread_id` 时每个用户各拿到一个新的 32 位 hex；`enabled=False` 时行为回到旧版（无令牌 200、不过滤、不写归属行、`owner_user_id` 为 `None`） |

设计取舍：**`_AppTestCase` 基类持有 `setUp/tearDown/login/claim`，`ApiTest` 与 `AuthApiTest` 各自继承**——第一版把 `AuthApiTest` 写成 `ApiTest` 的子类，结果 8 个旧用例被重跑一遍，既慢又让失败信息指错地方。哈希迭代次数在测试里降到 1000，算法与校验逻辑不变。

### 3.12 会话历史压缩单测（51 个）

`tests/test_history_compressor.py`，**全程离线**：摘要器用注入的 `FakeSummarizer`（记录每次入参与 config，返回固定摘要），不调 DashScope。压缩这类代码的风险不在「压不压得动」，而在「压错了会不会污染回答」，所以用例按七个分组铺满（`PlanFoldTest` 6 / `BuildTranscriptTest` 4 / `CompressorViewTest` 17 / `UnsourcedIdentifierTest` 8 / `MergeSummaryIntoSystemTest` 3 / `CompressHistoryMiddlewareTest` 7 / `SummarizeHistoryStatsTest` 6）：

| 分组 | 关键验证点 |
|---|---|
| `PlanFoldTest` | 低于 `trigger` 返回 `None`；边界不越过最后一条真实用户消息（**当前轮不折**）；保留段不以 `ToolMessage` 开头（**不拆工具组**，否则模型会收到没有调用记录的孤儿结果）；`after == before - boundary`（摘要不占对话条数）；护栏注入的消息不算「真实用户消息」 |
| `BuildTranscriptTest` | 工具结果只留工具名不留正文（检索片段体积最大、时效最短）；`AIMessage(tool_calls)` 写成 `[调用工具] name`；护栏消息不进稿；超长消息截断加 `…`；空内容行丢掉 |
| `CompressorViewTest` | 未触发/关闭时原样返回；触发时视图只剩保留段、摘要走 `stats["summary_text"]`；折叠区没变**复用缓存且摘要 token 为 0**；折叠区变长时**从原文重写而不是链式合并**（断言第二次入参里没有 `【已有摘要】`/`【新增对话】`，但含最早那轮的原文）；不同 thread 不共享缓存；超出 `cache_threads` 淘汰最旧；摘要器抛异常 / 返回空 → 退回完整历史且 `fallback=True`，**失败路径也要把已花掉的 token 记进账**；`usage_sink` 同时收到摘要用量；对话稿超 `max_transcript_chars` 时截断只影响送出去的部分、不影响编造校验用的完整稿；配置从 `agent_conf["history"]` 读取（含 `max_summary_chars` 与配置文件一致）；**摘要调用一定带 `SUMMARY_STREAM_TAG`**（见 P18） |
| `UnsourcedIdentifierTest` | 凭空出现的 `XG900`/`U123456`/ISO 日期被抓出来；对话稿里真实存在的标识符放行；**紧跟中文的标识符（`E05为悬崖异常`）不被误判**（见 P20）；`2.4G`、`3-6 个月` 这类正常写法不误伤；被拦下时 `reason="unsourced_identifiers"` 且 `unsourced` 记进统计 |
| `MergeSummaryIntoSystemTest` | 摘要接在 system 消息**末尾**而不是替换它；入参是纯字符串 / `None` 都能处理；模板里带「不要在回答里复述或照搬它的格式」 |
| `CompressHistoryMiddlewareTest` | context 里没有 compressor 时透传；触发时 `override` **同时**改 `system_message` 与 `messages`；**原 request 不被就地修改**（`override` 是返回新对象）；摘要为空/被校验拦下时不 `override`（否则会把完整历史截掉）；一轮多次模型调用各记一条 stats；摘要器失败仍用完整历史调模型 |
| `SummarizeHistoryStatsTest` | 一轮多次调用聚合成一条：`compressed_calls`/`fallback_calls`/`duration_ms` 求和，`before_messages` 取最大、`after_messages` 取最小；全是 fallback 时带上 `reason` 与去重后的 `unsourced`（上限 10 个）；空列表返回 `no_model_call` |

设计取舍：`_Request` 是手写的假 request（只有 `messages`/`system_message`/`runtime.context`/`override`），不去构造真的 `ModelRequest`——中间件只用到这四个属性，造假对象能让用例聚焦在「有没有正确 override」上。`FakeSummarizer` 的默认回复刻意只用 `_history()` 里真实出现过的词，否则会被编造校验拦下、导致一批无关用例失败（这个坑踩过一次）。

### 3.13 引用溯源单测（36 个）

分散在 5 个文件里，**全部离线**（不检索、不调模型）：`tests/test_citations.py` 22 个（`BuildSourcesTest` 6 / `LoadHistoryCitationTest` 6 / `MergeSourcesTest` 7 / `SummarizeRetrievalStatsTest` 3），`test_middleware.py` 的 `CitationCaptureTest` 7 个，`test_rag_service.py` 的 `RagServiceCitationTest` 3 个，`test_chat_runner.py` 的 `ChatRunnerCitationTest` 3 个，`test_hybrid_retrieval.py` 的 `RerankStageTest` 多 1 个。

| 分组 | 关键验证点 |
|---|---|
| `BuildSourcesTest` | 编号跟检索顺序一致；**绝对路径只留文件名**（Windows 反斜杠与 POSIX 正斜杠都要能剥掉，否则会把服务器目录结构推到浏览器）；没有 `rerank_score` 时是 `None` 而不是 `0`（0 会被读成「完全不相关」）；预览压掉换行并截到 `PREVIEW_CHARS`；缺元数据时字段是空串而不是 `KeyError` |
| `MergeSourcesTest` | 同一段资料被两次检索命中只留一条，**分数取较高的那次**；顺序按首次出现、编号连续重排；`None` 分数能被后来的真实分数替换；超过 `MAX_SOURCES` 截断；`None`/字符串/空列表这类脏输入被忽略；**不就地修改入参** |
| `SummarizeRetrievalStatsTest` | 空列表返回 `{"calls": 0}`；候选数、重叠数、三段延迟、重排 token 求和；`mode`/`index_size` 取最后一次；`rerank_top_score` 取最大值（代表这一轮最强的一条证据）；`rerank` 是去重排序后拼接（`fallback,ok` 说明这一轮有一次重排挂了）；非 dict 项被忽略 |
| `CitationCaptureTest` | artifact 里的 `sources`/`retrieval` 被收进 `context`；两次调用按顺序累积；**没有 artifact 的工具（`get_weather`）不会凭空造出 `sources_raw` 键**；引用为空时不收集（无证据兜底不该留下「检索过一次」的假记录）；工具抛异常 / 被护栏拦截时都不收集；`context` 为 `None` 时工具照常返回 artifact 不报错 |
| `RagServiceCitationTest` | 正常路径返回 `(回答, 引用, 统计)` 三元组、统计原样透出、分数原样带出；**越界拒答返回空引用**（那条回答一个字都没用到检索结果，列出「参考资料」等于暗示它有依据）；无证据兜底返回空引用 |
| `ChatRunnerCitationTest` | `done` 带 `sources`；有检索时多一条 `kind="retrieval"` 的 span，`duration_ms` = 向量+BM25+重排三段相加（**不含生成**，与 tool span 的整段墙钟时间互补），`detail` 里带引用条数；result 事件没有 `retrieval` 时 `sources` 是 `[]` 且不落空 span；`retrieval={"calls":0}` 同样不落 span |
| `LoadHistoryCitationTest` | 引用挂到**紧随工具结果之后的那条最终回答**上；护栏注入的 `HumanMessage` 不清空待定引用（它出现在工具结果和最终回答之间，当成新一轮就会把这一轮的引用顺手清掉）；一轮两次检索的引用被合并；**新的真实用户消息会清掉上一轮没被消费的引用**（模型没给出最终回答时不能顺延到下一轮）；没有 artifact 的旧会话不产生 `sources` 键；artifact 里 `sources` 为空时跳过 |
| `RerankStageTest`（新增 1 个） | 重排后返回的是**带分数的副本**，正文与来源不变；重排失败/关闭时返回原对象、**不带分数**；混合模式下连续两次检索，**缓存语料的 metadata 始终不含 `rerank_score`**——原地写会把「上一个 query 的相关度」留给下一个 query，而这个 bug 只在混合模式出现（向量模式的假检索器每次给新列表） |

设计取舍：`load_history` 的用例用 `object.__new__(ReactAgent)` 绕过 `__init__`（真构造会打开 Chroma 并灌库），只注入一个实现 `get_state` 的 `FakeGraph`。`RagServiceCitationTest` 沿用 `test_rag_service.py` 的做法：构造真服务对象但把 `chain` 换成 `StubChain`、把 `retrieve_with_stats` 换成返回固定文档的 lambda；`tearDown` 里用 `__dict__.pop` 删除实例属性而不是赋回，否则实例属性会一直遮住类上的真方法、污染后面的用例。

## 4. 怎么运行

```bash
# 1) 只校验数据集，不调用模型
python eval/run_eval.py --validate-only

# 2) 纯函数与数据集单测，不需要 API Key
python -m unittest discover -s tests -v

# 3) 冒烟：只跑前 10 道可答题
python eval/run_eval.py --limit 10

# 4) 完整检索基线
python eval/run_eval.py --ks 1,3,5,8
python eval/run_eval.py --dataset eval/dataset_hard.jsonl --ks 1,3,5,8

# 5) 回答级冒烟：每类前 2 题 / 报告 1 题
python eval/run_answer_eval.py --types answerable,negative --limit 2
python eval/run_answer_eval.py --types report --limit 1

# 6) 完整回答级基线
python eval/run_answer_eval.py
python eval/run_answer_eval.py --dataset eval/dataset_hard.jsonl \
  --out eval/results/hard_answer.json

# 7) pass^k：同一题独立跑 k 次
python eval/run_answer_eval.py --dataset eval/dataset_hard.jsonl \
  --repeat 3 --out eval/results/hard_pass3.json

# 8) k 敏感性：只改回答级检索条数，重排与切分不动
python eval/run_answer_eval.py --dataset eval/dataset_hard.jsonl \
  --k 5 --out eval/results/hard_answer_m2_k5.json

# 9) 标定无证据阈值（会真实调用 embedding + 重排，约 40 秒）
python eval/calibrate_min_score.py

# 10) 检索 A/B：改 config/rag.yml 的 retrieval.mode / rerank.enabled 后重跑第 4 步
#     mode: hybrid + rerank.enabled: true   → 当前生产配置
#     mode: hybrid + rerank.enabled: false  → 只看混合召回
#     mode: vector + rerank.enabled: false  → 回到 M2 之前的纯向量
```

> A/B 靠改配置而不是加命令行开关：管线只有一条代码路径，`mode=vector` 时 BM25 路返回空列表、RRF 融合退化成向量原序，因此三档对比不会引入「测试专用分支」。改完记得把配置恢复成生产值。

结果写入：

- `eval/results/baseline.json`：检索聚合指标 + 逐题排名 + Top 来源；
- `eval/results/baseline.md`：检索人读版，含失败题；
- `eval/results/answer_baseline.json`：回答级聚合指标 + 逐题回答、判分理由、报告事实核对；
- `eval/results/answer_baseline.md`：回答级人读版，含失败样例与延迟；
- `eval/results/hard_baseline.*`：hard split 检索基线；
- `eval/results/hard_answer.*`：hard split 回答级基线；
- `eval/results/hard_pass3.*`：hard split pass^3，含严格通过率与波动题；
- `eval/results/report_pass3.*`：easy 报告题 pass^3，用于验证报告子流程稳定性；
- `eval/results/{easy,hard}_m2a_chunking.*`：M2-A QA 感知切分后的检索结果（145 chunks）；
- `eval/results/{easy,hard}_m2b_hybrid.*`：M2-B 加上 BM25 + RRF 混合召回；
- `eval/results/{easy,hard}_m2c_rerank.*`：M2-C 加上 `gte-rerank-v2` 重排；
- `eval/results/{easy,hard}_m2d_final.*`：M2-D 最终配置复跑（与 m2c 一致，确认兜底不影响检索）；
- `eval/results/{easy,hard}_answer_m2.*`：M2 后的回答级结果；
- `eval/results/hard_answer_m2_k5.*`：hard 回答级 k=5 敏感性对照；
- `eval/results/min_score_calibration.*`：重排分分布与阈值建议。

## 5. 本次结果

### 5.1 测试

| 项 | 结果 |
|---|---|
| 数据集校验 | `VALIDATION_OK rows=80`、`VALIDATION_OK rows=25`，0 个错误 |
| 单元测试 | **247 个全部通过，约 3 秒**（M2 前 84 个；M2 后 126 个：新增切分 8、混合检索 20、重排 9、RAG 服务 5；身份上线后 160 个：新增 `test_auth.py` 17、`AuthApiTest` 10、报告归属 4、报告参数补齐 3；历史压缩上线后 211 个：新增 `test_history_compressor.py` 51；引用溯源上线后 247 个：新增 `test_citations.py` 22、`CitationCaptureTest` 7、`RagServiceCitationTest` 3、`ChatRunnerCitationTest` 3、重排副本回归 1） |
| 向量库 | **145 个向量**、145 个唯一 ID、145 段唯一内容、5 个来源/md5 组合、`hnsw:space=cosine`、`split_sig=95ec9f1c38e8`（M2 前：66 个向量、默认 L2） |
| 索引幂等 | 同一语料第二次 `load_document()` 全部跳过，不重新 embedding、不产生重复 ID |
| easy 回答级评测 | 80 题全部执行，生成与判分错误 0（M2 后复跑见 5.13） |
| hard 回答级评测 | 25 题全部执行，生成与判分错误 0 |
| HTTP 冒烟 | QA / 报告 / 越界三条链路 + `/health`、`/history`、`/traces`、`/trace/{id}`、`/usage` 均返回正常 |
| Python 编译 | `py_compile` 通过 |

### 5.2 检索基线（M2 前的起点）

> 以下是 2026-11-07 的旧基线，保留作为对比起点；M2 之后的数字见 **5.13**。

**easy split**：55 道可答题，66 chunks，`text-embedding-v4`，Top-8：

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall | 65.45% | 94.55% | 98.18% | 98.18% |
| Span MRR | 65.45% | 78.79% | 79.52% | 79.52% |
| Source Recall | 85.45% | 100.00% | 100.00% | 100.00% |

延迟：P50 126.05 ms / P95 296.50 ms。

**hard split**：20 道可答题，同一向量库与 embedding：

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall | 55.00% | 80.00% | 90.00% | 90.00% |
| Span MRR | 55.00% | 66.67% | 69.17% | 69.17% |
| Source Recall | 75.00% | 100.00% | 100.00% | 100.00% |

延迟：P50 156.92 ms / P95 607.88 ms。

easy 与 hard 的差距是 10.45–19.09 个百分点（@1 / @8），而 Source Recall@3 都是 100%，说明改写问法后文件仍能找到，掉分主要在片段级——固定 200 字切分把跨章节信息拆散，这是 M2 QA 感知切分的直接依据。

### 5.3 一个检索失败（已在 M2-A 修复）

`qa-042`「拖布多久清洗一次？」是旧基线里唯一在 Top-8 完全未命中片段的题：

- 正确答案在 `扫拖一体机器人100问.txt` 的 Q3；
- 旧版 200 字硬切把 Q1、Q2、Q3 挤进同一个 chunk，主题被稀释；
- Top-8 召回的是 Q12（拖布能不能机洗）、Q13（多久换一次）等相邻主题；
- 这不是标签错，而是**固定字符切分混合问答对**的实证，直接对应 M2 的 QA 感知切分。

**结果：QA 感知切分单独就把这题修好了**，`span_rank` 从 `None`（Top-8 未命中）变成 **1**，且 M2-B/C/D 三个阶段都稳定在第 1 位；回答级 correctness 也从 0 → 1 → **2**。这是「诊断到层 → 对症下药 → 用同一道题验证」的一次完整闭环，不需要动 embedding 或加候选数。

### 5.4 回答级基线（旧，2026-11-07）

`qwen-max` 生成与判分，RAG 检 Top-3，报告走确定性 `report_pipeline`；80 题全部完成（2026-11-07 13:36:55）：

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| answerable | 55 | 98.18% | 完全正确 80.00%；完全有据 98.18%；幻觉 0.00% |
| negative | 15 | 86.67% | 拒答正确率 86.67% |
| report | 10 | 100.00% | 字段完整 100.00%；忠实 100.00% |
| **合计** | **80** | **96.25%** | - |

知识问答：correctness=2 有 44 题、=1 有 10 题、=0 有 1 题（`qa-042`）；faithfulness=2 有 54 题、=1 有 1 题、=0 有 0 题。关键词命中率 83.33% 仅作诊断，不计分。

报告：四个字段全部 10/10 覆盖，平均字段覆盖率 100%，平均事实片段覆盖率 99%；完全有据率 70%，可执行建议率 100%，**报告子流程执行率 100%**，`fetch` 参数正确率 100%。10 道题全部 `route=report_pipeline`，不再出现漏调标记工具。

延迟（生成 P50 / P95）：answerable 1677 / 3269 ms，negative 1071 / 2207 ms，report **7251 / 10224 ms**；判分 P50 均在 1.0–1.4 秒。报告 P50 比旧版 12492 ms 下降 **42%**，详见 5.6。

### 5.5 回答级失败（旧基线）

| 题号 | 现象 | 根因 |
|---|---|---|
| `qa-042` | 未回答清洗频率，判 correctness=0 | 检索 Top-8 未命中 Q3，生成器拿不到答案（5.3） |
| `neg-004` | 直接写出 Python 爬虫代码 | 越界任务没有明确拒绝，提示词缺边界示例 |
| `neg-009` | 追问要翻译的文本，准备执行翻译 | 同上，把超范围任务当成了待补全请求 |

这轮 `neg-011`（先说资料没有洗衣机信息，随后又给通用排查建议）被 judge 判为已拒答并通过，说明拒答边界本身有波动；`neg-004` / `neg-009` 连续两轮稳定失败，是需要靠提示词解决的确定性问题。`qa-042` 这次从 correctness=1 掉到 0，也说明检索没修好时，回答通过与否取决于模型有没有“猜对”。

报告题这次没有字段缺失，也没有路由回退：`特征` 10/10，`pipeline_ok_rate` 从旧版的 90% 提升到 100%。仍有 3 道题 faithfulness=1（`rep-008/009/010`），表现为个别数值推断，不是字段缺失；完全有据率 70%。

其中报告流程问题已记为 [P7](pitfalls.md)；拒答问题来自提示词边界，报告字段问题已由确定性报告子流程消除。

### 5.6 报告延迟剖析

**结论：主因是最终报告的输出长度，不是检索或提示词本身的长度。**

对 `rep-001` 的同一条请求做回调剖析（旧提示词）：

| 阶段 | 耗时 | 输出 |
|---|---|---|
| 模型第 1 轮（选工具） | 3.50 s | 131 字，`fill_context_for_report` + `fetch_external_data` |
| 工具执行 | 5 ms | 217 字 CSV 记录 |
| 模型第 2 轮（写报告） | 10.90 s | 579 字 |
| 合计 | 14.4 s | - |

最终写作占 **76%**，工具执行可以忽略。旧基线 10 道报告题的长度与耗时相关系数为 **0.82**；新提示词把平均长度从 535 字压到 425 字后，相关降到 **-0.19**（单次样本，API 抖动占主导），说明长度不再是唯一变量，但单请求剖析仍显示最终写作是大头。

提示词确实会放大延迟，但方式不是「提示词太长」，而是「提示词诱导了多余的一轮工具调用」：新版提示词第一版保留 `rag_summarize` 说明，模型在取数后又检索一次，轮次从 2 变成 3，同一请求耗时从 14.4 秒涨到 **24.1 秒**；移除无关工具后回到 **9.5 秒**。

已做的优化，分三步：

1. **提示词层**：固定五段模板（特征 / 清洁效率 / 耗材 / 环比 / 建议）；报告阶段只保留 `fetch_external_data`，禁止 `rag_summarize`、天气、定位；用户已给 ID/月份时跳过 `get_user_id` / `get_current_month`。
2. **兜底层**：`fetch_external_data` 成功即切换报告提示词，避免模型漏调标记工具导致字段缺失。
3. **结构层（当前）**：删除 `fill_context_for_report`，报告请求走 `report_pipeline`——解析参数、直接取数、把记录注入系统提示词、只调用一次模型写作。工具轮次由代码固定，不再有「选工具」那一轮。

三个版本的对比（同 10 道报告题，单次运行）：

| 指标 | 旧提示词 + ReAct | 优化提示词 + ReAct | 确定性报告子流程 |
|---|---|---|---|
| 平均回答长度 | 535 字 | 425 字 | **379 字** |
| 字段完整率 | 50% | 100% | 100% |
| 报告子流程执行率 | - | 90%（`fill` 可漏） | **100%** |
| 生成 P50 | 15.3 s | 12.5 s | **7.25 s（-42%）** |
| 生成 P95 | 18.0 s | 16.5 s | **10.22 s（-38%）** |
| 生成均值 | 14.3 s | 13.0 s | **7.31 s（-44%）** |

当前 10 道题全部 `route=report_pipeline`、`fetch_args_ok=True`，字段 10/10；`rep-001` 的 400 字报告耗时 7.26 秒。P50 从 12.5 秒降到 7.25 秒，主要来自两处：少了一次「模型选工具」的往返，以及模板把平均输出从 425 字压到 379 字。

边界：这是单次运行，API 抖动仍会影响绝对值，不能把 7.25 秒当成稳定上限；报告写作本身仍要 5–10 秒。M3 上线 SSE 后，报告首 token 实测 0.44 秒、完整回答 7.5 秒——用户不再盯着空白等待，但总时长仍由输出长度决定，要再压到 5 秒内需要更快的报告模型或更短模板。

### 5.7 hard split 回答基线

`qwen-max` 生成与判分，RAG 检 Top-3；25 题（20 可答 + 5 负样本）全部完成：

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| answerable | 20 | 95.00% | 完全正确 75.00%；完全有据 90.00%；幻觉 0.00% |
| negative | 5 | 100.00% | 拒答正确率 100.00% |
| **合计** | **25** | **96.00%** | - |

知识问答：correctness=2 有 15 题、=1 有 4 题、=0 有 1 题；faithfulness=2 有 18 题、=1 有 2 题、=0 有 0 题；关键词命中率 67.50%。延迟：answerable P50 1706 / P95 3374 ms，negative P50 976 / P95 2230 ms。

唯一失败是 `hard-qa-016`（儿童锁/边刷防护）：检索 Top-8 未命中，回答漏了核心要点。另一道检索未命中的 `hard-qa-007`（老人操作方式）回答被判 correctness=1、通过，属于模型用常识补上了部分内容——这类“检索失败但回答通过”不能算检索修好，和 `qa-042` 是同一个问题。

hard 的完全正确率 75.00% 明显低于 easy 的 80.00%，关键词命中率 67.50% 也低于 easy 的 83.33%；这组题才更能反映改写问法下的真实水平。

### 5.8 pass^k：单次通过会骗人

对 hard 25 题、easy 10 道报告题各独立跑 3 次：

| 数据集 | 题数 | 严格 pass^3 | 至少一次通过 | 平均单次通过 | 波动题 |
|---|---|---|---|---|---|
| hard answerable | 20 | 95.00% | 100.00% | 96.67% | 1 |
| hard negative | 5 | **60.00%** | 100.00% | 86.67% | 2 |
| hard 合计 | 25 | **88.00%** | 100.00% | 94.67% | 3 |
| easy report | 10 | **100.00%** | 100.00% | 100.00% | 0 |

最刺眼的是 hard negative：单次基线是 **5/5 全部拒答**，pass^3 严格只有 **3/5（60%）**，另外两道都是「3 次里 2 次通过」：

- `hard-neg-002`（能不能自己爬楼梯）：有一轮没有明确说「不能」，而是给了高度 1.5–2 cm、需要装坡道之类的操作建议，被 judge 判为没有拒答；
- `hard-neg-004`（手机无线充电）：有一轮说「不确定设备是否支持」，随后继续给充电注意事项，越界建议让拒答不彻底。

`hard-qa-016` 则是 3 次里只对 1 次：检索没命中时，模型有时能靠常识补出儿童锁，有时不能。报告题严格 pass^3 = 100%、0 道波动，和确定性报告子流程的设计一致——控制流固定后，报告不再受「模型这次记不记得调工具」影响。

这就是 pass^k 的价值：**单次 100% 不等于稳定 100%**。hard negative 的 5/5 是运气好，严格 pass^3 的 60% 才是当前真实稳定性。easy 80 题没有跑 pass^3，原因是成本约为单次的 3 倍；hard 25 题 + 报告 10 题先用较小成本回答「稳不稳」。

### 5.9 M3 / M4 / M5 实跑

服务用 `uvicorn server.app:app --port 8123` 启动，前端与接口走同一条 SSE 链路。三条请求各跑一次（2026-11-08）：

| 链路 | 路由 | 首 token | 端到端 | 回答长度 | token | 成本（CNY） | 工具 |
|---|---|---|---|---|---|---|---|
| 知识问答（滤网更换） | `qa` | 3.67 s | 4.79 s | 86 字 | 3,415 | 0.008786 | `rag_summarize` 1 次 |
| 使用报告（1001 / 2026-06） | `report` | 0.44 s | 7.54 s | 406 字 | 837 | 0.004183 | `fetch_external_data` 1 次 |
| 越界（写快速排序代码） | `out_of_scope` | 0.02 s | 0.03 s | 99 字 | 0 | 不计 | 0 次 |

观察：

- **QA 首 token 慢在 RAG 工具**：路由立即返回，检索约 2.5–3 秒，拿到资料后模型才开始流式输出；不是前端或 SSE 卡住。
- **报告首 token 快**：确定性流程先取数，取数完成后立即开始写作；用户先看到报告开头，完整 406 字在 7.5 秒结束。
- **越界不烧 token**：25 ms 返回固定拒答，`usage_source=none`、`cost=None`，`/usage` 不会把它算成“供应商漏报用量”。
- **账本一致**：同一时段 `/usage?days=1` 汇总 3 次请求、4,252 tokens、0.012969 元，与三条 trace 相加一致。
- `/trace/{id}` 返回路由 span + 工具 span；`/history/{thread}` 能恢复一问一答（报告/拒答经 `update_state` 写回同一 thread）。

测试覆盖：84 个单测全部通过；其中 API/事件流 13 个用 fake runner，不依赖模型与网络。**未测**：真实浏览器交互、断线重连、多用户并发、Streamlit 页面（仅代码路径检查）、护栏在真实模型反复试错下的行为（只在单测里构造拦截状态）。

### 5.10 M4 确定性拒答后的 easy 复跑

2026-11-08 用同一套 easy 80 题复跑（结果文件 `eval/results/answer_after_m3m4m5.json` / `.md`）：

| 类型 | 样本 | 旧通过率 | 复跑通过率 | 关键指标（复跑） |
|---|---|---|---|---|
| answerable | 55 | 98.18% | **100.00%** | 完全正确 81.82%；完全有据 100.00%；幻觉 0.00% |
| negative | 15 | 86.67% | **100.00%** | 拒答正确率 100.00%；生成 P50 135ms |
| report | 10 | 100.00% | **100.00%** | 字段完整 100.00%；完全有据 90.00% |
| **合计** | **80** | **96.25%** | **100.00%** | - |

两个变化必须拆开归因：

- **negative 100% 是规则带来的，不是模型进步**：15 道 easy 负样本全部命中 `is_out_of_scope_query`，`RagSummarizeService.answer` 直接返回固定话术，不调用生成模型，所以 P50 135ms、0 token。hard 5 道负样本是“能力边界”问题，规则不拦，仍是模型判断，pass^3 只有 60%——easy 的 100% 不能代表真实边界能力。
- **answerable 100% 是单次通过率，不是检索修好**：完全正确率仍是 81.82%（45 题），`qa-042` 这次是 correctness=1 的部分通过，检索 Top-8 仍未命中 Q3。与 5.3 的结论一致，M2 的 QA 切分仍是待办。

> 后续：M2 已完成，这题的检索与判分都修好了，easy 完全正确率升到 90.91%，见 **5.13**。

报告复跑：四个字段 10/10，平均事实片段覆盖率 97%，忠实率 100%，完全有据率 90%，子流程执行率 100%，`fetch` 参数正确率 100%；生成 P50 7.38 秒、P95 11.88 秒。

### 5.11 会话侧栏与回答区简化

2026-11-08 验证「历史会话可选 + 首问命名 + 去掉最近 trace / 工具 chip / 统计行」：

| 项 | 结果 |
|---|---|
| 自动测试 | 80 个单测全部通过；新增会话索引 5 个、`/sessions` 接口 1 个 |
| `/sessions` | 真实库返回 1 条：标题「拖布多久清洗一次？」、2 轮、时间 2026-11-08T11:59:47.876Z |
| 点击切换 | 点击列表第二条后 URL 更新为新 thread，4 条历史气泡恢复，当前项高亮 |
| 新会话 | 发一句越界问题后，列表出现「帮我写一段 Python 爬虫代码 · 刚刚 · 1 轮」，不调用模型 |
| 刷新恢复 | 刷新页面后仍停在该会话，列表与消息都能恢复 |
| 回答区 | 正文章节之外的 `#trace-list`、`.tool-chip`、`.stats` 数量均为 0；工具与 token 只留后端接口 |
| 未测 | 宽屏视觉效果（in-app 浏览器无可视视口，只做了 DOM 断言）、快速连续切换的竞态、几十条会话的滚动性能 |

### 5.12 删除历史会话

2026-11-10 验证「侧栏悬停删除 + 二次确认 + checkpoint/trace 同步清理」：

| 项 | 结果 |
|---|---|
| 自动测试 | 84 个单测全部通过；新增会话索引删除 2 个、trace 删除 1 个、`DELETE /sessions/{id}` 1 个 |
| 接口删除 | 临时会话返回 `{"checkpoints":1,"writes":0,"traces":1,"spans":1}`；之后 `/sessions` 不再返回该线程，`/trace/{id}` 为 404 |
| 参数与不存在 | 非 32 位 hex 返回 422；合法但不存在的 ID 返回 404；Agent 未就绪返回 503 |
| 当前会话删除 | 确认文案带上会话标题；删除后列表 3 → 2，URL 切到新的 32 位 thread，当前项变为「新会话（未发送）」 |
| 非当前会话删除 | 列表少一条，URL 与当前会话不变；删除按钮只出现在真实历史会话上，未发送的占位项没有 |
| 数据隔离 | 只用临时线程 `aaaa…`/`bbbb…`/`dddd…` 做实测；用户自己的两条会话未改动 |
| 未测 | 原生确认框的实际渲染与键盘操作（in-app 浏览器视口宽为 0，侧栏被媒体查询隐藏，只断言了确认文案与后续请求）、删除与正在流式生成的并发、超大线程的删除耗时 |

真实浏览器之外的替代验证：用 `innerWidth=0` 的 in-app 浏览器直接调用行内按钮点击，并临时替换 `window.confirm` 记录文案、返回 `true`；轮询 DOM 直到目标行消失，再回查 `/sessions`、`/trace` 与 checkpoint 表。

### 5.13 M2 检索升级前后对比

2026-11-13 完成 M2：**QA 感知切分 → BM25 + 向量 RRF 混合召回 → `gte-rerank-v2` 重排 → 无证据兜底**，并把向量库距离从默认 L2 换成 cosine。四段每改一段都用同一套 easy 80 题 + hard 25 题重跑，不靠「感觉变准了」。

#### 分阶段检索指标

easy（55 道可答题，145 chunks，k=8）：

| 阶段 | Span@1 | Span@3 | Span@5 | Span@8 | Source@3 | 检索 P50 |
|---|---|---|---|---|---|---|
| 起点：200 字硬切 + 纯向量 L2 | 65.45% | 94.55% | 98.18% | 98.18% | 100.00% | 126.05 ms |
| ① QA 感知切分 | 67.27% | 87.27% | 94.55% | 98.18% | 98.18% | 135.16 ms |
| ② + BM25 / RRF 混合召回 | 69.09% | 81.82% | 96.36% | 100.00% | 94.55% | 146.49 ms |
| ③ + 交叉编码器重排 | **96.36%** | **98.18%** | **100.00%** | **100.00%** | **100.00%** | 265.09 ms |
| ④ + 无证据兜底（复跑） | 96.36% | 98.18% | 100.00% | 100.00% | 100.00% | 261.47 ms |

hard（20 道可答题，同一向量库）：

| 阶段 | Span@1 | Span@3 | Span@5 | Span@8 | Source@3 | 检索 P50 |
|---|---|---|---|---|---|---|
| 起点 | 55.00% | 80.00% | 90.00% | 90.00% | 100.00% | 156.92 ms |
| ① QA 感知切分 | 50.00% | 85.00% | 85.00% | 85.00% | 85.00% | 148.01 ms |
| ② + 混合召回 | 65.00% | 85.00% | 85.00% | 85.00% | 90.00% | 138.82 ms |
| ③ + 重排 | **80.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 262.55 ms |
| ④ + 兜底（复跑） | 80.00% | 100.00% | 100.00% | 100.00% | 100.00% | 232.18 ms |

**最容易讲错的一条结论：切分和混合召回都不是主要杠杆，重排才是；但只看首尾两个数字，会误判成「切分是负收益」。**

① 单独上 QA 切分时，easy Span@3 从 94.55% 掉到 87.27%、hard Span@1 从 55% 掉到 50%。逐题看排名分布才发现掉分的方式很特别（easy 55 题 / hard 20 题，数字是题数）：

| 阶段 | easy @1 | easy Top-3 内 | easy Top-8 未命中 | hard @1 | hard Top-3 内 | hard Top-8 未命中 |
|---|---|---|---|---|---|---|
| 起点（66 chunks） | 36 | 52 | 1 | 11 | 16 | 2 |
| ① 只加切分（145 chunks） | 37 | **48↓** | 1 | 10 | 17 | **3↑** |
| ② 再加混合召回 | 38 | **45↓** | **0** | 13 | 17 | 3 |
| ③ 再加重排 | **53** | **54** | 0 | **16** | **20** | **0** |

**easy Span@1 是一路上升的（36 → 37 → 38 → 53），掉的是 @2/@3 那一批**：chunk 从 66 变 145、粒度变细之后，纯向量对「半相关」的相邻块打分变高，正确块被推到第 4–7 位（② 阶段 easy 有 7 题正好落在第 4 位）。也就是说，**切分改善的是块的质量，不是块的排序**；排序要靠混合召回把关键词强命中的块拉进候选池（② 把 easy 的 Top-8 未命中压到 0、Source Recall@8 到 100%，但片段级 Span@3 反而再掉到 81.82%，hard 仍有 3 题未命中），再靠重排重新打分（③ 一次性把 easy Span@1 拉到 96.36%、hard Span@3 拉到 100%、两边 Top-8 未命中都归零）。

反过来说，**如果只做了 ① 就收工，结论会是「切分没用」，那是错的**：`qa-042` 这道旧基线里 Top-8 完全召不回的题，恰好是 ① 单独就修好的（未命中 → @1）；而 `hard-qa-016` 在 ① 之后依然未命中，要到 ③ 才修好。切分是重排能生效的前提（候选池里得有干净的问答单元），三段是配套的。详见 [pitfalls.md](pitfalls.md) P16。

代价：检索 P50 从 126 ms 涨到 265 ms，多出来的约 120–140 ms 全部是重排的一次同步 HTTP 调用；相对回答级 1.7–1.9 秒的生成耗时占比不到 8%，但**重排失败会静默退回 RRF 序**，质量掉分不会报错，这是当前最需要盯的风险（见 `docs/design.md` 第 7 节）。

#### 逐题变化：改好的与变差的

| 数据集 | 排名变好 | 排名变差 | 不变 | Top-8 完全未命中 |
|---|---|---|---|---|
| easy 55 题 | 18 | 1 | 36 | 1 → **0** |
| hard 20 题 | 8 | 2 | 10 | 2 → **0** |

变好的代表：

- `qa-042`（拖布清洗频率）：Top-8 未命中 → **Span@1**，仅靠 ① 就修好（见 5.3）；
- `hard-qa-016`（儿童锁 / 边刷防护）：Top-8 未命中 → ② 后 Source@4 仍无片段 → ③ 后 **Span@1**。这题是 5.7 里 hard 唯一的失败题，也是 pass^3 只过 1/3 的波动源，现在回答级 correctness=2；
- `hard-qa-007`（老人操作方式）：Top-8 未命中 → **Span@1**。旧基线里它是「检索失败但模型靠常识蒙对」的典型，现在不再需要蒙。

变差的三题（诚实记录，都是排名小幅后退、仍在 Top-8 内）：

| ID | 变化 | 根因判断 |
|---|---|---|
| `qa-023`（拖布怎么清洗和晾干） | Span@1 → **@4** | gold 来源是 `维护保养.txt` + `扫拖一体机器人100问.txt`、关键词是「洗衣袋 / 高温烘干」；重排把 `扫拖一体机器人100问.txt` 里几个「拖布清洗」块排到了前三位，真正带 gold 关键词的「每周维护 / 晾干」块被压到第 4 |
| `hard-qa-009`（尘袋倒空后继续用） | Span@1 → @2 | 重排给同一文件内相邻的集尘主题块打了几乎同分，属于同分抖动 |
| `hard-qa-020`（进水后能否开机试） | Span@1 → @2 | 同上，故障排除文件里有多个进水相关块 |

三题都仍在 Top-4 内、回答级都通过（`qa-023` correctness=1、另两题 correctness=2），所以聚合指标是净上升；但它们说明**重排不是单调改进器**，换语料后要重跑逐题对比而不是只看汇总分。

#### 回答级前后对比

| 数据集 | 指标 | M2 前 | M2 后 |
|---|---|---|---|
| easy answerable（55） | 通过率 | 100.00% | 100.00% |
| | 完全正确（correctness=2） | 81.82% | **90.91%** |
| | 完全有据（faithfulness=2） | 100.00% | 98.18% |
| | 幻觉（faithfulness=0） | 0.00% | 0.00% |
| easy negative（15） | 拒答正确率 | 100.00%（规则） | 100.00%（规则） |
| easy report（10） | 字段完整 / 完全有据 | 100.00% / 90.00% | 100.00% / 100.00% |
| hard answerable（20） | 通过率 | 95.00% | **100.00%** |
| | 完全正确 | 75.00% | **95.00%** |
| | 完全有据 | 90.00% | 95.00% |
| hard negative（5） | 拒答正确率 | 100.00%（单次） | 100.00%（单次） |

两处必须拆开归因，不能直接说「M2 让回答变好了」：

- **easy 完全有据从 100% 掉到 98.18%，是 M2 的真实代价**：唯一一题 `qa-017`（回充失败怎么办），重排后上下文更窄更聚焦，模型在正确答案之外补了一句资料里没有的「电池老化也可能影响充电效果」。检索变准会让模型更敢往下推，这是提示词约束的问题，不是检索退步。
- **negative 100% 依然主要是规则功劳**：15 道 easy 负样本仍全部被 `is_out_of_scope_query` 拦下、0 token；hard 5 道是「能力边界」问题，现在多了一层确定性兜底（下节），但**本轮 repeat=1，没有重跑 pass^3**，所以 5.8 里 hard negative 严格 pass^3 = 60% 这个数还没有被更新，不能说稳定性已解决。

生成延迟基本没变：easy answerable P50 1637 → 1868 ms、hard 1706 → 1754 ms，都在 API 抖动范围内；easy negative P50 135 → **259 ms**，涨的是检索（重排那一次调用），因为评测口径的 `gen_latency_ms` 含 `retrieve()`，即使最终不调生成模型也要先检索。

#### 无证据兜底：阈值是标定出来的

`min_score = 0.16` 不是拍的，由 `eval/calibrate_min_score.py` 跑 105 道题标定（`eval/results/min_score_calibration.md`）：

| 分组 | 样本 | 最低分 | P50 | 最高分 |
|---|---|---|---|---|
| answerable | 75 | **0.1673** | 0.5273 | 0.9510 |
| negative（未被路由拦下） | 5 | 0.1261 | 0.1473 | **0.1578** |
| negative（已被 `scope_router` 拦下） | 15 | 0.0061 | - | 0.4521 |
| report | 10 | 0.0801 | 0.1398 | 0.1798 |

- 两组可分，间隔只有 **0.0095（约 4% 余量）**，脚本建议取中点 **0.1626**，实际配 **0.16**（偏向少误伤可答题）。
- 已被路由拦下的 15 条 negative **不参与阈值计算**：它们根本走不到重排，算进来会把阈值抬到 0.45 以上、误伤一大片可答题。第一版标定就是因为没区分这点，报出了 `separable: false`。
- 实跑复核（2026-11-13）：5 道 in-scope 负样本全部 `docs=0`、返回 `NO_EVIDENCE_ANSWER`、**不调用生成模型**（最高分 0.1578 / 0.1487 / 0.1473 / 0.1355 / 0.1261）；分数最低的两道可答题 `hard-qa-016`（0.1673）与 `qa-008`（0.1811–0.1812）仍各保留 1 条片段并正常作答。
- 同一题连跑 3 次，重排最高分在 **±0.0001** 级抖动（0.1812 / 0.1811 / 0.1812）；抖动量级远小于 4% 余量，不会翻转判定，但说明标定值本身带测量噪声。
- 边界很薄：`report` 组最低分 0.0801 低于阈值，但报告走 `report_pipeline` 不经过 RAG 检索，不受影响；换语料、换重排模型或改切分参数后**必须重跑标定**。

#### k 敏感性：k=3 保留

hard 20 题分别用 k=3 和 k=5 跑回答级：

| k | 完全正确 | 完全有据 | 关键词命中率 |
|---|---|---|---|
| 3（生产值） | 95.00% | 95.00% | 77.50% |
| 5 | 90.00% | 100.00% | 75.00% |

k=5 多给了两条上下文，`hard-qa-011` 的 faithfulness 从 1 修到 2，但 `hard-qa-017` 反而从 2 掉到 1（信息变多后漏掉了更换周期）。**一升一降、样本 20 题，属于噪声**，没有证据支持改 k；保持 k=3 还能少喂 40% 上下文、省 token。

#### 未测 / 待补

- **easy 全量 pass^3 未跑**：成本约为单次 3 倍，本轮只跑了 repeat=1；hard pass^3 也还是 M2 前的旧数据（5.8），M2 后的稳定性尚未量化。
- **重排降级路径只做了单测**，没有实测真实限流/超时下的端到端表现，也没有降级告警。
- **BM25 索引是进程内的**：多 worker 部署时各自建索引，语料更新后没有跨进程失效机制，本轮只在单进程下验证。
- **`min_score` 只有 4% 余量**，扰动测试只做了 1 题 × 3 次（抖动 ±0.0001），没有覆盖全部边界样本，也没有测重排模型版本更新后的分数漂移。
- **切分只在当前 5 个语料文件上验证**，`test_real_data_files_lose_no_line` 保证不丢行，但没验证新增文档类型（表格、Markdown 标题层级）时的切分行为。

### 5.14 身份登录与会话归属（2026-11-13）

**测什么**：登录态是否真的挡在业务接口前面、会话是否按账号隔离、报告链路是否拦得住「查别人的使用记录」，以及前端登录/退出/403 的实际表现。

**怎么测**：三层。① `tests/test_auth.py` 17 个 + `tests/test_api.py::AuthApiTest` 10 个纯单测（临时库、`iterations=1000`、fake agent，不联网）；② `uvicorn server.app:app --port 8123` 起真服务、真模型，用脚本按 HTTP 打完整流程；③ 浏览器打开页面，用结构性快照 + JS 驱动真实 DOM 事件走一遍登录/切换/发送/删除/退出/越权。

**结果（自动测试）**：

| 项 | 结果 |
|---|---|
| 单元测试 | **160 个全部通过，约 4.5 秒**（比 126 个时慢 3.5 秒，主要是 `TestClient` 生命周期会拉起真 Agent 与 145 chunk 向量库，以及 PBKDF2 20 万次迭代） |
| 种子幂等 | 已存在的账号不再重新哈希：冷启动 0.184 秒、二次启动 **0.005 秒** |
| 公开接口 | `/health`、`/auth/bootstrap`、`/`、`/static` 无令牌 200；`bootstrap` 返回的每个账号只有 `username / display_name / user_id` 三个键，整个响应里搜不到 `pbkdf2_sha256` |
| 无令牌 | `/sessions`、`/usage`、`/traces`、`/auth/me`、`/history/{id}`、`/chat/stream` 全部 **401**；`Authorization: nonsense` 与裸令牌同样 401 |
| 登录 | 密码错误 401；`demo` / `alice` 各拿到不同令牌，`expires_in_hours=12`；`/auth/me` 回 `demo / ID 1001` |

**结果（HTTP 实跑，真模型）**：

| 场景 | 观测 |
|---|---|
| demo 问答一次 | SSE `meta → tool×2 → delta×31 → done`；`thread_id` 是服务端新发的 32 位 hex（不再共用 `default`）；`usage` 3,516 tokens、成本 ¥0.009706、trace 落库 |
| alice 碰 demo 的会话 | `/history` **403**、`/chat/stream` **403**、`DELETE /sessions/{id}` **403**；`/sessions` 里也看不到这条；agent 侧收到的 `owner_user_id` 始终是 `1001`，从未出现 `1002` |
| 旧会话归属 | 启动日志 `[auth] 已把 2 个旧会话归到 legacy_owner 名下`；demo 侧栏 2 条、alice 侧栏 **0 条** |
| 越权报告 | alice 问「生成用户 1001 的报告」→ 固定拒答（同时点名 1001 与 1002）、`tool_calls=0`、`usage.source="none"`、成本为空、端到端 **47 ms**（零取数零模型） |
| 自己的报告 | demo 问「帮我看下我 2026-10 的使用报告」→ 取数 1 次 + 写作 1 次、823 tokens、380 字正文、trace `route=report / status=ok`。**修 P17 之前这里返回的是越权拒答** |
| 零成本探针 | 问没有记录的 2020-01：demo 的回答带 `1001`、alice 的带 `1002`，两者都 `usage.source="none"`——用空记录分支反查参数解析结果，不花 token |
| 登出 | 令牌立即失效（`/auth/me` 401），另一个账号的令牌不受影响 |

**结果（浏览器实跑）**：登录面板渲染出标题、说明、账号/密码框、两个演示账号 chip 与提示语；错误密码时面板留着并显示「账号或密码不正确。」且不写令牌；chip 只填用户名并把焦点移到密码框（不自动填密码）；demo 登录后面板消失、顶栏显示「演示用户 / ID 1001」；侧栏 2 条旧会话可点击切换并加载出 2 条历史消息，URL 与 `localStorage` 同步；发一条越界问题走 SSE 流式拒答、会话数变 3、用量面板刷新到 9 次 / 12,024 tokens；× 删除该测试会话后回到 2 条；退出后令牌清空、面板重现、侧栏清空；alice 登录显示「Alice / ID 1002」、0 条会话；alice 强行打开 demo 的 thread（改 `localStorage` + URL 重载）时出现横幅「原会话属于其他账号，已为你开启新会话。」，消息区只剩欢迎语（**没有泄露任何历史内容**）并自动换新 thread；再切回 demo，2 条会话正常、横幅不残留。

**用途**：这条链路是「多用户隔离」的最小可验证证据——面试追问「你怎么保证 A 看不到 B 的对话」时，可以当场演示 401/403、侧栏过滤和零 token 的越权拒答，而不是只讲设计。

**清理**：验证过程造的 4 个会话及其 trace 全部用 `DELETE /sessions/{id}` 删掉；另有 1 条被截断成 12 位的 thread_id（脚本 bug 造成的孤儿线程，接口删不掉）手工从 `checkpoints` 与 `thread_owner` 清掉。库里只剩用户原有的 2 条会话，归属 `demo`。

**未测 / 待补**：

- **视觉层面没有像素级检查**：in-app 浏览器没有可见视口（`NATIVE_BROWSER_VIEWPORT_UNAVAILABLE`），截图与真实点击都不可用，只能用结构性快照 + JS 驱动 DOM 事件；登录面板的配色、间距、移动端断点属于**未实测**。
- **多用户并发未测**：两个账号同时写同一个 thread、令牌表并发写入、SQLite WAL 下的锁竞争都只在单请求串行下验证过。
- **令牌过期的前端行为只测了后端**：单测用 `token_ttl_hours=-1` 验证过期令牌查不到，浏览器侧只验证了 401 会退回登录面板，没验证「用到一半过期」的流式中断表现。
- **限流还没有**（下一步 #29），所以没有 429 相关测试。
- **观测接口不按账号隔离**：alice 登录后 `/usage` 仍显示全量 8 次 / 12,024 tokens，其中大部分是 demo 产生的；这是已知边界（`docs/design.md` 第 7 节第 20 条），本轮没有改。
- **Streamlit 入口未测**：它直接调 `ReactAgent`、不走 HTTP，因此不受鉴权影响（等价于 `enabled: false` 的行为），本轮没有实跑。
- **`enabled: false` 只在单测里验证**（无令牌 200、不过滤、不写归属行、`owner_user_id` 为 `None`），没有起真服务复跑。

### 5.15 会话历史压缩（2026-11-13）

**测什么**：长会话折叠成「要点清单 + 最近原文」之后，三件事必须同时成立——① 早期事实还能被召回（压缩不能把第一轮问的 E05 丢掉）；② 摘要不许编造对话里没有的型号/用户 ID/日期；③ checkpoint 与界面历史保持完整、摘要正文不许出现在用户看到的回答里。另外要量化「到底省了多少 token、摘要自己花了多少」。

**怎么测**：三层，全部可复跑。

1. **离线单测 51 个**（`tests/test_history_compressor.py`，摘要器用注入的假对象，不调 DashScope）：折叠边界规则、对话稿构造、缓存复用/从原文重写/失败降级、编造校验、system 消息拼接、中间件接线、统计聚合。
2. **端到端实跑**（`log/verify_history.py`，真模型真 HTTP）：新开一个 32 位 hex thread，连续 8 轮问答把消息数推过触发线，再打 3 发召回探针——问第一轮的错误码、问第四轮的边刷周期、以及一发**对照题**（问「我提到过我家住在哪个城市吗」，正确答案是「没有」，用来抓编造）。每轮回答都过 `FORBIDDEN` 清单（`U123456`/`XG900`/`E203`/`2024-04-05`/`1.2.3`，都是曾经真实被编造出来的值）和 `contamination()`（回答开头连续 bullet ≥8 行、或含摘要标记就算污染）。最后拉 `/history/{thread}` 检查条数、摘要泄漏、丢问题。报告写 UTF-8 文件 `log/verify_history_report.md`，**控制台只输出 ASCII**——本机控制台会把中文打成乱码，前面有两次因为读乱码而误判。
3. **确定性 payload A/B**（`log/probe_payload.py <thread_id>`）：跨轮对比 token 是不可靠的，因为每轮是否触发检索会让 payload 差几千 token（第一版就是拿「关掉压缩跑 8 轮」和「开着跑 8 轮」比总量，被工具调用差异污染了）。改成**同一条 thread、同一个当前问题**，分别按完整视图和压缩视图各调一次 qwen-max，读供应商返回的真实 `input_tokens`，再按 `config/observability.yml` 的单价算钱。
4. **摘要指令调参**（`log/probe_cap.py`）：拿真实 dump 出来的对话稿（`log/dump_summary.py <thread_id>` → `log/dump_transcript.txt`）对比两版指令 × 两个字数上限 × 各跑 2 次，同时看耗时、字数、bullet 条数、输出 tokens，以及 E05/边刷/清水箱三个早期事实是否留住。

**结果**：

| 项 | 观测 |
|---|---|
| 单元测试 | 211 个全通过（历史压缩占 51 个），约 2.4 秒 |
| 触发行为 | 8 轮里第 1–4 轮 `below_trigger`（3/7/11/15 条消息），第 5–8 轮全部触发，`before` 19/21/25/29 → `after` 恒为 8 |
| 召回 | `recall_turn1_error_code` 命中 E05、`recall_turn4_side_brush` 命中「3 至 6 个月」、对照题正确回答「您并没有提到」；三发都 `fabricated=[]`、`unsourced=None` |
| 回答污染 | 11 轮全部 `clean`，`answer_contamination=none`（修 P18 之前第 5 轮起每轮都被摘要正文污染） |
| 界面历史 | 22 条（11 问 11 答，一条不少）、摘要泄漏 **0** 条、丢失提问 **0** 条、持久化历史里的编造标识符 **0** 个 |
| token（36 条消息） | 完整视图 **3072** input → 压缩视图 **1700** input，省 1372（**-44.7%**）；摘要自身 1485 in / 237 out（qwen-flash） |
| 成本（每轮） | 主模型省 ¥0.003293，摘要花 ¥0.000578，**净省 ¥0.002715** |
| 摘要耗时 | 指令改成「一轮问答只写一条」后 0.8–3.4s（改之前 5.7–6.9s）；摘要字数 117–349（改之前 593–690，且无视 600 上限实际写到 1074–1197） |
| 编造校验精度 | trace 库里 38 次 compress span：17 次真压缩、12 次 `below_trigger`、8 次 `disabled`、**只有 1 次被校验拦下**（接受率 94%），说明校验没有过度触发 |
| 浏览器 | 打开那条 11 轮会话，`.messages` 渲染 22 行、`summaryMarkerPresent=false`、`worstLeadingBulletRun=0`、E05 出现 3 次，首尾两条回答都是正常话术 |

**为什么这么测**：压缩的收益（省 token）很好量化，风险（丢事实、编造、污染回答）却很容易漏——它不报错，只是**回答悄悄变差**。所以召回探针和 `FORBIDDEN` 清单是这套验证的主角，token 数字只是附带产出。对照题（问一个对话里从没出现过的信息）专门用来抓编造：如果摘要凭空补了字段，这一题就会答出内容而不是「您没提过」。

**用途**：面试追问「长会话怎么办」时，可以直接给出「36 条消息时 payload -44.7%、每轮净省 ¥0.0027、11 轮后界面历史零丢失零泄漏」这组数字，以及「为什么不用官方 `SummarizationMiddleware`」（它会用 `RemoveMessage` 真的改写 state，早期对话就从界面消失了）。

**未测 / 待补**：

- **超长会话未测**：只跑到 36 条 checkpoint 消息。`max_transcript_chars=12000` 的截断分支有单测覆盖，但没有真实百轮会话的实测数据；摘要每轮从原文重写，输入随会话长度线性增长（`docs/design.md` 第 7 节第 5 条已记为已知短板）。
- **摘要质量的语义评测没有做**：只做了「关键事实字符串是否命中」和「有没有编造标识符」，没有用 LLM-as-judge 去评摘要的完整性/忠实度，也没纳入 `run_answer_eval.py` 的指标体系。
- **缓存淘汰只在单测里验证**：`cache_threads=64` 的滚动淘汰是纯内存 dict，多 worker / 重启后缓存全丢（会退化成每轮重算，功能不受影响），没有跨进程共享。
- **视觉层面没有像素级检查**：in-app 浏览器无可见视口（`NATIVE_BROWSER_VIEWPORT_UNAVAILABLE`），截图与真实点击不可用，只能用结构性快照 + JS 驱动 DOM。
- **摘要模型单点**：只实测了 qwen-flash。qwen-turbo 在本账号下拿不到（HTTPError/429），qwen-max/qwen-plus 太慢（同一段稿 5.0s / 3.3–6.6s），换模型需要复跑 `log/probe_cap.py`。
- **`enabled: false` 的回归没有自动化**：关闭压缩走的是原路径，单测覆盖了 `disabled` 分支，但没有跑一遍完整评测集对比「开/关压缩」的回答质量差异（需要 easy 全量复跑，成本原因暂缓，见第 8 节）。

### 5.16 回答引用溯源（2026-11-13）

**测什么**：一条回答下面的「引用来源」面板，必须做到四件事——① **该有引用时一定有**（这一轮真的检索过），② **不该有引用时绝不能有**（越界拒答、报告路由、无证据兜底、模型自己答的），③ 引用条目里的文件/小节/相关度和检索那一刻的结果一致，且**不泄漏本地绝对路径**，④ 刷新页面或切换会话之后引用还在（不能只是流式过程中闪一下）。另外要确认引用走的是 `ToolMessage.artifact` 通道，**一个 token 都没多花**。

**怎么测**：四层，前三层可复跑。

1. **离线单测 36 个**（`tests/test_citations.py` 22、`CitationCaptureTest` 7、`RagServiceCitationTest` 3、`ChatRunnerCitationTest` 3、重排副本回归 1）：纯函数 `build_sources` / `merge_sources` / `summarize_retrieval_stats` 的边界（路径归一化、预览折叠空白与截断、去重保留更高分、多次调用聚合、空列表）、中间件从 artifact 收集引用（含护栏拦截/工具报错时不收集）、`answer_with_citations` 三条分支（正常/越界/无证据）、`chat_runner` 落 `rag:retrieve` span、以及 `load_history` 从 checkpoint 还原引用。全部不调 DashScope，不碰网络。
2. **序列化往返（离线）**：`log/check_artifact_serde.py` 用 LangGraph 真实的 `JsonPlusSerializer` 把一个带 artifact 的 `ToolMessage` dump 再 load，验证 `ARTIFACT_EQUAL=True`、`SCORE_TYPE=float`、`CONTENT_ONLY_HAS_ARTIFACT=False`（artifact 的内容没有混进 `content`）。**这一步必须在写功能之前做**——整个方案建立在「artifact 会被存进 checkpoint」这个假设上，假设不成立就得换设计。
3. **端到端实跑**（`log/verify_citations.py`，真模型真 Chroma 真 HTTP）：5 发覆盖全部路由（错误码 QA、拖布 QA、越界、超纲、报告），逐轮校验 `bool(sources) == bool(有 rag:retrieve span)` 这条**核心不变式**，再做结构校验（必填键齐全、`file` 里不含 `\` 或 `/`、`score` 落在 0~1、编号从 1 连续）、拒答/报告路由不许自称有依据、span 的 `detail["sources"]` 与面板条数一致；最后拉 `/history/{thread}` 验证往返——带引用的条数应当等于检索过的轮数，且引用只挂在 assistant 消息上。报告写 UTF-8 文件 `log/verify_citations_report.md`，**控制台只输出 ASCII**，跑完自动删掉自己的 thread。
4. **浏览器结构验证**（手工驱动 DOM）：真发一轮问答，读 `.notice-target` 下 `details.sources` 的结构、summary 文案、每条的文件/小节/分数；刷新页面看面板是否复原；再打开一条 **#28 之前的旧会话**看降级表现；`list_console_messages` 查有无报错。

**结果**：

| 项 | 观测 |
|---|---|
| 单元测试 | 247 个全通过（引用溯源占 36 个），约 3 秒 |
| 端到端 | `EXIT=0`、`PROBLEMS 0`；5 轮路由 `['qa','qa','out_of_scope','qa','report']`，引用条数 `[3, 3, 0, 0, 0]` |
| 错误码轮 | 首条 `故障排除.txt / 错误代码 / E05 悬崖或轮组异常`，相关度 **0.891**；次两条 E04 0.1772、E03 0.1743——阈值 0.16 之上但分数差一个量级，说明重排真的把 E05 排出来了 |
| 拖布轮 | 首条 `扫拖一体机器人100问.txt / 拖地与水箱 / Q4` 0.383，第三条 Q12 0.2716；回答里的「装洗衣袋、避免高温烘干」正是 Q12 的原文，**溯源对得上** |
| 检索 span | `duration_ms=252.93`（=召回 149.09 + BM25 0.09 + 重排 103.75）、`calls=1, mode=hybrid, index_size=145, vector_candidates=20, bm25_candidates=14, fused_candidates=24, overlap=10, rerank_kept=6, rerank=ok, rerank_top_score=0.891, sources=3`（`rerank_kept=6` 但只展示 3 条，因为 k=3） |
| 历史往返 | `HISTORY_WITH_SOURCES 2` / 检索过的轮数 `2`，一致；引用只出现在 assistant 消息上，user 消息 0 条 |
| token | 引用全程走 artifact，`content` 里没有引用文本；`done` 事件的 `usage` 与上线前同题型一致（未新增开销） |
| 浏览器 | summary 为 `▶引用来源3`，展开后 3 条，每条渲染出 `维护保养.txt / 耗材更换周期（参考）· 1-7 / 相关度 0.334` 三段；刷新后面板数量 1、内容一致 |
| 旧会话降级 | 打开 #28 之前的会话（22 行、11 条 assistant 回答），渲染 **0 个**引用面板——不报错、不出空壳 |
| 清理 | 实跑脚本自删 thread，返回 `{'checkpoints': 25, 'writes': 29, 'traces': 5, 'spans': 13}`；浏览器那条另删，返回 `{'checkpoints': 9, 'writes': 11, 'traces': 1, 'spans': 4}` |

**为什么这么测**：引用面板最坏的失效方式不是「不显示」，而是**显示了假的**——给一条越界拒答挂上「参考资料」，等于在暗示它有依据，比不挂更糟。所以验证的主角是那条不变式（`bool(sources) == bool(retrieval_span)`），而不是「能不能看到面板」。第一版脚本里我写死了一条 `no_evidence:sources_empty` 的断言，实跑才发现「超纲问题」这一轮模型压根没调 `rag_summarize`（`tool_calls=0`、无检索 span），0 引用本来就是对的结果，是断言的前提错了——**零引用有四种合法成因，逐轮对着 span 判才站得住**。序列化往返之所以单列一层，是因为它是整个方案的地基；地基塌了，后面所有代码都要重写。

**用途**：面试追问「RAG 怎么保证不胡说」时，可以给出「每轮都落一条 `rag:retrieve` span，带 mode / 候选数 / 重叠数 / 重排保留数 / top score，界面上引用条目和 span 数量对得上；引用走 artifact 通道零 token 开销，且随 checkpoint 持久化，刷新和切会话都不丢」这组可验证的事实，而不是只说「加了引用」。

**未测 / 待补**：

- **视觉层面没有像素级检查**：宿主视口是 0×0（`window.innerWidth === 0`），`click` 与 `take_screenshot` 直接报 `NATIVE_BROWSER_VIEWPORT_UNAVAILABLE`，CSS 过渡也永远不走帧——展开箭头的 `transform` 读出来是单位矩阵，一度被我误判成「样式没生效」，最后靠内联 `transition:none` 强制取值才确认 `rotate(90deg)` 是对的（见 `docs/pitfalls.md` P22）。**面板的观感、间距、窄屏/移动端布局都属于未验证项。**
- **句级引用映射没有做**：现在只能回答「这条回答用了这 3 段」，不能回答「这句话出自第几段」。要做就得改生成提示词让模型输出角标，那会动到 M2 的回答级基线，属于另一次评测投入。
- **`MAX_SOURCES=12` 的截断只在单测里覆盖**：实跑最多一轮 3 条（k=3），截断分支没有被真实流量打到。
- **artifact 给 checkpoint 增了体积**：一轮一条引用约 0.5~1 KB，长会话下 checkpoint 会持续变大。只做了定性观测，没有量化百轮会话的膨胀曲线，也没有做 artifact 的淘汰策略。
- **重排降级时的分数缺失**：`rerank=fallback` 或 `disabled` 时后端给 `score=null`，界面按「不显示分数」处理（单测覆盖），但没有真的把重排关掉跑一遍端到端确认。

## 6. 这些测试有什么用

1. **改动前先知道现状。** M2 之前能说“easy Top-1 65.45%、Top-3 94.55%，hard Top-1 55.00%、Top-8 90.00%”，而不是“感觉挺准”；正是这几个数字指出了「文件能找到、片段找不到」。
2. **改动后能证明是否变好，也能证明哪一步没用。** M2 的切分、混合召回、重排各跑一次同一套 easy + hard 题，才敢说「切分单独上会让 Span@3 掉 7 个百分点、重排才是主要杠杆」（5.13）；只看最终分是得不出这个结论的。
3. **失败能定位到层。** hard 的 Source@3 是 100% 但 Span@8 只有 90%，问题在片段级；`qa-042` 因此能用一次切分改动修好；报告字段完整但忠实度不足，问题在生成而不是取数；pass^k 波动题则说明单次通过不可靠。
4. **面试有证据。** 题量、标注口径、指标定义、复现命令、失败案例都能当场打开；hard split 与 pass^k 能回答“是不是只会背原句、是不是碰巧通过”；分阶段结果文件能回答“这个提升到底是哪一步带来的”。
5. **防止回归。** 改配置或换模型后，先跑两套 `--validate-only` 和 247 个单测，再跑检索与回答级基线；换语料或换重排模型还要重跑 `calibrate_min_score.py`，避免“修一个坏三个”。

## 7. 设计取舍与边界

- **不用 pytest**：当前用标准库 `unittest`，零额外依赖；后续测试变多再引入。
- **单测不调模型**：保证快、稳定、可在无网络时跑；真实检索和回答放到单独的端到端命令。
- **gold 不绑 chunk id**：切分策略一变 id 就失效；改用来源 + 章节/关键词。
- **easy 与 hard 分开报**：easy 题目多来自知识库原问句，分数偏高；hard 是改写与跨节组合，25 题样本仍小。两者都不能直接当作真实用户分布。
- **同源判分**：生成与判分都是 `qwen-max`，存在自偏好；已用 gold answer、明确 rubric、`temperature=0` 降低，但未做人工双标。
- **粗粒度 rubric**：correctness / faithfulness 是 0/1/2，同题跨运行可能波动 1 分；hard split 用 pass^3 暴露波动，easy 仍只跑单次。
- **规则路由有覆盖边界**：`is_report_query` 覆盖「报告 / 使用记录 / 使用情况」和「ID + 月份 + 数据词」，换一种完全不含这些词的说法会漏判并回退 ReAct；当前用 80 题逐题断言做回归。
- **报告字段是词面下界**：标签 + 数值 + 单位归一能抓漏字段，但不判断语义等价；方向与编造交给 LLM judge。
- **两种检索口径**：检索基线用 k=8 看召回上限，回答级用 k=3 复现生产链路，两个数字不能直接相减。
- **报告题都显式给 ID/月份**：`get_user_id` / `get_current_month` 的随机 mock 分支未被覆盖。
- **BM25 用「单字 + 双字」近似中文分词**：不引 `jieba`/`rank_bm25`，零新依赖、可完全离线单测；代价是词表偏大、`E05` 这类字母数字型号靠整词命中而不是靠分词，长尾同义词也召不回——这一层缺口由重排补。
- **RRF 而不是分数加权融合**：只用名次不用分值，向量分与 BM25 分尺度不同也不需要归一化；代价是丢掉分数的绝对强弱信息，所以融合后必须再过一次重排。
- **重排失败静默退回 RRF 序**：可用性优先，不让一次外部调用失败变成 500；代价是质量掉分不报错，只能靠 stats 里的 `rerank=fallback` 事后发现，当前没有告警。
- **`min_score` 是全局单一阈值**：不按问题类型分档。报告题的重排分天然偏低（最低 0.0801），但报告走确定性子流程不经过 RAG 检索，所以不冲突；如果以后有别的链路复用这条检索管线，阈值要重新评估。
- **阈值标定排除已被路由拦下的负样本**：那 15 条根本走不到重排，算进来会把阈值抬高、误伤可答题；代价是标定脚本必须知道 `scope_router` 的行为，两个模块耦合了一层。
- **身份是演示级、单测用低迭代次数**：`tests/test_auth.py` 与 `AuthApiTest` 都传 `iterations=1000`（生产配置 20 万），否则 17 + 10 个用例光哈希就要几秒；哈希算法与校验逻辑不变，只是慢函数被调快了。账号写死在 `config/auth.yml`，没有注册、刷新令牌与角色权限，测的是「隔离与越权拦截」而不是「账号体系」。
- **鉴权相关单测都用临时库**：`AuthStore` 与 `TraceStore` 都指向 `tempfile` 目录，绝不碰 `checkpoints/` 里的真实身份库与会话；HTTP 实跑才用真库，跑完把造的会话删干净（见 5.14 的清理段）。

## 8. 下一步

1. ~~**M2 检索优化**：QA 感知切分、BM25 + 向量 RRF、交叉编码器重排。~~ 已完成，四段前后对比见 5.13。
2. **M2 后重跑 pass^k**：5.8 的 hard pass^3（严格 88%、negative 60%）是 M2 之前的数据。检索修好后 `hard-qa-016` 这类波动源应该消失，需要用 `--repeat 3` 重新量化稳定性，这是当前评测侧最该补的一格。
3. **easy 全量 pass^k**：easy 80 题仍只跑单次，成本约为单次 3 倍，适合按周或发版前跑。
4. **扩大 hard split**：25 题能暴露问题，但样本仍小；下一步加入更多跨章节组合、近似干扰项和错别字问法，并把越界路由准确率纳入同一套基线。
5. **越界边界补强**：`neg-004` / `neg-009` 已由确定性拒答覆盖（见 5.10），`hard-neg-001/002/003/004/005` 已由无证据兜底覆盖（见 5.13）；剩下的是「知识库里有相邻内容、但答案其实不存在」这类灰区，仍靠模型判断。
6. **重排可观测性**：~~重排分低于阈值时把 `top_score` 写进 trace span~~（已完成，`rag:retrieve` span 现在带 `rerank_top_score` 与 `rerank` 状态，见 5.16）；剩下的是给 `rerank=fallback` 加计数与告警，便于事后判断是「真没证据」还是「重排挂了」。
7. **服务化收尾**：~~给 `/chat/stream` 加登录态与 `thread_id` 归属校验~~（已完成，见 5.14）；~~会话历史压缩~~（已完成，见 5.15）；~~回答引用溯源~~（已完成，见 5.16）；剩下按用户限流、CI、trace 换 Postgres 或接 OpenTelemetry、观测数据按账号隔离、补浏览器端自动化测试（当前浏览器验证靠手工驱动 DOM，截图能力受限于宿主视口）。
8. 把两套 `--validate-only` + 247 个单测 + `calibrate_min_score.py` 接进 CI，检索与回答级基线作为人工回归。
