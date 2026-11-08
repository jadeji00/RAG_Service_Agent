# 测试与评测说明

> 测试要回答三个问题：**评测集和标签有没有错？指标算得对不对？系统现在到底是什么水平？**
>
> 这份文档说明当前测了什么、怎么测、结果如何、为什么这样设计，以及哪些还没测。

## 1. 为什么要测试

1. **评测集是尺子，尺子错了会带偏优化。** 如果 gold 标错，调切分、换 embedding、加重排都可能朝着错误方向走；所以先把数据校验做成自动化。
2. **RAG 失败要能定位到层。** 是文件没召回、片段没命中，还是答案生成了但没引用？不分层就只会说“感觉不太准”。
3. **改检索必须有回归基线。** M2 要做 QA 切分、混合召回和重排，没有同一套题和同一套指标，就无法证明改动是进步还是退步。
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
| 观测底座单测 | `tests/test_observability.py` | 用量归一/提取/去重、成本估算、trace/span 落库与查询 | 否 | 毫秒级 |
| 事件流单测 | `tests/test_chat_runner.py` | SSE 事件序列、trace 落库、错误/取消、缺用量不伪造成本 | 否（fake agent） | 毫秒级 |
| API 单测 | `tests/test_api.py` | health、chat SSE、历史、trace、usage、503 not_ready | 否（fake runner） | 毫秒级 |
| pass^k 单测 | `tests/test_passk.py` | 严格通过、至少一次、波动题、repeat=1 | 否 | 毫秒级 |
| 检索端到端 | `eval/run_eval.py` | 真实 embedding + Chroma 的 Recall/MRR/延迟 | 是（仅 embedding） | 约 10 秒 / 套 |
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
```

结果写入：

- `eval/results/baseline.json`：检索聚合指标 + 逐题排名 + Top 来源；
- `eval/results/baseline.md`：检索人读版，含失败题；
- `eval/results/answer_baseline.json`：回答级聚合指标 + 逐题回答、判分理由、报告事实核对；
- `eval/results/answer_baseline.md`：回答级人读版，含失败样例与延迟；
- `eval/results/hard_baseline.*`：hard split 检索基线；
- `eval/results/hard_answer.*`：hard split 回答级基线；
- `eval/results/hard_pass3.*`：hard split pass^3，含严格通过率与波动题；
- `eval/results/report_pass3.*`：easy 报告题 pass^3，用于验证报告子流程稳定性。

## 5. 本次结果

### 5.1 测试

| 项 | 结果 |
|---|---|
| 数据集校验 | `VALIDATION_OK rows=80`、`VALIDATION_OK rows=25`，0 个错误 |
| 单元测试 | 74 个全部通过，约 0.5 秒（新增护栏 7、越界 6、观测 7、事件流 5、API 6，中间件扩到 8） |
| 向量库 | 66 个向量、66 个唯一 ID、66 段唯一内容、5 个来源/md5 组合 |
| easy 回答级评测 | 80 题全部执行，生成与判分错误 0（确定性拒答上线后的复跑见 5.9） |
| hard 回答级评测 | 25 题全部执行，生成与判分错误 0 |
| HTTP 冒烟 | QA / 报告 / 越界三条链路 + `/health`、`/history`、`/traces`、`/trace/{id}`、`/usage` 均返回正常 |
| Python 编译 | `py_compile` 通过 |

### 5.2 检索基线

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

### 5.3 一个检索失败

`qa-042`「拖布多久清洗一次？」是唯一在 Top-8 完全未命中的题：

- 正确答案在 `扫拖一体机器人100问.txt` 的 Q3；
- 当前 200 字硬切把 Q1、Q2、Q3 挤进同一个 chunk，主题被稀释；
- Top-8 召回的是 Q12（拖布能不能机洗）、Q13（多久换一次）等相邻主题；
- 这不是标签错，而是**固定字符切分混合问答对**的实证，直接对应 M2 的 QA 感知切分。

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

测试覆盖：74 个单测全部通过；其中 API/事件流 11 个用 fake runner，不依赖模型与网络。**未测**：真实浏览器交互、断线重连、多用户并发、Streamlit 页面（仅代码路径检查）、护栏在真实模型反复试错下的行为（只在单测里构造拦截状态）。

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

报告复跑：四个字段 10/10，平均事实片段覆盖率 97%，忠实率 100%，完全有据率 90%，子流程执行率 100%，`fetch` 参数正确率 100%；生成 P50 7.38 秒、P95 11.88 秒。

## 6. 这些测试有什么用

1. **改动前先知道现状。** easy 检索能说“Top-1 65.45%，Top-3 94.55%”，hard 检索能说“Top-1 55.00%，Top-8 90.00%”，回答能说“easy 通过率 96.25%、hard 通过率 96.00%”，而不是“感觉挺准”。
2. **改动后能证明是否变好。** M2 的切分、混合召回、重排都跑同一套 easy + hard 题；报告子流程跑同一套 10 题，成对报效果和延迟。
3. **失败能定位到层。** hard 的 Source@3 是 100% 但 Span@8 只有 90%，问题在片段级；报告字段完整但忠实度不足，问题在生成而不是取数；pass^k 波动题则说明单次通过不可靠。
4. **面试有证据。** 题量、标注口径、指标定义、复现命令、失败案例都能当场打开；hard split 与 pass^k 能回答“是不是只会背原句、是不是碰巧通过”。
5. **防止回归。** 改配置或换模型后，先跑两套 `--validate-only` 和 74 个单测，再跑基线，避免“修一个坏三个”。

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

## 8. 下一步

1. **M2 检索优化**：QA 感知切分、BM25 + 向量 RRF、交叉编码器重排；用同一套 easy + hard 题对比效果与延迟，仍是当前最优先。
2. **easy 全量 pass^k**：当前 easy 80 题只跑单次，hard 25 题跑 pass^3。easy 题量大、成本高，适合按周或发版前跑。
3. **扩大 hard split**：25 题能暴露问题，但样本仍小；下一步加入更多跨章节组合、近似干扰项和错别字问法，并把越界路由准确率纳入同一套基线。
4. **越界边界补强**：`neg-004` / `neg-009` 已由确定性拒答覆盖（见 5.10）；`hard-neg-002/004` 这类“能力边界”问题仍需提示词或规则补充。
5. **服务化收尾**：给 `/chat/stream` 加鉴权、限流与 `thread_id` 归属校验；trace 换 Postgres 或接 OpenTelemetry；补浏览器端自动化测试。
6. 把两套 `--validate-only` + 74 个单测接进 CI，基线作为人工回归。
