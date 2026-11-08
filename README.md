# 智扫通 · 扫地机器人智能客服

基于 LangChain 1.x + LangGraph 的 RAG 智能客服示例：既能回答扫地/扫拖机器人的选购、使用、故障与保养问题，也能按用户和月份生成使用情况报告。

## 功能

- **知识库问答**：自动把 `data/` 下的 TXT（可配置扩展 PDF）切片、向量化，存入 Chroma；回答前检索最相关的 3 个片段。
- **多轮对话**：LangGraph checkpointer 把会话写入 SQLite，按 `thread_id` 隔离；侧栏列出历史会话（标题取首个问题），点击即可切换，刷新页面也能恢复。
- **报告生成**：规则路由识别报告意图后走确定性流程——解析用户 ID/月份、直接取数、把记录注入报告提示词，模型只负责一次写作；不依赖模型是否记得调用标记工具。
- **流式 HTTP 服务**：FastAPI + SSE，按 token 推送回答，同时推送路由、工具开始/结束、错误与最终统计；自带单文件前端，支持会话切换与用量查看。
- **工具护栏**：调用前校验参数、限制单轮工具预算、拦截重复成功调用、失败最多重试一次；连续拦截后强制模型收口回答。
- **越界路由**：编程、翻译、医疗、投资、订票等请求直接走确定性拒答，不调用模型，也不消耗 token。
- **trace 与 token 账本**：每次请求落一条 trace，记录路由、工具、首 token、端到端延迟、token 用量与按配置单价估算的成本，可按天/路由查询。
- **启动自检**：向量库为空会自动重建；源文件新增、修改、删除都会同步到向量库。
- **错误兜底**：工具异常、模型异常、知识库初始化失败都会记录日志并返回友好提示，不会把堆栈直接抛给用户。

## 目录结构

```text
AI_RAG_Agent/
├─ app.py                  # Streamlit 界面入口（可选）
├─ server/
│  ├─ app.py               # FastAPI 入口：SSE、历史、trace、用量
│  ├─ chat_runner.py       # 把 Agent 事件流包装成 trace/成本/done 事件
│  ├─ sse.py               # SSE 编码
│  └─ static/index.html    # 单文件前端
├─ agent/
│  ├─ react_agent.py       # Agent 组装、路由、流式事件、会话列表/历史
│  ├─ report_router.py     # 报告意图识别 + 用户 ID/月份抽取（纯函数）
│  ├─ report_pipeline.py   # 确定性报告流程：解析 → 取数 → 单次写作
│  ├─ scope_router.py      # 报告/越界/问答三路确定性路由
│  ├─ guardrails.py        # 参数校验、工具预算、重复调用拦截
│  └─ tools/
│     ├─ agent_tools.py    # RAG、报告数据、用户/月份/天气等工具
│     └─ middleware.py     # 工具监控、护栏、日志、报告提示词兜底切换
├─ observability/
│  ├─ usage.py             # token 用量提取与去重
│  ├─ pricing.py           # 按配置单价估算成本
│  └─ store.py             # SQLite trace/span 与用量汇总
├─ rag/
│  ├─ rag_service.py       # 检索 + 总结链（越界问题确定性拒答）
│  └─ vector_store.py      # Chroma 同步、去重、自愈
├─ model/factory.py        # ChatTongyi / DashScope Embedding
├─ prompts/                # 系统提示词、RAG 提示词、报告提示词
├─ config/                 # 模型、向量库、提示词、Agent、观测配置
├─ data/                   # 知识库文件 + external/records.csv
├─ eval/                   # 80 题 easy + 25 题 hard 评测集、脚本与基线
├─ docs/                   # 设计、踩坑、测试说明
├─ tests/                  # 纯函数、护栏、API 与事件流测试
├─ utils/                  # 配置、日志、路径、文件、提示词加载
└─ requirements.txt
```

## 环境要求

- Python 3.11（本项目在 3.11.9 验证）
- DashScope API Key（通义千问 + text-embedding-v4）

## 安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

国内网络如果下载慢，可临时加清华镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 配置密钥

复制示例文件并填入自己的 Key：

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

`.env` 中的变量：

```dotenv
DASHSCOPE_API_KEY=sk-...
```

`utils/config_handler.py` 在导入配置时会自动加载项目根目录的 `.env`。**不要把 `.env` 提交到仓库**（已在 `.gitignore` 中忽略）。

## 运行

推荐用 FastAPI + SSE 入口（前端、接口、trace 一体）：

```bash
.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000/>。首次启动会做知识库自检，页面顶部的状态灯会显示模型是否就绪；启动失败时接口不会直接崩，`/health` 会返回可读原因。

旧的 Streamlit 入口仍保留：

```bash
streamlit run app.py
```

也可以单独跑一次 Agent：

```bash
python -m agent.react_agent
```

### HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务与模型状态；未就绪返回 503 |
| POST | `/chat/stream` | 请求体 `{"query": "...", "thread_id": "..."}`，返回 SSE |
| GET | `/sessions?limit=20` | 历史会话列表；标题取首个问题（限 1–50） |
| GET | `/history/{thread_id}` | 还原某个会话的历史消息 |
| GET | `/traces?limit=20` | 最近 trace 列表（限 1–100） |
| GET | `/trace/{trace_id}` | 单条 trace + spans；不存在返回 404 |
| GET | `/usage?days=7` | 近 N 天 token/成本/延迟汇总（限 1–90） |

`/chat/stream` 的 SSE 事件：

| 事件 | 关键字段 | 含义 |
|---|---|---|
| `meta` | `trace_id`, `thread_id`, `route`, `model` | 请求开始，路由已确定 |
| `tool` | `name`, `status`, `args`, `ok`, `blocked`, `duration_ms` | 工具开始/结束 |
| `delta` | `text` | 回答增量 |
| `notice` | `level`, `text` | 降级提示（如报告流程失败转问答） |
| `done` | `answer`, `latency_ms`, `first_token_ms`, `usage`, `cost_cny`, `tool_calls` | 正常结束与统计 |
| `error` | `code`, `message`, `trace_id` | 失败，对外只给通用提示 |

前端用 `fetch` + `ReadableStream` 解析 SSE，支持停止生成、新会话、历史恢复；右侧列出最近会话（标题取首个问题），点击可直接切换，近 7 天用量与单条 trace 入口分别保留在侧栏和每条回答底部。

## 数据与配置

- `data/*.txt|pdf`：知识库源文件；默认只索引 TXT（`config/chroma.yml` 的 `allow_knowledge_file_type`）。启动时按文件 md5 判断是否需要重新入库，内容变化会先删旧向量再写入；文件被删除后对应向量也会清理。
- `data/external/records.csv`：报告用的使用记录，必需列：`用户ID, 特征, 清洁效率, 耗材, 对比, 时间`。
- `config/chroma.yml`：集合名、持久化目录、Top-K、分片大小与分隔符。
- `config/agent.yml`：外部数据路径、SQLite checkpointer 路径（默认 `checkpoints/agent_memory.sqlite3`）。
- `config/rag.yml`：对话模型与 embedding 模型名称。
- `config/observability.yml`：trace 数据库路径（默认 `checkpoints/observability.sqlite3`）、模型单价与币种。单价按官方价目表填写；未配置单价的模型只记 token，不估算成本。

## 护栏与越界

- **参数校验**：`fetch_external_data` 要求 4 位 `user_id` 与 `YYYY-MM`；`get_weather` 要求非空城市名；`rag_summarize` 要求非空 query；无参工具不接受额外参数。
- **预算与去重**：单轮最多 6 次工具调用；完全相同的成功调用会被拦截；失败允许换参数重试一次。
- **强制收口**：连续 2 次被拦截后，向模型注入一次「停止调工具、直接回答或说明不知道」的提示，且只注入一次。
- **越界拒答**：`agent/scope_router.py` 用确定性规则识别编程、翻译、医疗、投资、订票、其他家电故障等请求，直接返回固定话术，不调用模型。规则宁可漏判也不误伤——例如「拖布能用洗衣机洗吗」属于正常问答，只有「上门修洗衣机」才判越界。

护栏状态是**每次请求独立**的，不跨会话累计；规则是启发式，复杂越界请求仍可能落到模型。

## 观测与成本

每个请求在 `traces` 表落一条记录，工具与路由各落 span：

- **延迟**：端到端 `latency_ms`、首个 token `first_token_ms`、首尾 token 间隔 `llm_ms`。
- **用量**：从 DashScope 响应的 `response_metadata.token_usage` 读取，按 `request_id` 去重，跨多次模型调用累加。
- **成本**：仅当拿到供应商用量且模型有配置单价时估算；缺数据就留空，不用默认值伪造。
- **越界请求**：`usage_source = "none"`，代表本来就没调用模型，不计入「供应商漏报用量」。

当前账本是单机 SQLite（WAL 模式，每次操作独立连接），够演示和本地复盘；多实例部署应换成 Postgres 或接 OpenTelemetry。

## 评测

仓库自带两套评测集：**easy 80 题**（55 可回答 / 15 负样本 / 10 报告）与 **hard 25 题**（20 可回答 / 5 负样本）。hard 用改写问法、跨章节组合和领域邻近负样本，专门看模型是不是只会背原句。

```bash
python eval/run_eval.py --validate-only          # 校验 easy gold 标签
python eval/run_eval.py --ks 1,3,5,8             # easy 检索基线
python eval/run_eval.py --dataset eval/dataset_hard.jsonl --ks 1,3,5,8
python eval/run_answer_eval.py                   # easy 回答级基线（约 5 分钟）
python eval/run_answer_eval.py --dataset eval/dataset_hard.jsonl \
  --out eval/results/hard_answer.json             # hard 回答级基线
python eval/run_answer_eval.py --dataset eval/dataset_hard.jsonl \
  --repeat 3 --out eval/results/hard_pass3.json   # pass^3
```

检索基线（66 chunks、`text-embedding-v4`、k=8）：

| 数据集 | Span Recall @1 / @3 / @8 | Source Recall @1 / @3 |
|---|---|---|
| easy（55 题） | 65.45% / 94.55% / 98.18% | 85.45% / 100.00% |
| hard（20 题） | 55.00% / 80.00% / 90.00% | 75.00% / 100.00% |

回答级基线（`qwen-max`，RAG Top-3，报告走确定性 `report_pipeline`）：

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| easy answerable | 55 | 100.00% | 完全正确 81.82%；完全有据 100.00%；幻觉 0.00% |
| easy negative | 15 | 100.00% | 拒答正确率 100.00%（确定性越界路由） |
| easy report | 10 | 100.00% | 字段完整 100.00%；忠实 100.00% |
| easy 合计 | 80 | 100.00% | - |
| hard answerable | 20 | 95.00% | 完全正确 75.00%；完全有据 90.00%；幻觉 0.00% |
| hard negative | 5 | 100.00% | 拒答正确率 100.00% |
| hard 合计 | 25 | 96.00% | - |

报告字段用确定性规则核对（标签 + 数值 + 单位归一），语义忠实度才交给 LLM judge。P7 已从结构上修复：报告请求不再让模型选工具，而是由 `is_report_query` 路由到 `report_pipeline`（解析 ID/月份 → 直接取数 → 记录注入提示词 → 一次写作）；报告字段完整率 100%，子流程执行率 100%，生成 P50 **7.38 秒**，首 token 约 0.44 秒。

需要看清两个口径：easy negative 的 100% 来自 M4 的确定性越界路由（15 道 easy 负样本都被规则覆盖，不再调用模型），不代表开放式边界已经解决；hard negative 仍是模型判断，pass^3 只有 60%。easy answerable 这次 55/55 通过，但完全正确率仍是 81.82%，`qa-042` 只是 partial 通过，检索切分问题（M2）没有修。

pass^k（同题独立跑 3 次）暴露了单次数字的乐观：hard negative 单次 5/5，严格 pass^3 只有 **3/5（60%）**，波动来自「没有明确拒绝、继续给跨领域建议」；hard answerable 严格 95%，easy 报告严格 **100% 且零波动**。当前真正的短板是硬负样本的能力边界（`hard-neg-002` / `hard-neg-004`）和片段级检索（`qa-042` / `hard-qa-016`）。评测口径、失败题分析见 `eval/README.md`，测试分层、P7 新旧机制与 pass^k 结果见 `docs/testing.md`；下一步是 M2 的 QA 感知切分、混合召回与重排。

## 当前阶段的说明

`get_user_id`、`get_current_month`、`get_user_location` 目前仍是演示用的随机模拟值，用于跑通报告流程。接入真实业务时，应改为由登录态/服务端时间/定位服务提供，而不是让模型猜测。

服务化仍是单机演示形态：没有鉴权、限流和多用户隔离，`thread_id` 由客户端生成；trace 存本地 SQLite。直接暴露公网前，至少需要补上接口鉴权、按用户限流、`thread_id` 归属校验，并把账本换成共享存储。

## 排障

- **启动提示“知识库初始化失败”**：检查 `data/` 是否有 TXT（或已启用的文件类型）、`.env` 是否配置 Key、`log/` 下最新日志的具体报错。
- **页面顶部显示“未就绪”**：请求 `GET /health` 看具体错误；常见原因是 API Key 缺失、向量库目录不可写。
- **对话没有记忆**：确认已安装 `langgraph-checkpoint-sqlite`；未安装时会退回内存记忆，进程重启即丢失。FastAPI 与 Streamlit 共用同一个 checkpointer 文件。
- **成本显示“—”**：说明没有拿到供应商 token 用量，或 `config/observability.yml` 里没有该模型单价；这是刻意不估算，不是漏记。
- **端口被占用**：FastAPI 用 `--port 8001`；Streamlit 用 `streamlit run app.py --server.port 8502`。
