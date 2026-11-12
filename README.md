# 智扫通 · 扫地机器人智能客服

基于 LangChain 1.x + LangGraph 的 RAG 智能客服示例：既能回答扫地/扫拖机器人的选购、使用、故障与保养问题，也能按用户和月份生成使用情况报告。

## 功能

- **知识库问答**：按问答对 / 错误码块 / 编号清单做 QA 感知切分（5 个 TXT → 145 chunks），向量化存入 Chroma（cosine）；回答前走「向量 + BM25 双路召回 → RRF 融合 → `gte-rerank-v2` 重排」，取最相关的 3 个片段交给模型。
- **无证据兜底**：重排最高分低于标定阈值 `min_score` 时，直接返回固定的「知识库里没有相关内容」话术，不调用生成模型、不编造。
- **多轮对话**：LangGraph checkpointer 把会话写入 SQLite，按 `thread_id` 隔离；侧栏列出历史会话（标题取首个问题），点击即可切换，刷新页面也能恢复；每条会话右侧常驻 × 按钮可删除，二次确认后连同该会话的 trace 一起清理。
- **登录与会话归属**：演示级账号（PBKDF2-SHA256 加盐哈希 + 随机令牌，存 SQLite，重启不掉线）；会话按 `thread_id` 认领归属，别人换个线程 ID 也读不到你的历史，侧栏只列自己的会话；登录账号绑定的业务用户 ID 会拦下「查别人使用报告」的请求——不取数、不调模型、零 token。
- **报告生成**：规则路由识别报告意图后走确定性流程——解析用户 ID/月份、直接取数、把记录注入报告提示词，模型只负责一次写作；不依赖模型是否记得调用标记工具。
- **流式 HTTP 服务**：FastAPI + SSE，按 token 推送回答，同时推送路由、工具开始/结束、错误与最终统计；自带单文件前端，支持会话切换与用量查看。
- **工具护栏**：调用前校验参数、限制单轮工具预算、拦截重复成功调用、失败最多重试一次；连续拦截后强制模型收口回答。
- **越界路由**：编程、翻译、医疗、投资、订票等请求直接走确定性拒答，不调用模型，也不消耗 token。
- **trace 与 token 账本**：每次请求落一条 trace，记录路由、工具、首 token、端到端延迟、token 用量与按配置单价估算的成本，可按天/路由查询。
- **启动自检**：向量库为空会自动重建；源文件新增、修改、删除都会同步到向量库；改了切分参数或距离度量（`split_sig` / `hnsw:space` 变化）会整批换 ID 重灌，新旧 chunk 不会互相覆盖。
- **错误兜底**：工具异常、模型异常、知识库初始化失败都会记录日志并返回友好提示，不会把堆栈直接抛给用户。

## 目录结构

```text
AI_RAG_Agent/
├─ app.py                  # Streamlit 界面入口（可选）
├─ server/
│  ├─ app.py               # FastAPI 入口：登录、SSE、历史、trace、用量
│  ├─ auth.py              # 演示级身份：密码哈希、令牌、会话归属
│  ├─ chat_runner.py       # 把 Agent 事件流包装成 trace/成本/done 事件
│  ├─ sse.py               # SSE 编码
│  └─ static/index.html    # 单文件前端（含登录面板）
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
│  ├─ rag_service.py       # 检索 + 总结链（越界拒答、无证据兜底）
│  ├─ hybrid_retriever.py  # 向量 + BM25 → RRF 融合 → 重排 → 截断
│  ├─ bm25.py              # 手写 BM25：ASCII 整词 + 中文单字/双字
│  ├─ reranker.py          # DashScope gte-rerank-v2，失败退回融合序
│  ├─ chunking.py          # QA 感知切分与切分签名
│  └─ vector_store.py      # Chroma 同步、去重、自愈
├─ model/factory.py        # ChatTongyi / DashScope Embedding
├─ prompts/                # 系统提示词、RAG 提示词、报告提示词
├─ config/                 # 模型、向量库、提示词、Agent、观测配置
├─ data/                   # 知识库文件 + external/records.csv
├─ eval/                   # 80 题 easy + 25 题 hard 评测集、脚本、阈值标定与基线
├─ docs/                   # 设计、踩坑、测试说明
├─ tests/                  # 纯函数、切分、检索管线、护栏、API 与事件流测试
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
| GET | `/health` | 服务与模型状态；未就绪返回 503（公开） |
| GET | `/auth/bootstrap` | 是否启用登录、演示账号列表与提示语，不含任何密码字段（公开） |
| POST | `/auth/login` | 请求体 `{"username": "...", "password": "..."}`，返回 `token` 与用户信息；密码错误 401 |
| POST | `/auth/logout` | 撤销当前令牌（需登录） |
| GET | `/auth/me` | 当前登录身份（需登录） |
| POST | `/chat/stream` | 请求体 `{"query": "...", "thread_id": "..."}`，返回 SSE；`thread_id` 可省略，服务端会发一个新的 |
| GET | `/sessions?limit=20` | 历史会话列表；标题取首个问题，只返回当前账号名下的会话（限 1–50） |
| DELETE | `/sessions/{thread_id}` | 删除一个历史会话及其 trace；非法 ID 返回 422，不存在返回 404 |
| GET | `/history/{thread_id}` | 还原某个会话的历史消息 |
| GET | `/traces?limit=20` | 最近 trace 列表（限 1–100） |
| GET | `/trace/{trace_id}` | 单条 trace + spans；不存在返回 404 |
| GET | `/usage?days=7` | 近 N 天 token/成本/延迟汇总（限 1–90） |

除 `/health`、`/auth/bootstrap`、`/auth/login` 和静态页面外，其余接口都要带 `Authorization: Bearer <token>`：缺失或过期返回 401，访问别人名下的会话返回 403。`config/auth.yml` 里把 `enabled` 改成 `false` 可以整体关掉鉴权（评测脚本、CLI、Streamlit 走这条路），此时所有接口都不校验、会话也不做归属过滤。

`/chat/stream` 的 SSE 事件：

| 事件 | 关键字段 | 含义 |
|---|---|---|
| `meta` | `trace_id`, `thread_id`, `route`, `model` | 请求开始，路由已确定 |
| `tool` | `name`, `status`, `args`, `ok`, `blocked`, `duration_ms` | 工具开始/结束 |
| `delta` | `text` | 回答增量 |
| `notice` | `level`, `text` | 降级提示（如报告流程失败转问答） |
| `done` | `answer`, `latency_ms`, `first_token_ms`, `usage`, `cost_cny`, `tool_calls` | 正常结束与统计 |
| `error` | `code`, `message`, `trace_id` | 失败，对外只给通用提示 |

前端用 `fetch` + `ReadableStream` 解析 SSE，支持停止生成、新会话、历史恢复；右侧列出最近会话（标题取首个问题），点击可直接切换，每条会话右侧常驻 × 删除按钮（二次确认，删除当前会话后自动开新会话），并展示近 7 天用量。回答气泡只保留正文，不渲染工具调用与耗时/token/trace 统计；这些数据仍通过 `/trace/{id}` 与 `/usage` 查询。

启用鉴权后，未登录会先弹出登录面板（演示账号可一键填入用户名），登录后顶栏显示当前账号与绑定的用户 ID，并提供退出按钮；令牌存在 `localStorage`，401 会自动退回登录面板，403（会话属于别人）会提示并直接开一个新会话。

## 数据与配置

- `data/*.txt|pdf`：知识库源文件；默认只索引 TXT（`config/chroma.yml` 的 `allow_knowledge_file_type`）。启动时按文件 md5 + 切分签名判断是否需要重新入库，内容或切分方式变化会先删旧向量再写入；文件被删除后对应向量也会清理。
- `data/external/records.csv`：报告用的使用记录，必需列：`用户ID, 特征, 清洁效率, 耗材, 对比, 时间`。
- `config/chroma.yml`：集合名、持久化目录、Top-K（默认 3）、距离度量（`cosine`）、切分模式（`qa_aware` / `recursive`）、单元字数上限与分隔符。
- `config/rag.yml`：对话模型与 embedding 模型名称；`retrieval` 段配置混合召回（`mode` / 两路候选数 / `rrf_k`）与重排（`enabled` / `model` / `candidates` / `min_score`）。
- `config/agent.yml`：外部数据路径、SQLite checkpointer 路径（默认 `checkpoints/agent_memory.sqlite3`）。
- `config/observability.yml`：trace 数据库路径（默认 `checkpoints/observability.sqlite3`）、模型单价与币种。单价按官方价目表填写；未配置单价的模型只记 token，不估算成本。
- `config/auth.yml`：是否启用登录、身份库路径（默认 `checkpoints/auth.sqlite3`，已 gitignore）、令牌有效期、PBKDF2 迭代次数、演示账号（用户名 / 显示名 / 密码 / 绑定的业务用户 ID）与旧会话归属者 `legacy_owner`。演示密码可以用环境变量 `AUTH_DEMO_PASSWORD` 覆盖（只对新写入的账号生效）；要彻底重置，删掉身份库重启即可。

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
python eval/calibrate_min_score.py                # 标定无证据阈值（约 40 秒）
python -m unittest discover -s tests              # 160 个单测，不需要 API Key
```

检索基线（145 chunks、`text-embedding-v4`、向量 + BM25 + RRF + `gte-rerank-v2`、k=8）：

| 数据集 | Span Recall @1 / @3 / @8 | Source Recall @1 / @3 | 检索 P50 |
|---|---|---|---|
| easy（55 题） | **96.36%** / 98.18% / 100.00% | 98.18% / 100.00% | 261 ms |
| hard（20 题） | **80.00%** / 100.00% / 100.00% | 85.00% / 100.00% | 232 ms |

M2 之前的起点是 easy Span@1 65.45% / @3 94.55% / @8 98.18%，hard 55.00% / 80.00% / 90.00%（66 chunks、纯向量 L2、P50 126–157 ms）。分阶段拆开后一个反直觉的结论是：**QA 切分单独上线会让 easy Span@3 从 94.55% 掉到 87.27%**——chunk 从 66 变 145、粒度变细之后，正确块被挤到第 4–7 位，但 Span@1 其实是上升的（36 → 37 题）；混合召回把「找不找得到」补齐（两边 Top-8 未命中归零），真正的排序跃升来自重排（easy Span@1 +27.27 个百分点）。逐题看，easy 18 题排名变好 / 1 题变差，hard 8 好 / 2 差，Top-8 完全未命中从 3 题降到 **0 题**。代价是检索 P50 多约 120–140 ms（重排的一次同步调用）。完整四阶段对比见 `docs/testing.md` §5.13，踩坑与误判风险见 `docs/pitfalls.md` P13–P16。

回答级基线（`qwen-max`，RAG Top-3，报告走确定性 `report_pipeline`）：

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| easy answerable | 55 | 100.00% | 完全正确 **90.91%**；完全有据 98.18%；幻觉 0.00% |
| easy negative | 15 | 100.00% | 拒答正确率 100.00%（确定性越界路由） |
| easy report | 10 | 100.00% | 字段完整 100.00%；忠实 100.00% |
| easy 合计 | 80 | **100.00%** | - |
| hard answerable | 20 | **100.00%** | 完全正确 **95.00%**；完全有据 95.00%；幻觉 0.00% |
| hard negative | 5 | 100.00% | 拒答正确率 100.00% |
| hard 合计 | 25 | **100.00%** | - |

M2 前是 easy 完全正确 81.82% / 完全有据 100.00%，hard 通过率 95.00% / 完全正确 75.00% / 完全有据 90.00%。`qa-042`（拖布清洗频率）与 `hard-qa-016`（儿童锁 + App 防护）这两道旧基线里 Top-8 完全召不回片段的题，现在都是 Span@1 且 correctness=2。

报告字段用确定性规则核对（标签 + 数值 + 单位归一），语义忠实度才交给 LLM judge。报告请求不让模型选工具，而是由 `is_report_query` 路由到 `report_pipeline`（解析 ID/月份 → 直接取数 → 记录注入提示词 → 一次写作）：字段完整率 100%，子流程执行率 100%，生成 P50 约 6.9 秒，首 token 约 0.44 秒。

三个口径必须讲清，否则数字会被误读：

1. **easy negative 的 100% 来自确定性越界路由**（15 道全被规则覆盖、0 token），不代表开放式边界已解决。
2. **hard negative 现在多了一层无证据兜底**：5 道能力边界负样本的重排最高分（0.1261–0.1578）全部低于标定阈值 `min_score = 0.16`，返回固定话术且不调用生成模型；而 75 道可答题的最低分是 0.1673，两组可分但**余量只有约 4%**，换语料或换重排模型必须重跑 `calibrate_min_score.py`。
3. **pass^k 数据还是 M2 之前的**：hard negative 单次 5/5、严格 pass^3 只有 **3/5（60%）**；M2 后本轮 `repeat=1`，稳定性尚未重新量化。easy 报告题严格 pass^3 100% 且零波动。

另外记录两处 M2 的真实代价，不是纯赚：easy 完全有据从 100% 掉到 98.18%（`qa-017` 在更聚焦的上下文外补了一句「电池老化」）；`qa-023` 的片段排名从 @1 退到 @4（重排把同主题的「拖布清洗」块排到了带 gold 关键词的「晾干 / 每周维护」块前面）。k=3 vs k=5 做过敏感性对照，一升一降属噪声，保持 k=3。

评测口径、失败题分析与阈值标定过程见 `eval/README.md`；测试分层、M2 四阶段前后对比与 pass^k 结果见 `docs/testing.md`。

## 当前阶段的说明

`get_user_id`、`get_current_month`、`get_user_location` 目前仍是演示用的随机模拟值，用于跑通报告流程。接入真实业务时，应改为由登录态/服务端时间/定位服务提供，而不是让模型猜测。报告链路已经接上了登录态：账号绑定了业务用户 ID 时，「我的报告」直接用绑定号取数，查别人的 ID 会被拒答。

身份是**演示级**的：账号写死在 `config/auth.yml`，没有注册、刷新令牌、角色权限和找回密码，公网部署前必须改密码或删库重建。服务化仍是单机形态：没有限流，`thread_id` 由客户端生成（服务端只校验归属），trace 与用量是全局的、不按账号隔离——任何登录用户都能在 `/usage` 看到所有人的 token 账本。直接暴露公网前，至少还需要按用户限流、把观测数据按账号隔离，并把账本换成共享存储。

## 排障

- **启动提示“知识库初始化失败”**：检查 `data/` 是否有 TXT（或已启用的文件类型）、`.env` 是否配置 Key、`log/` 下最新日志的具体报错。
- **页面顶部显示“未就绪”**：请求 `GET /health` 看具体错误；常见原因是 API Key 缺失、向量库目录不可写。
- **对话没有记忆**：确认已安装 `langgraph-checkpoint-sqlite`；未安装时会退回内存记忆，进程重启即丢失。FastAPI 与 Streamlit 共用同一个 checkpointer 文件。
- **成本显示“—”**：说明没有拿到供应商 token 用量，或 `config/observability.yml` 里没有该模型单价；这是刻意不估算，不是漏记。
- **正常问题却回“知识库里没有找到相关内容”**：重排最高分低于 `config/rag.yml` 的 `retrieval.rerank.min_score`（当前 0.16）。先跑 `python eval/calibrate_min_score.py` 看分数分布，再决定是调阈值还是补语料；不要直接调高阈值绕过。
- **回答质量突然下降但没报错**：重排调用失败会静默退回 RRF 融合序，日志里有 `rerank` 警告、检索 stats 里是 `rerank=fallback`；检查网络与 DashScope 额度。
- **接口一律 401**：令牌缺失、过期（默认 12 小时）或身份库被删。前端会自动退回登录面板；用 curl 时记得带 `Authorization: Bearer <token>`，裸令牌和其他 scheme 都不认。
- **侧栏会话突然空了**：多半是换了账号——会话按归属过滤，只列自己名下的。登录功能上线前的旧会话会在启动时归到 `config/auth.yml` 的 `legacy_owner`（默认 `demo`），日志里能看到「已把 N 个旧会话归到 legacy_owner 名下」。
- **报告回「只能查询当前登录账号自己的使用记录」**：请求里的用户 ID 和登录账号绑定的 ID 不一致。换成自己的 ID，或用那个 ID 对应的账号登录；不需要这层校验就把 `config/auth.yml` 的 `enabled` 设为 `false`。
- **想换掉演示密码**：设环境变量 `AUTH_DEMO_PASSWORD` 后删掉 `checkpoints/auth.sqlite3` 重启（已存在的账号不会被覆盖）。
- **端口被占用**：FastAPI 用 `--port 8001`；Streamlit 用 `streamlit run app.py --server.port 8502`。
