# 智扫通 · 扫地机器人智能客服

基于 LangChain 1.x + LangGraph 的 RAG 智能客服示例：既能回答扫地/扫拖机器人的选购、使用、故障与保养问题，也能按用户和月份生成使用情况报告。

## 功能

- **知识库问答**：自动把 `data/` 下的 TXT（可配置扩展 PDF）切片、向量化，存入 Chroma；回答前检索最相关的 3 个片段。
- **多轮对话**：LangGraph checkpointer 把会话写入 SQLite，按 `thread_id` 隔离；刷新页面可恢复同一段对话。
- **报告生成**：通过工具调用获取用户 ID、月份与使用记录，由中间件切换到报告提示词。
- **启动自检**：向量库为空会自动重建；源文件新增、修改、删除都会同步到向量库。
- **错误兜底**：工具异常、模型异常、知识库初始化失败都会记录日志并返回友好提示，不会把堆栈直接抛给用户。

## 目录结构

```text
AI_RAG_Agent/
├─ app.py                  # Streamlit 界面入口
├─ agent/
│  ├─ react_agent.py       # Agent 组装、checkpointer、会话历史读取
│  └─ tools/
│     ├─ agent_tools.py    # RAG、报告数据、用户/月份/天气等工具
│     └─ middleware.py     # 工具监控、日志、报告提示词切换
├─ rag/
│  ├─ rag_service.py       # 检索 + 总结链
│  └─ vector_store.py      # Chroma 同步、去重、自愈
├─ model/factory.py        # ChatTongyi / DashScope Embedding
├─ prompts/                # 系统提示词、RAG 提示词、报告提示词
├─ config/                 # 模型、向量库、提示词、Agent 配置
├─ data/                   # 知识库文件 + external/records.csv
├─ eval/                   # 80 题评测集、检索评测脚本与基线
├─ docs/                   # 设计、踩坑、测试说明
├─ tests/                  # 纯函数与数据校验测试
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

```bash
streamlit run app.py
```

浏览器打开终端提示的地址即可。侧边栏会显示当前会话 ID，点「开启新会话」会换一个新的 `thread_id`。

也可以单独跑一次 Agent：

```bash
python -m agent.react_agent
```

## 数据与配置

- `data/*.txt|pdf`：知识库源文件；默认只索引 TXT（`config/chroma.yml` 的 `allow_knowledge_file_type`）。启动时按文件 md5 判断是否需要重新入库，内容变化会先删旧向量再写入；文件被删除后对应向量也会清理。
- `data/external/records.csv`：报告用的使用记录，必需列：`用户ID, 特征, 清洁效率, 耗材, 对比, 时间`。
- `config/chroma.yml`：集合名、持久化目录、Top-K、分片大小与分隔符。
- `config/agent.yml`：外部数据路径、SQLite checkpointer 路径（默认 `checkpoints/agent_memory.sqlite3`）。
- `config/rag.yml`：对话模型与 embedding 模型名称。

## 评测

仓库自带 80 题评测集（55 道可回答 / 15 道负样本 / 10 道报告），并把「检索对不对」变成可复现的数字：

```bash
python eval/run_eval.py --validate-only     # 只校验 gold 标签
python eval/run_eval.py --ks 1,3,5,8        # 跑完整检索基线
```

当前基线（55 道可回答、66 chunks、`text-embedding-v4`）：

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall | 65.45% | 94.55% | 98.18% | 98.18% |
| Span MRR | 65.45% | 78.79% | 79.52% | 79.52% |
| Source Recall | 85.45% | 100.00% | 100.00% | 100.00% |

评测口径、失败题分析见 `eval/README.md`，测试分层与用例说明见 `docs/testing.md`。当前题目多来自知识库原问句，属于 easy split；下一步是 QA 感知切分、混合召回与重排。

## 当前阶段的说明

`get_user_id`、`get_current_month`、`get_user_location` 目前仍是演示用的随机模拟值，用于跑通报告流程。接入真实业务时，应改为由登录态/服务端时间/定位服务提供，而不是让模型猜测。

## 排障

- **启动提示“知识库初始化失败”**：检查 `data/` 是否有 TXT（或已启用的文件类型）、`.env` 是否配置 Key、`log/` 下最新日志的具体报错。
- **对话没有记忆**：确认已安装 `langgraph-checkpoint-sqlite`；未安装时会退回内存记忆，进程重启即丢失。
- **端口被占用**：`streamlit run app.py --server.port 8502`。
