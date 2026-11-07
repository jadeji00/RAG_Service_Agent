# 评测

> 目标：把「检索对不对、答得对不对」变成可复现的数字。当前已完成 **easy / hard 两套检索与回答基线**，并支持 `--repeat` 跑 **pass^k**。
>
> 测试分层、39 个用例与结果说明见 [docs/testing.md](../docs/testing.md)；报告延迟剖析见该文档 5.6 节。

## 1. 数据集

两套数据集，schema 相同：

| 文件 | 题量 | 定位 |
|---|---|---|
| `eval/dataset.jsonl` | 80（answerable 55 / negative 15 / report 10） | **easy split**：多数题目接近知识库原问句，用于稳定回归 |
| `eval/dataset_hard.jsonl` | 25（answerable 20 / negative 5） | **hard split**：改写问法、跨章节组合、领域邻近负样本，用于看真实理解 |

easy split 的用途：

| 类型 | 数量 | 用途 |
|---|---|---|
| `answerable` | 55 | 知识库可回答，评检索与回答 |
| `negative` | 15 | 超范围问题，评拒答 |
| `report` | 10 | 按用户/月生成报告，评字段完整性 |

### 字段

```json
{
  "id": "qa-001",
  "type": "answerable",
  "question": "60㎡以下的小户型选扫地机器人，重点看什么？",
  "gold_sources": ["选购指南.txt"],
  "gold_sections": ["一、先明确需求"],
  "gold_terms": ["60㎡以下", "性价比"],
  "gold_answer": "……",
  "tags": ["选购", "小户型"]
}
```

- `gold_sources`：允许的正确来源文件，可多个。
- `gold_sections`：章节标题或题号，如 `二、每周维护`、`Q3`、`【E01 碰撞传感器异常】`。
- `gold_terms`：答案关键词；有它时以关键词为准，避免只命中章节标题。
- **不标 chunk id**：切分策略一变 id 就失效，gold 必须能跨切分复用。

报告题额外有 `user_id`、`month`、`expected_fields`，gold 由 `records.csv` 对应行生成。

## 2. 指标

| 指标 | 定义 |
|---|---|
| Span Recall@k | 前 k 条里是否存在「来源命中 + 答案关键词/章节命中」的片段 |
| Span MRR@k | 首个相关片段排名的倒数均值，未命中记 0 |
| Source Recall@k | 前 k 条里是否出现正确来源文件 |
| 延迟 | 单题检索（embedding + Chroma）的 P50 / P95 |

每题只有一个 gold 答案集合，所以 **Recall@k 等价于 hit rate**，不是多文档召回率。Span 与 Source 分开报，是为了区分「文件没召回」和「文件召回但段落不对」。

### 回答级指标

`eval/run_answer_eval.py` 对三类题分别计分：

| 指标 | 定义 |
|---|---|
| correctness 0/1/2 | LLM judge 对照 gold answer：2 完全正确、1 部分正确、0 错误/答非所问 |
| faithfulness 0/1/2 | 回答中的事实能否在检索上下文（或 CSV 原始记录）中找到依据 |
| refusal | 负样本是否明确拒答，或说明产品不具备该能力并拉回业务范围 |
| 字段完整率 | 报告是否覆盖 `expected_fields`：每个字段至少命中 1 条事实，且命中率 ≥ 50% |
| 报告忠实率 | 报告中的数字、状态、对比是否来自 CSV；编造记 0 |
| 建议率 | 报告是否给出至少一条针对该用户数据的可执行建议 |
| 报告子流程执行率 | 报告请求是否命中 `is_report_query` 并走确定性 `report_pipeline` |
| `fetch_args_ok` | 取数调的 `user_id` / `month` 是否与题目给定值一致 |
| 通过率 | answerable = correctness ≥ 1 且 faithfulness ≥ 1；negative = refusal = 1；report = 字段完整且 faithfulness ≥ 1 |

报告字段用**确定性规则**核对（标签 + 数值 + 单位归一），不交给模型；语义忠实度才用 LLM judge。这样既能抓住漏字段，也能避免模型「自己判自己」。

### pass^k

同一道题独立跑 k 次（`--repeat k`），每题记录 `pass_count`：

| 指标 | 定义 | 回答的问题 |
|---|---|---|
| 严格 pass^k | k 次全部通过 | 系统是否**稳定**会 |
| 至少一次通过 | k 次里至少 1 次通过 | 系统是否**有可能**会 |
| 平均单次通过率 | `Σ pass_count / (题数 × k)` | 单次运行的期望水平 |
| 波动题 | `0 < pass_count < k` | 哪些题靠运气 |

单次通过率会掩盖不稳定；严格 pass^k 与「至少一次」一起报，才能区分「稳定会」和「碰巧对」。

## 3. 当前基线

- 日期：2026-11-06
- 向量库：`agent` / **66 chunks**（只索引 5 个 txt；本地 PDF 不参与）
- Embedding：`text-embedding-v4`；检索上限 k=8
- 样本：55 道 answerable

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall | 65.45% | 94.55% | 98.18% | 98.18% |
| Span MRR | 65.45% | 78.79% | 79.52% | 79.52% |
| Source Recall | 85.45% | 100.00% | 100.00% | 100.00% |

检索延迟：P50 **126.05 ms** / P95 **296.50 ms**。完整逐题结果见 `eval/results/baseline.json`，可读版见 `eval/results/baseline.md`。

### 已知失败：`qa-042`（拖布多久清洗一次）

- 目标片段是 `扫拖一体机器人100问.txt` 的 Q3。
- 现配置按 200 字硬切，Q3 和 Q1、Q2 挤在同一个 chunk 里，主题被稀释；Top-8 没有召回 Q3。
- 这不是 gold 标错，而是 **固定字符切分切断/混合问答对** 的实证，M2 的 QA 感知切分需要解决。

### hard split 检索基线

- 日期：2026-11-07；数据集：`eval/dataset_hard.jsonl`，20 道 answerable
- 其余口径与 easy 相同（66 chunks、`text-embedding-v4`、k=8）

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall | 55.00% | 80.00% | 90.00% | 90.00% |
| Span MRR | 55.00% | 66.67% | 69.17% | 69.17% |
| Source Recall | 75.00% | 100.00% | 100.00% | 100.00% |

检索延迟：P50 **156.92 ms** / P95 **607.88 ms**。完整结果见 `eval/results/hard_baseline.json` / `.md`。

改写问法后 Span Recall@1 从 easy 的 65.45% 降到 55.00%，@8 从 98.18% 降到 90.00%；Source Recall@3 仍是 100%，说明文件基本找得到，**差距主要出在片段级匹配**，与 M2 的 QA 切分方向一致。Top-8 完全未命中两道：

- `hard-qa-007`「老人不想用手机 App，选机型看哪些操作方式」——问法已偏离原文的「语音控制/实体按键」；
- `hard-qa-016`「小孩在家，边刷和 App 分别防什么」——跨章节组合（安全 + App），固定字符 chunk 没有同时覆盖两部分。

### 回答级基线

- 日期：2026-11-07 13:36:55
- 生成 / 判分：`qwen-max`；RAG 生成 Top-3；报告走确定性 `report_pipeline`
- 样本：answerable 55 / negative 15 / report 10，执行与判分错误 0
- 结果文件：`eval/results/answer_baseline.json` / `.md`

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| answerable | 55 | 98.18% | 完全正确 80.00%；完全有据 98.18%；幻觉 0.00% |
| negative | 15 | 86.67% | 拒答正确率 86.67% |
| report | 10 | 100.00% | 字段完整 100.00%；忠实 100.00% |
| **合计** | **80** | **96.25%** | - |

知识问答：44 题完全正确、10 题部分正确、1 题错误（`qa-042`）；faithfulness=2 有 54 题、=1 有 1 题、=0 有 0 题；关键词命中率 83.33%。`qa-042` 这次被判错，和检索 Top-8 未命中 Q3 一致——切分问题终于体现在回答分上，而不是靠模型猜对掩盖。

拒答失败 2 道：`neg-004` 直接写 Python 爬虫，`neg-009` 追问要翻译的文本；`neg-011` 本轮被判为已拒答并通过，说明边界样本有波动，稳定失败的是前两道。hard 5 道负样本全部拒答。

报告：四个字段全部 10/10 覆盖，平均事实片段覆盖率 99%，忠实率 100%，完全有据率 70%，可执行建议率 100%，**报告子流程执行率 100%**，`fetch` 参数正确率 100%，10 道题全部 `route=report_pipeline`。报告生成 P50 **7.25 秒**、P95 **10.22 秒**，比旧版 12.5 / 16.5 秒明显下降；剖析见 `docs/testing.md` 5.6 节。

### hard split 回答基线

- 日期：2026-11-07；数据集：`eval/dataset_hard.jsonl`（20 可答 + 5 负样本）
- 结果文件：`eval/results/hard_answer.json` / `.md`

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| answerable | 20 | 95.00% | 完全正确 75.00%；完全有据 90.00%；幻觉 0.00% |
| negative | 5 | 100.00% | 拒答正确率 100.00% |
| **合计** | **25** | **96.00%** | - |

唯一失败是 `hard-qa-016`（儿童锁/边刷防护），检索 Top-8 未命中；`hard-qa-007` 检索也未命中，但回答被判部分正确并通过。hard 的完全正确率（75.00%）与关键词命中率（67.50%）都低于 easy，说明改写问法确实更难。

### pass^k 结果

同题独立跑 3 次（`--repeat 3`）：

| 数据集 | 题数 | 严格 pass^3 | 至少一次通过 | 平均单次通过 | 波动题 |
|---|---|---|---|---|---|
| hard answerable | 20 | 95.00% | 100.00% | 96.67% | `hard-qa-016` |
| hard negative | 5 | **60.00%** | 100.00% | 86.67% | `hard-neg-002` / `hard-neg-004` |
| hard 合计 | 25 | **88.00%** | 100.00% | 94.67% | 3 道 |
| easy report | 10 | **100.00%** | 100.00% | 100.00% | 0 道 |

hard negative 单次是 5/5，严格 pass^3 只有 3/5：两次失败都是「没有明确拒绝，而是继续给跨领域操作建议」。报告题 pass^3 全过且零波动，说明确定性报告子流程把控制流固定后，稳定性显著优于旧 ReAct 链路。结果文件：`hard_pass3.json` / `report_pass3.json`。

## 4. 复现

```bash
# 只校验数据和 gold 标签，不调用模型
python eval/run_eval.py --validate-only

# 先跑 10 题冒烟
python eval/run_eval.py --limit 10

# 完整检索基线
python eval/run_eval.py --ks 1,3,5,8

# 回答级冒烟：每类只跑前 2 题
python eval/run_answer_eval.py --types answerable,negative --limit 2
python eval/run_answer_eval.py --types report --limit 1

# 完整回答级基线（55 + 15 + 10，约 5 分钟）
python eval/run_answer_eval.py

# hard split：检索 + 回答
python eval/run_eval.py --dataset eval/dataset_hard.jsonl --ks 1,3,5,8
python eval/run_answer_eval.py --dataset eval/dataset_hard.jsonl \
  --out eval/results/hard_answer.json

# pass^k：同一题独立跑 3 次
python eval/run_answer_eval.py --dataset eval/dataset_hard.jsonl \
  --repeat 3 --out eval/results/hard_pass3.json

# 纯函数单测（不需要 API Key）
python -m unittest discover -s tests
```

输出：

- `eval/results/baseline.json`：检索聚合指标 + 逐题排名 + Top 来源；
- `eval/results/baseline.md`：检索人读版，失败题可直接用于分析；
- `eval/results/answer_baseline.json`：回答级聚合指标 + 逐题回答、判分理由、报告事实核对；
- `eval/results/answer_baseline.md`：回答级人读版，含失败样例与延迟；
- `eval/results/hard_baseline.*` / `hard_answer.*`：hard split 的检索与回答基线；
- `eval/results/hard_pass3.*`：hard split 的 pass^3（严格 / 至少一次 / 波动题）；
- `eval/results/report_pass3.*`：easy 报告题的 pass^3（验证报告子流程稳定性）。

## 5. 新增题目

1. 在 `eval/dataset.jsonl` 追加一行，`id` 不重复；
2. 填 `gold_sources` + `gold_sections`/`gold_terms`，**不要写 chunk id**；
3. 跑 `--validate-only`，它会检查每个章节/关键词是否仍能在对应源文件中找到；
4. 跑评测，确认新题不会因为标签错误制造假失败。

## 6. 限制与下一步

- **easy / hard 分开报**：easy 多数题目来自知识库原问句，分数偏高；hard 是改写与跨节组合，25 题样本仍小，不能当作真实用户分布。
- **同源判分**：生成与判分默认都是 `qwen-max`，存在自偏好；已用 gold answer、明确 rubric 与 `temperature=0` 降低，但未做人工双标与一致性系数。
- **粗粒度 rubric**：correctness / faithfulness 是 0/1/2，同一条回答跨运行可能有 1 分波动；hard split 用 pass^3 暴露波动，easy 目前只跑单次。
- **单 gold 集合**：真实场景可能有多个正确片段；当前 Recall@k 等价于 hit rate。
- **报告字段是词面下界**：标签 + 数值 + 单位归一能抓漏字段，但不判断语义等价；方向与编造交给 LLM judge。
- **报告题都显式给出 user_id / month**：`get_user_id` / `get_current_month` 的随机 mock 分支未被覆盖。
- **报告生成 P50 约 7.25 秒**：确定性流程已去掉选工具往返，剩余成本主要是最终报告的输出长度；当前只统计总耗时，未拆首 token 与流式延迟。
- **规则路由有覆盖边界**：`is_report_query` 只覆盖显式报告词和「ID + 月份 + 数据词」，换一种完全不含这些词的说法会回退 ReAct；用 80 题逐题断言路由结果做回归。
- **两种检索口径**：检索基线用 k=8 看召回上限；回答级用 k=3 复现生产链路，不能直接相减。
- **下一步（M2）**：QA 感知切分 + BM25/向量 RRF + 重排，用同一套 easy + hard 题前后对比，效果与延迟一起报；报告链路继续做流式与更快模型。
