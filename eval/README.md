# 评测

> 目标：把「检索对不对、答得对不对」变成可复现的数字。当前已完成**检索基线**与**回答级基线**；hard split 与 pass^k 尚未加入。
>
> 测试分层、27 个用例与结果说明见 [docs/testing.md](../docs/testing.md)；报告延迟剖析见该文档 5.6 节。

## 1. 数据集

`eval/dataset.jsonl`，共 **80 题**：

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
| 通过率 | answerable = correctness ≥ 1 且 faithfulness ≥ 1；negative = refusal = 1；report = 字段完整且 faithfulness ≥ 1 |

报告字段用**确定性规则**核对（标签 + 数值 + 单位归一），不交给模型；语义忠实度才用 LLM judge。这样既能抓住漏字段，也能避免模型「自己判自己」。

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

### 回答级基线

- 日期：2026-11-06 21:53:38
- 生成 / 判分：`qwen-max`，`temperature=0`；RAG 生成 Top-3，报告走完整 Agent
- 样本：answerable 55 / negative 15 / report 10，执行与判分错误 0
- 结果文件：`eval/results/answer_baseline.json` / `.md`

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| answerable | 55 | 100.00% | 完全正确 83.64%；完全有据 98.18%；幻觉 0.00% |
| negative | 15 | 80.00% | 拒答正确率 80.00% |
| report | 10 | 100.00% | 字段完整 100.00%；忠实 100.00% |
| **合计** | **80** | **96.25%** | - |

知识问答：46 题完全正确、9 题部分正确、0 题错误；faithfulness=2 有 54 题、=1 有 1 题、=0 有 0 题。`qa-042` 这次判为部分正确并通过，但检索仍然 Top-8 未命中 Q3，说明切分问题没有被回答通过掩盖。

拒答的 3 道失败是边界问题：`neg-004` 直接写 Python 爬虫，`neg-009` 追问要翻译的文本，`neg-011` 先说资料没有洗衣机信息、随后又给通用排查建议；三者都没有明确把话题拉回扫地/扫拖机器人范围。同一批 15 道负样本单独重跑一次是 13/15（86.67%），`neg-011` 从 0 变 1，稳定失败的是 `neg-004` / `neg-009`——这说明单次判分存在波动，也是当前只报单次、尚未做 pass^k 的局限。

报告字段完整率从 50% 提升到 100%：四个字段全部 10/10 覆盖，工具序列正确率 90%，`fetch` 参数正确率 100%。`rep-005` 仍跳过了 `fill_context_for_report`，但 `fetch_external_data` 成功后会兜底切换报告提示词，所以字段没有缺失（P7）。报告忠实率 100%，完全有据率 70%，`rep-002/003/010` 仍有个别数值推断。报告生成 P50 12.5 秒、P95 16.5 秒，剖析见 `docs/testing.md` 5.6 节。

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

# 完整回答级基线（55 + 15 + 10，约 8 分钟）
python eval/run_answer_eval.py

# 纯函数单测（不需要 API Key）
python -m unittest discover -s tests
```

输出：

- `eval/results/baseline.json`：检索聚合指标 + 逐题排名 + Top 来源；
- `eval/results/baseline.md`：检索人读版，失败题可直接用于分析；
- `eval/results/answer_baseline.json`：回答级聚合指标 + 逐题回答、判分理由、报告事实核对；
- `eval/results/answer_baseline.md`：回答级人读版，含失败样例与延迟。

## 5. 新增题目

1. 在 `eval/dataset.jsonl` 追加一行，`id` 不重复；
2. 填 `gold_sources` + `gold_sections`/`gold_terms`，**不要写 chunk id**；
3. 跑 `--validate-only`，它会检查每个章节/关键词是否仍能在对应源文件中找到；
4. 跑评测，确认新题不会因为标签错误制造假失败。

## 6. 限制与下一步

- **easy split**：多数题目直接来自知识库原问句，分数偏高；下一步补改写问法、跨节组合的 hard split。
- **同源判分**：生成与判分默认都是 `qwen-max`，存在自偏好；已用 gold answer、明确 rubric 与 `temperature=0` 降低，但未做人工双标与一致性系数。
- **粗粒度 rubric**：correctness / faithfulness 是 0/1/2，同一条回答跨运行可能有 1 分波动；当前只跑单次，未做 pass^k。
- **单 gold 集合**：真实场景可能有多个正确片段；当前 Recall@k 等价于 hit rate。
- **报告字段是词面下界**：标签 + 数值 + 单位归一能抓漏字段，但不判断语义等价；方向与编造交给 LLM judge。
- **报告题都显式给出 user_id / month**：`get_user_id` / `get_current_month` 的随机 mock 分支未被覆盖。
- **报告生成 P50 约 12 秒**：主要成本是最终报告的输出长度，不是检索；当前只统计总耗时，未拆首 token 与流式延迟。
- **报告工具序列仍可能不完整**：`fetch_external_data` 已能兜底切换报告提示词，字段不再缺失，但 `fill_context_for_report` 仍可能被模型跳过。
- **两种检索口径**：检索基线用 k=8 看召回上限；回答级用 k=3 复现生产链路，不能直接相减。
- **下一步（M2）**：QA 感知切分 + BM25/向量 RRF + 重排，用同一套 80 题前后对比，效果与延迟一起报；报告链路则走 M4 子图与流式。
