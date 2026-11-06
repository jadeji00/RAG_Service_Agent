# 评测

> 目标：把「答得对不对」变成可复现的数字。当前已完成**检索基线**；回答级评测（faithfulness / 拒答）与 hard split 尚未加入。
>
> 测试分层、10 个用例与结果说明见 [docs/testing.md](../docs/testing.md)。

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

## 4. 复现

```bash
# 只校验数据和 gold 标签，不调用模型
python eval/run_eval.py --validate-only

# 先跑 10 题冒烟
python eval/run_eval.py --limit 10

# 完整基线
python eval/run_eval.py --ks 1,3,5,8

# 纯函数单测（不需要 API Key）
python -m unittest discover -s tests
```

输出：

- `eval/results/baseline.json`：聚合指标 + 逐题排名 + Top 来源；
- `eval/results/baseline.md`：表格版，失败题可直接用于分析。

## 5. 新增题目

1. 在 `eval/dataset.jsonl` 追加一行，`id` 不重复；
2. 填 `gold_sources` + `gold_sections`/`gold_terms`，**不要写 chunk id**；
3. 跑 `--validate-only`，它会检查每个章节/关键词是否仍能在对应源文件中找到；
4. 跑评测，确认新题不会因为标签错误制造假失败。

## 6. 限制与下一步

- **easy split**：多数题目直接来自知识库原问句，分数偏高；下一步补改写问法、跨节组合的 hard split。
- **只评检索**：负样本与报告题尚未计分；回答级评测需要 `judge.py`（rubric + 引用校验 + 拒答判定）。
- **单 gold 集合**：真实场景可能多个片段都正确，后续可引入多 gold 的 Recall 计算。
- **没有 pass^k**：稳定性要同题重复跑多次，当前尚未做。
- 报告题 gold 直接来自 mock CSV，换数据后需重新生成对应题。
