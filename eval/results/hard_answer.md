# 回答级评测基线

- 生成时间：2026-11-07 13:29:46
- 数据集：`eval\dataset_hard.jsonl`（answerable 20 / negative 5 / report 0）
- 生成模型：`qwen-max`；判分模型：`qwen-max`
- RAG 检索：Top-3；向量库：`agent` / 66 chunks
- 报告链路：确定性报告子流程（路由 → `fetch_external_data` → 一次写作；未命中路由时回退 ReAct）
- 重复次数：1（>1 时计算 pass^k）

## 总览

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| answerable | 20 | 95.00% | 完全正确 75.00%；完全有据 90.00%；幻觉 0.00% |
| negative | 5 | 100.00% | 拒答正确率 100.00% |
| report | 0 | - | 字段完整 -；忠实 - |
| **合计** | **25** | **96.00%** | - |

> 通过口径：answerable = correctness ≥ 1 且 faithfulness ≥ 1；negative = 明确拒答；report = 期望字段全部覆盖且 faithfulness ≥ 1。

## 知识问答

| 指标 | 数值 |
|---|---|
| 参与判分 | 20 / 20 |
| correctness=2（完全正确） | 15 题 |
| correctness=1（部分正确） | 4 题 |
| correctness=0（错误） | 1 题 |
| 平均 correctness | 1.70 |
| faithfulness=2（全部有据） | 18 题 |
| faithfulness=1（少量推断） | 2 题 |
| faithfulness=0（编造） | 0 题 |
| 关键词命中率（诊断，不计分） | 67.50% |

## 拒答

| 指标 | 数值 |
|---|---|
| 参与判分 | 5 / 5 |
| 拒答正确率（LLM judge） | 100.00% |
| 规则拒答信号命中率（诊断） | 100.00% |

## 报告

| 字段 | 覆盖题数 | 覆盖率 |
|---|---|---|

| 指标 | 数值 |
|---|---|
| 字段完整率 | - |
| 平均字段覆盖率 | - |
| 平均事实片段覆盖率 | - |
| 忠实率（faithfulness ≥ 1） | - |
| 完全有据率（faithfulness = 2） | - |
| 可执行建议率 | - |
| 报告子流程执行率（路由 → fetch） | - |
| fetch 参数正确率 | - |

## 延迟

| 链路 | 生成 P50 | 生成 P95 | 判分 P50 | 判分 P95 |
|---|---|---|---|---|
| answerable | 1706 ms | 3374 ms | 1447 ms | 1675 ms |
| negative | 976 ms | 2230 ms | 1187 ms | 1422 ms |
| report | - ms | - ms | - ms | - ms |

## 失败样例

| ID | 类型 | 问题 | 判分 | 原因 / 回答摘要 |
|---|---|---|---|---|
| hard-qa-016 | answerable | 家里有小孩，边刷和 App 分别要防什么？ | correctness=0, faithfulness=1 | 模型回答未提及儿童锁和边刷防护，与标准答案不符；信息主要来源于检索上下文。 |

## 复现

```bash
python eval/run_answer_eval.py --types answerable,negative --limit 2
python eval/run_answer_eval.py --types report --limit 1
python eval/run_answer_eval.py
python -m unittest discover -s tests
```

## 限制

- 生成与判分默认同为 `qwen-max`，存在自偏好；已用 gold answer、明确 rubric 和低 temperature 降低，但未做人工双标与一致性系数。
- correctness / faithfulness 是 0/1/2 粗粒度 rubric，同一条回答在不同运行间可能有 1 分波动；本次 repeat=1，未做 pass^k。
- gold answer 是单一参考答案，语义正确但表述不同的回答可能被判为 partial。
- 报告题都显式给出 user_id / month；`get_user_id` / `get_current_month` 的随机 mock分支未被覆盖。
- 报告字段覆盖是词面匹配（标签 + 数值），不是语义等价；语义忠实度由 LLM judge 判断。
- 报告请求正常走确定性子流程，工具序列由代码保证；`pipeline_ok_rate` 下降只可能来自路由未命中或子流程异常回退。
- 报告生成耗时主要来自最终输出长度；当前只统计总耗时，未拆首 token 与流式延迟。
- answerable / negative 走 `RagSummarizeService` 的 Top-3 生成链路；检索基线用 k=8，两者口径不同，不能直接相减。
