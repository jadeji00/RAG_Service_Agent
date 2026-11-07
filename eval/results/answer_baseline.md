# 回答级评测基线

- 生成时间：2026-11-07 13:36:55
- 数据集：`eval\dataset.jsonl`（answerable 55 / negative 15 / report 10）
- 生成模型：`qwen-max`；判分模型：`qwen-max`
- RAG 检索：Top-3；向量库：`agent` / 66 chunks
- 报告链路：确定性报告子流程（路由 → `fetch_external_data` → 一次写作；未命中路由时回退 ReAct）
- 重复次数：1（>1 时计算 pass^k）

## 总览

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| answerable | 55 | 98.18% | 完全正确 80.00%；完全有据 98.18%；幻觉 0.00% |
| negative | 15 | 86.67% | 拒答正确率 86.67% |
| report | 10 | 100.00% | 字段完整 100.00%；忠实 100.00% |
| **合计** | **80** | **96.25%** | - |

> 通过口径：answerable = correctness ≥ 1 且 faithfulness ≥ 1；negative = 明确拒答；report = 期望字段全部覆盖且 faithfulness ≥ 1。

## 知识问答

| 指标 | 数值 |
|---|---|
| 参与判分 | 55 / 55 |
| correctness=2（完全正确） | 44 题 |
| correctness=1（部分正确） | 10 题 |
| correctness=0（错误） | 1 题 |
| 平均 correctness | 1.78 |
| faithfulness=2（全部有据） | 54 题 |
| faithfulness=1（少量推断） | 1 题 |
| faithfulness=0（编造） | 0 题 |
| 关键词命中率（诊断，不计分） | 83.33% |

## 拒答

| 指标 | 数值 |
|---|---|
| 参与判分 | 15 / 15 |
| 拒答正确率（LLM judge） | 86.67% |
| 规则拒答信号命中率（诊断） | 66.67% |

## 报告

| 字段 | 覆盖题数 | 覆盖率 |
|---|---|---|
| 特征 | 10 / 10 | 100.00% |
| 清洁效率 | 10 / 10 | 100.00% |
| 耗材 | 10 / 10 | 100.00% |
| 对比 | 10 / 10 | 100.00% |

| 指标 | 数值 |
|---|---|
| 字段完整率 | 100.00% |
| 平均字段覆盖率 | 100.00% |
| 平均事实片段覆盖率 | 99.00% |
| 忠实率（faithfulness ≥ 1） | 100.00% |
| 完全有据率（faithfulness = 2） | 70.00% |
| 可执行建议率 | 100.00% |
| 报告子流程执行率（路由 → fetch） | 100.00% |
| fetch 参数正确率 | 100.00% |

## 延迟

| 链路 | 生成 P50 | 生成 P95 | 判分 P50 | 判分 P95 |
|---|---|---|---|---|
| answerable | 1677 ms | 3269 ms | 1422 ms | 1962 ms |
| negative | 1071 ms | 2207 ms | 1041 ms | 1292 ms |
| report | 7251 ms | 10224 ms | 1189 ms | 1731 ms |

## 失败样例

| ID | 类型 | 问题 | 判分 | 原因 / 回答摘要 |
|---|---|---|---|---|
| qa-042 | answerable | 拖布多久清洗一次？ | correctness=0, faithfulness=1 | 未回答清洗频率，与标准答案核心要点不符；大部分内容基于检索上下文。 |
| neg-004 | negative | 帮我写一段 Python 爬虫代码。 | refusal=0 | 模型尝试完成了编写Python爬虫代码的任务，超出了服务范围。 |
| neg-009 | negative | 帮我把这段英文翻译成中文。 | refusal=0 | 模型尝试完成翻译任务，而未说明该请求不在服务范围内。 |

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
