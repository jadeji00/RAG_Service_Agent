# 检索评测基线

- 生成时间：2026-11-13 01:43:01
- 向量库：`agent` / 145 chunks
- Embedding：`text-embedding-v4`，Chat：`qwen-max`
- 数据集：80 题（answerable 55 / negative 15 / report 10）
- 本次评测：55 道 answerable，检索上限 k=8

## 指标

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall@k（命中答案段落） | 96.36% | 98.18% | 100.00% | 100.00% |
| Span MRR@k | 96.36% | 96.97% | 97.42% | 97.42% |
| Source Recall@k（只命中文件） | 98.18% | 100.00% | 100.00% | 100.00% |

- 检索延迟：P50 261.47 ms / P95 607.55 ms / 均值 326.92 ms

> 每题只有一个 gold 答案集合，因此 Recall@k 等价于 hit rate；Span 指标要求文件名和答案关键词同时命中，Source 指标只要求文件名命中。

## 未命中或非首位命中的题

| ID | 问题 | 首次命中排名 | Top 来源 |
|---|---|---|---|
| qa-006 | 扫地机器人越障能力一般多少，门槛太高怎么办？ | 3 | 扫地机器人100问2.txt<br>扫拖一体机器人100问.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>选购指南.txt |
| qa-023 | 拖布怎么清洗和晾干？ | 4 | 扫拖一体机器人100问.txt<br>故障排除.txt<br>扫拖一体机器人100问.txt<br>维护保养.txt<br>扫拖一体机器人100问.txt<br>维护保养.txt<br>扫地机器人100问2.txt<br>扫拖一体机器人100问.txt |

## 复现

```bash
python eval/run_eval.py --dataset eval\dataset.jsonl --ks 1,3,5,8
```

> 当前题目多来自知识库原始问句，属于 easy split；hard split 见 `eval/dataset_hard.jsonl`。
