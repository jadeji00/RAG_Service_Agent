# 检索评测基线

- 生成时间：2026-11-13 01:43:09
- 向量库：`agent` / 145 chunks
- Embedding：`text-embedding-v4`，Chat：`qwen-max`
- 数据集：25 题（answerable 20 / negative 5 / report 0）
- 本次评测：20 道 answerable，检索上限 k=8

## 指标

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall@k（命中答案段落） | 80.00% | 100.00% | 100.00% | 100.00% |
| Span MRR@k | 80.00% | 89.17% | 89.17% | 89.17% |
| Source Recall@k（只命中文件） | 85.00% | 100.00% | 100.00% | 100.00% |

- 检索延迟：P50 232.18 ms / P95 378.73 ms / 均值 297.1 ms

> 每题只有一个 gold 答案集合，因此 Recall@k 等价于 hit rate；Span 指标要求文件名和答案关键词同时命中，Source 指标只要求文件名命中。

## 未命中或非首位命中的题

| ID | 问题 | 首次命中排名 | Top 来源 |
|---|---|---|---|
| hard-qa-001 | 家里是复式，机器每次换层都要重新建图，选的时候该看什么？ | 2 | 扫地机器人100问2.txt<br>选购指南.txt<br>选购指南.txt<br>维护保养.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>故障排除.txt |
| hard-qa-003 | 宠物换毛季主刷总缠毛，选刷子和日常清理各要注意什么？ | 3 | 扫地机器人100问2.txt<br>故障排除.txt<br>选购指南.txt<br>维护保养.txt<br>维护保养.txt<br>扫地机器人100问2.txt |
| hard-qa-009 | 自动集尘的尘袋倒空后继续用，会影响什么？ | 2 | 扫拖一体机器人100问.txt<br>扫拖一体机器人100问.txt |
| hard-qa-020 | 机器进水后，能不能先开机试试还能不能用？ | 2 | 扫地机器人100问2.txt<br>故障排除.txt<br>故障排除.txt<br>扫地机器人100问2.txt |

## 复现

```bash
python eval/run_eval.py --dataset eval\dataset_hard.jsonl --ks 1,3,5,8
```

> 本文件是 hard split：题目经过改写或跨章节组合，不照抄知识库原问句；和 easy split 使用同一套指标，分数下降是预期结果。
