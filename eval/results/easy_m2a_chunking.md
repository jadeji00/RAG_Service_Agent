# 检索评测基线

- 生成时间：2026-11-13 01:05:02
- 向量库：`agent` / 145 chunks
- Embedding：`text-embedding-v4`，Chat：`qwen-max`
- 数据集：80 题（answerable 55 / negative 15 / report 10）
- 本次评测：55 道 answerable，检索上限 k=8

## 指标

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall@k（命中答案段落） | 67.27% | 87.27% | 94.55% | 98.18% |
| Span MRR@k | 67.27% | 76.67% | 78.48% | 79.09% |
| Source Recall@k（只命中文件） | 83.64% | 98.18% | 100.00% | 100.00% |

- 检索延迟：P50 135.16 ms / P95 428.64 ms / 均值 186.9 ms

> 每题只有一个 gold 答案集合，因此 Recall@k 等价于 hit rate；Span 指标要求文件名和答案关键词同时命中，Source 指标只要求文件名命中。

## 未命中或非首位命中的题

| ID | 问题 | 首次命中排名 | Top 来源 |
|---|---|---|---|
| qa-001 | 60㎡以下的小户型选扫地机器人，重点看什么？ | 2 | 选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>扫地机器人100问2.txt |
| qa-002 | 90㎡以上的户型选扫地机器人，要重点关注什么？ | 2 | 选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>维护保养.txt |
| qa-003 | 养宠家庭选扫地机器人，重点看哪些配置？ | 2 | 选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>维护保养.txt |
| qa-006 | 扫地机器人越障能力一般多少，门槛太高怎么办？ | 4 | 扫地机器人100问2.txt<br>选购指南.txt<br>扫拖一体机器人100问.txt<br>选购指南.txt<br>故障排除.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>故障排除.txt |
| qa-007 | 扫地机器人吸力多大够用？ | 4 | 选购指南.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt |
| qa-008 | 胶刷和毛刷怎么选？ | 4 | 扫拖一体机器人100问.txt<br>扫拖一体机器人100问.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>扫拖一体机器人100问.txt<br>扫地机器人100问2.txt<br>选购指南.txt |
| qa-009 | 复式或多楼层怎么选扫地机器人？ | 2 | 选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>扫拖一体机器人100问.txt<br>选购指南.txt<br>选购指南.txt<br>扫地机器人100问2.txt |
| qa-011 | 扫地机器人报 E01 错误怎么处理？ | 6 | 扫地机器人100问2.txt<br>故障排除.txt<br>维护保养.txt<br>扫拖一体机器人100问.txt<br>故障排除.txt<br>故障排除.txt<br>故障排除.txt<br>维护保养.txt |
| qa-012 | 扫地机器人报 E02 是什么原因？ | 3 | 扫地机器人100问2.txt<br>故障排除.txt<br>故障排除.txt<br>故障排除.txt<br>故障排除.txt<br>扫拖一体机器人100问.txt<br>维护保养.txt<br>故障排除.txt |
| qa-015 | 扫地机器人原地打转不前进怎么办？ | 2 | 扫地机器人100问2.txt<br>故障排除.txt<br>扫地机器人100问2.txt<br>故障排除.txt<br>扫拖一体机器人100问.txt<br>维护保养.txt<br>维护保养.txt<br>故障排除.txt |
| qa-016 | 扫地机器人无法开机或无法充电怎么办？ | 4 | 扫地机器人100问2.txt<br>故障排除.txt<br>维护保养.txt<br>故障排除.txt<br>故障排除.txt<br>扫拖一体机器人100问.txt<br>扫地机器人100问2.txt<br>维护保养.txt |
| qa-018 | 扫地机器人频繁掉线怎么办？ | 未命中 | 扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>故障排除.txt<br>扫拖一体机器人100问.txt<br>扫地机器人100问2.txt<br>维护保养.txt<br>维护保养.txt<br>故障排除.txt |
| qa-020 | 扫地机器人电池鼓包怎么办？ | 2 | 扫地机器人100问2.txt<br>故障排除.txt<br>故障排除.txt<br>维护保养.txt<br>扫拖一体机器人100问.txt<br>维护保养.txt<br>故障排除.txt<br>扫地机器人100问2.txt |
| qa-021 | 每天清扫后要做哪些维护？ | 6 | 维护保养.txt<br>维护保养.txt<br>维护保养.txt<br>维护保养.txt<br>扫拖一体机器人100问.txt<br>维护保养.txt<br>维护保养.txt<br>扫拖一体机器人100问.txt |
| qa-027 | 梅雨季使用扫地机器人要注意什么？ | 2 | 扫地机器人100问2.txt<br>维护保养.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>维护保养.txt<br>选购指南.txt<br>扫拖一体机器人100问.txt<br>扫地机器人100问2.txt |
| ... | 其余 3 题见 JSON | - | - |

## 复现

```bash
python eval/run_eval.py --dataset eval\dataset.jsonl --ks 1,3,5,8
```

> 当前题目多来自知识库原始问句，属于 easy split；hard split 见 `eval/dataset_hard.jsonl`。
