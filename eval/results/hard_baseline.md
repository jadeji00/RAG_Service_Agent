# 检索评测基线

- 生成时间：2026-11-07 13:31:01
- 向量库：`agent` / 66 chunks
- Embedding：`text-embedding-v4`，Chat：`qwen-max`
- 数据集：25 题（answerable 20 / negative 5 / report 0）
- 本次评测：20 道 answerable，检索上限 k=8

## 指标

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall@k（命中答案段落） | 55.00% | 80.00% | 90.00% | 90.00% |
| Span MRR@k | 55.00% | 66.67% | 69.17% | 69.17% |
| Source Recall@k（只命中文件） | 75.00% | 100.00% | 100.00% | 100.00% |

- 检索延迟：P50 136.29 ms / P95 261.47 ms / 均值 161.57 ms

> 每题只有一个 gold 答案集合，因此 Recall@k 等价于 hit rate；Span 指标要求文件名和答案关键词同时命中，Source 指标只要求文件名命中。

## 未命中或非首位命中的题

| ID | 问题 | 首次命中排名 | Top 来源 |
|---|---|---|---|
| hard-qa-001 | 家里是复式，机器每次换层都要重新建图，选的时候该看什么？ | 2 | 扫地机器人100问2.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>维护保养.txt<br>选购指南.txt |
| hard-qa-003 | 宠物换毛季主刷总缠毛，选刷子和日常清理各要注意什么？ | 4 | 维护保养.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>维护保养.txt<br>扫地机器人100问2.txt<br>维护保养.txt<br>故障排除.txt |
| hard-qa-004 | 家里是黑色地毯，机器报 E05，第一步该擦哪个传感器？ | 3 | 故障排除.txt<br>故障排除.txt<br>故障排除.txt<br>故障排除.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>故障排除.txt |
| hard-qa-007 | 家里老人不想用手机 App，选机型要看哪些操作方式？ | 未命中 | 扫地机器人100问2.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>故障排除.txt |
| hard-qa-008 | 面粉撒了一地，能直接让扫地机器人吸吗？ | 2 | 扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>选购指南.txt |
| hard-qa-012 | 厨房地面有油，日常能不能直接交给机器人拖？ | 4 | 选购指南.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>扫拖一体机器人100问.txt<br>扫拖一体机器人100问.txt<br>扫拖一体机器人100问.txt<br>扫拖一体机器人100问.txt<br>故障排除.txt |
| hard-qa-014 | 滤网水洗后马上装回去，会有什么问题？ | 2 | 扫拖一体机器人100问.txt<br>维护保养.txt<br>维护保养.txt<br>扫地机器人100问2.txt<br>扫拖一体机器人100问.txt<br>维护保养.txt<br>选购指南.txt<br>扫拖一体机器人100问.txt |
| hard-qa-016 | 家里有小孩，边刷和 App 分别要防什么？ | 未命中 | 扫拖一体机器人100问.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>选购指南.txt<br>选购指南.txt<br>选购指南.txt<br>扫拖一体机器人100问.txt |
| hard-qa-019 | 3 厘米高的门槛，扫地机器人能直接过去吗？ | 2 | 扫地机器人100问2.txt<br>扫拖一体机器人100问.txt<br>选购指南.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt<br>扫地机器人100问2.txt |

## 复现

```bash
python eval/run_eval.py --dataset eval\dataset_hard.jsonl --ks 1,3,5,8
```

> 本文件是 hard split：题目经过改写或跨章节组合，不照抄知识库原问句；和 easy split 使用同一套指标，分数下降是预期结果。
