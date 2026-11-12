# min_score 阈值标定

- 生成时间：2026-11-13 01:38:38
- 数据集：eval/dataset.jsonl, eval/dataset_hard.jsonl
- 重排模型：`gte-rerank-v2`，检索 k=3
- 样本数：105

## 各组最高重排分分布

| 分组 | 样本 | 最小 | P5 | P50 | 最大 |
|---|---|---|---|---|---|
| answerable | 75 | 0.1673 | 0.1948 | 0.5273 | 0.951 |
| negative | 20 | 0.0061 | 0.0061 | 0.1473 | 0.4521 |
| report | 10 | 0.0801 | 0.0801 | 0.1398 | 0.1798 |

## 阈值建议

- 建议 `min_score = 0.1626`（分离边界 0.1673，两组可分时取间隔中点留余量）
- answerable 保留率：100.00%
- 未被路由拦下的 negative 拒答率：100.00%（样本 5 条）
- 已被服务范围路由拦下、不参与阈值计算的 negative：15 条
- 两组是否完全可分：是
- answerable 最低分 0.1673 / negative 最高分 0.1578

## 边界样本（分数最低的 answerable 与分数最高的 negative）

| 分组 | ID | 最高重排分 | 路由已拦 | 问题 |
|---|---|---|---|---|
| answerable | hard-qa-016 | 0.1673 | 否 | 家里有小孩，边刷和 App 分别要防什么？ |
| answerable | qa-008 | 0.1811 | 否 | 胶刷和毛刷怎么选？ |
| answerable | hard-qa-019 | 0.1835 | 否 | 3 厘米高的门槛，扫地机器人能直接过去吗？ |
| answerable | hard-qa-014 | 0.1841 | 否 | 滤网水洗后马上装回去，会有什么问题？ |
| answerable | qa-006 | 0.1948 | 否 | 扫地机器人越障能力一般多少，门槛太高怎么办？ |
| negative | hard-neg-001 | 0.1578 | 否 | 扫地机器人能除甲醛吗？ |
| negative | hard-neg-002 | 0.1487 | 否 | 它能自己爬楼梯上下楼吗？ |
| negative | hard-neg-004 | 0.1473 | 否 | 能帮我给手机无线充电吗？ |
| negative | hard-neg-003 | 0.1355 | 否 | 这台机器能当空气净化器用吗？ |
| negative | hard-neg-005 | 0.1261 | 否 | 它能检测家里的煤气泄漏吗？ |
| negative | neg-011 | 0.4521 | 是 | 洗衣机报 E01 错误怎么处理？ |
| negative | neg-015 | 0.1988 | 是 | 扫地机器人能做饭吗？ |
| negative | neg-002 | 0.1948 | 是 | 扫地机器人能帮我洗碗吗？ |

## 复现

```bash
python eval/calibrate_min_score.py
```
