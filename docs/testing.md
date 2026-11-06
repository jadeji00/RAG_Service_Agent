# 测试与评测说明

> 测试要回答三个问题：**评测集和标签有没有错？指标算得对不对？系统现在到底是什么水平？**
>
> 这份文档说明当前测了什么、怎么测、结果如何、为什么这样设计，以及哪些还没测。

## 1. 为什么要测试

1. **评测集是尺子，尺子错了会带偏优化。** 如果 gold 标错，调切分、换 embedding、加重排都可能朝着错误方向走；所以先把数据校验做成自动化。
2. **RAG 失败要能定位到层。** 是文件没召回、片段没命中，还是答案生成了但没引用？不分层就只会说“感觉不太准”。
3. **改检索必须有回归基线。** M2 要做 QA 切分、混合召回和重排，没有同一套题和同一套指标，就无法证明改动是进步还是退步。
4. **面试要报数字，数字要能复现。** 题量、标注方式、指标定义、运行命令、失败案例，缺一项都会被追问穿。

## 2. 测试分层

| 层 | 文件 | 测什么 | 是否调用模型 | 速度 |
|---|---|---|---|---|
| 数据契约 | `eval/run_eval.py --validate-only` | 80 题 schema、ID、gold 标签、报告数据 | 否 | 毫秒级 |
| 纯函数单测 | `tests/test_eval_metrics.py` | 匹配规则、排名、MRR、分位数 | 否 | 毫秒级 |
| 数据集单测 | `tests/test_dataset.py` | 题量分布、ID 唯一、gold 与当前数据一致 | 否 | 毫秒级 |
| 检索端到端 | `eval/run_eval.py` | 真实 embedding + Chroma 的 Recall/MRR/延迟 | 是（仅 embedding） | 约 10 秒 |
| 回答级评测 | 尚未实现 | 拒答、faithfulness、报告字段完整性 | 是 | 待定 |

**为什么把单测和端到端分开**：单测不依赖 API Key、网络和向量库，提交前可以随时跑；端到端会调用 embedding，结果受数据和模型影响，单独作为基线命令执行。

## 3. 具体测了什么

### 3.1 数据集契约（`validate_rows`）

| 检查项 | 规则 | 失败示例 |
|---|---|---|
| 题型 | 只能是 `answerable / negative / report` | 拼错 `answerble` |
| ID | 全库唯一 | 两题都写 `qa-001` |
| 问题 | 非空且不重复 | 复制粘贴后忘改 |
| gold_answer | 非空 | 只写标签没写答案 |
| 来源文件 | `gold_sources` 必须存在 | 改名 `选购指南.txt` 后没同步 |
| 章节/题号 | 必须能在某个 gold 来源中找到 | 数据改版后 `Q3` 消失 |
| 关键词 | 必须能在某个 gold 来源中找到 | 写了原文没有的“高温烘烤” |
| 多来源题 | 章节/关键词命中任意一个来源即可 | 把两个来源做笛卡尔积误报 |
| 负样本 | `expected` 必须是 `refuse` | 写成 `answer` |
| 报告题 | `user_id + month` 必须存在于 `records.csv` | 选了不存在的 `1011/2026-12` |
| 报告字段 | `expected_fields` 必须是 CSV 真实列 | 写了 `总结` 但 CSV 没这列 |

这一步是**防止“假失败”**：如果标签本身错了，跑出来的低分不能用来指导优化。

### 3.2 纯函数单测（10 个）

| 测试 | 验证点 | 为什么重要 |
|---|---|---|
| `test_normalize_removes_whitespace` | 换行/空格不影响匹配 | 中文标题常被切分器换行 |
| `test_source_basename_handles_windows_path` | Windows 绝对路径取文件名 | metadata 里是 `D:\...`，gold 里是文件名 |
| `test_term_match_requires_gold_source` | 关键词命中但来源不对，判不相关 | 防止别的文件“碰巧”出现同词 |
| `test_section_fallback_when_no_terms` | 无关键词时按章节/题号匹配 | Q&A 文件按 Q 号标注 |
| `test_rank_is_one_based` | 排名从 1 开始 | MRR 的分母不能从 0 开始 |
| `test_summarize_ranks` | Recall@k 与 MRR@k 的数学 | 指标算错，后面全错 |
| `test_percentile` | P50/P95 取最近秩 | 延迟统计不引入 numpy |
| `test_dataset_size_and_distribution` | 80 题 = 55/15/10 | 防止误删或漏加题 |
| `test_all_ids_unique` | ID 无重复 | 逐题结果能对上 |
| `test_gold_labels_match_current_data` | 80 题 gold 全部可追溯 | 数据一改就能发现标签过期 |

### 3.3 检索端到端

对 55 道 `answerable` 题：

1. 用题目原文调用 `RagSummarizeService.retrieve(question, k=8)`；
2. 取回 Top-8 片段，按 `gold_sources` + `gold_terms`/`gold_sections` 判断相关性；
3. 记录首次命中排名、Top 来源、单题延迟；
4. 汇总 Span Recall@k、Span MRR@k、Source Recall@k 和 P50/P95。

**Span 与 Source 分开报**的原因：Source 高但 Span 低，说明文件找对了、片段没切对；两者都低，才是召回或 embedding 的问题。

## 4. 怎么运行

```bash
# 1) 只校验数据集，不调用模型
python eval/run_eval.py --validate-only

# 2) 纯函数与数据集单测，不需要 API Key
python -m unittest discover -s tests -v

# 3) 冒烟：只跑前 10 道可答题
python eval/run_eval.py --limit 10

# 4) 完整检索基线
python eval/run_eval.py --ks 1,3,5,8
```

结果写入：

- `eval/results/baseline.json`：聚合指标 + 逐题排名 + Top 来源；
- `eval/results/baseline.md`：人读版，含失败题。

## 5. 本次结果

### 5.1 测试

| 项 | 结果 |
|---|---|
| 数据集校验 | `VALIDATION_OK rows=80`，0 个错误 |
| 单元测试 | 10 个全部通过，约 0.004 秒 |
| 向量库 | 66 个向量、66 个唯一 ID、66 段唯一内容、5 个来源/md5 组合 |
| Python 编译 | `py_compile` 通过 |

### 5.2 检索基线

55 道可答题，66 chunks，`text-embedding-v4`，Top-8：

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall | 65.45% | 94.55% | 98.18% | 98.18% |
| Span MRR | 65.45% | 78.79% | 79.52% | 79.52% |
| Source Recall | 85.45% | 100.00% | 100.00% | 100.00% |

延迟：P50 126.05 ms / P95 296.50 ms。

### 5.3 一个真实失败

`qa-042`「拖布多久清洗一次？」是唯一在 Top-8 完全未命中的题：

- 正确答案在 `扫拖一体机器人100问.txt` 的 Q3；
- 当前 200 字硬切把 Q1、Q2、Q3 挤进同一个 chunk，主题被稀释；
- Top-8 召回的是 Q12（拖布能不能机洗）、Q13（多久换一次）等相邻主题；
- 这不是标签错，而是**固定字符切分混合问答对**的实证，直接对应 M2 的 QA 感知切分。

## 6. 这些测试有什么用

1. **改动前先知道现状。** 现在能说“Top-1 65.45%，Top-3 94.55%”，而不是“感觉挺准”。
2. **改动后能证明是否变好。** M2 的切分、混合召回、重排都跑同一套 55 题，成对报效果和延迟。
3. **失败能定位。** Source 100% 但 Span 98.18%，说明问题在片段级而不是文件级；`qa-042` 直接指向切分策略。
4. **面试有证据。** 题量、标注口径、指标定义、复现命令、失败案例都能当场打开。
5. **防止回归。** 改配置或换模型后，先跑 `--validate-only` 和单测，再跑基线，避免“修一个坏三个”。

## 7. 设计取舍与边界

- **不用 pytest**：当前用标准库 `unittest`，零额外依赖；后续测试变多再引入。
- **单测不调模型**：保证快、稳定、可在无网络时跑；真实检索放到单独的端到端命令。
- **gold 不绑 chunk id**：切分策略一变 id 就失效；改用来源 + 章节/关键词。
- **当前是 easy split**：题目多来自知识库原问句，分数偏高，不能当作真实用户分布。
- **只评了检索**：负样本拒答、报告字段完整性、回答 faithfulness 尚未计分。
- **没有 pass^k**：稳定性要同题重复跑多次，当前未做。

## 8. 下一步

1. `eval/judge.py`：对回答做 rubric 评分，补拒答与 faithfulness；
2. 报告题评字段完整性，负样本评“是否拒答 + 是否引导回业务”；
3. 增加 hard split（改写问法、跨节组合）；
4. 同题重复 5 次，报 pass^5；
5. 把 `--validate-only` + 单测接进 CI，基线作为人工回归。
