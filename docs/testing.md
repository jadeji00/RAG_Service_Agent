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
| 检索纯函数单测 | `tests/test_eval_metrics.py` | 匹配规则、排名、MRR、分位数 | 否 | 毫秒级 |
| 判分纯函数单测 | `tests/test_judge.py` | JSON 解析、单位归一、报告字段核对、拒答信号 | 否 | 毫秒级 |
| 数据集单测 | `tests/test_dataset.py` | 题量分布、ID 唯一、gold 与当前数据一致 | 否 | 毫秒级 |
| 检索端到端 | `eval/run_eval.py` | 真实 embedding + Chroma 的 Recall/MRR/延迟 | 是（仅 embedding） | 约 10 秒 |
| 回答级端到端 | `eval/run_answer_eval.py` | 55 问答 + 15 拒答 + 10 报告；correctness / faithfulness / 拒答 / 字段完整性 | 是 | 约 8 分钟 |

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

### 3.2 数据与检索单测（10 个）

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

### 3.3 判分单测（12 个）

| 测试 | 验证点 | 为什么重要 |
|---|---|---|
| `test_extract_json_from_code_block` | 兼容 ```json 代码块 | 模型常把 JSON 包进 Markdown |
| `test_extract_json_from_prose` | 兼容前后解释文字 | 判分器要能提取真正对象 |
| `test_extract_json_invalid` | 非法输出返回 None | 不把错误输出当 0 分 |
| `test_compact_removes_markdown_and_unifies_units` | 去 Markdown、统一 `%`/`㎡` | 报告排版不同不等于事实缺失 |
| `test_complete_answer_covers_all_fields` | 完整报告四字段全过 | 核对逻辑的正例 |
| `test_paraphrased_units_still_match` | 「1 个百分点」「主刷剩余寿命」 | 防止改写被误判缺字段 |
| `test_missing_comparison_is_detected` | 缺对比字段能抓出来 | 核对逻辑的负例 |
| `test_positive_change_without_plus_sign` | CSV `+1%` 对报告「提升 1%」 | 正负号不能导致漏判 |
| `test_baseline_month_without_comparison` | 「基线数据，无对比」 | 基线月没有变化是合理的 |
| `test_refusal_phrases` | 拒答/无法回答/不在范围 | 规则信号用于诊断 |
| `test_normal_answer_is_not_refusal` | 正常回答不误报 | 避免规则把业务回答当拒答 |
| `test_load_records_indexes_user_and_month` | CSV 按 `(用户ID, 月份)` 建索引 | 报告题取数不能串行 |

### 3.4 检索端到端

对 55 道 `answerable` 题：

1. 用题目原文调用 `RagSummarizeService.retrieve(question, k=8)`；
2. 取回 Top-8 片段，按 `gold_sources` + `gold_terms`/`gold_sections` 判断相关性；
3. 记录首次命中排名、Top 来源、单题延迟；
4. 汇总 Span Recall@k、Span MRR@k、Source Recall@k 和 P50/P95。

**Span 与 Source 分开报**的原因：Source 高但 Span 低，说明文件找对了、片段没切对；两者都低，才是召回或 embedding 的问题。

### 3.5 回答级端到端

三条链路分开跑，避免用同一种「满意」口径糊在一起：

| 类型 | 实际走的链路 | 怎么判 |
|---|---|---|
| answerable（55） | `RagSummarizeService.retrieve(k=3)` + `answer()` | LLM judge 对照 gold answer 打 correctness / faithfulness |
| negative（15） | 同一条 RAG 链路 | LLM judge 判是否拒答；规则关键词只做诊断 |
| report（10） | 完整 `ReactAgent`，唯一 `thread_id`，`.invoke()` 取最终 `AIMessage` | 确定性核对字段 + LLM judge 判忠实度与建议 |

几个关键取舍：

- **问答/拒答只评 RAG 子链路**：避免 Agent 的工具选择噪声掩盖生成问题；报告题才评完整控制流。
- **报告题不走 `execute_stream`**：当前 `stream_mode="values"` 会把工具消息和中间思考一起吐出（见 P3），评测直接取最后一条无 `tool_calls` 的 `AIMessage`。
- **报告字段不交给模型判**：CSV 是事实源，标签 + 数值 + 单位归一能稳定抓漏字段；模型只判「数字是否忠实、建议是否可执行」。
- **判分器强制 JSON、失败重试一次**：解析失败记为 `judge_error`，不静默当 0 分。

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

# 5) 回答级冒烟：每类前 2 题 / 报告 1 题
python eval/run_answer_eval.py --types answerable,negative --limit 2
python eval/run_answer_eval.py --types report --limit 1

# 6) 完整回答级基线
python eval/run_answer_eval.py
```

结果写入：

- `eval/results/baseline.json`：检索聚合指标 + 逐题排名 + Top 来源；
- `eval/results/baseline.md`：检索人读版，含失败题；
- `eval/results/answer_baseline.json`：回答级聚合指标 + 逐题回答、判分理由、报告事实核对；
- `eval/results/answer_baseline.md`：回答级人读版，含失败样例与延迟。

## 5. 本次结果

### 5.1 测试

| 项 | 结果 |
|---|---|
| 数据集校验 | `VALIDATION_OK rows=80`，0 个错误 |
| 单元测试 | 22 个全部通过，约 0.006 秒 |
| 向量库 | 66 个向量、66 个唯一 ID、66 段唯一内容、5 个来源/md5 组合 |
| 回答级评测 | 80 题全部执行，生成与判分错误 0 |
| Python 编译 | `py_compile` 通过 |

### 5.2 检索基线

55 道可答题，66 chunks，`text-embedding-v4`，Top-8：

| 指标 | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| Span Recall | 65.45% | 94.55% | 98.18% | 98.18% |
| Span MRR | 65.45% | 78.79% | 79.52% | 79.52% |
| Source Recall | 85.45% | 100.00% | 100.00% | 100.00% |

延迟：P50 126.05 ms / P95 296.50 ms。

### 5.3 一个检索失败

`qa-042`「拖布多久清洗一次？」是唯一在 Top-8 完全未命中的题：

- 正确答案在 `扫拖一体机器人100问.txt` 的 Q3；
- 当前 200 字硬切把 Q1、Q2、Q3 挤进同一个 chunk，主题被稀释；
- Top-8 召回的是 Q12（拖布能不能机洗）、Q13（多久换一次）等相邻主题；
- 这不是标签错，而是**固定字符切分混合问答对**的实证，直接对应 M2 的 QA 感知切分。

### 5.4 回答级基线

`qwen-max` 生成与判分，RAG 检 Top-3，报告走完整 Agent；80 题全部完成：

| 类型 | 样本 | 通过率 | 关键指标 |
|---|---|---|---|
| answerable | 55 | 98.18% | 完全正确 81.82%；完全有据 98.18%；幻觉 0.00% |
| negative | 15 | 86.67% | 拒答正确率 86.67% |
| report | 10 | 50.00% | 字段完整 50.00%；忠实 100.00% |
| **合计** | **80** | **90.00%** | - |

知识问答：correctness=2 有 45 题、=1 有 9 题、=0 有 1 题；faithfulness=2 有 54 题、=1 有 1 题、=0 有 0 题。关键词命中率 82.29% 仅作诊断，不计分。

报告：字段覆盖为 特征 6/10、清洁效率 10/10、耗材 9/10、对比 8/10；平均字段覆盖率 82.50%，平均事实片段覆盖率 73.00%；完全有据率 90%，可执行建议率 100%，报告工具序列正确率 50%，`fetch` 参数正确率 100%。

延迟（生成 P50 / P95）：answerable 1614 / 2517 ms，negative 1053 / 3607 ms，report 15321 / 18032 ms；判分 P50 均在 1.0–1.4 秒。

### 5.5 回答级失败

| 题号 | 现象 | 根因 |
|---|---|---|
| `qa-042` | 回答拖布更换周期，没答“每次拖地后清洗” | Top-3 检索未命中 Q3，与 5.3 同源 |
| `neg-004` | 直接写出 Python 爬虫代码 | 越界任务没有明确拒绝，提示词缺边界示例 |
| `neg-009` | 追问要翻译的文本，准备执行翻译 | 同上，把超范围任务当成了待补全请求 |
| `rep-002/003/006/007/008` | 缺 `特征` 字段 | 5/10 漏调 `fill_context_for_report`，报告提示词没有切换 |
| `rep-004` | 缺 `耗材` 字段 | 工具调用完整，但报告正文漏写该字段 |
| `rep-003` | 编造滤网状态变化，faithfulness=1 | 报告未逐项对齐 CSV |

其中报告流程问题已记为 [P7](pitfalls.md)；拒答问题来自提示词边界，报告字段问题一半来自流程、一半来自字段覆盖约束。

## 6. 这些测试有什么用

1. **改动前先知道现状。** 检索能说“Top-1 65.45%，Top-3 94.55%”，回答能说“通过率 98.18%、幻觉 0%”，而不是“感觉挺准”。
2. **改动后能证明是否变好。** M2 的切分、混合召回、重排都跑同一套 55 题；报告流程修复也跑同一套 10 题，成对报效果和延迟。
3. **失败能定位到层。** Source 100% 但 Span 98.18%，问题在片段级；回答通过率 98.18% 但报告只有 50%，问题在 Agent 流程；报告忠实 100% 但字段完整 50%，问题在覆盖而不是编造。
4. **面试有证据。** 题量、标注口径、指标定义、复现命令、失败案例都能当场打开。
5. **防止回归。** 改配置或换模型后，先跑 `--validate-only` 和 22 个单测，再跑基线，避免“修一个坏三个”。

## 7. 设计取舍与边界

- **不用 pytest**：当前用标准库 `unittest`，零额外依赖；后续测试变多再引入。
- **单测不调模型**：保证快、稳定、可在无网络时跑；真实检索和回答放到单独的端到端命令。
- **gold 不绑 chunk id**：切分策略一变 id 就失效；改用来源 + 章节/关键词。
- **当前是 easy split**：题目多来自知识库原问句，分数偏高，不能当作真实用户分布。
- **同源判分**：生成与判分都是 `qwen-max`，存在自偏好；已用 gold answer、明确 rubric、`temperature=0` 降低，但未做人工双标。
- **粗粒度 rubric + 单次运行**：correctness / faithfulness 是 0/1/2，同题跨运行可能波动 1 分；尚未做 pass^k。
- **报告字段是词面下界**：标签 + 数值 + 单位归一能抓漏字段，但不判断语义等价；方向与编造交给 LLM judge。
- **两种检索口径**：检索基线用 k=8 看召回上限，回答级用 k=3 复现生产链路，两个数字不能直接相减。
- **报告题都显式给 ID/月份**：`get_user_id` / `get_current_month` 的随机 mock 分支未被覆盖。

## 8. 下一步

1. 修 P7：让报告流程强制经过 `fill_context_for_report`，或在评测前确定性地补齐上下文；
2. 增加 hard split（改写问法、跨节组合），降低 easy split 带来的虚高；
3. 同题重复 5 次，报 pass^5，区分“真会”与“碰巧一次对”；
4. 把 `--validate-only` + 22 个单测接进 CI，基线作为人工回归；
5. M2 检索优化（QA 切分、混合召回、重排）用同一套题对比效果与延迟。
