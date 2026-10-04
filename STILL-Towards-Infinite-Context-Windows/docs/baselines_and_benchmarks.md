# 基线与四个 benchmark 的准备状态

本轮先比较完整 KV、STILL 风格蒸馏、完整上下文 OPD。证据教师训练留到
这三个基线建立后。默认冻结同一 Qwen3-4B，仅学习每层 512 个缓存位置的压缩器。
后续已补齐四种方法的统一 benchmark 接口；准备、磁盘权重加载与运行命令见
[benchmark suite](benchmark_suite.md)。下方基线定义和早期三组冒烟结果保留。

## 三个基线的定义

| 方法 / CLI 名称 | 训练轨迹 | 教师 | 损失 | 评测时如何回答 |
| --- | --- | --- | --- | --- |
| Qwen3-4B / `full_context` | 不训练 | 无 | 无 | 完整文档的未压缩 KV |
| STILL 风格 / `still` | 完整教师预先生成的固定回答 | 完整文档 | forward KL | 问题无关的文档压缩 KV |
| 普通 KV OPD / `full` | 当前压缩学生采样的回答 | 完整文档 | 等权 JSD | 同样的文档压缩 KV |

`still` 是**原始蒸馏目标在 QASPER 上的适配基线**，不是论文发布 checkpoint
的原样复现。[STILL 论文](https://arxiv.org/html/2606.07878) 使用 MCQ 答案侧的
forward KL，主训练采用 top-200 教师词表支持并包含答案 token。本适配使用
完整词表 forward KL，并由完整 Qwen 教师生成固定的开放式答案前缀；不加 gold
答案 CE。原复现仓库 benchmark 默认的 `CE-only` 也不同于这一论文目标。

两个训练分支从完全相同的初始化开始，重置优化器和随机种子，采用相同的文档
日程、问题集合、学习率、更新次数、缓存预算和生成上限。学生回放保留梯度，
教师回放停止梯度；完整教师回答只在训练前生成一次，并保存 token IDs、数据和
backbone 哈希。训练不读取 gold 答案或证据文本作为 prompt；QASPER 输入检查
仍保留同一证据合格子集，保证之后的证据教师对照使用同一批样本。

评测统一 greedy 解码、关闭 thinking、相同完整文档与问题。所有方法只做一次
文档 prefill，三个基线均支持在该文档缓存上回答多个问题。STILL 与 OPD 的
每篇文档只压缩一次。每个问题使用独立的运行缓存，物理缓存位置与原文 RoPE
位置分开维护。

STILL vs 普通 OPD 同时改变轨迹来源和散度，比较的是整套 OPD 训练方案。
不同损失数值不能直接作为质量排名。报告单独记录固定教师回答准备时间，
避免遗漏 STILL 的离线数据准备成本。

## 运行入口

在 `STILL-Towards-Infinite-Context-Windows` 目录运行，使用新的输出目录：

```bash
.venv/bin/python scripts/run_opd_comparison.py \
  --methods still full \
  --num-latents 512 --steps 1 \
  --train-documents 1 --eval-documents 1 --questions-per-document 2 \
  --max-new-tokens 16 \
  --output-dir outputs/opd/baselines_smoke_4b
```

`full_context` 自动作为不训练的质量参照。以上命令不运行证据教师诊断或训练。
原 `--teacher-context full/evidence/both` 接口保留；它与新 `--methods` 互斥。
将来 `--methods still full evidence` 可以得到四组比较。

输出包含 `full_context_result.json`、`still_result.json`、`full_result.json`、
`summary.json`、`comparison.md`、两个 compactor checkpoint，以及
`still_teacher_trajectories.json`。checkpoint 仅包含压缩器、优化器状态和配置。

## 四个 benchmark 是否合理

这四个 benchmark 相互补充，但只有 QASPER 直接评测同一论文缓存服务多个原始
问题。其他三个主要检查跨任务泛化和检索能力；应按它们自己的上下文实例评测，
不能把不同 needle 放置或不同文档混用一份缓存。

| Benchmark | 适合验证什么 | 指标与协议 | 当前仓库状态 |
| --- | --- | --- | --- |
| QASPER 留出集 | 一份文档缓存能否服务多个问题 | 官方 max-reference Answer F1；额外文档平均 F1 | 数据已下载；四种方法的统一缓存评测已接通 |
| LongBench v2 | 真实任务的理解与推理泛化 | 官方选择题 Accuracy，按任务/难度/长度分组 | 官方数据已下载，新增按完整上下文分组的转换和当前缓存评测；复用 `kvpress` scorer |
| RULER | 检索、追踪、多目标聚合随长度与预算的变化 | 逐任务官方 string-match 分数及聚合；不把所有任务都视为严格 EM | 新增官方 13 类任务的 Qwen tokenizer 生成、缓存评测与官方匹配评分 |
| NoLiMa | 少词面重合情况下的关联检索 | 按所选官方配置的 EM/contains 等规则汇总正确率，按长度和放置深度报告 | 新增官方 needle/haystack 下载、放置、缓存评测和配置评分 |

本机已缓存 QASPER、完整 LongBench-v2、RULER 的生成资源和 NoLiMa 官方资源。
冒烟使用明确声明的小样本；数据文件、来源 revision 与覆盖数量见新 suite 的
`provenance.json` / `selection.json`，不代表完整官方 benchmark 已评测。

可以复用的代码：

- [QASPER 文档与评分接口](../src/still/data/qasper.py)、
  [当前统一 KV 评测](../src/still/train/opd.py)。
- [LongBench-v2 转换与模板](../../kvpress/evaluation/benchmarks/longbenchv2/create_huggingface_dataset.py)、
  [评分代码](../../kvpress/evaluation/benchmarks/longbenchv2/calculate_metrics.py)。
- [RULER 现有资产](../../kvpress/evaluation/benchmarks/ruler/README.md)、
  [评分代码](../../kvpress/evaluation/benchmarks/ruler/calculate_metrics.py)。
- [cartridges 修改版 RULER](../../cartridges/cartridges/data/ruler/README.md)。

`kvpress` 的 loader 指向经过转换的 HF 数据集；转换脚本中的 `push_to_hub`
是发布步骤，不是本地准备所必需的步骤。当前接入保存本地转换数据，复用
官方数据、prompt 和评分逻辑，再桥接到已有 `CompactKVCache` 回答路径，
无需重装或复刻整个训练栈。当前 registry 中的 `CompactorPress` 也不是本项目
的 STILL Perceiver，不能仅换一个 press 名称就认为已接通。

## 需要明确的评测边界

1. **QASPER：** dev 用于调参，test 用于最终报告。当前 test JSONL 是原始
   官方 test 的筛选子集：4K–8K source tokens、每篇至少两个有完整可映射
   证据的可回答问题，200 篇论文、693 个问题。应明确报告为该子集，而非
   全部官方 QASPER test。dev 为 121 篇/433 问；train 为 309 篇/946 问。
2. **LongBench v2：** 官方 503 道选择题的上下文从 8K 到 2M **words**，
   不等同于 tokens。Qwen3-4B 原生上下文为 32,768 tokens，当前工程冒烟验证
   到 8K source tokens。第一轮应按 Qwen tokenizer 选择可完整读取的样本，
   预留问题、模板和生成空间，报告覆盖数及分组结果；超长样本不默默截断。
   该 HF 数据集的 split 名为 `train`，但在本项目中只用作外部评测。
3. **RULER：** 现有 `kvpress` 生成脚本默认 Llama tokenizer，应用 Qwen3
   tokenizer 重新控制长度。优先 4K、8K 与每层 256/512/1024 个缓存位置的
   网格，再单独验证 16K、32K。保持 needle 数量、深度、复杂度、种子和数据
   在方法间一致。压缩预算变化需要相应预算的 compactor，不能随意切片旧 latent
   权重当作等价的重新训练结果。
4. **NoLiMa：** 复用官方 needle、haystack、放置和评分配置；不能仅把普通
   RULER NIAH 改名。先跑短上下文与完整 KV 参照，判断 backbone 是否具备
   该问题的基础关联推理能力，再分析压缩损失。

LongBench v2 和 NoLiMa 中，即使完整 KV 的模型也可能答错。因此完整 KV 是
性能参照，不是数学上的质量上界。所有外部 benchmark 都只评测训练后的
固定压缩器，不在测试文档或测试问题上优化参数。

官方协议来源：[LongBench v2](https://github.com/THUDM/LongBench)、
[RULER](https://github.com/NVIDIA/RULER)、
[NoLiMa](https://github.com/adobe-research/NoLiMa)、
[NoLiMa 评分实现](https://github.com/adobe-research/NoLiMa/blob/main/evaluation/async_evaluate.py)、
[Qwen3-4B 模型说明](https://huggingface.co/Qwen/Qwen3-4B)。

## 本轮验证结果（2026-10-04）

52 项测试通过，修改的 Python 文件通过 Ruff 和 whitespace 检查。测试覆盖
forward-KL 方向与归一化、教师停止梯度、一次 prefill 多问题、完整 KV 与完整
输入的无缓存生成一致、倒序提问缓存隔离、固定教师前缀不调用学生采样、
gold/evidence 不进入训练 prompt，
以及三个基线 runner 的端到端 checkpoint 和输出检查。

真实 Qwen3-4B / H200 NVL 冒烟采用 4131-token 训练论文、一篇独立 dev 论文、
每篇两个问题，每层 512 个缓存位置、16-token 回答上限、每个训练方法一次更新：

| 方法 | dev 文档 F1 | 训练分支峰值显存 GiB | 检查 |
| --- | ---: | ---: | --- |
| 完整 KV | 0.5000 | 9.41（评测峰值） | 同一评测接口，无参数更新 |
| STILL 风格 KL | 0.1357 | 23.98 | 36 层梯度有限，压缩器更新，checkpoint 回读一致 |
| 普通 OPD | 0.0926 | 24.05 | 36 层梯度有限，压缩器更新，checkpoint 回读一致 |

训练与评测峰值显存不是同一种成本，不能用本表当推理效率排名。单篇 dev、
两个问题、一次更新和短回答上限不能支持方法优劣结论。两个训练方法共享的
backbone SHA256 均保持为
`91e6502c7b75f2011e35650e5c275b3b3be9743edae1acf8e295d64007760f57`。

本机原始报告：[comparison.md](../outputs/opd/baselines_smoke_4b/comparison.md)、
[summary.json](../outputs/opd/baselines_smoke_4b/summary.json)。生成 artifacts 不进入 git。
