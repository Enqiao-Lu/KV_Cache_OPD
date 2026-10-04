# 四种方法的统一 benchmark 接口

完整 Qwen3-4B、STILL 风格 KL、普通 OPD、证据 OPD 使用同一评测入口。
后三种方法加载训练脚本保存的压缩器 checkpoint；完整 KV 无需 checkpoint。
推理只读取完整文档与问题，不使用证据或答案标签。方法定义和 STILL 的
QASPER 适配差异见 [基线说明](baselines_and_benchmarks.md)。

## 入口与准备

以下命令均在 `STILL-Towards-Infinite-Context-Windows` 目录执行。
现有环境已具备模型与 QASPER；新环境先安装依赖，再运行原准备脚本：

```bash
uv pip install --python .venv/bin/python -e '.[train,benchmarks,dev]'
.venv/bin/python scripts/prepare_opd.py
```

数据准备入口为 [prepare_benchmarks.py](../scripts/prepare_benchmarks.py)。
默认准备四个 benchmark；`--benchmarks` 可选一个或多个。所有文件保存到
本地 `outputs/`，不发布数据集。

本轮小样本的可复现准备命令：

```bash
.venv/bin/python scripts/prepare_benchmarks.py \
  --benchmarks qasper longbench_v2 \
  --document-limit 2 --question-limit 2 --max-new-tokens 16 \
  --output-dir outputs/benchmarks/smoke_data

.venv/bin/python scripts/prepare_benchmarks.py \
  --benchmarks ruler --lengths 4096 --samples 1 --max-new-tokens 16 \
  --output-dir outputs/benchmarks/smoke_data

.venv/bin/python scripts/prepare_benchmarks.py \
  --benchmarks nolima --lengths 1024 4096 --depths 0.25 0.75 \
  --samples 2 --max-new-tokens 16 \
  --output-dir outputs/benchmarks/smoke_data
```

QASPER 默认使用已准备的 `outputs/opd/data/dev.jsonl`；用 `--qasper-data`
切换到 test。LongBench v2 默认下载官方 HF 数据，也可通过
`--longbench-source path/to/raw.jsonl` 读取本地官方 JSON/JSONL。
RULER 与 NoLiMa 首次运行会下载固定版本的官方仓库到
`outputs/benchmarks/vendor/`；`--vendor-dir` 可改变缓存目录。

每个 benchmark 写入 `<output-dir>/<benchmark>/prepared/`：

- `source/documents.jsonl`、`source/provenance.json`：转换后的完整候选数据及来源。
- `documents.jsonl`：模型窗口内的完整上下文子集。
- `provenance.json`、`selection.json`：实际文件哈希、选择数量、排除长度与未扫描数量。
- 原始下载、生成命令和日志保存在 `source/` 下或声明的来源路径。

上下文窗口默认 32,768 tokens，计算时包含 chat 模板、问题、回答前缀及生成
预留空间。不截断文档。`--document-limit 0 --question-limit 0` 表示全部；
限制数量时会记录未扫描部分，不能将这次覆盖数解释为完整官方集的覆盖率。

## 四种方法的磁盘权重评测

先沿用已有训练入口产生三个相同初始化、相同预算的冒烟 checkpoint：

```bash
.venv/bin/python scripts/run_opd_comparison.py \
  --methods still full evidence --num-latents 512 --max-new-tokens 16 \
  --output-dir outputs/opd/four_methods_smoke_4b
```

默认只训练一篇 train 文档、两个问题、一次更新。它用于检查接口，不提供
可据以排名的正式权重。训练数据与 dev 分开；外部 benchmark 不参与训练。

[evaluate_benchmarks.py](../scripts/evaluate_benchmarks.py) 加载保存的磁盘权重：

```bash
.venv/bin/python scripts/evaluate_benchmarks.py \
  --data-dir outputs/benchmarks/smoke_data \
  --checkpoint-dir outputs/opd/four_methods_smoke_4b \
  --methods full_context still full evidence \
  --num-latents 512 --max-new-tokens 16 \
  --output-dir outputs/benchmarks/smoke_4b
```

默认方法和 benchmark 已包含全部四组。也可指定 `--benchmarks ruler nolima`
或 `--methods full_context full`。`--still-checkpoint`、`--full-checkpoint`、
`--evidence-checkpoint` 支持分别加载其他路径。

加载器校验 model ID、固定 revision、backbone 全参数 SHA256、方法、真实/微型
模型标志、缓存预算和压缩器结构。缺失或不兼容的权重直接报错，不以随机参数
替代。比较中的压缩器预算必须一致。评测禁用梯度，验证 backbone 未改变。

正式评测可使用同一入口，切换正式训练的 checkpoint 与更完整的准备集。
不传 `--max-new-tokens` 时使用任务默认上限：LongBench v2 为 16，RULER 使用
各任务的官方上限，其他任务为 128。显式的短上限会记录到逐题结果；聚合任务
可能因回答截断而少得分，不能据此报告正式能力。

## 数据与评分协议

| Benchmark | 数据/生成 | 评分与分组 |
| --- | --- | --- |
| QASPER | 原有官方划分的完整论文筛选集，按论文共享缓存 | 官方 max-reference Answer F1，额外文档平均 F1 |
| LongBench v2 | 官方 503 题；相同完整上下文合并，保留 0-shot 选项与回答格式 | 复用 `kvpress` 中官方 Accuracy，按难度、长度、domain/sub-domain 报告 |
| RULER | 固定 NVIDIA 仓库，Qwen tokenizer，全部 13 类生成器 | 官方 all-reference recall / QA best-reference matching；清理控制字符，逐任务和任务平均 |
| NoLiMa | 固定官方 needle、五本 shuffled haystack、原 BookHaystack 放置算法 | 官方配置的大小写敏感 `contains`；同时支持 EM、lastline_EM、lastline_contains，按长度、深度、needle 等分组 |

RULER 官方 few-shot 示例保留在上下文中；最后的目标问题和回答前缀在缓存外。
有些示例包含与目标相同的通用问题措辞，不应把它们误删。

NoLiMa 的长度是**插入 needle 与 chat 模板之前的 haystack tokens**；本轮使用
1024/4096，而官方 YAML 的 1K/4K 参数为 1000/4000，来源记录明确保存此覆盖。
`--samples` 在 NoLiMa 表示官方测试 variant 数，在 RULER 表示每个任务的样本数。
NoLiMa 默认标准 shuffled/non-distractor 配置；可用 `--nolima-config` 指定兼容
官方配置。distractor 与其他 haystack 变体会明确报错，需单独声明协议。

外部任务的上下文段适配到 Qwen system cache，问题放在 user continuation；
RULER 的官方 answer prefix 接到 assistant 开头。该 chat 适配在四种方法间
一致，并记录在 provenance，属于本项目的 KV 评测协议。QASPER 保持训练时的
原 chat 格式；thinking 均关闭。benchmark 分数统一储存为 0–1，RULER 另存
官方百分制分数。完整 KV 是模型性能参照，不保证所有任务答对。

官方来源：[LongBench v2](https://github.com/THUDM/LongBench)、
[RULER](https://github.com/NVIDIA/RULER)、[NoLiMa](https://github.com/adobe-research/NoLiMa)。
仓库与 HF 数据 revision、tokenizer 与资源文件哈希保存在本机 provenance 中。

## 输出

`<output-dir>/<method>/<benchmark>/predictions.jsonl` 保存逐题回答、token IDs、
参考答案、得分、任务 metadata、源缓存/压缩缓存长度及生成上限。
同目录的 `summary.json` 保存指标与分组；输出根目录的 `summary.json` 和
`comparison.md` 汇总四种方法、四个任务、数据哈希和 checkpoint 来源。

实现文件：[统一缓存推理](../src/still/benchmarks/suite.py)、
[LongBench v2](../src/still/benchmarks/longbench_v2.py)、
[RULER](../src/still/benchmarks/ruler.py)、[NoLiMa](../src/still/benchmarks/nolima.py)。

## 已验证的冒烟（2026-10-04）

145 项测试通过，修改的 Python 文件通过 Ruff 和 diff 检查。独立审阅发现的
来源哈希、backbone 校验、固定版本 LFS 下载及 prompt-style 校验问题均有
先失败后通过的回归测试；最终审阅未发现剩余 Important 问题。

H200 NVL 上，真实冻结 Qwen3-4B、每层 512 个缓存位置，三个训练方法各一次
更新；评测 16-token 上限。四种方法使用完全相同的以下样本：

| Benchmark | 文档/上下文 | 问题 | 覆盖 |
| --- | ---: | ---: | --- |
| QASPER dev 子集 | 2 | 4 | 同一论文多个问题 |
| LongBench v2 子集 | 2 | 2 | 前 22 个上下文中 20 个因窗口排除；其余 440 个未扫描 |
| RULER | 13 | 13 | 全部 13 类，每类 1 题，声明的长度 4096 |
| NoLiMa 标准子集 | 8 | 8 | 2 个官方 variant × 1024/4096 haystack tokens × 25%/75% 深度 |

16 个方法/benchmark 组合全部成功，总计 108 条逐题预测，均生成至少一个
token、得分有限，压缩缓存均为 512，磁盘 checkpoint 哈希校验通过。
评测前后 backbone SHA256 为
`91e6502c7b75f2011e35650e5c275b3b3be9743edae1acf8e295d64007760f57`。

| 方法 | QASPER 问题 F1 | LongBench v2 Accuracy | RULER 任务平均 | NoLiMa Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 完整 KV | 0.4325 | 0.0000 | 0.6397 | 0.6250 |
| STILL 风格 KL | 0.1144 | 0.5000 | 0.1538 | 0.0000 |
| 普通 OPD | 0.0928 | 0.5000 | 0.1538 | 0.0000 |
| 证据 OPD | 0.0928 | 0.5000 | 0.1538 | 0.0000 |

这些分数只用于确认生成和评分链路；两道选择题、单步训练和短回答不支持
方法排名。RULER 生成器在 FWE 日志中存在上游 logging-format 提示，但返回
成功，输出经结构、长度与评分检查；不影响实际样本或推理。

本机完整报告：[comparison.md](../outputs/benchmarks/smoke_4b/comparison.md)、
[summary.json](../outputs/benchmarks/smoke_4b/summary.json)、
[逐组验证记录](../outputs/benchmarks/smoke_4b/smoke_validation.json)。
数据与权重位于 `outputs/benchmarks/smoke_data/` 和
`outputs/opd/four_methods_smoke_4b/`，均不进入 Git。
