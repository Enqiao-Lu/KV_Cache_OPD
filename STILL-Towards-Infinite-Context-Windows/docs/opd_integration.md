# Qwen3-4B / STILL KV OPD 整合

## 入口与实验定义

`scripts/run_opd_comparison.py` 在同一训练实现中切换教师上下文：

| 项目 | `full` 基线 | `evidence` 方案 |
| --- | --- | --- |
| 冻结 backbone | Qwen/Qwen3-4B | 同一 checkpoint |
| 学生读取 | 整篇文档压缩缓存 | 相同 |
| 教师读取 | 未压缩完整文档 | 该问题的未压缩原文证据 |
| 可训练对象 | STILL 压缩器 | 相同 |
| 缓存预算 | 默认每层 512 个位置 | 相同 |
| 压缩器输入 | 文档 KV，不含问题 | 相同 |
| 损失 | 学生轨迹上的等权完整词表 JSD | 相同 |
| 归一化 | 先对回答 token 平均，再对问题平均 | 相同 |
| 标签 | 仅用于数据筛选、教师诊断与评测 | 相同 |

这里的“原始 OPD”是**完整上下文教师的 KV OPD 对照**。工作区 `OPSD/opsd_trainer.py` 的原始训练器使用解答特权上下文并更新 LLM/LoRA 权重；直接运行它会改变实验变量。本整合保留其 on-policy 分布匹配思想，测试通过 AST 提取纯 JSD 方法验证数值一致，不安装该训练器的 TRL、DeepSpeed、vLLM 训练栈。

两组使用相同初始压缩器、种子、文档日程、优化器参数和生成设置，并检查第一步所有问题的学生轨迹一致。更新后轨迹可以随各自策略变化，这是 on-policy 训练的一部分。

## 复用接口与必要修正

- `src/still/core/still.py`：每层一个 `StillLayerCompactor`，两层 Perceiver block；Qwen3 的 head dimension 为 128，因此内部维度为 256。固定可学习 latent → cross-attention → self-attention/RMSNorm → K/V/beta 投影。现有复现没有独立 FFN，本次保持该结构。
- `src/still/core/cache.py`：`CompactKVCache`、缓存标准化、`DynamicCache` 构建和序列化。
- `src/still/attention_bias.py`：beta 加性 bias，以及 GQA KV head 到 query head 的展开。
- `src/still/chat.py`：验证 system 文档前缀与 question continuation 的精确 token 拼接。
- `src/still/train/still.py`：已有种子与训练文档日程函数。旧离线 KL/CE 训练入口保留。

修正了两个共享 helper：RoPE 从相邻维度旋转改为 Hugging Face Qwen3 的 split-half 约定；本地 tokenizer 通过直接的 `enable_thinking=False` 参数关闭 thinking，原嵌套参数没有传入模板变量。回归测试与 Qwen3 原生 RoPE 和模板输出逐项比较。

压缩 key 的 latent positions 仍分布在原始 source positions 上。新的 OPD 接口显式区分：

```text
压缩缓存物理长度 = m
新增 token 的 cache_position 从 m 开始
新增 token 的 position_ids 从原始 source prefix 长度开始
```

每个问题、采样、教师回放、学生回放分别创建新的 `DynamicCache` 容器。不能把已追加问题/回答 token 的运行缓存用于下一问题。采样和教师计算使用 `no_grad`；学生回放穿过冻结 LLM 将梯度传回压缩器，JSD 的 mixture 保留学生梯度。模型声明的两个 EOS ID 都能终止生成，最后的结束 token 仍参与蒸馏。

这些 RoPE/模板修正改变了旧实现的行为；此前基于旧约定训练的压缩器和缓存不应直接用于本次公平对照。

## 环境与下载

在本仓库目录执行：

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -e '.[train,dev]'
.venv/bin/python scripts/prepare_opd.py
.venv/bin/python -m pytest tests -q
```

已建立 `.venv`，Python 3.12，Torch 2.10.0+cu128，Transformers 4.57.6。环境清单位于 `outputs/opd/environment.freeze.txt`。`openai` 加入 train extra，因为现有包初始化会导入已有客户端；新 OPD 路径使用本地 Hugging Face 模型。所有 GPU 冒烟在 H200 NVL 上运行。

精确 backbone revision：`1cfa9a7208912126459214e8b04321603b3df60c`。

下载包含三片 safetensors 权重、配置、generation config、tokenizer 和 merges，位于：

```text
/home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
```

模型与数据使用官方来源：[Qwen3-4B 模型说明](https://huggingface.co/Qwen/Qwen3-4B)、[QASPER 数据说明](https://huggingface.co/datasets/allenai/qasper)。准备脚本直接下载官方 QASPER v0.3 train/dev/test 原始归档，保留官方 evaluator，不执行远程数据集脚本。原始文件和 SHA256 位于 `outputs/opd/data/raw/` 与 `outputs/opd/data/manifest.json`。

## 数据约定

`src/still/data/qasper.py` 把同一篇论文的多个原始问题放在同一条 JSONL 中。只要某条 answerable annotation 的**全部** evidence 都能映射到正文、标题或摘要，就保留这条完整证据链；不接受缺失段落、表格占位符或部分证据链。混合标注中其他人的 `Unanswerable` 参考答案仍保留用于官方 max-reference F1，不作为教师证据。

该检查验证原文出处和标注完整性；语义充分性依赖人工证据标注，本次没有额外声称自动证明语义充分性。教师答案诊断用于观察证据是否有效，不是训练损失。

默认要求每篇至少两个有效问题，source prefix 为 4096–8192 tokens，包括固定 system 模板。超长论文整体过滤，完整文档不裁剪，也不根据问题调整文档内容。官方划分没有文档 ID 交叉。

| 划分 | 保留论文 | 保留问题 |
| --- | ---: | ---: |
| train | 309 | 946 |
| dev | 121 | 433 |
| test | 200 | 693 |

训练只读取 `document`、`question`、`evidence`；`answers` 不进入任何学生或教师 prompt。测试用替换全部 gold answer 的方式验证训练损失不变。评测使用官方 token F1 的最大参考分数，同时输出问题平均和文档平均。

## 运行与结果文件

真实模型默认冒烟：一篇训练论文、一篇开发论文，每篇两个问题，每组一次 AdamW 更新，回答上限 16 tokens，每层 512 个位置。

```bash
.venv/bin/python scripts/run_opd_comparison.py \
  --teacher-context both --num-latents 512 --max-new-tokens 16 \
  --output-dir outputs/opd/smoke_4b
```

小模型检查（真实 tokenizer、随机两层 Qwen3，验证接口）：

```bash
.venv/bin/python scripts/run_opd_comparison.py \
  --tiny --num-latents 8 --max-new-tokens 3 \
  --output-dir outputs/opd/smoke_tiny
```

8K 检查使用训练集中 8147 tokens 的论文，输入文件已经写到 `outputs/opd/data/smoke_8k_train.jsonl`：

```bash
.venv/bin/python scripts/run_opd_comparison.py \
  --train-data outputs/opd/data/smoke_8k_train.jsonl \
  --teacher-context both --num-latents 512 --max-new-tokens 8 \
  --output-dir outputs/opd/smoke_8k
```

每个输出目录包含：

- `summary.json` / `comparison.md`：匹配设置、训练 loss、梯度、更新前后开发集 F1、峰值显存和检查结果。
- `teacher_diagnostics.json`：同一初始学生轨迹上的 student/full/evidence JSD；双方教师独立生成的答案和 F1。
- `full_result.json` / `evidence_result.json`：逐步、逐问题记录及生成 token IDs。
- `full_compactor.pt` / `evidence_compactor.pt`：压缩器权重、优化器状态和运行 metadata，已用 `weights_only=True` 回读并检查参数哈希。

新 checkpoint 不包含 backbone 权重；推理时加载相同 backbone 并构建相同预算的压缩器。可以直接使用以下现有接口，文档原始 KV 在完成压缩后释放：

```python
import torch
from still.core import StillCompactor
from still.train.opd import build_document_cache

checkpoint = torch.load(path, map_location='cpu', weights_only=True)
compactor = StillCompactor.from_model_config(model.config, num_latents=512).to(model.device)
compactor.load_state_dict(checkpoint['state_dict'])
with torch.no_grad():
    full, compact = build_document_cache(model, tokenizer, compactor, document, config)
    position_start = full.num_tokens
    del full
    # 对该文档的每个新问题调用 question_ids 和 rollout；共享 compact。
    # compact.save(...) / CompactKVCache.load(...) 可用于持久化缓存。
```

命令行的 `--steps`、`--train-documents`、`--eval-documents`、`--questions-per-document`、`--num-latents` 可以控制后续实验规模。默认 smoke 只取各划分开头的一篇文档和两个问题；正式实验应覆盖足够训练文档和完整留出集，使用相同预算/种子比较多次运行。

当前冒烟只验证工程链路。不同教师下 JSD 的大小不是两种方法质量的排名；一两篇论文、一次更新和截断的短回答 F1 不能支持方法有效性的结论。

## 已验证的冒烟结果（2026-10-04）

最终代码通过 45 项仓库测试、修改文件的 Ruff 检查和 `git diff --check`。全部 2072 个保留问题的参考答案及 token F1 与下载的官方 evaluator 对照一致；空回答、混合 `Unanswerable` 标注、全部模型 EOS、同前缀采样/回放与相同上下文教师控制组均有回归检查。

真实 backbone 有 4,022,468,096 个冻结参数，压缩器有 44,909,604 个可训练参数。每层压缩 K/V 为 `[1, 8, 512, 128]`，共 36 层。

| 原文 source tokens | 教师 | 单步 JSD | 梯度范数（clip 前） | 有梯度的层 | 峰值显存 GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4131 | full | 0.303787 | 39.0426 | 36 | 24.01 |
| 4131 | evidence | 0.257823 | 58.8893 | 36 | 24.10 |
| 8147 | full | 0.292762 | 45.4318 | 36 | 35.71 |
| 8147 | evidence | 0.296028 | 46.1157 | 36 | 35.17 |

两组均完成学生采样、教师/学生回放、反向传播、AdamW 更新、checkpoint 保存/回读及开发论文评测。所有梯度有限；压缩器参数哈希改变；backbone 全参数 SHA256 始终为 `91e6502c7b75f2011e35650e5c275b3b3be9743edae1acf8e295d64007760f57`；两组第一步轨迹一致。4K 和 8K 检查各自采用 16 和 8 个生成 token 上限，不能将两行不同长度的 JSD 当作同一质量实验。

详细可点击报告：[4K 对照](../outputs/opd/smoke_4b/comparison.md)、[8K 对照](../outputs/opd/smoke_8k/comparison.md)、[4K 教师诊断](../outputs/opd/smoke_4b/teacher_diagnostics.json)。这些是本机生成的 artifacts，`outputs/` 不进入 git。

## 旧 benchmark 接口

原 MCQ benchmark 对外部 cartridges 复现的导入改为在调用该分支时加载，报表与 STILL 模块无需安装它即可导入。路径可以通过 `STILL_CARTRIDGES_ROOT` 指定，默认工作区的 sibling `cartridges`。当前 sibling 是另一种目录/API 布局；旧 MCQ cartridges 分支仍需要原兼容复现，本次新的 KV OPD 两组对照没有该依赖。
