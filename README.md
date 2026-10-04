# KVCache OPD

第一版整合位于 `STILL-Towards-Infinite-Context-Windows`，复用其 Perceiver 压缩器、KV 缓存与 attention bias 接口。

- `full`：完整文档教师的 KV OPD 基线。
- `evidence`：逐问题原文证据教师的 KV OPD。
- 两组共享冻结的 Qwen3-4B、问题无关的文档压缩器、采样设置、完整词表 JSD 和缓存预算。

配置、数据筛选、可复现命令和冒烟结果见 [整合说明](STILL-Towards-Infinite-Context-Windows/docs/opd_integration.md)。

当前先建立三个基线：完整 KV、STILL 风格固定教师 forward KL、完整上下文 OPD。
[三个基线与 benchmark 核查](STILL-Towards-Infinite-Context-Windows/docs/baselines_and_benchmarks.md)
记录了统一比较入口、已验证的冒烟结果和 QASPER / LongBench v2 / RULER / NoLiMa 的接入状态。

四种方法在四个 benchmark 上的统一磁盘权重评测见
[benchmark 接口与命令](STILL-Towards-Infinite-Context-Windows/docs/benchmark_suite.md)。
准备数据与压缩器 checkpoint 后运行：

```bash
cd STILL-Towards-Infinite-Context-Windows
.venv/bin/python scripts/evaluate_benchmarks.py \
  --data-dir outputs/benchmarks/smoke_data \
  --checkpoint-dir outputs/opd/four_methods_smoke_4b \
  --num-latents 512 --max-new-tokens 16 \
  --output-dir outputs/benchmarks/smoke_4b
```

前三个基线的训练冒烟入口：

```bash
cd STILL-Towards-Infinite-Context-Windows
.venv/bin/python scripts/run_opd_comparison.py --methods still full \
  --output-dir outputs/opd/baselines_smoke_4b
```

后续两个 OPD 教师对照仍可使用：

```bash
cd STILL-Towards-Infinite-Context-Windows
.venv/bin/python scripts/run_opd_comparison.py --teacher-context both
```

默认每组仅执行一次更新，属于训练前冒烟测试。正式训练效果需要在足够训练步数和完整留出集上另行评测。
