# KVCache OPD

第一版整合位于 `STILL-Towards-Infinite-Context-Windows`，复用其 Perceiver 压缩器、KV 缓存与 attention bias 接口。

- `full`：完整文档教师的 KV OPD 基线。
- `evidence`：逐问题原文证据教师的 KV OPD。
- 两组共享冻结的 Qwen3-4B、问题无关的文档压缩器、采样设置、完整词表 JSD 和缓存预算。

配置、数据筛选、可复现命令和冒烟结果见 [整合说明](STILL-Towards-Infinite-Context-Windows/docs/opd_integration.md)。

```bash
cd STILL-Towards-Infinite-Context-Windows
.venv/bin/python scripts/run_opd_comparison.py --teacher-context both
```

默认每组仅执行一次更新，属于训练前冒烟测试。正式训练效果需要在足够训练步数和完整留出集上另行评测。
