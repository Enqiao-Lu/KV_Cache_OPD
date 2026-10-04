# Final MCQ Benchmark

Definitions:
- `Mean query total latency` is the amortized end-to-end per-question cost, including method-specific preparation when a method has one.
- `Mean online query latency` is per-question question-answering latency after the method-specific artifact is already ready.
- For `full_context` and `truncation`, `mean online query latency` still includes building the prompt-side KV state for that question because there is no reusable artifact.
- For `STILL` and `cartridge`, `mean online query latency` starts after the compact cache artifact already exists and measures only the question-answering pass against that artifact.
- `Mean target preparation` is the one-time cost to prepare a held-out corpus/page before answering its benchmark questions.
- `One-time reusable training` applies only to STILL and is the cost to train the reusable compactor once on the training split.
- `Mean target total seconds` combines `mean target preparation` plus all benchmark-question inference for one held-out target, excluding STILL's one-time reusable training.

| Method | Accuracy | Compression vs Full | Mean KV Bytes | Mean Query Total Latency (ms) | Mean Online Query Latency (ms) | Mean Query Prefill (ms) | Mean Query Decode tok/s | Mean Prompt Tokens | Mean Completion Tokens | Target Prep Kind | Mean Target Preparation (s) | Mean Target Total Seconds | One-Time Reusable Training (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| full_context | 0.950 | 1.000 | 410928168 | 178.388 | 178.388 | 79.829 | 59.637 | 2786.785 | 5.835 | none | 0.000 | 1.784 | n/a |
| truncation_1024 | 0.775 | 2.458 | 167212892 | 133.736 | 133.736 | 32.708 | 59.139 | 1133.985 | 5.960 | none | 0.000 | 1.337 | n/a |
| still_1024_ce_only | 0.315 | 2.721 | 150994944 | 102.479 | 84.517 | 30.605 | 80.760 | 75.285 | 3.200 | still_cache_build | 0.180 | 1.025 | 274.215 |
| cartridge_1024 | 0.885 | 2.736 | 150198681 | 2537.627 | 189.468 | 24.821 | 57.334 | 75.285 | 8.975 | per_target_optimization | 23.482 | 25.376 | n/a |

## STILL Settings

- Latents: 1024
- Steps: 300
- Learning rate: 2e-05
- Seed: 1
- Validation examples: 32
- Validation interval: 50
- Best checkpoint step: 300
- KL weight: 0.0
- Exact-token CE weight: 1.0
- Best validation loss: 0.915
- One-time reusable training seconds: 274.215
- Mean per-target cache build seconds: 0.180

## Latency Interpretation

- Each held-out target contributes about 10.000 benchmark questions.
- Truncation and full-context methods pay the retained prompt prefill cost on every question.
- STILL pays one cache-build step per target, then answers the target's questions from the compact cache with only the short continuation online.
- Equal compact budget (`1024`) does not imply equal online work: truncation still processes roughly a thousand prompt tokens per query, while STILL processes a much shorter prompt plus the reused compact cache.
- `Mean online query latency` is the right number if you want to isolate question-answering speed after any artifact is already built. `Mean query total latency` is the right number if you want operational end-to-end cost per query.
- Decode lengths differ materially in the current benchmark. The prediction artifacts now log raw decoded text and generated token ids so normalized MCQ letters can be audited against the actual generation trace.

## Cartridge Settings

- Tokens: 1024
- Steps per held-out target: 240
- Mean per-target optimization seconds: 23.482
- Cartridge has no reusable training stage; every new target corpus pays this optimization cost again.
