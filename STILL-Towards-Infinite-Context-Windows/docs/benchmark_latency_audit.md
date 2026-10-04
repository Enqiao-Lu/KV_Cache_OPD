# Benchmark Latency Audit

This note explains the STILL versus truncation latency gap in the final MCQ benchmark and fixes the metric semantics used by the tracked report.

## What Each Metric Includes

- `mean_query_continuation_latency_ms`
  - Includes only the online question-answering path after the method-specific artifact is ready.
  - For `full_context` and `truncation`, this still includes prompt prefill for that question because there is no reusable artifact.
  - For `STILL`, this is the short user continuation against a prebuilt compact cache.
  - For `cartridge`, this is the short user continuation against the optimized per-target cartridge.
  - In markdown/README output this is labeled `Mean online query latency` to make the timing boundary explicit.

- `mean_target_preparation_seconds`
  - `full_context`: `0`
  - `truncation`: `0`
  - `STILL`: one compact-cache build per held-out target
  - `cartridge`: one per-target optimization run

- `mean_target_total_seconds`
  - `mean_target_preparation_seconds + questions_per_target * mean_query_continuation_latency_ms / 1000`
  - Excludes `STILL` reusable compactor training

- `one_time_reusable_training_seconds`
  - Applies only to `STILL`
  - Paid once across unseen targets, not once per target

- `mean_query_total_latency_ms`
  - The corrected headline metric
  - `mean_target_total_seconds * 1000 / questions_per_target`
  - This is the operational per-query cost after amortizing any per-target preparation across the benchmark's questions for that target

## Root Cause Of The STILL vs Truncation Gap

The gap is real under the benchmark contract. It is not primarily a timer bug.

`truncation_1024` and `still_1024_ce_only` both use a `1024`-token compact budget, but they do not perform the same online work:

- `truncation_1024` reprefills roughly `1024` context tokens on every question
- `STILL` builds the compact cache once per target, then answers each question from a much shorter prompt plus the reused compact cache

That difference shows up directly in prompt length:

- `truncation_1024`: mean prompt tokens `1133.985`
- `still_1024_ce_only`: mean prompt tokens `75.285`

So the earlier comparison of:

- STILL continuation latency: `84.517 ms`
- truncation continuation latency: `133.736 ms`

is not comparing identical online work. The corrected end-to-end metric is:

- STILL total per-query latency: `102.479 ms`
- truncation total per-query latency: `133.736 ms`

The cache-build step matters, but it is small enough in this benchmark that STILL remains faster even after amortization.

## Decode-Length Audit

The earlier aggregate numbers also hid a normalization effect.

The benchmark normalizes MCQ generations down to a single letter, but the raw decoded output can be longer.

A direct CPU diagnostic on the first held-out sample produced:

- `full_context`: raw prediction `"<think>\n\n</think>\n\nC<|im_end|>"`, `6` generated tokens
- `truncation_1024`: raw prediction `"<think>\n\n</think>\n\nC<|im_end|>"`, `6` generated tokens
- `STILL`: raw prediction `"D<|im_end|>"`, `2` generated tokens in the tracked final run

So the `2 vs 6` token difference is real. The extra tokens in full/truncation are mostly the `<think>` wrapper plus the final chat terminator, even though the normalized prediction is only `A/B/C/D`.

This does not fully explain the latency gap by itself; the larger driver is that STILL processes a much shorter prompt online. But the shorter generation does contribute.

## Timing Boundary Audit

The timing boundaries were inspected in:

- `scripts/run_benchmark.py`
- `src/still/eval/baseline.py`
- `src/still/eval/truncation.py`
- `src/still/eval/still.py`

Findings:

- CUDA synchronization is already present around the timed forward passes.
- The reported continuation latency comes from the per-row evaluator and is not mixing in reusable training.
- STILL cache-build time was already tracked separately and included in the per-target total; the main problem was that the headline report label made the continuation metric look like the end-to-end metric.
- The benchmark now logs decode-debug fields for future runs:
  - `raw_prediction`
  - `generated_token_ids`
  - `normalized_prediction`
  - `finish_reason`

## Current Limitation

The best STILL result in the tracked report comes from reusing the best checkpoint found in the seed sweep rather than from a fresh retrain embedded inside the 4-way benchmark.

That is deliberate. The benchmark was rerun on `GPU 4` with `--skip-still-train` so the final timing comparison would reflect the best validated STILL checkpoint instead of being contaminated by a worse retrain from the same code path.
