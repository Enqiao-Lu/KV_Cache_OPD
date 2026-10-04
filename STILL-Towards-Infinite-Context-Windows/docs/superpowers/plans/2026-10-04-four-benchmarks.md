# Four Benchmark Interfaces Implementation Plan

**Goal:** Run full Qwen3-4B, STILL-style KL, full-teacher OPD and evidence-teacher
OPD checkpoints through QASPER, LongBench v2, RULER and NoLiMa, with real-model smoke evidence.

**Approved scope:** Complete the adapters and smoke every method/benchmark as requested.
Reuse official data and scoring, existing frozen Qwen/cache/Perceiver interfaces and installed dependencies.
Prepare evaluation data locally; no dataset publication and no formal training campaign.

**Design:** Benchmark adapters emit document-grouped JSONL with `benchmark`,
`document_id`, `document`, `prompt_style` (`qasper` or `context`), and `questions`.
Each question contains `question_id`, `question`, `answers`, `metric`, `metadata`
and optional `answer_prefix`. For context style, `document` is the complete system
prefix; questions are encoded as user continuations. Context keys never contain
the target question or answer. QASPER retains the original training chat format.
A single checkpoint evaluator prefills/compresses once per document, isolates
runtime caches per question, continues source RoPE positions, applies official
metric adapters, and writes per-question results plus grouped summaries.

### Task 1: Independent data and scoring adapters

- [x] LongBench v2: create `src/still/benchmarks/longbench_v2.py` and
  `tests/test_longbench_v2.py`; load official HF/local data, retain complete
  contexts and 0-shot MCQ prompt, preserve difficulty/domain/length, reuse scorer.
- [x] RULER: create `src/still/benchmarks/ruler.py` and `tests/test_ruler_adapter.py`;
  reuse official model-tokenizer generation for all 13 tasks, separate context
  and question, preserve task metadata and answer prefix, reuse all/part matching.
- [x] NoLiMa: create `src/still/benchmarks/nolima.py` and `tests/test_nolima.py`;
  load official needles/haystacks/configuration, generate local length/depth
  samples, preserve configured scoring and exact source placement.
- [x] For each adapter, write a small source-shaped fixture and scoring tests,
  observe failures first, implement, and verify no question/gold enters cache.

### Task 2: Unified inference and checkpoint loading

- [x] Add `src/still/benchmarks/suite.py` and `tests/test_benchmark_suite.py`.
  Validate JSONL, contexts, IDs, metrics, and token limits before inference.
- [x] Test independent uncached full-Qwen generation against full-KV inference;
  test question order/cache isolation, compaction once, and no gradients at evaluation.
- [x] Load checkpoint `state_dict` plus metadata; reject incompatible model revision,
  method, architecture, or latent budget. Never silently evaluate random weights.
- [x] Add `scripts/prepare_benchmarks.py` and `scripts/evaluate_benchmarks.py`.
  Support all four benchmarks, explicit checkpoint paths and subset/context limits.
  Save data/checkpoint hashes, filtering coverage, prompts, predictions and scores.
- [x] Add tiny-model end-to-end CLI regression and run the full existing suite.

### Task 3: Real smoke and documentation

- [x] Create matched one-step `still`, `full`, `evidence` 512-slot checkpoints with
  the existing `run_opd_comparison.py --methods still full evidence` runner.
- [x] Prepare real QASPER heldout, native-window-fitting LongBench v2, all RULER
  tasks at a short declared length, and official NoLiMa length/depth samples.
- [x] Run all four checkpoints/methods across all four prepared datasets on H200.
  Require generated tokens, finite scores, identical sample sets, saved artifacts,
  and a frozen backbone. Smoke scores are not an effectiveness claim.
- [x] Add reproducible prepare/train/evaluate commands to README and
  `docs/benchmark_suite.md`; update stale readiness documentation.
- [x] Run complete tests, Ruff, diff checks and an independent review; integrate
  the verified changes back into the user's workspace.

**Verification commands:** Use the original project `.venv/bin/python` with
`PYTHONPATH=src` in this worktree. Run `python -m pytest -q` and
`python -m ruff check` for changed Python files. Real outputs live in the original
project's ignored `outputs/benchmarks/` and `outputs/opd/four_methods_smoke_4b/`.

**Boundaries:** No truncation disguised as full official evaluation. Filter whole
LongBench examples with a stated context limit and generation reserve. Preserve
RULER's official task-specific all/part scoring. NoLiMa is the official benchmark,
not renamed literal NIAH. No test-document optimization or evidence at inference.

## Completion evidence

- 145 tests passed; Ruff and diff checks passed.
- Independent review findings repaired with observed failing/passing regressions:
  source/selection provenance separation, mandatory backbone hashes, pinned LFS
  checkout bypass, benchmark-specific prompt style, malformed-object diagnostics.
- Real Qwen3-4B: all 16 method/benchmark groups completed; 108 predictions, finite
  scores, matching sample sets, 512 compact slots, verified disk checkpoints and
  unchanged full backbone SHA256. Data: 4 QASPER questions, 2 LongBench-v2, all
  13 RULER tasks, 8 official NoLiMa length/depth instances.
- Review final: no remaining Important findings. Results are engineering smoke
  evidence from one-step weights, not a comparative effectiveness conclusion.
