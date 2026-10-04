# Evidence-privileged KV OPD integration plan

**Goal:** Prepare a runnable Qwen3-4B / STILL comparison before substantive training.

**Architecture:** Reuse `StillCompactor`, `CompactKVCache`, and the per-layer beta attention interface. One document-level OPD trainer supports full-context and evidence-context teachers; the backbone stays frozen and every document shares one question-independent compact cache across its questions. Preserve the existing offline MCQ trainer.

**Stack:** Python 3.12, Torch 2.10.0, Transformers 4.57.6, Hugging Face Hub/Datasets, pytest.

**Authorized scope:** User supplied the design and requested integration, downloads and smoke tests. Execute inline in the existing clean STILL checkout; no full training run or remote publication.

- [x] Validate Hugging Face Qwen3 RoPE and chat-template behavior with failing regressions; fix only the affected shared helpers.
- [x] Add `src/still/data/qasper.py`: group original documents/questions, retain complete source-matched evidence annotations, keep answers exclusively as evaluation metadata, filter whole overlong documents rather than clipping evidence. Preserve official splits and report filtering.
- [x] Add `src/still/train/opd.py`: full-vocabulary equal-weight JSD on sampled answer positions, student sampling under no-grad, detached teacher, differentiable replay, mean over tokens then questions, one document cache per step, explicit physical cache positions and original RoPE continuation positions, fresh cache containers for every branch/question.
- [x] Add regression tests with a tiny real Qwen3 model for teacher routing, branch isolation, query-independent shared cache, finite nonzero compactor gradients and frozen backbone. Cross-check the JSD math with the existing OPSD implementation without importing its TRL/DeepSpeed stack.
- [x] Add model/data preparation and one comparison CLI: exact Qwen3-4B checkpoint, 512 slots by default, matched seed/initialization/sampling settings, checkpoint save/load, before/after evaluation with question- and document-averaged answer F1, and paired teacher diagnostics on the same initial student trajectories.
- [x] Download model/tokenizer/config and original QASPER splits, run repository tests, tiny smoke, real-model multi-question backward/update smoke, and 4K–8K document-length checks. Save exact commands and machine-readable outcomes; do not interpret smoke scores as research conclusions.

## Acceptance evidence

`pytest tests` must pass. Real smoke must record both teacher modes, full per-layer compact shapes, source/evidence lengths, normalized finite JSD, positive finite gradient norms, changed compactor state, unchanged backbone state, fresh cache isolation, disabled thinking, teacher quality diagnostics and peak GPU memory. Evaluation must share one cache across questions in a document and use held-out documents. All commands and source revisions must be documented for repeatability.

## Implementation decisions

- Baseline means full-context-teacher KV OPD, with the same frozen backbone and compactor as evidence OPD. The existing solution-privileged OPSD trainer updates model weights and is not this controlled baseline.
- Reuse the existing two-block compactor exactly (internal dimension `2 * head_dim`, 256 for Qwen3-4B); do not add an FFN absent from this reproduction. Document this deviation from the proposed architecture.
- Keep question sampling and training loss normalization separate from evaluation answer labels. Source matching checks evidence provenance, not semantic sufficiency; do not claim automatic verification proves sufficiency.

## Final verification

- 45 repository tests passed. Modified Python files pass Ruff; git diff --check is clean.
- Official evaluator cross-check covers all 2072 retained QASPER questions.
- Real Qwen3-4B, 512 slots, two questions: both modes pass at 4131 and 8147 source tokens. Full backbone SHA256 is unchanged; all 36 compactor layers have finite nonzero gradients; checkpoints round-trip; initial trajectories match.
- Independent read-only review found EOS-set handling and mixed Unanswerable evaluation references; both were reproduced, fixed, and regression tested.
- Complete commands, artifacts and limits are in docs/opd_integration.md. No substantive training or research-effectiveness claim.
