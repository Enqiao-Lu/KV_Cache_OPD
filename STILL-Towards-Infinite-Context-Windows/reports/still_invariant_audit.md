# STILL Invariant Audit

Generated: 2026-04-04

This matrix maps the Baseten blog requirements to the current local implementation status before the repair pass.

| Invariant | Expected Behavior | Current Status | Evidence |
| --- | --- | --- | --- |
| Chat-template-consistent prefix split | The cached system/context prefix and the user continuation must come from the same chat template tokenization used by baseline generation. | Correct | `src/still/chat.py`, `src/still/train/still.py`, `src/still/eval/still.py` now split system and user tokens using `encode_system_prefix` / `encode_user_continuation`. |
| On-policy teacher supervision | Teacher answer text and logits should come from the frozen teacher model, not copied labels. | Incorrect | `src/still/benchmarks/text_benchmark.py:generate_teacher_answers` currently discards client/model inputs and copies `expected_answer` directly. |
| RoPE unrotate / compress / rerotate | Keys should be inverse-RoPE'd before compression and re-RoPE'd at latent positions afterward. | Correct | `src/still/core/still.py:apply_rope`, `StillLayerCompactor.forward`. |
| No final latent normalization | The latent path should not apply a final normalization layer after perceiver-style attention. | Correct | `src/still/core/still.py` has no terminal norm in `StillLayerCompactor`. |
| Identity-style value path | Initialization should make the initial compactor behave like a near-copy route through the value path. | Partially correct | `v_proj`, `out_proj`, `key_head`, `value_head` are identity/near-identity initialized in `src/still/core/still.py`, but routing init is incomplete. |
| Matching q/k routing bias at init | Query and key projections need the blog's matching bias direction so each latent initially routes to nearby positions instead of globally averaging. | Incorrect | `src/still/core/still.py` sets `k_proj.bias = 10.0` but leaves `q_proj.bias = 0.0`. |
| Zero-init residual/output behavior | Residual/output projections that should start inactive must be zero-initialized so the identity route dominates initially. | Partially correct | `SelfAttentionBlock.out_proj` and `bias_head` are zero-initialized, but init-time locality remains unverified until probes are added. |
| Additive beta attention bias | The compact cache must include a learned additive attention-bias term applied at inference and training. | Partially correct | `src/still/attention_bias.py` injects beta now, but it was previously omitted and later coerced through a boolean mask path. |
| Float-valued mask semantics for beta | Beta must be added to float attention logits, not cast into a boolean mask. | Correct after local patch, previously incorrect | `src/still/attention_bias.py:_merge_still_bias` now converts boolean masks to additive float masks before beta injection. |
| Non-zero beta gradient flow | `bias_head` parameters must receive gradient during training and move away from zero in saved checkpoints. | Incorrect before float-mask patch, unverified after patch | Earlier checkpoints in `outputs/wiki_snapshot_phase1/runs/*/still_*` had zero `bias_head` norms; needs regression tests and a fresh deterministic training check. |
| Stable raw-output logging | Raw decode text and cleaned text should both be logged so collapse modes are visible. | Incorrect | `src/still/eval/still.py` currently only records cleaned completions in `prediction`. |
| Deterministic tiny repro harness | Tiny diagnostic runs should record seed, dataset hash, model/tokenizer revision, and stable metrics for before/after comparisons. | Incorrect | Current ad hoc diagnostic outputs do not have a dedicated deterministic manifest or raw-metric bundle. |

## Priority Failures

1. `generate_teacher_answers` is not on-policy and must be fixed before tuning.
2. `q_proj.bias` initialization does not match the identity-routing design described in the blog.
3. `bias_head` gradient flow must be revalidated after the float-mask fix.
4. The tiny diagnostic harness needs deterministic manifests and raw-output metrics.
