from typing import Any, Callable

import torch


def _repeat_bias_heads(layer_module, bias: torch.Tensor) -> torch.Tensor:
    """Expand KV-head biases to attention-head shape when grouped-query attention is used."""
    # Some decoder architectures have fewer KV heads than attention heads. The
    # compact cache stores one beta value per KV head, so here we expand it to
    # the full attention-head shape expected by the attention kernel.
    num_attention_heads = int(layer_module.config.num_attention_heads)
    if bias.shape[1] == num_attention_heads:
        return bias
    return bias.repeat_interleave(layer_module.num_key_value_groups, dim=1)


def _merge_still_bias(
    *,
    layer_module,
    attention_mask: torch.Tensor | None,
    hidden_states: torch.Tensor,
    past_key_values,
    still_layer_biases: list[torch.Tensor] | None,
) -> torch.Tensor | None:
    """Inject STILL's additive beta bias into the model attention mask for one layer.

    ``still_layer_biases`` contains the per-layer ``beta`` tensors produced by
    the compactor. Beta does not add new content to the cache. Instead, it
    shifts the attention logits over the compact tokens so the frozen model can
    upweight or downweight particular latent slots after compression.

    Intuitively:
    - compact keys/values say *what* information each latent slot stores
    - beta says *how much* the model should trust or prefer each latent slot

    This is why beta is merged into the additive attention mask/logits rather
    than being appended to the KV tensors themselves.
    """
    if not still_layer_biases:
        return attention_mask

    layer_bias = still_layer_biases[layer_module.layer_idx]
    if layer_bias is None:
        return attention_mask

    if layer_bias.dim() != 3:
        raise ValueError("STILL attention bias must have shape [batch, kv_heads, compact_tokens].")

    batch_size, _, compact_len = layer_bias.shape
    query_len = hidden_states.shape[1]
    # The attention kernel sees the whole effective key axis: compact cache
    # positions from the prefix plus any new decode-time tokens appended later.
    if attention_mask is not None:
        total_key_len = int(attention_mask.shape[-1])
    else:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        total_key_len = past_seen_tokens + query_len
    trailing_zeros = total_key_len - compact_len
    if trailing_zeros < 0:
        raise ValueError("STILL attention bias is longer than the effective cache length.")

    bias = _repeat_bias_heads(layer_module, layer_bias)
    # Broadcast beta across the current query positions so each fresh token sees
    # the same compact-cache preference offsets.
    bias = bias[:, :, None, :].expand(batch_size, bias.shape[1], query_len, compact_len)
    if trailing_zeros:
        # Beta is defined only over the compact prefix. Newly generated decode
        # tokens should not inherit any extra preference, so append zeros.
        bias = torch.cat(
            [
                bias,
                torch.zeros(
                    batch_size,
                    bias.shape[1],
                    query_len,
                    trailing_zeros,
                    device=bias.device,
                    dtype=bias.dtype,
                ),
            ],
            dim=-1,
        )

    if attention_mask is None:
        return bias.to(hidden_states.dtype)
    if attention_mask.dtype == torch.bool:
        # HF attention often uses a boolean mask. Convert it to the additive
        # float form before adding beta, otherwise beta would be silently
        # coerced away by boolean semantics.
        additive_mask = torch.zeros(
            attention_mask.shape,
            device=attention_mask.device,
            dtype=hidden_states.dtype,
        )
        additive_mask = additive_mask.masked_fill(~attention_mask, torch.finfo(hidden_states.dtype).min)
        attention_mask = additive_mask
    return attention_mask + bias.to(attention_mask.dtype)


def enable_still_attention_bias(model) -> None:
    """Patch a decoder-only Hugging Face model so it accepts STILL layer biases.

    The model is not given a new persistent attribute containing beta. Instead,
    this function wraps each decoder layer's self-attention forward method so a
    caller can pass ``still_layer_biases=...`` as a per-forward keyword
    argument. The wrapper merges the correct layer's beta tensor into that
    layer's attention mask just before the original attention code runs.
    """
    if getattr(model, "_still_attention_bias_enabled", False):
        return

    decoder_layers = getattr(getattr(model, "model", None), "layers", None)
    if decoder_layers is None:
        raise TypeError("enable_still_attention_bias expects a decoder-only causal LM with model.layers.")

    def _wrap_attention(self_attn, original_forward: Callable[..., Any]):
        def forward_with_still_bias(
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            # Pull the shared list of per-layer beta tensors off the kwargs and
            # select the slice for this specific decoder layer.
            still_layer_biases = kwargs.pop("still_layer_biases", None)
            attention_mask = _merge_still_bias(
                layer_module=self_attn,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                still_layer_biases=still_layer_biases,
            )
            return original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        return forward_with_still_bias

    for decoder_layer in decoder_layers:
        self_attn = decoder_layer.self_attn
        original_forward: Callable[..., Any] = self_attn.forward
        # Monkey-patch the attention module once so later training/eval calls
        # can thread beta through the normal model(...) API.
        self_attn.forward = _wrap_attention(self_attn, original_forward)

    model._still_attention_bias_enabled = True
