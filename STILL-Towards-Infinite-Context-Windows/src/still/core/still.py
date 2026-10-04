"""Core STILL compactor implementation.

This module implements the reusable neural KV-cache compactor used throughout the
benchmark. The high-level idea is:

1. Run the frozen language model once to obtain the full KV cache for a long
   context.
2. For each transformer layer, compress that dense cache into a much smaller
   fixed-size representation.
3. Save the compact keys, compact values, and an additive attention bias
   (``beta``) so later queries can reuse that compressed memory.

The compression itself is RoPE-aware. Cached keys are first "unrotated" so the
perceiver blocks operate in a position-neutral space. After the latent
computation is finished, the compact keys are rotated back at their new latent
positions so the frozen model can consume them as if they were normal KV-cache
entries.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from still.core.cache import CompactKVCache, normalize_past_key_values


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply the standard RoPE half-rotation to the last dimension."""
    # Hugging Face Qwen3/Llama rotate halves, not adjacent even/odd pairs.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_frequencies(dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """Construct the inverse-frequency vector used by RoPE."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))


def _rope_cos_sin(
    positions: torch.Tensor,
    *,
    dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cosine and sine tables for a set of token positions."""
    inv_freq = _rope_frequencies(dim, theta, positions.device)
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.float32), emb.sin().to(torch.float32)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply or invert RoPE for arbitrary positions on the last hidden dimension.

    Args:
        x: Tensor whose final dimension is the RoPE head dimension.
        positions: Absolute token positions associated with ``x``.
        theta: Standard RoPE base parameter from the model config.
        inverse: When ``True``, undo an existing RoPE rotation instead of
            applying one. STILL uses this to move cached keys into an
            unrotated space before compression.
    """
    cos, sin = _rope_cos_sin(positions, dim=x.shape[-1], theta=theta)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if inverse:
        sin = -sin
    x_float = x.to(torch.float32)
    return ((x_float * cos) + (_rotate_half(x_float) * sin)).to(x.dtype)


class SelfAttentionBlock(nn.Module):
    """Latent-only self-attention used inside each perceiver block.

    The cross-attention step lets each latent pull information from the dense
    KV sequence. The follow-up self-attention step lets latents coordinate with
    one another before the next block or the final projections.
    """
    def __init__(self, dim: int, *, zero_output_init: bool = True) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        if zero_output_init:
            nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix information across latent slots.

        ``x`` has shape ``[num_heads, num_latents, latent_dim]`` in this repo.
        The code treats each KV head independently and lets the latent slots for
        that head attend to one another.
        """
        scale = 1.0 / math.sqrt(x.shape[-1])
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        return self.out_proj(torch.matmul(weights, v))


class CrossAttentionBlock(nn.Module):
    """RoPE-aware latent-to-KV cross-attention block.

    This is the part of STILL that reads the dense cache. Learned latent slots
    act as the queries and the concatenated ``[unrotated key | value]`` tensor
    acts as the key/value memory.

    The initialization is intentionally structured:
    - ``q_proj`` and ``k_proj`` weights start at zero and rely on bias vectors
      to create a simple locality prior.
    - ``v_proj`` starts as identity so the block can initially behave like a
      near-copy path.
    - ``out_proj`` is identity only for the first block, then zero for later
      blocks so the architecture starts in a stable regime.
    """
    def __init__(
        self,
        *,
        dim: int,
        rope_theta: float,
        active_init: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self._reset_parameters(active_init=active_init)

    def _reset_parameters(self, *, active_init: bool) -> None:
        """Initialize the cross-attention path with blog-style routing priors."""
        query_direction = torch.ones(self.dim, dtype=self.q_proj.bias.dtype)
        query_direction = query_direction / query_direction.norm(p=2)
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.k_proj.weight)
        self.q_proj.bias.data.copy_(query_direction)
        self.k_proj.bias.data.copy_(query_direction * 10.0)
        nn.init.eye_(self.v_proj.weight)
        if active_init:
            nn.init.eye_(self.out_proj.weight)
        else:
            nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        latents: torch.Tensor,
        kv_input: torch.Tensor,
        *,
        latent_positions: torch.Tensor,
        token_positions: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Let learned latents read from the dense per-token KV input.

        Args:
            latents: ``[num_heads, num_latents, latent_dim]``
            kv_input: ``[num_heads, seq_len, latent_dim]`` where the final
                dimension is ``[unrotated key | value]``.
            latent_positions: Evenly spaced positions assigned to the latent
                slots so RoPE remains meaningful after compression.
            token_positions: Original positions of the dense cache entries.
        """
        q = apply_rope(self.q_proj(latents), latent_positions, theta=self.rope_theta)
        k = apply_rope(self.k_proj(kv_input), token_positions, theta=self.rope_theta)
        v = self.v_proj(kv_input)
        scale = 1.0 / math.sqrt(self.dim)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        outputs = self.out_proj(torch.matmul(weights, v))
        if return_attention_weights:
            return outputs, weights
        return outputs


class PerceiverBlock(nn.Module):
    """One perceiver block in the layer compactor.

    Each block performs:
    1. latent-to-dense cross-attention
    2. residual connection + RMSNorm
    3. latent self-attention
    4. residual connection + RMSNorm

    The repo uses two of these blocks per layer, which corresponds to the
    README notation ``Z_l^(0) -> Z_l^(1) -> Z_l^(2)``.
    """
    def __init__(
        self,
        *,
        dim: int,
        rope_theta: float,
        active_init: bool,
    ) -> None:
        super().__init__()
        self.cross_attn = CrossAttentionBlock(dim=dim, rope_theta=rope_theta, active_init=active_init)
        self.cross_norm = nn.RMSNorm(dim)
        self.self_attn = SelfAttentionBlock(dim, zero_output_init=True)
        self.self_norm = nn.RMSNorm(dim)

    def forward(
        self,
        latents: torch.Tensor,
        kv_input: torch.Tensor,
        *,
        latent_positions: torch.Tensor,
        token_positions: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Run one full perceiver update on the latent state."""
        if return_attention_weights:
            cross_out, weights = self.cross_attn(
                latents,
                kv_input,
                latent_positions=latent_positions,
                token_positions=token_positions,
                return_attention_weights=True,
            )
        else:
            cross_out = self.cross_attn(
                latents,
                kv_input,
                latent_positions=latent_positions,
                token_positions=token_positions,
            )
            weights = None
        latents = self.cross_norm(latents + cross_out)
        latents = self.self_norm(latents + self.self_attn(latents))
        if return_attention_weights:
            return latents, weights
        return latents


class StillLayerCompactor(nn.Module):
    """Compress one transformer's layer dense cache into fixed-size memory.

    Input:
    - keys: ``[batch, kv_heads, seq_len, head_dim]``
    - values: ``[batch, kv_heads, seq_len, head_dim]``

    Output:
    - compact keys: ``[batch, kv_heads, num_latents, head_dim]``
    - compact values: ``[batch, kv_heads, num_latents, head_dim]``
    - beta bias: ``[batch, kv_heads, num_latents]``

    ``beta`` is an additive attention term consumed later at inference time so
    the frozen model can compensate for information lost during compression.
    """
    def __init__(
        self,
        *,
        head_dim: int,
        num_latents: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        latent_dim = head_dim * 2
        self.num_latents = num_latents
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.latent_dim = latent_dim

        self.latents = nn.Parameter(torch.zeros(num_latents, latent_dim))
        self.blocks = nn.ModuleList(
            [
                PerceiverBlock(
                    dim=latent_dim,
                    rope_theta=rope_theta,
                    active_init=(idx == 0),
                )
                for idx in range(2)
            ]
        )
        self.key_head = nn.Linear(latent_dim, head_dim, bias=False)
        self.value_head = nn.Linear(latent_dim, head_dim, bias=False)
        self.bias_head = nn.Linear(latent_dim, 1, bias=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize the layer compactor close to an identity readout.

        The latent table starts at zero. The output projections are wired so
        the first half of the latent vector maps to keys and the second half
        maps to values. This makes the value path easy to interpret and helps
        debugging because early checkpoints often resemble a crude copy-based
        compressor rather than arbitrary noise.
        """
        # The output heads start as a near-identity split of [unrotated key | value].
        nn.init.zeros_(self.latents)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

        self.key_head.weight.data.zero_()
        self.key_head.weight.data[:, : self.head_dim] = torch.eye(self.head_dim)
        self.value_head.weight.data.zero_()
        self.value_head.weight.data[:, self.head_dim :] = torch.eye(self.head_dim)

    def _latent_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Assign evenly spaced absolute positions to the latent slots.

        The compact cache needs positions because the frozen model expects RoPE-
        rotated keys. Using evenly spaced latent positions is the repo's simple
        deterministic scheme for saying "latent slot i stands in for this part
        of the original sequence."
        """
        if self.num_latents == 1:
            return torch.zeros(1, device=device, dtype=torch.long)
        values = torch.linspace(
            0,
            max(seq_len - 1, 0),
            steps=self.num_latents,
            device=device,
            dtype=torch.float32,
        )
        return values.round().to(torch.long)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compress one layer's dense cache into compact keys, values, and beta.

        The computation order is:
        1. undo RoPE on the dense keys
        2. concatenate ``[key_unrotated | value]`` per token
        3. expand the learned latent table across KV heads
        4. run two perceiver blocks
        5. project the final latent state into compact keys, values, and beta
        6. reapply RoPE to compact keys at the latent positions
        """
        if keys.dim() != 4 or values.dim() != 4:
            raise ValueError("Expected keys and values shaped [batch, heads, tokens, dim].")
        batch_size, num_heads, seq_len, head_dim = keys.shape
        if batch_size != 1:
            raise ValueError("STILL currently supports batch_size=1.")
        if head_dim != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, found {head_dim}.")

        output_dtype = keys.dtype
        module_dtype = self.blocks[0].cross_attn.q_proj.weight.dtype
        token_positions = torch.arange(seq_len, device=keys.device, dtype=torch.long)
        latent_positions = self._latent_positions(seq_len, keys.device)
        # The blog's RoPE fix is unrotate -> compress with perceiver RoPE -> rerotate.
        unrotated_keys = apply_rope(keys, token_positions, theta=self.rope_theta, inverse=True)
        # The perceiver reads both key and value information at once, so the
        # last dimension is doubled here.
        kv_input = torch.cat([unrotated_keys, values], dim=-1)
        kv_input = kv_input.squeeze(0).reshape(num_heads, seq_len, head_dim * 2).to(module_dtype)
        # One latent table is shared across heads, then expanded so each head
        # compresses its own sequence independently.
        latents = self.latents.unsqueeze(0).expand(num_heads, -1, -1).to(module_dtype)

        weights = None
        for block_idx, block in enumerate(self.blocks):
            if return_attention_weights and block_idx == 0:
                latents, weights = block(
                    latents,
                    kv_input,
                    latent_positions=latent_positions,
                    token_positions=token_positions,
                    return_attention_weights=True,
                )
            else:
                latents = block(
                    latents,
                    kv_input,
                    latent_positions=latent_positions,
                    token_positions=token_positions,
                )

        # ``latents`` is now ``Z_l^(2)`` in the README notation.
        compact_keys = self.key_head(latents)
        compact_values = self.value_head(latents)
        compact_biases = self.bias_head(latents).squeeze(-1)
        # Compact keys must be rerotated at their evenly spaced latent positions before the LLM consumes them.
        compact_keys = apply_rope(compact_keys, latent_positions, theta=self.rope_theta)
        return (
            compact_keys.unsqueeze(0).to(output_dtype),
            compact_values.unsqueeze(0).to(output_dtype),
            compact_biases.unsqueeze(0).to(output_dtype),
            weights.unsqueeze(0).to(output_dtype),
        ) if return_attention_weights else (
            compact_keys.unsqueeze(0).to(output_dtype),
            compact_values.unsqueeze(0).to(output_dtype),
            compact_biases.unsqueeze(0).to(output_dtype),
        )


@dataclass(frozen=True)
class StillBuildResult:
    """Metadata returned when a trained STILL compactor builds a cache artifact.

    This dataclass is mostly a convenience container for callers that want both
    the compact cache and the timing/source-token metadata from a build step.
    """
    cache: CompactKVCache
    build_seconds: float
    num_source_tokens: int


class StillCompactor(nn.Module):
    """Top-level STILL compactor spanning every transformer layer.

    The frozen language model exposes a stack of per-layer KV tensors. This
    module holds one :class:`StillLayerCompactor` per transformer layer and
    applies them independently, returning a :class:`CompactKVCache` that can be
    saved and later attached back to the same base model.
    """
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        head_dim: int,
        num_latents: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.layers = nn.ModuleList(
            [
                StillLayerCompactor(
                    head_dim=head_dim,
                    num_latents=num_latents,
                    rope_theta=rope_theta,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    @classmethod
    def from_model_config(cls, model_config, *, num_latents: int) -> "StillCompactor":
        """Construct the compactor directly from a Hugging Face model config."""
        head_dim = getattr(
            model_config,
            "head_dim",
            model_config.hidden_size // model_config.num_attention_heads,
        )
        rope_theta = float(getattr(model_config, "rope_theta", 10000.0))
        return cls(
            num_hidden_layers=model_config.num_hidden_layers,
            head_dim=head_dim,
            num_latents=num_latents,
            rope_theta=rope_theta,
        )

    def forward(self, past_key_values) -> CompactKVCache:
        """Compress the full per-layer cache stack into a serializable compact cache.

        ``past_key_values`` is expected to come directly from the frozen base
        model after prefilling the long context. The method normalizes that
        cache format, runs one layer compactor per transformer layer, and
        returns the compact memory artifact used later during STILL inference.
        """
        normalized = normalize_past_key_values(past_key_values)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        for layer_module, (layer_keys, layer_values) in zip(self.layers, normalized, strict=True):
            # Each transformer layer gets its own compact keys, values, and additive beta bias.
            compact_keys, compact_values, compact_biases = layer_module(layer_keys, layer_values)
            keys.append(compact_keys)
            values.append(compact_values)
            biases.append(compact_biases)
        # ``detach_tensors=False`` keeps the training graph alive when the
        # compactor is used inside the STILL loss. Callers that save the cache
        # later will explicitly clone/detach inside ``CompactKVCache.save``.
        return CompactKVCache(keys=keys, values=values, biases=biases, detach_tensors=False)
