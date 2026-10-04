import torch

from still.core.still import StillCompactor


def _fake_past() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randn(1, 2, 16, 8, dtype=torch.float32),
            torch.randn(1, 2, 16, 8, dtype=torch.float32),
        ),
        (
            torch.randn(1, 2, 16, 8, dtype=torch.float32),
            torch.randn(1, 2, 16, 8, dtype=torch.float32),
        ),
    ]


class _Config:
    num_hidden_layers = 2
    hidden_size = 16
    num_attention_heads = 2
    num_key_value_heads = 2
    head_dim = 8
    rope_theta = 10000.0


def test_still_compactor_reduces_sequence_length() -> None:
    compactor = StillCompactor.from_model_config(_Config, num_latents=4)
    compact_cache = compactor(_fake_past())
    assert compact_cache.num_layers == 2
    assert compact_cache.num_tokens == 4
    assert compact_cache.keys[0].shape == (1, 2, 4, 8)
    assert compact_cache.values[0].shape == (1, 2, 4, 8)
    assert compact_cache.biases[0].shape == (1, 2, 4)


def test_still_identity_init_uses_matching_qk_bias_direction() -> None:
    compactor = StillCompactor.from_model_config(_Config, num_latents=4)
    cross_attn = compactor.layers[0].blocks[0].cross_attn
    q_bias = cross_attn.q_proj.bias.detach()
    k_bias = cross_attn.k_proj.bias.detach()
    assert torch.count_nonzero(q_bias) > 0
    assert torch.allclose(k_bias, q_bias * 10.0)


def test_still_identity_init_routes_latents_monotonically_by_position() -> None:
    compactor = StillCompactor.from_model_config(_Config, num_latents=4)
    layer = compactor.layers[0]
    _, _, _, weights = layer(_fake_past()[0][0], _fake_past()[0][1], return_attention_weights=True)
    mean_weights = weights.squeeze(0).mean(dim=0)
    argmax_positions = mean_weights.argmax(dim=-1).tolist()
    assert argmax_positions == sorted(argmax_positions)
