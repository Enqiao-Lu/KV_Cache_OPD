import torch

from still.attention_bias import _merge_still_bias


class _Layer:
    layer_idx = 0
    num_key_value_groups = 2

    class config:
        num_attention_heads = 4


class _Past:
    def __init__(self, seq_length: int) -> None:
        self._seq_length = seq_length

    def get_seq_length(self) -> int:
        return self._seq_length


def test_merge_still_bias_repeats_kv_heads_and_zero_pads_noncompact_tokens() -> None:
    hidden_states = torch.zeros(1, 3, 8)
    attention_mask = torch.zeros(1, 1, 3, 8)
    still_layer_biases = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])]

    merged = _merge_still_bias(
        layer_module=_Layer(),
        attention_mask=attention_mask,
        hidden_states=hidden_states,
        past_key_values=_Past(seq_length=5),
        still_layer_biases=still_layer_biases,
    )

    assert merged is not None
    assert merged.shape == (1, 4, 3, 8)
    assert torch.allclose(merged[0, 0, :, :2], torch.tensor([[1.0, 2.0]]).expand(3, 2))
    assert torch.allclose(merged[0, 1, :, :2], torch.tensor([[1.0, 2.0]]).expand(3, 2))
    assert torch.allclose(merged[0, 2, :, :2], torch.tensor([[3.0, 4.0]]).expand(3, 2))
    assert torch.allclose(merged[0, 3, :, :2], torch.tensor([[3.0, 4.0]]).expand(3, 2))
    assert torch.count_nonzero(merged[..., 2:]) == 0


def test_merge_still_bias_converts_boolean_mask_to_additive_float_mask() -> None:
    hidden_states = torch.zeros(1, 1, 8, dtype=torch.float32)
    attention_mask = torch.tensor([[[[True, True, False]]]])
    still_layer_biases = [torch.tensor([[[0.5, -0.25]]], dtype=torch.float32)]

    merged = _merge_still_bias(
        layer_module=_Layer(),
        attention_mask=attention_mask,
        hidden_states=hidden_states,
        past_key_values=_Past(seq_length=2),
        still_layer_biases=still_layer_biases,
    )

    assert merged is not None
    assert merged.dtype == torch.float32
    assert torch.allclose(merged[0, 0, 0, :2], torch.tensor([0.5, -0.25]))
    assert merged[0, 0, 0, 2] < -1e20
