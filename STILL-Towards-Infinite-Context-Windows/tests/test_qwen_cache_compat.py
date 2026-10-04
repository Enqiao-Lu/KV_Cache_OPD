import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb

from still.core.still import apply_rope


def test_compactor_rope_matches_qwen3_and_inverts_cached_keys():
    config = Qwen3Config(hidden_size=32, num_attention_heads=4, head_dim=8)
    rotary = Qwen3RotaryEmbedding(config)
    keys = torch.randn(1, 2, 7, 8)
    positions = torch.arange(7)
    cos, sin = rotary(keys, positions[None])
    _, expected = apply_rotary_pos_emb(keys, keys, cos, sin)
    actual = apply_rope(keys, positions, theta=config.rope_theta)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        apply_rope(expected, positions, theta=config.rope_theta, inverse=True), keys
    )


def test_chat_split_disables_qwen3_thinking():
    from transformers import AutoTokenizer

    from still.chat import encode_system_prefix, encode_user_continuation

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    prefix = encode_system_prefix(tokenizer, "Document context.")
    continuation = encode_user_continuation(
        tokenizer, system_prompt="Document context.", user_message="Question?"
    )
    expected = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Document context."},
            {"role": "user", "content": "Question?"},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert prefix[0].tolist() + continuation[0].tolist() == expected
    assert tokenizer.decode(continuation[0]).endswith("<think>\n\n</think>\n\n")
