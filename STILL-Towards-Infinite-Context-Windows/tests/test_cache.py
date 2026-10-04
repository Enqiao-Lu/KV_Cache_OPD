import torch

from still.core import CompactKVCache


def test_compact_cache_roundtrip(tmp_path) -> None:
    cache = CompactKVCache(
        keys=[torch.randn(1, 2, 4, 8)],
        values=[torch.randn(1, 2, 4, 8)],
        biases=[torch.randn(1, 2, 4)],
        metadata={"name": "demo"},
    )
    path = tmp_path / "cache.pt"
    cache.save(path)
    restored = CompactKVCache.load(path)
    assert restored.metadata["name"] == "demo"
    assert restored.keys[0].shape == (1, 2, 4, 8)
    assert restored.biases[0].shape == (1, 2, 4)
    assert restored.canonical_kv_bytes() == cache.canonical_kv_bytes()
