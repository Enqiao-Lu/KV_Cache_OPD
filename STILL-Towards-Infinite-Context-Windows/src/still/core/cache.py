from dataclasses import dataclass
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class AttentionShape:
    """Minimal attention-shape metadata needed for canonical KV accounting."""
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype_bytes: int = 2


def infer_attention_shape(model) -> AttentionShape:
    """Infer the KV-cache tensor shape from a Hugging Face model config."""
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return AttentionShape(
        num_hidden_layers=config.num_hidden_layers,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=head_dim,
    )


def normalize_past_key_values(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Convert cache objects into a legacy list of key/value tensors per layer."""
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    return [(layer[0], layer[1]) for layer in past_key_values]


class CompactKVCache:
    """Serializable compact cache bundle holding per-layer keys, values, and beta biases."""
    def __init__(
        self,
        *,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        biases: list[torch.Tensor] | None = None,
        metadata: dict[str, object] | None = None,
        detach_tensors: bool = True,
    ) -> None:
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same number of layers.")
        if not keys:
            raise ValueError("CompactKVCache requires at least one layer.")
        if biases is not None and len(biases) != len(keys):
            raise ValueError("biases must match the number of layers.")
        if detach_tensors:
            self.keys = [tensor.detach().clone() for tensor in keys]
            self.values = [tensor.detach().clone() for tensor in values]
        else:
            self.keys = list(keys)
            self.values = list(values)
        self.biases = (
            [tensor.detach().clone() for tensor in biases]
            if biases is not None
            else [torch.zeros_like(key[..., 0]) for key in keys]
        )
        if not detach_tensors and biases is not None:
            self.biases = list(biases)
        self.metadata = dict(metadata or {})

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def num_tokens(self) -> int:
        return int(self.keys[0].shape[-2])

    def to(self, device: str | torch.device) -> "CompactKVCache":
        self.keys = [tensor.to(device) for tensor in self.keys]
        self.values = [tensor.to(device) for tensor in self.values]
        self.biases = [tensor.to(device) for tensor in self.biases]
        return self

    def as_legacy_past_key_values(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple((key, value) for key, value in zip(self.keys, self.values, strict=True))

    def as_cache(self, model_config) -> DynamicCache:
        return DynamicCache(ddp_cache_data=self.as_legacy_past_key_values(), config=model_config)

    def canonical_kv_bytes(self) -> int:
        total = 0
        for key, value in zip(self.keys, self.values, strict=True):
            total += key.numel() * key.element_size()
            total += value.numel() * value.element_size()
        return total

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "keys": [tensor.detach().cpu() for tensor in self.keys],
                "values": [tensor.detach().cpu() for tensor in self.values],
                "biases": [tensor.detach().cpu() for tensor in self.biases],
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "CompactKVCache":
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        cache = cls(
            keys=checkpoint["keys"],
            values=checkpoint["values"],
            biases=checkpoint.get("biases"),
            metadata=checkpoint.get("metadata"),
        )
        if device is not None:
            cache.to(device)
        return cache
