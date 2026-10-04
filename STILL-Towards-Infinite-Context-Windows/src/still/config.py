import os
from dataclasses import dataclass
from typing import Literal


RunMode = Literal["smoke", "full"]


@dataclass(frozen=True)
class CompatibilityMatrix:
    """Pinned runtime and model versions used by the benchmark scripts."""
    python: str
    torch: str
    transformers: str
    vllm: str
    model_id: str


DEFAULT_MODEL_ID = os.environ.get("STILL_MODEL_ID", "Qwen/Qwen3-4B")
DEFAULT_VLLM_PORT = int(os.environ.get("STILL_VLLM_PORT", "8000"))
DISALLOWED_GPUS = (6, 7)
PREFERRED_GPU_ORDER = (3, 5)
GPU_USED_MEMORY_LIMIT_MIB = {
    "smoke": 1024,
    "full": 4096,
}
GPU_UTILIZATION_LIMIT = {
    "smoke": 0,
    "full": 0,
}
REQUIRED_PORTS = {
    "smoke": DEFAULT_VLLM_PORT,
    "full": DEFAULT_VLLM_PORT,
}
TOKENIZER_PROBE_TEXT = "The quick brown fox jumps over the lazy dog."

DEFAULT_MATRIX = CompatibilityMatrix(
    python="3.12",
    torch=os.environ.get("STILL_TORCH_VERSION", "2.10.0"),
    transformers=os.environ.get("STILL_TRANSFORMERS_VERSION", "4.57.6"),
    vllm=os.environ.get("STILL_VLLM_VERSION", "0.19.0"),
    model_id=DEFAULT_MODEL_ID,
)


def resolve_wandb_mode(mode: RunMode, env: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Choose the correct Weights & Biases mode for smoke versus full runs."""
    env = dict(os.environ if env is None else env)
    warnings: list[str] = []
    api_key = env.get("WANDB_API_KEY")

    if mode == "smoke":
        if not api_key:
            warnings.append("WANDB_API_KEY is not set; using offline mode for smoke runs.")
        return "offline", warnings

    if not api_key:
        raise RuntimeError("WANDB_API_KEY is required for full runs.")
    return "online", warnings
