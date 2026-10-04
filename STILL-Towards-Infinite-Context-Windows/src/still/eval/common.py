import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

SYSTEM_PROMPT = """Please answer the user's question using only the provided context.

<context>
{context}
</context>

Follow the requested answer format exactly. Do not emit <think> tags or chain-of-thought."""

class EvalRecord(BaseModel):
    """Normalized evaluation row stored for every benchmark prediction."""
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    method: str
    prediction: str
    gold: list[str]
    exact_match: bool
    canonical_kv_bytes: int
    compression_ratio: float
    prefill_ms: float | None = None
    decode_tokens_per_second: float | None = None
    total_latency_ms: float | None = None
    prompt_tokens: int
    completion_tokens: int
    metadata: dict[str, Any]


def decode_finish_reason(
    *,
    generated_ids: list[int],
    eos_token_id: int | None,
    max_completion_tokens: int,
) -> str:
    """Label why greedy decoding stopped for one evaluation row."""
    if not generated_ids:
        return "no_tokens"
    if eos_token_id is not None and generated_ids[-1] == eos_token_id:
        return "eos_token"
    if len(generated_ids) >= max_completion_tokens:
        return "max_completion_tokens"
    return "stopped"


def build_decode_debug_metadata(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    generated_ids: list[int],
    normalized_prediction: str,
    eos_token_id: int | None,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Capture raw decode details so latency and normalization artifacts stay inspectable."""
    raw_prediction = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return {
        "raw_prediction": raw_prediction,
        "generated_token_ids": generated_ids,
        "normalized_prediction": normalized_prediction,
        "finish_reason": decode_finish_reason(
            generated_ids=generated_ids,
            eos_token_id=eos_token_id,
            max_completion_tokens=max_completion_tokens,
        ),
        "prediction_mode": row.get("prediction_mode"),
    }


def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation prompts from JSONL."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}.")
    return rows


def write_eval_records(path: str | Path, records: list[EvalRecord]) -> None:
    """Persist evaluation records as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build the standard full-context chat messages for one evaluation row."""
    user_prompt = f"/no_think\n{row['query']}\n\n{row['answer_prompt']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=row["context"])},
        {"role": "user", "content": user_prompt},
    ]


def build_compact_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build the user-side prompt used when context lives in a compact cache artifact."""
    user_prompt = f"/no_think\n{row['query']}\n\n{row['answer_prompt']}"
    return [{"role": "user", "content": user_prompt}]


def normalize_generated_text_for_row(row: dict[str, Any], text: str) -> str:
    """Clean a raw generation and coerce MCQ rows down to a letter prediction."""
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", " ").replace("</think>", " ")
    cleaned = re.sub(r"^(?:assistant:\s*)+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if row.get("prediction_mode") == "mcq_letter":
        match = re.search(r"\b([A-D])\b", cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return ""
    return cleaned


def canonical_kv_bytes(
    *,
    num_tokens: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Compute the canonical byte size of a dense KV cache for one prompt length."""
    return (
        num_tokens
        * num_hidden_layers
        * num_key_value_heads
        * head_dim
        * 2
        * dtype_bytes
    )


def normalize_prediction(text: str) -> list[str]:
    """Normalize free-form answers before exact-match comparison."""
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", " ").replace("</think>", " ")
    number_matches = re.findall(r"\d+", cleaned)
    if number_matches:
        return sorted(number_matches)
    return [re.sub(r"\s+", " ", cleaned).strip().lower()]


def exact_match(prediction: str, gold: list[str]) -> bool:
    """Compare a prediction against all gold answers under repo-specific normalization rules."""
    normalized_gold: list[str] = []
    for item in gold:
        normalized_gold.extend(normalize_prediction(str(item)))
    return normalize_prediction(prediction) == sorted(normalized_gold)
