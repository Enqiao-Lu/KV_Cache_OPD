import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from still.attention_bias import enable_still_attention_bias
from still.chat import chat_template_kwargs, encode_user_continuation
from still.config import DEFAULT_MATRIX
from still.core import CompactKVCache
from still.eval.common import (
    EvalRecord,
    SYSTEM_PROMPT,
    build_decode_debug_metadata,
    build_messages,
    canonical_kv_bytes,
    exact_match,
    load_eval_rows,
    normalize_generated_text_for_row,
    write_eval_records,
)


def _head_dim(model_config) -> int:
    """Infer a model's per-head KV dimension across config variants."""
    return getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )


def _sync_if_cuda(device: str) -> None:
    """Synchronize CUDA work so recorded timings reflect completed kernels."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def run_still_eval(
    *,
    eval_path: str | Path,
    cache_path: str | Path,
    output_path: str | Path,
    device: str,
    sample_id: str | None = None,
    max_samples: int | None = None,
    max_completion_tokens: int = 128,
    model=None,
    tokenizer=None,
) -> list[EvalRecord]:
    """Evaluate a prepared STILL compact cache against held-out prompts for one sample or split."""
    rows = load_eval_rows(eval_path)
    if sample_id is not None:
        rows = [row for row in rows if row["sample_id"] == sample_id]
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError("No evaluation rows selected for STILL evaluation.")

    owns_model = model is None or tokenizer is None
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = model or AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    if owns_model:
        model.to(device)
        model.eval()
    enable_still_attention_bias(model)
    compact_cache = CompactKVCache.load(cache_path, device=device)

    eos_token_id = tokenizer.eos_token_id
    records: list[EvalRecord] = []
    with torch.inference_mode():
        for row in rows:
            baseline_prompt = tokenizer.apply_chat_template(
                build_messages(row),
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs(),
            )
            baseline_prompt_tokens = len(tokenizer.encode(baseline_prompt, add_special_tokens=False))
            input_ids = encode_user_continuation(
                tokenizer,
                system_prompt=SYSTEM_PROMPT.format(context=row["context"]),
                user_message=f"/no_think\n{row['query']}\n\n{row['answer_prompt']}",
            ).to(device)

            _sync_if_cuda(device)
            started = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                past_key_values=compact_cache.as_cache(model.config),
                still_layer_biases=compact_cache.biases,
                use_cache=True,
            )
            _sync_if_cuda(device)
            prefill_ms = (time.perf_counter() - started) * 1000.0

            generated_ids: list[int] = []
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            _sync_if_cuda(device)
            decode_started = time.perf_counter()
            for _ in range(max_completion_tokens):
                token_id = int(next_token.item())
                generated_ids.append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    break
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    still_layer_biases=compact_cache.biases,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            _sync_if_cuda(device)
            decode_seconds = time.perf_counter() - decode_started

            raw_completion_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            completion_text = normalize_generated_text_for_row(row, raw_completion_text)
            decode_tokens_per_second = None
            if decode_seconds > 0 and generated_ids:
                decode_tokens_per_second = len(generated_ids) / decode_seconds

            baseline_bytes = canonical_kv_bytes(
                num_tokens=baseline_prompt_tokens,
                num_hidden_layers=model.config.num_hidden_layers,
                num_key_value_heads=model.config.num_key_value_heads,
                head_dim=_head_dim(model.config),
            )
            compact_bytes = compact_cache.canonical_kv_bytes()
            records.append(
                EvalRecord(
                    prompt_id=f"{row['sample_id']}::{row['row_hash']}",
                    method="still_hf_matched",
                    prediction=completion_text,
                    gold=[str(item) for item in row["answers"]],
                    exact_match=exact_match(completion_text, row["answers"]),
                    canonical_kv_bytes=compact_bytes,
                    compression_ratio=baseline_bytes / compact_bytes,
                    prefill_ms=prefill_ms,
                    decode_tokens_per_second=decode_tokens_per_second,
                    total_latency_ms=prefill_ms + (decode_seconds * 1000.0),
                    prompt_tokens=int(input_ids.shape[-1]),
                    completion_tokens=len(generated_ids),
                    metadata={
                        "sample_id": row["sample_id"],
                        "question_id": row.get("question_id"),
                        "query": row["query"],
                        "baseline_canonical_kv_bytes": baseline_bytes,
                        "device": device,
                        "cache_path": str(Path(cache_path).resolve()),
                        **build_decode_debug_metadata(
                            row=row,
                            tokenizer=tokenizer,
                            generated_ids=generated_ids,
                            normalized_prediction=completion_text,
                            eos_token_id=eos_token_id,
                            max_completion_tokens=max_completion_tokens,
                        ),
                    },
                )
            )

    write_eval_records(output_path, records)
    return records
