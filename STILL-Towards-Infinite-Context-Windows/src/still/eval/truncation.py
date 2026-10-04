import json
from pathlib import Path

from transformers import AutoTokenizer

from still.config import DEFAULT_MATRIX
from still.data.common import write_jsonl
from still.eval.baseline import run_local_hf_matched_eval
from still.eval.common import load_eval_rows


def build_truncated_eval_rows(
    *,
    eval_path: str | Path,
    output_path: str | Path,
    context_token_budget: int,
) -> list[dict[str, object]]:
    """Trim each evaluation context down to a fixed token budget."""
    rows = load_eval_rows(eval_path)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    truncated_rows: list[dict[str, object]] = []
    for row in rows:
        token_ids = tokenizer.encode(row["context"], add_special_tokens=False)
        if len(token_ids) > context_token_budget:
            token_ids = token_ids[-context_token_budget:]
        truncated_rows.append(
            {
                **row,
                "context": tokenizer.decode(token_ids),
                "truncation_metadata": {
                    "context_token_budget": context_token_budget,
                    "original_context_tokens": len(
                        tokenizer.encode(row["context"], add_special_tokens=False)
                    ),
                    "retained_context_tokens": len(token_ids),
                },
            }
        )
    write_jsonl(Path(output_path), truncated_rows)
    return truncated_rows


def run_truncation_eval(
    *,
    eval_path: str | Path,
    output_path: str | Path,
    device: str,
    context_token_budget: int,
    max_samples: int | None = None,
    max_completion_tokens: int = 128,
) -> list:
    """Evaluate the truncation baseline by rebuilding rows then reusing the full-context path."""
    truncated_eval_path = Path(output_path).with_name("truncated_eval_rows.jsonl")
    build_truncated_eval_rows(
        eval_path=eval_path,
        output_path=truncated_eval_path,
        context_token_budget=context_token_budget,
    )
    records = run_local_hf_matched_eval(
        eval_path=truncated_eval_path,
        output_path=output_path,
        device=device,
        max_samples=max_samples,
        max_completion_tokens=max_completion_tokens,
    )
    for record in records:
        record.method = "truncation_hf_matched"
    # rewrite after method mutation
    Path(output_path).write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    return records
