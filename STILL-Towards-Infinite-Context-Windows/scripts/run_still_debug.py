#!/usr/bin/env python3
"""Focused STILL-only training/debug runner.

This script exists so STILL quality debugging does not require paying for the
full four-method benchmark every time. It can either train a new compactor with
decode-aware checkpoint selection or reuse an existing checkpoint, then build
per-target caches and evaluate the held-out MCQ set.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from still.config import DEFAULT_MATRIX  # noqa: E402
from still.data.common import write_json  # noqa: E402
from still.eval.common import EvalRecord, SYSTEM_PROMPT, load_eval_rows  # noqa: E402
from still.eval.still import run_still_eval  # noqa: E402
from still.train import build_still_cache, train_still  # noqa: E402

DEFAULT_STILL_DATASET = (
    ROOT
    / "outputs"
    / "final_mcq_benchmark"
    / "runs"
    / "final_mcq_v1"
    / "prepared"
    / "still_mcq"
    / "combined_train_dataset.jsonl"
)
DEFAULT_EVAL_ROWS = (
    ROOT
    / "outputs"
    / "final_mcq_benchmark"
    / "runs"
    / "final_mcq_v1"
    / "prepared"
    / "still_mcq"
    / "combined_eval_rows.jsonl"
)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _system_prompt_for_sample(rows: list[dict[str, Any]], sample_id: str) -> str:
    sample_rows = [row for row in rows if row["sample_id"] == sample_id]
    if not sample_rows:
        raise ValueError(f"No eval rows found for sample_id={sample_id}")
    return SYSTEM_PROMPT.format(context=sample_rows[0]["context"])


def _summarize(records: list[EvalRecord], cache_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "accuracy": mean(int(record.exact_match) for record in records),
        "mean_query_prefill_ms": mean(record.prefill_ms for record in records if record.prefill_ms is not None),
        "mean_query_continuation_latency_ms": mean(
            record.total_latency_ms for record in records if record.total_latency_ms is not None
        ),
        "mean_prompt_tokens": mean(record.prompt_tokens for record in records),
        "mean_completion_tokens": mean(record.completion_tokens for record in records),
        "mean_target_preparation_seconds": mean(
            item["build_seconds"] for item in cache_manifest.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/debug STILL without running the full benchmark.")
    parser.add_argument("--train-dataset-path", default=str(DEFAULT_STILL_DATASET))
    parser.add_argument("--eval-rows-path", default=str(DEFAULT_EVAL_ROWS))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "still_debug"))
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--still-latents", type=int, default=1024)
    parser.add_argument("--still-steps", type=int, default=920)
    parser.add_argument("--still-learning-rate", type=float, default=2e-5)
    parser.add_argument("--still-kl-weight", type=float, default=0.0)
    parser.add_argument("--still-exact-token-ce-weight", type=float, default=1.0)
    parser.add_argument("--still-compactor-path")
    parser.add_argument("--validation-examples", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--max-completion-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-heldout-eval", action="store_true")
    args = parser.parse_args()

    train_dataset_path = Path(args.train_dataset_path)
    eval_rows_path = Path(args.eval_rows_path)
    _require_file(train_dataset_path, "STILL training dataset")
    _require_file(eval_rows_path, "evaluation rows")
    compactor_path = Path(args.still_compactor_path).resolve() if args.still_compactor_path else None
    if compactor_path is not None:
        _require_file(compactor_path, "STILL compactor checkpoint")

    output_root = Path(args.output_root)
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if compactor_path is None:
        train_summary = train_still(
            dataset_path=train_dataset_path,
            output_dir=run_dir / "train",
            device=args.device,
            num_latents=args.still_latents,
            learning_rate=args.still_learning_rate,
            steps=args.still_steps,
            seed=args.seed,
            validation_examples=args.validation_examples,
            validation_interval=args.validation_interval,
            kl_weight=args.still_kl_weight,
            exact_token_ce_weight=args.still_exact_token_ce_weight,
            max_completion_tokens=args.max_completion_tokens,
        )
        compactor_path = Path(train_summary["compactor_path"])
    else:
        summary_path = compactor_path.parent / "still_summary.json"
        if summary_path.is_file():
            train_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            train_summary["compactor_path"] = str(compactor_path)
        else:
            train_summary = {
                "compactor_path": str(compactor_path),
                "num_latents": args.still_latents,
                "steps": None,
                "best_loss": None,
                "train_seconds": None,
            }

    if args.skip_heldout_eval:
        write_json(
            run_dir / "run_manifest.json",
            {
                "run_name": run_name,
                "train_summary": train_summary,
                "compactor_path": str(compactor_path),
                "heldout_eval_skipped": True,
            },
        )
        print(json.dumps({"run_dir": str(run_dir.resolve()), "train_summary": train_summary}, indent=2))
        return 0

    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)

    eval_rows = load_eval_rows(eval_rows_path)
    sample_ids = sorted({row["sample_id"] for row in eval_rows})
    cache_manifest: dict[str, Any] = {}
    prediction_parts: list[Path] = []
    for sample_id in sample_ids:
        cache_summary = build_still_cache(
            compactor_path=compactor_path,
            system_prompt=_system_prompt_for_sample(eval_rows, sample_id),
            output_path=run_dir / "caches" / f"{sample_id}.pt",
            device=args.device,
            model=model,
            tokenizer=tokenizer,
        )
        prediction_path = run_dir / "predictions_parts" / f"{sample_id}.jsonl"
        run_still_eval(
            eval_path=eval_rows_path,
            cache_path=cache_summary["cache_path"],
            output_path=prediction_path,
            device=args.device,
            sample_id=sample_id,
            max_completion_tokens=args.max_completion_tokens,
            model=model,
            tokenizer=tokenizer,
        )
        cache_manifest[sample_id] = cache_summary
        prediction_parts.append(prediction_path)

    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for path in prediction_parts:
            handle.write(path.read_text(encoding="utf-8"))
    records = [
        EvalRecord.model_validate_json(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = _summarize(records, cache_manifest)
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_name": run_name,
            "train_summary": train_summary,
            "compactor_path": str(compactor_path),
            "summary_path": str((run_dir / "summary.json").resolve()),
            "predictions_path": str(predictions_path.resolve()),
            "summary": summary,
        },
    )
    print(json.dumps({"run_dir": str(run_dir.resolve()), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
