#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CARTRIDGES_ROOT = Path(os.environ.get("STILL_CARTRIDGES_ROOT", ROOT.parent / "cartridges"))
sys.path.insert(0, str(CARTRIDGES_ROOT / "src"))

from still.config import DEFAULT_MATRIX  # noqa: E402
from still.data.common import write_json  # noqa: E402
from still.eval import run_local_hf_matched_eval, run_still_eval, run_truncation_eval  # noqa: E402
from still.eval.common import (  # noqa: E402
    EvalRecord,
    SYSTEM_PROMPT,
    exact_match,
    load_eval_rows,
    normalize_generated_text_for_row,
    write_eval_records,
)
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
DEFAULT_CARTRIDGE_DATASET = (
    ROOT
    / "outputs"
    / "final_mcq_benchmark"
    / "runs"
    / "final_mcq_v1"
    / "prepared"
    / "cartridge_mcq"
    / "combined_train_dataset.jsonl"
)


def _require_file(path: Path, label: str) -> None:
    """Fail fast when a required prepared dataset or artifact is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _system_prompt_for_sample(rows: list[dict[str, Any]], sample_id: str) -> str:
    """Recover the per-target system prompt used to build a STILL cache."""
    sample_rows = [row for row in rows if row["sample_id"] == sample_id]
    if not sample_rows:
        raise ValueError(f"No eval rows found for sample_id={sample_id}")
    context = sample_rows[0]["context"]
    return SYSTEM_PROMPT.format(context=context)


def _run_still_benchmark(
    *,
    train_dataset_path: Path,
    eval_rows_path: Path,
    output_dir: Path,
    device: str,
    num_latents: int,
    steps: int,
    learning_rate: float,
    validation_examples: int,
    validation_interval: int,
    max_completion_tokens: int,
    seed: int,
    kl_weight: float,
    exact_token_ce_weight: float,
    compactor_path: Path | None = None,
    skip_train: bool = False,
) -> tuple[list[EvalRecord], dict[str, Any], dict[str, Any]]:
    """Train the reusable STILL compactor, build per-target caches, and evaluate them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_train:
        if compactor_path is None:
            raise ValueError("--skip-still-train requires --still-compactor-path.")
        summary_path = compactor_path.parent / "still_summary.json"
        checkpoint_state = torch.load(compactor_path, map_location="cpu", weights_only=False)
        checkpoint_metadata = dict(checkpoint_state.get("metadata", {}))
        if summary_path.is_file():
            train_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            train_summary["compactor_path"] = str(compactor_path.resolve())
        else:
            train_summary = {
                "dataset_path": str(train_dataset_path.resolve()),
                "compactor_path": str(compactor_path.resolve()),
                "steps": checkpoint_metadata.get("steps"),
                "num_latents": checkpoint_metadata.get("num_latents", num_latents),
                "best_loss": None,
                "train_seconds": None,
                "kl_weight": kl_weight,
                "exact_token_ce_weight": exact_token_ce_weight,
                "best_validation_summary": None,
            }
        train_summary.setdefault("dataset_path", str(train_dataset_path.resolve()))
        train_summary.setdefault("steps", checkpoint_metadata.get("steps", steps))
        train_summary.setdefault("num_latents", checkpoint_metadata.get("num_latents", num_latents))
        train_summary.setdefault("learning_rate", checkpoint_metadata.get("learning_rate", learning_rate))
        train_summary.setdefault("seed", checkpoint_metadata.get("seed", seed))
        train_summary.setdefault("validation_examples", checkpoint_metadata.get("validation_examples", validation_examples))
        train_summary.setdefault("validation_interval", checkpoint_metadata.get("validation_interval", validation_interval))
        train_summary.setdefault("kl_weight", checkpoint_metadata.get("kl_weight", kl_weight))
        train_summary.setdefault(
            "exact_token_ce_weight",
            checkpoint_metadata.get("exact_token_ce_weight", exact_token_ce_weight),
        )
    else:
        train_summary = train_still(
            dataset_path=train_dataset_path,
            output_dir=output_dir / "train",
            device=device,
            num_latents=num_latents,
            learning_rate=learning_rate,
            steps=steps,
            seed=seed,
            validation_examples=validation_examples,
            validation_interval=validation_interval,
            kl_weight=kl_weight,
            exact_token_ce_weight=exact_token_ce_weight,
            max_completion_tokens=max_completion_tokens,
        )

    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)

    eval_rows = load_eval_rows(eval_rows_path)
    sample_ids = sorted({row["sample_id"] for row in eval_rows})
    prediction_parts: list[Path] = []
    cache_manifest: dict[str, Any] = {}

    for sample_id in sample_ids:
        # STILL pays one reusable training cost, then one cheap cache-build pass per target page.
        cache_summary = build_still_cache(
            compactor_path=str(compactor_path.resolve()) if skip_train and compactor_path else train_summary["compactor_path"],
            system_prompt=_system_prompt_for_sample(eval_rows, sample_id),
            output_path=output_dir / "caches" / f"{sample_id}.pt",
            device=device,
            model=model,
            tokenizer=tokenizer,
        )
        prediction_path = output_dir / "predictions_parts" / f"{sample_id}.jsonl"
        run_still_eval(
            eval_path=eval_rows_path,
            cache_path=cache_summary["cache_path"],
            output_path=prediction_path,
            device=device,
            sample_id=sample_id,
            max_completion_tokens=max_completion_tokens,
            model=model,
            tokenizer=tokenizer,
        )
        prediction_parts.append(prediction_path)
        cache_manifest[sample_id] = cache_summary

    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as out:
        for path in prediction_parts:
            out.write(path.read_text(encoding="utf-8"))
    records = [
        EvalRecord.model_validate_json(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records, train_summary, cache_manifest


def _run_cartridge_benchmark(
    *,
    combined_train_dataset_path: Path,
    eval_rows_path: Path,
    output_dir: Path,
    device: str,
    cartridge_tokens: int,
    train_steps: int,
    learning_rate: float,
    validation_examples: int,
    validation_interval: int,
    max_completion_tokens: int,
    seed: int,
) -> tuple[list[EvalRecord], dict[str, Any]]:
    """Optimize one cartridge per held-out target and evaluate it on matching rows."""
    # Optional external reproduction; reporting and STILL must import independently.
    from cartridges.eval.cartridge import run_cartridge_eval as ext_run_cartridge_eval
    from cartridges.train.cartridge import train_cartridge

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = load_eval_rows(eval_rows_path)
    sample_ids = sorted({row["sample_id"] for row in eval_rows})
    eval_rows_by_prompt = {f"{row['sample_id']}::{row['row_hash']}": row for row in eval_rows}
    all_records: list[EvalRecord] = []
    train_manifest: dict[str, Any] = {}

    for sample_id in sample_ids:
        # Cartridges does not amortize across targets: each held-out page gets its own optimization run.
        sample_out = output_dir / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        train_started = time.perf_counter()
        train_summary = train_cartridge(
            dataset_path=combined_train_dataset_path,
            output_dir=sample_out / "train",
            slice_id=sample_id,
            device=device,
            cartridge_tokens=cartridge_tokens,
            learning_rate=learning_rate,
            steps=train_steps,
            seed=seed,
            validation_examples=validation_examples,
            validation_interval=validation_interval,
        )
        train_seconds = time.perf_counter() - train_started
        raw_records = ext_run_cartridge_eval(
            eval_path=eval_rows_path,
            cartridge_path=train_summary["cartridge_path"],
            output_path=sample_out / "predictions_raw.jsonl",
            device=device,
            sample_id=sample_id,
            max_completion_tokens=max_completion_tokens,
        )
        normalized_records: list[EvalRecord] = []
        for raw_record in raw_records:
            data = raw_record.model_dump()
            row = eval_rows_by_prompt[data["prompt_id"]]
            normalized_records.append(_normalize_cartridge_record(raw_record=raw_record, row=row))
        write_eval_records(sample_out / "predictions.jsonl", normalized_records)
        all_records.extend(normalized_records)
        train_manifest[sample_id] = {
            **train_summary,
            "train_seconds": train_seconds,
        }

    predictions_path = output_dir / "predictions.jsonl"
    write_eval_records(predictions_path, all_records)
    return all_records, train_manifest


def _mean_or_none(values: list[float | None]) -> float | None:
    """Return the arithmetic mean of the non-null values, if any."""
    filtered = [value for value in values if value is not None]
    return mean(filtered) if filtered else None


def _mean_questions_per_target(records: list[EvalRecord]) -> float:
    """Infer how many questions each held-out target contributes to the benchmark."""
    counts = Counter(record.prompt_id.split("::", 1)[0] for record in records)
    return mean(counts.values())


def _normalize_cartridge_record(
    *,
    raw_record: EvalRecord,
    row: dict[str, Any],
) -> EvalRecord:
    """Add benchmark-local normalization and decode-debug metadata to cartridge records."""
    data = raw_record.model_dump()
    raw_prediction = data["prediction"]
    prediction = normalize_generated_text_for_row(row, raw_prediction)
    data["prediction"] = prediction
    data["exact_match"] = exact_match(prediction, row["answers"])
    data["metadata"] = {
        **data["metadata"],
        "raw_prediction": raw_prediction,
        "generated_token_ids": None,
        "normalized_prediction": prediction,
        "finish_reason": None,
        "prediction_mode": row.get("prediction_mode"),
        "decode_debug_source": "external_cartridge_eval",
    }
    return EvalRecord.model_validate(data)


def _summarize_method(
    *,
    method_name: str,
    records: list[EvalRecord],
    baseline_mean_bytes: float,
    mean_target_preparation_seconds: float,
    target_preparation_kind: str,
    mean_questions_per_target: float,
    one_time_reusable_training_seconds: float | None = None,
) -> dict[str, Any]:
    """Collapse per-row evaluation records into one report row with aligned cost accounting."""
    mean_bytes = mean(record.canonical_kv_bytes for record in records)
    compression = baseline_mean_bytes / mean_bytes if mean_bytes else None
    mean_query_continuation_latency_ms = _mean_or_none(
        [record.total_latency_ms for record in records]
    )
    mean_query_prefill_ms = _mean_or_none([record.prefill_ms for record in records])
    mean_query_decode_tokens_per_second = _mean_or_none(
        [record.decode_tokens_per_second for record in records]
    )
    mean_target_total_seconds = (
        mean_target_preparation_seconds
        + ((mean_query_continuation_latency_ms or 0.0) * mean_questions_per_target / 1000.0)
    )
    mean_query_total_latency_ms = (
        mean_target_total_seconds * 1000.0 / mean_questions_per_target
        if mean_questions_per_target
        else None
    )
    return {
        "method": method_name,
        "rows": len(records),
        "accuracy": mean(int(record.exact_match) for record in records),
        "mean_canonical_kv_bytes": mean_bytes,
        "compression_vs_full": compression,
        "mean_query_prefill_ms": mean_query_prefill_ms,
        "mean_query_continuation_latency_ms": mean_query_continuation_latency_ms,
        "mean_query_total_latency_ms": mean_query_total_latency_ms,
        "mean_query_decode_tokens_per_second": mean_query_decode_tokens_per_second,
        "mean_prompt_tokens": mean(record.prompt_tokens for record in records),
        "mean_completion_tokens": mean(record.completion_tokens for record in records),
        "mean_questions_per_target": mean_questions_per_target,
        "target_preparation_kind": target_preparation_kind,
        "mean_target_preparation_seconds": mean_target_preparation_seconds,
        "mean_target_total_seconds": mean_target_total_seconds,
        "mean_target_eval_seconds_excluding_reusable_training": mean_target_total_seconds,
        "one_time_reusable_training_seconds": one_time_reusable_training_seconds,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    """Render optional floating-point report values with a stable fallback."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_setting(value: Any) -> str:
    """Render optional report settings without forcing a numeric format."""
    if value is None:
        return "n/a"
    return str(value)


def _write_report(
    *,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    still_train_summary: dict[str, Any],
    still_cache_manifest: dict[str, Any],
    cartridge_train_manifest: dict[str, Any],
    artifacts: dict[str, str],
) -> None:
    """Write the final machine-readable summary and human-readable comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_per_target = summaries[0]["mean_questions_per_target"] if summaries else None
    first_cartridge_manifest = next(iter(cartridge_train_manifest.values()))
    cartridge_tokens = first_cartridge_manifest.get("cartridge_tokens")
    if cartridge_tokens is None:
        cartridge_method = next(
            (row["method"] for row in summaries if str(row["method"]).startswith("cartridge_")),
            None,
        )
        if cartridge_method is not None:
            try:
                cartridge_tokens = int(str(cartridge_method).rsplit("_", 1)[-1])
            except ValueError:
                cartridge_tokens = None
    write_json(
        output_dir / "summary.json",
        {
            "benchmark_protocol": {
                "questions_per_target": questions_per_target,
                "definitions": {
                    "mean_query_total_latency_ms": (
                        "Operational per-question latency with method-specific preparation "
                        "amortized across the benchmark questions for one target."
                    ),
                    "mean_query_continuation_latency_ms": (
                        "Per-question online question-answering latency after any method-specific "
                        "build/optimization artifact is already ready. For full/truncation this "
                        "still includes prompt prefill for that question; for STILL/cartridge it "
                        "means querying with the prebuilt compact cache already loaded."
                    ),
                    "mean_target_preparation_seconds": "One-time cost to prepare one held-out target before answering its benchmark questions.",
                    "one_time_reusable_training_seconds": "Reusable training cost paid once across targets. Applies only to STILL.",
                    "mean_target_total_seconds": (
                        "Mean target preparation plus all benchmark-question inference for one held-out "
                        "target, excluding STILL reusable training."
                    ),
                    "mean_target_eval_seconds_excluding_reusable_training": (
                        "Compatibility alias for mean_target_total_seconds."
                    ),
                },
                "notes": [
                    "Cartridges has no reusable training stage. Its expensive cost is per-target optimization and must be paid again for every new corpus/page.",
                    "STILL separates one-time reusable compactor training from per-target cache build.",
                    "Truncation still reprefills roughly the retained long context on every query, while STILL reuses a compact cache and only runs the short continuation online.",
                    "Decode length also affects observed latency. This benchmark now logs raw decoded text and generated token ids so normalized single-letter MCQ answers can be audited.",
                ],
            },
            "methods": summaries,
            "artifacts": artifacts,
            "still_train_summary": still_train_summary,
            "still_cache_manifest": still_cache_manifest,
            "cartridge_train_manifest": cartridge_train_manifest,
        },
    )

    lines = [
        "# Final MCQ Benchmark",
        "",
        "Definitions:",
        "- `Mean query total latency` is the amortized end-to-end per-question cost, including method-specific preparation when a method has one.",
        "- `Mean online query latency` is per-question question-answering latency after the method-specific artifact is already ready.",
        "- For `full_context` and `truncation`, `mean online query latency` still includes building the prompt-side KV state for that question because there is no reusable artifact.",
        "- For `STILL` and `cartridge`, `mean online query latency` starts after the compact cache artifact already exists and measures only the question-answering pass against that artifact.",
        "- `Mean target preparation` is the one-time cost to prepare a held-out corpus/page before answering its benchmark questions.",
        "- `One-time reusable training` applies only to STILL and is the cost to train the reusable compactor once on the training split.",
        "- `Mean target total seconds` combines `mean target preparation` plus all benchmark-question inference for one held-out target, excluding STILL's one-time reusable training.",
        "",
        "| Method | Accuracy | Compression vs Full | Mean KV Bytes | Mean Query Total Latency (ms) | Mean Online Query Latency (ms) | Mean Query Prefill (ms) | Mean Query Decode tok/s | Mean Prompt Tokens | Mean Completion Tokens | Target Prep Kind | Mean Target Preparation (s) | Mean Target Total Seconds | One-Time Reusable Training (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {_fmt(row['accuracy'])} | {_fmt(row['compression_vs_full'])} | "
            f"{int(row['mean_canonical_kv_bytes'])} | {_fmt(row['mean_query_total_latency_ms'])} | "
            f"{_fmt(row['mean_query_continuation_latency_ms'])} | {_fmt(row['mean_query_prefill_ms'])} | "
            f"{_fmt(row['mean_query_decode_tokens_per_second'])} | {_fmt(row['mean_prompt_tokens'])} | "
            f"{_fmt(row['mean_completion_tokens'])} | "
            f"{row['target_preparation_kind']} | {_fmt(row['mean_target_preparation_seconds'])} | "
            f"{_fmt(row['mean_target_total_seconds'])} | "
            f"{_fmt(row['one_time_reusable_training_seconds'])} |"
        )

    lines.extend(
        [
            "",
            "## STILL Settings",
            "",
            f"- Latents: {_fmt_setting(still_train_summary.get('num_latents'))}",
            f"- Steps: {_fmt_setting(still_train_summary.get('steps'))}",
            f"- Learning rate: {_fmt_setting(still_train_summary.get('learning_rate'))}",
            f"- Seed: {_fmt_setting(still_train_summary.get('seed'))}",
            f"- Validation examples: {_fmt_setting(still_train_summary.get('validation_examples'))}",
            f"- Validation interval: {_fmt_setting(still_train_summary.get('validation_interval'))}",
            f"- Best checkpoint step: {_fmt_setting(still_train_summary.get('best_step'))}",
            f"- KL weight: {_fmt_setting(still_train_summary.get('kl_weight'))}",
            f"- Exact-token CE weight: {_fmt_setting(still_train_summary.get('exact_token_ce_weight'))}",
            f"- Best validation loss: {_fmt(still_train_summary.get('best_loss'))}",
            f"- One-time reusable training seconds: {_fmt(still_train_summary.get('train_seconds'))}",
            f"- Mean per-target cache build seconds: {_fmt(mean(item['build_seconds'] for item in still_cache_manifest.values()))}",
            "",
            "## Latency Interpretation",
            "",
            f"- Each held-out target contributes about {_fmt(questions_per_target)} benchmark questions.",
            "- Truncation and full-context methods pay the retained prompt prefill cost on every question.",
            "- STILL pays one cache-build step per target, then answers the target's questions from the compact cache with only the short continuation online.",
            "- Equal compact budget (`1024`) does not imply equal online work: truncation still processes roughly a thousand prompt tokens per query, while STILL processes a much shorter prompt plus the reused compact cache.",
            "- `Mean online query latency` is the right number if you want to isolate question-answering speed after any artifact is already built. `Mean query total latency` is the right number if you want operational end-to-end cost per query.",
            "- Decode lengths differ materially in the current benchmark. The prediction artifacts now log raw decoded text and generated token ids so normalized MCQ letters can be audited against the actual generation trace.",
            "",
            "## Cartridge Settings",
            "",
            f"- Tokens: {cartridge_tokens if cartridge_tokens is not None else 'n/a'}",
            f"- Steps per held-out target: {first_cartridge_manifest['steps']}",
            f"- Mean per-target optimization seconds: {_fmt(mean(item['train_seconds'] for item in cartridge_train_manifest.values()))}",
            "- Cartridge has no reusable training stage; every new target corpus pays this optimization cost again.",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _still_method_name(latents: int, kl_weight: float, exact_token_ce_weight: float) -> str:
    """Name the STILL variant according to the active training loss terms."""
    if kl_weight == 0.0 and exact_token_ce_weight > 0.0:
        return f"still_{latents}_ce_only"
    return f"still_{latents}"


def main() -> int:
    """Run the final full-vs-truncation-vs-STILL-vs-cartridge benchmark from prepared MCQ data."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the final MCQ benchmark for full context, truncation, STILL, and cartridges "
            "from prepared datasets. This path skips the slow bootstrap/data-generation stage."
        )
    )
    parser.add_argument("--train-dataset-path", default=str(DEFAULT_STILL_DATASET))
    parser.add_argument("--eval-rows-path", default=str(DEFAULT_EVAL_ROWS))
    parser.add_argument("--cartridge-train-dataset-path", default=str(DEFAULT_CARTRIDGE_DATASET))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "final_mcq_benchmark"))
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--truncation-budget", type=int, default=1024)
    parser.add_argument("--still-latents", type=int, default=1024)
    parser.add_argument("--still-steps", type=int, default=300)
    parser.add_argument("--still-learning-rate", type=float, default=2e-5)
    parser.add_argument("--still-kl-weight", type=float, default=0.0)
    parser.add_argument("--still-exact-token-ce-weight", type=float, default=1.0)
    parser.add_argument("--still-validation-examples", type=int, default=32)
    parser.add_argument("--still-validation-interval", type=int, default=50)
    parser.add_argument("--still-seed", type=int, default=1)
    parser.add_argument("--still-compactor-path")
    parser.add_argument("--skip-still-train", action="store_true")
    parser.add_argument("--cartridge-tokens", type=int, default=1024)
    parser.add_argument("--cartridge-steps", type=int, default=240)
    parser.add_argument("--cartridge-learning-rate", type=float, default=3e-3)
    parser.add_argument("--validation-examples", type=int, default=8)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--max-completion-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_dataset_path = Path(args.train_dataset_path)
    eval_rows_path = Path(args.eval_rows_path)
    cartridge_train_dataset_path = Path(args.cartridge_train_dataset_path)
    still_compactor_path = Path(args.still_compactor_path).resolve() if args.still_compactor_path else None
    _require_file(train_dataset_path, "STILL training dataset")
    _require_file(eval_rows_path, "evaluation rows")
    _require_file(cartridge_train_dataset_path, "cartridge training dataset")
    if still_compactor_path is not None:
        _require_file(still_compactor_path, "STILL compactor checkpoint")

    output_root = Path(args.output_root)
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Full context establishes the quality and byte-accounting baseline.
    full_records = run_local_hf_matched_eval(
        eval_path=eval_rows_path,
        output_path=run_dir / "full_context" / "predictions.jsonl",
        device=args.device,
        max_completion_tokens=args.max_completion_tokens,
    )
    # Truncation reuses the same evaluator after clipping context length to the token budget.
    trunc_records = run_truncation_eval(
        eval_path=eval_rows_path,
        output_path=run_dir / "truncation" / "predictions.jsonl",
        device=args.device,
        context_token_budget=args.truncation_budget,
        max_completion_tokens=args.max_completion_tokens,
    )
    # STILL first learns a reusable compactor, then materializes one compact cache per held-out target.
    still_records, still_train_summary, still_cache_manifest = _run_still_benchmark(
        train_dataset_path=train_dataset_path,
        eval_rows_path=eval_rows_path,
        output_dir=run_dir / "still",
        device=args.device,
        num_latents=args.still_latents,
        steps=args.still_steps,
        learning_rate=args.still_learning_rate,
        validation_examples=args.still_validation_examples,
        validation_interval=args.still_validation_interval,
        max_completion_tokens=args.max_completion_tokens,
        seed=args.still_seed,
        kl_weight=args.still_kl_weight,
        exact_token_ce_weight=args.still_exact_token_ce_weight,
        compactor_path=still_compactor_path,
        skip_train=args.skip_still_train or still_compactor_path is not None,
    )
    # Cartridges optimizes a separate compact cache for every held-out target corpus.
    cartridge_records, cartridge_train_manifest = _run_cartridge_benchmark(
        combined_train_dataset_path=cartridge_train_dataset_path,
        eval_rows_path=eval_rows_path,
        output_dir=run_dir / "cartridge",
        device=args.device,
        cartridge_tokens=args.cartridge_tokens,
        train_steps=args.cartridge_steps,
        learning_rate=args.cartridge_learning_rate,
        validation_examples=args.validation_examples,
        validation_interval=args.validation_interval,
        max_completion_tokens=args.max_completion_tokens,
        seed=args.seed,
    )

    baseline_mean_bytes = mean(record.canonical_kv_bytes for record in full_records)
    mean_questions_per_target = _mean_questions_per_target(full_records)
    still_training_seconds = still_train_summary.get("train_seconds")
    still_build_seconds = mean(item["build_seconds"] for item in still_cache_manifest.values())
    cartridge_training_seconds = mean(
        item["train_seconds"] for item in cartridge_train_manifest.values()
    )
    summaries = [
        _summarize_method(
            method_name="full_context",
            records=full_records,
            baseline_mean_bytes=baseline_mean_bytes,
            mean_target_preparation_seconds=0.0,
            target_preparation_kind="none",
            mean_questions_per_target=mean_questions_per_target,
        ),
        _summarize_method(
            method_name=f"truncation_{args.truncation_budget}",
            records=trunc_records,
            baseline_mean_bytes=baseline_mean_bytes,
            mean_target_preparation_seconds=0.0,
            target_preparation_kind="none",
            mean_questions_per_target=mean_questions_per_target,
        ),
        _summarize_method(
            method_name=_still_method_name(
                args.still_latents,
                args.still_kl_weight,
                args.still_exact_token_ce_weight,
            ),
            records=still_records,
            baseline_mean_bytes=baseline_mean_bytes,
            mean_target_preparation_seconds=still_build_seconds,
            target_preparation_kind="still_cache_build",
            mean_questions_per_target=mean_questions_per_target,
            one_time_reusable_training_seconds=still_training_seconds,
        ),
        _summarize_method(
            method_name=f"cartridge_{args.cartridge_tokens}",
            records=cartridge_records,
            baseline_mean_bytes=baseline_mean_bytes,
            mean_target_preparation_seconds=cartridge_training_seconds,
            target_preparation_kind="per_target_optimization",
            mean_questions_per_target=mean_questions_per_target,
        ),
    ]
    artifacts = {
        "train_dataset_path": str(train_dataset_path.resolve()),
        "eval_rows_path": str(eval_rows_path.resolve()),
        "cartridge_train_dataset_path": str(cartridge_train_dataset_path.resolve()),
        "full_predictions": str((run_dir / "full_context" / "predictions.jsonl").resolve()),
        "truncation_predictions": str((run_dir / "truncation" / "predictions.jsonl").resolve()),
        "still_predictions": str((run_dir / "still" / "predictions.jsonl").resolve()),
        "still_train_summary": str((run_dir / "still" / "train" / "still_summary.json").resolve()),
        "cartridge_predictions": str((run_dir / "cartridge" / "predictions.jsonl").resolve()),
        "still_compactor_path": str(still_compactor_path) if still_compactor_path else None,
    }
    _write_report(
        output_dir=run_dir / "report",
        summaries=summaries,
        still_train_summary=still_train_summary,
        still_cache_manifest=still_cache_manifest,
        cartridge_train_manifest=cartridge_train_manifest,
        artifacts=artifacts,
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_name": run_name,
            "train_dataset_path": str(train_dataset_path.resolve()),
            "eval_rows_path": str(eval_rows_path.resolve()),
            "cartridge_train_dataset_path": str(cartridge_train_dataset_path.resolve()),
            "summary_path": str((run_dir / "report" / "summary.json").resolve()),
            "comparison_path": str((run_dir / "report" / "comparison.md").resolve()),
            "summaries": summaries,
        },
    )
    print(json.dumps({"run_dir": str(run_dir.resolve()), "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
