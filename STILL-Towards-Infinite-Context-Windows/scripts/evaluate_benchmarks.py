"""Evaluate frozen full KV and saved STILL/OPD checkpoints on four benchmarks."""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from run_opd_comparison import MODEL_ID, MODEL_REVISION, parameter_digest
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from still.benchmarks.suite import (
    METRICS,
    evaluate_documents,
    file_digest,
    filter_documents,
    load_compactor_checkpoint,
    load_documents,
)
from still.train.still import _set_training_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/benchmarks/data"))
    parser.add_argument("--benchmarks", nargs="+", choices=list(METRICS), default=list(METRICS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["full_context", "still", "full", "evidence"],
        default=["full_context", "still", "full", "evidence"],
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("outputs/opd/four_methods_smoke_4b")
    )
    for method in ("still", "full", "evidence"):
        parser.add_argument(f"--{method}-checkpoint", type=Path)
    parser.add_argument("--num-latents", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks/smoke_4b"))
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--document-limit", type=int, default=0)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    if len(args.methods) != len(set(args.methods)) or len(args.benchmarks) != len(
        set(args.benchmarks)
    ):
        parser.error("methods and benchmarks must be unique")
    if not 1 <= args.max_context_tokens <= 32768:
        parser.error("native context limit must be between 1 and 32768")
    paths = {
        method: getattr(args, f"{method}_checkpoint")
        or args.checkpoint_dir / f"{method}_compactor.pt"
        for method in args.methods
        if method != "full_context"
    }
    for method, path in paths.items():
        if not path.is_file():
            parser.error(f"missing {method} checkpoint: {path}")
    datasets, provenance = {}, {}
    for benchmark in args.benchmarks:
        path = args.data_dir / benchmark / "prepared" / "documents.jsonl"
        rows = load_documents(path)
        if any(row["benchmark"] != benchmark for row in rows):
            parser.error(f"{path}: benchmark field mismatch")
        datasets[benchmark] = rows
        selection = path.parent / "selection.json"
        provenance[benchmark] = {
            "path": str(path.resolve()),
            "sha256": file_digest(path),
            "preparation": json.loads(selection.read_text()) if selection.exists() else None,
        }
    _set_training_seed(0)
    snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.model_max_length = args.max_context_tokens
    if args.tiny:
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=16384,
                eos_token_id=tokenizer.eos_token_id,
                attention_dropout=0.0,
            )
        )
        model.config._attn_implementation = args.attn_implementation
    else:
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        )
    model.to(args.device).eval().requires_grad_(False)
    context_limit = min(args.max_context_tokens, model.config.max_position_embeddings)
    for benchmark, rows in datasets.items():
        selected, coverage = filter_documents(
            tokenizer,
            rows,
            max_context_tokens=context_limit,
            max_new_tokens=args.max_new_tokens,
            limit=args.document_limit,
            question_limit=args.question_limit,
        )
        if not selected:
            raise ValueError(f"{benchmark}: no whole documents fit the context window")
        datasets[benchmark] = selected
        provenance[benchmark]["evaluation_coverage"] = coverage
    backbone_hash = parameter_digest(model)
    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tiny": args.tiny,
        "enable_thinking": False,
        "decoding": "greedy",
        "command": sys.argv,
        "max_context_tokens": context_limit,
        "backbone_sha256": backbone_hash,
        "datasets": provenance,
        "checkpoints": {},
    }
    results = {}
    budgets = set()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        compactor = None
        if method != "full_context":
            compactor, info = load_compactor_checkpoint(
                model,
                paths[method],
                method=method,
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                tiny=args.tiny,
                num_latents=args.num_latents,
                backbone_sha256=backbone_hash,
            )
            budgets.add(info["num_latents"])
            if len(budgets) > 1:
                raise ValueError("comparison checkpoint latent budgets differ")
            trained = set(info.get("train_documents", []))
            if any(row["document_id"] in trained for row in datasets.get("qasper", [])):
                raise ValueError(f"{method}: QASPER evaluation overlaps training documents")
            metadata["checkpoints"][method] = info
        results[method] = {}
        for benchmark, rows in datasets.items():
            print(f"Evaluating {method} / {benchmark}: {len(rows)} documents", flush=True)
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model.device)
            result = evaluate_documents(
                model,
                tokenizer,
                compactor,
                rows,
                max_new_tokens=args.max_new_tokens,
                max_context_tokens=context_limit,
            )
            output = args.output_dir / method / benchmark
            output.mkdir(parents=True, exist_ok=True)
            predictions = result.pop("predictions")
            (output / "predictions.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
            )
            result["predictions_path"] = str((output / "predictions.jsonl").resolve())
            result["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(model.device) / 2**30
                if model.device.type == "cuda"
                else None
            )
            results[method][benchmark] = result
            (output / "summary.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "method": method,
                        "benchmark": benchmark,
                        "questions": result["questions"],
                        "score": result["score"],
                    }
                ),
                flush=True,
            )
        del compactor
    metadata["backbone_unchanged"] = parameter_digest(model) == backbone_hash
    if not metadata["backbone_unchanged"]:
        raise RuntimeError("frozen backbone changed during evaluation")
    summary = {"metadata": metadata, "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Four-method benchmark comparison",
        "",
        "Scores use each benchmark's metric and selected subset; "
        "short smoke runs do not rank methods.",
        "",
        "| method | " + " | ".join(datasets) + " |",
        "| --- | " + " | ".join("---" for _ in datasets) + " |",
    ]
    for method, scores in results.items():
        lines.append(
            "| "
            + method
            + " | "
            + " | ".join(
                f"{scores[benchmark]['score']:.4f} ({scores[benchmark]['questions']} questions)"
                for benchmark in datasets
            )
            + " |"
        )
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
