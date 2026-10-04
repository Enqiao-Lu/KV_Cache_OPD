#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from still.benchmarks import (  # noqa: E402
    aligned_expected_answer_records,
    build_training_dataset,
    generate_bootstrap_questions,
)
from still.config import DEFAULT_MATRIX  # noqa: E402
from still.data import heldout_snapshot_entries, train_snapshot_entries  # noqa: E402
from still.data.common import write_json, write_jsonl  # noqa: E402


def _subset_names_from_root(path: Path) -> list[str]:
    """Reuse the experiment ordering from an existing prepared dataset root."""
    return sorted(
        child.name
        for child in path.iterdir()
        if child.is_dir() and (child / "teacher_answers.jsonl").is_file()
    )


def main() -> int:
    """Build a dense exact-answer training dataset by generating many extractive questions per article."""
    parser = argparse.ArgumentParser(
        description="Generate a denser exact-answer STILL training dataset with many extractive questions per article."
    )
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=["train", "heldout"], default="train")
    parser.add_argument("--base-url", default="http://127.0.0.1:8005/v1")
    parser.add_argument("--api-key", default="still-local")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--questions-per-article", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=48)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument(
        "--subset-from-root",
        help="Optional existing train root whose experiment directories define the article subset/order to densify.",
    )
    args = parser.parse_args()

    snapshot_root = Path(args.snapshot_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.split == "train":
        base_entries = train_snapshot_entries(snapshot_root, limit=500)
    else:
        base_entries = heldout_snapshot_entries(snapshot_root, limit=20)
    if args.subset_from_root:
        # Reusing an older root keeps article identity and ordering stable across rebuilt datasets.
        subset_names = _subset_names_from_root(Path(args.subset_from_root))
        if args.max_experiments is not None:
            subset_names = subset_names[: args.max_experiments]
        entry_by_name = {str(entry["experiment_name"]): entry for entry in base_entries}
        selected_entries = [entry_by_name[name] for name in subset_names if name in entry_by_name]
    else:
        selected_entries = base_entries[: args.max_experiments] if args.max_experiments is not None else base_entries

    teacher_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    teacher_model.to(args.device)
    teacher_model.eval()

    combined_rows: list[dict] = []
    manifest_rows: list[dict[str, object]] = []
    for entry in selected_entries:
        experiment_name = str(entry["experiment_name"])
        corpus_text = Path(str(entry["data_path"])).read_text(encoding="utf-8")
        question_path = output_root / experiment_name / "questions.txt"
        answer_path = output_root / experiment_name / "teacher_answers.jsonl"
        train_dataset_path = output_root / experiment_name / "train_dataset.jsonl"

        bootstrap_examples = generate_bootstrap_questions(
            corpus_text=corpus_text,
            eval_spec=[],
            output_path=question_path,
            base_url=args.base_url,
            api_key=args.api_key,
            num_questions=args.questions_per_article,
        )
        answer_records = aligned_expected_answer_records(
            bootstrap_examples=bootstrap_examples,
            max_completion_tokens=args.max_completion_tokens,
            tokenizer=teacher_tokenizer,
        )
        # The dense exact dataset stores teacher logprobs for the gold answer tokens, not free-form generations.
        write_jsonl(answer_path, answer_records)
        rows = build_training_dataset(
            corpus_text=corpus_text,
            slice_id=experiment_name,
            answer_records=answer_records,
            output_path=train_dataset_path,
            device=args.device,
            top_logprobs=args.top_logprobs,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
        )
        combined_rows.extend(rows)
        manifest_rows.append(
            {
                "experiment_name": experiment_name,
                "questions_requested": args.questions_per_article,
                "bootstrap_examples": len(bootstrap_examples),
                "answer_records": len(answer_records),
                "train_rows": len(rows),
            }
        )

    combined_path = output_root / "combined_train_dataset.jsonl"
    write_jsonl(combined_path, combined_rows)
    write_json(
        output_root / "dense_manifest.json",
        {
            "questions_per_article": args.questions_per_article,
            "max_experiments": args.max_experiments,
            "subset_from_root": str(Path(args.subset_from_root).resolve()) if args.subset_from_root else None,
            "experiments": manifest_rows,
            "combined_rows": len(combined_rows),
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root.resolve()),
                "split": args.split,
                "questions_per_article": args.questions_per_article,
                "experiments": len(manifest_rows),
                "combined_rows": len(combined_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
