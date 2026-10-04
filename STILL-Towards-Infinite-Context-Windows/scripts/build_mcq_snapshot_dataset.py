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
    build_mcq_answer_records,
    build_mcq_eval_rows,
    build_training_dataset,
)
from still.config import DEFAULT_MATRIX  # noqa: E402
from still.data import heldout_snapshot_entries, train_snapshot_entries  # noqa: E402
from still.data.common import write_json, write_jsonl  # noqa: E402
from still.eval.common import load_eval_rows  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    """Load newline-delimited JSON records from disk."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    """Convert exact-answer corpora into deterministic MCQ train and evaluation datasets."""
    parser = argparse.ArgumentParser(description="Build deterministic MCQ train/eval datasets from exact-answer corpora.")
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--train-exact-root", required=True)
    parser.add_argument("--eval-exact-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--num-options", type=int, default=4)
    parser.add_argument("--max-experiments", type=int)
    args = parser.parse_args()

    snapshot_root = Path(args.snapshot_root)
    train_exact_root = Path(args.train_exact_root)
    eval_exact_path = Path(args.eval_exact_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_entries = train_snapshot_entries(snapshot_root, limit=500)
    heldout_entries = heldout_snapshot_entries(snapshot_root, limit=20)
    entry_by_name = {
        str(entry["experiment_name"]): entry
        for entry in [*train_entries, *heldout_entries]
    }
    experiment_dirs = sorted(path for path in train_exact_root.iterdir() if path.is_dir())
    if args.max_experiments is not None:
        experiment_dirs = experiment_dirs[: args.max_experiments]

    per_article_records: dict[str, list[dict]] = {}
    global_answer_pool: list[str] = []
    for experiment_dir in experiment_dirs:
        teacher_answers_path = experiment_dir / "teacher_answers.jsonl"
        if not teacher_answers_path.is_file():
            continue
        answer_records = _load_jsonl(teacher_answers_path)
        if not answer_records:
            continue
        per_article_records[experiment_dir.name] = answer_records
        global_answer_pool.extend(str(record.get("expected_answer") or record.get("assistant_text", "")) for record in answer_records)

    teacher_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    teacher_model.to(args.device)
    teacher_model.eval()

    combined_train_rows: list[dict] = []
    manifest_rows: list[dict[str, object]] = []
    for experiment_name, answer_records in per_article_records.items():
        entry = entry_by_name.get(experiment_name)
        if entry is None:
            continue
        corpus_text = Path(str(entry["data_path"])).read_text(encoding="utf-8")
        mcq_answers = build_mcq_answer_records(
            answer_records=answer_records,
            sample_id=experiment_name,
            global_answer_pool=global_answer_pool,
            num_options=args.num_options,
        )
        if not mcq_answers:
            continue
        answer_path = output_root / experiment_name / "teacher_answers.jsonl"
        train_path = output_root / experiment_name / "train_dataset.jsonl"
        write_jsonl(answer_path, mcq_answers)
        # MCQ training rows still carry teacher token logprobs, but the target answer is now the option letter.
        rows = build_training_dataset(
            corpus_text=corpus_text,
            slice_id=experiment_name,
            answer_records=mcq_answers,
            output_path=train_path,
            device=args.device,
            top_logprobs=args.top_logprobs,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
        )
        combined_train_rows.extend(rows)
        manifest_rows.append(
            {
                "experiment_name": experiment_name,
                "mcq_rows": len(rows),
            }
        )

    combined_train_path = output_root / "combined_train_dataset.jsonl"
    write_jsonl(combined_train_path, combined_train_rows)

    eval_rows = load_eval_rows(eval_exact_path)
    mcq_eval_rows = build_mcq_eval_rows(
        eval_rows=eval_rows,
        global_answer_pool=global_answer_pool,
        num_options=args.num_options,
    )
    mcq_eval_path = output_root / "combined_eval_rows.jsonl"
    write_jsonl(mcq_eval_path, mcq_eval_rows)

    write_json(
        output_root / "mcq_manifest.json",
        {
            "train_experiments": len(manifest_rows),
            "combined_train_rows": len(combined_train_rows),
            "combined_eval_rows": len(mcq_eval_rows),
            "num_options": args.num_options,
            "experiments": manifest_rows,
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root.resolve()),
                "train_experiments": len(manifest_rows),
                "combined_train_rows": len(combined_train_rows),
                "combined_eval_rows": len(mcq_eval_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
