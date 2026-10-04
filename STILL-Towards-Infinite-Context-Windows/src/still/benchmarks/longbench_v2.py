"""Full-context LongBench v2 preparation and the local official accuracy metric."""

import importlib.util
import json
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from still.data.common import file_sha256, stable_hash, write_json, write_jsonl

DATASET_ID = "THUDM/LongBench-v2"
REFERENCE_DIR = (
    Path(__file__).resolve().parents[4]
    / "kvpress/evaluation/benchmarks/longbenchv2"
)
METADATA_FIELDS = ("difficulty", "length", "domain", "sub_domain")

# Exact 0-shot templates from the adjacent kvpress adapter, derived from
# https://github.com/THUDM/LongBench/blob/main/prompts/0shot.txt.
CONTEXT_TEMPLATE = """Please read the following text and answer the question below.
<text>
{context}
</text>

"""
QUESTION_TEMPLATE = """What is the correct answer to this question: {question}
Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}

Format your response as follows: "The correct answer is (insert answer here)."""


def convert_longbench(rows: Iterable[dict]) -> list[dict]:
    """Group source-shaped rows by complete context, keeping all targets in questions."""
    documents: dict[str, dict] = {}
    question_ids: set[str] = set()
    for row in rows:
        for field in ("context", "question", "choice_A", "choice_B", "choice_C", "choice_D"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"LongBench v2 requires a nonempty string {field}.")
        if row.get("answer") not in ("A", "B", "C", "D"):
            raise ValueError("LongBench v2 answer must be A, B, C, or D.")
        if row.get("_id") is None or str(row["_id"]) == "":
            raise ValueError("LongBench v2 requires a question_id in _id.")
        question_id = str(row["_id"])
        if question_id in question_ids:
            raise ValueError(f"Duplicate LongBench v2 question_id: {question_id}")
        question_ids.add(question_id)
        document_id = f"longbench_v2:{stable_hash(row['context'])}"
        if document_id not in documents:
            documents[document_id] = {
                "benchmark": "longbench_v2",
                "document_id": document_id,
                "document": CONTEXT_TEMPLATE.format(context=row["context"]),
                "prompt_style": "context",
                "questions": [],
            }
        documents[document_id]["questions"].append(
            {
                "question_id": question_id,
                "question": QUESTION_TEMPLATE.format(
                    question=row["question"],
                    A=row["choice_A"],
                    B=row["choice_B"],
                    C=row["choice_C"],
                    D=row["choice_D"],
                ),
                "answers": [row["answer"]],
                "metric": "longbench_accuracy",
                "metadata": {key: row.get(key) for key in METADATA_FIELDS},
            }
        )
    return list(documents.values())


def prepare_longbench(output_dir: Path, *, source: Path | None = None) -> dict:
    """Prepare official train or local JSON/JSONL data without shortening any context."""
    provenance: dict[str, Any] = {
        "benchmark": "longbench_v2",
        "context_policy": "full_context_no_truncation",
        "prompt_source": "kvpress/evaluation/benchmarks/longbenchv2/create_huggingface_dataset.py",
        "prompt_sha256": stable_hash({"context": CONTEXT_TEMPLATE, "question": QUESTION_TEMPLATE}),
        "scorer_source": str(REFERENCE_DIR / "calculate_metrics.py"),
        "scorer_sha256": file_sha256(REFERENCE_DIR / "calculate_metrics.py"),
    }
    if source is None:
        from datasets import load_dataset

        rows = load_dataset(DATASET_ID, split="train")
        provenance.update(
            source=DATASET_ID,
            split="train",
            dataset_fingerprint=getattr(rows, "_fingerprint", None),
        )
    else:
        source = Path(source).resolve()
        with source.open(encoding="utf-8") as handle:
            if source.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in handle if line.strip()]
            else:
                rows = json.load(handle)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("LongBench v2 source must contain an array or JSONL of source rows.")
        provenance.update(source=str(source), source_sha256=file_sha256(source))
    documents = convert_longbench(rows)
    output_dir = Path(output_dir)
    documents_path = output_dir / "documents.jsonl"
    provenance_path = output_dir / "provenance.json"
    write_jsonl(documents_path, documents)
    provenance.update(
        source_rows=sum(len(document["questions"]) for document in documents),
        document_count=len(documents),
        documents_sha256=file_sha256(documents_path),
    )
    write_json(provenance_path, provenance)
    return {
        "documents": documents,
        "provenance": provenance,
        "documents_path": str(documents_path.resolve()),
        "provenance_path": str(provenance_path.resolve()),
    }


@lru_cache(maxsize=1)
def _official_scorer():
    """Load only the pure reference file; importing kvpress initializes model code."""
    path = REFERENCE_DIR / "calculate_metrics.py"
    spec = importlib.util.spec_from_file_location("_still_longbench_v2_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score


def score_longbench(prediction: str, answers: list[str]) -> float:
    """Return the exact local official 0/1 score against any accepted option."""
    score = _official_scorer()
    return float(any(score(prediction, answer) for answer in answers))


def summarize_longbench(predictions: list[dict]) -> dict:
    """Report mean accuracy and coverage for the original benchmark categories."""
    summary = {
        "average": mean(row["score"] for row in predictions) if predictions else None,
        "count": len(predictions),
    }
    for field in METADATA_FIELDS:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in predictions:
            value = row.get("metadata", {}).get(field)
            if value is not None:
                groups[str(value)].append(float(row["score"]))
        summary[f"by_{field}"] = {
            key: {"average": mean(scores), "count": len(scores)}
            for key, scores in sorted(groups.items())
        }
    return summary
