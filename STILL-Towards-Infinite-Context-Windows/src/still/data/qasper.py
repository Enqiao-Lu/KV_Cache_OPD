"""Document-level QASPER input; gold answers are evaluation metadata only."""

import json
import re
import string
from collections import Counter
from pathlib import Path


def _normalized_space(text: str) -> str:
    return " ".join(text.split())


def _answer_text(answer: dict) -> str:
    if answer.get("unanswerable"):
        return ""
    if answer.get("extractive_spans"):
        return ", ".join(answer["extractive_spans"])
    if answer.get("free_form_answer", "").strip():
        return answer["free_form_answer"].strip()
    if answer.get("yes_no") is not None:
        return "Yes" if answer["yes_no"] else "No"
    return ""


def prepare_paper(document_id: str, paper: dict, *, split: str, min_questions: int = 2):
    """Keep complete text evidence from one annotator, never a partial evidence chain.

    Exclude non-text/table-only evidence and unanswerable questions. Exact
    whitespace-normalized source matching verifies provenance, not sufficiency.
    """
    paragraphs = [paper["title"], paper["abstract"]]
    for section in paper["full_text"]:
        paragraphs.append(section["section_name"])
        paragraphs.extend(section["paragraphs"])
    paragraphs = [p for p in paragraphs if p is not None and p.strip()]
    source = {_normalized_space(p): p for p in paragraphs if p.strip()}
    questions = []
    for qa in paper["qas"]:
        annotations = [a["answer"] for a in qa["answers"]]
        # Preserve every official evaluation reference, including disagreements.
        # Unanswerable annotations remain excluded from teacher evidence selection.
        answers = list(
            dict.fromkeys(
                filter(
                    None,
                    (
                        "Unanswerable" if a.get("unanswerable") else _answer_text(a)
                        for a in annotations
                    ),
                )
            )
        )
        evidence = None
        for annotation in annotations:
            parts = annotation.get("evidence", [])
            if not _answer_text(annotation) or not parts:
                continue
            if all(_normalized_space(p) in source for p in parts):
                evidence = list(dict.fromkeys(source[_normalized_space(p)] for p in parts))
                break
        if evidence is not None:
            questions.append(
                {
                    "question_id": qa["question_id"],
                    "question": qa["question"],
                    "evidence": "\n\n".join(evidence),
                    "evidence_paragraphs": evidence,
                    "answers": answers,
                }
            )
    if len(questions) < min_questions:
        return None
    return {
        "document_id": document_id,
        "document": "\n\n".join(paragraphs),
        "split": split,
        "questions": questions,
    }


def load_documents(path: str | Path) -> list[dict]:
    documents = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = row["document_id"]
        if doc_id in seen:
            raise ValueError(f"duplicate document_id: {doc_id}")
        seen.add(doc_id)
        if not row["document"].strip() or not row["questions"]:
            raise ValueError(f"{doc_id}: empty document/questions")
        question_ids = set()
        for qa in row["questions"]:
            if qa["question_id"] in question_ids or not qa["question"].strip():
                raise ValueError(f"{doc_id}: duplicate or empty question")
            question_ids.add(qa["question_id"])
            parts = qa.get("evidence_paragraphs", [qa["evidence"]])
            if not parts or not all(p.strip() and p in row["document"] for p in parts):
                raise ValueError(f"{doc_id}: evidence must come from the document")
            if qa["evidence"] != "\n\n".join(parts):
                raise ValueError(f"{doc_id}: evidence and evidence_paragraphs disagree")
        documents.append(row)
    if not documents:
        raise ValueError(f"No documents in {path}")
    return documents


def answer_f1(prediction: str, answers: list[str]) -> float:
    """QASPER/SQuAD answer token F1, maximizing over reference annotations."""

    def tokens(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    predicted = tokens(prediction)
    scores = []
    for answer in answers:
        gold = tokens(answer)
        common = sum((Counter(predicted) & Counter(gold)).values())
        if not predicted or not gold:
            scores.append(0.0)
        else:
            scores.append(2 * common / (len(predicted) + len(gold)))
    return max(scores, default=0.0)
