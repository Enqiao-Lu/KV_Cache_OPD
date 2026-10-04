import copy

import pytest


def paper():
    return {
        "title": "Example",
        "abstract": "Overview.",
        "full_text": [
            {"section_name": "Results", "paragraphs": ["Alpha evidence.", "Beta evidence."]}
        ],
        "qas": [
            {
                "question_id": "q1",
                "question": "Which alpha?",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": ["Alpha"],
                            "free_form_answer": "",
                            "yes_no": None,
                            "evidence": ["Alpha evidence."],
                        }
                    }
                ],
            },
            {
                "question_id": "q2",
                "question": "Which beta?",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "free_form_answer": "Beta",
                            "yes_no": None,
                            "evidence": ["Beta evidence."],
                        }
                    }
                ],
            },
        ],
    }


def test_qasper_keeps_one_document_multiple_questions_and_source_evidence():
    from still.data.qasper import prepare_paper

    result = prepare_paper("doc1", paper(), split="train")
    assert len(result["questions"]) == 2
    assert result["document_id"] == "doc1"
    assert result["split"] == "train"
    assert result["questions"][0]["evidence"] == "Alpha evidence."
    assert result["questions"][1]["answers"] == ["Beta"]
    assert all(q["evidence"] in result["document"] for q in result["questions"])


def test_qasper_rejects_missing_or_incomplete_evidence_without_fallback():
    from still.data.qasper import prepare_paper

    value = paper()
    value["qas"][0]["answers"][0]["answer"]["evidence"].append("Missing table or paragraph")
    assert prepare_paper("doc1", value, split="train") is None
    value = copy.deepcopy(paper())
    value["qas"][0]["answers"][0]["answer"]["unanswerable"] = True
    assert prepare_paper("doc1", value, split="train") is None


def test_qasper_accepts_null_section_names_in_original_release():
    from still.data.qasper import prepare_paper

    value = paper()
    value["full_text"][0]["section_name"] = None
    result = prepare_paper("doc1", value, split="train")
    assert result is not None
    assert len(result["questions"]) == 2


def test_f1_matches_official_qasper_empty_answer_convention():
    from still.data.qasper import answer_f1

    assert answer_f1("", [""]) == 0
    assert answer_f1("The", ["a"]) == 0
    assert answer_f1("ALPHA!", ["Alpha", "Beta"]) == 1
    assert answer_f1("Alpha code", ["Alpha"]) == pytest.approx(2 / 3)


def test_qasper_preserves_mixed_unanswerable_evaluation_references():
    from still.data.qasper import prepare_paper

    value = paper()
    value["qas"][0]["answers"].append({"answer": {"unanswerable": True}})
    result = prepare_paper("doc1", value, split="dev")
    assert result["questions"][0]["answers"] == ["Alpha", "Unanswerable"]
    assert result["questions"][0]["evidence"] == "Alpha evidence."


def test_opd_data_loader_rejects_answer_only_evidence_and_duplicate_ids(tmp_path):
    import json

    from still.data.qasper import load_documents

    row = {
        "document_id": "d",
        "document": "Alpha source.",
        "split": "train",
        "questions": [
            {
                "question_id": "q",
                "question": "Q?",
                "evidence": "Not in document",
                "answers": ["Alpha"],
            }
        ],
    }
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="evidence"):
        load_documents(path)
    row["questions"][0]["evidence"] = "Alpha source."
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_documents(path)
