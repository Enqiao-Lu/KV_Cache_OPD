import ast
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

REFERENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "kvpress/evaluation/benchmarks/longbenchv2"
)


def _adapter():
    spec = importlib.util.find_spec("still.benchmarks.longbench_v2")
    assert spec is not None, "LongBench v2 adapter is missing"
    return importlib.import_module("still.benchmarks.longbench_v2")


def _row(**changes):
    return {
        "_id": "sample-0",
        "context": "Full source context\n\nLast paragraph stays intact.",
        "question": "UNIQUE QUESTION: Which action resolved the dispute?",
        "choice_A": "UNIQUE OPTION ALPHA",
        "choice_B": "UNIQUE OPTION BRAVO",
        "choice_C": "UNIQUE OPTION CHARLIE",
        "choice_D": "UNIQUE OPTION DELTA",
        "answer": "B",
        "difficulty": "hard",
        "length": "long",
        "domain": "Single-Document QA",
        "sub_domain": "Finance",
        **changes,
    }


def _official_templates():
    tree = ast.parse((REFERENCE_DIR / "create_huggingface_dataset.py").read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"context_template", "question_template"}
    }


def test_source_rows_group_by_complete_context_with_official_prompts():
    adapter = _adapter()
    rows = [_row(), _row(_id="sample-1", question="Another question?", answer="D")]
    documents = adapter.convert_longbench(iter(rows))
    templates = _official_templates()

    assert len(documents) == 1
    document = documents[0]
    assert document["benchmark"] == "longbench_v2"
    assert isinstance(document["document_id"], str)
    assert document["prompt_style"] == "context"
    assert document["document"] == templates["context_template"].format(context=rows[0]["context"])
    assert len(document["questions"]) == 2
    first = document["questions"][0]
    assert first["question_id"] == "sample-0"
    assert first["question"] == templates["question_template"].format(
        question=rows[0]["question"],
        A=rows[0]["choice_A"],
        B=rows[0]["choice_B"],
        C=rows[0]["choice_C"],
        D=rows[0]["choice_D"],
    )
    assert first["answers"] == ["B"]
    assert first["metric"] == "longbench_accuracy"
    assert first["metadata"] == {
        key: rows[0][key] for key in ("difficulty", "length", "domain", "sub_domain")
    }


def test_cache_document_never_contains_question_choices_or_gold_annotations():
    adapter = _adapter()
    row = _row()
    document = adapter.convert_longbench([row])[0]
    for key in ("question", "choice_A", "choice_B", "choice_C", "choice_D"):
        assert row[key] not in document["document"]
    assert "The correct answer is" not in document["document"]
    changed = adapter.convert_longbench([_row(answer="A", question="Changed question?")])[0]
    assert changed["document"] == document["document"]
    assert changed["document_id"] == document["document_id"]


def test_different_contexts_stay_separate_without_truncation():
    adapter = _adapter()
    context = "prefix\n" + "x" * 100_000 + "\nTAIL SENTINEL"
    documents = adapter.convert_longbench([_row(), _row(_id="sample-1", context=context)])
    assert len(documents) == 2
    assert documents[0]["document_id"] != documents[1]["document_id"]
    assert context in documents[1]["document"]


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_prepare_reads_local_source_and_writes_hashed_provenance(tmp_path, suffix):
    adapter = _adapter()
    source = tmp_path / f"raw{suffix}"
    rows = [_row()]
    text = json.dumps(rows) if suffix == ".json" else json.dumps(rows[0]) + "\n"
    source.write_text(text, encoding="utf-8")
    prepared = adapter.prepare_longbench(tmp_path / "prepared", source=source)

    assert prepared["documents"] == adapter.convert_longbench(rows)
    assert prepared["provenance"]["source"] == str(source.resolve())
    assert len(prepared["provenance"]["source_sha256"]) == 64
    assert prepared["provenance"]["source_rows"] == 1
    assert prepared["provenance"]["context_policy"] == "full_context_no_truncation"
    written = [
        json.loads(line) for line in Path(prepared["documents_path"]).read_text().splitlines()
    ]
    assert written == prepared["documents"]
    manifest = json.loads(Path(prepared["provenance_path"]).read_text())
    assert manifest == prepared["provenance"]


def test_prepare_default_loads_official_train_split(monkeypatch, tmp_path):
    adapter = _adapter()
    import datasets

    class _Dataset(list):
        _fingerprint = "source-fingerprint"

    calls = []

    def load_dataset(name, *, split):
        calls.append((name, split))
        return _Dataset([_row()])

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    prepared = adapter.prepare_longbench(tmp_path)
    assert calls == [("THUDM/LongBench-v2", "train")]
    assert prepared["provenance"]["source"] == "THUDM/LongBench-v2"
    assert prepared["provenance"]["split"] == "train"
    assert prepared["provenance"]["dataset_fingerprint"] == "source-fingerprint"


def test_scoring_has_exact_local_official_parity():
    adapter = _adapter()
    spec = importlib.util.spec_from_file_location(
        "longbench_reference_score", REFERENCE_DIR / "calculate_metrics.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    predictions = [
        "The correct answer is (B).",
        "**The correct answer is B**.",
        "Reasoning. The correct answer is (A). The correct answer is (B).",
        "The correct answer is BOGUS",
        "B",
        "(B)",
        "The correct answer is (b).",
        "the correct answer is B.",
        "",
        "The correct answer is (D).",
    ]
    for prediction in predictions:
        assert adapter.score_longbench(prediction, ["B"]) == float(reference.score(prediction, "B"))
    assert adapter.score_longbench("The correct answer is (D).", ["B", "D"]) == 1.0
    assert adapter.score_longbench("The correct answer is (D).", []) == 0.0


def test_summary_reports_each_metadata_group_on_zero_to_one_scale():
    adapter = _adapter()
    records = [
        {"score": 1.0, "metadata": _row()},
        {"score": 0.0, "metadata": _row(difficulty="easy", length="short")},
        {"score": 1.0, "metadata": _row(domain="Code", sub_domain="Python")},
    ]
    summary = adapter.summarize_longbench(records)
    assert summary["count"] == 3
    assert summary["average"] == pytest.approx(2 / 3)
    assert summary["by_difficulty"]["hard"] == {"average": 1.0, "count": 2}
    assert summary["by_difficulty"]["easy"] == {"average": 0.0, "count": 1}
    assert summary["by_length"]["short"] == {"average": 0.0, "count": 1}
    assert summary["by_domain"]["Single-Document QA"] == {"average": 0.5, "count": 2}
    assert summary["by_sub_domain"]["Python"] == {"average": 1.0, "count": 1}
    assert adapter.summarize_longbench([])["average"] is None


@pytest.mark.parametrize(
    "rows, message",
    [
        ([_row(answer="E")], "answer"),
        ([_row(), _row()], "question_id"),
        ([_row(context=None)], "context"),
    ],
)
def test_invalid_source_rows_fail_before_preparation(rows, message):
    with pytest.raises(ValueError, match=message):
        _adapter().convert_longbench(rows)
