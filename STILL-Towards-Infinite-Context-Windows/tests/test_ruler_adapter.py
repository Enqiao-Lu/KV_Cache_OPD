import hashlib
import importlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]
QA_INSTRUCTION = (
    "Answer the question based on the given documents. Only give me the answer "
    "and do not output any other words."
)


def _adapter():
    assert importlib.util.find_spec("still.benchmarks.ruler"), "RULER adapter is missing"
    return importlib.import_module("still.benchmarks.ruler")


def _source_row(task):
    """Source-shaped base-template rows; CWE and VT retain official demonstrations."""
    if task.startswith("niah"):
        plural = task in {"niah_multivalue", "niah_multiquery"}
        kind = "uuids" if task in {"niah_single_3", "niah_multikey_3"} else "numbers"
        kind = kind if plural else kind[:-1]
        start = "Some" if plural else "A"
        verb = "are" if plural else "is"
        document = (
            f"{start} special magic {kind} {verb} hidden within the following text. "
            f"Make sure to memorize it. I will quiz you about the {kind} afterwards.\n"
            f"The grass is green. One of the special magic {kind} for hidden-key is: 7654321.\n"
        )
        question = (
            f"What {'are all' if plural else 'is'} the special magic {kind} "
            "for hidden-key mentioned in the provided text?"
        )
        prefix = f" The special magic {kind} for hidden-key mentioned in the provided text {verb}"
    elif task == "vt":
        instruction = (
            "Memorize and track the chain(s) of variable assignment hidden in the following text."
        )
        document = (
            f"{instruction}\n\nVAR DEMO = 12345\n"
            "Question: Find all variables that are assigned the value 12345 in the text above."
            " Answer: According to the chain(s) of variable assignment in the text above, "
            "1 variables are assigned the value 12345, they are: DEMO\n\n"
            f"{instruction}\n\nVAR HELLO = 67890\nVAR WORLD = VAR HELLO\n"
        )
        question = (
            "Question: Find all variables that are assigned the value 67890 in the text above."
        )
        prefix = (
            " Answer: According to the chain(s) of variable assignment in the text above, "
            "2 variables are assigned the value 67890, they are: "
        )
    elif task == "cwe":
        instruction = (
            "Below is a numbered list of words. In these words, some appear more often than "
            "others. Memorize the ones that appear most often."
        )
        question = "Question: What are the 10 most common words in the above list?"
        prefix = " Answer: The top 10 words that appear most often in the list are:"
        document = (
            f"{instruction}\n1. demonstration 2. demonstration\n{question}{prefix} "
            f"1. demonstration\n{instruction}\n1. target 2. target\n"
        )
    elif task == "fwe":
        document = (
            "Read the following coded text and track the frequency of each coded word. "
            "Find the three most frequently appeared coded words. ABC DEF ABC GHI\n"
        )
        question = (
            "Question: Do not provide any explanation. Please ignore the dots '....'. "
            "What are the three most frequently appeared words in the above coded text?"
        )
        prefix = (
            " Answer: According to the coded text above, the three most frequently "
            "appeared words are:"
        )
    else:
        document = (
            f"{QA_INSTRUCTION}\n\nThe following are given documents.\n\n"
            "Document 1:\nA document mentions Question: inside source text.\n\n"
        )
        question = f"{QA_INSTRUCTION}\n\nQuestion: What is the fixture query?"
        prefix = " Answer:"
    return (
        {
            "index": 7,
            "input": document + question,
            "outputs": ["7654321", "alternative"],
            "length": 4000,
            "answer_prefix": prefix,
            "token_position_answer": 17,
        },
        document,
        question,
        prefix,
    )


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("inline_prefix", [False, True])
def test_all_thirteen_tasks_split_only_final_query(task, inline_prefix):
    row, document, question, prefix = _source_row(task)
    if inline_prefix:
        row["input"] += row.pop("answer_prefix")
    converted = _adapter().convert_ruler([row], task=task)[0]
    assert converted["document"] == document
    assert converted["prompt_style"] == "context"
    assert converted["benchmark"] == "ruler"
    qa = converted["questions"][0]
    assert qa["question"] == question
    assert qa["answer_prefix"] == prefix
    assert qa["answers"] == row["outputs"]
    assert qa["metric"] == ("ruler_part" if task.startswith("qa_") else "ruler_all")
    assert qa["metadata"]["task"] == task
    assert qa["metadata"]["length"] == 4000
    assert qa["metadata"]["index"] == 7
    assert qa["metadata"]["token_position_answer"] == 17
    assert qa["metadata"]["max_new_tokens"] > 0
    assert (
        converted["document"] + qa["question"] + qa["answer_prefix"] == document + question + prefix
    )


@pytest.mark.parametrize(
    "row",
    [
        {"input": "unrecognized prompt", "outputs": ["answer"]},
        {
            "input": (
                "source\nWhat is the special magic number for key mentioned in the provided text?"
            ),
            "outputs": [],
        },
    ],
)
def test_malformed_source_rows_fail_instead_of_compressing_query(row):
    with pytest.raises(ValueError):
        _adapter().convert_ruler([row], task="niah_single_1")


def test_rejects_unknown_task():
    with pytest.raises(ValueError, match="task"):
        _adapter().convert_ruler([], task="niah_imaginary")


def test_rejects_separate_prefix_from_modified_template():
    row = _source_row("niah_single_1")[0]
    row["answer_prefix"] = " Target answer: 7654321"
    with pytest.raises(ValueError, match="prefix"):
        _adapter().convert_ruler([row], task="niah_single_1")


def test_qa_uses_last_complete_question_marker_even_when_quoted_in_document():
    row, document, question, prefix = _source_row("qa_1")
    quoted = f"{QA_INSTRUCTION}\n\nQuestion: A quoted document question? Answer: quoted\n\n"
    row["input"] = document + quoted + question
    converted = _adapter().convert_ruler([row], task="qa_1")[0]
    assert converted["document"] == document + quoted
    assert converted["questions"][0]["question"] == question
    assert converted["questions"][0]["answer_prefix"] == prefix


@pytest.mark.parametrize("partial", [False, True])
def test_scoring_matches_official_case_insensitive_reference_recall(partial):
    # NVIDIA/RULER scripts/eval/synthetic/constants.py, Apache-2.0.
    predictions = ["A RED car", "BLUE", "unknown", "red RED"]
    references = [["red", "blue", "car"], ["blue", "cyan"], ["red"], ["red", "red"]]
    individual = [
        _adapter().score_ruler(pred, ref, partial=partial)
        for pred, ref in zip(predictions, references, strict=True)
    ]
    official = round(
        sum(
            max(float(r.lower() in pred.lower()) for r in ref)
            if partial
            else sum(float(r.lower() in pred.lower()) for r in ref) / len(ref)
            for pred, ref in zip(predictions, references, strict=True)
        )
        / len(predictions)
        * 100,
        2,
    )
    assert round(sum(individual) / len(individual) * 100, 2) == official
    assert individual[0] == (1.0 if partial else pytest.approx(2 / 3))


def test_scoring_preserves_official_nonprintable_postprocess_and_rejects_empty_gold():
    assert _adapter().score_ruler("  RED\x00CAR ", ["red\ncar"]) == 1.0
    with pytest.raises(ValueError, match="answers"):
        _adapter().score_ruler("prediction", [])


def test_summary_rounds_after_averaging_and_keeps_task_metric():
    summary = _adapter().summarize_ruler(
        [
            {"metadata": {"task": "cwe"}, "prediction": "red", "answers": ["red", "blue", "car"]},
            {
                "metadata": {"task": "cwe"},
                "prediction": "red car",
                "answers": ["red", "blue", "car"],
            },
            {"task": "qa_1", "pred": "red", "outputs": ["red", "blue", "car"]},
        ]
    )
    assert summary["per_task"]["cwe"]["official_score"] == 50.0
    assert summary["per_task"]["qa_1"]["official_score"] == 100.0
    assert summary["score"] == 0.75
    assert summary["count"] == 3


def _fake_vendor(tmp_path):
    vendor = tmp_path / "vendor"
    base = vendor / "scripts" / "data" / "synthetic"
    base.mkdir(parents=True)
    (base / "niah.py").write_text("# fake generator\n")
    (base / "constants.py").write_text(
        "TASKS = {'niah': {'tokens_to_generate': 128, 'template': '{context}\\n{query}', "
        "'answer_prefix': ' The special magic number'}}\n"
    )
    (vendor / "scripts" / "synthetic.yaml").write_text(
        "niah_single_1:\n  task: niah\n  args:\n    type_haystack: noise\n"
    )
    return vendor


def test_prepare_calls_official_generator_with_active_python_and_saves_provenance(
    monkeypatch, tmp_path
):
    adapter = _adapter()
    vendor = _fake_vendor(tmp_path)
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        calls.append(command)
        assert command[0] == sys.executable
        assert kwargs.get("shell", False) is False
        save = Path(command[command.index("--save_dir") + 1]) / "niah_single_1" / "validation.jsonl"
        save.parent.mkdir(parents=True, exist_ok=True)
        row = _source_row("niah_single_1")[0]
        save.write_text(json.dumps(row) + "\n")
        return subprocess.CompletedProcess(command, 0, stdout="generated", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", run)
    result = adapter.prepare_ruler(
        tmp_path / "prepared",
        vendor_dir=vendor,
        tokenizer_path="/cached/Qwen3-4B",
        lengths=[4096],
        samples=1,
        tasks=["niah_single_1"],
        seed=123,
    )
    assert len(result["documents"]) == 1
    assert "--template" in calls[0]
    assert calls[0][calls[0].index("--random_seed") + 1] == "123"
    assert calls[0][calls[0].index("--tokenizer_type") + 1] == "hf"
    assert result["documents"][0]["questions"][0]["metadata"]["requested_length"] == 4096
    assert Path(result["documents_path"]).is_file()
    provenance = json.loads(Path(result["provenance_path"]).read_text())
    assert provenance["tokenizer_path"] == "/cached/Qwen3-4B"
    assert provenance["seed"] == 123
    assert provenance["tasks"] == ["niah_single_1"]
    assert provenance["vendor_revision"] == "abc123"
    assert provenance["sources"][0]["sha256"]


def test_prepare_does_not_accept_generator_missing_output(monkeypatch, tmp_path):
    adapter = _adapter()
    vendor = _fake_vendor(tmp_path)
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="abc123", stderr=""
        ),
    )
    with pytest.raises((RuntimeError, FileNotFoundError)):
        adapter.prepare_ruler(
            tmp_path / "prepared",
            vendor_dir=vendor,
            tokenizer_path="qwen",
            lengths=[4096],
            samples=1,
            tasks=["niah_single_1"],
        )


@pytest.mark.parametrize("valid_hash", [True, False])
def test_materializes_official_lfs_json_and_checks_pointer_hash(monkeypatch, tmp_path, valid_hash):
    adapter = _adapter()
    content = b'{"1": "word"}'
    digest = hashlib.sha256(content).hexdigest() if valid_hash else "0" * 64
    pointer = (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(content)}\n"
    )
    path = tmp_path / "english_words.json"
    path.write_text(pointer)
    monkeypatch.setattr(
        adapter.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(content)
    )
    if valid_hash:
        adapter._download_json(path, ["https://official.example/english_words.json"])
        assert path.read_bytes() == content
    else:
        with pytest.raises(RuntimeError, match="hash"):
            adapter._download_json(path, ["https://official.example/english_words.json"])
        assert path.read_text() == pointer
