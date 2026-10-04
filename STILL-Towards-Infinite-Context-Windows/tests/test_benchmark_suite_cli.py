import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def test_four_checkpoints_evaluate_four_benchmarks_from_disk(tmp_path):
    root = Path(__file__).resolve().parents[1]
    train_row = {
        "document_id": "train",
        "document": "The red code is ALPHA. The blue code is BETA.",
        "questions": [
            {
                "question_id": "r",
                "question": "What is the red code?",
                "answers": ["ALPHA"],
                "evidence": "The red code is ALPHA.",
            },
            {
                "question_id": "b",
                "question": "What is the blue code?",
                "answers": ["BETA"],
                "evidence": "The blue code is BETA.",
            },
        ],
    }
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text(json.dumps(train_row) + "\n")
    dev.write_text(json.dumps(train_row | {"document_id": "dev"}) + "\n")
    env = os.environ | {"PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"}

    def run(script, *arguments):
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / script), *map(str, arguments)],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    checkpoints = tmp_path / "checkpoints"
    run(
        "run_opd_comparison.py",
        "--methods",
        "still",
        "full",
        "evidence",
        "--tiny",
        "--device",
        "cpu",
        "--train-data",
        train,
        "--eval-data",
        dev,
        "--num-latents",
        "4",
        "--max-new-tokens",
        "2",
        "--output-dir",
        checkpoints,
    )
    metrics = {
        "qasper": "qasper_f1",
        "longbench_v2": "longbench_accuracy",
        "ruler": "ruler_all",
        "nolima": "nolima_contains",
    }
    data = tmp_path / "data"
    run(
        "prepare_benchmarks.py",
        "--benchmarks",
        "qasper",
        "--qasper-data",
        dev,
        "--document-limit",
        "1",
        "--question-limit",
        "1",
        "--max-new-tokens",
        "2",
        "--output-dir",
        data,
    )
    for benchmark, metric in metrics.items():
        if benchmark == "qasper":
            continue
        path = data / benchmark / "prepared" / "documents.jsonl"
        path.parent.mkdir(parents=True)
        row = {
            "benchmark": benchmark,
            "document_id": f"{benchmark}-heldout",
            "document": train_row["document"],
            "prompt_style": "qasper" if benchmark == "qasper" else "context",
            "questions": [
                {
                    "question_id": "r",
                    "question": "What is the red code?",
                    "answers": ["ALPHA"],
                    "metric": metric,
                    "metadata": {
                        "task": "niah_single_1",
                        "difficulty": "easy",
                        "length": "short",
                        "depth": 0.5,
                        "needle_id": "n",
                    },
                }
            ],
        }
        path.write_text(json.dumps(row) + "\n")
    output = tmp_path / "evaluation"
    run(
        "evaluate_benchmarks.py",
        "--data-dir",
        data,
        "--checkpoint-dir",
        checkpoints,
        "--tiny",
        "--device",
        "cpu",
        "--max-new-tokens",
        "2",
        "--output-dir",
        output,
    )
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["results"]) == {"full_context", "still", "full", "evidence"}
    assert summary["metadata"]["backbone_unchanged"]
    ids = {}
    for method, results in summary["results"].items():
        assert set(results) == set(metrics)
        for benchmark, result in results.items():
            assert result["questions"] == 1 and 0 <= result["score"] <= 1
            path = output / method / benchmark / "predictions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert rows[0]["generated_tokens"] > 0
            ids.setdefault(benchmark, []).append([r["question_id"] for r in rows])
        if method != "full_context":
            assert summary["metadata"]["checkpoints"][method]["method"] == method
    assert all(all(group == groups[0] for group in groups) for groups in ids.values())
    assert (output / "comparison.md").exists()


def test_prepare_preserves_full_source_and_selected_provenance(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "raw.jsonl"
    rows = [
        {
            "_id": str(i),
            "context": context,
            "question": "Choose the red code.",
            "choice_A": "ALPHA",
            "choice_B": "BETA",
            "choice_C": "GAMMA",
            "choice_D": "DELTA",
            "answer": "A",
            "difficulty": "easy",
            "length": "short",
            "domain": "QA",
        }
        for i, context in enumerate(["The red code is ALPHA.", "irrelevant words " * 500])
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = tmp_path / "data"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/prepare_benchmarks.py"),
            "--benchmarks",
            "longbench_v2",
            "--longbench-source",
            str(source),
            "--max-context-tokens",
            "256",
            "--max-new-tokens",
            "2",
            "--output-dir",
            str(data),
        ],
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    prepared = data / "longbench_v2" / "prepared"
    selected = prepared / "documents.jsonl"
    full = prepared / "source" / "documents.jsonl"
    assert len(full.read_text().splitlines()) == 2
    assert len(selected.read_text().splitlines()) == 1
    for path in (full, selected):
        provenance = json.loads((path.parent / "provenance.json").read_text())
        assert provenance["documents_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendor_pin_checkout_works_without_git_lfs(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from prepare_benchmarks import VENDOR_REVISIONS, vendor_checkout

    remote = tmp_path / "remote"
    remote.mkdir()
    for key, value in {
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_KEY_0": "filter.lfs.process",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_KEY_1": "filter.lfs.required",
        "GIT_CONFIG_VALUE_1": "true",
        "GIT_CONFIG_KEY_2": "filter.lfs.smudge",
        "GIT_CONFIG_VALUE_2": "false",
    }.items():
        monkeypatch.setenv(key, value)

    def git(*args):
        return subprocess.run(
            [
                "git",
                "-c",
                "filter.lfs.process=",
                "-c",
                "filter.lfs.required=false",
                "-c",
                "filter.lfs.clean=cat",
                "-c",
                "filter.lfs.smudge=cat",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                *args,
            ],
            cwd=remote,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    git("init")
    (remote / ".gitattributes").write_text("asset.txt filter=lfs\n")
    (remote / "asset.txt").write_text("first version\n")
    git("add", ".")
    git("commit", "-m", "first")
    first = git("rev-parse", "HEAD")
    (remote / "asset.txt").write_text("second version\n")
    git("add", ".")
    git("commit", "-m", "second")
    monkeypatch.setitem(VENDOR_REVISIONS, "RULER", first)
    checkout = vendor_checkout(tmp_path / "vendor", "RULER", remote.as_uri())
    assert (checkout / "asset.txt").read_text() == "first version\n"
