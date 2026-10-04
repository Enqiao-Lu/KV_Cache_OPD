import json
import os
import subprocess
import sys
from pathlib import Path


def test_runner_compares_three_baselines_without_evidence_training(tmp_path):
    root = Path(__file__).resolve().parents[1]
    row = {
        "document_id": "training-paper",
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
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text(json.dumps(row) + "\n")
    dev.write_text(json.dumps(row | {"document_id": "held-out-paper"}) + "\n")
    output = tmp_path / "results"
    env = os.environ | {"PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_opd_comparison.py"),
            "--methods",
            "still",
            "full",
            "--tiny",
            "--device",
            "cpu",
            "--train-data",
            str(train),
            "--eval-data",
            str(dev),
            "--num-latents",
            "4",
            "--max-new-tokens",
            "2",
            "--output-dir",
            str(output),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["results"]) == {"full_context", "still", "full"}
    assert summary["results"]["full_context"]["evaluation"]["questions"] == 2
    assert summary["results"]["full_context"]["train_seconds"] == 0
    assert summary["results"]["still"]["objective"] == "forward_kl"
    assert summary["results"]["full"]["objective"] == "jsd"
    for method in ("still", "full"):
        arm = summary["results"][method]
        assert arm["backbone_unchanged"] and arm["checkpoint_roundtrip"]
        assert arm["initial_compactor_sha256"] == summary["metadata"]["initial_compactor_sha256"]
        assert arm["after_evaluation"]["questions"] == 2
        assert arm["steps"][0]["gradient"]["finite"]
    references = json.loads((output / "still_teacher_trajectories.json").read_text())
    for qa in summary["results"]["still"]["steps"][0]["questions"]:
        assert qa["answer_token_ids"] == references["answers"]["training-paper"][qa["question_id"]]
        assert qa["trajectory_source"] == "full_teacher"
    assert not (output / "teacher_diagnostics.json").exists()
    assert not (output / "evidence_compactor.pt").exists()
