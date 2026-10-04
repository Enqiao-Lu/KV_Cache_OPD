import json

from still.eval.truncation import build_truncated_eval_rows


def test_build_truncated_eval_rows_adds_metadata(tmp_path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    rows = [
        {
            "sample_id": "demo",
            "context": "one two three four five six seven eight nine ten",
            "query": "What is the question?",
            "answer_prompt": "Answer briefly.",
            "answers": ["ten"],
            "question_id": "q1",
            "row_hash": "abc",
        }
    ]
    eval_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out_path = tmp_path / "truncated.jsonl"
    truncated_rows = build_truncated_eval_rows(
        eval_path=eval_path,
        output_path=out_path,
        context_token_budget=4,
    )
    assert truncated_rows[0]["truncation_metadata"]["context_token_budget"] == 4
    assert "retained_context_tokens" in truncated_rows[0]["truncation_metadata"]
