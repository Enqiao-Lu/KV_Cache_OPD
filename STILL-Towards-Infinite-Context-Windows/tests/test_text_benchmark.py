import json
from pathlib import Path

from still.benchmarks.text_benchmark import (
    aligned_expected_answer_records,
    build_mcq_answer_records,
    build_mcq_eval_rows,
    generate_teacher_answers,
)
from still.clients.vllm_openai import ChatCompletionResult


class _FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[dict[str, object]] = []
        self.tokenizer = self

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template_kwargs):
        del tokenize, add_generation_prompt, chat_template_kwargs
        return "\n".join(item["content"] for item in messages)

    def _tokenize_via_server(self, prompt: str):
        return list(range(len(prompt.split())))

    def chat(self, *, messages, max_completion_tokens, temperature, run_mode):
        self.calls.append(
            {
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
                "run_mode": run_mode,
            }
        )
        return ChatCompletionResult(
            text="Teacher answer",
            token_ids=[1, 2],
            token_logprobs=[],
            raw_logprobs=[],
            usage=None,
            finish_reason="stop",
            logprob_source="none",
        )

    def close(self) -> None:
        return None


def test_generate_teacher_answers_uses_teacher_client(monkeypatch, tmp_path: Path) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "still.benchmarks.text_benchmark.VLLMClient",
        lambda *args, **kwargs: fake_client,
    )
    output_path = tmp_path / "teacher_answers.jsonl"

    rows = generate_teacher_answers(
        corpus_text="Context text",
        bootstrap_examples=[{"question": "What is the answer?", "expected_answer": "Copied answer"}],
        output_path=output_path,
        base_url="http://127.0.0.1:8000/v1",
        api_key="still-local",
        max_completion_tokens=12,
        model_context_limit=128,
    )

    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["temperature"] == 0.0
    assert rows[0]["assistant_text"] == "Teacher answer"
    assert rows[0]["expected_answer"] == "Copied answer"
    assert rows[0]["teacher_max_completion_tokens_used"] <= 12
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["assistant_text"] == "Teacher answer"


def test_aligned_expected_answer_records_use_bootstrap_targets() -> None:
    records = aligned_expected_answer_records(
        bootstrap_examples=[
            {
                "question": "What is the capital?",
                "expected_answer": "Paris",
            }
        ],
        max_completion_tokens=48,
    )

    assert len(records) == 1
    assert records[0]["question"] == "What is the capital?"
    assert records[0]["assistant_text"] == "Paris"
    assert records[0]["expected_answer"] == "Paris"
    assert records[0]["teacher_finish_reason"] == "aligned_expected_answer"
    assert records[0]["teacher_answer_matches_expected"] is True
    assert "/no_think\nWhat is the capital?\n\n" in records[0]["user_message"]


def test_build_mcq_answer_records_constructs_label_targets() -> None:
    records = build_mcq_answer_records(
        answer_records=[
            {"question": "When was it founded?", "expected_answer": "1998"},
            {"question": "When was it acquired?", "expected_answer": "2001"},
            {"question": "When was it closed?", "expected_answer": "2007"},
            {"question": "When was it reopened?", "expected_answer": "2010"},
        ],
        sample_id="sample_a",
        global_answer_pool=["1998", "2001", "2007", "2010", "2012"],
    )

    assert len(records) == 4
    first = records[0]
    assert first["assistant_text"] in {"A", "B", "C", "D"}
    assert first["expected_answer"] == first["assistant_text"]
    assert len(first["mcq_options"]) == 4
    assert any(option["text"] == "1998" for option in first["mcq_options"])
    assert "Answer with only the single capital letter" in first["user_message"]


def test_build_mcq_eval_rows_sets_prediction_mode_and_letter_gold() -> None:
    rows = build_mcq_eval_rows(
        eval_rows=[
            {
                "sample_id": "heldout_a",
                "context": "ctx",
                "query": "When was it founded?",
                "answer_prompt": "old",
                "answers": ["1998"],
                "question_id": "q1",
                "row_hash": "h1",
            },
            {
                "sample_id": "heldout_a",
                "context": "ctx",
                "query": "When was it acquired?",
                "answer_prompt": "old",
                "answers": ["2001"],
                "question_id": "q2",
                "row_hash": "h2",
            },
            {
                "sample_id": "heldout_a",
                "context": "ctx",
                "query": "When was it closed?",
                "answer_prompt": "old",
                "answers": ["2007"],
                "question_id": "q3",
                "row_hash": "h3",
            },
            {
                "sample_id": "heldout_a",
                "context": "ctx",
                "query": "When was it reopened?",
                "answer_prompt": "old",
                "answers": ["2010"],
                "question_id": "q4",
                "row_hash": "h4",
            },
        ],
        global_answer_pool=["1998", "2001", "2007", "2010"],
    )

    assert len(rows) == 4
    assert all(row["prediction_mode"] == "mcq_letter" for row in rows)
    assert all(row["answers"][0] in {"A", "B", "C", "D"} for row in rows)
    assert all("Options:" in row["query"] for row in rows)
