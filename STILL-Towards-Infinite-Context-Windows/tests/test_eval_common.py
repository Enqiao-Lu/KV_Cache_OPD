from types import SimpleNamespace

from still.eval.common import build_decode_debug_metadata, exact_match


def test_exact_match_normalizes_gold_text_for_mcq_letters() -> None:
    assert exact_match("C", ["C"])
    assert exact_match("c", ["C"])
    assert exact_match("1998", ["1998"])


def test_build_decode_debug_metadata_records_raw_tokens_and_finish_reason() -> None:
    tokenizer = SimpleNamespace(decode=lambda token_ids, skip_special_tokens=False: "A<|im_end|>")
    metadata = build_decode_debug_metadata(
        row={"prediction_mode": "mcq_letter"},
        tokenizer=tokenizer,
        generated_ids=[32, 151645],
        normalized_prediction="A",
        eos_token_id=151645,
        max_completion_tokens=32,
    )
    assert metadata["raw_prediction"] == "A<|im_end|>"
    assert metadata["generated_token_ids"] == [32, 151645]
    assert metadata["normalized_prediction"] == "A"
    assert metadata["finish_reason"] == "eos_token"
