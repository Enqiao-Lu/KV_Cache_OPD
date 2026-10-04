import json

import torch

from still.train.still import (
    TrainingExample,
    _build_filtered_training_schedule,
    _build_training_schedule,
    _distillation_loss,
    _selection_key,
    _validation_metrics_from_predictions,
    _write_validation_artifacts,
)


def test_training_schedule_covers_dataset_before_repeating() -> None:
    schedule = _build_training_schedule(num_examples=8, steps=8, seed=7)
    assert sorted(schedule) == list(range(8))


def test_training_schedule_is_deterministic_and_extends_by_shuffled_epochs() -> None:
    schedule_a = _build_training_schedule(num_examples=5, steps=12, seed=3)
    schedule_b = _build_training_schedule(num_examples=5, steps=12, seed=3)
    assert schedule_a == schedule_b
    assert len(set(schedule_a[:5])) == 5
    assert len(set(schedule_a[5:10])) == 5


def test_filtered_training_schedule_preserves_shuffle_and_skips_validation_rows() -> None:
    examples = [
        TrainingExample(
            record_id=f"r{idx}",
            slice_id=f"s{idx}",
            system_prompt="system",
            user_message="user",
            assistant_text="A",
            assistant_token_ids=[1],
            prediction_mode="mcq_letter",
        )
        for idx in range(5)
    ]
    schedule = _build_filtered_training_schedule(
        examples=examples,
        steps=4,
        seed=3,
        validation_record_ids={"r1", "r4"},
    )
    assert len(schedule) == 4
    assert all(examples[idx].record_id not in {"r1", "r4"} for idx in schedule)


def test_selection_key_prefers_structurally_valid_checkpoint_over_lower_loss_collapse() -> None:
    collapsed = {
        "passes_structural_decode_gates": False,
        "validation_accuracy": 0.0,
        "validation_loss": 0.5,
        "empty_prediction_rate": 1.0,
    }
    valid = {
        "passes_structural_decode_gates": True,
        "validation_accuracy": 0.25,
        "validation_loss": 1.0,
        "empty_prediction_rate": 0.0,
    }
    assert _selection_key(valid) > _selection_key(collapsed)


def test_validation_metrics_capture_empty_predictions_and_repetition() -> None:
    metrics = _validation_metrics_from_predictions(
        [
            {
                "raw_prediction": ",,,,,",
                "normalized_prediction": "",
                "completion_tokens": 32,
                "finish_reason": "max_completion_tokens",
                "gold_prediction": "A",
            },
            {
                "raw_prediction": "B<|im_end|>",
                "normalized_prediction": "B",
                "completion_tokens": 2,
                "finish_reason": "eos_token",
                "gold_prediction": "B",
            },
        ]
    )
    assert metrics["validation_accuracy"] == 0.5
    assert metrics["empty_prediction_rate"] == 0.5
    assert metrics["max_completion_stop_rate"] == 0.5
    assert metrics["first_token_ad_rate"] == 0.5
    assert metrics["top_repeated_raw_predictions"][0]["count"] == 1


def test_write_validation_artifacts_persists_predictions_and_summary(tmp_path) -> None:
    predictions_path, summary_path = _write_validation_artifacts(
        output_dir=tmp_path,
        step=100,
        validation_predictions=[{"record_id": "r1", "raw_prediction": "A"}],
        validation_summary={"validation_accuracy": 1.0},
    )
    assert predictions_path.is_file()
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["validation_accuracy"] == 1.0


def test_distillation_loss_penalizes_non_option_argmax_for_mcq_tokens() -> None:
    teacher_logits = torch.zeros((2, 8), dtype=torch.float32)
    target_token_ids = [1, 7]

    good_student_logits = torch.full((2, 8), -10.0, dtype=torch.float32)
    good_student_logits[0, 1] = 10.0
    good_student_logits[0, 0] = 9.0
    good_student_logits[1, 7] = 10.0

    bad_student_logits = good_student_logits.clone()
    bad_student_logits[0, 0] = 10.5
    bad_student_logits[0, 1] = 10.0

    good_loss = _distillation_loss(
        teacher_logits=teacher_logits,
        student_logits=good_student_logits,
        target_token_ids=target_token_ids,
        kl_weight=0.0,
        exact_token_ce_weight=1.0,
    )
    bad_loss = _distillation_loss(
        teacher_logits=teacher_logits,
        student_logits=bad_student_logits,
        target_token_ids=target_token_ids,
        kl_weight=0.0,
        exact_token_ce_weight=1.0,
    )

    assert bad_loss.item() > good_loss.item()
