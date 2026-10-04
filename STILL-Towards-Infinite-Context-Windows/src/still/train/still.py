import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from still.attention_bias import enable_still_attention_bias
from still.chat import encode_system_prefix, encode_user_continuation
from still.config import DEFAULT_MATRIX
from still.core import CompactKVCache, StillCompactor
from still.data.common import stable_hash, write_json
from still.eval.common import normalize_generated_text_for_row


@dataclass(frozen=True)
class TrainingExample:
    """One teacher-supervised training row consumed by the STILL trainer."""
    record_id: str
    slice_id: str
    system_prompt: str
    user_message: str
    assistant_text: str
    assistant_token_ids: list[int]
    prediction_mode: str | None


@dataclass(frozen=True)
class ValidationExample:
    """One held-out training row used only for checkpoint selection."""
    training_example: TrainingExample
    gold_prediction: str


def _set_training_seed(seed: int) -> None:
    """Seed Python and Torch so repeated training runs stay comparable."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_training_schedule(num_examples: int, steps: int, seed: int) -> list[int]:
    """Build a shuffled training schedule that covers the dataset before repeating."""
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    rng = random.Random(seed)
    schedule: list[int] = []
    while len(schedule) < steps:
        epoch_indices = list(range(num_examples))
        rng.shuffle(epoch_indices)
        schedule.extend(epoch_indices)
    return schedule[:steps]


def _build_filtered_training_schedule(
    *,
    examples: list[TrainingExample],
    steps: int,
    seed: int,
    validation_record_ids: set[str],
) -> list[int]:
    """Preserve the original shuffled order while skipping held-out validation rows."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    rng = random.Random(seed)
    schedule: list[int] = []
    while len(schedule) < steps:
        epoch_indices = list(range(len(examples)))
        rng.shuffle(epoch_indices)
        for example_idx in epoch_indices:
            if examples[example_idx].record_id in validation_record_ids:
                continue
            schedule.append(example_idx)
            if len(schedule) == steps:
                break
    return schedule


def load_training_examples(path: str | Path) -> list[TrainingExample]:
    """Load the JSONL training dataset emitted by build_training_dataset."""
    rows: list[TrainingExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                TrainingExample(
                    record_id=row["record_id"],
                    slice_id=row["slice_ids"][0],
                    system_prompt=row["system_prompt"],
                    user_message=row["messages"][0]["content"],
                    assistant_text=row["messages"][1]["content"],
                    assistant_token_ids=[int(token_id) for token_id in row["assistant_token_ids"]],
                    prediction_mode=row.get("metadata", {}).get("prediction_mode"),
                )
            )
    if not rows:
        raise ValueError(f"No training examples found in {path}.")
    return rows


def _gold_prediction_for_example(example: TrainingExample) -> str:
    """Normalize the gold answer into the same form used for decoded validation outputs."""
    return normalize_generated_text_for_row(
        {"prediction_mode": example.prediction_mode},
        example.assistant_text,
    )


def _split_examples_for_validation(
    examples: list[TrainingExample],
    *,
    validation_examples: int,
    seed: int,
) -> tuple[list[TrainingExample], list[ValidationExample]]:
    """Reserve a deterministic held-out validation subset from distinct slice ids when possible."""
    if validation_examples >= len(examples):
        raise ValueError("validation_examples must be smaller than the full dataset.")

    by_slice: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        by_slice[example.slice_id].append(example)
    for grouped_examples in by_slice.values():
        grouped_examples.sort(key=lambda example: example.record_id)

    rng = random.Random(seed)
    slice_ids = sorted(by_slice)
    rng.shuffle(slice_ids)

    validation_records: list[TrainingExample] = []
    selected_record_ids: set[str] = set()
    for slice_id in slice_ids:
        candidate = by_slice[slice_id][0]
        validation_records.append(candidate)
        selected_record_ids.add(candidate.record_id)
        if len(validation_records) == validation_examples:
            break

    if len(validation_records) < validation_examples:
        remaining_examples = [
            example
            for example in examples
            if example.record_id not in selected_record_ids
        ]
        remaining_examples.sort(key=lambda example: example.record_id)
        rng.shuffle(remaining_examples)
        for example in remaining_examples:
            validation_records.append(example)
            selected_record_ids.add(example.record_id)
            if len(validation_records) == validation_examples:
                break

    training_examples = [
        example for example in examples if example.record_id not in selected_record_ids
    ]
    validation_subset = [
        ValidationExample(
            training_example=example,
            gold_prediction=_gold_prediction_for_example(example),
        )
        for example in validation_records
    ]
    return training_examples, validation_subset


def _kl_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Compute token-level KL divergence from teacher logits to student logits."""
    teacher = F.softmax(teacher_logits.float(), dim=-1)
    student = F.log_softmax(student_logits.float(), dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean")


def _exact_token_ce_loss(
    student_logits: torch.Tensor,
    target_token_ids: list[int],
) -> torch.Tensor:
    """Compute cross-entropy against the exact answer-token sequence."""
    targets = torch.tensor(target_token_ids, device=student_logits.device, dtype=torch.long)
    return F.cross_entropy(student_logits.float(), targets, reduction="mean")


def _distillation_loss(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_token_ids: list[int],
    kl_weight: float,
    exact_token_ce_weight: float,
) -> torch.Tensor:
    """Combine KL and exact-token losses according to the configured training weights."""
    loss = student_logits.new_tensor(0.0, dtype=torch.float32)
    if kl_weight > 0.0:
        loss = loss + (kl_weight * _kl_loss(teacher_logits, student_logits))
    if exact_token_ce_weight > 0.0:
        # MCQ rows are evaluated by greedy decoding, so training must penalize
        # any non-letter token winning the full-vocabulary argmax. The previous
        # A/B/C/D-only loss let punctuation or control tokens dominate while the
        # correct letter merely ranked highest inside the restricted slice.
        ce_term = _exact_token_ce_loss(student_logits, target_token_ids)
        loss = loss + (exact_token_ce_weight * ce_term)
    return loss


def _teacher_and_student_logits(
    *,
    model,
    tokenizer,
    compactor: StillCompactor,
    example: TrainingExample,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, CompactKVCache, int]:
    """Run frozen teacher and compact-cache student forwards for one training example."""
    context_ids = encode_system_prefix(tokenizer, example.system_prompt).to(device)
    prompt_ids = encode_user_continuation(
        tokenizer,
        system_prompt=example.system_prompt,
        user_message=example.user_message,
    ).to(device)

    if len(example.assistant_token_ids) > 1:
        assistant_prefix = torch.tensor(
            [example.assistant_token_ids[:-1]],
            device=device,
            dtype=prompt_ids.dtype,
        )
        model_input = torch.cat([prompt_ids, assistant_prefix], dim=-1)
    else:
        model_input = prompt_ids

    with torch.no_grad():
        full_outputs = model(input_ids=context_ids, use_cache=True)
        teacher_outputs = model(
            input_ids=model_input,
            past_key_values=full_outputs.past_key_values,
            use_cache=False,
        )

    compact_cache = compactor(full_outputs.past_key_values)
    student_outputs = model(
        input_ids=model_input,
        past_key_values=compact_cache.as_cache(model.config),
        still_layer_biases=compact_cache.biases,
        use_cache=False,
    )
    target_len = len(example.assistant_token_ids)
    start_idx = prompt_ids.shape[-1] - 1
    end_idx = start_idx + target_len
    return (
        teacher_outputs.logits[0, start_idx:end_idx, :],
        student_outputs.logits[0, start_idx:end_idx, :],
        compact_cache,
        int(context_ids.shape[-1]),
    )


def _decode_student_prediction(
    *,
    model,
    tokenizer,
    compactor: StillCompactor,
    example: TrainingExample,
    device: str,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Greedily decode one validation answer from the compact-cache student path."""
    context_ids = encode_system_prefix(tokenizer, example.system_prompt).to(device)
    prompt_ids = encode_user_continuation(
        tokenizer,
        system_prompt=example.system_prompt,
        user_message=example.user_message,
    ).to(device)
    eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        full_outputs = model(input_ids=context_ids, use_cache=True)
        compact_cache = compactor(full_outputs.past_key_values)
        outputs = model(
            input_ids=prompt_ids,
            past_key_values=compact_cache.as_cache(model.config),
            still_layer_biases=compact_cache.biases,
            use_cache=True,
        )
        generated_ids: list[int] = []
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        for _ in range(max_completion_tokens):
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                still_layer_biases=compact_cache.biases,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    raw_prediction = tokenizer.decode(generated_ids, skip_special_tokens=False)
    normalized_prediction = normalize_generated_text_for_row(
        {"prediction_mode": example.prediction_mode},
        raw_prediction,
    )
    return {
        "raw_prediction": raw_prediction,
        "normalized_prediction": normalized_prediction,
        "generated_token_ids": generated_ids,
        "completion_tokens": len(generated_ids),
        "finish_reason": (
            "eos_token"
            if generated_ids and eos_token_id is not None and generated_ids[-1] == eos_token_id
            else "max_completion_tokens"
            if len(generated_ids) >= max_completion_tokens
            else "stopped"
        ),
    }


def _validation_metrics_from_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize validation decode behavior for checkpoint selection."""
    total = len(predictions)
    if total == 0:
        raise ValueError("Validation predictions must not be empty.")
    raw_counter = Counter(prediction["raw_prediction"] for prediction in predictions)
    first_token_letters = 0
    empty_predictions = 0
    max_stop_predictions = 0
    correct_predictions = 0
    for prediction in predictions:
        normalized = prediction["normalized_prediction"]
        if normalized in {"A", "B", "C", "D"}:
            first_token_letters += 1
        if not normalized:
            empty_predictions += 1
        if prediction["finish_reason"] == "max_completion_tokens":
            max_stop_predictions += 1
        if normalized == prediction["gold_prediction"]:
            correct_predictions += 1
    top_repeated = [
        {"raw_prediction": raw_prediction, "count": count}
        for raw_prediction, count in raw_counter.most_common(5)
    ]
    dominant_count = top_repeated[0]["count"] if top_repeated else 0
    return {
        "validation_accuracy": correct_predictions / total,
        "empty_prediction_rate": empty_predictions / total,
        "max_completion_stop_rate": max_stop_predictions / total,
        "first_token_ad_rate": first_token_letters / total,
        "mean_completion_tokens": sum(
            prediction["completion_tokens"] for prediction in predictions
        )
        / total,
        "dominant_raw_output_rate": dominant_count / total,
        "top_repeated_raw_predictions": top_repeated,
    }


def _passes_structural_decode_gates(metrics: dict[str, Any]) -> bool:
    """Reject checkpoints that show the collapse modes already observed locally."""
    return (
        metrics["empty_prediction_rate"] <= 0.25
        and metrics["max_completion_stop_rate"] <= 0.25
        and metrics["first_token_ad_rate"] >= 0.75
        and metrics["dominant_raw_output_rate"] <= 0.5
    )


def _selection_key(validation_summary: dict[str, Any]) -> tuple[float, float, float, float]:
    """Rank checkpoint candidates by structural validity first, then quality, then loss."""
    passes = 1.0 if validation_summary["passes_structural_decode_gates"] else 0.0
    return (
        passes,
        float(validation_summary["validation_accuracy"]),
        -float(validation_summary["validation_loss"]),
        -float(validation_summary["empty_prediction_rate"]),
    )


def _write_validation_artifacts(
    *,
    output_dir: Path,
    step: int,
    validation_predictions: list[dict[str, Any]],
    validation_summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist per-checkpoint validation artifacts so selector decisions are inspectable."""
    step_dir = output_dir / "validation" / f"step_{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = step_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in validation_predictions:
            handle.write(json.dumps(row))
            handle.write("\n")
    summary_path = step_dir / "summary.json"
    write_json(summary_path, validation_summary)
    return predictions_path, summary_path


def collect_compactor_gradient_stats(
    *,
    model,
    tokenizer,
    compactor: StillCompactor,
    example: TrainingExample,
    device: str,
    kl_weight: float = 1.0,
    exact_token_ce_weight: float = 1.0,
) -> dict[str, float | bool]:
    """Probe whether key STILL parameters receive finite gradients at initialization."""
    compactor.zero_grad(set_to_none=True)
    teacher_logits, student_logits, _, _ = _teacher_and_student_logits(
        model=model,
        tokenizer=tokenizer,
        compactor=compactor,
        example=example,
        device=device,
    )
    loss = _distillation_loss(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        target_token_ids=example.assistant_token_ids,
        kl_weight=kl_weight,
        exact_token_ce_weight=exact_token_ce_weight,
    )
    loss.backward()

    def _norm(parameter_name: str) -> float:
        parameter = dict(compactor.named_parameters())[parameter_name]
        if parameter.grad is None:
            return 0.0
        return float(parameter.grad.float().norm().item())

    stats = {
        "loss": float(loss.item()),
        "bias_head_weight_grad_norm": _norm("layers.0.bias_head.weight"),
        "bias_head_bias_grad_norm": _norm("layers.0.bias_head.bias"),
        "q_proj_bias_grad_norm": _norm("layers.0.blocks.0.cross_attn.q_proj.bias"),
        "k_proj_bias_grad_norm": _norm("layers.0.blocks.0.cross_attn.k_proj.bias"),
    }
    stats["all_finite"] = all(torch.isfinite(torch.tensor(value)) for value in stats.values())
    compactor.zero_grad(set_to_none=True)
    return stats


def collect_initial_locality_stats(
    *,
    compactor: StillCompactor,
    past_key_values,
) -> dict[str, float]:
    """Measure whether the initial latent routing stays near its intended positions."""
    first_layer = compactor.layers[0]
    compact_keys, compact_values, compact_biases, attention_weights = first_layer(
        past_key_values[0][0],
        past_key_values[0][1],
        return_attention_weights=True,
    )
    del compact_keys, compact_values, compact_biases
    weights = attention_weights.squeeze(0).mean(dim=0)
    seq_len = weights.shape[-1]
    latent_positions = first_layer._latent_positions(seq_len, weights.device).to(torch.float32)
    argmax_positions = weights.argmax(dim=-1).to(torch.float32)
    mean_position_error = float((argmax_positions - latent_positions).abs().mean().item())
    max_position_error = float((argmax_positions - latent_positions).abs().max().item())
    return {
        "mean_position_error": mean_position_error,
        "max_position_error": max_position_error,
    }


def build_still_cache(
    *,
    compactor_path: str | Path,
    system_prompt: str,
    output_path: str | Path,
    device: str = "cuda:0",
    model=None,
    tokenizer=None,
    compactor=None,
    compactor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prefill the full model once, compact the cache, and save the compact artifact."""
    owns_model = model is None or tokenizer is None
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = model or AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    if owns_model:
        model.to(device)
        model.eval()
    enable_still_attention_bias(model)

    if compactor is None:
        compactor = StillCompactor.from_model_config(model.config, num_latents=1)
        state_dict = torch.load(compactor_path, map_location="cpu", weights_only=False)
        if "state_dict" not in state_dict:
            raise ValueError(f"Expected a STILL compactor checkpoint at {compactor_path}.")
        metadata = dict(state_dict.get("metadata", {}))
        num_latents = int(metadata["num_latents"])
        compactor = StillCompactor.from_model_config(model.config, num_latents=num_latents)
        compactor.load_state_dict(state_dict["state_dict"])
        compactor.to(device)
        compactor.eval()
    else:
        metadata = dict(compactor_metadata or {})

    context_ids = encode_system_prefix(tokenizer, system_prompt).to(device)
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=context_ids, use_cache=True)
        # Cache building is a single forward pass of the frozen model plus one compactor pass.
        cache = compactor(outputs.past_key_values)
    build_seconds = time.perf_counter() - started
    cache.metadata.update(
        {
            **metadata,
            "source_prompt_tokens": int(context_ids.shape[-1]),
            "build_seconds": build_seconds,
        }
    )
    cache.save(output_path)
    return {
        "cache_path": str(Path(output_path).resolve()),
        "build_seconds": build_seconds,
        "source_prompt_tokens": int(context_ids.shape[-1]),
        "compact_tokens": cache.num_tokens,
        "canonical_kv_bytes": cache.canonical_kv_bytes(),
    }


def train_still(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    num_latents: int = 1024,
    learning_rate: float = 2e-4,
    steps: int = 120,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    validation_examples: int = 16,
    validation_interval: int = 10,
    kl_weight: float = 1.0,
    exact_token_ce_weight: float = 1.0,
    max_completion_tokens: int = 32,
) -> dict[str, Any]:
    """Train the reusable STILL compactor on the prepared supervision dataset."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if validation_examples <= 0:
        raise ValueError("validation_examples must be positive.")
    if validation_interval <= 0:
        raise ValueError("validation_interval must be positive.")
    if exact_token_ce_weight < 0:
        raise ValueError("exact_token_ce_weight must be non-negative.")
    if kl_weight < 0:
        raise ValueError("kl_weight must be non-negative.")
    if max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive.")

    _set_training_seed(seed)
    examples = load_training_examples(dataset_path)
    training_examples, validation_subset = _split_examples_for_validation(
        examples,
        validation_examples=min(validation_examples, len(examples) - 1),
        seed=seed,
    )
    validation_record_ids = {
        item.training_example.record_id for item in validation_subset
    }
    # Preserve the original full-dataset shuffle order so validation holdout
    # does not accidentally change the first few training examples and memory profile.
    training_schedule = _build_filtered_training_schedule(
        examples=examples,
        steps=steps,
        seed=seed,
        validation_record_ids=validation_record_ids,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    enable_still_attention_bias(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    compactor = StillCompactor.from_model_config(model.config, num_latents=num_latents)
    compactor.to(device)
    optimizer = AdamW(compactor.parameters(), lr=learning_rate)
    initial_locality_stats = None
    initial_gradient_stats = None

    loss_history: list[float] = []
    validation_history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_step = 0
    best_validation_summary: dict[str, Any] | None = None
    best_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in compactor.state_dict().items()
    }
    start_time = time.perf_counter()

    with torch.inference_mode():
        context_ids = encode_system_prefix(tokenizer, examples[0].system_prompt).to(device)
        full_outputs = model(input_ids=context_ids, use_cache=True)
        # These diagnostics make it obvious when identity routing or beta gradients are broken.
        initial_locality_stats = collect_initial_locality_stats(
            compactor=compactor,
            past_key_values=full_outputs.past_key_values,
        )
    initial_gradient_stats = collect_compactor_gradient_stats(
        model=model,
        tokenizer=tokenizer,
        compactor=compactor,
        example=examples[0],
        device=device,
        kl_weight=kl_weight,
        exact_token_ce_weight=exact_token_ce_weight,
    )

    for step_idx, example_idx in enumerate(training_schedule):
        example = examples[example_idx]
        # Teacher logits come from the full cache; student logits come from the compact cache built by STILL.
        teacher_logits, student_logits, compact_cache, source_tokens = _teacher_and_student_logits(
            model=model,
            tokenizer=tokenizer,
            compactor=compactor,
            example=example,
            device=device,
        )
        loss = _distillation_loss(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            target_token_ids=example.assistant_token_ids,
            kl_weight=kl_weight,
            exact_token_ce_weight=exact_token_ce_weight,
        )
        loss_history.append(float(loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compactor.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed_step = step_idx + 1
        should_validate = completed_step == steps or completed_step % validation_interval == 0
        if should_validate:
            with torch.inference_mode():
                validation_losses: list[float] = []
                validation_predictions: list[dict[str, Any]] = []
                for validation_item in validation_subset:
                    validation_example = validation_item.training_example
                    teacher_logits, student_logits, _, _ = _teacher_and_student_logits(
                        model=model,
                        tokenizer=tokenizer,
                        compactor=compactor,
                        example=validation_example,
                        device=device,
                    )
                    validation_losses.append(
                        float(
                            _distillation_loss(
                                teacher_logits=teacher_logits,
                                student_logits=student_logits,
                                target_token_ids=validation_example.assistant_token_ids,
                                kl_weight=kl_weight,
                                exact_token_ce_weight=exact_token_ce_weight,
                            ).item()
                        )
                    )
                    decode_result = _decode_student_prediction(
                        model=model,
                        tokenizer=tokenizer,
                        compactor=compactor,
                        example=validation_example,
                        device=device,
                        max_completion_tokens=max_completion_tokens,
                    )
                    validation_predictions.append(
                        {
                            "record_id": validation_example.record_id,
                            "slice_id": validation_example.slice_id,
                            "gold_prediction": validation_item.gold_prediction,
                            **decode_result,
                        }
                    )
            # The saved checkpoint is the best validation compactor, not simply the last training step.
            validation_loss = sum(validation_losses) / len(validation_losses)
            validation_metrics = _validation_metrics_from_predictions(validation_predictions)
            validation_summary = {
                "step": completed_step,
                "validation_loss": validation_loss,
                "recent_train_loss": float(loss.item()),
                "passes_structural_decode_gates": _passes_structural_decode_gates(
                    validation_metrics
                ),
                **validation_metrics,
            }
            predictions_path, summary_path = _write_validation_artifacts(
                output_dir=output_dir,
                step=completed_step,
                validation_predictions=validation_predictions,
                validation_summary=validation_summary,
            )
            validation_summary["predictions_path"] = str(predictions_path.resolve())
            validation_summary["summary_path"] = str(summary_path.resolve())
            validation_history.append(validation_summary)
            if best_validation_summary is None or _selection_key(
                validation_summary
            ) > _selection_key(best_validation_summary):
                best_loss = validation_loss
                best_step = completed_step
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in compactor.state_dict().items()
                }
                best_validation_summary = dict(validation_summary)

    compactor.load_state_dict(best_state_dict)
    checkpoint_path = output_dir / "still_compactor.pt"
    torch.save(
        {
            "state_dict": compactor.state_dict(),
            "metadata": {
                "num_latents": num_latents,
                "dataset_path": str(Path(dataset_path).resolve()),
                "model_id": DEFAULT_MATRIX.model_id,
                "seed": seed,
            },
        },
        checkpoint_path,
    )

    train_seconds = time.perf_counter() - start_time
    summary = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "compactor_path": str(checkpoint_path.resolve()),
        "steps": steps,
        "num_latents": num_latents,
        "learning_rate": learning_rate,
        "seed": seed,
        "validation_examples": len(validation_subset),
        "validation_interval": validation_interval,
        "initial_loss": loss_history[0],
        "best_loss": best_loss,
        "best_step": best_step,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "validation_history": validation_history,
        "loss_decreased": loss_history[-1] < loss_history[0],
        "train_seconds": train_seconds,
        "kl_weight": kl_weight,
        "exact_token_ce_weight": exact_token_ce_weight,
        "max_completion_tokens": max_completion_tokens,
        "num_examples": len(examples),
        "num_training_examples": len(training_examples),
        "num_validation_examples": len(validation_subset),
        "unique_examples_seen": len(set(training_schedule)),
        "source_prompt_tokens": source_tokens,
        "compact_tokens": compact_cache.num_tokens,
        "compact_kv_bytes": compact_cache.canonical_kv_bytes(),
        "initial_gradient_stats": initial_gradient_stats,
        "initial_locality_stats": initial_locality_stats,
        "validation_record_ids": [
            item.training_example.record_id for item in validation_subset
        ],
        "checkpoint_selection_rule": (
            "passes_structural_decode_gates > validation_accuracy > "
            "-validation_loss > -empty_prediction_rate"
        ),
        "best_validation_summary": best_validation_summary,
        "manifest_hash": stable_hash(
            {
                "dataset_path": str(Path(dataset_path).resolve()),
                "steps": steps,
                "seed": seed,
                "num_latents": num_latents,
            }
        ),
    }
    write_json(output_dir / "still_summary.json", summary)
    return summary
