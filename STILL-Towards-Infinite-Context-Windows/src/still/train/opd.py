"""Question-independent KV compression with full/evidence on-policy teachers.

Reuse STILL's compactor and cache interface; only the teacher's prefix changes.
Physical cache offsets count compact slots, while RoPE positions continue after
the original source prefix because compact keys retain source-space positions.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from still.attention_bias import enable_still_attention_bias
from still.chat import encode_system_prefix, encode_user_continuation
from still.core import CompactKVCache
from still.core.cache import normalize_past_key_values
from still.data.qasper import answer_f1
from still.eval.common import SYSTEM_PROMPT


@dataclass(frozen=True)
class OPDConfig:
    teacher_context: str = "full"
    max_source_tokens: int = 8192
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

    def __post_init__(self):
        if self.teacher_context not in {"full", "evidence"}:
            raise ValueError("teacher_context must be full or evidence")
        if self.max_source_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("token limits must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("invalid sampling configuration")


def jsd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Equal-weight full-vocabulary JSD, mean over answer positions (natural log).

    Same beta=0.5 objective as OPSDTrainer.generalized_jsd_loss. Compute in
    float32; the teacher is detached, but the mixture retains student gradients.
    Sampling temperature does not change the distillation distributions.
    """
    student = F.log_softmax(student_logits.float(), dim=-1)
    teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    mixture = torch.logaddexp(student, teacher) - 0.6931471805599453
    divergence = (student.exp() * (student - mixture) + teacher.exp() * (teacher - mixture)).sum(
        dim=-1
    ) * 0.5
    return divergence.mean()


def forward_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary KL(teacher || student), averaged over answer positions."""
    student = F.log_softmax(student_logits.float(), dim=-1)
    teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    return (teacher.exp() * (teacher - student)).sum(dim=-1).mean()


def system_prompt(context: str) -> str:
    return SYSTEM_PROMPT.format(context=context)


def question_ids(tokenizer, context: str, question: str, device) -> torch.Tensor:
    return encode_user_continuation(
        tokenizer,
        system_prompt=system_prompt(context),
        user_message=question + "\n\nAnswer concisely using only the provided context.",
    ).to(device)


@torch.no_grad()
def prefill(model, input_ids: torch.Tensor) -> CompactKVCache:
    output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    layers = normalize_past_key_values(output.past_key_values)
    return CompactKVCache(
        keys=[k for k, _ in layers],
        values=[v for _, v in layers],
        detach_tensors=False,
        metadata={"uncompressed": True},
    )


def _forward(model, cache, ids, *, position_start, use_cache, runtime_cache=None, logits_to_keep=1):
    past = runtime_cache if runtime_cache is not None else cache.as_cache(model.config)
    physical_start = past.get_seq_length()
    return model(
        input_ids=ids,
        past_key_values=past,
        use_cache=use_cache,
        position_ids=torch.arange(
            position_start, position_start + ids.shape[-1], device=ids.device
        )[None],
        cache_position=torch.arange(
            physical_start, physical_start + ids.shape[-1], device=ids.device
        ),
        still_layer_biases=None if cache.metadata.get("uncompressed") else cache.biases,
        logits_to_keep=logits_to_keep,
    )


@torch.no_grad()
def rollout(
    model,
    cache,
    prompt_ids,
    *,
    position_start: int,
    config: OPDConfig,
    eos_token_id: int | None,
    greedy: bool = False,
) -> list[int]:
    """Sample from an isolated runtime cache; never mutate the document artifact."""
    declared_eos = getattr(model.generation_config, "eos_token_id", None)
    declared_eos = declared_eos if declared_eos is not None else eos_token_id
    terminal_ids = set(declared_eos if isinstance(declared_eos, (list, tuple)) else [declared_eos])
    output = _forward(model, cache, prompt_ids, position_start=position_start, use_cache=True)
    generated = []
    for index in range(config.max_new_tokens):
        logits = output.logits[:, -1].float()
        if greedy:
            token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / config.temperature
            if config.top_k:
                cutoff = logits.topk(min(config.top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, -torch.inf)
            if config.top_p < 1:
                sorted_logits, indices = logits.sort(dim=-1, descending=True)
                remove = sorted_logits.softmax(-1).cumsum(-1) > config.top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                logits = logits.masked_fill(
                    torch.zeros_like(remove).scatter(1, indices, remove), -torch.inf
                )
            token = torch.multinomial(logits.softmax(-1), num_samples=1)
        token_id = int(token.item())
        generated.append(token_id)
        if token_id in terminal_ids or index + 1 == config.max_new_tokens:
            break
        output = _forward(
            model,
            cache,
            token,
            position_start=position_start + prompt_ids.shape[-1] + index,
            use_cache=True,
            runtime_cache=output.past_key_values,
        )
    return generated


def replay_logits(model, cache, prompt_ids, answer_ids: list[int], *, position_start: int):
    """Predict exactly y[0..T-1] on q + y[:-1], with a fresh cache container."""
    if not answer_ids:
        raise ValueError("cannot replay an empty answer")
    prefix = torch.tensor([answer_ids[:-1]], dtype=prompt_ids.dtype, device=prompt_ids.device)
    ids = torch.cat([prompt_ids, prefix], dim=-1)
    return _forward(
        model,
        cache,
        ids,
        position_start=position_start,
        use_cache=False,
        logits_to_keep=len(answer_ids),
    ).logits[0]


def build_full_document_cache(model, tokenizer, document: dict, config: OPDConfig):
    """Prefill a complete document once, without a question or answer label."""
    model.eval().requires_grad_(False)
    enable_still_attention_bias(model)
    ids = encode_system_prefix(tokenizer, system_prompt(document["document"])).to(model.device)
    source_tokens = int(ids.shape[-1])
    if source_tokens > config.max_source_tokens:
        raise ValueError(
            f"{document['document_id']}: {source_tokens} source tokens exceed "
            f"{config.max_source_tokens}; filter whole documents during preparation"
        )
    full_cache = prefill(model, ids)
    full_cache.metadata.update(
        {"source_tokens": source_tokens, "document_id": document["document_id"]}
    )
    return full_cache


def build_document_cache(model, tokenizer, compactor, document: dict, config: OPDConfig):
    full_cache = build_full_document_cache(model, tokenizer, document, config)
    compact = compactor(full_cache.as_cache(model.config))
    compact.metadata.update(
        {"source_tokens": full_cache.num_tokens, "document_id": document["document_id"]}
    )
    return full_cache, compact


@torch.no_grad()
def generate_teacher_answers(model, tokenizer, document: dict, config: OPDConfig):
    """Generate fixed full-teacher trajectories for the QASPER STILL adaptation.

    Prepare once before training. Neither gold answers nor evidence enter the
    prompt; these trajectories are independent of the changing compact student.
    """
    full = build_full_document_cache(model, tokenizer, document, config)
    return {
        qa["question_id"]: rollout(
            model,
            full,
            question_ids(tokenizer, document["document"], qa["question"], model.device),
            position_start=full.num_tokens,
            config=config,
            eos_token_id=tokenizer.eos_token_id,
            greedy=True,
        )
        for qa in document["questions"]
    }


def document_loss(
    model,
    tokenizer,
    compactor,
    document: dict,
    config: OPDConfig,
    *,
    teacher_answers: dict[str, list[int]] | None = None,
):
    """Compress once; train on student OPD or fixed full-teacher STILL prefixes."""
    if teacher_answers is not None:
        if config.teacher_context != "full":
            raise ValueError("fixed teacher answers require a full-context teacher")
        for qa in document["questions"]:
            if not teacher_answers.get(qa["question_id"]):
                raise ValueError(f"missing or empty teacher answer for {qa['question_id']}")
    full_cache, compact = build_document_cache(model, tokenizer, compactor, document, config)
    losses, questions = [], []
    for qa in document["questions"]:
        prompt = question_ids(tokenizer, document["document"], qa["question"], model.device)
        if teacher_answers is None:
            answer_ids = rollout(
                model,
                compact,
                prompt,
                position_start=full_cache.num_tokens,
                config=config,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            answer_ids = list(teacher_answers[qa["question_id"]])
        with torch.no_grad():
            if config.teacher_context == "full":
                teacher_cache = full_cache
            else:
                evidence_ids = encode_system_prefix(tokenizer, system_prompt(qa["evidence"])).to(
                    model.device
                )
                teacher_cache = prefill(model, evidence_ids)
            teacher_logits = replay_logits(
                model, teacher_cache, prompt, answer_ids, position_start=teacher_cache.num_tokens
            )
        student_logits = replay_logits(
            model, compact, prompt, answer_ids, position_start=full_cache.num_tokens
        )
        loss_fn = jsd_loss if teacher_answers is None else forward_kl_loss
        loss = loss_fn(student_logits, teacher_logits)
        losses.append(loss)
        questions.append(
            {
                "question_id": qa["question_id"],
                "teacher_context": config.teacher_context,
                "trajectory_source": "student" if teacher_answers is None else "full_teacher",
                "objective": "jsd" if teacher_answers is None else "forward_kl",
                "teacher_tokens": teacher_cache.num_tokens,
                "generated_tokens": len(answer_ids),
                "answer_token_ids": answer_ids,
                "student_answer" if teacher_answers is None else "teacher_answer": tokenizer.decode(
                    answer_ids, skip_special_tokens=True
                ),
                "loss": float(loss.detach()),
            }
        )
    if not losses:
        raise ValueError("document must have at least one question")
    loss = torch.stack(losses).mean()
    return loss, {
        "document_id": document["document_id"],
        "source_tokens": full_cache.num_tokens,
        "compact_tokens": compact.num_tokens,
        "compact_shapes": [list(k.shape) for k in compact.keys],
        "loss": float(loss.detach()),
        "questions": questions,
    }


def gradient_stats(compactor) -> dict:
    gradients = [p.grad for p in compactor.parameters() if p.grad is not None]
    return {
        "finite": bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients),
        "norm": float(torch.sqrt(sum(g.float().square().sum() for g in gradients)))
        if gradients
        else 0.0,
        "layers_with_gradient": sum(
            any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in layer.parameters())
            for layer in compactor.layers
        ),
    }


@torch.no_grad()
def evaluate(model, tokenizer, compactor, documents: list[dict], config: OPDConfig):
    """Shared greedy QASPER evaluator; compactor=None evaluates uncompressed KV."""
    predictions, document_scores = [], []
    for document in documents:
        if compactor is None:
            cache = build_full_document_cache(model, tokenizer, document, config)
            position_start = cache.num_tokens
        else:
            full, cache = build_document_cache(model, tokenizer, compactor, document, config)
            position_start = full.num_tokens
            del full
        scores = []
        for qa in document["questions"]:
            prompt = question_ids(tokenizer, document["document"], qa["question"], model.device)
            ids = rollout(
                model,
                cache,
                prompt,
                position_start=position_start,
                config=config,
                eos_token_id=tokenizer.eos_token_id,
                greedy=True,
            )
            answer = tokenizer.decode(ids, skip_special_tokens=True)
            score = answer_f1(answer, qa.get("answers", []))
            scores.append(score)
            predictions.append(
                {
                    "document_id": document["document_id"],
                    "question_id": qa["question_id"],
                    "answer": answer,
                    "f1": score,
                }
            )
        document_scores.append(sum(scores) / len(scores))
    return {
        "question_f1": sum(p["f1"] for p in predictions) / len(predictions),
        "document_f1": sum(document_scores) / len(document_scores),
        "documents": len(documents),
        "questions": len(predictions),
        "predictions": predictions,
    }


@torch.no_grad()
def compare_teachers(model, tokenizer, compactor, document, config):
    """Paired teacher diagnostics on identical initial student trajectories."""
    full, compact = build_document_cache(model, tokenizer, compactor, document, config)
    rows = []
    for qa in document["questions"]:
        prompt = question_ids(tokenizer, document["document"], qa["question"], model.device)
        sampled = rollout(
            model,
            compact,
            prompt,
            position_start=full.num_tokens,
            config=config,
            eos_token_id=tokenizer.eos_token_id,
        )
        student = replay_logits(model, compact, prompt, sampled, position_start=full.num_tokens)
        evidence = prefill(
            model, encode_system_prefix(tokenizer, system_prompt(qa["evidence"])).to(model.device)
        )
        full_logits = replay_logits(model, full, prompt, sampled, position_start=full.num_tokens)
        evidence_logits = replay_logits(
            model, evidence, prompt, sampled, position_start=evidence.num_tokens
        )
        row = {
            "question_id": qa["question_id"],
            "student_trajectory": sampled,
            "jsd_student_full": float(jsd_loss(student, full_logits)),
            "jsd_student_evidence": float(jsd_loss(student, evidence_logits)),
            "jsd_full_evidence": float(jsd_loss(full_logits, evidence_logits)),
            "full_tokens": full.num_tokens,
            "evidence_tokens": evidence.num_tokens,
        }
        for name, cache in [("student", compact), ("full", full), ("evidence", evidence)]:
            ids = rollout(
                model,
                cache,
                prompt,
                position_start=full.num_tokens if name == "student" else cache.num_tokens,
                config=config,
                eos_token_id=tokenizer.eos_token_id,
                greedy=True,
            )
            answer = tokenizer.decode(ids, skip_special_tokens=True)
            row[name + "_answer"] = answer
            row[name + "_f1"] = answer_f1(answer, qa.get("answers", []))
        rows.append(row)
    return {"document_id": document["document_id"], "config": asdict(config), "questions": rows}
