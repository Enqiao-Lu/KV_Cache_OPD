import ast
import copy
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from still.core import StillCompactor


def tiny_components():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            attention_dropout=0.0,
            max_position_embeddings=2048,
            eos_token_id=tokenizer.eos_token_id,
        )
    )
    model.config._attn_implementation = "eager"
    compactor = StillCompactor.from_model_config(model.config, num_latents=4)
    return model, tokenizer, compactor


def document():
    return {
        "document_id": "d",
        "document": "The red code is ALPHA. The blue code is BETA.",
        "questions": [
            {
                "question_id": "r",
                "question": "What is the red code?",
                "evidence": "The red code is ALPHA.",
                "answers": ["ALPHA"],
            },
            {
                "question_id": "b",
                "question": "What is the blue code?",
                "evidence": "The blue code is BETA.",
                "answers": ["BETA"],
            },
        ],
    }


def test_jsd_matches_opsd_loss_and_detaches_teacher():
    from still.train.opd import jsd_loss

    # Reuse the existing pure OPSD method without importing TRL/DeepSpeed/vLLM.
    opsd_path = Path(__file__).resolve().parents[2] / "OPSD" / "opsd_trainer.py"
    tree = ast.parse(opsd_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OPSDTrainer")
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generalized_jsd_loss"
    )
    method.decorator_list = []
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(opsd_path), "exec"), namespace)
    student = torch.randn(5, 17, requires_grad=True)
    teacher = torch.randn(5, 17, requires_grad=True)
    actual = jsd_loss(student, teacher)
    expected = namespace["generalized_jsd_loss"](student, teacher.detach(), beta=0.5)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert student.grad is not None and student.grad.norm() > 0
    assert teacher.grad is None
    assert 0 <= actual <= torch.log(torch.tensor(2.0))
    assert jsd_loss(student.detach(), student.detach()).abs() < 1e-7


@pytest.mark.parametrize("teacher_context", ["full", "evidence"])
def test_document_opd_updates_only_compactor_and_compresses_once(teacher_context):
    from still.train.opd import OPDConfig, document_loss

    model, tokenizer, compactor = tiny_components()
    backbone = {n: p.detach().clone() for n, p in model.named_parameters()}
    initial = {n: p.detach().clone() for n, p in compactor.named_parameters()}
    calls = []
    handle = compactor.register_forward_hook(lambda *_: calls.append(1))
    loss, metrics = document_loss(
        model,
        tokenizer,
        compactor,
        document(),
        OPDConfig(teacher_context=teacher_context, max_new_tokens=3),
    )
    assert calls == [1]
    assert len(metrics["questions"]) == 2
    assert metrics["compact_tokens"] == 4
    assert torch.isfinite(loss)
    assert metrics["loss"] == pytest.approx(sum(q["loss"] for q in metrics["questions"]) / 2)
    assert all(q["generated_tokens"] > 0 for q in metrics["questions"])
    assert all(q["teacher_context"] == teacher_context for q in metrics["questions"])
    if teacher_context == "evidence":
        assert all(q["teacher_tokens"] < metrics["source_tokens"] for q in metrics["questions"])
    optimizer = torch.optim.AdamW(compactor.parameters(), lr=1e-3)
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    grads = [p.grad for p in compactor.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    assert all(layer.bias_head.weight.grad.abs().sum() > 0 for layer in compactor.layers)
    optimizer.step()
    assert any(not torch.equal(initial[n], p) for n, p in compactor.named_parameters())
    assert all(torch.equal(backbone[n], p) for n, p in model.named_parameters())
    handle.remove()


def test_replay_preserves_source_cache_and_uses_original_rope_positions():
    from still.attention_bias import enable_still_attention_bias
    from still.train.opd import prefill, replay_logits

    model, tokenizer, compactor = tiny_components()
    model.eval().requires_grad_(False)
    enable_still_attention_bias(model)
    source_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70]])
    full = prefill(model, source_ids)
    compact = compactor(full.as_cache(model.config))
    original = [k.detach().clone() for k in compact.keys]
    prompt = torch.tensor([[11, 12]])
    answer = [13, 14, 15]
    actual = replay_logits(model, compact, prompt, answer, position_start=7)
    joined = torch.tensor([[11, 12, 13, 14]])
    expected = model(
        input_ids=joined,
        past_key_values=compact.as_cache(model.config),
        position_ids=torch.arange(7, 11)[None],
        cache_position=torch.arange(4, 8),
        still_layer_biases=compact.biases,
        use_cache=False,
    ).logits[0, 1:4]
    torch.testing.assert_close(actual, expected)
    again = replay_logits(model, compact, prompt, answer, position_start=7)
    torch.testing.assert_close(actual, again)
    assert compact.num_tokens == 4 and full.num_tokens == 7
    assert all(
        torch.equal(before, after) for before, after in zip(original, compact.keys, strict=True)
    )


def test_training_does_not_read_gold_answers():
    from still.train.opd import OPDConfig, document_loss

    model, tokenizer, compactor = tiny_components()
    row = document()
    changed = copy.deepcopy(row)
    for q in changed["questions"]:
        q["answers"] = ["POISONED LABEL THAT MUST NEVER ENTER TRAINING"]
    config = OPDConfig(max_new_tokens=2)
    torch.manual_seed(5)
    first, _ = document_loss(model, tokenizer, compactor, row, config)
    torch.manual_seed(5)
    second, _ = document_loss(model, tokenizer, compactor, changed, config)
    torch.testing.assert_close(first, second)


def test_full_and_evidence_teachers_agree_when_contexts_are_identical():
    from still.train.opd import OPDConfig, document_loss

    model, tokenizer, compactor = tiny_components()
    row = document()
    for q in row["questions"]:
        q["evidence"] = row["document"]
    torch.manual_seed(7)
    full, first = document_loss(
        model, tokenizer, compactor, row, OPDConfig(teacher_context="full", max_new_tokens=2)
    )
    torch.manual_seed(7)
    evidence, second = document_loss(
        model, tokenizer, compactor, row, OPDConfig(teacher_context="evidence", max_new_tokens=2)
    )
    torch.testing.assert_close(full, evidence)
    assert [q["answer_token_ids"] for q in first["questions"]] == [
        q["answer_token_ids"] for q in second["questions"]
    ]


def test_rollout_logits_match_same_prefix_replay():
    from still.attention_bias import enable_still_attention_bias
    from still.train.opd import OPDConfig, prefill, replay_logits, rollout

    model, tokenizer, compactor = tiny_components()
    model.eval().requires_grad_(False)
    enable_still_attention_bias(model)
    full = prefill(model, torch.tensor([[10, 20, 30, 40, 50, 60, 70]]))
    compact = compactor(full.as_cache(model.config))
    prompt = torch.tensor([[11, 12]])
    captured = []
    hook = model.register_forward_hook(lambda _, __, out: captured.append(out.logits[0, -1]))
    generated = rollout(
        model,
        compact,
        prompt,
        position_start=7,
        config=OPDConfig(max_new_tokens=3),
        eos_token_id=None,
    )
    hook.remove()
    replay = replay_logits(model, compact, prompt, generated, position_start=7)
    torch.testing.assert_close(torch.stack(captured), replay, rtol=1e-4, atol=1e-5)
    assert all(not logits.requires_grad for logits in captured)
    assert replay.requires_grad


@pytest.mark.parametrize("terminal_id", [5, 6])
def test_rollout_honors_all_model_declared_eos_tokens(terminal_id):
    from still.attention_bias import enable_still_attention_bias
    from still.train.opd import OPDConfig, prefill, rollout

    model, tokenizer, _ = tiny_components()
    model.eval().requires_grad_(False)
    model.generation_config.eos_token_id = [5, 6]
    enable_still_attention_bias(model)
    full = prefill(model, torch.tensor([[10, 20, 30]]))

    def force_terminal(_, __, output):
        output.logits.fill_(-100)
        output.logits[..., terminal_id] = 100

    hook = model.register_forward_hook(force_terminal)
    generated = rollout(
        model,
        full,
        torch.tensor([[11, 12]]),
        position_start=3,
        config=OPDConfig(max_new_tokens=3),
        eos_token_id=tokenizer.eos_token_id,
        greedy=True,
    )
    hook.remove()
    assert generated == [terminal_id]


def test_forward_kl_matches_teacher_to_student_and_detaches_teacher():
    from still.train.opd import forward_kl_loss

    student = torch.tensor([[0.2, -0.5, 1.4], [1.0, 0.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[1.2, 0.3, -0.7], [-0.1, 1.5, 0.4]], requires_grad=True)
    p = teacher.detach().softmax(-1)
    expected = (p * (p.log() - student.log_softmax(-1))).sum(-1).mean()
    actual = forward_kl_loss(student, teacher)
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, forward_kl_loss(teacher, student))
    actual.backward()
    assert student.grad is not None and student.grad.norm() > 0
    assert teacher.grad is None
    assert forward_kl_loss(student.detach(), student.detach()).abs() < 1e-7


def test_full_kv_evaluation_reuses_one_prefill_and_matches_direct_generation():
    from still.data.qasper import answer_f1
    from still.eval.common import SYSTEM_PROMPT
    from still.train.opd import OPDConfig, evaluate

    model, tokenizer, _ = tiny_components()
    row = document()
    config = OPDConfig(max_new_tokens=2)
    prefills = []

    def count_prefills(_, args, kwargs):
        if kwargs.get("past_key_values") is None:
            prefills.append(1)

    hook = model.register_forward_pre_hook(count_prefills, with_kwargs=True)
    actual = evaluate(model, tokenizer, None, [row], config)
    hook.remove()
    assert prefills == [1]
    expected = []
    for qa in row["questions"]:
        # Independent reference: no cache builder, split prompt or rollout helper.
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(context=row["document"])},
                {
                    "role": "user",
                    "content": (
                        qa["question"] + "\n\nAnswer concisely using only the provided context."
                    ),
                },
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)
        ids = []
        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                token = model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[
                    :, -1
                ].argmax(-1, keepdim=True)
                ids.append(int(token.item()))
                if ids[-1] == tokenizer.eos_token_id:
                    break
                inputs = torch.cat([inputs, token], dim=-1)
        answer = tokenizer.decode(ids, skip_special_tokens=True)
        expected.append(answer)
        assert actual["predictions"][len(expected) - 1]["f1"] == answer_f1(answer, qa["answers"])
    assert [p["answer"] for p in actual["predictions"]] == expected
    assert actual["documents"] == 1 and actual["questions"] == 2
    assert actual["question_f1"] == actual["document_f1"]
    reversed_row = copy.deepcopy(row)
    reversed_row["questions"].reverse()
    reversed_result = evaluate(model, tokenizer, None, [reversed_row], config)
    assert [p["answer"] for p in reversed_result["predictions"]] == expected[::-1]


def test_fixed_full_teacher_answers_ignore_gold_and_evidence():
    from still.train.opd import OPDConfig, generate_teacher_answers

    model, tokenizer, _ = tiny_components()
    row = document()
    poisoned = copy.deepcopy(row)
    for qa in poisoned["questions"]:
        qa["answers"] = ["POISONED GOLD"]
        qa["evidence"] = "POISONED EVIDENCE"
    config = OPDConfig(max_new_tokens=2)
    answers = generate_teacher_answers(model, tokenizer, row, config)
    assert answers == generate_teacher_answers(model, tokenizer, poisoned, config)
    assert set(answers) == {"r", "b"}
    assert all(1 <= len(tokens) <= 2 for tokens in answers.values())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_still_replays_fixed_teacher_prefixes_without_student_sampling(monkeypatch):
    from still.train import opd

    model, tokenizer, compactor = tiny_components()
    row = document()
    references = {"r": [101, 102], "b": [103, 104, 105]}
    config = opd.OPDConfig(max_new_tokens=3)

    def reject_sampling(*args, **kwargs):
        raise AssertionError("fixed-trajectory STILL must not sample from the student")

    monkeypatch.setattr(opd, "rollout", reject_sampling)
    calls = []
    hook = compactor.register_forward_hook(lambda *_: calls.append(1))
    loss, log = opd.document_loss(
        model, tokenizer, compactor, row, config, teacher_answers=references
    )
    hook.remove()
    assert calls == [1]
    assert [q["answer_token_ids"] for q in log["questions"]] == list(references.values())
    assert all(q["trajectory_source"] == "full_teacher" for q in log["questions"])
    assert all(q["objective"] == "forward_kl" for q in log["questions"])
    assert all("student_answer" not in q for q in log["questions"])
    assert log["loss"] == pytest.approx(sum(q["loss"] for q in log["questions"]) / 2)
    changed = copy.deepcopy(row)
    for qa in changed["questions"]:
        qa["answers"] = ["POISONED GOLD"]
        qa["evidence"] = "POISONED EVIDENCE"
    other, _ = opd.document_loss(
        model, tokenizer, compactor, changed, config, teacher_answers=references
    )
    torch.testing.assert_close(loss, other)
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert opd.gradient_stats(compactor)["layers_with_gradient"] == len(compactor.layers)
    assert opd.gradient_stats(compactor)["finite"]


@pytest.mark.parametrize("references", [{}, {"r": [], "b": [103]}])
def test_still_rejects_missing_or_empty_teacher_answers(references):
    from still.train.opd import OPDConfig, document_loss

    model, tokenizer, compactor = tiny_components()
    with pytest.raises(ValueError, match="teacher answer"):
        document_loss(
            model, tokenizer, compactor, document(), OPDConfig(), teacher_answers=references
        )
