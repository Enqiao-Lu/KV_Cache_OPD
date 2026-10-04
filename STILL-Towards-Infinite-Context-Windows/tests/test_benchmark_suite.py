import copy
import json

import pytest
import torch
from test_opd import tiny_components


def sample():
    return {
        "benchmark": "qasper",
        "document_id": "paper",
        "document": "The red code is ALPHA. The blue code is BETA.",
        "prompt_style": "qasper",
        "questions": [
            {
                "question_id": "red",
                "question": "What is the red code?",
                "answers": ["ALPHA"],
                "metric": "qasper_f1",
                "metadata": {},
            },
            {
                "question_id": "blue",
                "question": "What is the blue code?",
                "answers": ["BETA"],
                "metric": "qasper_f1",
                "metadata": {},
            },
        ],
    }


@pytest.mark.parametrize("style", ["qasper", "context"])
@pytest.mark.parametrize("prefix", ["", "The response is: "])
def test_full_cache_matches_uncached_chat_and_question_order(style, prefix):
    from still.benchmarks.suite import evaluate_documents
    from still.eval.common import SYSTEM_PROMPT

    model, tokenizer, _ = tiny_components()
    row = sample() | {"prompt_style": style}
    for qa in row["questions"]:
        qa["answer_prefix"] = prefix
    actual = evaluate_documents(model, tokenizer, None, [row], max_new_tokens=2)
    system = SYSTEM_PROMPT.format(context=row["document"]) if style == "qasper" else row["document"]
    expected = []
    for qa in row["questions"]:
        question = qa["question"]
        if style == "qasper":
            question += "\n\nAnswer concisely using only the provided context."
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(text + prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        generated = []
        with torch.no_grad():
            for _ in range(2):
                token = (
                    model(input_ids=ids, use_cache=False, logits_to_keep=1)
                    .logits[:, -1]
                    .argmax(-1, keepdim=True)
                )
                generated.append(int(token.item()))
                if generated[-1] == tokenizer.eos_token_id:
                    break
                ids = torch.cat([ids, token], -1)
        expected.append(prefix + tokenizer.decode(generated, skip_special_tokens=True))
    assert [p["prediction"] for p in actual["predictions"]] == expected
    row["questions"].reverse()
    reversed_result = evaluate_documents(model, tokenizer, None, [row], max_new_tokens=2)
    assert [p["prediction"] for p in reversed_result["predictions"]] == expected[::-1]
    assert actual["questions"] == 2 and actual["documents"] == 1


def test_compresses_once_and_gold_evidence_are_only_scoring_inputs():
    from still.benchmarks.suite import evaluate_documents

    model, tokenizer, compactor = tiny_components()
    row = sample()
    calls = []
    hook = compactor.register_forward_hook(lambda *_: calls.append(1))
    result = evaluate_documents(model, tokenizer, compactor, [row], max_new_tokens=2)
    hook.remove()
    assert calls == [1]
    changed = copy.deepcopy(row)
    for qa in changed["questions"]:
        qa["answers"] = ["POISONED GOLD"]
        qa["evidence"] = "POISONED PRIVILEGED EVIDENCE"
    other = evaluate_documents(model, tokenizer, compactor, [changed], max_new_tokens=2)
    assert [p["prediction"] for p in result["predictions"]] == [
        p["prediction"] for p in other["predictions"]
    ]
    assert all(p["cache_tokens"] == 4 for p in result["predictions"])
    assert all(p["generated_tokens"] > 0 for p in result["predictions"])
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in compactor.parameters())


def test_context_filter_preserves_whole_documents_and_accounts_for_questions():
    from still.benchmarks.suite import filter_documents, prompt_tokens

    _, tokenizer, _ = tiny_components()
    row = sample()
    count = max(prompt_tokens(tokenizer, row, qa) for qa in row["questions"])
    selected, coverage = filter_documents(
        tokenizer, [row], max_context_tokens=count + 2, max_new_tokens=2
    )
    assert selected == [row] and coverage["selected_documents"] == 1
    selected, coverage = filter_documents(
        tokenizer, [row], max_context_tokens=count + 1, max_new_tokens=2
    )
    assert selected == [] and coverage["excluded_context_documents"] == 1
    assert row["document"] == sample()["document"]


def test_jsonl_validation_rejects_duplicate_questions_and_wrong_metric(tmp_path):
    from still.benchmarks.suite import load_documents

    path = tmp_path / "data.jsonl"
    row = sample()
    path.write_text(json.dumps(row) + "\n")
    assert load_documents(path) == [row]
    row["questions"][1]["question_id"] = "red"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_documents(path)
    row = sample()
    row["questions"][0]["metric"] = "longbench_accuracy"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="metric"):
        load_documents(path)
    row = sample() | {"prompt_style": "context"}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="prompt_style"):
        load_documents(path)


@pytest.mark.parametrize(
    "bad_row",
    [
        None,
        [],
        {
            "benchmark": "qasper",
            "prompt_style": "qasper",
            "document_id": "x",
            "document": "context",
            "questions": [None],
        },
    ],
)
def test_malformed_objects_raise_schema_errors(tmp_path, bad_row):
    from still.benchmarks.suite import load_documents

    path = tmp_path / "malformed.jsonl"
    path.write_text(json.dumps(bad_row) + "\n")
    with pytest.raises(ValueError, match="object"):
        load_documents(path)


def test_checkpoint_loading_is_strict_and_never_uses_random_fallback(tmp_path):
    from still.benchmarks.suite import load_compactor_checkpoint

    model, _, trained = tiny_components()
    path = tmp_path / "full.pt"
    metadata = {
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "test-pinned-revision",
        "tiny": True,
        "num_latents": 4,
        "method": "full",
        "backbone_sha256": "a" * 64,
    }
    torch.save({"state_dict": trained.state_dict(), "metadata": metadata}, path)
    options = {
        "method": "full",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "test-pinned-revision",
        "tiny": True,
    }
    loaded, info = load_compactor_checkpoint(model, path, **options)
    assert info["num_latents"] == 4 and len(info["checkpoint_sha256"]) == 64
    assert all(
        torch.equal(a, b) for a, b in zip(trained.parameters(), loaded.parameters(), strict=True)
    )
    with pytest.raises(ValueError, match="method"):
        load_compactor_checkpoint(model, path, **(options | {"method": "evidence"}))
    with pytest.raises(ValueError, match="revision"):
        load_compactor_checkpoint(model, path, **(options | {"model_revision": "other"}))
    with pytest.raises(ValueError, match="tiny"):
        load_compactor_checkpoint(model, path, **(options | {"tiny": False}))
    with pytest.raises(ValueError, match="latent"):
        load_compactor_checkpoint(model, path, **options, num_latents=8)
    with pytest.raises(FileNotFoundError):
        load_compactor_checkpoint(model, tmp_path / "missing.pt", **options)


def test_checkpoint_requires_backbone_hash_and_checks_actual_backbone(tmp_path):
    from still.benchmarks.suite import load_compactor_checkpoint

    model, _, trained = tiny_components()
    path = tmp_path / "full.pt"
    metadata = {
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "pinned",
        "tiny": True,
        "num_latents": 4,
        "method": "full",
    }
    options = {
        "method": "full",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "pinned",
        "tiny": True,
    }
    torch.save({"state_dict": trained.state_dict(), "metadata": metadata}, path)
    with pytest.raises(ValueError, match="backbone_sha256"):
        load_compactor_checkpoint(model, path, **options)
    metadata["backbone_sha256"] = "a" * 64
    torch.save({"state_dict": trained.state_dict(), "metadata": metadata}, path)
    with pytest.raises(ValueError, match="backbone"):
        load_compactor_checkpoint(model, path, **options, backbone_sha256="b" * 64)
