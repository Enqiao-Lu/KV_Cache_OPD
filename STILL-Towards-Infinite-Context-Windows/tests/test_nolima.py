import ast
import importlib.util
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

OFFICIAL_SCORER = '''
def _evaluate_response(self, response: str, gold_answers=None) -> int:
    if gold_answers is None:
        gold_answers = self.gold_answers
    if self.metric == "EM":
        return int(response.strip() in gold_answers)
    elif self.metric == "contains":
        return int(any([gold_answer in response for gold_answer in gold_answers]))
    elif self.metric == "lastline_EM":
        return int(response.strip().split("\\n")[-1] in gold_answers)
    elif self.metric == "lastline_contains":
        return int(any([gold_answer in response.strip().split("\\n")[-1]
                        for gold_answer in gold_answers]))
    else:
        raise ValueError(f"Invalid metric: {self.metric}")
'''


@pytest.mark.parametrize("mode", ["EM", "contains", "lastline_EM", "lastline_contains"])
@pytest.mark.parametrize("prediction", [" Alice ", "ALICE", "Alice.\nBob", "Reasoning\nAlice", ""])
def test_scoring_matches_official_response_method(mode, prediction):
    from still.benchmarks.nolima import score_nolima

    namespace = {}
    exec(compile(ast.parse(OFFICIAL_SCORER), "official_nolima_scorer", "exec"), namespace)
    expected = namespace["_evaluate_response"](
        SimpleNamespace(metric=mode), prediction, ["Alice", "Megan"]
    )
    assert score_nolima(prediction, ["Alice", "Megan"], mode=mode) == expected


def test_unknown_score_mode_is_rejected():
    from still.benchmarks.nolima import score_nolima

    with pytest.raises(ValueError, match="metric"):
        score_nolima("Alice", ["Alice"], mode="lowercase_em")


class CharacterTokenizer:
    name_or_path = "fixture-character-tokenizer"
    init_kwargs = {}

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("\n".join(m["content"] for m in messages))


@pytest.fixture
def official_data_fixture(tmp_path, monkeypatch):
    # Exact non-distractor placement arithmetic from official BookHaystack.
    vendor = tmp_path / "vendor"
    (vendor / "data").mkdir(parents=True)
    (vendor / "evaluation/run_config").mkdir(parents=True)
    (vendor / "data/book_haystack.py").write_text('''
from pathlib import Path
import numpy as np
class BookHaystack:
    def __init__(self, book_path):
        self.text = Path(book_path).read_text()
        self.text_encoded = None
    def generate_w_needle_placement(self, needle, token_count_func, encoding_func,
                                   decoding_func, context_length, shift=0,
                                   depth=0.5, static_depth=-1, distractor=None):
        assert static_depth == -1 and distractor is None
        if self.text_encoded is None:
            self.text_encoded = encoding_func(self.text)
            self.text_tokens = [decoding_func([i]) for i in self.text_encoded]
            valid = [0]
            for i in range(len(self.text_tokens)):
                if "\\n" in self.text_tokens[i]:
                    valid.append(i)
            self.valid_positions_token_counts = np.array(valid)
        delta = (self.valid_positions_token_counts - shift) / (context_length + 1) - depth
        delta[delta < 0] = np.inf
        closest = np.argmin(delta)
        start = self.valid_positions_token_counts[closest] - int(np.round(context_length * depth))
        end = start + context_length
        assert start >= 0 and end <= len(self.text_encoded)
        pre = decoding_func(self.text_encoded[start:self.valid_positions_token_counts[closest]+1])
        pre = pre[:max(pre.rfind("\\n"), 0)]
        post = decoding_func(self.text_encoded[self.valid_positions_token_counts[closest]+1:end])
        return {"text": pre + " " + needle + "\\n" + post,
                "static_depth": static_depth,
                "token_depth": int(np.round(context_length * depth)), "depth": depth,
                "context_length_wo_needle": context_length}
''')
    template = (
        "You will answer a question based on the following book snippet:\n\n{haystack}"
        "\n\nUse the information provided in the book snippet to answer the question."
        "\n\nQuestion: {question}\n\n Return only the final answer."
    )
    (vendor / "evaluation/run_tests.py").write_text(f"DEFAULT_TASK_TEMPLATE = {template!r}\n")
    (vendor / "evaluation/async_evaluate.py").write_text(OFFICIAL_SCORER)
    config = vendor / "evaluation/run_config/multi_test_config_book_4K.yaml"
    config.write_text(
        "context_length: 4000\nmetric: contains\nuse_default_system_prompt: true\n"
        "needle_set_path: ../data/needlesets/needle_set.json\n"
        "haystack_dir: ../data/haystack/rand_shuffle\n"
    )
    raw = tmp_path / "raw_fixture"
    (raw / "needlesets").mkdir(parents=True)
    (raw / "haystack/rand_shuffle").mkdir(parents=True)
    (raw / "haystack/LICENSES.md").write_text("Fixture license provenance.")
    experiments = [{
        "id": "0401", "reasoning_type": "world_knowledge", "system_prompt": "",
        "needle": "Actually, {CHAR} lives next to {1}.",
        "character_set": ["Alice", "Bob"],
        "questions": {"onehop": "Which character has been to {2}?"},
        "tests": {
            "T1": {"input_args": ["the Kiasma museum", "Helsinki"]},
            "T2": {"input_args": ["the European Central Bank", "Frankfurt"]},
        },
    }]
    (raw / "needlesets/needle_set.json").write_text(json.dumps(experiments))
    for index in range(1, 6):
        (raw / f"haystack/rand_shuffle/rand_book_{index}.txt").write_text(
            "A line about ships sailing.\nA second line about a forest.\n" * 200
        )
    import huggingface_hub
    import transformers

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(raw))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: CharacterTokenizer()
    )
    return vendor, raw, template, config


def test_prepare_uses_official_placement_and_keeps_question_out_of_context(
    tmp_path, official_data_fixture
):
    from still.benchmarks.nolima import DATA_REVISION, prepare_nolima

    vendor, raw, template, _ = official_data_fixture
    prepared = prepare_nolima(
        tmp_path / "prepared", vendor_dir=vendor, tokenizer_path="fixture",
        lengths=[80, 40], depths=[0.25, 0.75], samples=2,
    )
    documents = prepared["documents"]
    assert len(documents) == 8
    assert len({row["document_id"] for row in documents}) == 8
    assert prepared["dataset_revision"] == DATA_REVISION
    assert prepared["metric"] == "contains"
    assert prepared["length_unit"] == "tokenizer haystack tokens before needle and prompt"
    spec = importlib.util.spec_from_file_location(
        "fixture_placement", vendor / "data/book_haystack.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = CharacterTokenizer()
    for row in documents:
        qa = row["questions"][0]
        meta = qa["metadata"]
        assert row["benchmark"] == "nolima" and row["prompt_style"] == "context"
        assert qa["metric"] == "nolima_contains"
        assert qa["answers"] == [meta["selected_character"]]
        assert "Which character has been to" not in row["document"]
        assert "Question:" not in row["document"]
        assert qa["question"].startswith("Question: Which character")
        assert "Return only the final answer." in qa["question"]
        book = module.BookHaystack(str(raw / f"haystack/rand_shuffle/{meta['haystack_id']}"))
        placement = book.generate_w_needle_placement(
            meta["needle"], len, tokenizer.encode, tokenizer.decode,
            context_length=meta["length"], depth=meta["depth"],
        )
        context = template.partition("\n\nQuestion: ")[0].format(haystack=placement["text"])
        assert row["document"] == "You are a helpful assistant\n\n" + context
        assert meta["placement"]["token_depth"] == round(meta["length"] * meta["depth"])
        assert meta["prompt_tokens"] > meta["length"]
    repeated = prepare_nolima(
        tmp_path / "repeat", vendor_dir=vendor, tokenizer_path="fixture",
        lengths=[80, 40], depths=[0.25, 0.75], samples=2,
    )
    assert repeated["documents"] == documents


def test_prepare_preserves_configured_scoring(tmp_path, official_data_fixture):
    from still.benchmarks.nolima import prepare_nolima

    vendor, _, _, config = official_data_fixture
    config.write_text(config.read_text().replace("metric: contains", "metric: lastline_EM"))
    prepared = prepare_nolima(
        tmp_path / "prepared", vendor_dir=vendor, tokenizer_path="fixture",
        lengths=[80], depths=[0.5], samples=1, config_path=config,
    )
    assert prepared["metric"] == "lastline_EM"
    assert prepared["documents"][0]["questions"][0]["metric"] == "nolima_lastline_EM"


def test_pinned_tokenizer_identity_and_files_are_in_provenance(tmp_path, official_data_fixture):
    from still.benchmarks.nolima import prepare_nolima

    vendor, _, _, _ = official_data_fixture
    revision = "a" * 40
    snapshot = tmp_path / "models--Qwen--Qwen3-4B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text('{"fixture": true}')
    prepared = prepare_nolima(
        tmp_path / "prepared", vendor_dir=vendor, tokenizer_path=str(snapshot),
        lengths=[80], depths=[0.5], samples=1,
    )
    assert prepared["tokenizer_model_id"] == "Qwen/Qwen3-4B"
    assert prepared["tokenizer_revision"] == revision
    assert prepared["tokenizer_files_sha256"]["tokenizer.json"] == sha256(
        (snapshot / "tokenizer.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("depths", [[-0.1], [1.1], []])
def test_invalid_placement_depths_are_rejected(tmp_path, depths):
    from still.benchmarks.nolima import prepare_nolima

    with pytest.raises(ValueError, match="depth"):
        prepare_nolima(
            tmp_path, vendor_dir=tmp_path, tokenizer_path="fixture",
            lengths=[80], depths=depths, samples=1,
        )


def test_summary_groups_accuracy_by_length_depth_and_needle():
    from still.benchmarks.nolima import summarize_nolima

    rows = [
        {"score": 1.0, "metadata": {"length": 4096, "depth": 0.25, "needle_id": "A"}},
        {"score": 0.0, "metadata": {"length": 4096, "depth": 0.75, "needle_id": "B"}},
        {"score": 1.0, "metadata": {"length": 1024, "depth": 0.25, "needle_id": "A"}},
    ]
    result = summarize_nolima(rows)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["questions"] == 3
    assert result["by_length"]["4096"] == {"accuracy": 0.5, "questions": 2}
    assert result["by_depth"]["0.25"] == {"accuracy": 1.0, "questions": 2}
    assert result["by_needle_id"]["B"] == {"accuracy": 0.0, "questions": 1}
    assert summarize_nolima([])["questions"] == 0
