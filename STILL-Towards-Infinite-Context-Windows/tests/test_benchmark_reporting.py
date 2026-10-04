import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from still.eval.common import EvalRecord


def _load_run_benchmark_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/run_benchmark.py for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(*, prompt_id: str, total_latency_ms: float) -> EvalRecord:
    return EvalRecord(
        prompt_id=prompt_id,
        method="demo",
        prediction="A",
        gold=["A"],
        exact_match=True,
        canonical_kv_bytes=1024,
        compression_ratio=1.0,
        prefill_ms=10.0,
        decode_tokens_per_second=50.0,
        total_latency_ms=total_latency_ms,
        prompt_tokens=64,
        completion_tokens=2,
        metadata={
            "raw_prediction": "A<|im_end|>",
            "generated_token_ids": [32, 151645],
            "normalized_prediction": "A",
            "finish_reason": "eos_token",
        },
    )


def test_summarize_method_amortizes_target_preparation_into_query_total() -> None:
    module = _load_run_benchmark_module()
    records = [
        _record(prompt_id="sample-1::row-1", total_latency_ms=50.0),
        _record(prompt_id="sample-1::row-2", total_latency_ms=70.0),
        _record(prompt_id="sample-2::row-3", total_latency_ms=50.0),
        _record(prompt_id="sample-2::row-4", total_latency_ms=70.0),
    ]
    summary = module._summarize_method(
        method_name="still_1024_ce_only",
        records=records,
        baseline_mean_bytes=2048.0,
        mean_target_preparation_seconds=0.2,
        target_preparation_kind="still_cache_build",
        mean_questions_per_target=2.0,
        one_time_reusable_training_seconds=10.0,
    )

    assert summary["mean_query_continuation_latency_ms"] == 60.0
    assert summary["mean_target_total_seconds"] == 0.32
    assert summary["mean_target_eval_seconds_excluding_reusable_training"] == 0.32
    assert summary["mean_query_total_latency_ms"] == 160.0


def test_normalize_cartridge_record_adds_decode_debug_fields() -> None:
    module = _load_run_benchmark_module()
    raw_record = EvalRecord(
        prompt_id="sample-1::row-1",
        method="cartridge_hf_matched",
        prediction="B",
        gold=["B"],
        exact_match=True,
        canonical_kv_bytes=1024,
        compression_ratio=1.0,
        prefill_ms=11.0,
        decode_tokens_per_second=40.0,
        total_latency_ms=55.0,
        prompt_tokens=64,
        completion_tokens=3,
        metadata={"sample_id": "sample-1"},
    )
    normalized = module._normalize_cartridge_record(
        raw_record=raw_record,
        row={"answers": ["B"], "prediction_mode": "mcq_letter"},
    )

    assert normalized.metadata["raw_prediction"] == "B"
    assert normalized.metadata["generated_token_ids"] is None
    assert normalized.metadata["normalized_prediction"] == "B"
    assert normalized.metadata["finish_reason"] is None


def test_run_still_benchmark_skips_training_when_checkpoint_is_provided(tmp_path, monkeypatch) -> None:
    module = _load_run_benchmark_module()
    checkpoint_path = tmp_path / "still_compactor.pt"
    module.torch.save({"state_dict": {}, "metadata": {"num_latents": 1024}}, checkpoint_path)
    summary_path = checkpoint_path.parent / "still_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "compactor_path": str(checkpoint_path),
                "num_latents": 1024,
                "steps": 920,
                "best_loss": 0.5,
                "train_seconds": 123.0,
            }
        ),
        encoding="utf-8",
    )

    def _fail_train(**kwargs):
        raise AssertionError("train_still should not be called when a checkpoint path is provided.")

    class DummyModel:
        config = SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=1,
            num_attention_heads=1,
            hidden_size=8,
        )

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(module, "train_still", _fail_train)
    monkeypatch.setattr(
        module,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyModel()),
    )
    monkeypatch.setattr(
        module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        module,
        "load_eval_rows",
        lambda path: [{"sample_id": "sample-1", "context": "ctx"}],
    )
    monkeypatch.setattr(
        module,
        "build_still_cache",
        lambda **kwargs: {
            "cache_path": str((tmp_path / "cache.pt").resolve()),
            "build_seconds": 0.1,
            "source_prompt_tokens": 10,
            "compact_tokens": 1024,
            "canonical_kv_bytes": 1024,
        },
    )

    def _fake_run_still_eval(**kwargs):
        record = _record(prompt_id="sample-1::row-1", total_latency_ms=50.0)
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text(record.model_dump_json() + "\n", encoding="utf-8")
        return [record]

    monkeypatch.setattr(module, "run_still_eval", _fake_run_still_eval)

    records, train_summary, cache_manifest = module._run_still_benchmark(
        train_dataset_path=tmp_path / "train.jsonl",
        eval_rows_path=tmp_path / "eval.jsonl",
        output_dir=tmp_path / "out",
        device="cpu",
        num_latents=1024,
        steps=920,
        learning_rate=2e-5,
        validation_examples=32,
        validation_interval=100,
        max_completion_tokens=32,
        seed=0,
        kl_weight=0.0,
        exact_token_ce_weight=1.0,
        compactor_path=checkpoint_path,
        skip_train=True,
    )

    assert len(records) == 1
    assert train_summary["compactor_path"] == str(checkpoint_path.resolve())
    assert train_summary["train_seconds"] == 123.0
    assert cache_manifest["sample-1"]["build_seconds"] == 0.1
