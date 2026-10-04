"""Document-grouped benchmark inference using the existing STILL KV interface."""

import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from still.attention_bias import enable_still_attention_bias
from still.chat import encode_system_prefix, encode_user_continuation
from still.core import StillCompactor
from still.data.qasper import answer_f1
from still.train.opd import OPDConfig, prefill, rollout, system_prompt

METRICS = {
    "qasper": {"qasper_f1"},
    "longbench_v2": {"longbench_accuracy"},
    "ruler": {"ruler_all", "ruler_part"},
    "nolima": {"nolima_contains", "nolima_EM", "nolima_lastline_EM", "nolima_lastline_contains"},
}


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_documents(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty benchmark dataset: {path}")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("benchmark document must be an object")
        benchmark = row.get("benchmark")
        if benchmark not in METRICS or row.get("prompt_style") not in {"qasper", "context"}:
            raise ValueError("unsupported benchmark or prompt_style")
        expected_style = "qasper" if benchmark == "qasper" else "context"
        if row["prompt_style"] != expected_style:
            raise ValueError(
                f"incompatible prompt_style for {benchmark}: expected {expected_style}"
            )
        doc_id = row.get("document_id")
        if not isinstance(doc_id, str) or not doc_id or (benchmark, doc_id) in seen:
            raise ValueError("missing or duplicate document_id")
        seen.add((benchmark, doc_id))
        if not isinstance(row.get("document"), str) or not row["document"].strip():
            raise ValueError(f"{doc_id}: empty document")
        questions = row.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"{doc_id}: no questions")
        question_ids = set()
        for qa in questions:
            if not isinstance(qa, dict):
                raise ValueError(f"{doc_id}: question must be an object")
            qid = qa.get("question_id")
            if not isinstance(qid, str) or not qid or qid in question_ids:
                raise ValueError(f"{doc_id}: missing or duplicate question_id")
            question_ids.add(qid)
            if not isinstance(qa.get("question"), str) or not qa["question"].strip():
                raise ValueError(f"{qid}: empty question")
            answers = qa.get("answers")
            if (
                not isinstance(answers, list)
                or not answers
                or not all(isinstance(answer, str) and answer for answer in answers)
            ):
                raise ValueError(f"{qid}: answers must be nonempty strings")
            if qa.get("metric") not in METRICS[benchmark]:
                raise ValueError(f"{qid}: incompatible metric for {benchmark}")
            if not isinstance(qa.get("metadata", {}), dict):
                raise ValueError(f"{qid}: metadata must be an object")
            if not isinstance(qa.get("answer_prefix", ""), str):
                raise ValueError(f"{qid}: answer_prefix must be a string")
    return rows


def prepare_qasper(path: str | Path) -> dict:
    from still.data.qasper import load_documents as load_qasper

    documents = load_qasper(path)
    for row in documents:
        row.update(benchmark="qasper", prompt_style="qasper")
        for qa in row["questions"]:
            qa.update(metric="qasper_f1", metadata={})
    return {
        "documents": documents,
        "provenance": {"source": str(Path(path).resolve()), "source_sha256": file_digest(path)},
    }


def _system(document: dict) -> str:
    if document["prompt_style"] == "qasper":
        return system_prompt(document["document"])
    return document["document"]


def _question_ids(tokenizer, document: dict, qa: dict):
    question = qa["question"]
    if document["prompt_style"] == "qasper":
        question += "\n\nAnswer concisely using only the provided context."
    return encode_user_continuation(
        tokenizer,
        system_prompt=_system(document),
        user_message=question,
        assistant_prefix=qa.get("answer_prefix", ""),
    )


def generation_limit(qa: dict, override: int | None) -> int:
    default = 16 if qa["metric"] == "longbench_accuracy" else 128
    value = (
        override if override is not None else qa.get("metadata", {}).get("max_new_tokens", default)
    )
    if not isinstance(value, int) or value < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    return value


def prompt_tokens(tokenizer, document: dict, qa: dict) -> int:
    return int(encode_system_prefix(tokenizer, _system(document)).shape[-1]) + int(
        _question_ids(tokenizer, document, qa).shape[-1]
    )


def filter_documents(
    tokenizer,
    documents: list[dict],
    *,
    max_context_tokens: int,
    max_new_tokens: int | None,
    limit: int = 0,
    question_limit: int = 0,
) -> tuple[list[dict], dict]:
    """Select complete documents; stop at limit and disclose unexamined coverage."""
    if max_context_tokens < 1 or min(limit, question_limit) < 0:
        raise ValueError("invalid context or sample limits")
    # Over-window examples are measured for exclusion, never sent through the model.
    tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
    selected, excluded, scanned = [], [], 0
    for original in documents:
        if limit and len(selected) >= limit:
            break
        row = copy.deepcopy(original)
        if question_limit:
            row["questions"] = row["questions"][:question_limit]
        scanned += 1
        required = max(
            prompt_tokens(tokenizer, row, qa) + generation_limit(qa, max_new_tokens)
            for qa in row["questions"]
        )
        if required > max_context_tokens:
            excluded.append({"document_id": row["document_id"], "required_tokens": required})
        else:
            selected.append(row)
    return selected, {
        "input_documents": len(documents),
        "scanned_documents": scanned,
        "unexamined_documents": len(documents) - scanned,
        "selected_documents": len(selected),
        "selected_questions": sum(len(row["questions"]) for row in selected),
        "excluded_context_documents": len(excluded),
        "excluded": excluded,
        "max_context_tokens": max_context_tokens,
        "max_new_tokens_override": max_new_tokens,
        "truncated": False,
    }


def score_question(prediction: str, qa: dict) -> float:
    metric, answers = qa["metric"], qa["answers"]
    if metric == "qasper_f1":
        return answer_f1(prediction, answers)
    if metric == "longbench_accuracy":
        from still.benchmarks.longbench_v2 import score_longbench

        return score_longbench(prediction, answers)
    if metric.startswith("ruler_"):
        from still.benchmarks.ruler import score_ruler

        return score_ruler(prediction, answers, partial=metric == "ruler_part")
    if metric.startswith("nolima_"):
        from still.benchmarks.nolima import score_nolima

        return score_nolima(prediction, answers, mode=metric.removeprefix("nolima_"))
    raise ValueError(f"unsupported metric: {metric}")


def summarize_predictions(benchmark: str, predictions: list[dict]) -> dict:
    if benchmark == "qasper":
        scores = {}
        for row in predictions:
            scores.setdefault(row["document_id"], []).append(row["score"])
        return {
            "question_f1": sum(row["score"] for row in predictions) / len(predictions),
            "document_f1": sum(sum(values) / len(values) for values in scores.values())
            / len(scores),
        }
    if benchmark == "longbench_v2":
        from still.benchmarks.longbench_v2 import summarize_longbench

        return summarize_longbench(predictions)
    if benchmark == "ruler":
        from still.benchmarks.ruler import summarize_ruler

        return summarize_ruler(predictions)
    from still.benchmarks.nolima import summarize_nolima

    return summarize_nolima(predictions)


@torch.no_grad()
def evaluate_documents(
    model,
    tokenizer,
    compactor,
    documents: list[dict],
    *,
    max_new_tokens: int | None = None,
    max_context_tokens: int = 32768,
) -> dict:
    if not documents or len({row["benchmark"] for row in documents}) != 1:
        raise ValueError("evaluate one nonempty benchmark at a time")
    model.eval().requires_grad_(False)
    enable_still_attention_bias(model)
    if compactor is not None:
        compactor.eval().requires_grad_(False)
    predictions, started = [], time.perf_counter()
    for row in documents:
        source_ids = encode_system_prefix(tokenizer, _system(row)).to(model.device)
        source_tokens = int(source_ids.shape[-1])
        prompts = [_question_ids(tokenizer, row, qa).to(model.device) for qa in row["questions"]]
        for qa, prompt in zip(row["questions"], prompts, strict=True):
            if (
                source_tokens + prompt.shape[-1] + generation_limit(qa, max_new_tokens)
                > max_context_tokens
            ):
                raise ValueError(f"{row['document_id']}: complete prompt exceeds context limit")
        full = prefill(model, source_ids)
        cache = full if compactor is None else compactor(full.as_cache(model.config))
        if compactor is not None:
            del full
        for qa, prompt in zip(row["questions"], prompts, strict=True):
            limit = generation_limit(qa, max_new_tokens)
            ids = rollout(
                model,
                cache,
                prompt,
                position_start=source_tokens,
                config=OPDConfig(max_source_tokens=max_context_tokens, max_new_tokens=limit),
                eos_token_id=tokenizer.eos_token_id,
                greedy=True,
            )
            prediction = qa.get("answer_prefix", "") + tokenizer.decode(
                ids, skip_special_tokens=True
            )
            predictions.append(
                {
                    "benchmark": row["benchmark"],
                    "document_id": row["document_id"],
                    "question_id": qa["question_id"],
                    "prediction": prediction,
                    "answers": qa["answers"],
                    "metric": qa["metric"],
                    "score": score_question(prediction, qa),
                    "metadata": qa.get("metadata", {}),
                    "source_tokens": source_tokens,
                    "prompt_tokens": source_tokens + prompt.shape[-1],
                    "cache_tokens": cache.num_tokens,
                    "generated_tokens": len(ids),
                    "generated_token_ids": ids,
                    "max_new_tokens": limit,
                }
            )
        del cache
    benchmark = documents[0]["benchmark"]
    metrics = summarize_predictions(benchmark, predictions)
    mean = sum(row["score"] for row in predictions) / len(predictions)
    return {
        "benchmark": benchmark,
        "documents": len(documents),
        "questions": len(predictions),
        "score": metrics.get("score", metrics.get("average", mean)),
        "metrics": metrics,
        "seconds": time.perf_counter() - started,
        "predictions": predictions,
    }


def load_compactor_checkpoint(
    model,
    path: str | Path,
    *,
    method: str,
    model_id: str,
    model_revision: str,
    tiny: bool = False,
    num_latents: int | None = None,
    backbone_sha256: str | None = None,
) -> tuple[StillCompactor, dict]:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    saved_hash = metadata.get("backbone_sha256")
    if (
        not isinstance(saved_hash, str)
        or len(saved_hash) != 64
        or any(char not in "0123456789abcdef" for char in saved_hash)
    ):
        raise ValueError("checkpoint backbone_sha256 is missing or malformed")
    if backbone_sha256 is not None and saved_hash != backbone_sha256:
        raise ValueError("checkpoint backbone_sha256 differs from the loaded model")
    checks = {"model_id": model_id, "model_revision": model_revision, "tiny": tiny}
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint {key} mismatch: expected {expected!r}")
    saved_method = metadata.get("method", metadata.get("teacher_context"))
    if method not in {"still", "full", "evidence"} or saved_method != method:
        raise ValueError(f"checkpoint method mismatch: expected {method!r}")
    slots = metadata.get("num_latents")
    if (
        not isinstance(slots, int)
        or slots < 1
        or (num_latents is not None and slots != num_latents)
    ):
        raise ValueError("checkpoint latent budget mismatch")
    compactor = StillCompactor.from_model_config(model.config, num_latents=slots)
    try:
        compactor.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"checkpoint compactor architecture mismatch: {error}") from error
    compactor.to(model.device).eval().requires_grad_(False)
    return compactor, metadata | {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": file_digest(path),
    }
