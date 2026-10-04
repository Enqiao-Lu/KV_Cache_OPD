"""Official NoLiMa needles, book placement and case-sensitive response scoring.

The context/question split adapts the released user prompt to a reusable Qwen
system cache. Requested lengths count haystack tokens before needle insertion;
they are not the final chat length or the released YAMLs' decimal K labels.
"""

import ast
import hashlib
import importlib.util
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

DATASET_ID = "amodaresi/NoLiMa"
DATA_REVISION = "378115b1f136b6ba78f90f78682bc55f70ec3ddd"
OFFICIAL_REPOSITORY = "https://github.com/adobe-research/NoLiMa"
SCORE_MODES = {"EM", "contains", "lastline_EM", "lastline_contains"}


def score_nolima(prediction: str, answers: list[str], *, mode: str = "contains") -> float:
    """Match NoLiMa_Tester._evaluate_response, including whitespace and case."""
    if mode == "EM":
        return float(prediction.strip() in answers)
    if mode == "contains":
        return float(any(answer in prediction for answer in answers))
    if mode == "lastline_EM":
        return float(prediction.strip().split("\n")[-1] in answers)
    if mode == "lastline_contains":
        return float(any(answer in prediction.strip().split("\n")[-1] for answer in answers))
    raise ValueError(f"Invalid metric: {mode}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_template(vendor_dir: Path) -> str:
    tree = ast.parse((vendor_dir / "evaluation/run_tests.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_TASK_TEMPLATE"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Official DEFAULT_TASK_TEMPLATE not found")


def _variants(experiments: list[dict], seed: int):
    for experiment in experiments:
        for question_type, question in experiment["questions"].items():
            for test_id, test in experiment["tests"].items():
                needle, retrieval = experiment["needle"], question
                for index, argument in enumerate(test["input_args"], 1):
                    needle = needle.replace("{" + str(index) + "}", argument)
                    retrieval = retrieval.replace("{" + str(index) + "}", argument)
                yield {
                    "experiment": experiment,
                    "needle_id": f"{experiment['id']}_{test_id}",
                    "question_type": question_type,
                    "needle": needle,
                    "retrieval_question": retrieval,
                    "answers": test.get("gold_answers", []),
                    "seed": seed + int(experiment["id"][:4]),
                }


def prepare_nolima(
    output_dir: Path,
    *,
    vendor_dir: Path,
    tokenizer_path: str,
    lengths: list[int],
    depths: list[float],
    samples: int,
    seed: int = 42,
    config_path: Path | None = None,
) -> dict:
    """Prepare ``samples`` official test variants per length/depth, in file order.

    Variants cycle through the five released shuffled books. Character RNG uses
    the official base_seed + experiment ID + book index and resets per variant
    and length; the requested depth order controls successive character draws.
    Only standard non-distractor needles are supported in this adapter.
    """
    if not depths or any(not math.isfinite(d) or not 0 <= d <= 1 for d in depths):
        raise ValueError("depths must be nonempty fractions in [0, 1]")
    if not lengths or any(not isinstance(length, int) or length < 1 for length in lengths):
        raise ValueError("lengths must contain positive token counts")
    if samples < 1 or len(set(lengths)) != len(lengths) or len(set(depths)) != len(depths):
        raise ValueError("samples must be positive and length/depth grids must be unique")
    vendor_dir, output_dir = Path(vendor_dir), Path(output_dir)
    config_path = Path(
        config_path or vendor_dir / "evaluation/run_config/multi_test_config_book_4K.yaml"
    )
    configuration = yaml.safe_load(config_path.read_text())
    metric = configuration.get("metric", "EM")
    if metric not in SCORE_MODES:
        raise ValueError(f"Invalid metric: {metric}")
    needle_file = "needlesets/" + Path(configuration["needle_set_path"]).name
    if Path(configuration["haystack_dir"]).name != "rand_shuffle":
        raise ValueError("Only the official rand_shuffle haystacks are supported")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    raw = Path(snapshot_download(
        repo_id=DATASET_ID, repo_type="dataset", revision=DATA_REVISION,
        allow_patterns=[needle_file, "haystack/rand_shuffle/*", "haystack/LICENSES.md"],
        local_dir=output_dir / "raw",
    ))
    experiments = json.loads((raw / needle_file).read_text())
    variants = list(_variants(experiments, seed))
    if samples > len(variants):
        raise ValueError(f"Requested {samples} variants but only {len(variants)} are available")
    selected = variants[:samples]
    if any("distractors" in variant["experiment"] for variant in selected):
        raise ValueError("Distractor needle sets require a separate declared protocol")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def count_tokens(text):
        return len(encode(text))

    placement_path = vendor_dir / "data/book_haystack.py"
    spec = importlib.util.spec_from_file_location("nolima_official_book", placement_path)
    placement_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(placement_module)
    books = sorted((raw / "haystack/rand_shuffle").glob("rand_book_*.txt"))
    if not books:
        raise ValueError("No official NoLiMa shuffled books found")
    haystacks = [placement_module.BookHaystack(str(book)) for book in books]
    default_template = _default_template(vendor_dir)
    documents = []
    for length in lengths:
        for index, variant in enumerate(selected):
            experiment = variant["experiment"]
            book_index = index % len(books)
            sample_seed = variant["seed"] + book_index
            rng = np.random.RandomState(sample_seed)
            template = experiment.get(
                "task_template", configuration.get("task_template") or default_template
            )
            before, separator, after = template.partition("{question}")
            if not separator or "{question}" in after or "{haystack}" not in before:
                raise ValueError("NoLiMa task template must place haystack before one question")
            split = before.rfind("\n\nQuestion:")
            context_template = before[:split] if split >= 0 else before
            question_prefix = before[split:].lstrip("\n") if split >= 0 else ""
            for depth in depths:
                needle, retrieval = variant["needle"], variant["retrieval_question"]
                character = None
                answers = variant["answers"]
                if "{CHAR}" in needle:
                    character = str(rng.choice(experiment["character_set"]))
                    needle = needle.replace("{CHAR}", character)
                    retrieval = retrieval.replace("{CHAR}", character)
                    answers = [character]
                if not isinstance(answers, list) or not answers:
                    raise ValueError(f"No answer targets for {variant['needle_id']}")
                placement = haystacks[book_index].generate_w_needle_placement(
                    needle=needle, token_count_func=count_tokens, encoding_func=encode,
                    decoding_func=tokenizer.decode, context_length=length, depth=depth,
                    shift=configuration.get("shift", 0),
                    static_depth=configuration.get("static_depth", -1),
                )
                system = (
                    "You are a helpful assistant"
                    if configuration.get("use_default_system_prompt", False)
                    else experiment["system_prompt"]
                )
                context = context_template.format(haystack=placement["text"])
                context = (system + "\n\n" if system else "") + context
                question = question_prefix + retrieval + after
                messages = [{"role": "system", "content": context}]
                source_tokens = len(tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False, enable_thinking=False,
                ))
                messages.append({"role": "user", "content": question})
                prompt_tokens = len(tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                ))
                document_id = (
                    f"nolima:{variant['needle_id']}:{variant['question_type']}:"
                    f"{books[book_index].stem}:{length}:{depth:g}:{sample_seed}"
                )
                documents.append({
                    "benchmark": "nolima", "document_id": document_id,
                    "document": context, "prompt_style": "context",
                    "questions": [{
                        "question_id": document_id + ":question", "question": question,
                        "answers": answers, "metric": "nolima_" + metric,
                        "metadata": {
                            "length": length, "depth": depth, "needle_id": variant["needle_id"],
                            "question_type": variant["question_type"],
                            "reasoning_type": experiment.get("reasoning_type"),
                            "haystack_id": books[book_index].name, "seed": sample_seed,
                            "needle": needle, "selected_character": character,
                            "placement": {k: v for k, v in placement.items() if k != "text"},
                            "source_tokens": source_tokens, "prompt_tokens": prompt_tokens,
                        },
                    }],
                })
    git = subprocess.run(
        ["git", "-C", str(vendor_dir), "rev-parse", "HEAD"], capture_output=True, text=True,
    )
    source_paths = [raw / needle_file, *books, raw / "haystack/LICENSES.md"]
    vendor_paths = [placement_path, vendor_dir / "evaluation/run_tests.py",
                    vendor_dir / "evaluation/async_evaluate.py", config_path]
    tokenizer_dir = Path(tokenizer_path)
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    tokenizer_model_id = tokenizer_path if not tokenizer_dir.is_dir() else None
    if tokenizer_dir.parent.name == "snapshots":
        tokenizer_revision = tokenizer_dir.name
        tokenizer_model_id = tokenizer_dir.parent.parent.name.removeprefix("models--").replace(
            "--", "/"
        )
    tokenizer_files = [
        tokenizer_dir / name
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if (tokenizer_dir / name).is_file()
    ]
    provenance = {
        "benchmark": "nolima", "dataset_id": DATASET_ID, "dataset_revision": DATA_REVISION,
        "official_repository": OFFICIAL_REPOSITORY,
        "vendor_revision": git.stdout.strip() if git.returncode == 0 else None,
        "source_files_sha256": {str(path.relative_to(raw)): _sha256(path) for path in source_paths},
        "vendor_files_sha256": {str(path): _sha256(path) for path in vendor_paths},
        "tokenizer_path": tokenizer_path, "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_model_id": tokenizer_model_id, "tokenizer_revision": tokenizer_revision,
        "tokenizer_files_sha256": {path.name: _sha256(path) for path in tokenizer_files},
        "metric": metric, "configuration": configuration, "seed": seed,
        "lengths": lengths, "depths": depths, "samples_per_length_depth": samples,
        "available_variants": len(variants),
        "selection": "first official test variants in file order",
        "length_unit": "tokenizer haystack tokens before needle and prompt",
        "chat_adaptation": "official default/needle system instruction and pre-question task "
                           "text moved into Qwen system cache; question and final-answer "
                           "instruction remain in user continuation; thinking disabled",
        "documents_count": len(documents),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents)
    )
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return {"documents": documents, **provenance}


def summarize_nolima(predictions: list[dict]) -> dict:
    """Accuracy plus the sample counts for length, depth and needle groups."""
    def summarize(rows):
        return {"accuracy": sum(row["score"] for row in rows) / len(rows) if rows else 0.0,
                "questions": len(rows)}

    result = summarize(predictions)
    for field in ("length", "depth", "needle_id", "haystack_id", "question_type"):
        groups = defaultdict(list)
        for row in predictions:
            value = row.get("metadata", {}).get(field)
            if value is not None:
                groups[str(value)].append(row)
        result["by_" + field] = {key: summarize(rows) for key, rows in groups.items()}
    return result
