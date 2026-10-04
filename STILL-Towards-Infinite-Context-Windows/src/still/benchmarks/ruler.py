"""Local NVIDIA RULER generation, context/query conversion, and official scoring.

Templates and substring-recall metrics follow NVIDIA/RULER (Apache-2.0):
https://github.com/NVIDIA/RULER/tree/main/scripts
Generation uses the caller's tokenizer, the official task parameters, and base
templates. The requested length includes the official generation reserve; it is
not a promise that every generated input has exactly that many tokens.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean

from still.data.common import file_sha256, write_json, write_jsonl

RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
_MAX_NEW_TOKENS = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
_QA_INSTRUCTION = (
    "Answer the question based on the given documents. Only give me the answer "
    "and do not output any other words."
)
# Anchored complete suffixes, copied from official data/synthetic/constants.py.
# Only the final query is split: CWE and VT may contain completed demonstrations.
_TAILS = {
    "niah": re.compile(
        r"(?P<question>What (?:is|are all) the special magic "
        r"(?P<kind>numbers?|uuids?|words?) for (?P<query>[^\n]+?) "
        r"mentioned in the provided text\?)"
        r"(?P<prefix> The special magic (?P=kind) for (?P=query) "
        r"mentioned in the provided text (?:is|are))?\Z"
    ),
    "vt": re.compile(
        r"(?P<question>Question: Find all variables that are assigned the value "
        r"(?P<query>\d+) in the text above\.)"
        r"(?P<prefix> Answer: According to the chain\(s\) of variable assignment "
        r"in the text above, \d+ variables are assigned the value (?P=query), they are: )?\Z"
    ),
    "cwe": re.compile(
        r"(?P<question>Question: What are the 10 most common words in the above list\?)"
        r"(?P<prefix> Answer: The top 10 words that appear most often in the list are:)?\Z"
    ),
    "fwe": re.compile(
        r"(?P<question>Question: Do not provide any explanation\. Please ignore the dots "
        r"'\.\.\.\.'\. What are the three most frequently appeared words in the above coded text\?)"
        r"(?P<prefix> Answer: According to the coded text above, the three most frequently "
        r"appeared words are:)?\Z"
    ),
    "qa": re.compile(
        r"(?P<question>" + re.escape(_QA_INSTRUCTION) + r"\n\nQuestion: [\s\S]+?)"
        r"(?P<prefix> Answer:)?\Z"
    ),
}
# Lookahead finds overlapping candidates when source text quotes a task query.
_TAIL_STARTS = {
    category: re.compile(f"(?={pattern.pattern})") for category, pattern in _TAILS.items()
}


def _category(task: str) -> str:
    if task not in RULER_TASKS:
        raise ValueError(f"Unknown RULER task: {task}")
    return task.split("_")[0]


def convert_ruler(rows: Iterable[dict], *, task: str) -> list[dict]:
    """Split official base-template rows into a cache prefix and final query.

    Supports both current separate answer_prefix rows and older inline prefixes.
    Source instructions and completed few-shot examples stay in the prefix. The
    target query and its answer prefix never enter the document cache. Unknown or
    modified prompt templates fail closed instead of caching an unsplit query.
    """
    category = _category(task)
    converted = []
    for ordinal, row in enumerate(rows):
        text = row.get("input")
        answers = row.get("outputs")
        if not isinstance(text, str) or not isinstance(answers, list) or not answers:
            raise ValueError(f"RULER {task} row {ordinal} needs input and nonempty outputs")
        if not all(isinstance(answer, str) and answer for answer in answers):
            raise ValueError(f"RULER {task} row {ordinal} has invalid answers")
        matches = list(_TAIL_STARTS[category].finditer(text))
        if not matches:
            raise ValueError(
                f"RULER {task} row {ordinal} does not match the official base template"
            )
        match = matches[-1]
        document = text[: match.start()]
        question = match["question"]
        inline_prefix = match["prefix"] or ""
        prefix = row.get("answer_prefix", inline_prefix)
        if not document.strip() or not isinstance(prefix, str):
            raise ValueError(f"RULER {task} row {ordinal} has an invalid context or answer prefix")
        if inline_prefix and prefix != inline_prefix:
            raise ValueError(f"RULER {task} row {ordinal} has conflicting answer prefixes")
        if _TAILS[category].fullmatch(question + prefix) is None:
            raise ValueError(f"RULER {task} row {ordinal} has an unofficial answer prefix")
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"input", "outputs", "answer_prefix"}
        }
        metadata.update(task=task, max_new_tokens=_MAX_NEW_TOKENS[category])
        # Current NIAH's index is a character offset, not a unique row ordinal.
        identity = (
            f"ruler:{task}:{row.get('requested_length', row.get('length', 'unknown'))}:{ordinal}"
        )
        converted.append(
            {
                "benchmark": "ruler",
                "document_id": identity,
                "document": document,
                "prompt_style": "context",
                "questions": [
                    {
                        "question_id": identity + ":q0",
                        "question": question,
                        "answers": list(answers),
                        "metric": "ruler_part" if category == "qa" else "ruler_all",
                        "answer_prefix": prefix,
                        "metadata": metadata,
                    }
                ],
            }
        )
    return converted


def score_ruler(prediction: str, answers: list[str], *, partial: bool = False) -> float:
    """Official all-reference recall or QA best-reference match, scaled to 0..1.

    Equivalent to NVIDIA/RULER scripts/eval/synthetic/constants.py (Apache-2.0),
    with evaluate.py's printable-character postprocessing. Per-row results stay
    unrounded; the official percent rounding happens after task averaging.
    """
    if not answers or not all(isinstance(answer, str) and answer for answer in answers):
        raise ValueError("RULER answers must be nonempty strings")
    prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip().lower()
    matches = [float(answer.lower() in prediction) for answer in answers]
    return max(matches) if partial else sum(matches) / len(matches)


def summarize_ruler(predictions: list[dict]) -> dict:
    """Report official rounded task percentages and their macro mean (0..1)."""
    grouped = defaultdict(list)
    for row in predictions:
        task = row.get("task", row.get("metadata", {}).get("task"))
        category = _category(task)
        prediction = row.get("prediction", row.get("pred"))
        answers = row.get("answers", row.get("outputs"))
        if not isinstance(prediction, str):
            raise ValueError("RULER prediction must be a string")
        grouped[task].append(score_ruler(prediction, answers, partial=category == "qa"))
    per_task = {}
    for task, scores in grouped.items():
        official = round(mean(scores) * 100, 2)
        per_task[task] = {
            "count": len(scores),
            "score": official / 100,
            "official_score": official,
            "string_match": official,
        }
    return {
        "benchmark": "ruler",
        "count": len(predictions),
        "per_task": per_task,
        "score": mean([item["score"] for item in per_task.values()]) if per_task else None,
    }


def _download_json(path: Path, urls: list[str]) -> None:
    expected_hash = None
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing.startswith("version https://git-lfs.github.com/spec/v1\n"):
            expected_hash = re.search(r"oid sha256:([a-f0-9]{64})", existing).group(1)
        else:
            json.loads(existing)
            return
    errors = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            json.loads(content)
            if expected_hash and hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError("Downloaded RULER LFS asset has an unexpected hash")
            path.write_bytes(content)
            return
        except (OSError, ValueError) as error:
            errors.append(str(error))
    raise RuntimeError(f"Unable to download RULER asset {path.name}: {errors}")


def _prepare_assets(
    vendor_dir: Path, configurations: dict, tasks: list[str], revision: str
) -> list[Path]:
    asset_dir = vendor_dir / "scripts" / "data" / "synthetic" / "json"
    paths = []
    essay = any(configurations[task]["args"].get("type_haystack") == "essay" for task in tasks)
    if essay:
        import nltk

        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True, raise_on_error=True)
        path = asset_dir / "PaulGrahamEssays.json"
        if not path.is_file():
            subprocess.run(
                [sys.executable, str(asset_dir / "download_paulgraham_essay.py")],
                cwd=asset_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
        if not json.loads(path.read_text(encoding="utf-8")).get("text", "").strip():
            raise RuntimeError("Official RULER essay download produced an empty corpus")
        paths.append(path)
    if "qa_1" in tasks:
        path = asset_dir / "squad.json"
        _download_json(path, ["https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"])
        paths.append(path)
    if "qa_2" in tasks:
        path = asset_dir / "hotpotqa.json"
        _download_json(
            path,
            [
                "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
                "7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
                "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            ],
        )
        paths.append(path)
    if "cwe" in tasks:
        path = asset_dir / "english_words.json"
        _download_json(
            path,
            [
                f"https://media.githubusercontent.com/media/NVIDIA/RULER/{revision}/"
                "scripts/data/synthetic/json/english_words.json"
            ],
        )
        paths.append(path)
    return paths


def prepare_ruler(
    output_dir: Path,
    *,
    vendor_dir: Path,
    tokenizer_path: str,
    lengths: list[int],
    samples: int,
    tasks: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Run official CPU generators and write documents.jsonl plus provenance.json.

    vendor_dir must contain a local NVIDIA/RULER checkout. Missing official QA or
    essay assets are downloaded locally; nothing is uploaded. Generator failures
    propagate. Run each generator directly with this Python interpreter, avoiding
    upstream prepare.py's shell quoting and swallowed subprocess failures.
    """
    import yaml

    output_dir, vendor_dir = Path(output_dir).resolve(), Path(vendor_dir).resolve()
    tasks = list(RULER_TASKS) if tasks is None else list(tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("RULER tasks must be a nonempty unique list")
    for task in tasks:
        _category(task)
    if samples < 1 or not lengths or any(length < 1 for length in lengths):
        raise ValueError("RULER lengths and samples must be positive")
    if len(set(lengths)) != len(lengths):
        raise ValueError("RULER lengths must be unique")
    script_dir = vendor_dir / "scripts" / "data" / "synthetic"
    constants_path = script_dir / "constants.py"
    spec = importlib.util.spec_from_file_location("ruler_data_constants", constants_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    config_path = vendor_dir / "scripts" / "synthetic.yaml"
    configurations = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    revision = subprocess.run(
        ["git", "-C", str(vendor_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assets = _prepare_assets(vendor_dir, configurations, tasks, revision)
    documents, sources, commands = [], [], []
    for length in lengths:
        for task in tasks:
            config = configurations[task]
            base = constants.TASKS[config["task"]]
            script = script_dir / f"{config['task']}.py"
            save_dir = output_dir / "raw" / str(length)
            command = [
                sys.executable,
                str(script),
                "--save_dir",
                str(save_dir),
                "--save_name",
                task,
                "--subset",
                "validation",
                "--tokenizer_path",
                tokenizer_path,
                "--tokenizer_type",
                "hf",
                "--max_seq_length",
                str(length),
                "--tokens_to_generate",
                str(base["tokens_to_generate"]),
                "--num_samples",
                str(samples),
                "--random_seed",
                str(seed),
                "--template",
                base["template"] + base.get("answer_prefix", ""),
            ]
            for key, value in config["args"].items():
                command += [f"--{key}", str(value)]
            if config["task"] == "qa":
                command += ["--pre_samples", "0"]
            result = subprocess.run(
                command, cwd=vendor_dir, check=True, capture_output=True, text=True, timeout=900
            )
            log_path = output_dir / "logs" / str(length) / f"{task}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
            source_path = save_dir / task / "validation.jsonl"
            rows = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != samples:
                raise RuntimeError(f"RULER {task} generated {len(rows)} rows, expected {samples}")
            for row in rows:
                if row.get("length", length + 1) > length:
                    raise RuntimeError(f"RULER {task} exceeds requested length {length}")
                row["requested_length"] = length
            documents.extend(convert_ruler(rows, task=task))
            sources.append(
                {
                    "task": task,
                    "length": length,
                    "path": str(source_path),
                    "sha256": file_sha256(source_path),
                    "rows": len(rows),
                }
            )
            commands.append(command)
    packages = {}
    for package in (
        "transformers",
        "tokenizers",
        "numpy",
        "nltk",
        "wonderwords",
        "scipy",
        "tenacity",
        "html2text",
        "beautifulsoup4",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    generator_paths = [
        config_path,
        constants_path,
        *sorted({script_dir / (configurations[task]["task"] + ".py") for task in tasks}),
    ]
    generator_paths.extend(
        path
        for path in [
            script_dir.parent / "tokenizer.py",
            script_dir.parent / "manifest_utils.py",
            vendor_dir / "scripts" / "eval" / "synthetic" / "constants.py",
        ]
        if path.is_file()
    )
    tokenizer_dir = Path(tokenizer_path)
    tokenizer_files = [
        path
        for path in sorted(tokenizer_dir.glob("*"))
        if path.is_file()
        and path.name
        in {
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "config.json",
        }
    ]
    provenance = {
        "benchmark": "ruler",
        "source_url": "https://github.com/NVIDIA/RULER",
        "vendor_dir": str(vendor_dir),
        "vendor_revision": revision,
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": "hf",
        "model_template_type": "base",
        "tokenizer_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in tokenizer_files
        ],
        "lengths": lengths,
        "samples_per_task_per_length": samples,
        "tasks": tasks,
        "seed": seed,
        "python_version": sys.version,
        "packages": packages,
        "commands": commands,
        "sources": sources,
        "generator_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in generator_paths
        ],
        "assets": [{"path": str(path), "sha256": file_sha256(path)} for path in assets],
        "length_definition": "official input plus answer prefix plus task generation reserve",
        "scorer_source": "NVIDIA/RULER scripts/eval/synthetic/constants.py",
    }
    documents_path, provenance_path = output_dir / "documents.jsonl", output_dir / "provenance.json"
    write_jsonl(documents_path, documents)
    provenance["documents_sha256"] = file_sha256(documents_path)
    write_json(provenance_path, provenance)
    return {
        "documents": documents,
        "documents_path": str(documents_path),
        "provenance": provenance,
        "provenance_path": str(provenance_path),
    }
