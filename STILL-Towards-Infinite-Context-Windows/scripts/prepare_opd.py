"""Download the exact backbone and original QASPER, preserving official splits."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from still.chat import encode_system_prefix
from still.data.qasper import prepare_paper
from still.eval.common import SYSTEM_PROMPT

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ARCHIVES = [
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/opd/data"))
    parser.add_argument("--min-source-tokens", type=int, default=4096)
    parser.add_argument("--max-source-tokens", type=int, default=8192)
    parser.add_argument("--min-questions", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.min_source_tokens <= args.max_source_tokens or args.min_questions < 1:
        parser.error("invalid length/question limits")
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "*.jinja"],
        max_workers=4,
    )
    print(f"Model ready: {snapshot}", flush=True)
    archive_hashes = {}
    for url in ARCHIVES:
        archive = raw_dir / url.rsplit("/", 1)[-1]
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(archive)
        archive_hashes[url] = hashlib.sha256(archive.read_bytes()).hexdigest()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if member.isfile() and name in {
                    "qasper-train-v0.3.json",
                    "qasper-dev-v0.3.json",
                    "qasper-test-v0.3.json",
                    "qasper_evaluator.py",
                }:
                    with tar.extractfile(member) as source, (raw_dir / name).open("wb") as target:
                        shutil.copyfileobj(source, target)
        print(f"Data archive ready: {archive}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": snapshot,
        "source_archives_sha256": archive_hashes,
        "preparation": vars(args).copy(),
        "splits": {},
        "evidence_validation": "complete annotation, text source matching only",
    }
    report["preparation"]["output_dir"] = str(args.output_dir)
    document_ids = set()
    for split in ["train", "dev", "test"]:
        papers = json.loads((raw_dir / f"qasper-{split}-v0.3.json").read_text())
        kept, filtered_questions, filtered_length = [], 0, 0
        for doc_id, paper in papers.items():
            if doc_id in document_ids:
                raise ValueError(f"Official splits overlap at document {doc_id}")
            document_ids.add(doc_id)
            row = prepare_paper(doc_id, paper, split=split, min_questions=args.min_questions)
            if row is None:
                filtered_questions += 1
                continue
            length = encode_system_prefix(
                tokenizer, SYSTEM_PROMPT.format(context=row["document"])
            ).shape[-1]
            if not args.min_source_tokens <= length <= args.max_source_tokens:
                filtered_length += 1
                continue
            row["source_tokens"] = length
            kept.append(row)
        output = args.output_dir / f"{split}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
        stats = {
            "input_documents": len(papers),
            "kept_documents": len(kept),
            "kept_questions": sum(len(r["questions"]) for r in kept),
            "filtered_evidence_or_question_count": filtered_questions,
            "filtered_length": filtered_length,
            "min_tokens": min((r["source_tokens"] for r in kept), default=None),
            "max_tokens": max((r["source_tokens"] for r in kept), default=None),
        }
        report["splits"][split] = stats
        print(split, stats, flush=True)
        if not kept:
            raise ValueError(f"No usable documents in {split}")
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
