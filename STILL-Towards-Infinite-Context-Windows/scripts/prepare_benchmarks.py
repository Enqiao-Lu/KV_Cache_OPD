"""Prepare local official benchmark samples for the shared KV evaluator."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download
from run_opd_comparison import MODEL_ID, MODEL_REVISION
from transformers import AutoTokenizer

from still.benchmarks.suite import METRICS, file_digest, filter_documents, prepare_qasper

VENDOR_REVISIONS = {
    "RULER": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
    "NoLiMa": "cb14780b249fecf2851127b2101a062c1b2c6430",
}


def vendor_checkout(base: Path, name: str, url: str) -> Path:
    path = base / name
    if not (path / ".git").is_dir():
        base.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "-c",
                "filter.lfs.required=false",
                "-c",
                "filter.lfs.process=",
                "-c",
                "filter.lfs.smudge=cat",
                "clone",
                "--depth",
                "1",
                url,
                str(path),
            ],
            check=True,
            env=os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"},
        )
        revision = VENDOR_REVISIONS[name]
        subprocess.run(["git", "fetch", "--depth", "1", "origin", revision], cwd=path, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "filter.lfs.required=false",
                "-c",
                "filter.lfs.process=",
                "-c",
                "filter.lfs.smudge=cat",
                "checkout",
                "--detach",
                revision,
            ],
            cwd=path,
            check=True,
            env=os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"},
        )
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True, capture_output=True
    ).stdout.strip()
    if actual != VENDOR_REVISIONS[name]:
        raise ValueError(f"{name}: cached vendor revision differs from {VENDOR_REVISIONS[name]}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=list(METRICS), default=list(METRICS))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks/data"))
    parser.add_argument("--vendor-dir", type=Path, default=Path("outputs/benchmarks/vendor"))
    parser.add_argument("--qasper-data", type=Path, default=Path("outputs/opd/data/dev.jsonl"))
    parser.add_argument("--longbench-source", type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096])
    parser.add_argument("--depths", nargs="+", type=float, default=[0.25, 0.75])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--ruler-tasks", nargs="+")
    parser.add_argument("--nolima-config", type=Path)
    parser.add_argument("--document-limit", type=int, default=0)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.samples, *args.lengths, args.max_context_tokens) < 1:
        parser.error("sample counts, lengths and context limits must be positive")
    if min(args.document_limit, args.question_limit) < 0:
        parser.error("limits must be nonnegative; zero means all")
    if not 1 <= args.max_context_tokens <= 32768:
        parser.error("this evaluator uses Qwen3-4B's native window, at most 32768 tokens")
    snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.model_max_length = 32768
    manifest = {}
    for benchmark in args.benchmarks:
        output = (args.output_dir / benchmark / "prepared").resolve()
        output.mkdir(parents=True, exist_ok=True)
        source_output = output / "source"
        source_output.mkdir(parents=True, exist_ok=True)
        print(f"Preparing {benchmark}", flush=True)
        if benchmark == "qasper":
            result = prepare_qasper(args.qasper_data)
        elif benchmark == "longbench_v2":
            from still.benchmarks.longbench_v2 import prepare_longbench

            result = prepare_longbench(source_output, source=args.longbench_source)
        elif benchmark == "ruler":
            from still.benchmarks.ruler import prepare_ruler

            vendor = vendor_checkout(args.vendor_dir, "RULER", "https://github.com/NVIDIA/RULER")
            result = prepare_ruler(
                source_output,
                vendor_dir=vendor.resolve(),
                tokenizer_path=snapshot,
                lengths=args.lengths,
                samples=args.samples,
                tasks=args.ruler_tasks,
                seed=args.seed,
            )
        else:
            from still.benchmarks.nolima import prepare_nolima

            vendor = vendor_checkout(
                args.vendor_dir, "NoLiMa", "https://github.com/adobe-research/NoLiMa"
            )
            result = prepare_nolima(
                source_output,
                vendor_dir=vendor.resolve(),
                tokenizer_path=snapshot,
                lengths=args.lengths,
                depths=args.depths,
                samples=args.samples,
                seed=args.seed,
                config_path=args.nolima_config,
            )
        source_path = source_output / "documents.jsonl"
        if benchmark == "qasper":
            source_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result["documents"])
            )
        provenance = result.get("provenance", {k: v for k, v in result.items() if k != "documents"})
        provenance |= {
            "documents_path": str(source_path),
            "documents_sha256": file_digest(source_path),
            "document_count": len(result["documents"]),
            "question_count": sum(len(row["questions"]) for row in result["documents"]),
        }
        (source_output / "provenance.json").write_text(json.dumps(provenance, indent=2))
        selected, coverage = filter_documents(
            tokenizer,
            result["documents"],
            max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,
            limit=args.document_limit,
            question_limit=args.question_limit,
        )
        if not selected:
            raise ValueError(f"{benchmark}: no complete examples fit the requested window")
        path = output / "documents.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        manifest[benchmark] = {
            "path": str(path),
            "sha256": file_digest(path),
            "coverage": coverage,
            "source_provenance": provenance,
            "documents_sha256": file_digest(path),
            "document_count": len(selected),
            "question_count": sum(len(row["questions"]) for row in selected),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "enable_thinking": False,
            "chat_adaptation": "cached system-context prefix and user question continuation",
        }
        (output / "selection.json").write_text(json.dumps(manifest[benchmark], indent=2))
        (output / "provenance.json").write_text(json.dumps(manifest[benchmark], indent=2))
        print(json.dumps({"benchmark": benchmark, "coverage": coverage}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text()) | manifest
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
