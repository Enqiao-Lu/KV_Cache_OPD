#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from still.data.wikipedia_snapshot import SnapshotConfig, build_wikipedia_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a reproducible local snapshot from wikimedia/wikipedia 20231101.en."
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "wikipedia_snapshot_20231101_en"),
        help="Directory where the snapshot split and manifest are written.",
    )
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
        help="Data root used to resolve local held-out experiments (e.g., wikipedia_india).",
    )
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--heldout-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scan", type=int, default=100_000)
    parser.add_argument("--min-chars", type=int, default=4_000)
    parser.add_argument("--max-chars", type=int, default=40_000)
    parser.add_argument("--near-dup-jaccard-threshold", type=float, default=0.9)
    parser.add_argument(
        "--include-heldout-experiments",
        default="wikipedia_india",
        help="Comma-separated local experiments to pin into held-out split.",
    )
    args = parser.parse_args()

    include_heldout_experiments = [
        item.strip() for item in args.include_heldout_experiments.split(",") if item.strip()
    ]
    config = SnapshotConfig(
        train_size=args.train_size,
        heldout_size=args.heldout_size,
        seed=args.seed,
        max_scan=args.max_scan,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        near_dup_jaccard_threshold=args.near_dup_jaccard_threshold,
    )
    manifest = build_wikipedia_snapshot(
        output_dir=args.output_dir,
        data_root=args.data_root,
        include_heldout_experiments=include_heldout_experiments,
        config=config,
    )
    print(json.dumps({"output_dir": args.output_dir, "manifest_hash": manifest["manifest_hash"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
