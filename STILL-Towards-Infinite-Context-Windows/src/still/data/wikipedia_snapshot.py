import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from still.data.common import stable_hash, write_json


def _normalize_text(text: str) -> str:
    """Normalize article text for exact and near-duplicate checks."""
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9 ]+", "", lowered)
    return lowered.strip()


def _tokenize_for_similarity(text: str, *, max_tokens: int = 512) -> set[str]:
    """Tokenize normalized text into a bounded set for Jaccard overlap checks."""
    normalized = _normalize_text(text)
    tokens = normalized.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return set(tokens)


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def _slugify(value: str) -> str:
    """Convert a title into a filesystem-safe experiment suffix."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "untitled"


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration controlling Wikipedia snapshot selection and deduplication."""
    train_size: int = 500
    heldout_size: int = 20
    seed: int = 0
    max_scan: int = 100_000
    min_chars: int = 4_000
    max_chars: int = 40_000
    near_dup_jaccard_threshold: float = 0.9
    dataset_name: str = "wikimedia/wikipedia"
    dataset_subset: str = "20231101.en"


def _deterministic_score(*, seed: int, key: str) -> int:
    """Assign a reproducible random-looking score used to sample streaming rows."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def _trim_top_k_by_score(
    *,
    candidates: list[dict[str, Any]],
    candidate: dict[str, Any],
    k: int,
) -> None:
    """Maintain the best-scoring candidate pool while streaming the dataset."""
    if len(candidates) < k:
        candidates.append(candidate)
        return
    max_idx = max(range(len(candidates)), key=lambda idx: candidates[idx]["score"])
    if candidate["score"] < candidates[max_idx]["score"]:
        candidates[max_idx] = candidate


def _collect_stream_candidates(
    *,
    config: SnapshotConfig,
    needed_count: int,
) -> tuple[list[dict[str, Any]], int]:
    """Scan the streaming Wikipedia dataset and keep candidate articles."""
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_subset,
        split="train",
        streaming=True,
    )
    over_sample_target = max(needed_count * 5, needed_count + 200)
    selected: list[dict[str, Any]] = []
    scanned = 0
    for row in dataset:
        scanned += 1
        if scanned > config.max_scan:
            break
        text = (row.get("text") or "").strip()
        if len(text) < config.min_chars or len(text) > config.max_chars:
            continue
        page_id = str(row.get("id") or row.get("url") or row.get("title") or "")
        if not page_id:
            continue
        candidate = {
            "source": "hf",
            "dataset_name": config.dataset_name,
            "dataset_subset": config.dataset_subset,
            "article_id": page_id,
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "text": text,
            "score": _deterministic_score(seed=config.seed, key=page_id),
        }
        _trim_top_k_by_score(candidates=selected, candidate=candidate, k=over_sample_target)
    selected.sort(key=lambda item: item["score"])
    return selected, scanned


def _dedupe_candidates(
    *,
    candidates: Iterable[dict[str, Any]],
    near_dup_jaccard_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove exact and near-duplicate articles from a candidate pool."""
    accepted: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen_exact: dict[str, dict[str, Any]] = {}
    accepted_tokens: list[set[str]] = []

    for candidate in candidates:
        normalized = _normalize_text(candidate["text"])
        exact_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        candidate["exact_hash"] = exact_hash
        candidate_tokens = _tokenize_for_similarity(candidate["text"])

        exact_hit = seen_exact.get(exact_hash)
        if exact_hit is not None:
            duplicates.append(
                {
                    "type": "exact",
                    "first_article_id": exact_hit["article_id"],
                    "second_article_id": candidate["article_id"],
                }
            )
            continue

        near_hit_idx = None
        near_sim = 0.0
        for idx, token_set in enumerate(accepted_tokens):
            similarity = _jaccard_similarity(candidate_tokens, token_set)
            if similarity >= near_dup_jaccard_threshold:
                near_hit_idx = idx
                near_sim = similarity
                break
        if near_hit_idx is not None:
            duplicates.append(
                {
                    "type": "near",
                    "first_article_id": accepted[near_hit_idx]["article_id"],
                    "second_article_id": candidate["article_id"],
                    "similarity": round(near_sim, 4),
                }
            )
            continue

        seen_exact[exact_hash] = candidate
        accepted_tokens.append(candidate_tokens)
        accepted.append(candidate)
    return accepted, duplicates


def _load_existing_experiment(
    *,
    data_root: Path,
    experiment_name: str,
) -> dict[str, Any]:
    """Load a pinned local held-out corpus and treat it like a snapshot record."""
    experiment_dir = data_root / experiment_name
    data_path = experiment_dir / "data.txt"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing corpus for held-out experiment: {data_path}")
    text = data_path.read_text(encoding="utf-8")
    return {
        "source": "local",
        "dataset_name": "local",
        "dataset_subset": "local",
        "article_id": f"local::{experiment_name}",
        "title": experiment_name,
        "url": "",
        "text": text,
        "score": _deterministic_score(seed=0, key=f"local::{experiment_name}"),
    }


def _compute_cross_split_leakage(
    *,
    train_records: list[dict[str, Any]],
    heldout_records: list[dict[str, Any]],
    near_dup_jaccard_threshold: float,
) -> dict[str, Any]:
    """Check train and held-out splits for exact or near-duplicate leakage."""
    train_exact = {record["exact_hash"]: record for record in train_records}
    train_tokens = [
        (
            record["article_id"],
            _tokenize_for_similarity(record["text"]),
        )
        for record in train_records
    ]
    exact_pairs: list[dict[str, str]] = []
    near_pairs: list[dict[str, Any]] = []

    for heldout in heldout_records:
        exact_hit = train_exact.get(heldout["exact_hash"])
        if exact_hit is not None:
            exact_pairs.append(
                {
                    "train_article_id": exact_hit["article_id"],
                    "heldout_article_id": heldout["article_id"],
                }
            )
            continue
        heldout_tokens = _tokenize_for_similarity(heldout["text"])
        for train_article_id, train_token_set in train_tokens:
            similarity = _jaccard_similarity(heldout_tokens, train_token_set)
            if similarity >= near_dup_jaccard_threshold:
                near_pairs.append(
                    {
                        "train_article_id": train_article_id,
                        "heldout_article_id": heldout["article_id"],
                        "similarity": round(similarity, 4),
                    }
                )
                break

    return {
        "exact_overlap_count": len(exact_pairs),
        "near_overlap_count": len(near_pairs),
        "near_dup_jaccard_threshold": near_dup_jaccard_threshold,
        "exact_overlap_pairs": exact_pairs,
        "near_overlap_pairs": near_pairs,
        "passed": len(exact_pairs) == 0 and len(near_pairs) == 0,
    }


def _materialize_records(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    split: str,
) -> list[dict[str, Any]]:
    """Write selected snapshot records into the repo's train/heldout directory layout."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        if record["source"] == "local" and record["title"] == "wikipedia_india":
            experiment_name = "wikipedia_india"
        else:
            suffix = _slugify(record["title"] or record["article_id"])
            experiment_name = f"wiki_{idx:04d}_{suffix[:60]}".strip("_")
        experiment_dir = split_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        data_path = experiment_dir / "data.txt"
        metadata_path = experiment_dir / "metadata.json"
        data_path.write_text(record["text"], encoding="utf-8")
        metadata = {
            "experiment_name": experiment_name,
            "split": split,
            "source": record["source"],
            "dataset_name": record["dataset_name"],
            "dataset_subset": record["dataset_subset"],
            "article_id": record["article_id"],
            "title": record["title"],
            "url": record["url"],
            "text_chars": len(record["text"]),
            "exact_hash": record["exact_hash"],
            "source_score": record["score"],
        }
        write_json(metadata_path, metadata)
        materialized.append(
            {
                **metadata,
                "relative_data_path": str(data_path.relative_to(output_dir)),
                "relative_metadata_path": str(metadata_path.relative_to(output_dir)),
                "entry_hash": stable_hash(
                    {
                        "experiment_name": experiment_name,
                        "split": split,
                        "article_id": record["article_id"],
                        "exact_hash": record["exact_hash"],
                    }
                ),
            }
        )
    return materialized


def build_wikipedia_snapshot(
    *,
    output_dir: str | Path,
    data_root: str | Path,
    include_heldout_experiments: list[str] | None = None,
    config: SnapshotConfig | None = None,
) -> dict[str, Any]:
    """Build the reproducible Wikipedia train and held-out snapshot used by the benchmark."""
    config = config or SnapshotConfig()
    include_heldout_experiments = include_heldout_experiments or ["wikipedia_india"]
    if config.heldout_size < len(include_heldout_experiments):
        raise ValueError("heldout_size must be >= number of include_heldout_experiments.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)

    heldout_from_hf = config.heldout_size - len(include_heldout_experiments)
    required_unique = config.train_size + heldout_from_hf
    candidates, scanned = _collect_stream_candidates(config=config, needed_count=required_unique)
    deduped, duplicates = _dedupe_candidates(
        candidates=candidates,
        near_dup_jaccard_threshold=config.near_dup_jaccard_threshold,
    )
    if len(deduped) < required_unique:
        raise RuntimeError(
            f"Not enough unique candidates after dedupe ({len(deduped)} < {required_unique}). "
            f"Increase max_scan from {config.max_scan}."
        )

    heldout_records = deduped[:heldout_from_hf]
    train_records = deduped[heldout_from_hf : heldout_from_hf + config.train_size]
    for experiment_name in include_heldout_experiments:
        heldout_records.append(
            _load_existing_experiment(
                data_root=data_root,
                experiment_name=experiment_name,
            )
        )
    train_records, train_dropped = _dedupe_candidates(
        candidates=train_records,
        near_dup_jaccard_threshold=config.near_dup_jaccard_threshold,
    )
    if len(train_records) < config.train_size:
        raise RuntimeError(
            f"Train records dropped below requested size after final dedupe: "
            f"{len(train_records)} < {config.train_size}"
        )
    train_records = train_records[: config.train_size]
    heldout_records, heldout_dropped = _dedupe_candidates(
        candidates=heldout_records,
        near_dup_jaccard_threshold=config.near_dup_jaccard_threshold,
    )

    leakage = _compute_cross_split_leakage(
        train_records=train_records,
        heldout_records=heldout_records,
        near_dup_jaccard_threshold=config.near_dup_jaccard_threshold,
    )
    if not leakage["passed"]:
        raise RuntimeError(
            "Leakage check failed between train and heldout splits. "
            "Adjust seed/size/threshold."
        )

    train_entries = _materialize_records(records=train_records, output_dir=output_dir, split="train")
    heldout_entries = _materialize_records(
        records=heldout_records,
        output_dir=output_dir,
        split="heldout",
    )

    manifest = {
        "schema_version": "still.wikipedia_snapshot.v1",
        "dataset_source": {
            "dataset_name": config.dataset_name,
            "dataset_subset": config.dataset_subset,
            "split": "train",
            "streaming": True,
        },
        "selection_config": {
            "train_size": config.train_size,
            "heldout_size": config.heldout_size,
            "seed": config.seed,
            "max_scan": config.max_scan,
            "min_chars": config.min_chars,
            "max_chars": config.max_chars,
            "near_dup_jaccard_threshold": config.near_dup_jaccard_threshold,
            "include_heldout_experiments": include_heldout_experiments,
        },
        "scan_stats": {
            "rows_scanned": scanned,
            "candidate_pool_size": len(candidates),
            "deduped_pool_size": len(deduped),
            "dropped_within_pool": duplicates,
            "dropped_train_after_split": train_dropped,
            "dropped_heldout_after_split": heldout_dropped,
        },
        "split_stats": {
            "train_count": len(train_entries),
            "heldout_count": len(heldout_entries),
        },
        "leakage_summary": leakage,
        "train_entries": train_entries,
        "heldout_entries": heldout_entries,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(output_dir / "snapshot_manifest.json", manifest)
    (output_dir / "train_ids.txt").write_text(
        "\n".join(entry["experiment_name"] for entry in train_entries) + "\n",
        encoding="utf-8",
    )
    (output_dir / "heldout_ids.txt").write_text(
        "\n".join(entry["experiment_name"] for entry in heldout_entries) + "\n",
        encoding="utf-8",
    )
    return manifest
