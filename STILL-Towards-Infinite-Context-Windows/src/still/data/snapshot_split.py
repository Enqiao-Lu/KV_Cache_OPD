import json
from pathlib import Path
from typing import Any


def load_snapshot_manifest(snapshot_root: str | Path) -> dict[str, Any]:
    """Load and validate the materialized Wikipedia snapshot manifest."""
    snapshot_root = Path(snapshot_root)
    manifest_path = snapshot_root / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "still.wikipedia_snapshot.v1":
        raise ValueError(f"Unsupported snapshot schema: {payload.get('schema_version')}")
    return payload


def _entry_dir(snapshot_root: Path, split: str, experiment_name: str) -> Path:
    """Resolve the directory that stores one snapshot entry."""
    return snapshot_root / split / experiment_name


def list_snapshot_entries(
    snapshot_root: str | Path,
    *,
    split: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return normalized metadata rows for one snapshot split."""
    payload = load_snapshot_manifest(snapshot_root)
    key = "train_entries" if split == "train" else "heldout_entries"
    entries = list(payload[key])
    if limit is not None:
        entries = entries[:limit]
    root = Path(snapshot_root)
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        experiment_dir = _entry_dir(root, split, entry["experiment_name"])
        normalized.append(
            {
                **entry,
                "split": split,
                "experiment_dir": str(experiment_dir.resolve()),
                "data_path": str((experiment_dir / "data.txt").resolve()),
                "metadata_path": str((experiment_dir / "metadata.json").resolve()),
            }
        )
    return normalized


def train_snapshot_entries(
    snapshot_root: str | Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper for the training portion of the snapshot."""
    return list_snapshot_entries(snapshot_root, split="train", limit=limit)


def heldout_snapshot_entries(
    snapshot_root: str | Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper for the held-out portion of the snapshot."""
    return list_snapshot_entries(snapshot_root, split="heldout", limit=limit)
