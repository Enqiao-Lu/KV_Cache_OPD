import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> str:
    """Serialize JSON deterministically so manifests and hashes stay stable."""
    if is_dataclass(data):
        data = asdict(data)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Any) -> str:
    """Hash arbitrary JSON-like data with deterministic canonicalization."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Compute a streaming SHA256 digest for a file on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Write indented JSON, creating parent directories first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write canonical JSONL rows, creating parent directories first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
