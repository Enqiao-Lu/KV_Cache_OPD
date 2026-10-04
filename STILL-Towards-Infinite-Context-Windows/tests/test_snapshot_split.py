import json

from still.data.snapshot_split import heldout_snapshot_entries, load_snapshot_manifest, train_snapshot_entries


def test_snapshot_manifest_can_be_loaded() -> None:
    payload = load_snapshot_manifest("data/wikipedia_snapshot_20231101_en")
    assert payload["schema_version"] == "still.wikipedia_snapshot.v1"
    assert payload["split_stats"]["train_count"] == 500
    assert payload["split_stats"]["heldout_count"] == 20


def test_snapshot_entry_helpers_resolve_paths() -> None:
    train_entries = train_snapshot_entries("data/wikipedia_snapshot_20231101_en", limit=2)
    heldout_entries = heldout_snapshot_entries("data/wikipedia_snapshot_20231101_en", limit=2)
    assert len(train_entries) == 2
    assert len(heldout_entries) == 2
    assert train_entries[0]["data_path"].endswith("data.txt")
    assert heldout_entries[0]["metadata_path"].endswith("metadata.json")
