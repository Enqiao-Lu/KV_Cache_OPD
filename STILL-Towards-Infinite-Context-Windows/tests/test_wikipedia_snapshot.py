from still.data.wikipedia_snapshot import (
    _compute_cross_split_leakage,
    _dedupe_candidates,
)


def _candidate(article_id: str, text: str) -> dict[str, object]:
    return {
        "source": "hf",
        "dataset_name": "wikimedia/wikipedia",
        "dataset_subset": "20231101.en",
        "article_id": article_id,
        "title": article_id,
        "url": f"https://example.org/{article_id}",
        "text": text,
        "score": 1,
    }


def test_dedupe_candidates_drops_exact_duplicates() -> None:
    candidates = [
        _candidate("a1", "Alpha beta gamma"),
        _candidate("a2", "Alpha beta gamma"),
        _candidate("a3", "Completely different"),
    ]
    accepted, dropped = _dedupe_candidates(candidates=candidates, near_dup_jaccard_threshold=0.9)
    assert len(accepted) == 2
    assert len(dropped) == 1
    assert dropped[0]["type"] == "exact"


def test_leakage_summary_detects_cross_split_exact_overlap() -> None:
    train, _ = _dedupe_candidates(
        candidates=[_candidate("train_1", "One two three"), _candidate("train_2", "Delta epsilon")],
        near_dup_jaccard_threshold=0.9,
    )
    heldout, _ = _dedupe_candidates(
        candidates=[_candidate("heldout_1", "One two three"), _candidate("heldout_2", "Unique text")],
        near_dup_jaccard_threshold=0.9,
    )
    leakage = _compute_cross_split_leakage(
        train_records=train,
        heldout_records=heldout,
        near_dup_jaccard_threshold=0.9,
    )
    assert leakage["exact_overlap_count"] == 1
    assert leakage["passed"] is False
