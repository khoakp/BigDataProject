"""Choose a model checkpoint using validation quality alone."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def _has_test_field(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Checkpoint record field names must be strings")
            lower = key.lower()
            if lower.startswith("test") or any(marker in lower for marker in ("_test", ".test", "-test")):
                return True
            if _has_test_field(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_test_field(child) for child in value)
    return False


def _metric(record: Mapping[str, object], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    if not 0 <= value <= 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return float(value)


def select_checkpoint(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Return the best record by validation NDCG@20, Recall@20, earliest step.

    Required fields are ``step``, ``validation_recall_20`` and
    ``validation_ndcg_20``. Test fields are forbidden even when nested in
    metadata, so test results cannot accidentally affect model selection.
    """
    best: dict[str, object] | None = None
    best_rank: tuple[float, float, int] | None = None
    steps: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each checkpoint record must be a mapping")
        if _has_test_field(record):
            raise ValueError("Test fields are forbidden during checkpoint selection")
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if step in steps:
            raise ValueError(f"Duplicate checkpoint step: {step}")
        steps.add(step)
        recall = _metric(record, "validation_recall_20")
        ndcg = _metric(record, "validation_ndcg_20")
        rank = (ndcg, recall, -step)
        if best_rank is None or rank > best_rank:
            best, best_rank = dict(record), rank
    if best is None:
        raise ValueError("At least one checkpoint record is required")
    return best
