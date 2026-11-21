"""Train-only catalog and interaction history for the Streamlit demo."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb

from src.graph.full_builder import file_sha256


def verify_history_source(source: Path, expected_sha256: str) -> None:
    """Reject a history file that is not the canonical source used for training."""
    if not source.is_file() or file_sha256(source) != expected_sha256:
        raise ValueError("History Parquet does not match the REC-01 training source")


def train_rated_ids(source: Path, user_raw_id: int, train_end_ts: int) -> set[int]:
    """All movies rated by this user before cutoff, including negative ratings."""
    with duckdb.connect() as con:
        rows = con.execute(
            "SELECT DISTINCT movie_raw_id FROM read_parquet(?) "
            "WHERE user_raw_id = ? AND event_ts <= ?",
            [str(source), user_raw_id, train_end_ts],
        ).fetchall()
    return {movie_id for (movie_id,) in rows}


def train_history(source: Path, user_raw_id: int, train_end_ts: int,
                  movies: dict[int, tuple[str | None, str | None]], *,
                  limit: int = 100) -> dict:
    """Return only train-time ratings, newest first, with metadata added later."""
    if limit < 1 or limit > 1000:
        raise ValueError("History limit must be between 1 and 1000")
    with duckdb.connect() as con:
        total, positive = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE rating >= 4.0) "
            "FROM read_parquet(?) WHERE user_raw_id = ? AND event_ts <= ?",
            [str(source), user_raw_id, train_end_ts],
        ).fetchone()
        events = con.execute(
            "SELECT event_id, event_ts, movie_raw_id, rating FROM read_parquet(?) "
            "WHERE user_raw_id = ? AND event_ts <= ? "
            "ORDER BY event_ts DESC, event_id DESC LIMIT ?",
            [str(source), user_raw_id, train_end_ts, limit],
        ).fetchall()
    rows = []
    for event_id, timestamp, movie_id, rating in events:
        title, genres = movies.get(movie_id, (None, None))
        rows.append({"event_id": event_id, "time_utc": datetime.fromtimestamp(
            timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "movie_raw_id": movie_id, "title": title, "genres": genres,
            "rating": float(rating), "positive_for_graph": rating >= 4.0})
    return {"total_train_ratings": total, "positive_train_ratings": positive,
            "rows": rows, "limit": limit}


def train_catalog(movie_nodes: dict[int, int], popularity: dict[int, int],
                  movies: dict[int, tuple[str | None, str | None]]) -> list[dict]:
    """Present the exact train-movie candidate universe used by inference."""
    rows = []
    for movie_id in sorted(movie_nodes, key=lambda item: (-popularity.get(item, 0), item)):
        title, genres = movies.get(movie_id, (None, None))
        rows.append({"movie_raw_id": movie_id, "title": title, "genres": genres,
                     "train_positive_users": popularity.get(movie_id, 0)})
    return rows
