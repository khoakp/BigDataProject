"""Validate canonical event contracts in DuckDB without loading rows into Python."""
from pathlib import Path

import duckdb


FIELDS = ("event_id", "event_ts", "user_raw_id", "movie_raw_id", "rating", "source_row_id")


def validate_events(path: Path, *, start_id: int = 0) -> int:
    """Validate one ordered, contiguous canonical Parquet range."""
    with duckdb.connect() as con:
        con.execute("SET threads=1")
        schema = {name: dtype for name, dtype, *_ in con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}
        if any(name not in schema for name in FIELDS):
            raise ValueError("Canonical event schema is incomplete")
        if any(schema[name] not in ("BIGINT", "INTEGER") for name in FIELDS if name != "rating"):
            raise ValueError("Canonical IDs/timestamps must use integer types")
        count, invalid = con.execute("""
            WITH ordered AS (
                SELECT *, row_number() OVER (ORDER BY event_id) - 1 + ? AS expected_id,
                  lag(event_ts) OVER (ORDER BY event_id) AS prior_ts,
                  lag(user_raw_id) OVER (ORDER BY event_id) AS prior_user,
                  lag(movie_raw_id) OVER (ORDER BY event_id) AS prior_movie,
                  lag(source_row_id) OVER (ORDER BY event_id) AS prior_source
                FROM read_parquet(?)
            ) SELECT count(*), count(*) FILTER (WHERE
                event_id IS NULL OR event_id != expected_id OR event_ts IS NULL OR event_ts <= 0 OR
                user_raw_id IS NULL OR user_raw_id <= 0 OR
                movie_raw_id IS NULL OR movie_raw_id <= 0 OR
                source_row_id IS NULL OR source_row_id < 0 OR
                rating IS NULL OR NOT (rating BETWEEN 0.5 AND 5.0 AND
                                       rating * 2 = floor(rating * 2)) OR
                (prior_ts IS NOT NULL AND
                  (event_ts, user_raw_id, movie_raw_id, source_row_id) <=
                  (prior_ts, prior_user, prior_movie, prior_source))) FROM ordered
        """, [start_id, str(path)]).fetchone()
    if count == 0 or invalid:
        raise ValueError("Invalid canonical event IDs, fields or ordering")
    return count
