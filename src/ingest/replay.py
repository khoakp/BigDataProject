"""Slice an existing canonical event range without renumbering event IDs."""
from pathlib import Path

import duckdb


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def slice_events(source: Path, target: Path, start: int, end: int):
    if start < 0 or end <= start:
        raise ValueError("Invalid event range")
    with duckdb.connect() as con:
        con.execute("SET threads=1")
        con.execute(f"COPY (SELECT * FROM read_parquet('{sql_path(source)}') WHERE event_id >= {start} "
                    f"AND event_id < {end} ORDER BY event_id) TO '{sql_path(target)}' (FORMAT PARQUET)")
        count, minimum, maximum = con.execute("SELECT count(*),min(event_id),max(event_id) FROM read_parquet(?)",
                                               [str(target)]).fetchone()
    if (count, minimum, maximum) != (end - start, start, end - 1):
        raise AssertionError("Canonical slice does not cover requested event IDs")
