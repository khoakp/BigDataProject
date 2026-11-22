"""Full rebuild reference for the historical-positive bipartite graph (D03)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
from src.artifacts import file_sha256
from src.ingest.canonical import validate_events


POSITIVE_THRESHOLD = 4.0


def rebuild(events: Path, output: Path | None = None, threshold: float = POSITIVE_THRESHOLD,
            *, validate: bool = True) -> dict:
    """Build state from all supplied canonical events; no incremental state is read."""
    if validate:
        validate_events(events)
    if output is not None and any((output / name).exists() for name in
                                  ("registry.parquet", "edges.parquet", "degrees.parquet", "manifest.json")):
        raise FileExistsError("Full rebuild output already contains graph artifacts")
    with duckdb.connect() as con:
        return _rebuild(con, events, output, threshold)


def _rebuild(con, events, output, threshold):
    if threshold != POSITIVE_THRESHOLD:
        raise ValueError("D03 policy is fixed at rating >= 4.0")
    con.execute("SET threads=1")
    path = str(events.resolve()).replace("\\", "/").replace("'", "''")
    con.execute(f"CREATE TEMP VIEW events AS SELECT * FROM read_parquet('{path}')")
    count, min_id, max_id = con.execute("SELECT count(*), min(event_id), max(event_id) FROM events").fetchone()
    if count == 0 or min_id != 0 or max_id != count - 1:
        raise ValueError("Expected nonempty canonical prefix with contiguous event IDs")
    con.execute("""
        CREATE TEMP TABLE positive_edges AS
        SELECT user_raw_id, movie_raw_id, min(event_id) AS first_event_id
        FROM events WHERE rating >= 4.0
        GROUP BY user_raw_id, movie_raw_id
    """)
    # A node is registered only when its first positive edge is inserted.
    # At equal first_event_id, allocate the user endpoint before the movie endpoint.
    con.execute("""
        CREATE TEMP TABLE registry AS
        WITH appearances AS (
          SELECT 'user' AS node_type, user_raw_id AS raw_id, min(first_event_id) AS first_event_id
          FROM positive_edges GROUP BY user_raw_id
          UNION ALL
          SELECT 'movie', movie_raw_id, min(first_event_id)
          FROM positive_edges GROUP BY movie_raw_id
        )
        SELECT row_number() OVER (ORDER BY first_event_id, CASE node_type WHEN 'user' THEN 0 ELSE 1 END, raw_id) - 1 AS node_id,
               node_type, raw_id, first_event_id
        FROM appearances
    """)
    con.execute("""
        CREATE TEMP TABLE edges AS
        SELECT u.node_id AS user_node_id, m.node_id AS movie_node_id,
               p.user_raw_id, p.movie_raw_id, p.first_event_id
        FROM positive_edges p
        JOIN registry u ON u.node_type='user' AND u.raw_id=p.user_raw_id
        JOIN registry m ON m.node_type='movie' AND m.raw_id=p.movie_raw_id
    """)
    con.execute("""
        CREATE TEMP TABLE degrees AS
        SELECT node_id, count(*) AS degree FROM (
          SELECT user_node_id AS node_id FROM edges
          UNION ALL SELECT movie_node_id FROM edges
        ) GROUP BY node_id
    """)
    edge_count = con.execute("SELECT count(*) FROM edges").fetchone()[0]
    degree_user, degree_movie = con.execute("""
        SELECT coalesce(sum(CASE WHEN r.node_type='user' THEN d.degree ELSE 0 END),0),
               coalesce(sum(CASE WHEN r.node_type='movie' THEN d.degree ELSE 0 END),0)
        FROM degrees d JOIN registry r USING (node_id)
    """).fetchone()
    if not (edge_count == degree_user == degree_movie):
        raise AssertionError("Bipartite degree invariant failed")
    users, movies = con.execute("SELECT count(*) FILTER (WHERE node_type='user'), count(*) FILTER (WHERE node_type='movie') FROM registry").fetchone()
    positive_events = con.execute("SELECT count(*) FROM events WHERE rating >= 4.0").fetchone()[0]
    result = {"events": count, "positive_events": positive_events, "edges": edge_count,
              "users": users, "movies": movies, "nodes": users + movies,
              "sum_user_degree": degree_user, "sum_movie_degree": degree_movie,
              "threshold": threshold, "input_sha256": file_sha256(events)}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for table, order in (("registry", "node_id"), ("edges", "user_raw_id, movie_raw_id"), ("degrees", "node_id")):
            target = str((output / f"{table}.parquet").resolve()).replace("\\", "/").replace("'", "''")
            con.execute(f"COPY (SELECT * FROM {table} ORDER BY {order}) TO '{target}' (FORMAT PARQUET)")
            result[f"{table}_sha256"] = file_sha256(output / f"{table}.parquet")
        (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=Path("data/processed/canonical_events/canonical_100000.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/full_reference_100000"))
    args = parser.parse_args()
    print(json.dumps(rebuild(args.events, args.out), indent=2))
