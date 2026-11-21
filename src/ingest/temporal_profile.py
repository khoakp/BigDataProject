"""Read-only MovieLens temporal profiling for v2 cutoff selection.

All cutoffs are UTC half-open boundaries: base event_ts < cutoff, replay >= cutoff.
Only a fresh output directory is accepted. No canonical or graph artifact is changed.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


CUTOFFS = ("2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01")
FIELDS = "userId,movieId,rating,timestamp"


def utc_second(day: str) -> int:
    return calendar.timegm(datetime.strptime(day, "%Y-%m-%d").timetuple())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(con, query: str, path: Path) -> None:
    escaped = str(path.resolve()).replace("\\", "/").replace("'", "''")
    con.execute(f"COPY ({query}) TO '{escaped}' (HEADER, DELIMITER ',')")


def _create_first(con, name: str, select: str, key: str, where: str = "") -> None:
    con.execute(f"CREATE TABLE {name} AS SELECT {select}, min(event_ts) AS first_ts, "
                f"max(event_ts) AS last_ts FROM events {where} GROUP BY {key}")


def _monthly_query() -> str:
    return """
      WITH event_month AS (
        SELECT strftime(to_timestamp(event_ts), '%Y-%m') AS month,
          count(*) AS rating_events,
          count(*) FILTER (WHERE rating >= 4.0) AS positive_events,
          count(DISTINCT user_raw_id) AS distinct_users,
          count(DISTINCT movie_raw_id) AS distinct_movies
        FROM events GROUP BY 1
      ), first_users AS (
        SELECT strftime(to_timestamp(first_ts), '%Y-%m') AS month, count(*) AS n
        FROM first_user GROUP BY 1
      ), first_movies AS (
        SELECT strftime(to_timestamp(first_ts), '%Y-%m') AS month, count(*) AS n
        FROM first_movie GROUP BY 1
      ), first_graph_users AS (
        SELECT strftime(to_timestamp(first_ts), '%Y-%m') AS month, count(*) AS n
        FROM first_positive_user GROUP BY 1
      ), first_graph_movies AS (
        SELECT strftime(to_timestamp(first_ts), '%Y-%m') AS month, count(*) AS n
        FROM first_positive_movie GROUP BY 1
      ), first_edges AS (
        SELECT strftime(to_timestamp(first_ts), '%Y-%m') AS month, count(*) AS n
        FROM first_positive_edge GROUP BY 1
      )
      SELECT e.month, e.rating_events, e.positive_events,
        e.distinct_users, e.distinct_movies,
        coalesce(u.n,0) AS first_users, coalesce(m.n,0) AS first_movies,
        coalesce(gu.n,0) AS first_positive_graph_users,
        coalesce(gm.n,0) AS first_positive_graph_movies,
        coalesce(edge.n,0) AS new_positive_edges,
        sum(e.rating_events) OVER (ORDER BY e.month) AS cumulative_events,
        sum(e.positive_events) OVER (ORDER BY e.month) AS cumulative_positive_events,
        sum(coalesce(u.n,0)) OVER (ORDER BY e.month) AS cumulative_users,
        sum(coalesce(m.n,0)) OVER (ORDER BY e.month) AS cumulative_movies,
        sum(coalesce(gu.n,0)) OVER (ORDER BY e.month) AS cumulative_positive_graph_users,
        sum(coalesce(gm.n,0)) OVER (ORDER BY e.month) AS cumulative_positive_graph_movies,
        sum(coalesce(edge.n,0)) OVER (ORDER BY e.month) AS cumulative_positive_edges
      FROM event_month e
      LEFT JOIN first_users u USING (month)
      LEFT JOIN first_movies m USING (month)
      LEFT JOIN first_graph_users gu USING (month)
      LEFT JOIN first_graph_movies gm USING (month)
      LEFT JOIN first_edges edge USING (month)
      ORDER BY e.month
    """


def _yearly_query() -> str:
    return """
      WITH year_entities AS (
        SELECT strftime(to_timestamp(event_ts), '%Y') AS year,
          count(DISTINCT user_raw_id) AS distinct_users,
          count(DISTINCT movie_raw_id) AS distinct_movies
        FROM events GROUP BY 1
      ), annual AS (
        SELECT left(month,4) AS year, sum(rating_events) AS rating_events,
          sum(positive_events) AS positive_events,
          sum(first_users) AS first_users, sum(first_movies) AS first_movies,
          sum(first_positive_graph_users) AS first_positive_graph_users,
          sum(first_positive_graph_movies) AS first_positive_graph_movies,
          sum(new_positive_edges) AS new_positive_edges,
          max(cumulative_events) AS cumulative_events,
          max(cumulative_positive_events) AS cumulative_positive_events,
          max(cumulative_users) AS cumulative_users,
          max(cumulative_movies) AS cumulative_movies,
          max(cumulative_positive_graph_users) AS cumulative_positive_graph_users,
          max(cumulative_positive_graph_movies) AS cumulative_positive_graph_movies,
          max(cumulative_positive_edges) AS cumulative_positive_edges,
          count(*) AS months_with_events
        FROM monthly GROUP BY 1
      )
      SELECT a.*, e.distinct_users, e.distinct_movies
      FROM annual a JOIN year_entities e USING (year) ORDER BY year
    """


def _cutoff_row(con, day: str, maximum_ts: int) -> dict:
    cutoff = utc_second(day)
    event = con.execute("""SELECT count(*) FILTER (WHERE event_ts < ?),
        count(*) FILTER (WHERE event_ts >= ?),
        count(*) FILTER (WHERE event_ts >= ? AND rating >= 4.0),
        count(*) FILTER (WHERE event_ts = ?) FROM events""",
                        [cutoff, cutoff, cutoff, cutoff]).fetchone()
    def entity(table: str) -> tuple[int, int, int]:
        return con.execute(f"SELECT count(*) FILTER (WHERE first_ts < ?), "
                           f"count(*) FILTER (WHERE last_ts >= ?), "
                           f"count(*) FILTER (WHERE first_ts >= ?) FROM {table}",
                           [cutoff, cutoff, cutoff]).fetchone()
    users = entity("first_user")
    movies = entity("first_movie")
    graph_users = entity("first_positive_user")
    graph_movies = entity("first_positive_movie")
    edges = con.execute("SELECT count(*) FILTER (WHERE first_ts < ?), "
                        "count(*) FILTER (WHERE first_ts >= ?) FROM first_positive_edge",
                        [cutoff, cutoff]).fetchone()
    return {"cutoff_utc": day, "cutoff_unix": cutoff,
            "base_events": event[0], "replay_events": event[1],
            "replay_positive_events": event[2], "events_exactly_at_cutoff": event[3],
            "replay_span_days": round(max(0, maximum_ts-cutoff+1)/86400, 2),
            "base_distinct_users": users[0], "base_distinct_movies": movies[0],
            "base_graph_users": graph_users[0], "base_graph_movies": graph_movies[0],
            "base_graph_nodes": graph_users[0]+graph_movies[0],
            "base_positive_edges": edges[0], "base_sparse_entries": 2*edges[0],
            "replay_active_users": users[1], "replay_new_users": users[2],
            "replay_new_user_fraction": users[2]/users[1] if users[1] else None,
            "replay_active_movies": movies[1], "replay_new_movies": movies[2],
            "replay_new_movie_fraction": movies[2]/movies[1] if movies[1] else None,
            "replay_positive_active_users": graph_users[1],
            "replay_new_graph_users": graph_users[2],
            "replay_new_graph_user_fraction": graph_users[2]/graph_users[1] if graph_users[1] else None,
            "replay_positive_active_movies": graph_movies[1],
            "replay_new_graph_movies": graph_movies[2],
            "replay_new_graph_movie_fraction": graph_movies[2]/graph_movies[1] if graph_movies[1] else None,
            "replay_new_positive_edges": edges[1]}


def profile(source: Path, output: Path, *, cutoffs: tuple[str, ...] = CUTOFFS,
            memory_limit: str = "1GB", threads: int = 2) -> dict:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig") as stream:
        if stream.readline().strip() != FIELDS:
            raise ValueError("Unexpected ratings.csv header")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    db = output / "profile.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("SET TimeZone='UTC'")
        con.execute(f"SET threads={int(threads)}")
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET temp_directory='{str(output / 'spill').replace(chr(92), '/')}'")
        con.execute("""CREATE TABLE events AS SELECT userId::BIGINT AS user_raw_id,
            movieId::BIGINT AS movie_raw_id, rating::DOUBLE AS rating,
            timestamp::BIGINT AS event_ts FROM read_csv(?, header=true, auto_detect=false,
            columns={'userId':'BIGINT','movieId':'BIGINT','rating':'DOUBLE','timestamp':'BIGINT'})""",
                    [str(source)])
        validity = con.execute("""SELECT count(*) AS rows,
            count(*) FILTER (WHERE user_raw_id IS NULL OR user_raw_id <= 0 OR
              movie_raw_id IS NULL OR movie_raw_id <= 0 OR rating IS NULL OR
              rating NOT BETWEEN 0.5 AND 5 OR rating*2 != floor(rating*2) OR
              event_ts IS NULL OR event_ts <= 0) AS invalid_rows,
            min(event_ts), max(event_ts),
            count(*) FILTER (WHERE rating >= 4.0) AS positive_events
            FROM events""").fetchone()
        if validity[1]:
            raise ValueError(f"Invalid MovieLens rows: {validity[1]}")
        _create_first(con, "first_user", "user_raw_id", "user_raw_id")
        _create_first(con, "first_movie", "movie_raw_id", "movie_raw_id")
        _create_first(con, "first_positive_user", "user_raw_id", "user_raw_id", "WHERE rating >= 4.0")
        _create_first(con, "first_positive_movie", "movie_raw_id", "movie_raw_id", "WHERE rating >= 4.0")
        _create_first(con, "first_positive_edge", "user_raw_id, movie_raw_id",
                      "user_raw_id, movie_raw_id", "WHERE rating >= 4.0")
        con.execute("CREATE TABLE monthly AS " + _monthly_query())
        _csv(con, "SELECT * FROM monthly ORDER BY month", output / "monthly.csv")
        con.execute("CREATE TABLE yearly AS " + _yearly_query())
        _csv(con, "SELECT * FROM yearly ORDER BY year", output / "yearly.csv")
        rows = [_cutoff_row(con, day, validity[3]) for day in cutoffs]
        con.register("cutoff_rows", __import__("pandas").DataFrame(rows))
        _csv(con, "SELECT * FROM cutoff_rows ORDER BY cutoff_utc", output / "cutoffs.csv")
        totals = con.execute("SELECT count(*), sum(rating_events), sum(positive_events), "
                             "sum(new_positive_edges), max(cumulative_users), "
                             "max(cumulative_movies), max(cumulative_positive_graph_users), "
                             "max(cumulative_positive_graph_movies) FROM monthly").fetchone()
        edge_count = con.execute("SELECT count(*) FROM first_positive_edge").fetchone()[0]
        if (totals[1] != validity[0] or totals[2] != validity[4] or
                totals[3] != edge_count or any(r["base_events"]+r["replay_events"] != validity[0] or
                                               r["base_positive_edges"]+r["replay_new_positive_edges"] != edge_count
                                               for r in rows)):
            raise AssertionError("Temporal aggregation reconciliation failed")
        result = {"source": str(source), "source_bytes": source.stat().st_size,
                  "source_sha256": file_sha256(source), "rows": validity[0],
                  "positive_events": validity[4], "unique_positive_edges": edge_count,
                  "distinct_users": totals[4], "distinct_movies": totals[5],
                  "positive_graph_users": totals[6], "positive_graph_movies": totals[7],
                  "min_event_ts": validity[2], "max_event_ts": validity[3],
                  "months_with_events": totals[0], "positive_threshold": 4.0,
                  "timezone": "UTC", "cutoff_rule": "base < cutoff; replay >= cutoff",
                  "entity_first_observed_rule": "minimum event_ts; ties do not affect period counts",
                  "edge_rule": "distinct (user_raw_id,movie_raw_id) with any rating >= 4.0; first positive timestamp",
                  "memory_limit": memory_limit, "threads": threads,
                  "outputs_sha256": {name: file_sha256(output / name) for name in
                                     ("monthly.csv", "yearly.csv", "cutoffs.csv")}}
        (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    db.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(profile(args.source, args.out, memory_limit=args.memory_limit,
                             threads=args.threads), indent=2))


if __name__ == "__main__":
    main()
