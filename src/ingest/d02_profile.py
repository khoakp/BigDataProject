"""D02: profile MovieLens ratings and export deterministic chronological prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import duckdb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(source: Path, out: Path, *, max_prefix: int = 1_000_000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"SET temp_directory='{(out / 'tmp').as_posix()}'")
    source_sql = str(source.resolve()).replace("'", "''").replace("\\", "/")
    with source.open("r", encoding="utf-8-sig") as stream:
        header = stream.readline().strip()
    if header != "userId,movieId,rating,timestamp":
        raise ValueError(f"Unexpected header: {header}")
    # One scanner, one thread and insertion order preserve the CSV physical row order.
    con.execute(f"""
        CREATE TEMP TABLE source AS
        SELECT row_number() OVER () - 1 AS source_row_id,
               userId::BIGINT AS user_raw_id, movieId::BIGINT AS movie_raw_id,
               rating::DOUBLE AS rating, timestamp::BIGINT AS event_ts
        FROM read_csv('{source_sql}', header=true, auto_detect=false,
            columns={{'userId':'BIGINT','movieId':'BIGINT','rating':'DOUBLE','timestamp':'BIGINT'}})
    """)
    stats = con.execute("""
        SELECT count(*) AS rows, count(*) FILTER (WHERE user_raw_id IS NULL) AS null_user,
          count(*) FILTER (WHERE movie_raw_id IS NULL) AS null_movie,
          count(*) FILTER (WHERE rating IS NULL) AS null_rating,
          count(*) FILTER (WHERE event_ts IS NULL) AS null_timestamp,
          min(rating) AS min_rating, max(rating) AS max_rating,
          count(*) FILTER (WHERE rating NOT BETWEEN 0.5 AND 5 OR rating*2 != floor(rating*2)) AS invalid_rating,
          min(event_ts) AS min_timestamp, max(event_ts) AS max_timestamp,
          count(*) FILTER (WHERE event_ts <= 0) AS invalid_timestamp,
          count(DISTINCT user_raw_id) AS users, count(DISTINCT movie_raw_id) AS movies,
          count(DISTINCT event_ts) AS distinct_timestamps,
          count(DISTINCT (user_raw_id, movie_raw_id)) AS distinct_pairs
        FROM source
    """).fetchone()
    keys = [d[0] for d in con.description]
    profile = dict(zip(keys, stats))
    profile["duplicate_pair_rows"] = profile["rows"] - profile["distinct_pairs"]
    profile["timestamp_tie_rows"] = profile["rows"] - profile["distinct_timestamps"]
    profile["source_order_descents"] = con.execute("""
        SELECT count(*) FROM (SELECT event_ts, lag(event_ts) OVER (ORDER BY source_row_id) AS prior FROM source)
        WHERE event_ts < prior
    """).fetchone()[0]
    if any(profile[k] for k in ("null_user", "null_movie", "null_rating", "null_timestamp", "invalid_rating", "invalid_timestamp")):
        raise ValueError("Source validation failed; see profile")
    if max_prefix > profile["rows"]:
        raise ValueError("Prefix exceeds source rows")
    con.execute(f"""
        CREATE TEMP TABLE canonical_prefix AS
        SELECT row_number() OVER (ORDER BY event_ts, user_raw_id, movie_raw_id, source_row_id) - 1 AS event_id,
          event_ts, user_raw_id, movie_raw_id, rating::FLOAT AS rating, source_row_id
        FROM source
        ORDER BY event_ts, user_raw_id, movie_raw_id, source_row_id
        LIMIT {int(max_prefix)}
    """)
    # Positive threshold remains undecided (DEC-005); batch_id belongs to replay, not D02.
    prefix_sizes = sorted({min(100_000, max_prefix), max_prefix})
    for n in prefix_sizes:
        target = (out / f"canonical_{n}.parquet").as_posix().replace("'", "''")
        con.execute(f"COPY (SELECT * FROM canonical_prefix WHERE event_id < {n} ORDER BY event_id) TO '{target}' (FORMAT PARQUET)")
    profile["prefixes"] = {}
    for n in prefix_sizes:
        path = out / f"canonical_{n}.parquet"
        profile["prefixes"][str(n)] = {"rows": n, "sha256": sha256(path), "bytes": path.stat().st_size}
    profile["source_sha256"] = sha256(source)
    profile["transform"] = {"version": "d02-v1", "threads": 1, "source_row_id_base": 0,
                            "order_by": ["event_ts", "user_raw_id", "movie_raw_id", "source_row_id"],
                            "event_id_base": 0, "rating_dtype": "FLOAT", "positive_threshold": "not_chosen"}
    (out / "manifest.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../ml-32m/ratings.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/canonical_events"))
    args = parser.parse_args()
    result = run(args.source, args.out)
    print(json.dumps(result, indent=2))
