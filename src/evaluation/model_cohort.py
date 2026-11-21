"""Deterministic pre-cutoff LightGCN workload cohorts and post-cutoff coverage.

Entity selection uses only positive edges before the UTC cutoff. Future positives
are read later solely to measure fixed-cohort coverage, never to select nodes.
Each exported cohort is a standalone bipartite graph for GPU feasibility probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


DEFAULT_CUTOFF_TS = 1577836800  # 2020-01-01 00:00:00 UTC
DEFAULT_BUDGETS = (500_000, 2_000_000, 5_000_000, 10_000_000, 13_644_889)
CSV_COLUMNS = "{'userId':'BIGINT','movieId':'BIGINT','rating':'DOUBLE','timestamp':'BIGINT'}"
SELECTION_BASIS = {
    "movie_degree": "pre-cutoff positive movie degree DESC, raw movie ID ASC; all incident edges",
    "user_degree": "pre-cutoff positive user degree DESC, raw user ID ASC; all incident edges",
    "user_recency": "pre-cutoff last positive user timestamp DESC, degree DESC, raw user ID ASC; all incident edges",
    "chronological_prefix": "pre-cutoff positive edge first timestamp ASC, raw user/movie IDs ASC; edge subset",
    "hash_sample": "pre-cutoff positive pair DuckDB hash(user,movie,7) ASC, raw user/movie IDs ASC; edge subset",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export(con, query: str, path: Path) -> None:
    escaped = str(path.resolve()).replace("\\", "/").replace("'", "''")
    con.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET)")


def _row(con, budget: int, selected_units: int, full_base_movies: int,
         future_pairs: int) -> dict:
    con.execute("CREATE OR REPLACE TEMP TABLE nodes AS "
                "WITH raw_nodes AS ("
                "  SELECT DISTINCT 'user' AS node_type, user_raw_id AS raw_id FROM selected_edges "
                "  UNION ALL SELECT DISTINCT 'movie', movie_raw_id FROM selected_edges"
                ") SELECT row_number() OVER (ORDER BY CASE node_type WHEN 'user' THEN 0 ELSE 1 END, raw_id)-1 "
                "AS node_id, node_type, raw_id FROM raw_nodes")
    con.execute("CREATE OR REPLACE TEMP TABLE graph_edges AS "
                "SELECT u.node_id AS user_node_id, m.node_id AS movie_node_id, e.first_event_ts "
                "FROM selected_edges e JOIN nodes u ON u.node_type='user' AND u.raw_id=e.user_raw_id "
                "JOIN nodes m ON m.node_type='movie' AND m.raw_id=e.movie_raw_id")
    counts = con.execute("SELECT count(*), count(*) FILTER (WHERE node_type='user'), "
                         "count(*) FILTER (WHERE node_type='movie') FROM nodes").fetchone()
    edge_count = con.execute("SELECT count(*) FROM graph_edges").fetchone()[0]
    coverage = con.execute("""
      WITH annotated AS (
        SELECT f.user_raw_id, f.movie_raw_id,
          u.raw_id IS NOT NULL AS in_user,
          m.raw_id IS NOT NULL AS in_movie,
          e.user_raw_id IS NOT NULL AS train_seen,
          bu.user_raw_id IS NOT NULL AS in_base_user,
          bm.movie_raw_id IS NOT NULL AS in_base_movie
        FROM future_positive f
        LEFT JOIN nodes u ON u.node_type='user' AND u.raw_id=f.user_raw_id
        LEFT JOIN nodes m ON m.node_type='movie' AND m.raw_id=f.movie_raw_id
        LEFT JOIN selected_edges e ON e.user_raw_id=f.user_raw_id AND e.movie_raw_id=f.movie_raw_id
        LEFT JOIN base_users bu ON bu.user_raw_id=f.user_raw_id
        LEFT JOIN base_movies bm ON bm.movie_raw_id=f.movie_raw_id
      )
      SELECT count(*),
        count(*) FILTER (WHERE NOT in_user),
        count(*) FILTER (WHERE in_user AND NOT in_movie),
        count(*) FILTER (WHERE in_user AND in_movie AND train_seen),
        count(*) FILTER (WHERE in_user AND in_movie AND NOT train_seen),
        count(*) FILTER (WHERE in_movie),
        count(DISTINCT user_raw_id) FILTER (WHERE in_user AND in_movie AND NOT train_seen),
        count(DISTINCT movie_raw_id) FILTER (WHERE in_user AND in_movie AND NOT train_seen),
        count(*) FILTER (WHERE NOT in_user AND NOT in_base_user),
        count(*) FILTER (WHERE NOT in_user AND in_base_user),
        count(*) FILTER (WHERE in_user AND NOT in_movie AND NOT in_base_movie),
        count(*) FILTER (WHERE in_user AND NOT in_movie AND in_base_movie)
      FROM annotated
    """).fetchone()
    if coverage[0] != future_pairs or sum(coverage[1:5]) != future_pairs:
        raise AssertionError("Coverage categories do not partition future positive pairs")
    if coverage[1] != coverage[8] + coverage[9] or coverage[2] != coverage[10] + coverage[11]:
        raise AssertionError("Cold/omitted categories do not partition cohort exclusions")
    return {"edge_budget": budget, "selected_rank_units": selected_units,
            "nodes": counts[0], "users": counts[1], "movies": counts[2],
            "positive_edges": edge_count, "sparse_entries": 2 * edge_count,
            "base_movie_catalog_coverage": counts[2] / full_base_movies,
            "future_positive_pairs": coverage[0],
            "outside_model_user_pairs": coverage[1],
            "cold_base_user_pairs": coverage[8],
            "omitted_base_user_pairs": coverage[9],
            "outside_model_movie_pairs_after_user": coverage[2],
            "cold_base_movie_pairs_after_user": coverage[10],
            "omitted_base_movie_pairs_after_user": coverage[11],
            "train_seen_repeat_pairs": coverage[3],
            "evaluable_future_pairs": coverage[4],
            "evaluable_fraction": coverage[4] / future_pairs if future_pairs else None,
            "future_pair_movie_catalog_coverage": coverage[5] / future_pairs if future_pairs else None,
            "evaluable_warm_users": coverage[6], "evaluable_warm_movies": coverage[7]}


def _select_edges(con, strategy: str, budget: int, base_edges: int,
                  full_rank_units: int) -> int:
    """Select a prefix of a ranking computed entirely from base-positive edges."""
    if strategy in ("movie_degree", "user_degree", "user_recency"):
        units = (full_rank_units if budget >= base_edges else
                 con.execute("SELECT coalesce(max(rank),0) FROM unit_rank "
                             "WHERE cumulative_edges <= ?", [budget]).fetchone()[0])
        if units == 0:
            raise ValueError("Budget smaller than first ranked entity's degree")
        key = "movie_raw_id" if strategy == "movie_degree" else "user_raw_id"
        con.execute("CREATE OR REPLACE TEMP TABLE selected_edges AS "
                    f"SELECT e.* FROM base_edges e JOIN unit_rank r ON r.raw_id=e.{key} "
                    "WHERE r.rank <= ?", [units])
    else:
        units = min(budget, base_edges)
        con.execute("CREATE OR REPLACE TEMP TABLE selected_edges AS "
                    "SELECT e.* FROM base_edges e JOIN edge_rank r "
                    "USING (user_raw_id,movie_raw_id) WHERE r.rank <= ?", [units])
    return units


def prepare_cohorts(source: Path, output: Path, *, cutoff_ts: int = DEFAULT_CUTOFF_TS,
                    edge_budgets: tuple[int, ...] = DEFAULT_BUDGETS,
                    strategy: str = "movie_degree", export_graphs: bool = True,
                    memory_limit: str = "1GB", threads: int = 2) -> dict:
    """Export immutable pre-cutoff induced graphs and inspect fixed-cohort future labels.

    Strategies use only base-positive data: top movies by degree, top users by
    degree/recency, chronological edge prefix, or seeded edge hash. Entity
    strategies select the largest whole-entity prefix with E <= budget.
    """
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if cutoff_ts <= 0 or not edge_budgets or any(b <= 0 for b in edge_budgets):
        raise ValueError("Expected positive cutoff and edge budgets")
    if tuple(sorted(set(edge_budgets))) != tuple(edge_budgets):
        raise ValueError("Edge budgets must be strictly increasing")
    if strategy not in ("movie_degree", "user_degree", "user_recency",
                        "chronological_prefix", "hash_sample"):
        raise ValueError("Unknown selection strategy")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hash = file_sha256(source)
    semantics = ("complete_induced_pre_cutoff_positive" if strategy in
                 ("movie_degree", "user_degree", "user_recency") else
                 "edge_subset_not_complete_induced_graph")
    db = output / "work.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("SET TimeZone='UTC'")
        con.execute(f"SET threads={int(threads)}")
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET temp_directory='{str(output / 'spill').replace(chr(92), '/')}'")
        con.execute("CREATE TABLE base_edges AS SELECT userId::BIGINT AS user_raw_id, "
                    "movieId::BIGINT AS movie_raw_id, min(timestamp::BIGINT) AS first_event_ts, "
                    "max(timestamp::BIGINT) AS last_positive_ts "
                    f"FROM read_csv(?, header=true, auto_detect=false, columns={CSV_COLUMNS}) "
                    "WHERE timestamp < ? AND rating >= 4.0 GROUP BY 1,2",
                    [str(source), cutoff_ts])
        base_edges, full_base_movies = con.execute(
            "SELECT count(*), count(DISTINCT movie_raw_id) FROM base_edges").fetchone()
        con.execute("CREATE TABLE base_users AS SELECT DISTINCT user_raw_id FROM base_edges")
        con.execute("CREATE TABLE base_movies AS SELECT DISTINCT movie_raw_id FROM base_edges")
        if strategy in ("movie_degree", "user_degree", "user_recency"):
            key = "movie_raw_id" if strategy == "movie_degree" else "user_raw_id"
            order = ("degree DESC, raw_id" if strategy != "user_recency" else
                     "last_positive_ts DESC, degree DESC, raw_id")
            con.execute("CREATE TABLE unit_rank AS WITH counts AS ("
                        f"SELECT {key} AS raw_id, count(*) AS degree, "
                        "max(last_positive_ts) AS last_positive_ts FROM base_edges GROUP BY 1) "
                        "SELECT raw_id, row_number() OVER (ORDER BY " + order + ") AS rank, "
                        "sum(degree) OVER (ORDER BY " + order + ") AS cumulative_edges "
                        "FROM counts")
            full_rank_units = con.execute("SELECT count(*) FROM unit_rank").fetchone()[0]
        else:
            ordering = ("first_event_ts,user_raw_id,movie_raw_id" if strategy == "chronological_prefix" else
                        "hash(user_raw_id,movie_raw_id,7),user_raw_id,movie_raw_id")
            con.execute("CREATE TABLE edge_rank AS SELECT user_raw_id,movie_raw_id, "
                        "row_number() OVER (ORDER BY " + ordering + ") AS rank FROM base_edges")
            full_rank_units = base_edges
        con.execute("CREATE TABLE future_positive AS SELECT userId::BIGINT AS user_raw_id, "
                    "movieId::BIGINT AS movie_raw_id "
                    f"FROM read_csv(?, header=true, auto_detect=false, columns={CSV_COLUMNS}) "
                    "WHERE timestamp >= ? AND rating >= 4.0 GROUP BY 1,2",
                    [str(source), cutoff_ts])
        future_pairs = con.execute("SELECT count(*) FROM future_positive").fetchone()[0]
        rows = []
        for budget in edge_budgets:
            units = _select_edges(con, strategy, budget, base_edges, full_rank_units)
            row = _row(con, budget, units, full_base_movies, future_pairs)
            if export_graphs:
                subdir = output / f"edges_{budget}"
                subdir.mkdir()
                _export(con, "SELECT node_id,node_type,raw_id FROM nodes ORDER BY node_id", subdir / "nodes.parquet")
                _export(con, "SELECT user_node_id,movie_node_id,first_event_ts FROM graph_edges "
                        "ORDER BY user_node_id,movie_node_id", subdir / "edges.parquet")
                row["nodes_sha256"] = file_sha256(subdir / "nodes.parquet")
                row["edges_sha256"] = file_sha256(subdir / "edges.parquet")
                (subdir / "manifest.json").write_text(json.dumps({
                    "source_sha256": source_hash, "cutoff_ts": cutoff_ts,
                    "positive_threshold": 4.0, "selection": strategy,
                    "selection_basis": SELECTION_BASIS[strategy],
                    "graph_semantics": semantics, **row}, indent=2), encoding="utf-8")
            rows.append(row)
        result = {"source": str(source), "source_sha256": source_hash,
                  "cutoff_ts": cutoff_ts, "positive_threshold": 4.0,
                  "selection": strategy,
                  "selection_basis": SELECTION_BASIS[strategy],
                  "graph_semantics": semantics,
                  "future_usage": "coverage only; never selection",
                  "base_positive_edges": base_edges, "base_positive_movies": full_base_movies,
                  "future_positive_pairs": future_pairs, "cohorts": rows}
        (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    db.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--cutoff-ts", type=int, default=DEFAULT_CUTOFF_TS)
    parser.add_argument("--edge-budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--strategy", choices=("movie_degree", "user_degree", "user_recency",
                                              "chronological_prefix", "hash_sample"), default="movie_degree")
    parser.add_argument("--no-export-graphs", action="store_true")
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    result = prepare_cohorts(args.source, args.out, cutoff_ts=args.cutoff_ts,
                             edge_budgets=tuple(args.edge_budgets),
                             strategy=args.strategy, export_graphs=not args.no_export_graphs,
                             memory_limit=args.memory_limit, threads=args.threads)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
