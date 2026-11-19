"""D05 exact logical-state comparison; version lineage is checked separately."""
import hashlib
import json
from pathlib import Path

import duckdb


def checksum(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def full_state(output: Path) -> dict:
    with duckdb.connect() as con:
        return {name: con.execute(
            f"SELECT {columns} FROM read_parquet(?) ORDER BY ALL", [str(output / f"{name}.parquet")]
        ).fetchall() for name, columns in (
            ("registry", "node_id,node_type,raw_id,first_event_id"),
            ("edges", "user_node_id,movie_node_id,first_event_id"),
            ("degrees", "node_id,degree"))}


def incremental_state(graph) -> dict:
    return {name: graph.con.execute(f"SELECT {columns} FROM {table} ORDER BY ALL").fetchall()
            for name, table, columns in (
                ("registry", "node_registry", "node_id,node_type,raw_id,first_event_id"),
                ("edges", "edge_state", "user_node_id,movie_node_id,first_event_id"),
                ("degrees", "degree_state", "node_id,degree"))}


def compare_states(reference: dict, actual: dict) -> dict:
    """Compare complete sorted rows, not counts or hashes alone."""
    checks = {}
    for name in ("registry", "edges", "degrees"):
        left, right = sorted(reference[name]), sorted(actual[name])
        checks[name] = {"equal": left == right, "full_rows": len(left), "incremental_rows": len(right),
                        "full_sha256": checksum(left), "incremental_sha256": checksum(right)}
    checks["pass"] = all(checks[name]["equal"] for name in ("registry", "edges", "degrees"))
    return checks
