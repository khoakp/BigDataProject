"""Immutable normalized graph snapshots and explicit freshness (D06).

The source graph DB and these snapshot tables share one transaction. Disk exports
are standalone model artifacts; this module does not implement checkpoint replay.
"""
import json
from pathlib import Path

import duckdb

from .equivalence import checksum, incremental_state
from .full_builder import file_sha256
from .normalize import ATOL, SparseAdjacency, normalize_state


def _initialize(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_graph_snapshots (
            model_graph_version INTEGER PRIMARY KEY, manifest VARCHAR NOT NULL);
        CREATE TABLE IF NOT EXISTS model_graph_nodes (
            model_graph_version INTEGER, node_id BIGINT, node_type VARCHAR,
            raw_id BIGINT, first_event_id BIGINT,
            PRIMARY KEY(model_graph_version,node_id));
        CREATE TABLE IF NOT EXISTS model_graph_coo (
            model_graph_version INTEGER, source BIGINT, target BIGINT, weight DOUBLE,
            PRIMARY KEY(model_graph_version,source,target));
    """)


def canonical_status(graph) -> dict:
    row = graph.con.execute("SELECT graph_version,max_event_id,last_order_key FROM applied_batches ORDER BY graph_version DESC LIMIT 1").fetchone()
    return {"canonical_version": row[0] if row else 0,
            "watermark_event_id": row[1] if row else -1,
            "max_event_ts": json.loads(row[2])[0] if row else None}


def materialize_normalized(graph, *, expected_version: int, provenance: dict) -> dict:
    """Capture ONLY current canonical version, never label current edges as old ones."""
    con = graph.con
    con.execute("BEGIN TRANSACTION")
    try:
        _initialize(con)
        source = canonical_status(graph)
        if type(expected_version) is not int or expected_version != source["canonical_version"]:
            raise ValueError("Requested version is not the current canonical version")
        state = incremental_state(graph)
        adjacency = normalize_state(state)
        manifest = {**source, "model_graph_version": expected_version,
                    "model_checkpoint_version": None,
                    "shape": list(adjacency.shape), "nnz": len(adjacency.entries),
                    "positive_edge_count": len(state["edges"]), "dtype": "float64",
                    "index_dtype": "int64", "normalization": "symmetric_degree_v1",
                    "normalization_owner": "graph_materializer", "self_loops": False,
                    "threshold": 4.0, "atol": ATOL,
                    "registry_sha256": checksum(state["registry"]),
                    "edge_sha256": checksum(state["edges"]),
                    "degree_sha256": checksum(state["degrees"]),
                    "coo_sha256": checksum(adjacency.entries), "provenance": provenance}
        existing = con.execute("SELECT manifest FROM model_graph_snapshots WHERE model_graph_version=?", [expected_version]).fetchone()
        if existing:
            stored = json.loads(existing[0])
            if stored != manifest:
                raise ValueError("Snapshot version already exists with different content/provenance")
            load_snapshot(graph,expected_version)
        else:
            con.execute("INSERT INTO model_graph_snapshots VALUES (?,?)",[expected_version,json.dumps(manifest,sort_keys=True)])
            if state["registry"]:
                con.execute("""INSERT INTO model_graph_nodes
                    SELECT ?,node_id,node_type,raw_id,first_event_id FROM node_registry""",
                            [expected_version])
            if adjacency.entries:
                con.execute("""INSERT INTO model_graph_coo
                    SELECT ?,e.user_node_id,e.movie_node_id,
                           1.0/sqrt(CAST(du.degree AS DOUBLE)*dm.degree)
                    FROM edge_state e
                    JOIN degree_state du ON du.node_id=e.user_node_id
                    JOIN degree_state dm ON dm.node_id=e.movie_node_id
                    UNION ALL
                    SELECT ?,e.movie_node_id,e.user_node_id,
                           1.0/sqrt(CAST(du.degree AS DOUBLE)*dm.degree)
                    FROM edge_state e
                    JOIN degree_state du ON du.node_id=e.user_node_id
                    JOIN degree_state dm ON dm.node_id=e.movie_node_id""",
                            [expected_version,expected_version])
        con.execute("COMMIT")
        return manifest
    except Exception:
        con.execute("ROLLBACK")
        raise


def _validate_payload(manifest, nodes, entries):
    n = len(nodes)
    if (manifest["shape"] != [n,n] or [r[0] for r in nodes] != list(range(n)) or
            manifest["nnz"] != len(entries) or checksum(nodes) != manifest["registry_sha256"] or
            checksum(entries) != manifest["coo_sha256"]):
        raise ValueError("Snapshot payload does not match its manifest")
    return SparseAdjacency((n,n),tuple(entries))


def load_snapshot(graph, model_graph_version: int):
    """Load old or current snapshot with its own node map, shape and watermark."""
    _initialize(graph.con)
    row = graph.con.execute("SELECT manifest FROM model_graph_snapshots WHERE model_graph_version=?",[model_graph_version]).fetchone()
    if not row:
        raise ValueError("Unknown model graph version")
    manifest = json.loads(row[0])
    nodes = graph.con.execute("SELECT node_id,node_type,raw_id,first_event_id FROM model_graph_nodes WHERE model_graph_version=? ORDER BY node_id",[model_graph_version]).fetchall()
    entries = graph.con.execute("SELECT source,target,weight FROM model_graph_coo WHERE model_graph_version=? ORDER BY source,target",[model_graph_version]).fetchall()
    return manifest,nodes,_validate_payload(manifest,nodes,entries)


def freshness(graph, model_graph_version: int) -> dict:
    manifest, _, _ = load_snapshot(graph,model_graph_version)
    current = canonical_status(graph)
    if model_graph_version > current["canonical_version"]:
        raise ValueError("Model snapshot is ahead of canonical state")
    return {**current, "model_graph_version": model_graph_version,
            "model_watermark_event_id": manifest["watermark_event_id"],
            "staleness_batches": current["canonical_version"]-model_graph_version,
            "staleness_events": current["watermark_event_id"]-manifest["watermark_event_id"]}


def export_snapshot(graph, model_graph_version: int, output: Path) -> dict:
    """Publish a fresh directory; manifest is written last as the completion marker."""
    manifest, _, _ = load_snapshot(graph,model_graph_version)
    output.mkdir(parents=True,exist_ok=False)
    for name, table, columns, order in (
        ("nodes", "model_graph_nodes", "node_id,node_type,raw_id,first_event_id", "node_id"),
        ("adjacency", "model_graph_coo", "source,target,weight", "source,target")):
        path = str((output / f"{name}.parquet").resolve()).replace("\\","/").replace("'","''")
        graph.con.execute(f"COPY (SELECT {columns} FROM {table} WHERE model_graph_version=? ORDER BY {order}) TO '{path}' (FORMAT PARQUET)",[model_graph_version])
    exported = {**manifest,"files_sha256":{name:file_sha256(output / name)
                                          for name in ("nodes.parquet","adjacency.parquet")}}
    temp = output / "manifest.json.tmp"
    temp.write_text(json.dumps(exported,indent=2,sort_keys=True),encoding="utf-8")
    temp.replace(output / "manifest.json")
    return exported


def load_export(output: Path):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name in ("nodes.parquet","adjacency.parquet"):
        if file_sha256(output / name) != manifest["files_sha256"][name]:
            raise ValueError("Export file hash mismatch")
    with duckdb.connect() as con:
        nodes = con.execute("SELECT * FROM read_parquet(?) ORDER BY node_id",[str(output / "nodes.parquet")]).fetchall()
        entries = con.execute("SELECT * FROM read_parquet(?) ORDER BY source,target",[str(output / "adjacency.parquet")]).fetchall()
    return manifest,nodes,_validate_payload(manifest,nodes,entries)
