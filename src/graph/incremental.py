"""Transactional, single-writer incremental historical-positive graph (D04)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb


class IncrementalGraph:
    """Own one connection; use as a context manager to release the database lock."""

    def __init__(self, database: Path | str = ":memory:"):
        self.con = duckdb.connect(str(database))
        self.con.execute("SET threads=1")
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS node_registry (
                node_id BIGINT PRIMARY KEY, node_type VARCHAR NOT NULL,
                raw_id BIGINT NOT NULL, first_event_id BIGINT NOT NULL,
                created_version INTEGER NOT NULL, UNIQUE(node_type, raw_id));
            CREATE TABLE IF NOT EXISTS edge_state (
                user_node_id BIGINT NOT NULL, movie_node_id BIGINT NOT NULL,
                first_event_id BIGINT NOT NULL, created_version INTEGER NOT NULL,
                PRIMARY KEY(user_node_id, movie_node_id));
            CREATE TABLE IF NOT EXISTS degree_state (
                node_id BIGINT PRIMARY KEY, degree BIGINT NOT NULL,
                updated_version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS applied_batches (
                batch_id INTEGER PRIMARY KEY, min_event_id BIGINT NOT NULL,
                max_event_id BIGINT NOT NULL, batch_hash VARCHAR NOT NULL,
                graph_version INTEGER NOT NULL UNIQUE, events_processed BIGINT NOT NULL,
                positive_events BIGINT NOT NULL, inserted_edges BIGINT NOT NULL,
                last_order_key VARCHAR NOT NULL);
        """)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def apply_batch(self, batch_id: int, events: Path) -> dict:
        """Hash the canonical row payload, then apply once in a DB transaction.

        Batch IDs are nonnegative unique identifiers. Event IDs must continue the
        previous watermark without gaps; nonempty all-negative batches are valid.
        Empty batches are rejected. Threshold is fixed at 4.0 (DEC-005).
        """
        if type(batch_id) is not int or batch_id < 0:
            raise ValueError("batch_id must be a nonnegative integer")
        path = str(events.resolve()).replace("\\", "/").replace("'", "''")
        self.con.execute(f"""CREATE OR REPLACE TEMP TABLE incoming AS
            SELECT event_id, event_ts, user_raw_id, movie_raw_id, rating, source_row_id
            FROM read_parquet('{path}') ORDER BY event_id""")
        schema = self.con.execute("DESCRIBE incoming").fetchall()
        integer_columns = {"event_id", "event_ts", "user_raw_id", "movie_raw_id", "source_row_id"}
        if any(name in integer_columns and dtype not in ("BIGINT", "INTEGER") for name, dtype, *_ in schema):
            raise ValueError("Canonical IDs/timestamps must use integer types")
        # The hash covers all events, including negatives and source-row identity.
        digest = hashlib.sha256()
        first_id = last_id = None
        first_key = last_key = None
        positive = count = 0
        cursor = self.con.execute("SELECT * FROM incoming ORDER BY event_id")
        while rows := cursor.fetchmany(8192):
            for eid, ts, user, movie, rating, source_row in rows:
                if any(v is None for v in (eid, ts, user, movie, rating, source_row)):
                    raise ValueError("Null canonical field")
                if not (user > 0 and movie > 0 and ts > 0 and source_row >= 0 and
                        0.5 <= rating <= 5 and rating * 2 == int(rating * 2)):
                    raise ValueError("Invalid canonical field")
                key = (ts, user, movie, source_row)
                if last_id is not None and (eid != last_id + 1 or key <= last_key):
                    raise ValueError("Noncontiguous event IDs or noncanonical ordering")
                if first_id is None:
                    first_id, first_key = eid, key
                last_id, last_key = eid, key
                count += 1
                positive += rating >= 4
                digest.update((json.dumps([eid, ts, user, movie, float(rating), source_row], separators=(",", ":")) + "\n").encode())
        if not count:
            raise ValueError("Empty batch")
        batch_hash = digest.hexdigest()
        self.con.execute("BEGIN TRANSACTION")
        try:
            existing = self.con.execute("SELECT batch_hash,graph_version FROM applied_batches WHERE batch_id=?", [batch_id]).fetchone()
            if existing:
                if existing[0] != batch_hash:
                    raise ValueError("Conflicting batch hash for existing batch_id")
                self.con.execute("COMMIT")
                return {"status": "NO_OP", "batch_id": batch_id, "batch_hash": batch_hash,
                        "graph_version": existing[1], "inserted_edges": 0}
            previous = self.con.execute("SELECT max_event_id,graph_version,last_order_key FROM applied_batches ORDER BY graph_version DESC LIMIT 1").fetchone()
            if first_id != (previous[0] + 1 if previous else 0):
                raise ValueError("Batch does not continue event watermark")
            if previous and first_key <= tuple(json.loads(previous[2])):
                raise ValueError("Batch violates canonical ordering at boundary")
            version = previous[1] + 1 if previous else 1
            self.con.execute("""CREATE OR REPLACE TEMP TABLE new_edges AS
                SELECT i.user_raw_id, i.movie_raw_id, min(i.event_id) AS first_event_id
                FROM incoming i WHERE i.rating>=4.0 AND NOT EXISTS (
                    SELECT 1 FROM edge_state e
                    JOIN node_registry u ON u.node_id=e.user_node_id
                    JOIN node_registry m ON m.node_id=e.movie_node_id
                    WHERE u.raw_id=i.user_raw_id AND m.raw_id=i.movie_raw_id)
                GROUP BY i.user_raw_id,i.movie_raw_id""")
            offset = self.con.execute("SELECT coalesce(max(node_id)+1,0) FROM node_registry").fetchone()[0]
            self.con.execute("""INSERT INTO node_registry
                WITH endpoints AS (
                    SELECT 'user' AS node_type,user_raw_id AS raw_id,min(first_event_id) AS first_event_id FROM new_edges GROUP BY user_raw_id
                    UNION ALL SELECT 'movie',movie_raw_id,min(first_event_id) FROM new_edges GROUP BY movie_raw_id
                ), unseen AS (
                    SELECT * FROM endpoints x WHERE NOT EXISTS (
                        SELECT 1 FROM node_registry r WHERE r.node_type=x.node_type AND r.raw_id=x.raw_id)
                )
                SELECT ? + row_number() OVER (ORDER BY first_event_id, CASE node_type WHEN 'user' THEN 0 ELSE 1 END,raw_id)-1,
                       node_type,raw_id,first_event_id,? FROM unseen""", [offset, version])
            self.con.execute("""CREATE OR REPLACE TEMP TABLE inserted AS
                SELECT u.node_id AS user_node_id,m.node_id AS movie_node_id,n.first_event_id
                FROM new_edges n JOIN node_registry u ON u.node_type='user' AND u.raw_id=n.user_raw_id
                JOIN node_registry m ON m.node_type='movie' AND m.raw_id=n.movie_raw_id""")
            inserted = self.con.execute("SELECT count(*) FROM inserted").fetchone()[0]
            self.con.execute("INSERT INTO edge_state SELECT *, ? FROM inserted", [version])
            self.con.execute("""INSERT INTO degree_state
                SELECT node_id,count(*),? FROM (
                    SELECT user_node_id AS node_id FROM inserted UNION ALL SELECT movie_node_id FROM inserted
                ) GROUP BY node_id
                ON CONFLICT(node_id) DO UPDATE SET degree=degree_state.degree+excluded.degree,
                    updated_version=excluded.updated_version""", [version])
            self.con.execute("INSERT INTO applied_batches VALUES (?,?,?,?,?,?,?,?,?)",
                             [batch_id,first_id,last_id,batch_hash,version,count,positive,inserted,json.dumps(last_key)])
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return {"status": "APPLIED", "batch_id": batch_id, "batch_hash": batch_hash,
                "graph_version": version, "events_processed": count,
                "positive_events": positive, "inserted_edges": inserted}

    def snapshot(self) -> dict:
        """Ordered state for auditing/tests; the ledger is the version authority."""
        return {name: self.con.execute(f"SELECT * FROM {name} ORDER BY ALL").fetchall()
                for name in ("node_registry", "edge_state", "degree_state", "applied_batches")}
