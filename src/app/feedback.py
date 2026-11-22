"""Append-only demo ratings on a fork of the sealed REC-01 train graph.

Every submitted request is written as an immutable intent before its Parquet
batch is applied. Replaying an intent after interruption uses D04's batch
ledger and returns NO_OP instead of changing degrees twice. The Kaggle graph,
model snapshot, and selected LightGCN checkpoint are never modified.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import duckdb

from src.artifacts import file_sha256
from src.graph.incremental import IncrementalGraph
from src.graph.snapshot import canonical_status


_WRITE_LOCK = threading.RLock()  # Streamlit runs one server process for this demo.
_RATING_FIELDS = {"user_raw_id", "movie_raw_id", "rating"}


def _write_json_atomic(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _write_batch(path: Path, rows: list[list]) -> None:
    temp = path.with_name(path.stem + ".tmp.parquet")
    with duckdb.connect() as con:
        con.execute("CREATE TABLE events(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,"
                    "movie_raw_id BIGINT,rating FLOAT,source_row_id BIGINT)")
        con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", rows)
        escaped = str(temp).replace("\\", "/").replace("'", "''")
        con.execute(f"COPY events TO '{escaped}' (FORMAT PARQUET)")
    os.replace(temp, path)


def _batch_rows(path: Path) -> list[list]:
    with duckdb.connect() as con:
        return [list(row) for row in con.execute(
            "SELECT event_id,event_ts,user_raw_id,movie_raw_id,rating,source_row_id "
            "FROM read_parquet(?) ORDER BY event_id", [str(path)]).fetchall()]


class FeedbackStore:
    """Single-process, single-writer fork of a verified REC-01 train graph."""

    def __init__(self, run_dir: Path, source_events: Path, state_dir: Path):
        self.run_dir = Path(run_dir)
        self.source_events = Path(source_events)
        self.state_dir = Path(state_dir)
        self.graph_path = self.state_dir / "graph.duckdb"
        self.meta_path = self.state_dir / "state.json"
        self.intents_dir = self.state_dir / "intents"
        self.batches_dir = self.state_dir / "batches"

        report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
        seal = json.loads((self.run_dir / "output_manifest.json").read_text(encoding="utf-8"))
        if report["status"] != "PASS" or seal["run_id"] != report["run_id"]:
            raise ValueError("REC-01 report/seal mismatch")
        self.run_id = report["run_id"]
        self.source_sha256 = report["source"]["sha256"]
        self.base_version = report["graph"]["version"]
        self.base_watermark = report["protocol"]["train_events"] - 1
        self.train_end_ts = report["protocol"]["train_end_ts"]
        self.checkpoint_step = report["selected_checkpoint"]["step"]
        source_graph = self.run_dir / "graph_build/graph.duckdb"
        self.base_graph_sha256 = seal["files_sha256"]["graph_build/graph.duckdb"]
        if file_sha256(source_graph) != self.base_graph_sha256:
            raise ValueError("Sealed REC-01 graph database hash mismatch")
        if file_sha256(self.source_events) != self.source_sha256:
            raise ValueError("Canonical source differs from REC-01 training source")

        with _WRITE_LOCK:
            if self.meta_path.exists():
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                expected = {"run_id": self.run_id, "source_sha256": self.source_sha256,
                            "base_graph_sha256": self.base_graph_sha256,
                            "base_version": self.base_version,
                            "base_watermark": self.base_watermark,
                            "checkpoint_step": self.checkpoint_step}
                if meta != expected or not self.graph_path.is_file():
                    raise ValueError("Feedback state does not match the REC-01 artifact")
            else:
                if self.state_dir.exists() and any(self.state_dir.iterdir()):
                    raise ValueError("Incomplete feedback state; do not overwrite it")
                self.state_dir.mkdir(parents=True, exist_ok=True)
                staged = self.state_dir / "graph.tmp.duckdb"
                shutil.copyfile(source_graph, staged)
                os.replace(staged, self.graph_path)
                _write_json_atomic(self.meta_path, {
                    "run_id": self.run_id, "source_sha256": self.source_sha256,
                    "base_graph_sha256": self.base_graph_sha256,
                    "base_version": self.base_version,
                    "base_watermark": self.base_watermark,
                    "checkpoint_step": self.checkpoint_step,
                })
            self.intents_dir.mkdir(exist_ok=True)
            self.batches_dir.mkdir(exist_ok=True)
            with IncrementalGraph(self.graph_path) as graph:
                status = canonical_status(graph)
                if (status["canonical_version"] < self.base_version or
                        status["watermark_event_id"] < self.base_watermark):
                    raise ValueError("Feedback graph is behind the sealed train graph")
            self._recover_locked()

    def _intents(self) -> list[dict]:
        intents = [json.loads(path.read_text(encoding="utf-8"))
                   for path in self.intents_dir.glob("*.json")]
        intents.sort(key=lambda item: item["batch_id"])
        return intents

    def _apply_intent(self, intent: dict) -> dict:
        if intent["run_id"] != self.run_id or not intent["rows"]:
            raise ValueError("Feedback intent provenance mismatch")
        batch = self.batches_dir / f"batch_{intent['batch_id']:05d}.parquet"
        if batch.exists():
            if _batch_rows(batch) != intent["rows"]:
                raise ValueError("Feedback batch differs from durable intent")
        else:
            _write_batch(batch, intent["rows"])
        with IncrementalGraph(self.graph_path) as graph:
            return graph.apply_batch(intent["batch_id"], batch)

    def _recover_locked(self) -> None:
        for intent in self._intents():
            self._apply_intent(intent)

    def status(self) -> dict:
        with _WRITE_LOCK:
            self._recover_locked()
            with IncrementalGraph(self.graph_path) as graph:
                state = canonical_status(graph)
                edges = graph.con.execute("SELECT count(*) FROM edge_state").fetchone()[0]
                latest = graph.con.execute(
                    "SELECT batch_id,inserted_edges FROM applied_batches "
                    "ORDER BY graph_version DESC LIMIT 1").fetchone()
            return {"canonical_graph_version": state["canonical_version"],
                    "model_graph_version": self.base_version,
                    "checkpoint_step": self.checkpoint_step,
                    "staleness_batches": state["canonical_version"] - self.base_version,
                    "new_feedback_events": state["watermark_event_id"] - self.base_watermark,
                    "positive_edges": edges, "latest_batch_id": latest[0],
                    "latest_batch_inserted_edges": latest[1],
                    "refresh_status": "checkpoint_unchanged"}

    def applied_ratings(self) -> list[dict]:
        with _WRITE_LOCK:
            self._recover_locked()
            return [{"event_id": row[0], "event_ts": row[1], "user_raw_id": row[2],
                     "movie_raw_id": row[3], "rating": row[4],
                     "batch_id": intent["batch_id"]}
                    for intent in self._intents() for row in intent["rows"]]

    def apply_ratings(self, ratings: list[dict], request_id: str, *,
                      now_ts: int | None = None) -> dict:
        """Persist then apply one canonical batch; retrying request_id is idempotent."""
        try:
            request_id = str(uuid.UUID(request_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("request_id must be a UUID") from exc
        if type(ratings) is not list or not 1 <= len(ratings) <= 20:
            raise ValueError("Submit between 1 and 20 ratings per batch")
        if now_ts is not None and (type(now_ts) is not int or now_ts <= 0):
            raise ValueError("Invalid feedback timestamp")
        normalized = []
        for item in ratings:
            if type(item) is not dict or set(item) != _RATING_FIELDS:
                raise ValueError("Rating requires user_raw_id, movie_raw_id and rating")
            user, movie, rating = item["user_raw_id"], item["movie_raw_id"], item["rating"]
            if (type(user) is not int or user <= 0 or type(movie) is not int or movie <= 0 or
                    type(rating) not in (int, float) or not 0.5 <= rating <= 5.0 or
                    rating * 2 != int(rating * 2)):
                raise ValueError("Invalid user/movie ID or half-star rating")
            normalized.append((user, movie, float(rating)))
        if len({(user, movie) for user, movie, _ in normalized}) != len(normalized):
            raise ValueError("Duplicate user/movie pair within feedback batch")
        normalized.sort(key=lambda item: (item[0], item[1]))

        with _WRITE_LOCK:
            self._recover_locked()
            intent_path = self.intents_dir / f"{request_id}.json"
            if intent_path.exists():
                intent = json.loads(intent_path.read_text(encoding="utf-8"))
                saved = [(row[2], row[3], row[4]) for row in intent["rows"]]
                if saved != normalized:
                    raise ValueError("request_id reused with different ratings")
                result = self._apply_intent(intent)
                return {**result, "request_id": request_id}

            existing_pairs = {(row[2], row[3]) for intent in self._intents()
                              for row in intent["rows"]}
            requested_pairs = {(user, movie) for user, movie, _ in normalized}
            if requested_pairs & existing_pairs:
                raise ValueError("Movie already rated in this demo branch")
            with duckdb.connect() as con:
                con.execute("CREATE TEMP TABLE requested(user_raw_id BIGINT,movie_raw_id BIGINT)")
                con.executemany("INSERT INTO requested VALUES (?,?)",
                                [(user, movie) for user, movie, _ in normalized])
                overlap = con.execute(
                    "SELECT count(*) FROM read_parquet(?) e JOIN requested r "
                    "USING (user_raw_id,movie_raw_id) WHERE e.event_ts <= ?",
                    [str(self.source_events), self.train_end_ts]).fetchone()[0]
            if overlap:
                raise ValueError("Movie already rated in the train history")

            with IncrementalGraph(self.graph_path) as graph:
                last = graph.con.execute(
                    "SELECT batch_id,max_event_id,last_order_key FROM applied_batches "
                    "ORDER BY graph_version DESC LIMIT 1").fetchone()
            batch_id, first_event = last[0] + 1, last[1] + 1
            previous_ts = json.loads(last[2])[0]
            timestamp = max(int(time.time()) if now_ts is None else now_ts,
                            previous_ts + 1)
            if type(timestamp) is not int or timestamp <= 0:
                raise ValueError("Invalid feedback timestamp")
            rows = [[first_event + index, timestamp, user, movie, rating,
                     first_event + index]
                    for index, (user, movie, rating) in enumerate(normalized)]
            intent = {"run_id": self.run_id, "request_id": request_id,
                      "batch_id": batch_id, "rows": rows}
            _write_json_atomic(intent_path, intent)
            result = self._apply_intent(intent)
            return {**result, "request_id": request_id}
