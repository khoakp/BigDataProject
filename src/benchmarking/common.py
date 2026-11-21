"""Shared, experiment-independent measurement and canonical slicing utilities."""
from __future__ import annotations

import threading
from pathlib import Path

import duckdb
import psutil


ROOT = Path(__file__).resolve().parents[2]


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


class RssSampler:
    def __init__(self, interval=.005):
        self.interval = interval
        self.process = psutil.Process()
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.stop.wait(self.interval)

    def __enter__(self):
        self._run_once()
        self.thread.start()
        return self

    def _run_once(self):
        self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self._run_once()
