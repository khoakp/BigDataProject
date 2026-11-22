"""Shared, experiment-independent measurement and canonical slicing utilities."""
from __future__ import annotations

import threading
from pathlib import Path

import psutil
from src.ingest.replay import slice_events, sql_path


ROOT = Path(__file__).resolve().parents[2]


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
