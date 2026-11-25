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


class PreflightSampler:
    """Small process-tree sampler; DuckDB and phase timing stay with the caller."""

    def __init__(self, *, interval=.1, device='cpu', spill_dir: Path | None = None):
        self.interval = interval
        self.device = device
        self.spill_dir = spill_dir
        self.parent = psutil.Process()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss_bytes = 0
        self.min_available_ram_bytes = None
        self.min_gpu_free_fraction = None
        self.gpu_peak_allocated_bytes = None
        self.gpu_peak_reserved_bytes = None
        self.gpu_free_bytes = None
        self.gpu_total_bytes = None
        self.gpu_error = None
        self.peak_spill_bytes = 0
        self.min_disk_free_bytes = None
        self.spill_error = None

    def sample(self):
        try:
            family = [self.parent, *self.parent.children(recursive=True)]
            rss = sum(process.memory_info().rss for process in family if process.is_running())
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        except psutil.Error:
            pass  # A child may exit between enumeration and sampling.
        available = psutil.virtual_memory().available
        self.min_available_ram_bytes = (available if self.min_available_ram_bytes is None
                                        else min(self.min_available_ram_bytes, available))
        if self.spill_dir is not None:
            import shutil
            try:
                used = sum(path.stat().st_size for path in self.spill_dir.rglob('*')
                           if path.is_file()) if self.spill_dir.is_dir() else 0
                self.peak_spill_bytes = max(self.peak_spill_bytes, used)
                free = shutil.disk_usage(self.spill_dir.parent).free
                self.min_disk_free_bytes = (free if self.min_disk_free_bytes is None
                                            else min(self.min_disk_free_bytes, free))
            except OSError as exc:
                self.spill_error = f'{type(exc).__name__}: {exc}'
        if self.device == 'cuda:0':
            try:
                import torch
                free, total = torch.cuda.mem_get_info(0)
                self.gpu_free_bytes, self.gpu_total_bytes = free, total
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                self.gpu_peak_allocated_bytes = max(self.gpu_peak_allocated_bytes or 0, allocated)
                self.gpu_peak_reserved_bytes = max(self.gpu_peak_reserved_bytes or 0, reserved)
                fraction = free / total
                self.min_gpu_free_fraction = (fraction if self.min_gpu_free_fraction is None
                                              else min(self.min_gpu_free_fraction, fraction))
            except Exception as exc:
                self.gpu_error = f'{type(exc).__name__}: {exc}'

    def _run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.interval)

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()

    def record(self):
        return dict(process_tree_peak_rss_bytes=self.peak_rss_bytes,
                    min_host_available_ram_bytes=self.min_available_ram_bytes,
                    gpu_min_free_fraction=self.min_gpu_free_fraction,
                    gpu_peak_allocated_bytes=self.gpu_peak_allocated_bytes,
                    gpu_peak_reserved_bytes=self.gpu_peak_reserved_bytes,
                    gpu_free_bytes=self.gpu_free_bytes,
                    gpu_total_bytes=self.gpu_total_bytes,
                    gpu_unavailable_reason=(self.gpu_error or
                        ('CPU measurement; no CUDA device sampled' if self.device == 'cpu' else None)),
                    sampling_interval_s=self.interval,
                    peak_spill_bytes=self.peak_spill_bytes if self.spill_dir else None,
                    spill_unavailable_reason=(self.spill_error if self.spill_dir else
                                              'No spill directory configured'),
                    min_disk_free_bytes=self.min_disk_free_bytes,
                    limitation='Sampling can miss short peaks; children exiting between samples are omitted')
