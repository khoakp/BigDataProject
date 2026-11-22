"""Historical GPU feasibility CLI; implementation lives in benchmarking."""
import psutil  # compatibility for existing test patches

from src.benchmarking.gpu_feasibility import (
    _cpu_memory_snapshot, _cgroup_memory_files, _read_memory_value,
    AvailableMemorySampler, file_sha256, load_cohort, main, run,
)


if __name__ == "__main__":
    main()
