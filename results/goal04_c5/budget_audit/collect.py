"""Read-only Kaggle C3/C4 version audit; run with the Kaggle CLI environment."""
from __future__ import annotations

import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest


ROOT = Path(__file__).resolve().parent
KERNELS = {
    "c3": ("moviegraph-v2-c3-real-5m-preflight", range(1, 4)),
    "c4": ("moviegraph-v2-c4-single-t4-training", range(1, 5)),
}


def main():
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    rows = []
    for phase, (slug, versions) in KERNELS.items():
        for version in versions:
            ref = f"khoakp/{slug}/{version}"
            request = ApiGetKernelRequest()
            request.user_name = "khoakp"
            request.kernel_slug = slug
            request.version_label = api._kernel_version_label(str(version))
            metadata = client.kernels.kernels_api_client.get_kernel(request).metadata
            status = api.kernels_status(ref)
            log = api.kernels_logs(ref)
            files = api.kernels_list_files(ref)
            rows.append({
                "phase": phase, "version": version, "ref": ref,
                "status": str(status.status),
                "failure_message": status.failure_message,
                "metadata_last_run_time": str(metadata.last_run_time),
                "metadata_current_version_number": metadata.current_version_number,
                "log": log,
                "files": [{"name": item.name, "creation_date": str(item.creation_date)}
                          for item in files.files],
            })
    target = ROOT / "api-evidence.json"
    target.write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    print(target, len(rows), "versions")


if __name__ == "__main__":
    main()
