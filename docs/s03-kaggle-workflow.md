# S03 — Chuyển REC-01 sang Kaggle GPU và kiểm chứng artifact

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

**Trạng thái 21/11/2026:** bundle đã được tải lên [Dataset riêng tư REC-01](https://www.kaggle.com/datasets/khoakp/rec01-movielens-1m-s03-verified-bundle). [Notebook Version 3](https://www.kaggle.com/code/khoakp/rec01-movielens-1m-gpu-validation?scriptVersionId=372012165) chạy GPU T4 x2, kết thúc `REC01_KAGGLE_PASS`, `seal-output PASS`; artifact tải về đã đạt `verify-output PASS`. Giao thức chính đã được Project Owner duyệt ở [S01](s01-evaluation-spec.md). Script `scripts/rec01_kaggle_prep.py` chỉ đóng gói canonical prefix 1M và mã Python cần thiết; không lấy token, credential hoặc raw `ratings.csv`.

**Bundle đã xác minh tại máy này:** `data/processed/rec01_kaggle_bundle_s03/`, `verify-input=PASS`, `rec01_source.zip` SHA-256 `07f05600dfe8dfaf108e927ff3f41ebe926afeead84f8b9bd3946b060170d49f`. Thư mục `data/processed/rec01_kaggle_bundle/` là lượt đóng gói chẩn đoán trước khi chốt code verifier; **không dùng để upload**. Nếu source code thay đổi sau khi đóng gói, tạo bundle mới với `--out` mới và cập nhật hash thay vì dùng bản cũ.

## 1. Tạo bundle trên máy local

Từ root repo `BigDataProject`, sau khi code REC-01 đã qua tests:

```powershell
..\venv\Scripts\python.exe scripts\rec01_kaggle_prep.py prepare --source data\processed\canonical_events\canonical_1000000.parquet --out data\processed\rec01_kaggle_bundle_s03
..\venv\Scripts\python.exe scripts\rec01_kaggle_prep.py verify-input --bundle data\processed\rec01_kaggle_bundle_s03
```

`--out` phải mới/trống. Script kiểm SHA-256 canonical `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`, 1.000.000 event IDs liên tiếp, 478.972 timestamps phân biệt, cutoffs `840100781`/`841267862` và split counts `802988/102046/94966`. Bundle chứa:

- `canonical_1000000.parquet` và `d02_manifest.json`.
- `src/` và `scripts/rec01_kaggle_prep.py` chỉ gồm mã `.py`, cùng `rec01_source.zip` có nội dung tương đương.
- `input_manifest.json`: hash từng file mã, ZIP, canonical source; commit hiện tại và cờ worktree dirty. Code hash là bằng chứng chính khi code chưa được commit.

Bundle nằm dưới `data/processed/` nên không tự vào Git. Giữ một bản bundle và hash manifest để đối chiếu sau khi tải kết quả về.

## 2. Gắn bundle vào Kaggle Notebook

Tạo **private Dataset** bằng trang **New Dataset** với đúng bốn tệp ở gốc bundle: `canonical_1000000.parquet`, `d02_manifest.json`, `input_manifest.json`, `rec01_source.zip`. Tạo Notebook Python và thêm dataset trong **File → Add input**. Trên Dataset này, Kaggle đã giải nén ZIP vào `rec01_source/src/` và `rec01_source/scripts/`, không giữ ZIP ở đường dẫn gốc. Cell dưới đây cũng xử lý trường hợp ZIP còn nguyên hoặc cây mã nằm trực tiếp ở gốc. [Kaggle Datasets](https://www.kaggle.com/docs/datasets), [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

Chọn GPU trong Notebook session options; nếu cần cài DuckDB bằng `pip`, bật Internet cho phiên đó hoặc gắn wheel phù hợp. Cell dưới đây cũng được lưu ở [`notebooks/rec01_kaggle.ipynb`](../notebooks/rec01_kaggle.ipynb) để tái lập bản nháp Kaggle. Đổi `REC01-Kaggle-1M` thành run ID mới, ghi lại Notebook URL/version, accelerator và package versions trong log S03.

```python
from pathlib import Path
import importlib.metadata
import json
import shutil
import subprocess
import sys
import zipfile

matches = list(Path('/kaggle/input').rglob('input_manifest.json'))
assert len(matches) == 1, f'Expected one REC-01 bundle; found {matches}'
bundle = matches[0].parent
project = Path('/kaggle/working/rec01_project')
project.mkdir(exist_ok=False)
for name in ('canonical_1000000.parquet', 'd02_manifest.json', 'input_manifest.json'):
    shutil.copy2(bundle / name, project / name)
archive = bundle / 'rec01_source.zip'
if archive.is_file():
    shutil.copy2(archive, project / archive.name)
    with zipfile.ZipFile(archive) as source_zip:
        source_zip.extractall(project)
else:
    source_tree = bundle / 'rec01_source'
    if not (source_tree / 'src').is_dir():
        source_tree = bundle
    for name in ('src', 'scripts'):
        shutil.copytree(source_tree / name, project / name)

try:
    duckdb_version = importlib.metadata.version('duckdb')
except importlib.metadata.PackageNotFoundError:
    duckdb_version = None
if duckdb_version != '1.5.6':
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'duckdb==1.5.6'], check=True)

import torch
assert torch.cuda.is_available(), 'GPU is not enabled in this Kaggle notebook'
print(json.dumps({'bundle': str(bundle), 'python': sys.version,
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0),
                  'duckdb': importlib.metadata.version('duckdb')}, sort_keys=True), flush=True)

subprocess.run([sys.executable, '-u', '-m', 'scripts.rec01_kaggle_prep',
                'verify-input', '--bundle', str(project)], cwd=project, check=True)
run = project / 'REC01-Kaggle-1M'
subprocess.run([sys.executable, '-u', '-m', 'src.models.rec01_train',
                '--source', str(project / 'canonical_1000000.parquet'),
                '--out', str(run), '--device', 'cuda'], cwd=project, check=True)
subprocess.run([sys.executable, '-u', '-m', 'scripts.rec01_kaggle_prep',
                'seal-output', '--bundle', str(project), '--run', str(run)],
               cwd=project, check=True)
print('REC01_KAGGLE_PASS', run, flush=True)
```

Lịch sử thực tế trên [Notebook REC-01](https://www.kaggle.com/code/khoakp/rec01-movielens-1m-gpu-validation): [Version 1](https://www.kaggle.com/code/khoakp/rec01-movielens-1m-gpu-validation?scriptVersionId=371924331) chạy 17 giây rồi lỗi `FileNotFoundError` vì tìm `src/` ở gốc Dataset, trong khi Kaggle đặt mã dưới `rec01_source/`. [Version 2](https://www.kaggle.com/code/khoakp/rec01-movielens-1m-gpu-validation?scriptVersionId=371924863) đã chạy REC-01 đến hết huấn luyện và chọn checkpoint bước 300, nhưng `seal-output` lỗi `ModuleNotFoundError` cho `src` do gọi script theo đường dẫn khiến `sys.path` bắt đầu ở `scripts/`. Cell trên gọi script dưới dạng module từ root project. Một số lần thử gửi Version 3 qua công cụ trình duyệt bị automatic approval review từ chối; sau đó Notebook Version 3 đã xuất hiện trong tài khoản và chạy thành công. Version 3 có `output_manifest.json` và được xác minh local nên là kết quả S03 cuối.

`rec01_train` khẳng định lại giao thức đã duyệt, train tối đa 500 bước, lưu checkpoint ở 100/300/500, chọn bằng validation NDCG@20, load lại checkpoint rồi mới đánh giá test. Output gồm `report.json`, `rec01_quality.csv`, `rankings.json`, `model_ids.json`, ba checkpoint và `graph_build/`. `seal-output` chỉ thành công nếu report PASS, graph Full/Incremental equivalence PASS, nguồn/code/selection/cohorts và checkpoint hashes khớp; sau đó ghi `output_manifest.json` cho mọi file. Nếu run lỗi, giữ report ERROR và logs để phân tích, không báo cáo nó như kết quả.

Lưu phiên bản Notebook bằng **Save & Run All** để Kaggle chạy lại từ trạng thái sạch và giữ output dưới `/kaggle/working`. Kaggle mô tả phiên bản này gồm code, logs, output files, data sources và môi trường; tải output của phiên bản đã hoàn tất từ pane Output. [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

## 3. Chuyển về máy và xác minh

Tải toàn bộ thư mục `rec01_project/REC01-Kaggle-1M` từ Notebook Output, giải nén vào một thư mục local mới dưới `data/processed/rec01_kaggle_runs/` (không ghi đè). Sau đó:

```powershell
..\venv\Scripts\python.exe -m scripts.rec01_kaggle_prep verify-output --bundle data\processed\rec01_kaggle_bundle_s03 --run data\processed\rec01_kaggle_runs\version_3\rec01_project\REC01-Kaggle-1M
```

**Kết quả thực tế:** Output archive 55.238.120 byte ở `data/processed/rec01_kaggle_runs/version_3/results.zip`, SHA-256 `25b3f97606aa330fc5892195e7afea7b825f48818c45b31b8b947d3f6172fe49`; 141 ZIP entries được giải nén vào `version_3/`. Lệnh trên trả `verify-output PASS` với source SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`. Run dùng Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, DuckDB 1.5.6; Kaggle hiển thị runtime 1 phút 35 giây. GPU trong report là Tesla T4, còn Notebook session ghi T4 x2; không diễn giải đây là huấn luyện phân tán trên hai GPU.

Verifier kiểm file list/hash từ `output_manifest.json`, hash của từng checkpoint và ID map, source/config/code provenance, ba điểm validation, checkpoint đã reload, graph equivalence, hai methods/cùng warm cohorts và sáu raw CSV rows. Nó xác nhận **tính toàn vẹn artifact được chuyển về**. Số đo và giới hạn chất lượng nằm ở [S04](s04-rec01-results.md); test Popularity nhỉnh hơn LightGCN trong run này.

Nếu không có quyền Kaggle, quota hoặc dependency install thất bại, ghi lý do và chạy cùng CLI trên GPU laptop ở scale đã kiểm chứng với `--device cuda`, gắn nhãn `local GPU fallback`. Không gọi run đó là Kaggle. Không đổi source/cutoff/candidate/selection rule sau khi xem test.
