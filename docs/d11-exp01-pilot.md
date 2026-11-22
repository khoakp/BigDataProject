# D11 — EXP-01 pilot harness

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

## Protocol triển khai

Entry point: `src/benchmarking/exp01.py`. Grid đã đăng ký trong [kế hoạch](04-project-plan.md): `H={100k,500k,1M}`, `Δ={1k,10k}`. Input cho mọi cặp là cùng canonical prefix từ `ratings.csv` MovieLens 32M, đã sort theo D02. Trước khi đo, harness cắt history `H`, delta `(H,H+Δ]` và combined `H+Δ`; dựng DuckDB state chứa history bằng một batch D04. Chi phí và kích thước prebuild ghi riêng từng `H`.

Phép đo Full gọi D03 `rebuild(combined)` rồi xuất `registry`, `edges`, `degrees` Parquet. Phép đo Incremental copy prebuilt DB **trước timer**, apply delta qua D04 rồi xuất ba bảng cùng schema và thứ tự. Cả hai có manifest và hashes cho output; chi phí export nằm trong timer. `movies.csv`, normalization, LightGCN và metadata lookup nằm ngoài EXP-01. Sau timer, so sánh chính xác từng dòng registry/edges/degrees ở cùng cutoff; dòng sai bị gắn `CORRECTNESS_FAIL` và không dùng để tính speedup.

Mỗi cặp chạy một warm-up và một lượt đo cho mỗi phương án, mỗi lượt ở process mới. Thứ tự Full/Incremental được shuffle với seed 7 và giữ như nhau giữa warm-up và measured của cặp. DuckDB `threads=1`; cache policy là `warm-cache-uncontrolled` vì không reset OS page cache. RSS là peak lấy mẫu mỗi 5 ms, **không phải** đỉnh tuyệt đối. `app_bytes_read` là kích thước Parquet input; `export_bytes_written` là kích thước ba Parquet output và manifest; `db_growth_bytes` là mức tăng kích thước file DB, không phải số byte ghi thực tế. `os_read_bytes`/`os_write_bytes` là delta process I/O counters; các số này không tráo với application payload. CPU time lấy từ `time.process_time()`, wall time từ `time.perf_counter()`.

Lệnh từ root repo:

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_exp01 -v
..\venv\Scripts\python.exe -m src.benchmarking.exp01 pilot --ratings ..\ml-32m\ratings.csv --out data/processed/d11_exp01/D11-pilot-grid --csv results/raw/exp01_pilot_d11.csv
```

Để chạy lại, dùng `--out` mới và `--csv` mới, vì đầu ra benchmark không được ghi chồng. D12 mới làm repetitions và phân tích speedup/crossover. Một measured repetition cho mỗi cặp trong D11 chỉ là pilot, chưa đủ độ tin cậy để kết luận phương án nào nhanh hơn.

## Kết quả

Run `D11-pilot-grid` ngày 20/11/2026 **PASS**: 24/24 rows (sáu cặp × hai methods × warm-up/measured) có `status=PASS` và exact correctness `True`. Raw rows ở `results/raw/exp01_pilot_d11.csv`; manifest ở `data/processed/d11_exp01/D11-pilot-grid/report.json`. D02 đã tạo canonical 1.010.000 events từ `../ml-32m/ratings.csv`; source raw SHA-256 `91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee`, canonical prefix SHA-256 `6c2aa8d38a014be75125a2d694077ef364cba5465bf2da1da4d7966d60e175c1`. Python 3.10.11, DuckDB 1.5.6, Intel64 Family 6 Model 183, RAM 16.780.570.624 bytes, Windows 10 build 26300, single thread, seed 7. Git HEAD `123942550cf8192e5cc85f1c6f1432216ff39ab5`, worktree dirty; config hash `7809d045e13ba0c6f128d6ac0a7d91f011cf1e46a02755427b8edb2752a7c703`. Hash của harness `exp01.py` tại lúc chạy: `d32313b785630a1ba36cd7ce9b64526b65419311c90f7d5cbfceae60df1633a9`. Schema Parquet Full/Incremental của ba bảng đã được đối chiếu bằng `DESCRIBE`, cùng tên cột, kiểu và thứ tự.

| H | Δ | Full wall (s) | Incremental wall (s) | Full / Inc sampled peak RSS (MiB) | Correctness |
|---:|---:|---:|---:|---:|---|
| 100k | 1k | 0,0927 | 0,1547 | 55,9 / 57,5 | PASS |
| 100k | 10k | 0,1268 | 0,2448 | 56,5 / 67,2 | PASS |
| 500k | 1k | 0,2040 | 0,2385 | 80,3 / 77,5 | PASS |
| 500k | 10k | 0,1725 | 0,2808 | 76,8 / 76,1 | PASS |
| 1M | 1k | 0,2750 | 0,2831 | 107,4 / 89,0 | PASS |
| 1M | 10k | 0,3519 | 0,3493 | 103,2 / 102,6 | PASS |

Một số đo Full ở 500k+10k thấp hơn 500k+1k: đây là biến thiên của pilot một repetition, không được suy diễn là thêm events giúp Full nhanh hơn. Prebuild H một lần, **ngoài timer**: 0,842 s/4,21 MB ở 100k; 3,896 s/9,97 MB ở 500k; 7,551 s/18,10 MB ở 1M. CSV còn có CPU time, input/export bytes, DB growth và OS I/O counters từng row. Incremental ghi thêm DB state; ví dụ 1M+10k DB tăng 13.893.632 bytes, ngoài ~3.977.589 export bytes. Đây là kết quả đo trên một máy/cấu hình và warm cache không kiểm soát; chưa công bố speedup hoặc crossover. D12 sẽ chạy repetitions và phân tích spread.
