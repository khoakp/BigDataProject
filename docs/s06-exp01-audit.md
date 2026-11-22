# S06 — Kiểm kê EXP-01 để tái dùng raw benchmark D12

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

**Kết luận 21/11/2026:** S06 dùng lại hai run EXP-01 D12 đã hoàn thành, không chạy thêm benchmark hay đổi số liệu. [`scripts/audit_exp01_results.py`](../scripts/audit_exp01_results.py) kiểm tra độc lập CSV/report, pairing, provenance, summary, số byte export và SHA-256 của từng Parquet so với manifest. Core `D12-repeat-grid`: **72/72 rows, 36/36 cặp PASS**; scale `D12-scale-2m`: **24/24 rows, 12/12 cặp PASS**. D11 là pilot và không dùng để kết luận speedup. [D12](d12-exp01-results.md) giữ kết quả, biểu đồ và diễn giải chi tiết.

## Phạm vi đo và tính công bằng

- Mỗi cặp `(H, Δ)` dùng cùng canonical history + delta, cùng máy Windows/Intel64, DuckDB 1.5.6, một thread, một warm-up và năm measured repetitions/method. Thứ tự method được shuffle; cache policy `warm-cache-uncontrolled`.
- Full Rebuild xử lý H+Δ events; Incremental copy prebuilt history DB **trước timer** rồi apply Δ. Timer của cả hai gồm cập nhật graph và xuất cùng ba bảng Parquet `registry/edges/degrees`. Exact logical state được so sau timer ở từng cặp. Prebuild cost ghi riêng, không nằm trong tỷ số update wall time.
- CSV có wall/CPU seconds, sampled peak RSS (5 ms), application input bytes, Parquet export bytes, Incremental DB growth, process OS I/O và `events_per_s`. `events_per_s` dùng **H+Δ** cho Full nhưng **Δ** cho Incremental; chỉ diễn giải trong mỗi method, không so hai throughput này như cùng mẫu số. `app_bytes_read`, DB growth và OS I/O là ba khái niệm khác nhau.
- Hai run dùng cùng nguồn raw `ratings.csv` và harness SHA-256 `e1a63d1af28f973337c9cf78a349ccb8b068af299cbe7e6d323ed504af44d4c2`, nhưng khác canonical prefix SHA/config; giữ riêng summary và không ghép raw như một run.

## Bằng chứng đã kiểm lại

| Run | Grid | Raw CSV SHA-256 | Kết quả audit |
|---|---|---|---|
| `D12-repeat-grid` | H=100k/500k/1M, Δ=1k/10k | `fb0f83c6fd4415daa77e40275eeaea6d0393588bf3cc6ab35a9e5ea954ac260c` | 72 rows, 36 cặp, 6 cases PASS |
| `D12-scale-2m` | H=2M, Δ=1k/10k | `4e95552d63dd3e006652dc31df5035bcefea921acaf6735ac1aa77634fc7ef35` | 24 rows, 12 cặp, 2 cases PASS |

`results/raw/exp01_summary_d12.csv` SHA-256 `590ff3ec77cf2621de2f9b415c7d4fa06b9b75e695346b5dc0d1d76dd35ac973`; scale summary SHA-256 `0380a7383703db129fc18bc43fdcbc2b084f243c99d0a9a3501d3cf205f90a69`. Script tái tính các trường summary từ raw và đối chiếu từng giá trị. Mỗi cặp Full/Incremental có logical registry/edge/degree hashes trùng; từng Parquet hiện có khớp SHA-256 trong manifest của chính nó, và paired manifest hashes trùng.

Từ root repo:

```powershell
..\venv\Scripts\python.exe -m scripts.audit_exp01_results --raw results/raw/exp01_full_vs_incremental.csv --summary results/raw/exp01_summary_d12.csv --report data/processed/d12_exp01/D12-repeat-grid/report.json
..\venv\Scripts\python.exe -m scripts.audit_exp01_results --raw results/raw/exp01_scale2m_d12.csv --summary results/raw/exp01_scale2m_summary_d12.csv --report data/processed/d12_exp01/D12-scale-2m/report.json
```

**Giới hạn:** SHA-256 xác nhận các file local khớp manifest do harness tạo; không chứng minh cache được kiểm soát, sampled RSS là peak thật, hay hiệu năng trên máy khác/32M. Worktree lúc benchmark dirty nhưng harness hash và config/source hashes đã ghi. S07 sẽ tổng hợp scalability từ đúng các run này, giữ điều kiện đo và giới hạn cạnh mỗi claim.
