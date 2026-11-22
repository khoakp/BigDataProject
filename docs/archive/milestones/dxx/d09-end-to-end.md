# D09 — Incremental graph đến metadata Top-K

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](../../../current/scope-and-decisions.md) · [Tiến độ v2](../../../current/progress-and-evidence.md).

Ngày chạy: 20/11/2026 (Asia/Saigon). Entry point: `src/models/d09_pipeline.py`. D09 lấy canonical prefix, cắt train bằng event time theo D08, chia train thành batch 10.000 events và gọi `IncrementalGraph.apply_batch()`. Graph version đến từ ledger D04. Tại train cutoff, registry/edges/degrees được so sánh exact với Full Rebuild; sau đó D06 normalization, LightGCN training/evaluation và Top-K raw IDs chạy theo D08. `movies.csv` được validate trước khi tra cứu theo `movieId`; metadata chỉ bổ sung sau ranking. Output gồm `graph.duckdb`, `model.pt`, `topk_raw.json`, `topk_enriched.json`, `report.json` và `integration_report.json` trong thư mục ngoài Git.

Chạy từ root `BigDataProject`:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
..\venv\Scripts\python.exe -m src.models.d09_pipeline --out data/processed/d09_incremental_100k --device cuda --steps 100
..\venv\Scripts\python.exe -m src.models.d09_pipeline --source data/processed/canonical_events/canonical_1000000.parquet --out data/processed/d09_incremental_1m --device cuda --steps 100
```

| Prefix | Train events | D04 graph version | Full equivalence | Raw / enriched rows | Metadata found | Ranking/metrics invariant |
|---|---:|---:|---|---:|---:|---|
| 100k | 86.771 | 9 | PASS | 300 / 300 | 300 | PASS |
| 1M | 802.988 | 81 | PASS | 8.850 / 8.850 | 8.850 | PASS |

`movies.csv` thực có 87.585 unique/non-null movie IDs, 0 title/genres trống, 7.080 nhãn `(no genres listed)`. Trên hai Top-K runs, thiếu metadata ID/title/genres đều bằng 0. Test toy cũng bao gồm movie ID không có metadata, title/genres trống và duplicate `movieId`: dòng thiếu vẫn giữ rank/score, duplicate bị reject trước enrichment. T-08 so chunked scoring với full-score reference độc lập trên toy, kể cả score bằng nhau; T-10/11/12 và tổng 25 unit tests PASS.

Run 1M: canonical source SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`; movies source SHA-256 `b37ca1abc7798de741138ed252b62f69f7e37c84b8a8fab1b82d409b4c6c5cc2`; checkpoint SHA-256 của lần chạy cuối `f2e7aba0b8e1ce715c6332bb735fc0c18b32189684f038ba9cc7b8d7e9ab5448`. PyTorch 2.8.0+cu126, CUDA, seed 7, 64 chiều, 3 lớp, 100 bước BPR. Git HEAD khi chạy `123942550cf8192e5cc85f1c6f1432216ff39ab5` là commit trước thay đổi D08/D09 chưa commit. Full cấu hình và metrics ở `data/processed/d09_incremental_1m/report.json`.

**Giới hạn:** Đây là E2E trên chronological prefixes 100k/1M, chưa chạy toàn bộ 32M. Warm cohort bị loại nhiều cold users như [D08](d08-temporal-baseline.md); metadata coverage 100% chỉ đúng cho Top-K ở hai runs này. Version 9/81 là số batch train đã apply, không phải event count. D10 còn clean rerun/demo và đóng gói MVP.
