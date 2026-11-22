# D02 — Profile ratings và canonical prefix

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

Ngày chạy: 19/11/2026. Input: `../ml-32m/ratings.csv`, SHA-256 `91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee`; MD5 nguồn xem [source-manifest](source-manifest.md). Python 3.10.11, DuckDB 1.5.6 (`../venv`), một DuckDB thread. Code: [`src/ingest/d02_profile.py`](../src/ingest/d02_profile.py); test: [`tests/test_d02.py`](../tests/test_d02.py). HEAD trước thay đổi: `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b`; chưa commit D02.

## Profile toàn bộ ratings.csv

| Metric | Kết quả |
|---|---:|
| Rows | 32,000,204 |
| Distinct users | 200,948 |
| Distinct movies trong ratings | 84,432 |
| Null ở bốn cột | 0 |
| Rating min / max | 0.5 / 5.0 |
| Rating ngoài [0.5, 5.0] hoặc lệch bước 0.5 | 0 |
| Timestamp min / max (Unix seconds) | 789652004 / 1697164147 |
| Timestamp ≤ 0 | 0 |
| Distinct timestamps | 25,976,523 |
| Rows vượt số timestamp phân biệt | 6,023,681 |
| Distinct `(userId,movieId)` | 32,000,204 |
| Rows vượt số cặp phân biệt | 0 |
| Lần timestamp giảm khi duyệt source order | 15,647,103 |

`84,432` là số movie ID xuất hiện trong **ratings**, khác với `87,585` movies trong thống kê toàn dataset do README công bố. `6,023,681` là `rows - distinct timestamps`, không phải số nhóm timestamp có tie. Không có cặp user–movie trùng trong file đã kiểm tra, nhưng logic graph ở D03 vẫn cần dedup để đúng với hợp đồng và dữ liệu batch khác. Source không theo thứ tự thời gian, nên cần sort rõ ràng.

## Canonical contract và output

`source_row_id` là vị trí dòng CSV (0-based) từ scan một thread với `preserve_insertion_order=true`. Sort theo `(event_ts, user_raw_id, movie_raw_id, source_row_id)`, rồi cấp `event_id` 0-based. Output chứa `event_id`, `event_ts`, `user_raw_id`, `movie_raw_id`, `rating` dạng float32, `source_row_id`. `is_positive` chưa được cấp vì ngưỡng DEC-005 còn đề xuất; `batch_id` chỉ gán ở replay. Cấu hình và hash có trong `data/processed/canonical_events/manifest.json` (ngoài Git).

| Prefix | Rows | Bytes Parquet | SHA-256 của file, hai run giống nhau |
|---|---:|---:|---|
| 100k | 100,000 | 1,210,758 | `8a01bc6f132362f44ab5a206575fb16ab7f34ce44c6bd224b418ea076c7bf519` |
| 1M | 1,000,000 | 12,994,021 | `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae` |

Kiểm tra 1M prefix: `event_id` min/max `0/999999`, 1,000,000 `source_row_id` phân biệt, 0 vi phạm thứ tự. Toy test T-01 dùng timestamp ties và hai dòng giống nhau: `source_row_id` theo `event_id` là `[1,2,3,0]`; hai lần xuất Parquet cho cùng toy input có SHA-256 giống nhau. Full data được chạy hai lần và hash của cả hai prefix không đổi. Điều này kiểm chứng tính lặp lại **trong cấu hình hiện tại**; không bảo đảm hash Parquet giống nhau khi đổi phiên bản DuckDB hoặc writer settings.

## Tái lập

```powershell
..\venv\Scripts\python.exe src\ingest\d02_profile.py
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Chạy lại script tạo cùng output trong `data/processed/canonical_events/`. Input SHA-256 và transform manifest phải được đối chiếu trước khi so hash. Không dùng prefix 1M như MovieLens 1M độc lập. Chưa tạo full canonical 32M Parquet; D02 chỉ yêu cầu prefix 100k/1M.
