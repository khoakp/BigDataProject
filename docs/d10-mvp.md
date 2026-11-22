# D10 — Clean rerun và MVP gate G3

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

## Luồng và artifact

`src/models/d10_mvp.py` bắt đầu từ `../ml-32m/ratings.csv`, chạy lại D02 canonicalization trong thư mục mới, rồi gọi D09 để xử lý train-only D04 incremental graph, exact Full equivalence, D06 normalized adjacency, LightGCN, raw Top-K và metadata Top-K. `movies.csv` được validate trước enrichment. Run manifest lưu input/code/config hashes, môi trường, các lần thử device, đường dẫn và SHA-256 của artifacts. Checkpoint chứa embedding weights và graph version; `model_ids.json` xuất ánh xạ `model_index → node_type, raw_id, first_event_id` để inspect độc lập.

Lệnh từ root `BigDataProject`:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
..\venv\Scripts\python.exe -m src.models.d10_mvp --out data/processed/d10_mvp/D10-clean-100k --device cuda --steps 100 --prefix-events 100000
..\venv\Scripts\python.exe -m src.models.d10_demo data/processed/d10_mvp/D10-clean-100k
```

Chọn `--out` **mới/trống** cho mỗi clean rerun. Nếu GPU OOM hoặc CUDA lỗi, manifest giữ một attempt `OOM`/`ERROR` và dùng thư mục fallback CPU riêng; không biến run GPU lỗi thành PASS. Test injection xác nhận cả hai attempt được ghi khi fallback cũng thất bại. D07 đã đo memory GPU theo phase; D10 chưa đặt một kết quả peak VRAM mới.

## Kịch bản demo 3–5 phút (khoảng 4 phút)

1. **0:00–0:40 — Bài toán và nguồn.** Giới thiệu `ratings.csv` MovieLens 32M là lịch sử gốc. Mở `run_manifest.json`, chỉ source hashes, `git_head`/`git_dirty`, phiên bản PyTorch/DuckDB và cấu hình prefix 100k. Giải thích vì sao cần canonical sort theo event time.
2. **0:40–1:25 — Graph có version.** Mở `pipeline/report.json`: train cutoff chỉ dùng quá khứ; D04 ledger có graph version, registry/edges/degrees khớp Full Rebuild. Nhắc `model_ids.json` cố định ánh xạ raw ID sang embedding index.
3. **1:25–2:15 — Model và đánh giá.** Chỉ `model.pt`, normalized graph, BPR 100 bước và `valid/test` Recall@10/NDCG@10. Nêu warm cohort và excluded cold users, không suy rộng metric.
4. **2:15–3:15 — Gợi ý phim.** Chạy lệnh `python -m src.models.d10_demo ...`; cho thấy movie ID, score, title/genres và metadata coverage. Mở một dòng trong `topk_raw.json` và `topk_enriched.json` để chỉ raw IDs/rank/score giữ nguyên.
5. **3:15–4:00 — Tính đúng và giới hạn.** Chỉ `integration_report.json` có ranking/metric invariance và source hash. Kết luận G3 trên 100k; nhấn mạnh chưa có EXP-01/02 benchmark hay model baseline comparison.

Lệnh demo kiểm SHA-256 của tất cả artifacts trong manifest trước khi hiển thị; không train lại. Có thể dùng `--user <raw_id>` cho user thuộc validation warm cohort.

## Kết quả clean rerun

Run `D10-clean-100k` hoàn thành ngày 20/11/2026: **PASS**, CUDA attempt duy nhất PASS; từ `ratings.csv` gốc đến 100.000 canonical events, train graph version 9 với 2.562 nodes, 52.014 edges, exact Full/Incremental equivalence PASS. Checkpoint SHA-256 `60afdffe70797192a51a62e8d5d9a337eb48cb45cb2cfde71eeb64637ece587f`; `model_ids.json` có 2.562 mappings liên tiếp. Raw/enriched Top-K có 300/300 dòng, metadata found 300, ranking fields và metrics không đổi. Valid Recall@10/NDCG@10 = 0,03491/0,06418 trên 20 warm users. Test có 10 warm users, cả hai metric bằng 0.

`run_manifest.json` ghi ratings SHA-256 `91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee`, movies SHA-256 `b37ca1abc7798de741138ed252b62f69f7e37c84b8a8fab1b82d409b4c6c5cc2`, canonical prefix SHA-256 `8a01bc6f132362f44ab5a206575fb16ab7f34ce44c6bd224b418ea076c7bf519`, config/code hashes và SHA-256 cho 12 artifacts. Python 3.10.11, DuckDB 1.5.6, PyTorch 2.8.0+cu126 và RTX 4050 Laptop GPU. Git HEAD `123942550cf8192e5cc85f1c6f1432216ff39ab5`, `git_dirty=true`: chưa commit các thay đổi D08–D10 nên code hashes trong manifest là provenance chính xác của run. Thời gian từ `started_utc` đến `finished_utc` khoảng 56,5 giây; đây là thời gian run này, không phải benchmark EXP-01.

Lệnh demo đã chạy thành công và xác minh đủ 12 artifact hashes. User validation mẫu `12879` có Top-10 với raw movie IDs, score và metadata; vị trí 1 là `movieId=296` (*Pulp Fiction (1994)*). Suite 27 tests PASS, gồm T-07/08/10/11/12 và injection T-09. **G3 MVP PASS trong phạm vi prefix 100k**; không khẳng định chất lượng model trên toàn bộ MovieLens 32M.
