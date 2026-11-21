# S11 — Integration, failure paths và scale demo

**Trạng thái 21/11/2026:** DONE trong phạm vi REC-01 canonical prefix 1M và một Streamlit server process. [`scripts/s11_integration_probe.py`](../scripts/s11_integration_probe.py) chạy batch 20 feedback ratings trên bản sao sealed Kaggle graph, dựng Full Rebuild độc lập ở cùng train cutoff cộng batch, rồi so từng dòng registry/edges/degrees. [`test_s11_feedback_failures.py`](../tests/test_s11_feedback_failures.py) kiểm gián đoạn, đầu vào sai, sửa intent và hai request đồng thời trong cùng tiến trình.

## Input, output và cách chạy

Từ root repo `BigDataProject`:

```powershell
..\venv\Scripts\python.exe -m scripts.s11_integration_probe --out data/processed/s11_integration/my-run
..\venv\Scripts\python.exe -m unittest discover -s tests -q
```

`--out` phải mới. Run thật dùng `REC01-Kaggle-1M` [Kaggle Version 3](s03-kaggle-workflow.md), source SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`, sealed train graph SHA-256 `375e2ce4c150b3a2c77733030752a1f7fc18b5b3ab8bba1ba114fcfecbca7e16`. Output ngoài Git: `data/processed/s11_integration/rec01_1m_batch20_v1/report.json`, `feedback/`, combined Parquet và Full Rebuild Parquets. Combined events SHA-256 `eb8d2014a76e7acf2ddb135a82d8c4b124e50467a12da6e31cac88657067de70`; số hash này gắn với batch timestamp/UUID của run đã lưu, không kỳ vọng giống khi chạy lại.

## Kết quả đã đo

| Kiểm tra | Trước | Sau batch 20 ratings | Kết quả |
|---|---:|---:|---|
| Canonical graph version | 81 | 82 | Tăng đúng một version |
| Model graph version / checkpoint step | 81 / 300 | 81 / 300 | Giữ nguyên |
| Positive edges | 392.626 | 392.636 | 10 ratings `5,0` tạo 10 cạnh; 10 ratings `2,5` không tạo cạnh |
| Registry rows | — | 14.388 | Full = Incremental từng dòng |
| Edge rows | — | 392.636 | Full = Incremental từng dòng |
| Degree rows | — | 14.388 | Full = Incremental từng dòng |
| Replay cùng UUID | — | `NO_OP` | Graph version không tăng lần nữa |
| Model snapshots trong branch | `[81]` | `[81]` | Chưa materialize graph mới |

Warm LightGCN Top-20 và cold Popularity Top-20 trước/sau batch bằng nhau. Sealed Kaggle graph vẫn giữ cùng SHA-256. Một lượt chạy trên máy này ghi `load_and_copy=2,621 s`, `apply_batch=0,183 s`, `full_reference_and_compare=0,994 s`; đây là diagnostic của một batch nhỏ, **không phải** EXP-01 benchmark, không so tốc độ Full/Incremental và không suy rộng lên 32M.

## Failure paths, giới hạn và diễn giải

- Giả lập dừng sau khi intent JSON được ghi nhưng trước Parquet/apply: bản graph chưa đổi; mở store lần sau replay intent và áp dụng một lần.
- Payload thiếu/thừa trường, rating/ID/timestamp sai bị reject trước khi ghi; sửa nội dung intent sau khi batch đã tồn tại bị phát hiện khi so Parquet; cùng UUID khác ratings bị từ chối.
- Hai request qua hai threads trên **cùng server process** nhận versions 82/83 và event IDs 802988/802989 liên tục. Lock chỉ trong tiến trình; chưa bảo đảm nhiều server process hoặc multi-host.
- Full reference dựng từ 802.988 train events và 20 feedback events, không đưa validation/test tương lai vào graph. Đây là chứng cứ correctness ở 1M+20, không chứng minh performance hay quality sau feedback. Chưa có REC-02 model refresh.

Toàn bộ **60 local tests PASS** ngày 21/11/2026, bao gồm S09/S10 AppTest và S11 failure tests. `report.json` của probe là bằng chứng thô cho counts/hashes/timings; bảng ở đây chỉ tóm tắt các trường đã kiểm.

**Bản nháp báo cáo:** Trên train graph REC-01 từ canonical prefix 1M, một batch 20 rating mới làm canonical graph tăng từ version 81 lên 82. Full Rebuild trên train events cộng batch tạo đúng cùng registry, edge state và degree state như incremental update; 10 rating đạt ngưỡng thêm 10 cạnh, 10 rating thấp hơn không thêm cạnh. Checkpoint LightGCN vẫn thuộc model graph version 81 và Top-20 giữ nguyên. Điều này xác nhận cập nhật trạng thái graph trong điều kiện thử nghiệm, không xác nhận model đã thích nghi với feedback.
