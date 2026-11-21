# S02 — Evaluator Top-20 và validation-only smoke

**Ngày:** 21/11/2026. **Trạng thái:** S02 evaluator PASS; **không phải REC-01 final quality run**. [S01 protocol](s01-evaluation-spec.md) đã được Project Owner duyệt. Probe này chỉ kiểm tra đường tích hợp bằng checkpoint D08 cũ 100 bước, chưa có validation checkpoint selection và không đọc nhãn test.

## Đã triển khai

- `src/evaluation/baselines.py`: Popularity đếm unique train-positive users/movie, xếp theo count giảm và raw movie ID tăng; bắt buộc candidate set đúng toàn bộ train movie catalog, dùng cùng warm user và seen mask.
- `src/evaluation/rec01.py`: gọi LightGCN và Popularity trên cùng train state, future validation window, cohort/candidates/seen; trả Top-20 và macro Recall/NDCG; từ chối graph edge có `first_event_id` vượt train boundary.
- `src/evaluation/checkpoint_selection.py`: chọn bản ghi validation NDCG@20 cao nhất, rồi Recall@20, rồi bước sớm hơn; từ chối test fields và metric không hợp lệ. S03 sẽ gắn helper vào các candidate checkpoints.
- `src/evaluation/s02_probe.py`: xác minh hash canonical prefix 1M, cutoff/counts, train graph/checkpoint cũ; truy vấn Parquet chỉ đến `valid_end_ts`, rồi tính validation Top-20 trên CUDA. Không query test partition.

## Kết quả probe thật, chỉ validation

Run chính `data/processed/s02_probe/validation_only_d08_1m_verified/report.json` có status PASS và label `S02_VALIDATION_ONLY_SMOKE_NOT_REC01`; lượt `validation_only_d08_1m` trước đó là chẩn đoán, đã được thay bằng run có xác minh đủ graph Parquet hashes. Input SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`, checkpoint D08 SHA-256 `6516e8f1b72cb8da24daae9e609db7d83a8665f15a6cffe7a68cfd033eb1280d`; train/validation events 802.988/102.046. Chỉ có 482 warm validation users, 3.333 warm pairs từ 47.300 future positives và 773 train movie candidates.

| Cách xếp hạng | Validation Recall@20 | Validation NDCG@20 |
|---|---:|---:|
| LightGCN checkpoint D08 100 bước | 0,11949 | 0,08200 |
| Popularity cùng train graph/cohort | 0,12070 | 0,08534 |

Popularity nhỉnh hơn trong **smoke validation** này. Không đổi cấu hình hay protocol để làm đẹp số. Chưa kết luận LightGCN thua ở REC-01 vì D08 checkpoint cũ chỉ có 100 bước, chưa chọn bằng validation; không có test metric, scale khác và catalog lớn hơn chưa đo. Cohort warm nhỏ là giới hạn quan trọng.

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests
..\venv\Scripts\python.exe -m src.evaluation.s02_probe --source data/processed/canonical_events/canonical_1000000.parquet --previous-run data/processed/d08_baseline_1m --out data/processed/s02_probe/my-fresh-run --device cuda
```

S03 cần tạo/lưu ba candidate checkpoints ở bước 100/300/500, chọn bằng validation và chỉ sau đó mới tính test. S04 mới có thể công bố REC-01 comparison chính.
