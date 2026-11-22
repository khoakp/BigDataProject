# D08 — Temporal split và LightGCN baseline

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

Ngày chạy: 20/11/2026 (Asia/Saigon). Mã nguồn: `src/evaluation/temporal.py`, `src/models/d08_baseline.py`. Lệnh từ gốc `BigDataProject`:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
..\venv\Scripts\python.exe -m src.models.d08_baseline --steps 100 --device cuda
..\venv\Scripts\python.exe -m src.models.d08_baseline --source data/processed/canonical_events/canonical_1000000.parquet --out data/processed/d08_baseline_1m --steps 100 --device cuda
```

Cutoff chọn theo 80% và 90% **distinct timestamps** của canonical prefix. Train gồm `event_ts <= train_end`; validation là `(train_end, valid_end]`; test là `> valid_end`. Tất cả events cùng timestamp thuộc cùng một phía. Full Rebuild và normalization chỉ dùng train prefix. Model có 64 chiều, 3 propagation layers, Adam lr 0,001, 100 bước BPR × 1024 triples, seed 7. Negative chỉ lấy từ train movie catalog và không thuộc positive train edges của user. Evaluation chỉ dùng warm user/movie đã có embedding trong train, bỏ train-seen positives khỏi candidates, Top-10 được chấm theo movie ID thô; không đọc metadata. Recall@10 và NDCG@10 là macro average trên warm users có ít nhất một future positive chưa seen. Checkpoint cố định tại train cutoff cho **cả** valid và test; test là một horizon xa hơn từ cùng snapshot, không retrain bằng validation.

| Prefix | Train / valid / test events | Train nodes / edges | Warm users valid / test | Recall@10 valid / test | NDCG@10 valid / test |
|---|---:|---:|---:|---:|---:|
| 100k | 86.771 / 6.376 / 6.853 | 2.562 / 52.014 | 20 / 10 | 0,03491 / 0 | 0,06418 / 0 |
| 1M | 802.988 / 102.046 / 94.966 | 14.388 / 392.626 | 482 / 403 | 0,06390 / 0,07154 | 0,06567 / 0,07092 |

Run 1M: source SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`; Git HEAD trước thay đổi D08 `123942550cf8192e5cc85f1c6f1432216ff39ab5`; PyTorch 2.8.0+cu126 / CUDA. Raw manifest, raw Top-K và checkpoint ngoài Git: `data/processed/d08_baseline_1m/`. Run 100k tương tự ở `data/processed/d08_baseline/`. Unit suite 22 tests PASS, gồm kiểm tra train graph và negative catalog không chứa future item.

**Giới hạn:** Trong 1M, 47.300 valid future positive pairs chỉ còn 3.333 warm pairs; 43.723 pairs bị loại do cold user, 244 do cold movie. Test có 44.865 future pairs, 2.069 warm pairs; 42.176 cold user và 620 cold movie. Vì vậy metric chỉ mô tả cohort nhỏ, không đại diện toàn bộ future events. BPR loss giảm không chứng minh chất lượng ranking. Đây là baseline ngắn, chưa tune và chưa so với Popularity/BPR-MF. Raw Top-K D08 chưa JOIN `movies.csv`; việc này thuộc D09.
