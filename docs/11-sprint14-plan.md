# Kế hoạch S01–S14 theo định hướng v1.0

**Bắt đầu:** 21/11/2026. **Ngân sách dự kiến:** 14 ngày làm việc, 5–6 giờ/ngày; đây không phải ngày nộp chính thức. [Nguồn định hướng đã thống nhất](10-direction-update-v1.md). [Kế hoạch D01–D21](04-project-plan.md) và log D01–D13 là lịch sử, không đổi số hiệu thí nghiệm.

## Mục tiêu và thứ tự ưu tiên

**Hai trụ cột bắt buộc:** Big Data Engineering (MovieLens 32M raw/canonical, DuckDB/Parquet, Full/Incremental correctness, EXP-01) và Graph-based Deep Recommender Systems (LightGCN đã train, REC-01 vs Popularity, checkpoint/Top-20 inference). Streamlit là bề mặt demo tương tác. MovieLens 32M là *nguồn*; model có thể train ở scale thấp hơn có chứng cứ tài nguyên. Rating/graph mới không đồng nghĩa checkpoint/embedding đã làm mới.

- **P0:** end-to-end pipeline; LightGCN training trên Kaggle GPU nếu khả dụng; temporal Top-20 Recall/NDCG và Popularity cùng protocol; checkpoint/config/ID mapping; EXP-01 logs; Streamlit gợi ý, rating và graph refresh demo; tests, báo cáo, slides, reproducibility.
- **P1:** ItemKNN, training/inference profiling, REC-02 stale vs propagation-only, version dashboard, mở rộng benchmark.
- **P2:** PinSage/XSimGCL, full 32M model training, Spark/Hadoop/Kafka/Neo4j, nâng cao EXP-02, multi-GPU, production deployment/cold-start.
- **Đã có nhưng không thay thế P0:** D13 EXP-02 R=1/R=5. Giữ mã và kết quả lịch sử; không dùng chi phí normalization làm đại diện thời gian train LightGCN.

## Sprint và acceptance evidence

| Sprint | Công việc | Bằng chứng cần có | Trạng thái 21/11 |
|---|---|---|---|
| S01 | Audit code, khóa REC-01 protocol | [Evaluation spec](s01-evaluation-spec.md), gap matrix, cutoffs/candidate/selection rule | DONE; Project Owner đã duyệt protocol |
| S02 | Evaluator Top-20 và Popularity | Unit toy + leakage tests; hai methods cùng candidate/mask/cohort | DONE: [validation-only smoke](s02-validation-smoke.md) và dùng lại trong REC-01 |
| S03 | Kaggle LightGCN training | Script/notebook, environment manifest, candidate checkpoints, ID map, graph version, hashes; fallback ghi rõ nếu Kaggle không có | DONE: [Kaggle Version 3](s03-kaggle-workflow.md), `seal-output` và local `verify-output` PASS |
| S04 | REC-01 quality evaluation | Validation selection, test độc lập, raw Recall/NDCG@20 vs Popularity, exclusions | DONE: [sáu rows và cohort exclusions](s04-rec01-results.md); Popularity nhỉnh hơn trên test |
| S05 | Recommendation pipeline | Load checkpoint, Top-20 warm-user inference, metadata LEFT JOIN, cold fallback | DONE: [loader và JSON CLI](s05-recommendation-pipeline.md), 8 toy tests và smoke trên artifact Kaggle thật PASS |
| S06 | EXP-01 benchmark | D11–D12 đã có raw 100k–2M; audit để tái dùng hợp lệ | DONE: [audit D12](s06-exp01-audit.md) 96/96 rows, 48/48 cặp PASS; giữ raw runs cũ |
| S07 | Benchmark analysis/scalability | Figures/claims từ raw, correctness và limitations | DONE: [Engineering conclusions](s07-exp01-engineering-conclusions.md), figures tái sinh từ raw; gate DE PASS trong phạm vi H≤2M |
| S08 | REC-02 freshness | Stale vs propagation-only trên warm nodes, chi phí và quality cùng protocol nếu đủ | TODO, P1 |
| S09 | Streamlit MVP | Chọn warm user/luồng demo, catalog/history, Top-20 title/genres, version state | DONE: [app, tests và browser smoke](s09-streamlit-mvp.md); snapshot chỉ đọc |
| S10 | Rating và graph refresh demo | Append interaction hợp lệ, batch apply/idempotency, graph version tăng; refresh policy thật, không hứa model đổi nếu chưa refresh | DONE: [feedback store, UI và tests](s10-rating-graph-refresh.md); gate app PASS trong phạm vi local một tiến trình |
| S11 | Integration/scaling | E2E tests và failure paths, scale theo headroom thực | DONE: [probe 1M+20 ratings](s11-integration-scaling.md), exact Full/Incremental và 60 tests PASS; không suy rộng 32M |
| S12 | Feature freeze/report | Pin môi trường/artifacts, báo cáo với evidence | REPORT READY: [báo cáo IUH DOCX/PDF](s12-reproducibility.md), nội dung và render QA xong; còn metadata/chữ ký của chủ dự án |
| S13 | Slides/Q&A | Slides từ số liệu thực, rehearsal | READY: deck Spaces 12 slide, [PDF và Q&A](s13-defense-qa.md); chủ dự án cần điền thông tin và luyện trình bày |
| S14 | Final QA/submission | Reproduce từ README, package source/config/logs/checkpoint/demo | PACKAGE QA: [gói private và kiểm hash](s14-submission.md); formal submission chờ metadata/chữ ký và nơi nộp |

Không bỏ một thành phần P0 để mở rộng model scale. Khi Kaggle không khả dụng, ghi lỗi/quota và chạy training trên GPU laptop ở scale đã kiểm chứng; không trình bày là run Kaggle. EXP-01 chỉ so trên cùng phần cứng và cùng output contract. REC-01 là quality, REC-02 là freshness; giữ EXP-01/02 cũ theo tên cũ.

## Trạng thái code lúc mở sprint

- **Có thể tái dùng:** D02 deterministic canonical prefix; D04/05 graph state correctness; D06 sparse normalization; D08 temporal train-only và macro metric generic K; D10 checkpoint + ID map/CLI Top-10; D11–D12 EXP-01; D13 EXP-02.
- **Cần xây tiếp:** Streamlit; ghi rating mới và graph refresh; báo cáo tổng hợp. Popularity/Top-20, validation checkpoint selection, Kaggle workflow, reusable checkpoint inference và [REC-01](s04-rec01-results.md) đã có. [Kiểm kê ban đầu S01](s01-evaluation-spec.md).
- **Rủi ro kiểm chứng sớm:** warm cohort trên prefix 1M nhỏ (D08: 482 valid, 403 test users), dù model chạy nhanh. Báo cáo phải nêu candidate universe chỉ gồm 773 train movies và phần tương tác cold bị loại. Không so 100k/1M metrics như cùng một protocol.

## Quy tắc thực thi

Mỗi sprint cập nhật `docs/05-work-log.md`, `docs/07-experiment-log.md` khi có run, `docs/08-report-notes.md`, `docs/09-report-evidence.md`; dẫn raw/hashes/config thật. Trước khi kết luận một gate PASS, chạy test có phạm vi tương ứng và kiểm tra artifact/reload. Không tự commit/push. Các quyết định mới về cutoff, candidate universe, model refresh hoặc rating semantics phải có ADR và không được chỉnh sau khi xem test để làm đẹp metric. REC-01 S01 đã được Project Owner duyệt.
